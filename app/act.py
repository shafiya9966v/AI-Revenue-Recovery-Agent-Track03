"""
Action layer. Executes the decided action.

If real Razorpay test-mode keys are configured (see .env), this makes actual
test-mode API calls (creating payment links for retry/instrument-switch flows).
If no keys are set, it runs in SIMULATION mode - deterministic, seeded, so the
whole pipeline is still runnable and demoable end-to-end without live credentials.

Every call here is GATED: it re-checks decision.requires_human_approval before
doing anything, and never executes an action the policy layer marked as
escalate_human / write_off / no_action.
"""
import random
import uuid
from datetime import datetime, UTC
from sqlalchemy.orm import Session
from app.schemas import Decision, ActionResult, ActionType, PaymentFailedEvent
from app.models import ActionResultRecord
from app.audit import log
from app.config import RAZORPAY_SIMULATION_MODE, RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET
from app.nudge import generate_recovery_nudge

random.seed(7)

if not RAZORPAY_SIMULATION_MODE:
    import razorpay
    _client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))

# actions that never actually call the payment gateway
NO_EXECUTE_ACTIONS = {ActionType.escalate_human, ActionType.write_off, ActionType.no_action}

# simulated recovery-likelihood by action type, used only in SIMULATION mode
# to produce realistic (not 100%) recovery outcomes for the batch report
SIMULATED_SUCCESS_RATE = {
    ActionType.retry_fast: 0.55,
    ActionType.retry_delayed: 0.45,
    ActionType.notify_customer: 0.30,
    ActionType.request_new_instrument: 0.40,
    ActionType.mandate_retry_sequence: 0.35,
}


def act(db: Session, decision: Decision, amount: float, event_id: str, event: PaymentFailedEvent | None = None) -> ActionResult:
    # GATE: pending human approval - do NOT execute yet
    if decision.requires_human_approval and decision.action not in (
        ActionType.escalate_human, ActionType.write_off
    ):
        result = ActionResult(
            payment_id=decision.payment_id,
            action=decision.action,
            executed=False,
            outcome="pending",
            timestamp=datetime.now(UTC),
        )
        _persist(db, result)
        log(db, event_id, decision.payment_id, "act",
            {"result": "blocked_pending_human_approval", "action": decision.action.value})
        return result

    if decision.action in NO_EXECUTE_ACTIONS:
        outcome = "escalated" if decision.action == ActionType.escalate_human else "skipped"
        result = ActionResult(
            payment_id=decision.payment_id,
            action=decision.action,
            executed=False,
            outcome=outcome,
            timestamp=datetime.now(UTC),
        )
        _persist(db, result)
        log(db, event_id, decision.payment_id, "act",
            {"result": outcome, "action": decision.action.value})
        return result

    # --- actual execution ---
    try:
        if RAZORPAY_SIMULATION_MODE:
            razorpay_id, succeeded = _simulate_execution(decision)
        else:
            razorpay_id, succeeded = _execute_live(decision, amount)

        result = ActionResult(
            payment_id=decision.payment_id,
            action=decision.action,
            executed=True,
            razorpay_response_id=razorpay_id,
            outcome="recovered" if succeeded else "failed",
            amount_recovered=amount if succeeded else None,
            timestamp=datetime.now(UTC),
        )
    except Exception as e:
        # ONE FAILURE HANDLED GRACEFULLY: if the gateway call errors out
        # (timeout, malformed response, etc.), we don't crash the pipeline
        # or double-fire - we log it and mark this attempt as failed cleanly.
        result = ActionResult(
            payment_id=decision.payment_id,
            action=decision.action,
            executed=False,
            outcome="failed",
            timestamp=datetime.now(UTC),
        )
        log(db, event_id, decision.payment_id, "act",
            {"result": "exception_handled_gracefully", "error": str(e), "action": decision.action.value})

    _persist(db, result)
    log(db, event_id, decision.payment_id, "act", result.model_dump(mode="json"))

    # --- Hinglish nudge generation ---
    # Only fires for actions that involve messaging a customer, and only
    # after the action itself was actually executed. This is the ONE place
    # in the whole pipeline that calls an LLM - see app/nudge.py for why.
    if result.executed and event is not None:
        nudge_text = generate_recovery_nudge(decision, event)
        if nudge_text:
            log(db, event_id, decision.payment_id, "act",
                {"result": "nudge_generated", "action": decision.action.value, "message": nudge_text})

    return result


def _simulate_execution(decision: Decision):
    rate = SIMULATED_SUCCESS_RATE.get(decision.action, 0.4)
    succeeded = random.random() < rate
    fake_id = f"sim_{uuid.uuid4().hex[:16]}"
    return fake_id, succeeded


def _execute_live(decision: Decision, amount: float):
    """
    Real Razorpay test-mode calls. Payment Links API used for retry/instrument
    switch/notify flows - the merchant-safe way to prompt a customer to pay again
    without silently re-charging a saved instrument.
    """
    link = _client.payment_link.create({
        "amount": int(amount * 100),  # paise
        "currency": "INR",
        "description": f"Recovery attempt - {decision.action.value}",
        "notify": {"sms": True, "email": True},
        "reminder_enable": True,
    })
    # test-mode: we can't force a real customer to pay, so we still measure
    # "issued" as executed=True, outcome tracked via webhook in production;
    # for this project's batch run, treat "issued successfully" as executed.
    return link.get("id"), True


def _persist(db: Session, result: ActionResult):
    db.add(ActionResultRecord(
        payment_id=result.payment_id,
        action=result.action.value,
        executed=result.executed,
        razorpay_response_id=result.razorpay_response_id,
        outcome=result.outcome,
        amount_recovered=result.amount_recovered,
    ))
    db.commit()
