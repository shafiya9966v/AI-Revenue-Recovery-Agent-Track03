"""
Decision layer. Pure deterministic policy - NO ML here, on purpose.

This is where "bounded and gated" is actually enforced. Even if the classifier
is confident and wrong, these gates are the last line of defense:
  1. Never exceed the per-cause max-attempts cap.
  2. Never auto-act on fraud_suspected - always escalate to a human.
  3. Low-confidence diagnosis -> escalate instead of guessing.
  4. High-value payments -> require human approval regardless of cause.
"""
from datetime import datetime, UTC
from sqlalchemy.orm import Session
from app.schemas import DiagnosisResult, Decision, ActionType, RootCause
from app.models import PaymentAttemptCounter
from app.audit import log
from app.config import POLICY_TABLE, CONFIDENCE_THRESHOLD, HIGH_VALUE_APPROVAL_LIMIT_INR


def _get_or_create_counter(db: Session, payment_id: str) -> PaymentAttemptCounter:
    counter = db.query(PaymentAttemptCounter).filter_by(payment_id=payment_id).first()
    if counter is None:
        counter = PaymentAttemptCounter(payment_id=payment_id, attempts_made=0, stopped=False)
        db.add(counter)
        db.commit()
        db.refresh(counter)
    return counter


def decide(db: Session, diagnosis: DiagnosisResult, amount: float) -> Decision:
    counter = _get_or_create_counter(db, diagnosis.payment_id)
    policy = POLICY_TABLE[diagnosis.predicted_root_cause.value]

    reason_parts = [
        f"root_cause={diagnosis.predicted_root_cause.value}",
        f"confidence={diagnosis.confidence:.2f}",
    ]

    # GATE 1: low confidence -> escalate regardless of predicted cause
    if diagnosis.confidence < CONFIDENCE_THRESHOLD:
        decision = Decision(
            payment_id=diagnosis.payment_id,
            root_cause=diagnosis.predicted_root_cause,
            confidence=diagnosis.confidence,
            action=ActionType.escalate_human,
            attempt_number=counter.attempts_made + 1,
            requires_human_approval=True,
            reason=" | ".join(reason_parts + [
                f"below confidence threshold ({CONFIDENCE_THRESHOLD}) -> routed to human review"
            ]),
        )
        _log_decision(db, diagnosis, decision)
        return decision

    # GATE 2: fraud is never auto-actioned, no matter what
    if diagnosis.predicted_root_cause == RootCause.fraud_suspected:
        decision = Decision(
            payment_id=diagnosis.payment_id,
            root_cause=diagnosis.predicted_root_cause,
            confidence=diagnosis.confidence,
            action=ActionType.escalate_human,
            attempt_number=counter.attempts_made + 1,
            requires_human_approval=True,
            reason=" | ".join(reason_parts + ["fraud_suspected -> hard rule: always escalate, never auto-retry"]),
        )
        _log_decision(db, diagnosis, decision)
        return decision

    # GATE 3: unrecoverable -> write off, no action
    if diagnosis.predicted_root_cause == RootCause.unrecoverable:
        decision = Decision(
            payment_id=diagnosis.payment_id,
            root_cause=diagnosis.predicted_root_cause,
            confidence=diagnosis.confidence,
            action=ActionType.write_off,
            attempt_number=counter.attempts_made + 1,
            requires_human_approval=False,
            reason=" | ".join(reason_parts + ["classified unrecoverable -> logged and written off, no action taken"]),
        )
        _log_decision(db, diagnosis, decision)
        return decision

    # GATE 4: attempts cap already hit for this payment -> stop, escalate
    max_attempts = policy["max_attempts"]
    if counter.attempts_made >= max_attempts:
        counter.stopped = True
        db.commit()
        decision = Decision(
            payment_id=diagnosis.payment_id,
            root_cause=diagnosis.predicted_root_cause,
            confidence=diagnosis.confidence,
            action=ActionType.escalate_human,
            attempt_number=counter.attempts_made,
            requires_human_approval=True,
            reason=" | ".join(reason_parts + [
                f"max_attempts ({max_attempts}) reached -> stopping automated recovery, escalated"
            ]),
        )
        _log_decision(db, diagnosis, decision)
        return decision

    # GATE 5: high-value payment -> require human approval before acting
    requires_approval = amount > HIGH_VALUE_APPROVAL_LIMIT_INR

    action = ActionType(policy["action"])
    decision = Decision(
        payment_id=diagnosis.payment_id,
        root_cause=diagnosis.predicted_root_cause,
        confidence=diagnosis.confidence,
        action=action,
        attempt_number=counter.attempts_made + 1,
        requires_human_approval=requires_approval,
        reason=" | ".join(reason_parts + [
            f"action={action.value}",
            f"attempt {counter.attempts_made + 1}/{max_attempts}",
            f"high_value_approval_required={requires_approval}" if requires_approval else "standard auto-execution",
        ]),
    )

    # advance the counter only when we actually intend to act
    counter.attempts_made += 1
    counter.last_action_at = datetime.now(UTC)
    db.commit()

    _log_decision(db, diagnosis, decision)
    return decision


def _log_decision(db: Session, diagnosis: DiagnosisResult, decision: Decision):
    from app.models import DecisionRecord
    db.add(DecisionRecord(
        payment_id=decision.payment_id,
        root_cause=decision.root_cause.value,
        confidence=decision.confidence,
        action=decision.action.value,
        attempt_number=decision.attempt_number,
        requires_human_approval=decision.requires_human_approval,
        reason=decision.reason,
    ))
    db.commit()
    log(db, diagnosis.event_id, decision.payment_id, "decide", decision.model_dump(mode="json"))
