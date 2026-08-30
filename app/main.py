from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database import get_db, init_db
from app.schemas import PaymentFailedEvent, ActionResult
from app.detect import detect
from app.diagnose import diagnose
from app.decide import decide
from app.act import act
from app.models import ActionResultRecord, DecisionRecord, EventRecord

app = FastAPI(title="AI Revenue Recovery Agent")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.on_event("startup")
def startup():
    init_db()


def run_pipeline(db: Session, event: PaymentFailedEvent) -> ActionResult | None:
    """The full Detect -> Diagnose -> Decide -> Act pipeline for one event."""
    accepted = detect(db, event)
    if not accepted:
        return None

    diagnosis = diagnose(event)

    from app.models import DiagnosisRecord
    db.add(DiagnosisRecord(
        event_id=diagnosis.event_id,
        payment_id=diagnosis.payment_id,
        predicted_root_cause=diagnosis.predicted_root_cause.value,
        confidence=diagnosis.confidence,
        model_version=diagnosis.model_version,
    ))
    db.commit()

    decision = decide(db, diagnosis, amount=event.amount)
    result = act(db, decision, amount=event.amount, event_id=event.event_id, event=event)
    return result


@app.post("/webhook/payment-failed")
def webhook_payment_failed(event: PaymentFailedEvent, db: Session = Depends(get_db)):
    """Entry point for a Razorpay-style payment.failed webhook."""
    result = run_pipeline(db, event)
    if result is None:
        return {"status": "dropped", "reason": "duplicate_or_stopped"}
    return {"status": "processed", "outcome": result.outcome, "action": result.action.value}


@app.get("/metrics")
def metrics(db: Session = Depends(get_db)):
    total_events = db.query(func.count(EventRecord.event_id)).scalar() or 0
    total_amount_at_risk = db.query(func.sum(EventRecord.amount)).scalar() or 0.0

    recovered = db.query(ActionResultRecord).filter_by(outcome="recovered").all()
    total_recovered_amount = sum(r.amount_recovered or 0 for r in recovered)
    recovered_count = len(recovered)

    failed_actions = db.query(func.count(ActionResultRecord.id)).filter_by(outcome="failed").scalar() or 0
    escalated = db.query(func.count(ActionResultRecord.id)).filter_by(outcome="escalated").scalar() or 0
    pending = db.query(func.count(ActionResultRecord.id)).filter_by(outcome="pending").scalar() or 0
    skipped = db.query(func.count(ActionResultRecord.id)).filter_by(outcome="skipped").scalar() or 0

    root_cause_breakdown = dict(
        db.query(DecisionRecord.root_cause, func.count(DecisionRecord.id))
        .group_by(DecisionRecord.root_cause).all()
    )
    action_breakdown = dict(
        db.query(DecisionRecord.action, func.count(DecisionRecord.id))
        .group_by(DecisionRecord.action).all()
    )

    return {
        "total_events_processed": total_events,
        "total_amount_at_risk_inr": round(total_amount_at_risk, 2),
        "total_amount_recovered_inr": round(total_recovered_amount, 2),
        "recovery_rate_pct": round(recovered_count / total_events * 100, 2) if total_events else 0,
        "recovered_count": recovered_count,
        "failed_action_count": failed_actions,
        "escalated_count": escalated,
        "pending_approval_count": pending,
        "skipped_count": skipped,
        "root_cause_breakdown": root_cause_breakdown,
        "action_breakdown": action_breakdown,
    }


@app.get("/audit/{payment_id}")
def audit_trail(payment_id: str, db: Session = Depends(get_db)):
    from app.models import AuditLog
    entries = db.query(AuditLog).filter_by(payment_id=payment_id).order_by(AuditLog.created_at).all()
    return [
        {"stage": e.stage, "payload": e.payload, "timestamp": e.created_at.isoformat()}
        for e in entries
    ]


@app.get("/health")
def health():
    return {"status": "ok"}