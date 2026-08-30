from datetime import datetime, UTC
from sqlalchemy.orm import Session
from app.models import AuditLog


def log(db: Session, event_id: str, payment_id: str, stage: str, payload: dict):
    """
    Single choke point for the audit trail. Every stage (detect/diagnose/decide/act)
    calls this so we always have a complete, queryable explanation for any money action.
    """
    entry = AuditLog(
        event_id=event_id,
        payment_id=payment_id,
        stage=stage,
        payload=payload,
        created_at=datetime.now(UTC),
    )
    db.add(entry)
    db.commit()
    return entry
