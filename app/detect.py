"""
Detection layer. Deliberately simple/deterministic - no ML here.
Its job is just: is this event actually "revenue at risk", and have we
already processed it (idempotency)?
"""
from sqlalchemy.orm import Session
from app.schemas import PaymentFailedEvent
from app.models import EventRecord, PaymentAttemptCounter
from app.audit import log


def is_duplicate(db: Session, event: PaymentFailedEvent) -> bool:
    existing = db.query(EventRecord).filter_by(event_id=event.event_id).first()
    return existing is not None


def detect(db: Session, event: PaymentFailedEvent) -> bool:
    """
    Returns True if this event should proceed to diagnosis, False if it should
    be dropped (duplicate event, or payment already stopped by policy).
    """
    if is_duplicate(db, event):
        log(db, event.event_id, event.payment_id, "detect",
            {"result": "dropped_duplicate_event"})
        return False

    counter = db.query(PaymentAttemptCounter).filter_by(payment_id=event.payment_id).first()
    if counter and counter.stopped:
        log(db, event.event_id, event.payment_id, "detect",
            {"result": "dropped_payment_already_stopped"})
        return False

    # persist the raw event
    record = EventRecord(
        event_id=event.event_id,
        payment_id=event.payment_id,
        merchant_id=event.merchant_id,
        order_id=event.order_id,
        amount=event.amount,
        instrument_type=event.instrument_type.value,
        error_code=event.error_code,
        attempt_number=event.attempt_number,
        customer_id=event.customer_id,
        customer_past_successful_payments=event.customer_past_successful_payments,
    )
    db.add(record)
    db.commit()

    log(db, event.event_id, event.payment_id, "detect",
        {"result": "accepted", "amount": event.amount, "error_code": event.error_code})
    return True
