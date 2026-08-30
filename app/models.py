from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, JSON
from datetime import datetime, UTC
from app.database import Base


class EventRecord(Base):
    __tablename__ = "events"
    event_id = Column(String, primary_key=True)
    payment_id = Column(String, index=True)
    merchant_id = Column(String)
    order_id = Column(String)
    amount = Column(Float)
    instrument_type = Column(String)
    error_code = Column(String)
    attempt_number = Column(Integer)
    customer_id = Column(String)
    customer_past_successful_payments = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


class DiagnosisRecord(Base):
    __tablename__ = "diagnoses"
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String, index=True)
    payment_id = Column(String, index=True)
    predicted_root_cause = Column(String)
    confidence = Column(Float)
    model_version = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class DecisionRecord(Base):
    __tablename__ = "decisions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    payment_id = Column(String, index=True)
    root_cause = Column(String)
    confidence = Column(Float)
    action = Column(String)
    attempt_number = Column(Integer)
    requires_human_approval = Column(Boolean)
    reason = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class ActionResultRecord(Base):
    __tablename__ = "action_results"
    id = Column(Integer, primary_key=True, autoincrement=True)
    payment_id = Column(String, index=True)
    action = Column(String)
    executed = Column(Boolean)
    razorpay_response_id = Column(String, nullable=True)
    outcome = Column(String)
    amount_recovered = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class AuditLog(Base):
    __tablename__ = "audit_log"
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String, index=True)
    payment_id = Column(String, index=True)
    stage = Column(String)  # detect | diagnose | decide | act
    payload = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class PaymentAttemptCounter(Base):
    """Tracks attempts-so-far per payment_id to enforce stopping rules / idempotency."""
    __tablename__ = "attempt_counters"
    payment_id = Column(String, primary_key=True)
    attempts_made = Column(Integer, default=0)
    last_action_at = Column(DateTime, nullable=True)
    stopped = Column(Boolean, default=False)
