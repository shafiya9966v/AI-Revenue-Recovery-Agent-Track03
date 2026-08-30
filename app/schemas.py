"""
Shared data contracts for the pipeline: Detect -> Diagnose -> Decide -> Act -> Audit.

Every stage reads/writes these typed models. Nothing malformed can flow through
the pipeline because Pydantic validates at every boundary. This is the concrete
mechanism behind "every money action is bounded and gated."
"""
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime, UTC
from typing import Optional


class InstrumentType(str, Enum):
    card = "card"
    upi = "upi"
    netbanking = "netbanking"
    wallet = "wallet"
    emandate = "emandate"


class RootCause(str, Enum):
    insufficient_funds = "insufficient_funds"
    bank_side_transient = "bank_side_transient"
    card_expired = "card_expired"
    customer_abandoned = "customer_abandoned"
    mandate_failed = "mandate_failed"
    gateway_issue = "gateway_issue"
    fraud_suspected = "fraud_suspected"
    unrecoverable = "unrecoverable"


class ActionType(str, Enum):
    retry_fast = "retry_fast"
    retry_delayed = "retry_delayed"
    notify_customer = "notify_customer"
    request_new_instrument = "request_new_instrument"
    mandate_retry_sequence = "mandate_retry_sequence"
    escalate_human = "escalate_human"
    write_off = "write_off"
    no_action = "no_action"


class PaymentFailedEvent(BaseModel):
    event_id: str
    payment_id: str
    merchant_id: str
    order_id: str
    amount: float = Field(gt=0)
    currency: str = "INR"
    instrument_type: InstrumentType
    error_code: str
    attempt_number: int = Field(ge=1)
    time_since_last_attempt_min: Optional[float] = None
    customer_id: str
    customer_past_successful_payments: int = Field(ge=0, default=0)
    preferred_language: Optional[str] = "hinglish"
    timestamp: datetime

    # ground-truth labels, only present in synthetic/eval data, never used
    # by the model at inference time
    true_root_cause: Optional[RootCause] = None
    true_recoverable: Optional[bool] = None


class DiagnosisResult(BaseModel):
    event_id: str
    payment_id: str
    predicted_root_cause: RootCause
    confidence: float = Field(ge=0.0, le=1.0)
    model_version: str


class Decision(BaseModel):
    payment_id: str
    root_cause: RootCause
    confidence: float
    action: ActionType
    attempt_number: int
    requires_human_approval: bool
    reason: str  # human-readable explanation, always populated


class ActionResult(BaseModel):
    payment_id: str
    action: ActionType
    executed: bool
    razorpay_response_id: Optional[str] = None
    outcome: str  # "recovered" | "failed" | "pending" | "skipped" | "escalated"
    amount_recovered: Optional[float] = None
    timestamp: datetime


class AuditLogEntry(BaseModel):
    event_id: str
    payment_id: str
    stage: str  # "detect" | "diagnose" | "decide" | "act"
    payload: dict
    timestamp: datetime
