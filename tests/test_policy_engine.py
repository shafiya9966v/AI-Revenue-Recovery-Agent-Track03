"""
Tests targeted at the money-safety-critical gates in app/decide.py, not full
coverage. These are the assertions that matter most for "would you trust it."
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.decide import decide
from app.schemas import DiagnosisResult, RootCause, ActionType


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def _diag(payment_id, root_cause, confidence):
    return DiagnosisResult(
        event_id=f"evt_{payment_id}",
        payment_id=payment_id,
        predicted_root_cause=root_cause,
        confidence=confidence,
        model_version="test",
    )


def test_fraud_never_auto_retried(db):
    diagnosis = _diag("pay_fraud1", RootCause.fraud_suspected, 0.95)
    decision = decide(db, diagnosis, amount=5000)
    assert decision.action == ActionType.escalate_human
    assert decision.requires_human_approval is True


def test_low_confidence_escalates_regardless_of_cause(db):
    diagnosis = _diag("pay_lowconf", RootCause.bank_side_transient, 0.4)
    decision = decide(db, diagnosis, amount=1000)
    assert decision.action == ActionType.escalate_human


def test_high_value_requires_approval(db):
    diagnosis = _diag("pay_highval", RootCause.bank_side_transient, 0.9)
    decision = decide(db, diagnosis, amount=75000)
    assert decision.requires_human_approval is True


def test_max_attempts_enforced(db):
    payment_id = "pay_capcheck"
    # bank_side_transient has max_attempts=3 in the policy table
    for _ in range(3):
        diagnosis = _diag(payment_id, RootCause.bank_side_transient, 0.9)
        decision = decide(db, diagnosis, amount=1000)
        assert decision.action == ActionType.retry_fast

    # 4th attempt must be stopped, not retried again
    diagnosis = _diag(payment_id, RootCause.bank_side_transient, 0.9)
    decision = decide(db, diagnosis, amount=1000)
    assert decision.action == ActionType.escalate_human
    assert "max_attempts" in decision.reason


def test_unrecoverable_never_actioned(db):
    diagnosis = _diag("pay_dead", RootCause.unrecoverable, 0.99)
    decision = decide(db, diagnosis, amount=2000)
    assert decision.action == ActionType.write_off
    assert decision.requires_human_approval is False
