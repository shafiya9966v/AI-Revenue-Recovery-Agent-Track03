"""
Diagnosis layer. Wraps the trained RandomForest classifier behind a clean function.
This is the ONLY stage in the pipeline that uses ML - detect/decide/act are all
deterministic. That separation is deliberate (see README "AI judgment" section).
"""
import joblib
import numpy as np
import pandas as pd
from app.schemas import PaymentFailedEvent, DiagnosisResult, RootCause
from app.config import MODEL_PATH, ENCODER_PATH

MODEL_VERSION = "rf_v1_2026-08-24"

_model_bundle = None
_encoders = None


def _load():
    global _model_bundle, _encoders
    if _model_bundle is None:
        _model_bundle = joblib.load(MODEL_PATH)
        _encoders = joblib.load(ENCODER_PATH)
    return _model_bundle, _encoders


def _safe_encode(encoder, value: str) -> int:
    """Falls back to -1 for categories the encoder never saw during training,
    rather than crashing the pipeline on unseen error codes/instruments."""
    if value in encoder.classes_:
        return int(encoder.transform([value])[0])
    return -1


def diagnose(event: PaymentFailedEvent) -> DiagnosisResult:
    bundle, encoders = _load()
    clf = bundle["model"]
    feature_cols = bundle["feature_cols"]

    row = {
        "amount": event.amount,
        "instrument_type_enc": _safe_encode(encoders["instrument_type"], event.instrument_type.value),
        "error_code_enc": _safe_encode(encoders["error_code"], event.error_code),
        "attempt_number": event.attempt_number,
        "time_since_last_attempt_min": event.time_since_last_attempt_min
        if event.time_since_last_attempt_min is not None else -1,
        "customer_past_successful_payments": event.customer_past_successful_payments,
    }
    X = pd.DataFrame([[row[c] for c in feature_cols]], columns=feature_cols)

    proba = clf.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    confidence = float(proba[pred_idx])
    predicted_label = encoders["target"].inverse_transform([pred_idx])[0]

    return DiagnosisResult(
        event_id=event.event_id,
        payment_id=event.payment_id,
        predicted_root_cause=RootCause(predicted_label),
        confidence=round(confidence, 4),
        model_version=MODEL_VERSION,
    )
