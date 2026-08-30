import os
from dotenv import load_dotenv

load_dotenv()

# --- Razorpay ---
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "")
RAZORPAY_SIMULATION_MODE = not (RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET)

# --- Policy thresholds (the hard gates) ---
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
HIGH_VALUE_APPROVAL_LIMIT_INR = float(os.getenv("HIGH_VALUE_APPROVAL_LIMIT_INR", "50000"))

# --- Retry caps and cooldowns per root cause (minutes) ---
# This table is the single source of truth for the Decision layer's gates.
POLICY_TABLE = {
    "insufficient_funds":  {"action": "retry_delayed",           "max_attempts": 3, "cooldown_min": 240},
    "bank_side_transient": {"action": "retry_fast",               "max_attempts": 3, "cooldown_min": 15},
    "card_expired":        {"action": "request_new_instrument",   "max_attempts": 2, "cooldown_min": 1440},
    "customer_abandoned":  {"action": "notify_customer",          "max_attempts": 2, "cooldown_min": 360},
    "mandate_failed":      {"action": "mandate_retry_sequence",   "max_attempts": 3, "cooldown_min": 1440},
    "gateway_issue":       {"action": "retry_fast",               "max_attempts": 1, "cooldown_min": 0},
    "fraud_suspected":     {"action": "escalate_human",           "max_attempts": 0, "cooldown_min": 0},
    "unrecoverable":       {"action": "no_action",                "max_attempts": 0, "cooldown_min": 0},
}

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./revenue_recovery.db")

MODEL_PATH = os.getenv("MODEL_PATH", "ml/root_cause_model.joblib")
ENCODER_PATH = os.getenv("ENCODER_PATH", "ml/feature_encoders.joblib")
