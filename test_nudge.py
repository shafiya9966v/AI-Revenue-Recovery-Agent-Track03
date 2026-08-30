"""
Standalone test for the Hinglish nudge generator - confirms whether it's
actually calling the Mistral API or running in template-fallback mode.

Run this directly:
    python test_nudge.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from app.nudge import generate_recovery_nudge, NUDGE_SIMULATION_MODE, MISTRAL_MODEL
from app.schemas import Decision, PaymentFailedEvent, RootCause, ActionType, InstrumentType
from datetime import datetime, UTC

print("=" * 60)
print("NUDGE MODULE STATUS CHECK")
print("=" * 60)
print(f"NUDGE_SIMULATION_MODE : {NUDGE_SIMULATION_MODE}")
print(f"MISTRAL_MODEL         : {MISTRAL_MODEL}")

if NUDGE_SIMULATION_MODE:
    print()
    print("[WARNING] No MISTRAL_API_KEY detected in your .env file.")
    print("   The nudge generator will use FIXED FALLBACK TEMPLATES,")
    print("   not real Mistral API calls.")
else:
    print()
    print("[SUCCESS] MISTRAL_API_KEY detected. Will attempt a REAL API call below.")

print("=" * 60)

# Build fake decision
fake_decision = Decision(
    payment_id="pay_llm_test",
    root_cause=RootCause.card_expired,
    confidence=0.88,
    action=ActionType.request_new_instrument,
    attempt_number=1,
    requires_human_approval=False,
    reason="test call",
)

# Test 1: Hinglish
print("\n--- TEST 1: HINGLISH NUDGE ---")
fake_event_hinglish = PaymentFailedEvent(
    event_id="evt_llm_test_hi",
    payment_id="pay_llm_test",
    merchant_id="merchant_1",
    order_id="order_llm_test",
    amount=1850,
    instrument_type=InstrumentType.card,
    error_code="BAD_REQUEST_ERROR:card_expired",
    attempt_number=1,
    customer_id="cust_1",
    customer_past_successful_payments=4,
    preferred_language="hinglish",
    timestamp=datetime.now(UTC),
)
message_hinglish = generate_recovery_nudge(fake_decision, fake_event_hinglish)
print(f"Hinglish Output:\n  {message_hinglish}\n")

# Test 2: Telugu-English
print("--- TEST 2: TELUGU-ENGLISH NUDGE ---")
fake_event_telugu = PaymentFailedEvent(
    event_id="evt_llm_test_te",
    payment_id="pay_llm_test",
    merchant_id="merchant_1",
    order_id="order_llm_test",
    amount=1850,
    instrument_type=InstrumentType.card,
    error_code="BAD_REQUEST_ERROR:card_expired",
    attempt_number=1,
    customer_id="cust_1",
    customer_past_successful_payments=4,
    preferred_language="telugu_english",
    timestamp=datetime.now(UTC),
)
message_telugu = generate_recovery_nudge(fake_decision, fake_event_telugu)
print(f"Telugu-English Output:\n  {message_telugu}\n")

# Test 3: English
print("--- TEST 3: ENGLISH NUDGE ---")
fake_event_english = PaymentFailedEvent(
    event_id="evt_llm_test_en",
    payment_id="pay_llm_test",
    merchant_id="merchant_1",
    order_id="order_llm_test",
    amount=1850,
    instrument_type=InstrumentType.card,
    error_code="BAD_REQUEST_ERROR:card_expired",
    attempt_number=1,
    customer_id="cust_1",
    customer_past_successful_payments=4,
    preferred_language="english",
    timestamp=datetime.now(UTC),
)
message_english = generate_recovery_nudge(fake_decision, fake_event_english)
print(f"English Output:\n  {message_english}\n")

print("=" * 60)
print("Running variability check for Hinglish...")
outputs = [generate_recovery_nudge(fake_decision, fake_event_hinglish) for _ in range(3)]
all_identical = len(set(outputs)) == 1
for i, o in enumerate(outputs, 1):
    print(f"  Run {i}: {o}")

print()
if all_identical:
    print("-> All 3 outputs are IDENTICAL. This strongly suggests FALLBACK TEMPLATE mode")
    print("  (or MISTRAL_API_KEY is missing/invalid).")
else:
    print("-> Outputs VARY between runs. This confirms REAL Mistral API calls are")
    print("  being made.")
print("=" * 60)