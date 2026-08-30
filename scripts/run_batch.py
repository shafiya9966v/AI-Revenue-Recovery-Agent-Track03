"""
Runs the full Detect -> Diagnose -> Decide -> Act pipeline over a held-out
batch of synthetic events (default: last 60 rows of the dataset, kept separate
from the 400 used to train the classifier's train/test split - i.e. this batch
represents "new" incoming events the deployed system has never touched).

Produces the honest metrics report required by the track's evaluation bar:
recovery rate, rupees recovered, false-positive cost, escalation accuracy,
and an explicit exception list.
"""
import sys, os, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from datetime import datetime, UTC
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

from app.database import Base
from app.schemas import PaymentFailedEvent, InstrumentType
from app.main import run_pipeline

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "..", "data", "payment_failures.csv")
BATCH_DB_PATH = os.path.join(HERE, "..", "batch_run.db")


def load_batch(n=60):
    df = pd.read_csv(DATA_PATH)
    # take the LAST n rows as the "held-out live batch" - never seen during
    # classifier training/eval, since train_classifier.py does its own
    # internal 75/25 split on the full file. This is a clean, separate slice
    # standing in for "new events arriving after deployment."
    return df.tail(n).reset_index(drop=True)


def row_to_event(row) -> PaymentFailedEvent:
    return PaymentFailedEvent(
        event_id=str(uuid.uuid4()),
        payment_id=row["payment_id"],
        merchant_id=row["merchant_id"],
        order_id=row["order_id"],
        amount=row["amount"],
        instrument_type=InstrumentType(row["instrument_type"]),
        error_code=row["error_code"],
        attempt_number=int(row["attempt_number"]),
        time_since_last_attempt_min=(
            None if pd.isna(row["time_since_last_attempt_min"])
            else float(row["time_since_last_attempt_min"])
        ),
        customer_id=row["customer_id"],
        customer_past_successful_payments=int(row["customer_past_successful_payments"]),
        timestamp=datetime.now(UTC),
        true_root_cause=row["true_root_cause"],
        true_recoverable=bool(row["true_recoverable"]),
    )


def main():
    if os.path.exists(BATCH_DB_PATH):
        os.remove(BATCH_DB_PATH)
    engine = create_engine(f"sqlite:///{BATCH_DB_PATH}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    batch = load_batch(60)
    results = []

    correct_diagnoses = 0
    fraud_correctly_escalated = 0
    fraud_total = 0
    false_positive_retries = 0  # retried something that was actually unrecoverable/fraud
    recovered_amount = 0.0
    at_risk_amount = 0.0
    exceptions = []

    from app.diagnose import diagnose as diagnose_fn

    for _, row in batch.iterrows():
        event = row_to_event(row)
        at_risk_amount += event.amount

        diagnosis = diagnose_fn(event)
        predicted = diagnosis.predicted_root_cause.value
        true_cause = event.true_root_cause

        if predicted == true_cause:
            correct_diagnoses += 1

        if true_cause == "fraud_suspected":
            fraud_total += 1

        result = run_pipeline(db, event)

        if result is None:
            exceptions.append({
                "payment_id": event.payment_id, "reason": "dropped_by_detect_layer"
            })
            continue

        if true_cause == "fraud_suspected" and result.outcome == "escalated":
            fraud_correctly_escalated += 1

        if result.outcome == "recovered":
            recovered_amount += result.amount_recovered or 0

        # false positive: system attempted a retry/notify action on a case
        # that was actually unrecoverable or fraud (ground truth), i.e. wasted effort
        if true_cause in ("unrecoverable", "fraud_suspected") and result.executed:
            false_positive_retries += 1
            exceptions.append({
                "payment_id": event.payment_id,
                "reason": f"executed action on true_cause={true_cause} (should not have)",
                "predicted_cause": predicted,
            })

        if result.outcome == "failed":
            exceptions.append({
                "payment_id": event.payment_id,
                "reason": "action executed but did not recover the payment",
                "action": result.action.value,
            })

        results.append({
            "payment_id": event.payment_id,
            "true_cause": true_cause,
            "predicted_cause": predicted,
            "confidence": diagnosis.confidence,
            "action": result.action.value,
            "outcome": result.outcome,
            "amount": event.amount,
        })

    n = len(batch)
    print("=" * 70)
    print(f"BATCH RUN REPORT  ({n} held-out events)")
    print("=" * 70)
    print(f"Root-cause diagnosis accuracy:      {correct_diagnoses}/{n}  ({correct_diagnoses/n*100:.1f}%)")
    if fraud_total:
        print(f"Fraud correctly escalated:          {fraud_correctly_escalated}/{fraud_total}  "
              f"({fraud_correctly_escalated/fraud_total*100:.1f}%)")
    print(f"Total amount at risk:               INR {at_risk_amount:,.2f}")
    print(f"Total amount recovered:             INR {recovered_amount:,.2f}")
    print(f"Recovery rate (of at-risk value):   {recovered_amount/at_risk_amount*100:.1f}%")
    print(f"False-positive actions (cost):      {false_positive_retries} "
          f"(action taken on unrecoverable/fraud ground truth)")
    print(f"Exceptions logged:                  {len(exceptions)}")
    print("=" * 70)

    report_df = pd.DataFrame(results)
    report_path = os.path.join(HERE, "..", "batch_report.csv")
    report_df.to_csv(report_path, index=False)
    print(f"\nFull per-record report -> {report_path}")

    if exceptions:
        print(f"\nException list (first 10 of {len(exceptions)}):")
        for e in exceptions[:10]:
            print(f"  - {e}")

    exc_df = pd.DataFrame(exceptions)
    exc_path = os.path.join(HERE, "..", "batch_exceptions.csv")
    exc_df.to_csv(exc_path, index=False)
    print(f"Exception list -> {exc_path}")


if __name__ == "__main__":
    main()
