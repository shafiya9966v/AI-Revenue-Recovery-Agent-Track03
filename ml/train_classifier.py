"""
Trains the root-cause classifier on the synthetic dataset and reports honest
precision/recall/confusion matrix on a held-out test split.

Model choice: RandomForestClassifier (tree-based).
  - Interpretable via feature_importances_ (explainability requirement).
  - Handles categorical + numeric features without heavy preprocessing.
  - Trains in seconds on a few hundred rows - no need for a heavier
    boosting library (XGBoost/LightGBM) at this data scale; swapping in
    GradientBoostingClassifier or XGBoost later is a one-line change if
    the dataset grows and needs it. This is a deliberate scope decision,
    not a limitation we're hiding.
"""
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "..", "data", "payment_failures.csv")
MODEL_PATH = os.path.join(HERE, "root_cause_model.joblib")
ENCODER_PATH = os.path.join(HERE, "feature_encoders.joblib")

FEATURE_COLS = [
    "amount",
    "instrument_type",
    "error_code",
    "attempt_number",
    "time_since_last_attempt_min",
    "customer_past_successful_payments",
]
TARGET_COL = "true_root_cause"


def load_and_prepare(df: pd.DataFrame):
    df = df.copy()
    df["time_since_last_attempt_min"] = df["time_since_last_attempt_min"].fillna(-1)

    encoders = {}
    for col in ["instrument_type", "error_code"]:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])
        encoders[col] = le

    target_encoder = LabelEncoder()
    df["target_enc"] = target_encoder.fit_transform(df[TARGET_COL])
    encoders["target"] = target_encoder

    feature_cols_final = [
        "amount", "instrument_type_enc", "error_code_enc",
        "attempt_number", "time_since_last_attempt_min",
        "customer_past_successful_payments",
    ]
    X = df[feature_cols_final]
    y = df["target_enc"]
    return X, y, encoders, feature_cols_final


def main():
    df = pd.read_csv(DATA_PATH)
    X, y, encoders, feature_cols = load_and_prepare(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    target_names = encoders["target"].classes_

    print("=" * 60)
    print(f"Held-out test accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        index=target_names, columns=target_names
    ))

    print("\nFeature importances:")
    for name, importance in sorted(
        zip(feature_cols, clf.feature_importances_), key=lambda x: -x[1]
    ):
        print(f"  {name:35s} {importance:.3f}")

    joblib.dump({"model": clf, "feature_cols": feature_cols}, MODEL_PATH)
    joblib.dump(encoders, ENCODER_PATH)
    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved encoders -> {ENCODER_PATH}")


if __name__ == "__main__":
    main()
