# AI Revenue Recovery Agent
### Track 03 — Payment Degradation → Root Cause → Recovery Action

An agent that detects revenue at risk from failed payments, diagnoses *why*
each payment failed, decides the right recovery intervention under hard
safety gates, executes it against Razorpay's test-mode APIs, and logs every
decision to a queryable audit trail.

---

## 1. Problem Statement

Revenue loss from failed payments rarely happens in one clean step, and most
systems respond to it in one of two bad ways:

- **Ignore it** — the payment fails, nothing happens, the revenue is gone.
- **Blindly retry everything** — expensive in gateway fees, annoying to
  customers, and actively dangerous when the failure is fraud (you do not
  want to keep re-attempting a charge on a fraudulent card).

The piece missing from both approaches is **diagnosis before action**. A
payment that failed because of a temporary bank timeout needs a completely
different response than one that failed because a card expired, or one that
is actually fraud. This project builds that missing diagnosis-and-decision
layer.

---

## 2. Solution Approach

The system is built as a five-stage pipeline, not a single monolithic
"AI agent":

```
DETECT → DIAGNOSE (ML) → DECIDE (policy engine) → ACT (Razorpay) → AUDIT
```

**The central design decision:** only the Diagnose stage uses machine
learning. Detection, policy decisions, and action execution are all plain,
deterministic code. Money-safety guarantees — never exceed a retry cap,
never auto-retry a fraud case, always escalate low-confidence diagnoses —
should not depend on a model being right. They are enforced by code that
cannot be talked out of it.

The one exception, deliberately scoped, is a small LLM-powered customer
messaging feature (see Section 7) — because natural language generation is
exactly the kind of low-stakes, creative task an LLM is suited for, and
nowhere near a financial decision.

---

## 3. Root-Cause Taxonomy

| Root Cause | Recoverable? | Default Action |
|---|---|---|
| insufficient_funds | Yes | Delayed retry |
| bank_side_transient | Yes | Fast retry |
| card_expired | Yes | Request new payment instrument |
| customer_abandoned | Yes | Notify / reminder |
| mandate_failed | Yes | Mandate retry sequence |
| gateway_issue | Yes | Single immediate retry |
| fraud_suspected | No | Always escalate to human, never auto-retried |
| unrecoverable | No | Write off, log only |

Eight causes, chosen to be fine-grained enough to map to distinct actions,
and coarse enough for a classifier to learn reliably from a few hundred
labeled examples. Two of the eight (fraud_suspected, unrecoverable) are
deliberately "dead ends" for automation — this is where the system is
designed to know when to stop, not just how to act.

---

## 4. Methodology

### 4.1 Data

No public dataset exists with labeled "why did this payment fail" data, so
a synthetic generator (data/generate_synthetic_data.py) produces it,
calibrated to realistic Indian digital-payments failure patterns:
transient bank/UPI errors and insufficient funds dominate (about 20% each);
fraud is rare but high-severity (about 5%). Roughly 6% label noise is
injected deliberately so the classifier cannot simply memorize a 1:1
error-code lookup — it has to learn genuine signal from overlapping,
ambiguous cases, the same way real payment data would behave.

### 4.2 Model

A RandomForestClassifier (scikit-learn) predicts root cause from six
features: amount, instrument type, error code, attempt number, time since
last attempt, and customer payment history. Chosen deliberately over a
heavier model (XGBoost, deep learning) because:

- It is interpretable via feature_importances_ — a direct explainability
  requirement.
- It trains in seconds on a dataset this size — no need for a bigger tool
  than the data warrants.
- This is itself a judgment call worth stating: the right-sized model for
  the data you actually have, not the most sophisticated one available.

### 4.3 Policy Engine

A hand-written policy table (app/config.py) maps each root cause to an
action, a max-attempt cap, and a cooldown period. app/decide.py applies
five hard gates, in order, before any action fires:

1. Diagnosis confidence below 0.6 -> escalate to human, do not guess.
2. fraud_suspected -> always escalate, regardless of confidence.
3. unrecoverable -> write off, no action.
4. Payment already at its max-attempts cap -> stop, escalate.
5. Amount above Rs.50,000 -> require human approval before any action executes.

These are asserted directly with automated tests
(tests/test_policy_engine.py), not just described — see Section 9.

---

## 5. Architecture

```
                    EVENT SOURCE
        Razorpay test-mode webhook: payment.failed
                        |
                        v
        DETECT  (app/detect.py) — deterministic
   Deduplicates events, drops already-stopped payments
                        |
                        v
        DIAGNOSE  (app/diagnose.py) — ML
     RandomForest -> root_cause + confidence score
                        |
                        v
        DECIDE  (app/decide.py) — deterministic policy engine
      5 hard gates -> action + human-readable reason string
                        |
                        v
        ACT  (app/act.py) — deterministic execution
   Razorpay test-mode API (or simulation fallback)
   Wrapped in try/except — one gateway failure handled cleanly
        -> NUDGE (app/nudge.py) — the one LLM call, Hinglish messaging
                        |
                        v
        AUDIT  (app/audit.py) — every stage logs here
        Queryable via GET /audit/{payment_id}
```

---

## 6. Directory Structure

```
revenue-recovery-agent/
├── README.md
├── requirements.txt
├── .env.example
├── test_nudge.py                    # standalone LLM integration check
├── data/
│   ├── generate_synthetic_data.py
│   └── payment_failures.csv
├── ml/
│   ├── train_classifier.py
│   ├── root_cause_model.joblib
│   └── feature_encoders.joblib
├── app/
│   ├── main.py            # FastAPI app — webhook, metrics, audit endpoints
│   ├── schemas.py          # Pydantic contracts for every pipeline stage
│   ├── config.py           # policy table + thresholds
│   ├── database.py / models.py
│   ├── detect.py
│   ├── diagnose.py
│   ├── decide.py           # the policy engine — hard gates
│   ├── act.py              # Razorpay execution + nudge trigger
│   ├── nudge.py             # Hinglish LLM messaging — the one LLM call
│   └── audit.py
├── tests/
│   └── test_policy_engine.py
├── scripts/
│   └── run_batch.py
├── dashboard/
│   └── index.html
└── batch_report.csv / batch_exceptions.csv
```

---

## 7. AI Judgment — where AI is used, and where it deliberately is not

| Task | Tool used | Why |
|---|---|---|
| Root-cause diagnosis | ML (RandomForest) | Genuine pattern-recognition problem — no fixed rule can separate 8 overlapping failure classes reliably |
| Retry caps, fraud escalation, approval gates | Deterministic code | Money-safety guarantees must be reproducible and cannot depend on a model being right every time |
| Action execution (API calls) | Deterministic code | Execution is a mechanical step — no judgment needed, only correctness |
| Customer recovery messaging | LLM (Mistral) | Low-stakes, creative natural-language task — exactly what an LLM is suited for, and nowhere near a financial decision |
| Policy orchestration | Plain Python, no agent framework | The decision logic is a lookup table with gates — using a heavy agent framework here would add complexity without adding capability |

The single LLM touchpoint (app/nudge.py) generates short Hinglish recovery
messages for customer-facing actions only (notify_customer,
request_new_instrument, mandate_retry_sequence). It runs in a deterministic
template-fallback mode when no API key is configured, and falls back to the
same templates if a live API call fails — so a language model outage never
breaks the recovery workflow, only slightly degrades the message quality.

---

## 8. Failure Recovery — what broke, and what was done about it

**1. Classifier weak spot, disclosed not hidden.**
The gateway_issue class has the weakest F1-score (about 0.54) among the 8
root causes, because it overlaps heavily with bank_side_transient in
feature space — both often present as generic processing errors. Rather
than hide this, the confidence gate in app/decide.py catches it directly:
any diagnosis below 0.6 confidence is routed to human escalation instead of
being auto-actioned. The weakness in the model does not propagate into a
bad automated decision.

**2. Live API failure, handled gracefully.**
app/act.py wraps every live Razorpay execution call in a try/except. If the
call raises — timeout, malformed response, rate limit — the pipeline does
not crash or leave a payment in an inconsistent state. It logs the
exception to the audit trail (result: exception_handled_gracefully), marks
the attempt as outcome: failed (visible in metrics, not silently dropped),
and the same max-attempts cap still applies to any subsequent retry.

**3. LLM output not matching the required register (caught during testing).**
An early version of the Hinglish nudge prompt produced plain English output
instead of genuine Hindi-English code-switching. This was caught by running
the diagnostic script (test_nudge.py) and comparing outputs across repeated
calls — the fix was a stronger, example-anchored prompt. This is included
here as a real instance of iterating on a failure during development, not a
theoretical failure mode.

---

## 9. Automated Tests

tests/test_policy_engine.py — 5 tests targeted specifically at the
money-safety-critical gates, not full code coverage:

```
test_fraud_never_auto_retried                       PASSED
test_low_confidence_escalates_regardless_of_cause    PASSED
test_high_value_requires_approval                    PASSED
test_max_attempts_enforced                           PASSED
test_unrecoverable_never_actioned                    PASSED
```

Each test calls the real decide() function against a fresh in-memory
database, not a mock — these assertions verify actual pipeline behavior.

---

## 10. Results (held-out batch of 60 synthetic events)

```
Root-cause diagnosis accuracy:      54/60  (90.0%)
Fraud correctly escalated:          7/7    (100.0%)
Total amount at risk:               Rs.528,912.90
Total amount recovered:             Rs.104,942.79
Recovery rate (of at-risk value):   19.8%
False-positive actions (cost):      0
Exceptions logged:                  16
```

Classifier held-out test-split accuracy (separate from the batch run,
ml/train_classifier.py): 83% across all 8 classes, with near-perfect
precision/recall on fraud_suspected and insufficient_funds.

**On synthetic data:** this dataset is generated, not real merchant data —
no such labeled dataset is publicly available. These metrics demonstrate
the pipeline's correctness and the model's ability to learn genuine signal
from realistic, noisy data — not production-grade real-world accuracy,
which would require validation against actual merchant failure logs.

**On the 19.8% recovery rate:** this is a conservative floor, not an
optimistic estimate — it reflects deliberately modest simulated
success-rates per action type (see SIMULATED_SUCCESS_RATE in app/act.py),
chosen to avoid overstating results.

---

## 11. How to Run

```bash
pip install -r requirements.txt --break-system-packages
cp .env.example .env   # optional — blank keys run everything in simulation mode

python data/generate_synthetic_data.py     # generate synthetic dataset
python ml/train_classifier.py               # train + evaluate the classifier
pytest tests/ -v                             # run the safety-gate tests
python scripts/run_batch.py                  # run the full pipeline, get metrics
python test_nudge.py                          # verify the LLM nudge integration
uvicorn app.main:app --reload                  # start the API
```

Then open dashboard/index.html in a browser (with the API running) for the
visual metrics dashboard, or http://localhost:8000/docs for the interactive
Swagger UI to trigger the pipeline live.

Without Razorpay or Mistral API keys in .env, both the Act layer and the
Nudge layer automatically run in deterministic simulation/fallback mode —
the entire pipeline is fully runnable and demoable without any live
credentials.

---

## 12. Tech Stack Summary

| Tool | Purpose |
|---|---|
| FastAPI + Pydantic | Pipeline orchestration, strict schema validation at every stage boundary |
| scikit-learn RandomForestClassifier | Root-cause diagnosis — interpretable, fast, right-sized for the data |
| SQLAlchemy + SQLite | Structured, queryable audit trail |
| Razorpay Python SDK | Test-mode Payment Links API, with simulation fallback |
| Mistral API | The one LLM touchpoint — Hinglish customer messaging, with template fallback |
| pytest | Targeted tests on the money-safety-critical policy gates |
| Chart.js (vanilla HTML) | Lightweight metrics dashboard, no build step |

---

## 13. What's Next (beyond hackathon scope)

- Regional-language expansion beyond Hinglish (Telugu, Tamil, etc. — tested
  informally and the underlying model handles this well already)
- A learned recovery-probability scorer replacing the fixed simulated
  success rates in app/act.py
- NPCI-compliant mandate-retry spacing rules for mandate_failed cases
- A human-approval queue UI for high-value and low-confidence escalations
- Validation against real (anonymized) merchant failure logs, not synthetic
  data
