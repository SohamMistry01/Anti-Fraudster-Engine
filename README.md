# Credit Card Fraud Detection: Production-Ready Inference, Explainability & Anomaly Pipeline

A production-grade machine learning inference engine, explainability suite, and REST API for real-time European cardholder transaction fraud scoring and anomaly triage.

Powered by a champion **XGBoost Classifier** (PR-AUC **0.8209**, ROC-AUC **0.9766**), an unsupervised **Isolation Forest** anomaly baseline (ROC-AUC **0.9422**), **TreeSHAP** local explainability with interactive **Plotly** charts, and a low-latency **FastAPI** service.

---

## 1. Quickstart & Setup

### Activate Environment & Install Dependencies
```bash
# Windows
.\venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
pip install -e .
```

### Run the Complete Test Suite
```bash
pytest -v
```
*(44 unit & integration tests covering scoring, explainability, anomaly detection, and API endpoints)*

---

## 2. Reusable Python Scoring Module

The inference engine provides a singleton `FraudPredictor` and top-level `predict_transaction()` / `predict_batch()` functions:

```python
from src.inference.predictor import predict_transaction

# Sample raw transaction (Amount in USD, Time in seconds, V1..V28)
transaction = {
    "Amount": 149.62,
    "Time": 406.0,
    "V1": -2.312227, "V2": 1.951992, "V3": -1.609851, "V4": 3.997906,
    "V5": -0.522188, "V6": -1.426545, "V7": -2.537387, "V8": 1.391657,
    "V9": -2.770089, "V10": -2.772272, "V11": 3.202033, "V12": -2.899907,
    "V13": -0.595222, "V14": -4.289254, "V15": 0.389724, "V16": -1.140747,
    "V17": -2.830056, "V18": -0.016822, "V19": 0.416956, "V20": 0.126911,
    "V21": 0.517232, "V22": -0.035049, "V23": -0.465211, "V24": 0.320198,
    "V25": 0.044519, "V26": 0.177840, "V27": 0.261145, "V28": -0.143276
}

# Score with default threshold (0.5)
result = predict_transaction(transaction, threshold="default")
print(result)
# Output:
# {
#     "fraud_probability": 0.999709,
#     "is_fraud": True,
#     "risk_category": "CRITICAL",
#     "threshold_used": 0.5,
#     "threshold_name": "default",
#     "model_version": "1.0.0",
#     "model_name": "XGBoost Classifier"
# }
```

### Operational Deployment Modes

| Threshold ($\tau$) | Operational Mode | Focus & Rationale |
| :---: | :--- | :--- |
| **`cost_optimal` ($\tau = 0.182$)** | Minimum Financial Loss | Minimizes chargeback fees ($150 FN vs $3 FP); captures maximum fraud |
| **`default` ($\tau = 0.500$)** | Standard Classification | Balanced baseline threshold |
| **`f2_optimal` ($\tau = 0.552$)** | High Precision / Recall | 90.59% Precision, 81.05% Recall, cuts false alarms to single digits |
| **`f1_optimal` ($\tau = 0.798$)** | Low-Friction Mode | 93.75% Precision; prioritizes minimizing false customer card declines |

---

## 3. Explainability with TreeSHAP & Plotly

```python
from src.explainability.shap_explainer import get_explainer

explainer = get_explainer()
explanation = explainer.explain_transaction(transaction, top_n=5, generate_plot=True)

print("Evidence Summary:", explanation["evidence_summary"])
print("Top Fraud Drivers:", explanation["top_positive_contributors"])
# Interactive Plotly figure accessible via:
# explanation["plot_html"]  # Standalone embeddable HTML snippet
# explanation["plot_dict"]  # JSON dictionary for frontend visual frameworks
```

> [!IMPORTANT]
> **Data Integrity Principle**:
> Features `V1` through `V28` are anonymized principal components derived via PCA. The pipeline strictly articulates statistical facts (e.g. median deviations and directional log-odds shifts) without fabricating fictitious domain narratives (such as unverified merchant categories, card velocities, or foreign IP addresses).

---

## 4. Unsupervised Anomaly Detection with Isolation Forest

Provides an independent outlier signal trained without class labels:

```python
from src.anomaly.detector import get_anomaly_detector

detector = get_anomaly_detector()
anomaly_res = detector.score_transaction(transaction)
print(anomaly_res)
# Output:
# {
#     "anomaly_score": 0.437,           # Normalized [0.0 = Standard, 1.0 = Extreme Outlier]
#     "raw_score": 0.1271,             # Raw decision_function score
#     "is_anomaly": False,             # Binary outlier flag
#     "anomaly_tier": "NORMAL",        # Categorical tier (NORMAL, ELEVATED, OUTLIER)
#     "algorithm": "IsolationForest"
# }
```

---

## 5. Supervised vs. Unsupervised Model Comparison

Run the comparative evaluation on unseen holdout transactions:

```python
from src.anomaly.comparison import compare_models, generate_comparison_report

comparison = compare_models()
print(generate_comparison_report(comparison))
```

### Empirical Leaderboard on Unseen Test Set (56,746 transactions)

| Evaluation Metric | XGBoost (Supervised Champion) | Isolation Forest (Unsupervised Baseline) | Superiority |
| :--- | :---: | :---: | :--- |
| **PR-AUC (Primary)** | **0.8209** | 0.1557 | Supervised (+0.6652) |
| **ROC-AUC** | **0.9766** | 0.9422 | Supervised (+0.0344) |
| **Precision** | **86.52%** | 26.47% | Supervised (+60.05%) |
| **Recall** | **81.05%** | 28.42% | Supervised (+52.63%) |
| **F1-Score** | **0.8370** | 0.2741 | Supervised (+0.5629) |

### Signal Concordance & Overlap Insights:
- **Both Models Agree (28 transactions):** Yields **96.4% Precision** (27 of 28 are ground-truth fraud).
- **Supervised Only (XGBoost):** Flags 50 true frauds missed by Isolation Forest, demonstrating that fraudsters intentionally keep spending subtle ($9.82 median) to evade global outlier detection.
- **Unsupervised Only (Isolation Forest):** Flags 74 rare statistical anomalies (extreme values or unusual multivariate vectors) representing novel behavioral deviations rather than historical fraud templates.

---

## 6. Integrated Interpretable Risk Triage (4 Quadrants)

The `IntegratedRiskAssessor` evaluates both models independently before synthesizing an operational action:

```python
from src.explainability.risk_assessor import get_risk_assessor

assessor = get_risk_assessor()
assessment = assessor.assess_transaction(transaction)
print(assessment["operational_quadrant"])  # e.g. CONFIRMED_HIGH_RISK, SUBTLE_IN_DISTRIBUTION_FRAUD, NOVEL_ANOMALY_OUTLIER, CONFIRMED_LEGITIMATE
print(assessment["recommended_action"])    # e.g. DECLINE_TRANSACTION, STEP_UP_2FA, APPROVE
```

```
                        Anomaly Score (Isolation Forest)
                                 ▲
                     NOVEL       │    CONFIRMED
                     ANOMALY     │    HIGH RISK
                 (Step-Up 2FA)   │  (Auto-Decline)
                                 │
             ────────────────────┼────────────────────► Fraud Probability (XGBoost)
                                 │
                   STANDARD      │     SUBTLE
                  LEGITIMATE     │      FRAUD
                (Auto-Approve)   │  (Challenge/Review)
                                 │
```

---

## 7. Real-Time FastAPI Service

### Start the API Server
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger documentation is available at: **`http://localhost:8000/docs`**

### API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service readiness and model status |
| `GET` | `/model/metadata` | Benchmark metrics, operational thresholds, and top predictive features |
| `POST` | `/predict` | Real-time single transaction scoring |
| `POST` | `/predict/batch` | High-throughput vectorized batch transaction scoring |
| `POST` | `/predict/explain` | TreeSHAP feature attributions and interactive Plotly chart |
| `POST` | `/anomaly/score` | Independent Isolation Forest unsupervised anomaly scoring |
| `POST` | `/evaluate/combined` | Multi-signal decision triage combining supervised, anomaly, and SHAP evidence |
| `GET` | `/anomaly/comparison` | On-demand comparative benchmark between XGBoost and Isolation Forest |