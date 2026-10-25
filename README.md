# Credit Card Fraud Detection: Production-Ready Inference Pipeline

A production-grade machine learning inference engine and REST API for real-time European cardholder transaction fraud scoring.

Powered by a champion **XGBoost Classifier** (PR-AUC **0.8209**, ROC-AUC **0.9766**), feature-aware preprocessing (`RobustScaler` + `SimpleImputer`), Pydantic validation schemas, and a low-latency FastAPI service.

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

### Run the Test Suite
```bash
pytest -v
```

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

## 3. Real-Time FastAPI Service

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

### Sample cURL Request

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "transaction": {
         "Amount": 149.62,
         "Time": 406.0,
         "V1": -2.312, "V2": 1.952, "V3": -1.610, "V4": 3.998,
         "V5": -0.522, "V6": -1.427, "V7": -2.537, "V8": 1.392,
         "V9": -2.770, "V10": -2.772, "V11": 3.202, "V12": -2.900,
         "V13": -0.595, "V14": -4.289, "V15": 0.390, "V16": -1.141,
         "V17": -2.830, "V18": -0.017, "V19": 0.417, "V20": 0.127,
         "V21": 0.517, "V22": -0.035, "V23": -0.465, "V24": 0.320,
         "V25": 0.045, "V26": 0.178, "V27": 0.261, "V28": -0.143
       },
       "threshold": "cost_optimal"
     }'
```

---

## 4. Pipeline Architecture & Guarantees

```
Incoming Request
       │
       ▼
[ Pydantic Validation ] ──► Rejects NaN, Inf, negative raw amounts, unknown fields
       │
       ▼
[ Feature Alignment ]   ──► Enforces exact 30-feature order (Amount, Time, V1..V28)
       │
       ▼
[ Robust Preprocessing] ──► Median Impute + RobustScaler on Amount & Time
       │
       ▼
[ XGBoost Classifier ]  ──► Probability calculation via serialized Booster
       │
       ▼
[ Decision & Tiering ]  ──► Operational thresholding & LOW/MEDIUM/HIGH/CRITICAL tiers
       │
       ▼
Formatted Response
```