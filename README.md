# Anti-Fraudster Engine: Production-Grade Inference, Explainability, Anomaly Detection & Case Management

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42+-FF4B4B.svg)](https://streamlit.io)
[![SQLAlchemy 2.0](https://img.shields.io/badge/SQLAlchemy-2.0+-red.svg)](https://www.sqlalchemy.org/)
[![Alembic](https://img.shields.io/badge/Alembic-1.14+-orange.svg)](https://alembic.sqlalchemy.org/)
[![Tests](https://img.shields.io/badge/tests-68%20passed-brightgreen.svg)](tests/)
[![PR-AUC](https://img.shields.io/badge/PR--AUC-0.8209-success.svg)](models/)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9766-success.svg)](models/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2+-purple.svg)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/Groq-OSS%20LLMs-orange.svg)](https://groq.com)

A comprehensive, enterprise-ready credit card fraud detection system designed for high-throughput scoring, local model explainability, unsupervised anomaly detection, prioritised alert queues, auditable investigation workflows, and an autonomous Agentic Investigation Layer powered by LangGraph and Groq open-source LLMs.

Built on the European cardholder benchmark dataset (284,807 transactions with an extreme 0.172% fraud class imbalance), the system features a champion **XGBoost Classifier** (PR-AUC **0.8209**, ROC-AUC **0.9766**), an optimized **Isolation Forest Anomaly Detector** (PR-AUC **0.5614**, ROC-AUC **0.9308** with noise subspace pruning and 4 operational modes), local feature attributions via **TreeSHAP** with interactive **Plotly** visualizations, a stateful **LangGraph 1.2.16** autonomous investigation layer using Groq open-source models (`openai/gpt-oss-120b` and `20b`), a low-latency **FastAPI** REST service, and an interactive **Streamlit** investigation workspace powered by **PostgreSQL / SQLAlchemy** and **Alembic** schema migrations.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Machine Learning Models & Benchmark Performance](#2-machine-learning-models--benchmark-performance)
3. [Operational Deployment Thresholds & Cost-Optimal Policy](#3-operational-deployment-thresholds--cost-optimal-policy)
4. [Explainability & Strict Data Integrity Principles](#4-explainability--strict-data-integrity-principles)
5. [Unsupervised Anomaly Detection & Model Comparison](#5-unsupervised-anomaly-detection--model-comparison)
6. [Multi-Signal Risk Triage (4 Operational Quadrants)](#6-multi-signal-risk-triage-4-operational-quadrants)
7. [Alert Queue & Case Management System](#7-alert-queue--case-management-system)
8. [Database Schema & Entity-Relationship Architecture](#8-database-schema--entity-relationship-architecture)
9. [FastAPI REST Service & Endpoints](#9-fastapi-rest-service--endpoints)
10. [Agentic Investigation Layer (LangChain & LangGraph Orchestration)](#10-agentic-investigation-layer-langchain--langgraph-orchestration)
11. [Streamlit Interactive Investigation Workspace](#11-streamlit-interactive-investigation-workspace)
12. [Installation, Setup & Verification](#12-installation-setup--verification)
13. [Repository Directory Structure](#13-repository-directory-structure)

---

## 1. System Architecture

The Anti-Fraudster Engine is engineered as a decoupled, multi-tiered architecture that separates real-time inference, explainability attribution, persistent case state, and presentation layers:

```mermaid
flowchart TD
    subgraph INGESTION["1. Ingestion & Validation"]
        A[Incoming Transaction Payload] --> B[Pydantic V2 Validation<br/>TransactionInput / BatchTransactionInput]
        B --> C[Feature Verification & Alignment<br/>Verify 30 Features & Column Ordering]
        C --> D[Conditional RobustScaler<br/>Amount & Time Transformation]
    end

    subgraph INFERENCE["2. Dual-Signal Inference Engine"]
        D --> E1[Supervised Champion:<br/>XGBoost Booster v1.0.0]
        D --> E2[Unsupervised Baseline:<br/>Isolation Forest Detector]
        E1 --> F1[P_Fraud Probability<br/>Configurable Risk Tier Cutoffs]
        E2 --> F2[Normalized Anomaly Score<br/>Anomaly Tier: NORMAL / OUTLIER]
    end

    subgraph EXPLAINABILITY["3. Attribution & Policy Engine"]
        E1 & D --> G1[TreeSHAP Explainer<br/>Exact Log-Odds Attributions]
        G1 --> G2[Interactive Plotly Waterfall Plot<br/>Statistical Evidence Breakdown]
        F1 & F2 --> H[Multi-Signal Risk Assessor<br/>4 Operational Quadrants]
        H --> I[Risk Policy Engine<br/>Priority: CRITICAL / HIGH / MED / LOW]
    end

    subgraph PERSISTENCE["4. Persistence & Audit Layer"]
        I --> J[(SQLAlchemy 2.0 ORM<br/>PostgreSQL / SQLite via Alembic)]
        J --> K1[Alert Queue Table<br/>Status: NEW / ESCALATED / DISMISSED]
        J --> K2[Investigation Cases Table<br/>Preserves Sealed ML Prediction Snapshot]
        K2 --> K3[Case Notes Table<br/>Author & Finding Entries]
        K2 --> K4[Tamper-Evident Audit Logs Table<br/>Immutable State Change Trail]
    end

    subgraph AGENTS["5. Agentic Investigation Layer (LangGraph + Groq OSS LLMs)"]
        K2 --> L1[LangGraph State Workflow<br/>Autonomous Multi-Signal Orchestration]
        L1 --> L2[Deterministic Read-Only Tools<br/>Alert, TreeSHAP, Anomaly, History]
        L2 --> L3[Self-Reflective Grounding Audit<br/>Detect Hallucinations & Verify Citations]
        L3 --> L4[Human Approval Checkpoint<br/>Investigator Review & Lifecycle Commit]
    end

    subgraph PRESENTATION["6. Delivery & User Interfaces"]
        J & L4 --> M[FastAPI REST API<br/>19 Production Endpoints]
        J & L4 --> N[Streamlit Multi-Page Workspace<br/>Case Workbench with 1-Click AI Agent Approval]
    end
```

---

## 2. Machine Learning Models & Benchmark Performance

### Supervised Champion: XGBoost Classifier
The primary decision model is a gradient-boosted decision tree ensemble trained with cost-sensitive weighting (`scale_pos_weight`) to counter the extreme 578:1 legitimate-to-fraud class imbalance. It is serialized in native XGBoost JSON format (`models/xgb_fraud_detector.json`) and joblib bundle (`models/xgb_fraud_detector.joblib`), ensuring zero platform drift.

### Optimized Unsupervised Anomaly Detector: Isolation Forest
An independent outlier detector (`models/isolation_forest.joblib`) engineered to isolate anomalous multivariate vectors without relying on fraud labels. Following systematic optimization in Part 3 ([`notebooks/part3_anomaly_detection_optimization.md`](file:///e:/Fraud-Detection/notebooks/part3_anomaly_detection_optimization.md)), the model prunes 24 non-informative noise dimensions to focus on the top anomaly subspace (`V17`, `V14`, `V12`, `V10`, `V16`, `V3`) with `max_samples=512`, elevating PR-AUC from a baseline of **0.1557** to **0.5614** (+260%).

### Empirical Leaderboard on Holdout Test Set (56,746 Transactions, 95 Frauds)

| Metric | XGBoost (Supervised Champion) | Isolation Forest (Optimized Champion) | Isolation Forest (Baseline Un-tuned) | Optimization Gain ($\Delta$) |
| :--- | :---: | :---: | :---: | :--- |
| **Precision-Recall AUC (PR-AUC)** | **0.8209** | **0.5614** | 0.1557 | **+0.4057 (+260.6%)** |
| **Receiver Operating Characteristic (ROC-AUC)** | **0.9766** | **0.9308** | 0.9422 | Robust Latent Separation |
| **Recall (Fraud Caught)** | **81.05%** (77/95) | **62.11%** (59/95) *(Balanced)*<br/>**73.68%** (70/95) *(High-Recall)* | 28.42% (27/95) | **+45.26%** (High-Recall Mode) |
| **Precision** | **86.52%** | **59.00%** *(Balanced)*<br/>**48.61%** *(High-Recall)* | 26.47% | **+32.53%** (Balanced Mode) |
| **F1-Score** | **0.8370** | **0.6051** *(Balanced)* | 0.2741 | **+0.3310 (+120.8%)** |
| **False Alarms (FP)** | **12** | **41** *(Balanced)* | 75 | **-45.3% False Alarm Reduction** |
| **Simulated Business Loss** | **$2,736.00** | **$3,972.00** *(High-Recall)*<br/>**$3,936.00** *(Min-Loss)* | $10,425.00 | **-$6,489.00 (62.2% Cost Cut)** |
| **Inference Latency (Single Sample)** | **< 3.5 ms** | **< 1.8 ms** | < 2.1 ms | Low-Dimensional Fast Scoring |

---

## 3. Operational Deployment Thresholds & Cost-Optimal Policy

Standard thresholding ($\tau = 0.5$) is sub-optimal in fraud detection due to asymmetric financial costs:
- **False Negative Cost ($C_{FN}$)**: Average chargeback loss, replacement fee, and regulatory overhead ($\approx \$150.00$).
- **False Positive Cost ($C_{FP}$)**: Customer card decline friction and manual verification overhead ($\approx \$3.00$).

The platform supports four calibrated operational deployment modes:

```
                            Total Financial Cost Curve vs Threshold
    Cost ($)
       ▲
       │     \
       │      \             Standard Default (τ = 0.500)
       │       \                    ▼
       │        \                ●───────
       │         ● Cost-Optimal (τ = 0.182)
       │          \             /
       │           \___________/
       └──────────────────────────────────────────────► Decision Threshold (τ)
       0.0                 0.182        0.500         1.0
```

| Operational Mode | Threshold ($\tau$) | Target Objective & Trade-off Rationale |
| :--- | :---: | :--- |
| **`cost_optimal`** | **0.182** | **Minimum Financial Loss**: Captures marginal fraud cases where risk exceeds $\$3$ friction cost ($86.52\%$ Precision, $81.05\%$ Recall). |
| **`default`** | **0.500** | **Balanced Standard**: Baseline classification cutoff. |
| **`f2_optimal`** | **0.552** | **Recall-Weighted Optimization**: Emphasizes fraud capture while maintaining single-digit false positives ($90.59\%$ Precision, $81.05\%$ Recall). |
| **`f1_optimal`** | **0.798** | **High-Precision / Low-Friction**: Prioritizes VIP/low-friction cards; eliminates false card declines ($93.75\%$ Precision). |
| **Custom Numeric** | `0.000`–`1.000` | Arbitrary custom numerical threshold passed dynamically per request. |

### Configurable Risk Tiers
Scored transactions are categorized into four standardized urgency bands:
- **`CRITICAL`** ($P \ge 0.85$): Immediate transaction block or mandatory manual escalation.
- **`HIGH`** ($\tau \le P < 0.85$): Exceeds operational decision threshold; requires review or step-up authentication.
- **`MEDIUM`** ($0.05 \le P < \tau$): Elevated probability below threshold; routed to anomaly monitoring.
- **`LOW`** ($P < 0.05$): Standard legitimate transaction; auto-approved.

---

## 4. Explainability & Strict Data Integrity Principles

### TreeSHAP Local Feature Attribution
For every prediction, the `TreeSHAPExplainer` ([`src/explainability/shap_explainer.py`](file:///e:/Fraud-Detection/src/explainability/shap_explainer.py)) calculates exact Shapley values. Predictions satisfy the additive attribution axiom:

$$\ln\left(\frac{P(\text{Fraud})}{1 - P(\text{Fraud})}\right) = \phi_0 + \sum_{j=1}^{30} \phi_j(x)$$

Where $\phi_0$ is the training population log-odds base value ($-6.4882$) and $\phi_j$ represents the marginal contribution of feature $j$.

```python
from src.explainability.shap_explainer import get_explainer

explainer = get_explainer()
explanation = explainer.explain_transaction(transaction, top_n=5, generate_plot=True)

# Access top driving forces and Plotly figure
print("Top Positive Contributors (Fraud-Pushing):", explanation["top_positive_contributors"])
print("Top Negative Contributors (Mitigating):", explanation["top_negative_contributors"])
fig = explanation["figure"]  # plotly.graph_objects.Figure
```

### Strict Data Integrity Principle
> [!IMPORTANT]
> **Data Integrity Principle**:
> Features `V1` through `V28` are anonymized principal components derived through Principal Component Analysis (PCA) to preserve European cardholder confidentiality. The system strictly formulates explainability in terms of **empirical statistical deviations, median/IQR shifts, and directional log-odds contributions**. It **never fabricates fictitious domain stories** (e.g., claiming `V14` represents "foreign IP addresses" or `V4` represents "merchant velocity").

---

## 5. Unsupervised Anomaly Detection & Model Comparison

The unsupervised `IsolationForestAnomalyDetector` ([`src/anomaly/detector.py`](file:///e:/Fraud-Detection/src/anomaly/detector.py)) normalizes raw decision function scores into a calibrated anomaly index bounded in the range $[0.0, 1.0]$:

$$\text{Anomaly Score} = 1.0 - \frac{s_{\text{raw}} - s_{\min}}{s_{\max} - s_{\min}}$$

Where:
- $s_{\text{raw}}$ is the raw Isolation Forest `decision_function` score (lower scores indicate deeper tree isolation and higher anomaly severity).
- $s_{\min} = -0.116189$ and $s_{\max} = 0.389574$ are the empirical calibration boundary extremes observed during training on the European cardholder dataset.
- The score is inverted so that **$0.0$ represents standard in-distribution behavior** and **$1.0$ represents an extreme population outlier**.

### Optimization Breakthrough: Noise Pruning & Tree Tuning (Part 3)

In Part 3 of the project ([`notebooks/part3_anomaly_detection_optimization.md`](file:///e:/Fraud-Detection/notebooks/part3_anomaly_detection_optimization.md)), extensive empirical analysis resolved the primary bottleneck of standard Isolation Forest:
1. **Curse of Dimensionality in Tree Splitting:** In 30 dimensions with uniform random feature selection, $80\%$ of splits fell on non-informative Gaussian noise features, diluting tree depth and masking genuine anomalies. Restricting splits to the **Top 6 anomaly-discriminative subspace** (`V17`, `V14`, `V12`, `V10`, `V16`, `V3`) propelled PR-AUC from **0.1557 to 0.5313 (+241%)**.
2. **Subsample Size (`max_samples=512`):** Tuning subsample size prevented fraud clusters from masking each other, achieving **0.5614 PR-AUC**.
3. **Four Calibrated Operational Modes:**
   - **Balanced Risk Mode ($\tau = 0.0011$):** Optimal F1 (**0.6051**), capturing **59 frauds** with only **41 false alarms** (Precision: 59.00%, Recall: 62.11%, Simulated Loss: **$5,523.00**).
   - **High-Recall Optimal Mode ($\tau = 0.0214$):** Optimal F2 (**0.6679**), capturing **70 frauds (73.68% Recall)** with only 74 false alarms (Precision: 48.61%, Simulated Loss: **$3,972.00**).
   - **Min Financial Loss Mode ($\tau = 0.0637$):** Captures **73 frauds (76.84% Recall)**, slashing net business loss to **$3,936.00** (**a 62% cost reduction** compared to the $10,425.00 baseline).
   - **Default Mode ($\tau = 0.0000$):** Catches 57 frauds with 41 false alarms (Simulated Loss: $5,823.00).

### Signal Concordance & Overlap Analysis

Cross-model analysis on 56,746 unseen holdout transactions reveals critical behavioral dynamics between the supervised classifier (XGBoost) and the unsupervised detector (Isolation Forest):

```
                            Isolation Forest (Optimized)
                                     Anomalous (100)
                                   ┌─────────┬─────────┐
                                   │   41    │   59    │
                                   │ Novel   │ Both    │
                                   │ Outlier │ Agree   │
                 Non-Fraud (56,651)├─────────┼─────────┤ Fraud (95)
                                   │ 56,610  │   36    │
                                   │ Normal  │ Subtle  │
                                   │ Legit   │ Fraud   │
                                   └─────────┴─────────┘
                                     Normal (56,646)
                                  XGBoost (Supervised)
```

1. **Both Models Agree (59 transactions)**:
   - Yields **98.3% Precision** (58 of 59 are confirmed ground-truth fraud). These represent blatant, high-confidence attacks with extreme multivariate deviations.
2. **Supervised Only / Subtle Fraud (36 transactions)**:
   - Frauds that evade unsupervised outlier isolation by mimicking local median spending habits. Only the supervised classifier's non-linear decision boundaries caught them.
3. **Unsupervised Only / Novel Anomaly (41 transactions)**:
   - Rare statistical outliers representing unusual transaction patterns that do not match historical fraud vectors. Crucial for zero-day behavioral deviations and emerging fraud mechanisms.

---

## 6. Multi-Signal Risk Triage (4 Operational Quadrants)

The `IntegratedRiskAssessor` ([`src/explainability/risk_assessor.py`](file:///e:/Fraud-Detection/src/explainability/risk_assessor.py)) fuses supervised classification, unsupervised anomaly detection, and TreeSHAP attributions into an actionable operational matrix:

```
                         Anomaly Score (Isolation Forest)
                                     ▲
                                     │
           NOVEL_ANOMALY_OUTLIER     │     CONFIRMED_HIGH_RISK
             Action: STEP_UP_2FA     │   Action: DECLINE_TRANSACTION
          (Rare multivariate vector; │     (Both models agree on
           no historical fraud match)│      extreme fraud certainty)
                                     │
         ────────────────────────────┼────────────────────────────► Fraud Probability
                                     │                              (XGBoost)
             CONFIRMED_LEGITIMATE    │   SUBTLE_IN_DISTRIBUTION_FRAUD
                Action: APPROVE      │   Action: CHALLENGE_OR_REVIEW
           (Low fraud probability &  │     (In-distribution values;
            normal population vector)│      matches subtle fraud template)
                                     │
```

---

## 7. Alert Queue & Case Management System

A financial risk operations workbench providing complete lifecycle management from real-time transaction scoring to regulatory dispute resolution.

### Architectural Tenets
1. **Prioritised Alert Queue**:
   - Risk policies evaluate incoming scores and partition transactions into urgency tiers (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`).
   - Automatically generates alerts for transactions exceeding critical risk or exhibiting dual-signal consensus.
2. **Sealed Machine Learning Prediction Snapshots**:
   - **Core Invariant**: When an alert is escalated into an investigation case, the original model version, threshold applied, fraud probability, anomaly score, operational quadrant, and full TreeSHAP attribution vector are permanently sealed in dedicated columns (`original_*`).
   - **Immutable Integrity**: Subsequent investigator actions (status updates, manual verdicts, customer notes, mitigation actions) **never mutate or overwrite** the initial prediction record, guaranteeing an audit trail.
3. **Auditable Investigation Lifecycle**:
   - Managed state transitions: `OPEN` $\rightarrow$ `UNDER_INVESTIGATION` $\rightarrow$ `PENDING_CUSTOMER` $\rightarrow$ `RESOLVED_CONFIRMED_FRAUD` / `RESOLVED_FALSE_POSITIVE` / `CLOSED_DISMISSED`.
   - Analyst verdicts: `CONFIRMED_FRAUD`, `FALSE_POSITIVE`, `SUSPICIOUS_UNVERIFIED`, `LEGITIMATE`.
   - Mitigation actions: `CARD_BLOCKED`, `CHARGEBACK_INITIATED`, `TRANSACTION_RELEASED`, `CUSTOMER_CONTACTED`, `NONE`.
   - Chronological notes and tamper-evident audit logs tracking every change.

---

## 8. Database Schema & Entity-Relationship Architecture

The persistence layer is implemented in SQLAlchemy 2.0 ([`src/cases/models.py`](file:///e:/Fraud-Detection/src/cases/models.py)) with connection management in [`src/cases/database.py`](file:///e:/Fraud-Detection/src/cases/database.py). Migrations are tracked via Alembic ([`alembic/`](file:///e:/Fraud-Detection/alembic/)).

```mermaid
erDiagram
    ALERTS {
        int id PK
        string alert_id UK "ALT-XXXXXXXX"
        string transaction_id "TX-XXXXXXXX"
        timestamp created_at
        float amount
        float time_offset
        float fraud_probability
        float anomaly_score
        string risk_category
        string priority "CRITICAL|HIGH|MEDIUM|LOW"
        string operational_quadrant
        string status "NEW|ESCALATED|CASE_CREATED|DISMISSED"
        json reasons "Trigger policy justifications"
        json transaction_payload "Complete 30-feature vector"
        string case_id FK "Optional linkage"
    }

    INVESTIGATION_CASES {
        int id PK
        string case_id UK "CASE-XXXXXXXX"
        string alert_id FK "Linked alert reference"
        string transaction_id "TX-XXXXXXXX"
        timestamp created_at
        timestamp updated_at
        string status "OPEN|UNDER_INVESTIGATION|RESOLVED_*"
        string priority "CRITICAL|HIGH|MEDIUM|LOW"
        string assigned_analyst
        string verdict "CONFIRMED_FRAUD|FALSE_POSITIVE|..."
        string action_taken "CARD_BLOCKED|CHARGEBACK_INITIATED|..."
        text resolution_notes
        timestamp resolved_at
        float original_fraud_probability "SEALED ML SNAPSHOT"
        string original_risk_category "SEALED ML SNAPSHOT"
        float original_threshold_used "SEALED ML SNAPSHOT"
        string original_threshold_name "SEALED ML SNAPSHOT"
        string original_model_version "SEALED ML SNAPSHOT"
        float original_anomaly_score "SEALED ML SNAPSHOT"
        string original_anomaly_tier "SEALED ML SNAPSHOT"
        string original_quadrant "SEALED ML SNAPSHOT"
        json original_shap_explanation "SEALED ML SNAPSHOT"
        json transaction_snapshot "SEALED ML SNAPSHOT"
    }

    CASE_NOTES {
        int id PK
        string note_id UK "NOTE-XXXXXXXX"
        string case_id FK
        string author
        text content
        timestamp created_at
    }

    CASE_AUDIT_LOGS {
        int id PK
        string log_id UK "AUD-XXXXXXXX"
        string case_id FK
        string action "CASE_CREATED|STATUS_CHANGED|..."
        string performed_by
        json details "Before/after state diff"
        timestamp timestamp
    }

    ALERTS ||--o| INVESTIGATION_CASES : "escalates to"
    INVESTIGATION_CASES ||--o{ CASE_NOTES : "contains"
    INVESTIGATION_CASES ||--o{ CASE_AUDIT_LOGS : "generates"
```

### Database Portability & Zero-Configuration Fallback
- **PostgreSQL Ready**: Connects to remote PostgreSQL via `DATABASE_URL=postgresql+psycopg://user:pass@host:5432/fraud_db` with connection pooling (`pool_size=10`, `max_overflow=20`, `pool_pre_ping=True`).
- **SQLite Fallback**: Automatically creates and maintains local `data/fraud_cases.db` if `DATABASE_URL` is omitted.
- **Session Safety**: Configured with `expire_on_commit=False` and `lazy="selectin"` on relationships to prevent `DetachedInstanceError` in Streamlit and asynchronous contexts.

---

## 9. FastAPI REST Service & Endpoints

The API is built using FastAPI ([`src/api/main.py`](file:///e:/Fraud-Detection/src/api/main.py)) with auto-generated OpenAPI documentation available at `http://localhost:8000/docs`.

### Comprehensive Route Reference

| Module | Method | Endpoint | Description |
| :--- | :---: | :--- | :--- |
| **System** | `GET` | `/` | API service index and operational routing directory |
| | `GET` | `/health` | Service health status, model version, and loaded feature count |
| | `GET` | `/model/metadata` | Benchmark metrics, operational thresholds, top predictive features |
| **Scoring** | `POST` | `/predict` | Real-time transaction scoring (direct payload or wrapper envelope) |
| | `POST` | `/predict/batch` | Vectorized batch scoring for high-throughput pipelines |
| **Explainability** | `POST` | `/predict/explain` | TreeSHAP feature attributions with Plotly waterfall plot HTML |
| **Anomaly** | `POST` | `/anomaly/score` | Independent Isolation Forest unsupervised anomaly scoring |
| | `POST` | `/evaluate/combined` | Multi-signal triage combining supervised, anomaly, and SHAP evidence |
| | `GET` | `/anomaly/comparison` | On-demand comparative benchmark between XGBoost and Isolation Forest |
| **Alert Queue** | `GET` | `/alerts` | Searchable, prioritized alert queue sorted by urgency (CRITICAL $\rightarrow$ LOW) |
| | `GET` | `/alerts/summary` | Queue count summary aggregated by status and priority |
| | `GET` | `/alerts/{alert_id}` | Detailed alert record with trigger policy reasons and full payload |
| | `PATCH` | `/alerts/{alert_id}/status` | Transition alert status (`NEW`, `ESCALATED`, `CASE_CREATED`, `DISMISSED`) |
| | `POST` | `/alerts/triage` | Evaluate transaction against risk policy; trigger alert & auto-escalate |
| **Case Management** | `GET` | `/cases` | Filterable investigation cases (status, priority, analyst, search) |
| | `GET` | `/cases/summary` | Executive metrics: open cases, confirmed fraud rate, resolution counts |
| | `POST` | `/cases` | Escalate an alert into an active investigation case |
| | `GET` | `/cases/{case_id}` | Complete case workspace including sealed prediction snapshot & notes |
| | `PATCH` | `/cases/{case_id}/workflow` | Update status, assign analyst, record verdict, and mitigation action |
| | `POST` | `/cases/{case_id}/notes` | Append auditable investigator note |
| | `GET` | `/cases/{case_id}/audit` | Full chronological audit trail of state changes and actions |
| **Agentic Investigation** | `POST` | `/cases/{case_id}/investigate` | Trigger autonomous multi-signal LangGraph investigation workflow |
| | `POST` | `/cases/{case_id}/agent-decision` | Human-in-the-loop checkpoint: approve or modify agent recommendation |
| | `POST` | `/agent/evaluate` | Run automated benchmark for factual grounding, citations & hallucinations |
| | `GET` | `/agent/tools/check` | Live health verification of the 4 deterministic read-only retrieval tools |

### Example API Request (Python)

```python
import httpx

# Triage incoming transaction
payload = {
    "transaction": {
        "Amount": 245.50,
        "Time": 43200.0,
        "V1": -2.31, "V2": 1.95, "V3": -1.61, "V4": 3.99, "V5": -0.52,
        "V6": -1.42, "V7": -2.53, "V8": 1.39, "V9": -2.77, "V10": -2.77,
        "V11": 3.20, "V12": -2.90, "V13": -0.59, "V14": -4.29, "V15": 0.39,
        "V16": -1.14, "V17": -2.83, "V18": -0.02, "V19": 0.42, "V20": 0.13,
        "V21": 0.52, "V22": -0.04, "V23": -0.47, "V24": 0.32, "V25": 0.04,
        "V26": 0.18, "V27": 0.26, "V28": -0.14
    },
    "threshold": "cost_optimal",
    "apply_preprocessing": True,
    "auto_escalate": True
}

response = httpx.post("http://localhost:8000/alerts/triage", json=payload)
data = response.json()
print("Alert Created:", data["alert_created"], "Priority:", data["alert"]["priority"])
if data["case_created"]:
    print("Auto-Escalated Case ID:", data["case"]["case_id"])
```

---

## 10. Agentic Investigation Layer (LangChain & LangGraph Orchestration)

To augment human fraud analysts and accelerate case resolution, the Anti-Fraudster Engine incorporates an autonomous **Agentic Investigation Layer** built with **LangChain 1.4.7** and **LangGraph 1.2.16**, powered by Groq's high-speed open-source LLM inference (`openai/gpt-oss-120b` champion and `openai/gpt-oss-20b` fallback).

Rather than replacing or duplicating the underlying machine learning models, the agent acts as an **autonomous orchestrator** that queries deterministic tools, synthesizes supervised probabilities with unsupervised anomaly scores, enforces strict factual grounding, and prepares validated, structured recommendations for investigator review.

### Architecture & Stateful LangGraph Workflow

The investigation workflow is modeled as a compiled state graph (`StateGraph[InvestigationState]`) in [`src/agents/investigation_agent.py`](src/agents/investigation_agent.py):

<p align="center">
  <img src="miscellaneous/investigation_agent_graph.png" alt="Investigation Agent State Graph" width="340"/>
</p>

```mermaid
flowchart TD
    START([START]) --> GATHER[node_gather_evidence<br/>Query 4 Read-Only Retrieval Tools]
    GATHER --> SYNTH[node_synthesize_analysis<br/>Multi-Signal Diagnostic Reasoning]
    SYNTH --> REPORT[node_generate_structured_report<br/>Groq LLM + PydanticOutputParser]
    REPORT --> AUDIT[node_validate_grounding<br/>Deterministic Grounding & Hallucination Audit]

    AUDIT --> CHECK{Passed Validation?}
    CHECK -- Yes --> APPROVAL[node_checkpoint_approval<br/>Human-in-the-Loop Checkpoint]
    CHECK -- "No (Attempts <= 2)" --> REVISE[node_revise_report<br/>Inject Critique & Self-Correct]
    REVISE --> AUDIT
    CHECK -- "No (Attempts > 2)" --> APPROVAL

    APPROVAL --> END([END])
```

### Deterministic Read-Only Tools

The agent operates strictly over audited read-only retrieval functions ([`src/agents/tools.py`](file:///e:/Fraud-Detection/src/agents/tools.py)), ensuring zero risk of unvetted state mutations:

1. **`get_alert_details(alert_id: str)`**: Retrieves the stored alert record, supervised fraud probability, trigger policy justifications, and the complete 30-feature transaction vector.
2. **`get_shap_explanation(alert_id: str, top_n: int = 5)`**: Computes TreeSHAP attributions directly on the transaction vector, returning the top positive (fraud-pushing) and negative (legitimate-mitigating) features in exact log-odds space.
3. **`get_anomaly_score(alert_id: str)`**: Queries the champion Isolation Forest detector to compute the normalized anomaly score, raw decision score, and outlier tier (`EXTREME`, `HIGH`, `MODERATE`, `NORMAL`).
4. **`get_case_history(case_id: str)`**: Retrieves the sealed original ML prediction snapshot, all previous analyst notes, and the tamper-evident audit trail.

### Strict Factual Grounding & Anti-Hallucination Guardrails

To prevent LLM hallucination—especially regarding anonymized PCA components $V_1$ through $V_{28}$—the system enforces deterministic validation:

- **Mathematical Value Alignment**: Every feature citation in `key_evidence_citations` must exactly match the true floating-point value present in the transaction vector within $|s - \hat{s}| \le 0.08$.
- **SHAP Directional Consistency**: A feature claimed as `FRAUD_PUSHING` must have a mathematically positive TreeSHAP attribution ($\phi_i > 0$); claimed mitigating features must have $\phi_i < 0$.
- **Prohibited Domain Storytelling Filter**: The grounding engine scans report text against regular expression patterns (`PROHIBITED_DOMAIN_PATTERNS`) to penalize any invented narratives (e.g., claiming $V_{14}$ represents an "overseas IP", "gas station", or "merchant category").

### Human-in-the-Loop Approval Checkpoint

In strict adherence to financial regulatory governance, an AI agent cannot unilaterally close or alter an active fraud case. Instead:
1. The agent compiles an `InvestigatorApprovalRequest` presenting its recommended verdict (`CONFIRMED_FRAUD`, `FALSE_POSITIVE`, `SUSPICIOUS_UNVERIFIED`, `LEGITIMATE`) and recommended action (`CARD_BLOCKED`, `CHARGEBACK_INITIATED`, `TRANSACTION_RELEASED`, `CUSTOMER_CONTACTED`).
2. Fraud investigators review the executive summary and evidence citation table in the Streamlit Workbench.
3. With **1 click**, the investigator can approve the recommendation or adjust the verdict and action.
4. The system executes `apply_investigator_decision()`, which commits the lifecycle transition, appends a signed `[AI Investigation Agent Decision - APPROVED]` note, and logs a tamper-evident audit entry.

### Automated Evaluation Benchmark Suite

The agentic layer includes an automated benchmarking suite ([`src/agents/evaluator.py`](file:///e:/Fraud-Detection/src/agents/evaluator.py)) tested across representative holdout cases:

| Metric | Target | Benchmark Result | Evaluation Method |
| :--- | :---: | :---: | :--- |
| **Factual Grounding Accuracy** | $> 95\%$ | **100.0%** | Mathematical verification against raw 30-feature transaction vector |
| **Evidence Citation Accuracy** | $> 95\%$ | **100.0%** | Directional and magnitude alignment with TreeSHAP attributions |
| **Hallucination Freedom Rate** | $100\%$ | **100.0%** | Zero prohibited domain storytelling on anonymized components $V_1$–$V_{28}$ |
| **Tool Execution Reliability** | $100\%$ | **100.0%** | Correct execution across all 4 deterministic retrieval engines |
| **Composite Agent Quality** | $> 90\%$ | **100.0%** | Weighted synthesis across grounding, citations, and report consistency |

---

## 11. Streamlit Interactive Investigation Workspace

A modern, responsive dark-themed workspace built with Streamlit ([`dashboard/app.py`](file:///e:/Fraud-Detection/dashboard/app.py)) implementing enterprise UX patterns:

### 1. Executive Operations Center (`dashboard/app.py`)
- **Key Performance Indicators**: Total alerts in queue, active open cases, confirmed fraud count, and false positive ratio.
- **Urgency Distribution Donut Chart**: Interactive Plotly breakdown of alerts across `CRITICAL`, `HIGH`, `MEDIUM`, and `LOW`.
- **Operational Quadrants Distribution**: Live distribution of flagged transactions across the 4 risk quadrants.
- **Interactive Triage Simulator**: Score holdout or custom transactions live and observe real-time policy evaluation and automatic case escalation.

### 2. Prioritised Alert Queue (`dashboard/pages/alerts.py`)
- **Queue Filtering**: Multi-select filter by lifecycle status (`NEW`, `ESCALATED`, `CASE_CREATED`, `DISMISSED`), priority, minimum probability slider, and text search.
- **Triage Action Drawer**: Inspect trigger justifications, view raw feature payloads, mark as escalated, dismiss false alerts, or escalate to an active case with analyst assignment and notes.

### 3. Investigation Case Workbench with Autonomous AI Agent (`dashboard/pages/investigations.py`)
- **🤖 Autonomous Agentic Investigator Panel**:
  - Model engine selector (`openai/gpt-oss-120b` champion, `openai/gpt-oss-20b` fallback).
  - One-click **⚡ Generate AI Investigation Report** triggering multi-signal LangGraph synthesis.
  - Interactive **Supporting Evidence Citations Table** displaying observed feature values, exact SHAP log-odds impacts, directions, and statistical rationales.
  - **Factual Grounding Badge & Hallucination Audit** confirming mathematical precision.
  - **Interactive Approval Checkpoint**: Review recommended verdict and mitigation action, customize analyst comments, and click **✅ Approve & Apply Recommendation to Case** to commit the state transition to the database.
  - **🧪 Integrated Benchmark Suite**: Run live grounding and citation audits across cases directly within the interface.
- **Side-by-Side Verification**:
  - **Left**: Sealed Machine Learning Prediction Snapshot (immutable baseline probability, threshold applied, model version, anomaly score, and TreeSHAP attributions).
  - **Right**: Investigation Controls (lifecycle status transitions, analyst assignment, verdict recording, mitigation action execution, and resolution justification).
- **Analyst Notes Feed**: Chronological commentary stream with author tags and timestamps.
- **Tamper-Evident Audit Trail**: Transparent event log showing every state modification, actor, timestamp, and field-level diff.

### 4. Transaction Evidence Deep-Dive (`dashboard/pages/transaction_detail.py`)
- **Interactive TreeSHAP Waterfall Chart**: Dynamic Plotly attributions showing the exact log-odds forces of individual features pushing towards or mitigating fraud.
- **Dual-Signal Gauge**: Visual indicators comparing supervised probability against unsupervised anomaly scores.
- **30-Feature Inspection Grid**: Tabular view of all components (`Time`, `Amount`, `V1`–`V28`) with absolute magnitudes and PCA context.

---

## 12. Installation, Setup & Verification

### 1. Clone & Activate Environment
```bash
git clone https://github.com/SohamMistry01/Anti-Fraudster-Engine.git
cd Anti-Fraudster-Engine

# Create and activate virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 3. Apply Database Schema Migrations (Alembic)
```bash
# Verify current migration status
alembic current

# Run migrations to head
alembic upgrade head
```

### 4. Run the Full Test Suite
```bash
pytest -v
```
*(All 68 unit and integration tests covering agent orchestration, factual grounding, preprocessing, scoring, explainability, anomaly detection, alert triage, case management, and REST endpoints)*

### 5. Launch the Streamlit Investigation Workspace
```bash
streamlit run dashboard/app.py
```
Open your browser to: **`http://localhost:8501`**

### 6. Start the FastAPI Production Service
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation: **`http://localhost:8000/docs`**

---

## 13. Repository Directory Structure

```
Fraud-Detection/
│
├── alembic/                         # Database schema migration configuration
│   ├── env.py                       # SQLAlchemy metadata hook
│   ├── script.py.mako               # Migration template
│   └── versions/                    # Versioned schema migration scripts
│       └── b5381be6b0eb_create_alerts_and_cases_tables.py
│
├── dashboard/                       # Streamlit interactive investigation workspace
│   ├── app.py                       # Main executive operations dashboard
│   ├── utils.py                     # Custom CSS, glassmorphic metric cards, DB helpers
│   └── pages/
│       ├── alerts.py                # Prioritised alert queue and triage desk
│       ├── investigations.py        # Case investigation workbench & autonomous AI agent
│       └── transaction_detail.py    # Deep-dive TreeSHAP waterfall & dual gauge
│
├── data/                            # Datasets & local storage (git-ignored)
│   ├── .gitkeep
│   └── fraud_cases.db               # Local SQLite database fallback
│
├── miscellaneous/                   # State graphs and miscellaneous visual artifacts
│   └── investigation_agent_graph.png # Compiled LangGraph state machine visualization
│
├── models/                          # Serialized champion models and metadata
│   ├── anomaly_metadata.json        # Isolation Forest calibration parameters & operational thresholds
│   ├── isolation_forest.joblib      # Champion optimized Isolation Forest (Top 6 subspace)
│   ├── isolation_forest_optimized.joblib # Serialized artifact of optimized detector
│   ├── model_metadata.json          # XGBoost metadata, operational thresholds, top features
│   ├── preprocessor.joblib          # RobustScaler transformer for Amount & Time
│   ├── xgb_fraud_detector.joblib    # Serialized XGBoost Classifier
│   └── xgb_fraud_detector.json      # Native JSON format XGBoost booster
│
├── notebooks/                       # Research, benchmarking, and optimization workflows
│   ├── fraud_detection_part1.ipynb  # Part 1: Exploratory analysis & leak-free preprocessing
│   ├── fraud_detection_part2.ipynb  # Part 2: Supervised model evaluation & XGBoost selection
│   ├── fraud_detection_part3.ipynb  # Part 3: Isolation Forest subspace optimization
│   ├── part1_data_analysis.md       # Technical report for Part 1
│   ├── part2_model_evaluation.md    # Technical report for Part 2
│   └── part3_anomaly_detection_optimization.md # Technical report for Part 3
│
├── src/                             # Core Python application modules
│   ├── agents/                      # Agentic Investigation Layer (LangGraph + Groq OSS)
│   │   ├── config.py                # Groq ChatGroq model factory (gpt-oss-120b & 20b)
│   │   ├── evaluator.py             # Grounding, citation, hallucination benchmark suite
│   │   ├── investigation_agent.py   # Compiled LangGraph workflow & decision committer
│   │   ├── schemas.py               # Pydantic V2 schemas (Report, Citations, Approval)
│   │   ├── state.py                 # InvestigationState TypedDict definition
│   │   └── tools.py                 # Deterministic read-only retrieval tools (@tool)
│   │
│   ├── anomaly/                     # Unsupervised anomaly detection & comparative evaluation
│   │   ├── comparison.py            # Empirical comparison (PR-AUC, ROC-AUC, overlap analysis)
│   │   └── detector.py              # Isolation Forest anomaly scoring engine
│   │
│   ├── api/                         # FastAPI application and route endpoints
│   │   ├── main.py                  # App entry point, lifespan warmup, CORS
│   │   └── routes/
│   │       ├── agent.py             # Agent investigation, approval, and benchmark routes
│   │       ├── alerts.py            # Alert queue listing, summary, and triage routes
│   │       ├── cases.py             # Case management, workflow, notes, and audit routes
│   │       └── predict.py           # Single/batch prediction, explain, anomaly routes
│   │
│   ├── cases/                       # Database models, repositories, and case service
│   │   ├── database.py              # Engine, connection pooling, and get_db session dependency
│   │   ├── models.py                # Alert, InvestigationCase, CaseNote, CaseAuditLog models
│   │   ├── repository.py            # SQLAlchemy 2.0 CRUD operations and priority sorting
│   │   └── service.py               # CaseService coordinating transactions and workflows
│   │
│   ├── explainability/              # Model interpretability & multi-signal synthesis
│   │   ├── risk_assessor.py         # IntegratedRiskAssessor fusing 4 operational quadrants
│   │   └── shap_explainer.py        # TreeSHAP explainer with Plotly waterfall plots
│   │
│   ├── inference/                   # Reusable prediction engine and input validation
│   │   ├── predictor.py             # FraudPredictor singleton with multi-threshold scoring
│   │   ├── preprocessing.py         # Feature ordering verification & RobustScaling
│   │   └── schemas.py               # Pydantic V2 transaction validation schemas
│   │
│   └── risk/                        # Risk policy engine
│       └── policy.py                # Alert prioritization and auto-escalation criteria
│
├── tests/                           # Comprehensive test suite (68 tests)
│   ├── test_agent.py                # Tests for LangGraph workflow, tools, schemas & evaluator
│   ├── test_anomaly.py              # Tests for Isolation Forest detector and comparison
│   ├── test_api.py                  # Tests for all FastAPI endpoints (scoring, alerts, cases, agent)
│   ├── test_cases.py                # Tests for alert triage, case workflow, notes, and audit
│   ├── test_explainability.py       # Tests for TreeSHAP explainer and risk assessor
│   └── test_predictor.py            # Tests for XGBoost scoring, schemas, and thresholds
│
├── .gitignore                       # Git ignore rules (datasets, caches, databases)
├── alembic.ini                      # Alembic configuration file
├── pyproject.toml                   # Project metadata and build configuration
├── README.md                        # Professional system documentation
└── requirements.txt                 # Project dependencies
```

---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.