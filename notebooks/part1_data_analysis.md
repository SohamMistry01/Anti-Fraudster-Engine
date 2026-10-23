# Part 1: Exploratory Data Analysis & Feature-Aware Preprocessing Report

**Project:** Credit Card Fraud Detection  
**Notebook:** [`notebooks/fraud_detection_part1.ipynb`](file:///e:/Fraud-Detection/notebooks/fraud_detection_part1.ipynb)  
**Dataset Source:** [`data/creditcard.csv`](file:///e:/Fraud-Detection/data/creditcard.csv)  
**Artifacts Generated:**  
- [`data/train_data.csv`](file:///e:/Fraud-Detection/data/train_data.csv) (226,980 rows × 31 columns, 120.53 MB)
- [`data/test_data.csv`](file:///e:/Fraud-Detection/data/test_data.csv) (56,746 rows × 31 columns, 30.13 MB)

---

## 1. Executive Summary

In Part 1 of the Credit Card Fraud Detection project, we conducted a rigorous end-to-end data analysis, sanity verification, anomaly inspection, and feature-aware preprocessing pipeline on the European cardholder transaction dataset.

The core challenge of this dataset is extreme class imbalance (**0.167% fraud**, or approximately 1 fraud case for every 599 legitimate transactions). Rather than applying standard generic transformations, we performed deep behavioral profiling on monetary amounts, transaction timing, and PCA components, leading to a modular, leak-free pipeline tailored to the specific mathematical and domain properties of each feature category.

---

## 2. Step-by-Step Workflow & Cell Outputs

### **Cell 1: Environment Setup & Data Ingestion**
- **Objective:** Load the raw credit card transactions securely and verify dimensions across environments.
- **Implementation:** Imported core data science libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`) and implemented a directory-agnostic path resolver (`data/creditcard.csv` vs `../data/creditcard.csv`).
- **Output:**
  - Raw records: **284,807 rows**
  - Features: **31 columns** (30 numerical features + 1 target `Class`)

---

### **Cell 2: Data Types, Missing Values & Duplicate Scan**
- **Objective:** Audit null values, data types, memory consumption, and record uniqueness.
- **Key Findings:**
  - **Data Types:** 30 features are `float64`, target `Class` is `int64`. Total in-memory footprint is **67.4 MB**.
  - **Explicit Missing Values:** **0 NaN / null values** across all 31 columns.
  - **Duplicate Rows:** **1,081 duplicate rows (0.38%)** detected.

---

### **Cell 3: Masked Value Checks & Duplicate Resolution**
- **Objective:** Verify potential masked anomalies (`np.inf`, `-np.inf`, negative monetary amounts, negative timestamps) and safely resolve duplicates without destroying rare positive signals.
- **Key Findings:**
  - Infinite values count: **0** across all numerical fields.
  - Negative values in `Amount`: **0**
  - Negative values in `Time`: **0**
  - **Duplicate Class Breakdown:**
    - Legitimate duplicates: **1,062 rows**
    - Fraudulent duplicates: **19 rows**
- **Action Taken:** Deduped records using `drop_duplicates(keep='first')` to preserve the first occurrence of identical transactions while eliminating data leakage for model training.
- **Cleaned Dataset Dimensions:** **283,726 rows × 31 columns** (1,081 rows removed).

---

### **Cell 4: Target Distribution & Imbalance Quantification**
- **Objective:** Quantify the minority class severity and visualize the class distribution.
- **Quantitative Results:**
  - **Legitimate (Class 0):** 283,253 transactions (**99.833%**)
  - **Fraudulent (Class 1):** 473 transactions (**0.167%**)
  - **Imbalance Ratio:** **1 : 598.8** (1 fraud case per ~599 legitimate transactions).
- **Visualization:** Linear vs. Log-scale bar charts confirmed that linear scaling completely obscures the minority class, reinforcing that standard accuracy metrics would yield misleading 99.83% scores while predicting zero fraud.

---

### **Cell 5: Transaction Amount (`Amount`) Deep Dive**
- **Objective:** Analyze monetary behavior distinguishing fraud from normal cardholder spending.
- **Statistical Breakdown:**

| Metric | Legitimate (Class 0) | Fraudulent (Class 1) | Key Takeaway |
| :--- | :--- | :--- | :--- |
| **Count** | 283,253 | 473 | - |
| **Mean** | $88.41 | $123.87 | Fraud average is ~40% higher than normal purchases |
| **Std Dev** | $250.38 | $260.21 | Both exhibit high variance |
| **Median (50%)** | $22.00 | **$9.82** | Median fraud is much lower due to card-testing transactions |
| **75th Percentile** | $77.46 | $105.89 | Upper quartile fraud shifts higher |
| **95th Percentile** | $365.00 | $655.82 | Substantial concentration in the $100–$650 range |
| **99th Percentile** | $1,018.06 | $1,364.14 | Higher top percentiles |
| **Max** | **$25,691.16** | **$2,125.87** | **Fraudsters intentionally cap spending under ~$2,150** |

- **Distribution Insights:**
  1. **Bimodal fraud density:** Fraud clusters around micro-transactions ($0 to $1) to test stolen card validity, followed by a secondary peak between $100 and $500.
  2. **Ceiling avoidance:** Legitimate transactions reach up to $25,691.16, whereas fraud completely cuts off at $2,125.87 to evade instant bank authorization flags.

---

### **Cell 6: Temporal Dynamics & Circadian Rhythm Analysis (`Time`)**
- **Objective:** Investigate whether transactions reflect human daily activity patterns over the 48-hour recording span.
- **Key Insights:**
  - Extracted feature `Hour_of_Day = ((Time / 3600) % 24)`.
  - **Legitimate Circadian Cycle:** Normal transactions drop steeply to ~1% between hours **2:00 AM and 6:00 AM** (sleeping hours) and remain consistently between 5.5%–6.2% throughout daytime hours (9:00 AM to 10:00 PM).
  - **Fraud Anomalies:** Fraud does not respect human sleep cycles. Noticeable spikes occur during early morning hours (e.g., hour 2:00 AM and hour 11:00 AM), indicating bot automated attacks or card exploitation when victims are asleep.

---

### **Cell 7: Correlation Analysis with Target (`Class`)**
- **Objective:** Identify linear predictive relationships between the 28 orthogonal PCA features and fraud.
- **Top Negatively Correlated Features:**
  - `V17`: **-0.3135**
  - `V14`: **-0.2934**
  - `V12`: **-0.2507**
  - `V10`: **-0.2070**
  - `V16`: **-0.1872**
- **Top Positively Correlated Features:**
  - `V11`: **+0.1491**
  - `V4`:  **+0.1293**
  - `V2`:  **+0.0846**
  - `V19`: **+0.0336**
  - `V8`:  **+0.0331**

---

### **Cell 8: Discriminative Separation & Outlier Risk Evaluation**
- **Objective:** Evaluate distribution separation on top features (`V17`, `V14`, `V12`, `V11`) and assess outlier removal implications.
- **Median Differences:**
  - `V14`: Legit median `+0.051` vs Fraud median **`-6.591`** (Δ = **-6.642**)
  - `V12`: Legit median `+0.141` vs Fraud median **`-5.437`** (Δ = **-5.578**)
  - `V17`: Legit median `-0.065` vs Fraud median **`-5.158`** (Δ = **-5.093**)
  - `V11`: Legit median `-0.034` vs Fraud median **`+3.526`** (Δ = **+3.560**)
- **Critical Risk Identified:** Many observations that appear as extreme statistical outliers in the general population are actually **ground-truth fraud transactions**. Standard unsupervised outlier removal techniques (such as IQR trimming or Isolation Forest on the raw dataset) would severely delete positive fraud cases.

---

### **Cell 9: Feature-Aware Preprocessing Pipeline Architecture**
- **Objective:** Construct a scikit-learn `ColumnTransformer` applying targeted strategies based on feature mathematics:

```
                          ┌─── [Amount, Time] ───> SimpleImputer(strategy='median') ───> RobustScaler()
                          │
Raw Features (30) ────────┼
                          │
                          └─── [V1 to V28] ──────> SimpleImputer(strategy='mean') ─────> Passthrough
```

- **Rationale:**
  - **`Amount` & `Time`:** Imputed with **median** to prevent outlier skew (e.g., $25,000 purchases), scaled via **`RobustScaler`** (IQR-based centering) to prevent heavy compression of standard values.
  - **`V1` – `V28`:** Imputed with **mean** (expected mathematical value of PCA components is 0), passed through without additional scaling since PCA components are already standardized.

---

### **Cell 10: Stratified Partitioning & Leak-Free Transformation**
- **Objective:** Ensure zero data leakage between training and evaluation splits.
- **Split Configuration:** 80% Train, 20% Test with `stratify=y, random_state=42`.
- **Partition Verification:**
  - **Train Set:** 226,980 rows | 378 frauds (**0.167%**)
  - **Test Set:** 56,746 rows | 95 frauds (**0.167%**)
- **Leak-Free Transformation:**
  - Preprocessor fitted strictly on `X_train` (`fit_transform`).
  - `X_test` transformed using frozen parameters from `X_train` (`transform`).
  - Total remaining NaNs in train & test: **0**.

---

### **Cell 11: Export of Production-Ready Partitions**
- **Objective:** Serialize clean, scaled datasets for Part 2 model experimentation.
- **Exported Files in [`data/`](file:///e:/Fraud-Detection/data/):**
  - **`train_data.csv`:** 226,980 rows × 31 columns (120.53 MB)
  - **`test_data.csv`:** 56,746 rows × 31 columns (30.13 MB)

---

## 3. Comprehensive Feature Characteristic Matrix

| Feature Group | Features | Type | Distribution / Behavior | Missing Strategy | Scaling Method |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Monetary** | `Amount` | Continuous Float | Severe right skew ($0 to $25k), bimodal in fraud | Median Imputation | `RobustScaler` (IQR) |
| **Temporal** | `Time` | Continuous Float | Monotonic seconds, distinct diurnal cycle | Median Imputation | `RobustScaler` (IQR) |
| **Principal Components** | `V1`–`V28` | Continuous Float | Orthogonal, zero-centered, heavy fraud separation | Mean Imputation | Passthrough (Pre-scaled) |
| **Target** | `Class` | Binary Integer | Severe imbalance (99.83% : 0.167%) | None (Target) | Categorical Binary (0 / 1) |

---

## 4. Roadmap for Part 2: Model Training & Evaluation Phase

With clean, preprocessed data ready in [`data/train_data.csv`](file:///e:/Fraud-Detection/data/train_data.csv) and [`data/test_data.csv`](file:///e:/Fraud-Detection/data/test_data.csv), Part 2 will execute the following stages:

### **Stage 1: Handling Severe Class Imbalance**
- **Algorithmic Balancing:**
  - Cost-sensitive learning via `class_weight='balanced'` in linear models and `scale_pos_weight` (~599) in XGBoost / CatBoost.
- **Resampling Techniques (Applied strictly to `X_train` within cross-validation folds):**
  - **SMOTE** (Synthetic Minority Over-sampling Technique)
  - **SMOTE + Tomek Links** / **SMOTE + ENN** (oversampling combined with borderline noise cleanup)
  - **Random Under-Sampling** (as a fast benchmark)

### **Stage 2: Model Architecture Benchmark**
1. **Baseline Model:** Logistic Regression with L2 regularization (fast, interpretable benchmark).
2. **Ensemble Bagging:** Random Forest Classifier (robust against noisy decision boundaries).
3. **Gradient Boosted Trees (Primary Contenders):**
   - **XGBoost:** Excellent handle of sparse/extreme values with direct `scale_pos_weight`.
   - **CatBoost / LightGBM:** High training efficiency and built-in handling of complex continuous feature interactions.

### **Stage 3: Evaluation Metrics Selection**
*Note: Accuracy is strictly disqualified (a model predicting 100% Class 0 attains 99.83% accuracy).*
- **Primary Metric:** **PR-AUC (Precision-Recall Area Under Curve / Average Precision)** — the gold standard for rare-event detection where false positives and false negatives carry disproportionate business weight.
- **Secondary Metrics:**
  - **ROC-AUC:** Overall ranking discrimination.
  - **Recall at Fixed Precision:** (e.g., what % of fraud can we capture if we tolerate a maximum 20% false alarm rate?).
  - **F1-Score & F2-Score:** F2-score places higher emphasis on recall (missing a fraud is financially worse than a false alert).

### **Stage 4: Optimal Threshold Tuning & Cost-Benefit Analysis**
- Models by default predict fraud if probability $\ge 0.5$. In fraud detection, threshold calibration (e.g., testing thresholds from 0.05 to 0.50) is vital to maximize economic utility.
- Construct a business cost matrix:
  - *Cost of False Negative:* Direct loss of stolen funds (average ~$124+ chargebacks).
  - *Cost of False Positive:* Minor friction / cardholder notification cost (~$2 to $5).
- Plot Cost Curves across thresholds to find the business-optimal decision boundary.
