"""Comprehensive test suite verifying model loading, feature ordering, preprocessing,

scoring accuracy on saved holdout examples, and invalid input handling.
"""

from pathlib import Path
import random
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from pydantic import ValidationError

from src.inference.predictor import FraudPredictor, get_predictor, predict_transaction
from src.inference.preprocessing import (
    FEATURE_NAMES,
    PCA_FEATURES,
    SKEWED_FEATURES,
    load_preprocessor,
    preprocess_input,
    verify_features,
)
from src.inference.schemas import RiskCategory, TransactionInput


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture(scope="session")
def predictor() -> FraudPredictor:
    """Fixture providing a singleton FraudPredictor instance."""
    return get_predictor()


@pytest.fixture(scope="session")
def holdout_test_df() -> pd.DataFrame:
    """Load saved holdout evaluation dataset (data/test_data.csv)."""
    test_csv = Path("data/test_data.csv")
    if not test_csv.exists():
        pytest.skip("data/test_data.csv not available")
    return pd.read_csv(test_csv)


@pytest.fixture(scope="session")
def raw_creditcard_df() -> pd.DataFrame:
    """Load sample raw creditcard dataset (data/creditcard.csv)."""
    raw_csv = Path("data/creditcard.csv")
    if not raw_csv.exists():
        pytest.skip("data/creditcard.csv not available")
    return pd.read_csv(raw_csv, nrows=100)


# =====================================================================
# 1. Model & Metadata Loading and Verification Tests
# =====================================================================

def test_model_and_metadata_loading(predictor: FraudPredictor):
    """Verify model artifacts and metadata load cleanly with expected keys."""
    assert predictor.model is not None, "Model failed to load"
    assert predictor.metadata is not None, "Metadata failed to load"
    assert predictor.model_name == "XGBoost Classifier"
    assert predictor.model_version == "1.0.0"
    assert len(predictor.feature_names) == 30


def test_load_from_native_json_booster():
    """Verify FraudPredictor initializes cleanly from native JSON model artifact."""
    json_path = Path("models/xgb_fraud_detector.json")
    if not json_path.exists():
        pytest.skip("models/xgb_fraud_detector.json not found")
    predictor_json = FraudPredictor(model_path=json_path)
    assert predictor_json.model is not None
    assert isinstance(predictor_json.model, xgb.Booster)


def test_feature_ordering_matches_training(predictor: FraudPredictor):
    """Verify the feature sequence strictly matches training booster features."""
    expected_order = ["Amount", "Time"] + [f"V{i}" for i in range(1, 29)]
    assert predictor.feature_names == expected_order

    # Verify booster / sklearn feature names
    if hasattr(predictor.model, "feature_names_in_"):
        model_features = list(predictor.model.feature_names_in_)
        assert model_features == expected_order, "Sklearn model feature order does not match!"


def test_metadata_metrics_and_thresholds(predictor: FraudPredictor):
    """Verify operational thresholds and evaluation metrics from model training."""
    meta = predictor.metadata
    assert "evaluation_metrics" in meta
    metrics = meta["evaluation_metrics"]
    assert metrics["pr_auc"] == pytest.approx(0.8209, abs=1e-3)
    assert metrics["roc_auc"] == pytest.approx(0.9766, abs=1e-3)

    assert "operational_thresholds" in meta
    thresholds = meta["operational_thresholds"]
    assert "default" in thresholds
    assert "cost_optimal" in thresholds
    assert thresholds["default"] == 0.5
    assert thresholds["cost_optimal"] == pytest.approx(0.182, abs=1e-3)
    assert thresholds["f1_optimal"] == pytest.approx(0.798, abs=1e-3)
    assert thresholds["f2_optimal"] == pytest.approx(0.552, abs=1e-3)


# =====================================================================
# 2. Preprocessing & Feature Alignment Tests
# =====================================================================

def test_preprocessor_structure():
    """Verify preprocessor contains correct transformers for skewed and PCA features."""
    preprocessor = load_preprocessor()
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert "skewed" in transformer_names
    assert "pca" in transformer_names


def test_feature_reordering_resilience(predictor: FraudPredictor, holdout_test_df: pd.DataFrame):
    """Verify that dictionaries with scrambled/shuffled key orders produce identical predictions."""
    sample_row = holdout_test_df.iloc[0].drop(labels=["Class"]).to_dict()

    # Create shuffled dictionary
    keys = list(sample_row.keys())
    random.seed(123)
    random.shuffle(keys)
    shuffled_row = {k: sample_row[k] for k in keys}

    res_canonical = predictor.predict_transaction(sample_row, apply_preprocessing=False)
    res_shuffled = predictor.predict_transaction(shuffled_row, apply_preprocessing=False)

    assert res_canonical["fraud_probability"] == pytest.approx(res_shuffled["fraud_probability"], abs=1e-6)
    assert res_canonical["is_fraud"] == res_shuffled["is_fraud"]
    assert res_canonical["risk_category"] == res_shuffled["risk_category"]


def test_verify_features_detects_missing_column():
    """Verify verify_features raises ValueError when any expected feature is omitted."""
    incomplete_data = {"Amount": 10.0, "Time": 100.0, "V1": 0.5}  # missing V2..V28
    with pytest.raises(ValueError, match="Input is missing"):
        verify_features(incomplete_data)


def test_verify_features_detects_infinite_values():
    """Verify verify_features rejects Infinite values."""
    data = {f"V{i}": 0.0 for i in range(1, 29)}
    data["Amount"] = float("inf")
    data["Time"] = 100.0
    with pytest.raises(ValueError, match="Infinite values"):
        verify_features(data)


# =====================================================================
# 3. Holdout Prediction & Scoring Verification Tests
# =====================================================================

def test_holdout_legitimate_transactions(predictor: FraudPredictor, holdout_test_df: pd.DataFrame):
    """Verify that known ground-truth legitimate transactions score low fraud probability."""
    legit_df = holdout_test_df[holdout_test_df["Class"] == 0].head(10)
    for idx, row in legit_df.iterrows():
        tx_dict = row.drop(labels=["Class"]).to_dict()
        res = predictor.predict_transaction(tx_dict, threshold="default", apply_preprocessing=False)

        assert res["is_fraud"] is False, f"False positive on legitimate row {idx}"
        assert res["fraud_probability"] < 0.10, f"Expected low fraud prob, got {res['fraud_probability']}"
        assert res["risk_category"] == RiskCategory.LOW.value
        assert res["threshold_used"] == 0.5
        assert res["model_version"] == "1.0.0"


def test_holdout_fraudulent_transactions(predictor: FraudPredictor, holdout_test_df: pd.DataFrame):
    """Verify that known ground-truth fraud transactions score high fraud probability and flag True."""
    fraud_df = holdout_test_df[holdout_test_df["Class"] == 1].head(5)
    high_confidence_catches = 0

    for idx, row in fraud_df.iterrows():
        tx_dict = row.drop(labels=["Class"]).to_dict()
        res = predictor.predict_transaction(tx_dict, threshold="default", apply_preprocessing=False)

        if res["is_fraud"]:
            high_confidence_catches += 1
            assert res["risk_category"] in [RiskCategory.HIGH.value, RiskCategory.CRITICAL.value]

    # Model caught 77 out of 95 frauds at 0.5 (81.05% recall); verify high capture rate on head
    assert high_confidence_catches >= 3, "Fraud detection recall lower than expected on sample frauds"


def test_standalone_predict_transaction_function(holdout_test_df: pd.DataFrame):
    """Verify top-level predict_transaction function returns all specified contract fields."""
    sample_row = holdout_test_df.iloc[0].drop(labels=["Class"]).to_dict()
    result = predict_transaction(sample_row, threshold="cost_optimal", apply_preprocessing=False)

    required_keys = {
        "fraud_probability",
        "is_fraud",
        "risk_category",
        "threshold_used",
        "threshold_name",
        "model_version",
        "model_name",
    }
    assert required_keys.issubset(result.keys())
    assert isinstance(result["fraud_probability"], float)
    assert isinstance(result["is_fraud"], bool)
    assert isinstance(result["risk_category"], str)
    assert result["threshold_used"] == pytest.approx(0.182, abs=1e-3)
    assert result["threshold_name"] == "cost_optimal"
    assert result["model_version"] == "1.0.0"
    assert result["model_name"] == "XGBoost Classifier"


def test_batch_prediction_matches_single(predictor: FraudPredictor, holdout_test_df: pd.DataFrame):
    """Verify vectorized batch scoring matches sequential single scoring."""
    sample_rows = [holdout_test_df.iloc[i].drop(labels=["Class"]).to_dict() for i in range(5)]

    single_results = [
        predictor.predict_transaction(row, threshold="default", apply_preprocessing=False)
        for row in sample_rows
    ]
    batch_result = predictor.predict_batch(sample_rows, threshold="default", apply_preprocessing=False)

    assert batch_result["total_transactions"] == 5
    for single, batch in zip(single_results, batch_result["predictions"]):
        assert single["fraud_probability"] == pytest.approx(batch["fraud_probability"], abs=1e-6)
        assert single["is_fraud"] == batch["is_fraud"]
        assert single["risk_category"] == batch["risk_category"]


# =====================================================================
# 4. Thresholds & Configurable Risk Categories Tests
# =====================================================================

def test_operational_threshold_modes(predictor: FraudPredictor, holdout_test_df: pd.DataFrame):
    """Verify operational threshold modes resolve correctly and adjust binary classification."""
    # Find a sample with intermediate fraud probability (e.g. between 0.20 and 0.40)
    fraud_df = holdout_test_df[holdout_test_df["Class"] == 1]
    borderline_row = None
    for _, row in fraud_df.iterrows():
        tx = row.drop(labels=["Class"]).to_dict()
        p = predictor.predict_transaction(tx, apply_preprocessing=False)["fraud_probability"]
        if 0.15 <= p <= 0.45:
            borderline_row = tx
            break

    if borderline_row is not None:
        # At cost_optimal (0.182), borderline should be flagged fraud
        res_cost = predictor.predict_transaction(borderline_row, threshold="cost_optimal", apply_preprocessing=False)
        # At f1_optimal (0.798), borderline should NOT be flagged fraud
        res_f1 = predictor.predict_transaction(borderline_row, threshold="f1_optimal", apply_preprocessing=False)

        assert res_cost["is_fraud"] is True
        assert res_f1["is_fraud"] is False
        assert res_cost["threshold_used"] == pytest.approx(0.182, abs=1e-3)
        assert res_f1["threshold_used"] == pytest.approx(0.798, abs=1e-3)


def test_custom_numerical_threshold(predictor: FraudPredictor, holdout_test_df: pd.DataFrame):
    """Verify custom float threshold (e.g. 0.35) is respected."""
    sample_row = holdout_test_df.iloc[0].drop(labels=["Class"]).to_dict()
    res = predictor.predict_transaction(sample_row, threshold=0.35, apply_preprocessing=False)
    assert res["threshold_used"] == 0.35
    assert res["threshold_name"] == "custom"


def test_invalid_threshold_raises_error(predictor: FraudPredictor, holdout_test_df: pd.DataFrame):
    """Verify unknown threshold names or out-of-range floats raise ValueError."""
    sample_row = holdout_test_df.iloc[0].drop(labels=["Class"]).to_dict()

    with pytest.raises(ValueError, match="Unknown threshold name"):
        predictor.predict_transaction(sample_row, threshold="unsupported_mode", apply_preprocessing=False)

    with pytest.raises(ValueError, match="Threshold value must be between"):
        predictor.predict_transaction(sample_row, threshold=1.5, apply_preprocessing=False)

    with pytest.raises(ValueError, match="Threshold value must be between"):
        predictor.predict_transaction(sample_row, threshold=-0.1, apply_preprocessing=False)


def test_configurable_risk_category_cutoffs(predictor: FraudPredictor):
    """Verify custom risk category cutoffs override defaults properly."""
    custom_levels = {"low": 0.05, "medium": 0.20, "high": 0.60}

    assert predictor.calculate_risk_category(0.02, 0.5, custom_levels) == RiskCategory.LOW.value
    assert predictor.calculate_risk_category(0.10, 0.5, custom_levels) == RiskCategory.MEDIUM.value
    assert predictor.calculate_risk_category(0.35, 0.5, custom_levels) == RiskCategory.HIGH.value
    assert predictor.calculate_risk_category(0.75, 0.5, custom_levels) == RiskCategory.CRITICAL.value


# =====================================================================
# 5. Malformed Inputs & Schema Validation Tests
# =====================================================================

def test_negative_amount_for_raw_transaction(predictor: FraudPredictor, holdout_test_df: pd.DataFrame):
    """Verify raw transaction with negative Amount is rejected when apply_preprocessing=True."""
    tx = holdout_test_df.iloc[0].drop(labels=["Class"]).to_dict()
    tx["Amount"] = -50.0
    tx["Time"] = 100.0

    with pytest.raises(ValueError, match="Raw transaction Amount cannot be negative"):
        predictor.predict_transaction(tx, apply_preprocessing=True)


def test_negative_time_for_raw_transaction(predictor: FraudPredictor, holdout_test_df: pd.DataFrame):
    """Verify raw transaction with negative Time is rejected when apply_preprocessing=True."""
    tx = holdout_test_df.iloc[0].drop(labels=["Class"]).to_dict()
    tx["Amount"] = 50.0
    tx["Time"] = -100.0

    with pytest.raises(ValueError, match="Raw transaction Time cannot be negative"):
        predictor.predict_transaction(tx, apply_preprocessing=True)


def test_nan_feature_rejected_by_pydantic():
    """Verify Pydantic TransactionInput rejects NaN features."""
    data = {f"V{i}": 0.0 for i in range(1, 29)}
    data["Amount"] = float("nan")
    data["Time"] = 100.0

    with pytest.raises(ValidationError, match="must not be NaN"):
        TransactionInput(**data)


def test_extra_unexpected_fields_rejected_by_pydantic():
    """Verify Pydantic TransactionInput rejects extra extraneous fields (extra='forbid')."""
    data = {f"V{i}": 0.0 for i in range(1, 29)}
    data["Amount"] = 100.0
    data["Time"] = 0.0
    data["unauthorized_field"] = "malicious_injection"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TransactionInput(**data)
