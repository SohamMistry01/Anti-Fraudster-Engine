"""Test suite for FastAPI REST API endpoints covering health, metadata, and scoring."""

from pathlib import Path
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


@pytest.fixture(scope="session")
def holdout_test_df() -> pd.DataFrame:
    """Load saved holdout evaluation dataset for API testing."""
    test_csv = Path("data/test_data.csv")
    if not test_csv.exists():
        pytest.skip("data/test_data.csv not available")
    return pd.read_csv(test_csv)


def test_health_endpoint():
    """Verify GET /health returns ready status and model details."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_version"] == "1.0.0"
    assert data["model_name"] == "XGBoost Classifier"
    assert data["features_count"] == 30


def test_model_metadata_endpoint():
    """Verify GET /model/metadata exposes metrics, operational thresholds, and features."""
    response = client.get("/model/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "XGBoost Classifier"
    assert data["model_version"] == "1.0.0"
    assert data["features_count"] == 30
    assert "evaluation_metrics" in data
    assert "operational_thresholds" in data
    assert data["operational_thresholds"]["default"] == 0.5
    assert data["operational_thresholds"]["cost_optimal"] == pytest.approx(0.182, abs=1e-3)
    assert "V14" in data["top_5_predictive_features"]


def test_predict_single_direct_payload(holdout_test_df: pd.DataFrame):
    """Verify POST /predict with direct transaction payload and apply_preprocessing=False."""
    legit_row = holdout_test_df[holdout_test_df["Class"] == 0].iloc[0].drop(labels=["Class"]).to_dict()

    response = client.post("/predict?apply_preprocessing=false", json=legit_row)
    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    assert "is_fraud" in data
    assert "risk_category" in data
    assert "threshold_used" in data
    assert "model_version" in data
    assert data["is_fraud"] is False
    assert data["fraud_probability"] < 0.05
    assert data["risk_category"] == "LOW"


def test_predict_single_wrapper_payload(holdout_test_df: pd.DataFrame):
    """Verify POST /predict with PredictionRequest envelope structure."""
    legit_row = holdout_test_df[holdout_test_df["Class"] == 0].iloc[0].drop(labels=["Class"]).to_dict()
    payload = {
        "transaction": legit_row,
        "threshold": "cost_optimal",
        "apply_preprocessing": False,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["threshold_used"] == pytest.approx(0.182, abs=1e-3)
    assert data["threshold_name"] == "cost_optimal"


def test_predict_fraud_sample(holdout_test_df: pd.DataFrame):
    """Verify POST /predict correctly flags ground-truth fraud sample."""
    fraud_row = holdout_test_df[holdout_test_df["Class"] == 1].iloc[0].drop(labels=["Class"]).to_dict()
    payload = {
        "transaction": fraud_row,
        "threshold": "default",
        "apply_preprocessing": False,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_fraud"] is True
    assert data["fraud_probability"] > 0.90
    assert data["risk_category"] in ["HIGH", "CRITICAL"]


def test_predict_batch_endpoint(holdout_test_df: pd.DataFrame):
    """Verify POST /predict/batch scores a list of transactions."""
    rows = [
        holdout_test_df.iloc[i].drop(labels=["Class"]).to_dict()
        for i in range(4)
    ]
    payload = {
        "transactions": rows,
        "threshold": "default",
        "apply_preprocessing": False,
    }

    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_transactions"] == 4
    assert "predictions" in data
    assert len(data["predictions"]) == 4


def test_predict_malformed_input_returns_422():
    """Verify malformed input (missing required features) returns HTTP 422."""
    bad_payload = {"Amount": 50.0}  # missing Time and V1..V28
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_invalid_threshold_returns_400(holdout_test_df: pd.DataFrame):
    """Verify invalid threshold parameter returns HTTP 400 or 422."""
    legit_row = holdout_test_df.iloc[0].drop(labels=["Class"]).to_dict()
    payload = {
        "transaction": legit_row,
        "threshold": "non_existent_mode",
        "apply_preprocessing": False,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in [400, 422]


def test_api_predict_explain_endpoint(holdout_test_df: pd.DataFrame):
    """Verify POST /predict/explain returns SHAP feature contributions and Plotly HTML."""
    legit_row = holdout_test_df[holdout_test_df["Class"] == 0].iloc[0].drop(labels=["Class"]).to_dict()
    response = client.post("/predict/explain?apply_preprocessing=false", json=legit_row)
    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    assert "top_positive_contributors" in data
    assert "top_negative_contributors" in data
    assert "evidence_summary" in data
    assert "plot_html" in data
    assert data["plot_html"] is not None


def test_api_anomaly_score_endpoint(holdout_test_df: pd.DataFrame):
    """Verify POST /anomaly/score returns normalized anomaly score and flag."""
    legit_row = holdout_test_df[holdout_test_df["Class"] == 0].iloc[0].drop(labels=["Class"]).to_dict()
    response = client.post("/anomaly/score?apply_preprocessing=false", json=legit_row)
    assert response.status_code == 200
    data = response.json()
    assert "anomaly_score" in data
    assert "is_anomaly" in data
    assert "anomaly_tier" in data
    assert data["algorithm"] == "IsolationForest"
    assert data["is_anomaly"] is False


def test_api_evaluate_combined_endpoint(holdout_test_df: pd.DataFrame):
    """Verify POST /evaluate/combined integrates supervised, unsupervised, and SHAP evidence."""
    legit_row = holdout_test_df[holdout_test_df["Class"] == 0].iloc[0].drop(labels=["Class"]).to_dict()
    response = client.post("/evaluate/combined?apply_preprocessing=false", json=legit_row)
    assert response.status_code == 200
    data = response.json()
    assert "assessment_summary" in data
    assert "operational_quadrant" in data
    assert "recommended_action" in data
    assert "supervised_signal" in data
    assert "unsupervised_anomaly_signal" in data
    assert data["operational_quadrant"] == "CONFIRMED_LEGITIMATE"
    assert data["recommended_action"] == "APPROVE"


def test_api_anomaly_comparison_endpoint():
    """Verify GET /anomaly/comparison returns comparative metrics and Plotly charts."""
    response = client.get("/anomaly/comparison?subsample_n=200")
    assert response.status_code == 200
    data = response.json()
    assert "supervised_metrics" in data
    assert "anomaly_metrics" in data
    assert "overlap_analysis" in data
    assert "metrics_plot_html" in data
    assert "bivariate_plot_html" in data

