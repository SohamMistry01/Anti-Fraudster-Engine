"""Unit and integration tests for SHAP explainability and Plotly feature attribution."""

from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.explainability.risk_assessor import IntegratedRiskAssessor, get_risk_assessor
from src.explainability.shap_explainer import FraudSHAPExplainer, get_explainer


@pytest.fixture(scope="session")
def explainer() -> FraudSHAPExplainer:
    """Fixture providing initialized TreeSHAP explainer."""
    return get_explainer()


@pytest.fixture(scope="session")
def holdout_test_df() -> pd.DataFrame:
    """Load saved holdout evaluation dataset (data/test_data.csv)."""
    test_csv = Path("data/test_data.csv")
    if not test_csv.exists():
        pytest.skip("data/test_data.csv not available")
    return pd.read_csv(test_csv)


def test_shap_explainer_initialization(explainer: FraudSHAPExplainer):
    """Verify TreeExplainer initializes with model and extracts valid base value."""
    assert explainer.explainer is not None
    assert explainer.predictor is not None
    assert isinstance(explainer.base_value, float)


def test_explain_transaction_legitimate(explainer: FraudSHAPExplainer, holdout_test_df: pd.DataFrame):
    """Verify SHAP explanation structure on legitimate transaction."""
    legit_row = holdout_test_df[holdout_test_df["Class"] == 0].iloc[0].drop(labels=["Class"]).to_dict()

    res = explainer.explain_transaction(legit_row, apply_preprocessing=False, top_n=5, generate_plot=True)

    assert "fraud_probability" in res
    assert res["fraud_probability"] < 0.05
    assert res["is_fraud"] is False
    assert "top_positive_contributors" in res
    assert "top_negative_contributors" in res
    assert "all_contributions" in res
    assert len(res["all_contributions"]) == 30

    # Verify Plotly figure outputs
    assert res["plot_dict"] is not None
    assert res["plot_html"] is not None
    assert "<div id=" in res["plot_html"] or "plotly" in res["plot_html"].lower()


def test_explain_transaction_fraud(explainer: FraudSHAPExplainer, holdout_test_df: pd.DataFrame):
    """Verify that known fraud sample identifies key fraud-driving features (e.g. V14, V10)."""
    fraud_row = holdout_test_df[holdout_test_df["Class"] == 1].iloc[0].drop(labels=["Class"]).to_dict()

    res = explainer.explain_transaction(fraud_row, apply_preprocessing=False, top_n=5)

    assert res["is_fraud"] is True
    assert res["fraud_probability"] > 0.85
    assert len(res["top_positive_contributors"]) > 0

    # Verify that top positive contributor contains key predictive components (V14 or V10)
    top_pos_features = [c["feature"] for c in res["top_positive_contributors"]]
    assert any(feat in top_pos_features for feat in ["V14", "V10", "V12", "V4"]), (
        f"Expected top predictive features in positive contributors, got {top_pos_features}"
    )


def test_statistical_evidence_no_fabrication(explainer: FraudSHAPExplainer, holdout_test_df: pd.DataFrame):
    """Verify that generated evidence descriptions adhere strictly to mathematical facts.

    Integrity rule: Does NOT invent merchant names, cardholder personas, or fake velocities.
    """
    fraud_row = holdout_test_df[holdout_test_df["Class"] == 1].iloc[0].drop(labels=["Class"]).to_dict()
    res = explainer.explain_transaction(fraud_row, apply_preprocessing=False)

    summary = res["evidence_summary"].lower()
    # Ensure no fabricated domain buzzwords
    forbidden_terms = ["foreign ip", "card velocity", "merchant country", "suspicious atm", "cardholder age"]
    for term in forbidden_terms:
        assert term not in summary, f"Found unverified fabricated term '{term}' in evidence summary!"

    # Verify explicitly mentions PCA / statistical log-odds
    assert "pca" in summary or "statistical" in summary or "log-odds" in summary


def test_shap_plotly_chart_generation(explainer: FraudSHAPExplainer):
    """Verify create_shap_waterfall_plot builds valid Plotly Figure."""
    sample_contributions = [
        {"feature": "V14", "feature_value": -4.28, "shap_value": 3.12, "direction": "INCREASES_FRAUD_RISK"},
        {"feature": "V4", "feature_value": 3.99, "shap_value": 1.45, "direction": "INCREASES_FRAUD_RISK"},
        {"feature": "V11", "feature_value": 1.20, "shap_value": -0.85, "direction": "DECREASES_FRAUD_RISK"},
    ]

    fig = explainer.create_shap_waterfall_plot(
        top_contributions=sample_contributions,
        base_value=0.0,
        fraud_prob=0.92,
        is_fraud=True,
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "bar"
    assert fig.data[0].orientation == "h"


def test_integrated_risk_assessor_quadrants(holdout_test_df: pd.DataFrame):
    """Verify IntegratedRiskAssessor categorizes transactions and recommends appropriate actions."""
    assessor = get_risk_assessor()

    # Legitimate sample -> CONFIRMED_LEGITIMATE
    legit_row = holdout_test_df[holdout_test_df["Class"] == 0].iloc[0].drop(labels=["Class"]).to_dict()
    legit_eval = assessor.assess_transaction(legit_row, apply_preprocessing=False)

    assert legit_eval["operational_quadrant"] == "CONFIRMED_LEGITIMATE"
    assert legit_eval["recommended_action"] == "APPROVE"
    assert "supervised_signal" in legit_eval
    assert "unsupervised_anomaly_signal" in legit_eval
    assert "shap_explanation" in legit_eval

    # Fraud sample -> CONFIRMED_HIGH_RISK or SUBTLE_IN_DISTRIBUTION_FRAUD
    fraud_row = holdout_test_df[holdout_test_df["Class"] == 1].iloc[0].drop(labels=["Class"]).to_dict()
    fraud_eval = assessor.assess_transaction(fraud_row, apply_preprocessing=False)

    assert fraud_eval["operational_quadrant"] in ["CONFIRMED_HIGH_RISK", "SUBTLE_IN_DISTRIBUTION_FRAUD"]
    assert fraud_eval["recommended_action"] in ["DECLINE_TRANSACTION", "CHALLENGE_OR_REVIEW"]
    assert fraud_eval["supervised_signal"]["is_fraud"] is True
