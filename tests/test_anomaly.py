"""Unit and integration tests for Isolation Forest anomaly detection and model comparison."""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.anomaly.comparison import compare_models, generate_comparison_report
from src.anomaly.detector import AnomalyDetector, get_anomaly_detector


@pytest.fixture(scope="session")
def detector() -> AnomalyDetector:
    """Fixture providing initialized AnomalyDetector."""
    return get_anomaly_detector()


@pytest.fixture(scope="session")
def holdout_test_df() -> pd.DataFrame:
    """Load saved holdout evaluation dataset (data/test_data.csv)."""
    test_csv = Path("data/test_data.csv")
    if not test_csv.exists():
        pytest.skip("data/test_data.csv not available")
    return pd.read_csv(test_csv)


def test_anomaly_detector_loading(detector: AnomalyDetector):
    """Verify IsolationForest model and metadata load with calibration bounds."""
    assert detector.model is not None
    assert detector.metadata is not None
    assert detector.min_score < detector.max_score
    assert detector.metadata["algorithm"] == "IsolationForest"


def test_score_transaction_legitimate(detector: AnomalyDetector, holdout_test_df: pd.DataFrame):
    """Verify normal legitimate transaction yields low anomaly score and is_anomaly=False."""
    legit_row = holdout_test_df[holdout_test_df["Class"] == 0].iloc[0].drop(labels=["Class"]).to_dict()

    res = detector.score_transaction(legit_row, apply_preprocessing=False)

    assert "anomaly_score" in res
    assert "raw_score" in res
    assert "is_anomaly" in res
    assert "anomaly_tier" in res
    assert 0.0 <= res["anomaly_score"] <= 1.0
    assert res["is_anomaly"] is False
    assert res["anomaly_tier"] in ["NORMAL", "ELEVATED"]


def test_score_transaction_extreme_outlier(detector: AnomalyDetector, holdout_test_df: pd.DataFrame):
    """Verify an extreme synthetic statistical outlier is flagged as an anomaly."""
    # Synthesize extreme multi-feature outlier
    outlier_tx = {f"V{i}": 15.0 for i in range(1, 29)}
    outlier_tx["Amount"] = 100.0  # extreme scaled or raw value
    outlier_tx["Time"] = 50.0

    res = detector.score_transaction(outlier_tx, apply_preprocessing=False)

    assert res["is_anomaly"] is True
    assert res["anomaly_score"] > 0.70
    assert res["anomaly_tier"] == "OUTLIER"


def test_score_batch_vectorized(detector: AnomalyDetector, holdout_test_df: pd.DataFrame):
    """Verify vectorized batch scoring matches sequential outputs."""
    sample_rows = [
        holdout_test_df.iloc[i].drop(labels=["Class"]).to_dict()
        for i in range(5)
    ]

    single_results = [
        detector.score_transaction(r, apply_preprocessing=False)
        for r in sample_rows
    ]
    batch_result = detector.score_batch(sample_rows, apply_preprocessing=False)

    assert batch_result["total_transactions"] == 5
    assert len(batch_result["predictions"]) == 5
    for s, b in zip(single_results, batch_result["predictions"]):
        assert s["anomaly_score"] == pytest.approx(b["anomaly_score"], abs=1e-4)
        assert s["is_anomaly"] == b["is_anomaly"]
        assert s["anomaly_tier"] == b["anomaly_tier"]


def test_plot_anomaly_distribution(detector: AnomalyDetector):
    """Verify plot_anomaly_distribution generates valid Plotly Figure."""
    fig = detector.plot_anomaly_distribution(current_score=0.45)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    assert fig.data[0].type == "histogram"


def test_compare_models_metrics():
    """Verify compare_models computes scikit-learn metrics and generates comparison charts."""
    # Run on subsample for speed during unit test suite
    comp = compare_models(subsample_n=300)

    assert "supervised_metrics" in comp
    assert "anomaly_metrics" in comp
    assert "overlap_analysis" in comp

    sup = comp["supervised_metrics"]
    ano = comp["anomaly_metrics"]

    # Verify metrics exist and are within mathematical range [0, 1]
    for key in ["pr_auc", "roc_auc", "precision", "recall", "f1_score"]:
        assert 0.0 <= sup[key] <= 1.0
        assert 0.0 <= ano[key] <= 1.0

    # Overlap concordance
    con = comp["overlap_analysis"]["concordance"]
    assert "both_flagged_count" in con
    assert "xgb_only_count" in con
    assert "iso_only_count" in con

    # Plotly figures
    assert comp["metrics_plot_dict"] is not None
    assert comp["bivariate_plot_dict"] is not None
    assert len(comp["metrics_plot_html"]) > 1000
    assert len(comp["bivariate_plot_html"]) > 1000

    # Report text
    report = generate_comparison_report(comp)
    assert "Supervised vs. Unsupervised Model Comparison Audit" in report
    assert "PR-AUC" in report
    assert "Concordance" in report
