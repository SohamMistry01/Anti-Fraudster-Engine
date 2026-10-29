"""Comprehensive automated test suite for Fraud Pattern Analysis Agent.

Verifies:
- Deterministic cohort statistics computation (amounts, percentiles, error rates, PCA means)
- Emerging anomaly and statistical shift detection (severity scoring, relative change)
- Pydantic V2 schemas (TimeWindowCohort, DetectedPatternShift, PatternFindingItem, PatternAnalysisReport)
- Factual grounding evaluation, numeric fidelity, and domain hallucination rejection (anonymized PCA guardrails)
- LangGraph stateful pattern analysis graph compilation and sequential node execution
- End-to-end run_pattern_analysis execution with resilient fallback guarantee
- FastAPI endpoints (/agent/pattern-analysis/run, /agent/pattern-analysis/latest, /agent/pattern-analysis/evaluate)
"""

from typing import Any, Dict, List
import pytest
from fastapi.testclient import TestClient

from src.agents.config import (
    DEFAULT_AGENT_MODEL,
    DISTRIBUTED_PIPELINE_OPTION,
    PATTERN_COHORT_MODEL,
    PATTERN_EXECUTIVE_MODEL,
    PATTERN_FINDINGS_MODEL,
    SUPPORTED_MODELS,
)
from src.agents.evaluator import evaluate_pattern_grounding
from src.agents.pattern_agent import (
    PROHIBITED_DOMAIN_PATTERNS,
    build_pattern_analysis_graph,
    compute_cohort_statistics,
    detect_statistical_shifts,
    formulate_pattern_findings,
    get_pattern_analysis_graph,
    node_compute_pattern_statistics,
    node_detect_anomalous_trends,
    node_gather_historical_data,
    node_publish_analysis_report,
    node_synthesize_findings,
    node_validate_evidence,
    run_pattern_analysis,
    save_pattern_graph_image_if_needed,
    synthesize_cohort_and_error_trends,
    synthesize_executive_summary,
)
from src.agents.schemas import (
    DetectedPatternShift,
    PatternAnalysisReport,
    PatternCategory,
    PatternFindingItem,
    ShiftSeverity,
    StatisticalConfidence,
    TimeWindowCohort,
)
from src.agents.state import PatternAnalysisState
from src.agents.tools import get_historical_pattern_data
from src.api.main import app

client = TestClient(app)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_historical_records() -> List[Dict[str, Any]]:
    """Generate a controlled cohort of historical alerts for deterministic statistical testing."""
    records = []
    # 10 baseline records: lower amounts, lower anomalies
    for i in range(10):
        records.append({
            "alert_id": f"ALT-TEST-B{i:02d}",
            "transaction_id": f"TX-TEST-B{i:02d}",
            "created_at": f"2026-10-25T{i:02d}:00:00Z",
            "amount": 100.0 + i * 10.0,
            "time_offset": float(i * 100),
            "fraud_probability": 0.20 if i < 8 else 0.85,
            "anomaly_score": 0.30 if i < 8 else 0.75,
            "priority": "LOW" if i < 8 else "CRITICAL",
            "operational_quadrant": "CONFIRMED_LEGITIMATE" if i < 8 else "CONFIRMED_HIGH_RISK",
            "status": "RESOLVED",
            "verdict": "FALSE_POSITIVE" if i < 8 else "CONFIRMED_FRAUD",
            "transaction_payload": {"Time": i * 100.0, "Amount": 100.0 + i * 10.0, "V14": -0.5, "V10": -0.2, "V4": 0.1},
        })

    # 10 recent records: surge in amounts, surge in anomalies
    for i in range(10):
        records.append({
            "alert_id": f"ALT-TEST-R{i:02d}",
            "transaction_id": f"TX-TEST-R{i:02d}",
            "created_at": f"2026-10-26T{i:02d}:00:00Z",
            "amount": 400.0 + i * 20.0,
            "time_offset": float(1000 + i * 100),
            "fraud_probability": 0.80 if i < 6 else 0.25,
            "anomaly_score": 0.85 if i < 6 else 0.35,
            "priority": "CRITICAL" if i < 6 else "LOW",
            "operational_quadrant": "CONFIRMED_HIGH_RISK" if i < 6 else "CONFIRMED_LEGITIMATE",
            "status": "RESOLVED",
            "verdict": "CONFIRMED_FRAUD" if i < 6 else "FALSE_POSITIVE",
            "transaction_payload": {"Time": 1000.0 + i * 100.0, "Amount": 400.0 + i * 20.0, "V14": -3.8, "V10": -2.5, "V4": 2.2},
        })

    return records


# =============================================================================
# DETERMINISTIC STATISTICAL COMPUTATION TESTS
# =============================================================================

def test_compute_cohort_statistics(sample_historical_records: List[Dict[str, Any]]):
    """Verify deterministic calculation of cohort amounts, percentiles, probabilities, and error rates."""
    baseline = sample_historical_records[:10]
    recent = sample_historical_records[10:]

    b_stats = compute_cohort_statistics(baseline, window_name="Baseline Window")
    r_stats = compute_cohort_statistics(recent, window_name="Recent Window")

    # Count verification
    assert b_stats["alert_count"] == 10
    assert r_stats["alert_count"] == 10

    # Monetary exposure
    assert b_stats["mean_amount"] < r_stats["mean_amount"]
    assert b_stats["median_amount"] == 145.0
    assert r_stats["median_amount"] == 490.0

    # High anomaly share (Anomaly Score >= 0.70)
    assert b_stats["high_anomaly_share_pct"] == 20.0  # 2 of 10
    assert r_stats["high_anomaly_share_pct"] == 60.0  # 6 of 10

    # Supervised fraud share (P(Fraud) >= 0.70)
    assert b_stats["high_fraud_share_pct"] == 20.0  # 2 of 10
    assert r_stats["high_fraud_share_pct"] == 60.0  # 6 of 10

    # Verified outcomes & False Positive Rate
    assert b_stats["verified_outcomes_count"] == 10
    assert b_stats["confirmed_fraud_count"] == 2
    assert b_stats["false_positive_count"] == 8
    assert b_stats["false_positive_rate"] == 0.8
    assert b_stats["precision"] == 0.2

    assert r_stats["verified_outcomes_count"] == 10
    assert r_stats["confirmed_fraud_count"] == 6
    assert r_stats["false_positive_count"] == 4
    assert r_stats["false_positive_rate"] == 0.4
    assert r_stats["precision"] == 0.6

    # PCA feature averages
    assert "V14" in b_stats["pca_means"]
    assert b_stats["pca_means"]["V14"] == -0.5
    assert r_stats["pca_means"]["V14"] == -3.8


def test_compute_cohort_statistics_empty():
    """Verify safe fallback for empty cohort records."""
    stats = compute_cohort_statistics([], window_name="Empty Window")
    assert stats["alert_count"] == 0
    assert stats["mean_amount"] == 0.0
    assert stats["false_positive_rate"] is None


def test_detect_statistical_shifts(sample_historical_records: List[Dict[str, Any]]):
    """Verify deterministic shift detection and severity classification between cohorts."""
    baseline = compute_cohort_statistics(sample_historical_records[:10], window_name="Baseline")
    recent = compute_cohort_statistics(sample_historical_records[10:], window_name="Recent")

    shifts = detect_statistical_shifts(baseline=baseline, recent=recent)
    assert len(shifts) >= 4

    shift_names = [s["metric_name"] for s in shifts]
    assert any("High-Anomaly" in name for name in shift_names)
    assert any("Monetary Exposure" in name for name in shift_names)
    assert any("False Positive Rate" in name for name in shift_names)

    # Check high anomaly shift magnitude
    anom_shift = next(s for s in shifts if "High-Anomaly" in s["metric_name"])
    assert anom_shift["baseline_value"] == 20.0
    assert anom_shift["recent_value"] == 60.0
    assert anom_shift["absolute_change"] == 40.0
    assert anom_shift["relative_change_pct"] == 200.0
    assert anom_shift["severity"] == ShiftSeverity.CRITICAL.value
    assert anom_shift["is_significant"] is True


# =============================================================================
# PYDANTIC SCHEMA VALIDATION TESTS
# =============================================================================

def test_pattern_schemas_valid():
    """Verify validation of Pydantic models for cohort metrics, shifts, and reports."""
    cohort = TimeWindowCohort(
        window_name="Recent Window",
        alert_count=50,
        mean_amount=320.50,
        median_amount=210.00,
        p90_amount=850.00,
        mean_fraud_probability=0.72,
        high_fraud_share_pct=42.0,
        mean_anomaly_score=0.68,
        high_anomaly_share_pct=38.0,
        tier_distribution={"CRITICAL": 25, "HIGH": 15, "MEDIUM": 10},
        quadrant_distribution={"CONFIRMED_HIGH_RISK": 30, "SUBTLE_IN_DISTRIBUTION_FRAUD": 20},
        verified_outcomes_count=20,
        confirmed_fraud_count=16,
        false_positive_count=4,
        false_positive_rate=0.20,
        precision=0.80,
    )
    assert cohort.alert_count == 50
    assert cohort.precision == 0.80

    shift = DetectedPatternShift(
        metric_name="High-Anomaly Alert Share",
        baseline_value=15.0,
        recent_value=45.0,
        absolute_change=30.0,
        relative_change_pct=200.0,
        severity=ShiftSeverity.CRITICAL.value,
        is_significant=True,
        shift_summary="Tripling of high anomaly transactions.",
    )
    assert shift.severity == "CRITICAL"

    finding = PatternFindingItem(
        pattern_title="Concentration of High-Anomaly Latent Vectors",
        category=PatternCategory.ANOMALY_CONCENTRATION.value,
        measured_pattern="High anomaly alerts shifted from 15.0% to 45.0% (+200.0% relative).",
        supporting_statistics={"baseline": 15.0, "recent": 45.0},
        statistical_confidence=StatisticalConfidence.HIGH.value,
        dataset_limitations="European Credit Card dataset contains anonymized PCA features V1-V28.",
        recommended_followup="Adjust triage composite urgency threshold and review Isolation Forest bounds.",
    )
    assert finding.category == "ANOMALY_CONCENTRATION"

    report = PatternAnalysisReport(
        report_id="PAT-TEST0001",
        executive_summary="Executive synthesis of emerging anomaly concentration across historical windows.",
        baseline_window=cohort,
        recent_window=cohort,
        detected_shifts=[shift],
        findings=[finding],
        model_error_analysis="False positive rate decreased significantly.",
        dataset_limitations_acknowledged=True,
        grounding_audit_passed=True,
    )
    assert report.report_id == "PAT-TEST0001"
    assert len(report.findings) == 1


def test_pattern_schema_invalid_category():
    """Verify validation fails on unsupported pattern categories."""
    with pytest.raises(ValueError):
        PatternFindingItem(
            pattern_title="Invalid Test",
            category="UNSUPPORTED_CATEGORY",
            measured_pattern="Measured pattern statement.",
            statistical_confidence="HIGH",
            dataset_limitations="Anonymized PCA features V1-V28.",
            recommended_followup="Actionable follow-up recommendation.",
        )


# =============================================================================
# GROUNDING AUDIT & ANTI-HALLUCINATION TESTS
# =============================================================================

def test_evaluate_pattern_grounding_clean(sample_historical_records: List[Dict[str, Any]]):
    """Verify that a compliant report citing exact numbers and anonymized PCA constraints passes audit."""
    b_stats = compute_cohort_statistics(sample_historical_records[:10], window_name="Baseline Window")
    r_stats = compute_cohort_statistics(sample_historical_records[10:], window_name="Recent Window")

    b_cohort = TimeWindowCohort(**{k: v for k, v in b_stats.items() if k in TimeWindowCohort.model_fields})
    r_cohort = TimeWindowCohort(**{k: v for k, v in r_stats.items() if k in TimeWindowCohort.model_fields})

    clean_report = {
        "report_id": "PAT-CLEAN01",
        "executive_summary": "Historical analysis across 20 alerts indicates an increase in high anomaly transactions.",
        "baseline_window": b_cohort.model_dump(),
        "recent_window": r_cohort.model_dump(),
        "detected_shifts": [],
        "findings": [
            {
                "pattern_title": "Surge in Anomaly Score Latent Components",
                "category": "ANOMALY_CONCENTRATION",
                "measured_pattern": "High anomaly alerts rose from 20.0% to 60.0% (+200.0% relative).",
                "supporting_statistics": {"baseline": 20.0, "recent": 60.0},
                "statistical_confidence": "HIGH",
                "dataset_limitations": "European Credit Card dataset contains anonymized PCA features V1-V28.",
                "recommended_followup": "Review unsupervised Isolation Forest threshold and queue prioritization.",
            }
        ],
        "model_error_analysis": "Verified false positive rate declined from 80.0% to 40.0%.",
        "dataset_limitations_acknowledged": True,
        "grounding_audit_passed": True,
    }

    audit = evaluate_pattern_grounding(
        pattern_report=clean_report,
        computed_stats={"baseline_cohort": b_stats, "recent_cohort": r_stats},
    )
    assert audit["grounding_passed"] is True
    assert audit["hallucination_freedom_score"] == 1.0
    assert audit["numeric_fidelity_score"] == 1.0
    assert audit["overall_score"] >= 0.95


def test_evaluate_pattern_grounding_detects_prohibited_storytelling():
    """Verify that claims inventing merchant identities, locations, or fraud rings are caught and penalized."""
    hallucinated_report = {
        "report_id": "PAT-HALLUC01",
        "executive_summary": "We discovered a fraud ring operating across gas stations with foreign IP addresses.",
        "baseline_window": {"alert_count": 10, "mean_amount": 100.0, "high_anomaly_share_pct": 20.0},
        "recent_window": {"alert_count": 10, "mean_amount": 400.0, "high_anomaly_share_pct": 60.0},
        "findings": [
            {
                "pattern_title": "Cardholder Traveling to Overseas Store",
                "category": "ANOMALY_CONCENTRATION",
                "measured_pattern": "Transactions at particular merchant category codes indicate organized crime syndicates.",
                "supporting_statistics": {"amount": 400.0},
                "statistical_confidence": "HIGH",
                "dataset_limitations": "PCA features V1-V28 represent merchant store locations.",
                "recommended_followup": "Block merchant category code.",
            }
        ],
        "dataset_limitations_acknowledged": False,
    }

    audit = evaluate_pattern_grounding(
        pattern_report=hallucinated_report,
        computed_stats={
            "baseline_cohort": {"alert_count": 10, "mean_amount": 100.0, "high_anomaly_share_pct": 20.0},
            "recent_cohort": {"alert_count": 10, "mean_amount": 400.0, "high_anomaly_share_pct": 60.0},
        },
    )

    assert audit["grounding_passed"] is False
    assert audit["hallucination_freedom_score"] < 1.0
    assert any("prohibited domain narrative" in issue for issue in audit["issues"])


# =============================================================================
# LANGGRAPH WORKFLOW EXECUTION TESTS
# =============================================================================

def test_build_pattern_analysis_graph():
    """Verify LangGraph compilation and structure for the Pattern Analysis Agent."""
    workflow = build_pattern_analysis_graph()
    assert workflow is not None

    compiled = get_pattern_analysis_graph()
    assert compiled is not None


def test_save_pattern_graph_image_if_needed(tmp_path):
    """Verify state graph visualization PNG is generated and written to disk."""
    out_path = tmp_path / "test_pattern_graph.png"
    saved = save_pattern_graph_image_if_needed(output_path=out_path)
    assert saved == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0



def test_graph_node_sequential_flow(sample_historical_records: List[Dict[str, Any]]):
    """Verify sequential node data transformations through gather, compute, detect, and validate."""
    # 1. Gather node
    state: PatternAnalysisState = {
        "messages": [],
        "time_horizon_hours": None,
        "baseline_ratio": 0.5,
        "min_alerts_threshold": 5,
        "model_name": DEFAULT_AGENT_MODEL,
        "raw_historical_records": sample_historical_records,
        "baseline_records": sample_historical_records[:10],
        "recent_records": sample_historical_records[10:],
        "time_windows_metadata": {
            "total_historical_records": 20,
            "baseline_count": 10,
            "recent_count": 10,
        },
        "computed_statistics": {},
        "detected_shifts": [],
        "structured_report": None,
        "validation_passed": False,
        "validation_errors": [],
        "validation_attempts": 0,
        "published_report": None,
    }

    # 2. Compute statistics node
    stats_out = node_compute_pattern_statistics(state)
    assert "computed_statistics" in stats_out
    state["computed_statistics"] = stats_out["computed_statistics"]

    # 3. Detect anomalous trends node
    shifts_out = node_detect_anomalous_trends(state)
    assert "detected_shifts" in shifts_out
    assert len(shifts_out["detected_shifts"]) > 0
    state["detected_shifts"] = shifts_out["detected_shifts"]

    # 4. Validate evidence node on a clean synthetic report
    state["structured_report"] = {
        "report_id": "PAT-NODE-TEST",
        "executive_summary": "Historical analysis across evaluated window cohorts.",
        "baseline_window": state["computed_statistics"]["baseline_cohort"],
        "recent_window": state["computed_statistics"]["recent_cohort"],
        "detected_shifts": state["detected_shifts"],
        "findings": [
            {
                "pattern_title": "Surge in High-Anomaly Latent Components",
                "category": "ANOMALY_CONCENTRATION",
                "measured_pattern": "High anomaly alerts rose from 20.0% to 60.0%.",
                "supporting_statistics": {"shift": 40.0},
                "statistical_confidence": "HIGH",
                "dataset_limitations": "European Credit Card dataset contains anonymized PCA features V1-V28.",
                "recommended_followup": "Review unsupervised Isolation Forest threshold.",
            }
        ],
        "dataset_limitations_acknowledged": True,
    }

    val_out = node_validate_evidence(state)
    assert val_out["validation_passed"] is True
    assert len(val_out["validation_errors"]) == 0
    state["validation_passed"] = True

    # 5. Publish node
    pub_out = node_publish_analysis_report(state)
    assert "published_report" in pub_out
    assert pub_out["published_report"]["report_id"] == "PAT-NODE-TEST"
    assert pub_out["published_report"]["grounding_audit_passed"] is True


def test_run_pattern_analysis_end_to_end():
    """Verify full end-to-end execution of the pattern analysis workflow with existing database records."""
    report = run_pattern_analysis(min_alerts_threshold=5)
    assert report is not None
    assert "report_id" in report
    assert "baseline_window" in report
    assert "recent_window" in report
    assert "findings" in report
    assert len(report["findings"]) >= 1
    assert report.get("dataset_limitations_acknowledged") is True
    assert "models_used" in report
    assert report["models_used"]["pipeline_mode"] == "Distributed Multi-LLM"
    assert report["models_used"]["cohort_model"] == PATTERN_COHORT_MODEL
    assert report["models_used"]["findings_model"] == PATTERN_FINDINGS_MODEL
    assert report["models_used"]["executive_model"] == PATTERN_EXECUTIVE_MODEL


def test_distributed_multi_llm_subtasks(sample_historical_records: List[Dict[str, Any]]):
    """Verify modular distributed sub-task execution with specialized model assignments."""
    b_stats = compute_cohort_statistics(sample_historical_records[:10], window_name="Baseline")
    r_stats = compute_cohort_statistics(sample_historical_records[10:], window_name="Recent")
    b_cohort = TimeWindowCohort(**{k: v for k, v in b_stats.items() if k in TimeWindowCohort.model_fields})
    r_cohort = TimeWindowCohort(**{k: v for k, v in r_stats.items() if k in TimeWindowCohort.model_fields})
    shifts = detect_statistical_shifts(baseline=b_stats, recent=r_stats)

    # 1. Sub-task 1: Cohort Trends and Verified Error Rate Analysis
    insights, error_analysis = synthesize_cohort_and_error_trends(b_cohort, r_cohort, model_name=PATTERN_COHORT_MODEL)
    assert len(insights) > 10
    assert len(error_analysis) > 10

    # 2. Sub-task 2: Structured Pattern Findings Formulation
    findings = formulate_pattern_findings(shifts, b_stats, r_stats, model_name=PATTERN_FINDINGS_MODEL)
    assert len(findings) >= 1
    assert all(isinstance(f, PatternFindingItem) for f in findings)
    assert all("V1-V28" in f.dataset_limitations or "PCA" in f.dataset_limitations for f in findings)

    # 3. Sub-task 3: Executive Leadership Brief Synthesis
    exec_summary = synthesize_executive_summary(
        cohort_insights=insights,
        error_analysis=error_analysis,
        findings=findings,
        b_cohort=b_cohort,
        r_cohort=r_cohort,
        model_name=PATTERN_EXECUTIVE_MODEL,
    )
    assert len(exec_summary) >= 25



# =============================================================================
# FASTAPI ENDPOINT TESTS
# =============================================================================

def test_api_run_pattern_analysis_endpoint():
    """Verify POST /agent/pattern-analysis/run executes and returns structured findings."""
    response = client.post(
        "/agent/pattern-analysis/run",
        json={
            "time_horizon_hours": None,
            "baseline_ratio": 0.5,
            "min_alerts_threshold": 5,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "report_id" in data
    assert "findings" in data["data"]


def test_api_get_latest_pattern_report_endpoint():
    """Verify GET /agent/pattern-analysis/latest retrieves published report."""
    response = client.get("/agent/pattern-analysis/latest")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "report_id" in data["data"]


def test_api_evaluate_pattern_report_endpoint():
    """Verify POST /agent/pattern-analysis/evaluate audits report grounding."""
    # First get latest report
    get_res = client.get("/agent/pattern-analysis/latest")
    latest_report = get_res.json()["data"]

    eval_res = client.post(
        "/agent/pattern-analysis/evaluate",
        json={"report": latest_report},
    )
    assert eval_res.status_code == 200
    eval_data = eval_res.json()
    assert eval_data["status"] == "success"
    assert "audit" in eval_data
    assert "grounding_passed" in eval_data["audit"]
