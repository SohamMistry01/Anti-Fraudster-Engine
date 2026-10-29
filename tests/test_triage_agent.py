"""Comprehensive automated tests for Fraud Triage & Prioritization Agent.

Verifies:
- Deterministic multi-signal prioritization, scoring, and tier categorization
- Investigator workload calculation and capacity-aware matching
- Pydantic V2 schemas (AlertTriageItem, QueueOverview, TriageBatchReport, etc.)
- Triage grounding evaluation, monotonicity verification, and domain hallucination audit
- LangGraph stateful triage graph compilation and node transitions
- Human-in-the-loop supervisor approval checkpoint and batch decision application
- FastAPI endpoints for triage run, decision application, and workload tracking
"""

from typing import Any, Dict, List
import pytest
from fastapi.testclient import TestClient

from src.agents.config import DEFAULT_AGENT_MODEL, SUPPORTED_MODELS
from src.agents.evaluator import evaluate_triage_grounding
from src.agents.schemas import (
    AlertTriageItem,
    InvestigatorWorkload,
    QueueOverview,
    TriageAction,
    TriageBatchReport,
    TriageDecisionApplication,
    TriagePriorityTier,
)
from src.agents.state import TriageState
from src.agents.tools import (
    DEFAULT_INVESTIGATOR_ROSTER,
    get_investigator_workloads,
    get_pending_alerts_for_triage,
)
from src.agents.triage_agent import (
    apply_triage_decisions,
    build_triage_graph,
    compute_deterministic_triage,
    get_triage_graph,
    match_investigator_for_alert,
    node_checkpoint_triage_approval,
    node_deterministic_ranking,
    node_gather_triage_queue,
    node_validate_triage_grounding,
)
from src.api.main import app
from src.cases.database import get_db_context
from src.cases.models import AlertPriority, AlertStatus, CaseStatus
from src.cases.repository import AlertRepository, CaseRepository
from src.cases.service import get_case_service

client = TestClient(app)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def setup_triage_test_alerts() -> List[str]:
    """Create a set of diverse alerts with varied risk profiles for triage testing."""
    created_alert_ids = []
    with get_db_context() as db:
        # Alert 1: Critical high-dollar, high-probability fraud
        a1 = AlertRepository.create(db, {
            "amount": 1450.00,
            "time_offset": 500.0,
            "fraud_probability": 0.945,
            "anomaly_score": 0.880,
            "risk_category": "CRITICAL",
            "priority": AlertPriority.CRITICAL.value,
            "operational_quadrant": "CONFIRMED_HIGH_RISK",
            "status": AlertStatus.NEW.value,
            "reasons": ["Extreme fraud probability", "Statistical population outlier"],
            "transaction_payload": {
                **{"Time": 500.0, "Amount": 1450.0},
                **{f"V{i}": (-5.2 if i in [10, 12, 14, 17] else 0.05) for i in range(1, 29)},
            },
        })
        created_alert_ids.append(a1.alert_id)

        # Alert 2: Moderate fraud probability, elevated anomaly
        a2 = AlertRepository.create(db, {
            "amount": 320.00,
            "time_offset": 1200.0,
            "fraud_probability": 0.520,
            "anomaly_score": 0.720,
            "risk_category": "HIGH",
            "priority": AlertPriority.HIGH.value,
            "operational_quadrant": "NOVEL_ANOMALY_OUTLIER",
            "status": AlertStatus.NEW.value,
            "reasons": ["Elevated anomaly score"],
            "transaction_payload": {
                **{"Time": 1200.0, "Amount": 320.0},
                **{f"V{i}": (-1.2 if i == 14 else 1.5 if i == 4 else 0.02) for i in range(1, 29)},
            },
        })
        created_alert_ids.append(a2.alert_id)

        # Alert 3: Low probability, benign
        a3 = AlertRepository.create(db, {
            "amount": 25.50,
            "time_offset": 3000.0,
            "fraud_probability": 0.020,
            "anomaly_score": 0.150,
            "risk_category": "LOW",
            "priority": AlertPriority.LOW.value,
            "operational_quadrant": "CONFIRMED_LEGITIMATE",
            "status": AlertStatus.NEW.value,
            "reasons": ["Routine low risk transaction"],
            "transaction_payload": {
                **{"Time": 3000.0, "Amount": 25.5},
                **{f"V{i}": 0.01 for i in range(1, 29)},
            },
        })
        created_alert_ids.append(a3.alert_id)

    return created_alert_ids


# =============================================================================
# 1. DETERMINISTIC PRIORITIZATION & SCORING TESTS
# =============================================================================

def test_compute_deterministic_triage_critical():
    """Verify deterministic scoring maps high-probability high-dollar alert to TIER_1_CRITICAL."""
    alert_mock = {
        "alert_id": "ALT-TEST-001",
        "fraud_probability": 0.92,
        "anomaly_score": 0.85,
        "amount": 1200.0,
        "priority": "CRITICAL",
        "operational_quadrant": "CONFIRMED_HIGH_RISK",
        "case_id": None,
        "reasons": ["High probability", "Large amount"],
    }
    result = compute_deterministic_triage(alert_mock)

    assert result["alert_id"] == "ALT-TEST-001"
    assert result["priority_tier"] == TriagePriorityTier.TIER_1_CRITICAL.value
    assert result["composite_urgency_score"] >= 75.0
    assert result["recommended_action"] == TriageAction.ESCALATE_TO_CASE.value


def test_compute_deterministic_triage_low_risk():
    """Verify benign alert is categorized into TIER_4_LOW with DISMISS or MONITOR action."""
    alert_mock = {
        "alert_id": "ALT-TEST-002",
        "fraud_probability": 0.015,
        "anomaly_score": 0.12,
        "amount": 15.0,
        "priority": "LOW",
        "operational_quadrant": "CONFIRMED_LEGITIMATE",
        "case_id": None,
        "reasons": [],
    }
    result = compute_deterministic_triage(alert_mock)

    assert result["priority_tier"] == TriagePriorityTier.TIER_4_LOW.value
    assert result["composite_urgency_score"] < 25.0
    assert result["recommended_action"] in [TriageAction.DISMISS_FALSE_POSITIVE.value, TriageAction.MONITOR_ONLY.value]


def test_compute_deterministic_triage_existing_case():
    """Verify alert with linked case is flagged with PRIORITIZE_CASE_INVESTIGATION."""
    alert_mock = {
        "alert_id": "ALT-TEST-003",
        "fraud_probability": 0.65,
        "anomaly_score": 0.60,
        "amount": 200.0,
        "priority": "HIGH",
        "case_id": "CASE-ACTIVE-01",
        "reasons": ["Elevated metrics"],
    }
    result = compute_deterministic_triage(alert_mock)
    assert result["recommended_action"] == TriageAction.PRIORITIZE_CASE_INVESTIGATION.value


# =============================================================================
# 2. WORKLOAD-AWARE ASSIGNMENT TESTS
# =============================================================================

def test_get_investigator_workloads():
    """Verify get_investigator_workloads retrieves roster with live capacity ratios."""
    roster = get_investigator_workloads.invoke({})
    assert isinstance(roster, list)
    assert len(roster) >= 4

    for inv in roster:
        assert "investigator_id" in inv
        assert "name" in inv
        assert "role" in inv
        assert "active_cases" in inv
        assert "max_capacity" in inv
        assert "capacity_utilization" in inv
        assert inv["capacity_utilization"] >= 0.0


def test_match_investigator_for_alert():
    """Verify workload-aware assignment matches critical tier alerts to senior specialists."""
    roster = [
        {
            "investigator_id": "senior_specialist",
            "name": "Senior Lead",
            "role": "Senior Fraud Specialist",
            "specialties": ["TIER_1_CRITICAL"],
            "active_cases": 2,
            "max_capacity": 10,
            "capacity_utilization": 0.20,
            "is_available": True,
        },
        {
            "investigator_id": "junior_analyst",
            "name": "Junior Dev",
            "role": "Junior Triage Analyst",
            "specialties": ["TIER_4_LOW"],
            "active_cases": 1,
            "max_capacity": 15,
            "capacity_utilization": 0.07,
            "is_available": True,
        },
    ]

    ranking = {
        "alert_id": "ALT-TEST-MATCH",
        "priority_tier": TriagePriorityTier.TIER_1_CRITICAL.value,
        "operational_quadrant": "CONFIRMED_HIGH_RISK",
        "fraud_probability": 0.95,
    }

    inv_id, rationale = match_investigator_for_alert(ranking, roster)
    assert inv_id == "senior_specialist"
    assert "Senior Lead" in rationale
    assert "TIER_1_CRITICAL" in rationale


# =============================================================================
# 3. SCHEMAS VALIDATION TESTS
# =============================================================================

def test_alert_triage_item_schema():
    """Verify AlertTriageItem schema validates strictly and normalizes enums."""
    item = AlertTriageItem(
        alert_id="ALT-12345678",
        rank=1,
        composite_urgency_score=88.5,
        priority_tier="TIER_1_CRITICAL",
        fraud_probability=0.912,
        anomaly_score=0.835,
        amount=750.0,
        policy_reasons=["High probability cutoff"],
        triage_explanation="Ranked #1 due to critical fraud probability (91.2%) and severe anomaly score.",
        recommended_action="ESCALATE_TO_CASE",
        recommended_investigator="analyst_mistry",
        assignment_rationale="Matched to senior specialist.",
    )
    assert item.rank == 1
    assert item.priority_tier == TriagePriorityTier.TIER_1_CRITICAL.value
    assert item.recommended_action == TriageAction.ESCALATE_TO_CASE.value


def test_triage_batch_report_schema():
    """Verify complete TriageBatchReport schema instantiation."""
    overview = QueueOverview(
        total_alerts_evaluated=2,
        tier_distribution={"TIER_1_CRITICAL": 1, "TIER_2_HIGH": 1},
        queue_risk_synthesis="Queue presents high threat concentration in critical financial tiers.",
        workload_bottlenecks="Adequate investigator capacity available.",
    )
    report = TriageBatchReport(
        batch_id="BATCH-TEST-001",
        queue_overview=overview,
        prioritized_queue=[],
        grounding_audit_passed=True,
    )
    assert report.batch_id == "BATCH-TEST-001"
    assert report.queue_overview.total_alerts_evaluated == 2


# =============================================================================
# 4. GROUNDING EVALUATOR TESTS
# =============================================================================

def test_evaluate_triage_grounding_clean():
    """Verify evaluate_triage_grounding approves clean, monotonic, story-free report."""
    mock_report = {
        "queue_overview": {
            "queue_risk_synthesis": "Queue exhibits elevated statistical deviations across top alerts.",
            "workload_bottlenecks": "Investigators operating within optimal bandwidth limits.",
        },
        "prioritized_queue": [
            {
                "alert_id": "ALT-1",
                "rank": 1,
                "composite_urgency_score": 90.0,
                "fraud_probability": 0.90,
                "anomaly_score": 0.80,
                "triage_explanation": "Ranked #1 with statistical probability 0.90.",
            },
            {
                "alert_id": "ALT-2",
                "rank": 2,
                "composite_urgency_score": 60.0,
                "fraud_probability": 0.50,
                "anomaly_score": 0.60,
                "triage_explanation": "Ranked #2 with statistical probability 0.50.",
            },
        ],
    }

    true_alerts = [
        {"alert_id": "ALT-1", "fraud_probability": 0.90, "anomaly_score": 0.80},
        {"alert_id": "ALT-2", "fraud_probability": 0.50, "anomaly_score": 0.60},
    ]

    audit = evaluate_triage_grounding(mock_report, true_alerts)
    assert audit["grounding_passed"] is True
    assert audit["ranking_monotonicity_score"] == 1.0
    assert audit["hallucination_freedom_score"] == 1.0
    assert audit["overall_score"] >= 0.95


def test_evaluate_triage_grounding_detects_prohibited_storytelling():
    """Verify grounding audit catches and penalizes unverified PCA domain storytelling."""
    mock_report = {
        "queue_overview": {
            "queue_risk_synthesis": "Alerts triggered due to foreign IP address and stolen card at a gas station.",
            "workload_bottlenecks": "None",
        },
        "prioritized_queue": [
            {
                "alert_id": "ALT-1",
                "rank": 1,
                "composite_urgency_score": 90.0,
                "fraud_probability": 0.90,
                "anomaly_score": 0.80,
                "triage_explanation": "Transaction occurred at an electronics shop with international IP.",
            },
        ],
    }

    true_alerts = [{"alert_id": "ALT-1", "fraud_probability": 0.90, "anomaly_score": 0.80}]
    audit = evaluate_triage_grounding(mock_report, true_alerts)

    assert audit["grounding_passed"] is False
    assert audit["hallucination_freedom_score"] < 1.0
    assert any("prohibited domain storytelling" in issue.lower() for issue in audit["issues"])


# =============================================================================
# 5. LANGGRAPH WORKFLOW & NODE TESTS
# =============================================================================

def test_build_triage_graph():
    """Verify LangGraph triage workflow compiles with all expected nodes."""
    graph = get_triage_graph()
    assert graph is not None

    nodes = graph.get_graph().nodes
    assert "gather_triage_queue" in nodes
    assert "deterministic_ranking" in nodes
    assert "generate_triage_explanations" in nodes
    assert "validate_triage_grounding" in nodes
    assert "revise_triage" in nodes
    assert "checkpoint_approval" in nodes


def test_graph_node_deterministic_ranking():
    """Verify deterministic ranking node orders items descending by urgency score."""
    raw = [
        {"alert_id": "ALT-LOW", "fraud_probability": 0.1, "anomaly_score": 0.2, "amount": 10.0, "priority": "LOW"},
        {"alert_id": "ALT-HIGH", "fraud_probability": 0.9, "anomaly_score": 0.8, "amount": 900.0, "priority": "CRITICAL"},
    ]
    state: TriageState = {
        "messages": [],
        "filter_status": "NEW",
        "max_alerts": 10,
        "include_workload": True,
        "model_name": DEFAULT_AGENT_MODEL,
        "raw_alerts": raw,
        "investigators": DEFAULT_INVESTIGATOR_ROSTER,
        "deterministic_rankings": [],
        "structured_batch_report": None,
        "validation_passed": False,
        "validation_errors": [],
        "validation_attempts": 0,
        "approval_status": "PENDING",
        "final_decisions": None,
    }

    result = node_deterministic_ranking(state)
    rankings = result["deterministic_rankings"]

    assert len(rankings) == 2
    assert rankings[0]["alert_id"] == "ALT-HIGH"
    assert rankings[0]["rank"] == 1
    assert rankings[1]["alert_id"] == "ALT-LOW"
    assert rankings[1]["rank"] == 2
    assert rankings[0]["composite_urgency_score"] > rankings[1]["composite_urgency_score"]


# =============================================================================
# 6. HUMAN OVERSIGHT & DECISION APPLICATION TESTS
# =============================================================================

def test_apply_triage_decisions(setup_triage_test_alerts: List[str]):
    """Verify supervisor triage decisions escalate alerts to cases and dismiss benign alerts."""
    critical_id = setup_triage_test_alerts[0]
    low_id = setup_triage_test_alerts[2]

    decisions = [
        {
            "alert_id": critical_id,
            "action": TriageAction.ESCALATE_TO_CASE.value,
            "assigned_investigator": "analyst_mistry",
            "supervisor_notes": "Supervisor approved escalation to Soham Mistry.",
        },
        {
            "alert_id": low_id,
            "action": TriageAction.DISMISS_FALSE_POSITIVE.value,
            "supervisor_notes": "Supervisor dismissed benign low value alert.",
        },
    ]

    res = apply_triage_decisions(decisions=decisions, supervisor_id="lead_supervisor")

    assert res["applied_count"] == 2
    assert len(res["cases_created"]) == 1
    assert res["cases_created"][0]["alert_id"] == critical_id
    assert res["cases_created"][0]["assigned_analyst"] == "analyst_mistry"

    # Verify database state
    with get_db_context() as db:
        alt_low = AlertRepository.get_by_id(db, low_id)
        assert alt_low.status == AlertStatus.DISMISSED.value

        alt_crit = AlertRepository.get_by_id(db, critical_id)
        assert alt_crit.status == AlertStatus.CASE_CREATED.value
        assert alt_crit.case_id is not None


# =============================================================================
# 7. FASTAPI API ROUTE ENDPOINTS TESTS
# =============================================================================

def test_api_get_workloads_endpoint():
    """Verify GET /agent/triage/workloads returns investigator workloads."""
    response = client.get("/agent/triage/workloads")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "investigators" in data
    assert len(data["investigators"]) >= 4


def test_api_apply_triage_decisions_endpoint(setup_triage_test_alerts: List[str]):
    """Verify POST /agent/triage/apply processes supervisor decisions."""
    alert_id = setup_triage_test_alerts[1]
    payload = {
        "supervisor_id": "test_supervisor",
        "decisions": [
            {
                "alert_id": alert_id,
                "action": "ESCALATE_TO_CASE",
                "assigned_investigator": "lead_analyst",
                "supervisor_notes": "API test escalation.",
            }
        ],
    }
    response = client.post("/agent/triage/apply", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "success"
    assert res_data["results"]["applied_count"] == 1
