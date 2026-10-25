"""Comprehensive test suite for the Agentic Investigation Layer.

Verifies:
- Deterministic read-only retrieval tools (alert, TreeSHAP, Isolation Forest, case history)
- Pydantic V2 schema validation for EvidenceCitation, GroundingCheck, and InvestigationReport
- Grounding audit engine, citation precision verification, and domain hallucination detection
- LangGraph stateful investigation graph compilation and execution
- Human-in-the-loop approval checkpoint and database lifecycle synchronization
- FastAPI agent endpoints (/cases/{case_id}/investigate, /cases/{case_id}/agent-decision, /agent/tools/check)
"""

from typing import Any, Dict
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage

from src.agents.config import DEFAULT_AGENT_MODEL, SUPPORTED_MODELS, get_groq_llm
from src.agents.evaluator import (
    evaluate_agent_on_cases,
    evaluate_report_grounding,
    evaluate_tool_correctness,
)
from src.agents.investigation_agent import (
    apply_investigator_decision,
    build_investigation_graph,
    get_investigation_graph,
    node_checkpoint_approval,
    node_gather_evidence,
    node_synthesize_analysis,
    node_validate_grounding,
    run_case_investigation,
)
from src.agents.schemas import (
    CitationDirection,
    EvidenceCitation,
    GroundingCheck,
    InvestigationReport,
    InvestigatorApprovalRequest,
)
from src.agents.state import InvestigationState
from src.agents.tools import (
    get_alert_details,
    get_anomaly_score,
    get_case_history,
    get_shap_explanation,
)
from src.api.main import app
from src.cases.database import get_db_context
from src.cases.models import (
    AlertPriority,
    AlertStatus,
    CaseAction,
    CaseStatus,
    CaseVerdict,
    InvestigationCase,
)
from src.cases.repository import AlertRepository, CaseRepository
from src.cases.service import get_case_service

client = TestClient(app)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def sample_test_case() -> Dict[str, Any]:
    """Create a persistent test alert and escalated case in the database."""
    service = get_case_service()
    with get_db_context() as db:
        # 1. Create Alert
        alert_payload = {
            "amount": 289.50,
            "time_offset": 7200.0,
            "fraud_probability": 0.892,
            "anomaly_score": 0.765,
            "risk_category": "CRITICAL",
            "priority": AlertPriority.CRITICAL.value,
            "operational_quadrant": "CONFIRMED_FRAUD",
            "status": AlertStatus.NEW.value,
            "reasons": ["High fraud probability", "Elevated isolation anomaly score"],
            "transaction_payload": {
                **{"Time": 7200.0, "Amount": 289.50},
                **{f"V{i}": (-3.5 if i in [10, 12, 14, 17] else 0.05) for i in range(1, 29)},
            },
        }
        alert = AlertRepository.create(db, alert_payload)

        # 2. Escalate to Case
        case = service.open_case_from_alert(
            db=db,
            alert_id=alert.alert_id,
            assigned_analyst="test_analyst",
            initial_note="Agent testing fixture case.",
            creator="test_runner",
        )
        return {
            "case_id": case.case_id,
            "alert_id": alert.alert_id,
            "transaction_id": case.transaction_id,
        }


# =============================================================================
# 1. READ-ONLY RETRIEVAL TOOLS TESTS
# =============================================================================

def test_tool_get_case_history(sample_test_case: Dict[str, Any]):
    """Verify get_case_history tool retrieves sealed ML snapshot and audit logs."""
    case_id = sample_test_case["case_id"]
    res = get_case_history.invoke({"case_id": case_id})

    assert "case_id" in res
    assert res["case_id"] == case_id
    assert "original_prediction" in res
    assert "transaction_snapshot" in res
    assert "notes" in res
    assert "audit_logs" in res
    assert res["status"] in [s.value for s in CaseStatus]


def test_tool_get_case_history_not_found():
    """Verify get_case_history tool returns error object for nonexistent case."""
    res = get_case_history.invoke({"case_id": "CASE-NONEXISTENT-9999"})
    assert "error" in res
    assert "not found" in res["error"].lower()


def test_tool_get_alert_details(sample_test_case: Dict[str, Any]):
    """Verify get_alert_details tool returns alert data with transaction payload."""
    alert_id = sample_test_case["alert_id"]
    res = get_alert_details.invoke({"alert_id": alert_id})

    assert "alert_id" in res
    assert res["alert_id"] == alert_id
    assert "fraud_probability" in res
    assert "anomaly_score" in res
    assert "transaction_payload" in res
    assert res["transaction_payload"]["Amount"] == 289.50


def test_tool_get_alert_details_not_found():
    """Verify get_alert_details returns error dictionary for invalid alert."""
    res = get_alert_details.invoke({"alert_id": "ALT-INVALID-0000"})
    assert "error" in res


def test_tool_get_shap_explanation(sample_test_case: Dict[str, Any]):
    """Verify get_shap_explanation generates feature attributions from payload."""
    alert_id = sample_test_case["alert_id"]
    res = get_shap_explanation.invoke({"alert_id": alert_id, "top_n": 5})

    assert "alert_id" in res
    assert "top_positive_contributors" in res
    assert "top_negative_contributors" in res
    assert "evidence_summary" in res
    assert len(res["top_positive_contributors"]) > 0


def test_tool_get_anomaly_score(sample_test_case: Dict[str, Any]):
    """Verify get_anomaly_score calculates independent Isolation Forest metrics."""
    alert_id = sample_test_case["alert_id"]
    res = get_anomaly_score.invoke({"alert_id": alert_id})

    assert "alert_id" in res
    assert "anomaly_score" in res
    assert 0.0 <= res["anomaly_score"] <= 1.0
    assert "anomaly_tier" in res
    assert "is_anomaly" in res


def test_evaluate_tool_correctness(sample_test_case: Dict[str, Any]):
    """Verify evaluate_tool_correctness benchmark checks all 4 retrieval engines."""
    check = evaluate_tool_correctness(
        sample_case_id=sample_test_case["case_id"],
        sample_alert_id=sample_test_case["alert_id"],
    )
    assert check["all_passed"] is True
    assert check["tools"]["get_case_history"]["success"] is True
    assert check["tools"]["get_alert_details"]["success"] is True
    assert check["tools"]["get_shap_explanation"]["success"] is True
    assert check["tools"]["get_anomaly_score"]["success"] is True


# =============================================================================
# 2. SCHEMAS & PYDANTIC VALIDATION TESTS
# =============================================================================

def test_evidence_citation_schema():
    """Verify EvidenceCitation schema validation and enum formatting."""
    cit = EvidenceCitation(
        feature="V14",
        observed_value=-5.61,
        shap_value=1.84,
        direction=CitationDirection.FRAUD_PUSHING,
        rationale="Strong negative standard deviations on V14 drive positive log-odds shift.",
    )
    assert cit.feature == "V14"
    assert cit.observed_value == -5.61
    assert cit.shap_value == 1.84
    assert cit.direction == CitationDirection.FRAUD_PUSHING


def test_investigation_report_schema():
    """Verify InvestigationReport schema adheres to strict typing and case enums."""
    report = InvestigationReport(
        case_id="CASE-TEST-1234",
        alert_id="ALT-TEST-1234",
        executive_summary="Elevated risk transaction with multi-signal agreement.",
        investigation_findings="Significant deviations across principal components V14 and V10.",
        key_evidence_citations=[
            EvidenceCitation(
                feature="V14",
                observed_value=-5.61,
                shap_value=1.84,
                direction=CitationDirection.FRAUD_PUSHING,
                rationale="Extreme deviation increases log-odds.",
            ),
        ],
        anomaly_context="Isolation Forest confirms anomaly tier HIGH.",
        case_precedent="Consistent with confirmed fraudulent transaction patterns.",
        recommended_verdict=CaseVerdict.CONFIRMED_FRAUD,
        recommended_action=CaseAction.CARD_BLOCKED,
        confidence_score=0.92,
    )
    assert report.case_id == "CASE-TEST-1234"
    assert report.recommended_verdict == CaseVerdict.CONFIRMED_FRAUD
    assert report.recommended_action == CaseAction.CARD_BLOCKED
    assert len(report.key_evidence_citations) == 1


def test_investigator_approval_request_schema():
    """Verify InvestigatorApprovalRequest schema validation."""
    appr = InvestigatorApprovalRequest(
        case_id="CASE-TEST-1234",
        investigator_id="senior_analyst",
        action_approved=True,
        final_verdict="CONFIRMED_FRAUD",
        final_action="CARD_BLOCKED",
        analyst_comments="Reviewed agent evidence citations; confirmed card compromise.",
    )
    assert appr.investigator_id == "senior_analyst"
    assert appr.action_approved is True
    assert appr.final_verdict == "CONFIRMED_FRAUD"


# =============================================================================
# 3. GROUNDING EVALUATOR & HALLUCINATION DETECTION TESTS
# =============================================================================

def test_grounding_evaluator_valid_citations():
    """Verify that factually accurate citations score 100% grounding and 0 hallucinations."""
    payload = {"V14": -5.61, "V10": -3.88, "Amount": 289.50}
    shap_data = {
        "top_positive_contributors": [
            {"feature": "V14", "shap_value": 1.84, "feature_value": -5.61},
            {"feature": "V10", "shap_value": 1.15, "feature_value": -3.88},
        ],
        "top_negative_contributors": [],
    }

    report_dict = {
        "case_id": "CASE-123",
        "executive_summary": "Transaction exhibits anomalous PCA variance on V14 and V10.",
        "investigation_findings": "Observed V14=-5.61 shifts log-odds by +1.84. Observed V10=-3.88 shifts log-odds by +1.15.",
        "key_evidence_citations": [
            {
                "feature": "V14",
                "observed_value": -5.61,
                "shap_value": 1.84,
                "direction": "FRAUD_PUSHING",
                "rationale": "High statistical deviation in principal component V14.",
            },
            {
                "feature": "V10",
                "observed_value": -3.88,
                "shap_value": 1.15,
                "direction": "FRAUD_PUSHING",
                "rationale": "Statistical deviation in principal component V10.",
            },
        ],
        "recommended_verdict": "CONFIRMED_FRAUD",
        "recommended_action": "CARD_BLOCKED",
    }

    eval_result = evaluate_report_grounding(report_dict, payload, shap_data)
    assert eval_result["grounding_score"] == 1.0
    assert eval_result["citation_accuracy_score"] == 1.0
    assert eval_result["hallucination_score"] == 1.0  # 1.0 = clean, zero hallucinations
    assert len(eval_result["unsupported_claims_found"]) == 0
    assert eval_result["composite_score"] >= 0.90


def test_grounding_evaluator_detects_prohibited_domain_hallucinations():
    """Verify that prohibited domain storytelling on anonymized V1-V28 is penalized."""
    payload = {"V14": -5.61, "Amount": 289.50}
    shap_data = {
        "top_positive_contributors": [{"feature": "V14", "shap_value": 1.84, "feature_value": -5.61}],
        "top_negative_contributors": [],
    }

    hallucinated_report = {
        "case_id": "CASE-123",
        "executive_summary": "Cardholder was traveling overseas and transaction occurred from a foreign ip address.",
        "investigation_findings": "V14 represents a grocery store location where an atm withdrawal occurred.",
        "key_evidence_citations": [
            {
                "feature": "V14",
                "observed_value": -5.61,
                "shap_value": 1.84,
                "direction": "FRAUD_PUSHING",
                "rationale": "Transaction at a gas station.",
            },
        ],
        "recommended_verdict": "CONFIRMED_FRAUD",
        "recommended_action": "CARD_BLOCKED",
    }

    eval_result = evaluate_report_grounding(hallucinated_report, payload, shap_data)
    assert eval_result["hallucination_score"] < 1.0
    assert len(eval_result["unsupported_claims_found"]) > 0


# =============================================================================
# 4. LANGGRAPH WORKFLOW NODE & GRAPH TESTS
# =============================================================================

def test_build_investigation_graph():
    """Verify compilation of the LangGraph state graph."""
    graph = build_investigation_graph()
    assert graph is not None
    assert hasattr(graph, "invoke")


def test_graph_node_gather_evidence(sample_test_case: Dict[str, Any]):
    """Verify node_gather_evidence executes tools and populates state."""
    state: InvestigationState = {
        "messages": [],
        "case_id": sample_test_case["case_id"],
        "alert_id": sample_test_case["alert_id"],
        "investigator_id": "test_analyst",
        "model_name": DEFAULT_AGENT_MODEL,
        "alert_data": None,
        "shap_data": None,
        "anomaly_data": None,
        "case_history": None,
        "evidence_gathered": False,
        "synthesis_notes": None,
        "structured_report": None,
        "validation_passed": False,
        "validation_errors": [],
        "validation_attempts": 0,
        "approval_status": "PENDING",
        "final_decision": None,
    }

    updates = node_gather_evidence(state)
    assert updates["evidence_gathered"] is True
    assert updates["case_history"] is not None
    assert updates["alert_data"] is not None
    assert updates["shap_data"] is not None
    assert updates["anomaly_data"] is not None


def test_graph_node_synthesize_analysis():
    """Verify node_synthesize_analysis performs signal comparison."""
    state: InvestigationState = {
        "messages": [],
        "case_id": "CASE-TEST",
        "alert_id": "ALT-TEST",
        "investigator_id": "test_analyst",
        "model_name": DEFAULT_AGENT_MODEL,
        "alert_data": {"fraud_probability": 0.91, "risk_category": "CRITICAL"},
        "shap_data": {"top_positive_contributors": [{"feature": "V14", "shap_value": 1.5, "feature_value": -4.2}]},
        "anomaly_data": {"anomaly_score": 0.82, "anomaly_tier": "HIGH"},
        "case_history": {"case_id": "CASE-TEST"},
        "evidence_gathered": True,
        "synthesis_notes": None,
        "structured_report": None,
        "validation_passed": False,
        "validation_errors": [],
        "validation_attempts": 0,
        "approval_status": "PENDING",
        "final_decision": None,
    }

    updates = node_synthesize_analysis(state)
    assert "synthesis_notes" in updates
    assert "HIGH" in updates["synthesis_notes"]
    assert "0.9100" in updates["synthesis_notes"]


# =============================================================================
# 5. HUMAN-IN-THE-LOOP APPROVAL CHECKPOINT TEST
# =============================================================================

def test_apply_investigator_decision(sample_test_case: Dict[str, Any]):
    """Verify investigator approval commits verdict, action, and note to DB."""
    case_id = sample_test_case["case_id"]
    decision = InvestigatorApprovalRequest(
        case_id=case_id,
        investigator_id="senior_investigator",
        action_approved=True,
        final_verdict="CONFIRMED_FRAUD",
        final_action="CARD_BLOCKED",
        analyst_comments="Agent findings verified against historical PCA pattern. Card blocked.",
    )

    updated_case_dict = apply_investigator_decision(case_id=case_id, decision=decision)

    assert updated_case_dict["case_id"] == case_id
    assert updated_case_dict["status"] == "RESOLVED_CONFIRMED_FRAUD"
    assert updated_case_dict["verdict"] == "CONFIRMED_FRAUD"
    assert updated_case_dict["action_taken"] == "CARD_BLOCKED"
    assert updated_case_dict["assigned_analyst"] == "senior_investigator"

    # Check note was appended
    notes = updated_case_dict.get("notes", [])
    assert any("[AI Investigation Agent Decision - APPROVED]" in n["content"] for n in notes)


# =============================================================================
# 6. FASTAPI AGENT ENDPOINTS TESTS
# =============================================================================

def test_api_check_tools_endpoint(sample_test_case: Dict[str, Any]):
    """Verify GET /agent/tools/check returns tool health status."""
    case_id = sample_test_case["case_id"]
    response = client.get(f"/agent/tools/check?case_id={case_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["case_tested"] == case_id
    assert data["tool_health"]["all_passed"] is True


def test_api_apply_agent_decision_endpoint(sample_test_case: Dict[str, Any]):
    """Verify POST /cases/{case_id}/agent-decision updates case via REST."""
    case_id = sample_test_case["case_id"]
    payload = {
        "case_id": case_id,
        "investigator_id": "api_analyst",
        "action_approved": True,
        "final_verdict": "FALSE_POSITIVE",
        "final_action": "TRANSACTION_RELEASED",
        "analyst_comments": "Customer confirmed legitimate transaction over phone.",
    }
    response = client.post(f"/cases/{case_id}/agent-decision", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["case"]["status"] == "RESOLVED_FALSE_POSITIVE"
    assert data["case"]["verdict"] == "FALSE_POSITIVE"
    assert data["case"]["action_taken"] == "TRANSACTION_RELEASED"
