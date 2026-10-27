"""Stateful Investigation Agent Workflow using LangGraph and Groq open-source LLMs.

Orchestrates read-only tools, multi-signal synthesis, structured report generation,
grounding verification, and human-in-the-loop investigator approval checkpoints.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, START, StateGraph

from src.agents.config import DEFAULT_AGENT_MODEL, get_groq_llm
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
from src.cases.database import get_db_context
from src.cases.repository import CaseRepository
from src.cases.service import get_case_service

logger = logging.getLogger(__name__)

# Keywords indicating prohibited domain storytelling for anonymized PCA features V1-V28
PROHIBITED_DOMAIN_PATTERNS = [
    r"\b(foreign|international|overseas)\s+ip\b",
    r"\bgas\s+station\b",
    r"\bgrocery\s+store\b",
    r"\batm\s+withdrawal\b",
    r"\bjewelry\s+store\b",
    r"\belectronics\s+shop\b",
    r"\bcardholder\s+(is\s+traveling|went\s+to|lives\s+in)\b",
    r"\bmerchant\s+category\s+code\b",
    r"\bv\d+\s+represents\s+(a\s+)?(location|device|country|merchant|ip)\b",
]


# =============================================================================
# GRAPH NODE FUNCTIONS
# =============================================================================

def node_gather_evidence(state: InvestigationState) -> Dict[str, Any]:
    """Retrieve all deterministic facts using the 4 read-only tools."""
    case_id = state["case_id"]
    logger.info("Gathering evidence for case: %s", case_id)

    # 1. Fetch Case History & Sealed ML Snapshot
    case_history = get_case_history.invoke({"case_id": case_id})
    if "error" in case_history:
        return {
            "case_history": case_history,
            "evidence_gathered": False,
            "validation_passed": False,
            "validation_errors": [case_history["error"]],
        }

    alert_id = case_history.get("alert_id") or state.get("alert_id")

    # 2. Fetch Alert Details
    alert_data = None
    if alert_id:
        alert_data = get_alert_details.invoke({"alert_id": alert_id})

    # 3. Fetch TreeSHAP Explanation
    shap_data = None
    if alert_id:
        shap_data = get_shap_explanation.invoke({"alert_id": alert_id, "top_n": 5})

    # 4. Fetch Unsupervised Anomaly Score
    anomaly_data = None
    if alert_id:
        anomaly_data = get_anomaly_score.invoke({"alert_id": alert_id})

    return {
        "alert_id": alert_id,
        "case_history": case_history,
        "alert_data": alert_data,
        "shap_data": shap_data,
        "anomaly_data": anomaly_data,
        "evidence_gathered": True,
    }


def node_synthesize_analysis(state: InvestigationState) -> Dict[str, Any]:
    """Perform preliminary multi-signal reasoning over gathered facts."""
    if not state.get("evidence_gathered"):
        return {"synthesis_notes": "Evidence gathering failed."}

    case = state["case_history"]
    alert = state.get("alert_data") or {}
    shap = state.get("shap_data") or {}
    anomaly = state.get("anomaly_data") or {}

    orig = case.get("original_prediction", {})
    p_fraud = orig.get("fraud_probability", alert.get("fraud_probability", 0.0))
    a_score = orig.get("anomaly_score", anomaly.get("anomaly_score", 0.0))
    a_tier = orig.get("anomaly_tier", anomaly.get("anomaly_tier", "NORMAL"))
    quadrant = orig.get("operational_quadrant", alert.get("operational_quadrant", "UNKNOWN"))

    pos_drivers = [
        f"{c['feature']} ({c['feature_value']:+.2f}, SHAP impact {c['shap_value']:+.2f})"
        for c in shap.get("top_positive_contributors", [])
    ]
    neg_drivers = [
        f"{c['feature']} ({c['feature_value']:+.2f}, SHAP impact {c['shap_value']:+.2f})"
        for c in shap.get("top_negative_contributors", [])
    ]

    notes = (
        f"Multi-Signal Evidence Summary:\n"
        f"- Supervised P(Fraud): {p_fraud:.4f} (Operational Quadrant: {quadrant})\n"
        f"- Isolation Forest Anomaly Score: {a_score:.4f} (Tier: {a_tier})\n"
        f"- Top Fraud-Pushing Components: {'; '.join(pos_drivers) if pos_drivers else 'None'}\n"
        f"- Mitigating Factors: {'; '.join(neg_drivers) if neg_drivers else 'None'}\n"
        f"- Transaction: Amount=EUR {alert.get('amount', 0.0):.2f}, Time={alert.get('time_offset', 0.0):.1f}s\n"
        f"- Trigger Policy Reasons: {'; '.join(alert.get('reasons', []))}"
    )

    return {"synthesis_notes": notes}


def node_generate_structured_report(state: InvestigationState) -> Dict[str, Any]:
    """Generate structured InvestigationReport adhering strictly to the Pydantic schema."""
    case = state["case_history"]
    alert = state.get("alert_data") or {}
    shap = state.get("shap_data") or {}
    anomaly = state.get("anomaly_data") or {}
    synthesis = state.get("synthesis_notes", "")

    model_name = state.get("model_name", DEFAULT_AGENT_MODEL)
    llm = get_groq_llm(model_name=model_name, temperature=0.1)

    parser = PydanticOutputParser(pydantic_object=InvestigationReport)

    prompt_tmpl = PromptTemplate(
        template="""You are an expert fraud investigation agent for a tier-1 financial institution.
Your task is to conduct an auditable, evidence-grounded investigation of Case {case_id}.

--- EVIDENCE COLLECTED VIA DETERMINISTIC ENGINES ---
{synthesis}

--- STRICT DATA INTEGRITY DIRECTIVES (ZERO TOLERANCE) ---
1. Features V1-V28 are anonymized PCA components to protect cardholder privacy.
   You must NEVER invent unverified domain stories (e.g. "Paris store", "card stolen at gas station", "foreign IP address", "merchant velocity").
   Express evidence strictly in terms of empirical deviations, log-odds forces, and statistical anomalies.
2. Every cited feature and value MUST match the real observed transaction vector.
3. Recommended verdict MUST be one of: CONFIRMED_FRAUD, FALSE_POSITIVE, SUSPICIOUS_UNVERIFIED, LEGITIMATE.
4. Recommended action MUST be one of: CARD_BLOCKED, CHARGEBACK_INITIATED, TRANSACTION_RELEASED, CUSTOMER_CONTACTED, NONE.

{format_instructions}
""",
        input_variables=["case_id", "synthesis"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt_tmpl | llm | parser

    try:
        report: InvestigationReport = chain.invoke({
            "case_id": state["case_id"],
            "synthesis": synthesis,
        })
        return {
            "structured_report": report.model_dump(),
            "validation_errors": [],
        }
    except Exception as exc:
        logger.warning("Structured report parsing error: %s. Attempting fallback parse.", exc)
        return {
            "validation_passed": False,
            "validation_errors": [f"Parsing error: {exc}"],
        }


def node_validate_grounding(state: InvestigationState) -> Dict[str, Any]:
    """Audit the generated report for factual grounding, citation accuracy, and unsupported stories."""
    report_dict = state.get("structured_report")
    if not report_dict:
        return {
            "validation_passed": False,
            "validation_attempts": state.get("validation_attempts", 0) + 1,
            "validation_errors": ["Report was not generated."],
        }

    errors: List[str] = []
    alert_payload = (state.get("alert_data") or {}).get("transaction_payload", {})
    shap_data = state.get("shap_data") or {}
    true_shap_features = {c["feature"]: c for c in shap_data.get("top_positive_contributors", []) + shap_data.get("top_negative_contributors", [])}

    # 1. Audit Evidence Citations
    citations = report_dict.get("key_evidence_citations", [])
    if not citations:
        errors.append("No evidence citations provided in report.")

    for cit in citations:
        feat = cit.get("feature", "")
        # Check feature existence in payload
        if alert_payload and feat not in alert_payload:
            errors.append(f"Cited feature '{feat}' does not exist in transaction payload.")

        # Check directional correctness against true SHAP values
        if feat in true_shap_features:
            true_shap = true_shap_features[feat]["shap_value"]
            claimed_dir = cit.get("direction", "")
            if true_shap > 0 and claimed_dir == CitationDirection.LEGITIMATE_MITIGATING.value:
                errors.append(f"Feature '{feat}' has positive SHAP impact ({true_shap:+.2f}) but was cited as mitigating.")
            elif true_shap < 0 and claimed_dir == CitationDirection.FRAUD_PUSHING.value:
                errors.append(f"Feature '{feat}' has negative SHAP impact ({true_shap:+.2f}) but was cited as fraud-pushing.")

    # 2. Check for Hallucinated Domain Stories on V1-V28
    full_text = " ".join([
        report_dict.get("executive_summary", ""),
        report_dict.get("justification", ""),
        report_dict.get("supervised_signal_analysis", ""),
    ]).lower()

    for pattern in PROHIBITED_DOMAIN_PATTERNS:
        match = re.search(pattern, full_text)
        if match:
            errors.append(f"Prohibited fictitious narrative detected: '{match.group(0)}'. Features V1-V28 must remain purely statistical.")

    attempts = state.get("validation_attempts", 0) + 1
    passed = len(errors) == 0

    if passed:
        logger.info("Grounding audit passed with 0 unsupported claims.")
    else:
        logger.warning("Grounding audit detected %d issues (attempt %d).", len(errors), attempts)

    return {
        "validation_passed": passed,
        "validation_attempts": attempts,
        "validation_errors": errors,
    }


def node_revise_report(state: InvestigationState) -> Dict[str, Any]:
    """Correct report based on grounding validation feedback."""
    logger.info("Revising report due to grounding audit feedback: %s", state.get("validation_errors"))
    model_name = state.get("model_name", DEFAULT_AGENT_MODEL)
    llm = get_groq_llm(model_name=model_name, temperature=0.0)

    parser = PydanticOutputParser(pydantic_object=InvestigationReport)
    errors_str = "\n".join(f"- {e}" for e in state.get("validation_errors", []))

    prompt_tmpl = PromptTemplate(
        template="""Your previous investigation report had factual grounding or narrative integrity issues.
Please revise the report to eliminate all errors listed below.

ERRORS TO CORRECT:
{errors}

ORIGINAL EVIDENCE:
{synthesis}

{format_instructions}
""",
        input_variables=["errors", "synthesis"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt_tmpl | llm | parser
    try:
        revised = chain.invoke({
            "errors": errors_str,
            "synthesis": state.get("synthesis_notes", ""),
        })
        return {
            "structured_report": revised.model_dump(),
            "validation_errors": [],
        }
    except Exception as exc:
        return {"validation_errors": [f"Revision parsing error: {exc}"]}


def node_checkpoint_approval(state: InvestigationState) -> Dict[str, Any]:
    """Package validated report for investigator review and checkpoint sign-off."""
    report = state.get("structured_report") or {}
    logger.info("Report packaged for investigator checkpoint: case_id=%s, verdict=%s", state["case_id"], report.get("recommended_verdict"))
    return {
        "approval_status": "PENDING",
    }


def should_continue_validation(state: InvestigationState) -> Literal["revise_report", "checkpoint_approval"]:
    """Conditional edge deciding whether to revise or present for human checkpoint."""
    if state.get("validation_passed", False):
        return "checkpoint_approval"
    if state.get("validation_attempts", 0) < 2:
        return "revise_report"
    return "checkpoint_approval"


# =============================================================================
# COMPILE LANGGRAPH WORKFLOW
# =============================================================================

def build_investigation_graph() -> StateGraph:
    """Build and compile the multi-step investigation workflow."""
    workflow = StateGraph(InvestigationState)

    workflow.add_node("gather_evidence", node_gather_evidence)
    workflow.add_node("synthesize_analysis", node_synthesize_analysis)
    workflow.add_node("generate_structured_report", node_generate_structured_report)
    workflow.add_node("validate_grounding", node_validate_grounding)
    workflow.add_node("revise_report", node_revise_report)
    workflow.add_node("checkpoint_approval", node_checkpoint_approval)

    # Edge definitions
    workflow.add_edge(START, "gather_evidence")
    workflow.add_edge("gather_evidence", "synthesize_analysis")
    workflow.add_edge("synthesize_analysis", "generate_structured_report")
    workflow.add_edge("generate_structured_report", "validate_grounding")

    workflow.add_conditional_edges(
        "validate_grounding",
        should_continue_validation,
        {
            "revise_report": "revise_report",
            "checkpoint_approval": "checkpoint_approval",
        },
    )

    workflow.add_edge("revise_report", "validate_grounding")
    workflow.add_edge("checkpoint_approval", END)

    compiled = workflow.compile()
    save_graph_image_if_needed(compiled)
    return compiled


def save_graph_image_if_needed(compiled_graph=None, output_path: Optional[Path] = None) -> Optional[Path]:
    """Save the state graph visualization image to miscellaneous/ if not already present.

    Ensures the graph diagram is generated once and skipped if already saved.
    """
    if output_path is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        output_dir = project_root / "miscellaneous"
        output_path = output_dir / "investigation_agent_graph.png"
    else:
        output_path = Path(output_path)
        output_dir = output_path.parent

    # If the file is already present, do not regenerate
    if output_path.exists():
        return output_path

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        if compiled_graph is None:
            compiled_graph = get_investigation_graph()
        g = compiled_graph.get_graph()
        png_bytes = g.draw_mermaid_png()
        output_path.write_bytes(png_bytes)
        logger.info("Saved investigation agent state graph image to %s", output_path)
        return output_path
    except Exception as exc:
        logger.warning("Could not export graph image to %s: %s", output_path, exc)
        return None


# Cached compiled graph singleton
_COMPILED_GRAPH = None


def get_investigation_graph():
    """Retrieve or compile the LangGraph investigation workflow."""
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_investigation_graph()
    return _COMPILED_GRAPH


# =============================================================================
# HIGH-LEVEL CONVENIENCE EXECUTION METHODS
# =============================================================================

def run_case_investigation(
    case_id: str,
    model_name: Optional[str] = None,
    investigator_id: str = "lead_analyst",
) -> Dict[str, Any]:
    """Execute complete end-to-end agentic investigation for a case."""
    graph = get_investigation_graph()

    initial_state: InvestigationState = {
        "messages": [HumanMessage(content=f"Investigate Case {case_id}")],
        "case_id": case_id,
        "alert_id": None,
        "investigator_id": investigator_id,
        "model_name": model_name or DEFAULT_AGENT_MODEL,
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

    final_state = graph.invoke(initial_state)
    return {
        "case_id": case_id,
        "model_used": final_state.get("model_name"),
        "evidence": {
            "alert": final_state.get("alert_data"),
            "shap": final_state.get("shap_data"),
            "anomaly": final_state.get("anomaly_data"),
        },
        "report": final_state.get("structured_report"),
        "validation_passed": final_state.get("validation_passed"),
        "validation_errors": final_state.get("validation_errors"),
        "approval_status": final_state.get("approval_status"),
    }


def apply_investigator_decision(
    case_id: str,
    decision: InvestigatorApprovalRequest,
) -> Dict[str, Any]:
    """Apply an investigator's review decision to the case in the database."""
    service = get_case_service()
    with get_db_context() as db:
        # Determine status transition based on verdict
        if decision.final_verdict == "CONFIRMED_FRAUD":
            new_status = "RESOLVED_CONFIRMED_FRAUD"
        elif decision.final_verdict in ["FALSE_POSITIVE", "LEGITIMATE"]:
            new_status = "RESOLVED_FALSE_POSITIVE"
        else:
            new_status = "UNDER_INVESTIGATION"

        # 1. Update Case Workflow
        updated_case = service.update_case_workflow(
            db=db,
            case_id=case_id,
            status=new_status,
            assigned_analyst=decision.investigator_id,
            verdict=decision.final_verdict,
            action_taken=decision.final_action,
            resolution_notes=decision.analyst_comments or "Investigator reviewed and applied AI Agent recommendation.",
            updated_by=decision.investigator_id,
        )

        # 2. Add Official Note
        approval_label = "APPROVED" if decision.action_approved else "MODIFIED/OVERRIDDEN"
        note_content = (
            f"[AI Investigation Agent Decision - {approval_label}]\n"
            f"Verdict: {decision.final_verdict}\n"
            f"Action Taken: {decision.final_action}\n"
            f"Investigator Notes: {decision.analyst_comments or 'None provided.'}"
        )
        service.add_note(
            db=db,
            case_id=case_id,
            author=decision.investigator_id,
            content=note_content,
        )

        db.expire_all()
        fresh_case = CaseRepository.get_by_id(db, case_id)
        return fresh_case.to_dict(include_relations=True) if fresh_case else updated_case.to_dict(include_relations=True)
