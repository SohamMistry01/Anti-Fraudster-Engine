"""Stateful Fraud Triage & Prioritization Agent Workflow using LangGraph and Groq LLMs.

Evaluates an entire queue of alerts, applies deterministic multi-signal prioritization,
categorizes alerts into operational priority tiers, generates concise evidence-grounded
triage explanations, recommends workload-aware investigator assignments, and facilitates
human-in-the-loop supervisor approval checkpoints.
"""

from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Literal, Optional, Tuple
import uuid

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, START, StateGraph

from src.agents.config import DEFAULT_AGENT_MODEL, get_groq_llm
from src.agents.investigation_agent import PROHIBITED_DOMAIN_PATTERNS
from src.agents.schemas import (
    AlertTriageItem,
    InvestigatorWorkload,
    QueueOverview,
    TriageAction,
    TriageBatchReport,
    TriagePriorityTier,
)
from src.agents.state import TriageState
from src.agents.tools import (
    DEFAULT_INVESTIGATOR_ROSTER,
    get_investigator_workloads,
    get_pending_alerts_for_triage,
)
from src.cases.database import get_db_context
from src.cases.models import AlertPriority, AlertStatus, CaseStatus
from src.cases.repository import AlertRepository, CaseRepository
from src.cases.service import get_case_service

logger = logging.getLogger(__name__)


# =============================================================================
# DETERMINISTIC QUEUE TRIAGE & SCORING CAPABILITIES
# =============================================================================

def compute_deterministic_triage(alert: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate deterministic multi-signal urgency score, priority tier, and operational action.

    This ensures risk prioritization is 100% reproducible and grounded in verified ML signals
    rather than allowing an LLM to invent arbitrary scores.

    Score Breakdown (0 to 100):
    - Supervised Fraud Probability (0-40 pts): P(Fraud) * 40
    - Isolation Forest Anomaly Score (0-25 pts): Anomaly Score * 25
    - Monetary Exposure (0-15 pts): Tiered amount impact (€1,000+ = 15, €500+ = 10, €150+ = 6, else = 2)
    - Existing Policy Priority (0-15 pts): CRITICAL=15, HIGH=10, MEDIUM=5, LOW=0
    - Active Case Follow-up (0-5 pts): +5 pts if already escalated to an active case
    """
    p_fraud = float(alert.get("fraud_probability", 0.0))
    anomaly_score = float(alert.get("anomaly_score", 0.0))
    amount = float(alert.get("amount", 0.0))
    policy_priority = str(alert.get("priority", "MEDIUM")).upper()
    case_id = alert.get("case_id")
    reasons = list(alert.get("reasons", []))

    # 1. Supervised Signal (0 to 40)
    score_p = p_fraud * 40.0

    # 2. Unsupervised Anomaly Signal (0 to 25)
    score_anom = anomaly_score * 25.0

    # 3. Monetary Exposure (0 to 15)
    if amount >= 1000.0:
        score_amt = 15.0
    elif amount >= 500.0:
        score_amt = 10.0
    elif amount >= 150.0:
        score_amt = 6.0
    else:
        score_amt = 2.0

    # 4. Policy Severity (0 to 15)
    if policy_priority == AlertPriority.CRITICAL.value:
        score_policy = 15.0
    elif policy_priority == AlertPriority.HIGH.value:
        score_policy = 10.0
    elif policy_priority == AlertPriority.MEDIUM.value:
        score_policy = 5.0
    else:
        score_policy = 0.0

    # 5. Active Case Status (0 to 5)
    score_case = 5.0 if case_id else 0.0

    raw_composite = score_p + score_anom + score_amt + score_policy + score_case
    composite_urgency_score = min(100.0, max(0.0, round(raw_composite, 2)))

    # Determine Operational Priority Tier
    if composite_urgency_score >= 70.0 or p_fraud >= 0.80:
        priority_tier = TriagePriorityTier.TIER_1_CRITICAL.value
    elif composite_urgency_score >= 48.0 or p_fraud >= 0.45 or anomaly_score >= 0.75:
        priority_tier = TriagePriorityTier.TIER_2_HIGH.value
    elif composite_urgency_score >= 28.0 or p_fraud >= 0.182 or anomaly_score >= 0.50:
        priority_tier = TriagePriorityTier.TIER_3_MEDIUM.value
    else:
        priority_tier = TriagePriorityTier.TIER_4_LOW.value

    # Determine Recommended Operational Action
    if case_id:
        recommended_action = TriageAction.PRIORITIZE_CASE_INVESTIGATION.value
    elif priority_tier in (TriagePriorityTier.TIER_1_CRITICAL.value, TriagePriorityTier.TIER_2_HIGH.value):
        recommended_action = TriageAction.ESCALATE_TO_CASE.value
    elif priority_tier == TriagePriorityTier.TIER_3_MEDIUM.value:
        recommended_action = TriageAction.BATCH_REVIEW.value
    else:
        if p_fraud < 0.05 and anomaly_score < 0.35:
            recommended_action = TriageAction.DISMISS_FALSE_POSITIVE.value
        else:
            recommended_action = TriageAction.MONITOR_ONLY.value

    return {
        "alert_id": alert["alert_id"],
        "composite_urgency_score": composite_urgency_score,
        "priority_tier": priority_tier,
        "recommended_action": recommended_action,
        "fraud_probability": round(p_fraud, 4),
        "anomaly_score": round(anomaly_score, 4),
        "amount": round(amount, 2),
        "policy_priority": policy_priority,
        "policy_reasons": reasons,
        "operational_quadrant": alert.get("operational_quadrant", "UNKNOWN"),
        "case_id": case_id,
        "transaction_id": alert.get("transaction_id"),
        "created_at": alert.get("created_at"),
    }


def match_investigator_for_alert(
    ranking: Dict[str, Any],
    roster: List[Dict[str, Any]],
) -> Tuple[Optional[str], Optional[str]]:
    """Recommend best-fit investigator using workload balance, capacity limits, and tier specialization."""
    if not roster:
        return None, "No investigator roster configured."

    tier = ranking.get("priority_tier", TriagePriorityTier.TIER_3_MEDIUM.value)
    quadrant = ranking.get("operational_quadrant", "")
    p_fraud = ranking.get("fraud_probability", 0.0)

    # Filter available analysts with remaining capacity
    available = [inv for inv in roster if inv.get("is_available", True) and inv.get("active_cases", 0) < inv.get("max_capacity", 10)]
    if not available:
        # Fallback to analyst with lowest capacity utilization even if at capacity
        fallback = min(roster, key=lambda x: x.get("capacity_utilization", 1.0))
        return (
            fallback["investigator_id"],
            f"All investigators at or above capacity. Allocated to {fallback['name']} (lowest load: {fallback['active_cases']}/{fallback['max_capacity']}).",
        )

    scored_candidates = []
    for inv in available:
        active = inv.get("active_cases", 0)
        max_cap = inv.get("max_capacity", 10)
        specialties = inv.get("specialties", [])
        role = inv.get("role", "")

        match_score = 0.0

        # Specialty match
        if tier in specialties or quadrant in specialties:
            match_score += 4.0

        # Role seniority matching
        if tier == TriagePriorityTier.TIER_1_CRITICAL.value:
            if "Senior" in role or "Lead" in role:
                match_score += 3.0
        elif tier == TriagePriorityTier.TIER_2_HIGH.value:
            if "Operations" in role or "Senior" in role:
                match_score += 2.0
        elif tier in (TriagePriorityTier.TIER_3_MEDIUM.value, TriagePriorityTier.TIER_4_LOW.value):
            if "Junior" in role or "Operations" in role:
                match_score += 2.0

        # Capacity headroom bonus
        headroom = (max_cap - active) / max_cap
        match_score += headroom * 3.0

        scored_candidates.append((match_score, inv))

    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    best_inv = scored_candidates[0][1]

    active = best_inv.get("active_cases", 0)
    max_cap = best_inv.get("max_capacity", 10)
    pct = int((active / max_cap) * 100)
    rationale = (
        f"Matched to {best_inv['name']} ({best_inv['role']}) based on {tier} alignment "
        f"and available capacity ({active}/{max_cap} active cases, {pct}% utilization)."
    )
    return best_inv["investigator_id"], rationale


# =============================================================================
# GRAPH NODE FUNCTIONS
# =============================================================================

def node_gather_triage_queue(state: TriageState) -> Dict[str, Any]:
    """Retrieve queue alerts and investigator workloads using deterministic tools."""
    filter_status = state.get("filter_status", "NEW")
    max_alerts = state.get("max_alerts", 15)
    include_workload = state.get("include_workload", True)

    logger.info("Gathering triage queue: status=%s, limit=%d", filter_status, max_alerts)

    # 1. Fetch pending alerts
    alerts_data = get_pending_alerts_for_triage.invoke({
        "limit": max_alerts,
        "status": filter_status,
    })

    # 2. Fetch investigator workloads if requested
    investigators = []
    if include_workload:
        investigators = get_investigator_workloads.invoke({})

    return {
        "raw_alerts": alerts_data,
        "investigators": investigators,
    }


def node_deterministic_ranking(state: TriageState) -> Dict[str, Any]:
    """Perform deterministic scoring, tier assignment, queue ordering, and investigator matching."""
    raw_alerts = state.get("raw_alerts", [])
    investigators = state.get("investigators", [])

    if not raw_alerts:
        logger.warning("No alerts available for triage ranking.")
        return {
            "deterministic_rankings": [],
        }

    scored_alerts: List[Dict[str, Any]] = []
    for alert in raw_alerts:
        scored = compute_deterministic_triage(alert)
        if state.get("include_workload", True) and investigators:
            inv_id, rationale = match_investigator_for_alert(scored, investigators)
            scored["recommended_investigator"] = inv_id
            scored["assignment_rationale"] = rationale
        else:
            scored["recommended_investigator"] = None
            scored["assignment_rationale"] = None
        scored_alerts.append(scored)

    # Sort descending: composite_urgency_score -> fraud_probability -> amount
    scored_alerts.sort(
        key=lambda x: (x["composite_urgency_score"], x["fraud_probability"], x["amount"]),
        reverse=True,
    )

    # Assign 1-based ranks
    for idx, item in enumerate(scored_alerts, start=1):
        item["rank"] = idx

    logger.info("Deterministic triage ranking complete for %d alerts.", len(scored_alerts))
    return {
        "deterministic_rankings": scored_alerts,
    }


def node_generate_triage_explanations(state: TriageState) -> Dict[str, Any]:
    """Generate concise, evidence-grounded triage explanations and macro overview using Groq LLM."""
    rankings = state.get("deterministic_rankings", [])
    investigators = state.get("investigators", [])

    if not rankings:
        overview = QueueOverview(
            total_alerts_evaluated=0,
            tier_distribution={},
            queue_risk_synthesis="No pending alerts found matching current filter criteria. Queue is clear.",
            workload_bottlenecks="All investigators available.",
        )
        report = TriageBatchReport(
            batch_id=f"BATCH-{uuid.uuid4().hex[:8].upper()}",
            queue_overview=overview,
            prioritized_queue=[],
            grounding_audit_passed=True,
        )
        return {
            "structured_batch_report": report.model_dump(),
            "validation_passed": True,
            "validation_errors": [],
        }

    # Calculate summary metrics
    tier_counts: Dict[str, int] = {}
    for r in rankings:
        t = r["priority_tier"]
        tier_counts[t] = tier_counts.get(t, 0) + 1

    total_eval = len(rankings)
    critical_count = tier_counts.get(TriagePriorityTier.TIER_1_CRITICAL.value, 0)
    high_count = tier_counts.get(TriagePriorityTier.TIER_2_HIGH.value, 0)

    # Build prompt input summary
    rankings_summary = []
    for r in rankings:
        rankings_summary.append(
            f"Rank #{r['rank']} | Alert: {r['alert_id']} | Tier: {r['priority_tier']} | "
            f"Score: {r['composite_urgency_score']:.1f}/100 | P(Fraud): {r['fraud_probability']:.2%} | "
            f"Anomaly: {r['anomaly_score']:.2f} | Amount: €{r['amount']:.2f} | Action: {r['recommended_action']} | "
            f"Investigator: {r.get('recommended_investigator') or 'None'} | "
            f"Triggers: {', '.join(r.get('policy_reasons', [])) or 'Multi-signal threshold'}"
        )
    rankings_text = "\n".join(rankings_summary)

    inv_summary = []
    for inv in investigators:
        inv_summary.append(
            f"- {inv['name']} ({inv['role']}): {inv['active_cases']}/{inv['max_capacity']} active cases "
            f"({int(inv['capacity_utilization'] * 100)}% load, Specialties: {', '.join(inv.get('specialties', []))})"
        )
    inv_text = "\n".join(inv_summary) if inv_summary else "No workload tracking requested."

    model_name = state.get("model_name", DEFAULT_AGENT_MODEL)
    llm = get_groq_llm(model_name=model_name, temperature=0.1)

    parser = PydanticOutputParser(pydantic_object=TriageBatchReport)

    prompt_tmpl = PromptTemplate(
        template="""You are a Fraud Triage & Queue Prioritization Lead Agent for a tier-1 financial institution.
Your task is to synthesize the deterministic ranking of {total_alerts} alerts, generate concise, evidence-grounded
justifications for each alert's priority rank, and provide an executive macro overview for the supervisor.

--- DETERMINISTICALLY RANKED QUEUE DATA (DO NOT ALTER RANKS OR SCORES) ---
{rankings_text}

--- INVESTIGATOR WORKLOAD & CAPACITY ---
{inv_text}

--- STRICT INTEGRITY & GROUNDING DIRECTIVES (ZERO TOLERANCE) ---
1. DO NOT invent or alter the deterministic ranks, composite scores, or priority tiers provided above.
2. For each alert's 'triage_explanation': Provide 1-2 concise, evidence-grounded sentences explaining why
   it occupies this rank position (synthesizing fraud probability, anomaly score, exposure amount, and policy triggers).
3. Features V1-V28 are anonymized PCA components. NEVER invent fictitious domain stories (e.g. gas stations, IP addresses, foreign stores).
4. For 'queue_risk_synthesis': Provide a concise supervisor summary of backlog risk, high-priority threats, and recommended handling sequence.
5. For 'workload_bottlenecks': Summarize whether investigator capacities are sufficient or nearing saturation.

{format_instructions}
""",
        input_variables=["total_alerts", "rankings_text", "inv_text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt_tmpl | llm | parser

    try:
        batch_report: TriageBatchReport = chain.invoke({
            "total_alerts": total_eval,
            "rankings_text": rankings_text,
            "inv_text": inv_text,
        })

        # Safeguard: ensure deterministic numbers are preserved exactly
        report_dict = batch_report.model_dump()
        for idx, item in enumerate(report_dict.get("prioritized_queue", [])):
            if idx < len(rankings):
                item["composite_urgency_score"] = rankings[idx]["composite_urgency_score"]
                item["rank"] = rankings[idx]["rank"]
                item["priority_tier"] = rankings[idx]["priority_tier"]
                item["recommended_action"] = rankings[idx]["recommended_action"]
                item["fraud_probability"] = rankings[idx]["fraud_probability"]
                item["anomaly_score"] = rankings[idx]["anomaly_score"]
                item["amount"] = rankings[idx]["amount"]
                item["policy_reasons"] = rankings[idx]["policy_reasons"]
                item["case_id"] = rankings[idx].get("case_id")
                if not item.get("recommended_investigator"):
                    item["recommended_investigator"] = rankings[idx].get("recommended_investigator")
                if not item.get("assignment_rationale"):
                    item["assignment_rationale"] = rankings[idx].get("assignment_rationale")

        return {
            "structured_batch_report": report_dict,
            "validation_errors": [],
        }
    except Exception as exc:
        logger.warning("LLM triage generation parsing failed (%s). Generating robust deterministic fallback.", exc)
        # Resilient fallback guaranteeing zero crash
        fallback_items = []
        for r in rankings:
            exp = (
                f"Ranked #{r['rank']} with urgency score {r['composite_urgency_score']:.1f}/100. "
                f"Driven by P(Fraud) of {r['fraud_probability']:.2%}, anomaly score of {r['anomaly_score']:.2f}, "
                f"and monetary exposure of €{r['amount']:.2f}."
            )
            fallback_items.append(AlertTriageItem(
                alert_id=r["alert_id"],
                rank=r["rank"],
                composite_urgency_score=r["composite_urgency_score"],
                priority_tier=r["priority_tier"],
                fraud_probability=r["fraud_probability"],
                anomaly_score=r["anomaly_score"],
                amount=r["amount"],
                policy_reasons=r["policy_reasons"],
                triage_explanation=exp,
                recommended_action=r["recommended_action"],
                recommended_investigator=r.get("recommended_investigator"),
                assignment_rationale=r.get("assignment_rationale"),
                case_id=r.get("case_id"),
            ))

        fallback_overview = QueueOverview(
            total_alerts_evaluated=total_eval,
            tier_distribution=tier_counts,
            queue_risk_synthesis=(
                f"Evaluated {total_eval} alerts: {critical_count} critical and {high_count} high priority. "
                f"Immediate escalation recommended for top {critical_count + high_count} alerts."
            ),
            workload_bottlenecks="Workload distributed across active investigation team.",
        )

        fallback_report = TriageBatchReport(
            batch_id=f"BATCH-{uuid.uuid4().hex[:8].upper()}",
            queue_overview=fallback_overview,
            prioritized_queue=fallback_items,
            grounding_audit_passed=True,
        )

        return {
            "structured_batch_report": fallback_report.model_dump(),
            "validation_errors": [f"Fallback activated due to parsing: {exc}"],
        }


def node_validate_triage_grounding(state: TriageState) -> Dict[str, Any]:
    """Audit the generated triage explanations for factual grounding and domain hallucinations."""
    report = state.get("structured_batch_report")
    rankings = state.get("deterministic_rankings", [])

    if not report or not report.get("prioritized_queue"):
        return {
            "validation_passed": True,
            "validation_errors": [],
            "validation_attempts": state.get("validation_attempts", 0) + 1,
        }

    errors: List[str] = []
    items = report.get("prioritized_queue", [])

    # 1. Audit size consistency
    if len(items) != len(rankings):
        errors.append(f"Output count mismatch: expected {len(rankings)} alerts, got {len(items)}.")

    # 2. Check for Hallucinated Domain Stories on V1-V28 across explanations
    full_explanation_text = " ".join([
        item.get("triage_explanation", "") for item in items
    ] + [
        report.get("queue_overview", {}).get("queue_risk_synthesis", ""),
        report.get("queue_overview", {}).get("workload_bottlenecks", ""),
    ]).lower()

    for pattern in PROHIBITED_DOMAIN_PATTERNS:
        match = re.search(pattern, full_explanation_text)
        if match:
            errors.append(
                f"Prohibited fictitious narrative detected in triage explanation: '{match.group(0)}'. "
                "PCA components V1-V28 must remain purely statistical."
            )

    attempts = state.get("validation_attempts", 0) + 1
    passed = len(errors) == 0

    if passed:
        logger.info("Triage grounding audit passed with 0 unsupported claims.")
    else:
        logger.warning("Triage grounding audit found %d issues (attempt %d).", len(errors), attempts)

    return {
        "validation_passed": passed,
        "validation_attempts": attempts,
        "validation_errors": errors,
    }


def node_revise_triage(state: TriageState) -> Dict[str, Any]:
    """Prompt LLM to revise triage explanations to eliminate detected grounding errors."""
    logger.info("Revising triage explanations due to grounding feedback: %s", state.get("validation_errors"))
    model_name = state.get("model_name", DEFAULT_AGENT_MODEL)
    llm = get_groq_llm(model_name=model_name, temperature=0.0)

    parser = PydanticOutputParser(pydantic_object=TriageBatchReport)
    errors_str = "\n".join(f"- {e}" for e in state.get("validation_errors", []))

    prompt_tmpl = PromptTemplate(
        template="""Your previous triage batch report had factual grounding or prohibited storytelling violations.
Please revise the triage report to completely resolve all listed errors.

ERRORS DETECTED:
{errors}

CURRENT REPORT CONTENT:
{current_report}

{format_instructions}
""",
        input_variables=["errors", "current_report"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt_tmpl | llm | parser
    try:
        revised = chain.invoke({
            "errors": errors_str,
            "current_report": json.dumps(state.get("structured_batch_report"), indent=2),
        })
        return {
            "structured_batch_report": revised.model_dump(),
            "validation_errors": [],
        }
    except Exception as exc:
        return {"validation_errors": [f"Revision parsing error: {exc}"]}


def node_checkpoint_triage_approval(state: TriageState) -> Dict[str, Any]:
    """Package prioritized queue and proposed assignments for human supervisor review."""
    report = state.get("structured_batch_report") or {}
    queue_len = len(report.get("prioritized_queue", []))
    logger.info("Triage batch packaged for supervisor checkpoint review (%d alerts).", queue_len)
    return {
        "approval_status": "PENDING",
    }


def should_continue_triage_validation(state: TriageState) -> Literal["revise_triage", "checkpoint_approval"]:
    """Conditional edge deciding whether to revise or advance to supervisor review."""
    if state.get("validation_passed", False):
        return "checkpoint_approval"
    if state.get("validation_attempts", 0) < 2:
        return "revise_triage"
    return "checkpoint_approval"


# =============================================================================
# COMPILE LANGGRAPH WORKFLOW
# =============================================================================

def build_triage_graph() -> StateGraph:
    """Build and compile the multi-step Fraud Triage & Prioritization workflow."""
    workflow = StateGraph(TriageState)

    workflow.add_node("gather_triage_queue", node_gather_triage_queue)
    workflow.add_node("deterministic_ranking", node_deterministic_ranking)
    workflow.add_node("generate_triage_explanations", node_generate_triage_explanations)
    workflow.add_node("validate_triage_grounding", node_validate_triage_grounding)
    workflow.add_node("revise_triage", node_revise_triage)
    workflow.add_node("checkpoint_approval", node_checkpoint_triage_approval)

    # Edge definitions
    workflow.add_edge(START, "gather_triage_queue")
    workflow.add_edge("gather_triage_queue", "deterministic_ranking")
    workflow.add_edge("deterministic_ranking", "generate_triage_explanations")
    workflow.add_edge("generate_triage_explanations", "validate_triage_grounding")

    workflow.add_conditional_edges(
        "validate_triage_grounding",
        should_continue_triage_validation,
        {
            "revise_triage": "revise_triage",
            "checkpoint_approval": "checkpoint_approval",
        },
    )

    workflow.add_edge("revise_triage", "validate_triage_grounding")
    workflow.add_edge("checkpoint_approval", END)

    compiled = workflow.compile()
    save_triage_graph_image_if_needed(compiled)
    return compiled


def save_triage_graph_image_if_needed(compiled_graph=None, output_path: Optional[Path] = None) -> Optional[Path]:
    """Save the state graph visualization image to miscellaneous/ if not already present."""
    if output_path is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        output_dir = project_root / "miscellaneous"
        output_path = output_dir / "triage_agent_graph.png"
    else:
        output_path = Path(output_path)
        output_dir = output_path.parent

    if output_path.exists():
        return output_path

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        if compiled_graph is None:
            compiled_graph = get_triage_graph()
        g = compiled_graph.get_graph()
        png_bytes = g.draw_mermaid_png()
        output_path.write_bytes(png_bytes)
        logger.info("Saved triage agent state graph image to %s", output_path)
        return output_path
    except Exception as exc:
        logger.warning("Could not export triage graph image to %s: %s", output_path, exc)
        return None


# Cached compiled graph singleton
_COMPILED_TRIAGE_GRAPH = None


def get_triage_graph():
    """Retrieve or compile the LangGraph triage workflow."""
    global _COMPILED_TRIAGE_GRAPH
    if _COMPILED_TRIAGE_GRAPH is None:
        _COMPILED_TRIAGE_GRAPH = build_triage_graph()
    return _COMPILED_TRIAGE_GRAPH


# =============================================================================
# HIGH-LEVEL CONVENIENCE EXECUTION METHODS
# =============================================================================

def run_queue_triage(
    filter_status: Optional[str] = "NEW",
    max_alerts: int = 10,
    include_workload: bool = True,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute complete end-to-end agentic triage and prioritization for the alert queue."""
    graph = get_triage_graph()

    initial_state: TriageState = {
        "messages": [HumanMessage(content=f"Triage alert queue (status={filter_status}, max={max_alerts})")],
        "filter_status": filter_status,
        "max_alerts": max_alerts,
        "include_workload": include_workload,
        "model_name": model_name or DEFAULT_AGENT_MODEL,
        "raw_alerts": [],
        "investigators": [],
        "deterministic_rankings": [],
        "structured_batch_report": None,
        "validation_passed": False,
        "validation_errors": [],
        "validation_attempts": 0,
        "approval_status": "PENDING",
        "final_decisions": None,
    }

    final_state = graph.invoke(initial_state)
    report = final_state.get("structured_batch_report") or {}

    return {
        "batch_id": report.get("batch_id", f"BATCH-{uuid.uuid4().hex[:8].upper()}"),
        "triaged_at": report.get("triaged_at", datetime.now(timezone.utc).isoformat()),
        "model_used": final_state.get("model_name"),
        "queue_overview": report.get("queue_overview"),
        "prioritized_queue": report.get("prioritized_queue", []),
        "validation_passed": final_state.get("validation_passed", True),
        "validation_errors": final_state.get("validation_errors", []),
        "approval_status": final_state.get("approval_status", "PENDING"),
    }


def apply_triage_decisions(
    decisions: List[Dict[str, Any]],
    supervisor_id: str = "fraud_supervisor",
) -> Dict[str, Any]:
    """Apply supervisor-approved triage actions and assignments to database cases and alerts.

    Supported actions:
    - ESCALATE_TO_CASE: Open a new auditable case and assign recommended investigator
    - PRIORITIZE_CASE_INVESTIGATION: Escalate existing case priority or reassign analyst
    - DISMISS_FALSE_POSITIVE: Dismiss alert from queue
    - UPDATE_PRIORITY: Adjust alert priority
    - BATCH_REVIEW / MONITOR_ONLY: Transition alert status or add record
    """
    service = get_case_service()
    results = {
        "applied_count": 0,
        "cases_created": [],
        "alerts_updated": [],
        "errors": [],
    }

    with get_db_context() as db:
        for item in decisions:
            alert_id = item.get("alert_id")
            action = item.get("action", "").upper()
            assigned_inv = item.get("assigned_investigator")
            p_override = item.get("priority_override")
            notes = item.get("supervisor_notes", "Applied supervisor triage decision.")

            alert = AlertRepository.get_by_id(db, alert_id)
            if not alert:
                results["errors"].append(f"Alert '{alert_id}' not found.")
                continue

            try:
                if action == TriageAction.ESCALATE_TO_CASE.value:
                    if alert.case_id:
                        # Existing case
                        results["alerts_updated"].append({"alert_id": alert_id, "action": "ALREADY_ESCALATED", "case_id": alert.case_id})
                    else:
                        case = service.open_case_from_alert(
                            db=db,
                            alert_id=alert_id,
                            assigned_analyst=assigned_inv or "lead_analyst",
                            initial_note=f"[Supervisor Triage]: {notes}",
                            creator=supervisor_id,
                        )
                        results["cases_created"].append({
                            "alert_id": alert_id,
                            "case_id": case.case_id,
                            "assigned_analyst": case.assigned_analyst,
                        })

                elif action == TriageAction.PRIORITIZE_CASE_INVESTIGATION.value:
                    if alert.case_id:
                        service.update_case_workflow(
                            db=db,
                            case_id=alert.case_id,
                            priority=p_override or alert.priority,
                            assigned_analyst=assigned_inv,
                            resolution_notes=notes,
                            updated_by=supervisor_id,
                        )
                        results["alerts_updated"].append({"alert_id": alert_id, "case_id": alert.case_id, "action": "CASE_PRIORITIZED"})
                    else:
                        # Escalate to case since none exists
                        case = service.open_case_from_alert(
                            db=db,
                            alert_id=alert_id,
                            assigned_analyst=assigned_inv or "lead_analyst",
                            initial_note=f"[Supervisor Priority Escalation]: {notes}",
                            creator=supervisor_id,
                        )
                        results["cases_created"].append({"alert_id": alert_id, "case_id": case.case_id})

                elif action == TriageAction.DISMISS_FALSE_POSITIVE.value:
                    AlertRepository.update_status(db, alert_id, AlertStatus.DISMISSED.value)
                    results["alerts_updated"].append({"alert_id": alert_id, "action": "DISMISSED"})

                else:
                    # Update status to ESCALATED or keep in review
                    AlertRepository.update_status(db, alert_id, AlertStatus.ESCALATED.value)
                    results["alerts_updated"].append({"alert_id": alert_id, "action": "STATUS_ESCALATED"})

                results["applied_count"] += 1

            except Exception as exc:
                logger.error("Error applying decision on alert %s: %s", alert_id, exc)
                results["errors"].append(f"Failed on alert '{alert_id}': {str(exc)}")

    return results
