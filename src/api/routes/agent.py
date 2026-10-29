"""FastAPI route endpoints for the Agentic Investigation Layer.

Provides endpoints to trigger autonomous investigations, validate evidence citations,
execute human-in-the-loop approval decisions, and evaluate agent factual grounding.
"""

import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.agents.config import DEFAULT_AGENT_MODEL, SUPPORTED_MODELS
from src.agents.evaluator import (
    evaluate_agent_on_cases,
    evaluate_tool_correctness,
)
from src.agents.investigation_agent import (
    apply_investigator_decision,
    run_case_investigation,
)
from src.agents.schemas import InvestigatorApprovalRequest
from src.cases.database import get_db
from src.cases.repository import CaseRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["Agentic Investigation Layer"])


class TriggerInvestigationRequest(BaseModel):
    """Payload to trigger an AI-driven investigation on a case."""
    model_name: Optional[str] = Field(
        default=DEFAULT_AGENT_MODEL,
        description=f"Groq LLM model to use. Options: {SUPPORTED_MODELS}",
    )
    investigator_id: Optional[str] = Field(
        default="lead_analyst",
        description="ID of the investigator initiating or overseeing the agent run",
    )


class EvaluateAgentRequest(BaseModel):
    """Payload to trigger agent factual grounding & citation accuracy benchmark."""
    case_ids: Optional[List[str]] = Field(
        default=None,
        description="Explicit list of case IDs to benchmark. If None, top recent cases are evaluated.",
    )
    model_name: Optional[str] = Field(
        default=DEFAULT_AGENT_MODEL,
        description="Groq model identifier to benchmark",
    )
    limit: int = Field(default=3, ge=1, le=10, description="Max number of cases to evaluate")


@router.post(
    "/cases/{case_id}/investigate",
    summary="Trigger Agentic Case Investigation",
    description=(
        "Executes a stateful LangGraph workflow that gathers evidence via read-only tools, "
        "synthesizes multi-signal ML outputs (XGBoost, SHAP, Isolation Forest), verifies factual "
        "grounding, and prepares a structured report with human approval checkpoints."
    ),
)
def investigate_case_endpoint(
    case_id: str,
    payload: TriggerInvestigationRequest = TriggerInvestigationRequest(),
    db: Session = Depends(get_db),
):
    """Run agentic investigation workflow on a specific case."""
    # Verify case exists
    case = CaseRepository.get_by_id(db, case_id)
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Investigation case '{case_id}' not found.",
        )

    try:
        result = run_case_investigation(
            case_id=case_id,
            model_name=payload.model_name,
            investigator_id=payload.investigator_id or "lead_analyst",
        )
        return {
            "status": "success",
            "case_id": case_id,
            "data": result,
        }
    except Exception as exc:
        logger.exception("Agentic investigation failed for case %s: %s", case_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent investigation failed: {str(exc)}",
        )


@router.post(
    "/cases/{case_id}/agent-decision",
    summary="Approve or Override Agent Recommendation (Human-in-the-Loop)",
    description=(
        "Applies the investigator's approval or modification of the agent's recommended verdict "
        "and mitigation action directly to the case, logging an auditable decision record."
    ),
)
def apply_agent_decision_endpoint(
    case_id: str,
    decision: InvestigatorApprovalRequest,
    db: Session = Depends(get_db),
):
    """Apply human review decision on agent report to case management system."""
    case = CaseRepository.get_by_id(db, case_id)
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Investigation case '{case_id}' not found.",
        )

    try:
        updated_case = apply_investigator_decision(case_id=case_id, decision=decision)
        return {
            "status": "success",
            "message": "Investigator decision successfully applied to case.",
            "case": updated_case,
        }
    except Exception as exc:
        logger.exception("Failed to apply decision for case %s: %s", case_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record investigator decision: {str(exc)}",
        )


@router.post(
    "/agent/evaluate",
    summary="Evaluate Agent Factual Grounding & Citations",
    description=(
        "Runs an automated benchmark across representative cases testing factual grounding, "
        "TreeSHAP evidence citation accuracy, hallucination detection on anonymized PCA features, "
        "and report usefulness."
    ),
)
def evaluate_agent_endpoint(
    payload: EvaluateAgentRequest = EvaluateAgentRequest(),
):
    """Run benchmark evaluation suite over representative cases."""
    try:
        benchmark_results = evaluate_agent_on_cases(
            case_ids=payload.case_ids,
            model_name=payload.model_name,
            limit=payload.limit,
        )
        return {
            "status": "success",
            "benchmark": benchmark_results,
        }
    except Exception as exc:
        logger.exception("Agent evaluation benchmark failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation benchmark failed: {str(exc)}",
        )


@router.get(
    "/agent/tools/check",
    summary="Check Read-Only Tool Health",
    description="Validates that all 4 deterministic read-only tools function correctly on existing DB records.",
)
def check_tools_endpoint(
    case_id: Optional[str] = Query(None, description="Optional case ID to test. If omitted, uses first found case."),
    db: Session = Depends(get_db),
):
    """Test health of all 4 retrieval tools."""
    if not case_id:
        cases, _ = CaseRepository.list_cases(db=db, limit=1)
        if not cases:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No cases available in database to test tools.",
            )
        sample_case = cases[0]
        case_id = sample_case.case_id
        alert_id = sample_case.alert_id or "ALERT-UNKNOWN"
    else:
        sample_case = CaseRepository.get_by_id(db, case_id)
        if not sample_case:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Case {case_id} not found.",
            )
        alert_id = sample_case.alert_id or "ALERT-UNKNOWN"

    results = evaluate_tool_correctness(sample_case_id=case_id, sample_alert_id=alert_id)
    return {
        "status": "online" if results["all_passed"] else "degraded",
        "case_tested": case_id,
        "alert_tested": alert_id,
        "tool_health": results,
    }


# =============================================================================
# FRAUD TRIAGE & PRIORITIZATION AGENT ENDPOINTS
# =============================================================================

class TriggerTriageRequest(BaseModel):
    """Payload to trigger agentic queue triage and prioritization."""
    filter_status: Optional[str] = Field(default="NEW", description="Alert status filter (e.g. 'NEW', 'ESCALATED', or 'ALL')")
    max_alerts: int = Field(default=15, ge=1, le=50, description="Maximum alerts to evaluate and rank")
    include_workload: bool = Field(default=True, description="Whether to include workload-aware investigator assignment")
    model_name: Optional[str] = Field(default=DEFAULT_AGENT_MODEL, description="Groq model to use")


@router.post(
    "/agent/triage/run",
    summary="Trigger Agentic Alert Queue Triage & Prioritization",
    description=(
        "Executes stateful LangGraph workflow that retrieves pending alerts, applies deterministic "
        "multi-signal prioritization, categorizes alerts into operational priority tiers, "
        "generates grounded explanations, and recommends workload-aware investigator assignments."
    ),
)
def run_triage_endpoint(payload: TriggerTriageRequest = TriggerTriageRequest()) -> Dict[str, Any]:
    """Execute queue triage agent workflow."""
    from src.agents.triage_agent import run_queue_triage

    try:
        result = run_queue_triage(
            filter_status=payload.filter_status,
            max_alerts=payload.max_alerts,
            include_workload=payload.include_workload,
            model_name=payload.model_name,
        )
        return {
            "status": "success",
            "data": result,
        }
    except Exception as exc:
        logger.exception("Queue triage agent run failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Triage agent failed: {str(exc)}",
        )


@router.post(
    "/agent/triage/apply",
    summary="Apply Supervisor Triage Review Decisions",
    description="Batch apply approved triage actions, opening cases, assigning investigators, and updating alerts.",
)
def apply_triage_decisions_endpoint(
    payload: Dict[str, Any],
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Apply supervisor triage decisions to database."""
    from src.agents.triage_agent import apply_triage_decisions

    decisions = payload.get("decisions", [])
    supervisor_id = payload.get("supervisor_id", "fraud_supervisor")

    if not decisions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payload must include a non-empty 'decisions' list.",
        )

    try:
        results = apply_triage_decisions(decisions=decisions, supervisor_id=supervisor_id)
        return {
            "status": "success",
            "results": results,
        }
    except Exception as exc:
        logger.exception("Failed to apply triage decisions: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to apply triage decisions: {str(exc)}",
        )


@router.get(
    "/agent/triage/workloads",
    summary="Get Real-Time Investigator Workloads",
    description="Retrieve live investigator active case counts, capacity limits, and utilization ratios.",
)
def get_workloads_endpoint() -> Dict[str, Any]:
    """Fetch current investigator workloads."""
    from src.agents.tools import get_investigator_workloads

    try:
        workloads = get_investigator_workloads.invoke({})
        return {
            "status": "success",
            "total_investigators": len(workloads),
            "investigators": workloads,
        }
    except Exception as exc:
        logger.exception("Failed to retrieve investigator workloads: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workload retrieval failed: {str(exc)}",
        )

