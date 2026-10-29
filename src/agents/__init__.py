"""Agentic Investigation Layer for Anti-Fraudster Engine.

Provides multi-signal autonomous reasoning, tool orchestration, structured report generation,
factual grounding verification, and human-in-the-loop investigator approval checkpoints.
"""

from src.agents.config import (
    DEFAULT_AGENT_MODEL,
    FALLBACK_AGENT_MODEL,
    SUPPORTED_MODELS,
    get_groq_api_key,
    get_groq_llm,
)
from src.agents.evaluator import (
    evaluate_agent_on_cases,
    evaluate_report_grounding,
    evaluate_tool_correctness,
    evaluate_triage_grounding,
)
from src.agents.investigation_agent import (
    apply_investigator_decision,
    build_investigation_graph,
    get_investigation_graph,
    run_case_investigation,
)
from src.agents.schemas import (
    AlertTriageItem,
    CitationDirection,
    EvidenceCitation,
    GroundingCheck,
    InvestigationReport,
    InvestigatorApprovalRequest,
    InvestigatorWorkload,
    QueueOverview,
    TriageAction,
    TriageApprovalBatchRequest,
    TriageBatchReport,
    TriageDecisionApplication,
    TriagePriorityTier,
)
from src.agents.state import InvestigationState, TriageState
from src.agents.tools import (
    DEFAULT_INVESTIGATOR_ROSTER,
    INVESTIGATION_TOOLS,
    TRIAGE_TOOLS,
    get_alert_details,
    get_anomaly_score,
    get_case_history,
    get_investigator_workloads,
    get_pending_alerts_for_triage,
    get_shap_explanation,
)
from src.agents.triage_agent import (
    apply_triage_decisions,
    build_triage_graph,
    compute_deterministic_triage,
    get_triage_graph,
    match_investigator_for_alert,
    run_queue_triage,
)

__all__ = [
    "DEFAULT_AGENT_MODEL",
    "FALLBACK_AGENT_MODEL",
    "SUPPORTED_MODELS",
    "get_groq_api_key",
    "get_groq_llm",
    "InvestigationState",
    "TriageState",
    "EvidenceCitation",
    "CitationDirection",
    "GroundingCheck",
    "InvestigationReport",
    "InvestigatorApprovalRequest",
    "TriagePriorityTier",
    "TriageAction",
    "InvestigatorWorkload",
    "AlertTriageItem",
    "QueueOverview",
    "TriageBatchReport",
    "TriageDecisionApplication",
    "TriageApprovalBatchRequest",
    "get_alert_details",
    "get_shap_explanation",
    "get_anomaly_score",
    "get_case_history",
    "get_pending_alerts_for_triage",
    "get_investigator_workloads",
    "DEFAULT_INVESTIGATOR_ROSTER",
    "INVESTIGATION_TOOLS",
    "TRIAGE_TOOLS",
    "build_investigation_graph",
    "get_investigation_graph",
    "run_case_investigation",
    "apply_investigator_decision",
    "build_triage_graph",
    "get_triage_graph",
    "compute_deterministic_triage",
    "match_investigator_for_alert",
    "run_queue_triage",
    "apply_triage_decisions",
    "evaluate_tool_correctness",
    "evaluate_report_grounding",
    "evaluate_triage_grounding",
    "evaluate_agent_on_cases",
]

