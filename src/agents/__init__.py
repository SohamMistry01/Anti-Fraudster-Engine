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
)
from src.agents.investigation_agent import (
    apply_investigator_decision,
    build_investigation_graph,
    get_investigation_graph,
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

__all__ = [
    "DEFAULT_AGENT_MODEL",
    "FALLBACK_AGENT_MODEL",
    "SUPPORTED_MODELS",
    "get_groq_api_key",
    "get_groq_llm",
    "InvestigationState",
    "EvidenceCitation",
    "CitationDirection",
    "GroundingCheck",
    "InvestigationReport",
    "InvestigatorApprovalRequest",
    "get_alert_details",
    "get_shap_explanation",
    "get_anomaly_score",
    "get_case_history",
    "build_investigation_graph",
    "get_investigation_graph",
    "run_case_investigation",
    "apply_investigator_decision",
    "evaluate_tool_correctness",
    "evaluate_report_grounding",
    "evaluate_agent_on_cases",
]
