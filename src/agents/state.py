"""LangGraph State Definition for the Agentic Investigation Workflow."""

from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class InvestigationState(TypedDict):
    """Execution state tracking the complete lifecycle of an automated investigation."""

    # Chat history / Message stream
    messages: Annotated[List[BaseMessage], add_messages]

    # Target entity identifiers
    case_id: str
    alert_id: Optional[str]
    investigator_id: str
    model_name: str

    # Deterministic evidence retrieved via read-only tools
    alert_data: Optional[Dict[str, Any]]
    shap_data: Optional[Dict[str, Any]]
    anomaly_data: Optional[Dict[str, Any]]
    case_history: Optional[Dict[str, Any]]

    # Analytical synthesis and intermediate thoughts
    evidence_gathered: bool
    synthesis_notes: Optional[str]

    # Structured Pydantic report
    structured_report: Optional[Dict[str, Any]]

    # Grounding check and validation status
    validation_passed: bool
    validation_errors: List[str]
    validation_attempts: int

    # Human-in-the-loop checkpoint
    approval_status: str  # PENDING, APPROVED, MODIFIED, REJECTED
    final_decision: Optional[Dict[str, Any]]
