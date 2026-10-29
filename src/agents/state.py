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


class TriageState(TypedDict):
    """Execution state tracking the complete lifecycle of alert queue triage and prioritization."""

    # Chat history / Message stream
    messages: Annotated[List[BaseMessage], add_messages]

    # Queue filters & configuration
    filter_status: Optional[str]
    max_alerts: int
    include_workload: bool
    model_name: str

    # Deterministic facts retrieved via tools
    raw_alerts: List[Dict[str, Any]]
    investigators: List[Dict[str, Any]]

    # Deterministic risk-based prioritization & matching
    deterministic_rankings: List[Dict[str, Any]]

    # LLM-generated structured triage report (grounded explanations & macro summary)
    structured_batch_report: Optional[Dict[str, Any]]

    # Grounding check and validation status
    validation_passed: bool
    validation_errors: List[str]
    validation_attempts: int

    # Human oversight checkpoint
    approval_status: str  # PENDING, APPROVED, OVERRIDDEN
    final_decisions: Optional[List[Dict[str, Any]]]


class PatternAnalysisState(TypedDict):
    """Execution state tracking the complete lifecycle of historical fraud pattern and trend analysis."""

    # Chat history / Message stream
    messages: Annotated[List[BaseMessage], add_messages]

    # Analysis scope & configuration parameters
    time_horizon_hours: Optional[int]
    baseline_ratio: float
    min_alerts_threshold: int
    model_name: str

    # Deterministic historical records retrieved from storage
    raw_historical_records: List[Dict[str, Any]]
    baseline_records: List[Dict[str, Any]]
    recent_records: List[Dict[str, Any]]
    time_windows_metadata: Dict[str, Any]

    # Deterministic cohort statistics & detected shifts
    computed_statistics: Dict[str, Any]
    detected_shifts: List[Dict[str, Any]]

    # Structured Pydantic report & LLM synthesis
    structured_report: Optional[Dict[str, Any]]

    # Grounding check and validation status
    validation_passed: bool
    validation_errors: List[str]
    validation_attempts: int

    # Final publication
    published_report: Optional[Dict[str, Any]]


