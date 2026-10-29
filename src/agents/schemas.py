"""Pydantic V2 Schemas for Structured Output Validation in the Agentic Investigation Layer."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

from src.cases.models import CaseAction, CaseVerdict


class CitationDirection(str, Enum):
    """Directional force of a feature contribution."""
    FRAUD_PUSHING = "FRAUD_PUSHING"
    LEGITIMATE_MITIGATING = "LEGITIMATE_MITIGATING"


class EvidenceCitation(BaseModel):
    """Factual statistical attribution citation for an individual feature."""
    feature: str = Field(..., description="Feature name (e.g., 'V14', 'V17', 'Amount')")
    feature_value: float = Field(..., description="Observed feature value in transaction")
    shap_impact: float = Field(..., description="Exact SHAP attribution value in log-odds space")
    direction: CitationDirection = Field(..., description="Direction of risk impact")
    statistical_rationale: str = Field(
        ...,
        description="Factual statistical interpretation without unverified domain narratives",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "feature_value" not in data and "observed_value" in data:
                data["feature_value"] = data["observed_value"]
            if "shap_impact" not in data and "shap_value" in data:
                data["shap_impact"] = data["shap_value"]
            if "statistical_rationale" not in data and "rationale" in data:
                data["statistical_rationale"] = data["rationale"]
        return data

    @property
    def observed_value(self) -> float:
        return self.feature_value

    @property
    def shap_value(self) -> float:
        return self.shap_impact

    @property
    def rationale(self) -> str:
        return self.statistical_rationale


class GroundingCheck(BaseModel):
    """Self-audit checking for factual grounding and absence of hallucinated domain stories."""
    claims_verified: List[str] = Field(
        default_factory=list,
        description="List of factual claims verified against database and SHAP evidence",
    )
    respects_anonymization: bool = Field(
        default=True,
        description="True if V1-V28 are strictly treated as statistical PCA components",
    )
    unsupported_claims_found: List[str] = Field(
        default_factory=list,
        description="Any unverified assumptions or hallucinated domain narratives detected",
    )


class InvestigationReport(BaseModel):
    """Complete structured investigation report produced by the agent."""
    case_id: str = Field(..., description="Unique case identifier (e.g. 'CASE-XXXXXXXX')")
    alert_id: str = Field(..., description="Unique linked alert identifier")
    executive_summary: str = Field(
        ...,
        min_length=20,
        description="High-level synthesis of transaction risk and evidence findings",
    )
    supervised_signal_analysis: str = Field(
        ...,
        description="Analysis of XGBoost probability and calibrated threshold comparison",
    )
    anomaly_signal_analysis: str = Field(
        ...,
        description="Analysis of Isolation Forest unsupervised score and population outlier tier",
    )
    operational_quadrant: str = Field(
        ...,
        description="Operational risk quadrant: CONFIRMED_HIGH_RISK, SUBTLE_IN_DISTRIBUTION_FRAUD, NOVEL_ANOMALY_OUTLIER, or CONFIRMED_LEGITIMATE",
    )
    key_evidence_citations: List[EvidenceCitation] = Field(
        ...,
        min_length=1,
        description="Primary feature attributions driving the recommendation",
    )
    recommended_verdict: str = Field(
        ...,
        description="Recommended verdict: CONFIRMED_FRAUD, FALSE_POSITIVE, SUSPICIOUS_UNVERIFIED, or LEGITIMATE",
    )
    recommended_action: str = Field(
        ...,
        description="Recommended action: CARD_BLOCKED, CHARGEBACK_INITIATED, TRANSACTION_RELEASED, CUSTOMER_CONTACTED, or NONE",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence level in recommendation from 0.0 to 1.0",
    )
    justification: str = Field(
        ...,
        min_length=20,
        description="Detailed investigative rationale linking evidence to verdict and action",
    )
    investigation_findings: Optional[str] = Field(
        default=None,
        description="Detailed investigative findings",
    )
    anomaly_context: Optional[str] = Field(
        default=None,
        description="Anomaly context details",
    )
    case_precedent: Optional[str] = Field(
        default=None,
        description="Historical case precedent",
    )
    grounding_check: GroundingCheck = Field(
        default_factory=GroundingCheck,
        description="Self-audit of evidence grounding and strict anonymization integrity",
    )
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO UTC timestamp of report generation",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_report_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "justification" not in data and "investigation_findings" in data:
                data["justification"] = data["investigation_findings"]
            elif "investigation_findings" not in data and "justification" in data:
                data["investigation_findings"] = data["justification"]
            if "supervised_signal_analysis" not in data:
                data["supervised_signal_analysis"] = data.get("justification", "Supervised signal analysis completed.")
            if "anomaly_signal_analysis" not in data:
                data["anomaly_signal_analysis"] = data.get("anomaly_context", "Anomaly signal analysis completed.")
            if "operational_quadrant" not in data:
                data["operational_quadrant"] = "CONFIRMED_HIGH_RISK"
        return data

    @field_validator("recommended_verdict")
    @classmethod
    def validate_verdict(cls, v: str) -> str:
        clean = v.upper().strip()
        valid = [item.value for item in CaseVerdict]
        if clean not in valid:
            raise ValueError(f"Invalid verdict '{v}'. Must be one of {valid}")
        return clean

    @field_validator("recommended_action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        clean = v.upper().strip()
        valid = [item.value for item in CaseAction]
        if clean not in valid:
            raise ValueError(f"Invalid action '{v}'. Must be one of {valid}")
        return clean


class InvestigatorApprovalRequest(BaseModel):
    """Payload to record an investigator's review, approval, and application of report recommendations."""
    case_id: str
    action_approved: bool = Field(default=True, description="Whether investigator approves agent recommendations")
    final_verdict: str
    final_action: str
    analyst_comments: Optional[str] = None
    investigator_id: str = Field(default="lead_investigator")


# =============================================================================
# FRAUD TRIAGE & PRIORITIZATION AGENT SCHEMAS
# =============================================================================

class TriagePriorityTier(str, Enum):
    """Operational priority levels for queued alerts."""
    TIER_1_CRITICAL = "TIER_1_CRITICAL"
    TIER_2_HIGH = "TIER_2_HIGH"
    TIER_3_MEDIUM = "TIER_3_MEDIUM"
    TIER_4_LOW = "TIER_4_LOW"


class TriageAction(str, Enum):
    """Recommended next operational step for an alert in the queue."""
    ESCALATE_TO_CASE = "ESCALATE_TO_CASE"
    PRIORITIZE_CASE_INVESTIGATION = "PRIORITIZE_CASE_INVESTIGATION"
    BATCH_REVIEW = "BATCH_REVIEW"
    DISMISS_FALSE_POSITIVE = "DISMISS_FALSE_POSITIVE"
    MONITOR_ONLY = "MONITOR_ONLY"


class InvestigatorWorkload(BaseModel):
    """Investigator profile, specialization, and real-time workload capacity."""
    investigator_id: str = Field(..., description="Unique investigator username/ID")
    name: str = Field(..., description="Full display name")
    role: str = Field(..., description="Operational title or seniority")
    specialties: List[str] = Field(default_factory=list, description="Risk tiers or fraud patterns of expertise")
    active_cases: int = Field(default=0, ge=0, description="Current number of active open cases")
    max_capacity: int = Field(default=10, gt=0, description="Maximum concurrent case capacity")
    capacity_utilization: float = Field(default=0.0, ge=0.0, description="Ratio of active cases to max capacity (0.0 to 1.0+)")
    is_available: bool = Field(default=True, description="Whether the investigator is actively on duty")


class AlertTriageItem(BaseModel):
    """Prioritized queue entry with deterministic scores, tier, and evidence-grounded explanation."""
    alert_id: str = Field(..., description="Unique alert identifier (e.g. 'ALT-XXXXXXXX')")
    rank: int = Field(..., ge=1, description="Deterministic rank position in the alert queue (1=highest urgency)")
    composite_urgency_score: float = Field(..., ge=0.0, le=100.0, description="Deterministic multi-signal urgency score (0-100)")
    priority_tier: str = Field(..., description="Operational priority tier: TIER_1_CRITICAL, TIER_2_HIGH, TIER_3_MEDIUM, TIER_4_LOW")
    fraud_probability: float = Field(..., ge=0.0, le=1.0, description="Supervised fraud probability from XGBoost")
    anomaly_score: float = Field(..., ge=0.0, le=1.0, description="Unsupervised outlier score from Isolation Forest")
    amount: float = Field(..., ge=0.0, description="Transaction monetary exposure in Euros")
    policy_reasons: List[str] = Field(default_factory=list, description="Deterministic policy rules triggered")
    triage_explanation: str = Field(
        ...,
        min_length=15,
        description="Concise, evidence-grounded justification for rank, priority, and next steps without ungrounded stories",
    )
    recommended_action: str = Field(
        ...,
        description="Recommended operational action: ESCALATE_TO_CASE, PRIORITIZE_CASE_INVESTIGATION, BATCH_REVIEW, DISMISS_FALSE_POSITIVE, MONITOR_ONLY",
    )
    recommended_investigator: Optional[str] = Field(
        default=None,
        description="Recommended investigator based on workload, availability, and expertise",
    )
    assignment_rationale: Optional[str] = Field(
        default=None,
        description="Evidence-grounded rationale for the workload-aware investigator assignment",
    )
    case_id: Optional[str] = Field(default=None, description="Linked case ID if already escalated")

    @field_validator("priority_tier")
    @classmethod
    def validate_tier(cls, v: str) -> str:
        clean = v.upper().strip()
        valid = [item.value for item in TriagePriorityTier]
        if clean not in valid:
            raise ValueError(f"Invalid priority tier '{v}'. Must be one of {valid}")
        return clean

    @field_validator("recommended_action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        clean = v.upper().strip()
        valid = [item.value for item in TriageAction]
        if clean not in valid:
            raise ValueError(f"Invalid recommended action '{v}'. Must be one of {valid}")
        return clean


class QueueOverview(BaseModel):
    """Executive macro synthesis of the triaged alert backlog."""
    total_alerts_evaluated: int = Field(..., ge=0, description="Total count of evaluated alerts in this batch")
    tier_distribution: Dict[str, int] = Field(default_factory=dict, description="Counts of alerts per priority tier")
    queue_risk_synthesis: str = Field(
        ...,
        min_length=20,
        description="Concise executive summary of queue risk distribution, dominant threats, and backlog urgency",
    )
    workload_bottlenecks: Optional[str] = Field(
        default=None,
        description="Summary of analyst workload utilization, available capacity, and potential bottlenecks",
    )


class TriageBatchReport(BaseModel):
    """Complete structured output produced by the Fraud Triage & Prioritization Agent."""
    batch_id: str = Field(..., description="Unique batch identifier for the triage run")
    triaged_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO UTC timestamp of triage run",
    )
    queue_overview: QueueOverview = Field(..., description="Executive macro overview of the triage queue")
    prioritized_queue: List[AlertTriageItem] = Field(
        default_factory=list,
        description="Ordered list of triaged alerts sorted by priority rank",
    )
    grounding_audit_passed: bool = Field(
        default=True,
        description="Whether the triage explanations passed factual grounding verification",
    )


class TriageDecisionApplication(BaseModel):
    """Single alert action decision submitted by an investigator or supervisor."""
    alert_id: str
    action: str = Field(..., description="Action to take: ESCALATE_TO_CASE, DISMISS_FALSE_POSITIVE, UPDATE_PRIORITY, etc.")
    assigned_investigator: Optional[str] = None
    priority_override: Optional[str] = None
    supervisor_notes: Optional[str] = None


class TriageApprovalBatchRequest(BaseModel):
    """Batch payload to execute human supervisor review approvals and assignments."""
    supervisor_id: str = Field(default="fraud_supervisor", description="ID of supervisor approving decisions")
    decisions: List[TriageDecisionApplication] = Field(..., min_length=1, description="Decisions to apply across alerts")


# =============================================================================
# FRAUD PATTERN ANALYSIS AGENT SCHEMAS
# =============================================================================

class PatternCategory(str, Enum):
    """Classification of detected fraud patterns and distribution shifts."""
    ANOMALY_CONCENTRATION = "ANOMALY_CONCENTRATION"
    VOLUME_VELOCITY = "VOLUME_VELOCITY"
    MODEL_PERFORMANCE_SHIFT = "MODEL_PERFORMANCE_SHIFT"
    EXPOSURE_DRIFT = "EXPOSURE_DRIFT"
    STATISTICAL_DISTRIBUTION_SHIFT = "STATISTICAL_DISTRIBUTION_SHIFT"
    RISK_TIER_MIGRATION = "RISK_TIER_MIGRATION"


class ShiftSeverity(str, Enum):
    """Urgency severity tier for detected pattern shifts."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class StatisticalConfidence(str, Enum):
    """Confidence level of statistical findings."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class TimeWindowCohort(BaseModel):
    """Deterministic summary statistics for a discrete temporal window cohort."""
    window_name: str = Field(..., description="Label of the time window (e.g. 'Baseline Window', 'Recent Window')")
    start_time: Optional[str] = Field(default=None, description="Start timestamp of the window")
    end_time: Optional[str] = Field(default=None, description="End timestamp of the window")
    alert_count: int = Field(default=0, ge=0, description="Total count of alerts in the window")
    mean_amount: float = Field(default=0.0, ge=0.0, description="Mean transaction amount in Euros")
    median_amount: float = Field(default=0.0, ge=0.0, description="Median transaction amount in Euros")
    p90_amount: float = Field(default=0.0, ge=0.0, description="90th percentile transaction amount")
    mean_fraud_probability: float = Field(default=0.0, ge=0.0, le=1.0, description="Mean supervised fraud probability")
    high_fraud_share_pct: float = Field(default=0.0, ge=0.0, le=100.0, description="Percentage of alerts with P(Fraud) >= 0.70")
    mean_anomaly_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Mean unsupervised anomaly score")
    high_anomaly_share_pct: float = Field(default=0.0, ge=0.0, le=100.0, description="Percentage of alerts with Anomaly Score >= 0.70")
    tier_distribution: Dict[str, int] = Field(default_factory=dict, description="Counts of alerts per priority tier")
    quadrant_distribution: Dict[str, int] = Field(default_factory=dict, description="Counts of alerts per operational quadrant")
    verified_outcomes_count: int = Field(default=0, ge=0, description="Count of cases with verified outcomes in this window")
    confirmed_fraud_count: int = Field(default=0, ge=0, description="Count of confirmed fraud cases")
    false_positive_count: int = Field(default=0, ge=0, description="Count of verified false positive cases")
    false_positive_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="False positive rate among verified cases")
    precision: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Precision among verified cases")


class DetectedPatternShift(BaseModel):
    """Measured statistical shift between baseline and recent window cohorts."""
    metric_name: str = Field(..., description="Descriptive identifier of the measured metric")
    baseline_value: float = Field(..., description="Metric value during the baseline comparison window")
    recent_value: float = Field(..., description="Metric value during the recent analysis window")
    absolute_change: float = Field(..., description="recent_value - baseline_value")
    relative_change_pct: float = Field(..., description="Percentage shift relative to baseline: ((recent - baseline) / baseline) * 100")
    severity: str = Field(default="INFO", description="Urgency severity: CRITICAL, HIGH, MEDIUM, LOW, INFO")
    is_significant: bool = Field(default=False, description="Whether the shift meets statistical significance thresholds")
    shift_summary: str = Field(..., min_length=10, description="Concise deterministic summary of the shift")

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        clean = v.upper().strip()
        valid = [item.value for item in ShiftSeverity]
        if clean not in valid:
            raise ValueError(f"Invalid severity '{v}'. Must be one of {valid}")
        return clean


class PatternFindingItem(BaseModel):
    """Evidence-backed analytical finding synthesized from computed pattern shifts."""
    pattern_title: str = Field(..., min_length=5, description="Clear, descriptive title for the discovered pattern")
    category: str = Field(..., description="Category: ANOMALY_CONCENTRATION, VOLUME_VELOCITY, MODEL_PERFORMANCE_SHIFT, EXPOSURE_DRIFT, STATISTICAL_DISTRIBUTION_SHIFT, RISK_TIER_MIGRATION")
    measured_pattern: str = Field(..., min_length=15, description="Precise measured pattern citing exact counts, rates, and window comparisons")
    supporting_statistics: Dict[str, Any] = Field(default_factory=dict, description="Underlying numerical facts, counts, percentages, and comparison timeframes")
    statistical_confidence: str = Field(default="MEDIUM", description="Confidence level: HIGH, MEDIUM, LOW")
    dataset_limitations: str = Field(
        ...,
        min_length=15,
        description="Explicit limitation stating V1-V28 are anonymized PCA components and do not identify merchants, locations, or cardholder behaviors",
    )
    recommended_followup: str = Field(..., min_length=15, description="Actionable follow-up steps (rules, thresholds, retraining, or queue handling)")

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        clean = v.upper().strip()
        valid = [item.value for item in PatternCategory]
        if clean not in valid:
            raise ValueError(f"Invalid category '{v}'. Must be one of {valid}")
        return clean

    @field_validator("statistical_confidence")
    @classmethod
    def validate_confidence(cls, v: str) -> str:
        clean = v.upper().strip()
        valid = [item.value for item in StatisticalConfidence]
        if clean not in valid:
            raise ValueError(f"Invalid statistical_confidence '{v}'. Must be one of {valid}")
        return clean


class CohortTrendAnalysisOutput(BaseModel):
    """Intermediate analytical synthesis of cohort dynamics and model error shifts (processed by specialized LLM)."""
    cohort_trend_insights: str = Field(
        ...,
        min_length=15,
        description="Analysis of cohort differences between baseline and recent periods (amounts, velocity, anomaly concentrations)",
    )
    model_error_analysis: str = Field(
        ...,
        min_length=15,
        description="Focused analysis of verified false positive rates, precision trends, and detection efficacy",
    )


class PatternFindingsBatchOutput(BaseModel):
    """Batch container for structured pattern findings (processed by specialized LLM)."""
    findings: List[PatternFindingItem] = Field(
        ...,
        min_length=1,
        description="Formulated list of evidence-backed findings strictly adhering to anonymized PCA constraints",
    )


class PatternAnalysisReport(BaseModel):
    """Complete structured report produced by the Fraud Pattern Analysis Agent."""
    report_id: str = Field(..., description="Unique report identifier (e.g. 'PAT-XXXXXXXX')")
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO UTC timestamp of report generation",
    )
    executive_summary: str = Field(
        ...,
        min_length=25,
        description="High-level synthesis of historical trends, emerging shifts, and operational impacts",
    )
    baseline_window: TimeWindowCohort = Field(..., description="Cohort metrics for the baseline comparison period")
    recent_window: TimeWindowCohort = Field(..., description="Cohort metrics for the recent evaluation period")
    detected_shifts: List[DetectedPatternShift] = Field(
        default_factory=list,
        description="List of measured statistical shifts between cohorts",
    )
    findings: List[PatternFindingItem] = Field(
        ...,
        min_length=1,
        description="Structured, evidence-backed findings interpreting computed pattern shifts",
    )
    model_error_analysis: Optional[str] = Field(
        default=None,
        description="Analysis of verified false positive and false negative trends across cohorts",
    )
    dataset_limitations_acknowledged: bool = Field(
        default=True,
        description="Affirmation that anonymized PCA features are treated strictly as mathematical components",
    )
    grounding_audit_passed: bool = Field(
        default=True,
        description="Whether all findings strictly adhere to computed numbers without hallucinated domain narratives",
    )
    models_used: Optional[Dict[str, str]] = Field(
        default=None,
        description="Breakdown of models used across the distributed pattern analysis pipeline",
    )



