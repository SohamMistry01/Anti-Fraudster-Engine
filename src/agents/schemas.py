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
