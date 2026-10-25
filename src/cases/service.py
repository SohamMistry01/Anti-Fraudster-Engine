"""Case service coordinating transaction risk assessment, alert queue generation,

case creation, and analyst workflows.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from sqlalchemy.orm import Session

from src.cases.models import (
    Alert,
    AlertPriority,
    AlertStatus,
    CaseAction,
    CaseStatus,
    CaseVerdict,
    InvestigationCase,
)
from src.cases.repository import AlertRepository, CaseRepository
from src.explainability.risk_assessor import IntegratedRiskAssessor, get_risk_assessor
from src.risk.policy import evaluate_alert_policy, should_auto_escalate_case

logger = logging.getLogger(__name__)


class CaseService:
    """Service orchestrating alert triage, immutable case recording, and investigation lifecycles."""

    def __init__(self, risk_assessor: Optional[IntegratedRiskAssessor] = None) -> None:
        """Initialize case service with the integrated risk assessor."""
        self.assessor = risk_assessor or get_risk_assessor()

    def process_and_triage_transaction(
        self,
        db: Session,
        transaction: Dict[str, Any],
        threshold: Union[float, str] = "default",
        apply_preprocessing: bool = True,
        auto_escalate: bool = True,
    ) -> Dict[str, Any]:
        """Score incoming transaction, evaluate risk policies, and generate prioritized alert/case.

        Returns comprehensive diagnostic dictionary including the alert and case if created.
        """
        # 1. Multi-signal risk assessment
        assessment = self.assessor.assess_transaction(
            transaction=transaction,
            threshold=threshold,
            apply_preprocessing=apply_preprocessing,
            generate_plots=False,
        )

        supervised = assessment["supervised_signal"]
        anomaly = assessment["unsupervised_anomaly_signal"]
        shap_data = assessment["shap_explanation"]

        # 2. Risk policy evaluation
        should_alert, priority, reasons = evaluate_alert_policy(
            prediction=supervised,
            anomaly=anomaly,
            transaction=transaction,
        )

        alert_record: Optional[Alert] = None
        case_record: Optional[InvestigationCase] = None

        if should_alert:
            # 3. Create persistent Alert in queue
            alert_payload = {
                "amount": float(transaction.get("Amount", 0.0)),
                "time_offset": float(transaction.get("Time", 0.0)),
                "fraud_probability": supervised["fraud_probability"],
                "anomaly_score": anomaly["anomaly_score"],
                "risk_category": supervised["risk_category"],
                "priority": priority,
                "operational_quadrant": assessment["operational_quadrant"],
                "status": AlertStatus.NEW.value,
                "reasons": reasons,
                "transaction_payload": transaction,
            }
            alert_record = AlertRepository.create(db, alert_payload)

            # 4. Auto-escalate to case if policy criteria are met
            if auto_escalate and should_auto_escalate_case(priority, supervised["fraud_probability"]):
                case_record = self.open_case_from_alert(
                    db=db,
                    alert_id=alert_record.alert_id,
                    creator="auto_policy_engine",
                    initial_note=f"Auto-escalated based on priority={priority}. Trigger reasons: {'; '.join(reasons)}",
                    apply_preprocessing=apply_preprocessing,
                )

        return {
            "assessment": assessment,
            "alert_created": alert_record is not None,
            "alert": alert_record.to_dict() if alert_record else None,
            "case_created": case_record is not None,
            "case": case_record.to_dict(include_relations=False) if case_record else None,
        }

    def open_case_from_alert(
        self,
        db: Session,
        alert_id: str,
        assigned_analyst: Optional[str] = None,
        initial_note: Optional[str] = None,
        creator: str = "analyst",
        apply_preprocessing: Optional[bool] = None,
    ) -> InvestigationCase:
        """Promote an alert into an active investigation case, preserving original inference snapshot."""
        alert = AlertRepository.get_by_id(db, alert_id)
        if not alert:
            raise ValueError(f"Alert '{alert_id}' not found.")

        # Check if already escalated
        if alert.case_id:
            existing = CaseRepository.get_by_id(db, alert.case_id)
            if existing:
                return existing

        # Extract features and compute full explainability snapshot for case preservation
        tx_data = alert.transaction_payload

        # Auto-detect preprocessing requirement if not explicitly provided
        if apply_preprocessing is None:
            if float(tx_data.get("Amount", 0.0)) < 0.0 or float(tx_data.get("Time", 0.0)) < 0.0:
                prep = False
            else:
                prep = True
        else:
            prep = apply_preprocessing

        shap_details = self.assessor.explainer.explain_transaction(
            transaction=tx_data,
            apply_preprocessing=prep,
            top_n=10,
            generate_plot=False,
        )

        case_data = {
            "alert_id": alert.alert_id,
            "transaction_id": alert.transaction_id,
            "status": CaseStatus.OPEN.value,
            "priority": alert.priority,
            "assigned_analyst": assigned_analyst,
            # Immutable Original Prediction Snapshot
            "original_fraud_probability": alert.fraud_probability,
            "original_risk_category": alert.risk_category,
            "original_threshold_used": shap_details["threshold_used"],
            "original_threshold_name": shap_details["threshold_name"],
            "original_model_version": self.assessor.predictor.model_version,
            "original_anomaly_score": alert.anomaly_score,
            "original_anomaly_tier": (
                "OUTLIER" if alert.anomaly_score >= 0.75
                else "ELEVATED" if alert.anomaly_score >= 0.50
                else "NORMAL"
            ),
            "original_quadrant": alert.operational_quadrant,
            "original_shap_explanation": {
                "base_value": shap_details["base_value"],
                "top_positive_contributors": shap_details["top_positive_contributors"],
                "top_negative_contributors": shap_details["top_negative_contributors"],
                "evidence_summary": shap_details["evidence_summary"],
            },
            "transaction_snapshot": tx_data,
        }

        note_text = initial_note or f"Case opened from alert {alert.alert_id} (Trigger: {'; '.join(alert.reasons)})"

        case = CaseRepository.create(
            db=db,
            case_data=case_data,
            initial_note=note_text,
            creator=creator,
        )

        # Update alert with case linkage
        AlertRepository.update_status(
            db=db,
            alert_id=alert.alert_id,
            status=AlertStatus.CASE_CREATED.value,
            case_id=case.case_id,
        )

        return case

    def update_case_workflow(
        self,
        db: Session,
        case_id: str,
        status: Optional[str] = None,
        assigned_analyst: Optional[str] = None,
        verdict: Optional[str] = None,
        action_taken: Optional[str] = None,
        resolution_notes: Optional[str] = None,
        updated_by: str = "analyst",
    ) -> InvestigationCase:
        """Update case investigation progress and record state changes."""
        case = CaseRepository.update_case(
            db=db,
            case_id=case_id,
            status=status,
            assigned_analyst=assigned_analyst,
            verdict=verdict,
            action_taken=action_taken,
            resolution_notes=resolution_notes,
            updated_by=updated_by,
        )
        if not case:
            raise ValueError(f"InvestigationCase '{case_id}' does not exist.")
        return case

    def add_note(
        self,
        db: Session,
        case_id: str,
        author: str,
        content: str,
    ) -> Dict[str, Any]:
        """Add an analyst note to a case."""
        note = CaseRepository.add_note(db, case_id=case_id, author=author, content=content)
        return note.to_dict()


# Cached singleton
_CASE_SERVICE_INSTANCE: Optional[CaseService] = None


def get_case_service() -> CaseService:
    """Retrieve or initialize the cached singleton CaseService."""
    global _CASE_SERVICE_INSTANCE
    if _CASE_SERVICE_INSTANCE is None:
        _CASE_SERVICE_INSTANCE = CaseService()
    return _CASE_SERVICE_INSTANCE
