"""Read-only deterministic retrieval tools for the Agentic Investigation Layer.

These tools orchestrate the existing inference, explainability, anomaly detection,
and persistence foundations without duplicating business logic.
"""

import logging
from typing import Any, Dict, List, Optional
from langchain_core.tools import tool

from src.anomaly.detector import get_anomaly_detector
from src.cases.database import get_db_context
from src.cases.repository import AlertRepository, CaseRepository
from src.explainability.shap_explainer import get_explainer

logger = logging.getLogger(__name__)


@tool
def get_alert_details(alert_id: str) -> Dict[str, Any]:
    """Retrieve complete metadata and 30-feature vector for an alert.

    Args:
        alert_id: Unique alert identifier (e.g. 'ALT-XXXXXXXX').

    Returns:
        Dictionary containing alert status, urgency priority, fraud probability,
        anomaly score, trigger policy reasons, amount, time, and full transaction payload.
    """
    with get_db_context() as db:
        alert = AlertRepository.get_by_id(db, alert_id)
        if not alert:
            return {"error": f"Alert '{alert_id}' was not found in the queue."}
        return alert.to_dict()


@tool
def get_shap_explanation(alert_id: str, top_n: int = 5) -> Dict[str, Any]:
    """Calculate and return exact TreeSHAP feature attributions for an alert's transaction.

    Args:
        alert_id: Unique alert identifier (e.g. 'ALT-XXXXXXXX').
        top_n: Number of top driving and mitigating features to return (default: 5).

    Returns:
        Dictionary containing the baseline log-odds, top positive fraud-pushing features,
        top negative mitigating features, and statistical evidence summary.
    """
    with get_db_context() as db:
        alert = AlertRepository.get_by_id(db, alert_id)
        if not alert:
            return {"error": f"Alert '{alert_id}' not found."}
        tx_data = alert.transaction_payload

    # Auto-detect preprocessing: negative values indicate RobustScaled data
    apply_prep = not (float(tx_data.get("Amount", 0.0)) < 0.0 or float(tx_data.get("Time", 0.0)) < 0.0)

    explainer = get_explainer()
    explanation = explainer.explain_transaction(
        transaction=tx_data,
        top_n=top_n,
        apply_preprocessing=apply_prep,
        generate_plot=False,
    )

    return {
        "alert_id": alert_id,
        "base_value": explanation["base_value"],
        "fraud_probability": explanation["fraud_probability"],
        "risk_category": explanation["risk_category"],
        "top_positive_contributors": explanation["top_positive_contributors"],
        "top_negative_contributors": explanation["top_negative_contributors"],
        "evidence_summary": explanation["evidence_summary"],
    }


@tool
def get_anomaly_score(alert_id: str) -> Dict[str, Any]:
    """Retrieve the independent Isolation Forest unsupervised anomaly assessment.

    Args:
        alert_id: Unique alert identifier (e.g. 'ALT-XXXXXXXX').

    Returns:
        Dictionary with normalized anomaly score (0.0=normal, 1.0=extreme outlier),
        raw decision function score, boolean is_anomaly flag, and anomaly tier.
    """
    with get_db_context() as db:
        alert = AlertRepository.get_by_id(db, alert_id)
        if not alert:
            return {"error": f"Alert '{alert_id}' not found."}
        tx_data = alert.transaction_payload

    apply_prep = not (float(tx_data.get("Amount", 0.0)) < 0.0 or float(tx_data.get("Time", 0.0)) < 0.0)

    detector = get_anomaly_detector()
    score_data = detector.score_transaction(
        transaction=tx_data,
        apply_preprocessing=apply_prep,
    )

    return {
        "alert_id": alert_id,
        "anomaly_score": score_data["anomaly_score"],
        "raw_score": score_data["raw_score"],
        "is_anomaly": score_data["is_anomaly"],
        "anomaly_tier": score_data["anomaly_tier"],
        "algorithm": score_data["algorithm"],
    }


@tool
def get_case_history(case_id: str) -> Dict[str, Any]:
    """Retrieve full investigation case details, including sealed ML snapshot, notes, and audit logs.

    Args:
        case_id: Unique case identifier (e.g. 'CASE-XXXXXXXX').

    Returns:
        Dictionary containing current status, priority, assigned analyst, verdict,
        action taken, sealed original prediction snapshot, investigator notes feed,
        and chronological audit trail.
    """
    with get_db_context() as db:
        case = CaseRepository.get_by_id(db, case_id)
        if not case:
            return {"error": f"Case '{case_id}' was not found in the repository."}
        return case.to_dict(include_relations=True)


INVESTIGATION_TOOLS = [
    get_alert_details,
    get_shap_explanation,
    get_anomaly_score,
    get_case_history,
]
