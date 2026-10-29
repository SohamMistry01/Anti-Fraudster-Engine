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


# =============================================================================
# TRIAGE & PRIORITIZATION RETRIEVAL TOOLS
# =============================================================================

DEFAULT_INVESTIGATOR_ROSTER: List[Dict[str, Any]] = [
    {
        "investigator_id": "analyst_mistry",
        "name": "Soham Mistry",
        "role": "Senior Fraud Specialist",
        "specialties": ["TIER_1_CRITICAL", "NOVEL_ANOMALY_OUTLIER", "HIGH_EXPOSURE"],
        "max_capacity": 10,
        "is_available": True,
    },
    {
        "investigator_id": "lead_analyst",
        "name": "Sarah Connor",
        "role": "Lead Investigator",
        "specialties": ["TIER_1_CRITICAL", "CONFIRMED_HIGH_RISK", "COMPLEX_ESCALATION"],
        "max_capacity": 8,
        "is_available": True,
    },
    {
        "investigator_id": "analyst_chen",
        "name": "David Chen",
        "role": "Fraud Operations Analyst",
        "specialties": ["TIER_2_HIGH", "TIER_3_MEDIUM", "SUBTLE_IN_DISTRIBUTION_FRAUD"],
        "max_capacity": 12,
        "is_available": True,
    },
    {
        "investigator_id": "analyst_patel",
        "name": "Amina Patel",
        "role": "Junior Triage Analyst",
        "specialties": ["TIER_3_MEDIUM", "TIER_4_LOW", "FAST_TRACK_VERIFICATION"],
        "max_capacity": 15,
        "is_available": True,
    },
]


@tool
def get_pending_alerts_for_triage(limit: int = 15, status: Optional[str] = "NEW") -> List[Dict[str, Any]]:
    """Retrieve pending or unescalated alerts from the database queue for triage evaluation.

    Args:
        limit: Maximum number of alerts to fetch (default: 15).
        status: Status filter ('NEW', 'ESCALATED', or None for all active alerts).

    Returns:
        List of alert dictionaries containing multi-signal scores and payload attributes.
    """
    with get_db_context() as db:
        s_val = None if status in (None, "ALL", "") else status.upper()
        alerts, _ = AlertRepository.list_alerts(db=db, status=s_val, limit=limit)
        return [a.to_dict() for a in alerts]


@tool
def get_investigator_workloads() -> List[Dict[str, Any]]:
    """Retrieve current active case counts, capacity limits, and utilization for all fraud investigators.

    Returns:
        List of investigator workload dictionaries with real-time active case counts and capacity ratios.
    """
    from sqlalchemy import func
    from src.cases.models import CaseStatus, InvestigationCase

    active_statuses = [
        CaseStatus.OPEN.value,
        CaseStatus.UNDER_INVESTIGATION.value,
        CaseStatus.PENDING_CUSTOMER.value,
    ]

    with get_db_context() as db:
        counts = dict(
            db.query(InvestigationCase.assigned_analyst, func.count(InvestigationCase.id))
            .filter(InvestigationCase.status.in_(active_statuses))
            .group_by(InvestigationCase.assigned_analyst)
            .all()
        )

    roster: List[Dict[str, Any]] = []
    for inv in DEFAULT_INVESTIGATOR_ROSTER:
        inv_id = inv["investigator_id"]
        active = int(counts.get(inv_id, 0))
        max_cap = int(inv.get("max_capacity", 10))
        util = round(active / max(max_cap, 1), 2)
        roster.append({
            "investigator_id": inv_id,
            "name": inv["name"],
            "role": inv["role"],
            "specialties": list(inv.get("specialties", [])),
            "active_cases": active,
            "max_capacity": max_cap,
            "capacity_utilization": util,
            "is_available": inv.get("is_available", True) and active < max_cap,
        })

    return roster


TRIAGE_TOOLS = [
    get_pending_alerts_for_triage,
    get_investigator_workloads,
    get_alert_details,
]


# =============================================================================
# FRAUD PATTERN ANALYSIS RETRIEVAL TOOLS
# =============================================================================

@tool
def get_historical_pattern_data(
    limit: int = 500,
    time_horizon_hours: Optional[int] = None,
) -> Dict[str, Any]:
    """Retrieve historical alerts, transaction scores, and resolved case outcomes for pattern analysis.

    Args:
        limit: Maximum number of historical records to retrieve (default: 500).
        time_horizon_hours: Optional lookback window filter in hours (e.g. 72, 168).

    Returns:
        Dictionary containing historical records, alert counts, case verdict breakdowns,
        and temporal span metadata.
    """
    from datetime import datetime, timedelta, timezone
    from src.cases.models import Alert, InvestigationCase

    with get_db_context() as db:
        query = db.query(Alert)
        if time_horizon_hours is not None and time_horizon_hours > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=time_horizon_hours)
            query = query.filter(Alert.created_at >= cutoff)

        alerts = query.order_by(Alert.created_at.asc()).limit(limit).all()

        cases = db.query(InvestigationCase).all()
        cases_by_alert = {c.alert_id: c for c in cases if c.alert_id}
        cases_by_tx = {c.transaction_id: c for c in cases if c.transaction_id}

        records: List[Dict[str, Any]] = []
        for a in alerts:
            rec = a.to_dict()
            linked_case = cases_by_alert.get(a.alert_id) or cases_by_tx.get(a.transaction_id)
            if linked_case:
                rec["case_id"] = linked_case.case_id
                rec["case_status"] = linked_case.status
                rec["verdict"] = linked_case.verdict
                rec["action_taken"] = linked_case.action_taken
                rec["resolved_at"] = (
                    linked_case.resolved_at.isoformat() if linked_case.resolved_at else None
                )
            else:
                rec["case_status"] = None
                rec["verdict"] = None
                rec["action_taken"] = None
                rec["resolved_at"] = None

            records.append(rec)

    return {
        "total_records": len(records),
        "records": records,
    }


PATTERN_TOOLS = [
    get_historical_pattern_data,
    get_alert_details,
]


