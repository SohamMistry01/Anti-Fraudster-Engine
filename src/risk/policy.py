"""Risk policy engine determining alert triggers, priority ranking, and automatic case escalation."""

from typing import Any, Dict, List, Tuple

from src.cases.models import AlertPriority


def evaluate_alert_policy(
    prediction: Dict[str, Any],
    anomaly: Dict[str, Any],
    transaction: Dict[str, Any],
) -> Tuple[bool, str, List[str]]:
    """Evaluate multi-signal risk policies to determine if an alert should be created and its priority.

    Args:
        prediction: Supervised XGBoost scoring result.
        anomaly: Unsupervised Isolation Forest scoring result.
        transaction: Raw or preprocessed transaction features dictionary.

    Returns:
        Tuple of (should_alert: bool, priority: str, reasons: List[str])
    """
    p_fraud = float(prediction.get("fraud_probability", 0.0))
    is_fraud = bool(prediction.get("is_fraud", False))
    thresh_used = float(prediction.get("threshold_used", 0.5))

    anom_score = float(anomaly.get("anomaly_score", 0.0))
    is_anom = bool(anomaly.get("is_anomaly", False))
    quadrant = prediction.get("risk_category", "")

    amt = float(transaction.get("Amount", 0.0))
    v14 = float(transaction.get("V14", 0.0))
    v4 = float(transaction.get("V4", 0.0))
    v10 = float(transaction.get("V10", 0.0))

    reasons: List[str] = []

    # 1. Supervised probability rules
    if p_fraud >= 0.80:
        reasons.append(f"Critical fraud probability: {p_fraud:.2%} (well above 0.80 cutoff).")
    elif is_fraud:
        reasons.append(f"Fraud probability ({p_fraud:.2%}) exceeds decision threshold ({thresh_used:.3f}).")
    elif p_fraud >= 0.182:
        reasons.append(f"Fraud probability ({p_fraud:.2%}) exceeds cost-optimal threshold (0.182).")

    # 2. Unsupervised outlier rules
    if is_anom or anom_score >= 0.75:
        reasons.append(f"Isolation Forest flagged statistical population outlier (anomaly score: {anom_score:.2f}).")
    elif anom_score >= 0.60:
        reasons.append(f"Elevated unsupervised anomaly score: {anom_score:.2f}.")

    # 3. High-dollar exposure rules
    if amt >= 500.0 and (p_fraud >= 0.182 or anom_score >= 0.60):
        reasons.append(f"High-exposure monetary transaction (${amt:.2f}) with elevated risk metrics.")

    # 4. Statistical feature deviations on dominant indicators
    if v14 <= -3.5:
        reasons.append(f"Severe negative deviation in primary fraud indicator V14 ({v14:+.2f}).")
    if v4 >= 3.5:
        reasons.append(f"Elevated positive deviation in primary fraud driver V4 ({v4:+.2f}).")
    if v10 <= -3.0:
        reasons.append(f"Depressed negative value in top indicator V10 ({v10:+.2f}).")

    # Decide if alert should fire
    should_alert = len(reasons) > 0 or p_fraud >= 0.182 or is_anom

    if not should_alert:
        return False, AlertPriority.LOW.value, []

    # 5. Priority determination
    if p_fraud >= 0.80 or (p_fraud >= 0.50 and (amt >= 500.0 or anom_score >= 0.75)):
        priority = AlertPriority.CRITICAL.value
    elif p_fraud >= 0.50 or anom_score >= 0.75:
        priority = AlertPriority.HIGH.value
    elif p_fraud >= 0.182 or anom_score >= 0.60:
        priority = AlertPriority.MEDIUM.value
    else:
        priority = AlertPriority.LOW.value

    return True, priority, reasons


def should_auto_escalate_case(priority: str, fraud_probability: float) -> bool:
    """Determine whether an incoming alert should automatically open a case immediately."""
    return priority == AlertPriority.CRITICAL.value or fraud_probability >= 0.85
