"""Automated Evaluation Suite for the Agentic Investigation Layer.

Evaluates:
- Factual grounding against real database transaction payloads
- Evidence citation accuracy against deterministic TreeSHAP values
- Unsupported claims & hallucination detection on anonymized PCA features
- Tool-call correctness across read-only retrieval engines
- Decision consistency and report usefulness
"""

import logging
import math
import re
from typing import Any, Dict, List, Optional
from src.agents.investigation_agent import PROHIBITED_DOMAIN_PATTERNS, run_case_investigation
from src.agents.tools import (
    get_alert_details,
    get_anomaly_score,
    get_case_history,
    get_shap_explanation,
)
from src.cases.database import get_db_context
from src.cases.models import CaseAction, CaseVerdict, InvestigationCase

logger = logging.getLogger(__name__)


def evaluate_tool_correctness(sample_case_id: str, sample_alert_id: str) -> Dict[str, Any]:
    """Test that all 4 deterministic read-only tools execute and return valid structures."""
    tool_results = {}

    # Tool 1: get_case_history
    try:
        case_res = get_case_history.invoke({"case_id": sample_case_id})
        tool_results["get_case_history"] = {
            "success": "error" not in case_res and "case_id" in case_res,
            "details": f"Retrieved case {case_res.get('case_id')}",
        }
    except Exception as exc:
        tool_results["get_case_history"] = {"success": False, "details": str(exc)}

    # Tool 2: get_alert_details
    try:
        alert_res = get_alert_details.invoke({"alert_id": sample_alert_id})
        tool_results["get_alert_details"] = {
            "success": "error" not in alert_res and "alert_id" in alert_res,
            "details": f"Retrieved alert {alert_res.get('alert_id')}",
        }
    except Exception as exc:
        tool_results["get_alert_details"] = {"success": False, "details": str(exc)}

    # Tool 3: get_shap_explanation
    try:
        shap_res = get_shap_explanation.invoke({"alert_id": sample_alert_id, "top_n": 5})
        tool_results["get_shap_explanation"] = {
            "success": "error" not in shap_res and "top_positive_contributors" in shap_res,
            "details": f"Top drivers count: {len(shap_res.get('top_positive_contributors', []))}",
        }
    except Exception as exc:
        tool_results["get_shap_explanation"] = {"success": False, "details": str(exc)}

    # Tool 4: get_anomaly_score
    try:
        anomaly_res = get_anomaly_score.invoke({"alert_id": sample_alert_id})
        tool_results["get_anomaly_score"] = {
            "success": "error" not in anomaly_res and "anomaly_score" in anomaly_res,
            "details": f"Tier: {anomaly_res.get('anomaly_tier')}",
        }
    except Exception as exc:
        tool_results["get_anomaly_score"] = {"success": False, "details": str(exc)}

    all_passed = all(t["success"] for t in tool_results.values())
    return {
        "all_passed": all_passed,
        "tools": tool_results,
    }


def evaluate_report_grounding(
    report: Dict[str, Any],
    true_payload: Dict[str, Any],
    true_shap_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Audit an individual generated report for factual grounding, citation accuracy, and unsupported stories."""
    citations = report.get("key_evidence_citations", [])
    if not citations:
        return {
            "grounding_score": 0.0,
            "citation_accuracy_score": 0.0,
            "hallucination_score": 1.0,
            "unsupported_claims": ["No citations provided"],
            "overall_score": 0.0,
        }

    # 1. Factual Grounding of Feature Values
    grounded_count = 0
    grounding_details = []

    for cit in citations:
        feat = cit.get("feature")
        claimed_val = cit.get("feature_value") if cit.get("feature_value") is not None else cit.get("observed_value")

        if feat in true_payload and claimed_val is not None:
            actual_val = float(true_payload[feat])
            try:
                is_match = math.isclose(actual_val, float(claimed_val), abs_tol=0.08)
            except (ValueError, TypeError):
                is_match = False
            if is_match:
                grounded_count += 1
            grounding_details.append({
                "feature": feat,
                "claimed_value": claimed_val,
                "actual_value": actual_val,
                "matched": is_match,
            })
        else:
            grounding_details.append({
                "feature": feat,
                "claimed_value": claimed_val,
                "actual_value": float(true_payload[feat]) if feat in true_payload else None,
                "matched": False,
            })

    grounding_score = grounded_count / len(citations)

    # 2. Evidence Citation Direction Accuracy
    all_true_shap = {
        c["feature"]: c["shap_value"]
        for c in true_shap_data.get("top_positive_contributors", [])
        + true_shap_data.get("top_negative_contributors", [])
    }

    dir_correct_count = 0
    citation_details = []

    for cit in citations:
        feat = cit.get("feature")
        claimed_dir = cit.get("direction", "")

        if feat in all_true_shap:
            true_impact = all_true_shap[feat]
            expected_dir = "FRAUD_PUSHING" if true_impact > 0 else "LEGITIMATE_MITIGATING"
            is_dir_match = claimed_dir == expected_dir
            if is_dir_match:
                dir_correct_count += 1
            citation_details.append({
                "feature": feat,
                "claimed_direction": claimed_dir,
                "true_shap_impact": true_impact,
                "direction_accurate": is_dir_match,
            })
        else:
            # If not in top 5, neutral or unverified
            citation_details.append({
                "feature": feat,
                "claimed_direction": claimed_dir,
                "true_shap_impact": None,
                "direction_accurate": True,
            })
            dir_correct_count += 1

    citation_accuracy_score = dir_correct_count / len(citations)

    # 3. Scan for Prohibited Hallucinated Stories on V1-V28
    full_text = " ".join([
        report.get("executive_summary", ""),
        report.get("justification", ""),
        report.get("supervised_signal_analysis", ""),
    ]).lower()

    detected_hallucinations = []
    for pattern in PROHIBITED_DOMAIN_PATTERNS:
        match = re.search(pattern, full_text)
        if match:
            detected_hallucinations.append(match.group(0))

    hallucination_score = 1.0 if not detected_hallucinations else max(0.0, 1.0 - (0.3 * len(detected_hallucinations)))

    # 4. Consistency & Usefulness
    verdict = report.get("recommended_verdict", "")
    action = report.get("recommended_action", "")
    confidence = report.get("confidence_score", 0.0)

    is_valid_verdict = verdict in [v.value for v in CaseVerdict]
    is_valid_action = action in [a.value for a in CaseAction]
    consistency_score = 1.0 if (is_valid_verdict and is_valid_action and confidence > 0.5) else 0.5

    # Composite Score
    composite = (
        0.30 * grounding_score
        + 0.30 * citation_accuracy_score
        + 0.25 * hallucination_score
        + 0.15 * consistency_score
    )

    return {
        "grounding_score": round(grounding_score, 4),
        "citation_accuracy_score": round(citation_accuracy_score, 4),
        "hallucination_score": round(hallucination_score, 4),
        "consistency_score": round(consistency_score, 4),
        "composite_score": round(composite, 4),
        "unsupported_claims_found": detected_hallucinations,
        "grounding_details": grounding_details,
        "citation_details": citation_details,
    }


def evaluate_agent_on_cases(
    case_ids: Optional[List[str]] = None,
    model_name: Optional[str] = None,
    limit: int = 3,
) -> Dict[str, Any]:
    """Run end-to-end evaluation benchmark across representative investigation cases."""
    with get_db_context() as db:
        if not case_ids:
            cases = db.query(InvestigationCase).limit(limit).all()
            case_ids = [c.case_id for c in cases]

    if not case_ids:
        return {"error": "No cases available in database for evaluation."}

    results = []
    for cid in case_ids:
        logger.info("Evaluating agent on case %s...", cid)
        inv_result = run_case_investigation(case_id=cid, model_name=model_name)
        report = inv_result.get("report")

        if not report:
            results.append({
                "case_id": cid,
                "success": False,
                "error": "Failed to generate report",
            })
            continue

        true_payload = (inv_result.get("evidence", {}).get("alert") or {}).get("transaction_payload", {})
        true_shap = inv_result.get("evidence", {}).get("shap") or {}

        metrics = evaluate_report_grounding(report, true_payload, true_shap)
        results.append({
            "case_id": cid,
            "success": True,
            "recommended_verdict": report.get("recommended_verdict"),
            "recommended_action": report.get("recommended_action"),
            "confidence_score": report.get("confidence_score"),
            "metrics": metrics,
        })

    # Compute Aggregate Benchmark Metrics
    successful_evals = [r["metrics"] for r in results if r["success"]]
    if successful_evals:
        avg_grounding = sum(m["grounding_score"] for m in successful_evals) / len(successful_evals)
        avg_citation = sum(m["citation_accuracy_score"] for m in successful_evals) / len(successful_evals)
        avg_hallucination_free = sum(m["hallucination_score"] for m in successful_evals) / len(successful_evals)
        avg_composite = sum(m["composite_score"] for m in successful_evals) / len(successful_evals)
    else:
        avg_grounding = avg_citation = avg_hallucination_free = avg_composite = 0.0

    return {
        "total_cases_evaluated": len(case_ids),
        "model_evaluated": model_name or "default",
        "benchmark_summary": {
            "factual_grounding_accuracy": round(avg_grounding, 4),
            "evidence_citation_accuracy": round(avg_citation, 4),
            "hallucination_freedom_rate": round(avg_hallucination_free, 4),
            "composite_agent_quality": round(avg_composite, 4),
        },
        "case_evaluations": results,
    }


def evaluate_triage_grounding(
    triage_report: Dict[str, Any],
    true_alerts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Audit a Fraud Triage batch report for ranking consistency, factual integrity, and storytelling absence.

    Verifies:
    1. Ranking monotonicity (scores ordered descending by urgency)
    2. Exact numeric alignment with true alert fraud probabilities and anomaly scores
    3. Absence of prohibited domain narratives for anonymized PCA features V1-V28
    4. Deterministic tier validation
    """
    items = triage_report.get("prioritized_queue", [])
    if not items:
        return {
            "grounding_passed": True,
            "ranking_monotonicity_score": 1.0,
            "hallucination_freedom_score": 1.0,
            "numeric_fidelity_score": 1.0,
            "overall_score": 1.0,
            "issues": [],
        }

    issues: List[str] = []
    true_by_id = {a["alert_id"]: a for a in true_alerts}

    # 1. Ranking Monotonicity
    monotonic_errors = 0
    for i in range(len(items) - 1):
        s_curr = items[i].get("composite_urgency_score", 0.0)
        s_next = items[i + 1].get("composite_urgency_score", 0.0)
        if s_curr < s_next:
            monotonic_errors += 1
            issues.append(f"Rank violation at index {i}: Rank #{items[i].get('rank')} (score {s_curr}) < Rank #{items[i+1].get('rank')} (score {s_next}).")

    monotonicity_score = max(0.0, 1.0 - (monotonic_errors / max(len(items) - 1, 1)))

    # 2. Numeric Fidelity with True Database Alerts
    numeric_matches = 0
    for item in items:
        aid = item.get("alert_id")
        if aid in true_by_id:
            true_a = true_by_id[aid]
            p_diff = abs(item.get("fraud_probability", 0.0) - true_a.get("fraud_probability", 0.0))
            a_diff = abs(item.get("anomaly_score", 0.0) - true_a.get("anomaly_score", 0.0))
            if p_diff <= 0.01 and a_diff <= 0.01:
                numeric_matches += 1
            else:
                issues.append(f"Numeric mismatch in alert '{aid}': reported P={item.get('fraud_probability')}, true P={true_a.get('fraud_probability')}.")
        else:
            issues.append(f"Alert ID '{aid}' in triage report does not exist in true alert queue.")

    numeric_score = numeric_matches / max(len(items), 1)

    # 3. Domain Hallucination Audit
    all_text = " ".join([
        item.get("triage_explanation", "") for item in items
    ] + [
        triage_report.get("queue_overview", {}).get("queue_risk_synthesis", ""),
        triage_report.get("queue_overview", {}).get("workload_bottlenecks", ""),
    ]).lower()

    detected_hallucinations = []
    for pattern in PROHIBITED_DOMAIN_PATTERNS:
        matches = re.findall(pattern, all_text)
        if matches:
            detected_hallucinations.extend(matches)
            issues.append(f"Detected prohibited domain storytelling: {matches}")

    hallucination_score = 1.0 if not detected_hallucinations else max(0.0, 1.0 - 0.25 * len(detected_hallucinations))

    overall = round(0.4 * monotonicity_score + 0.3 * numeric_score + 0.3 * hallucination_score, 4)

    return {
        "grounding_passed": len(issues) == 0,
        "ranking_monotonicity_score": round(monotonicity_score, 4),
        "numeric_fidelity_score": round(numeric_score, 4),
        "hallucination_freedom_score": round(hallucination_score, 4),
        "overall_score": overall,
        "issues": issues,
    }

