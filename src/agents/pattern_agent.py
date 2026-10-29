"""Stateful Fraud Pattern Analysis Agent Workflow using LangGraph and Groq LLMs.

Discovers recurring statistical patterns, performs cohort and temporal trend analysis,
detects emerging distribution shifts and anomalies across historical transaction and alert data,
and generates evidence-backed findings strictly adhering to anonymized PCA dataset constraints.
"""

from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Literal, Optional, Tuple
import uuid

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, START, StateGraph

from src.agents.config import (
    DEFAULT_AGENT_MODEL,
    DISTRIBUTED_PIPELINE_OPTION,
    FALLBACK_AGENT_MODEL,
    PATTERN_COHORT_MODEL,
    PATTERN_EXECUTIVE_MODEL,
    PATTERN_FINDINGS_MODEL,
    get_groq_llm,
)
from src.agents.schemas import (
    CohortTrendAnalysisOutput,
    DetectedPatternShift,
    PatternAnalysisReport,
    PatternCategory,
    PatternFindingItem,
    PatternFindingsBatchOutput,
    ShiftSeverity,
    StatisticalConfidence,
    TimeWindowCohort,
)
from src.agents.state import PatternAnalysisState
from src.agents.tools import get_historical_pattern_data

logger = logging.getLogger(__name__)

# Extended anti-hallucination patterns for European Credit Card Fraud PCA dataset
PROHIBITED_DOMAIN_PATTERNS = [
    r"\b(foreign|international|overseas)\s+ip\b",
    r"\bgas\s+station\b",
    r"\bgrocery\s+store\b",
    r"\batm\s+withdrawal\b",
    r"\bjewelry\s+store\b",
    r"\belectronics\s+shop\b",
    r"\bcardholder\s+(is\s+traveling|went\s+to|lives\s+in|behavior|profile|demographic)\b",
    r"\bmerchant\s+(category\s+code|name|identity|location|cluster)\b",
    r"\bv\d+\s+represents\s+(a\s+)?(location|device|country|merchant|ip)\b",
    r"\bfraud\s+ring(s)?\b",
    r"\borganized\s+crime\s+syndicate(s)?\b",
    r"\bparticular\s+merchant\b",
    r"\bgeographic(al)?\s+(location|region|country|city)\b",
]


# =============================================================================
# DETERMINISTIC STATISTICAL COMPUTATION ENGINE
# =============================================================================

def compute_cohort_statistics(
    records: List[Dict[str, Any]],
    window_name: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> Dict[str, Any]:
    """Calculate deterministic summary statistics, distributions, and error metrics for a cohort.

    Ensures all numbers, medians, percentiles, and rates are strictly reproducible and grounded.
    """
    if not records:
        return {
            "window_name": window_name,
            "start_time": start_time,
            "end_time": end_time,
            "alert_count": 0,
            "mean_amount": 0.0,
            "median_amount": 0.0,
            "p90_amount": 0.0,
            "std_amount": 0.0,
            "mean_fraud_probability": 0.0,
            "high_fraud_share_pct": 0.0,
            "mean_anomaly_score": 0.0,
            "high_anomaly_share_pct": 0.0,
            "tier_distribution": {},
            "quadrant_distribution": {},
            "verified_outcomes_count": 0,
            "confirmed_fraud_count": 0,
            "false_positive_count": 0,
            "false_positive_rate": None,
            "precision": None,
            "pca_means": {},
        }

    count = len(records)
    amounts = [float(r.get("amount", 0.0)) for r in records]
    p_frauds = [float(r.get("fraud_probability", 0.0)) for r in records]
    anomalies = [float(r.get("anomaly_score", 0.0)) for r in records]

    # Monetary metrics
    mean_amt = round(float(np.mean(amounts)), 2)
    median_amt = round(float(np.median(amounts)), 2)
    p90_amt = round(float(np.percentile(amounts, 90)), 2)
    std_amt = round(float(np.std(amounts)), 2)

    # Supervised fraud probability metrics
    mean_p = round(float(np.mean(p_frauds)), 4)
    high_fraud_count = sum(1 for p in p_frauds if p >= 0.70)
    high_fraud_share_pct = round((high_fraud_count / count) * 100.0, 2)

    # Unsupervised anomaly metrics
    mean_anom = round(float(np.mean(anomalies)), 4)
    high_anom_count = sum(1 for a in anomalies if a >= 0.70)
    high_anomaly_share_pct = round((high_anom_count / count) * 100.0, 2)


    # Distributions
    tier_dist: Dict[str, int] = {}
    quad_dist: Dict[str, int] = {}
    for r in records:
        tier = str(r.get("priority", "MEDIUM")).upper()
        tier_dist[tier] = tier_dist.get(tier, 0) + 1

        quad = str(r.get("operational_quadrant", "UNKNOWN")).upper()
        quad_dist[quad] = quad_dist.get(quad, 0) + 1

    # Verified case outcomes & model error metrics
    verified_records = [r for r in records if r.get("verdict") in ("CONFIRMED_FRAUD", "FALSE_POSITIVE")]
    verified_count = len(verified_records)
    confirmed_count = sum(1 for r in verified_records if r.get("verdict") == "CONFIRMED_FRAUD")
    fp_count = sum(1 for r in verified_records if r.get("verdict") == "FALSE_POSITIVE")

    fpr = round(fp_count / verified_count, 4) if verified_count > 0 else None
    precision = round(confirmed_count / verified_count, 4) if verified_count > 0 else None

    # PCA feature distributions for top informative features
    pca_means: Dict[str, float] = {}
    key_features = ["V14", "V10", "V4", "V12", "V17"]
    for feat in key_features:
        vals = []
        for r in records:
            payload = r.get("transaction_payload") or {}
            if feat in payload:
                try:
                    vals.append(float(payload[feat]))
                except (ValueError, TypeError):
                    pass
        if vals:
            pca_means[feat] = round(float(np.mean(vals)), 4)

    return {
        "window_name": window_name,
        "start_time": start_time,
        "end_time": end_time,
        "alert_count": count,
        "mean_amount": mean_amt,
        "median_amount": median_amt,
        "p90_amount": p90_amt,
        "std_amount": std_amt,
        "mean_fraud_probability": mean_p,
        "high_fraud_share_pct": high_fraud_share_pct,
        "mean_anomaly_score": mean_anom,
        "high_anomaly_share_pct": high_anomaly_share_pct,
        "tier_distribution": tier_dist,
        "quadrant_distribution": quad_dist,
        "verified_outcomes_count": verified_count,
        "confirmed_fraud_count": confirmed_count,
        "false_positive_count": fp_count,
        "false_positive_rate": fpr,
        "precision": precision,
        "pca_means": pca_means,
    }


def detect_statistical_shifts(
    baseline: Dict[str, Any],
    recent: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Deterministically detect significant metric shifts between baseline and recent windows."""
    shifts: List[Dict[str, Any]] = []

    # 1. High-Anomaly Concentration Shift
    b_anom = float(baseline.get("high_anomaly_share_pct", 0.0))
    r_anom = float(recent.get("high_anomaly_share_pct", 0.0))
    diff_anom = round(r_anom - b_anom, 2)
    rel_anom = round(((r_anom - b_anom) / max(b_anom, 0.01)) * 100.0, 2) if b_anom > 0 else (100.0 if r_anom > 0 else 0.0)

    sev_anom = ShiftSeverity.INFO.value
    sig_anom = False
    if abs(diff_anom) >= 15.0 or abs(rel_anom) >= 40.0:
        sev_anom = ShiftSeverity.CRITICAL.value if rel_anom > 0 else ShiftSeverity.HIGH.value
        sig_anom = True
    elif abs(diff_anom) >= 8.0 or abs(rel_anom) >= 20.0:
        sev_anom = ShiftSeverity.MEDIUM.value
        sig_anom = True

    shifts.append({
        "metric_name": "High-Anomaly Alert Share (Anomaly Score >= 0.70)",
        "baseline_value": b_anom,
        "recent_value": r_anom,
        "absolute_change": diff_anom,
        "relative_change_pct": rel_anom,
        "severity": sev_anom,
        "is_significant": sig_anom,
        "shift_summary": (
            f"High-anomaly transactions shifted from {b_anom:.1f}% to {r_anom:.1f}% "
            f"({'+' if diff_anom >= 0 else ''}{diff_anom:.1f} pts, {'+' if rel_anom >= 0 else ''}{rel_anom:.1f}% relative)."
        ),
    })

    # 2. Supervised Fraud Probability Shift
    b_p = float(baseline.get("high_fraud_share_pct", 0.0))
    r_p = float(recent.get("high_fraud_share_pct", 0.0))
    diff_p = round(r_p - b_p, 2)
    rel_p = round(((r_p - b_p) / max(b_p, 0.01)) * 100.0, 2) if b_p > 0 else (100.0 if r_p > 0 else 0.0)

    sev_p = ShiftSeverity.INFO.value
    sig_p = False
    if abs(diff_p) >= 15.0 or abs(rel_p) >= 35.0:
        sev_p = ShiftSeverity.HIGH.value
        sig_p = True
    elif abs(diff_p) >= 7.0 or abs(rel_p) >= 15.0:
        sev_p = ShiftSeverity.MEDIUM.value
        sig_p = True

    shifts.append({
        "metric_name": "High Supervised Fraud Share (P(Fraud) >= 0.70)",
        "baseline_value": b_p,
        "recent_value": r_p,
        "absolute_change": diff_p,
        "relative_change_pct": rel_p,
        "severity": sev_p,
        "is_significant": sig_p,
        "shift_summary": (
            f"High-probability alerts shifted from {b_p:.1f}% to {r_p:.1f}% "
            f"({'+' if diff_p >= 0 else ''}{diff_p:.1f} pts, {'+' if rel_p >= 0 else ''}{rel_p:.1f}% relative)."
        ),
    })

    # 3. Monetary Exposure Drift (Mean Amount)
    b_amt = float(baseline.get("mean_amount", 0.0))
    r_amt = float(recent.get("mean_amount", 0.0))
    diff_amt = round(r_amt - b_amt, 2)
    rel_amt = round(((r_amt - b_amt) / max(b_amt, 0.01)) * 100.0, 2) if b_amt > 0 else 0.0

    sev_amt = ShiftSeverity.INFO.value
    sig_amt = False
    if abs(rel_amt) >= 35.0:
        sev_amt = ShiftSeverity.HIGH.value
        sig_amt = True
    elif abs(rel_amt) >= 15.0:
        sev_amt = ShiftSeverity.MEDIUM.value
        sig_amt = True

    shifts.append({
        "metric_name": "Mean Transaction Monetary Exposure (€)",
        "baseline_value": b_amt,
        "recent_value": r_amt,
        "absolute_change": diff_amt,
        "relative_change_pct": rel_amt,
        "severity": sev_amt,
        "is_significant": sig_amt,
        "shift_summary": (
            f"Mean transaction amount changed from €{b_amt:.2f} to €{r_amt:.2f} "
            f"({'+' if diff_amt >= 0 else ''}€{diff_amt:.2f}, {'+' if rel_amt >= 0 else ''}{rel_amt:.1f}% relative)."
        ),
    })

    # 4. Verified False Positive Rate Shift (if outcome data present)
    b_fpr = baseline.get("false_positive_rate")
    r_fpr = recent.get("false_positive_rate")
    if b_fpr is not None and r_fpr is not None:
        diff_fpr = round(r_fpr - b_fpr, 4)
        rel_fpr = round(((r_fpr - b_fpr) / max(b_fpr, 0.001)) * 100.0, 2) if b_fpr > 0 else 0.0

        sev_fpr = ShiftSeverity.INFO.value
        sig_fpr = False
        if abs(diff_fpr) >= 0.15 or abs(rel_fpr) >= 30.0:
            sev_fpr = ShiftSeverity.CRITICAL.value if diff_fpr > 0 else ShiftSeverity.HIGH.value
            sig_fpr = True
        elif abs(diff_fpr) >= 0.08 or abs(rel_fpr) >= 15.0:
            sev_fpr = ShiftSeverity.MEDIUM.value
            sig_fpr = True

        shifts.append({
            "metric_name": "Verified False Positive Rate (FP / Total Verified Cases)",
            "baseline_value": round(float(b_fpr), 4),
            "recent_value": round(float(r_fpr), 4),
            "absolute_change": diff_fpr,
            "relative_change_pct": rel_fpr,
            "severity": sev_fpr,
            "is_significant": sig_fpr,
            "shift_summary": (
                f"Verified FPR changed from {b_fpr:.1%} to {r_fpr:.1%} "
                f"({'+' if diff_fpr >= 0 else ''}{diff_fpr * 100:.1f} percentage pts)."
            ),
        })

    # 5. Alert Influx Volume Velocity
    b_cnt = int(baseline.get("alert_count", 0))
    r_cnt = int(recent.get("alert_count", 0))
    diff_cnt = r_cnt - b_cnt
    rel_cnt = round(((r_cnt - b_cnt) / max(b_cnt, 1)) * 100.0, 2)
    sev_cnt = ShiftSeverity.MEDIUM.value if abs(rel_cnt) >= 30.0 else ShiftSeverity.INFO.value

    shifts.append({
        "metric_name": "Alert Queue Volume Velocity",
        "baseline_value": float(b_cnt),
        "recent_value": float(r_cnt),
        "absolute_change": float(diff_cnt),
        "relative_change_pct": rel_cnt,
        "severity": sev_cnt,
        "is_significant": abs(rel_cnt) >= 30.0,
        "shift_summary": (
            f"Alert volume changed from {b_cnt} alerts to {r_cnt} alerts ({'+' if diff_cnt >= 0 else ''}{diff_cnt} alerts, "
            f"{'+' if rel_cnt >= 0 else ''}{rel_cnt:.1f}% relative)."
        ),
    })

    # 6. PCA Component Drift (V14 and V10)
    b_pca = baseline.get("pca_means", {})
    r_pca = recent.get("pca_means", {})
    for feat in ["V14", "V10", "V4"]:
        if feat in b_pca and feat in r_pca:
            bv = float(b_pca[feat])
            rv = float(r_pca[feat])
            d_val = round(rv - bv, 4)
            # Statistical drift on standard PCA component
            if abs(d_val) >= 0.5:
                shifts.append({
                    "metric_name": f"PCA Component Drift ({feat})",
                    "baseline_value": bv,
                    "recent_value": rv,
                    "absolute_change": d_val,
                    "relative_change_pct": round(((rv - bv) / max(abs(bv), 0.01)) * 100.0, 2),
                    "severity": ShiftSeverity.HIGH.value if abs(d_val) >= 1.0 else ShiftSeverity.MEDIUM.value,
                    "is_significant": True,
                    "shift_summary": (
                        f"Mean {feat} component shifted from {bv:.3f} to {rv:.3f} "
                        f"(shift of {'+' if d_val >= 0 else ''}{d_val:.3f} in PCA latent space)."
                    ),
                })

    return shifts


# =============================================================================
# GRAPH NODE FUNCTIONS
# =============================================================================

def node_gather_historical_data(state: PatternAnalysisState) -> Dict[str, Any]:
    """Retrieve historical alerts and verified case outcomes, splitting into cohorts."""
    time_horizon = state.get("time_horizon_hours")
    baseline_ratio = state.get("baseline_ratio", 0.5)
    min_alerts = state.get("min_alerts_threshold", 10)

    logger.info("Gathering historical pattern data: lookback=%s hrs, baseline_ratio=%.2f", time_horizon, baseline_ratio)

    # 1. Fetch from deterministic retrieval tool
    data = get_historical_pattern_data.invoke({
        "limit": 500,
        "time_horizon_hours": time_horizon,
    })
    records: List[Dict[str, Any]] = list(data.get("records", []))

    # 2. Resilient fallback: If database has fewer than min_alerts, seed or pull holdout data
    if len(records) < min_alerts:
        logger.warning("Database has only %d records (< %d threshold). Loading holdout dataset.", len(records), min_alerts)
        try:
            from dashboard.utils import load_holdout_samples
            holdout_df = load_holdout_samples(n_samples=min_alerts * 2)
            if not holdout_df.empty:
                for idx, row in holdout_df.iterrows():
                    amt = float(row.get("Amount", 50.0))
                    t_off = float(row.get("Time", idx * 100.0))
                    # Deterministic mock score for synthetic test fallback
                    is_fraud_val = int(row.get("Class", 0))
                    p_f = 0.88 if is_fraud_val == 1 else 0.04
                    anom = 0.78 if is_fraud_val == 1 else 0.22
                    records.append({
                        "alert_id": f"ALT-HOLD-{idx:04d}",
                        "transaction_id": f"TX-HOLD-{idx:04d}",
                        "created_at": f"2026-10-25T{idx % 24:02d}:00:00Z",
                        "amount": amt,
                        "time_offset": t_off,
                        "fraud_probability": p_f,
                        "anomaly_score": anom,
                        "risk_category": "CRITICAL" if p_f >= 0.70 else "LOW",
                        "priority": "CRITICAL" if p_f >= 0.70 else "LOW",
                        "operational_quadrant": "CONFIRMED_HIGH_RISK" if is_fraud_val == 1 else "CONFIRMED_LEGITIMATE",
                        "status": "RESOLVED",
                        "verdict": "CONFIRMED_FRAUD" if is_fraud_val == 1 else "FALSE_POSITIVE",
                        "case_id": f"CASE-HOLD-{idx:04d}",
                        "transaction_payload": row.to_dict(),
                    })
        except Exception as exc:
            logger.error("Could not load holdout fallback: %s", exc)

    # Sort chronologically by created_at or time_offset
    records.sort(key=lambda r: (str(r.get("created_at") or ""), float(r.get("time_offset") or 0.0)))

    total = len(records)
    split_idx = max(1, min(total - 1, int(total * baseline_ratio))) if total > 1 else total

    baseline_records = records[:split_idx]
    recent_records = records[split_idx:] if total > 1 else records

    time_meta = {
        "total_historical_records": total,
        "baseline_count": len(baseline_records),
        "recent_count": len(recent_records),
        "baseline_start": baseline_records[0].get("created_at") if baseline_records else None,
        "baseline_end": baseline_records[-1].get("created_at") if baseline_records else None,
        "recent_start": recent_records[0].get("created_at") if recent_records else None,
        "recent_end": recent_records[-1].get("created_at") if recent_records else None,
    }

    logger.info("Historical data partitioned: %d baseline, %d recent records.", len(baseline_records), len(recent_records))
    return {
        "raw_historical_records": records,
        "baseline_records": baseline_records,
        "recent_records": recent_records,
        "time_windows_metadata": time_meta,
    }


def node_compute_pattern_statistics(state: PatternAnalysisState) -> Dict[str, Any]:
    """Calculate deterministic cohort statistics, percentiles, error rates, and distributions."""
    baseline_records = state.get("baseline_records", [])
    recent_records = state.get("recent_records", [])
    meta = state.get("time_windows_metadata", {})

    b_stats = compute_cohort_statistics(
        records=baseline_records,
        window_name="Baseline Window",
        start_time=meta.get("baseline_start"),
        end_time=meta.get("baseline_end"),
    )

    r_stats = compute_cohort_statistics(
        records=recent_records,
        window_name="Recent Window",
        start_time=meta.get("recent_start"),
        end_time=meta.get("recent_end"),
    )

    computed = {
        "baseline_cohort": b_stats,
        "recent_cohort": r_stats,
        "metadata": meta,
    }

    logger.info(
        "Computed pattern stats: Baseline (N=%d, high_anom=%.1f%%, mean_amt=€%.2f), Recent (N=%d, high_anom=%.1f%%, mean_amt=€%.2f)",
        b_stats["alert_count"], b_stats["high_anomaly_share_pct"], b_stats["mean_amount"],
        r_stats["alert_count"], r_stats["high_anomaly_share_pct"], r_stats["mean_amount"],
    )

    return {
        "computed_statistics": computed,
    }


def node_detect_anomalous_trends(state: PatternAnalysisState) -> Dict[str, Any]:
    """Detect statistical distribution shifts, volume anomalies, and performance changes."""
    stats = state.get("computed_statistics", {})
    b_stats = stats.get("baseline_cohort", {})
    r_stats = stats.get("recent_cohort", {})

    shifts = detect_statistical_shifts(baseline=b_stats, recent=r_stats)
    logger.info("Detected %d pattern shifts across cohorts.", len(shifts))

    return {
        "detected_shifts": shifts,
    }


# =============================================================================
# DISTRIBUTED MULTI-LLM PATTERN SYNTHESIS SUB-TASKS
# =============================================================================

def synthesize_cohort_and_error_trends(
    b_cohort: TimeWindowCohort,
    r_cohort: TimeWindowCohort,
    model_name: str = PATTERN_COHORT_MODEL,
) -> Tuple[str, str]:
    """Sub-task 1: Synthesize cohort dynamics and verified model error trends using specialized LLM (qwen/qwen3.8-27b).

    Focuses strictly on numeric cohort comparison and outcome validation shifts.
    Token footprint: ~250 input tokens (vs ~5,500 monolithic).
    """
    cohort_summary = (
        f"BASELINE COHORT: alerts={b_cohort.alert_count}, mean_amt=€{b_cohort.mean_amount:.2f}, "
        f"median_amt=€{b_cohort.median_amount:.2f}, p90_amt=€{b_cohort.p90_amount:.2f}, "
        f"high_anomaly_share={b_cohort.high_anomaly_share_pct:.1f}%, high_fraud_share={b_cohort.high_fraud_share_pct:.1f}%, "
        f"verified_cases={b_cohort.verified_outcomes_count}, confirmed_fraud={b_cohort.confirmed_fraud_count}, "
        f"false_positives={b_cohort.false_positive_count}, fpr={f'{b_cohort.false_positive_rate:.1%}' if b_cohort.false_positive_rate is not None else 'N/A'}.\n"
        f"RECENT COHORT: alerts={r_cohort.alert_count}, mean_amt=€{r_cohort.mean_amount:.2f}, "
        f"median_amt=€{r_cohort.median_amount:.2f}, p90_amt=€{r_cohort.p90_amount:.2f}, "
        f"high_anomaly_share={r_cohort.high_anomaly_share_pct:.1f}%, high_fraud_share={r_cohort.high_fraud_share_pct:.1f}%, "
        f"verified_cases={r_cohort.verified_outcomes_count}, confirmed_fraud={r_cohort.confirmed_fraud_count}, "
        f"false_positives={r_cohort.false_positive_count}, fpr={f'{r_cohort.false_positive_rate:.1%}' if r_cohort.false_positive_rate is not None else 'N/A'}."
    )

    llm = get_groq_llm(model_name=model_name, temperature=0.1)
    parser = PydanticOutputParser(pydantic_object=CohortTrendAnalysisOutput)

    prompt_tmpl = PromptTemplate(
        template="""You are a Quantitative Fraud Pattern Analyst.
Compare the baseline cohort vs. recent cohort metrics below and generate:
1. 'cohort_trend_insights': 1-2 concise sentences comparing alert volume, monetary exposure (mean, median, p90), and high-anomaly concentrations. Cite exact figures.
2. 'model_error_analysis': 1-2 concise sentences analyzing verified outcomes (confirmed fraud vs false positives, false positive rate, precision). If no verified outcomes exist, state that outcome verification is in progress.

--- COHORT METRICS ---
{cohort_summary}

--- STRICT CONSTRAINTS ---
- Cite exact numbers from the metrics above; never hallucinate or invent figures.
- European Credit Card dataset features V1-V28 are anonymized mathematical PCA components. Never mention specific merchants, locations, or cardholder travel.

{format_instructions}
""",
        input_variables=["cohort_summary"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    try:
        chain = prompt_tmpl | llm | parser
        res: CohortTrendAnalysisOutput = chain.invoke({"cohort_summary": cohort_summary})
        return res.cohort_trend_insights, res.model_error_analysis
    except Exception as exc:
        logger.warning("Cohort trend LLM synthesis failed on model '%s' (%s). Using deterministic fallback.", model_name, exc)
        insights = (
            f"Baseline cohort evaluated {b_cohort.alert_count} alerts (mean amount €{b_cohort.mean_amount:.2f}, "
            f"{b_cohort.high_anomaly_share_pct:.1f}% high-anomaly), compared to {r_cohort.alert_count} alerts "
            f"(mean amount €{r_cohort.mean_amount:.2f}, {r_cohort.high_anomaly_share_pct:.1f}% high-anomaly) in the recent cohort."
        )
        err_analysis = (
            f"Verified false positive rate is {r_cohort.false_positive_rate:.1%}" if r_cohort.false_positive_rate is not None
            else "Outcome verification in progress for recent cohort."
        )
        return insights, err_analysis


def formulate_pattern_findings(
    shifts: List[Dict[str, Any]],
    b_stats: Dict[str, Any],
    r_stats: Dict[str, Any],
    model_name: str = PATTERN_FINDINGS_MODEL,
) -> List[PatternFindingItem]:
    """Sub-task 2: Formulate itemized evidence-backed findings using specialized LLM (openai/gpt-oss-20b).

    Strictly applies anonymized PCA constraints and cites exact shift metrics.
    Token footprint: ~400 input tokens (vs ~5,500 monolithic).
    """
    shifts_summary = "\n".join(
        f"- {s['metric_name']}: Baseline={s['baseline_value']}, Recent={s['recent_value']} "
        f"({'+' if s['absolute_change'] >= 0 else ''}{s['absolute_change']}), "
        f"Rel Shift={s['relative_change_pct']}%, Severity={s['severity']}, Significant={s['is_significant']} "
        f"--> Summary: {s['shift_summary']}"
        for s in shifts
    )

    if not shifts_summary.strip():
        shifts_summary = "No significant metric shifts detected between baseline and recent cohorts."

    llm = get_groq_llm(model_name=model_name, temperature=0.1)
    parser = PydanticOutputParser(pydantic_object=PatternFindingsBatchOutput)

    prompt_tmpl = PromptTemplate(
        template="""You are a Fraud Risk Modeler specializing in statistical pattern discovery.
Interpret the detected statistical shifts between baseline and recent cohorts and formulate 1 to 4 evidence-backed findings.

--- DETECTED STATISTICAL SHIFTS ---
{shifts_summary}

--- STRICT CONSTRAINTS (ZERO TOLERANCE) ---
1. GROUNDING: Cite exact baseline, recent, and relative change percentages from the shifts above.
2. ANONYMIZED PCA RESTRICTION (CRITICAL): The European Credit Card Fraud dataset contains anonymized PCA features (V1–V28).
   - You MUST NOT claim to discover particular merchants, store types (e.g. gas stations, jewelry stores, electronics),
     geographic locations, foreign IPs, customer travel, or fraud rings from those features alone.
   - PCA components MUST be described strictly as mathematical PCA components reflecting latent feature correlations.
3. STRUCTURE:
   - 'pattern_title': Clear title (e.g., 'Surge in Isolation Forest Unsupervised Anomalies')
   - 'category': One of ANOMALY_CONCENTRATION, VOLUME_VELOCITY, MODEL_PERFORMANCE_SHIFT, EXPOSURE_DRIFT, STATISTICAL_DISTRIBUTION_SHIFT, RISK_TIER_MIGRATION
   - 'measured_pattern': 1-2 sentences with exact numbers and comparison periods
   - 'supporting_statistics': Dictionary containing baseline, recent, and relative change values
   - 'statistical_confidence': 'HIGH', 'MEDIUM', or 'LOW'
   - 'dataset_limitations': Explicit affirmation of anonymized PCA constraints
   - 'recommended_followup': Concrete operational recommendation

{format_instructions}
""",
        input_variables=["shifts_summary"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    try:
        chain = prompt_tmpl | llm | parser
        res: PatternFindingsBatchOutput = chain.invoke({"shifts_summary": shifts_summary})
        if res.findings:
            return res.findings
    except Exception as exc:
        logger.warning("Pattern findings LLM formulation failed on model '%s' (%s). Using deterministic fallback.", model_name, exc)

    # Resilient fallback formulation
    fallback_findings: List[PatternFindingItem] = []
    for s in shifts:
        if s.get("is_significant"):
            cat = PatternCategory.ANOMALY_CONCENTRATION.value
            if "Amount" in s["metric_name"]:
                cat = PatternCategory.EXPOSURE_DRIFT.value
            elif "Volume" in s["metric_name"]:
                cat = PatternCategory.VOLUME_VELOCITY.value
            elif "False Positive" in s["metric_name"]:
                cat = PatternCategory.MODEL_PERFORMANCE_SHIFT.value
            elif "PCA" in s["metric_name"]:
                cat = PatternCategory.STATISTICAL_DISTRIBUTION_SHIFT.value

            fallback_findings.append(PatternFindingItem(
                pattern_title=f"Emerging Shift: {s['metric_name']}",
                category=cat,
                measured_pattern=s["shift_summary"],
                supporting_statistics={
                    "baseline_value": s["baseline_value"],
                    "recent_value": s["recent_value"],
                    "relative_change_pct": s["relative_change_pct"],
                },
                statistical_confidence="HIGH" if abs(s["relative_change_pct"]) >= 30.0 else "MEDIUM",
                dataset_limitations=(
                    "European Credit Card dataset contains anonymized PCA features V1-V28. "
                    "This finding reflects purely mathematical distribution shifts without real merchant or cardholder identities."
                ),
                recommended_followup=(
                    f"Review risk threshold and queue triage routing for {s['metric_name']} "
                    f"to prevent operational backlog or detection degradation."
                ),
            ))

    if not fallback_findings:
        fallback_findings.append(PatternFindingItem(
            pattern_title="Stable Baseline Operational Risk Profile",
            category=PatternCategory.STATISTICAL_DISTRIBUTION_SHIFT.value,
            measured_pattern=(
                f"Evaluated {b_stats.get('alert_count', 0)} baseline and {r_stats.get('alert_count', 0)} recent alerts. "
                "Metric shifts remain within standard operational bounds."
            ),
            supporting_statistics={
                "baseline_alerts": b_stats.get("alert_count", 0),
                "recent_alerts": r_stats.get("alert_count", 0),
            },
            statistical_confidence="HIGH",
            dataset_limitations="Features V1-V28 are anonymized mathematical PCA components.",
            recommended_followup="Continue automated surveillance and monitor subsequent window cohorts.",
        ))

    return fallback_findings


def synthesize_executive_summary(
    cohort_insights: str,
    error_analysis: str,
    findings: List[PatternFindingItem],
    b_cohort: TimeWindowCohort,
    r_cohort: TimeWindowCohort,
    model_name: str = PATTERN_EXECUTIVE_MODEL,
) -> str:
    """Sub-task 3: High-level leadership synthesis using executive LLM (openai/gpt-oss-120b).

    Distills macro dynamics into a clean 2-3 sentence executive brief.
    Token footprint: ~200 input tokens (vs ~5,500 monolithic).
    """
    findings_bullets = "\n".join(
        f"- {f.pattern_title} ({f.category}): {f.measured_pattern}"
        for f in findings[:4]
    )

    llm = get_groq_llm(model_name=model_name, temperature=0.1)
    prompt_tmpl = PromptTemplate(
        template="""You are the Chief Risk Officer AI for a tier-1 financial institution.
Synthesize the following quantitative cohort insights and pattern findings into a concise 2-3 sentence executive summary for senior risk leadership.

COHORT TRENDS:
{cohort_insights}

VERIFIED ERROR DYNAMICS:
{error_analysis}

KEY FINDINGS:
{findings_bullets}

CRITICAL: Keep strictly grounded in measured data. Features V1-V28 are anonymized PCA components; never mention specific merchants, locations, or fraud rings.
Respond with ONLY the 2-3 sentence executive summary.
""",
        input_variables=["cohort_insights", "error_analysis", "findings_bullets"],
    )

    try:
        chain = prompt_tmpl | llm
        response = chain.invoke({
            "cohort_insights": cohort_insights,
            "error_analysis": error_analysis,
            "findings_bullets": findings_bullets,
        })
        text = response.content if hasattr(response, "content") else str(response)
        summary = text.strip()
        # Ensure clean text
        if len(summary) >= 25:
            return summary
    except Exception as exc:
        logger.warning("Executive summary LLM synthesis failed on model '%s' (%s). Using deterministic fallback.", model_name, exc)

    return (
        f"Historical analysis reveals a {r_cohort.high_anomaly_share_pct:.1f}% concentration of high-anomaly alerts "
        f"in the recent cohort (vs. {b_cohort.high_anomaly_share_pct:.1f}% baseline), with mean monetary exposure at €{r_cohort.mean_amount:.2f}. "
        f"Active findings highlight emerging distribution shifts across unsupervised anomaly scores and latent PCA vectors."
    )


def node_synthesize_findings(state: PatternAnalysisState) -> Dict[str, Any]:
    """Synthesize evidence-backed findings and structured report using distributed multi-LLM orchestration."""
    stats = state.get("computed_statistics", {})
    b_stats = stats.get("baseline_cohort", {})
    r_stats = stats.get("recent_cohort", {})
    shifts = state.get("detected_shifts", [])
    meta = state.get("time_windows_metadata", {})

    # Convert to TimeWindowCohort models
    b_cohort = TimeWindowCohort(**{k: v for k, v in b_stats.items() if k in TimeWindowCohort.model_fields})
    r_cohort = TimeWindowCohort(**{k: v for k, v in r_stats.items() if k in TimeWindowCohort.model_fields})
    shift_models = [DetectedPatternShift(**s) for s in shifts]

    # Model resolution: Distributed pipeline or explicit user single-model override
    user_model = state.get("model_name")
    if user_model and user_model != DISTRIBUTED_PIPELINE_OPTION and user_model in [
        "openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"
    ]:
        cohort_model = user_model
        findings_model = user_model
        executive_model = user_model
        pipeline_mode = f"Single Model ({user_model})"
    else:
        cohort_model = PATTERN_COHORT_MODEL
        findings_model = PATTERN_FINDINGS_MODEL
        executive_model = PATTERN_EXECUTIVE_MODEL
        pipeline_mode = "Distributed Multi-LLM"

    models_used = {
        "pipeline_mode": pipeline_mode,
        "cohort_model": cohort_model,
        "findings_model": findings_model,
        "executive_model": executive_model,
    }

    logger.info("Executing pattern synthesis in mode '%s' across models: %s", pipeline_mode, models_used)

    # Sub-task 1: Cohort Trends and Verified Error Rate Analysis (qwen/qwen3.8-27b)
    cohort_insights, error_analysis = synthesize_cohort_and_error_trends(
        b_cohort=b_cohort,
        r_cohort=r_cohort,
        model_name=cohort_model,
    )

    # Sub-task 2: Structured Evidence-Backed Findings Formulation (openai/gpt-oss-20b)
    findings = formulate_pattern_findings(
        shifts=shifts,
        b_stats=b_stats,
        r_stats=r_stats,
        model_name=findings_model,
    )

    # Sub-task 3: Executive Leadership Brief Synthesis (openai/gpt-oss-120b)
    exec_summary = synthesize_executive_summary(
        cohort_insights=cohort_insights,
        error_analysis=error_analysis,
        findings=findings,
        b_cohort=b_cohort,
        r_cohort=r_cohort,
        model_name=executive_model,
    )

    report_id = f"PAT-{uuid.uuid4().hex[:8].upper()}"
    report = PatternAnalysisReport(
        report_id=report_id,
        executive_summary=exec_summary,
        baseline_window=b_cohort,
        recent_window=r_cohort,
        detected_shifts=shift_models,
        findings=findings,
        model_error_analysis=error_analysis,
        dataset_limitations_acknowledged=True,
        grounding_audit_passed=True,
        models_used=models_used,
    )

    return {
        "structured_report": report.model_dump(),
        "validation_errors": [],
    }


def node_validate_evidence(state: PatternAnalysisState) -> Dict[str, Any]:
    """Audit the generated report for factual grounding, prohibited domain storytelling, and dataset limits."""
    report = state.get("structured_report") or {}
    errors: List[str] = []

    # 1. Audit Prohibited Domain Storytelling & Hallucinations on Anonymized PCA Features
    all_text_blobs = [
        report.get("executive_summary", ""),
        report.get("model_error_analysis", "") or "",
    ]
    for finding in report.get("findings", []):
        all_text_blobs.append(finding.get("pattern_title", ""))
        all_text_blobs.append(finding.get("measured_pattern", ""))
        all_text_blobs.append(finding.get("dataset_limitations", ""))
        all_text_blobs.append(finding.get("recommended_followup", ""))

    full_text = " ".join(all_text_blobs).lower()

    for pattern in PROHIBITED_DOMAIN_PATTERNS:
        match = re.search(pattern, full_text)
        if match:
            errors.append(
                f"Prohibited domain narrative detected: '{match.group(0)}'. "
                "Dataset contains anonymized PCA features V1-V28. "
                "The agent must not claim to discover particular merchants, locations, customer behaviors, or fraud rings."
            )

    # 2. Check dataset limitations acknowledgement
    if not report.get("dataset_limitations_acknowledged", False):
        errors.append("Report failed to acknowledge dataset limitations regarding anonymized PCA features.")

    # 3. Check that at least one finding exists
    if not report.get("findings"):
        errors.append("Report contains 0 findings.")

    attempts = state.get("validation_attempts", 0) + 1
    passed = len(errors) == 0

    if passed:
        logger.info("Pattern analysis grounding audit passed with 0 unsupported claims.")
    else:
        logger.warning("Pattern analysis grounding audit found %d issues (attempt %d).", len(errors), attempts)

    return {
        "validation_passed": passed,
        "validation_attempts": attempts,
        "validation_errors": errors,
    }


def node_revise_findings(state: PatternAnalysisState) -> Dict[str, Any]:
    """Prompt LLM to revise findings to eliminate detected grounding or storytelling errors."""
    logger.info("Revising pattern findings due to grounding feedback: %s", state.get("validation_errors"))
    user_model = state.get("model_name")
    if user_model and user_model != DISTRIBUTED_PIPELINE_OPTION and user_model in [
        "openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"
    ]:
        model_name = user_model
    else:
        model_name = PATTERN_FINDINGS_MODEL

    llm = get_groq_llm(model_name=model_name, temperature=0.0)
    parser = PydanticOutputParser(pydantic_object=PatternAnalysisReport)

    errors_str = "\n".join(f"- {e}" for e in state.get("validation_errors", []))

    prompt_tmpl = PromptTemplate(
        template="""Your previous pattern analysis report had factual grounding or prohibited domain storytelling violations.
Please revise the report to completely eliminate all listed errors. Remember that V1-V28 are anonymized PCA components;
never mention specific merchants, locations, customer travel, or fraud rings.

ERRORS DETECTED:
{errors}

CURRENT REPORT CONTENT:
{current_report}

{format_instructions}
""",
        input_variables=["errors", "current_report"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt_tmpl | llm | parser
    try:
        revised: PatternAnalysisReport = chain.invoke({
            "errors": errors_str,
            "current_report": json.dumps(state.get("structured_report"), indent=2),
        })
        rev_dict = revised.model_dump()
        rev_dict["dataset_limitations_acknowledged"] = True
        return {
            "structured_report": rev_dict,
            "validation_errors": [],
        }
    except Exception as exc:
        return {"validation_errors": [f"Revision parsing error: {exc}"]}


def node_publish_analysis_report(state: PatternAnalysisState) -> Dict[str, Any]:
    """Publish and finalize the validated structured pattern analysis report."""
    report = state.get("structured_report") or {}

    report_id = report.get("report_id") or f"PAT-{uuid.uuid4().hex[:8].upper()}"
    report["report_id"] = report_id
    report["grounding_audit_passed"] = state.get("validation_passed", True)
    report["published_at"] = datetime.now(timezone.utc).isoformat()

    logger.info("Published final pattern analysis report: %s", report_id)
    return {
        "published_report": report,
    }


def should_continue_validation(state: PatternAnalysisState) -> Literal["revise_findings", "publish_analysis_report"]:
    """Conditional edge deciding whether to revise or advance to publication."""
    if state.get("validation_passed", False):
        return "publish_analysis_report"
    if state.get("validation_attempts", 0) < 2:
        return "revise_findings"
    return "publish_analysis_report"


# =============================================================================
# COMPILE LANGGRAPH WORKFLOW
# =============================================================================

def build_pattern_analysis_graph() -> StateGraph:
    """Build and compile the multi-step Fraud Pattern Analysis workflow."""
    workflow = StateGraph(PatternAnalysisState)

    workflow.add_node("gather_historical_data", node_gather_historical_data)
    workflow.add_node("compute_pattern_statistics", node_compute_pattern_statistics)
    workflow.add_node("detect_anomalous_trends", node_detect_anomalous_trends)
    workflow.add_node("synthesize_findings", node_synthesize_findings)
    workflow.add_node("validate_evidence", node_validate_evidence)
    workflow.add_node("revise_findings", node_revise_findings)
    workflow.add_node("publish_analysis_report", node_publish_analysis_report)

    # Sequential edges as specified in architecture
    workflow.add_edge(START, "gather_historical_data")
    workflow.add_edge("gather_historical_data", "compute_pattern_statistics")
    workflow.add_edge("compute_pattern_statistics", "detect_anomalous_trends")
    workflow.add_edge("detect_anomalous_trends", "synthesize_findings")
    workflow.add_edge("synthesize_findings", "validate_evidence")

    # Conditional validation edge
    workflow.add_conditional_edges(
        "validate_evidence",
        should_continue_validation,
        {
            "revise_findings": "revise_findings",
            "publish_analysis_report": "publish_analysis_report",
        },
    )
    workflow.add_edge("revise_findings", "validate_evidence")
    workflow.add_edge("publish_analysis_report", END)

    return workflow


_compiled_pattern_graph = None


def get_pattern_analysis_graph():
    """Retrieve singleton compiled instance of pattern analysis graph."""
    global _compiled_pattern_graph
    if _compiled_pattern_graph is None:
        _compiled_pattern_graph = build_pattern_analysis_graph().compile()
    return _compiled_pattern_graph


def run_pattern_analysis(
    time_horizon_hours: Optional[int] = None,
    baseline_ratio: float = 0.5,
    min_alerts_threshold: int = 10,
    model_name: str = DISTRIBUTED_PIPELINE_OPTION,
) -> Dict[str, Any]:
    """Execute complete end-to-end Fraud Pattern Analysis Agent workflow.

    Args:
        time_horizon_hours: Optional lookback window filter (e.g., 72, 168).
        baseline_ratio: Fraction of historical records allocated to baseline cohort (default: 0.5).
        min_alerts_threshold: Minimum alerts required for statistical confidence (default: 10).
        model_name: Groq model name for synthesis and interpretation.

    Returns:
        Dictionary containing published pattern report, cohort metrics, and detected shifts.
    """
    graph = get_pattern_analysis_graph()

    initial_state: PatternAnalysisState = {
        "messages": [HumanMessage(content="Discover emerging fraud patterns, cohort trends, and distribution shifts.")],
        "time_horizon_hours": time_horizon_hours,
        "baseline_ratio": baseline_ratio,
        "min_alerts_threshold": min_alerts_threshold,
        "model_name": model_name,
        "raw_historical_records": [],
        "baseline_records": [],
        "recent_records": [],
        "time_windows_metadata": {},
        "computed_statistics": {},
        "detected_shifts": [],
        "structured_report": None,
        "validation_passed": False,
        "validation_errors": [],
        "validation_attempts": 0,
        "published_report": None,
    }

    final_state = graph.invoke(initial_state)
    return final_state.get("published_report") or final_state.get("structured_report") or {}
