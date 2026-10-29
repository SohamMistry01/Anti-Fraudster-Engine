"""Anti-Fraudster | Fraud Pattern Analysis Agent Workbench.

Discovers emerging patterns across historical transaction and alert data, performs
cohort and temporal trend analysis, detects statistical distribution shifts and anomalies,
and presents evidence-backed findings strictly adhering to anonymized PCA dataset constraints.
"""

from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure repository root is on path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dashboard.utils import (
    get_priority_badge_html,
    get_status_badge_html,
    inject_custom_styles,
    render_metric_card,
)
from src.agents.config import DEFAULT_AGENT_MODEL, SUPPORTED_MODELS
from src.agents.evaluator import evaluate_pattern_grounding
from src.agents.pattern_agent import run_pattern_analysis

st.set_page_config(
    page_title="Anti-Fraudster | Pattern Analysis Agent",
    page_icon="📈",
    layout="wide",
)

inject_custom_styles()

# =============================================================================
# HEADER
# =============================================================================
st.markdown('<div class="hero-title">Fraud Pattern Analysis Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">'
    'Autonomous historical pattern discovery, cohort trend analysis, emerging distribution shift detection, '
    'and evidence-backed findings.'
    '</div>',
    unsafe_allow_html=True,
)

# Dataset limitation disclaimer callout
st.info(
    "Dataset Constraint: The European Credit Card Fraud dataset contains anonymized PCA features (V1–V28) "
    "and does not provide real merchant or cardholder identities. Findings represent deterministic mathematical "
    "distribution shifts rather than real-world merchant names, geographic locations, or customer behaviors.",
    icon="ℹ️",
)

# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================
with st.sidebar:
    st.markdown("### 📈 **Pattern Analysis Controls**")
    st.markdown(
        "<span style='color: #94a3b8; font-size: 0.85rem;'>"
        "Configure temporal windows and LLM interpretation parameters."
        "</span>",
        unsafe_allow_html=True,
    )
    st.divider()

    time_horizon_option = st.selectbox(
        "Lookback Horizon",
        options=["All Available History", "Last 24 Hours", "Last 48 Hours", "Last 72 Hours", "Last 7 Days (168h)"],
        index=0,
    )

    hours_map = {
        "All Available History": None,
        "Last 24 Hours": 24,
        "Last 48 Hours": 48,
        "Last 72 Hours": 72,
        "Last 7 Days (168h)": 168,
    }
    selected_hours = hours_map[time_horizon_option]

    baseline_ratio = st.slider(
        "Baseline Cohort Split Ratio",
        min_value=0.2,
        max_value=0.8,
        value=0.5,
        step=0.05,
        help="Proportion of chronological records allocated to baseline comparison window.",
    )

    min_alerts = st.number_input(
        "Min Alerts Threshold",
        min_value=5,
        max_value=200,
        value=10,
        step=5,
        help="Minimum historical alerts required for statistical confidence.",
    )

    selected_model = st.selectbox(
        "Reasoning Model",
        options=SUPPORTED_MODELS,
        index=SUPPORTED_MODELS.index(DEFAULT_AGENT_MODEL) if DEFAULT_AGENT_MODEL in SUPPORTED_MODELS else 0,
    )

    st.divider()
    run_clicked = st.button("🚀 Run Pattern Analysis Agent", type="primary", width="stretch")

# =============================================================================
# EXECUTION OR CACHE RETRIEVAL
# =============================================================================
if "latest_pattern_report" not in st.session_state:
    st.session_state["latest_pattern_report"] = None

if run_clicked or st.session_state["latest_pattern_report"] is None:
    with st.spinner("Executing Fraud Pattern Analysis Agent workflow across historical data..."):
        try:
            report_data = run_pattern_analysis(
                time_horizon_hours=selected_hours,
                baseline_ratio=baseline_ratio,
                min_alerts_threshold=min_alerts,
                model_name=selected_model,
            )
            st.session_state["latest_pattern_report"] = report_data
            st.success("Pattern analysis complete and verified!")
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

report = st.session_state.get("latest_pattern_report")

if not report:
    st.warning("No pattern analysis report generated yet. Click 'Run Pattern Analysis Agent' in the sidebar.")
    st.stop()

# =============================================================================
# EXECUTIVE SUMMARY
# =============================================================================
with st.container(border=True):
    col_hdr1, col_hdr2 = st.columns([3, 1])
    with col_hdr1:
        st.markdown(f"#### 📋 **Executive Summary • Report `{report.get('report_id')}`**")
        st.markdown(f"**Generated:** `{report.get('generated_at')}`")
        st.markdown(f"> *{report.get('executive_summary')}*")
    with col_hdr2:
        audit_passed = report.get("grounding_audit_passed", True)
        if audit_passed:
            st.markdown(
                "<div style='text-align: right; margin-top: 10px;'>"
                "<span style='background: rgba(16, 185, 129, 0.2); color: #10b981; padding: 6px 12px; border-radius: 8px; font-weight: 600; font-size: 0.85rem; border: 1px solid rgba(16, 185, 129, 0.4);'>"
                "✓ Factual Grounding Verified"
                "</span>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='text-align: right; margin-top: 10px;'>"
                "<span style='background: rgba(245, 158, 11, 0.2); color: #f59e0b; padding: 6px 12px; border-radius: 8px; font-weight: 600; font-size: 0.85rem;'>"
                "⚠ Grounding Review Pending"
                "</span>"
                "</div>",
                unsafe_allow_html=True,
            )

st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# COHORT COMPARISON METRICS
# =============================================================================
b_cohort = report.get("baseline_window", {})
r_cohort = report.get("recent_window", {})

st.markdown("### 📊 **Cohort Comparison: Baseline vs. Recent Window**")

col_b, col_r = st.columns(2)

with col_b:
    with st.container(border=True):
        st.markdown(f"#### ⏳ **{b_cohort.get('window_name', 'Baseline Window')}**")
        st.caption(f"{b_cohort.get('start_time', 'Start')} → {b_cohort.get('end_time', 'End')}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Alert Count", f"{b_cohort.get('alert_count', 0)}")
        m2.metric("Mean Amount", f"€{b_cohort.get('mean_amount', 0.0):.2f}")
        m3.metric("P90 Exposure", f"€{b_cohort.get('p90_amount', 0.0):.2f}")

        m4, m5, m6 = st.columns(3)
        m4.metric("High-Anomaly Share", f"{b_cohort.get('high_anomaly_share_pct', 0.0):.1f}%")
        m5.metric("High-Fraud Share", f"{b_cohort.get('high_fraud_share_pct', 0.0):.1f}%")
        b_fpr = b_cohort.get("false_positive_rate")
        m6.metric("Verified FPR", f"{b_fpr:.1%}" if b_fpr is not None else "N/A")

with col_r:
    with st.container(border=True):
        st.markdown(f"#### ⚡ **{r_cohort.get('window_name', 'Recent Window')}**")
        st.caption(f"{r_cohort.get('start_time', 'Start')} → {r_cohort.get('end_time', 'End')}")

        d_cnt = r_cohort.get("alert_count", 0) - b_cohort.get("alert_count", 0)
        d_amt = r_cohort.get("mean_amount", 0.0) - b_cohort.get("mean_amount", 0.0)
        d_p90 = r_cohort.get("p90_amount", 0.0) - b_cohort.get("p90_amount", 0.0)
        d_anom = r_cohort.get("high_anomaly_share_pct", 0.0) - b_cohort.get("high_anomaly_share_pct", 0.0)
        d_p = r_cohort.get("high_fraud_share_pct", 0.0) - b_cohort.get("high_fraud_share_pct", 0.0)

        m1, m2, m3 = st.columns(3)
        m1.metric("Alert Count", f"{r_cohort.get('alert_count', 0)}", delta=f"{d_cnt}")
        m2.metric("Mean Amount", f"€{r_cohort.get('mean_amount', 0.0):.2f}", delta=f"€{d_amt:.2f}")
        m3.metric("P90 Exposure", f"€{r_cohort.get('p90_amount', 0.0):.2f}", delta=f"€{d_p90:.2f}")

        m4, m5, m6 = st.columns(3)
        m4.metric("High-Anomaly Share", f"{r_cohort.get('high_anomaly_share_pct', 0.0):.1f}%", delta=f"{d_anom:+.1f}%")
        m5.metric("High-Fraud Share", f"{r_cohort.get('high_fraud_share_pct', 0.0):.1f}%", delta=f"{d_p:+.1f}%")
        r_fpr = r_cohort.get("false_positive_rate")
        d_fpr = (r_fpr - b_fpr) if (r_fpr is not None and b_fpr is not None) else None
        m6.metric(
            "Verified FPR",
            f"{r_fpr:.1%}" if r_fpr is not None else "N/A",
            delta=f"{d_fpr * 100:+.1f}%" if d_fpr is not None else None,
        )

st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# DETECTED STATISTICAL SHIFTS
# =============================================================================
st.markdown("### 🚨 **Emerging Anomaly & Distribution Shift Detections**")

shifts = report.get("detected_shifts", [])
if shifts:
    shift_rows = []
    for s in shifts:
        sev = s.get("severity", "INFO")
        sev_color = {
            "CRITICAL": "#ef4444",
            "HIGH": "#f97316",
            "MEDIUM": "#eab308",
            "LOW": "#3b82f6",
            "INFO": "#94a3b8",
        }.get(sev, "#94a3b8")

        badge = f"<span style='background: {sev_color}22; color: {sev_color}; padding: 3px 8px; border-radius: 6px; font-weight: 700; font-size: 0.75rem; border: 1px solid {sev_color}44;'>{sev}</span>"

        shift_rows.append({
            "Severity": badge,
            "Metric": s.get("metric_name"),
            "Baseline": f"{s.get('baseline_value'):.2f}",
            "Recent": f"{s.get('recent_value'):.2f}",
            "Absolute Shift": f"{s.get('absolute_change'):+.2f}",
            "Relative Shift": f"{s.get('relative_change_pct'):+.1f}%",
            "Significant": "✓ Yes" if s.get("is_significant") else "No",
            "Summary": s.get("shift_summary"),
        })

    shift_df = pd.DataFrame(shift_rows)
    st.write(shift_df.to_html(escape=False, index=False), unsafe_allow_html=True)
else:
    st.info("No significant statistical shifts detected between baseline and recent window.")

st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# EVIDENCE-BACKED FINDINGS
# =============================================================================
st.markdown("### 🔍 **Evidence-Backed Pattern Findings & Interpretations**")

findings = report.get("findings", [])
for idx, f in enumerate(findings, start=1):
    cat = f.get("category", "ANOMALY_CONCENTRATION")
    conf = f.get("statistical_confidence", "MEDIUM")
    conf_color = {"HIGH": "#10b981", "MEDIUM": "#f59e0b", "LOW": "#64748b"}.get(conf, "#64748b")

    with st.expander(f"Finding #{idx}: {f.get('pattern_title')} • [{cat}]", expanded=(idx <= 2)):
        st.markdown(f"**Measured Pattern:** {f.get('measured_pattern')}")

        supp = f.get("supporting_statistics", {})
        if supp:
            st.markdown(
                f"<div style='font-size: 0.85rem; color: #94a3b8; background: rgba(15, 23, 42, 0.6); padding: 8px 12px; border-radius: 8px; margin: 8px 0;'>"
                f"<b>Supporting Statistics:</b> {supp}"
                f"</div>",
                unsafe_allow_html=True,
            )

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(
                f"**Statistical Confidence:** "
                f"<span style='color: {conf_color}; font-weight: 700;'>{conf}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Dataset Limitations:** *{f.get('dataset_limitations')}*")
        with c2:
            st.markdown(f"**Recommended Follow-Up:**\n> {f.get('recommended_followup')}")

if report.get("model_error_analysis"):
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("#### 🎯 **Model Error & Verification Analysis**")
        st.write(report.get("model_error_analysis"))
