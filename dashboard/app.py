"""Anti-Fraudster | Executive Fraud Operations & Risk Triage Center.

Streamlit executive overview displaying alert queues, case lifecycle metrics,
priority distributions, and real-time transaction scoring simulator.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure repository root is on path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dashboard.utils import (
    get_priority_badge_html,
    get_status_badge_html,
    inject_custom_styles,
    load_holdout_samples,
    render_metric_card,
    seed_sample_database_data_if_needed,
)
from src.cases.database import DATABASE_URL, get_db_context
from src.cases.models import Alert, AlertPriority, AlertStatus, CaseStatus, CaseVerdict, InvestigationCase
from src.cases.repository import AlertRepository, CaseRepository
from src.cases.service import get_case_service
from src.inference.predictor import get_predictor

# Page configuration
st.set_page_config(
    page_title="Anti-Fraudster | Ops Workspace",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_styles()

# Automatically seed sample data if database is fresh
try:
    seed_sample_database_data_if_needed(target_count=5)
except Exception:
    pass

# =============================================================================
# SIDEBAR: SYSTEM HEALTH & QUICK NAVIGATION
# =============================================================================
with st.sidebar:
    st.markdown("### 🛡️ **Anti-Fraudster Engine**")
    st.markdown(
        "<span style='color: #94a3b8; font-size: 0.85rem;'>"
        "Tier-1 Financial Risk & Investigation Workbench"
        "</span>",
        unsafe_allow_html=True,
    )
    st.divider()

    # System Status Indicators
    st.markdown("#### **System Architecture**")
    db_type = "PostgreSQL" if "postgresql" in DATABASE_URL else "SQLite (Local)"
    st.markdown(f"**Database:** `{db_type}`")
    st.markdown("**Classifier:** `XGBoost v1.0.0`")
    st.markdown("**Unsupervised Signal:** `Isolation Forest`")
    st.markdown("**Cost-Optimal Threshold:** `0.182`")
    st.markdown("**Default Threshold:** `0.500`")

    st.divider()
    st.markdown("#### **Quick Actions**")
    if st.button("🔄 Refresh Workspace Data", use_container_width=True):
        st.rerun()

    if st.button("🌱 Seed Sample Alerts/Cases", use_container_width=True):
        added = seed_sample_database_data_if_needed(target_count=10)
        st.success(f"Seeded {added} sample records from holdout data.")
        st.rerun()

    st.divider()
    st.markdown(
        "<div style='font-size: 0.75rem; color: #64748b;'>"
        "Anti-Fraudster AI Platform • Audit Compliant • Preserves Immutable Inference Snapshots"
        "</div>",
        unsafe_allow_html=True,
    )

# =============================================================================
# MAIN HEADER
# =============================================================================
st.markdown('<div class="hero-title">Fraud Operations & Case Management</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">'
    'Real-time prioritised alert queues, auditable investigator workflows, and dual-signal anomaly intelligence.'
    '</div>',
    unsafe_allow_html=True,
)

# =============================================================================
# DATA FETCHING & KPI METRICS
# =============================================================================
with get_db_context() as db:
    alerts_summary = AlertRepository.get_summary_counts(db)
    cases_summary = CaseRepository.get_summary_metrics(db)
    recent_alerts_orm, _ = AlertRepository.list_alerts(db, limit=5)
    recent_cases_orm, _ = CaseRepository.list_cases(db, limit=5)
    recent_alerts = [a.to_dict() for a in recent_alerts_orm]
    recent_cases = [c.to_dict(include_relations=False) for c in recent_cases_orm]

    # Compute financial exposure and quadrant counts in one query pass
    all_alerts = db.query(Alert).all()
    exposure_sum = sum(max(0.0, a.amount) for a in all_alerts if a.status != AlertStatus.DISMISSED.value)
    quadrant_counts = {}
    for a in all_alerts:
        q = a.operational_quadrant or "CONFIRMED_LEGITIMATE"
        quadrant_counts[q] = quadrant_counts.get(q, 0) + 1


total_alerts = alerts_summary.get("total_alerts", 0)
new_alerts = alerts_summary.get("by_status", {}).get(AlertStatus.NEW.value, 0)
critical_alerts = alerts_summary.get("by_priority", {}).get(AlertPriority.CRITICAL.value, 0)

total_cases = cases_summary.get("total_cases", 0)
open_cases = (
    cases_summary.get("by_status", {}).get(CaseStatus.OPEN.value, 0)
    + cases_summary.get("by_status", {}).get(CaseStatus.UNDER_INVESTIGATION.value, 0)
)
confirmed_fraud = cases_summary.get("confirmed_fraud_cases", 0)
false_positives = cases_summary.get("false_positive_cases", 0)

# Display 4 Key Performance Indicators
col1, col2, col3, col4 = st.columns(4)

with col1:
    render_metric_card(
        title="Prioritised Alerts",
        value=f"{total_alerts}",
        subtitle=f"{new_alerts} new unreviewed • {critical_alerts} critical",
        icon="🚨",
        border_color="rgba(239, 68, 68, 0.4)" if critical_alerts > 0 else None,
    )

with col2:
    render_metric_card(
        title="Active Case Investigations",
        value=f"{open_cases}",
        subtitle=f"{total_cases} total created cases",
        icon="💼",
        border_color="rgba(96, 165, 250, 0.4)",
    )

with col3:
    render_metric_card(
        title="Confirmed Fraud Cases",
        value=f"{confirmed_fraud}",
        subtitle=f"{false_positives} false positives resolved",
        icon="⚖️",
        border_color="rgba(168, 85, 247, 0.4)",
    )

with col4:
    fp_ratio = (
        f"{(false_positives / (confirmed_fraud + false_positives) * 100):.1f}%"
        if (confirmed_fraud + false_positives) > 0
        else "0.0%"
    )
    render_metric_card(
        title="False Positive Rate",
        value=fp_ratio,
        subtitle="Analyst disposition efficiency",
        icon="🎯",
        border_color="rgba(16, 185, 129, 0.4)",
    )

st.write("")

# =============================================================================
# OPERATIONAL CHARTS: PRIORITY DONUT & QUADRANTS
# =============================================================================
chart_col1, chart_col2 = st.columns([1, 1])

with chart_col1:
    st.markdown("#### **Alert Queue by Urgency Priority**")
    priority_data = alerts_summary.get("by_priority", {})
    if any(priority_data.values()):
        p_df = pd.DataFrame([
            {"Priority": p, "Count": c}
            for p, c in priority_data.items()
        ])
        color_map = {
            "CRITICAL": "#ef4444",
            "HIGH": "#f97316",
            "MEDIUM": "#eab308",
            "LOW": "#06b6d4",
        }
        fig_priority = px.pie(
            p_df,
            names="Priority",
            values="Count",
            hole=0.55,
            color="Priority",
            color_discrete_map=color_map,
        )
        fig_priority.update_traces(
            textinfo="value+percent",
            hoverinfo="label+value+percent",
            marker=dict(line=dict(color="#0f172a", width=2)),
        )
        fig_priority.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10, l=10, r=10),
            height=260,
            showlegend=True,
            font=dict(color="#94a3b8", family="Plus Jakarta Sans"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_priority, use_container_width=True)
    else:
        st.info("No alert queue data to visualize yet.")

with chart_col2:
    st.markdown("#### **Operational Risk Quadrants Distribution**")
    if quadrant_counts:
        q_df = pd.DataFrame([
            {"Quadrant": q.replace("_", " "), "Alerts": cnt}
            for q, cnt in quadrant_counts.items()
        ])
        fig_quad = px.bar(
            q_df,
            x="Quadrant",
            y="Alerts",
            color="Quadrant",
            color_discrete_sequence=["#ef4444", "#a855f7", "#3b82f6", "#10b981"],
        )
        fig_quad.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10, l=10, r=10),
            height=260,
            showlegend=False,
            font=dict(color="#94a3b8", family="Plus Jakarta Sans"),
            xaxis=dict(showgrid=False, title=""),
            yaxis=dict(showgrid=True, gridcolor="rgba(148, 163, 184, 0.1)", title="Count"),
        )
        st.plotly_chart(fig_quad, use_container_width=True)
    else:
        st.info("No quadrant distribution available.")

st.divider()

# =============================================================================
# RECENT HIGH-URGENCY ALERTS & WORKSPACE JUMP
# =============================================================================
table_col1, table_col2 = st.columns([3, 2])

with table_col1:
    st.markdown("#### **🚨 Top Prioritised Alerts (Live Queue Preview)**")
    if recent_alerts:
        rows = []
        for a in recent_alerts:
            rows.append({
                "Alert ID": a["alert_id"],
                "Priority": a["priority"],
                "Fraud Prob": f"{a['fraud_probability']:.3f}",
                "Anomaly Score": f"{a['anomaly_score']:.3f}",
                "Risk Category": a["risk_category"],
                "Status": a["status"],
                "Case": a["case_id"] or "—",
            })
        df_preview = pd.DataFrame(rows)
        st.dataframe(
            df_preview,
            use_container_width=True,
            hide_index=True,
        )
        st.markdown(
            "👉 *Go to the **Alerts** page in the sidebar for full triage actions, filtering, and escalation.*"
        )
    else:
        st.info("Queue is currently clear. No alerts require immediate action.")

with table_col2:
    st.markdown("#### **💼 Active Investigation Cases**")
    if recent_cases:
        c_rows = []
        for c in recent_cases:
            orig_p = c.get("original_prediction", {}).get("fraud_probability", 0.0)
            c_rows.append({
                "Case ID": c["case_id"],
                "Status": c["status"].replace("_", " "),
                "Priority": c["priority"],
                "Analyst": c["assigned_analyst"] or "Unassigned",
                "Original P(Fraud)": f"{orig_p:.3f}",
            })
        df_c_preview = pd.DataFrame(c_rows)
        st.dataframe(
            df_c_preview,
            use_container_width=True,
            hide_index=True,
        )
        st.markdown(
            "👉 *Go to the **Investigations** page in the sidebar for the full analyst workbench and audit trail.*"
        )
    else:
        st.info("No investigation cases opened yet.")

st.divider()

# =============================================================================
# LIVE TRANSACTION TRIAGE SIMULATOR
# =============================================================================
st.markdown("### ⚡ **Real-Time Transaction Scoring & Triage Simulator**")
st.markdown(
    "<span style='color: #94a3b8; font-size: 0.9rem;'>"
    "Simulate scoring of an incoming card transaction through the full risk policy engine. "
    "Transactions exceeding critical thresholds are automatically converted into prioritised alerts and investigation cases."
    "</span>",
    unsafe_allow_html=True,
)

sim_col1, sim_col2 = st.columns([2, 3])

fraud_samples, legit_samples = load_holdout_samples(n_fraud=5, n_legit=5)

with sim_col1:
    sample_type = st.radio(
        "Choose Transaction Source:",
        ["Holdout Ground-Truth Fraud", "Holdout Ground-Truth Legitimate", "Manual Custom Input"],
        horizontal=True,
    )

    if sample_type == "Holdout Ground-Truth Fraud" and fraud_samples:
        sample_idx = st.selectbox(
            "Select Holdout Fraud Sample:",
            range(len(fraud_samples)),
            format_func=lambda i: f"Fraud Example #{i+1} (Amount: €{fraud_samples[i].get('Amount', 0):.2f})",
        )
        selected_tx = fraud_samples[sample_idx]
        apply_prep = False
    elif sample_type == "Holdout Ground-Truth Legitimate" and legit_samples:
        sample_idx = st.selectbox(
            "Select Holdout Legitimate Sample:",
            range(len(legit_samples)),
            format_func=lambda i: f"Legitimate Example #{i+1} (Amount: €{legit_samples[i].get('Amount', 0):.2f})",
        )
        selected_tx = legit_samples[sample_idx]
        apply_prep = False
    else:
        amt = st.number_input("Transaction Amount (€)", value=245.50, min_value=0.01, step=10.0)
        time_sec = st.number_input("Transaction Time (Seconds)", value=43200.0, min_value=0.0, step=60.0)
        # Construct synthetic PCA with elevated V14/V17 if desired
        selected_tx = {"Time": time_sec, "Amount": amt}
        for i in range(1, 29):
            selected_tx[f"V{i}"] = 0.0
        apply_prep = True

    threshold_choice = st.selectbox(
        "Decision Threshold:",
        [
            "cost_optimal (0.182) — Min. Financial Loss",
            "default (0.500) — Balanced Baseline",
            "f1_optimal (0.798) — High Precision",
            "f2_optimal (0.552) — High Recall",
            "custom (Manual Cutoff)",
        ],
        index=0,
    )
    threshold_key = threshold_choice.split()[0]
    if threshold_key == "custom":
        custom_thresh_val = st.number_input(
            "Custom Cutoff (0.0 to 1.0):", min_value=0.001, max_value=0.999, value=0.500, step=0.02
        )
        threshold_to_use = float(custom_thresh_val)
    else:
        threshold_to_use = threshold_key

    auto_escalate_check = st.checkbox("Auto-escalate CRITICAL alerts to Cases", value=True)

    triage_button = st.button("🚀 Score Transaction & Triage", type="primary", use_container_width=True)

with sim_col2:
    if triage_button:
        with st.spinner("Executing XGBoost classifier & Isolation Forest anomaly assessment..."):
            service = get_case_service()
            with get_db_context() as db:
                triage_res = service.process_and_triage_transaction(
                    db=db,
                    transaction=selected_tx,
                    threshold=threshold_to_use,
                    apply_preprocessing=apply_prep,
                    auto_escalate=auto_escalate_check,
                )

        assessment = triage_res["assessment"]
        supervised = assessment["supervised_signal"]
        anomaly = assessment["unsupervised_anomaly_signal"]

        p_fraud = supervised["fraud_probability"]
        a_score = anomaly["anomaly_score"]
        quadrant = assessment["operational_quadrant"]
        category = supervised["risk_category"]

        # Display Result Card
        st.markdown("#### **Triage Diagnostics**")
        st.markdown(
            """
            <style>
            .st-key-triage_diagnostics [data-testid="stMetricValue"],
            .st-key-triage_diagnostics [data-testid="stMetricValue"] > div,
            .st-key-triage-diagnostics [data-testid="stMetricValue"],
            .st-key-triage-diagnostics [data-testid="stMetricValue"] > div {
                font-size: 1.15rem !important;
                font-weight: 700 !important;
                white-space: normal !important;
                word-break: break-word !important;
                line-height: 1.25 !important;
            }
            .st-key-triage_diagnostics [data-testid="stMetricLabel"],
            .st-key-triage-diagnostics [data-testid="stMetricLabel"] {
                font-size: 0.82rem !important;
            }
            .st-key-triage_diagnostics [data-testid="stMetricDelta"],
            .st-key-triage-diagnostics [data-testid="stMetricDelta"] {
                font-size: 0.78rem !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        with st.container(key="triage_diagnostics"):
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric("P(Fraud) Supervised", f"{p_fraud:.4f}", delta=category)
                st.metric("Risk Category", category)
            with res_col2:
                st.metric("Anomaly Score (IForest)", f"{a_score:.4f}", delta=anomaly["anomaly_tier"])
                st.metric("Operational Quadrant", quadrant.replace("_", " "))

        if triage_res["alert_created"]:
            alt = triage_res["alert"]
            st.success(
                f"🚨 **Alert Created!** Alert ID: `{alt['alert_id']}` | Urgency: **{alt['priority']}**"
            )
            st.markdown(f"**Trigger Reasons:** {', '.join(alt['reasons'])}")

            if triage_res["case_created"]:
                cs = triage_res["case"]
                st.info(
                    f"💼 **Auto-Escalated to Case!** Case ID: `{cs['case_id']}` | Status: **{cs['status']}**"
                )
        else:
            st.balloons()
            st.success("✅ Transaction classified as Legitimate. No alert triggered.")
    else:
        st.info("Select a sample transaction and click **Score Transaction & Triage** to observe real-time policy evaluation.")
