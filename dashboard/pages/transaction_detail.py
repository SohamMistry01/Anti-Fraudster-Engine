"""Anti-Fraudster | Deep-Dive Transaction Detail & Explainability.

Provides detailed visual evidence inspection: interactive Plotly TreeSHAP waterfall
diagrams, Isolation Forest anomaly radar/gauges, and complete 30-feature statistical vectors.
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
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dashboard.utils import (
    get_priority_badge_html,
    get_status_badge_html,
    inject_custom_styles,
    load_holdout_samples,
    render_metric_card,
)
from src.cases.database import get_db_context
from src.cases.models import Alert, InvestigationCase
from src.explainability.risk_assessor import get_risk_assessor

st.set_page_config(
    page_title="Anti-Fraudster | Transaction Evidence Deep-Dive",
    page_icon="🔬",
    layout="wide",
)

inject_custom_styles()

st.markdown('<div class="hero-title">🔬 Transaction Evidence Deep-Dive</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">'
    'SHAP feature contributions, unsupervised anomaly distribution positioning, and strict statistical vector inspection.'
    '</div>',
    unsafe_allow_html=True,
)

# =============================================================================
# DATA SOURCE SELECTOR
# =============================================================================
source_mode = st.radio(
    "Choose Transaction Source:",
    ["Select from Existing Alert/Case", "Select Holdout Sample"],
    horizontal=True,
)

selected_tx: Optional[Dict[str, Any]] = None
tx_label: str = "Unknown"
apply_prep: bool = False

if source_mode == "Select from Existing Alert/Case":
    with get_db_context() as db:
        alerts_orm = db.query(Alert).order_by(Alert.id.desc()).limit(50).all()
        alerts = [a.to_dict() for a in alerts_orm]

    if not alerts:
        st.warning("No alerts in database. Falling back to holdout samples.")
        source_mode = "Select Holdout Sample"
    else:
        alert_choice = st.selectbox(
            "Select Alert to Deep-Dive:",
            alerts,
            format_func=lambda a: f"{a['alert_id']} | Urgency: {a['priority']} | P(Fraud): {a['fraud_probability']:.3f} | Quadrant: {a['operational_quadrant']}",
        )
        selected_tx = alert_choice["transaction_payload"]
        tx_label = f"Alert {alert_choice['alert_id']} (TX: {alert_choice['transaction_id']})"
        # If Time or Amount are negative, it's already preprocessed
        apply_prep = not (float(selected_tx.get("Amount", 0)) < 0 or float(selected_tx.get("Time", 0)) < 0)

if source_mode == "Select Holdout Sample":
    fraud_samples, legit_samples = load_holdout_samples(n_fraud=5, n_legit=5)
    sample_cat = st.selectbox("Sample Class:", ["Ground-Truth Fraud", "Ground-Truth Legitimate"])
    samples = fraud_samples if sample_cat == "Ground-Truth Fraud" else legit_samples
    idx = st.selectbox("Sample Index:", range(len(samples)), format_func=lambda i: f"Sample #{i+1} (Amount: €{samples[i].get('Amount', 0):.2f})")
    selected_tx = samples[idx]
    tx_label = f"Holdout {sample_cat} #{idx+1}"
    apply_prep = False

if not selected_tx:
    st.error("No transaction selected.")
    st.stop()

# Preprocessing control
apply_prep_choice = st.checkbox(
    "Apply RobustScaler Preprocessing (uncheck if transaction is already scaled)",
    value=apply_prep,
)

# =============================================================================
# RUN DUAL-SIGNAL ASSESSMENT & GENERATE PLOTS
# =============================================================================
assessor = get_risk_assessor()

with st.spinner("Generating TreeSHAP waterfall and anomaly distributions..."):
    assessment = assessor.assess_transaction(
        transaction=selected_tx,
        threshold="cost_optimal",
        apply_preprocessing=apply_prep_choice,
        generate_plots=True,
    )

supervised = assessment["supervised_signal"]
anomaly = assessment["unsupervised_anomaly_signal"]
shap_data = assessment["shap_explanation"]
quadrant = assessment["operational_quadrant"]
action = assessment["recommended_action"]
plots = assessment.get("plots", {})

# =============================================================================
# TOP METRICS
# =============================================================================
st.markdown(f"### **Diagnostic Overview: {tx_label}**")

st.markdown(
    """
    <style>
    .st-key-diagnostic_overview_cards .metric-card-value {
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        white-space: normal !important;
        word-break: break-word !important;
        line-height: 1.25 !important;
    }
    .st-key-diagnostic_overview_cards .metric-card-title {
        font-size: 0.78rem !important;
    }
    .st-key-diagnostic_overview_cards .metric-card-sub {
        font-size: 0.75rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container(key="diagnostic_overview_cards"):
    top_c1, top_c2, top_c3, top_c4 = st.columns(4)
    with top_c1:
        render_metric_card(
            title="Supervised P(Fraud)",
            value=f"{supervised['fraud_probability']:.4f}",
            subtitle=f"Tier: {supervised['risk_category']}",
            icon="🧠",
            border_color="#ef4444" if supervised['fraud_probability'] >= 0.182 else "#10b981",
            font_size="1.15rem",
        )
    with top_c2:
        render_metric_card(
            title="Anomaly Score",
            value=f"{anomaly['anomaly_score']:.4f}",
            subtitle=f"Tier: {anomaly['anomaly_tier']}",
            icon="📡",
            border_color="#a855f7" if anomaly['anomaly_score'] >= 0.6 else "#64748b",
            font_size="1.15rem",
        )
    with top_c3:
        render_metric_card(
            title="Operational Quadrant",
            value=quadrant.replace("_", " "),
            subtitle="Policy Routing",
            icon="🧭",
            border_color="#3b82f6",
            font_size="1.15rem",
        )
    with top_c4:
        render_metric_card(
            title="Recommended Action",
            value=action.replace("_", " ") if isinstance(action, str) else action,
            subtitle="Decision Engine",
            icon="⚡",
            border_color="#10b981" if action == "APPROVE" else "#ef4444",
            font_size="1.15rem",
        )

st.write("")

# =============================================================================
# SHAP WATERFALL & DUAL-SIGNAL GAUGE
# =============================================================================
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("#### **TreeSHAP Feature Attribution (Waterfall Plot)**")
    st.markdown(
        "<div style='font-size: 0.8rem; color: #94a3b8; margin-bottom: 0.5rem;'>"
        "Displays exact log-odds marginal contributions of individual PCA components to the final XGBoost score."
        "</div>",
        unsafe_allow_html=True,
    )

    plot_fig = plots.get("shap_plot_fig")
    plot_dict = plots.get("shap_plot_dict")
    plot_html = plots.get("shap_plot_html")

    if plot_fig is not None:
        st.plotly_chart(plot_fig, use_container_width=True)
    elif plot_dict is not None:
        st.plotly_chart(go.Figure(plot_dict), use_container_width=True)
    elif plot_html is not None:
        st.components.v1.html(plot_html, height=480, scrolling=True)
    else:
        st.info("No waterfall plot available.")

with col_right:
    st.markdown("#### **Dual-Signal Gauge & Anomaly Position**")
    st.markdown(
        "<div style='font-size: 0.8rem; color: #94a3b8; margin-bottom: 0.5rem;'>"
        "Supervised fraud probability vs Unsupervised Isolation Forest anomaly score."
        "</div>",
        unsafe_allow_html=True,
    )

    fig_gauge = go.Figure()
    fig_gauge.add_trace(go.Indicator(
        mode="gauge+number",
        value=supervised["fraud_probability"] * 100,
        number={'font': {'size': 22, 'color': '#f8fafc', 'family': 'Plus Jakarta Sans'}},
        title={'text': "P(Fraud) Supervised (%)", 'font': {'size': 12, 'color': '#94a3b8', 'family': 'Plus Jakarta Sans'}},
        domain={'x': [0.0, 0.43], 'y': [0, 1]},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickvals': [0, 50, 100],
                'tickfont': {'size': 9, 'color': '#94a3b8'},
                'tickcolor': "#94a3b8",
            },
            'bar': {'color': "#ef4444" if supervised["fraud_probability"] >= 0.5 else "#3b82f6"},
            'threshold': {
                'line': {'color': "#f97316", 'width': 3},
                'thickness': 0.75,
                'value': 18.2,  # Cost-optimal threshold
            },
        },
    ))
    fig_gauge.add_trace(go.Indicator(
        mode="gauge+number",
        value=anomaly["anomaly_score"] * 100,
        number={'font': {'size': 22, 'color': '#f8fafc', 'family': 'Plus Jakarta Sans'}},
        title={'text': "Anomaly Score (%)", 'font': {'size': 12, 'color': '#94a3b8', 'family': 'Plus Jakarta Sans'}},
        domain={'x': [0.57, 1.0], 'y': [0, 1]},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickvals': [0, 50, 100],
                'tickfont': {'size': 9, 'color': '#94a3b8'},
                'tickcolor': "#94a3b8",
            },
            'bar': {'color': "#a855f7"},
            'threshold': {
                'line': {'color': "#eab308", 'width': 3},
                'thickness': 0.75,
                'value': 50.0,
            },
        },
    ))
    fig_gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc", family="Plus Jakarta Sans"),
        height=220,
        margin=dict(t=25, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("#### **Evidence Summary & Synthesis**")
    assessment_summary = assessment.get("assessment_summary")
    if assessment_summary:
        st.info(assessment_summary)

    shap_summary = shap_data.get("evidence_summary")
    if shap_summary and shap_summary != assessment_summary:
        st.markdown(
            f"<div style='font-size: 0.82rem; color: #cbd5e1; background: rgba(30,41,59,0.5); padding: 0.8rem; border-radius: 6px; border-left: 3px solid #60a5fa; margin-top: 0.5rem;'>"
            f"<b>Statistical SHAP Attribution:</b><br>{shap_summary}"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Highlight primary factor breakdown
    evidence_breakdown = shap_data.get("evidence_breakdown", {})
    pos_factors = evidence_breakdown.get("fraud_driving_factors", [])
    neg_factors = evidence_breakdown.get("legitimate_mitigating_factors", [])
    if pos_factors or neg_factors:
        st.markdown("<div style='margin-top: 0.8rem;'><b>Top Contributing Components:</b></div>", unsafe_allow_html=True)
        if pos_factors:
            for f in pos_factors:
                st.markdown(f"- 🔴 `{f}`")
        if neg_factors:
            for f in neg_factors:
                st.markdown(f"- 🟢 `{f}`")

st.divider()

# =============================================================================
# 30-FEATURE STATISTICAL VECTOR INSPECTION
# =============================================================================
st.markdown("### 📊 **Complete 30-Feature Inspection Vector**")
st.markdown(
    "<div style='font-size: 0.82rem; color: #94a3b8; margin-bottom: 0.8rem;'>"
    "Features V1–V28 are anonymised principal components from PCA transformation to protect cardholder privacy. "
    "All values represent statistical variance components without fabricated domain assumptions."
    "</div>",
    unsafe_allow_html=True,
)

vec_rows = []
for k, v in selected_tx.items():
    if k == "Class":
        continue
    val_flt = float(v)
    vec_rows.append({
        "Feature": k,
        "Value": round(val_flt, 6),
        "Absolute Magnitude": round(abs(val_flt), 4),
        "Type": "Transaction Parameter" if k in ["Amount", "Time"] else "Anonymized PCA Component",
    })

vec_df = pd.DataFrame(vec_rows)
# Sort with Amount and Time first, then V1..V28
st.dataframe(
    vec_df,
    use_container_width=True,
    hide_index=True,
)
