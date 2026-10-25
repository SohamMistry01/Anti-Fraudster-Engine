"""Anti-Fraudster | Prioritised Alert Queue & Triage Desk.

Provides a triage workspace for reviewing, filtering, searching, and escalating
scored transactions into auditable investigation cases.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
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
from src.cases.database import get_db_context
from src.cases.models import Alert, AlertPriority, AlertStatus
from src.cases.repository import AlertRepository
from src.cases.service import get_case_service

st.set_page_config(
    page_title="Anti-Fraudster | Prioritised Alert Queue",
    page_icon="🚨",
    layout="wide",
)

inject_custom_styles()

st.markdown('<div class="hero-title">🚨 Prioritised Alert Queue</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">'
    'Triage incoming high-risk alerts ordered by urgency priority, financial risk, and model certainty.'
    '</div>',
    unsafe_allow_html=True,
)

# =============================================================================
# FILTERS & SEARCH CONTROL BAR
# =============================================================================
with st.container():
    f_col1, f_col2, f_col3, f_col4 = st.columns([2, 2, 3, 3])

    with f_col1:
        status_filter = st.selectbox(
            "Lifecycle Status",
            ["ALL", AlertStatus.NEW.value, AlertStatus.ESCALATED.value, AlertStatus.CASE_CREATED.value, AlertStatus.DISMISSED.value],
            index=0,
        )

    with f_col2:
        priority_filter = st.selectbox(
            "Urgency Priority",
            ["ALL", AlertPriority.CRITICAL.value, AlertPriority.HIGH.value, AlertPriority.MEDIUM.value, AlertPriority.LOW.value],
            index=0,
        )

    with f_col3:
        min_p = st.slider("Min Fraud Probability P(Fraud)", 0.0, 1.0, 0.0, step=0.05)

    with f_col4:
        search_query = st.text_input("🔍 Search Alert / Transaction / Case ID", placeholder="e.g. ALT- or CASE-")

# =============================================================================
# FETCH ALERTS FROM DATABASE
# =============================================================================
with get_db_context() as db:
    s_val = None if status_filter == "ALL" else status_filter
    p_val = None if priority_filter == "ALL" else priority_filter
    q_val = search_query.strip() if search_query.strip() else None

    alerts_orm, total_count = AlertRepository.list_alerts(
        db=db,
        status=s_val,
        priority=p_val,
        min_probability=min_p if min_p > 0.0 else None,
        search=q_val,
        limit=100,
    )
    alerts = [a.to_dict() for a in alerts_orm]
    summary_counts = AlertRepository.get_summary_counts(db)

# Header Metric Pills
m_col1, m_col2, m_col3, m_col4 = st.columns(4)
with m_col1:
    st.metric("Total in Queue", total_count)
with m_col2:
    st.metric("New (Unreviewed)", summary_counts.get("by_status", {}).get(AlertStatus.NEW.value, 0))
with m_col3:
    st.metric("Critical Urgency", summary_counts.get("by_priority", {}).get(AlertPriority.CRITICAL.value, 0))
with m_col4:
    st.metric("Escalated to Cases", summary_counts.get("by_status", {}).get(AlertStatus.CASE_CREATED.value, 0))

st.write("")

if not alerts:
    st.info("No alerts match the selected filter criteria. Try adjusting filters or search term.")
else:
    # Build display table
    table_data = []
    for a in alerts:
        table_data.append({
            "Alert ID": a["alert_id"],
            "Priority": a["priority"],
            "P(Fraud)": round(a["fraud_probability"], 4),
            "Anomaly Score": round(a["anomaly_score"], 4),
            "Risk Tier": a["risk_category"],
            "Operational Quadrant": a["operational_quadrant"].replace("_", " "),
            "Amount (€)": f"€{a['amount']:.2f}",
            "Status": a["status"],
            "Case ID": a["case_id"] or "—",
            "Created": a["created_at"][:16].replace("T", " ") if a.get("created_at") else "—",
        })

    df_alerts = pd.DataFrame(table_data)

    st.dataframe(
        df_alerts,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # =============================================================================
    # ALERT ACTION & INSPECTION DRAWER
    # =============================================================================
    st.markdown("### 🔍 **Inspect & Triage Selected Alert**")

    alert_ids = [a["alert_id"] for a in alerts]
    selected_alert_id = st.selectbox(
        "Select Alert to Inspect or Escalate:",
        alert_ids,
        format_func=lambda aid: f"{aid} | Priority: {[a['priority'] for a in alerts if a['alert_id'] == aid][0]} | P(Fraud): {[a['fraud_probability'] for a in alerts if a['alert_id'] == aid][0]:.3f}",
    )

    selected_alert = next((a for a in alerts if a["alert_id"] == selected_alert_id), None)

    if selected_alert:
        det_col1, det_col2 = st.columns([3, 2])

        with det_col1:
            st.markdown(
                f"""
                <div class="snapshot-box">
                    <div class="snapshot-header">🛡️ Alert Overview: {selected_alert['alert_id']}</div>
                    <p><b>Transaction ID:</b> <code>{selected_alert['transaction_id']}</code></p>
                    <p><b>Priority:</b> {get_priority_badge_html(selected_alert['priority'])} &nbsp;&nbsp; 
                       <b>Status:</b> {get_status_badge_html(selected_alert['status'])}</p>
                    <p><b>Fraud Probability:</b> <code>{selected_alert['fraud_probability']:.4f}</code> &nbsp;|&nbsp; 
                       <b>Anomaly Score:</b> <code>{selected_alert['anomaly_score']:.4f}</code></p>
                    <p><b>Operational Quadrant:</b> <code>{selected_alert['operational_quadrant']}</code></p>
                    <p><b>Trigger Policy Reasons:</b></p>
                    <ul>
                        {''.join(f'<li>{r}</li>' for r in selected_alert['reasons'])}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("📦 View Raw Transaction Feature Payload"):
                st.json(selected_alert["transaction_payload"])

        with det_col2:
            st.markdown("#### **Triage Actions**")

            if selected_alert["case_id"]:
                st.success(f"💼 Already escalated to Case: **`{selected_alert['case_id']}`**")
                st.markdown(
                    f"👉 *Navigate to the **Investigations** page to manage `{selected_alert['case_id']}`.*"
                )
            else:
                st.markdown("##### **Escalate to Investigation Case**")
                analyst_name = st.text_input("Assign Investigator:", value="analyst_mistry", key="assign_analyst")
                init_note = st.text_area(
                    "Initial Observation Note:",
                    value=f"Alert triggered via {', '.join(selected_alert['reasons'][:2])}.",
                    key="init_note",
                )

                if st.button("⚡ Open Investigation Case", type="primary", use_container_width=True):
                    with st.spinner("Escalating alert and capturing immutable prediction snapshot..."):
                        service = get_case_service()
                        with get_db_context() as db:
                            new_case = service.open_case_from_alert(
                                db=db,
                                alert_id=selected_alert["alert_id"],
                                assigned_analyst=analyst_name.strip() if analyst_name.strip() else None,
                                initial_note=init_note.strip() if init_note.strip() else None,
                                creator="analyst_ui",
                            )
                        st.success(f"Successfully opened Case **`{new_case.case_id}`**!")
                        st.rerun()

            st.write("")
            st.markdown("##### **Quick Status Transition**")
            btn_c1, btn_c2 = st.columns(2)
            with btn_c1:
                if st.button("Mark Escalated", use_container_width=True):
                    with get_db_context() as db:
                        AlertRepository.update_status(db, selected_alert["alert_id"], AlertStatus.ESCALATED.value)
                    st.success("Alert status updated to ESCALATED.")
                    st.rerun()
            with btn_c2:
                if st.button("Dismiss Alert", use_container_width=True):
                    with get_db_context() as db:
                        AlertRepository.update_status(db, selected_alert["alert_id"], AlertStatus.DISMISSED.value)
                    st.info("Alert dismissed from queue.")
                    st.rerun()
