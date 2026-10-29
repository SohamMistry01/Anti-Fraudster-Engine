"""Anti-Fraudster | Prioritised Alert Queue & Triage Desk.

Provides an operational workbench integrating the Fraud Triage & Prioritization Agent
for multi-signal queue ranking, workload-aware investigator assignment, grounded
explanations, and auditable human supervisor oversight.
"""

from datetime import datetime
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
from src.agents.config import DEFAULT_AGENT_MODEL, SUPPORTED_MODELS
from src.agents.schemas import TriageAction, TriagePriorityTier
from src.agents.tools import get_investigator_workloads
from src.agents.triage_agent import (
    apply_triage_decisions,
    compute_deterministic_triage,
    run_queue_triage,
)
from src.cases.database import get_db_context
from src.cases.models import Alert, AlertPriority, AlertStatus
from src.cases.repository import AlertRepository
from src.cases.service import get_case_service

st.set_page_config(
    page_title="Anti-Fraudster | Prioritised Alert Queue & Triage",
    page_icon="🚨",
    layout="wide",
)

inject_custom_styles()

# Custom Styling for Triage Desk
st.markdown(
    """
    <style>
    .triage-tier-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .tier-critical {
        background: rgba(239, 68, 68, 0.18);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.45);
    }
    .tier-high {
        background: rgba(249, 115, 22, 0.18);
        color: #fb923c;
        border: 1px solid rgba(249, 115, 22, 0.45);
    }
    .tier-medium {
        background: rgba(245, 158, 11, 0.18);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.45);
    }
    .tier-low {
        background: rgba(16, 185, 129, 0.18);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.45);
    }
    .action-badge {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.76rem;
        font-weight: 600;
        background: rgba(99, 102, 241, 0.18);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.35);
    }
    .score-meter {
        font-weight: 800;
        font-size: 1.15rem;
        color: #38bdf8;
    }
    .triage-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.75) 100%);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 10px;
        padding: 1.1rem;
        margin-bottom: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def get_tier_badge(tier: str) -> str:
    """Return colored HTML badge for operational priority tier."""
    t_clean = (tier or "").upper()
    if "CRITICAL" in t_clean or "TIER_1" in t_clean:
        return f'<span class="triage-tier-badge tier-critical">🔴 Tier 1 Critical</span>'
    elif "HIGH" in t_clean or "TIER_2" in t_clean:
        return f'<span class="triage-tier-badge tier-high">🟠 Tier 2 High</span>'
    elif "MEDIUM" in t_clean or "TIER_3" in t_clean:
        return f'<span class="triage-tier-badge tier-medium">🟡 Tier 3 Medium</span>'
    else:
        return f'<span class="triage-tier-badge tier-low">🟢 Tier 4 Low</span>'


st.markdown('<div class="hero-title">🚨 Prioritised Alert Queue & Triage Desk</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">'
    'Orchestrates multi-signal deterministic prioritization, AI queue triage explanations, '
    'workload-aware investigator assignment, and auditable supervisor checkpoints.'
    '</div>',
    unsafe_allow_html=True,
)

# Fetch quick summary metrics for top header
with get_db_context() as db:
    summary_counts = AlertRepository.get_summary_counts(db)

m_col1, m_col2, m_col3, m_col4 = st.columns(4)
with m_col1:
    st.metric("Total in Queue", summary_counts.get("total_alerts", 0))
with m_col2:
    st.metric("New (Pending Triage)", summary_counts.get("by_status", {}).get(AlertStatus.NEW.value, 0))
with m_col3:
    st.metric("Critical Urgency", summary_counts.get("by_priority", {}).get(AlertPriority.CRITICAL.value, 0))
with m_col4:
    st.metric("Escalated to Cases", summary_counts.get("by_status", {}).get(AlertStatus.CASE_CREATED.value, 0))

st.write("")

# Main Interface Tabs
tab_triage_agent, tab_manual_queue = st.tabs([
    "⚡ Fraud Triage & Prioritization Agent",
    "📋 Alert Queue Explorer & Manual Triage",
])

# =============================================================================
# TAB 1: FRAUD TRIAGE & PRIORITIZATION AGENT
# =============================================================================
with tab_triage_agent:
    st.markdown("### 🤖 **Agentic Queue Prioritization & Workload Matching**")
    st.caption(
        "Evaluates the active queue using deterministic multi-signal scoring, generates concise "
        "evidence-grounded triage rationales via Groq LLM, and proposes workload-balanced investigator assignments."
    )

    # 1. Investigator Roster Ribbon
    with st.container(border=True):
        st.markdown("##### 👥 **Live Fraud Investigator Workload & Capacity**")
        investigators_live = get_investigator_workloads.invoke({})
        inv_cols = st.columns(len(investigators_live))

        for idx, inv in enumerate(investigators_live):
            with inv_cols[idx]:
                active = inv["active_cases"]
                cap = inv["max_capacity"]
                pct = int(inv["capacity_utilization"] * 100)
                status_color = "#34d399" if pct < 50 else "#fbbf24" if pct < 80 else "#f87171"

                st.markdown(
                    f"""
                    <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(148, 163, 184, 0.15); border-radius: 8px; padding: 0.8rem;">
                        <div style="font-weight: 700; color: #f8fafc; font-size: 0.95rem;">{inv['name']}</div>
                        <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 0.4rem;">{inv['role']}</div>
                        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.3rem;">
                            <span style="color: #cbd5e1;">Load: <b>{active}/{cap}</b> cases</span>
                            <span style="color: {status_color}; font-weight: 700;">{pct}%</span>
                        </div>
                        <div style="background: rgba(51, 65, 85, 0.6); height: 6px; border-radius: 3px; overflow: hidden;">
                            <div style="width: {min(pct, 100)}%; height: 100%; background: {status_color};"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.write("")

    # 2. Agent Execution Controls
    with st.container(border=True):
        st.markdown("##### ⚙️ **Triage Agent Run Parameters**")
        p_c1, p_c2, p_c3, p_c4 = st.columns([3, 2, 2, 2])

        with p_c1:
            agent_model = st.selectbox(
                "LLM Reasoning Engine",
                SUPPORTED_MODELS,
                index=0,
                key="triage_model_select",
            )
        with p_c2:
            queue_status_filter = st.selectbox(
                "Alerts Status Filter",
                ["NEW", "ALL", "ESCALATED"],
                index=0,
                key="triage_status_filter",
            )
        with p_c3:
            max_alerts_input = st.slider(
                "Max Alerts to Prioritize",
                min_value=3,
                max_value=25,
                value=10,
                step=1,
                key="triage_max_alerts",
            )
        with p_c4:
            workload_toggle = st.toggle(
                "Workload Assignment",
                value=True,
                help="Dynamically recommend investigators based on live capacity and tier expertise.",
                key="triage_workload_toggle",
            )

        run_triage_btn = st.button(
            "⚡ Run Fraud Triage & Prioritization Agent",
            type="primary",
            width="stretch",
            key="btn_run_queue_triage",
        )

    # Agent Execution Trigger
    if run_triage_btn:
        with st.spinner(f"Orchestrating Fraud Triage Agent ({agent_model})... Gathering queue & workloads → Computing deterministic priorities → Generating grounded explanations → Verifying integrity"):
            try:
                triage_res = run_queue_triage(
                    filter_status=queue_status_filter,
                    max_alerts=max_alerts_input,
                    include_workload=workload_toggle,
                    model_name=agent_model,
                )
                st.session_state["triage_batch_results"] = triage_res
                st.toast("Fraud triage queue successfully evaluated and grounded!", icon="✅")
            except Exception as exc:
                st.error(f"Triage agent run failed: {exc}")

    triage_data = st.session_state.get("triage_batch_results")

    if triage_data and triage_data.get("prioritized_queue"):
        overview = triage_data.get("queue_overview") or {}
        prioritized = triage_data.get("prioritized_queue", [])
        v_passed = triage_data.get("validation_passed", True)

        st.write("")
        st.divider()

        # 3. Macro Queue Overview Header
        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(99, 102, 241, 0.35); border-radius: 12px; padding: 1.25rem; margin-bottom: 1.2rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.8rem;">
                    <div>
                        <span style="font-size: 1.2rem; font-weight: 800; color: #f8fafc;">📊 Executive Queue Synthesis</span>
                        &nbsp;&nbsp;
                        <span style="background: {'rgba(16, 185, 129, 0.2)' if v_passed else 'rgba(239, 68, 68, 0.2)'}; color: {'#34d399' if v_passed else '#f87171'}; border: 1px solid {'rgba(16, 185, 129, 0.4)' if v_passed else 'rgba(239, 68, 68, 0.4)'}; padding: 3px 10px; border-radius: 9999px; font-size: 0.8rem; font-weight: 600;">
                            {'✅ Factual Grounding Verified' if v_passed else '⚠️ Grounding Revision Triggered'}
                        </span>
                    </div>
                    <div style="font-size: 0.85rem; color: #94a3b8;">
                        Batch: <code>{triage_data.get('batch_id')}</code> &nbsp;|&nbsp;
                        Engine: <code>{triage_data.get('model_used')}</code> &nbsp;|&nbsp;
                        Triaged: <b>{len(prioritized)} alerts</b>
                    </div>
                </div>
                <div style="margin-top: 0.8rem; font-size: 0.95rem; color: #e2e8f0; line-height: 1.6; background: rgba(30, 41, 59, 0.5); padding: 0.85rem; border-radius: 6px;">
                    {overview.get('queue_risk_synthesis', 'Queue synthesis completed.')}
                </div>
                {f'<div style="margin-top: 0.5rem; font-size: 0.85rem; color: #94a3b8;"><i style="color: #38bdf8;">Workload Note:</i> {overview.get("workload_bottlenecks")}</div>' if overview.get('workload_bottlenecks') else ''}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 4. Human Oversight Checkpoint Form
        st.markdown("#### 🛡️ **Supervisor Review & Proposed Triage Decisions**")
        st.caption("Review deterministic rankings and evidence-grounded rationales. Override assignments or actions as needed before applying.")

        supervisor_id_input = st.text_input(
            "Supervisor Reviewer ID",
            value="lead_supervisor",
            key="supervisor_id_input",
        )

        available_investigators = [inv["investigator_id"] for inv in investigators_live]

        # Interactive Decision Matrix
        user_decisions: List[Dict[str, Any]] = []

        for idx, item in enumerate(prioritized):
            aid = item["alert_id"]
            rank = item["rank"]
            tier = item["priority_tier"]
            score = item["composite_urgency_score"]
            p_fraud = item["fraud_probability"]
            anom = item["anomaly_score"]
            amt = item["amount"]
            rec_action = item["recommended_action"]
            rec_inv = item.get("recommended_investigator")
            explanation = item.get("triage_explanation", "")
            rationale = item.get("assignment_rationale", "")
            has_case = item.get("case_id")

            with st.container(border=True):
                # Header row: Rank, Alert ID, Tier, Scores
                c_top1, c_top2, c_top3, c_top4, c_top5 = st.columns([1.5, 2.5, 2.5, 2.5, 3])

                with c_top1:
                    st.markdown(f"<div style='font-size: 1.3rem; font-weight: 800; color: #60a5fa;'>#{rank}</div>", unsafe_allow_html=True)
                with c_top2:
                    st.markdown(f"**`{aid}`**")
                    if has_case:
                        st.caption(f"Linked: `{has_case}`")
                    else:
                        st.caption("No case yet")
                with c_top3:
                    st.markdown(get_tier_badge(tier), unsafe_allow_html=True)
                with c_top4:
                    st.markdown(f"<span class='score-meter'>{score:.1f}</span> <small style='color: #64748b;'>/ 100</small>", unsafe_allow_html=True)
                    st.caption(f"P: **{p_fraud:.2%}** | Anom: **{anom:.2f}**")
                with c_top5:
                    st.markdown(f"**€{amt:,.2f}** exposure")
                    st.markdown(f"<span class='action-badge'>{rec_action}</span>", unsafe_allow_html=True)

                # Grounded explanation
                st.markdown(
                    f"""
                    <div style="background: rgba(15, 23, 42, 0.45); border-left: 3px solid #38bdf8; padding: 0.55rem 0.8rem; margin: 0.5rem 0; font-size: 0.88rem; color: #cbd5e1;">
                        💡 <b>Triage Rationale:</b> {explanation}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if rec_inv and rationale:
                    st.markdown(
                        f"""
                        <div style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 0.6rem;">
                            👤 <b>Recommended Assignment:</b> <code>{rec_inv}</code> &nbsp;—&nbsp; <i>{rationale}</i>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Supervisor Review Controls
                act_col1, act_col2, act_col3 = st.columns([3, 3, 4])

                with act_col1:
                    action_choices = [
                        TriageAction.ESCALATE_TO_CASE.value,
                        TriageAction.PRIORITIZE_CASE_INVESTIGATION.value,
                        TriageAction.BATCH_REVIEW.value,
                        TriageAction.DISMISS_FALSE_POSITIVE.value,
                        TriageAction.MONITOR_ONLY.value,
                    ]
                    def_action_idx = action_choices.index(rec_action) if rec_action in action_choices else 0
                    sel_action = st.selectbox(
                        "Review Decision",
                        action_choices,
                        index=def_action_idx,
                        key=f"act_sel_{aid}",
                        label_visibility="collapsed",
                    )

                with act_col2:
                    inv_choices = available_investigators
                    def_inv_idx = inv_choices.index(rec_inv) if rec_inv in inv_choices else 0
                    sel_inv = st.selectbox(
                        "Assign Investigator",
                        inv_choices,
                        index=def_inv_idx,
                        key=f"inv_sel_{aid}",
                        label_visibility="collapsed",
                    )

                with act_col3:
                    sel_note = st.text_input(
                        "Supervisor Note",
                        value=f"Triage Rank #{rank} ({tier}) approved.",
                        key=f"note_input_{aid}",
                        label_visibility="collapsed",
                    )

                user_decisions.append({
                    "alert_id": aid,
                    "action": sel_action,
                    "assigned_investigator": sel_inv,
                    "supervisor_notes": sel_note,
                })

        st.write("")

        # Batch Application Button
        b_c1, b_c2 = st.columns([3, 1])
        with b_c1:
            st.info(f"Ready to apply **{len(user_decisions)} triage decisions** to the active queue and case repository.")
        with b_c2:
            apply_all_btn = st.button(
                "✅ Approve & Apply Triage Decisions",
                type="primary",
                width="stretch",
                key="btn_apply_all_triage",
            )

        if apply_all_btn:
            with st.spinner("Executing supervisor triage decisions in database (Escalating cases, assigning analysts, updating queue)..."):
                try:
                    res = apply_triage_decisions(
                        decisions=user_decisions,
                        supervisor_id=supervisor_id_input.strip() or "fraud_supervisor",
                    )
                    st.success(
                        f"Successfully applied {res['applied_count']} triage actions! "
                        f"Opened {len(res['cases_created'])} new investigation cases."
                    )
                    # Clear session state and refresh
                    if "triage_batch_results" in st.session_state:
                        del st.session_state["triage_batch_results"]
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to apply decisions: {exc}")

    elif triage_data and not triage_data.get("prioritized_queue"):
        st.info("Queue is empty for the selected status filter. No pending alerts found.")


# =============================================================================
# TAB 2: MANUAL ALERT QUEUE EXPLORER & INSPECTION
# =============================================================================
with tab_manual_queue:
    st.markdown("### 📋 **Alert Queue Explorer**")

    # Filters & Search Control Bar
    with st.container():
        f_col1, f_col2, f_col3, f_col4 = st.columns([2, 2, 3, 3])

        with f_col1:
            status_filter = st.selectbox(
                "Lifecycle Status",
                ["ALL", AlertStatus.NEW.value, AlertStatus.ESCALATED.value, AlertStatus.CASE_CREATED.value, AlertStatus.DISMISSED.value],
                index=0,
                key="manual_status_filter",
            )

        with f_col2:
            priority_filter = st.selectbox(
                "Urgency Priority",
                ["ALL", AlertPriority.CRITICAL.value, AlertPriority.HIGH.value, AlertPriority.MEDIUM.value, AlertPriority.LOW.value],
                index=0,
                key="manual_priority_filter",
            )

        with f_col3:
            min_p = st.slider("Min Fraud Probability P(Fraud)", 0.0, 1.0, 0.0, step=0.05, key="manual_min_p")

        with f_col4:
            search_query = st.text_input("🔍 Search Alert / Transaction / Case ID", placeholder="e.g. ALT- or CASE-", key="manual_search")

    # Fetch Alerts from Database
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
            width="stretch",
            hide_index=True,
        )

        st.divider()

        # Alert Action & Inspection Drawer
        st.markdown("### 🔍 **Inspect & Triage Selected Alert**")

        alert_ids = [a["alert_id"] for a in alerts]
        selected_alert_id = st.selectbox(
            "Select Alert to Inspect or Escalate:",
            alert_ids,
            format_func=lambda aid: f"{aid} | Priority: {[a['priority'] for a in alerts if a['alert_id'] == aid][0]} | P(Fraud): {[a['fraud_probability'] for a in alerts if a['alert_id'] == aid][0]:.3f}",
            key="inspect_alert_select",
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
                    analyst_name = st.text_input("Assign Investigator:", value="analyst_mistry", key="assign_analyst_manual")
                    init_note = st.text_area(
                        "Initial Observation Note:",
                        value=f"Alert triggered via {', '.join(selected_alert['reasons'][:2])}.",
                        key="init_note_manual",
                    )

                    if st.button("⚡ Open Investigation Case", type="primary", width="stretch", key="btn_open_case_manual"):
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
                    if st.button("Mark Escalated", width="stretch", key="btn_quick_escalate"):
                        with get_db_context() as db:
                            AlertRepository.update_status(db, selected_alert["alert_id"], AlertStatus.ESCALATED.value)
                        st.success("Alert status updated to ESCALATED.")
                        st.rerun()
                with btn_c2:
                    if st.button("Dismiss Alert", width="stretch", key="btn_quick_dismiss"):
                        with get_db_context() as db:
                            AlertRepository.update_status(db, selected_alert["alert_id"], AlertStatus.DISMISSED.value)
                        st.info("Alert dismissed from queue.")
                        st.rerun()
