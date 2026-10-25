"""Anti-Fraudster | Investigation Workbench & Auditable Case Lifecycle.

Provides an auditable workspace for fraud investigators to review immutable prediction
evidence, update case statuses, record verdicts and mitigation actions, add analyst notes,
and inspect full historical audit trails.
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
from src.cases.models import (
    AlertPriority,
    CaseAction,
    CaseAuditLog,
    CaseNote,
    CaseStatus,
    CaseVerdict,
    InvestigationCase,
)
from src.cases.repository import CaseRepository
from src.cases.service import get_case_service

st.set_page_config(
    page_title="Anti-Fraudster | Investigation Workbench",
    page_icon="💼",
    layout="wide",
)

inject_custom_styles()

st.markdown('<div class="hero-title">💼 Investigation Case Workbench</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">'
    'Strictly preserves immutable machine learning prediction snapshots separately from investigator verdicts, actions, and notes.'
    '</div>',
    unsafe_allow_html=True,
)

# =============================================================================
# CASE FILTERS & SELECTOR
# =============================================================================
with st.container():
    c_col1, c_col2, c_col3 = st.columns([3, 3, 4])

    with c_col1:
        status_filter = st.selectbox(
            "Filter by Case Status",
            [
                "ALL",
                CaseStatus.OPEN.value,
                CaseStatus.UNDER_INVESTIGATION.value,
                CaseStatus.PENDING_CUSTOMER.value,
                CaseStatus.RESOLVED_CONFIRMED_FRAUD.value,
                CaseStatus.RESOLVED_FALSE_POSITIVE.value,
                CaseStatus.CLOSED_DISMISSED.value,
            ],
            index=0,
        )

    with c_col2:
        priority_filter = st.selectbox(
            "Filter by Priority",
            ["ALL", AlertPriority.CRITICAL.value, AlertPriority.HIGH.value, AlertPriority.MEDIUM.value, AlertPriority.LOW.value],
            index=0,
        )

    with c_col3:
        search_kw = st.text_input("🔍 Search Case / Alert / Analyst", placeholder="e.g. CASE- or analyst_")

with get_db_context() as db:
    s_val = None if status_filter == "ALL" else status_filter
    p_val = None if priority_filter == "ALL" else priority_filter
    q_val = search_kw.strip() if search_kw.strip() else None

    cases_orm, total_cases = CaseRepository.list_cases(
        db=db,
        status=s_val,
        priority=p_val,
        search=q_val,
        limit=100,
    )
    cases = [c.to_dict(include_relations=False) for c in cases_orm]

if not cases:
    st.warning("No investigation cases found matching the current filters.")
    st.info("Tip: Head over to the **Alerts** page to escalate high-risk alerts into new cases, or use the Simulator on the Home page.")
    st.stop()

# Case Selector
case_options = [c["case_id"] for c in cases]
selected_case_id = st.selectbox(
    "Select Case to Investigate:",
    case_options,
    format_func=lambda cid: f"{cid} | Status: {[c['status'] for c in cases if c['case_id'] == cid][0]} | Priority: {[c['priority'] for c in cases if c['case_id'] == cid][0]} | Analyst: {[c['assigned_analyst'] or 'Unassigned' for c in cases if c['case_id'] == cid][0]}",
)

# Fetch full case entity including notes and audit logs
with get_db_context() as db:
    case_obj = CaseRepository.get_by_id(db, selected_case_id)
    if not case_obj:
        st.error(f"Case {selected_case_id} not found.")
        st.stop()

    case_dict = case_obj.to_dict(include_relations=True)

# =============================================================================
# CASE HEADER BANNER
# =============================================================================
status_badge = get_status_badge_html(case_dict["status"])
priority_badge = get_priority_badge_html(case_dict["priority"])

st.markdown(
    f"""
    <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(148, 163, 184, 0.15); border-radius: 10px; padding: 1.2rem; margin-bottom: 1.5rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.8rem;">
            <div>
                <span style="font-size: 1.4rem; font-weight: 800; color: #f8fafc;">Case {case_dict['case_id']}</span>
                &nbsp;&nbsp; {status_badge} &nbsp;&nbsp; {priority_badge}
            </div>
            <div style="font-size: 0.85rem; color: #94a3b8;">
                <b>Created:</b> {case_dict['created_at'][:19] if case_dict['created_at'] else '—'} &nbsp;|&nbsp;
                <b>Last Updated:</b> {case_dict['updated_at'][:19] if case_dict['updated_at'] else '—'}
            </div>
        </div>
        <div style="margin-top: 0.8rem; font-size: 0.9rem; color: #cbd5e1; display: flex; gap: 2rem; flex-wrap: wrap;">
            <div><b>Alert Reference:</b> <code>{case_dict['alert_id'] or 'Manual Direct'}</code></div>
            <div><b>Transaction ID:</b> <code>{case_dict['transaction_id']}</code></div>
            <div><b>Assigned Analyst:</b> <code>{case_dict['assigned_analyst'] or 'Unassigned'}</code></div>
            <div><b>Verdict:</b> <code>{case_dict['verdict'] or 'Pending'}</code></div>
            <div><b>Action Taken:</b> <code>{case_dict['action_taken'] or 'None'}</code></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# TWO-PANE WORKSPACE:
# LEFT: IMMUTABLE PREDICTION SNAPSHOT
# RIGHT: INVESTIGATOR WORKFLOW & LIFECYCLE CONTROLS
# =============================================================================
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown("### 🔒 **Immutable Prediction Snapshot**")
    st.markdown(
        "<div style='font-size: 0.8rem; color: #60a5fa; margin-bottom: 0.8rem;'>"
        "SEALED INFERENCE BASELINE • Preserved strictly separately from analyst outcomes"
        "</div>",
        unsafe_allow_html=True,
    )

    orig = case_dict.get("original_prediction", {})

    snap_c1, snap_c2 = st.columns(2)
    with snap_c1:
        st.metric("Original P(Fraud)", f"{orig.get('fraud_probability', 0.0):.4f}")
        st.metric("Risk Category", orig.get("risk_category", "UNKNOWN"))
        st.metric("Model Version", orig.get("model_version", "1.0.0"))
    with snap_c2:
        st.metric("Original Anomaly Score", f"{orig.get('anomaly_score', 0.0):.4f}")
        st.metric("Anomaly Tier", orig.get("anomaly_tier", "UNKNOWN"))
        st.metric("Threshold Used", f"{orig.get('threshold_used', 0.5):.3f} ({orig.get('threshold_name', 'default')})")

    st.markdown(f"**Operational Quadrant:** `{orig.get('operational_quadrant', 'UNKNOWN')}`")

    # SHAP Evidence & Top Contributors
    shap_exp = orig.get("shap_explanation", {})
    if shap_exp:
        st.markdown("#### **SHAP Explainability Evidence**")
        st.info(shap_exp.get("evidence_summary", "No textual evidence recorded."))

        pos_contributors = shap_exp.get("top_positive_contributors", [])
        neg_contributors = shap_exp.get("top_negative_contributors", [])

        if pos_contributors:
            st.markdown("**Top Fraud-Pushing Features (Positive Log-Odds Impact):**")
            pos_df = pd.DataFrame(pos_contributors)
            st.dataframe(pos_df, hide_index=True, use_container_width=True)

        if neg_contributors:
            st.markdown("**Top Legitimate-Pushing Features (Negative Log-Odds Impact):**")
            neg_df = pd.DataFrame(neg_contributors)
            st.dataframe(neg_df, hide_index=True, use_container_width=True)

    with st.expander("📦 Original Transaction Snapshot Vector"):
        st.json(case_dict.get("transaction_snapshot", {}))

with right_col:
    st.markdown("### ✍️ **Investigator Workflow & Decision**")
    st.markdown(
        "<div style='font-size: 0.8rem; color: #a855f7; margin-bottom: 0.8rem;'>"
        "AUDITABLE LIFECYCLE MANAGEMENT • All updates generate immutable audit logs"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.form("case_workflow_form"):
        # Status Dropdown
        status_options = [s.value for s in CaseStatus]
        curr_status_idx = status_options.index(case_dict["status"]) if case_dict["status"] in status_options else 0
        new_status = st.selectbox("Update Case Lifecycle Stage:", status_options, index=curr_status_idx)

        # Analyst Assignment
        new_analyst = st.text_input("Assigned Analyst:", value=case_dict.get("assigned_analyst") or "analyst_mistry")

        # Verdict Dropdown
        verdict_options = ["None"] + [v.value for v in CaseVerdict]
        curr_verdict_idx = (
            verdict_options.index(case_dict["verdict"])
            if case_dict.get("verdict") in verdict_options
            else 0
        )
        new_verdict = st.selectbox("Final Case Verdict:", verdict_options, index=curr_verdict_idx)

        # Mitigation Action Dropdown
        action_options = ["None"] + [a.value for a in CaseAction]
        curr_action_idx = (
            action_options.index(case_dict["action_taken"])
            if case_dict.get("action_taken") in action_options
            else 0
        )
        new_action = st.selectbox("Mitigation Action Executed:", action_options, index=curr_action_idx)

        # Resolution Notes
        new_resolution_notes = st.text_area(
            "Resolution Summary / Justification:",
            value=case_dict.get("resolution_notes") or "",
            placeholder="Document why this verdict was reached and any correspondence with cardholder...",
        )

        analyst_operator = st.text_input("Your Analyst ID (for audit log):", value="analyst_mistry")

        submit_workflow = st.form_submit_button("💾 Save Case Workflow Updates", type="primary", use_container_width=True)

        if submit_workflow:
            with st.spinner("Recording state transition and appending audit trail..."):
                service = get_case_service()
                with get_db_context() as db:
                    service.update_case_workflow(
                        db=db,
                        case_id=case_dict["case_id"],
                        status=new_status,
                        assigned_analyst=new_analyst.strip() if new_analyst.strip() else None,
                        verdict=None if new_verdict == "None" else new_verdict,
                        action_taken=None if new_action == "None" else new_action,
                        resolution_notes=new_resolution_notes.strip() if new_resolution_notes.strip() else None,
                        updated_by=analyst_operator.strip() or "analyst",
                    )
            st.success("Case updated successfully!")
            st.rerun()

st.divider()

# =============================================================================
# ANALYST NOTES & AUDIT TRAIL
# =============================================================================
notes_col, audit_col = st.columns([1, 1], gap="large")

with notes_col:
    st.markdown("### 📝 **Investigator Notes & Comments**")

    # Add Note Form
    with st.form("add_note_form", clear_on_submit=True):
        note_author = st.text_input("Author:", value="analyst_mistry")
        note_content = st.text_area("Add Investigation Finding / Observation:", placeholder="Enter details...")
        note_submitted = st.form_submit_button("➕ Add Note", use_container_width=True)

        if note_submitted and note_content.strip():
            service = get_case_service()
            with get_db_context() as db:
                service.add_note(
                    db=db,
                    case_id=case_dict["case_id"],
                    author=note_author.strip() or "analyst",
                    content=note_content.strip(),
                )
            st.success("Note added.")
            st.rerun()

    # Display Notes Chronologically
    notes_list = case_dict.get("notes", [])
    if notes_list:
        st.markdown(f"**Total Notes ({len(notes_list)}):**")
        for n in notes_list:
            created_str = n.get("created_at", "")[:19].replace("T", " ")
            st.markdown(
                f"""
                <div class="note-bubble">
                    <div class="note-meta">👤 {n.get('author', 'analyst')} • 🕒 {created_str} • <code>{n.get('note_id')}</code></div>
                    <div class="note-content">{n.get('content')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No investigator notes recorded for this case yet.")

with audit_col:
    st.markdown("### 🛡️ **Tamper-Evident Case Audit Log**")
    st.markdown(
        "<div style='font-size: 0.8rem; color: #94a3b8; margin-bottom: 0.8rem;'>"
        "Chronological audit trail of all state transitions, assignments, verdicts, and notes."
        "</div>",
        unsafe_allow_html=True,
    )

    audit_logs = case_dict.get("audit_logs", [])
    if audit_logs:
        for al in audit_logs:
            ts = al.get("timestamp", "")[:19].replace("T", " ")
            act = al.get("action", "")
            perf = al.get("performed_by", "system")
            details = al.get("details", {})

            # Format details nicely
            diff_lines = []
            for k, v in details.items():
                if isinstance(v, dict) and "old" in v and "new" in v:
                    diff_lines.append(f"<code>{k}</code>: <span style='color:#ef4444;'>{v['old']}</span> → <span style='color:#10b981;'>{v['new']}</span>")
                else:
                    diff_lines.append(f"<code>{k}</code>: {v}")
            diff_html = "<br>".join(diff_lines) if diff_lines else "<em>No details</em>"

            st.markdown(
                f"""
                <div class="audit-item">
                    <div style="display: flex; justify-content: space-between; font-weight: 700; color: #60a5fa;">
                        <span>⚙️ {act}</span>
                        <span style="font-weight: 400; color: #94a3b8;">🕒 {ts}</span>
                    </div>
                    <div style="font-size: 0.78rem; color: #cbd5e1; margin-top: 0.2rem;">By: <b>{perf}</b></div>
                    <div style="margin-top: 0.35rem; font-size: 0.8rem; color: #e2e8f0;">{diff_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No audit logs recorded.")
