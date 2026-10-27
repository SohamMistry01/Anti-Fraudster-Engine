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
from src.agents.config import DEFAULT_AGENT_MODEL, SUPPORTED_MODELS
from src.agents.evaluator import evaluate_agent_on_cases
from src.agents.investigation_agent import (
    apply_investigator_decision,
    run_case_investigation,
)
from src.agents.schemas import InvestigatorApprovalRequest
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

st.markdown(
    """
    <style>
    .st-key-investigation_recommendations [data-testid="stMetricValue"],
    .st-key-investigation_recommendations [data-testid="stMetricValue"] > div,
    .st-key-prediction_snapshot_metrics [data-testid="stMetricValue"],
    .st-key-prediction_snapshot_metrics [data-testid="stMetricValue"] > div {
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        white-space: normal !important;
        word-break: break-word !important;
        line-height: 1.25 !important;
    }
    .st-key-investigation_recommendations [data-testid="stMetricLabel"],
    .st-key-prediction_snapshot_metrics [data-testid="stMetricLabel"] {
        font-size: 0.82rem !important;
    }
    .st-key-investigation_recommendations [data-testid="stMetricDelta"],
    .st-key-prediction_snapshot_metrics [data-testid="stMetricDelta"] {
        font-size: 0.78rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
# AGENTIC INVESTIGATION LAYER (LANGGRAPH + GROQ OPEN-SOURCE LLMs)
# =============================================================================
with st.container():
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, rgba(30, 27, 75, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
                    border: 1px solid rgba(139, 92, 246, 0.35); border-radius: 12px; padding: 1.3rem; margin-bottom: 1.5rem;
                    box-shadow: 0 4px 20px -2px rgba(124, 58, 237, 0.15);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.8rem;">
                <div style="font-size: 1.25rem; font-weight: 700; color: #e0e7ff; display: flex; align-items: center; gap: 0.6rem;">
                    <span>🤖</span> <span>Autonomous Agentic Investigator</span>
                    <span style="font-size: 0.72rem; background: rgba(139, 92, 246, 0.25); color: #c4b5fd; padding: 2px 8px; border-radius: 9999px; border: 1px solid rgba(139, 92, 246, 0.4);">
                        LangGraph 1.2.16 • Groq OSS LLMs
                    </span>
                </div>
                <div style="font-size: 0.82rem; color: #a5b4fc;">
                    Multi-signal synthesis • TreeSHAP & Isolation Forest evidence • Grounding verification
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ag_col1, ag_col2, ag_col3 = st.columns([3, 4, 3])
    with ag_col1:
        model_choice = st.selectbox(
            "Groq LLM Engine:",
            options=SUPPORTED_MODELS,
            index=0 if DEFAULT_AGENT_MODEL in SUPPORTED_MODELS else 0,
            key=f"agent_model_select_{case_dict['case_id']}",
        )
    with ag_col2:
        st.write("")
        st.write("")
        run_agent_clicked = st.button(
            "⚡ Generate AI Investigation Report",
            type="primary",
            use_container_width=True,
            key=f"btn_investigate_{case_dict['case_id']}",
        )
    with ag_col3:
        st.write("")
        st.write("")
        show_bench = st.checkbox("Show Evaluation Benchmark", value=False, key=f"toggle_bench_{case_dict['case_id']}")

    report_state_key = f"agent_report_{case_dict['case_id']}"
    if run_agent_clicked:
        with st.spinner(f"Orchestrating agent workflow with {model_choice}... (Gathering evidence → Synthesizing signals → Verifying factual grounding)"):
            try:
                agent_res = run_case_investigation(
                    case_id=case_dict["case_id"],
                    model_name=model_choice,
                    investigator_id=case_dict.get("assigned_analyst") or "lead_analyst",
                )
                st.session_state[report_state_key] = agent_res
                st.toast("Agent investigation completed and validated!", icon="✅")
            except Exception as exc:
                st.error(f"Investigation agent encountered an error: {exc}")

    agent_result = st.session_state.get(report_state_key)
    if agent_result and agent_result.get("report"):
        rep = agent_result["report"]
        v_passed = agent_result.get("validation_passed", True)
        v_errors = agent_result.get("validation_errors", [])

        with st.container():
            st.markdown(
                f"""
                <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 10px; padding: 1.2rem; margin-top: 0.5rem; margin-bottom: 1.2rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
                        <div>
                            <span style="font-size: 1.15rem; font-weight: 700; color: #f8fafc;">📋 Executive Investigation Summary</span>
                            &nbsp;&nbsp;
                            <span style="background: {'rgba(16, 185, 129, 0.2)' if v_passed else 'rgba(239, 68, 68, 0.2)'}; color: {'#34d399' if v_passed else '#f87171'}; border: 1px solid {'rgba(16, 185, 129, 0.4)' if v_passed else 'rgba(239, 68, 68, 0.4)'}; padding: 3px 10px; border-radius: 9999px; font-size: 0.8rem; font-weight: 600;">
                                {'✅ Factual Grounding Verified' if v_passed else '⚠️ Grounding Revision Needed'}
                            </span>
                        </div>
                        <div style="font-size: 0.85rem; color: #94a3b8;">
                            Engine: <code>{agent_result.get('model_used')}</code> &nbsp;|&nbsp;
                            Confidence: <b style="color: #38bdf8;">{int(rep.get('confidence_score', 0.85) * 100)}%</b>
                        </div>
                    </div>
                    <div style="margin-top: 0.8rem; font-size: 0.95rem; color: #e2e8f0; line-height: 1.6; background: rgba(30, 41, 59, 0.5); padding: 0.8rem; border-radius: 6px;">
                        {rep.get('executive_summary', 'No summary generated.')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.container(key="investigation_recommendations"):
                rec_c1, rec_c2, rec_c3 = st.columns([1, 1, 1])
                with rec_c1:
                    st.metric("Recommended Verdict", rep.get("recommended_verdict", "UNKNOWN"))
                with rec_c2:
                    st.metric("Recommended Mitigation", rep.get("recommended_action", "UNKNOWN"))
                with rec_c3:
                    st.metric("Grounding Audit Status", "PASSED (0 Hallucinations)" if v_passed else "FLAGGED")

            st.markdown("#### 🧠 **Synthesized Behavioral Findings**")
            st.markdown(
                f"<div style='font-size: 0.9rem; color: #cbd5e1; background: rgba(15, 23, 42, 0.4); padding: 0.8rem; border-radius: 6px; border-left: 3px solid #6366f1; margin-bottom: 1rem;'>"
                f"{rep.get('investigation_findings', 'No detailed findings.')}"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown("#### 🔍 **Supporting Evidence Citations (Factual Grounding)**")
            citations = rep.get("key_evidence_citations", [])
            if citations:
                cit_records = []
                for c in citations:
                    cit_records.append({
                        "Feature": c.get("feature"),
                        "Observed Value": round(c.get("observed_value"), 4) if isinstance(c.get("observed_value"), (int, float)) else c.get("observed_value"),
                        "SHAP Impact (Log-Odds)": round(c.get("shap_value"), 4) if isinstance(c.get("shap_value"), (int, float)) else c.get("shap_value"),
                        "Direction": c.get("direction"),
                        "Statistical Rationale": c.get("rationale"),
                    })
                st.dataframe(pd.DataFrame(cit_records), hide_index=True, use_container_width=True)

            if rep.get("anomaly_context") or rep.get("case_precedent"):
                ac_col1, ac_col2 = st.columns(2)
                with ac_col1:
                    st.markdown("**Anomaly Detection Context:**")
                    st.info(rep.get("anomaly_context", "N/A"))
                with ac_col2:
                    st.markdown("**Historical Case Precedent:**")
                    st.info(rep.get("case_precedent", "N/A"))

            # Human-in-the-Loop Approval Checkpoint
            st.markdown("---")
            st.markdown("### 🚦 **Investigator Approval Checkpoint**")
            st.markdown(
                "<div style='font-size: 0.82rem; color: #94a3b8; margin-bottom: 0.8rem;'>"
                "The agent recommends the verdict and action below based on verified ML evidence. "
                "Review and click <b>Approve & Apply</b> to commit these findings to the official case lifecycle."
                "</div>",
                unsafe_allow_html=True,
            )

            chk_col1, chk_col2 = st.columns([1, 1], gap="medium")
            with chk_col1:
                all_verdicts = [v.value for v in CaseVerdict]
                agent_v = rep.get("recommended_verdict")
                v_idx = all_verdicts.index(agent_v) if agent_v in all_verdicts else 0
                appr_verdict = st.selectbox("Approve or Adjust Verdict:", all_verdicts, index=v_idx, key=f"appr_v_{case_dict['case_id']}")

                all_actions = [a.value for a in CaseAction]
                agent_a = rep.get("recommended_action")
                a_idx = all_actions.index(agent_a) if agent_a in all_actions else 0
                appr_action = st.selectbox("Approve or Adjust Mitigation:", all_actions, index=a_idx, key=f"appr_a_{case_dict['case_id']}")

            with chk_col2:
                suggested_note = (
                    f"[Agent Recommendation Approved]\n"
                    f"Executive Summary: {rep.get('executive_summary')}\n"
                    f"Recommended Action: {appr_action}"
                )
                appr_notes = st.text_area(
                    "Investigator Sign-Off Comments:",
                    value=suggested_note,
                    key=f"appr_notes_{case_dict['case_id']}",
                    height=110,
                )

            appr_btn_col1, appr_btn_col2 = st.columns([2, 3])
            with appr_btn_col1:
                apply_approval = st.button(
                    "✅ Approve & Apply Recommendation to Case",
                    type="primary",
                    use_container_width=True,
                    key=f"btn_apply_appr_{case_dict['case_id']}",
                )

                if apply_approval:
                    with st.spinner("Committing investigator decision and updating case lifecycle..."):
                        req = InvestigatorApprovalRequest(
                            case_id=case_dict["case_id"],
                            investigator_id=case_dict.get("assigned_analyst") or "lead_analyst",
                            action_approved=True,
                            final_verdict=appr_verdict,
                            final_action=appr_action,
                            analyst_comments=appr_notes,
                        )
                        apply_investigator_decision(case_id=case_dict["case_id"], decision=req)
                    st.success("Investigator decision committed successfully!")
                    st.rerun()

            with st.expander("🔬 View Raw Agent Structured Payload & State"):
                st.json(agent_result)

    # Benchmark Suite Expander
    if show_bench:
        st.markdown("---")
        st.markdown("### 🧪 **Agent Factual Grounding & Citation Benchmark**")
        st.markdown(
            "<div style='font-size: 0.82rem; color: #94a3b8; margin-bottom: 0.8rem;'>"
            "Evaluates factual grounding against raw transaction vectors, citation precision against TreeSHAP values, "
            "hallucination-freedom on anonymized PCA features, and tool correctness across representative cases."
            "</div>",
            unsafe_allow_html=True,
        )

        b_c1, b_c2 = st.columns([3, 2])
        with b_c1:
            num_cases_eval = st.slider("Number of Cases to Benchmark:", min_value=1, max_value=5, value=2, key=f"bench_num_cases_{case_dict['case_id']}")
        with b_c2:
            st.write("")
            run_eval_clicked = st.button("🚀 Run Agent Evaluation Benchmark", use_container_width=True, key=f"btn_run_eval_{case_dict['case_id']}")

        if run_eval_clicked:
            with st.spinner("Benchmarking agent across representative cases..."):
                try:
                    bench_res = evaluate_agent_on_cases(limit=num_cases_eval, model_name=model_choice)
                    st.session_state["agent_benchmark_results"] = bench_res
                    st.toast("Evaluation benchmark completed!", icon="📊")
                except Exception as b_err:
                    st.error(f"Benchmark error: {b_err}")

        saved_bench = st.session_state.get("agent_benchmark_results")
        if saved_bench and "benchmark_summary" in saved_bench:
            bs = saved_bench["benchmark_summary"]
            bm1, bm2, bm3, bm4 = st.columns(4)
            with bm1:
                st.metric("Factual Grounding", f"{bs.get('factual_grounding_accuracy', 0.0) * 100:.1f}%")
            with bm2:
                st.metric("Citation Accuracy", f"{bs.get('evidence_citation_accuracy', 0.0) * 100:.1f}%")
            with bm3:
                st.metric("Hallucination-Free Rate", f"{bs.get('hallucination_freedom_rate', 0.0) * 100:.1f}%")
            with bm4:
                st.metric("Composite Agent Score", f"{bs.get('composite_agent_quality', 0.0) * 100:.1f}%")

            with st.expander("📊 Detailed Case-by-Case Benchmark Results"):
                st.json(saved_bench.get("case_evaluations", []))

    st.markdown("---")

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

    with st.container(key="prediction_snapshot_metrics"):
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
