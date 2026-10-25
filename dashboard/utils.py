"""Shared styling, helper functions, and data access utilities for the Streamlit Fraud Dashboard."""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import streamlit as st

from src.cases.database import get_db_context
from src.cases.models import (
    Alert,
    AlertPriority,
    AlertStatus,
    CaseAction,
    CaseAuditLog,
    CaseNote,
    CaseStatus,
    CaseVerdict,
    InvestigationCase,
)
from src.cases.repository import AlertRepository, CaseRepository
from src.cases.service import get_case_service


def inject_custom_styles() -> None:
    """Inject modern dark-theme enterprise CSS styling."""
    custom_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre, .stCodeBlock {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Gradient Headers */
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f43f5e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.0rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }

    /* Metric Cards */
    .metric-card-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.35);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .metric-card-container:hover {
        transform: translateY(-2px);
        border-color: rgba(96, 165, 250, 0.3);
    }

    .metric-card-title {
        color: #94a3b8;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    .metric-card-value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.1;
        margin-bottom: 0.35rem;
    }

    .metric-card-sub {
        font-size: 0.78rem;
        color: #64748b;
        font-weight: 500;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.22rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        white-space: nowrap;
    }

    .badge-critical {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(185, 28, 28, 0.3));
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }

    .badge-high {
        background: linear-gradient(135deg, rgba(249, 115, 22, 0.2), rgba(194, 65, 12, 0.3));
        color: #fdba74;
        border: 1px solid rgba(249, 115, 22, 0.4);
    }

    .badge-medium {
        background: linear-gradient(135deg, rgba(234, 179, 8, 0.2), rgba(161, 98, 7, 0.3));
        color: #fde047;
        border: 1px solid rgba(234, 179, 8, 0.4);
    }

    .badge-low {
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.2), rgba(14, 116, 144, 0.3));
        color: #67e8f9;
        border: 1px solid rgba(6, 182, 212, 0.4);
    }

    .badge-open {
        background: rgba(59, 130, 246, 0.15);
        color: #93c5fd;
        border: 1px solid rgba(59, 130, 246, 0.35);
    }

    .badge-investigating {
        background: rgba(168, 85, 247, 0.15);
        color: #d8b4fe;
        border: 1px solid rgba(168, 85, 247, 0.35);
    }

    .badge-confirmed {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.45);
    }

    .badge-fp {
        background: rgba(16, 185, 129, 0.15);
        color: #6ee7b7;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }

    .badge-dismissed {
        background: rgba(100, 116, 139, 0.2);
        color: #cbd5e1;
        border: 1px solid rgba(100, 116, 139, 0.35);
    }

    /* Snapshot Box */
    .snapshot-box {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(96, 165, 250, 0.25);
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
    }

    .snapshot-header {
        font-size: 0.85rem;
        font-weight: 700;
        color: #60a5fa;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Note Card */
    .note-bubble {
        background: rgba(30, 41, 59, 0.6);
        border-left: 3px solid #60a5fa;
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1rem;
        margin-bottom: 0.6rem;
    }

    .note-meta {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-bottom: 0.25rem;
        font-weight: 600;
    }

    .note-content {
        font-size: 0.9rem;
        color: #e2e8f0;
    }

    /* Audit item */
    .audit-item {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 6px;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.5rem;
        font-size: 0.82rem;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def render_metric_card(
    title: str,
    value: Any,
    subtitle: str = "",
    icon: str = "📊",
    border_color: Optional[str] = None,
) -> None:
    """Render an aesthetically pleasing glassmorphic metric card."""
    border_style = f"border-color: {border_color};" if border_color else ""
    html = f"""
    <div class="metric-card-container" style="{border_style}">
        <div class="metric-card-title">{icon} {title}</div>
        <div class="metric-card-value">{value}</div>
        <div class="metric-card-sub">{subtitle}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def get_priority_badge_html(priority: str) -> str:
    """Return HTML formatted priority pill badge."""
    p = str(priority).upper()
    css_class = f"badge-{p.lower()}" if p in ["CRITICAL", "HIGH", "MEDIUM", "LOW"] else "badge-low"
    return f'<span class="badge {css_class}">{p}</span>'


def get_status_badge_html(status: str) -> str:
    """Return HTML formatted status pill badge."""
    s = str(status).upper()
    if s in ["OPEN", "NEW"]:
        css_class = "badge-open"
    elif s in ["UNDER_INVESTIGATION", "PENDING_CUSTOMER", "ESCALATED"]:
        css_class = "badge-investigating"
    elif s in ["RESOLVED_CONFIRMED_FRAUD", "CONFIRMED_FRAUD"]:
        css_class = "badge-confirmed"
    elif s in ["RESOLVED_FALSE_POSITIVE", "FALSE_POSITIVE", "LEGITIMATE"]:
        css_class = "badge-fp"
    else:
        css_class = "badge-dismissed"
    return f'<span class="badge {css_class}">{s.replace("_", " ")}</span>'


@st.cache_data(ttl=600)
def load_holdout_samples(n_fraud: int = 5, n_legit: int = 5) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load sample holdout transactions from test_data.csv for simulation."""
    csv_path = Path("data/test_data.csv")
    if not csv_path.exists():
        return [], []

    df = pd.read_csv(csv_path)
    fraud_df = df[df["Class"] == 1].head(n_fraud)
    legit_df = df[df["Class"] == 0].head(n_legit)

    fraud_samples = [row.drop(labels=["Class"]).to_dict() for _, row in fraud_df.iterrows()]
    legit_samples = [row.drop(labels=["Class"]).to_dict() for _, row in legit_df.iterrows()]
    return fraud_samples, legit_samples


def seed_sample_database_data_if_needed(target_count: int = 6) -> int:
    """Seed sample alerts and cases if the database has fewer than target_count alerts."""
    with get_db_context() as db:
        current_count = db.query(Alert).count()
        if current_count >= target_count:
            return 0

    fraud_samples, legit_samples = load_holdout_samples(n_fraud=4, n_legit=3)
    if not fraud_samples and not legit_samples:
        return 0

    service = get_case_service()
    added = 0

    with get_db_context() as db:
        # Triage fraud samples (these will produce CRITICAL/HIGH alerts and auto-escalate)
        for i, sample in enumerate(fraud_samples):
            try:
                service.process_and_triage_transaction(
                    db=db,
                    transaction=sample,
                    threshold="default",
                    apply_preprocessing=False,  # test_data.csv is already preprocessed
                    auto_escalate=(i % 2 == 0),
                )
                added += 1
            except Exception as e:
                pass

        # Triage a couple legitimate samples
        for sample in legit_samples[:2]:
            try:
                service.process_and_triage_transaction(
                    db=db,
                    transaction=sample,
                    threshold="default",
                    apply_preprocessing=False,
                    auto_escalate=False,
                )
                added += 1
            except Exception as e:
                pass

    return added
