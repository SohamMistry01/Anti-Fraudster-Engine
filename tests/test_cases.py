"""Comprehensive test suite for Alert & Case Management, preserving immutable inference snapshots.

Verifies:
- Creation of prioritised alerts from risk policy
- Promotion of alerts to investigation cases
- Preservation of original inference prediction snapshot (separate from outcomes)
- Analyst workflow transitions (status, verdict, action_taken)
- Investigator notes and immutable audit log trails
"""

from pathlib import Path
import pandas as pd
import pytest
from sqlalchemy.orm import Session

from src.cases.database import get_db_context, SessionLocal
from src.cases.models import (
    Alert,
    AlertPriority,
    AlertStatus,
    CaseAction,
    CaseStatus,
    CaseVerdict,
    InvestigationCase,
)
from src.cases.repository import AlertRepository, CaseRepository
from src.cases.service import get_case_service
from src.risk.policy import evaluate_alert_policy, should_auto_escalate_case


@pytest.fixture(scope="session")
def sample_fraud_tx() -> dict:
    """Load sample fraud transaction from holdout test dataset."""
    csv_path = Path("data/test_data.csv")
    if not csv_path.exists():
        pytest.skip("data/test_data.csv not available")
    df = pd.read_csv(csv_path)
    fraud_row = df[df["Class"] == 1].iloc[0].drop(labels=["Class"]).to_dict()
    return fraud_row


@pytest.fixture(scope="session")
def sample_legit_tx() -> dict:
    """Load sample legitimate transaction from holdout test dataset."""
    csv_path = Path("data/test_data.csv")
    if not csv_path.exists():
        pytest.skip("data/test_data.csv not available")
    df = pd.read_csv(csv_path)
    legit_row = df[df["Class"] == 0].iloc[0].drop(labels=["Class"]).to_dict()
    return legit_row


def test_alert_repository_crud():
    """Verify Alert creation, querying, priority ordering, and status updates."""
    with get_db_context() as db:
        payload = {
            "amount": 199.99,
            "time_offset": 3600.0,
            "fraud_probability": 0.88,
            "anomaly_score": 0.72,
            "risk_category": "CRITICAL",
            "priority": AlertPriority.CRITICAL.value,
            "operational_quadrant": "CONFIRMED_FRAUD",
            "status": AlertStatus.NEW.value,
            "reasons": ["Supervised fraud probability 0.88 exceeds threshold"],
            "transaction_payload": {"Amount": 199.99, "Time": 3600.0},
        }
        alert = AlertRepository.create(db, payload)
        assert alert.alert_id.startswith("ALT-")
        assert alert.priority == "CRITICAL"
        assert alert.status == "NEW"

        # Query by ID
        fetched = AlertRepository.get_by_id(db, alert.alert_id)
        assert fetched is not None
        assert fetched.alert_id == alert.alert_id

        # Update status
        updated = AlertRepository.update_status(db, alert.alert_id, AlertStatus.ESCALATED.value)
        assert updated.status == AlertStatus.ESCALATED.value

        # List alerts with priority filter
        items, count = AlertRepository.list_alerts(db, priority=AlertPriority.CRITICAL.value)
        assert count >= 1
        assert any(i.alert_id == alert.alert_id for i in items)


def test_triage_transaction_auto_escalation(sample_fraud_tx: dict):
    """Verify that a high-risk fraud transaction generates an alert and auto-escalates to a case."""
    service = get_case_service()

    with get_db_context() as db:
        result = service.process_and_triage_transaction(
            db=db,
            transaction=sample_fraud_tx,
            threshold="default",
            apply_preprocessing=False,
            auto_escalate=True,
        )

        assert result["alert_created"] is True
        alert = result["alert"]
        assert alert["priority"] in [AlertPriority.CRITICAL.value, AlertPriority.HIGH.value]
        assert alert["fraud_probability"] > 0.90

        assert result["case_created"] is True
        case = result["case"]
        assert case["case_id"].startswith("CASE-")
        assert case["alert_id"] == alert["alert_id"]
        assert case["status"] == CaseStatus.OPEN.value


def test_preserve_immutable_original_prediction_separately(sample_fraud_tx: dict):
    """Critical Requirement: Verify original prediction snapshot is stored and NOT mutated by investigator actions."""
    service = get_case_service()

    with get_db_context() as db:
        # Create an alert
        result = service.process_and_triage_transaction(
            db=db,
            transaction=sample_fraud_tx,
            threshold="cost_optimal",
            apply_preprocessing=False,
            auto_escalate=False,
        )
        alert_id = result["alert"]["alert_id"]

        # Escalate into case
        case = service.open_case_from_alert(
            db=db,
            alert_id=alert_id,
            assigned_analyst="analyst_test",
            initial_note="Opening test investigation.",
            creator="test_runner",
            apply_preprocessing=False,
        )

        case_id = case.case_id
        original_prob = case.original_fraud_probability
        original_risk = case.original_risk_category
        original_model_ver = case.original_model_version

        assert original_prob > 0.90
        assert original_risk in ["HIGH", "CRITICAL"]
        assert original_model_ver == "1.0.0"
        assert case.original_shap_explanation is not None
        assert "evidence_summary" in case.original_shap_explanation

        # Now simulate investigator taking actions and resolving the case
        updated_case = service.update_case_workflow(
            db=db,
            case_id=case_id,
            status=CaseStatus.RESOLVED_CONFIRMED_FRAUD.value,
            verdict=CaseVerdict.CONFIRMED_FRAUD.value,
            action_taken=CaseAction.CARD_BLOCKED.value,
            resolution_notes="Contacted cardholder; verified unapproved card activity.",
            updated_by="lead_investigator",
        )

        # Verify investigation outcomes updated
        assert updated_case.status == CaseStatus.RESOLVED_CONFIRMED_FRAUD.value
        assert updated_case.verdict == CaseVerdict.CONFIRMED_FRAUD.value
        assert updated_case.action_taken == CaseAction.CARD_BLOCKED.value
        assert updated_case.resolved_at is not None

        # CRITICAL ASSERTION: The original prediction snapshot is 100% UNCHANGED and preserved
        assert updated_case.original_fraud_probability == original_prob
        assert updated_case.original_risk_category == original_risk
        assert updated_case.original_model_version == original_model_ver


def test_case_notes_and_audit_trail(sample_fraud_tx: dict):
    """Verify analyst notes and tamper-evident audit logs."""
    service = get_case_service()

    with get_db_context() as db:
        result = service.process_and_triage_transaction(
            db=db,
            transaction=sample_fraud_tx,
            threshold="default",
            apply_preprocessing=False,
            auto_escalate=True,
        )
        case_id = result["case"]["case_id"]

        # Add notes
        service.add_note(db, case_id=case_id, author="analyst_1", content="First review: V14 anomaly detected.")
        service.add_note(db, case_id=case_id, author="analyst_2", content="Second review: Card placed on temporary hold.")

        # Re-fetch full case
        case = CaseRepository.get_by_id(db, case_id)
        assert len(case.notes) >= 2
        note_contents = [n.content for n in case.notes]
        assert any("First review" in c for c in note_contents)
        assert any("Second review" in c for c in note_contents)

        # Verify audit logs
        assert len(case.audit_logs) >= 1
        actions = [a.action for a in case.audit_logs]
        assert "CASE_CREATED" in actions
        assert "NOTE_ADDED" in actions
