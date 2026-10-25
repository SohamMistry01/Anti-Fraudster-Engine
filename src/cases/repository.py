"""SQLAlchemy repository layer providing database operations for Alerts, Cases, Notes, and Audit Logs."""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy import case, desc, func, or_
from sqlalchemy.orm import Session

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
    utc_now,
)

logger = logging.getLogger(__name__)


class AlertRepository:
    """Repository managing persistent CRUD operations and queue queries for alerts."""

    @staticmethod
    def create(db: Session, alert_data: Dict[str, Any]) -> Alert:
        """Create and commit a new alert record."""
        alert = Alert(**alert_data)
        db.add(alert)
        db.commit()
        db.refresh(alert)
        logger.info("Created Alert %s (Priority=%s)", alert.alert_id, alert.priority)
        return alert

    @staticmethod
    def get_by_id(db: Session, alert_id: str) -> Optional[Alert]:
        """Fetch alert by unique alert_id or primary key id."""
        return db.query(Alert).filter(
            or_(Alert.alert_id == alert_id, Alert.id == alert_id if alert_id.isdigit() else False)
        ).first()

    @staticmethod
    def list_alerts(
        db: Session,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        min_probability: Optional[float] = None,
        max_probability: Optional[float] = None,
        search: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[Alert], int]:
        """Query alert queue with filtering, priority sorting, and pagination."""
        query = db.query(Alert)

        if status:
            query = query.filter(Alert.status == status.upper())
        if priority:
            query = query.filter(Alert.priority == priority.upper())
        if min_probability is not None:
            query = query.filter(Alert.fraud_probability >= min_probability)
        if max_probability is not None:
            query = query.filter(Alert.fraud_probability <= max_probability)
        if search:
            s_term = f"%{search}%"
            query = query.filter(
                or_(
                    Alert.alert_id.ilike(s_term),
                    Alert.transaction_id.ilike(s_term),
                    Alert.case_id.ilike(s_term),
                )
            )

        total_count = query.count()

        # Custom priority ordering: CRITICAL -> HIGH -> MEDIUM -> LOW, then latest first
        priority_case = case(
            (Alert.priority == AlertPriority.CRITICAL.value, 1),
            (Alert.priority == AlertPriority.HIGH.value, 2),
            (Alert.priority == AlertPriority.MEDIUM.value, 3),
            (Alert.priority == AlertPriority.LOW.value, 4),
            else_=5,
        )

        items = query.order_by(
            priority_case,
            desc(Alert.fraud_probability),
            desc(Alert.created_at),
        ).offset(offset).limit(limit).all()

        return items, total_count

    @staticmethod
    def update_status(
        db: Session,
        alert_id: str,
        status: str,
        case_id: Optional[str] = None,
    ) -> Optional[Alert]:
        """Update status and case assignment link for an alert."""
        alert = AlertRepository.get_by_id(db, alert_id)
        if not alert:
            return None

        alert.status = status.upper()
        if case_id:
            alert.case_id = case_id
        db.commit()
        db.refresh(alert)
        return alert

    @staticmethod
    def get_summary_counts(db: Session) -> Dict[str, Any]:
        """Aggregate alert counts by status and priority for executive metrics."""
        status_counts = dict(
            db.query(Alert.status, func.count(Alert.id)).group_by(Alert.status).all()
        )
        priority_counts = dict(
            db.query(Alert.priority, func.count(Alert.id)).group_by(Alert.priority).all()
        )
        total = db.query(func.count(Alert.id)).scalar() or 0
        return {
            "total_alerts": total,
            "by_status": status_counts,
            "by_priority": priority_counts,
        }


class CaseRepository:
    """Repository managing investigation cases, analyst notes, and auditable history."""

    @staticmethod
    def create(
        db: Session,
        case_data: Dict[str, Any],
        initial_note: Optional[str] = None,
        creator: str = "system",
    ) -> InvestigationCase:
        """Create new case with immutable prediction snapshot and initial audit entry."""
        case = InvestigationCase(**case_data)
        db.add(case)
        db.flush()

        # Create initial audit log
        audit = CaseAuditLog(
            case_id=case.case_id,
            action="CASE_CREATED",
            performed_by=creator,
            details={
                "initial_status": case.status,
                "priority": case.priority,
                "fraud_probability": case.original_fraud_probability,
                "anomaly_score": case.original_anomaly_score,
                "alert_id": case.alert_id,
            },
        )
        db.add(audit)

        if initial_note:
            note = CaseNote(
                case_id=case.case_id,
                author=creator,
                content=initial_note,
            )
            db.add(note)

        db.commit()
        db.refresh(case)
        logger.info("Opened InvestigationCase %s (Priority=%s)", case.case_id, case.priority)
        return case

    @staticmethod
    def get_by_id(db: Session, case_id: str) -> Optional[InvestigationCase]:
        """Fetch investigation case by unique case_id."""
        return db.query(InvestigationCase).filter(
            or_(
                InvestigationCase.case_id == case_id,
                InvestigationCase.id == case_id if case_id.isdigit() else False,
            )
        ).first()

    @staticmethod
    def list_cases(
        db: Session,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        analyst: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[InvestigationCase], int]:
        """Query cases with filtering, priority sorting, and pagination."""
        query = db.query(InvestigationCase)

        if status:
            query = query.filter(InvestigationCase.status == status.upper())
        if priority:
            query = query.filter(InvestigationCase.priority == priority.upper())
        if analyst:
            query = query.filter(InvestigationCase.assigned_analyst == analyst)
        if search:
            s_term = f"%{search}%"
            query = query.filter(
                or_(
                    InvestigationCase.case_id.ilike(s_term),
                    InvestigationCase.alert_id.ilike(s_term),
                    InvestigationCase.transaction_id.ilike(s_term),
                    InvestigationCase.assigned_analyst.ilike(s_term),
                )
            )

        total_count = query.count()

        priority_case = case(
            (InvestigationCase.priority == AlertPriority.CRITICAL.value, 1),
            (InvestigationCase.priority == AlertPriority.HIGH.value, 2),
            (InvestigationCase.priority == AlertPriority.MEDIUM.value, 3),
            (InvestigationCase.priority == AlertPriority.LOW.value, 4),
            else_=5,
        )

        items = query.order_by(
            priority_case,
            desc(InvestigationCase.updated_at),
        ).offset(offset).limit(limit).all()

        return items, total_count

    @staticmethod
    def update_case(
        db: Session,
        case_id: str,
        status: Optional[str] = None,
        assigned_analyst: Optional[str] = None,
        verdict: Optional[str] = None,
        action_taken: Optional[str] = None,
        resolution_notes: Optional[str] = None,
        updated_by: str = "analyst",
    ) -> Optional[InvestigationCase]:
        """Update case workflow attributes and write an auditable state transition log."""
        case = CaseRepository.get_by_id(db, case_id)
        if not case:
            return None

        audit_details: Dict[str, Any] = {}

        # 1. Status Transition
        if status and status.upper() != case.status:
            old_s = case.status
            new_s = status.upper()
            case.status = new_s
            audit_details["status_change"] = {"from": old_s, "to": new_s}

            # If moving to resolved/closed, record resolved_at timestamp
            if "RESOLVED" in new_s or "CLOSED" in new_s:
                case.resolved_at = utc_now()
            else:
                case.resolved_at = None

        # 2. Analyst Assignment
        if assigned_analyst is not None and assigned_analyst != case.assigned_analyst:
            audit_details["analyst_assignment"] = {
                "from": case.assigned_analyst,
                "to": assigned_analyst,
            }
            case.assigned_analyst = assigned_analyst

        # 3. Verdict & Actions
        if verdict:
            v_val = verdict.upper()
            audit_details["verdict"] = {"from": case.verdict, "to": v_val}
            case.verdict = v_val

        if action_taken:
            a_val = action_taken.upper()
            audit_details["action_taken"] = {"from": case.action_taken, "to": a_val}
            case.action_taken = a_val

        if resolution_notes:
            case.resolution_notes = resolution_notes
            audit_details["resolution_notes_updated"] = True

        case.updated_at = utc_now()

        # Write audit log if meaningful modifications occurred
        if audit_details:
            audit = CaseAuditLog(
                case_id=case.case_id,
                action="CASE_UPDATED",
                performed_by=updated_by,
                details=audit_details,
            )
            db.add(audit)

        db.commit()
        db.refresh(case)
        logger.info("Updated Case %s: %s", case.case_id, audit_details)
        return case

    @staticmethod
    def add_note(
        db: Session,
        case_id: str,
        author: str,
        content: str,
    ) -> CaseNote:
        """Append investigator note and record an auditable NOTE_ADDED event."""
        case = CaseRepository.get_by_id(db, case_id)
        if not case:
            raise ValueError(f"InvestigationCase '{case_id}' does not exist.")

        note = CaseNote(
            case_id=case.case_id,
            author=author,
            content=content,
        )
        db.add(note)

        audit = CaseAuditLog(
            case_id=case.case_id,
            action="NOTE_ADDED",
            performed_by=author,
            details={"preview": content[:80] + ("..." if len(content) > 80 else "")},
        )
        db.add(audit)

        case.updated_at = utc_now()
        db.commit()
        db.refresh(note)
        return note

    @staticmethod
    def get_summary_metrics(db: Session) -> Dict[str, Any]:
        """Calculate executive metrics on cases, resolution rates, and verdicts."""
        total = db.query(func.count(InvestigationCase.id)).scalar() or 0
        open_cases = db.query(func.count(InvestigationCase.id)).filter(
            InvestigationCase.status.in_([
                CaseStatus.OPEN.value,
                CaseStatus.UNDER_INVESTIGATION.value,
                CaseStatus.PENDING_CUSTOMER.value,
            ])
        ).scalar() or 0

        confirmed_fraud = db.query(func.count(InvestigationCase.id)).filter(
            InvestigationCase.verdict == CaseVerdict.CONFIRMED_FRAUD.value
        ).scalar() or 0

        false_positives = db.query(func.count(InvestigationCase.id)).filter(
            InvestigationCase.verdict == CaseVerdict.FALSE_POSITIVE.value
        ).scalar() or 0

        status_counts = dict(
            db.query(InvestigationCase.status, func.count(InvestigationCase.id))
            .group_by(InvestigationCase.status)
            .all()
        )

        return {
            "total_cases": total,
            "open_cases": open_cases,
            "confirmed_fraud_count": confirmed_fraud,
            "false_positive_count": false_positives,
            "precision_rate": round(confirmed_fraud / max(confirmed_fraud + false_positives, 1), 4),
            "by_status": status_counts,
        }
