"""SQLAlchemy database models for Alert and Case Management.

Maintains strict separation between the immutable initial inference prediction
and subsequent investigator actions, statuses, verdicts, and audit logs.
"""

from datetime import datetime, timezone
from enum import Enum
import uuid
from typing import Any, Dict, List, Optional
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from src.cases.database import Base


def utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


class AlertPriority(str, Enum):
    """Urgency level assigned to alerts based on financial risk and model certainty."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class AlertStatus(str, Enum):
    """Lifecycle state of an alert."""
    NEW = "NEW"
    ESCALATED = "ESCALATED"
    CASE_CREATED = "CASE_CREATED"
    DISMISSED = "DISMISSED"


class CaseStatus(str, Enum):
    """Investigation lifecycle stages for fraud cases."""
    OPEN = "OPEN"
    UNDER_INVESTIGATION = "UNDER_INVESTIGATION"
    PENDING_CUSTOMER = "PENDING_CUSTOMER"
    RESOLVED_CONFIRMED_FRAUD = "RESOLVED_CONFIRMED_FRAUD"
    RESOLVED_FALSE_POSITIVE = "RESOLVED_FALSE_POSITIVE"
    CLOSED_DISMISSED = "CLOSED_DISMISSED"


class CaseVerdict(str, Enum):
    """Final analyst judgment on case outcome."""
    CONFIRMED_FRAUD = "CONFIRMED_FRAUD"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    SUSPICIOUS_UNVERIFIED = "SUSPICIOUS_UNVERIFIED"
    LEGITIMATE = "LEGITIMATE"


class CaseAction(str, Enum):
    """Mitigation action executed by the fraud operations team."""
    CARD_BLOCKED = "CARD_BLOCKED"
    CHARGEBACK_INITIATED = "CHARGEBACK_INITIATED"
    TRANSACTION_RELEASED = "TRANSACTION_RELEASED"
    CUSTOMER_CONTACTED = "CUSTOMER_CONTACTED"
    NONE = "NONE"


class Alert(Base):
    """Prioritised alert generated from scoring rules and anomaly thresholds."""

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(String(36), unique=True, nullable=False, index=True, default=lambda: f"ALT-{uuid.uuid4().hex[:8].upper()}")
    transaction_id = Column(String(36), nullable=False, index=True, default=lambda: f"TX-{uuid.uuid4().hex[:8].upper()}")
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False, index=True)

    amount = Column(Float, nullable=False)
    time_offset = Column(Float, nullable=False)
    fraud_probability = Column(Float, nullable=False, index=True)
    anomaly_score = Column(Float, nullable=False)
    risk_category = Column(String(20), nullable=False)
    priority = Column(String(20), nullable=False, default=AlertPriority.HIGH.value, index=True)
    operational_quadrant = Column(String(50), nullable=False)
    status = Column(String(30), nullable=False, default=AlertStatus.NEW.value, index=True)

    # Detailed justification why alert was triggered
    reasons = Column(JSON, nullable=False, default=list)
    # Full 30-feature vector
    transaction_payload = Column(JSON, nullable=False, default=dict)

    # Optional reference to case if escalated
    case_id = Column(String(36), nullable=True, index=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert record to dictionary representation."""
        return {
            "id": self.id,
            "alert_id": self.alert_id,
            "transaction_id": self.transaction_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "amount": self.amount,
            "time_offset": self.time_offset,
            "fraud_probability": round(self.fraud_probability, 4),
            "anomaly_score": round(self.anomaly_score, 4),
            "risk_category": self.risk_category,
            "priority": self.priority,
            "operational_quadrant": self.operational_quadrant,
            "status": self.status,
            "reasons": self.reasons,
            "case_id": self.case_id,
            "transaction_payload": self.transaction_payload,
        }


class InvestigationCase(Base):
    """Investigation case preserving immutable prediction snapshot and tracking analyst workflow."""

    __tablename__ = "investigation_cases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(String(36), unique=True, nullable=False, index=True, default=lambda: f"CASE-{uuid.uuid4().hex[:8].upper()}")
    alert_id = Column(String(36), nullable=True, index=True)
    transaction_id = Column(String(36), nullable=False, index=True)

    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    # Lifecycle State
    status = Column(String(30), nullable=False, default=CaseStatus.OPEN.value, index=True)
    priority = Column(String(20), nullable=False, default=AlertPriority.HIGH.value, index=True)
    assigned_analyst = Column(String(100), nullable=True, index=True)

    # Investigation Outcome
    verdict = Column(String(50), nullable=True)
    action_taken = Column(String(50), nullable=True)
    resolution_notes = Column(Text, nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)

    # =========================================================================
    # IMMUTABLE ORIGINAL PREDICTION SNAPSHOT (Preserved separately from outcomes)
    # =========================================================================
    original_fraud_probability = Column(Float, nullable=False)
    original_risk_category = Column(String(20), nullable=False)
    original_threshold_used = Column(Float, nullable=False)
    original_threshold_name = Column(String(50), nullable=False)
    original_model_version = Column(String(50), nullable=False)
    original_anomaly_score = Column(Float, nullable=False)
    original_anomaly_tier = Column(String(50), nullable=False)
    original_quadrant = Column(String(50), nullable=False)
    original_shap_explanation = Column(JSON, nullable=False, default=dict)
    transaction_snapshot = Column(JSON, nullable=False, default=dict)

    # Relationships
    notes = relationship(
        "CaseNote",
        back_populates="case",
        cascade="all, delete-orphan",
        order_by="CaseNote.created_at.desc()",
        lazy="selectin",
    )
    audit_logs = relationship(
        "CaseAuditLog",
        back_populates="case",
        cascade="all, delete-orphan",
        order_by="CaseAuditLog.timestamp.desc()",
        lazy="selectin",
    )

    def to_dict(self, include_relations: bool = True) -> Dict[str, Any]:
        """Convert case entity to dictionary representation."""
        data = {
            "id": self.id,
            "case_id": self.case_id,
            "alert_id": self.alert_id,
            "transaction_id": self.transaction_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "status": self.status,
            "priority": self.priority,
            "assigned_analyst": self.assigned_analyst,
            "verdict": self.verdict,
            "action_taken": self.action_taken,
            "resolution_notes": self.resolution_notes,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "original_prediction": {
                "fraud_probability": round(self.original_fraud_probability, 4),
                "risk_category": self.original_risk_category,
                "threshold_used": self.original_threshold_used,
                "threshold_name": self.original_threshold_name,
                "model_version": self.original_model_version,
                "anomaly_score": round(self.original_anomaly_score, 4),
                "anomaly_tier": self.original_anomaly_tier,
                "operational_quadrant": self.original_quadrant,
                "shap_explanation": self.original_shap_explanation,
            },
            "transaction_snapshot": self.transaction_snapshot,
        }
        if include_relations:
            data["notes"] = [n.to_dict() for n in self.notes]
            data["audit_logs"] = [a.to_dict() for a in self.audit_logs]
        return data


class CaseNote(Base):
    """Auditable analyst commentary and findings during investigation."""

    __tablename__ = "case_notes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    note_id = Column(String(36), unique=True, nullable=False, index=True, default=lambda: f"NOTE-{uuid.uuid4().hex[:8].upper()}")
    case_id = Column(String(36), ForeignKey("investigation_cases.case_id"), nullable=False, index=True)
    author = Column(String(100), nullable=False, default="analyst")
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    case = relationship("InvestigationCase", back_populates="notes")

    def to_dict(self) -> Dict[str, Any]:
        """Convert note to dictionary representation."""
        return {
            "note_id": self.note_id,
            "case_id": self.case_id,
            "author": self.author,
            "content": self.content,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class CaseAuditLog(Base):
    """Immutable audit trail recording every state change, assignment, and action taken."""

    __tablename__ = "case_audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    log_id = Column(String(36), unique=True, nullable=False, index=True, default=lambda: f"AUD-{uuid.uuid4().hex[:8].upper()}")
    case_id = Column(String(36), ForeignKey("investigation_cases.case_id"), nullable=False, index=True)
    action = Column(String(50), nullable=False)
    performed_by = Column(String(100), nullable=False)
    details = Column(JSON, nullable=False, default=dict)
    timestamp = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    case = relationship("InvestigationCase", back_populates="audit_logs")

    def to_dict(self) -> Dict[str, Any]:
        """Convert audit log to dictionary representation."""
        return {
            "log_id": self.log_id,
            "case_id": self.case_id,
            "action": self.action,
            "performed_by": self.performed_by,
            "details": self.details,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
