"""FastAPI route endpoints for the Prioritised Alert Queue and Risk Triage."""

import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.cases.database import get_db
from src.cases.models import AlertPriority, AlertStatus
from src.cases.repository import AlertRepository
from src.cases.service import get_case_service
from src.inference.schemas import TransactionInput

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/alerts", tags=["Alert Queue & Risk Triage"])


class AlertStatusUpdate(BaseModel):
    """Payload to update an alert's queue status."""
    status: AlertStatus
    case_id: Optional[str] = None


class AlertTriageRequest(BaseModel):
    """Payload to evaluate a transaction against risk policies and trigger an alert if needed."""
    transaction: TransactionInput
    threshold: Optional[str] = Field(default="default", description="Operational decision threshold")
    apply_preprocessing: bool = Field(default=True, description="Whether to apply RobustScaler preprocessing")
    auto_escalate: bool = Field(default=True, description="Auto-create investigation case if priority is CRITICAL")


@router.get(
    "",
    summary="List Prioritised Alert Queue",
    description="Retrieve paginated alert queue ordered by urgency (CRITICAL -> HIGH -> MEDIUM -> LOW).",
)
def list_alerts_endpoint(
    status: Optional[str] = Query(None, description="Filter by status: NEW, ESCALATED, CASE_CREATED, DISMISSED"),
    priority: Optional[str] = Query(None, description="Filter by priority: CRITICAL, HIGH, MEDIUM, LOW"),
    min_probability: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum fraud probability"),
    max_probability: Optional[float] = Query(None, ge=0.0, le=1.0, description="Maximum fraud probability"),
    search: Optional[str] = Query(None, description="Search term for alert_id, transaction_id, or case_id"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """List prioritised alerts."""
    items, total = AlertRepository.list_alerts(
        db=db,
        status=status,
        priority=priority,
        min_probability=min_probability,
        max_probability=max_probability,
        search=search,
        limit=limit,
        offset=offset,
    )
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": [item.to_dict() for item in items],
    }


@router.get(
    "/summary",
    summary="Alert Queue Metrics Summary",
    description="Aggregated counts of alerts by status and priority.",
)
def get_alert_summary(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Get alert counts grouped by status and priority."""
    return AlertRepository.get_summary_counts(db)


@router.get(
    "/{alert_id}",
    summary="Get Alert Details",
    description="Retrieve complete alert attributes including trigger reasons and full transaction payload.",
)
def get_alert_endpoint(alert_id: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Get single alert by ID."""
    alert = AlertRepository.get_by_id(db, alert_id)
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert '{alert_id}' not found.",
        )
    return alert.to_dict()


@router.patch(
    "/{alert_id}/status",
    summary="Update Alert Status",
    description="Transition alert lifecycle status (e.g. from NEW to ESCALATED or DISMISSED).",
)
def update_alert_status_endpoint(
    alert_id: str,
    payload: AlertStatusUpdate,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Update alert status."""
    alert = AlertRepository.update_status(
        db=db,
        alert_id=alert_id,
        status=payload.status.value,
        case_id=payload.case_id,
    )
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert '{alert_id}' not found.",
        )
    return alert.to_dict()


@router.post(
    "/triage",
    summary="Triage Transaction & Trigger Alert",
    description="Evaluate transaction against multi-signal policies, create prioritized alert in queue, and auto-escalate case if critical.",
)
def triage_transaction_endpoint(
    payload: AlertTriageRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Evaluate transaction and push to alert queue if risk criteria met."""
    service = get_case_service()
    try:
        result = service.process_and_triage_transaction(
            db=db,
            transaction=payload.transaction.model_dump(),
            threshold=payload.threshold or "default",
            apply_preprocessing=payload.apply_preprocessing,
            auto_escalate=payload.auto_escalate,
        )
        return result
    except Exception as exc:
        logger.error("Triage error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Triage error: {exc}",
        ) from exc
