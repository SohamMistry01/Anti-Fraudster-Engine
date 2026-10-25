"""FastAPI route endpoints for Case Management, Investigator Statuses, Notes, and Audits."""

import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.cases.database import get_db
from src.cases.models import (
    AlertPriority,
    CaseAction,
    CaseStatus,
    CaseVerdict,
)
from src.cases.repository import CaseRepository
from src.cases.service import get_case_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cases", tags=["Investigation Case Management"])


class CreateCaseFromAlertRequest(BaseModel):
    """Payload to escalate an alert into an active investigation case."""
    alert_id: str = Field(..., description="Unique alert ID to escalate")
    assigned_analyst: Optional[str] = Field(None, description="Analyst to assign")
    initial_note: Optional[str] = Field(None, description="Initial investigative observation")
    creator: str = Field(default="analyst", description="Username or system creator")


class CaseWorkflowUpdate(BaseModel):
    """Payload to update case investigation progress, verdict, and mitigation action."""
    status: Optional[CaseStatus] = None
    assigned_analyst: Optional[str] = None
    verdict: Optional[CaseVerdict] = None
    action_taken: Optional[CaseAction] = None
    resolution_notes: Optional[str] = None
    updated_by: str = Field(default="analyst", description="Analyst making the update")


class AddCaseNoteRequest(BaseModel):
    """Payload to add an investigator note to a case."""
    author: str = Field(default="analyst", description="Author of the note")
    content: str = Field(..., min_length=1, description="Commentary or investigation finding")


@router.get(
    "",
    summary="List Investigation Cases",
    description="Retrieve paginated list of cases filtered by status, priority, or assigned analyst.",
)
def list_cases_endpoint(
    status: Optional[str] = Query(None, description="Filter by status: OPEN, UNDER_INVESTIGATION, PENDING_CUSTOMER, etc."),
    priority: Optional[str] = Query(None, description="Filter by priority: CRITICAL, HIGH, MEDIUM, LOW"),
    analyst: Optional[str] = Query(None, description="Filter by assigned analyst username"),
    search: Optional[str] = Query(None, description="Search term for case_id, transaction_id, or alert_id"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """List investigation cases."""
    items, total = CaseRepository.list_cases(
        db=db,
        status=status,
        priority=priority,
        analyst=analyst,
        search=search,
        limit=limit,
        offset=offset,
    )
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": [item.to_dict(include_relations=False) for item in items],
    }


@router.get(
    "/summary",
    summary="Case Executive Metrics",
    description="High-level metrics on case counts, open investigations, and confirmed fraud rate.",
)
def get_case_summary_endpoint(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Retrieve aggregate case status and resolution metrics."""
    return CaseRepository.get_summary_metrics(db)


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Open Case from Alert",
    description="Escalate an alert into an active investigation case, capturing an immutable prediction snapshot.",
)
def create_case_from_alert_endpoint(
    payload: CreateCaseFromAlertRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Promote an alert into an active investigation case."""
    service = get_case_service()
    try:
        case = service.open_case_from_alert(
            db=db,
            alert_id=payload.alert_id,
            assigned_analyst=payload.assigned_analyst,
            initial_note=payload.initial_note,
            creator=payload.creator,
        )
        return case.to_dict(include_relations=True)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Failed to open case: %s", exc, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Case creation error: {exc}") from exc


@router.get(
    "/{case_id}",
    summary="Get Case Details with Immutable Snapshot & Audit History",
    description="Retrieve complete case record including original prediction snapshot, investigator notes, and audit logs.",
)
def get_case_details_endpoint(case_id: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Retrieve full case details."""
    case = CaseRepository.get_by_id(db, case_id)
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"InvestigationCase '{case_id}' not found.",
        )
    return case.to_dict(include_relations=True)


@router.patch(
    "/{case_id}/workflow",
    summary="Update Case Workflow & Status",
    description="Transition case status, assign analyst, or record final verdict/action. Preserves full audit trail.",
)
def update_case_workflow_endpoint(
    case_id: str,
    payload: CaseWorkflowUpdate,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Update case lifecycle and record auditable event."""
    service = get_case_service()
    try:
        case = service.update_case_workflow(
            db=db,
            case_id=case_id,
            status=payload.status.value if payload.status else None,
            assigned_analyst=payload.assigned_analyst,
            verdict=payload.verdict.value if payload.verdict else None,
            action_taken=payload.action_taken.value if payload.action_taken else None,
            resolution_notes=payload.resolution_notes,
            updated_by=payload.updated_by,
        )
        return case.to_dict(include_relations=True)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Failed to update case: %s", exc, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Update error: {exc}") from exc


@router.post(
    "/{case_id}/notes",
    status_code=status.HTTP_201_CREATED,
    summary="Add Investigator Note",
    description="Append an analyst observation or finding to a case.",
)
def add_case_note_endpoint(
    case_id: str,
    payload: AddCaseNoteRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Add a note to a case."""
    service = get_case_service()
    try:
        note_dict = service.add_note(
            db=db,
            case_id=case_id,
            author=payload.author,
            content=payload.content,
        )
        return note_dict
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Failed to add note: %s", exc, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Note error: {exc}") from exc


@router.get(
    "/{case_id}/audit",
    summary="Get Case Audit Trail",
    description="Retrieve chronological list of all state changes, assignments, and notes for compliance auditability.",
)
def get_case_audit_endpoint(case_id: str, db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    """Retrieve full audit log for a case."""
    case = CaseRepository.get_by_id(db, case_id)
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"InvestigationCase '{case_id}' not found.",
        )
    return [log.to_dict() for log in case.audit_logs]
