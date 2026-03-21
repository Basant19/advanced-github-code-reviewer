"""
app/api/routes/hitl.py

HITL (Human-In-The-Loop) API Routes — P3 Production Version
-------------------------------------------------------------
Dedicated approve/reject endpoints for the Streamlit admin dashboard.

These are thin convenience wrappers around the /reviews/id/{id}/decision
endpoint. They exist so the Streamlit dashboard can call:
    POST /reviews/{id}/approve
    POST /reviews/{id}/reject

instead of the more verbose decision endpoint with a JSON body.

All business logic is delegated to review_service.decide_review() —
no graph, DB, or LangGraph code lives here.

Why Separate File
-----------------
hitl.py handles the approve/reject/status pattern for the dashboard UI.
review.py handles the general review CRUD + decision endpoint.
Both call the same service functions — no duplication of logic.

Endpoints
---------
POST /reviews/{review_id}/approve
    Approves a pending_hitl review.
    Delegates to decide_review(decision="approved").

POST /reviews/{review_id}/reject
    Rejects a pending_hitl review.
    Delegates to decide_review(decision="rejected").

GET  /reviews/{review_id}/status
    Read-only status check — safe to poll from dashboard.

Error Mapping
-------------
CustomException "not found"   → HTTP 404
CustomException "invalid"     → HTTP 400
CustomException "checkpoint"  → HTTP 409
Any other                     → HTTP 500
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Path
from fastapi import status as http_status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.core.exceptions import CustomException
from app.core.logger import get_logger
from app.services.review_service import decide_review, get_review

logger = get_logger(__name__)

router = APIRouter(prefix="/reviews", tags=["hitl"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class HITLDecisionRequest(BaseModel):
    """
    Optional request body for approve/reject endpoints.

    reviewer_note is stored for audit purposes (future P4/P5 use).
    Neither field is required — both endpoints work with empty body.
    """
    reviewer_note: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Optional note from the human reviewer — stored for audit trail",
    )


class HITLDecisionResponse(BaseModel):
    """
    Response returned by approve and reject endpoints.

    Includes verdict so the dashboard can display the final outcome
    immediately without a follow-up GET request.
    """
    review_id: int
    decision:  str
    verdict:   Optional[str]
    status:    str
    message:   str


class HITLStatusResponse(BaseModel):
    """
    Read-only status response for GET /reviews/{id}/status.
    Safe to poll from the dashboard without modifying state.
    """
    review_id:  int
    status:     str
    verdict:    Optional[str]
    thread_id:  Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]


# ── Helper ────────────────────────────────────────────────────────────────────

def _handle_error(e: Exception, context: str) -> None:
    """
    Map service exceptions to HTTP responses for HITL routes.

    Parameters
    ----------
    e : Exception
        Exception from service layer.
    context : str
        Calling function name for log context.

    Raises
    ------
    HTTPException
        Always raises — maps exception type to status code.
    """
    logger.error(
        "[hitl_route] Error in %s — type=%s message=%s",
        context, type(e).__name__, str(e),
        exc_info=True,
    )

    if isinstance(e, CustomException):
        msg = str(e).lower()

        if "not found" in msg:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=str(e),
            )

        if "invalid" in msg or "not in hitl" in msg or "pending" in msg:
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )

        if "checkpoint" in msg or "corrupted" in msg:
            raise HTTPException(
                status_code=http_status.HTTP_409_CONFLICT,
                detail="Graph checkpoint is corrupted or missing. "
                       "Trigger a new review.",
            )

        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

    raise HTTPException(
        status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="An internal server error occurred.",
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "/{review_id}/approve",
    response_model=HITLDecisionResponse,
    status_code=http_status.HTTP_200_OK,
    summary="Approve a pending review",
    description=(
        "Resumes the LangGraph workflow with decision='approved'. "
        "The graph executes hitl_node → verdict_node → END. "
        "A GitHub PR comment is posted with the AI findings. "
        "The review must be in 'pending_hitl' status."
    ),
)
async def approve_review(
    review_id: int = Path(..., gt=0, description="Review ID to approve"),
    body: HITLDecisionRequest = HITLDecisionRequest(),
    db: AsyncSession = Depends(get_db),
) -> HITLDecisionResponse:
    """
    Approve a pending_hitl review.

    Resumes the graph with decision='approved'. The verdict is determined
    by the AI findings — APPROVE if no issues, REQUEST_CHANGES if issues found.
    A GitHub comment is posted after successful completion.
    """
    logger.info(
        "[hitl_route] Approve request — review_id=%d note=%s",
        review_id,
        f"'{body.reviewer_note[:50]}...'"
        if body.reviewer_note and len(body.reviewer_note) > 50
        else f"'{body.reviewer_note}'",
    )

    try:
        review = await decide_review(
            review_id=review_id,
            decision="approved",
            db=db,
        )

        logger.info(
            "[hitl_route] Review approved — id=%d verdict=%s status=%s",
            review.id, review.verdict, review.status,
        )

        return HITLDecisionResponse(
            review_id=review.id,
            decision="approved",
            verdict=review.verdict,
            status=review.status,
            message=(
                f"Review approved. Verdict: {review.verdict}. "
                f"GitHub comment posted."
            ),
        )

    except Exception as e:
        _handle_error(e, "approve_review")


@router.post(
    "/{review_id}/reject",
    response_model=HITLDecisionResponse,
    status_code=http_status.HTTP_200_OK,
    summary="Reject a pending review",
    description=(
        "Resumes the LangGraph workflow with decision='rejected'. "
        "The graph produces HUMAN_REJECTED verdict. "
        "No GitHub comment is posted. "
        "The review must be in 'pending_hitl' status."
    ),
)
async def reject_review(
    review_id: int = Path(..., gt=0, description="Review ID to reject"),
    body: HITLDecisionRequest = HITLDecisionRequest(),
    db: AsyncSession = Depends(get_db),
) -> HITLDecisionResponse:
    """
    Reject a pending_hitl review.

    Resumes the graph with decision='rejected'. The verdict is set to
    HUMAN_REJECTED. No GitHub comment is posted — the PR is left unchanged.
    """
    logger.info(
        "[hitl_route] Reject request — review_id=%d note=%s",
        review_id,
        f"'{body.reviewer_note[:50]}...'"
        if body.reviewer_note and len(body.reviewer_note) > 50
        else f"'{body.reviewer_note}'",
    )

    try:
        review = await decide_review(
            review_id=review_id,
            decision="rejected",
            db=db,
        )

        logger.info(
            "[hitl_route] Review rejected — id=%d verdict=%s status=%s",
            review.id, review.verdict, review.status,
        )

        return HITLDecisionResponse(
            review_id=review.id,
            decision="rejected",
            verdict=review.verdict,
            status=review.status,
            message="Review rejected. No GitHub comment posted.",
        )

    except Exception as e:
        _handle_error(e, "reject_review")


@router.get(
    "/{review_id}/status",
    response_model=HITLStatusResponse,
    summary="Get review status",
    description=(
        "Read-only status check. Safe to poll from the dashboard. "
        "Returns current status, verdict, and thread_id."
    ),
)
async def get_review_status(
    review_id: int = Path(..., gt=0, description="Review ID"),
    db: AsyncSession = Depends(get_db),
) -> HITLStatusResponse:
    """
    Fetch the current status of a review.

    Read-only — does not modify the graph or review state.
    Safe to poll repeatedly from the Streamlit dashboard.
    """
    logger.info("[hitl_route] Status request — review_id=%d", review_id)

    try:
        review = await get_review(review_id=review_id, db=db)

        logger.debug(
            "[hitl_route] Status fetched — "
            "id=%d status=%s verdict=%s",
            review.id, review.status, review.verdict,
        )

        return HITLStatusResponse(
            review_id=review.id,
            status=review.status,
            verdict=review.verdict,
            thread_id=review.thread_id,
            created_at=(
                review.created_at.isoformat()
                if review.created_at else None
            ),
            updated_at=(
                review.updated_at.isoformat()
                if review.updated_at else None
            ),
        )

    except Exception as e:
        _handle_error(e, "get_review_status")