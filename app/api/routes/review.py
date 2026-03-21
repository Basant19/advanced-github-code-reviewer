"""
app/api/routes/review.py

Review API Routes — P3 Production Version
------------------------------------------
REST endpoints for triggering and retrieving PR reviews.

These routes are the HTTP interface between clients (Swagger UI,
Streamlit dashboard, GitHub webhook) and the review_service layer.
All business logic lives in review_service.py — routes only handle
HTTP concerns: request parsing, response formatting, error mapping.

Architecture
------------
Client → FastAPI Route → review_service → LangGraph + PostgreSQL

Endpoints
---------
POST   /reviews/trigger
    Manually trigger a PR review without a GitHub webhook.
    Returns immediately with status="pending_hitl" once the graph
    pauses at the HITL gate.

GET    /reviews/id/{review_id}
    Fetch a single review with all ReviewStep execution records.

POST   /reviews/id/{review_id}/decision
    Resume a pending_hitl review with human decision (approved/rejected).
    This is the primary HITL endpoint used by the Streamlit dashboard.

GET    /reviews/repo/{owner}/{repo}
    List all reviews for a repository, newest first.

GET    /reviews/
    List all reviews across all repositories — used by dashboard.

Error Mapping
-------------
CustomException "not found"   → HTTP 404
CustomException "invalid"     → HTTP 400
CustomException "pending_hitl" → HTTP 409 (conflict)
Any other CustomException     → HTTP 500
Unexpected exception          → HTTP 500

Async Notes
-----------
All endpoints are async — they use AsyncSession from get_db()
which connects to PostgreSQL via asyncpg.
"""

import json
from datetime import datetime
from typing import Optional, List, Any, NoReturn, Literal

from fastapi import APIRouter, Depends, HTTPException, Path
from fastapi import status as http_status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.core.exceptions import CustomException
from app.core.logger import get_logger
from app.services.review_service import (
    trigger_review,
    get_review,
    list_reviews,
    list_all_reviews,
    decide_review,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/reviews", tags=["reviews"])


# ── Request / Response Schemas ────────────────────────────────────────────────

class TriggerReviewRequest(BaseModel):
    """
    Request body for POST /reviews/trigger.

    Validates that owner/repo are non-empty strings and pr_number is positive.
    Strips whitespace from owner and repo to prevent silent mismatches.
    """
    owner: str = Field(
        ...,
        min_length=1,
        description="GitHub username or organization name",
        examples=["Basant19"],
    )
    repo: str = Field(
        ...,
        min_length=1,
        description="GitHub repository name (without owner prefix)",
        examples=["python_tuts"],
    )
    pr_number: int = Field(
        ...,
        gt=0,
        description="GitHub Pull Request number (must be positive)",
        examples=[1],
    )

    @field_validator("owner", "repo")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        """Strip leading/trailing whitespace to prevent silent GitHub 404s."""
        return v.strip()


class ReviewDecisionRequest(BaseModel):
    """
    Request body for POST /reviews/id/{review_id}/decision.

    Enforces that decision is exactly "approved" or "rejected" —
    no other values are accepted. This maps directly to the value
    injected into the LangGraph checkpoint via Command(resume=decision).
    """
    decision: Literal["approved", "rejected"] = Field(
        ...,
        description="Human decision: 'approved' continues to verdict, "
                    "'rejected' produces HUMAN_REJECTED verdict",
    )


class ReviewStepResponse(BaseModel):
    """
    Single workflow node execution record.

    output_data is stored as JSON text in PostgreSQL.
    It is parsed back to dict/list by _safe_parse_json() before returning.
    """
    model_config = ConfigDict(from_attributes=True)

    id:          int
    step_name:   str
    status:      str
    output_data: Any = None


class ReviewResponse(BaseModel):
    """
    Minimal review response — used in list endpoints and trigger response.
    """
    model_config = ConfigDict(from_attributes=True)

    id:         int
    status:     str
    verdict:    Optional[str] = None
    summary:    Optional[str] = None
    created_at: datetime


class ReviewDetailResponse(ReviewResponse):
    """
    Full review response — includes ReviewStep execution history.
    Used by GET /reviews/id/{review_id}.
    """
    steps: List[ReviewStepResponse] = []


class TriggerReviewResponse(BaseModel):
    """
    Response for POST /reviews/trigger.
    Includes review_id and status so the caller knows what to poll.
    """
    review_id: int
    status:    str
    message:   str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _handle_service_error(e: Exception, context: str) -> NoReturn:
    """
    Map service layer exceptions to appropriate HTTP responses.

    Called from every route's except block. Logs the error with context
    then raises HTTPException with the correct status code.

    Parameters
    ----------
    e : Exception
        Exception from service layer.
    context : str
        Route function name — included in log for traceability.

    Raises
    ------
    HTTPException
        Always raises — return type is NoReturn.
    """
    logger.error(
        "[review_route] Error in %s — type=%s message=%s",
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
                detail="Graph state is corrupted or missing. "
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


def _safe_parse_json(data: Any) -> Any:
    """
    Convert ReviewStep.output_data from stored text back to a Python object.

    output_data is stored as JSON string in PostgreSQL TEXT column.
    This function parses it back to dict/list for API response.
    Falls back gracefully on any parse failure — never crashes.

    Parameters
    ----------
    data : Any
        Raw value from database (str, dict, list, or None).

    Returns
    -------
    Any
        Parsed dict/list, or a fallback dict with the raw string.
    """
    if not data:
        return {}

    if isinstance(data, (dict, list)):
        return data

    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        # SandboxResult stringified objects
        if "SandboxResult" in str(data):
            return {"type": "execution_result", "raw": str(data)}
        return {"type": "text", "content": str(data)}


# ── Routes ────────────────────────────────────────────────────────────────────
# Route order matters — more specific paths must come before parameterized ones.
# /trigger must come before /{review_id} to avoid "trigger" being parsed as int.

@router.post(
    "/trigger",
    response_model=TriggerReviewResponse,
    status_code=http_status.HTTP_202_ACCEPTED,
    summary="Manually trigger a PR review",
    description=(
        "Triggers the AI review pipeline for a GitHub Pull Request. "
        "Returns immediately with status='pending_hitl' once the graph "
        "pauses at the HITL gate awaiting human approval."
    ),
)
async def trigger_new_review(
    request: TriggerReviewRequest,
    db: AsyncSession = Depends(get_db),
) -> TriggerReviewResponse:
    """
    Trigger the AI review workflow for a PR.

    The graph runs fetch → analyze → reflect → lint → [refactor] → PAUSE.
    Response is returned once the graph pauses at the HITL gate.
    Use POST /reviews/id/{review_id}/decision to approve or reject.
    """
    logger.info(
        "[review_route] Trigger review — %s/%s#%d",
        request.owner, request.repo, request.pr_number,
    )

    try:
        review = await trigger_review(
            owner=request.owner,
            repo=request.repo,
            pr_number=request.pr_number,
            db=db,
        )

        logger.info(
            "[review_route] Review triggered — id=%d status=%s",
            review.id, review.status,
        )

        return TriggerReviewResponse(
            review_id=review.id,
            status=review.status,
            message=(
                f"Review paused at HITL gate — "
                f"call POST /reviews/id/{review.id}/decision to approve or reject"
                if review.status == "pending_hitl"
                else f"Review {review.status}"
            ),
        )

    except Exception as e:
        _handle_service_error(e, "trigger_new_review")


@router.get(
    "/",
    response_model=List[dict],
    summary="List all reviews (dashboard)",
    description=(
        "Returns all reviews across all repositories, newest first. "
        "Used by the Streamlit admin dashboard."
    ),
)
async def list_all_reviews_route(
    db: AsyncSession = Depends(get_db),
) -> List[dict]:
    """List all reviews across all repositories for the dashboard."""
    logger.info("[review_route] List all reviews")

    try:
        return await list_all_reviews(db)
    except Exception as e:
        _handle_service_error(e, "list_all_reviews_route")


@router.get(
    "/repo/{owner}/{repo}",
    response_model=List[ReviewResponse],
    summary="List all reviews for a repository",
)
async def list_repo_reviews(
    owner: str,
    repo: str,
    db: AsyncSession = Depends(get_db),
) -> List[ReviewResponse]:
    """
    List all reviews for a specific repository, ordered newest first.
    Returns an empty list if the repository has no reviews.
    """
    logger.info("[review_route] List reviews — %s/%s", owner, repo)

    try:
        return await list_reviews(owner, repo, db)
    except Exception as e:
        _handle_service_error(e, "list_repo_reviews")


@router.get(
    "/id/{review_id}",
    response_model=ReviewDetailResponse,
    summary="Get a single review with execution steps",
)
async def get_review_details(
    review_id: int = Path(..., gt=0, description="Review ID"),
    db: AsyncSession = Depends(get_db),
) -> ReviewDetailResponse:
    """
    Fetch a single review by ID including all ReviewStep records.

    ReviewStep.output_data is parsed from JSON text back to dict/list
    before returning to prevent the client receiving raw JSON strings.
    """
    logger.info("[review_route] Get review — id=%d", review_id)

    try:
        review = await get_review(review_id, db)

        # Parse output_data from JSON string to dict/list
        for step in review.steps:
            step.output_data = _safe_parse_json(step.output_data)

        return review

    except Exception as e:
        _handle_service_error(e, "get_review_details")


@router.post(
    "/id/{review_id}/decision",
    response_model=ReviewResponse,
    summary="Submit human decision (approve or reject)",
    description=(
        "Resumes the LangGraph workflow after human review. "
        "The review must be in 'pending_hitl' status. "
        "Approved reviews proceed to verdict_node and post a GitHub comment. "
        "Rejected reviews produce HUMAN_REJECTED verdict with no GitHub comment."
    ),
)
async def submit_human_decision(
    review_id: int = Path(..., gt=0, description="Review ID"),
    request: ReviewDecisionRequest = None,
    db: AsyncSession = Depends(get_db),
) -> ReviewResponse:
    """
    Submit a human decision to resume a pending_hitl review.

    This is the primary HITL endpoint. The decision is injected into
    the LangGraph checkpoint via Command(resume=decision), which causes
    hitl_node to receive it via interrupt() return value.
    """
    logger.info(
        "[review_route] HITL decision — review_id=%d decision=%s",
        review_id, request.decision,
    )

    try:
        review = await decide_review(
            review_id=review_id,
            decision=request.decision,
            db=db,
        )

        logger.info(
            "[review_route] Decision processed — "
            "review_id=%d status=%s verdict=%s",
            review.id, review.status, review.verdict,
        )

        return review

    except Exception as e:
        _handle_service_error(e, "submit_human_decision")