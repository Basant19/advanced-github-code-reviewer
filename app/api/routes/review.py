"""
app/api/routes/review.py

Review API Routes
-----------------
REST endpoints responsible for triggering and retrieving
Pull Request reviews processed by the AI review pipeline.

These routes act as the HTTP interface between the client
(UI / GitHub webhook / CLI) and the review_service layer.

Architecture Layer
------------------
Client → FastAPI Routes → Service Layer → Database + LangGraph

Routes
------
POST   /reviews/trigger
    Manually trigger a review for a GitHub Pull Request.

GET    /reviews/{owner}/{repo}
    Retrieve all reviews belonging to a repository.

GET    /reviews/{review_id}
    Retrieve a specific review with step-by-step execution details.

Error Handling
--------------
Service layer raises CustomException which is converted into
appropriate HTTP responses here.

Async Notes
-----------
All endpoints use AsyncSession because the application runs on
async SQLAlchemy with asyncpg.
"""

from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException
from fastapi import status as http_status
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.core.exceptions import CustomException
from app.core.logger import get_logger
from app.services.review_service import (
    trigger_review,
    get_review,
    list_reviews,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/reviews", tags=["reviews"])


# ─────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────

class TriggerReviewRequest(BaseModel):
    """Request body for manually triggering a review."""
    owner: str
    repo: str
    pr_number: int


class ReviewStepResponse(BaseModel):
    """Represents a single step executed during the review workflow."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    step_name: str
    input_data: Optional[dict] = None
    output_data: Optional[dict] = None


class ReviewResponse(BaseModel):
    """Minimal review response used in list endpoints."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    status: str
    verdict: Optional[str] = None
    summary: Optional[str] = None
    created_at: Optional[str] = None


class ReviewDetailResponse(ReviewResponse):
    """Full review response including step execution history."""

    steps: List[ReviewStepResponse] = []


class TriggerReviewResponse(BaseModel):
    """Response returned immediately after triggering a review."""

    review_id: int
    status: str
    verdict: Optional[str] = None
    message: str


# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────

def _handle_service_error(e: Exception, context: str) -> None:
    """
    Converts service layer exceptions into HTTP responses.

    Parameters
    ----------
    e : Exception
        Exception raised by service layer.
    context : str
        Context describing where the error occurred.
    """

    if isinstance(e, CustomException):
        logger.error(
            "[review_route] %s | service_error=%s",
            context,
            str(e),
            exc_info=True
        )

        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=str(e)
            )

        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

    logger.error(
        "[review_route] %s | unexpected_error=%s",
        context,
        str(e),
        exc_info=True
    )

    raise HTTPException(
        status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Internal server error"
    )


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@router.post(
    "/trigger",
    response_model=TriggerReviewResponse,
    status_code=http_status.HTTP_202_ACCEPTED,
    summary="Manually trigger a PR review"
)
async def trigger_review_route(
    request: TriggerReviewRequest,
    db: AsyncSession = Depends(get_db),
) -> TriggerReviewResponse:
    """
    Triggers the AI review workflow for a GitHub Pull Request.

    The workflow is executed asynchronously via LangGraph.

    Possible states returned immediately:
        running
        pending_hitl
        completed

    HITL (Human-in-the-Loop):
    If the workflow reaches the HITL node, execution pauses and
    the review status becomes `pending_hitl`.
    """

    logger.info(
        "[review_route] Trigger review request received | repo=%s/%s pr=%s",
        request.owner,
        request.repo,
        request.pr_number,
    )

    try:
        review = await trigger_review(
            owner=request.owner,
            repo=request.repo,
            pr_number=request.pr_number,
            db=db,
        )

        message = (
            "Review suspended at HITL gate"
            if review.status == "pending_hitl"
            else "Review completed"
        )

        return TriggerReviewResponse(
            review_id=review.id,
            status=review.status,
            verdict=review.verdict,
            message=message,
        )

    except Exception as e:
        _handle_service_error(e, "trigger_review_route")


@router.get(
    "/{owner}/{repo}",
    response_model=List[ReviewResponse],
    summary="List all reviews for a repository",
)
async def list_reviews_route(
    owner: str,
    repo: str,
    db: AsyncSession = Depends(get_db),
) -> List[ReviewResponse]:
    """
    Returns all reviews associated with a repository.
    Results are ordered by most recent first.
    """

    logger.info(
        "[review_route] Fetch reviews | repo=%s/%s",
        owner,
        repo,
    )

    try:
        reviews = await list_reviews(owner=owner, repo=repo, db=db)

        return [
            ReviewResponse(
                id=r.id,
                status=r.status,
                verdict=r.verdict,
                summary=r.summary,
                created_at=str(r.created_at) if r.created_at else None,
            )
            for r in reviews
        ]

    except Exception as e:
        _handle_service_error(e, "list_reviews_route")


@router.get(
    "/{review_id}",
    response_model=ReviewDetailResponse,
    summary="Get a single review",
)
async def get_review_route(
    review_id: int,
    db: AsyncSession = Depends(get_db),
) -> ReviewDetailResponse:
    """
    Returns detailed information for a specific review.

    Includes the execution steps generated by the
    LangGraph workflow nodes.
    """

    logger.info(
        "[review_route] Fetch review details | review_id=%s",
        review_id,
    )

    try:
        review = await get_review(review_id=review_id, db=db)

        steps = [
            ReviewStepResponse(
                id=s.id,
                step_name=s.step_name,
                input_data=s.input_data,
                output_data=s.output_data,
            )
            for s in (review.steps or [])
        ]

        return ReviewDetailResponse(
            id=review.id,
            status=review.status,
            verdict=review.verdict,
            summary=review.summary,
            created_at=str(review.created_at) if review.created_at else None,
            steps=steps,
        )

    except Exception as e:
        _handle_service_error(e, "get_review_route")