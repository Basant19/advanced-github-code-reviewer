"""
review.py 
Location: E:\advanced-github-code-reviewer\app\api\routes\review.py

API Layer for Code Review Management:
✔ HITL-First: Dedicated /decision endpoint to resume paused graphs.
✔ Robust Parsing: _safe_parse_json prevents frontend 500s on malformed state data.
✔ Error Mapping: Converts CustomException messages into appropriate HTTP status codes.
✔ Validation: Strict Pydantic models for trigger and decision payloads.
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
    decide_review,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/reviews", tags=["reviews"])

# ─────────────────────────────────────────────
# SCHEMAS (Request/Response)
# ─────────────────────────────────────────────

class TriggerReviewRequest(BaseModel):
    owner: str = Field(..., min_length=1, description="GitHub Org or User")
    repo: str = Field(..., min_length=1, description="Repository Name")
    pr_number: int = Field(..., gt=0)

    @field_validator("owner", "repo")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()

class ReviewDecisionRequest(BaseModel):
    decision: Literal["approved", "rejected"] = Field(..., description="Human verdict to resume the graph")

class ReviewStepResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    step_name: str
    status: str
    output_data: Any = None  # Will be parsed into dict/list by _safe_parse_json

class ReviewResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    status: str
    verdict: Optional[str] = None
    summary: Optional[str] = None
    created_at: datetime

class ReviewDetailResponse(ReviewResponse):
    steps: List[ReviewStepResponse] = []

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _handle_service_error(e: Exception, context: str) -> NoReturn:
    """Maps internal service exceptions to FastAPI HTTPExceptions."""
    logger.error(f"[ROUTE ERROR] {context}: {str(e)}", exc_info=True)
    
    if isinstance(e, CustomException):
        err_msg = str(e).lower()
        if "not found" in err_msg:
            raise HTTPException(status_code=404, detail=str(e))
        if "invalid" in err_msg or "decision" in err_msg:
            raise HTTPException(status_code=400, detail=str(e))
        if "checkpoint" in err_msg:
            raise HTTPException(status_code=409, detail="Graph state is corrupted or missing.")
            
    raise HTTPException(status_code=500, detail="An internal error occurred in the review service.")

def _safe_parse_json(data: Any) -> Any:
    """Ensures DB text fields are converted back to JSON for the API response."""
    if not data: return {}
    if isinstance(data, (dict, list)): return data
    
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        # Fallback for stringified objects like SandboxResult
        if "SandboxResult" in str(data):
            return {"type": "execution_result", "raw": str(data)}
        return {"type": "text", "content": str(data)}

# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@router.post("/trigger", response_model=ReviewResponse, status_code=202)
async def trigger_new_review(request: TriggerReviewRequest, db: AsyncSession = Depends(get_db)):
    """Triggers the AI agent. Will likely stop at 'pending_hitl'."""
    try:
        review = await trigger_review(
            owner=request.owner, 
            repo=request.repo, 
            pr_number=request.pr_number, 
            db=db
        )
        return review
    except Exception as e:
        _handle_service_error(e, "trigger_new_review")

@router.get("/id/{review_id}", response_model=ReviewDetailResponse)
async def get_review_details(review_id: int = Path(..., gt=0), db: AsyncSession = Depends(get_db)):
    """Fetch a full review including all execution steps."""
    try:
        review = await get_review(review_id, db)
        # Manually parse steps to ensure JSON safety
        for step in review.steps:
            step.output_data = _safe_parse_json(step.output_data)
        return review
    except Exception as e:
        _handle_service_error(e, "get_review_details")

@router.post("/id/{review_id}/decision", response_model=ReviewResponse)
async def submit_human_decision(
    review_id: int = Path(..., gt=0), 
    req: ReviewDecisionRequest = None, 
    db: AsyncSession = Depends(get_db)
):
    """
    The Human-In-The-Loop endpoint. 
    Resumes the LangGraph workflow using the 'approved' or 'rejected' signal.
    """
    try:
        review = await decide_review(
            review_id=review_id, 
            decision=req.decision, 
            db=db
        )
        return review
    except Exception as e:
        _handle_service_error(e, "submit_human_decision")

@router.get("/repo/{owner}/{repo}", response_model=List[ReviewResponse])
async def list_repo_reviews(owner: str, repo: str, db: AsyncSession = Depends(get_db)):
    """List all review history for a specific repository."""
    try:
        return await list_reviews(owner, repo, db)
    except Exception as e:
        _handle_service_error(e, "list_repo_reviews")