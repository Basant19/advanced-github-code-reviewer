#E:\advanced-github-code-reviewer\app\services\review_service.py


import sys
import json
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any
from sqlalchemy.orm import selectinload
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Restored Imports
from app.graph.workflow import review_graph
from app.graph.state import build_initial_state
from app.mcp.github_client import GitHubClient # Restored
from app.db.models.repository import Repository
from app.db.models.pull_request import PullRequest
from app.db.models.review import Review
from app.db.models.review_step import ReviewStep
from app.core.exceptions import CustomException
from app.core.logger import get_logger
from langgraph.types import Command
# Add this alongside existing imports
from app.mcp.sandbox_client import SandboxResult

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# GraphInterrupt Handling
# ─────────────────────────────────────────────
try:
    from langgraph.errors import GraphInterrupt
except ImportError:
    try:
        from langgraph.types import GraphInterrupt
    except ImportError:
        class GraphInterrupt(Exception): pass

# ─────────────────────────────────────────────
# Helper Functions (Condensed)
# ─────────────────────────────────────────────

def serialize_output(data: Any) -> Any:
    """Ensure JSON-safe serialization and consistent structure."""
    if isinstance(data, list):
        return [serialize_output(item) for item in data]
    
    if isinstance(data, SandboxResult):
        return {
            "passed": data.passed,
            "output": data.output,
            "errors": data.errors,
            "exit_code": data.exit_code,
            "tool": data.tool
        }

    if hasattr(data, "dict"):
        return data.dict()

    # If it's a basic type, return as is (prevents the "value" wrapping bug)
    if isinstance(data, (str, int, float, bool)) or data is None:
        return data

    return str(data)

NODE_NAME_MAP = {
    "diff": "fetch_diff",
    "issues": "analyze_code",
    "suggestions": "analyze_code", # Added
    "lint_result": "lint_check",
    "patch": "refactor_patch",
    "validation_result": "validator",
    "summary": "summary",          # Added
    "verdict": "final_verdict"     # Added
}

async def _get_or_create_repository(db: AsyncSession, owner: str, repo: str) -> Repository:
    res = await db.execute(select(Repository).where(Repository.owner == owner, Repository.name == repo))
    obj = res.scalar_one_or_none()
    if obj: return obj
    obj = Repository(owner=owner, name=repo, url=f"https://github.com/{owner}/{repo}", default_branch="main")
    db.add(obj); await db.commit(); await db.refresh(obj); return obj

async def _get_or_create_pull_request(db: AsyncSession, repository: Repository, pr_number: int, metadata: Dict) -> PullRequest:
    res = await db.execute(select(PullRequest).where(PullRequest.repo_id == repository.id, PullRequest.pr_number == pr_number))
    pr = res.scalar_one_or_none()
    if pr:
        if metadata.get("title"): pr.title = metadata["title"]
        await db.commit(); return pr
    pr = PullRequest(repo_id=repository.id, pr_number=pr_number, title=metadata.get("title", "Untitled PR"), 
                     author=metadata.get("author", "unknown"), branch=metadata.get("head_branch", "unknown"), status="open")
    db.add(pr); await db.commit(); await db.refresh(pr); return pr

async def _persist_review_steps(db: AsyncSession, review: Review, state: Dict) -> None:
    res = await db.execute(select(ReviewStep.step_name).where(ReviewStep.review_id == review.id))
    existing = {row[0] for row in res.all()}
    
    steps = []
    for key, v in NODE_NAME_MAP.items():
        if key in state and v not in existing:
            # We store as raw data now, the API's _safe_parse_json handles the rest
            val = serialize_output(state[key])
            steps.append(ReviewStep(
                review_id=review.id, 
                step_name=v, 
                status="completed", 
                input_data="{}", 
                output_data=json.dumps(val) if not isinstance(val, str) else val
            ))
    
    if steps:
        db.add_all(steps)
        await db.flush() # Flush keeps the transaction open but pushes changes

# ─────────────────────────────────────────────
# Core Service Functions
# ─────────────────────────────────────────────

async def trigger_review(owner: str, repo: str, pr_number: int, db: AsyncSession) -> Review:
    review = None
    try:
        state = build_initial_state(owner, repo, pr_number)
        repo_obj = await _get_or_create_repository(db, owner, repo)
        pr_obj = await _get_or_create_pull_request(db, repo_obj, pr_number, {})
        
        review = Review(pull_request_id=pr_obj.id, reviewer="ai", status="running", thread_id=str(uuid.uuid4()))
        db.add(review); await db.commit(); await db.refresh(review)

        config = {"configurable": {"thread_id": review.thread_id}}

        try:
            # Execute the graph
            final_state = await review_graph.ainvoke(state, config=config)
        except GraphInterrupt:
            # 1. Capture the state AT THE MOMENT of interruption
            # LangGraph stores the state in the checkpointer
            snapshot = await review_graph.aget_state(config)
            
            # 2. Persist steps completed so far
            await _persist_review_steps(db, review, snapshot.values)
            
            # 3. Set correct status
            review.status = "pending_hitl"
            await db.commit()
            return review

        # IF COMPLETED WITHOUT INTERRUPT:
        await _persist_review_steps(db, review, final_state)
        review.verdict = final_state.get("verdict")
        review.summary = final_state.get("summary")
        review.status = "completed"
        
        # NEW: Post to GitHub on completion
        try:
            gh = GitHubClient()
            await gh.post_review_comment(owner, repo, pr_number, review.summary, review.verdict)
        except:
            logger.warning("Failed to post to GitHub, but review saved.")

        await db.commit()
        return review
    except Exception as e:
        if review: review.status = "failed"; await db.commit()
        raise CustomException(str(e), sys)

async def decide_review(review_id: int, decision: str, db: AsyncSession) -> Review:
    res = await db.execute(select(Review).where(Review.id == review_id))
    review = res.scalar_one_or_none()
    if not review or review.status != "pending_hitl": raise CustomException("Invalid Review State", sys)
    
    config = {"configurable": {"thread_id": review.thread_id}}
    
    # Resume the graph
    final_state = await review_graph.ainvoke(Command(resume=decision), config=config)
    
    await _persist_review_steps(db, review, final_state)
    review.verdict = final_state.get("verdict")
    review.summary = final_state.get("summary")
    
    # Final status based on human choice
    review.status = "completed" if decision == "approved" else "rejected"
    
    await db.commit(); await db.refresh(review); return review

async def get_review(review_id: int, db: AsyncSession) -> Review:
    res = await db.execute(select(Review).options(selectinload(Review.steps)).where(Review.id == review_id))
    review = res.scalar_one_or_none()
    if not review: raise CustomException("Not Found", sys)
    return review

async def list_all_reviews(db: AsyncSession) -> List[Dict]:
    res = await db.execute(select(Review, PullRequest, Repository).join(PullRequest).join(Repository).order_by(Review.created_at.desc()))
    return [{"id": r.id, "status": r.status, "verdict": r.verdict, "repo": f"{repo.owner}/{repo.name}", "pr_number": pr.pr_number, "title": pr.title} for r, pr, repo in res.all()]

async def list_reviews(owner: str, repo: str, db: AsyncSession) -> List[Review]:
    """List all reviews for a specific repository (required by review.py)."""
    try:
        # Find the repo first
        res = await db.execute(select(Repository).where(Repository.owner == owner, Repository.name == repo))
        repository = res.scalar_one_or_none()
        if not repository:
            return []

        # Get reviews linked to this repo via PullRequest
        res = await db.execute(
            select(Review)
            .join(PullRequest, Review.pull_request_id == PullRequest.id)
            .where(PullRequest.repo_id == repository.id)
            .order_by(Review.created_at.desc())
        )
        return res.scalars().all()
    except Exception as e:
        logger.error("[review_service] list_reviews failed", exc_info=True)
        raise CustomException(str(e), sys)