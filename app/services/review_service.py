"""
app/services/review_service.py

Review Orchestration Service — P3 Production Version
-----------------------------------------------------
The single entry point for triggering, persisting, and retrieving PR reviews.

Responsibilities
----------------
1. Create and persist Review records in PostgreSQL
2. Run the LangGraph review workflow via astream
3. Detect GraphInterrupt — set status=pending_hitl
4. Resume the graph via Command(resume=decision) on human decision
5. Persist final verdict and summary after completion
6. Post GitHub PR comment on approval (skipped on rejection)

HITL Lifecycle
--------------
trigger_review()
    Runs graph via astream until GraphInterrupt fires.
    Sets review.status = "pending_hitl".
    Returns immediately — does NOT block waiting for human.

decide_review()
    Called by POST /reviews/id/{id}/decision.
    Validates review.status == "pending_hitl".
    Resumes graph via Command(resume=decision).
    Graph executes hitl_node → verdict_node → END.
    Sets review.status = "completed" or "rejected".

Command(resume=) Format
-----------------------
LangGraph expects Command(resume=value) where value is what
interrupt() returns inside hitl_node. We pass the decision string
directly: Command(resume="approved") or Command(resume="rejected").

State Snapshot
--------------
After GraphInterrupt, review_graph.aget_state(config) returns the
full merged state at the checkpoint. This is used to persist
ReviewStep records even on the interrupted (pending_hitl) path.

Error Handling
--------------
All service functions wrap exceptions in CustomException.
GraphInterrupt is caught separately BEFORE Exception to ensure
it is never misidentified as a failure.
"""

import sys
import json
import uuid
from typing import List, Dict, Any, Optional

from sqlalchemy.orm import selectinload
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from langgraph.types import Command
from langgraph.errors import GraphInterrupt

from app.graph.workflow import get_review_graph
from app.graph.state import build_initial_state
from app.mcp.github_client import GitHubClient

from app.db.models.repository import Repository
from app.db.models.pull_request import PullRequest
from app.db.models.review import Review
from app.db.models.review_step import ReviewStep

from app.core.exceptions import CustomException
from app.core.logger import get_logger

logger = get_logger(__name__)


# ── Node-to-step name mapping ─────────────────────────────────────────────────
# Maps state keys to ReviewStep.step_name values for persistence.
# Only keys present in final state are persisted — missing keys are skipped.

NODE_NAME_MAP: Dict[str, str] = {
    "diff":              "fetch_diff",
    "issues":            "analyze_code",
    "suggestions":       "analyze_code",
    "lint_result":       "lint_check",
    "patch":             "refactor_patch",
    "validation_result": "validator",
    "summary":           "summary",
    "verdict":           "final_verdict",
}


# ── Serialization ─────────────────────────────────────────────────────────────

def _serialize(data: Any) -> Any:
    """
    Convert any state value to a JSON-serializable format.

    Handles SandboxResult objects, Pydantic models, primitives, and lists.
    Falls back to str() for unknown types — never raises.

    Parameters
    ----------
    data : Any
        Value from graph state.

    Returns
    -------
    Any
        JSON-serializable representation.
    """
    try:
        if isinstance(data, list):
            return [_serialize(item) for item in data]

        # SandboxResult dataclass
        if hasattr(data, "passed") and hasattr(data, "output"):
            return {
                "passed":      data.passed,
                "output":      data.output,
                "errors":      getattr(data, "errors", ""),
                "exit_code":   data.exit_code,
                "duration_ms": data.duration_ms,
                "tool":        getattr(data, "tool", ""),
            }

        if hasattr(data, "dict"):
            return data.dict()

        if isinstance(data, (str, int, float, bool)) or data is None:
            return data

        return str(data)

    except Exception:
        logger.warning(
            "[review_service] _serialize failed for type=%s — using str()",
            type(data).__name__,
        )
        return str(data)


# ── DB Helpers ────────────────────────────────────────────────────────────────

async def _get_or_create_repository(
    db: AsyncSession,
    owner: str,
    name: str,
) -> Repository:
    """
    Fetch an existing Repository record or create one if not found.

    Parameters
    ----------
    db : AsyncSession
    owner : str  GitHub owner (username or org)
    name : str   Repository name

    Returns
    -------
    Repository
        Existing or newly created record.

    Raises
    ------
    CustomException
        On database error.
    """
    try:
        result = await db.execute(
            select(Repository).where(
                Repository.owner == owner,
                Repository.name == name,
            )
        )
        repo = result.scalar_one_or_none()

        if repo:
            logger.debug(
                "[review_service] Repository found — %s/%s id=%d",
                owner, name, repo.id,
            )
            return repo

        repo = Repository(
            owner=owner,
            name=name,
            url=f"https://github.com/{owner}/{name}",
            default_branch="main",
        )
        db.add(repo)
        await db.commit()
        await db.refresh(repo)

        logger.info(
            "[review_service] Repository created — %s/%s id=%d",
            owner, name, repo.id,
        )
        return repo

    except Exception as e:
        logger.exception(
            "[review_service] _get_or_create_repository failed — %s/%s",
            owner, name,
        )
        raise CustomException(str(e), sys)


async def _get_or_create_pull_request(
    db: AsyncSession,
    repo: Repository,
    pr_number: int,
    title: Optional[str] = None,
    author: Optional[str] = None,
    branch: Optional[str] = None,
) -> PullRequest:
    """
    Fetch an existing PullRequest record or create a stub.

    On first call (before the diff is fetched), title/author/branch
    may be unknown — stub values are used and updated later.

    Parameters
    ----------
    db : AsyncSession
    repo : Repository   Parent repository record.
    pr_number : int     GitHub PR number.
    title : str         PR title (optional — uses stub if not provided).
    author : str        PR author (optional).
    branch : str        Head branch (optional).

    Returns
    -------
    PullRequest
        Existing or newly created record.

    Raises
    ------
    CustomException
        On database error.
    """
    try:
        result = await db.execute(
            select(PullRequest).where(
                PullRequest.repo_id == repo.id,
                PullRequest.pr_number == pr_number,
            )
        )
        pr = result.scalar_one_or_none()

        if pr:
            # Update with real metadata if provided
            if title:
                pr.title = title
            if author:
                pr.author = author
            await db.commit()
            logger.debug(
                "[review_service] PullRequest found — #%d id=%d",
                pr_number, pr.id,
            )
            return pr

        pr = PullRequest(
            repo_id=repo.id,
            pr_number=pr_number,
            title=title or f"PR #{pr_number}",
            author=author or "unknown",
            branch=branch or "unknown",
            status="open",
        )
        db.add(pr)
        await db.commit()
        await db.refresh(pr)

        logger.info(
            "[review_service] PullRequest created — #%d '%s' id=%d",
            pr_number, pr.title, pr.id,
        )
        return pr

    except Exception as e:
        logger.exception(
            "[review_service] _get_or_create_pull_request failed — #%d",
            pr_number,
        )
        raise CustomException(str(e), sys)


async def _persist_review_steps(
    db: AsyncSession,
    review: Review,
    state: Dict,
) -> None:
    """
    Persist completed workflow steps to ReviewStep table.

    Skips step_names already persisted for this review (idempotent).
    Uses NODE_NAME_MAP to map state keys to step names.

    Parameters
    ----------
    db : AsyncSession
    review : Review     Parent review record.
    state : Dict        Graph state dict (from snapshot.values or ainvoke result).

    Raises
    ------
    Exception
        Re-raised after logging — caller decides how to handle.
    """
    try:
        # Fetch already-persisted step names to avoid duplicates
        result = await db.execute(
            select(ReviewStep.step_name).where(
                ReviewStep.review_id == review.id
            )
        )
        existing = {row[0] for row in result.all()}

        steps = []
        for key, step_name in NODE_NAME_MAP.items():
            if key not in state:
                continue
            if step_name in existing:
                logger.debug(
                    "[review_service] Step '%s' already persisted — skipping",
                    step_name,
                )
                continue

            val = _serialize(state[key])
            steps.append(
                ReviewStep(
                    review_id=review.id,
                    step_name=step_name,
                    status="completed",
                    input_data="{}",
                    output_data=(
                        json.dumps(val)
                        if not isinstance(val, str)
                        else val
                    ),
                )
            )
            existing.add(step_name)  # prevent duplicates within same call

        if steps:
            db.add_all(steps)
            await db.flush()
            logger.info(
                "[review_service] Persisted %d ReviewStep record(s) "
                "for review_id=%d",
                len(steps), review.id,
            )
        else:
            logger.debug(
                "[review_service] No new steps to persist for review_id=%d",
                review.id,
            )

    except Exception:
        logger.exception(
            "[review_service] _persist_review_steps failed "
            "for review_id=%d",
            review.id,
        )
        raise


# ── Core Service Functions ────────────────────────────────────────────────────

async def trigger_review(
    owner: str,
    repo: str,
    pr_number: int,
    db: AsyncSession,
) -> Review:
    """
    Trigger the full PR review pipeline.

    Runs the LangGraph workflow via astream until GraphInterrupt fires
    at the hitl_node checkpoint. Sets review.status = "pending_hitl"
    and returns immediately — does not wait for human decision.

    Flow
    ----
    1. Create Repository + PullRequest records (or fetch existing)
    2. Create Review record with status="running"
    3. Stream graph until GraphInterrupt
    4. Fetch checkpoint snapshot via aget_state()
    5. Persist ReviewStep records from snapshot
    6. Set review.status = "pending_hitl"
    7. Return review

    Failure Handling
    ----------------
    On any exception (other than GraphInterrupt):
        review.status = "failed"
        CustomException is raised to the route handler

    Parameters
    ----------
    owner : str
    repo : str
    pr_number : int
    db : AsyncSession

    Returns
    -------
    Review
        Review record with status="pending_hitl".

    Raises
    ------
    CustomException
        On any unrecoverable error.
    """
    review = None
    graph = get_review_graph()

    try:
        logger.info(
            "[trigger_review] Starting — %s/%s#%d",
            owner, repo, pr_number,
        )

        # ── DB setup ──────────────────────────────────────────────────────────
        repo_obj = await _get_or_create_repository(db, owner, repo)
        pr_obj = await _get_or_create_pull_request(db, repo_obj, pr_number)

        review = Review(
            pull_request_id=pr_obj.id,
            reviewer="ai",
            status="running",
            thread_id=str(uuid.uuid4()),
        )
        db.add(review)
        await db.commit()
        await db.refresh(review)

        logger.info(
            "[trigger_review] Review record created — "
            "id=%d thread_id=%s",
            review.id, review.thread_id,
        )

        # ── Build initial state ───────────────────────────────────────────────
        state = build_initial_state(owner, repo, pr_number)
        config = {"configurable": {"thread_id": review.thread_id}}

        # ── Stream graph ──────────────────────────────────────────────────────
        # astream raises GraphInterrupt when interrupt_before fires.
        # We catch it here to set pending_hitl status.
        interrupted = False

        try:
            logger.info(
                "[trigger_review] Starting graph stream — review_id=%d",
                review.id,
            )
            async for chunk in graph.astream(state, config=config):
                logger.debug(
                    "[trigger_review] Stream chunk — keys=%s",
                    list(chunk.keys()) if isinstance(chunk, dict) else type(chunk).__name__,
                )

        except GraphInterrupt:
            interrupted = True
            logger.info(
                "[trigger_review] GraphInterrupt caught — "
                "review_id=%d will be set to pending_hitl",
                review.id,
            )

        # ── Fetch checkpoint state ────────────────────────────────────────────
        # Get full merged state from MemorySaver checkpoint.
        # This contains all node outputs accumulated so far.
        try:
            snapshot = await graph.aget_state(config)
            full_state = snapshot.values
            logger.debug(
                "[trigger_review] Snapshot fetched — "
                "state keys=%s",
                list(full_state.keys()),
            )
        except Exception:
            logger.warning(
                "[trigger_review] Could not fetch snapshot — "
                "skipping step persistence",
                exc_info=True,
            )
            full_state = {}

        # ── Persist steps ─────────────────────────────────────────────────────
        if full_state:
            try:
                await _persist_review_steps(db, review, full_state)
            except Exception:
                logger.warning(
                    "[trigger_review] Step persistence failed — "
                    "continuing without steps",
                    exc_info=True,
                )

        # ── Set HITL status ───────────────────────────────────────────────────
        if interrupted:
            review.status = "pending_hitl"
            await db.commit()
            logger.info(
                "[trigger_review] Review paused at HITL gate — "
                "id=%d status=pending_hitl",
                review.id,
            )
        else:
            # Graph completed without interrupting — should not happen
            # with interrupt_before set, but handle defensively.
            logger.warning(
                "[trigger_review] Graph completed without GraphInterrupt — "
                "forcing pending_hitl. "
                "Check interrupt_before=['hitl_node'] is set in workflow.py",
            )
            review.status = "pending_hitl"
            await db.commit()

        return review

    except Exception as e:
        logger.exception(
            "[trigger_review] Unrecoverable error — "
            "review_id=%s",
            review.id if review else "not_created",
        )
        if review:
            try:
                review.status = "failed"
                await db.commit()
            except Exception:
                logger.warning(
                    "[trigger_review] Could not set failed status",
                    exc_info=True,
                )
        raise CustomException(str(e), sys)


async def decide_review(
    review_id: int,
    decision: str,
    db: AsyncSession,
) -> Review:
    """
    Resume the graph after a human decision.

    Validates that the review is in pending_hitl state, then resumes
    the graph via Command(resume=decision). The graph executes hitl_node
    (which reads the decision) and then verdict_node.

    Decision Values
    ---------------
    "approved"  — human approved, verdict determined by AI findings
    "rejected"  — human rejected, verdict = HUMAN_REJECTED, no GitHub comment

    Flow
    ----
    1. Fetch Review, validate status == "pending_hitl"
    2. Resume graph via Command(resume=decision)
    3. Get final state from snapshot
    4. Update PullRequest with real metadata (title, author)
    5. Persist remaining ReviewStep records
    6. Set verdict, summary, status on Review
    7. Post GitHub comment (if not rejected)
    8. Return completed review

    Parameters
    ----------
    review_id : int
    decision : str      "approved" or "rejected"
    db : AsyncSession

    Returns
    -------
    Review
        Review record with status="completed" or "rejected".

    Raises
    ------
    CustomException
        If review not found, not in pending_hitl state, or graph fails.
    """
    graph = get_review_graph()

    try:
        logger.info(
            "[decide_review] Processing decision — "
            "review_id=%d decision=%s",
            review_id, decision,
        )

        # ── Fetch and validate review ─────────────────────────────────────────
        result = await db.execute(
            select(Review).where(Review.id == review_id)
        )
        review = result.scalar_one_or_none()

        if not review:
            raise CustomException(
                f"Review {review_id} not found", sys
            )

        if review.status != "pending_hitl":
            logger.error(
                "[decide_review] Invalid state — "
                "review_id=%d expected=pending_hitl actual=%s",
                review_id, review.status,
            )
            raise CustomException(
                f"Review {review_id} is not pending HITL approval — "
                f"current status: {review.status}",
                sys,
            )

        config = {"configurable": {"thread_id": review.thread_id}}

        # ── Resume graph ──────────────────────────────────────────────────────
        # Command(resume=decision) injects the decision value into
        # the graph checkpoint. interrupt() inside hitl_node returns it.
        logger.info(
            "[decide_review] Resuming graph — "
            "review_id=%d thread_id=%s decision=%s",
            review.id, review.thread_id, decision,
        )

        final_state = await graph.ainvoke(
            Command(resume=decision),
            config=config,
        )

        logger.info(
            "[decide_review] Graph resumed and completed — "
            "review_id=%d",
            review.id,
        )

        # ── Persist steps ─────────────────────────────────────────────────────
        if final_state:
            try:
                await _persist_review_steps(db, review, final_state)
            except Exception:
                logger.warning(
                    "[decide_review] Step persistence failed — continuing",
                    exc_info=True,
                )

        # ── Update review ─────────────────────────────────────────────────────
        review.verdict = final_state.get("verdict", "")
        review.summary = final_state.get("summary", "")
        review.status = "completed" if decision == "approved" else "rejected"

        logger.info(
            "[decide_review] Review finalized — "
            "id=%d status=%s verdict=%s",
            review.id, review.status, review.verdict,
        )

        # ── Post GitHub comment ───────────────────────────────────────────────
        # Skip on rejection — HUMAN_REJECTED verdict means no comment posted.
        if review.verdict != "HUMAN_REJECTED" and review.summary:
            try:
                # Fetch PR metadata to get owner/repo/pr_number
                pr_result = await db.execute(
                    select(PullRequest).where(
                        PullRequest.id == review.pull_request_id
                    )
                )
                pr = pr_result.scalar_one_or_none()

                repo_result = await db.execute(
                    select(Repository).where(
                        Repository.id == pr.repo_id
                    )
                )
                repo = repo_result.scalar_one_or_none()

                if pr and repo:
                    gh = GitHubClient()
                    gh.post_review_comment(
                        repo.owner,
                        repo.name,
                        pr.pr_number,
                        review.summary,
                    )
                    logger.info(
                        "[decide_review] GitHub comment posted — "
                        "%s/%s#%d",
                        repo.owner, repo.name, pr.pr_number,
                    )
            except Exception:
                logger.warning(
                    "[decide_review] GitHub comment failed — "
                    "review saved, comment not posted",
                    exc_info=True,
                )
        else:
            logger.info(
                "[decide_review] Skipping GitHub comment — "
                "verdict=%s",
                review.verdict,
            )

        await db.commit()
        await db.refresh(review)
        return review

    except CustomException:
        raise

    except Exception as e:
        logger.exception(
            "[decide_review] Unrecoverable error — review_id=%d",
            review_id,
        )
        raise CustomException(str(e), sys)


# ── Query Functions ───────────────────────────────────────────────────────────

async def get_review(
    review_id: int,
    db: AsyncSession,
) -> Review:
    """
    Fetch a single Review by ID, including its ReviewStep records.

    Parameters
    ----------
    review_id : int
    db : AsyncSession

    Returns
    -------
    Review
        Review with steps eagerly loaded.

    Raises
    ------
    CustomException
        If review not found or DB error.
    """
    try:
        logger.debug("[get_review] Fetching review_id=%d", review_id)

        result = await db.execute(
            select(Review)
            .options(selectinload(Review.steps))
            .where(Review.id == review_id)
        )
        review = result.scalar_one_or_none()

        if not review:
            raise CustomException(
                f"Review {review_id} not found", sys
            )

        return review

    except CustomException:
        raise

    except Exception as e:
        logger.exception(
            "[get_review] DB error — review_id=%d", review_id,
        )
        raise CustomException(str(e), sys)


async def list_reviews(
    owner: str,
    repo: str,
    db: AsyncSession,
) -> List[Review]:
    """
    List all reviews for a specific repository, newest first.

    Parameters
    ----------
    owner : str
    repo : str
    db : AsyncSession

    Returns
    -------
    List[Review]
        Empty list if repository not found.

    Raises
    ------
    CustomException
        On DB error.
    """
    try:
        logger.debug("[list_reviews] %s/%s", owner, repo)

        repo_result = await db.execute(
            select(Repository).where(
                Repository.owner == owner,
                Repository.name == repo,
            )
        )
        repository = repo_result.scalar_one_or_none()

        if not repository:
            logger.info(
                "[list_reviews] Repository not found — %s/%s",
                owner, repo,
            )
            return []

        result = await db.execute(
            select(Review)
            .join(PullRequest, Review.pull_request_id == PullRequest.id)
            .where(PullRequest.repo_id == repository.id)
            .order_by(Review.created_at.desc())
        )
        reviews = result.scalars().all()

        logger.info(
            "[list_reviews] Found %d review(s) — %s/%s",
            len(reviews), owner, repo,
        )
        return list(reviews)

    except Exception as e:
        logger.exception("[list_reviews] DB error — %s/%s", owner, repo)
        raise CustomException(str(e), sys)


async def list_all_reviews(
    db: AsyncSession,
) -> List[Dict]:
    """
    List all reviews across all repositories — used by the dashboard.

    Returns enriched dicts with PR and repo metadata for display.

    Parameters
    ----------
    db : AsyncSession

    Returns
    -------
    List[Dict]
        Each dict contains id, status, verdict, repo, pr_number, title.

    Raises
    ------
    CustomException
        On DB error.
    """
    try:
        logger.debug("[list_all_reviews] Fetching all reviews for dashboard")

        result = await db.execute(
            select(Review, PullRequest, Repository)
            .select_from(Review)
            .join(PullRequest, Review.pull_request_id == PullRequest.id)
            .join(Repository, PullRequest.repo_id == Repository.id)
            .order_by(Review.created_at.desc())
        )

        rows = result.all()

        reviews = [
            {
                "id":         r.id,
                "status":     r.status,
                "verdict":    r.verdict,
                "thread_id":  r.thread_id,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "repo":       f"{repo.owner}/{repo.name}",
                "pr_number":  pr.pr_number,
                "title":      pr.title,
                "author":     pr.author,
                "branch":     pr.branch,
            }
            for r, pr, repo in rows
        ]

        logger.info(
            "[list_all_reviews] Returning %d review(s)", len(reviews),
        )
        return reviews

    except Exception as e:
        logger.exception("[list_all_reviews] DB error")
        raise CustomException(str(e), sys)