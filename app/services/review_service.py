"""
app/services/review_service.py

Review Orchestration Service — P3 Production Version
-----------------------------------------------------
The single entry point for triggering, persisting, and retrieving PR reviews.

Responsibilities
----------------
1. Create and persist Review records in PostgreSQL
2. Run the LangGraph review workflow via astream
3. Detect HITL pause via snapshot.next — set status=pending_hitl
4. Resume the graph via Command(resume=decision) on human decision
5. Persist final verdict and summary after completion
6. Post GitHub PR comment on approval (skipped on rejection)

HITL Lifecycle
--------------
trigger_review()
    Runs graph via astream until the graph pauses at hitl_node.
    With interrupt_before=["hitl_node"], astream() stops yielding chunks
    silently — NO Python exception is raised. The pause is detected by
    calling graph.aget_state() after the stream ends and checking
    snapshot.next. If snapshot.next is non-empty, the graph is paused.
    Sets review.status = "pending_hitl" and returns immediately.

decide_review()
    Called by POST /reviews/id/{id}/decision.
    Validates review.status == "pending_hitl".
    Resumes graph via Command(resume=decision).
    Graph executes hitl_node → verdict_node → END.
    Sets review.status = "completed" or "rejected".

GraphInterrupt Detection — IMPORTANT
-------------------------------------
With interrupt_before=["hitl_node"] in workflow.py compile():
    - astream() does NOT raise GraphInterrupt as a Python exception.
    - astream() simply stops yielding chunks when the graph pauses.
    - The correct detection method is snapshot.next after the stream ends.
    - snapshot.next == ["hitl_node"] means graph is paused at HITL gate.
    - snapshot.next == [] means graph completed normally (should not happen
      with interrupt_before set unless error routing bypassed hitl_node).

GraphInterrupt is still imported for forward compatibility — it may be
raised in edge cases by older LangGraph versions or during forced resumes.

Command(resume=) Format
-----------------------
LangGraph expects Command(resume=value) where value is what
interrupt() returns inside hitl_node. We pass the decision string
directly: Command(resume="approved") or Command(resume="rejected").

State Snapshot
--------------
After the graph pauses, graph.aget_state(config) returns the
full merged state at the checkpoint. This is used to persist
ReviewStep records even on the interrupted (pending_hitl) path.

Error Handling
--------------
All service functions wrap exceptions in CustomException.
GraphInterrupt is imported and caught as a fallback safety net.
The primary interrupt detection is via snapshot.next (not exception).
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
# Both "issues" and "suggestions" map to "analyze_code" — the idempotency
# check in _persist_review_steps prevents duplicate step insertion.

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

    Handles SandboxResult dataclass objects, Pydantic models,
    primitives (str, int, float, bool, None), and lists recursively.
    Falls back to str() for any unknown type — never raises an exception.

    Parameters
    ----------
    data : Any
        Value from graph state dict.

    Returns
    -------
    Any
        JSON-serializable representation of the input value.
    """
    try:
        if isinstance(data, list):
            return [_serialize(item) for item in data]

        # SandboxResult dataclass — detected by duck typing
        if hasattr(data, "passed") and hasattr(data, "output"):
            return {
                "passed":      data.passed,
                "output":      data.output,
                "errors":      getattr(data, "errors", ""),
                "exit_code":   data.exit_code,
                "duration_ms": data.duration_ms,
                "tool":        getattr(data, "tool", ""),
            }

        # Pydantic model
        if hasattr(data, "dict"):
            return data.dict()

        # Primitives — already JSON serializable
        if isinstance(data, (str, int, float, bool)) or data is None:
            return data

        # Unknown type — safe fallback
        return str(data)

    except Exception:
        logger.warning(
            "[review_service] _serialize: unexpected failure for type=%s "
            "— falling back to str()",
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

    Uses SELECT before INSERT to avoid race conditions on concurrent
    webhook triggers for the same repository.

    Parameters
    ----------
    db : AsyncSession
        Active async database session.
    owner : str
        GitHub username or organization name.
    name : str
        Repository name (without owner prefix).

    Returns
    -------
    Repository
        Existing or newly created repository record.

    Raises
    ------
    CustomException
        Wraps any database error with file/line context.
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
                "[review_service] _get_or_create_repository: found — "
                "%s/%s id=%d",
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
            "[review_service] _get_or_create_repository: created — "
            "%s/%s id=%d",
            owner, name, repo.id,
        )
        return repo

    except Exception as e:
        logger.exception(
            "[review_service] _get_or_create_repository: DB error — "
            "%s/%s error=%s",
            owner, name, str(e),
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

    On first trigger (before fetch_diff_node runs), title/author/branch
    are unknown. Stub values are inserted and updated on subsequent triggers
    once real metadata is available from fetch_diff_node.

    Parameters
    ----------
    db : AsyncSession
        Active async database session.
    repo : Repository
        Parent repository record.
    pr_number : int
        GitHub Pull Request number (positive integer).
    title : str, optional
        PR title — uses "PR #{pr_number}" stub if not provided.
    author : str, optional
        PR author GitHub username — uses "unknown" if not provided.
    branch : str, optional
        Head branch name — uses "unknown" if not provided.

    Returns
    -------
    PullRequest
        Existing (possibly updated) or newly created PR record.

    Raises
    ------
    CustomException
        Wraps any database error with file/line context.
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
            # Update metadata if real values are now available
            updated = False
            if title and pr.title != title:
                pr.title = title
                updated = True
            if author and pr.author != author:
                pr.author = author
                updated = True
            if updated:
                await db.commit()
                logger.debug(
                    "[review_service] _get_or_create_pull_request: "
                    "metadata updated — #%d '%s'",
                    pr_number, pr.title,
                )
            else:
                logger.debug(
                    "[review_service] _get_or_create_pull_request: found — "
                    "#%d id=%d",
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
            "[review_service] _get_or_create_pull_request: created — "
            "#%d '%s' id=%d",
            pr_number, pr.title, pr.id,
        )
        return pr

    except Exception as e:
        logger.exception(
            "[review_service] _get_or_create_pull_request: DB error — "
            "#%d error=%s",
            pr_number, str(e),
        )
        raise CustomException(str(e), sys)


async def _persist_review_steps(
    db: AsyncSession,
    review: Review,
    state: Dict,
) -> None:
    """
    Persist completed workflow steps to the ReviewStep table.

    Idempotent — step_names already persisted for this review are skipped.
    Called twice per review lifecycle:
        1. After trigger_review() pauses at HITL gate (partial state)
        2. After decide_review() completes the graph (full state)

    The idempotency check ensures the second call only inserts new steps
    (verdict, summary) without duplicating steps from the first call
    (diff, issues, lint_result, etc).

    Parameters
    ----------
    db : AsyncSession
        Active async database session.
    review : Review
        Parent review record — review.id used to scope the query.
    state : Dict
        Graph state dict from snapshot.values or ainvoke() result.
        Keys are matched against NODE_NAME_MAP for persistence.

    Raises
    ------
    Exception
        Re-raised after logging — caller decides how to handle failure.
        Callers wrap this in a try/except with a warning log and continue.
    """
    try:
        # Fetch already-persisted step names to enforce idempotency
        result = await db.execute(
            select(ReviewStep.step_name).where(
                ReviewStep.review_id == review.id
            )
        )
        existing = {row[0] for row in result.all()}

        logger.debug(
            "[review_service] _persist_review_steps: "
            "review_id=%d existing_steps=%s",
            review.id, existing,
        )

        steps = []
        for key, step_name in NODE_NAME_MAP.items():
            if key not in state:
                logger.debug(
                    "[review_service] _persist_review_steps: "
                    "key '%s' not in state — skipping",
                    key,
                )
                continue

            if step_name in existing:
                logger.debug(
                    "[review_service] _persist_review_steps: "
                    "step '%s' already persisted — skipping",
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
                "[review_service] _persist_review_steps: "
                "persisted %d new step(s) for review_id=%d",
                len(steps), review.id,
            )
        else:
            logger.debug(
                "[review_service] _persist_review_steps: "
                "no new steps to persist for review_id=%d",
                review.id,
            )

    except Exception:
        logger.exception(
            "[review_service] _persist_review_steps: failed "
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
    Trigger the full PR review pipeline and pause at the HITL gate.

    Runs the LangGraph workflow via astream(). With interrupt_before=
    ["hitl_node"] set in workflow.py, the stream stops silently when the
    graph reaches hitl_node. No Python exception is raised. The pause is
    detected by calling graph.aget_state() and checking snapshot.next.

    If snapshot.next is non-empty (contains "hitl_node"), the graph is
    paused and the review is set to pending_hitl. If snapshot.next is
    empty, the graph completed without pausing (should not happen with
    interrupt_before set — logged as a warning and still set to pending_hitl
    for safety, requiring human decision before verdict is issued).

    Flow
    ----
    1. Fetch or create Repository + PullRequest records
    2. Create Review record with status="running"
    3. Stream graph via astream() until it pauses or completes
    4. Detect pause via snapshot.next (primary method)
    5. Persist ReviewStep records from snapshot.values
    6. Set review.status = "pending_hitl"
    7. Return review to route handler

    Parameters
    ----------
    owner : str
        GitHub username or organization name.
    repo : str
        Repository name.
    pr_number : int
        GitHub Pull Request number.
    db : AsyncSession
        Active async database session from get_db() dependency.

    Returns
    -------
    Review
        Review record with status="pending_hitl".

    Raises
    ------
    CustomException
        On any unrecoverable error (DB failure, GitHub API error, etc).
        review.status is set to "failed" before raising.
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
        # With interrupt_before=["hitl_node"], astream() stops yielding
        # when the graph pauses at hitl_node. No exception is raised.
        # We still wrap in try/except GraphInterrupt as a safety net for
        # edge cases in older LangGraph versions.
        logger.info(
            "[trigger_review] Starting graph stream — review_id=%d",
            review.id,
        )

        try:
            async for chunk in graph.astream(state, config=config):
                logger.debug(
                    "[trigger_review] Stream chunk received — "
                    "keys=%s review_id=%d",
                    list(chunk.keys()) if isinstance(chunk, dict)
                    else type(chunk).__name__,
                    review.id,
                )

        except GraphInterrupt:
            # Safety net — GraphInterrupt as Python exception (edge case)
            logger.info(
                "[trigger_review] GraphInterrupt raised as exception "
                "(edge case) — review_id=%d",
                review.id,
            )

        logger.info(
            "[trigger_review] Graph stream ended — "
            "fetching checkpoint snapshot review_id=%d",
            review.id,
        )

        # ── Detect interrupt via snapshot.next ────────────────────────────────
        # This is the PRIMARY and CORRECT interrupt detection method.
        # snapshot.next contains nodes waiting to execute.
        # Non-empty → graph paused. Empty → graph completed normally.
        try:
            snapshot = await graph.aget_state(config)
            full_state = snapshot.values
            next_nodes = list(snapshot.next)

            logger.info(
                "[trigger_review] Snapshot fetched — "
                "review_id=%d next_nodes=%s state_keys=%s",
                review.id,
                next_nodes,
                list(full_state.keys()),
            )

        except Exception:
            logger.warning(
                "[trigger_review] Could not fetch snapshot — "
                "defaulting to pending_hitl review_id=%d",
                review.id,
                exc_info=True,
            )
            full_state = {}
            next_nodes = ["hitl_node"]  # assume paused if snapshot fails

        # ── Persist ReviewStep records ────────────────────────────────────────
        if full_state:
            try:
                await _persist_review_steps(db, review, full_state)
            except Exception:
                logger.warning(
                    "[trigger_review] Step persistence failed — "
                    "continuing without steps review_id=%d",
                    review.id,
                    exc_info=True,
                )

        # ── Set review status based on snapshot.next ──────────────────────────
        interrupted = bool(next_nodes)

        if interrupted:
            logger.info(
                "[trigger_review] Graph paused at HITL gate — "
                "next_nodes=%s review_id=%d setting status=pending_hitl",
                next_nodes, review.id,
            )
            review.status = "pending_hitl"
            await db.commit()

        else:
            # Graph completed without pausing — unexpected with interrupt_before
            # set. Log clearly and still force pending_hitl for safety.
            logger.warning(
                "[trigger_review] Graph completed without pausing — "
                "next_nodes=[] review_id=%d. "
                "This is unexpected with interrupt_before=['hitl_node']. "
                "Forcing status=pending_hitl to require human decision.",
                review.id,
            )
            review.status = "pending_hitl"
            await db.commit()

        logger.info(
            "[trigger_review] Complete — "
            "review_id=%d status=%s interrupted=%s",
            review.id, review.status, interrupted,
        )

        return review

    except Exception as e:
        logger.exception(
            "[trigger_review] Unrecoverable error — "
            "review_id=%s error=%s",
            review.id if review else "not_created",
            str(e),
        )
        if review:
            try:
                review.status = "failed"
                await db.commit()
                logger.info(
                    "[trigger_review] Review marked as failed — id=%d",
                    review.id,
                )
            except Exception:
                logger.warning(
                    "[trigger_review] Could not mark review as failed — "
                    "id=%d",
                    review.id,
                    exc_info=True,
                )
        raise CustomException(str(e), sys)


async def decide_review(
    review_id: int,
    decision: str,
    db: AsyncSession,
) -> Review:
    """
    Resume the LangGraph graph after a human approve or reject decision.

    Validates the review is in pending_hitl state, then calls
    graph.ainvoke(Command(resume=decision)) to resume from the MemorySaver
    checkpoint. The graph executes hitl_node (which reads the decision via
    interrupt() return value) and then verdict_node, producing the final
    verdict and summary.

    After completion:
    - verdict and summary are persisted to the Review record
    - status is set to "completed" (approved) or "rejected" (rejected)
    - GitHub PR comment is posted unless verdict is HUMAN_REJECTED

    Decision Values
    ---------------
    "approved"  — graph continues, verdict determined by AI findings
                  (APPROVE if no issues, REQUEST_CHANGES if issues found)
    "rejected"  — verdict set to HUMAN_REJECTED, no GitHub comment posted

    Parameters
    ----------
    review_id : int
        ID of the Review record to resume.
    decision : str
        Human decision: "approved" or "rejected".
    db : AsyncSession
        Active async database session from get_db() dependency.

    Returns
    -------
    Review
        Updated review record with status="completed" or "rejected",
        verdict and summary populated.

    Raises
    ------
    CustomException
        If review not found, not in pending_hitl state, or graph fails.
        Re-raises CustomException from validation checks unchanged.
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
            logger.error(
                "[decide_review] Review not found — review_id=%d",
                review_id,
            )
            raise CustomException(
                f"Review {review_id} not found", sys
            )

        if review.status != "pending_hitl":
            logger.error(
                "[decide_review] Invalid review state — "
                "review_id=%d expected=pending_hitl actual=%s",
                review_id, review.status,
            )
            raise CustomException(
                f"Review {review_id} is not pending HITL approval — "
                f"current status: {review.status}. "
                f"Only reviews with status='pending_hitl' can be approved "
                f"or rejected.",
                sys,
            )

        config = {"configurable": {"thread_id": review.thread_id}}

        logger.info(
            "[decide_review] Resuming graph — "
            "review_id=%d thread_id=%s decision=%s",
            review.id, review.thread_id, decision,
        )

        # ── Resume graph via Command(resume=decision) ─────────────────────────
        # Command(resume=decision) injects the decision string into the
        # MemorySaver checkpoint. When hitl_node executes, interrupt()
        # returns the injected decision string.
        # ainvoke() is used here (not astream) since we want the final
        # merged state dict returned synchronously after completion.
        final_state = await graph.ainvoke(
            Command(resume=decision),
            config=config,
        )

        logger.info(
            "[decide_review] Graph completed — "
            "review_id=%d verdict=%s",
            review.id,
            final_state.get("verdict", "unknown"),
        )

        # ── Persist remaining ReviewStep records ──────────────────────────────
        # The first call in trigger_review() persisted steps up to lint.
        # This call persists verdict and summary (added after HITL resume).
        # Idempotency in _persist_review_steps prevents any duplicate inserts.
        if final_state:
            try:
                await _persist_review_steps(db, review, final_state)
            except Exception:
                logger.warning(
                    "[decide_review] Step persistence failed — "
                    "continuing without updated steps review_id=%d",
                    review.id,
                    exc_info=True,
                )

        # ── Update review record ──────────────────────────────────────────────
        review.verdict = final_state.get("verdict", "")
        review.summary = final_state.get("summary", "")
        review.status = "completed" if decision == "approved" else "rejected"

        logger.info(
            "[decide_review] Review finalized — "
            "id=%d status=%s verdict=%s summary_chars=%d",
            review.id,
            review.status,
            review.verdict,
            len(review.summary) if review.summary else 0,
        )

        # ── Post GitHub comment ───────────────────────────────────────────────
        # Skip posting if verdict is HUMAN_REJECTED — the human chose not
        # to publish this review to the PR.
        if review.verdict != "HUMAN_REJECTED" and review.summary:
            try:
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
                        "%s/%s#%d review_id=%d",
                        repo.owner, repo.name, pr.pr_number, review.id,
                    )
                else:
                    logger.warning(
                        "[decide_review] Could not post GitHub comment — "
                        "pr or repo not found review_id=%d",
                        review.id,
                    )

            except Exception:
                logger.warning(
                    "[decide_review] GitHub comment failed — "
                    "review saved, comment not posted review_id=%d",
                    review.id,
                    exc_info=True,
                )
        else:
            logger.info(
                "[decide_review] Skipping GitHub comment — "
                "verdict=%s review_id=%d",
                review.verdict, review.id,
            )

        await db.commit()
        await db.refresh(review)

        logger.info(
            "[decide_review] Complete — "
            "review_id=%d status=%s verdict=%s",
            review.id, review.status, review.verdict,
        )

        return review

    except CustomException:
        # Re-raise validation errors unchanged — do not wrap them again
        raise

    except Exception as e:
        logger.exception(
            "[decide_review] Unrecoverable error — "
            "review_id=%d error=%s",
            review_id, str(e),
        )
        raise CustomException(str(e), sys)


# ── Query Functions ───────────────────────────────────────────────────────────

async def get_review(
    review_id: int,
    db: AsyncSession,
) -> Review:
    """
    Fetch a single Review by ID with ReviewStep records eagerly loaded.

    Used by GET /reviews/id/{review_id} and GET /reviews/{id}/status.
    Steps are loaded via selectinload to avoid N+1 query issues.

    Parameters
    ----------
    review_id : int
        ID of the review to fetch.
    db : AsyncSession
        Active async database session.

    Returns
    -------
    Review
        Review record with steps eagerly loaded.

    Raises
    ------
    CustomException
        If review not found (404-equivalent) or DB error.
    """
    try:
        logger.debug(
            "[get_review] Fetching — review_id=%d", review_id,
        )

        result = await db.execute(
            select(Review)
            .options(selectinload(Review.steps))
            .where(Review.id == review_id)
        )
        review = result.scalar_one_or_none()

        if not review:
            logger.warning(
                "[get_review] Not found — review_id=%d", review_id,
            )
            raise CustomException(
                f"Review {review_id} not found", sys
            )

        logger.debug(
            "[get_review] Found — review_id=%d status=%s steps=%d",
            review_id, review.status, len(review.steps),
        )

        return review

    except CustomException:
        raise

    except Exception as e:
        logger.exception(
            "[get_review] DB error — review_id=%d error=%s",
            review_id, str(e),
        )
        raise CustomException(str(e), sys)


async def list_reviews(
    owner: str,
    repo: str,
    db: AsyncSession,
) -> List[Review]:
    """
    List all reviews for a specific repository, newest first.

    Returns an empty list if the repository has no records in the DB
    rather than raising an error — allows the dashboard to handle
    new repos gracefully.

    Parameters
    ----------
    owner : str
        GitHub username or organization name.
    repo : str
        Repository name.
    db : AsyncSession
        Active async database session.

    Returns
    -------
    List[Review]
        Reviews ordered by created_at descending.
        Empty list if repository not found.

    Raises
    ------
    CustomException
        On DB error (not on empty result).
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
                "[list_reviews] Repository not found — "
                "%s/%s returning empty list",
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
        logger.exception(
            "[list_reviews] DB error — %s/%s error=%s",
            owner, repo, str(e),
        )
        raise CustomException(str(e), sys)


async def list_all_reviews(
    db: AsyncSession,
) -> List[Dict]:
    """
    List all reviews across all repositories — used by the Streamlit dashboard.

    Returns enriched dicts with PR and repository metadata to avoid
    additional API calls from the dashboard for display purposes.

    Parameters
    ----------
    db : AsyncSession
        Active async database session.

    Returns
    -------
    List[Dict]
        Each dict contains: id, status, verdict, thread_id, created_at,
        repo (owner/name), pr_number, title, author, branch.
        Ordered by created_at descending (newest first).

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
        logger.exception(
            "[list_all_reviews] DB error — error=%s", str(e),
        )
        raise CustomException(str(e), sys)