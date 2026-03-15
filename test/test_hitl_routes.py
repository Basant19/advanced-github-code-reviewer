"""
test/test_hitl_routes.py

P3 — Tests for HITL approval/rejection endpoints.

Covers:
    POST /reviews/{id}/approve — happy path, not found, wrong status, graph failure
    POST /reviews/{id}/reject  — happy path, not found, wrong status, graph failure
    GET  /reviews/{id}/status  — happy path, not found
    _resume_graph              — state injection, error wrapping, None input
    hitl_node / should_refactor — workflow routing logic
    verdict_node P3            — human_decision routing via nodes.py directly

All DB and graph interactions are mocked — no real DB or LangGraph required.

Key patching strategy for TestGetPendingReview / TestGetReviewStatus
---------------------------------------------------------------------
hitl.py imports `select` and `Review` at MODULE LEVEL.  When the module is
already loaded, patching sys.modules has no effect — Python already resolved
the names.  Instead we patch the *module-level name* that hitl.py bound:

    patch("app.api.routes.hitl.select", _make_mock_select())

This replaces the `select` callable inside the hitl module with a chainable
mock whose return value flows through .where() → db.execute() without ever
touching SQLAlchemy's column coercion logic.  The real `select` is never
called, so passing a MagicMock Review class never raises ArgumentError.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_review_pending():
    """A Review ORM object in pending_hitl status."""
    review = MagicMock()
    review.id         = 42
    review.status     = "pending_hitl"
    review.thread_id  = "thread-abc-123"
    review.pr_number  = 7
    review.verdict    = None
    review.summary    = None
    review.created_at = "2026-03-01T10:00:00"
    review.updated_at = "2026-03-01T10:05:00"
    return review


@pytest.fixture
def mock_review_completed():
    """A Review ORM object already completed (not pending_hitl)."""
    review = MagicMock()
    review.id        = 99
    review.status    = "completed"
    review.thread_id = "thread-xyz-999"
    review.verdict   = "APPROVE"
    review.summary   = "Looks good!"
    return review


@pytest.fixture
def mock_db():
    """Async DB session mock."""
    db = AsyncMock()
    db.commit  = AsyncMock()
    db.refresh = AsyncMock()
    return db


@pytest.fixture
def graph_approved_state():
    """Final state returned by graph after approve."""
    return {
        "verdict":        "REQUEST_CHANGES",
        "summary":        "## AI Review\n\n1 issue found.",
        "human_decision": "approved",
    }


@pytest.fixture
def graph_rejected_state():
    """Final state returned by graph after reject."""
    return {
        "verdict":        "HUMAN_REJECTED",
        "summary":        "## Review Rejected by Human Reviewer",
        "human_decision": "rejected",
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_scalar_result(obj):
    """Build a mock SQLAlchemy scalar result returning obj."""
    result = MagicMock()
    result.scalar_one_or_none.return_value = obj
    return result


def _make_mock_select():
    """
    Return a chainable mock that replaces sqlalchemy.select inside hitl.py.

    hitl.py calls:  select(Review).where(Review.id == review_id)
    We need: mock_select(anything).where(anything) → a mock query object
    that db.execute() can consume (db.execute is an AsyncMock whose
    return_value is controlled separately via _make_db_with_result).
    """
    mock_query = MagicMock()
    mock_query.where.return_value = mock_query   # select(...).where(...) → same mock
    mock_select = MagicMock(return_value=mock_query)
    return mock_select


def _make_db_with_result(obj):
    """
    Build an AsyncSession mock whose execute() returns a scalar result
    containing `obj`.  Works alongside _make_mock_select() to let
    _get_pending_review / get_review_status run without a real DB.
    """
    db = AsyncMock()
    db.commit  = AsyncMock()
    db.refresh = AsyncMock()
    db.execute = AsyncMock(return_value=_make_scalar_result(obj))
    return db


def _call_verdict_node(
    human_decision,
    issues,
    suggestions,
    metadata,
    lint_result=None,
    validation_result=None,
    patch_text="",
    pr_number="?",
):
    """
    Call verdict_node directly with a fully constructed state dict.

    verdict_node lives in app.graph.nodes — it reads human_decision from state.
    lint_result / validation_result passed as None; node uses safe defaults.
    """
    from app.graph.nodes import verdict_node

    state = {
        "human_decision":    human_decision,
        "issues":            issues,
        "suggestions":       suggestions,
        "metadata":          metadata,
        "lint_result":       lint_result,
        "validation_result": validation_result,
        "patch":             patch_text,
        "pr_number":         pr_number,
    }
    return verdict_node(state)


# ── TestApproveReview ─────────────────────────────────────────────────────────

class TestApproveReview:
    """POST /reviews/{id}/approve"""

    @pytest.mark.asyncio
    async def test_approve_happy_path(
        self, mock_review_pending, mock_db, graph_approved_state
    ):
        """Approving a pending review resumes graph and returns verdict."""
        from app.api.routes.hitl import approve_review, HITLDecisionRequest

        with (
            patch("app.api.routes.hitl._get_pending_review",
                  new=AsyncMock(return_value=mock_review_pending)),
            patch("app.api.routes.hitl._resume_graph",
                  new=AsyncMock(return_value=graph_approved_state)),
        ):
            response = await approve_review(
                review_id=42,
                body=HITLDecisionRequest(reviewer_note="LGTM"),
                db=mock_db,
            )

        assert response.review_id == 42
        assert response.decision  == "approved"
        assert response.verdict   == "REQUEST_CHANGES"
        assert "approved" in response.message.lower()
        mock_db.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_approve_sets_review_status_completed(
        self, mock_review_pending, mock_db, graph_approved_state
    ):
        """After approve, review.status is set to 'completed'."""
        from app.api.routes.hitl import approve_review, HITLDecisionRequest

        with (
            patch("app.api.routes.hitl._get_pending_review",
                  new=AsyncMock(return_value=mock_review_pending)),
            patch("app.api.routes.hitl._resume_graph",
                  new=AsyncMock(return_value=graph_approved_state)),
        ):
            await approve_review(
                review_id=42, body=HITLDecisionRequest(), db=mock_db
            )

        assert mock_review_pending.status  == "completed"
        assert mock_review_pending.verdict == "REQUEST_CHANGES"

    @pytest.mark.asyncio
    async def test_approve_no_reviewer_note(
        self, mock_review_pending, mock_db, graph_approved_state
    ):
        """Approve works without a reviewer note (optional field)."""
        from app.api.routes.hitl import approve_review, HITLDecisionRequest

        with (
            patch("app.api.routes.hitl._get_pending_review",
                  new=AsyncMock(return_value=mock_review_pending)),
            patch("app.api.routes.hitl._resume_graph",
                  new=AsyncMock(return_value=graph_approved_state)),
        ):
            response = await approve_review(
                review_id=42, body=HITLDecisionRequest(), db=mock_db
            )

        assert response.decision == "approved"

    @pytest.mark.asyncio
    async def test_approve_graph_failure_raises_500(
        self, mock_review_pending, mock_db
    ):
        """If graph resume fails, a 500 HTTPException is raised."""
        from fastapi import HTTPException
        from app.api.routes.hitl import approve_review, HITLDecisionRequest
        from app.core.exceptions import CustomException

        with (
            patch("app.api.routes.hitl._get_pending_review",
                  new=AsyncMock(return_value=mock_review_pending)),
            patch("app.api.routes.hitl._resume_graph",
                  new=AsyncMock(side_effect=CustomException("Graph crashed"))),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await approve_review(
                    review_id=42, body=HITLDecisionRequest(), db=mock_db
                )

        assert exc_info.value.status_code == 500
        assert "Graph crashed" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_approve_missing_thread_id_raises_422(self, mock_db):
        """Review with no thread_id raises 422."""
        from fastapi import HTTPException
        from app.api.routes.hitl import approve_review, HITLDecisionRequest

        review_no_thread            = MagicMock()
        review_no_thread.status     = "pending_hitl"
        review_no_thread.thread_id  = None

        with patch(
            "app.api.routes.hitl._get_pending_review",
            new=AsyncMock(return_value=review_no_thread),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await approve_review(
                    review_id=42, body=HITLDecisionRequest(), db=mock_db
                )

        assert exc_info.value.status_code == 422

    @pytest.mark.asyncio
    async def test_approve_updates_summary(
        self, mock_review_pending, mock_db, graph_approved_state
    ):
        """review.summary is set to the graph final state summary."""
        from app.api.routes.hitl import approve_review, HITLDecisionRequest

        with (
            patch("app.api.routes.hitl._get_pending_review",
                  new=AsyncMock(return_value=mock_review_pending)),
            patch("app.api.routes.hitl._resume_graph",
                  new=AsyncMock(return_value=graph_approved_state)),
        ):
            await approve_review(
                review_id=42, body=HITLDecisionRequest(), db=mock_db
            )

        assert mock_review_pending.summary == graph_approved_state["summary"]


# ── TestRejectReview ──────────────────────────────────────────────────────────

class TestRejectReview:
    """POST /reviews/{id}/reject"""

    @pytest.mark.asyncio
    async def test_reject_happy_path(
        self, mock_review_pending, mock_db, graph_rejected_state
    ):
        """Rejecting a pending review resumes graph and returns HUMAN_REJECTED."""
        from app.api.routes.hitl import reject_review, HITLDecisionRequest

        with (
            patch("app.api.routes.hitl._get_pending_review",
                  new=AsyncMock(return_value=mock_review_pending)),
            patch("app.api.routes.hitl._resume_graph",
                  new=AsyncMock(return_value=graph_rejected_state)),
        ):
            response = await reject_review(
                review_id=42,
                body=HITLDecisionRequest(reviewer_note="Too many false positives"),
                db=mock_db,
            )

        assert response.review_id == 42
        assert response.decision  == "rejected"
        assert response.verdict   == "HUMAN_REJECTED"
        assert "rejected" in response.message.lower()
        mock_db.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_reject_sets_review_status_rejected(
        self, mock_review_pending, mock_db, graph_rejected_state
    ):
        """After reject, review.status is set to 'rejected'."""
        from app.api.routes.hitl import reject_review, HITLDecisionRequest

        with (
            patch("app.api.routes.hitl._get_pending_review",
                  new=AsyncMock(return_value=mock_review_pending)),
            patch("app.api.routes.hitl._resume_graph",
                  new=AsyncMock(return_value=graph_rejected_state)),
        ):
            await reject_review(
                review_id=42, body=HITLDecisionRequest(), db=mock_db
            )

        assert mock_review_pending.status  == "rejected"
        assert mock_review_pending.verdict == "HUMAN_REJECTED"

    @pytest.mark.asyncio
    async def test_reject_no_reviewer_note(
        self, mock_review_pending, mock_db, graph_rejected_state
    ):
        """Reject works without a reviewer note."""
        from app.api.routes.hitl import reject_review, HITLDecisionRequest

        with (
            patch("app.api.routes.hitl._get_pending_review",
                  new=AsyncMock(return_value=mock_review_pending)),
            patch("app.api.routes.hitl._resume_graph",
                  new=AsyncMock(return_value=graph_rejected_state)),
        ):
            response = await reject_review(
                review_id=42, body=HITLDecisionRequest(), db=mock_db
            )

        assert response.decision == "rejected"

    @pytest.mark.asyncio
    async def test_reject_graph_failure_raises_500(
        self, mock_review_pending, mock_db
    ):
        """If graph resume fails on reject, a 500 HTTPException is raised."""
        from fastapi import HTTPException
        from app.api.routes.hitl import reject_review, HITLDecisionRequest
        from app.core.exceptions import CustomException

        with (
            patch("app.api.routes.hitl._get_pending_review",
                  new=AsyncMock(return_value=mock_review_pending)),
            patch("app.api.routes.hitl._resume_graph",
                  new=AsyncMock(side_effect=CustomException("Graph crashed"))),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await reject_review(
                    review_id=42, body=HITLDecisionRequest(), db=mock_db
                )

        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_reject_missing_thread_id_raises_422(self, mock_db):
        """Review with no thread_id raises 422 on reject."""
        from fastapi import HTTPException
        from app.api.routes.hitl import reject_review, HITLDecisionRequest

        review_no_thread            = MagicMock()
        review_no_thread.status     = "pending_hitl"
        review_no_thread.thread_id  = None

        with patch(
            "app.api.routes.hitl._get_pending_review",
            new=AsyncMock(return_value=review_no_thread),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await reject_review(
                    review_id=42, body=HITLDecisionRequest(), db=mock_db
                )

        assert exc_info.value.status_code == 422


# ── TestGetPendingReview ──────────────────────────────────────────────────────

class TestGetPendingReview:
    """
    _get_pending_review helper — 404 and 409 cases.

    Patching strategy
    -----------------
    hitl.py binds `select` at module level (top-level import).
    patch("app.api.routes.hitl.select", ...) replaces that binding so
    SQLAlchemy's real select() — which rejects MagicMock as a column
    expression — is never called.
    db.execute() return value is controlled via _make_db_with_result().
    """

    @pytest.mark.asyncio
    async def test_not_found_raises_404(self):
        """Returns 404 when review does not exist."""
        from fastapi import HTTPException
        from app.api.routes.hitl import _get_pending_review

        db = _make_db_with_result(None)

        with patch("app.api.routes.hitl.select", _make_mock_select()):
            with pytest.raises(HTTPException) as exc_info:
                await _get_pending_review(review_id=999, db=db)

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_wrong_status_raises_409(self, mock_review_completed):
        """Returns 409 when review is already completed (not pending_hitl)."""
        from fastapi import HTTPException
        from app.api.routes.hitl import _get_pending_review

        db = _make_db_with_result(mock_review_completed)

        with patch("app.api.routes.hitl.select", _make_mock_select()):
            with pytest.raises(HTTPException) as exc_info:
                await _get_pending_review(review_id=99, db=db)

        assert exc_info.value.status_code == 409
        assert "pending_hitl" in exc_info.value.detail


# ── TestGetReviewStatus ───────────────────────────────────────────────────────

class TestGetReviewStatus:
    """GET /reviews/{id}/status"""

    @pytest.mark.asyncio
    async def test_status_happy_path(self, mock_review_pending):
        """Returns review status dict for a valid review."""
        from app.api.routes.hitl import get_review_status

        db = _make_db_with_result(mock_review_pending)

        with patch("app.api.routes.hitl.select", _make_mock_select()):
            result = await get_review_status(review_id=42, db=db)

        assert result["review_id"] == 42
        assert result["status"]    == "pending_hitl"
        assert result["thread_id"] == "thread-abc-123"

    @pytest.mark.asyncio
    async def test_status_not_found_raises_404(self):
        """Returns 404 when review does not exist."""
        from fastapi import HTTPException
        from app.api.routes.hitl import get_review_status

        db = _make_db_with_result(None)

        with patch("app.api.routes.hitl.select", _make_mock_select()):
            with pytest.raises(HTTPException) as exc_info:
                await get_review_status(review_id=0, db=db)

        assert exc_info.value.status_code == 404


# ── TestResumeGraph ───────────────────────────────────────────────────────────

class TestResumeGraph:
    """_resume_graph — unit tests for graph invocation helper."""

    @pytest.mark.asyncio
    async def test_resume_graph_injects_human_decision(self):
        """_resume_graph calls update_state with correct decision."""
        from app.api.routes.hitl import _resume_graph

        mock_graph              = MagicMock()
        mock_graph.update_state = MagicMock()
        mock_graph.ainvoke      = AsyncMock(
            return_value={"verdict": "APPROVE", "summary": "ok"}
        )

        with patch("app.api.routes.hitl.review_graph", mock_graph):
            result = await _resume_graph("thread-123", "approved")

        mock_graph.update_state.assert_called_once_with(
            {"configurable": {"thread_id": "thread-123"}},
            {"human_decision": "approved"},
        )
        assert result["verdict"] == "APPROVE"

    @pytest.mark.asyncio
    async def test_resume_graph_raises_custom_exception_on_failure(self):
        """_resume_graph wraps graph errors in CustomException."""
        from app.api.routes.hitl import _resume_graph
        from app.core.exceptions import CustomException

        mock_graph              = MagicMock()
        mock_graph.update_state = MagicMock()
        mock_graph.ainvoke      = AsyncMock(
            side_effect=RuntimeError("LangGraph exploded")
        )

        with patch("app.api.routes.hitl.review_graph", mock_graph):
            with pytest.raises(CustomException) as exc_info:
                await _resume_graph("thread-123", "rejected")

        assert "LangGraph exploded" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_resume_graph_passes_none_input_to_ainvoke(self):
        """Graph is resumed with None input (continues from checkpoint)."""
        from app.api.routes.hitl import _resume_graph

        mock_graph              = MagicMock()
        mock_graph.update_state = MagicMock()
        mock_graph.ainvoke      = AsyncMock(
            return_value={"verdict": "APPROVE", "summary": "x"}
        )

        with patch("app.api.routes.hitl.review_graph", mock_graph):
            await _resume_graph("thread-xyz", "approved")

        mock_graph.ainvoke.assert_awaited_once_with(
            None,
            config={"configurable": {"thread_id": "thread-xyz"}},
        )


# ── TestWorkflowHITLNode ──────────────────────────────────────────────────────

class TestWorkflowHITLNode:
    """hitl_node and should_refactor — workflow routing logic."""

    def test_hitl_node_calls_interrupt(self):
        """hitl_node must call LangGraph interrupt()."""
        from app.graph.workflow import hitl_node

        state = {"pr_number": 5, "owner": "acme", "repo": "myrepo"}

        with patch("app.graph.workflow.interrupt") as mock_interrupt:
            mock_interrupt.side_effect = Exception("interrupt called")
            with pytest.raises(Exception, match="interrupt called"):
                hitl_node(state)

        mock_interrupt.assert_called_once()

    def test_should_refactor_routes_to_hitl_on_pass(self):
        """should_refactor routes to hitl_node when validation passes."""
        from app.graph.workflow import should_refactor

        state = {
            "validation_result": {"passed": True},
            "reflection_count":  1,
        }
        assert should_refactor(state) == "hitl_node"

    def test_should_refactor_loops_on_fail(self):
        """should_refactor routes back to refactor_node when failed and count < 3."""
        from app.graph.workflow import should_refactor

        state = {
            "validation_result": {"passed": False},
            "reflection_count":  1,
        }
        assert should_refactor(state) == "refactor_node"

    def test_should_refactor_exits_loop_at_max(self):
        """should_refactor exits to hitl_node when reflection_count >= 3."""
        from app.graph.workflow import should_refactor

        state = {
            "validation_result": {"passed": False},
            "reflection_count":  3,
        }
        assert should_refactor(state) == "hitl_node"


# ── TestVerdictNodeP3 ─────────────────────────────────────────────────────────

class TestVerdictNodeP3:
    """
    verdict_node P3 changes — human_decision routing.

    Tests call verdict_node directly from app.graph.nodes (the real file
    you updated). No separate patch module needed — build_verdict logic
    lives inside verdict_node itself.

    lint_result / validation_result are passed as None so verdict_node
    uses its safe defaults (lint_passed=True when lint_result is None).
    """

    def test_rejected_verdict_is_human_rejected(self):
        """When human_decision='rejected', verdict is HUMAN_REJECTED."""
        result = _call_verdict_node(
            human_decision="rejected",
            issues=["issue1"],
            suggestions=[],
            metadata={"title": "Fix bug", "author": "dev"},
        )
        assert result["verdict"] == "HUMAN_REJECTED"
        assert "Rejected" in result["summary"]

    def test_approved_with_issues_gives_request_changes(self):
        """When human_decision='approved' with issues, verdict is REQUEST_CHANGES."""
        result = _call_verdict_node(
            human_decision="approved",
            issues=["unused variable"],
            suggestions=[],
            metadata={"title": "Add feature", "author": "dev"},
        )
        assert result["verdict"] == "REQUEST_CHANGES"
        assert "Human Approval" in result["summary"]

    def test_approved_no_issues_gives_approve(self):
        """When human_decision='approved' and no issues, verdict is APPROVE."""
        result = _call_verdict_node(
            human_decision="approved",
            issues=[],
            suggestions=["Consider renaming"],
            metadata={"title": "Refactor", "author": "dev"},
        )
        assert result["verdict"] == "APPROVE"

    def test_rejected_summary_contains_issue_count(self):
        """Rejected summary mentions how many issues were found."""
        result = _call_verdict_node(
            human_decision="rejected",
            issues=["i1", "i2", "i3"],
            suggestions=[],
            metadata={"title": "PR", "author": "dev"},
        )
        assert "3" in result["summary"]

    def test_none_human_decision_defaults_to_approved_path(self):
        """When human_decision is None, verdict_node still produces a verdict."""
        result = _call_verdict_node(
            human_decision=None,
            issues=[],
            suggestions=[],
            metadata={"title": "PR", "author": "dev"},
        )
        assert result["verdict"] in ("APPROVE", "REQUEST_CHANGES")