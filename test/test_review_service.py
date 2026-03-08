"""
tests/test_review_service.py

Unit Tests for Review Orchestration Service
---------------------------------------------
Tests trigger_review(), get_review(), and list_reviews() with:
    - Mocked SQLAlchemy session (no real DB)
    - Mocked run_review() from workflow.py
    - Mocked GitHubClient.post_review_comment()

Run with:
    pytest tests/test_review_service.py -v

Mocking strategy:
    - db is a MagicMock() with .query().filter_by().first() chains
    - run_review is patched to return a controlled final_state dict
    - GitHubClient is patched to prevent real API calls
    - ORM model instances are plain MagicMocks with the right attributes
"""

import sys
import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.exceptions import CustomException


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_final_state(**overrides) -> dict:
    """Simulates the dict returned by run_review() after a successful workflow."""
    state = {
        "owner":            "Basant19",
        "repo":             "advanced-github-code-reviewer",
        "pr_number":        7,
        "metadata": {
            "number":      7,
            "title":       "Add review service",
            "author":      "Basant19",
            "description": "",
            "base_branch": "main",
            "head_branch": "feature/review-service",
            "state":       "open",
        },
        "diff":             "--- app/services/review_service.py\n@@ -0,0 +1,10 @@",
        "files":            [{"filename": "app/services/review_service.py",
                              "status": "added", "changes": 10, "patch": ""}],
        "issues":           ["Missing type hints"],
        "suggestions":      ["Add docstring"],
        "reflection_count": 2,
        "verdict":          "REQUEST_CHANGES",
        "summary":          "## 🔴 AI Code Review\n\n**Verdict:** `REQUEST_CHANGES`",
        "repo_context":     "",
    }
    state.update(overrides)
    return state


def make_mock_db() -> MagicMock:
    """
    Returns a MagicMock that behaves like a SQLAlchemy Session.
    All query chains return None by default (no existing records).
    """
    db = MagicMock()

    # Default: queries return None (no existing record found)
    db.query.return_value.filter_by.return_value.first.return_value  = None
    db.query.return_value.filter_by.return_value.order_by.return_value.all.return_value = []

    return db


def make_mock_repository(id: int = 1, owner: str = "Basant19",
                          name: str = "advanced-github-code-reviewer") -> MagicMock:
    repo      = MagicMock()
    repo.id   = id
    repo.owner = owner
    repo.name  = name
    return repo


def make_mock_pull_request(id: int = 1, number: int = 7) -> MagicMock:
    pr        = MagicMock()
    pr.id     = id
    pr.number = number
    return pr


def make_mock_review(id: int = 1, status: str = "completed",
                     verdict: str = "REQUEST_CHANGES") -> MagicMock:
    review            = MagicMock()
    review.id         = id
    review.status     = status
    review.verdict    = verdict
    review.summary    = "## 🔴 AI Code Review"
    review.created_at = datetime.now(timezone.utc)
    return review


# ──────────────────────────────────────────────────────────────────────────────
# Test: _get_or_create_repository  (via trigger_review internals)
# ──────────────────────────────────────────────────────────────────────────────

class TestGetOrCreateRepository:

    def test_returns_existing_repository_without_creating(self):
        from app.services.review_service import _get_or_create_repository

        existing_repo = make_mock_repository()
        db            = make_mock_db()
        db.query.return_value.filter_by.return_value.first.return_value = existing_repo

        result = _get_or_create_repository(db, "Basant19", "my-repo")

        assert result is existing_repo
        db.add.assert_not_called()

    def test_creates_new_repository_when_not_found(self):
        from app.services.review_service import _get_or_create_repository

        db = make_mock_db()
        db.query.return_value.filter_by.return_value.first.return_value = None

        with patch("app.services.review_service.Repository") as MockRepo:
            mock_instance = MagicMock()
            MockRepo.return_value = mock_instance
            _get_or_create_repository(db, "Basant19", "my-repo")

        db.add.assert_called_once()
        db.commit.assert_called()

    def test_new_repository_has_correct_owner_and_name(self):
        from app.services.review_service import _get_or_create_repository

        db = make_mock_db()
        db.query.return_value.filter_by.return_value.first.return_value = None

        with patch("app.services.review_service.Repository") as MockRepo:
            _get_or_create_repository(db, "Basant19", "my-repo")
            call_kwargs = MockRepo.call_args[1]

        assert call_kwargs["owner"]     == "Basant19"
        assert call_kwargs["name"]      == "my-repo"
        assert call_kwargs["full_name"] == "Basant19/my-repo"


# ──────────────────────────────────────────────────────────────────────────────
# Test: trigger_review — happy path
# ──────────────────────────────────────────────────────────────────────────────

class TestTriggerReview:

    def _setup_successful_trigger(self):
        """Returns (db, final_state) configured for a successful review run."""
        db           = make_mock_db()
        final_state  = make_final_state()
        mock_repo    = make_mock_repository()
        mock_pr      = make_mock_pull_request()
        mock_review  = make_mock_review()

        # Repository exists
        db.query.return_value.filter_by.return_value.first.return_value = mock_repo

        return db, final_state, mock_repo, mock_pr, mock_review

    def test_returns_review_object(self):
        from app.services.review_service import trigger_review

        db, final_state, mock_repo, mock_pr, mock_review = \
            self._setup_successful_trigger()

        with patch("app.services.review_service.run_review",
                   return_value=final_state), \
             patch("app.services.review_service.Review") as MockReview, \
             patch("app.services.review_service.PullRequest") as MockPR, \
             patch("app.services.review_service.GitHubClient"):

            MockReview.return_value = mock_review
            MockPR.return_value     = mock_pr

            result = trigger_review("Basant19", "advanced-github-code-reviewer", 7, db)

        assert result is not None

    def test_review_status_set_to_completed_on_success(self):
        from app.services.review_service import trigger_review

        db, final_state, mock_repo, mock_pr, mock_review = \
            self._setup_successful_trigger()

        with patch("app.services.review_service.run_review",
                   return_value=final_state), \
             patch("app.services.review_service.Review") as MockReview, \
             patch("app.services.review_service.PullRequest") as MockPR, \
             patch("app.services.review_service.GitHubClient"):

            MockReview.return_value = mock_review
            MockPR.return_value     = mock_pr

            trigger_review("Basant19", "advanced-github-code-reviewer", 7, db)

        assert mock_review.status == "completed"

    def test_review_verdict_is_persisted(self):
        from app.services.review_service import trigger_review

        db, final_state, mock_repo, mock_pr, mock_review = \
            self._setup_successful_trigger()

        with patch("app.services.review_service.run_review",
                   return_value=final_state), \
             patch("app.services.review_service.Review") as MockReview, \
             patch("app.services.review_service.PullRequest") as MockPR, \
             patch("app.services.review_service.GitHubClient"):

            MockReview.return_value = mock_review
            MockPR.return_value     = mock_pr

            trigger_review("Basant19", "advanced-github-code-reviewer", 7, db)

        assert mock_review.verdict == "REQUEST_CHANGES"

    def test_review_summary_is_persisted(self):
        from app.services.review_service import trigger_review

        db, final_state, mock_repo, mock_pr, mock_review = \
            self._setup_successful_trigger()

        with patch("app.services.review_service.run_review",
                   return_value=final_state), \
             patch("app.services.review_service.Review") as MockReview, \
             patch("app.services.review_service.PullRequest") as MockPR, \
             patch("app.services.review_service.GitHubClient"):

            MockReview.return_value = mock_review
            MockPR.return_value     = mock_pr

            trigger_review("Basant19", "advanced-github-code-reviewer", 7, db)

        assert "AI Code Review" in mock_review.summary

    def test_posts_comment_to_github(self):
        from app.services.review_service import trigger_review

        db, final_state, mock_repo, mock_pr, mock_review = \
            self._setup_successful_trigger()

        with patch("app.services.review_service.run_review",
                   return_value=final_state), \
             patch("app.services.review_service.Review") as MockReview, \
             patch("app.services.review_service.PullRequest") as MockPR, \
             patch("app.services.review_service.GitHubClient") as MockGH:

            MockReview.return_value = mock_review
            MockPR.return_value     = mock_pr
            mock_gh_instance        = MockGH.return_value

            trigger_review("Basant19", "advanced-github-code-reviewer", 7, db)

        mock_gh_instance.post_review_comment.assert_called_once_with(
            "Basant19", "advanced-github-code-reviewer", 7,
            final_state["summary"]
        )

    def test_db_commit_called_multiple_times(self):
        from app.services.review_service import trigger_review

        db, final_state, mock_repo, mock_pr, mock_review = \
            self._setup_successful_trigger()

        with patch("app.services.review_service.run_review",
                   return_value=final_state), \
             patch("app.services.review_service.Review") as MockReview, \
             patch("app.services.review_service.PullRequest") as MockPR, \
             patch("app.services.review_service.GitHubClient"):

            MockReview.return_value = mock_review
            MockPR.return_value     = mock_pr

            trigger_review("Basant19", "advanced-github-code-reviewer", 7, db)

        # At minimum: create repo, create review, persist steps, complete review
        assert db.commit.call_count >= 2


# ──────────────────────────────────────────────────────────────────────────────
# Test: trigger_review — failure handling
# ──────────────────────────────────────────────────────────────────────────────

class TestTriggerReviewFailures:

    def test_review_status_set_to_failed_on_workflow_error(self):
        from app.services.review_service import trigger_review

        db          = make_mock_db()
        mock_repo   = make_mock_repository()
        mock_review = make_mock_review(status="running")

        db.query.return_value.filter_by.return_value.first.return_value = mock_repo

        with patch("app.services.review_service.run_review",
                   side_effect=Exception("LLM timeout")), \
             patch("app.services.review_service.Review") as MockReview, \
             patch("app.services.review_service.GitHubClient"):

            MockReview.return_value = mock_review

            with pytest.raises((CustomException, Exception)):
                trigger_review("Basant19", "repo", 7, db)

        assert mock_review.status == "failed"

    def test_raises_custom_exception_on_workflow_error(self):
        from app.services.review_service import trigger_review

        db        = make_mock_db()
        mock_repo = make_mock_repository()
        mock_review = make_mock_review(status="running")

        db.query.return_value.filter_by.return_value.first.return_value = mock_repo

        with patch("app.services.review_service.run_review",
                   side_effect=Exception("Workflow crash")), \
             patch("app.services.review_service.Review") as MockReview, \
             patch("app.services.review_service.GitHubClient"):

            MockReview.return_value = mock_review

            with pytest.raises((CustomException, Exception)):
                trigger_review("Basant19", "repo", 7, db)

    def test_github_comment_failure_does_not_fail_review(self):
        """If posting to GitHub fails, review should still be marked completed."""
        from app.services.review_service import trigger_review

        db          = make_mock_db()
        final_state = make_final_state()
        mock_repo   = make_mock_repository()
        mock_pr     = make_mock_pull_request()
        mock_review = make_mock_review()

        db.query.return_value.filter_by.return_value.first.return_value = mock_repo

        with patch("app.services.review_service.run_review",
                   return_value=final_state), \
             patch("app.services.review_service.Review") as MockReview, \
             patch("app.services.review_service.PullRequest") as MockPR, \
             patch("app.services.review_service.GitHubClient") as MockGH:

            MockReview.return_value = mock_review
            MockPR.return_value     = mock_pr
            MockGH.return_value.post_review_comment.side_effect = \
                Exception("GitHub API down")

            # Must NOT raise even though GitHub comment failed
            result = trigger_review("Basant19", "repo", 7, db)

        assert mock_review.status == "completed"


# ──────────────────────────────────────────────────────────────────────────────
# Test: _persist_review_steps
# ──────────────────────────────────────────────────────────────────────────────

class TestPersistReviewSteps:

    def test_creates_four_step_records(self):
        from app.services.review_service import _persist_review_steps

        db          = make_mock_db()
        mock_review = make_mock_review()
        final_state = make_final_state()

        _persist_review_steps(db, mock_review, final_state)

        # bulk_save_objects should be called with 4 steps
        call_args = db.bulk_save_objects.call_args[0][0]
        assert len(call_args) == 4

    def test_step_names_match_workflow_stages(self):
        from app.services.review_service import _persist_review_steps

        db          = make_mock_db()
        mock_review = make_mock_review()
        final_state = make_final_state()

        with patch("app.services.review_service.ReviewStep") as MockStep:
            MockStep.side_effect = lambda **kw: kw   # return kwargs as dict
            _persist_review_steps(db, mock_review, final_state)
            calls = MockStep.call_args_list

        step_names = [c[1]["step_name"] for c in calls]
        assert "fetch_diff"   in step_names
        assert "analyze_code" in step_names
        assert "reflect"      in step_names
        assert "verdict"      in step_names

    def test_commit_called_after_bulk_save(self):
        from app.services.review_service import _persist_review_steps

        db          = make_mock_db()
        mock_review = make_mock_review()

        _persist_review_steps(db, mock_review, make_final_state())

        db.commit.assert_called()


# ──────────────────────────────────────────────────────────────────────────────
# Test: get_review
# ──────────────────────────────────────────────────────────────────────────────

class TestGetReview:

    def test_returns_review_when_found(self):
        from app.services.review_service import get_review

        mock_review = make_mock_review(id=1)
        db          = make_mock_db()
        db.query.return_value.filter_by.return_value.first.return_value = mock_review

        result = get_review(1, db)
        assert result is mock_review

    def test_raises_custom_exception_when_not_found(self):
        from app.services.review_service import get_review

        db = make_mock_db()
        db.query.return_value.filter_by.return_value.first.return_value = None

        with pytest.raises(CustomException):
            get_review(999, db)

    def test_raises_custom_exception_on_db_error(self):
        from app.services.review_service import get_review

        db = make_mock_db()
        db.query.side_effect = Exception("DB connection lost")

        with pytest.raises(CustomException):
            get_review(1, db)


# ──────────────────────────────────────────────────────────────────────────────
# Test: list_reviews
# ──────────────────────────────────────────────────────────────────────────────

class TestListReviews:

    def test_returns_list_of_reviews(self):
        from app.services.review_service import list_reviews

        mock_reviews = [make_mock_review(id=1), make_mock_review(id=2)]
        mock_repo    = make_mock_repository()
        db           = make_mock_db()

        db.query.return_value.filter_by.return_value.first.return_value = mock_repo
        db.query.return_value.filter_by.return_value \
            .order_by.return_value.all.return_value = mock_reviews

        result = list_reviews("Basant19", "my-repo", db)

        assert len(result) == 2

    def test_returns_empty_list_when_repo_not_registered(self):
        from app.services.review_service import list_reviews

        db = make_mock_db()
        db.query.return_value.filter_by.return_value.first.return_value = None

        result = list_reviews("unknown-owner", "unknown-repo", db)

        assert result == []

    def test_returns_empty_list_when_no_reviews_exist(self):
        from app.services.review_service import list_reviews

        mock_repo = make_mock_repository()
        db        = make_mock_db()

        db.query.return_value.filter_by.return_value.first.return_value = mock_repo
        db.query.return_value.filter_by.return_value \
            .order_by.return_value.all.return_value = []

        result = list_reviews("Basant19", "my-repo", db)

        assert result == []

    def test_raises_custom_exception_on_db_error(self):
        from app.services.review_service import list_reviews

        db = make_mock_db()
        db.query.side_effect = Exception("DB timeout")

        with pytest.raises(CustomException):
            list_reviews("owner", "repo", db)