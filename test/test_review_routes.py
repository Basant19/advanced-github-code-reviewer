"""
tests/test_review_routes.py

Unit Tests for Review API Routes
-----------------------------------
Tests all three review endpoints using FastAPI TestClient.
Service layer is fully mocked — no DB, no LangGraph, no GitHub.

Run with:
    pytest tests/test_review_routes.py -v

What is mocked:
    - trigger_review()   → returns a controlled Review mock
    - get_review()       → returns a controlled Review mock or raises
    - list_reviews()     → returns a list of Review mocks
    - get_db()           → yields a MagicMock session
"""

import sys
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from fastapi import FastAPI

from app.api.routes.review import router
from app.api.deps import get_db
from app.core.exceptions import CustomException

# ──────────────────────────────────────────────────────────────────────────────
# Test app setup
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI()
app.include_router(router)


def override_get_db():
    yield MagicMock()


app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

NOW = datetime.now(timezone.utc)


def make_mock_review(
    id:        int = 1,
    pr_number: int = 7,
    status:    str = "completed",
    verdict:   str = "REQUEST_CHANGES",
    summary:   str = "##  AI Code Review",
    steps:     list = None,
) -> MagicMock:
    r              = MagicMock()
    r.id           = id
    r.pr_number    = pr_number
    r.status       = status
    r.verdict      = verdict
    r.summary      = summary
    r.created_at   = NOW
    r.completed_at = NOW
    r.steps        = steps or []
    return r


def make_mock_step(
    id:          int  = 1,
    step_name:   str  = "fetch_diff",
    input_data:  dict = None,
    output_data: dict = None,
) -> MagicMock:
    s              = MagicMock()
    s.id           = id
    s.step_name    = step_name
    s.input_data   = input_data  or {"pr_number": 7}
    s.output_data  = output_data or {"files_changed": 3}
    return s


# ──────────────────────────────────────────────────────────────────────────────
# Test: POST /reviews/trigger
# ──────────────────────────────────────────────────────────────────────────────

class TestTriggerReviewRoute:

    def test_returns_202_on_success(self):
        mock_review = make_mock_review()

        with patch("app.api.routes.review.trigger_review",
                   return_value=mock_review):
            response = client.post("/reviews/trigger", json={
                "owner": "Basant19", "repo": "my-repo", "pr_number": 7
            })

        assert response.status_code == 202

    def test_response_contains_review_id(self):
        mock_review = make_mock_review(id=42)

        with patch("app.api.routes.review.trigger_review",
                   return_value=mock_review):
            response = client.post("/reviews/trigger", json={
                "owner": "Basant19", "repo": "my-repo", "pr_number": 7
            })

        assert response.json()["review_id"] == 42

    def test_response_contains_verdict(self):
        mock_review = make_mock_review(verdict="APPROVE")

        with patch("app.api.routes.review.trigger_review",
                   return_value=mock_review):
            response = client.post("/reviews/trigger", json={
                "owner": "Basant19", "repo": "my-repo", "pr_number": 7
            })

        assert response.json()["verdict"] == "APPROVE"

    def test_response_contains_status(self):
        mock_review = make_mock_review(status="completed")

        with patch("app.api.routes.review.trigger_review",
                   return_value=mock_review):
            response = client.post("/reviews/trigger", json={
                "owner": "Basant19", "repo": "my-repo", "pr_number": 7
            })

        assert response.json()["status"] == "completed"

    def test_response_contains_message(self):
        mock_review = make_mock_review()

        with patch("app.api.routes.review.trigger_review",
                   return_value=mock_review):
            response = client.post("/reviews/trigger", json={
                "owner": "Basant19", "repo": "my-repo", "pr_number": 7
            })

        assert "message" in response.json()
        assert "Basant19" in response.json()["message"]

    def test_returns_500_on_service_error(self):
        with patch("app.api.routes.review.trigger_review",
                   side_effect=CustomException("Workflow failed", sys)):
            response = client.post("/reviews/trigger", json={
                "owner": "Basant19", "repo": "my-repo", "pr_number": 7
            })

        assert response.status_code == 500

    def test_returns_422_for_missing_fields(self):
        """Pydantic validation should reject incomplete request body."""
        response = client.post("/reviews/trigger", json={
            "owner": "Basant19"   # missing repo and pr_number
        })
        assert response.status_code == 422


# ──────────────────────────────────────────────────────────────────────────────
# Test: GET /reviews/{owner}/{repo}
# ──────────────────────────────────────────────────────────────────────────────

class TestListReviewsRoute:

    def test_returns_200_on_success(self):
        with patch("app.api.routes.review.list_reviews", return_value=[]):
            response = client.get("/reviews/Basant19/my-repo")

        assert response.status_code == 200

    def test_returns_list_of_reviews(self):
        mock_reviews = [make_mock_review(id=1), make_mock_review(id=2)]

        with patch("app.api.routes.review.list_reviews",
                   return_value=mock_reviews):
            response = client.get("/reviews/Basant19/my-repo")

        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2

    def test_returns_empty_list_when_no_reviews(self):
        with patch("app.api.routes.review.list_reviews", return_value=[]):
            response = client.get("/reviews/Basant19/unknown-repo")

        assert response.json() == []

    def test_each_review_has_required_fields(self):
        mock_reviews = [make_mock_review(id=1, pr_number=7, status="completed")]

        with patch("app.api.routes.review.list_reviews",
                   return_value=mock_reviews):
            response = client.get("/reviews/Basant19/my-repo")

        item = response.json()[0]
        assert "id"         in item
        assert "pr_number"  in item
        assert "status"     in item

    def test_returns_500_on_service_error(self):
        with patch("app.api.routes.review.list_reviews",
                   side_effect=CustomException("DB error", sys)):
            response = client.get("/reviews/Basant19/my-repo")

        assert response.status_code == 500

    def test_owner_and_repo_are_passed_to_service(self):
        with patch("app.api.routes.review.list_reviews",
                   return_value=[]) as mock_svc:
            client.get("/reviews/MyOwner/my-repo")

        call_kwargs = mock_svc.call_args[1]
        assert call_kwargs["owner"] == "MyOwner"
        assert call_kwargs["repo"]  == "my-repo"


# ──────────────────────────────────────────────────────────────────────────────
# Test: GET /reviews/{review_id}
# ──────────────────────────────────────────────────────────────────────────────

class TestGetReviewRoute:

    def test_returns_200_on_success(self):
        mock_review = make_mock_review(id=1)

        with patch("app.api.routes.review.get_review",
                   return_value=mock_review):
            response = client.get("/reviews/1")

        assert response.status_code == 200

    def test_response_contains_review_fields(self):
        mock_review = make_mock_review(id=1, pr_number=7,
                                        verdict="APPROVE", status="completed")

        with patch("app.api.routes.review.get_review",
                   return_value=mock_review):
            response = client.get("/reviews/1")

        data = response.json()
        assert data["id"]        == 1
        assert data["pr_number"] == 7
        assert data["verdict"]   == "APPROVE"
        assert data["status"]    == "completed"

    def test_response_contains_steps(self):
        steps = [
            make_mock_step(id=1, step_name="fetch_diff"),
            make_mock_step(id=2, step_name="analyze_code"),
            make_mock_step(id=3, step_name="reflect"),
            make_mock_step(id=4, step_name="verdict"),
        ]
        mock_review = make_mock_review(id=1, steps=steps)

        with patch("app.api.routes.review.get_review",
                   return_value=mock_review):
            response = client.get("/reviews/1")

        assert len(response.json()["steps"]) == 4

    def test_step_names_are_correct(self):
        steps = [
            make_mock_step(id=1, step_name="fetch_diff"),
            make_mock_step(id=2, step_name="verdict"),
        ]
        mock_review = make_mock_review(id=1, steps=steps)

        with patch("app.api.routes.review.get_review",
                   return_value=mock_review):
            response = client.get("/reviews/1")

        step_names = [s["step_name"] for s in response.json()["steps"]]
        assert "fetch_diff" in step_names
        assert "verdict"    in step_names

    def test_returns_404_when_not_found(self):
        with patch("app.api.routes.review.get_review",
                   side_effect=CustomException("Review id=999 not found", sys)):
            response = client.get("/reviews/999")

        assert response.status_code == 404

    def test_returns_500_on_unexpected_service_error(self):
        with patch("app.api.routes.review.get_review",
                   side_effect=CustomException("DB connection lost", sys)):
            response = client.get("/reviews/1")

        assert response.status_code == 500

    def test_review_id_is_passed_to_service(self):
        mock_review = make_mock_review(id=7)

        with patch("app.api.routes.review.get_review",
                   return_value=mock_review) as mock_svc:
            client.get("/reviews/7")

        call_kwargs = mock_svc.call_args[1]
        assert call_kwargs["review_id"] == 7

    def test_returns_empty_steps_list_when_no_steps(self):
        mock_review = make_mock_review(id=1, steps=[])

        with patch("app.api.routes.review.get_review",
                   return_value=mock_review):
            response = client.get("/reviews/1")

        assert response.json()["steps"] == []