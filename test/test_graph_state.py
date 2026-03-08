"""
tests/test_graph_state.py

Unit Tests for ReviewState and build_initial_state
----------------------------------------------------
Tests that:
    - ReviewState has all required fields with correct types
    - build_initial_state sets correct values for trigger inputs
    - build_initial_state sets safe defaults for all other fields
    - build_initial_state raises CustomException on invalid inputs
    - State fields can be updated as nodes would update them

Run with:
    pytest tests/test_graph_state.py -v
"""

import sys
import pytest
from pathlib import Path

# ── make sure project root is on sys.path ────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.graph.state import ReviewState, build_initial_state
from app.core.exceptions import CustomException


# ──────────────────────────────────────────────────────────────────────────────
# Shared fixture
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_state() -> ReviewState:
    """A correctly built initial state used across multiple tests."""
    return build_initial_state("Basant19", "advanced-github-code-reviewer", 7)


# ──────────────────────────────────────────────────────────────────────────────
# Test: build_initial_state — happy path
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildInitialState:

    def test_returns_a_dict(self, valid_state):
        assert isinstance(valid_state, dict)

    def test_trigger_fields_are_set_correctly(self, valid_state):
        assert valid_state["owner"]     == "Basant19"
        assert valid_state["repo"]      == "advanced-github-code-reviewer"
        assert valid_state["pr_number"] == 7

    def test_owner_is_stripped_of_whitespace(self):
        state = build_initial_state("  Basant19  ", "repo", 1)
        assert state["owner"] == "Basant19"

    def test_repo_is_stripped_of_whitespace(self):
        state = build_initial_state("owner", "  my-repo  ", 1)
        assert state["repo"] == "my-repo"

    # ── default values ────────────────────────────────────────────────────

    def test_metadata_defaults_to_empty_dict(self, valid_state):
        assert valid_state["metadata"] == {}

    def test_diff_defaults_to_empty_string(self, valid_state):
        assert valid_state["diff"] == ""

    def test_files_defaults_to_empty_list(self, valid_state):
        assert valid_state["files"] == []

    def test_issues_defaults_to_empty_list(self, valid_state):
        assert valid_state["issues"] == []

    def test_suggestions_defaults_to_empty_list(self, valid_state):
        assert valid_state["suggestions"] == []

    def test_reflection_count_defaults_to_zero(self, valid_state):
        assert valid_state["reflection_count"] == 0

    def test_verdict_defaults_to_empty_string(self, valid_state):
        assert valid_state["verdict"] == ""

    def test_summary_defaults_to_empty_string(self, valid_state):
        assert valid_state["summary"] == ""

    def test_repo_context_defaults_to_empty_string(self, valid_state):
        assert valid_state["repo_context"] == ""

    # ── all required keys are present ─────────────────────────────────────

    def test_all_required_keys_present(self, valid_state):
        required_keys = [
            "owner", "repo", "pr_number",
            "metadata", "diff", "files",
            "issues", "suggestions",
            "reflection_count",
            "verdict", "summary",
            "repo_context",
        ]
        for key in required_keys:
            assert key in valid_state, f"Missing key: {key}"


# ──────────────────────────────────────────────────────────────────────────────
# Test: build_initial_state — invalid inputs raise CustomException
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildInitialStateValidation:

    def test_empty_owner_raises_custom_exception(self):
        with pytest.raises(CustomException):
            build_initial_state("", "repo", 1)

    def test_empty_repo_raises_custom_exception(self):
        with pytest.raises(CustomException):
            build_initial_state("owner", "", 1)

    def test_pr_number_zero_raises_custom_exception(self):
        with pytest.raises(CustomException):
            build_initial_state("owner", "repo", 0)

    def test_negative_pr_number_raises_custom_exception(self):
        with pytest.raises(CustomException):
            build_initial_state("owner", "repo", -5)

    def test_non_integer_pr_number_raises_custom_exception(self):
        with pytest.raises(CustomException):
            build_initial_state("owner", "repo", "42")   # type: ignore

    def test_none_owner_raises_custom_exception(self):
        with pytest.raises(CustomException):
            build_initial_state(None, "repo", 1)         # type: ignore

    def test_none_repo_raises_custom_exception(self):
        with pytest.raises(CustomException):
            build_initial_state("owner", None, 1)        # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Test: state fields can be updated (simulates node behaviour)
# ──────────────────────────────────────────────────────────────────────────────

class TestStateUpdates:
    """
    Simulates what each LangGraph node does — reads state, returns updates.
    LangGraph merges these updates; here we do it manually to verify types.
    """

    def test_fetch_diff_node_can_update_metadata_diff_files(self, valid_state):
        # Simulate fetch_diff_node output
        valid_state["metadata"] = {
            "number": 7, "title": "Add state", "author": "Basant19",
            "description": "", "base_branch": "main",
            "head_branch": "feature/state", "state": "open",
        }
        valid_state["diff"]  = "--- app/graph/state.py (added)\n@@ -0,0 +1,10 @@"
        valid_state["files"] = [{"filename": "app/graph/state.py", "status": "added",
                                  "changes": 10, "patch": "@@ -0,0 +1,10 @@"}]

        assert valid_state["metadata"]["title"] == "Add state"
        assert "state.py" in valid_state["diff"]
        assert len(valid_state["files"]) == 1

    def test_analyze_code_node_can_update_issues_and_suggestions(self, valid_state):
        # Simulate analyze_code_node output
        valid_state["issues"]      = ["Missing type hints on line 5"]
        valid_state["suggestions"] = ["Use dataclass instead of plain dict"]

        assert len(valid_state["issues"])      == 1
        assert len(valid_state["suggestions"]) == 1

    def test_reflect_node_increments_reflection_count(self, valid_state):
        # Simulate reflect_node running twice
        valid_state["reflection_count"] += 1
        assert valid_state["reflection_count"] == 1

        valid_state["reflection_count"] += 1
        assert valid_state["reflection_count"] == 2

    def test_reflection_loop_stops_at_two(self, valid_state):
        # The conditional edge in workflow.py checks this
        valid_state["reflection_count"] = 2
        should_continue = valid_state["reflection_count"] < 2
        assert should_continue is False

    def test_verdict_node_can_set_verdict_and_summary(self, valid_state):
        # Simulate verdict_node output
        valid_state["verdict"] = "REQUEST_CHANGES"
        valid_state["summary"] = "## AI Review\n\n**Verdict:** REQUEST_CHANGES"

        assert valid_state["verdict"] == "REQUEST_CHANGES"
        assert "REQUEST_CHANGES" in valid_state["summary"]

    def test_verdict_approve_is_valid(self, valid_state):
        valid_state["verdict"] = "APPROVE"
        assert valid_state["verdict"] == "APPROVE"

    def test_repo_context_can_be_set_from_chromadb(self, valid_state):
        # Simulate vector memory injection
        valid_state["repo_context"] = "Similar pattern found in app/services/chat_service.py"
        assert "chat_service" in valid_state["repo_context"]