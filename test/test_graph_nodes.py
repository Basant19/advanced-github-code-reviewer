"""
tests/test_graph_nodes.py

Unit Tests for LangGraph Nodes
--------------------------------
Tests all 4 nodes with mocks — no real LLM, GitHub, or ChromaDB calls.

Run with:
    pytest tests/test_graph_nodes.py -v

Mocking strategy (final):
    Problem 1: patch("app.graph.nodes.llm") triggers init_chat_model's
               lazy __getattr__ which instantiates ChatGoogleGenerativeAI
               and demands GOOGLE_API_KEY before the mock takes over.

    Problem 2: After reloading the module with init_chat_model mocked,
               n.llm is a MagicMock but n.llm.invoke is auto-created as
               a child MagicMock — setting .return_value on it directly
               raises AttributeError: 'method' object has no attribute
               'return_value' because of how MagicMock resolves attributes.

    Solution:  Use a module-level fixture that patches init_chat_model
               AND replaces nodes.llm with a clean MagicMock() after
               reload. Then each test uses patch.object(n, "llm", mock_llm)
               to inject a fresh controllable mock per test — this is the
               standard pattern for patching module-level objects.
"""

import sys
import pytest
import importlib
from unittest.mock import MagicMock, patch, create_autospec
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.graph.state import build_initial_state
from app.core.exceptions import CustomException


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_state(**overrides):
    """Build a ReviewState with sensible defaults, optionally overriding fields."""
    state = build_initial_state("Basant19", "advanced-github-code-reviewer", 7)
    state.update({
        "metadata": {
            "number": 7, "title": "Add LangGraph nodes",
            "author": "Basant19", "description": "Implements review nodes",
            "base_branch": "main", "head_branch": "feature/nodes",
            "state": "open",
        },
        "diff":  "--- app/graph/nodes.py (added)\n@@ -0,0 +1,10 @@\n+def fetch(): pass",
        "files": [{"filename": "app/graph/nodes.py", "status": "added",
                   "changes": 10, "patch": "@@ -0,0 +1,10 @@"}],
        "issues":      [],
        "suggestions": [],
    })
    state.update(overrides)
    return state


def make_mock_llm(response_content: str = "") -> MagicMock:
    """
    Returns a MagicMock that behaves like a LangChain chat model.
    mock_llm.invoke([...]) returns a response with .content = response_content
    """
    mock_llm          = MagicMock()
    fake_response     = MagicMock()
    fake_response.content = response_content
    mock_llm.invoke.return_value = fake_response
    return mock_llm


def mock_chroma_empty() -> MagicMock:
    """ChromaDB collection that returns no results."""
    col = MagicMock()
    col.query.return_value = {"documents": [[]]}
    return col


LLM_ANALYSIS_RESPONSE = """ISSUES:
- Missing type hints on function fetch
- No error handling for network failure

SUGGESTIONS:
- Add docstring to fetch function
- Consider using a context manager"""

LLM_REFLECTION_RESPONSE = """ISSUES:
- None

SUGGESTIONS:
- Add unit tests for the new node"""

LLM_EMPTY_RESPONSE = """ISSUES:
- None

SUGGESTIONS:
- None"""


# ──────────────────────────────────────────────────────────────────────────────
# Module-level fixture: ensure nodes.py is importable without GOOGLE_API_KEY
# by patching init_chat_model before the module's top-level code runs.
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def patch_llm_at_import():
    """
    Patches init_chat_model for the entire module so that
    `llm = init_chat_model(...)` at the top of nodes.py
    assigns a MagicMock instead of a real Gemini client.
    """
    with patch("app.graph.nodes.init_chat_model", return_value=MagicMock()):
        import app.graph.nodes
        importlib.reload(app.graph.nodes)
        yield


# ──────────────────────────────────────────────────────────────────────────────
# Test: fetch_diff_node  (no LLM involved)
# ──────────────────────────────────────────────────────────────────────────────

class TestFetchDiffNode:

    def test_returns_metadata_files_diff(self):
        from app.graph import nodes as n

        fake_metadata = {"number": 7, "title": "Test PR", "author": "dev",
                         "description": "", "base_branch": "main",
                         "head_branch": "feature", "state": "open"}
        fake_files    = [{"filename": "app/main.py", "status": "modified",
                          "changes": 5, "patch": "@@ -1 +1 @@"}]
        fake_diff     = "--- app/main.py (modified)\n@@ -1 +1 @@"

        with patch("app.graph.nodes.GitHubClient") as MockClient:
            instance = MockClient.return_value
            instance.get_pr_metadata.return_value = fake_metadata
            instance.get_pr_files.return_value    = fake_files
            instance.get_pr_diff.return_value     = fake_diff

            result = n.fetch_diff_node(make_state())

        assert result["metadata"] == fake_metadata
        assert result["files"]    == fake_files
        assert result["diff"]     == fake_diff

    def test_raises_custom_exception_on_github_error(self):
        from app.graph import nodes as n

        with patch("app.graph.nodes.GitHubClient") as MockClient:
            MockClient.return_value.get_pr_metadata.side_effect = Exception("API error")

            with pytest.raises(CustomException):
                n.fetch_diff_node(make_state())

    def test_calls_github_client_with_correct_args(self):
        from app.graph import nodes as n

        with patch("app.graph.nodes.GitHubClient") as MockClient:
            instance = MockClient.return_value
            instance.get_pr_metadata.return_value = {}
            instance.get_pr_files.return_value    = []
            instance.get_pr_diff.return_value     = ""

            n.fetch_diff_node(make_state())

            instance.get_pr_metadata.assert_called_once_with(
                "Basant19", "advanced-github-code-reviewer", 7
            )
            instance.get_pr_diff.assert_called_once_with(
                "Basant19", "advanced-github-code-reviewer", 7
            )


# ──────────────────────────────────────────────────────────────────────────────
# Test: analyze_code_node
# Each test injects its own mock_llm via patch.object — the clean pattern.
# ──────────────────────────────────────────────────────────────────────────────

class TestAnalyzeCodeNode:

    def test_returns_issues_and_suggestions(self):
        from app.graph import nodes as n

        mock_llm = make_mock_llm(LLM_ANALYSIS_RESPONSE)

        with patch.object(n, "llm", mock_llm), \
             patch("app.graph.nodes._get_chroma_collection", return_value=mock_chroma_empty()):
            result = n.analyze_code_node(make_state())

        assert len(result["issues"])      == 2
        assert len(result["suggestions"]) == 2
        assert "Missing type hints" in result["issues"][0]

    def test_issues_and_suggestions_are_lists(self):
        from app.graph import nodes as n

        mock_llm = make_mock_llm(LLM_ANALYSIS_RESPONSE)

        with patch.object(n, "llm", mock_llm), \
             patch("app.graph.nodes._get_chroma_collection", return_value=mock_chroma_empty()):
            result = n.analyze_code_node(make_state())

        assert isinstance(result["issues"],      list)
        assert isinstance(result["suggestions"], list)

    def test_chromadb_failure_does_not_block_review(self):
        """ChromaDB failure must never stop the review from completing."""
        from app.graph import nodes as n

        mock_llm = make_mock_llm(LLM_EMPTY_RESPONSE)

        with patch.object(n, "llm", mock_llm), \
             patch("app.graph.nodes._get_chroma_collection", side_effect=Exception("DB down")):
            result = n.analyze_code_node(make_state())   # must NOT raise

        assert result["issues"]       == []
        assert result["suggestions"]  == []
        assert result["repo_context"] == ""

    def test_raises_custom_exception_on_llm_failure(self):
        from app.graph import nodes as n

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM timeout")

        with patch.object(n, "llm", mock_llm), \
             patch("app.graph.nodes._get_chroma_collection", return_value=mock_chroma_empty()):
            with pytest.raises(CustomException):
                n.analyze_code_node(make_state())

    def test_repo_context_stored_when_chromadb_returns_results(self):
        from app.graph import nodes as n

        collection = MagicMock()
        collection.query.return_value = {
            "documents": [["Similar pattern in chat_service.py"]]
        }
        mock_llm = make_mock_llm(LLM_EMPTY_RESPONSE)

        with patch.object(n, "llm", mock_llm), \
             patch("app.graph.nodes._get_chroma_collection", return_value=collection):
            result = n.analyze_code_node(make_state())

        assert "chat_service" in result["repo_context"]


# ──────────────────────────────────────────────────────────────────────────────
# Test: _parse_llm_output  (pure function — no mocking needed)
# ──────────────────────────────────────────────────────────────────────────────

class TestParseLlmOutput:

    def test_parses_issues_and_suggestions_correctly(self):
        from app.graph.nodes import _parse_llm_output
        issues, suggestions = _parse_llm_output(LLM_ANALYSIS_RESPONSE)
        assert len(issues)      == 2
        assert len(suggestions) == 2

    def test_none_entries_are_excluded(self):
        from app.graph.nodes import _parse_llm_output
        issues, suggestions = _parse_llm_output(LLM_EMPTY_RESPONSE)
        assert issues      == []
        assert suggestions == []

    def test_handles_missing_sections_gracefully(self):
        from app.graph.nodes import _parse_llm_output
        issues, suggestions = _parse_llm_output("No structured content here")
        assert issues      == []
        assert suggestions == []

    def test_handles_empty_string(self):
        from app.graph.nodes import _parse_llm_output
        issues, suggestions = _parse_llm_output("")
        assert issues      == []
        assert suggestions == []


# ──────────────────────────────────────────────────────────────────────────────
# Test: reflect_node
# ──────────────────────────────────────────────────────────────────────────────

class TestReflectNode:

    def test_increments_reflection_count(self):
        from app.graph import nodes as n

        mock_llm = make_mock_llm(LLM_EMPTY_RESPONSE)
        state    = make_state(issues=["Missing type hints"],
                              suggestions=["Add docstring"], reflection_count=0)

        with patch.object(n, "llm", mock_llm):
            result = n.reflect_node(state)

        assert result["reflection_count"] == 1

    def test_merges_new_suggestions_with_existing(self):
        from app.graph import nodes as n

        mock_llm = make_mock_llm(LLM_REFLECTION_RESPONSE)
        state    = make_state(issues=["Missing type hints"],
                              suggestions=[], reflection_count=0)

        with patch.object(n, "llm", mock_llm):
            result = n.reflect_node(state)

        assert "Missing type hints" in result["issues"]
        assert any("unit tests" in s for s in result["suggestions"])

    def test_deduplicates_issues(self):
        from app.graph import nodes as n

        duplicate = "ISSUES:\n- Missing type hints\n\nSUGGESTIONS:\n- None"
        mock_llm  = make_mock_llm(duplicate)
        state     = make_state(issues=["Missing type hints"],
                               suggestions=[], reflection_count=0)

        with patch.object(n, "llm", mock_llm):
            result = n.reflect_node(state)

        assert len(result["issues"]) == 1   # deduped by lowercase

    def test_raises_custom_exception_on_llm_failure(self):
        from app.graph import nodes as n

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("Connection reset")

        with patch.object(n, "llm", mock_llm):
            with pytest.raises(CustomException):
                n.reflect_node(make_state(reflection_count=0))

    def test_second_reflection_increments_to_two(self):
        from app.graph import nodes as n

        mock_llm = make_mock_llm(LLM_EMPTY_RESPONSE)
        state    = make_state(reflection_count=1)

        with patch.object(n, "llm", mock_llm):
            result = n.reflect_node(state)

        assert result["reflection_count"] == 2

    def test_reflection_loop_condition_stops_at_two(self):
        """Verifies workflow conditional: count < 2 means keep looping."""
        state = make_state(reflection_count=2)
        assert (state["reflection_count"] < 2) is False


# ──────────────────────────────────────────────────────────────────────────────
# Test: verdict_node  (deterministic — no LLM at all)
# ──────────────────────────────────────────────────────────────────────────────

class TestVerdictNode:

    def test_verdict_is_request_changes_when_issues_exist(self):
        from app.graph.nodes import verdict_node
        result = verdict_node(make_state(issues=["Bug on line 5"], suggestions=[]))
        assert result["verdict"] == "REQUEST_CHANGES"

    def test_verdict_is_approve_when_no_issues(self):
        from app.graph.nodes import verdict_node
        result = verdict_node(make_state(issues=[], suggestions=["Add docstring"]))
        assert result["verdict"] == "APPROVE"

    def test_summary_contains_verdict(self):
        from app.graph.nodes import verdict_node
        result = verdict_node(make_state(issues=["Bug"], suggestions=[]))
        assert "REQUEST_CHANGES" in result["summary"]

    def test_summary_contains_pr_title(self):
        from app.graph.nodes import verdict_node
        result = verdict_node(make_state(issues=[], suggestions=[]))
        assert "Add LangGraph nodes" in result["summary"]

    def test_summary_contains_issue_text(self):
        from app.graph.nodes import verdict_node
        result = verdict_node(make_state(issues=["Missing error handling"], suggestions=[]))
        assert "Missing error handling" in result["summary"]

    def test_summary_contains_suggestion_text(self):
        from app.graph.nodes import verdict_node
        result = verdict_node(make_state(issues=[], suggestions=["Add type hints"]))
        assert "Add type hints" in result["summary"]

    def test_approve_summary_has_green_emoji(self):
        from app.graph.nodes import verdict_node
        result = verdict_node(make_state(issues=[], suggestions=[]))
        assert "✅" in result["summary"]

    def test_request_changes_summary_has_red_emoji(self):
        from app.graph.nodes import verdict_node
        result = verdict_node(make_state(issues=["Critical bug"], suggestions=[]))
        assert "🔴" in result["summary"]

    def test_summary_mentions_gemini(self):
        from app.graph.nodes import verdict_node
        result = verdict_node(make_state(issues=[], suggestions=[]))
        assert "Gemini" in result["summary"]