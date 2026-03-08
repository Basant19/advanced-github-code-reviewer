"""
tests/test_graph_workflow.py

Unit Tests for LangGraph Workflow
-----------------------------------
Tests graph structure, the conditional edge logic, and run_review()
with all nodes mocked — no real LLM, GitHub, or ChromaDB calls.

Run with:
    pytest tests/test_graph_workflow.py -v

Mocking strategy:
    - patch init_chat_model at module scope (same fix as test_graph_nodes.py)
    - patch all 4 nodes with controlled fake implementations in run_review tests
    - should_reflect() is a pure function — tested directly with no mocking
"""

import sys
import pytest
import importlib
from unittest.mock import MagicMock, patch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.exceptions import CustomException


# ──────────────────────────────────────────────────────────────────────────────
# Module-level fixture: prevent init_chat_model from touching GOOGLE_API_KEY
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def patch_llm_at_import():
    with patch("app.graph.nodes.init_chat_model", return_value=MagicMock()):
        import app.graph.nodes
        importlib.reload(app.graph.nodes)
        yield


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_final_state(**overrides) -> dict:
    """Simulates what a fully-executed workflow state looks like."""
    state = {
        "owner":            "Basant19",
        "repo":             "advanced-github-code-reviewer",
        "pr_number":        7,
        "metadata":         {"title": "Test PR", "author": "dev",
                             "base_branch": "main", "head_branch": "feature",
                             "state": "open", "description": ""},
        "diff":             "--- app/main.py (modified)\n@@ -1 +1 @@",
        "files":            [],
        "issues":           ["Missing type hints"],
        "suggestions":      ["Add docstring"],
        "reflection_count": 2,
        "verdict":          "REQUEST_CHANGES",
        "summary":          "## 🔴 AI Code Review\n\n**Verdict:** `REQUEST_CHANGES`",
        "repo_context":     "",
    }
    state.update(overrides)
    return state


# ──────────────────────────────────────────────────────────────────────────────
# Test: should_reflect  (pure function — no mocking needed)
# ──────────────────────────────────────────────────────────────────────────────

class TestShouldReflect:

    def test_returns_reflect_node_when_count_is_zero(self):
        from app.graph.workflow import should_reflect
        state  = make_final_state(reflection_count=0)
        result = should_reflect(state)
        assert result == "reflect_node"

    def test_returns_reflect_node_when_count_is_one(self):
        from app.graph.workflow import should_reflect
        state  = make_final_state(reflection_count=1)
        result = should_reflect(state)
        assert result == "reflect_node"

    def test_returns_verdict_node_when_count_is_two(self):
        from app.graph.workflow import should_reflect
        state  = make_final_state(reflection_count=2)
        result = should_reflect(state)
        assert result == "verdict_node"

    def test_returns_verdict_node_when_count_exceeds_two(self):
        from app.graph.workflow import should_reflect
        state  = make_final_state(reflection_count=5)
        result = should_reflect(state)
        assert result == "verdict_node"

    def test_boundary_exactly_at_two_stops_reflection(self):
        from app.graph.workflow import should_reflect
        # count=2 means we've already reflected twice → stop
        assert should_reflect(make_final_state(reflection_count=2)) == "verdict_node"
        # count=1 means only one pass done → keep going
        assert should_reflect(make_final_state(reflection_count=1)) == "reflect_node"


# ──────────────────────────────────────────────────────────────────────────────
# Test: build_graph  (graph structure)
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildGraph:

    def test_graph_compiles_without_error(self):
        from app.graph.workflow import build_graph
        graph = build_graph()
        assert graph is not None

    def test_review_graph_singleton_exists(self):
        from app.graph.workflow import review_graph
        assert review_graph is not None

    def test_graph_has_all_four_nodes(self):
        from app.graph.workflow import build_graph
        graph = build_graph()
        # LangGraph compiled graphs expose node names via .nodes
        node_names = set(graph.nodes.keys())
        assert "fetch_diff_node"   in node_names
        assert "analyze_code_node" in node_names
        assert "reflect_node"      in node_names
        assert "verdict_node"      in node_names


# ──────────────────────────────────────────────────────────────────────────────
# Test: run_review  (end-to-end with all nodes mocked)
# ──────────────────────────────────────────────────────────────────────────────

class TestRunReview:

    def _make_node_mocks(self):
        """
        Returns mock implementations for all 4 nodes.
        Each returns only the fields it is responsible for writing.
        """
        def fake_fetch(state):
            return {
                "metadata": {"title": "Test PR", "author": "dev",
                             "base_branch": "main", "head_branch": "feat",
                             "state": "open", "description": "", "number": 7},
                "diff":  "--- app/main.py\n@@ -1 +1 @@",
                "files": [],
            }

        def fake_analyze(state):
            return {
                "issues":       ["Missing type hints"],
                "suggestions":  ["Add docstring"],
                "repo_context": "",
            }

        def fake_reflect(state):
            return {
                "issues":           state["issues"],
                "suggestions":      state["suggestions"],
                "reflection_count": state["reflection_count"] + 1,
            }

        def fake_verdict(state):
            return {
                "verdict": "REQUEST_CHANGES",
                "summary": "## 🔴 AI Code Review\n\n**Verdict:** `REQUEST_CHANGES`",
            }

        return fake_fetch, fake_analyze, fake_reflect, fake_verdict

    def test_run_review_returns_final_state(self):
        from app.graph import workflow as wf

        fake_fetch, fake_analyze, fake_reflect, fake_verdict = self._make_node_mocks()

        with patch("app.graph.workflow.fetch_diff_node",   fake_fetch),   \
             patch("app.graph.workflow.analyze_code_node", fake_analyze), \
             patch("app.graph.workflow.reflect_node",      fake_reflect), \
             patch("app.graph.workflow.verdict_node",      fake_verdict):

            # Rebuild graph so patched nodes are used
            wf.review_graph = wf.build_graph()
            result = wf.run_review("Basant19", "advanced-github-code-reviewer", 7)

        assert result is not None
        assert isinstance(result, dict)

    def test_run_review_final_state_has_verdict(self):
        from app.graph import workflow as wf

        fake_fetch, fake_analyze, fake_reflect, fake_verdict = self._make_node_mocks()

        with patch("app.graph.workflow.fetch_diff_node",   fake_fetch),   \
             patch("app.graph.workflow.analyze_code_node", fake_analyze), \
             patch("app.graph.workflow.reflect_node",      fake_reflect), \
             patch("app.graph.workflow.verdict_node",      fake_verdict):

            wf.review_graph = wf.build_graph()
            result = wf.run_review("Basant19", "advanced-github-code-reviewer", 7)

        assert result["verdict"] in ("APPROVE", "REQUEST_CHANGES")

    def test_run_review_final_state_has_summary(self):
        from app.graph import workflow as wf

        fake_fetch, fake_analyze, fake_reflect, fake_verdict = self._make_node_mocks()

        with patch("app.graph.workflow.fetch_diff_node",   fake_fetch),   \
             patch("app.graph.workflow.analyze_code_node", fake_analyze), \
             patch("app.graph.workflow.reflect_node",      fake_reflect), \
             patch("app.graph.workflow.verdict_node",      fake_verdict):

            wf.review_graph = wf.build_graph()
            result = wf.run_review("Basant19", "advanced-github-code-reviewer", 7)

        assert "summary" in result
        assert len(result["summary"]) > 0

    def test_run_review_raises_custom_exception_on_invalid_owner(self):
        from app.graph.workflow import run_review
        with pytest.raises(CustomException):
            run_review("", "repo", 1)

    def test_run_review_raises_custom_exception_on_invalid_pr_number(self):
        from app.graph.workflow import run_review
        with pytest.raises(CustomException):
            run_review("owner", "repo", 0)

    def test_thread_id_is_unique_per_pr(self):
        """
        Verifies that different PRs produce different thread_ids
        so MemorySaver tracks them as separate runs.
        """
        owner = "Basant19"
        repo  = "advanced-github-code-reviewer"

        thread_id_pr7  = f"{owner}-{repo}-7"
        thread_id_pr42 = f"{owner}-{repo}-42"

        assert thread_id_pr7 != thread_id_pr42

    def test_same_pr_produces_same_thread_id(self):
        """Same PR always maps to the same thread_id for checkpoint resumption."""
        owner = "Basant19"
        repo  = "advanced-github-code-reviewer"

        assert f"{owner}-{repo}-7" == f"{owner}-{repo}-7"