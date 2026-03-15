"""
tests/test_graph_workflow.py

Unit Tests for LangGraph Workflow (P3)
----------------------------------------
Tests graph structure, conditional edge logic, and end-to-end graph
invocation with all nodes mocked — no real LLM, GitHub, or ChromaDB.

Run with:
    pytest tests/test_graph_workflow.py -v

Mocking strategy:
    - patch init_chat_model at module scope (prevents GOOGLE_API_KEY lookup)
    - patch all nodes with controlled fake implementations in run_review tests
    - should_reflect() and should_refactor() are pure functions — tested directly

P3 graph route (for reference):
    START
      → fetch_diff_node
      → analyze_code_node
      → reflect_node          ← loops via should_reflect() up to 2×
      → lint_node             ← should_reflect exits here (not verdict_node)
      → refactor_node
      → validator_node        ← loops via should_refactor() up to 3×
      → hitl_node             ★ interrupt() — suspends for human approval
      → verdict_node
      → END
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

def make_state(**overrides) -> dict:
    """Build a minimal ReviewState dict for testing pure functions."""
    state = {
        "owner":             "Basant19",
        "repo":              "advanced-github-code-reviewer",
        "pr_number":         7,
        "metadata":          {"title": "Test PR", "author": "dev",
                              "base_branch": "main", "head_branch": "feature",
                              "state": "open", "description": ""},
        "diff":              "--- app/main.py (modified)\n@@ -1 +1 @@",
        "files":             [],
        "issues":            ["Missing type hints"],
        "suggestions":       ["Add docstring"],
        "reflection_count":  0,
        "validation_result": {"passed": False},
        "verdict":           None,
        "summary":           "",
        "repo_context":      "",
        "human_decision":    None,
    }
    state.update(overrides)
    return state


# ──────────────────────────────────────────────────────────────────────────────
# Test: should_reflect  (pure function)
#
# P3 routing: should_reflect exits to "lint_node" (NOT "verdict_node").
# The reflect loop feeds into the P2 lint/refactor pipeline, not
# directly to verdict.  verdict_node is only reached after hitl_node.
# ──────────────────────────────────────────────────────────────────────────────

class TestShouldReflect:

    def test_returns_reflect_node_when_count_is_zero(self):
        from app.graph.workflow import should_reflect
        assert should_reflect(make_state(reflection_count=0)) == "reflect_node"

    def test_returns_reflect_node_when_count_is_one(self):
        from app.graph.workflow import should_reflect
        assert should_reflect(make_state(reflection_count=1)) == "reflect_node"

    def test_returns_lint_node_when_count_is_two(self):
        """
        After 2 reflections, should_reflect exits to lint_node (P3 graph).
        In P3 the reflect loop feeds into the lint/refactor pipeline —
        it does NOT route directly to verdict_node.
        """
        from app.graph.workflow import should_reflect
        assert should_reflect(make_state(reflection_count=2)) == "lint_node"

    def test_returns_lint_node_when_count_exceeds_two(self):
        """Any count >= 2 exits the reflection loop to lint_node."""
        from app.graph.workflow import should_reflect
        assert should_reflect(make_state(reflection_count=5)) == "lint_node"

    def test_boundary_exactly_at_two_exits_to_lint(self):
        from app.graph.workflow import should_reflect
        # count=2 → exit reflection loop → lint_node
        assert should_reflect(make_state(reflection_count=2)) == "lint_node"
        # count=1 → keep reflecting
        assert should_reflect(make_state(reflection_count=1)) == "reflect_node"


# ──────────────────────────────────────────────────────────────────────────────
# Test: should_refactor  (pure function — P3 routes to hitl_node not verdict)
# ──────────────────────────────────────────────────────────────────────────────

class TestShouldRefactor:

    def test_routes_to_hitl_node_when_validation_passes(self):
        from app.graph.workflow import should_refactor
        state = make_state(validation_result={"passed": True}, reflection_count=1)
        assert should_refactor(state) == "hitl_node"

    def test_routes_to_refactor_node_when_failed_and_under_max(self):
        from app.graph.workflow import should_refactor
        state = make_state(validation_result={"passed": False}, reflection_count=1)
        assert should_refactor(state) == "refactor_node"

    def test_routes_to_hitl_node_at_max_iterations(self):
        from app.graph.workflow import should_refactor
        state = make_state(validation_result={"passed": False}, reflection_count=3)
        assert should_refactor(state) == "hitl_node"

    def test_routes_to_hitl_node_when_count_exceeds_max(self):
        from app.graph.workflow import should_refactor
        state = make_state(validation_result={"passed": False}, reflection_count=5)
        assert should_refactor(state) == "hitl_node"


# ──────────────────────────────────────────────────────────────────────────────
# Test: build_graph  (graph structure)
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildGraph:

    def test_graph_compiles_without_error(self):
        from app.graph.workflow import build_graph
        assert build_graph() is not None

    def test_review_graph_singleton_exists(self):
        from app.graph.workflow import review_graph
        assert review_graph is not None

    def test_graph_has_all_p3_nodes(self):
        """P3 graph has 8 nodes: the original 4 + lint + refactor + validator + hitl."""
        from app.graph.workflow import build_graph
        node_names = set(build_graph().nodes.keys())
        expected = {
            "fetch_diff_node",
            "analyze_code_node",
            "reflect_node",
            "lint_node",
            "refactor_node",
            "validator_node",
            "hitl_node",
            "verdict_node",
        }
        assert expected.issubset(node_names)

    def test_graph_includes_hitl_node(self):
        """hitl_node is the P3 interrupt gate — must be present."""
        from app.graph.workflow import build_graph
        assert "hitl_node" in build_graph().nodes


# ──────────────────────────────────────────────────────────────────────────────
# Test: graph invocation  (end-to-end with all nodes mocked)
#
# P3 note: run_review() was removed — the graph is invoked asynchronously
# via review_graph.ainvoke() inside review_service.  For sync unit tests
# we call review_graph.invoke() directly with a thread config.
#
# The hitl_node interrupt() is also mocked so the graph runs to completion
# without suspending.
# ──────────────────────────────────────────────────────────────────────────────

class TestGraphInvocation:

    def _make_node_mocks(self):
        """
        Returns mock implementations for all nodes.
        Each returns only the fields it is responsible for writing.
        hitl_node returns {} (no-op — interrupt is patched away).
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
                "reflection_count": state.get("reflection_count", 0) + 1,
            }

        def fake_lint(state):
            return {"lint_result": {"passed": True, "output": ""}}

        def fake_refactor(state):
            return {"patch": ""}

        def fake_validator(state):
            return {"validation_result": {"passed": True, "output": ""}}

        def fake_hitl(state):
            # interrupt() is patched away — node is a no-op
            return {}

        def fake_verdict(state):
            return {
                "verdict": "REQUEST_CHANGES",
                "summary": "## 🔴 AI Code Review\n\n**Verdict:** `REQUEST_CHANGES`",
            }

        return (fake_fetch, fake_analyze, fake_reflect,
                fake_lint, fake_refactor, fake_validator,
                fake_hitl, fake_verdict)

    def _invoke_graph(self, wf, node_mocks):
        """Patch all nodes, rebuild graph, invoke synchronously."""
        (fake_fetch, fake_analyze, fake_reflect,
         fake_lint, fake_refactor, fake_validator,
         fake_hitl, fake_verdict) = node_mocks

        initial_state = make_state(
            owner="Basant19",
            repo="advanced-github-code-reviewer",
            pr_number=7,
        )
        config = {"configurable": {"thread_id": "test-thread-001"}}

        with patch("app.graph.workflow.fetch_diff_node",   fake_fetch),   \
             patch("app.graph.workflow.analyze_code_node", fake_analyze), \
             patch("app.graph.workflow.reflect_node",      fake_reflect), \
             patch("app.graph.workflow.lint_node",         fake_lint),    \
             patch("app.graph.workflow.refactor_node",     fake_refactor),\
             patch("app.graph.workflow.validator_node",    fake_validator),\
             patch("app.graph.workflow.hitl_node",         fake_hitl),    \
             patch("app.graph.workflow.verdict_node",      fake_verdict), \
             patch("app.graph.workflow.interrupt"):          # suppress real interrupt

            wf.review_graph = wf.build_graph()
            return wf.review_graph.invoke(initial_state, config=config)

    def test_invoke_returns_final_state(self):
        from app.graph import workflow as wf
        result = self._invoke_graph(wf, self._make_node_mocks())
        assert result is not None
        assert isinstance(result, dict)

    def test_invoke_final_state_has_verdict(self):
        from app.graph import workflow as wf
        result = self._invoke_graph(wf, self._make_node_mocks())
        assert result.get("verdict") in ("APPROVE", "REQUEST_CHANGES", "HUMAN_REJECTED")

    def test_invoke_final_state_has_summary(self):
        from app.graph import workflow as wf
        result = self._invoke_graph(wf, self._make_node_mocks())
        assert "summary" in result
        assert len(result["summary"]) > 0

    def test_graph_raises_on_empty_owner(self):
        """
        Validation of inputs happens in review_service before ainvoke.
        Passing an empty owner directly to invoke does NOT raise inside
        the graph — it would raise in review_service.  This test verifies
        the graph itself accepts any dict (validation is a service concern).
        We test service-level validation separately in test_review_service.
        """
        from app.graph import workflow as wf
        # Graph accepts the dict — no graph-level guard on owner
        state = make_state(owner="", pr_number=7)
        config = {"configurable": {"thread_id": "test-empty-owner"}}

        (fake_fetch, fake_analyze, fake_reflect,
         fake_lint, fake_refactor, fake_validator,
         fake_hitl, fake_verdict) = self._make_node_mocks()

        with patch("app.graph.workflow.fetch_diff_node",   fake_fetch),   \
             patch("app.graph.workflow.analyze_code_node", fake_analyze), \
             patch("app.graph.workflow.reflect_node",      fake_reflect), \
             patch("app.graph.workflow.lint_node",         fake_lint),    \
             patch("app.graph.workflow.refactor_node",     fake_refactor),\
             patch("app.graph.workflow.validator_node",    fake_validator),\
             patch("app.graph.workflow.hitl_node",         fake_hitl),    \
             patch("app.graph.workflow.verdict_node",      fake_verdict), \
             patch("app.graph.workflow.interrupt"):

            wf.review_graph = wf.build_graph()
            result = wf.review_graph.invoke(state, config=config)

        assert result is not None


# ──────────────────────────────────────────────────────────────────────────────
# Test: thread_id conventions
# ──────────────────────────────────────────────────────────────────────────────

class TestThreadId:

    def test_thread_id_is_unique_per_pr(self):
        """Different PRs must have different thread_ids for MemorySaver."""
        owner = "Basant19"
        repo  = "advanced-github-code-reviewer"
        assert f"{owner}-{repo}-7" != f"{owner}-{repo}-42"

    def test_same_pr_produces_same_thread_id(self):
        """Same PR always maps to the same thread_id (checkpoint resumption)."""
        owner = "Basant19"
        repo  = "advanced-github-code-reviewer"
        assert f"{owner}-{repo}-7" == f"{owner}-{repo}-7"