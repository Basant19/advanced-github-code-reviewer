"""
tests/test_graph_nodes.py

Unit Tests for LangGraph Nodes
--------------------------------
Tests all nodes with mocks — no real LLM, GitHub, ChromaDB, or Docker calls.

P1 nodes (27 tests — unchanged):
    fetch_diff_node, analyze_code_node, _parse_llm_output,
    reflect_node, verdict_node

P2 nodes (new — added below P1 tests):
    lint_node, refactor_node, validator_node

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

    P2 extension:
               SandboxClient is also module-level in nodes.py.
               Same pattern applies: patch.object(n, "sandbox_client", mock_sc)
               where mock_sc is a MagicMock with .run_lint() and .run_tests()
               returning a SandboxResult directly.
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

# P2 additions — SandboxResult needed for make_sandbox_result helper
from app.sandbox.docker_runner import SandboxResult, SANDBOX_IMAGE


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


# ── P2 helpers ────────────────────────────────────────────────────────────────

def make_sandbox_result(passed: bool = True, tool: str = "lint") -> SandboxResult:
    """Returns a SandboxResult for use in P2 node tests."""
    return SandboxResult(
        passed=passed,
        output="All checks passed." if passed else "E501 line too long",
        errors="" if passed else "E501 line too long",
        exit_code=0 if passed else 1,
        duration_ms=500,
        tool=tool,
        image=SANDBOX_IMAGE,
    )


def make_mock_sandbox_client(
    lint_passed: bool = True,
    test_passed: bool = True,
) -> MagicMock:
    """
    Returns a MagicMock that behaves like SandboxClient.
    run_lint() and run_tests() return a SandboxResult directly.
    """
    mock_sc = MagicMock()
    mock_sc.run_lint.return_value  = make_sandbox_result(passed=lint_passed, tool="lint")
    mock_sc.run_tests.return_value = make_sandbox_result(passed=test_passed, tool="test")
    return mock_sc


# ── LLM response fixtures ─────────────────────────────────────────────────────

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

LLM_PATCH_RESPONSE = """\
--- a/app/utils.py
+++ b/app/utils.py
@@ -1,2 +1,4 @@
+def add(x, y):
+    return x + y
"""


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
# P1 — Test: fetch_diff_node  (no LLM involved)
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
# P1 — Test: analyze_code_node
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
# P1 — Test: _parse_llm_output  (pure function — no mocking needed)
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
# P1 — Test: reflect_node
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
# P1 — Test: verdict_node  (deterministic — no LLM at all)
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


# ══════════════════════════════════════════════════════════════════════════════
# P2 — NEW TESTS BELOW
# lint_node, refactor_node, validator_node
# Mocking pattern: patch.object(n, "sandbox_client", mock_sc)
# Mirrors exactly how llm is mocked above — module-level object replaced
# per-test so each test is fully independent.
# ══════════════════════════════════════════════════════════════════════════════

class TestLintNode:
    """
    Tests for lint_node (P2).

    lint_node calls sandbox_client.run_lint(state["diff"]) and writes
    lint_result (SandboxResult) and lint_passed (bool) to state.

    All tests mock sandbox_client so no Docker calls are made.
    """

    def test_lint_node_sets_lint_passed_true_when_ruff_passes(self):
        from app.graph import nodes as n

        with patch.object(n, "sandbox_client", make_mock_sandbox_client(lint_passed=True)):
            result = n.lint_node(make_state())

        assert result["lint_passed"] is True

    def test_lint_node_sets_lint_passed_false_when_ruff_fails(self):
        from app.graph import nodes as n

        with patch.object(n, "sandbox_client", make_mock_sandbox_client(lint_passed=False)):
            result = n.lint_node(make_state())

        assert result["lint_passed"] is False

    def test_lint_node_writes_sandbox_result_to_state(self):
        from app.graph import nodes as n

        with patch.object(n, "sandbox_client", make_mock_sandbox_client(lint_passed=True)):
            result = n.lint_node(make_state())

        assert result["lint_result"] is not None
        assert isinstance(result["lint_result"], SandboxResult)

    def test_lint_node_calls_run_lint_with_diff(self):
        """lint_node must pass state["diff"] — not state["patch"] — to run_lint."""
        from app.graph import nodes as n

        mock_sc = make_mock_sandbox_client()
        state   = make_state(diff="diff --git a/app/utils.py b/app/utils.py\n+def new(): pass")

        with patch.object(n, "sandbox_client", mock_sc):
            n.lint_node(state)

        mock_sc.run_lint.assert_called_once_with(state["diff"])

    def test_lint_node_never_calls_run_tests(self):
        """lint_node checks ruff only — pytest is not invoked at this stage."""
        from app.graph import nodes as n

        mock_sc = make_mock_sandbox_client()

        with patch.object(n, "sandbox_client", mock_sc):
            n.lint_node(make_state())

        mock_sc.run_tests.assert_not_called()

    def test_lint_node_raises_custom_exception_on_sandbox_error(self):
        """Docker infrastructure failures must surface as CustomException."""
        from app.graph import nodes as n
        from app.sandbox.docker_runner import SandboxError

        mock_sc = MagicMock()
        mock_sc.run_lint.side_effect = SandboxError("Docker not running", sys)

        with patch.object(n, "sandbox_client", mock_sc):
            with pytest.raises(CustomException):
                n.lint_node(make_state())

    def test_lint_node_result_has_tool_lint(self):
        from app.graph import nodes as n

        with patch.object(n, "sandbox_client", make_mock_sandbox_client(lint_passed=True)):
            result = n.lint_node(make_state())

        assert result["lint_result"].tool == "lint"

    def test_lint_node_result_output_populated_on_failure(self):
        from app.graph import nodes as n

        with patch.object(n, "sandbox_client", make_mock_sandbox_client(lint_passed=False)):
            result = n.lint_node(make_state())

        assert result["lint_result"].output != ""


class TestRefactorNode:
    """
    Tests for refactor_node (P2).

    refactor_node receives state["diff"] + state["issues"] + state["lint_result"]
    and calls Gemini (llm) to generate a corrective patch string.
    The patch is written to state["patch"].

    All tests mock llm via patch.object(n, "llm", mock_llm).
    """

    def test_refactor_node_writes_patch_string_to_state(self):
        from app.graph import nodes as n

        state = make_state(
            issues=["Missing type hints"],
            lint_result=make_sandbox_result(passed=False, tool="lint"),
        )

        with patch.object(n, "llm", make_mock_llm(LLM_PATCH_RESPONSE)):
            result = n.refactor_node(state)

        assert isinstance(result["patch"], str)
        assert len(result["patch"]) > 0

    def test_refactor_node_calls_llm_exactly_once(self):
        from app.graph import nodes as n

        mock_llm = make_mock_llm(LLM_PATCH_RESPONSE)
        state    = make_state(
            issues=["Missing type hints"],
            lint_result=make_sandbox_result(passed=False, tool="lint"),
        )

        with patch.object(n, "llm", mock_llm):
            n.refactor_node(state)

        mock_llm.invoke.assert_called_once()

    def test_refactor_node_prompt_contains_issues(self):
        """Gemini must be told what the issues are so it can fix them."""
        from app.graph import nodes as n

        mock_llm = make_mock_llm(LLM_PATCH_RESPONSE)
        state    = make_state(
            issues=["Missing type hints on function fetch"],
            suggestions=[],
            lint_result=make_sandbox_result(passed=False),
        )

        with patch.object(n, "llm", mock_llm):
            n.refactor_node(state)

        prompt_text = str(mock_llm.invoke.call_args)
        assert "Missing type hints" in prompt_text

    def test_refactor_node_prompt_contains_lint_output(self):
        """The ruff error output must be in the prompt so Gemini sees it."""
        from app.graph import nodes as n

        lint_result = make_sandbox_result(passed=False, tool="lint")
        mock_llm    = make_mock_llm(LLM_PATCH_RESPONSE)
        state       = make_state(
            issues=["E501"],
            lint_result=lint_result,
        )

        with patch.object(n, "llm", mock_llm):
            n.refactor_node(state)

        prompt_text = str(mock_llm.invoke.call_args)
        assert "E501" in prompt_text

    def test_refactor_node_raises_custom_exception_on_llm_failure(self):
        from app.graph import nodes as n

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("Gemini timeout")

        with patch.object(n, "llm", mock_llm):
            with pytest.raises(CustomException):
                n.refactor_node(make_state(
                    issues=["Bug"],
                    lint_result=make_sandbox_result(passed=False),
                ))

    def test_refactor_node_patch_contains_diff_markers(self):
        from app.graph import nodes as n

        with patch.object(n, "llm", make_mock_llm(LLM_PATCH_RESPONSE)):
            result = n.refactor_node(make_state(
                issues=["Missing type hints"],
                lint_result=make_sandbox_result(passed=False),
            ))

        assert "+++" in result["patch"]

    def test_refactor_node_does_not_call_sandbox_client(self):
        """refactor_node is LLM-only — no Docker sandbox at this step."""
        from app.graph import nodes as n

        mock_sc = make_mock_sandbox_client()

        with patch.object(n, "llm", make_mock_llm(LLM_PATCH_RESPONSE)), \
             patch.object(n, "sandbox_client", mock_sc):
            n.refactor_node(make_state(
                issues=["Bug"],
                lint_result=make_sandbox_result(passed=False),
            ))

        mock_sc.run_lint.assert_not_called()
        mock_sc.run_tests.assert_not_called()


class TestValidatorNode:
    """
    Tests for validator_node (P2).

    validator_node calls sandbox_client.run_tests(state["patch"]) — running
    ruff + pytest on the patch generated by refactor_node. It writes
    validation_result (SandboxResult) to state.

    Loop control:
      - Tests pass → reflection_count unchanged → workflow exits to verdict_node
      - Tests fail → reflection_count += 1     → workflow loops to refactor_node

    All tests mock sandbox_client via patch.object(n, "sandbox_client", mock_sc).
    """

    def test_validator_node_writes_validation_result_to_state(self):
        from app.graph import nodes as n

        state = make_state(patch=LLM_PATCH_RESPONSE, reflection_count=0)

        with patch.object(n, "sandbox_client", make_mock_sandbox_client(test_passed=True)):
            result = n.validator_node(state)

        assert isinstance(result["validation_result"], SandboxResult)

    def test_validator_node_calls_run_tests_with_patch(self):
        """validator_node must pass state["patch"] — not state["diff"] — to run_tests."""
        from app.graph import nodes as n

        mock_sc = make_mock_sandbox_client(test_passed=True)
        state   = make_state(patch=LLM_PATCH_RESPONSE, reflection_count=0)

        with patch.object(n, "sandbox_client", mock_sc):
            n.validator_node(state)

        mock_sc.run_tests.assert_called_once_with(state["patch"])

    def test_validator_node_never_calls_run_lint(self):
        """validator_node runs ruff+pytest together via run_tests — never run_lint."""
        from app.graph import nodes as n

        mock_sc = make_mock_sandbox_client(test_passed=True)

        with patch.object(n, "sandbox_client", mock_sc):
            n.validator_node(make_state(patch=LLM_PATCH_RESPONSE, reflection_count=0))

        mock_sc.run_lint.assert_not_called()

    def test_validator_node_does_not_increment_count_on_pass(self):
        """When tests pass, reflection_count must stay unchanged."""
        from app.graph import nodes as n

        state = make_state(patch=LLM_PATCH_RESPONSE, reflection_count=1)

        with patch.object(n, "sandbox_client", make_mock_sandbox_client(test_passed=True)):
            result = n.validator_node(state)

        assert result["reflection_count"] == 1

    def test_validator_node_increments_count_on_fail(self):
        """When tests fail, reflection_count += 1 so the workflow loops to refactor."""
        from app.graph import nodes as n

        state = make_state(patch=LLM_PATCH_RESPONSE, reflection_count=0)

        with patch.object(n, "sandbox_client", make_mock_sandbox_client(test_passed=False)):
            result = n.validator_node(state)

        assert result["reflection_count"] == 1

    def test_validator_node_raises_custom_exception_on_sandbox_error(self):
        from app.graph import nodes as n
        from app.sandbox.docker_runner import SandboxError

        mock_sc = MagicMock()
        mock_sc.run_tests.side_effect = SandboxError("Docker timeout", sys)

        with patch.object(n, "sandbox_client", mock_sc):
            with pytest.raises(CustomException):
                n.validator_node(make_state(patch=LLM_PATCH_RESPONSE, reflection_count=0))

    def test_validator_node_result_passed_true_on_pass(self):
        from app.graph import nodes as n

        with patch.object(n, "sandbox_client", make_mock_sandbox_client(test_passed=True)):
            result = n.validator_node(make_state(patch=LLM_PATCH_RESPONSE, reflection_count=0))

        assert result["validation_result"].passed is True

    def test_validator_node_result_passed_false_on_fail(self):
        from app.graph import nodes as n

        with patch.object(n, "sandbox_client", make_mock_sandbox_client(test_passed=False)):
            result = n.validator_node(make_state(patch=LLM_PATCH_RESPONSE, reflection_count=0))

        assert result["validation_result"].passed is False

    def test_validator_node_result_has_tool_test(self):
        from app.graph import nodes as n

        with patch.object(n, "sandbox_client", make_mock_sandbox_client(test_passed=True)):
            result = n.validator_node(make_state(patch=LLM_PATCH_RESPONSE, reflection_count=0))

        assert result["validation_result"].tool == "test"

    # ── Workflow routing logic — verified as pure conditionals ────────────────
    # These tests document the exact condition the workflow uses to decide
    # whether to loop back to refactor_node or exit to verdict_node.
    # They are pure logic tests — no node call, no mock needed.

    def test_routing_loops_when_failed_and_under_max_iterations(self):
        state = make_state(
            reflection_count=1,
            validation_result=make_sandbox_result(passed=False, tool="test"),
        )
        should_loop = (
            not state["validation_result"].passed
            and state["reflection_count"] < 3
        )
        assert should_loop is True

    def test_routing_exits_when_tests_pass(self):
        state = make_state(
            reflection_count=1,
            validation_result=make_sandbox_result(passed=True, tool="test"),
        )
        should_loop = (
            not state["validation_result"].passed
            and state["reflection_count"] < 3
        )
        assert should_loop is False

    def test_routing_exits_at_max_iterations_even_if_still_failing(self):
        """Hard stop at 3 iterations — prevents infinite refactor loops."""
        state = make_state(
            reflection_count=3,
            validation_result=make_sandbox_result(passed=False, tool="test"),
        )
        should_loop = (
            not state["validation_result"].passed
            and state["reflection_count"] < 3
        )
        assert should_loop is False