"""
tests/test_sandbox_client.py
=============================
Tests for app/mcp/sandbox_client.py

Two test classes:

  TestParseDiff
    Pure logic tests for _parse_diff() — no Docker, no mocks needed.
    Every branch: added lines, removed lines, context lines, headers,
    non-Python files, deleted files, new files, multi-file diffs,
    empty/malformed diffs.

  TestSandboxClientRun
    Tests run_lint() and run_tests() with DockerRunner mocked entirely.
    Verifies: correct RunType passed, correct files extracted and forwarded,
    SandboxResult returned unchanged, ParseError on bad diff,
    DockerRunner not called on parse failure, is_available() delegation.

Run all:
    pytest tests/test_sandbox_client.py -v

Run parse tests only:
    pytest tests/test_sandbox_client.py -v -k "ParseDiff"
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.mcp.sandbox_client import SandboxClient, ParseError
from app.sandbox.docker_runner import RunType, SandboxResult, SANDBOX_IMAGE
from app.core.exceptions import CustomException


# ─────────────────────────────────────────────────────────────────────────────
# Diff fixtures
# ─────────────────────────────────────────────────────────────────────────────

# Single Python file — adds lines
DIFF_SINGLE_FILE = """\
diff --git a/app/utils.py b/app/utils.py
index abc123..def456 100644
--- a/app/utils.py
+++ b/app/utils.py
@@ -1,3 +1,5 @@
 def existing():
     pass
+
+def add(x, y):
+    return x + y
"""

# Diff with both added and removed lines
DIFF_WITH_REMOVALS = """\
diff --git a/app/utils.py b/app/utils.py
--- a/app/utils.py
+++ b/app/utils.py
@@ -1,3 +1,3 @@
-def add(x,y):
+def add(x, y):
     return x + y
"""

# Two Python files changed
DIFF_TWO_FILES = """\
diff --git a/app/utils.py b/app/utils.py
--- a/app/utils.py
+++ b/app/utils.py
@@ -1,2 +1,2 @@
-def old():
+def new():
     pass
diff --git a/app/helpers.py b/app/helpers.py
--- a/app/helpers.py
+++ b/app/helpers.py
@@ -1,2 +1,2 @@
-x = 1
+x = 2
"""

# Mix of Python and non-Python files
DIFF_WITH_NON_PYTHON = """\
diff --git a/README.md b/README.md
--- a/README.md
+++ b/README.md
@@ -1,2 +1,2 @@
-Old readme
+New readme
diff --git a/app/utils.py b/app/utils.py
--- a/app/utils.py
+++ b/app/utils.py
@@ -1 +1 @@
-x = 1
+x = 2
"""

# Only a non-Python file — nothing to lint
DIFF_ONLY_NON_PYTHON = """\
diff --git a/docker-compose.yml b/docker-compose.yml
--- a/docker-compose.yml
+++ b/docker-compose.yml
@@ -1 +1 @@
-version: "3"
+version: "3.9"
"""

# File was deleted — target is /dev/null
DIFF_DELETED_FILE = """\
diff --git a/app/old.py b/app/old.py
--- a/app/old.py
+++ /dev/null
@@ -1,3 +0,0 @@
-def old():
-    pass
-
"""

# Only a deletion — no Python files to lint
DIFF_ONLY_DELETION = """\
diff --git a/app/old.py b/app/old.py
--- a/app/old.py
+++ /dev/null
@@ -1,2 +0,0 @@
-def old():
-    pass
"""

# Brand new file — source is /dev/null
DIFF_NEW_FILE = (
    "diff --git a/app/new_feature.py b/app/new_feature.py\n"
    "--- /dev/null\n"
    "+++ b/app/new_feature.py\n"
    "@@ -0,0 +1,3 @@\n"
    "+def new_feature():\n"
    "+    return True\n"
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_client() -> SandboxClient:
    return SandboxClient()


def make_sandbox_result(passed: bool = True, tool: str = "lint") -> SandboxResult:
    return SandboxResult(
        passed=passed,
        output="All checks passed." if passed else "E501 line too long",
        errors="" if passed else "E501 line too long",
        exit_code=0 if passed else 1,
        duration_ms=500,
        tool=tool,
        image=SANDBOX_IMAGE,
    )


def make_mock_runner(
    lint_passed: bool = True,
    test_passed: bool = True,
) -> MagicMock:
    """Returns a MagicMock DockerRunner with pre-set run() return values."""
    mock_runner = MagicMock()
    mock_runner.run.return_value = make_sandbox_result(
        passed=lint_passed, tool="lint"
    )
    # run() return value varies by run_type — we set a side_effect for accuracy
    def run_side_effect(files, run_type):
        if run_type == RunType.LINT:
            return make_sandbox_result(passed=lint_passed, tool="lint")
        return make_sandbox_result(passed=test_passed, tool="test")

    mock_runner.run.side_effect = run_side_effect
    return mock_runner


# ─────────────────────────────────────────────────────────────────────────────
# TestParseDiff — pure logic, no Docker
# ─────────────────────────────────────────────────────────────────────────────

class TestParseDiff:
    """
    Tests _parse_diff() in isolation — the diff-parsing algorithm.
    No mocking needed: this is a pure string-in, dict-out function.
    """

    # ── Basic extraction ─────────────────────────────────────────────────────

    def test_single_file_key_extracted(self):
        files = make_client()._parse_diff(DIFF_SINGLE_FILE)
        assert "app/utils.py" in files

    def test_added_lines_present_in_output(self):
        files = make_client()._parse_diff(DIFF_SINGLE_FILE)
        assert "def add(x, y):" in files["app/utils.py"]
        assert "return x + y"   in files["app/utils.py"]

    def test_context_lines_preserved(self):
        """Lines without +/- (context lines) must be kept in post-patch content."""
        files = make_client()._parse_diff(DIFF_SINGLE_FILE)
        assert "def existing():" in files["app/utils.py"]

    def test_removed_lines_excluded(self):
        """Lines starting with - must NOT appear — they are the pre-patch version."""
        files = make_client()._parse_diff(DIFF_WITH_REMOVALS)
        assert "add(x,y)" not in files["app/utils.py"]   # old signature
        assert "add(x, y)" in files["app/utils.py"]      # new signature

    def test_diff_header_lines_excluded(self):
        """diff --git, index, ---, +++ b/, @@ lines must not appear in content."""
        files   = make_client()._parse_diff(DIFF_SINGLE_FILE)
        content = files["app/utils.py"]
        assert "diff --git"   not in content
        assert "index abc123" not in content
        assert "@@ -1"        not in content
        assert "--- a/"       not in content
        assert "+++ b/"       not in content

    def test_returns_dict_type(self):
        result = make_client()._parse_diff(DIFF_SINGLE_FILE)
        assert isinstance(result, dict)

    def test_values_are_strings(self):
        result = make_client()._parse_diff(DIFF_SINGLE_FILE)
        for v in result.values():
            assert isinstance(v, str)

    def test_keys_have_no_leading_slash(self):
        """Extracted paths must be relative — docker_runner expects relative paths."""
        files = make_client()._parse_diff(DIFF_SINGLE_FILE)
        for key in files:
            assert not key.startswith("/")

    # ── Multi-file diffs ─────────────────────────────────────────────────────

    def test_two_python_files_both_extracted(self):
        files = make_client()._parse_diff(DIFF_TWO_FILES)
        assert "app/utils.py"   in files
        assert "app/helpers.py" in files

    def test_two_files_content_is_independent(self):
        files = make_client()._parse_diff(DIFF_TWO_FILES)
        assert "def new():" in files["app/utils.py"]
        assert "x = 2"     in files["app/helpers.py"]
        # Content must not bleed across files
        assert "x = 2"     not in files["app/utils.py"]

    def test_two_files_returns_two_keys(self):
        files = make_client()._parse_diff(DIFF_TWO_FILES)
        assert len(files) == 2

    # ── Non-Python files ─────────────────────────────────────────────────────

    def test_non_python_files_are_skipped(self):
        files = make_client()._parse_diff(DIFF_WITH_NON_PYTHON)
        assert "README.md"    not in files
        assert "app/utils.py" in files

    def test_only_non_python_raises_parse_error(self):
        with pytest.raises(ParseError):
            make_client()._parse_diff(DIFF_ONLY_NON_PYTHON)

    # ── Deleted files ────────────────────────────────────────────────────────

    def test_deleted_file_target_dev_null_is_skipped(self):
        """+++ /dev/null means file deleted — nothing to lint."""
        with pytest.raises(ParseError):
            make_client()._parse_diff(DIFF_ONLY_DELETION)

    def test_deletion_with_addition_returns_only_addition(self):
        """A diff with one deletion AND one addition — only the addition returned."""
        diff = DIFF_DELETED_FILE + DIFF_NEW_FILE
        files = make_client()._parse_diff(diff)
        assert "app/new_feature.py" in files
        assert "app/old.py"         not in files

    # ── New files ────────────────────────────────────────────────────────────

    def test_new_file_addition_fully_captured(self):
        """Brand new file (--- /dev/null) should be extracted correctly."""
        files = make_client()._parse_diff(DIFF_NEW_FILE)
        assert "app/new_feature.py" in files
        assert "def new_feature():" in files["app/new_feature.py"]

    # ── Edge cases ───────────────────────────────────────────────────────────

    def test_empty_string_raises_parse_error(self):
        with pytest.raises(ParseError):
            make_client()._parse_diff("")

    def test_whitespace_only_raises_parse_error(self):
        with pytest.raises(ParseError):
            make_client()._parse_diff("   \n\n\t  ")

    def test_non_diff_string_raises_parse_error(self):
        with pytest.raises(ParseError):
            make_client()._parse_diff("This is just a regular string")

    def test_parse_error_extends_custom_exception(self):
        """ParseError must be catchable as CustomException at the top level."""
        with pytest.raises(CustomException):
            make_client()._parse_diff("")

    def test_parse_error_is_subclass_of_custom_exception(self):
        assert issubclass(ParseError, CustomException)


# ─────────────────────────────────────────────────────────────────────────────
# TestSandboxClientRun — DockerRunner mocked
# ─────────────────────────────────────────────────────────────────────────────

class TestSandboxClientRun:
    """
    Tests for run_lint() and run_tests() with DockerRunner replaced by a mock.

    Verifies:
      - Correct RunType forwarded to DockerRunner.run()
      - Correct files forwarded (parsed from diff)
      - SandboxResult passed through unchanged
      - ParseError raised before DockerRunner is called on bad diff
      - is_available() delegates to the runner
    """

    # ── run_lint ─────────────────────────────────────────────────────────────

    def test_run_lint_calls_runner_with_lint_runtype(self):
        client         = make_client()
        mock_runner    = make_mock_runner(lint_passed=True)
        client._runner = mock_runner

        client.run_lint(DIFF_SINGLE_FILE)

        call_args = mock_runner.run.call_args
        assert call_args.kwargs["run_type"] == RunType.LINT

    def test_run_lint_passes_parsed_python_files_to_runner(self):
        client         = make_client()
        mock_runner    = make_mock_runner()
        client._runner = mock_runner

        client.run_lint(DIFF_SINGLE_FILE)

        call_args = mock_runner.run.call_args
        assert "app/utils.py" in call_args.kwargs["files"]

    def test_run_lint_returns_sandbox_result_unchanged(self):
        client         = make_client()
        expected       = make_sandbox_result(passed=True, tool="lint")
        mock_runner    = MagicMock()
        mock_runner.run.return_value = expected
        client._runner = mock_runner

        result = client.run_lint(DIFF_SINGLE_FILE)

        assert result is expected

    def test_run_lint_returns_fail_when_runner_fails(self):
        client         = make_client()
        mock_runner    = make_mock_runner(lint_passed=False)
        client._runner = mock_runner

        result = client.run_lint(DIFF_SINGLE_FILE)

        assert result.passed is False

    def test_run_lint_raises_parse_error_on_empty_diff(self):
        with pytest.raises(ParseError):
            make_client().run_lint("")

    def test_run_lint_raises_parse_error_on_non_python_only_diff(self):
        with pytest.raises(ParseError):
            make_client().run_lint(DIFF_ONLY_NON_PYTHON)

    def test_run_lint_never_calls_runner_on_parse_error(self):
        """DockerRunner must not be invoked if diff parsing fails."""
        client         = make_client()
        mock_runner    = make_mock_runner()
        client._runner = mock_runner

        with pytest.raises(ParseError):
            client.run_lint("")

        mock_runner.run.assert_not_called()

    def test_run_lint_skips_non_python_files_in_parsed_result(self):
        """README.md in the diff must not appear in the files dict sent to runner."""
        client         = make_client()
        mock_runner    = make_mock_runner()
        client._runner = mock_runner

        client.run_lint(DIFF_WITH_NON_PYTHON)

        call_kwargs = mock_runner.run.call_args.kwargs
        assert "README.md" not in call_kwargs["files"]

    # ── run_tests ────────────────────────────────────────────────────────────

    def test_run_tests_calls_runner_with_test_runtype(self):
        client         = make_client()
        mock_runner    = make_mock_runner(test_passed=True)
        client._runner = mock_runner

        client.run_tests(DIFF_SINGLE_FILE)

        call_args = mock_runner.run.call_args
        assert call_args.kwargs["run_type"] == RunType.TEST

    def test_run_tests_passes_parsed_python_files_to_runner(self):
        client         = make_client()
        mock_runner    = make_mock_runner()
        client._runner = mock_runner

        client.run_tests(DIFF_SINGLE_FILE)

        call_args = mock_runner.run.call_args
        assert "app/utils.py" in call_args.kwargs["files"]

    def test_run_tests_returns_sandbox_result_unchanged(self):
        client         = make_client()
        expected       = make_sandbox_result(passed=True, tool="test")
        mock_runner    = MagicMock()
        mock_runner.run.return_value = expected
        client._runner = mock_runner

        result = client.run_tests(DIFF_SINGLE_FILE)

        assert result is expected

    def test_run_tests_raises_parse_error_on_empty_patch(self):
        with pytest.raises(ParseError):
            make_client().run_tests("")

    def test_run_tests_never_calls_runner_on_parse_error(self):
        client         = make_client()
        mock_runner    = make_mock_runner()
        client._runner = mock_runner

        with pytest.raises(ParseError):
            client.run_tests("")

        mock_runner.run.assert_not_called()

    # ── run_lint vs run_tests use same parser ────────────────────────────────

    def test_both_methods_extract_same_files_from_same_diff(self):
        """run_lint and run_tests both call _parse_diff — verify same extraction."""
        client      = make_client()
        mock_runner = make_mock_runner()
        client._runner = mock_runner

        client.run_lint(DIFF_TWO_FILES)
        lint_files = mock_runner.run.call_args.kwargs["files"]

        client.run_tests(DIFF_TWO_FILES)
        test_files = mock_runner.run.call_args.kwargs["files"]

        assert set(lint_files.keys()) == set(test_files.keys())

    # ── is_available ─────────────────────────────────────────────────────────

    def test_is_available_returns_true_when_runner_says_true(self):
        client         = make_client()
        mock_runner    = MagicMock()
        mock_runner.is_available.return_value = True
        client._runner = mock_runner

        assert client.is_available() is True

    def test_is_available_returns_false_when_runner_says_false(self):
        client         = make_client()
        mock_runner    = MagicMock()
        mock_runner.is_available.return_value = False
        client._runner = mock_runner

        assert client.is_available() is False

    def test_is_available_delegates_to_runner(self):
        client         = make_client()
        mock_runner    = MagicMock()
        mock_runner.is_available.return_value = True
        client._runner = mock_runner

        client.is_available()

        mock_runner.is_available.assert_called_once()