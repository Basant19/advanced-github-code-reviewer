"""
app/mcp/sandbox_client.py
=========================
MCP wrapper between LangGraph nodes and DockerRunner.

RESPONSIBILITY
--------------
This file sits between nodes.py and docker_runner.py. It does two things:
  1. Parse a unified diff string into {filename: content} that docker_runner
     can write to disk and lint/test.
  2. Call DockerRunner.run() and return the SandboxResult to the node.

This file knows NOTHING about:
  - LangGraph state         (that is nodes.py's job)
  - Docker SDK              (that is docker_runner.py's job)
  - review records          (that is review_service.py's job)

WHY THIS LAYER EXISTS (MCP pattern)
------------------------------------
nodes.py should not know HOW the sandbox works — only WHAT to ask for.
docker_runner.py should not know what a diff is — only HOW to run a container.
sandbox_client.py is the translator between the two.

This is the Model Context Protocol pattern: a standardised interface that
the agent calls, with the implementation detail hidden behind it. If we
swap Docker for a different sandbox later, only this file changes.

CALL CHAIN
----------
  nodes.py
    lint_node      → sandbox_client.run_lint(diff)   → SandboxResult
    validator_node → sandbox_client.run_tests(patch)  → SandboxResult
      → docker_runner.py  write files → run container → SandboxResult

HOW DIFF PARSING WORKS
-----------------------
A GitHub unified diff looks like this:

    diff --git a/app/utils.py b/app/utils.py
    --- a/app/utils.py
    +++ b/app/utils.py
    @@ -1,3 +1,3 @@
    -def add(x,y):
    +def add(x, y):
         return x + y

We need the POST-PATCH version of each file to lint it.
To reconstruct it we take:
  - Lines starting with "+" (added lines)  → strip the leading "+"
  - Lines starting with " " (context lines) → keep as-is
  - Lines starting with "-" (removed lines) → skip entirely

This gives us what the file looks like AFTER the PR changes are applied.
We do NOT need a full patch-apply engine — we are linting, not merging.

EXCEPTION HANDLING
------------------
SandboxError and subclasses from docker_runner.py are allowed to propagate
up to the calling node. The node catches them and writes the failure to
the ReviewStep audit trail in PostgreSQL.

ParseError (defined here) is raised when the diff is malformed or empty
in a way that prevents any files from being extracted.
"""

from __future__ import annotations

import re
import sys
from typing import Optional

from app.core.exceptions import CustomException
from app.core.logger import get_logger
from app.sandbox.docker_runner import DockerRunner, RunType, SandboxResult

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ParseError(CustomException):
    """
    Raised when the unified diff cannot be parsed into any files.

    This means the diff is either empty, malformed, or contains only
    binary file changes that cannot be linted as Python source.

    Extends CustomException so the error carries file + line context.
    """
    pass


# ---------------------------------------------------------------------------
# SandboxClient
# ---------------------------------------------------------------------------

class SandboxClient:
    """
    MCP client for the Docker sandbox execution environment.

    Instantiated once per review run. Reuses the same DockerRunner
    instance across lint and test calls for that review — avoids
    re-connecting to the Docker daemon unnecessarily.

    Usage (from nodes.py)
    ---------------------
        client = SandboxClient()

        # In lint_node:
        lint_result = client.run_lint(state["diff"])

        # In validator_node:
        test_result = client.run_tests(state["patch"])
    """

    def __init__(self) -> None:
        self._runner = DockerRunner()

    # ── Public API ──────────────────────────────────────────────────────────

    def run_lint(self, diff: str) -> SandboxResult:
        """
        Parse a unified diff and run ruff on the post-patch file contents.

        Called by lint_node in nodes.py with state["diff"] — the raw
        unified diff text fetched from GitHub by fetch_diff_node.

        Parameters
        ----------
        diff : str
            Unified diff string from GitHub API (get_pr_diff output).

        Returns
        -------
        SandboxResult
            passed=True  → ruff found no issues — proceed to analyze_code_node
            passed=False → ruff found issues — route directly to refactor_node
                           (fail-fast: skip Gemini reviewer, save tokens)

        Raises
        ------
        ParseError    — diff is empty or contains no parseable Python files
        SandboxError  — Docker infrastructure failure
        """
        logger.info("sandbox_client: Starting lint run")

        files = self._parse_diff(diff)
        logger.info(
            f"sandbox_client: Parsed {len(files)} file(s) from diff for lint"
        )

        result = self._runner.run(files=files, run_type=RunType.LINT)

        logger.info(
            f"sandbox_client: Lint complete — {result.summary}"
        )
        return result

    def run_tests(self, patch: str) -> SandboxResult:
        """
        Parse a patch generated by refactor_node and run ruff + pytest.

        Called by validator_node in nodes.py with state["patch"] — the
        corrective patch generated by Gemini refactor_node.

        The command inside the container is:
            ruff check /sandbox && pytest /sandbox --timeout=10 -q
        ruff runs first — if it fails, pytest is skipped entirely.

        Parameters
        ----------
        patch : str
            Unified diff / patch string generated by Gemini refactor_node.
            Same format as a GitHub diff.

        Returns
        -------
        SandboxResult
            passed=True  → ruff + pytest both pass — proceed to verdict_node
            passed=False → failures found — loop back to refactor_node

        Raises
        ------
        ParseError    — patch is empty or contains no parseable Python files
        SandboxError  — Docker infrastructure failure
        """
        logger.info("sandbox_client: Starting test run")

        files = self._parse_diff(patch)
        logger.info(
            f"sandbox_client: Parsed {len(files)} file(s) from patch for test"
        )

        result = self._runner.run(files=files, run_type=RunType.TEST)

        logger.info(
            f"sandbox_client: Test run complete — {result.summary}"
        )
        return result

    def is_available(self) -> bool:
        """
        Returns True if Docker is running and the sandbox image is built.
        Delegates directly to DockerRunner.is_available().
        Used by tests to skip Docker-dependent tests gracefully.
        """
        return self._runner.is_available()

    # ── Private: Diff parsing ───────────────────────────────────────────────

    def _parse_diff(self, diff: str) -> dict[str, str]:
        """
        Parse a unified diff string into {relative_path: file_content}.

        Only Python files (.py) are extracted — ruff and pytest operate
        on Python source only. Other file types (markdown, yaml, json)
        are silently skipped.

        Algorithm
        ---------
        For each file in the diff:
          1. Find the +++ b/filename header to get the filename
          2. Collect all hunk lines:
             - lines starting with "+"  → add line (strip leading "+")
             - lines starting with " "  → context line (keep as-is)
             - lines starting with "-"  → removed line (skip entirely)
          3. Reconstruct file content from collected lines
          4. Store as files[filename] = content

        This gives us the post-patch version of each file — what the
        file looks like AFTER the PR changes are applied.

        Parameters
        ----------
        diff : str
            Unified diff string — from GitHub API or Gemini refactor output.

        Returns
        -------
        dict[str, str]
            {relative_file_path: reconstructed_file_content}
            Example: {"app/utils.py": "def add(x, y):\n    return x + y\n"}

        Raises
        ------
        ParseError
            If the diff is empty, contains no Python files, or is so
            malformed that no files could be extracted.
        """
        if not diff or not diff.strip():
            raise ParseError("Diff is empty — nothing to lint or test.", sys)

        files: dict[str, str] = {}
        current_file: Optional[str] = None
        current_lines: list[str] = []

        for line in diff.splitlines():

            # ── Detect start of a new file in the diff ──────────────────
            # "+++ b/app/utils.py" → extract "app/utils.py"
            if line.startswith("+++ b/"):
                # Save the previous file before starting a new one
                if current_file is not None:
                    self._store_file(files, current_file, current_lines)

                raw_path = line[6:]  # Strip "+++ b/"

                # Skip /dev/null — means file was deleted, nothing to lint
                if raw_path == "/dev/null":
                    current_file = None
                    current_lines = []
                    continue

                # Only process Python files
                if not raw_path.endswith(".py"):
                    logger.info(
                        f"sandbox_client: Skipping non-Python file — {raw_path}"
                    )
                    current_file = None
                    current_lines = []
                    continue

                current_file = raw_path
                current_lines = []
                logger.info(
                    f"sandbox_client: Parsing file — {current_file}"
                )
                continue

            # ── Skip diff headers — not source code ─────────────────────
            if (
                line.startswith("diff --git")
                or line.startswith("index ")
                or line.startswith("--- ")
                or line.startswith("@@ ")
                or line.startswith("\\ No newline")
            ):
                continue

            # ── Collect source lines for the current file ────────────────
            if current_file is None:
                continue

            if line.startswith("+"):
                # Added line — strip the leading "+"
                current_lines.append(line[1:])

            elif line.startswith(" "):
                # Context line — keep exactly as-is (strip one leading space)
                current_lines.append(line[1:])

            elif line.startswith("-"):
                # Removed line — skip (not in post-patch version)
                continue

        # Save the last file in the diff
        if current_file is not None:
            self._store_file(files, current_file, current_lines)

        if not files:
            raise ParseError(
                "Diff contained no Python files to lint or test. "
                "Only .py files are processed by the sandbox.",
                sys,
            )

        return files

    def _store_file(
        self,
        files: dict[str, str],
        filename: str,
        lines: list[str],
    ) -> None:
        """
        Join collected lines into a file content string and store in files dict.
        Skips files with no content — these are binary files or empty diffs.
        """
        content = "\n".join(lines)

        if not content.strip():
            logger.info(
                f"sandbox_client: Skipping empty file content — {filename}"
            )
            return

        files[filename] = content
        logger.info(
            f"sandbox_client: Stored {len(lines)} lines for {filename}"
        )