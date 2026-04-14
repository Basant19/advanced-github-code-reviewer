"""
app/mcp/sandbox_client.py
=========================
MCP wrapper between LangGraph nodes and DockerRunner.
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
    """Raised when the unified diff cannot be parsed."""
    pass

class SandboxRuntimeError(CustomException):
    """Raised when the Docker execution fails fundamentally."""
    pass


# ---------------------------------------------------------------------------
# SandboxClient
# ---------------------------------------------------------------------------

class SandboxClient:
    def __init__(self) -> None:
        self._runner = DockerRunner()

    # ──────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────

    def run_lint(self, diff: str) -> SandboxResult:
        """
        CRITICAL FIX: Processes a PR diff and runs linting inside the sandbox.
        This resolves the AttributeError in nodes.py.
        """
        logger.info("sandbox_client: Starting lint run (type=diff)")
        
        try:
            # 1. Parse the diff into file mapping
            files = self._parse_diff(diff)
            if not files:
                logger.warning("sandbox_client: No files extracted from diff, skipping sandbox.")
                return SandboxResult(passed=True, output="No python files to lint", tool="lint")

            # 2. Log payload for observability
            self._log_payload(files, "LINT")

            # 3. Execute in Docker
            result = self._runner.run(files=files, run_type=RunType.LINT)
            
            logger.info(
                "sandbox_client: Lint complete — %s exit_code=%d duration=%dms",
                "[LINT PASSED]" if result.passed else "[LINT FAILED]",
                result.exit_code,
                result.duration_ms,
            )
            return result

        except Exception as exc:
            logger.exception("sandbox_client: run_lint encountered a fatal error")
            # Return a failed result to ensure the graph knows the sandbox didn't verify the code
            return SandboxResult(
                passed=False, 
                output="", 
                errors=f"Sandbox execution failed: {str(exc)}",
                exit_code=1,
                tool="lint"
            )

    def run_lint_raw(self, filename: str, content: str) -> SandboxResult:
        logger.info("sandbox_client: Starting lint run (type=raw)")
        files = {filename: content}

        try:
            # If your DockerRunner uses Ruff/Flake8, it won't catch 'a / str(b)'.
            # Consider adding a type-checker like 'mypy' to the Docker image 
            # specifically for raw analysis.
            result = self._runner.run(files=files, run_type=RunType.LINT)
            return result
        except Exception as exc:
            logger.exception("[sandbox_debug] run_lint_raw failed for %s", filename)
            return SandboxResult(
                passed=False, # CHANGED: Fail closed so the system knows baseline is unverified
                output="",
                errors=f"Baseline check failed: {str(exc)}",
                exit_code=1,
                tool="lint",
            )

    def run_tests(self, patch: str) -> SandboxResult:
        """Runs tests by applying a patch/diff."""
        logger.info("sandbox_client: Starting test run")
        try:
            files = self._parse_diff(patch)
            self._log_payload(files, "TEST")
            result = self._runner.run(files=files, run_type=RunType.TEST)
            logger.info(f"sandbox_client: Test run complete — {result.summary}")
            return result
        except Exception as exc:
            logger.exception("sandbox_client: run_tests failed")
            return SandboxResult(passed=False, errors=str(exc), tool="test")

    def is_available(self) -> bool:
        return self._runner.is_available()

    # ──────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ──────────────────────────────────────────────────────────

    def _log_payload(self, files: dict[str, str], context: str) -> None:
        logger.info(f"[sandbox_debug] Sending {len(files)} file(s) to Docker for {context}")
        for filename, content in files.items():
            logger.info(f"  -> {filename} ({len(content)} chars)")

    def _parse_diff(self, diff: str) -> dict[str, str]:
        if not diff or not diff.strip():
            return {}

        files: dict[str, str] = {}
        current_file: Optional[str] = None
        current_lines: list[str] = []

        plus_plus_re = re.compile(r"^\+\+\+\s+(?:[abciw][\\/])?([^\s]+)")
        modified_re = re.compile(r"^---\s+([^\s]+)\s+\(modified\)")

        try:
            for line in diff.splitlines():
                match = plus_plus_re.match(line) or modified_re.match(line)

                if match:
                    # Save previous file buffer
                    if current_file and current_lines:
                        files[current_file] = "\n".join(current_lines).strip()

                    new_path = match.group(1).replace("\\", "/").strip()
                    if new_path == "/dev/null" or not new_path.endswith(".py"):
                        current_file = None
                        continue
                    
                    current_file = new_path
                    current_lines = []
                    continue

                if current_file is not None:
                    if (line.startswith("diff --git") or line.startswith("index ") or 
                        line.startswith("@@") or line.startswith("--- ")):
                        continue
                    
                    # Reconstruction logic
                    if line.startswith("+") and not line.startswith("+++"):
                        current_lines.append(line[1:])
                    elif line.startswith(" "):
                        current_lines.append(line[1:])
                    elif not line.startswith("-"):
                        current_lines.append(line)

            # Save final file
            if current_file and current_lines:
                files[current_file] = "\n".join(current_lines).strip()

            return files
        except Exception as e:
            logger.error(f"[parse_diff] Failed to parse diff: {str(e)}")
            raise ParseError(f"Diff parsing failed: {str(e)}")