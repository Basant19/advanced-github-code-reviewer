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
        logger.info("sandbox_client: Starting lint run")

        # 🔍 RAW DIFF DEBUG (CRITICAL)
        logger.debug(f"[sandbox_debug] RAW DIFF PREVIEW:\n{diff[:1000]}")

        files = self._parse_diff(diff)

        # 🔍 Payload snapshot
        self._log_payload(files, "LINT")

        result = self._runner.run(files=files, run_type=RunType.LINT)

        logger.info(f"sandbox_client: Lint complete — {result.summary}")
        return result

    def run_tests(self, patch: str) -> SandboxResult:
        logger.info("sandbox_client: Starting test run")

        # 🔍 RAW PATCH DEBUG
        logger.debug(f"[sandbox_debug] RAW PATCH PREVIEW:\n{patch[:1000]}")

        files = self._parse_diff(patch)

        # 🔍 Payload snapshot
        self._log_payload(files, "TEST")

        result = self._runner.run(files=files, run_type=RunType.TEST)

        logger.info(f"sandbox_client: Test run complete — {result.summary}")
        return result

    def is_available(self) -> bool:
        return self._runner.is_available()

    # ──────────────────────────────────────────────────────────
    # DEBUG HELPERS
    # ──────────────────────────────────────────────────────────

    def _log_payload(self, files: dict[str, str], context: str) -> None:
        """Log exactly what is sent to Docker."""
        if not files:
            logger.warning(f"[sandbox_debug] {context} payload is EMPTY")
            return

        logger.info(f"[sandbox_debug] Sending {len(files)} file(s) to Docker for {context}")

        for filename, content in files.items():
            logger.info(f"  -> {filename} ({len(content)} chars)")
            logger.debug(
                f"[sandbox_debug] CONTENT PREVIEW ({filename}):\n{content[:300]}"
            )

    # ──────────────────────────────────────────────────────────
    # DIFF PARSER
    # ──────────────────────────────────────────────────────────

    def _parse_diff(self, diff: str) -> dict[str, str]:
        """
        Production-grade unified diff parser

        Handles:
        ✔ Standard Git diff (+++ b/file.py)
        ✔ GitHub simplified diff (--- file.py (modified))
        ✔ New files (/dev/null → file.py)
        ✔ Deleted files (skipped safely)
        ✔ Simplified diffs (no context lines)
        ✔ Silent parsing failures (logged)

        Returns:
            dict[str, str] → {filename: reconstructed_content}
        """

        if not diff or not diff.strip():
            logger.warning("[parse_diff]  Empty diff received")
            return {}

        files: dict[str, str] = {}
        current_file: Optional[str] = None
        current_lines: list[str] = []

        # ─────────────────────────────────────────────
        # Patterns
        # ─────────────────────────────────────────────
        plus_plus_re = re.compile(r"^\+\+\+\s+(?:[abciw][\\/])?([^\s]+)")
        modified_re = re.compile(r"^---\s+([^\s]+)\s+\(modified\)")
        new_file_re = re.compile(r"^\+\+\+\s+(?:[abciw][\\/])?(.+)")  # for new files

        skipped_files = 0
        detected_files = 0

        logger.info("[parse_diff] 🚀 START parsing")
        logger.debug(f"[parse_diff] Diff preview (first 500 chars):\n{diff[:500]}")

        try:
            lines = diff.splitlines()

            for idx, line in enumerate(lines):

                # ─────────────────────────────
                # 1. Detect file headers
                # ─────────────────────────────
                plus_match = plus_plus_re.match(line)
                mod_match = modified_re.match(line)

                match = plus_match or mod_match

                if match:
                    new_path = match.group(1).replace("\\", "/").strip()
                    detected_files += 1

                    header_type = "+++" if plus_match else "--- (modified)"
                    logger.info(f"[parse_diff] Detected via {header_type}: {new_path}")

                    # Save previous file
                    if current_file:
                        content = "\n".join(current_lines).strip()

                        if content:
                            files[current_file] = content
                            logger.info(f"[parse_diff]  Saved file={current_file} lines={len(current_lines)}")
                        else:
                            logger.warning(f"[parse_diff]  EMPTY file skipped: {current_file}")

                    # Reset state
                    current_file = None
                    current_lines = []

                    # Skip deleted file
                    if new_path == "/dev/null":
                        logger.info("[parse_diff]  Skipping deleted file")
                        continue

                    # Skip non-python
                    if not new_path.endswith(".py"):
                        skipped_files += 1
                        logger.info(f"[parse_diff] ⏭ Skipping non-python: {new_path}")
                        continue

                    current_file = new_path
                    logger.info(f"[parse_diff]  Tracking file: {current_file}")
                    continue

                # ─────────────────────────────
                # 2. Skip metadata
                # ─────────────────────────────
                if (
                    line.startswith("diff --git")
                    or (line.startswith("--- ") and not mod_match)
                    or line.startswith("index ")
                    or line.startswith("@@")
                    or line.startswith("rename ")
                    or line.startswith("similarity ")
                    or line.startswith("Binary files")
                    or line.startswith("\\ No newline")
                ):
                    continue

                # ─────────────────────────────
                # 3. Collect content (CRITICAL FIX)
                # ─────────────────────────────
                if current_file is not None:

                    if line.startswith("+") and not line.startswith("+++"):
                        current_lines.append(line[1:])

                    elif line.startswith(" "):
                        current_lines.append(line[1:])

                    elif not line.startswith("-"):
                        #  FIX: handles simplified diffs (no prefix)
                        current_lines.append(line)

                    # else: skip removed lines (-)

            # ─────────────────────────────
            # 4. Save last file
            # ─────────────────────────────
            if current_file:
                content = "\n".join(current_lines).strip()

                if content:
                    files[current_file] = content
                    logger.info(f"[parse_diff]  Saved final file={current_file} lines={len(current_lines)}")
                else:
                    logger.warning(f"[parse_diff]  Final file EMPTY: {current_file}")

            # ─────────────────────────────
            # 5. Validation + Silent Failure Detection
            # ─────────────────────────────
            if not files:
                logger.error(
                    f"[parse_diff]  NO FILES PARSED | detected={detected_files} skipped={skipped_files}"
                )
                logger.debug(f"[parse_diff] Full diff dump (first 1000 chars):\n{diff[:1000]}")
                return {}

            # Detect suspicious parsing
            for fname, content in files.items():
                if len(content.strip()) < 20:
                    logger.warning(f"[parse_diff]  Suspicious small file: {fname} ({len(content)} chars)")

            # Debug preview of output
            for fname, content in files.items():
                preview = content[:120].replace("\n", "\\n")
                logger.debug(f"[parse_diff]  {fname} preview: {preview}")

            logger.info(
                f"[parse_diff]  SUCCESS extracted={len(files)} skipped={skipped_files}"
            )

            return files

        except Exception as e:
            logger.exception(f"[parse_diff]  CRASH: {str(e)}")
            raise ParseError(f"Diff parsing failed: {str(e)}", sys)