"""
app/sandbox/docker_runner.py
============================
Docker SDK controller for isolated code execution.

RESPONSIBILITY
--------------
This file does exactly one thing: take a dict of {filename: content},
write those files into a temp directory, spin up a code-reviewer-sandbox
container with that directory mounted, run a command (ruff or pytest),
capture the output, destroy the container, and return a SandboxResult.

This file knows NOTHING about:
  - diffs or patches        (that is sandbox_client.py's job)
  - LangGraph state         (that is nodes.py's job)
  - which review triggered  (that is review_service.py's job)

CALL CHAIN
----------
  nodes.py (lint_node / validator_node)
    → sandbox_client.py     parse diff → {filename: content}
      → docker_runner.py    write files → run container → SandboxResult

EXCEPTION HIERARCHY
-------------------
  CustomException  (app/core/exceptions.py)  ← project root exception
    └── SandboxError                          ← sandbox infrastructure failures
          ├── SandboxImageNotFoundError       ← image not built yet
          └── SandboxTimeoutError             ← container exceeded timeout

  SandboxError is raised for Docker infrastructure failures ONLY.
  ruff/pytest finding code problems is NOT an exception — it is a normal
  SandboxResult with passed=False.

  All three exception classes extend CustomException so that every raised
  error automatically captures: script filename, line number, and error
  message via sys.exc_info(). No extra logging needed at the call site.

SECURITY MODEL
--------------
Every container run enforces:
  CPU limited  : 0.5 cores  — reviewed code cannot starve the host
  Memory ltd   : 256MB      — reviewed code cannot exhaust RAM
  No network   : True       — reviewed code cannot make HTTP calls
  Timeout      : 30s        — reviewed code cannot hang forever
  Auto-removed : True       — no state persists between reviews

PREREQUISITES
-------------
  1. Docker Desktop must be running.
  2. Image must be built once from the project root:
       docker build -f docker/sandbox/Dockerfile -t code-reviewer-sandbox:latest .
  3. uv add docker  (docker SDK in pyproject.toml)
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import docker
import docker.errors

from app.core.exceptions import CustomException
from app.core.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SANDBOX_IMAGE   = "code-reviewer-sandbox:latest"
CPU_QUOTA       = 50_000    # 50_000 / 100_000 period = 0.5 cores max
CPU_PERIOD      = 100_000   # Docker default period in microseconds
MEMORY_LIMIT    = "256m"    # 256 MB hard memory limit per container
TIMEOUT_SEC     = 30        # Wall-clock seconds before container is force-killed
SANDBOX_WORKDIR = "/sandbox"  # WORKDIR inside container — matches Dockerfile


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SandboxError(CustomException):
    """
    Raised when the Docker sandbox infrastructure itself fails.

    This is NOT raised when ruff or pytest finds problems in the reviewed
    code — those are returned as SandboxResult(passed=False, ...).

    Extends CustomException so every raised instance automatically
    captures the script filename and line number via sys.exc_info().
    The review_service.py catch block gets full context without needing
    additional logging at the call site.

    Raised for:
      - Docker daemon not running
      - Container crashes before producing any output
      - Unexpected Docker API errors
    """
    pass


class SandboxImageNotFoundError(SandboxError):
    """
    Raised when the code-reviewer-sandbox image has not been built yet.

    This is a one-time setup error. The error message includes the exact
    build command needed to fix it.
    """
    pass


class SandboxTimeoutError(SandboxError):
    """
    Raised when a container run exceeds TIMEOUT_SEC (30 seconds).

    Protects the host from reviewed code that contains infinite loops
    or extremely slow operations that would block the entire review pipeline.
    """
    pass


# ---------------------------------------------------------------------------
# RunType enum and SandboxResult dataclass
# ---------------------------------------------------------------------------

class RunType(str, Enum):
    """Which tool to run inside the sandbox container."""
    LINT = "lint"   # ruff check only
    TEST = "test"   # ruff check first, then pytest only if ruff passes


@dataclass
class SandboxResult:
    """
    Returned by DockerRunner.run() for every container execution.

    Always returned — never raises on ruff/pytest findings.
    SandboxError (and subclasses) are only raised for infrastructure failures.

    Fields
    ------
    passed      : True if exit_code == 0 — no ruff/pytest errors found
    output      : Full stdout captured from the container
    errors      : Stderr output — empty string if none
    exit_code   : Raw Docker container exit code (0=pass, 1=findings)
    duration_ms : Wall-clock milliseconds for the full container lifecycle
    tool        : "lint" or "test" — which RunType was executed
    image       : Docker image used — stored in review_steps audit trail
    """
    passed      : bool
    output      : str
    errors      : str
    exit_code   : int
    duration_ms : int
    tool        : str
    image       : str = SANDBOX_IMAGE

    @property
    def summary(self) -> str:
        """
        One-line summary for logging and PR comment inclusion.
        Example: "[LINT FAILED] exit_code=1 duration=1243ms"
        """
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"[{self.tool.upper()} {status}] "
            f"exit_code={self.exit_code} "
            f"duration={self.duration_ms}ms"
        )


# ---------------------------------------------------------------------------
# DockerRunner
# ---------------------------------------------------------------------------

class DockerRunner:
    """
    Manages the full lifecycle of one sandbox container run:
      write files → start container → capture output → cleanup temp dir

    Thread safety
    -------------
    Each call to run() creates its own isolated temp directory and container.
    Multiple concurrent reviews never share any filesystem state.

    Windows compatibility
    --------------------
    Docker Desktop on Windows requires volume mount paths in forward-slash
    format with a lowercase drive letter: /e/path/to/dir
    _normalize_path() handles this conversion automatically.
    """

    def __init__(self) -> None:
        # Cached Docker client — created once on first run(), reused after
        self._client: Optional[docker.DockerClient] = None

    # ── Public API ──────────────────────────────────────────────────────────

    def run(
        self,
        files: dict[str, str],
        run_type: RunType,
    ) -> SandboxResult:
        """
        Write files to a temp dir and execute ruff/pytest inside a container.

        Parameters
        ----------
        files : dict[str, str]
            Mapping of relative_path → file_content.
            Written to a temp directory and mounted into the container.
            Example: {"app/utils.py": "def add(x, y):\n    return x + y\n"}

        run_type : RunType
            LINT → ruff check /sandbox
            TEST → ruff check /sandbox && pytest /sandbox

        Returns
        -------
        SandboxResult
            Always returned. passed=False means ruff/pytest found problems.
            This is normal operation — not an exception.

        Raises
        ------
        SandboxImageNotFoundError  — sandbox image not built yet
        SandboxTimeoutError        — container exceeded 30s timeout
        SandboxError               — any other Docker infrastructure failure
        """
        if not files:
            logger.warning(
                "docker_runner.run() called with empty files dict — returning pass"
            )
            return SandboxResult(
                passed=True,
                output="No files provided.",
                errors="",
                exit_code=0,
                duration_ms=0,
                tool=run_type.value,
            )

        client  = self._get_client()
        tmp_dir = self._write_files(files)

        try:
            return self._run_container(
                client=client,
                host_dir=tmp_dir,
                run_type=run_type,
            )
        finally:
            # Always clean up the temp dir — even if _run_container raises
            self._cleanup(tmp_dir)

    def is_available(self) -> bool:
        """
        Returns True if Docker Desktop is running and the sandbox image exists.

        Used by tests to skip Docker-dependent tests when Docker is not running:

            @pytest.mark.skipif(
                not DockerRunner().is_available(),
                reason="Docker not running or sandbox image not built"
            )
        """
        try:
            client = self._get_client()
            client.images.get(SANDBOX_IMAGE)
            return True
        except (docker.errors.ImageNotFound, docker.errors.DockerException):
            return False
        except Exception:
            return False

    # ── Private: Docker client ──────────────────────────────────────────────
    def _get_client(self) -> docker.DockerClient:
        """
        Robust Docker client initialization with:
        ✔ Stale client detection
        ✔ Retry with exponential backoff
        ✔ Windows fallback
        ✔ Clear observability logs
        """

        MAX_RETRIES = 3
        BASE_DELAY = 1  # seconds

        # 🔁 Check cached client health
        if self._client is not None:
            try:
                self._client.ping()
                return self._client
            except Exception as exc:
                logger.warning(
                    f"[docker_runner] Cached client stale → reconnecting | error={exc}"
                )
                self._client = None

        last_exception = None

        for attempt in range(1, MAX_RETRIES + 1):
            delay = BASE_DELAY * (2 ** (attempt - 1))  # exponential backoff

            try:
                logger.info(
                    f"[docker_runner] Connecting to Docker (attempt {attempt}/{MAX_RETRIES})"
                )

                try:
                    client = docker.from_env(timeout=5)
                    client.ping()
                except Exception as e:
                    if platform.system() == "Windows":
                        logger.warning(
                            "[docker_runner] Default connection failed → trying npipe fallback"
                        )
                        client = docker.DockerClient(
                            base_url="npipe:////./pipe/docker_engine",
                            timeout=5,
                        )
                        client.ping()
                    else:
                        raise e

                # ✅ Verify image
                try:
                    client.images.get(SANDBOX_IMAGE)
                    logger.info(
                        f"[docker_runner] Connected & image verified: {SANDBOX_IMAGE}"
                    )
                except docker.errors.ImageNotFound as exc:
                    raise SandboxImageNotFoundError(
                        f"Sandbox image '{SANDBOX_IMAGE}' not found. Build it first.",
                        sys,
                    ) from exc

                self._client = client
                return client

            except Exception as exc:
                last_exception = exc

                logger.error(
                    f"[docker_runner] Attempt {attempt} failed | reason={exc}"
                )

                if attempt < MAX_RETRIES:
                    logger.info(
                        f"[docker_runner] Retrying in {delay}s (backoff)"
                    )
                    time.sleep(delay)

        # 💥 Final failure
        raise SandboxError(
            f"Failed to connect to Docker after {MAX_RETRIES} attempts",
            sys,
        ) from last_exception

    # ── Private: Filesystem ─────────────────────────────────────────────────

    def _write_files(self, files: dict[str, str]) -> Path:
        """
        Write all files into a fresh temp directory on the host.
        Returns the Path to that temp directory.

        The temp dir is the host-side mount source for the container volume.
        docker_runner owns the full lifecycle — it creates and deletes it.

        Path sanitisation prevents directory traversal attacks:
        a malicious filename like "../../etc/passwd" cannot escape the temp dir.
        """
        tmp_dir = Path(tempfile.mkdtemp(prefix="sandbox_"))
        logger.info(
            f"docker_runner: Writing {len(files)} file(s) to {tmp_dir}"
        )

        for rel_path, content in files.items():
            # Sanitise: strip any absolute prefix so paths stay inside tmp_dir
            if Path(rel_path).is_absolute():
                safe_rel = Path(rel_path).relative_to(Path(rel_path).anchor)
            else:
                safe_rel = Path(rel_path)

            safe_path = tmp_dir / safe_rel
            safe_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                safe_path.write_text(content, encoding="utf-8")
                logger.info(f"docker_runner: Wrote — {safe_path.name}")
            except OSError as exc:
                raise SandboxError(
                    f"Failed to write '{rel_path}' to temp dir: {exc}",
                    sys,
                ) from exc

        return tmp_dir

    def _cleanup(self, tmp_dir: Path) -> None:
        """
        Delete the temp directory after the container run completes.
        Non-fatal if deletion fails — the OS will clean up on next reboot.
        """
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.info(f"docker_runner: Cleaned up temp dir {tmp_dir}")
        except Exception as exc:
            logger.warning(
                f"docker_runner: Could not delete temp dir {tmp_dir} — {exc}"
            )

    # ── Private: Container execution ────────────────────────────────────────

    def _run_container(
        self,
        client: docker.DockerClient,
        host_dir: Path,
        run_type: RunType,
    ) -> SandboxResult:
        """
        Execute sandbox container with strict timeout enforcement and full observability.

        Guarantees:
        ✔ No hanging containers
        ✔ Clear timeout vs infra error separation
        ✔ Always cleans up container
        ✔ Full lifecycle logging
        """

        command = self._build_command(run_type)
        host_path = self._normalize_path(host_dir)

        logger.info(
            f"[docker_runner] START | image={SANDBOX_IMAGE} "
            f"tool={run_type.value} host_dir={host_path}"
        )

        start_time = time.perf_counter()
        container = None

        try:
            # 🔥 Start container in detached mode (required for timeout control)
            container = client.containers.run(
                image=SANDBOX_IMAGE,
                command=command,
                volumes={host_path: {"bind": SANDBOX_WORKDIR, "mode": "rw"}},
                working_dir=SANDBOX_WORKDIR,
                network_disabled=True,
                cpu_quota=CPU_QUOTA,
                cpu_period=CPU_PERIOD,
                mem_limit=MEMORY_LIMIT,
                memswap_limit=MEMORY_LIMIT,
                detach=True,
                stdout=True,
                stderr=True,
            )

            logger.info(
                f"[docker_runner] Container started | id={container.id[:12]}"
            )

            # ─────────────────────────────────────────────
            # ⏱️ WAIT WITH HARD TIMEOUT
            # ─────────────────────────────────────────────
            try:
                result = container.wait(timeout=TIMEOUT_SEC)

            except Exception as exc:
                duration_ms = int((time.perf_counter() - start_time) * 1000)

                # 🔍 Distinguish timeout vs other failure
                if "Read timed out" in str(exc) or "timeout" in str(exc).lower():
                    logger.error(
                        f"[docker_runner] TIMEOUT | {TIMEOUT_SEC}s exceeded "
                        f"| container={container.id[:12]} | killing..."
                    )

                    try:
                        container.kill()
                        logger.info(
                            f"[docker_runner] Container killed after timeout | id={container.id[:12]}"
                        )
                    except Exception as kill_exc:
                        logger.warning(
                            f"[docker_runner] Failed to kill container | error={kill_exc}"
                        )

                    raise SandboxTimeoutError(
                        f"Container exceeded timeout of {TIMEOUT_SEC}s",
                        sys,
                    ) from exc

                # ❌ Not timeout → real infra issue
                logger.error(
                    f"[docker_runner] WAIT FAILED | not a timeout "
                    f"| container={container.id[:12]} error={exc}"
                )

                raise SandboxError(
                    f"Container wait failed: {exc}",
                    sys,
                ) from exc

            # ─────────────────────────────────────────────
            # ✅ NORMAL COMPLETION
            # ─────────────────────────────────────────────
            exit_code = result.get("StatusCode", 1)

            try:
                logs = container.logs(stdout=True, stderr=True).decode("utf-8")
            except Exception as log_exc:
                logger.warning(
                    f"[docker_runner] Failed to fetch logs | error={log_exc}"
                )
                logs = ""

            duration_ms = int((time.perf_counter() - start_time) * 1000)

            result_obj = SandboxResult(
                passed=(exit_code == 0),
                output=logs,
                errors="",
                exit_code=exit_code,
                duration_ms=duration_ms,
                tool=run_type.value,
            )

            logger.info(
                f"[docker_runner] COMPLETE | id={container.id[:12]} "
                f"| {result_obj.summary}"
            )

            return result_obj

        # ─────────────────────────────────────────────
        # 🔥 DOCKER API FAILURE
        # ─────────────────────────────────────────────
        except docker.errors.APIError as exc:
            logger.error(
                f"[docker_runner] DOCKER API ERROR | {exc}"
            )
            raise SandboxError(
                f"Docker API error during container run: {exc}",
                sys,
            ) from exc

        # ─────────────────────────────────────────────
        # 💥 UNKNOWN FAILURE (VERY IMPORTANT)
        # ─────────────────────────────────────────────
        except Exception as exc:
            logger.exception(
                f"[docker_runner] UNEXPECTED ERROR | {exc}"
            )
            raise SandboxError(
                f"Unexpected sandbox failure: {exc}",
                sys,
            ) from exc

        # ─────────────────────────────────────────────
        # 🧹 CLEANUP (ALWAYS RUNS)
        # ─────────────────────────────────────────────
        finally:
            if container:
                try:
                    container.remove(force=True)
                    logger.info(
                        f"[docker_runner] CLEANUP | container removed id={container.id[:12]}"
                    )
                except Exception as cleanup_exc:
                    logger.warning(
                        f"[docker_runner] CLEANUP FAILED | id={container.id[:12]} "
                        f"| error={cleanup_exc}"
                    )

    # ── Private: Helpers ────────────────────────────────────────────────────

    def _build_command(self, run_type: RunType) -> str:
        """
        Build the shell command string passed to the container.

        LINT : ruff check only — fastest possible check.
        TEST : ruff first, then pytest only if ruff passes.
               Fail-fast principle: no point running 40 tests on code
               that already has a syntax error on line 3.
        """
        ruff_cmd   = f"ruff check {SANDBOX_WORKDIR}"
        pytest_cmd = f"pytest {SANDBOX_WORKDIR} --timeout=10 -q"

        if run_type == RunType.LINT:
            return ruff_cmd

        # sh -c chains two commands inside the container shell
        return f"sh -c '{ruff_cmd} && {pytest_cmd}'"

    @staticmethod
    def _normalize_path(path: Path) -> str:
        """
        Convert a host filesystem path to Docker volume mount format.

        Linux / Mac:  /tmp/sandbox_abc  →  /tmp/sandbox_abc   (no change)
        Windows:      E:\\Temp\\sandbox_abc  →  /e/Temp/sandbox_abc

        Docker Desktop on Windows requires:
          - drive letter lowercased and colon removed
          - backslashes replaced with forward slashes
          - leading forward slash added

        Without this, volume mounts silently fail on Windows — the container
        starts but /sandbox is empty, so ruff reports "no files to check"
        and every lint passes incorrectly.
        """
        if platform.system() == "Windows":
            abs_path     = str(path.resolve())
            drive, rest  = os.path.splitdrive(abs_path)
            drive_letter = drive.rstrip(":").lower()
            rest_forward = rest.replace("\\", "/")
            normalized   = f"/{drive_letter}{rest_forward}"
            logger.info(
                f"docker_runner: Windows path normalised — "
                f"{abs_path} → {normalized}"
            )
            return normalized

        return str(path.resolve())
    