"""
app/sandbox/docker_runner.py
============================
Docker SDK controller for isolated code execution.

DOCKER-OUT-OF-DOCKER (DooD) FIX
---------------------------------
When running inside Docker Compose, the app container mounts
/var/run/docker.sock to spawn sibling sandbox containers.

Problem: files written to /tmp inside the app container are invisible
to the Docker host daemon — so sandbox containers get an empty /sandbox.

Fix: Write files to a named volume (sandbox_tmp) that is mounted into
the app container. Pass that same volume by name to sibling containers.
CRITICALLY: pass the exact subdirectory as working_dir AND as the
target for ruff/mypy commands — they must scan the right directory.

Environment variables (set in docker-compose.yml):
  SANDBOX_TMP_DIR     — mount point of sandbox_tmp inside app container
                        e.g. /sandbox_tmp
  SANDBOX_VOLUME_NAME — Docker volume name for sibling containers
                        e.g. advanced-github-code-reviewer_sandbox_tmp

When neither is set → bare-metal / local dev → original behaviour.
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
CPU_QUOTA       = 50_000
CPU_PERIOD      = 100_000
MEMORY_LIMIT    = "256m"
TIMEOUT_SEC     = 30
SANDBOX_WORKDIR = "/sandbox"   # Used in bare-metal mode only

# DooD configuration — read from environment
_SANDBOX_TMP_DIR     = os.environ.get("SANDBOX_TMP_DIR", "")
_SANDBOX_VOLUME_NAME = os.environ.get("SANDBOX_VOLUME_NAME", "")
_RUNNING_IN_DOCKER   = os.path.exists("/.dockerenv")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SandboxError(CustomException):
    pass

class SandboxImageNotFoundError(SandboxError):
    pass

class SandboxTimeoutError(SandboxError):
    pass


# ---------------------------------------------------------------------------
# RunType / SandboxResult
# ---------------------------------------------------------------------------

class RunType(str, Enum):
    LINT = "lint"
    TEST = "test"


@dataclass
class SandboxResult:
    passed      : bool
    output      : str
    errors      : str
    exit_code   : int
    duration_ms : int
    tool        : str
    image       : str = SANDBOX_IMAGE

    @property
    def summary(self) -> str:
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

    def __init__(self) -> None:
        self._client: Optional[docker.DockerClient] = None

    # ── Public ──────────────────────────────────────────────────────────────

    def run(self, files: dict[str, str], run_type: RunType) -> SandboxResult:
        if not files:
            logger.warning("docker_runner.run() called with empty files — returning pass")
            return SandboxResult(
                passed=True, output="No files provided.",
                errors="", exit_code=0, duration_ms=0, tool=run_type.value,
            )

        client  = self._get_client()
        tmp_dir = self._write_files(files)

        try:
            return self._run_container(client=client, host_dir=tmp_dir, run_type=run_type)
        finally:
            self._cleanup(tmp_dir)

    def is_available(self) -> bool:
        try:
            self._get_client().images.get(SANDBOX_IMAGE)
            return True
        except Exception:
            return False

    # ── Docker client ────────────────────────────────────────────────────────

    def _get_client(self) -> docker.DockerClient:
        MAX_RETRIES = 3

        if self._client is not None:
            try:
                self._client.ping()
                return self._client
            except Exception:
                self._client = None

        last_exc = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(f"[docker_runner] Connecting to Docker (attempt {attempt}/{MAX_RETRIES})")
                try:
                    client = docker.from_env(timeout=5)
                    client.ping()
                except Exception as e:
                    if platform.system() == "Windows":
                        client = docker.DockerClient(
                            base_url="npipe:////./pipe/docker_engine", timeout=5
                        )
                        client.ping()
                    else:
                        raise e

                try:
                    client.images.get(SANDBOX_IMAGE)
                    logger.info(f"[docker_runner] Connected & image verified: {SANDBOX_IMAGE}")
                except docker.errors.ImageNotFound as exc:
                    raise SandboxImageNotFoundError(
                        f"Sandbox image '{SANDBOX_IMAGE}' not found. Build it first.", sys
                    ) from exc

                self._client = client
                return client

            except Exception as exc:
                last_exc = exc
                logger.error(f"[docker_runner] Attempt {attempt} failed | reason={exc}")
                if attempt < MAX_RETRIES:
                    time.sleep(2 ** (attempt - 1))

        raise SandboxError(
            f"Failed to connect to Docker after {MAX_RETRIES} attempts", sys
        ) from last_exc

    # ── Filesystem ───────────────────────────────────────────────────────────

    def _write_files(self, files: dict[str, str]) -> Path:
        """
        Write files to temp directory.

        DooD mode  → base dir = /sandbox_tmp (shared named volume)
        Local mode → base dir = system temp (/tmp or OS equivalent)
        """
        if _SANDBOX_TMP_DIR:
            base = Path(_SANDBOX_TMP_DIR)
            base.mkdir(parents=True, exist_ok=True)
            tmp_dir = Path(tempfile.mkdtemp(prefix="sandbox_", dir=base))
            logger.info(
                f"docker_runner: Writing {len(files)} file(s) to "
                f"{tmp_dir} [DooD volume mode]"
            )
        else:
            tmp_dir = Path(tempfile.mkdtemp(prefix="sandbox_"))
            logger.info(
                f"docker_runner: Writing {len(files)} file(s) to "
                f"{tmp_dir} [bare-metal mode]"
            )

        for rel_path, content in files.items():
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
                    f"Failed to write '{rel_path}' to temp dir: {exc}", sys
                ) from exc

        return tmp_dir

    def _cleanup(self, tmp_dir: Path) -> None:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.info(f"docker_runner: Cleaned up temp dir {tmp_dir}")
        except Exception as exc:
            logger.warning(f"docker_runner: Could not delete temp dir {tmp_dir} — {exc}")

    # ── Container execution ──────────────────────────────────────────────────

    def _run_container(
        self,
        client: docker.DockerClient,
        host_dir: Path,
        run_type: RunType,
    ) -> SandboxResult:
        """
        Determine volume strategy and run the sandbox container.

        DooD mode  — mount named volume at /vdata, use /vdata/sandbox_XXXX
                     as working_dir AND pass it to ruff/mypy commands.
        Local mode — bind-mount host path directly to /sandbox.
        """

        # ── Decide volume strategy and working directory ──────────────────────
        if _RUNNING_IN_DOCKER and _SANDBOX_TMP_DIR and _SANDBOX_VOLUME_NAME:
            # The named volume is mounted at /vdata inside the sibling container.
            # host_dir.name is the unique subdirectory: e.g. "sandbox_abc123"
            # So the actual files live at /vdata/sandbox_abc123
            workdir     = f"/vdata/{host_dir.name}"
            volumes     = {_SANDBOX_VOLUME_NAME: {"bind": "/vdata", "mode": "rw"}}
            mode_label  = f"DooD — volume={_SANDBOX_VOLUME_NAME} workdir={workdir}"
        else:
            workdir    = SANDBOX_WORKDIR
            host_path  = self._normalize_path(host_dir)
            volumes    = {host_path: {"bind": SANDBOX_WORKDIR, "mode": "rw"}}
            mode_label = f"bare-metal — host_path={host_path}"

        # ── Build command pointing at the CORRECT workdir ─────────────────────
        command = self._build_command(run_type, workdir=workdir)

        logger.info(
            f"[docker_runner] START | image={SANDBOX_IMAGE} "
            f"tool={run_type.value} | {mode_label}"
        )

        start_time = time.perf_counter()
        container  = None

        try:
            container = client.containers.run(
                image=SANDBOX_IMAGE,
                command=command,
                volumes=volumes,
                working_dir=workdir,
                network_disabled=True,
                cpu_quota=CPU_QUOTA,
                cpu_period=CPU_PERIOD,
                mem_limit=MEMORY_LIMIT,
                memswap_limit=MEMORY_LIMIT,
                detach=True,
                stdout=True,
                stderr=True,
            )

            logger.info(f"[docker_runner] Container started | id={container.id[:12]}")

            try:
                result = container.wait(timeout=TIMEOUT_SEC)
            except Exception as exc:
                if "Read timed out" in str(exc) or "timeout" in str(exc).lower():
                    logger.error(
                        f"[docker_runner] TIMEOUT | {TIMEOUT_SEC}s exceeded "
                        f"| container={container.id[:12]} | killing..."
                    )
                    try:
                        container.kill()
                    except Exception as kill_exc:
                        logger.warning(f"[docker_runner] Failed to kill container | {kill_exc}")
                    raise SandboxTimeoutError(
                        f"Container exceeded timeout of {TIMEOUT_SEC}s", sys
                    ) from exc

                logger.error(
                    f"[docker_runner] WAIT FAILED | container={container.id[:12]} error={exc}"
                )
                raise SandboxError(f"Container wait failed: {exc}", sys) from exc

            exit_code = result.get("StatusCode", 1)

            try:
                logs = container.logs(stdout=True, stderr=True).decode("utf-8")
            except Exception as log_exc:
                logger.warning(f"[docker_runner] Failed to fetch logs | {log_exc}")
                logs = ""

            duration_ms = int((time.perf_counter() - start_time) * 1000)
            result_obj  = SandboxResult(
                passed=(exit_code == 0),
                output=logs,
                errors="",
                exit_code=exit_code,
                duration_ms=duration_ms,
                tool=run_type.value,
            )

            logger.info(
                f"[docker_runner] COMPLETE | id={container.id[:12]} | {result_obj.summary}"
            )
            return result_obj

        except docker.errors.APIError as exc:
            logger.error(f"[docker_runner] DOCKER API ERROR | {exc}")
            raise SandboxError(f"Docker API error: {exc}", sys) from exc

        except Exception as exc:
            logger.exception(f"[docker_runner] UNEXPECTED ERROR | {exc}")
            raise SandboxError(f"Unexpected sandbox failure: {exc}", sys) from exc

        finally:
            if container:
                try:
                    container.remove(force=True)
                    logger.info(
                        f"[docker_runner] CLEANUP | container removed id={container.id[:12]}"
                    )
                except Exception as cleanup_exc:
                    logger.warning(
                        f"[docker_runner] CLEANUP FAILED | id={container.id[:12]} | {cleanup_exc}"
                    )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _build_command(self, run_type: RunType, workdir: str = SANDBOX_WORKDIR) -> str:
        """
        Build the ruff/mypy/pytest command.

        workdir is passed explicitly so DooD mode points tools at
        /vdata/sandbox_XXXX instead of the hardcoded /sandbox.
        This is the critical fix — without it ruff scans /sandbox
        (empty) instead of the directory where files were written.
        """
        ruff_cmd   = f"ruff check {workdir} --exit-zero"
        mypy_cmd   = (
            f"mypy {workdir} "
            f"--ignore-missing-imports "
            f"--check-untyped-defs "
            f"--no-error-summary "
            f"--no-pretty"
        )
        pytest_cmd = f"pytest {workdir} --timeout=10 -q"

        if run_type == RunType.LINT:
            return f"sh -c '{ruff_cmd}; {mypy_cmd}'"
        return f"sh -c '{ruff_cmd}; {mypy_cmd} && {pytest_cmd}'"

    @staticmethod
    def _normalize_path(path: Path) -> str:
        """Windows bare-metal: E:\\path → /e/path. Linux/macOS: unchanged."""
        if platform.system() == "Windows":
            abs_path     = str(path.resolve())
            drive, rest  = os.path.splitdrive(abs_path)
            drive_letter = drive.rstrip(":").lower()
            rest_forward = rest.replace("\\", "/")
            normalized   = f"/{drive_letter}{rest_forward}"
            logger.info(
                f"docker_runner: Windows path normalized — {abs_path} → {normalized}"
            )
            return normalized
        return str(path.resolve())