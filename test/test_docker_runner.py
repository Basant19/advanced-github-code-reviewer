"""
tests/test_docker_runner.py
============================
Tests for app/sandbox/docker_runner.py

Two test classes:

  TestDockerRunnerUnit
    Pure unit tests — no Docker required. All Docker SDK calls mocked.
    Tests: path normalisation, command building, file writing, path
    sanitisation, empty-files guard, security constraints in container
    call, exception hierarchy, SandboxResult dataclass.

  TestDockerRunnerIntegration
    Real container tests — auto-skipped if Docker Desktop is not running
    or the sandbox image has not been built yet.
    Tests: actual lint pass, actual lint fail, actual test run, network
    isolation, multi-file lint.

Run all:
    pytest tests/test_docker_runner.py -v

Run unit only (no Docker needed):
    pytest tests/test_docker_runner.py -v -k "Unit"

Run integration only:
    pytest tests/test_docker_runner.py -v -k "Integration"

Build sandbox image (prerequisite for integration tests):
    docker build -f docker/sandbox/Dockerfile -t code-reviewer-sandbox:latest .
"""

import sys
import platform
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.sandbox.docker_runner import (
    DockerRunner,
    RunType,
    SandboxResult,
    SandboxError,
    SandboxImageNotFoundError,
    SandboxTimeoutError,
    SANDBOX_IMAGE,
    SANDBOX_WORKDIR,
    CPU_QUOTA,
    MEMORY_LIMIT,
)
from app.core.exceptions import CustomException


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

CLEAN_PYTHON = "def add(x, y):\n    return x + y\n"

# ruff will flag: bare imports on one line, spacing around colon, no space before return
DIRTY_PYTHON = "import os,sys\ndef bad(x,y) : return x+y\n"


def make_runner() -> DockerRunner:
    return DockerRunner()


def docker_available() -> bool:
    """True only if Docker Desktop is running AND sandbox image is built."""
    return DockerRunner().is_available()


def make_mock_docker_client(run_return_value=b"", run_side_effect=None):
    """
    Returns a MagicMock that looks like a connected Docker client
    with the sandbox image already present.
    """
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    mock_client.images.get.return_value = MagicMock()

    if run_side_effect is not None:
        mock_client.containers.run.side_effect = run_side_effect
    else:
        mock_client.containers.run.return_value = run_return_value

    return mock_client


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests — no real Docker
# ─────────────────────────────────────────────────────────────────────────────

class TestDockerRunnerUnit:

    # ── Exception hierarchy ──────────────────────────────────────────────────

    def test_sandbox_error_extends_custom_exception(self):
        assert issubclass(SandboxError, CustomException)

    def test_image_not_found_extends_sandbox_error(self):
        assert issubclass(SandboxImageNotFoundError, SandboxError)

    def test_timeout_error_extends_sandbox_error(self):
        assert issubclass(SandboxTimeoutError, SandboxError)

    def test_all_sandbox_exceptions_extend_custom_exception(self):
        """Catching CustomException at the top level must catch all sandbox errors."""
        assert issubclass(SandboxImageNotFoundError, CustomException)
        assert issubclass(SandboxTimeoutError, CustomException)

    # ── _build_command ───────────────────────────────────────────────────────

    def test_build_command_lint_is_ruff_only(self):
        command = make_runner()._build_command(RunType.LINT)
        assert command == f"ruff check {SANDBOX_WORKDIR}"
        assert "pytest" not in command

    def test_build_command_test_includes_ruff_and_pytest(self):
        command = make_runner()._build_command(RunType.TEST)
        assert "ruff check" in command
        assert "pytest"     in command

    def test_build_command_test_ruff_before_pytest(self):
        """ruff && pytest — ruff must appear before pytest (fail-fast order)."""
        command = make_runner()._build_command(RunType.TEST)
        assert command.index("ruff") < command.index("pytest")

    def test_build_command_test_uses_shell(self):
        """Two commands require sh -c to chain inside the container."""
        command = make_runner()._build_command(RunType.TEST)
        assert command.startswith("sh -c")

    def test_build_command_test_includes_pytest_timeout(self):
        command = make_runner()._build_command(RunType.TEST)
        assert "--timeout=10" in command

    def test_build_command_lint_points_to_sandbox_workdir(self):
        command = make_runner()._build_command(RunType.LINT)
        assert SANDBOX_WORKDIR in command

    def test_build_command_test_points_to_sandbox_workdir(self):
        command = make_runner()._build_command(RunType.TEST)
        assert SANDBOX_WORKDIR in command

    # ── _normalize_path ──────────────────────────────────────────────────────

    def test_normalize_path_linux_returns_absolute_string(self):
        if platform.system() == "Windows":
            pytest.skip("Linux path test — not applicable on Windows")
        path   = Path("/tmp/sandbox_test")
        result = make_runner()._normalize_path(path)
        assert result == str(path.resolve())
        assert "\\" not in result

    def test_normalize_path_windows_uses_forward_slashes(self):
        with patch("app.sandbox.docker_runner.platform.system", return_value="Windows"), \
             patch.object(Path, "resolve", return_value=Path("E:\\Temp\\sandbox_abc")):
            result = make_runner()._normalize_path(Path("E:\\Temp\\sandbox_abc"))
        assert "\\" not in result

    def test_normalize_path_windows_lowercases_drive_letter(self):
        with patch("app.sandbox.docker_runner.platform.system", return_value="Windows"), \
             patch.object(Path, "resolve", return_value=Path("C:\\Users\\basant\\sandbox")):
            result = make_runner()._normalize_path(Path("C:\\Users\\basant\\sandbox"))
        assert result.startswith("/c/")

    def test_normalize_path_windows_removes_colon(self):
        with patch("app.sandbox.docker_runner.platform.system", return_value="Windows"), \
             patch.object(Path, "resolve", return_value=Path("E:\\work\\sandbox")):
            result = make_runner()._normalize_path(Path("E:\\work\\sandbox"))
        assert ":" not in result

    # ── _write_files ─────────────────────────────────────────────────────────

    def test_write_files_creates_all_files_on_disk(self, tmp_path):
        runner = make_runner()
        files  = {"utils.py": CLEAN_PYTHON, "main.py": "x = 1\n"}

        with patch("app.sandbox.docker_runner.tempfile.mkdtemp",
                   return_value=str(tmp_path)):
            runner._write_files(files)

        assert (tmp_path / "utils.py").exists()
        assert (tmp_path / "main.py").exists()

    def test_write_files_content_is_correct(self, tmp_path):
        runner = make_runner()

        with patch("app.sandbox.docker_runner.tempfile.mkdtemp",
                   return_value=str(tmp_path)):
            runner._write_files({"utils.py": CLEAN_PYTHON})

        assert (tmp_path / "utils.py").read_text() == CLEAN_PYTHON

    def test_write_files_creates_nested_directories(self, tmp_path):
        runner = make_runner()

        with patch("app.sandbox.docker_runner.tempfile.mkdtemp",
                   return_value=str(tmp_path)):
            runner._write_files({"app/services/review.py": CLEAN_PYTHON})

        assert (tmp_path / "app" / "services" / "review.py").exists()

    def test_write_files_returns_path_object(self, tmp_path):
        runner = make_runner()

        with patch("app.sandbox.docker_runner.tempfile.mkdtemp",
                   return_value=str(tmp_path)):
            result = runner._write_files({"f.py": "x=1"})

        assert isinstance(result, Path)

    def test_write_files_sanitises_absolute_paths(self, tmp_path):
        """Absolute path in file dict must not escape the temp dir."""
        runner = make_runner()

        with patch("app.sandbox.docker_runner.tempfile.mkdtemp",
                   return_value=str(tmp_path)):
            result_path = runner._write_files({"/etc/passwd": "malicious"})

        # Any file written must live inside the temp dir
        for written in result_path.rglob("*"):
            assert str(tmp_path) in str(written)

    # ── empty files guard ────────────────────────────────────────────────────

    def test_run_returns_pass_for_empty_files_dict(self):
        result = make_runner().run(files={}, run_type=RunType.LINT)
        assert result.passed    is True
        assert result.exit_code == 0
        assert result.tool      == "lint"

    def test_run_empty_files_never_calls_docker(self):
        with patch("app.sandbox.docker_runner.docker.from_env") as mock_docker:
            make_runner().run(files={}, run_type=RunType.LINT)
            mock_docker.assert_not_called()

    # ── _get_client error paths ──────────────────────────────────────────────

    def test_get_client_raises_sandbox_error_when_docker_not_running(self):
        runner = make_runner()
        with patch("app.sandbox.docker_runner.docker.from_env",
                   side_effect=Exception("Cannot connect to daemon")):
            with pytest.raises(SandboxError):
                runner._get_client()

    def test_get_client_raises_image_not_found_when_image_missing(self):
        import docker.errors as de

        runner      = make_runner()
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.images.get.side_effect = de.ImageNotFound("missing")

        with patch("app.sandbox.docker_runner.docker.from_env",
                   return_value=mock_client):
            with pytest.raises(SandboxImageNotFoundError):
                runner._get_client()

    def test_get_client_image_not_found_message_includes_build_command(self):
        import docker.errors as de

        runner      = make_runner()
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.images.get.side_effect = de.ImageNotFound("missing")

        with patch("app.sandbox.docker_runner.docker.from_env",
                   return_value=mock_client):
            with pytest.raises(SandboxImageNotFoundError) as exc_info:
                runner._get_client()

        assert "docker build" in str(exc_info.value)

    def test_get_client_caches_after_first_call(self):
        runner      = make_runner()
        mock_client = make_mock_docker_client()

        with patch("app.sandbox.docker_runner.docker.from_env",
                   return_value=mock_client) as mock_from_env:
            runner._get_client()
            runner._get_client()   # second call — must reuse cache

        mock_from_env.assert_called_once()

    # ── is_available ─────────────────────────────────────────────────────────

    def test_is_available_false_when_docker_not_running(self):
        with patch("app.sandbox.docker_runner.docker.from_env",
                   side_effect=Exception("daemon not running")):
            assert make_runner().is_available() is False

    def test_is_available_false_when_image_missing(self):
        import docker.errors as de

        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.images.get.side_effect = de.ImageNotFound("missing")

        with patch("app.sandbox.docker_runner.docker.from_env",
                   return_value=mock_client):
            assert make_runner().is_available() is False

    # ── _run_container — mocked Docker SDK ──────────────────────────────────

    def test_run_returns_pass_result_on_exit_zero(self, tmp_path):
        mock_client = make_mock_docker_client(run_return_value=b"All checks passed.")

        with patch("app.sandbox.docker_runner.docker.from_env", return_value=mock_client), \
             patch("app.sandbox.docker_runner.tempfile.mkdtemp", return_value=str(tmp_path)), \
             patch("app.sandbox.docker_runner.shutil.rmtree"):
            result = make_runner().run(
                files={"app/utils.py": CLEAN_PYTHON},
                run_type=RunType.LINT,
            )

        assert result.passed    is True
        assert result.exit_code == 0
        assert result.tool      == "lint"
        assert "All checks passed" in result.output

    def test_run_returns_fail_result_on_container_error(self, tmp_path):
        import docker.errors as de

        err = de.ContainerError(
            container=MagicMock(),
            exit_status=1,
            command="ruff check /sandbox",
            image=SANDBOX_IMAGE,
            stderr=b"E501 line too long",
        )
        mock_client = make_mock_docker_client(run_side_effect=err)

        with patch("app.sandbox.docker_runner.docker.from_env", return_value=mock_client), \
             patch("app.sandbox.docker_runner.tempfile.mkdtemp", return_value=str(tmp_path)), \
             patch("app.sandbox.docker_runner.shutil.rmtree"):
            result = make_runner().run(
                files={"app/utils.py": DIRTY_PYTHON},
                run_type=RunType.LINT,
            )

        assert result.passed    is False
        assert result.exit_code == 1

    def test_run_raises_sandbox_error_on_api_error(self, tmp_path):
        import docker.errors as de

        mock_client = make_mock_docker_client(
            run_side_effect=de.APIError("Unexpected server error")
        )

        with patch("app.sandbox.docker_runner.docker.from_env", return_value=mock_client), \
             patch("app.sandbox.docker_runner.tempfile.mkdtemp", return_value=str(tmp_path)), \
             patch("app.sandbox.docker_runner.shutil.rmtree"):
            with pytest.raises(SandboxError):
                make_runner().run(
                    files={"app/utils.py": CLEAN_PYTHON},
                    run_type=RunType.LINT,
                )

    def test_cleanup_called_even_when_container_raises(self, tmp_path):
        """temp dir must be deleted regardless of container outcome."""
        import docker.errors as de

        mock_client = make_mock_docker_client(
            run_side_effect=de.APIError("boom")
        )

        with patch("app.sandbox.docker_runner.docker.from_env", return_value=mock_client), \
             patch("app.sandbox.docker_runner.tempfile.mkdtemp", return_value=str(tmp_path)), \
             patch("app.sandbox.docker_runner.shutil.rmtree") as mock_rmtree:
            with pytest.raises(SandboxError):
                make_runner().run(files={"app/utils.py": CLEAN_PYTHON}, run_type=RunType.LINT)

        mock_rmtree.assert_called_once()

    # ── Security constraints verified in container call args ─────────────────

    def _get_container_call_kwargs(self, tmp_path) -> dict:
        """Helper: run once with mocked Docker and return containers.run kwargs."""
        mock_client = make_mock_docker_client(run_return_value=b"")

        with patch("app.sandbox.docker_runner.docker.from_env", return_value=mock_client), \
             patch("app.sandbox.docker_runner.tempfile.mkdtemp", return_value=str(tmp_path)), \
             patch("app.sandbox.docker_runner.shutil.rmtree"):
            make_runner().run(files={"f.py": "x=1"}, run_type=RunType.LINT)

        return mock_client.containers.run.call_args.kwargs

    def test_security_network_disabled(self, tmp_path):
        kwargs = self._get_container_call_kwargs(tmp_path)
        assert kwargs.get("network_disabled") is True

    def test_security_cpu_quota_set(self, tmp_path):
        kwargs = self._get_container_call_kwargs(tmp_path)
        assert kwargs.get("cpu_quota") == CPU_QUOTA

    def test_security_mem_limit_set(self, tmp_path):
        kwargs = self._get_container_call_kwargs(tmp_path)
        assert kwargs.get("mem_limit") == MEMORY_LIMIT

    def test_security_memswap_equals_mem_limit(self, tmp_path):
        """memswap_limit == mem_limit disables swap entirely."""
        kwargs = self._get_container_call_kwargs(tmp_path)
        assert kwargs.get("memswap_limit") == MEMORY_LIMIT

    def test_security_container_auto_removed(self, tmp_path):
        kwargs = self._get_container_call_kwargs(tmp_path)
        assert kwargs.get("remove") is True

    def test_security_detach_false(self, tmp_path):
        """detach=False means we block until container exits — required for output capture."""
        kwargs = self._get_container_call_kwargs(tmp_path)
        assert kwargs.get("detach") is False

    # ── SandboxResult dataclass ──────────────────────────────────────────────

    def test_sandbox_result_summary_passed_format(self):
        result = SandboxResult(
            passed=True, output="ok", errors="",
            exit_code=0, duration_ms=500, tool="lint"
        )
        assert "LINT PASSED" in result.summary
        assert "exit_code=0" in result.summary
        assert "500ms"        in result.summary

    def test_sandbox_result_summary_failed_format(self):
        result = SandboxResult(
            passed=False, output="E501", errors="E501",
            exit_code=1, duration_ms=1200, tool="test"
        )
        assert "TEST FAILED"  in result.summary
        assert "exit_code=1"  in result.summary
        assert "1200ms"       in result.summary

    def test_sandbox_result_default_image_is_sandbox_image(self):
        result = SandboxResult(
            passed=True, output="", errors="",
            exit_code=0, duration_ms=0, tool="lint"
        )
        assert result.image == SANDBOX_IMAGE


# ─────────────────────────────────────────────────────────────────────────────
# Integration tests — real Docker + real sandbox image
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not docker_available(),
    reason="Docker not running or sandbox image not built — skipping integration tests",
)
class TestDockerRunnerIntegration:
    """
    Real end-to-end tests that spin up actual containers.

    Run these to verify the full container lifecycle on your machine,
    including Windows path normalisation.

    Prerequisite (run once):
        docker build -f docker/sandbox/Dockerfile -t code-reviewer-sandbox:latest .
    """

    def test_lint_passes_on_clean_python(self):
        result = make_runner().run(
            files={"app/utils.py": CLEAN_PYTHON},
            run_type=RunType.LINT,
        )
        assert result.passed    is True
        assert result.exit_code == 0
        assert result.tool      == "lint"

    def test_lint_fails_on_ruff_violations(self):
        result = make_runner().run(
            files={"app/utils.py": DIRTY_PYTHON},
            run_type=RunType.LINT,
        )
        assert result.passed    is False
        assert result.exit_code != 0

    def test_lint_result_output_is_populated_on_failure(self):
        result = make_runner().run(
            files={"app/utils.py": DIRTY_PYTHON},
            run_type=RunType.LINT,
        )
        assert result.output != "" or result.errors != ""

    def test_lint_duration_is_positive(self):
        result = make_runner().run(
            files={"app/utils.py": CLEAN_PYTHON},
            run_type=RunType.LINT,
        )
        assert result.duration_ms > 0

    def test_test_run_on_clean_code_with_no_tests(self):
        """pytest exit 0 or 5 (no tests collected) — both are acceptable passes."""
        result = make_runner().run(
            files={"app/utils.py": CLEAN_PYTHON},
            run_type=RunType.TEST,
        )
        assert result.exit_code in (0, 5)

    def test_test_run_fails_when_ruff_violations_exist(self):
        """ruff fails → pytest skipped (fail-fast). Overall result is failed."""
        result = make_runner().run(
            files={"app/utils.py": DIRTY_PYTHON},
            run_type=RunType.TEST,
        )
        assert result.passed is False

    def test_multiple_files_all_linted(self):
        """All files in the dict must be checked — not just the first."""
        result = make_runner().run(
            files={
                "app/utils.py":   CLEAN_PYTHON,
                "app/helpers.py": DIRTY_PYTHON,
            },
            run_type=RunType.LINT,
        )
        assert result.passed is False   # DIRTY_PYTHON has violations

    def test_result_image_field_matches_sandbox_image_constant(self):
        result = make_runner().run(
            files={"app/utils.py": CLEAN_PYTHON},
            run_type=RunType.LINT,
        )
        assert result.image == SANDBOX_IMAGE

    def test_container_has_no_network_access(self):
        """Code under review must not be able to make HTTP calls."""
        network_code = (
            "import urllib.request\n"
            "urllib.request.urlopen('http://example.com', timeout=3)\n"
        )
        result = make_runner().run(
            files={"app/net_test.py": network_code},
            run_type=RunType.TEST,
        )
        # Network is disabled — any successful HTTP response is a security failure
        assert "200" not in result.output