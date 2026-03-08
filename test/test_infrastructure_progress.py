"""
tests/test_infrastructure_progress.py

Infrastructure Progress Test Suite
------------------------------------
Purpose  : Validate the entire backend foundation and log detailed
           progress output to  logs/test_progress.log  so you can
           see exactly where the project stands at any point in time.

Run with:
    pytest tests/test_infrastructure_progress.py -v

Log file will be written to:
    logs/test_progress.log

Actual project structure (root = advanced-github-code-reviewer/):
    app/
        api/routes/         webhook.py, review.py, chat.py
        core/               config.py, logger.py, exceptions.py
        db/base.py, session.py
        db/models/          repository.py, pull_request.py, review.py,
                            review_step.py, thread.py, message.py, __init__.py
        graph/              state.py, nodes.py, workflow.py
        mcp/                github_client.py, filesystem_client.py, sandbox_client.py
        sandbox/            docker_runner.py
        services/           review_service.py, repository_service.py, chat_service.py
    schemas/                ← ROOT-LEVEL (NOT inside app/)
        chat_schema.py, repository_schema.py, pull_request_schema.py,
        review_schema.py, thread_schema.py, message_schema.py
    logs/
    tests/
    .env
    .env.example
    pyproject.toml
    README.md
"""

import sys
import logging
import importlib
from datetime import datetime
from pathlib import Path

import pytest

# ──────────────────────────────────────────────────────────────────────────────
# 0.  ENSURE ROOT IS ON sys.path
#     Needed so both  `app.*`  and  `schemas.*`  are importable
# ──────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent   # …/advanced-github-code-reviewer

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ──────────────────────────────────────────────────────────────────────────────
# 1.  LOG FILE SETUP
#     Creates  logs/test_progress.log  (appends each run — full history kept)
# ──────────────────────────────────────────────────────────────────────────────

LOG_DIR  = ROOT / "logs"
LOG_FILE = LOG_DIR / "test_progress.log"

LOG_DIR.mkdir(parents=True, exist_ok=True)

progress_logger = logging.getLogger("test_progress")
progress_logger.setLevel(logging.DEBUG)

if not progress_logger.handlers:
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    progress_logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)-8s | %(message)s"))
    progress_logger.addHandler(ch)


# ── tiny helpers ──────────────────────────────────────────────────────────────

def log_section(title: str) -> None:
    progress_logger.info("=" * 60)
    progress_logger.info(f"  {title}")
    progress_logger.info("=" * 60)


def log_pass(check: str) -> None:
    progress_logger.info(f"  ✅  PASS  →  {check}")


def log_fail(check: str, reason: str = "") -> None:
    msg = f"  ❌  FAIL  →  {check}"
    if reason:
        msg += f"  ({reason})"
    progress_logger.warning(msg)


def log_skip(check: str, reason: str = "") -> None:
    msg = f"  ⏭️   SKIP  →  {check}"
    if reason:
        msg += f"  ({reason})"
    progress_logger.info(msg)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  SESSION-LEVEL BANNER  (logged once per pytest run)
# ──────────────────────────────────────────────────────────────────────────────

def pytest_configure(config):
    progress_logger.info("")
    progress_logger.info("*" * 60)
    progress_logger.info("  ADVANCED GITHUB CODE REVIEWER")
    progress_logger.info("  Infrastructure Progress Check")
    progress_logger.info(f"  Run started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    progress_logger.info(f"  Project root: {ROOT}")
    progress_logger.info("*" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  HELPER — safe module importer
# ──────────────────────────────────────────────────────────────────────────────

def try_import(module_path: str):
    """Return (module | None, error_string | None)."""
    try:
        mod = importlib.import_module(module_path)
        return mod, None
    except Exception as exc:
        return None, str(exc)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  TEST CLASSES
# ──────────────────────────────────────────────────────────────────────────────


class TestProjectStructure:
    """
    Phase 0 — verify every expected file/folder exists on disk.

    NOTE: schemas/ lives at the PROJECT ROOT, not inside app/.
          The test checks the real path: <root>/schemas/chat_schema.py
    """

    REQUIRED_PATHS = [
        # ── core ──────────────────────────────────────────────────────────
        "app/core/config.py",
        "app/core/logger.py",
        "app/core/exceptions.py",
        # ── db ────────────────────────────────────────────────────────────
        "app/db/base.py",
        "app/db/session.py",
        "app/db/models/__init__.py",
        "app/db/models/repository.py",
        "app/db/models/pull_request.py",
        "app/db/models/review.py",
        "app/db/models/review_step.py",
        "app/db/models/thread.py",
        "app/db/models/message.py",
        # ── graph (stubs acceptable right now) ───────────────────────────
        "app/graph/state.py",
        "app/graph/nodes.py",
        "app/graph/workflow.py",
        # ── mcp ───────────────────────────────────────────────────────────
        "app/mcp/github_client.py",
        "app/mcp/filesystem_client.py",
        "app/mcp/sandbox_client.py",
        # ── sandbox ───────────────────────────────────────────────────────
        "app/sandbox/docker_runner.py",
        # ── services ──────────────────────────────────────────────────────
        "app/services/chat_service.py",
        "app/services/review_service.py",
        "app/services/repository_service.py",
        # ── schemas  (ROOT-LEVEL — not inside app/) ───────────────────────
        "schemas/chat_schema.py",
        "schemas/repository_schema.py",
        "schemas/pull_request_schema.py",
        "schemas/review_schema.py",
        "schemas/thread_schema.py",
        "schemas/message_schema.py",
        # ── project config ────────────────────────────────────────────────
        "pyproject.toml",
        ".env.example",
        "README.md",
    ]

    def test_all_required_files_exist(self):
        log_section("Phase 0 · Project Structure")
        missing = []

        for rel in self.REQUIRED_PATHS:
            full = ROOT / rel
            if full.exists():
                log_pass(rel)
            else:
                log_fail(rel, "file not found")
                missing.append(rel)

        if missing:
            progress_logger.warning(
                f"  {len(missing)} missing file(s) — see ❌ entries above"
            )
        else:
            progress_logger.info("  All required files present 🎉")

        assert not missing, f"Missing files: {missing}"


class TestCoreInfrastructure:
    """Phase 1 — logging, exceptions, config."""

    def test_logger_module_imports(self):
        log_section("Phase 1 · Core Infrastructure")
        mod, err = try_import("app.core.logger")
        if mod:
            log_pass("app.core.logger imports successfully")
        else:
            log_fail("app.core.logger", err)
        assert mod is not None, err

    def test_logger_has_get_logger(self):
        mod, _ = try_import("app.core.logger")
        if mod is None:
            log_fail("app.core.logger", "module could not be imported")
            pytest.fail("Module import failed")
        if hasattr(mod, "get_logger"):
            log_pass("logger.get_logger() function exists")
        else:
            log_skip("logger.get_logger()", "not yet defined — acceptable at this stage")
            pytest.skip("get_logger not yet defined")

    def test_exceptions_module_imports(self):
        mod, err = try_import("app.core.exceptions")
        if mod:
            log_pass("app.core.exceptions imports successfully")
        else:
            log_fail("app.core.exceptions", err)
        assert mod is not None, err

    def test_config_module_imports(self):
        mod, err = try_import("app.core.config")
        if mod:
            log_pass("app.core.config imports successfully")
        else:
            log_fail("app.core.config", err)
        assert mod is not None, err

    def test_config_has_settings(self):
        mod, _ = try_import("app.core.config")
        if mod is None:
            pytest.fail("Module import failed")
        if hasattr(mod, "settings"):
            log_pass("config.settings object exists")
        else:
            log_skip("config.settings", "not yet instantiated — acceptable")
            pytest.skip("settings not yet defined")


class TestDatabaseLayer:
    """Phase 2 — ORM base, session, all models."""

    def test_db_base_imports(self):
        log_section("Phase 2 · Database Layer")
        mod, err = try_import("app.db.base")
        if mod:
            log_pass("app.db.base imports successfully")
        else:
            log_fail("app.db.base", err)
        assert mod is not None, err

    def test_db_session_imports(self):
        mod, err = try_import("app.db.session")
        if mod:
            log_pass("app.db.session imports successfully")
        else:
            log_fail("app.db.session", err)
        assert mod is not None, err

    def test_models_init_imports(self):
        """__init__.py registers all models — critical for Alembic migrations."""
        mod, err = try_import("app.db.models")
        if mod:
            log_pass("app.db.models.__init__ imports successfully (model registry OK)")
        else:
            log_fail("app.db.models.__init__", err)
        assert mod is not None, err

    @pytest.mark.parametrize("model_path,label", [
        ("app.db.models.repository",   "Repository model"),
        ("app.db.models.pull_request", "PullRequest model"),
        ("app.db.models.review",       "Review model"),
        ("app.db.models.review_step",  "ReviewStep model"),
        ("app.db.models.thread",       "Thread model"),
        ("app.db.models.message",      "Message model"),
    ])
    def test_model_imports(self, model_path, label):
        mod, err = try_import(model_path)
        if mod:
            log_pass(f"{label} imports successfully")
        else:
            log_fail(label, err)
        assert mod is not None, f"{label} import failed: {err}"

    def test_repository_model_has_required_fields(self):
        mod, _ = try_import("app.db.models.repository")
        if mod is None:
            log_skip("Repository fields", "module import failed")
            pytest.skip("module not importable")

        cls = getattr(mod, "Repository", None)
        if cls is None:
            log_skip("Repository fields", "class 'Repository' not found in module")
            pytest.skip("Repository class not found")

        table = getattr(cls, "__table__", None)
        if table is None:
            log_skip("Repository fields", "not a mapped ORM class yet")
            pytest.skip("not a mapped ORM class")

        cols    = {c.name for c in table.columns}
        needed  = {"id", "name", "owner", "url"}
        missing = needed - cols
        if not missing:
            log_pass(f"Repository has required columns: {needed}")
        else:
            log_fail("Repository columns", f"missing: {missing}")
        assert not missing, f"Repository missing columns: {missing}"

    def test_message_model_has_role_field(self):
        mod, _ = try_import("app.db.models.message")
        if mod is None:
            log_skip("Message.role field", "module not importable")
            pytest.skip("module not importable")

        cls = getattr(mod, "Message", None)
        if cls is None:
            log_skip("Message.role field", "class not found")
            pytest.skip("Message class not found")

        table = getattr(cls, "__table__", None)
        if table is None:
            log_skip("Message.role field", "not a mapped ORM class")
            pytest.skip("not mapped")

        cols = {c.name for c in table.columns}
        if "role" in cols:
            log_pass("Message model has 'role' column")
        else:
            log_fail("Message.role", "column missing")
        assert "role" in cols

    def test_models_registry_exports_all(self):
        """Confirm __init__.py exports every model needed by the app."""
        mod, _ = try_import("app.db.models")
        if mod is None:
            pytest.skip("models __init__ not importable")

        expected = ["Repository", "PullRequest", "Review", "ReviewStep", "Thread", "Message"]
        missing  = [name for name in expected if not hasattr(mod, name)]

        if not missing:
            log_pass(f"Model registry exports all {len(expected)} models")
        else:
            log_fail("Model registry", f"missing exports: {missing}")
        assert not missing, f"Model registry missing: {missing}"


class TestSchemaLayer:
    """
    Phase 3 — Pydantic schemas.

    schemas/ is at the PROJECT ROOT:
        <root>/schemas/chat_schema.py   → import as  schemas.chat_schema
    NOT:
        app/schemas/chat_schema.py      (does not exist)
    """

    def test_chat_schema_imports(self):
        log_section("Phase 3 · Schema Layer  (schemas/ is at project root)")
        mod, err = try_import("schemas.chat_schema")
        if mod:
            log_pass("schemas.chat_schema imports successfully  [root-level path ✓]")
        else:
            log_fail("schemas.chat_schema", err)
        assert mod is not None, err

    def test_message_create_schema_exists(self):
        mod, _ = try_import("schemas.chat_schema")
        if mod is None:
            pytest.skip("module not importable")
        cls = getattr(mod, "MessageCreate", None)
        if cls:
            log_pass("MessageCreate schema class exists")
        else:
            log_skip("MessageCreate", "class not yet defined — acceptable")
            pytest.skip("MessageCreate not yet defined")

    def test_message_create_instantiation(self):
        mod, _ = try_import("schemas.chat_schema")
        if mod is None:
            pytest.skip("module not importable")
        MessageCreate = getattr(mod, "MessageCreate", None)
        if MessageCreate is None:
            pytest.skip("MessageCreate not yet defined")
        try:
            obj = MessageCreate(thread_id=1, role="user", content="Hello")
            assert obj.role == "user"
            log_pass("MessageCreate instantiates and validates correctly")
        except Exception as exc:
            log_fail("MessageCreate instantiation", str(exc))
            pytest.fail(str(exc))

    @pytest.mark.parametrize("schema_module,label", [
        ("schemas.repository_schema",   "repository_schema"),
        ("schemas.pull_request_schema", "pull_request_schema"),
        ("schemas.review_schema",       "review_schema"),
        ("schemas.thread_schema",       "thread_schema"),
        ("schemas.message_schema",      "message_schema"),
    ])
    def test_all_schema_files_importable(self, schema_module, label):
        mod, err = try_import(schema_module)
        if mod:
            log_pass(f"{label} imports successfully")
        else:
            log_fail(label, err)
        assert mod is not None, f"{label} import failed: {err}"


class TestServiceLayer:
    """Phase 4 — service classes."""

    def test_chat_service_imports(self):
        log_section("Phase 4 · Service Layer")
        mod, err = try_import("app.services.chat_service")
        if mod:
            log_pass("app.services.chat_service imports successfully")
        else:
            log_fail("app.services.chat_service", err)
        assert mod is not None, err

    def test_chat_service_class_exists(self):
        mod, _ = try_import("app.services.chat_service")
        if mod is None:
            pytest.skip("module not importable")
        cls = getattr(mod, "ChatService", None)
        if cls:
            log_pass("ChatService class exists")
        else:
            log_fail("ChatService", "class not found in module")
        assert cls is not None

    def test_review_service_imports(self):
        mod, err = try_import("app.services.review_service")
        if mod:
            log_pass("app.services.review_service imports successfully")
        else:
            log_fail("app.services.review_service", err)
        assert mod is not None, err

    def test_repository_service_imports(self):
        mod, err = try_import("app.services.repository_service")
        if mod:
            log_pass("app.services.repository_service imports successfully")
        else:
            log_fail("app.services.repository_service", err)
        assert mod is not None, err


class TestAgentLayer:
    """Phase 5 — LangGraph graph layer (stubs acceptable now)."""

    def test_graph_state_imports(self):
        log_section("Phase 5 · Agent / Graph Layer  (stubs OK at this stage)")
        mod, err = try_import("app.graph.state")
        if mod:
            log_pass("app.graph.state imports successfully")
        else:
            log_fail("app.graph.state", err)
        assert mod is not None, err

    def test_review_state_exists(self):
        mod, _ = try_import("app.graph.state")
        if mod is None:
            pytest.skip("module not importable")
        cls = getattr(mod, "ReviewState", None)
        if cls:
            log_pass("ReviewState TypedDict/class exists")
        else:
            log_skip("ReviewState", "not yet defined — implement in Commit 3")
            pytest.skip("ReviewState not yet defined")

    def test_graph_nodes_imports(self):
        mod, err = try_import("app.graph.nodes")
        if mod:
            log_pass("app.graph.nodes imports successfully")
        else:
            log_fail("app.graph.nodes", err)
        assert mod is not None, err

    def test_graph_workflow_imports(self):
        mod, err = try_import("app.graph.workflow")
        if mod:
            log_pass("app.graph.workflow imports successfully")
        else:
            log_fail("app.graph.workflow", err)
        assert mod is not None, err


class TestMCPLayer:
    """Phase 6 — MCP / GitHub client (stub acceptable now)."""

    def test_github_client_imports(self):
        log_section("Phase 6 · MCP Layer  (stubs OK at this stage)")
        mod, err = try_import("app.mcp.github_client")
        if mod:
            log_pass("app.mcp.github_client imports successfully")
        else:
            log_fail("app.mcp.github_client", err)
        assert mod is not None, err

    def test_github_client_class_exists(self):
        mod, _ = try_import("app.mcp.github_client")
        if mod is None:
            pytest.skip("module not importable")
        cls = getattr(mod, "GitHubClient", None)
        if cls:
            log_pass("GitHubClient class exists")
        else:
            log_skip("GitHubClient", "not yet defined — implement in Commit 2")
            pytest.skip("GitHubClient not yet defined")

    def test_filesystem_client_imports(self):
        mod, err = try_import("app.mcp.filesystem_client")
        if mod:
            log_pass("app.mcp.filesystem_client imports successfully")
        else:
            log_fail("app.mcp.filesystem_client", err)
        assert mod is not None, err

    def test_sandbox_client_imports(self):
        mod, err = try_import("app.mcp.sandbox_client")
        if mod:
            log_pass("app.mcp.sandbox_client imports successfully")
        else:
            log_fail("app.mcp.sandbox_client", err)
        assert mod is not None, err


class TestAPILayer:
    """Phase 7 — FastAPI routes (stubs acceptable now)."""

    def test_webhook_route_imports(self):
        log_section("Phase 7 · API / Routes Layer  (stubs OK at this stage)")
        mod, err = try_import("app.api.routes.webhook")
        if mod:
            log_pass("app.api.routes.webhook imports successfully")
        else:
            log_fail("app.api.routes.webhook", err)
        assert mod is not None, err

    def test_review_route_imports(self):
        mod, err = try_import("app.api.routes.review")
        if mod:
            log_pass("app.api.routes.review imports successfully")
        else:
            log_fail("app.api.routes.review", err)
        assert mod is not None, err

    def test_chat_route_imports(self):
        mod, err = try_import("app.api.routes.chat")
        if mod:
            log_pass("app.api.routes.chat imports successfully")
        else:
            log_fail("app.api.routes.chat", err)
        assert mod is not None, err


# ──────────────────────────────────────────────────────────────────────────────
# 5.  SESSION-LEVEL TEARDOWN — full summary written to log
# ──────────────────────────────────────────────────────────────────────────────

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    passed  = len(terminalreporter.stats.get("passed",  []))
    failed  = len(terminalreporter.stats.get("failed",  []))
    skipped = len(terminalreporter.stats.get("skipped", []))
    total   = passed + failed + skipped

    progress_logger.info("")
    progress_logger.info("─" * 60)
    progress_logger.info("  TEST RUN SUMMARY")
    progress_logger.info("─" * 60)
    progress_logger.info(f"  Total   : {total}")
    progress_logger.info(f"  Passed  : {passed}  ✅")
    progress_logger.info(f"  Failed  : {failed}  {'❌' if failed else '—'}")
    progress_logger.info(f"  Skipped : {skipped}  ⏭️")
    progress_logger.info("")

    phases = {
        "Phase 0 · Project Structure"  : "TestProjectStructure",
        "Phase 1 · Core Infrastructure": "TestCoreInfrastructure",
        "Phase 2 · Database Layer"     : "TestDatabaseLayer",
        "Phase 3 · Schema Layer"       : "TestSchemaLayer",
        "Phase 4 · Service Layer"      : "TestServiceLayer",
        "Phase 5 · Agent Layer"        : "TestAgentLayer",
        "Phase 6 · MCP Layer"          : "TestMCPLayer",
        "Phase 7 · API Routes"         : "TestAPILayer",
    }

    progress_logger.info("  PHASE READINESS")
    progress_logger.info("─" * 60)

    all_reports = (
        terminalreporter.stats.get("passed",  []) +
        terminalreporter.stats.get("failed",  []) +
        terminalreporter.stats.get("skipped", [])
    )

    for phase_label, class_name in phases.items():
        phase_reports = [r for r in all_reports if class_name in r.nodeid]
        if not phase_reports:
            progress_logger.info(f"  {phase_label:45s}  [ NO TESTS RAN ]")
            continue
        failures = [r for r in phase_reports if getattr(r, "outcome", "") == "failed"]
        if failures:
            progress_logger.warning(f"  {phase_label:45s}  [ NEEDS WORK  ❌ ]")
        else:
            progress_logger.info(   f"  {phase_label:45s}  [ READY       ✅ ]")

    progress_logger.info("─" * 60)
    progress_logger.info(f"  Log saved → {LOG_FILE}")
    progress_logger.info("─" * 60)
    progress_logger.info("")