"""
app/graph/nodes.py

LangGraph Agent Nodes
----------------------
Each function is one node in the review workflow graph.

P1 nodes (unchanged — 27 tests passing):
    fetch_diff_node      → fetches PR diff from GitHub
    analyze_code_node    → Gemini review with ChromaDB context
    reflect_node         → self-reflection pass (runs ×2 in P1 graph)
    verdict_node         → APPROVE / REQUEST_CHANGES + PR comment

P2 nodes (new):
    lint_node            → Docker sandbox ruff check on PR diff
    refactor_node        → Gemini generates corrective patch from findings
    validator_node       → Docker sandbox ruff + pytest on generated patch

Node execution order (P2 graph — defined in workflow.py):
    fetch_diff_node
        → analyze_code_node
        → lint_node
            FAIL (lint_passed=False) → refactor_node
            PASS (lint_passed=True)  → refactor_node  (with clean lint signal)
        → refactor_node
        → validator_node
            FAIL → loop back to refactor_node  (up to MAX_REFACTOR_ITERATIONS)
            PASS → verdict_node
        → verdict_node

Module-level objects (both patchable in tests):
    llm              — Gemini chat model via init_chat_model
    sandbox_client   — SandboxClient instance for lint_node and validator_node

    patch.object(n, "llm", mock_llm)            — replaces LLM for a test
    patch.object(n, "sandbox_client", mock_sc)  — replaces sandbox for a test

    Both must be module-level names — patch.object cannot patch a local
    variable created inside a function call like _get_sandbox_client().

API Key Flow:
    config.py loads GOOGLE_API_KEY from .env into settings AND os.environ.
    init_chat_model reads os.environ at invocation time.
    This survives LangGraph's internal model re-instantiation at runtime.
"""

import os
import sys

from langsmith import traceable
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

from app.graph.state import ReviewState
from app.mcp.github_client import GitHubClient
from app.core.config import settings
from app.core.exceptions import CustomException
from app.core.logger import get_logger

logger = get_logger(__name__)


# ── API key validation ────────────────────────────────────────────────────────

_google_api_key = os.environ.get("GOOGLE_API_KEY", "")

if not _google_api_key:
    logger.error(
        "[nodes] GOOGLE_API_KEY not found in os.environ — "
        "check config.py loaded correctly"
    )
    raise CustomException("GOOGLE_API_KEY is not set in environment", sys)

logger.info(
    f"[nodes] GOOGLE_API_KEY received — "
    f"length={len(_google_api_key)}, "
    f"prefix={_google_api_key[:6]}..."
)


# ── LLM initialisation ────────────────────────────────────────────────────────

llm = init_chat_model(
    model="gemini-2.5-flash",
    model_provider="google_genai",
    configurable_fields="any",
)

logger.info("LLM initialised — gemini-2.5-flash via init_chat_model")


# ── Sandbox client initialisation (P2) ───────────────────────────────────────
#
# sandbox_client is a MODULE-LEVEL object — required for patch.object in tests:
#     patch.object(n, "sandbox_client", mock_sc)
#
# This is the identical pattern to llm above. patch.object can only replace
# a name that already exists on the module — it cannot patch a local variable
# created inside _get_sandbox_client() at call time.
#
# Instantiation is safe even when Docker Desktop is not running:
#   SandboxClient.__init__ → DockerRunner.__init__ → self._client = None
#   docker.from_env() is only called inside DockerRunner.run(), never here.

from app.mcp.sandbox_client import SandboxClient
sandbox_client = SandboxClient()

logger.info("[nodes] SandboxClient initialised")


# ── ChromaDB ──────────────────────────────────────────────────────────────────

def _get_chroma_collection():
    import chromadb
    from chromadb.utils.embedding_functions import ChromaLangchainEmbeddingFunction
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    langchain_embedder = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=settings.google_api_key,
    )
    embedding_fn = ChromaLangchainEmbeddingFunction(
        embedding_function=langchain_embedder
    )
    client = chromadb.PersistentClient(path="./chroma_store")
    return client.get_or_create_collection(
        name="repo_context",
        embedding_function=embedding_fn,
    )


# ── Shared helpers ────────────────────────────────────────────────────────────

def _parse_llm_output(raw: str) -> tuple[list[str], list[str]]:
    """
    Parses ISSUES / SUGGESTIONS sections from structured LLM response.
    Returns (issues, suggestions). Falls back gracefully on bad format.
    Used by analyze_code_node, reflect_node, and refactor_node.
    """
    issues:      list[str] = []
    suggestions: list[str] = []
    current_section = None

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.upper().startswith("ISSUES"):
            current_section = "issues"
        elif line.upper().startswith("SUGGESTIONS"):
            current_section = "suggestions"
        elif line.startswith("-"):
            item = line.lstrip("- ").strip()
            if item.lower() == "none" or not item:
                continue
            if current_section == "issues":
                issues.append(item)
            elif current_section == "suggestions":
                suggestions.append(item)

    return issues, suggestions


# =============================================================================
# P1 NODES — UNTOUCHED
# All 27 tests for these nodes pass. Do not modify.
# =============================================================================

@traceable(name="fetch_diff_node", tags=["github", "fetch"])
def fetch_diff_node(state: ReviewState) -> dict:
    """
    Fetches PR metadata, changed files, and unified diff from GitHub.

    Reads  : state["owner"], state["repo"], state["pr_number"]
    Writes : state["metadata"], state["files"], state["diff"]
    """
    owner     = state["owner"]
    repo      = state["repo"]
    pr_number = state["pr_number"]

    logger.info(f"[fetch_diff_node] Starting — {owner}/{repo}#{pr_number}")

    try:
        client   = GitHubClient()
        metadata = client.get_pr_metadata(owner, repo, pr_number)
        files    = client.get_pr_files(owner, repo, pr_number)
        diff     = client.get_pr_diff(owner, repo, pr_number)

        logger.info(
            f"[fetch_diff_node] Done — "
            f"{len(files)} file(s), diff length={len(diff)} chars"
        )

        return {
            "metadata": metadata,
            "files":    files,
            "diff":     diff,
        }

    except Exception as e:
        logger.error(f"[fetch_diff_node] Failed: {e}")
        raise CustomException(str(e), sys)


@traceable(name="analyze_code_node", tags=["llm", "analysis"])
def analyze_code_node(state: ReviewState) -> dict:
    """
    Uses Gemini to analyze the PR diff and identify issues / suggestions.
    Injects ChromaDB repo context to make the review codebase-aware.

    Reads  : state["diff"], state["metadata"]
    Writes : state["issues"], state["suggestions"], state["repo_context"]
    """
    diff     = state["diff"]
    metadata = state["metadata"]

    logger.info(
        f"[analyze_code_node] Starting analysis — "
        f"PR: '{metadata.get('title', 'unknown')}'"
    )

    repo_context = ""
    try:
        collection   = _get_chroma_collection()
        query        = f"{metadata.get('title', '')} {diff[:500]}"
        results      = collection.query(query_texts=[query], n_results=3)
        docs         = results.get("documents", [[]])[0]
        repo_context = "\n\n".join(docs) if docs else ""

        if repo_context:
            logger.info(
                f"[analyze_code_node] ChromaDB returned "
                f"{len(docs)} context chunk(s)"
            )
        else:
            logger.info(
                "[analyze_code_node] No ChromaDB context — proceeding without"
            )

    except Exception as e:
        logger.warning(
            f"[analyze_code_node] ChromaDB query failed (non-fatal): {e}"
        )
        repo_context = ""

    system_prompt = """You are an expert code reviewer. Your job is to review
GitHub Pull Request diffs and provide structured, actionable feedback.

You must respond in EXACTLY this format — no extra text:

ISSUES:
- <issue 1>
- <issue 2>
(or "- None" if no issues found)

SUGGESTIONS:
- <suggestion 1>
- <suggestion 2>
(or "- None" if no suggestions)

Be specific. Reference filenames and line context where possible.
Focus on: correctness, security, maintainability, and Python best practices."""

    user_message = f"""PR Title: {metadata.get('title', 'N/A')}
Author: {metadata.get('author', 'N/A')}
Base branch: {metadata.get('base_branch', 'N/A')} ← {metadata.get('head_branch', 'N/A')}

{f'--- Codebase Context ---{chr(10)}{repo_context}{chr(10)}' if repo_context else ''}
--- PR Diff ---
{diff if diff else 'No diff available.'}"""

    try:
        response   = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ])
        raw_output = response.content
        logger.info("[analyze_code_node] LLM response received")

    except Exception as e:
        logger.error(f"[analyze_code_node] LLM call failed: {e}")
        raise CustomException(str(e), sys)

    issues, suggestions = _parse_llm_output(raw_output)

    logger.info(
        f"[analyze_code_node] Parsed — "
        f"{len(issues)} issue(s), {len(suggestions)} suggestion(s)"
    )

    return {
        "issues":       issues,
        "suggestions":  suggestions,
        "repo_context": repo_context,
    }


@traceable(name="reflect_node", tags=["llm", "reflection"])
def reflect_node(state: ReviewState) -> dict:
    """
    Self-reflection pass — asks Gemini to find anything it missed.
    Runs up to 2 times (controlled by reflection_count in workflow.py).

    Reads  : state["issues"], state["suggestions"], state["diff"],
             state["reflection_count"]
    Writes : state["issues"], state["suggestions"], state["reflection_count"]
    """
    current_count        = state["reflection_count"]
    existing_issues      = state["issues"]
    existing_suggestions = state["suggestions"]
    diff                 = state["diff"]

    logger.info(f"[reflect_node] Reflection pass #{current_count + 1} starting")

    reflection_prompt = f"""You previously reviewed a Pull Request and found:

ISSUES FOUND:
{chr(10).join(f'- {i}' for i in existing_issues) if existing_issues else '- None'}

SUGGESTIONS MADE:
{chr(10).join(f'- {s}' for s in existing_suggestions) if existing_suggestions else '- None'}

Here is the full diff again:
{diff[:3000]}

Critically reflect: did you miss any bugs, security issues, or important
improvements? Add only NEW findings not already listed above.

Respond in EXACTLY this format:

ISSUES:
- <new issue> (or "- None" if nothing new)

SUGGESTIONS:
- <new suggestion> (or "- None" if nothing new)"""

    try:
        response   = llm.invoke([HumanMessage(content=reflection_prompt)])
        raw_output = response.content
        logger.info("[reflect_node] Reflection LLM response received")

    except Exception as e:
        logger.error(f"[reflect_node] LLM call failed: {e}")
        raise CustomException(str(e), sys)

    new_issues, new_suggestions = _parse_llm_output(raw_output)

    merged_issues = list(
        {i.lower(): i for i in existing_issues + new_issues}.values()
    )
    merged_suggestions = list(
        {s.lower(): s for s in existing_suggestions + new_suggestions}.values()
    )

    logger.info(
        f"[reflect_node] Pass #{current_count + 1} complete — "
        f"added {len(new_issues)} issue(s), {len(new_suggestions)} suggestion(s)"
    )

    return {
        "issues":           merged_issues,
        "suggestions":      merged_suggestions,
        "reflection_count": current_count + 1,
    }


@traceable(name="verdict_node", tags=["verdict"])
def verdict_node(state: ReviewState) -> dict:
    """
    Produces the final APPROVE or REQUEST_CHANGES verdict and formats
    the full review comment that will be posted to GitHub.

    P2 update: if lint_result or validation_result are present in state,
    their summaries are included in the PR comment for full transparency.

    Reads  : state["issues"], state["suggestions"], state["metadata"],
             state["lint_result"]       (P2 — optional)
             state["validation_result"] (P2 — optional)
    Writes : state["verdict"], state["summary"]
    """
    issues      = state["issues"]
    suggestions = state["suggestions"]
    metadata    = state["metadata"]

    lint_result       = state.get("lint_result")
    validation_result = state.get("validation_result")

    logger.info(
        f"[verdict_node] Generating verdict — "
        f"{len(issues)} issue(s), {len(suggestions)} suggestion(s)"
    )

    verdict       = "REQUEST_CHANGES" if issues else "APPROVE"
    verdict_emoji = "✅" if verdict == "APPROVE" else "🔴"

    issues_section = (
        "\n".join(f"- {i}" for i in issues)
        if issues else "_No issues found._"
    )
    suggestions_section = (
        "\n".join(f"- {s}" for s in suggestions)
        if suggestions else "_No suggestions._"
    )

    sandbox_section = ""
    if lint_result or validation_result:
        sandbox_lines = ["### 🔬 Sandbox Results"]
        if lint_result:
            lint_icon = "✅" if lint_result.passed else "❌"
            sandbox_lines.append(
                f"- **Lint (ruff):** {lint_icon} `{lint_result.summary}`"
            )
        if validation_result:
            val_icon = "✅" if validation_result.passed else "❌"
            sandbox_lines.append(
                f"- **Tests (ruff + pytest):** {val_icon} "
                f"`{validation_result.summary}`"
            )
        sandbox_section = "\n" + "\n".join(sandbox_lines) + "\n"

    summary = f"""## {verdict_emoji} AI Code Review

**PR:** {metadata.get('title', 'N/A')}
**Author:** {metadata.get('author', 'N/A')}
**Verdict:** `{verdict}`

---
{sandbox_section}
### 🐛 Issues
{issues_section}

### 💡 Suggestions
{suggestions_section}

---
*Review generated by Advanced GitHub Code Reviewer · Powered by Gemini 2.5 Flash*"""

    logger.info(f"[verdict_node] Verdict: {verdict}")

    return {
        "verdict": verdict,
        "summary": summary,
    }


# =============================================================================
# P2 NODES — NEW
# =============================================================================

@traceable(name="lint_node", tags=["sandbox", "lint"])
def lint_node(state: ReviewState) -> dict:
    """
    Runs ruff on the post-patch version of changed files inside the Docker
    sandbox. Executes BEFORE the Gemini reviewer — fail-fast design:
    if ruff fails, route directly to refactor_node, skip the Gemini API call.

    Uses module-level sandbox_client (patchable in tests).

    Reads  : state["diff"]
    Writes : state["lint_result"], state["lint_passed"]
    Raises : CustomException on any sandbox or parse failure
    """
    diff = state["diff"]
    logger.info("[lint_node] Starting lint run via Docker sandbox")

    try:
        lint_result = sandbox_client.run_lint(diff)

        logger.info(
            f"[lint_node] Complete — "
            f"passed={lint_result.passed} "
            f"exit_code={lint_result.exit_code} "
            f"duration={lint_result.duration_ms}ms"
        )

        if lint_result.passed:
            logger.info("[lint_node] Lint PASSED")
        else:
            logger.info(
                f"[lint_node] Lint FAILED\nruff output:\n{lint_result.output}"
            )

        return {
            "lint_result": lint_result,
            "lint_passed": lint_result.passed,
        }

    except Exception as e:
        logger.error(f"[lint_node] Sandbox error: {e}")
        raise CustomException(str(e), sys)


@traceable(name="refactor_node", tags=["llm", "refactor"])
def refactor_node(state: ReviewState) -> dict:
    """
    Sends the diff, reviewer findings, and lint output to Gemini and asks
    it to generate a corrective patch that fixes all identified issues.

    Uses module-level llm (patchable in tests).
    Does NOT call sandbox_client — LLM-only node.

    Reads  : state["diff"], state["issues"], state["suggestions"],
             state["lint_result"], state["validation_result"],
             state["reflection_count"]
    Writes : state["patch"], state["reflection_count"]
    Raises : CustomException on LLM failure
    """
    diff              = state["diff"]
    issues            = state["issues"]
    suggestions       = state["suggestions"]
    lint_result       = state.get("lint_result")
    validation_result = state.get("validation_result")
    current_count     = state["reflection_count"]

    logger.info(
        f"[refactor_node] Generating patch — "
        f"iteration={current_count + 1} issues={len(issues)}"
    )

    lint_context = ""
    if lint_result and not lint_result.passed:
        lint_context = f"\n--- Lint Failures (ruff) ---\n{lint_result.output}\n"

    test_context = ""
    if validation_result and not validation_result.passed:
        test_context = (
            f"\n--- Test Failures (ruff + pytest) ---\n{validation_result.output}\n"
        )

    issues_text = (
        "\n".join(f"- {i}" for i in issues) if issues else "- None identified"
    )
    suggestions_text = (
        "\n".join(f"- {s}" for s in suggestions) if suggestions else "- None"
    )

    refactor_prompt = f"""You are an expert Python developer tasked with fixing
code issues identified in a Pull Request review.

--- Original PR Diff ---
{diff[:3000]}

--- Issues Found by Code Reviewer ---
{issues_text}

--- Suggestions ---
{suggestions_text}
{lint_context}{test_context}
Your task: Generate a corrective unified diff patch that fixes ALL of the
issues listed above. The patch must:
  1. Be a valid unified diff format (--- a/file, +++ b/file, @@ hunks)
  2. Fix every issue listed — do not skip any
  3. Follow Python best practices and PEP 8
  4. Not introduce new issues

Respond with ONLY the unified diff patch. No explanations, no markdown
code blocks, no extra text — just the raw diff starting with "diff --git"."""

    try:
        response  = llm.invoke([HumanMessage(content=refactor_prompt)])
        raw_patch = response.content.strip()
        logger.info(
            f"[refactor_node] Patch generated — "
            f"{len(raw_patch)} chars"
        )

    except Exception as e:
        logger.error(f"[refactor_node] LLM call failed: {e}")
        raise CustomException(str(e), sys)

    return {
        "patch":            raw_patch,
        "reflection_count": current_count + 1,
    }


@traceable(name="validator_node", tags=["sandbox", "validation"])
def validator_node(state: ReviewState) -> dict:
    """
    Runs ruff + pytest on the patch generated by refactor_node inside the
    Docker sandbox. The result determines whether the loop continues or
    the workflow proceeds to verdict_node.

    Uses module-level sandbox_client (patchable in tests).

    Loop control in workflow.py (should_refactor()):
      validation_result.passed=True  → verdict_node
      validation_result.passed=False AND reflection_count < MAX → refactor_node
      reflection_count >= MAX        → verdict_node

    Reads  : state["patch"], state["reflection_count"]
    Writes : state["validation_result"]
             state["reflection_count"] incremented on failure (for loop control)
    Raises : CustomException on sandbox or parse failure
    """
    patch         = state.get("patch", "")
    current_count = state["reflection_count"]

    logger.info(
        f"[validator_node] Starting validation — iteration={current_count}"
    )

    if not patch:
        logger.warning(
            "[validator_node] Empty patch from refactor_node — marking failed"
        )
        from app.sandbox.docker_runner import SandboxResult, RunType
        empty_result = SandboxResult(
            passed=False,
            output="No patch was generated by refactor_node.",
            errors="Empty patch",
            exit_code=1,
            duration_ms=0,
            tool=RunType.TEST.value,
        )
        return {
            "validation_result": empty_result,
            "reflection_count":  current_count + 1,
        }

    try:
        validation_result = sandbox_client.run_tests(patch)

        logger.info(
            f"[validator_node] Complete — "
            f"passed={validation_result.passed} "
            f"exit_code={validation_result.exit_code} "
            f"duration={validation_result.duration_ms}ms"
        )

        if validation_result.passed:
            logger.info(
                "[validator_node] Validation PASSED — routing to verdict_node"
            )
            return {
                "validation_result": validation_result,
                "reflection_count":  current_count,   # unchanged — no loop
            }
        else:
            logger.info(
                f"[validator_node] Validation FAILED — looping to refactor_node\n"
                f"output:\n{validation_result.output}"
            )
            return {
                "validation_result": validation_result,
                "reflection_count":  current_count + 1,
            }

    except Exception as e:
        logger.error(f"[validator_node] Sandbox error: {e}")
        raise CustomException(str(e), sys)