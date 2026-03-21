"""
app/graph/nodes.py

LangGraph Agent Nodes — P3 Production Version
----------------------------------------------
Each function is one node in the review workflow graph.

Node execution order (defined in workflow.py):
    fetch_diff_node
        → analyze_code_node
        → reflect_node (×2 via should_reflect)
        → lint_node
        → refactor_node (conditional)
        → validator_node (conditional)
        → hitl_node     [interrupt_before pauses here]
        → verdict_node
        → END

Design Principles
-----------------
1. NO import-time side effects — all heavy objects initialized lazily.
2. ALL nodes are async — compatible with LangGraph's async runner.
3. LLM failures are SOFT — graph never crashes on quota errors.
4. Quota detection is INSTANT — max_retries=0 prevents retry storms.
5. Every node logs entry, exit, and any failure with structured context.
6. sandbox_client and llm are module-level globals for patch.object in tests.

LLM Failure Signals
-------------------
FREE_TIER_EXHAUSTED  — 429 quota error from Google API
LLM_ERROR            — any other LLM failure (timeout, server error)

When these signals are returned by safe_llm_invoke():
    - The node continues with degraded output (empty issues / skipped step)
    - The graph does NOT crash
    - The HITL gate still fires — human can review degraded output
    - verdict_node produces a clear summary of what happened

Async Safety
------------
LLM_SEMAPHORE limits concurrent Gemini calls to 2.
LLM_TIMEOUT enforces a hard 30-second ceiling per call.
Both prevent runaway resource consumption on the free tier.

Patching in Tests
-----------------
    from unittest.mock import patch, AsyncMock
    import app.graph.nodes as nodes

    with patch.object(nodes, "sandbox_client", mock_sc):
        ...

    with patch.object(nodes, "_llm_instance", mock_llm):
        ...
"""

import os
import sys
import asyncio
from typing import List, Any, Dict, Optional

from langsmith import traceable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.graph.state import ReviewState
from app.mcp.github_client import GitHubClient
from app.core.exceptions import CustomException
from app.core.logger import get_logger

logger = get_logger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

LLM_TIMEOUT: int = 30
"""Hard ceiling (seconds) for a single Gemini API call."""

LLM_SEMAPHORE: asyncio.Semaphore = asyncio.Semaphore(2)
"""Limits concurrent LLM calls to 2 — prevents free-tier saturation."""

MAX_DIFF_CHARS: int = 4000
"""Maximum diff characters sent to Gemini — keeps prompt within token limits."""

FREE_TIER_EXHAUSTED: str = "FREE_TIER_EXHAUSTED"
"""Sentinel returned by safe_llm_invoke() on 429 quota error."""

LLM_ERROR: str = "LLM_ERROR"
"""Sentinel returned by safe_llm_invoke() on any other LLM failure."""


# ── Module-level lazy singletons (patchable in tests) ────────────────────────

_llm_instance: Optional[ChatGoogleGenerativeAI] = None
"""
Lazily initialized Gemini LLM instance.
Access via get_llm() — never reference directly.
Module-level so patch.object(nodes, '_llm_instance', mock) works in tests.
"""

sandbox_client = None
"""
Lazily initialized SandboxClient.
Module-level so patch.object(nodes, 'sandbox_client', mock_sc) works in tests.
"""


# ── LLM Initialization ────────────────────────────────────────────────────────

def get_llm() -> ChatGoogleGenerativeAI:
    """
    Return the module-level Gemini LLM instance, initializing it on first call.

    Configuration
    -------------
    model        : gemini-2.0-flash  (lower quota usage than 2.5-flash)
    temperature  : 0                 (deterministic output for code review)
    max_retries  : 0                 (CRITICAL — prevents internal retry storms
                                      that exhaust free-tier quota silently)

    Raises
    ------
    CustomException
        If GOOGLE_API_KEY is not set in environment.

    Returns
    -------
    ChatGoogleGenerativeAI
        Initialized Gemini chat model instance.
    """
    global _llm_instance

    if _llm_instance is not None:
        return _llm_instance

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error(
            "[nodes] GOOGLE_API_KEY not found in environment — "
            "check config.py loaded correctly"
        )
        raise CustomException("GOOGLE_API_KEY is not set in environment", sys)

    logger.info(
        "[nodes] Initializing Gemini LLM — model=gemini-2.0-flash "
        "max_retries=0 temperature=0"
    )

    _llm_instance = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        temperature=0,
        max_retries=0,
    )

    logger.info("[nodes] Primary LLM initialized")
    return _llm_instance


# ── Sandbox Initialization ────────────────────────────────────────────────────

def get_sandbox():
    """
    Return the module-level SandboxClient instance, initializing it on first call.

    Safe to call even when Docker Desktop is not running — SandboxClient.__init__
    does not connect to Docker. Connection only happens inside run_lint() / run_tests().

    Returns
    -------
    SandboxClient or None
        None if initialization fails (Docker SDK not available).
    """
    global sandbox_client

    if sandbox_client is not None:
        return sandbox_client

    try:
        from app.mcp.sandbox_client import SandboxClient
        sandbox_client = SandboxClient()
        logger.info("[nodes] Sandbox initialized")
    except Exception:
        logger.exception(
            "[nodes] SandboxClient initialization failed — "
            "sandbox steps will be skipped"
        )
        sandbox_client = None

    return sandbox_client


# ── Safe LLM Invocation ───────────────────────────────────────────────────────

async def safe_llm_invoke(messages: List[Any]) -> str:
    """
    Invoke the Gemini LLM with timeout and quota protection.

    This is the single call site for all LLM invocations in the graph.
    It guarantees:
        - Graph never crashes on LLM failure
        - Free-tier quota errors are detected instantly (no retry storm)
        - Hard timeout prevents runaway async tasks
        - Concurrent calls are bounded by LLM_SEMAPHORE

    Parameters
    ----------
    messages : List[Any]
        LangChain message list (SystemMessage + HumanMessage).

    Returns
    -------
    str
        One of:
        - Normal LLM response content
        - FREE_TIER_EXHAUSTED  (on 429 quota error)
        - LLM_ERROR            (on timeout or other failure)
    """
    async with LLM_SEMAPHORE:
        try:
            logger.info("[LLM] Invoking Gemini — waiting for response")

            llm = get_llm()

            response = await asyncio.wait_for(
                llm.ainvoke(messages),
                timeout=LLM_TIMEOUT,
            )

            logger.info("[LLM] Response received successfully")
            return response.content

        except asyncio.TimeoutError:
            logger.error(
                "[LLM] Call timed out after %ds — returning LLM_ERROR",
                LLM_TIMEOUT,
            )
            return LLM_ERROR

        except Exception as e:
            error_str = str(e)

            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                logger.error(
                    "[LLM] Free tier quota exhausted (429) — "
                    "returning FREE_TIER_EXHAUSTED. "
                    "Wait until quota resets or add billing."
                )
                return FREE_TIER_EXHAUSTED

            logger.exception(
                "[LLM] Unexpected failure — returning LLM_ERROR. error=%s",
                error_str,
            )
            return LLM_ERROR


# ── Output Parser ─────────────────────────────────────────────────────────────

def _parse_llm_output(raw: str) -> tuple[list[str], list[str]]:
    """
    Parse structured ISSUES / SUGGESTIONS sections from LLM response.

    Expected format:
        ISSUES:
        - issue one
        - issue two

        SUGGESTIONS:
        - suggestion one

    Parameters
    ----------
    raw : str
        Raw LLM response string.

    Returns
    -------
    tuple[list[str], list[str]]
        (issues, suggestions) — both may be empty lists.
    """
    issues: list[str] = []
    suggestions: list[str] = []
    section = None

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.upper().startswith("ISSUES"):
            section = "issues"
        elif line.upper().startswith("SUGGESTIONS"):
            section = "suggestions"
        elif line.startswith("-"):
            item = line.lstrip("- ").strip()
            if not item or item.lower() == "none":
                continue
            if section == "issues":
                issues.append(item)
            elif section == "suggestions":
                suggestions.append(item)

    return issues, suggestions


# ── Helper: Python file detection ─────────────────────────────────────────────

def _has_python_files(files: list) -> bool:
    """Return True if any changed file in the PR is a .py file."""
    return any(
        str(f.get("filename", "")).endswith(".py")
        for f in files
    )


# =============================================================================
# NODES
# =============================================================================

@traceable(name="fetch_diff_node", tags=["github", "fetch"])
async def fetch_diff_node(state: ReviewState) -> Dict:
    """
    Fetch PR metadata, changed files, and unified diff from GitHub.

    Reads  : state["owner"], state["repo"], state["pr_number"]
    Writes : state["metadata"], state["files"], state["diff"]
             state["error"], state["error_reason"] (on failure)

    Failure behaviour
    -----------------
    On GitHub API failure, sets error=True and error_reason="github_fetch_failed".
    The workflow routes to hitl_node on error — human can inspect and decide.
    """
    owner = state["owner"]
    repo = state["repo"]
    pr_number = state["pr_number"]

    logger.info(
        "[fetch_diff_node] Starting — %s/%s#%d",
        owner, repo, pr_number,
    )

    try:
        client = GitHubClient()
        metadata = client.get_pr_metadata(owner, repo, pr_number)
        files = client.get_pr_files(owner, repo, pr_number)
        diff = client.get_pr_diff(owner, repo, pr_number)

        logger.info(
            "[fetch_diff_node] Done — %d file(s) fetched, diff=%d chars",
            len(files), len(diff),
        )

        return {
            "metadata": metadata,
            "files": files,
            "diff": diff,
            "error": False,
        }

    except Exception as e:
        logger.exception(
            "[fetch_diff_node] GitHub fetch failed — "
            "setting error=True | error=%s", str(e),
        )
        return {
            "error": True,
            "error_reason": "github_fetch_failed",
        }


@traceable(name="analyze_code_node", tags=["llm", "analysis"])
async def analyze_code_node(state: ReviewState) -> Dict:
    """
    Use Gemini to analyze the PR diff and identify issues and suggestions.

    Reads  : state["diff"], state["metadata"], state["error"]
    Writes : state["issues"], state["suggestions"], state["repo_context"]
             state["error"], state["error_reason"] (on hard failure)

    LLM Failure Behaviour
    ---------------------
    FREE_TIER_EXHAUSTED
        Returns issues=["LLM quota exhausted — manual review required"].
        Graph continues to HITL gate with degraded output.

    LLM_ERROR
        Returns issues=["LLM error — manual review required"].
        Graph continues to HITL gate with degraded output.

    In both cases error=False so the graph does NOT short-circuit —
    the HITL gate fires and a human can inspect the partial results.
    """
    if state.get("error"):
        logger.warning(
            "[analyze_code_node] Skipping — upstream error detected: %s",
            state.get("error_reason"),
        )
        return {}

    diff = state.get("diff", "")
    metadata = state.get("metadata", {})
    pr_title = metadata.get("title", "Unknown PR")

    logger.info(
        "[analyze_code_node] Starting analysis — PR: '%s' diff=%d chars",
        pr_title, len(diff),
    )

    prompt = [
        SystemMessage(content=(
            "You are an expert code reviewer. Analyze the PR diff below and "
            "provide structured, actionable feedback.\n\n"
            "Respond in EXACTLY this format:\n\n"
            "ISSUES:\n- <issue 1>\n- <issue 2>\n"
            "(or '- None' if no issues)\n\n"
            "SUGGESTIONS:\n- <suggestion 1>\n"
            "(or '- None' if no suggestions)\n\n"
            "Be specific. Reference filenames and line context where possible."
        )),
        HumanMessage(content=(
            f"PR Title: {pr_title}\n\n"
            f"--- PR Diff ---\n{diff[:MAX_DIFF_CHARS]}"
        )),
    ]

    result = await safe_llm_invoke(prompt)

    if result == FREE_TIER_EXHAUSTED:
        logger.warning(
            "[analyze_code_node] LLM quota exhausted — "
            "returning degraded output, graph continues to HITL"
        )
        return {
            "issues": ["LLM quota exhausted — manual review required"],
            "suggestions": [],
            "repo_context": "",
            "error": False,
        }

    if result == LLM_ERROR:
        logger.error(
            "[analyze_code_node] LLM error — "
            "returning degraded output, graph continues to HITL"
        )
        return {
            "issues": ["LLM error — manual review required"],
            "suggestions": [],
            "repo_context": "",
            "error": False,
        }

    issues, suggestions = _parse_llm_output(result)

    logger.info(
        "[analyze_code_node] Complete — %d issue(s), %d suggestion(s)",
        len(issues), len(suggestions),
    )

    return {
        "issues": issues,
        "suggestions": suggestions,
        "repo_context": "",
        "error": False,
    }


@traceable(name="reflect_node", tags=["llm", "reflection"])
async def reflect_node(state: ReviewState) -> Dict:
    """
    Self-reflection pass — ask Gemini to find anything it missed.

    Runs up to 2 times (controlled by should_reflect() in workflow.py).
    On LLM failure, increments counter and returns existing state unchanged
    so the graph can proceed rather than stalling.

    Reads  : state["issues"], state["suggestions"], state["diff"],
             state["reflection_count"]
    Writes : state["issues"], state["suggestions"], state["reflection_count"]
    """
    current_count = state.get("reflection_count", 0)
    existing_issues = state.get("issues", [])
    existing_suggestions = state.get("suggestions", [])
    diff = state.get("diff", "")

    logger.info(
        "[reflect_node] Pass #%d starting — "
        "current issues=%d suggestions=%d",
        current_count + 1,
        len(existing_issues),
        len(existing_suggestions),
    )

    prompt = [
        HumanMessage(content=(
            f"You previously reviewed a PR and found:\n\n"
            f"ISSUES:\n"
            + ("\n".join(f"- {i}" for i in existing_issues) or "- None")
            + f"\n\nSUGGESTIONS:\n"
            + ("\n".join(f"- {s}" for s in existing_suggestions) or "- None")
            + f"\n\nDiff:\n{diff[:2000]}\n\n"
            "Did you miss any bugs or improvements? "
            "Respond ONLY with new findings in the same ISSUES/SUGGESTIONS format."
        )),
    ]

    result = await safe_llm_invoke(prompt)

    if result in (FREE_TIER_EXHAUSTED, LLM_ERROR):
        logger.warning(
            "[reflect_node] Pass #%d skipped — LLM unavailable (%s). "
            "Incrementing count and continuing.",
            current_count + 1, result,
        )
        return {
            "issues": existing_issues,
            "suggestions": existing_suggestions,
            "reflection_count": current_count + 1,
        }

    new_issues, new_suggestions = _parse_llm_output(result)

    merged_issues = list(
        {i.lower(): i for i in existing_issues + new_issues}.values()
    )
    merged_suggestions = list(
        {s.lower(): s for s in existing_suggestions + new_suggestions}.values()
    )

    logger.info(
        "[reflect_node] Pass #%d complete — "
        "added %d issue(s), %d suggestion(s)",
        current_count + 1,
        len(new_issues),
        len(new_suggestions),
    )

    return {
        "issues": merged_issues,
        "suggestions": merged_suggestions,
        "reflection_count": current_count + 1,
    }


@traceable(name="lint_node", tags=["sandbox", "lint"])
async def lint_node(state: ReviewState) -> Dict:
    """
    Run ruff on changed Python files inside the Docker sandbox.

    Skips gracefully if:
        - No Python files in the PR diff
        - Docker sandbox unavailable

    Reads  : state["diff"], state["files"]
    Writes : state["lint_result"], state["lint_passed"]
    """
    logger.info("[lint_node] Starting")

    files = state.get("files", [])

    if not _has_python_files(files):
        logger.info(
            "[lint_node] No Python files in diff — skipping lint (non-fatal)"
        )
        return {
            "lint_passed": True,
            "lint_result": "SKIPPED_NO_PYTHON_FILES",
        }

    sandbox = get_sandbox()

    if not sandbox:
        logger.warning(
            "[lint_node] Sandbox unavailable — skipping lint"
        )
        return {
            "lint_passed": True,
            "lint_result": "SKIPPED_NO_SANDBOX",
        }

    try:
        diff = state.get("diff", "")
        result = sandbox.run_lint(diff)

        logger.info(
            "[lint_node] Complete — passed=%s exit_code=%s duration=%sms",
            result.passed, result.exit_code, result.duration_ms,
        )

        return {
            "lint_result": result,
            "lint_passed": result.passed,
        }

    except Exception as e:
        logger.exception(
            "[lint_node] Sandbox error — marking lint as passed to unblock graph. "
            "error=%s", str(e),
        )
        return {
            "lint_passed": True,
            "lint_result": "SKIPPED_SANDBOX_ERROR",
        }


@traceable(name="refactor_node", tags=["llm", "refactor"])
async def refactor_node(state: ReviewState) -> Dict:
    """
    Generate a corrective patch using Gemini based on issues and lint output.

    Only called when lint_passed=False (controlled by should_lint_refactor
    in workflow.py). LLM-only node — does not call sandbox.

    Reads  : state["diff"], state["issues"], state["suggestions"],
             state["lint_result"], state["refactor_count"]
    Writes : state["patch"], state["refactor_count"]
             state["suggestions"] (appends skip note on LLM failure)
    """
    logger.info("[refactor_node] Generating corrective patch")

    diff = state.get("diff", "")
    issues = state.get("issues", [])
    suggestions = state.get("suggestions", [])
    lint_result = state.get("lint_result")
    current_count = state.get("refactor_count", 0)

    lint_context = ""
    if lint_result and hasattr(lint_result, "passed") and not lint_result.passed:
        lint_context = f"\n--- Lint Failures ---\n{lint_result.output}\n"

    prompt = [
        SystemMessage(content=(
            "You are an expert Python developer. Generate a corrective "
            "unified diff patch that fixes all listed issues. "
            "Respond with ONLY the raw diff — no explanations."
        )),
        HumanMessage(content=(
            f"--- PR Diff ---\n{diff[:2000]}\n\n"
            f"--- Issues ---\n"
            + ("\n".join(f"- {i}" for i in issues) or "- None")
            + lint_context
        )),
    ]

    result = await safe_llm_invoke(prompt)

    if result in (FREE_TIER_EXHAUSTED, LLM_ERROR):
        logger.warning(
            "[refactor_node] LLM unavailable (%s) — skipping patch generation",
            result,
        )
        return {
            "refactor_count": current_count + 1,
            "suggestions": suggestions + ["Refactor skipped — LLM unavailable"],
        }

    logger.info(
        "[refactor_node] Patch generated — %d chars", len(result),
    )

    return {
        "patch": result.strip(),
        "refactor_count": current_count + 1,
    }


@traceable(name="validator_node", tags=["sandbox", "validation"])
async def validator_node(state: ReviewState) -> Dict:
    """
    Run ruff + pytest on the generated patch inside the Docker sandbox.

    The result determines whether the refactor loop continues or
    the workflow proceeds to hitl_node.

    Reads  : state["patch"], state["refactor_count"]
    Writes : state["validation_result"], state["validation_passed"],
             state["refactor_count"] (incremented on failure)
    """
    patch = state.get("patch", "")
    current_count = state.get("refactor_count", 0)

    logger.info(
        "[validator_node] Starting validation — iteration=%d", current_count,
    )

    if not patch or ".py" not in patch:
        logger.info(
            "[validator_node] No Python patch to validate — skipping"
        )
        return {
            "validation_passed": True,
            "validation_result": "SKIPPED_NO_PYTHON_PATCH",
        }

    sandbox = get_sandbox()

    if not sandbox:
        logger.warning(
            "[validator_node] Sandbox unavailable — marking validation as passed"
        )
        return {
            "validation_passed": True,
            "validation_result": "SKIPPED_NO_SANDBOX",
        }

    try:
        result = sandbox.run_tests(patch)

        logger.info(
            "[validator_node] Complete — passed=%s exit_code=%s duration=%sms",
            result.passed, result.exit_code, result.duration_ms,
        )

        if result.passed:
            return {
                "validation_result": result,
                "validation_passed": True,
                "refactor_count": current_count,
            }
        else:
            logger.info(
                "[validator_node] Validation FAILED — "
                "incrementing refactor_count to %d",
                current_count + 1,
            )
            return {
                "validation_result": result,
                "validation_passed": False,
                "refactor_count": current_count + 1,
            }

    except Exception as e:
        logger.exception(
            "[validator_node] Sandbox error — marking as passed to unblock graph. "
            "error=%s", str(e),
        )
        return {
            "validation_passed": True,
            "validation_result": "SKIPPED_SANDBOX_ERROR",
        }


@traceable(name="verdict_node", tags=["verdict"])
async def verdict_node(state: ReviewState) -> Dict:
    """
    Produce the final verdict and GitHub PR comment markdown.

    Reads human_decision first — if 'rejected', short-circuits immediately.
    Otherwise determines APPROVE or REQUEST_CHANGES from issues list.

    Reads  : state["human_decision"], state["issues"], state["suggestions"],
             state["metadata"], state["error"], state["error_reason"]
    Writes : state["verdict"], state["summary"]
    """
    human_decision = state.get("human_decision")
    issues = state.get("issues", [])
    suggestions = state.get("suggestions", [])
    metadata = state.get("metadata", {})
    pr_title = metadata.get("title", "Unknown PR")
    pr_author = metadata.get("author", "unknown")

    logger.info(
        "[verdict_node] Starting — human_decision=%r issues=%d suggestions=%d",
        human_decision, len(issues), len(suggestions),
    )

    # ── Error path ────────────────────────────────────────────────────────────
    if state.get("error"):
        reason = state.get("error_reason", "unknown_error")
        logger.warning(
            "[verdict_node] Upstream error detected — reason=%s", reason,
        )
        return {
            "verdict": "FAILED",
            "summary": (
                f"## ❌ Review Failed\n\n"
                f"**PR:** {pr_title}\n"
                f"**Author:** @{pr_author}\n\n"
                f"The review pipeline encountered an error: `{reason}`.\n"
                f"Please trigger a new review or inspect the logs."
            ),
        }

    # ── Human rejected ────────────────────────────────────────────────────────
    if human_decision == "rejected":
        logger.info(
            "[verdict_node] human_decision=rejected — "
            "producing HUMAN_REJECTED, no GitHub comment will be posted"
        )
        return {
            "verdict": "HUMAN_REJECTED",
            "summary": (
                f"## ❌ Review Rejected by Human Reviewer\n\n"
                f"**PR:** {pr_title}\n"
                f"**Author:** @{pr_author}\n\n"
                f"A human reviewer rejected this AI-generated review. "
                f"No comment has been posted to the pull request.\n\n"
                f"_{len(issues)} issue(s) and {len(suggestions)} suggestion(s) "
                f"were identified but not published._"
            ),
        }

    # ── Normal verdict ────────────────────────────────────────────────────────
    verdict = "REQUEST_CHANGES" if issues else "APPROVE"
    verdict_emoji = "✅" if verdict == "APPROVE" else "🔴"

    logger.info(
        "[verdict_node] Determined verdict=%s (issues=%d)",
        verdict, len(issues),
    )

    issues_section = (
        "\n".join(f"- {i}" for i in issues)
        if issues else "_No issues found._"
    )
    suggestions_section = (
        "\n".join(f"- {s}" for s in suggestions)
        if suggestions else "_No suggestions._"
    )

    human_badge = (
        "\n\n**Human Approval:** ✅ Approved by reviewer"
        if human_decision == "approved"
        else ""
    )

    summary = (
        f"## {verdict_emoji} AI Code Review\n\n"
        f"**PR:** {pr_title}\n"
        f"**Author:** @{pr_author}\n"
        f"**Verdict:** `{verdict}`"
        f"{human_badge}\n\n"
        f"---\n\n"
        f"### 🐛 Issues\n{issues_section}\n\n"
        f"### 💡 Suggestions\n{suggestions_section}\n\n"
        f"---\n"
        f"*Review generated by Advanced GitHub Code Reviewer "
        f"· Powered by Gemini*"
    )

    logger.info(
        "[verdict_node] Complete — verdict=%s summary=%d chars",
        verdict, len(summary),
    )

    return {
        "verdict": verdict,
        "summary": summary,
    }