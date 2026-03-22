"""
app/graph/nodes.py

LangGraph Agent Nodes — P4 Production Version
----------------------------------------------
Each function is one node in the review workflow graph.

Node execution order (defined in workflow.py):
    fetch_diff_node
        → retrieve_context_node  ★ P4 — query ChromaDB for repo context
        → grade_context_node     ★ P4 — grade relevance of retrieved context
        → analyze_code_node          — Gemini analysis with graded context
        → reflect_node               — self-reflection (REFLECTION_PASSES=0 dev)
        → lint_node                  — ruff in Docker sandbox
        → refactor_node              — Gemini corrective patch (conditional)
        → validator_node             — ruff + pytest in Docker (conditional)
        → memory_write_node      ★ P4 — write findings to ChromaDB memory
        → hitl_node              [interrupt_before pauses here]
        → verdict_node
        → END

Design Principles
-----------------
1. NO import-time side effects — all heavy objects initialized lazily.
2. ALL nodes are async — compatible with LangGraph's async runner.
3. LLM failures are SOFT — graph never crashes on quota errors.
4. Quota detection is INSTANT — max_retries=0 prevents retry storms.
5. Every node logs entry, exit, and any failure with structured context.
6. sandbox_client and _llm_instance are module-level for patch.object in tests.
7. ChromaDB is OPTIONAL — all nodes skip gracefully if collection unavailable.

LLM Failure Signals
-------------------
FREE_TIER_EXHAUSTED  — 429 quota error from Google API
LLM_ERROR            — any other LLM failure (timeout, server error)

When these signals are returned by safe_llm_invoke():
    - The node continues with degraded output (empty issues / skipped step)
    - The graph does NOT crash
    - The HITL gate still fires — human can review degraded output
    - verdict_node produces a clear summary of what happened

ChromaDB / RAG (P4)
-------------------
retrieve_context_node queries ChromaDB with the PR diff as the search query.
grade_context_node asks Gemini to grade whether the retrieved context is
relevant to the current PR. If grade is "yes", context is passed to
analyze_code_node. If "no", an empty context is used instead (CRAG pattern).
memory_write_node writes the review findings (issues + verdict) to ChromaDB
after the review completes so future reviews on the same repo are context-aware.

Embedding Model
---------------
gemini-embedding-001 is used for ChromaDB embeddings.
This model is confirmed available on the project API key.
ChromaDB collection is initialized lazily — returns None on any failure.
All nodes that use ChromaDB check for None before calling.

Async Safety
------------
LLM_SEMAPHORE limits concurrent Gemini calls to 1 (free tier safe).
LLM_TIMEOUT enforces a hard 30-second ceiling per call.
A 2-second cooldown after each successful call prevents RPM burst.

Patching in Tests
-----------------
    from unittest.mock import patch, AsyncMock
    import app.graph.nodes as nodes

    with patch.object(nodes, "sandbox_client", mock_sc):
        ...

    with patch.object(nodes, "_llm_instance", mock_llm):
        ...

    with patch.object(nodes, "_chroma_collection", mock_collection):
        ...
"""

import os
import sys
import time
import uuid
import asyncio
from collections import deque
from typing import List, Any, Dict, Optional, Tuple

from langsmith import traceable
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage

from app.graph.state import ReviewState
from app.mcp.github_client import GitHubClient
from app.core.exceptions import CustomException
from app.core.logger import get_logger

logger = get_logger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

LLM_TIMEOUT: int = 30
"""Hard ceiling (seconds) for a single Gemini API call."""

LLM_SEMAPHORE: asyncio.Semaphore = asyncio.Semaphore(1)
"""
Limits concurrent LLM calls to 1 on the free tier.
Prevents RPM burst errors when multiple requests arrive simultaneously.
Increase to 2 when billing is enabled.
"""

MAX_DIFF_CHARS: int = 4000
"""Maximum diff characters sent to Gemini — keeps prompt within token limits."""

MAX_CONTEXT_CHARS: int = 2000
"""Maximum repo context characters injected into the analysis prompt."""

FREE_TIER_EXHAUSTED: str = "FREE_TIER_EXHAUSTED"
"""Sentinel returned by safe_llm_invoke() on 429 quota error."""

LLM_ERROR: str = "LLM_ERROR"
"""Sentinel returned by safe_llm_invoke() on timeout or unexpected failure."""

CONTEXT_GRADE_YES: str = "yes"
"""Grade returned by grade_context_node when retrieved context is relevant."""

CONTEXT_GRADE_NO: str = "no"
"""Grade returned by grade_context_node when retrieved context is not relevant."""


# ── Module-level lazy singletons (patchable in tests) ────────────────────────

_llm_instance: Optional[ChatGoogleGenerativeAI] = None
"""
Lazily initialized Gemini chat model instance.
Access via get_llm() — never reference this directly outside get_llm().
Module-level so patch.object(nodes, '_llm_instance', mock) works in tests.
"""

sandbox_client = None
"""
Lazily initialized SandboxClient.
Module-level so patch.object(nodes, 'sandbox_client', mock_sc) works in tests.
"""

_chroma_collection = None
"""
Lazily initialized ChromaDB collection.
Module-level so patch.object(nodes, '_chroma_collection', mock) works in tests.
Set to None if ChromaDB init fails — all RAG nodes check for None gracefully.
"""

_chroma_embedder: Optional[GoogleGenerativeAIEmbeddings] = None
"""
Lazily initialized Gemini embedding model for ChromaDB queries.
Uses gemini-embedding-001 — confirmed available on the project API key.
"""


# ── Burst Tracking ────────────────────────────────────────────────────────────

_LLM_CALL_HISTORY: deque = deque(maxlen=100)
"""Sliding window of LLM call timestamps for burst detection logging."""


def _record_llm_call() -> int:
    """
    Record an LLM call timestamp and return the burst count in the last 10s.

    Used only for observability logging — does not throttle calls.
    The semaphore handles actual concurrency limiting.

    Returns
    -------
    int
        Number of LLM calls recorded in the last 10 seconds.
    """
    now = time.time()
    _LLM_CALL_HISTORY.append(now)
    window_seconds = 10
    return sum(1 for t in _LLM_CALL_HISTORY if now - t <= window_seconds)


# ── LLM Initialization ────────────────────────────────────────────────────────

def get_llm() -> ChatGoogleGenerativeAI:
    """
    Return the module-level Gemini chat model, initializing on first call.

    Configuration
    -------------
    model        : gemini-2.5-flash-lite — confirmed working on project API key
    temperature  : 0                     — deterministic output for code review
    max_retries  : 0                     — CRITICAL: prevents internal SDK retry
                                           storms that silently exhaust free quota.
                                           safe_llm_invoke() handles retry logic.

    The module-level singleton pattern ensures the model is only initialized
    once per process lifetime, avoiding repeated API key validation overhead.

    Returns
    -------
    ChatGoogleGenerativeAI
        Initialized Gemini chat model instance.

    Raises
    ------
    CustomException
        If GOOGLE_API_KEY is not set in environment.
    """
    global _llm_instance

    if _llm_instance is not None:
        return _llm_instance

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error(
            "[nodes] get_llm: GOOGLE_API_KEY not found in environment — "
            "check that app/core/config.py loaded .env correctly"
        )
        raise CustomException("GOOGLE_API_KEY is not set in environment", sys)

    logger.info(
        "[nodes] get_llm: Initializing Gemini LLM — "
        "model=gemini-2.5-flash-lite max_retries=0 temperature=0"
    )

    _llm_instance = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        max_retries=0,
    )

    logger.info("[nodes] Primary LLM initialized successfully")
    return _llm_instance


# ── Sandbox Initialization ────────────────────────────────────────────────────

def get_sandbox():
    """
    Return the module-level SandboxClient, initializing on first call.

    Safe to call when Docker Desktop is not running — SandboxClient.__init__
    does not connect to Docker. Connection only happens inside run_lint()
    and run_tests() when they are actually invoked.

    Returns
    -------
    SandboxClient or None
        None if Docker SDK is not available or initialization fails.
        Callers must check for None before using.
    """
    global sandbox_client

    if sandbox_client is not None:
        return sandbox_client

    try:
        from app.mcp.sandbox_client import SandboxClient
        sandbox_client = SandboxClient()
        logger.info("[nodes] get_sandbox: SandboxClient initialized successfully")
    except Exception:
        logger.exception(
            "[nodes] get_sandbox: SandboxClient initialization failed — "
            "lint_node and validator_node will be skipped"
        )
        sandbox_client = None

    return sandbox_client


# ── ChromaDB Initialization ───────────────────────────────────────────────────

def get_chroma_collection() -> Tuple[Optional[Any], Optional[GoogleGenerativeAIEmbeddings]]:
    """
    Return the module-level ChromaDB collection and embedder, initializing lazily.

    Embedding Model
    ---------------
    Uses gemini-embedding-001 — confirmed available on the project API key.
    Do NOT use text-embedding-004 (not available on this key).
    Do NOT use gemini-embedding-2-preview (preview, unstable for production).

    ChromaDB Path
    -------------
    Persistent client at ./chroma_store — created on first call.
    Collection name: "repo_context"
    This collection is empty until the P4 indexing pipeline populates it.

    Returns
    -------
    Tuple[Optional[collection], Optional[embedder]]
        (collection, embedder) on success.
        (None, None) on any failure — all callers handle this gracefully.

    Notes
    -----
    Module-level so patch.object(nodes, '_chroma_collection', mock) works in tests.
    """
    global _chroma_collection, _chroma_embedder

    if _chroma_collection is not None:
        return _chroma_collection, _chroma_embedder

    try:
        import chromadb
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.warning(
                "[nodes] get_chroma_collection: GOOGLE_API_KEY not set — "
                "ChromaDB disabled"
            )
            return None, None

        logger.info(
            "[nodes] get_chroma_collection: Initializing ChromaDB — "
            "model=gemini-embedding-001 path=./chroma_store"
        )

        _chroma_embedder = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=api_key,
            task_type="RETRIEVAL_DOCUMENT",
        )

        client = chromadb.PersistentClient(path="./chroma_store")

        _chroma_collection = client.get_or_create_collection(
            name="repo_context",
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            "[nodes] get_chroma_collection: ChromaDB ready — "
            "collection=repo_context count=%d",
            _chroma_collection.count(),
        )

        return _chroma_collection, _chroma_embedder

    except Exception as e:
        logger.warning(
            "[nodes] get_chroma_collection: ChromaDB init failed — "
            "RAG nodes will be skipped. error=%s",
            str(e),
        )
        return None, None


# ── Safe LLM Invocation ───────────────────────────────────────────────────────

async def safe_llm_invoke(messages: List[Any]) -> str:
    """
    Invoke Gemini with quota protection, timeout, and burst prevention.

    This is the SINGLE call site for all LLM invocations in the graph.
    It guarantees:
        - Graph never crashes on LLM failure
        - Free-tier quota errors are detected instantly (no retry storm)
        - Hard 30s timeout prevents runaway async tasks
        - Concurrent calls bounded by LLM_SEMAPHORE (1 on free tier)
        - 2-second cooldown after success prevents RPM burst

    Parameters
    ----------
    messages : List[Any]
        LangChain message list (SystemMessage + HumanMessage).

    Returns
    -------
    str
        One of:
        - Normal LLM response content string
        - FREE_TIER_EXHAUSTED  (on 429 quota error — after 60s wait)
        - LLM_ERROR            (on timeout or unexpected failure)
    """
    call_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    logger.info(
        "[LLM][%s] REQUEST_START | queued_for_semaphore | msg_count=%d",
        call_id, len(messages),
    )

    queue_start = time.time()

    async with LLM_SEMAPHORE:
        queue_wait = time.time() - queue_start
        burst_count = _record_llm_call()

        logger.info(
            "[LLM][%s] ACQUIRED | queue_wait=%.2fs | "
            "concurrent_limit=%d | burst_last_10s=%d",
            call_id,
            queue_wait,
            LLM_SEMAPHORE._value,
            burst_count,
        )

        try:
            llm = get_llm()
            invoke_start = time.time()

            logger.info(
                "[LLM][%s] INVOKE_START | timeout=%ds",
                call_id, LLM_TIMEOUT,
            )

            response = await asyncio.wait_for(
                llm.ainvoke(messages),
                timeout=LLM_TIMEOUT,
            )

            invoke_time = time.time() - invoke_start
            total_time = time.time() - start_time

            logger.info(
                "[LLM][%s] SUCCESS | invoke_time=%.2fs | total_time=%.2fs",
                call_id, invoke_time, total_time,
            )

            # Cooldown prevents back-to-back RPM burst
            cooldown_seconds = 2
            logger.info(
                "[LLM][%s] COOLDOWN | sleeping=%ds to prevent RPM burst",
                call_id, cooldown_seconds,
            )
            await asyncio.sleep(cooldown_seconds)

            return response.content

        except asyncio.TimeoutError:
            total_time = time.time() - start_time
            logger.error(
                "[LLM][%s] TIMEOUT | exceeded=%ds | total_time=%.2fs — "
                "returning LLM_ERROR",
                call_id, LLM_TIMEOUT, total_time,
            )
            return LLM_ERROR

        except Exception as e:
            total_time = time.time() - start_time
            error_str = str(e)

            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                logger.error(
                    "[LLM][%s] RATE_LIMIT | 429 hit | total_time=%.2fs | "
                    "sleeping=60s before returning FREE_TIER_EXHAUSTED",
                    call_id, total_time,
                )
                await asyncio.sleep(60)
                return FREE_TIER_EXHAUSTED

            logger.exception(
                "[LLM][%s] FAILURE | total_time=%.2fs | error=%s — "
                "returning LLM_ERROR",
                call_id, total_time, error_str,
            )
            return LLM_ERROR

        finally:
            logger.info("[LLM][%s] REQUEST_END", call_id)


# ── Output Parser ─────────────────────────────────────────────────────────────

def _parse_llm_output(raw: str) -> Tuple[List[str], List[str]]:
    """
    Parse structured ISSUES / SUGGESTIONS sections from LLM response.

    Expected LLM response format:
        ISSUES:
        - issue one
        - issue two

        SUGGESTIONS:
        - suggestion one

    Handles case-insensitive section headers and skips "- None" lines.

    Parameters
    ----------
    raw : str
        Raw string content from LLM response.

    Returns
    -------
    Tuple[List[str], List[str]]
        (issues, suggestions) — both may be empty lists if none found.
    """
    issues: List[str] = []
    suggestions: List[str] = []
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
    """Return True if any changed file in the PR ends with .py."""
    return any(
        str(f.get("filename", "")).endswith(".py")
        for f in files
    )


# =============================================================================
# NODES — P1/P2 (unchanged)
# =============================================================================

@traceable(name="fetch_diff_node", tags=["github", "fetch"])
async def fetch_diff_node(state: ReviewState) -> Dict:
    """
    Fetch PR metadata, changed files, and unified diff from GitHub.

    Reads  : state["owner"], state["repo"], state["pr_number"]
    Writes : state["metadata"], state["files"], state["diff"]
             state["error"], state["error_reason"] (on failure)

    Failure Behaviour
    -----------------
    On GitHub API failure, sets error=True and error_reason="github_fetch_failed".
    check_error() in workflow.py routes directly to hitl_node on error.
    Human can inspect the partial state and decide what to do.
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


# =============================================================================
# NODES — P4 NEW
# =============================================================================

@traceable(name="retrieve_context_node", tags=["chromadb", "rag", "p4"])
async def retrieve_context_node(state: ReviewState) -> Dict:
    """
    Query ChromaDB for relevant codebase context for the current PR.

    Uses the PR diff as the search query to find semantically similar
    code chunks previously indexed from the same repository. Retrieved
    context is passed to grade_context_node for relevance grading before
    being injected into analyze_code_node.

    Reads  : state["diff"], state["metadata"], state["error"]
    Writes : state["raw_context"]  — raw retrieved docs (ungraded)

    P4 Notes
    --------
    ChromaDB is empty until the indexing pipeline (POST /repos/index) runs.
    If collection is empty or unavailable, raw_context is set to "" and
    the graph continues without context — this is the expected behavior
    before any repo has been indexed.

    Failure Behaviour
    -----------------
    Any ChromaDB failure sets raw_context="" and continues.
    The graph NEVER crashes on RAG failures — context is optional.
    """
    if state.get("error"):
        logger.warning(
            "[retrieve_context_node] Skipping — upstream error: %s",
            state.get("error_reason"),
        )
        return {"raw_context": ""}

    diff = state.get("diff", "")
    metadata = state.get("metadata", {})
    pr_title = metadata.get("title", "")
    owner = state.get("owner", "")
    repo = state.get("repo", "")

    logger.info(
        "[retrieve_context_node] Starting — %s/%s PR: '%s' diff=%d chars",
        owner, repo, pr_title, len(diff),
    )

    collection, embedder = get_chroma_collection()

    if not collection or not embedder:
        logger.info(
            "[retrieve_context_node] ChromaDB unavailable — "
            "skipping retrieval, raw_context=''"
        )
        return {"raw_context": ""}

    if collection.count() == 0:
        logger.info(
            "[retrieve_context_node] ChromaDB collection is empty — "
            "no context available yet. "
            "Run POST /repos/index to index the repository first."
        )
        return {"raw_context": ""}

    try:
        # Use diff + PR title as search query for best semantic match
        query_text = f"{pr_title}\n{diff[:1000]}"

        logger.info(
            "[retrieve_context_node] Querying ChromaDB — "
            "query_chars=%d collection_count=%d",
            len(query_text), collection.count(),
        )

        query_embedding = embedder.embed_query(query_text)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(3, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not docs:
            logger.info(
                "[retrieve_context_node] No results returned from ChromaDB"
            )
            return {"raw_context": ""}

        # Format retrieved context with source metadata
        context_parts = []
        for doc, meta, dist in zip(docs, metadatas, distances):
            source = meta.get("source", "unknown")
            context_parts.append(
                f"### {source} (similarity={1-dist:.2f})\n{doc}"
            )

        raw_context = "\n\n".join(context_parts)

        logger.info(
            "[retrieve_context_node] Retrieved %d chunk(s) — "
            "total_chars=%d",
            len(docs), len(raw_context),
        )

        return {"raw_context": raw_context}

    except Exception as e:
        logger.warning(
            "[retrieve_context_node] ChromaDB query failed — "
            "continuing without context. error=%s",
            str(e),
        )
        return {"raw_context": ""}


@traceable(name="grade_context_node", tags=["llm", "rag", "crag", "p4"])
async def grade_context_node(state: ReviewState) -> Dict:
    """
    Grade the relevance of retrieved ChromaDB context for the current PR.

    Implements the Corrective RAG (CRAG) pattern — ask Gemini whether
    the retrieved context is actually relevant to the PR diff before
    injecting it into the analysis prompt. This prevents irrelevant
    context from confusing the analyzer.

    If the grade is "yes":  repo_context = raw_context (passed to analyzer)
    If the grade is "no":   repo_context = ""          (no context injected)
    If LLM fails:           repo_context = raw_context (fail open — use context)

    Reads  : state["raw_context"], state["diff"], state["error"]
    Writes : state["repo_context"]  — graded context (empty if irrelevant)
             state["context_grade"] — "yes" | "no" | "skipped"

    Failure Behaviour
    -----------------
    On LLM failure, fails OPEN — passes raw_context through unchanged.
    This is intentional: better to include potentially irrelevant context
    than to drop relevant context due to a grading LLM failure.
    The HITL gate still fires so a human can review the output.
    """
    if state.get("error"):
        logger.warning(
            "[grade_context_node] Skipping — upstream error: %s",
            state.get("error_reason"),
        )
        return {"repo_context": "", "context_grade": "skipped"}

    raw_context = state.get("raw_context", "")
    diff = state.get("diff", "")

    # No context to grade — skip LLM call
    if not raw_context:
        logger.info(
            "[grade_context_node] No raw_context to grade — "
            "skipping LLM call, repo_context=''"
        )
        return {"repo_context": "", "context_grade": "skipped"}

    logger.info(
        "[grade_context_node] Grading retrieved context — "
        "raw_context_chars=%d diff_chars=%d",
        len(raw_context), len(diff),
    )

    prompt = [
        SystemMessage(content=(
            "You are a relevance grader for a code review system.\n\n"
            "Your job is to determine if retrieved codebase context is "
            "relevant to a given PR diff.\n\n"
            "Respond with ONLY one word: 'yes' or 'no'.\n"
            "- 'yes' if the context contains code, patterns, or conventions "
            "that would help review the PR diff.\n"
            "- 'no' if the context is unrelated to the PR changes."
        )),
        HumanMessage(content=(
            f"--- Retrieved Context ---\n{raw_context[:1000]}\n\n"
            f"--- PR Diff ---\n{diff[:1000]}\n\n"
            "Is this context relevant to reviewing this PR diff? "
            "Respond with ONLY 'yes' or 'no'."
        )),
    ]

    result = await safe_llm_invoke(prompt)

    # Handle LLM failures — fail open (use context)
    if result in (FREE_TIER_EXHAUSTED, LLM_ERROR):
        logger.warning(
            "[grade_context_node] LLM unavailable (%s) — "
            "failing open, passing raw_context through",
            result,
        )
        return {
            "repo_context": raw_context[:MAX_CONTEXT_CHARS],
            "context_grade": "skipped",
        }

    grade = result.strip().lower()

    # Normalize to yes/no — LLM sometimes adds punctuation
    if "yes" in grade:
        grade = CONTEXT_GRADE_YES
    elif "no" in grade:
        grade = CONTEXT_GRADE_NO
    else:
        logger.warning(
            "[grade_context_node] Unexpected grade response: '%s' — "
            "defaulting to 'yes' (fail open)",
            result.strip(),
        )
        grade = CONTEXT_GRADE_YES

    if grade == CONTEXT_GRADE_YES:
        logger.info(
            "[grade_context_node] Grade=YES — "
            "context is relevant, injecting into analyzer "
            "context_chars=%d",
            min(len(raw_context), MAX_CONTEXT_CHARS),
        )
        return {
            "repo_context": raw_context[:MAX_CONTEXT_CHARS],
            "context_grade": grade,
        }
    else:
        logger.info(
            "[grade_context_node] Grade=NO — "
            "context not relevant, discarding"
        )
        return {
            "repo_context": "",
            "context_grade": grade,
        }


@traceable(name="analyze_code_node", tags=["llm", "analysis"])
async def analyze_code_node(state: ReviewState) -> Dict:
    """
    Use Gemini to analyze the PR diff and identify issues and suggestions.

    P4 Update: Now includes graded repo_context in the analysis prompt
    when available, giving Gemini awareness of the existing codebase
    conventions and patterns.

    Reads  : state["diff"], state["metadata"], state["repo_context"],
             state["error"]
    Writes : state["issues"], state["suggestions"], state["repo_context"]
             state["error"], state["error_reason"] (on hard failure)

    LLM Failure Behaviour
    ---------------------
    FREE_TIER_EXHAUSTED → degraded output, graph continues to HITL
    LLM_ERROR           → degraded output, graph continues to HITL
    In both cases error=False — HITL gate fires for human inspection.
    """
    if state.get("error"):
        logger.warning(
            "[analyze_code_node] Skipping — upstream error: %s",
            state.get("error_reason"),
        )
        return {}

    diff = state.get("diff", "")
    metadata = state.get("metadata", {})
    pr_title = metadata.get("title", "Unknown PR")
    repo_context = state.get("repo_context", "")
    context_grade = state.get("context_grade", "skipped")

    logger.info(
        "[analyze_code_node] Starting analysis — "
        "PR: '%s' diff=%d chars context=%d chars context_grade=%s",
        pr_title, len(diff), len(repo_context), context_grade,
    )

    # Build context section for prompt
    context_section = ""
    if repo_context:
        context_section = (
            f"\n\n--- Codebase Context (from repository index) ---\n"
            f"{repo_context}\n\n"
            f"Use the above context to understand existing conventions "
            f"and patterns when reviewing the diff below.\n"
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
            f"PR Title: {pr_title}"
            f"{context_section}"
            f"\n\n--- PR Diff ---\n{diff[:MAX_DIFF_CHARS]}"
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
        "error": False,
    }


@traceable(name="reflect_node", tags=["llm", "reflection"])
async def reflect_node(state: ReviewState) -> Dict:
    """
    Self-reflection pass — ask Gemini to find anything it missed.

    Controlled by REFLECTION_PASSES in workflow.py.
    Set to 0 in development (skipped entirely).
    Set to 1 in staging/production for one additional pass.

    On LLM failure, increments counter and returns existing issues unchanged
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
        current_count + 1, len(existing_issues), len(existing_suggestions),
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

    # Deduplicate by lowercase key while preserving original case
    merged_issues = list(
        {i.lower(): i for i in existing_issues + new_issues}.values()
    )
    merged_suggestions = list(
        {s.lower(): s for s in existing_suggestions + new_suggestions}.values()
    )

    logger.info(
        "[reflect_node] Pass #%d complete — "
        "added %d issue(s), %d suggestion(s)",
        current_count + 1, len(new_issues), len(new_suggestions),
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
        - Docker sandbox unavailable (Docker Desktop not running)

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

    logger.info("[refactor_node] Patch generated — %d chars", len(result))

    return {
        "patch": result.strip(),
        "refactor_count": current_count + 1,
    }


@traceable(name="validator_node", tags=["sandbox", "validation"])
async def validator_node(state: ReviewState) -> Dict:
    """
    Run ruff + pytest on the generated patch inside the Docker sandbox.

    The result determines whether the refactor loop continues (validation
    failed, attempts remaining) or the workflow proceeds to memory_write_node.

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
        logger.info("[validator_node] No Python patch to validate — skipping")
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


@traceable(name="memory_write_node", tags=["chromadb", "memory", "p4"])
async def memory_write_node(state: ReviewState) -> Dict:
    """
    Write review findings to ChromaDB long-term memory after each review.

    After each completed review, key findings (issues, verdict, PR title)
    are embedded and stored in ChromaDB. Future reviews on the same repo
    will retrieve these findings via retrieve_context_node, making the
    reviewer progressively more codebase-aware over time.

    This node always returns an empty dict — it has no state outputs.
    It only writes to ChromaDB as a side effect.

    Reads  : state["issues"], state["suggestions"], state["verdict"],
             state["metadata"], state["owner"], state["repo"]
    Writes : {} (no state changes — ChromaDB write is a side effect)

    Failure Behaviour
    -----------------
    Any ChromaDB write failure is logged as a warning and the graph
    continues. Memory writes are best-effort — a failure here should
    never block the review from completing.

    Notes
    -----
    Each memory entry is stored with metadata:
        - source: "{owner}/{repo}"
        - type: "review_memory"
        - pr_title: PR title
        - verdict: final verdict
    This metadata enables future filtering by repo when querying.
    """
    if state.get("error"):
        logger.warning(
            "[memory_write_node] Skipping — upstream error: %s",
            state.get("error_reason"),
        )
        return {}

    issues = state.get("issues", [])
    suggestions = state.get("suggestions", [])
    metadata = state.get("metadata", {})
    pr_title = metadata.get("title", "Unknown PR")
    pr_author = metadata.get("author", "unknown")
    owner = state.get("owner", "")
    repo = state.get("repo", "")
    pr_number = state.get("pr_number", 0)
    verdict = state.get("verdict", "")

    logger.info(
        "[memory_write_node] Writing review memory — "
        "%s/%s#%d PR: '%s' issues=%d verdict=%s",
        owner, repo, pr_number, pr_title, len(issues), verdict,
    )

    collection, embedder = get_chroma_collection()

    if not collection or not embedder:
        logger.info(
            "[memory_write_node] ChromaDB unavailable — skipping memory write"
        )
        return {}

    if not issues and not suggestions:
        logger.info(
            "[memory_write_node] No issues or suggestions to store — "
            "skipping memory write (APPROVE verdict with no findings)"
        )
        return {}

    try:
        # Build memory document text
        issues_text = "\n".join(f"- {i}" for i in issues) if issues else "None"
        suggestions_text = (
            "\n".join(f"- {s}" for s in suggestions)
            if suggestions else "None"
        )

        memory_doc = (
            f"PR Review Memory — {owner}/{repo}#{pr_number}\n"
            f"Title: {pr_title}\n"
            f"Author: @{pr_author}\n"
            f"Verdict: {verdict}\n\n"
            f"Issues Found:\n{issues_text}\n\n"
            f"Suggestions:\n{suggestions_text}"
        )

        # Generate unique document ID
        doc_id = f"review_memory_{owner}_{repo}_{pr_number}_{int(time.time())}"

        # Embed and store
        embedding = embedder.embed_query(memory_doc)

        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[memory_doc],
            metadatas=[{
                "source":   f"{owner}/{repo}",
                "type":     "review_memory",
                "pr_title": pr_title,
                "pr_number": str(pr_number),
                "verdict":  verdict,
                "author":   pr_author,
            }],
        )

        logger.info(
            "[memory_write_node] Memory written successfully — "
            "doc_id=%s collection_count=%d",
            doc_id, collection.count(),
        )

    except Exception as e:
        logger.warning(
            "[memory_write_node] ChromaDB write failed — "
            "graph continues without memory write. error=%s",
            str(e),
        )

    return {}


@traceable(name="verdict_node", tags=["verdict"])
async def verdict_node(state: ReviewState) -> Dict:
    """
    Produce the final verdict and GitHub PR comment markdown.

    Checks human_decision first — rejected produces HUMAN_REJECTED.
    Otherwise determines APPROVE or REQUEST_CHANGES from issues list.
    On upstream error, produces FAILED verdict with error details.

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
    context_grade = state.get("context_grade", "skipped")

    logger.info(
        "[verdict_node] Starting — human_decision=%r issues=%d "
        "suggestions=%d context_grade=%s",
        human_decision, len(issues), len(suggestions), context_grade,
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
            "HUMAN_REJECTED, no GitHub comment posted"
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

    # Context badge — shows whether RAG context was used
    context_badge = ""
    if context_grade == CONTEXT_GRADE_YES:
        context_badge = "\n**Context:** 🧠 Codebase context used"
    elif context_grade == CONTEXT_GRADE_NO:
        context_badge = "\n**Context:** ⚪ Retrieved context not relevant"

    summary = (
        f"## {verdict_emoji} AI Code Review\n\n"
        f"**PR:** {pr_title}\n"
        f"**Author:** @{pr_author}\n"
        f"**Verdict:** `{verdict}`"
        f"{human_badge}"
        f"{context_badge}\n\n"
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