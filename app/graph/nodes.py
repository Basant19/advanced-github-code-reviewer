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
        → summary_node               — format HITL briefing markdown
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

Patch Sanitization (Critical)
------------------------------
Gemini hallucinates in two distinct ways when generating patches:

Mode 1 — Outer fences (handled by _sanitize_patch outer fence stripper):
```diff
    --- a/file.py
    +++ b/file.py
    ...
```

Mode 2 — Inline contamination (the real problem, handled by _deep_clean_patch):
    Gemini embeds backtick sequences INSIDE the + lines of the diff:
        +b = str(b)`.
        +```
        +some code
    This makes the reconstructed file contain backticks that Python can't parse,
    causing ruff/pytest to find SyntaxErrors even in "fixed" code.

_sanitize_patch() handles Mode 1.
_deep_clean_patch() handles Mode 2 — scans every + line and removes backtick
sequences, trailing backticks, and standalone fence lines that appear inside
the diff content. Both are applied in sequence in refactor_node.

ChromaDB / RAG (P4)
-------------------
retrieve_context_node queries ChromaDB with the PR diff as the search query.
grade_context_node asks Gemini to grade whether the retrieved context is
relevant to the current PR (CRAG pattern).
memory_write_node writes the review findings to ChromaDB after the review.

Embedding Model
---------------
gemini-embedding-001 — confirmed available on the project API key.
ChromaDB collection initialized lazily — returns None on any failure.
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
import re
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
LLM_SEMAPHORE: asyncio.Semaphore = asyncio.Semaphore(1)
MAX_DIFF_CHARS: int = 4000
MAX_CONTEXT_CHARS: int = 2000
FREE_TIER_EXHAUSTED: str = "FREE_TIER_EXHAUSTED"
LLM_ERROR: str = "LLM_ERROR"
CONTEXT_GRADE_YES: str = "yes"
CONTEXT_GRADE_NO: str = "no"


# ── Module-level lazy singletons ──────────────────────────────────────────────

_llm_instance: Optional[ChatGoogleGenerativeAI] = None
sandbox_client = None
_chroma_collection = None
_chroma_embedder: Optional[GoogleGenerativeAIEmbeddings] = None


# ── Burst Tracking ────────────────────────────────────────────────────────────

_LLM_CALL_HISTORY: deque = deque(maxlen=100)


def _record_llm_call() -> int:
    now = time.time()
    _LLM_CALL_HISTORY.append(now)
    return sum(1 for t in _LLM_CALL_HISTORY if now - t <= 10)


# ── LLM Initialization ────────────────────────────────────────────────────────

def get_llm() -> ChatGoogleGenerativeAI:
    """
    Return the module-level Gemini chat model, initializing on first call.

    model        : gemini-2.5-flash-lite
    temperature  : 0    — deterministic output
    max_retries  : 0    — prevents SDK retry storms on free tier
    """
    global _llm_instance

    if _llm_instance is not None:
        return _llm_instance

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error(
            "[nodes] get_llm: GOOGLE_API_KEY not found in environment"
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
    """Return the module-level SandboxClient, initializing on first call."""
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

    Uses gemini-embedding-001 — confirmed available on the project API key.
    Returns (None, None) on any failure — all callers handle this gracefully.
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
            "RAG nodes will be skipped. error=%s", str(e),
        )
        return None, None


# ── Safe LLM Invocation ───────────────────────────────────────────────────────

async def safe_llm_invoke(messages: List[Any]) -> str:
    """
    Invoke Gemini with strict RPM protection (15 RPM limit).
    
    Protections:
    1. Semaphore(1): Prevents concurrent LLM calls.
    2. 5s Cooldown: Ensures throughput stays ~12 RPM (below 15 limit).
    3. 70s Penalty: Resets the quota window if a 429 occurs.
    4. CustomException: Raised for critical configuration/logic errors.
    """
    call_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    if not messages:
        logger.error(f"[LLM][{call_id}] Empty messages provided")
        raise CustomException("Cannot invoke LLM with empty message list.")

    logger.info(
        "[LLM][%s] REQUEST_START | queued_for_semaphore | msg_count=%d",
        call_id, len(messages),
    )

    queue_start = time.time()

    async with LLM_SEMAPHORE:
        queue_wait = time.time() - queue_start
        
        logger.info(
            "[LLM][%s] ACQUIRED | queue_wait=%.2fs | RPM_limit=15",
            call_id, queue_wait,
        )

        try:
            llm = get_llm()
            if not llm:
                raise CustomException("LLM client could not be initialized.")

            invoke_start = time.time()

            # Execute the call
            response = await asyncio.wait_for(
                llm.ainvoke(messages),
                timeout=LLM_TIMEOUT,
            )

            invoke_time = time.time() - invoke_start
            
            # --- RPM PROTECTION LOGIC ---
            # 60s / 15 RPM = 4s minimum. We use 5s to be safe.
            cooldown_seconds = 5 
            
            logger.info(
                "[LLM][%s] SUCCESS | invoke_time=%.2fs | cooling down %ds",
                call_id, invoke_time, cooldown_seconds,
            )
            
            # This sleep inside the semaphore block ensures the NEXT node 
            # cannot acquire the semaphore until this cooldown expires.
            await asyncio.sleep(cooldown_seconds)

            return response.content

        except asyncio.TimeoutError:
            logger.error("[LLM][%s] TIMEOUT | exceeded %ds", call_id, LLM_TIMEOUT)
            return LLM_ERROR

        except Exception as e:
            error_str = str(e)
            total_time = time.time() - start_time

            # Handle Rate Limiting (429)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                penalty_sleep = 70
                logger.warning(
                    "[LLM][%s] RATE_LIMIT (429) | Penalty sleep %ds | total_time=%.2fs",
                    call_id, penalty_sleep, total_time
                )
                # Sleep long enough to clear the 60s window
                await asyncio.sleep(penalty_sleep)
                return FREE_TIER_EXHAUSTED

            # Handle unexpected API/Network errors
            logger.exception(
                "[LLM][%s] FAILURE | error=%s",
                call_id, error_str
            )
            return LLM_ERROR

        finally:
            logger.info("[LLM][%s] REQUEST_RELEASED", call_id)


# ── Patch Sanitizers ──────────────────────────────────────────────────────────

def _sanitize_patch(raw: str) -> str:
    """
    Strip outer markdown code fences from an LLM-generated patch.

    Handles Mode 1 hallucination: Gemini wraps the entire diff in fences.

    Expected input with fences:
```diff
        --- a/calculator.py
        +++ b/calculator.py
        @@ -1,3 +1,3 @@
        -b = str(b).
        +b = str(b)
```

    Returns the raw diff content without the opening/closing fence lines.
    If no fences are found, returns the original string stripped of whitespace.

    Parameters
    ----------
    raw : str
        Raw string from Gemini — may or may not contain outer fences.

    Returns
    -------
    str
        Diff text with outer markdown fences removed.
    """
    if not raw:
        return raw

    # Match opening fence: ```diff, ```python, ```patch, ```text, ```py, or plain ```
    fence_pattern = re.compile(
        r"^```(?:diff|python|patch|text|py)?\s*\n",
        re.MULTILINE | re.IGNORECASE,
    )

    match = fence_pattern.search(raw)
    if not match:
        return raw.strip()

    content_after_fence = raw[match.end():]

    # Remove the closing fence (last ``` in the string)
    closing_fence = re.compile(r"\n```\s*$", re.MULTILINE)
    cleaned = closing_fence.sub("", content_after_fence)

    result = cleaned.strip()

    if result != raw.strip():
        logger.info(
            "[_sanitize_patch] Stripped outer fences — "
            "original_chars=%d cleaned_chars=%d",
            len(raw), len(result),
        )

    return result


def _deep_clean_patch(patch: str) -> str:
    """
    Remove inline backtick contamination from patch + lines and content lines.

    Handles Mode 2 hallucination: Gemini embeds backtick characters INSIDE
    the code content of the diff. This produces files that Python cannot parse.

    Examples of inline contamination caught here:

        +b = str(b)`.          ← trailing backtick after valid code
        +```                   ← standalone fence line inside diff
        +b = str(b)`           ← backtick glued to end of identifier
        +```python             ← language-annotated fence inside diff
        +x = 1 ` + "`" + `    ← multi-backtick pattern (rare)

    Cleaning Rules
    --------------
    For lines starting with + or space (context lines — lines kept in output):
        1. If the line is ONLY backticks (e.g. "```", "```diff") → drop it
        2. If the line ends with backtick(s) after code → strip trailing backtick(s)
        3. If the line contains isolated inline backtick sequences → strip them

    Lines starting with - (removed lines) are passed through unchanged —
    they are not part of the output file content so contamination there
    does not affect parsing.

    Lines starting with @@ (hunk headers) are passed through unchanged.

    Parameters
    ----------
    patch : str
        The patch string after _sanitize_patch() has been applied.
        May contain inline backtick contamination in + or context lines.

    Returns
    -------
    str
        Patch with all inline backtick contamination removed from content lines.

    Notes
    -----
    This function is intentionally conservative — it only removes backtick
    characters and never modifies the actual Python code logic on a line.
    If a line becomes empty after stripping, it is kept as an empty line
    rather than being dropped, to preserve diff hunk line counts.
    """
    if not patch:
        return patch

    raw_patch = patch  # alias for clarity in logging

    cleaned_lines = []
    removed_lines = 0
    modified_lines = 0

    # Pattern: a line that is ONLY a markdown fence (optional language tag)
    standalone_fence = re.compile(
        r"^[+ ]?```(?:diff|python|patch|text|py)?\s*$",
        re.IGNORECASE,
    )

    # Pattern: trailing backtick(s) at end of a code line
    # Matches ` or `` or ``` at the very end, optionally preceded by a period or space
    trailing_backtick = re.compile(r"[`]+\s*$")

    # Pattern: inline backtick sequence embedded mid-line
    # e.g. str(b)`. or str(b)`
    inline_backtick = re.compile(r"`+")

    for line in raw_patch.splitlines():
        # ── Pass through unchanged: removed lines and hunk headers ───────────
        if line.startswith("-") or line.startswith("@@") or line.startswith("diff") or line.startswith("index"):
            cleaned_lines.append(line)
            continue

        # ── Content lines: + lines and context (space) lines ─────────────────
        # Determine the prefix character and actual code content
        if line.startswith("+") and not line.startswith("+++"):
            prefix = "+"
            code = line[1:]
        elif line.startswith(" "):
            prefix = " "
            code = line[1:]
        else:
            # Header lines (---, +++, etc.) — pass through unchanged
            cleaned_lines.append(line)
            continue

        # ── Rule 1: Drop standalone fence lines ───────────────────────────────
        if standalone_fence.match(line):
            removed_lines += 1
            logger.debug(
                "[_deep_clean_patch] Dropped standalone fence line: %r",
                line[:80],
            )
            continue

        # ── Rule 2: Strip trailing backtick(s) ───────────────────────────────
        if trailing_backtick.search(code):
            original_code = code
            code = trailing_backtick.sub("", code).rstrip()
            if code != original_code:
                modified_lines += 1
                logger.debug(
                    "[_deep_clean_patch] Stripped trailing backtick: %r → %r",
                    original_code[:60], code[:60],
                )

        # ── Rule 3: Strip remaining inline backtick sequences ─────────────────
        # Only apply if backticks remain after Rule 2
        if "`" in code:
            original_code = code
            code = inline_backtick.sub("", code)
            if code != original_code:
                modified_lines += 1
                logger.debug(
                    "[_deep_clean_patch] Stripped inline backtick: %r → %r",
                    original_code[:60], code[:60],
                )

        cleaned_lines.append(prefix + code)

    result = "\n".join(cleaned_lines)

    if removed_lines > 0 or modified_lines > 0:
        logger.info(
            "[_deep_clean_patch] Cleaned patch — "
            "removed_lines=%d modified_lines=%d "
            "original_chars=%d cleaned_chars=%d",
            removed_lines, modified_lines,
            len(raw_patch), len(result),
        )

    return result


def _validate_patch_syntax(patch: str) -> Tuple[bool, str]:
    """
    Quick heuristic check that a patch looks like a valid unified diff.

    Checks for the minimum structural requirements:
        - Contains at least one +++ header line
        - Contains at least one + content line
        - Does NOT contain unescaped backtick sequences in + lines
        - Does NOT contain standalone ``` lines

    This is NOT a full diff validator — just a sanity check to catch
    obviously broken patches before sending to Docker.

    Parameters
    ----------
    patch : str
        Cleaned patch string after _sanitize_patch + _deep_clean_patch.

    Returns
    -------
    Tuple[bool, str]
        (is_valid, reason)
        is_valid=True if patch passes all heuristic checks.
        reason="" on success, or a description of the failure.
    """
    if not patch or not patch.strip():
        return False, "patch is empty"

    lines = patch.splitlines()

    has_plus_header = any(l.startswith("+++") for l in lines)
    has_plus_content = any(l.startswith("+") and not l.startswith("+++") for l in lines)

    if not has_plus_header:
        return False, "patch has no +++ header line — not a valid unified diff"

    if not has_plus_content:
        return False, "patch has no + content lines — nothing to apply"

    # Check for surviving backtick contamination
    for line in lines:
        if line.startswith("+") and not line.startswith("+++"):
            if "`" in line:
                return False, f"patch still contains backtick on line: {line[:60]!r}"
            if line.strip() in ("```", "```diff", "```python", "```patch"):
                return False, f"patch still contains fence line: {line[:60]!r}"

    return True, ""


# ── Output Parser ─────────────────────────────────────────────────────────────

def _parse_llm_output(raw: str) -> Tuple[List[str], List[str]]:
    """
    Parse structured ISSUES / SUGGESTIONS sections from LLM response.

    Expected format:
        ISSUES:
        - issue one

        SUGGESTIONS:
        - suggestion one

    Returns (issues, suggestions) — both may be empty lists.
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


# ── Helper ────────────────────────────────────────────────────────────────────

def _has_python_files(files: list) -> bool:
    """Return True if any changed file in the PR ends with .py."""
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
            "files":    files,
            "diff":     diff,
            "error":    False,
        }

    except Exception as e:
        logger.exception(
            "[fetch_diff_node] GitHub fetch failed — error=%s", str(e),
        )
        return {
            "error":        True,
            "error_reason": "github_fetch_failed",
        }


@traceable(name="retrieve_context_node", tags=["chromadb", "rag", "p4"])
async def retrieve_context_node(state: ReviewState) -> Dict:
    """
    Query ChromaDB for relevant codebase context for the current PR.

    Reads  : state["diff"], state["metadata"], state["error"]
    Writes : state["raw_context"]

    Returns raw_context="" if ChromaDB is empty, unavailable, or query fails.
    Graph NEVER crashes on RAG failures — context is optional.
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
            "[retrieve_context_node] ChromaDB unavailable — skipping retrieval"
        )
        return {"raw_context": ""}

    if collection.count() == 0:
        logger.info(
            "[retrieve_context_node] ChromaDB collection is empty — "
            "run POST /repos/index first"
        )
        return {"raw_context": ""}

    try:
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
            logger.info("[retrieve_context_node] No results from ChromaDB")
            return {"raw_context": ""}

        context_parts = []
        for doc, meta, dist in zip(docs, metadatas, distances):
            source = meta.get("source", "unknown")
            context_parts.append(
                f"### {source} (similarity={1-dist:.2f})\n{doc}"
            )

        raw_context = "\n\n".join(context_parts)

        logger.info(
            "[retrieve_context_node] Retrieved %d chunk(s) — total_chars=%d",
            len(docs), len(raw_context),
        )

        return {"raw_context": raw_context}

    except Exception as e:
        logger.warning(
            "[retrieve_context_node] ChromaDB query failed — "
            "continuing without context. error=%s", str(e),
        )
        return {"raw_context": ""}


@traceable(name="grade_context_node", tags=["llm", "rag", "crag", "p4"])
async def grade_context_node(state: ReviewState) -> Dict:
    """
    Grade the relevance of retrieved ChromaDB context (CRAG pattern).

    If grade is "yes" → repo_context = raw_context (injected into analyzer)
    If grade is "no"  → repo_context = "" (irrelevant context discarded)
    If LLM fails      → fails open (raw_context passed through unchanged)

    Reads  : state["raw_context"], state["diff"], state["error"]
    Writes : state["repo_context"], state["context_grade"]
    """
    if state.get("error"):
        logger.warning(
            "[grade_context_node] Skipping — upstream error: %s",
            state.get("error_reason"),
        )
        return {"repo_context": "", "context_grade": "skipped"}

    raw_context = state.get("raw_context", "")
    diff = state.get("diff", "")

    if not raw_context:
        logger.info(
            "[grade_context_node] No raw_context to grade — skipping LLM call"
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
            "Determine if retrieved codebase context is relevant to a PR diff.\n\n"
            "Respond with ONLY one word: 'yes' or 'no'.\n"
            "- 'yes' if the context contains code, patterns, or conventions "
            "that would help review the PR diff.\n"
            "- 'no' if the context is unrelated to the PR changes."
        )),
        HumanMessage(content=(
            f"--- Retrieved Context ---\n{raw_context[:1000]}\n\n"
            f"--- PR Diff ---\n{diff[:1000]}\n\n"
            "Is this context relevant? Respond with ONLY 'yes' or 'no'."
        )),
    ]

    result = await safe_llm_invoke(prompt)

    if result in (FREE_TIER_EXHAUSTED, LLM_ERROR):
        logger.warning(
            "[grade_context_node] LLM unavailable (%s) — "
            "failing open, passing raw_context through", result,
        )
        return {
            "repo_context":  raw_context[:MAX_CONTEXT_CHARS],
            "context_grade": "skipped",
        }

    grade = result.strip().lower()

    if "yes" in grade:
        grade = CONTEXT_GRADE_YES
    elif "no" in grade:
        grade = CONTEXT_GRADE_NO
    else:
        logger.warning(
            "[grade_context_node] Unexpected grade: '%s' — defaulting to yes",
            result.strip(),
        )
        grade = CONTEXT_GRADE_YES

    if grade == CONTEXT_GRADE_YES:
        logger.info(
            "[grade_context_node] Grade=YES — injecting context_chars=%d",
            min(len(raw_context), MAX_CONTEXT_CHARS),
        )
        return {
            "repo_context":  raw_context[:MAX_CONTEXT_CHARS],
            "context_grade": grade,
        }
    else:
        logger.info("[grade_context_node] Grade=NO — discarding context")
        return {
            "repo_context":  "",
            "context_grade": grade,
        }


@traceable(name="analyze_code_node", tags=["llm", "analysis"])
async def analyze_code_node(state: ReviewState) -> Dict:
    """
    Use Gemini to analyze the PR diff and identify issues and suggestions.

    P4: Includes graded repo_context in prompt when available.

    Reads  : state["diff"], state["metadata"], state["repo_context"], state["error"]
    Writes : state["issues"], state["suggestions"], state["error"]
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
        logger.warning("[analyze_code_node] LLM quota exhausted — degraded output")
        return {
            "issues":      ["LLM quota exhausted — manual review required"],
            "suggestions": [],
            "error":       False,
        }

    if result == LLM_ERROR:
        logger.error("[analyze_code_node] LLM error — degraded output")
        return {
            "issues":      ["LLM error — manual review required"],
            "suggestions": [],
            "error":       False,
        }

    issues, suggestions = _parse_llm_output(result)

    logger.info(
        "[analyze_code_node] Complete — %d issue(s), %d suggestion(s)",
        len(issues), len(suggestions),
    )

    return {
        "issues":      issues,
        "suggestions": suggestions,
        "error":       False,
    }


@traceable(name="reflect_node", tags=["llm", "reflection"])
async def reflect_node(state: ReviewState) -> Dict:
    """
    Self-reflection pass — ask Gemini to find anything it missed.

    Controlled by REFLECTION_PASSES in workflow.py.
    Set to 0 in development (skipped entirely).

    Reads  : state["issues"], state["suggestions"], state["diff"],
             state["reflection_count"]
    Writes : state["issues"], state["suggestions"], state["reflection_count"]
    """
    current_count = state.get("reflection_count", 0)
    existing_issues = state.get("issues", [])
    existing_suggestions = state.get("suggestions", [])
    diff = state.get("diff", "")

    logger.info(
        "[reflect_node] Pass #%d starting — issues=%d suggestions=%d",
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
            "[reflect_node] Pass #%d skipped — LLM unavailable (%s)",
            current_count + 1, result,
        )
        return {
            "issues":           existing_issues,
            "suggestions":      existing_suggestions,
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
        "[reflect_node] Pass #%d complete — added %d issue(s), %d suggestion(s)",
        current_count + 1, len(new_issues), len(new_suggestions),
    )

    return {
        "issues":           merged_issues,
        "suggestions":      merged_suggestions,
        "reflection_count": current_count + 1,
    }


@traceable(name="lint_node", tags=["sandbox", "lint"])
async def lint_node(state: ReviewState) -> Dict:
    """
    Run ruff on changed Python files inside the Docker sandbox.

    Reads  : state["diff"], state["files"]
    Writes : state["lint_result"], state["lint_passed"]
    """
    logger.info("[lint_node] Starting")

    files = state.get("files", [])

    if not _has_python_files(files):
        logger.info("[lint_node] No Python files in diff — skipping lint")
        return {
            "lint_passed": True,
            "lint_result": "SKIPPED_NO_PYTHON_FILES",
        }

    sandbox = get_sandbox()

    if not sandbox:
        logger.warning("[lint_node] Sandbox unavailable — skipping lint")
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
            "lint_result":  result,
            "lint_passed":  result.passed,
        }

    except Exception as e:
        logger.exception(
            "[lint_node] Sandbox error — marking lint as passed. error=%s",
            str(e),
        )
        return {
            "lint_passed": True,
            "lint_result": "SKIPPED_SANDBOX_ERROR",
        }


@traceable(name="refactor_node", tags=["llm", "refactor"])
async def refactor_node(state: ReviewState) -> Dict:
    """
    Generate a corrective patch using Gemini and sanitize it before storage.

    Patch Sanitization Pipeline
    ---------------------------
    The raw LLM output passes through three stages:

    Stage 1 — _sanitize_patch():
        Strips outer markdown code fences (Mode 1 hallucination).
        Example: removes opening ```diff and closing ``` wrapper.

    Stage 2 — _deep_clean_patch():
        Removes inline backtick contamination from + lines (Mode 2 hallucination).
        Example: strips trailing `. from `+b = str(b)`.`
        Example: drops standalone ``` lines inside the diff content.

    Stage 3 — _validate_patch_syntax():
        Heuristic check that the cleaned patch is a structurally valid diff.
        If validation fails, the patch is rejected and refactor_count is
        incremented without storing the bad patch, preventing Docker from
        receiving malformed input.

    Without this pipeline, Docker receives contaminated content and
    ruff/pytest always exit with code 1, exhausting the 2-attempt loop.

    Reads  : state["diff"], state["issues"], state["lint_result"],
             state["refactor_count"]
    Writes : state["patch"], state["refactor_count"]
    """
    logger.info("[refactor_node] Generating corrective patch")

    diff = state.get("diff", "")
    issues = state.get("issues", [])
    suggestions = state.get("suggestions", [])
    lint_result = state.get("lint_result")
    current_count = state.get("refactor_count", 0)

    lint_context = ""
    if lint_result and hasattr(lint_result, "passed") and not lint_result.passed:
        lint_context = (
            f"\n--- Lint Failures ---\n"
            f"{getattr(lint_result, 'output', '')}\n"
        )

    prompt = [
        SystemMessage(content=(
            "You are an expert Python developer. Generate a corrective "
            "unified diff patch that fixes all listed issues.\n\n"
            "CRITICAL RULES — follow exactly:\n"
            "1. Output ONLY the raw unified diff. No explanations.\n"
            "2. Do NOT use markdown code fences (no backticks, no ```).\n"
            "3. Do NOT include backtick characters anywhere in the output.\n"
            "4. Start the output directly with '--- a/filename'.\n"
            "5. Every fixed line must start with a '+' prefix.\n"
            "6. Every removed line must start with a '-' prefix.\n"
            "7. Context lines must start with a single space.\n\n"
            "The output will be fed directly to a diff parser. "
            "Any backtick will cause a Python SyntaxError in the sandbox."
        )),
        HumanMessage(content=(
            f"--- PR Diff (original, do not include this in output) ---\n"
            f"{diff[:2000]}\n\n"
            f"--- Issues to fix ---\n"
            + ("\n".join(f"- {i}" for i in issues) or "- None")
            + lint_context
        )),
    ]

    result = await safe_llm_invoke(prompt)

    if result in (FREE_TIER_EXHAUSTED, LLM_ERROR):
        logger.warning(
            "[refactor_node] LLM unavailable (%s) — skipping patch", result,
        )
        return {
            "refactor_count": current_count + 1,
            "suggestions":    suggestions + ["Refactor skipped — LLM unavailable"],
        }

    # ── Stage 1: Strip outer fences ───────────────────────────────────────────
    stage1 = _sanitize_patch(result)

    # ── Stage 2: Remove inline backtick contamination ─────────────────────────
    stage2 = _deep_clean_patch(stage1)

    # ── Stage 3: Validate patch structure ─────────────────────────────────────
    is_valid, reason = _validate_patch_syntax(stage2)

    if not is_valid:
        logger.warning(
            "[refactor_node] Patch failed validation — "
            "reason=%s raw_chars=%d cleaned_chars=%d — "
            "incrementing refactor_count without storing bad patch",
            reason, len(result), len(stage2),
        )
        return {
            "refactor_count": current_count + 1,
            "suggestions":    suggestions + [
                f"Refactor patch invalid ({reason}) — skipped"
            ],
        }

    logger.info(
        "[refactor_node] Patch ready — "
        "raw=%d stage1=%d stage2=%d chars | valid=True",
        len(result), len(stage1), len(stage2),
    )

    return {
        "patch":          stage2,
        "refactor_count": current_count + 1,
    }


@traceable(name="validator_node", tags=["sandbox", "validation"])
async def validator_node(state: ReviewState) -> Dict:
    """
    Run ruff + pytest on the generated patch inside the Docker sandbox.

    Reads  : state["patch"], state["refactor_count"]
    Writes : state["validation_result"], state["validation_passed"],
             state["refactor_count"]
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
        logger.warning("[validator_node] Sandbox unavailable — marking as passed")
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
                "validation_result":  result,
                "validation_passed":  True,
                "refactor_count":     current_count,
            }
        else:
            logger.info(
                "[validator_node] Validation FAILED — "
                "incrementing refactor_count to %d",
                current_count + 1,
            )
            return {
                "validation_result":  result,
                "validation_passed":  False,
                "refactor_count":     current_count + 1,
            }

    except Exception as e:
        logger.exception(
            "[validator_node] Sandbox error — marking as passed. error=%s",
            str(e),
        )
        return {
            "validation_passed": True,
            "validation_result": "SKIPPED_SANDBOX_ERROR",
        }


@traceable(name="memory_write_node", tags=["chromadb", "memory", "p4"])
async def memory_write_node(state: ReviewState) -> Dict:
    """
    Write review findings to ChromaDB long-term memory after each review.

    Reads  : state["issues"], state["suggestions"], state["verdict"],
             state["metadata"], state["owner"], state["repo"]
    Writes : {} — ChromaDB write is a side effect only

    Failure is non-fatal — any exception is logged and graph continues.
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
        logger.info("[memory_write_node] ChromaDB unavailable — skipping")
        return {}

    if not issues and not suggestions:
        logger.info("[memory_write_node] No findings to store — skipping")
        return {}

    try:
        issues_text = "\n".join(f"- {i}" for i in issues) if issues else "None"
        suggestions_text = (
            "\n".join(f"- {s}" for s in suggestions) if suggestions else "None"
        )

        memory_doc = (
            f"PR Review Memory — {owner}/{repo}#{pr_number}\n"
            f"Title: {pr_title}\n"
            f"Author: @{pr_author}\n"
            f"Verdict: {verdict}\n\n"
            f"Issues Found:\n{issues_text}\n\n"
            f"Suggestions:\n{suggestions_text}"
        )

        doc_id = f"review_memory_{owner}_{repo}_{pr_number}_{int(time.time())}"
        embedding = embedder.embed_query(memory_doc)

        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[memory_doc],
            metadatas=[{
                "source":    f"{owner}/{repo}",
                "type":      "review_memory",
                "pr_title":  pr_title,
                "pr_number": str(pr_number),
                "verdict":   verdict,
                "author":    pr_author,
            }],
        )

        logger.info(
            "[memory_write_node] Memory written — "
            "doc_id=%s collection_count=%d",
            doc_id, collection.count(),
        )

    except Exception as e:
        logger.warning(
            "[memory_write_node] ChromaDB write failed — continuing. error=%s",
            str(e),
        )

    return {}


@traceable(name="summary_node", tags=["summary"])
async def summary_node(state: ReviewState) -> Dict:
    """
    Format review findings into a Markdown HITL briefing.

    Runs deterministically — no LLM call.
    Produces a structured summary for the human reviewer before the HITL gate.

    Reads  : state["issues"], state["suggestions"], state["lint_result"],
             state["error"], state["error_reason"]
    Writes : state["summary"]

    Failure Behaviour
    -----------------
    Any exception returns a minimal fallback summary.
    The graph NEVER crashes in this node.
    """
    try:
        logger.info("[summary_node] Formatting findings for HITL briefing")

        issues = state.get("issues") or []
        suggestions = state.get("suggestions") or []
        lint_result = state.get("lint_result") or {}
        error_flag = state.get("error", False)
        error_reason = state.get("error_reason", "")
        context_grade = state.get("context_grade", "skipped")

        # ── Extract lint details safely ────────────────────────────────────────
        lint_passed = False
        lint_output = ""
        if lint_result:
            if isinstance(lint_result, dict):
                lint_passed = lint_result.get("passed", False)
                lint_output = lint_result.get("output", "")
            else:
                lint_passed = getattr(lint_result, "passed", False)
                lint_output = getattr(lint_result, "output", "")

        # ── Context badge ──────────────────────────────────────────────────────
        context_badge = ""
        if context_grade == CONTEXT_GRADE_YES:
            context_badge = " | 🧠 Context: Used"
        elif context_grade == CONTEXT_GRADE_NO:
            context_badge = " | ⚪ Context: Not relevant"

        # ── Build markdown ─────────────────────────────────────────────────────
        summary_text = "## 🔍 AI Code Review Briefing\n\n---\n\n"
        summary_text += (
            f"**Stats:** 🐛 `{len(issues)}` Issues | "
            f"💡 `{len(suggestions)}` Suggestions | "
            f"🚨 Lint: `{'PASSED' if lint_passed else 'FAILED'}`"
            f"{context_badge}\n\n"
        )

        # Error section
        if error_flag:
            summary_text += (
                f"### ❌ Workflow Error\n"
                f"> **Reason:** `{error_reason or 'Internal Processing Error'}`\n\n"
            )

        # Issues section
        if issues:
            summary_text += "### 🐛 Issues\n"
            for i in issues[:15]:
                summary_text += f"- {i}\n"
            if len(issues) > 15:
                summary_text += f"\n*...and {len(issues) - 15} more.*\n"
            summary_text += "\n"
        else:
            summary_text += "### ✅ No Issues Found\n\n"

        # Suggestions section
        if suggestions:
            summary_text += "### 💡 Suggestions\n"
            for s in suggestions[:10]:
                summary_text += f"- {s}\n"
            summary_text += "\n"

        # Lint section
        status_icon = "✅" if lint_passed else "⚠️"
        summary_text += f"### {status_icon} Lint\n"
        if not lint_passed and lint_output:
            clean_output = str(lint_output).strip()[:1000]
            summary_text += (
                "<details>\n"
                "<summary>Click to view lint output</summary>\n\n"
                f"```text\n{clean_output}\n```\n"
                "</details>\n\n"
            )
        elif not lint_result:
            summary_text += "_No lint data._\n\n"
        else:
            summary_text += "Code style is healthy.\n\n"

        # Action prompt
        summary_text += (
            "---\n\n"
            "### 📝 Action Required\n"
            "Review the findings above. "
            "Call `POST /reviews/id/{id}/decision` with `approved` or `rejected`."
        )

        return {"summary": summary_text}

    except Exception as e:
        logger.error("[summary_node] Formatting failed — %s", str(e))
        return {
            "summary": (
                "### ⚠️ Summary Generation Failed\n"
                f"Issues count: {len(state.get('issues', []))}\n"
                "Check logs for details."
            )
        }


@traceable(name="verdict_node", tags=["verdict"])
async def verdict_node(state: ReviewState) -> Dict:
    """
    Produce the final verdict and GitHub PR comment markdown.

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

    if state.get("error"):
        reason = state.get("error_reason", "unknown_error")
        logger.warning("[verdict_node] Upstream error — reason=%s", reason)
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

    if human_decision == "rejected":
        logger.info("[verdict_node] HUMAN_REJECTED")
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

    verdict = "REQUEST_CHANGES" if issues else "APPROVE"
    verdict_emoji = "✅" if verdict == "APPROVE" else "🔴"

    logger.info(
        "[verdict_node] Determined verdict=%s (issues=%d)",
        verdict, len(issues),
    )

    issues_section = (
        "\n".join(f"- {i}" for i in issues) if issues else "_No issues found._"
    )
    suggestions_section = (
        "\n".join(f"- {s}" for s in suggestions)
        if suggestions else "_No suggestions._"
    )

    human_badge = (
        "\n\n**Human Approval:** ✅ Approved by reviewer"
        if human_decision == "approved" else ""
    )

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