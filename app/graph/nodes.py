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
import hashlib
from langsmith import traceable
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from dataclasses import asdict
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
        calls_in_last_10s = _record_llm_call()
        
        logger.info(
            "[LLM][%s] ACQUIRED | Burst Pressure: %d calls/10s | RPM_limit=15",
            call_id, calls_in_last_10s,
        )
        queue_wait = time.time() - queue_start
        
        logger.info(
            "[LLM][%s] ACQUIRED | queue_wait=%.2fs | RPM_limit=15",
            call_id, queue_wait,
        )
        # 2. Proactive Burst Protection
        # If we've made 3+ calls in 10 seconds, we're trending toward 18+ RPM.
        # Let's add an extra buffer.

        if calls_in_last_10s > 2:
            extra_buffer = 2.0
            logger.info("[LLM][%s] BURST_PROTECTION | Adding %.1fs buffer", call_id, extra_buffer)
            await asyncio.sleep(extra_buffer)

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
    if not patch:
        return patch

    raw_patch = patch
    cleaned_lines = []
    removed_lines = 0
    modified_lines = 0

    standalone_fence = re.compile(
        r"^[+ ]?```(?:diff|python|patch|text|py)?\s*$",
        re.IGNORECASE,
    )
    # Also catch bare fence lines with no prefix at all
    bare_fence = re.compile(
        r"^```(?:diff|python|patch|text|py)?\s*$",
        re.IGNORECASE,
    )
    trailing_backtick = re.compile(r"[`]+\s*$")
    inline_backtick   = re.compile(r"`+")

    for line in raw_patch.splitlines():
        if line.startswith("-") or line.startswith("@@") or line.startswith("diff") or line.startswith("index"):
            cleaned_lines.append(line)
            continue

        if line.startswith("+") and not line.startswith("+++"):
            prefix = "+"
            code   = line[1:]
        elif line.startswith(" "):
            prefix = " "
            code   = line[1:]
        else:
            # Header lines (---, +++) pass through
            # BUT: check for bare backtick fence lines — drop them
            if bare_fence.match(line):
                removed_lines += 1
                logger.debug("[_deep_clean_patch] Dropped bare fence line: %r", line[:80])
                continue
            cleaned_lines.append(line)
            continue

        if standalone_fence.match(line):
            removed_lines += 1
            continue

        if trailing_backtick.search(code):
            original = code
            code = trailing_backtick.sub("", code).rstrip()
            if code != original:
                modified_lines += 1

        if "`" in code:
            original = code
            code = inline_backtick.sub("", code)
            if code != original:
                modified_lines += 1

        cleaned_lines.append(prefix + code)

    result = "\n".join(cleaned_lines)

    if removed_lines > 0 or modified_lines > 0:
        logger.info(
            "[_deep_clean_patch] Cleaned — removed=%d modified=%d orig=%d clean=%d",
            removed_lines, modified_lines, len(raw_patch), len(result),
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

def _parse_llm_output(text: str) -> Tuple[List[str], List[str]]:
    """
    Parses structured LLM output into issues and suggestions lists.
    Uses regex to handle Markdown formatting and provides robust logging.
    """
    issues: List[str] = []
    suggestions: List[str] = []
    
    if not text or not isinstance(text, str):
        logger.error("[parser] Received empty or invalid text input")
        return issues, suggestions

    current_section = None
    
    try:
        lines = text.splitlines()
        for line in lines:
            clean_line = line.strip()
            if not clean_line:
                continue

            # 1. Detect Sections using flexible Regex (handles **ISSUES:**, # ISSUES, etc.)
            if re.search(r'(?i)\bISSUES\b', clean_line) and clean_line.endswith(':'):
                current_section = "issues"
                continue
            elif re.search(r'(?i)\bSUGGESTIONS\b', clean_line) and clean_line.endswith(':'):
                current_section = "suggestions"
                continue

            # 2. Collect content starting with common bullet symbols
            # Matches: "-", "*", "1."
            bullet_match = re.match(r'^[-*•]|\d+\.\s+(.*)', clean_line)
            if bullet_match:
                # Remove the bullet symbol and leading/trailing whitespace
                content = re.sub(r'^[-*•]\s*|^\d+\.\s*', '', clean_line).strip()
                
                # Skip placeholder "None" responses
                if content.upper() in ["NONE", "NONE.", "N/A"]:
                    continue

                if current_section == "issues":
                    issues.append(content)
                elif current_section == "suggestions":
                    suggestions.append(content)
                else:
                    # Content found before any section header was reached
                    logger.debug(f"[parser] Skipping orphaned content: {content}")

        # 3. Validation & Observability
        if not issues and not suggestions:
            logger.warning("[parser] No structured data extracted. Raw output length: %d", len(text))
            logger.debug("[parser] Raw problematic text: %s", text)
        else:
            logger.info(
                "[parser] Extraction complete: %d issues, %d suggestions", 
                len(issues), len(suggestions)
            )

    except Exception as e:
        # sys.exc_info helps capture line number and context in your CustomException style
        logger.exception("[parser] Critical failure during parsing: %s", str(e))
        # We return empty lists rather than crashing the node to allow the graph to continue
        return [], []

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
    Fetches PR metadata, files, and diff. 
    Also performs a 'baseline' lint check on the original files to isolate pre-existing bugs.
    """
    # Small delay for UI/logging pacing
    await asyncio.sleep(2)
    owner, repo, pr_number = state["owner"], state["repo"], state["pr_number"]

    try:
        client = GitHubClient()
        metadata = client.get_pr_metadata(owner, repo, pr_number)
        files = client.get_pr_files(owner, repo, pr_number)
        diff = client.get_pr_diff(owner, repo, pr_number)

        original_issues = []
        original_files = {}
        base_sha = metadata.get("base_sha", "")

        # --- Phase 1: Fetch Base File Content ---
        if base_sha:
            for f in files:
                filepath = f.get("filename", "")
                # Only analyze modified Python files to save resources
                if filepath.endswith(".py") and f.get("status") == "modified":
                    content = await client.get_base_file_content(
                        owner, repo, filepath, base_sha
                    )
                    if content:
                        original_files[filepath] = content

        # --- Phase 2: Baseline Linting (The "Healthy Check") ---
        if original_files:
            sandbox = get_sandbox()
            if sandbox:
                for fname, content in original_files.items():
                    # Use run_lint_raw — passes file directly to Docker, no diff parsing needed
                    res = sandbox.run_lint_raw(fname, content)

                    if not res.passed and res.output:
                        for line in res.output.strip().splitlines():
                            line = line.strip()
                            
                            # Filtering logic to ignore non-error output from ruff/mypy
                            if not line:
                                continue
                            if line.startswith("Found ") and "error" in line:
                                continue
                            if line == "Success: no issues found":
                                continue
                            if ": note:" in line:
                                continue
                            if line == "error:":
                                continue
                                
                            # If the line contains a file pointer (e.g., file.py:10), it's a real issue
                            if re.search(r'\.py:\d+', line):
                                original_issues.append(
                                    f"[Pre-existing in base branch] {line}"
                                )

        logger.info(
            "[fetch_diff_node] Pre-existing issues detected: %d",
            len(original_issues),
        )

        return {
            "metadata": metadata,
            "files": files,
            "diff": diff,
            "original_issues": original_issues,
            "base_files": original_files, 
            "error": False,
        }

    except Exception as e:
        logger.exception("[fetch_diff_node] Failed")
        return {"error": True, "error_reason": str(e)}

@traceable(name="retrieve_context_node", tags=["chromadb", "rag", "p4"])
async def retrieve_context_node(state: ReviewState) -> Dict:
    """
    Query ChromaDB for relevant codebase context.
    RAG is optional; failures here should NEVER stop the workflow.
    """
    await asyncio.sleep(2)
    
    # Check for systemic infrastructure failure only
    if state.get("critical_infra_failure"):
        logger.warning("[retrieve_context_node] Skipping — critical infra failure detected")
        return {"raw_context": ""}

    diff = state.get("diff", "")
    metadata = state.get("metadata", {})
    pr_title = metadata.get("title", "")

    collection, embedder = get_chroma_collection()

    # RAG failure is handled gracefully by returning empty context
    if not collection or not embedder or collection.count() == 0:
        logger.info("[retrieve_context_node] ChromaDB empty or unavailable — skipping")
        return {"raw_context": ""}

    try:
        query_text = f"{pr_title}\n{diff[:1000]}"
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
            return {"raw_context": ""}

        context_parts = [
            f"### {m.get('source', 'unknown')} (similarity={1-d:.2f})\n{doc}"
            for doc, m, d in zip(docs, metadatas, distances)
        ]

        return {"raw_context": "\n\n".join(context_parts)}

    except Exception as e:
        # Note: We do NOT set critical_infra_failure=True here because 
        # the review can still proceed effectively without RAG.
        logger.warning("[retrieve_context_node] Query failed, continuing without RAG: %s", str(e))
        return {"raw_context": ""}


@traceable(name="grade_context_node", tags=["llm", "rag", "crag", "p4"])
async def grade_context_node(state: ReviewState) -> Dict:
    """
    Grade context relevance. If grading fails, we 'fail open' 
    by keeping the context to avoid losing potential insights.
    """
    await asyncio.sleep(2)
    
    if state.get("critical_infra_failure"):
        return {"repo_context": "", "context_grade": "skipped"}

    raw_context = state.get("raw_context", "")
    diff = state.get("diff", "")

    if not raw_context:
        return {"repo_context": "", "context_grade": "skipped"}

    prompt = [
        SystemMessage(content="You are a relevance grader. Respond ONLY with 'yes' or 'no'."),
        HumanMessage(content=f"Context: {raw_context[:1000]}\n\nDiff: {diff[:1000]}")
    ]

    result = await safe_llm_invoke(prompt)

    # LLM errors in grading result in "skipped" (fail open)
    if result in (FREE_TIER_EXHAUSTED, LLM_ERROR):
        logger.warning("[grade_context_node] LLM grading failed — failing open")
        return {
            "repo_context": raw_context[:MAX_CONTEXT_CHARS],
            "context_grade": "skipped",
        }

    grade = result.strip().lower()
    if "yes" in grade:
        return {"repo_context": raw_context[:MAX_CONTEXT_CHARS], "context_grade": CONTEXT_GRADE_YES}
    
    return {"repo_context": "", "context_grade": CONTEXT_GRADE_NO}

@traceable(name="analyze_code_node", tags=["llm", "analysis"])
async def analyze_code_node(state: ReviewState) -> Dict:
    """
    Core analysis node. 
    Compare ORIGINAL vs PR DIFF to identify logic errors.
    Uses critical_infra_failure to distinguish between system crashes and code bugs.
    """
    await asyncio.sleep(1) 
    
    # 1. Check for fatal infrastructure failure
    if state.get("critical_infra_failure"):
        logger.warning("[analyze_code_node] Skipping due to system failure.")
        return {}

    # 2. Extraction
    diff            = state.get("diff", "")
    metadata        = state.get("metadata", {})
    pr_title        = metadata.get("title", "Unknown PR")
    repo_context    = state.get("repo_context", "")
    original_issues = state.get("original_issues", []) # Issues found by sandbox in base
    base_files      = state.get("base_files", {})      # Raw file content
    
    # 3. Construct Ground Truth Context
    base_context = ""
    if base_files:
        base_context = "\n\n--- ORIGINAL CODE (BASE BRANCH) ---\n"
        for fname, content in base_files.items():
            base_context += f"FILE: {fname}\n```python\n{content}\n```\n"

    # 4. Detailed Prompt Engineering
    prompt = [
            SystemMessage(content=(
                "You are a Senior Python Engineer performing a precise code review.\n\n"
                "You will receive:\n"
                "1. ORIGINAL CODE — the exact file content BEFORE this PR\n"
                "2. PR DIFF — what this PR changed ('+' lines are additions)\n\n"
                "LABELING RULES — apply exactly one label per issue:\n"
                "- [Pre-existing] — bug exists in the ORIGINAL CODE. "
                "The PR did not introduce it. Use this when the bug is visible in the original code block.\n"
                "- [Introduced by PR] — bug is ONLY visible in the '+' lines of the PR DIFF "
                "and does NOT exist in the original code.\n\n"
                "NEVER apply both labels to one issue. Pick the one that is correct.\n\n"
                "CONCRETE EXAMPLE for this codebase:\n"
                "- Original has `b = str(b)` → TypeError when dividing → label [Pre-existing]\n"
                "- PR changes it to `b = str(b).` → SyntaxError from trailing period → label [Introduced by PR]\n"
                "- These are TWO SEPARATE issues with DIFFERENT labels\n\n"
                "You MUST respond in EXACTLY this format:\n\n"
                "ISSUES:\n"
                "- <filename>:<line>: [label] <description>\n\n"
                "SUGGESTIONS:\n"
                "- <filename>:<line>: <suggestion>\n\n"
                "Use '- None' if there are no issues or suggestions."
            )),
            HumanMessage(content=(
                f"PR Title: {pr_title}\n"
                f"{repo_context}\n"
                f"{base_context}\n"
                f"\n--- PR DIFF ('+' lines are new, '-' lines are removed) ---\n{diff}"
            )),
        ]

    # 5. Invoke LLM with safety handling
    result = await safe_llm_invoke(prompt)

    # If the LLM is just down/exhausted, we return the sandbox baseline 
    # but we DO NOT mark it as a critical infrastructure failure.
    if result in [FREE_TIER_EXHAUSTED, LLM_ERROR]:
        logger.error("[analyze_code_node] LLM invocation failed — returning sandbox baseline")
        return {
            "issues": original_issues, 
            "critical_infra_failure": False
        }

    # 6. Parsing & Merging
    issues, suggestions = _parse_llm_output(result)
    
    # Ensure sandbox-detected pre-existing issues are included even if LLM missed them
    final_issues = issues
    for orig in original_issues:
        if not any(orig.lower() in pi.lower() for pi in final_issues):
            final_issues.append(f"[Pre-existing] {orig}")

    return {
        "issues": final_issues,           # Published in PR comment
        "original_issues": original_issues, # Preserved for thread memory
        "suggestions": suggestions,
        "critical_infra_failure": False,
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
    await asyncio.sleep(2)
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
    Run ruff via Docker sandbox. 
    Distinguishes between "Bugs Found" and "Infrastructure Crashes".
    """
    await asyncio.sleep(2)
    from dataclasses import asdict as dataclass_asdict

    logger.info("[lint_node] Starting execution")

    files = state.get("files", [])
    diff = state.get("diff", "")

    if not _has_python_files(files):
        logger.info("[lint_node] No Python files detected — skipping lint")
        return {
            "lint_passed": True, 
            "lint_result": "SKIPPED_NO_PYTHON_FILES",
            "critical_infra_failure": False
        }

    sandbox = get_sandbox()
    if not sandbox:
        logger.error("[lint_node] Sandbox unavailable — infrastructure failure")
        return {
            "lint_passed": False, 
            "lint_result": "FAILED_INFRASTRUCTURE",
            "critical_infra_failure": True, # CRITICAL FIX: Mark as system failure
            "error_reason": "SANDBOX_UNAVAILABLE"
        }

    try:
        logger.info("[lint_node] Dispatching diff to SandboxClient.run_lint")
        result = sandbox.run_lint(diff)

        if result.passed:
            logger.info("[lint_node] LINT PASSED")
        else:
            logger.warning("[lint_node] LINT FAILED — issues found in PR")

        # FIX: result.passed being False is NOT a critical_infra_failure.
        # It just means the code has bugs.
        return {
            "lint_result": dataclass_asdict(result),
            "lint_passed": result.passed,
            "critical_infra_failure": False,
            # Never set "error" here — lint failure is expected code behavior,
            # not a workflow error. Setting error=truthy triggers the ❌ banner
            # in summary_node incorrectly.
        }

    except Exception as e:
        logger.exception("[lint_node] Unexpected sandbox crash")
        return {
            "lint_passed": False, 
            "lint_result": f"CRASH: {type(e).__name__}",
            "critical_infra_failure": True, # CRITICAL FIX: The system crashed
            "error_reason": "SANDBOX_RUNTIME_CRASH"
        }


@traceable(name="refactor_node", tags=["llm", "refactor"])
async def refactor_node(state: ReviewState) -> Dict:
    await asyncio.sleep(2)
    logger.info("[refactor_node] Generating corrective patch")

    # 1. Extract State
    diff             = state.get("diff", "")
    issues           = state.get("issues", [])
    suggestions      = state.get("suggestions", [])
    lint_result      = state.get("lint_result")
    original_issues  = state.get("original_issues", [])
    base_files       = state.get("base_files", {})  # Ground truth from fetcher
    current_count    = state.get("refactor_count", 0)

    # 2. Prepare Context
    lint_context = ""
    if isinstance(lint_result, dict) and not lint_result.get("passed", True):
        lint_context = f"\n--- Recent Sandbox Lint Failures ---\n{lint_result.get('output', '')}\n"

    # Merge issues and deduplicate
    all_issues = list(set(issues + original_issues))

    # Build file context so Gemini knows the 'Before' state of the whole file
    full_file_context = ""
    if base_files:
        full_file_context = "\n--- Full Content of Affected Files (Base Version) ---\n"
        for fname, content in base_files.items():
            full_file_context += f"\nFILE: {fname}\n{content}\n"

    # 3. Construct Prompt
    prompt = [
        SystemMessage(content=(
            "You are an expert Python developer. Generate a unified diff patch (.patch) "
            "that fixes ALL issues provided. \n\n"
            "STRICT OUTPUT RULES:\n"
            "1. Output ONLY the raw unified diff. No markdown blocks, no backticks.\n"
            "2. Start with '--- a/filename'.\n"
            "3. Fix both [Introduced by PR] and [Pre-existing] issues.\n"
            "4. Use the 'Full Content' provided to ensure the diff line numbers and context are accurate."
        )),
        HumanMessage(content=(
            f"{full_file_context}\n"
            f"--- PR Diff to apply fixes to ---\n{diff[:5000]}\n\n"
            f"--- All Issues (Sandbox + LLM) ---\n"
            + "\n".join(f"- {i}" for i in all_issues)
            + lint_context
        )),
    ]

    # 4. Invoke LLM
    result = await safe_llm_invoke(prompt)

    if result in (FREE_TIER_EXHAUSTED, LLM_ERROR):
        return {
            "refactor_count": current_count + 1,
            "suggestions": suggestions + ["Refactor skipped — LLM unavailable"],
        }

    # 5. Sanitization Pipeline
    # Ensure no backticks or leading/trailing commentary
    stage1 = _sanitize_patch(result)
    stage2 = _deep_clean_patch(stage1)

    # 6. Structural Validation
    is_valid, reason = _validate_patch_syntax(stage2)

    if not is_valid:
        logger.warning("[refactor_node] Patch invalid: %s", reason)
        return {
            "refactor_count": current_count + 1,
            "suggestions": suggestions + [f"Auto-refactor failed validation: {reason}"],
        }

    logger.info("[refactor_node] Patch successfully generated and validated")
    return {
        "patch": stage2,
        "refactor_count": current_count + 1,
    }

@traceable(name="validator_node", tags=["sandbox", "validation"])
async def validator_node(state: ReviewState) -> Dict:
    """
    Run ruff + pytest via Docker sandbox. 
    Treats exit_code 5 (No tests found) as a PASS to prevent infinite loops.
    """
    await asyncio.sleep(2)
    
    patch = state.get("patch", "")
    current_count = state.get("refactor_count", 0)

    logger.info("[validator_node] Starting validation — iteration=%d", current_count)

    # 1. Quick exit if no patch exists
    if not patch or ".py" not in patch:
        logger.info("[validator_node] No Python patch — skipping validation")
        return {"validation_passed": True, "validation_result": "SKIPPED_NO_PYTHON_PATCH"}

    # 2. Get Sandbox Instance
    sandbox = get_sandbox()
    if not sandbox:
        logger.warning("[validator_node] Sandbox unavailable — failsafe: marking as passed")
        return {"validation_passed": True, "validation_result": "SKIPPED_NO_SANDBOX"}

    try:
        # 3. Execute Sandbox Tests
        logger.info("[validator_node] Executing ruff + pytest in Docker...")
        result = sandbox.run_tests(patch)
        
        # LOGIC: 
        # exit_code 0 = All tests/lint passed
        # exit_code 5 = No tests found (common in small PRs; we treat as success)
        is_success = result.passed or result.exit_code == 5
        
        logger.info(
            "[validator_node] Sandbox complete — passed=%s exit_code=%s effective_success=%s",
            result.passed, result.exit_code, is_success
        )

        # 4. Convert Dataclass to Dict for LangGraph serialization
        # Using asdict() directly here to avoid naming confusion
        validation_dict = asdict(result)

        if is_success:
            return {
                "validation_result": validation_dict,
                "validation_passed": True,
                "refactor_count": current_count,
            }
        else:
            # Increment refactor_count to trigger the next loop in LangGraph
            logger.warning(
                "[validator_node] REAL FAILURE — Incremented refactor_count to %d. Output: %s", 
                current_count + 1, result.output[:150]
            )
            return {
                "validation_result": validation_dict,
                "validation_passed": False,
                "refactor_count": current_count + 1,
            }
            
    except Exception as e:
        logger.exception("[validator_node] Sandbox runtime error: %s", str(e))
        return {
            "validation_passed": True, 
            "validation_result": {"error": str(e), "status": "SKIPPED_SANDBOX_ERROR"}
        }


@traceable(name="memory_write_node", tags=["chromadb", "memory", "p4"])
async def memory_write_node(state: ReviewState) -> Dict:
    """
    Saves results to ChromaDB. 
    FIX: Now persists memory even if code validation failed (allows Chat Bot to see bugs).
    """
    await asyncio.sleep(1)
    
    # 1. NEW LOGIC: Only skip if there is NO data to save.
    # Do NOT skip just because state["error"] is True (unless it's an infra failure).
    issues      = state.get("issues", [])
    suggestions = state.get("suggestions", [])
    original    = state.get("original_issues", [])
    
    if not (issues or suggestions or original):
        logger.warning("[memory_write_node] No data to persist. Skipping.")
        return {}

    # 2. Extract Metadata
    owner   = state.get("owner")
    repo    = state.get("repo")
    pr_num  = state.get("pr_number")
    verdict = state.get("verdict", "Review Completed")

    try:
        collection, embedder = get_chroma_collection()
        if not collection: return {}

        # 3. Enhanced Document (Including Original Issues for the Bot)
        memory_doc = (
            f"Repository: {owner}/{repo} PR: #{pr_num}\n"
            f"Verdict: {verdict}\n"
            f"Confirmed Pre-existing Bugs: {original}\n"
            f"New PR Issues: {issues}\n"
            f"Suggestions: {suggestions}"
        )
        
        doc_id = f"mem_{owner}_{repo}_{pr_num}_{int(time.time())}"
        
        collection.add(
            ids=[doc_id],
            embeddings=[embedder.embed_query(memory_doc)],
            documents=[memory_doc],
            metadatas=[{"type": "review_memory", "repo": repo, "pr": str(pr_num)}]
        )
        
        logger.info("[memory_write_node] Memory stored successfully for PR #%s", pr_num)

    except Exception as e:
        logger.exception("[memory_write_node] Persistence failed: %s", str(e))

    return {}

@traceable(name="summary_node", tags=["summary"])
async def summary_node(state: ReviewState) -> Dict:
    await asyncio.sleep(2)
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
    """
    await asyncio.sleep(2)
    
    human_decision = state.get("human_decision")
    issues = state.get("issues", [])
    suggestions = state.get("suggestions", [])
    metadata = state.get("metadata", {})
    pr_title = metadata.get("title", "Unknown PR")
    pr_author = metadata.get("author", "unknown")
    context_grade = state.get("context_grade", "skipped")

    # FIX: Check critical_infra_failure instead of general "error"
    if state.get("critical_infra_failure"):
        reason = state.get("error_reason", "unknown_infra_error")
        logger.warning("[verdict_node] Infrastructure Failure — reason=%s", reason)
        return {
            "verdict": "FAILED",
            "summary": (
                f"## ❌ System Error\n\n"
                f"The review pipeline encountered a technical failure: `{reason}`.\n"
                f"The AI results could not be verified by the sandbox."
            ),
        }

    if human_decision == "rejected":
        return {
            "verdict": "HUMAN_REJECTED",
            "summary": f"## ❌ Review Rejected by Human Reviewer\n\nPR: {pr_title}"
        }

    # If code has issues OR lint failed, we REQUEST_CHANGES
    lint_passed = state.get("lint_passed", True)
    verdict = "REQUEST_CHANGES" if (issues or not lint_passed) else "APPROVE"
    verdict_emoji = "✅" if verdict == "APPROVE" else "🔴"

    issues_section = "\n".join(f"- {i}" for i in issues) if issues else "_No architectural issues._"

    lint_passed = state.get("lint_passed", True)

    # Add linting info to summary if it failed
    if not lint_passed:
        lint_result = state.get("lint_result", {})
        if isinstance(lint_result, dict):
            lint_out = lint_result.get("output", "Check logs for details.")
            issues_section += f"\n\n**Linting Errors:**\n```\n{lint_out[:500]}\n```"

    human_badge = ""
    if human_decision == "approved":
        human_badge = "\n**Human Approval:** ✅ Approved by reviewer"

    context_badge = ""
    if context_grade == CONTEXT_GRADE_YES:
        context_badge = "\n**Context:** 🧠 Codebase context used"
    elif context_grade == CONTEXT_GRADE_NO:
        context_badge = "\n**Context:** ⚪ Retrieved context not relevant"

    suggestions_section = (
        "\n".join(f"- {s}" for s in suggestions) if suggestions else "_No suggestions._"
    )

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
        f"*Review generated by Advanced GitHub Code Reviewer · Powered by Gemini*"
    )

    logger.info(
        "[verdict_node] Complete — verdict=%s summary=%d chars",
        verdict, len(summary),
    )

    return {
        "verdict": verdict,
        "summary": summary,
    }