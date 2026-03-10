"""
app/graph/nodes.py

LangGraph Agent Nodes
----------------------
Each function is one node in the review workflow graph.

Node execution order (defined in workflow.py):
    fetch_diff_node
        → analyze_code_node   (uses ChromaDB context + Gemini)
        → reflect_node        (loops up to 2 times)
        → verdict_node        (produces final APPROVE / REQUEST_CHANGES)

Every node:
    - Receives the full ReviewState
    - Returns a PARTIAL state dict (LangGraph merges it)
    - Is decorated with @traceable so every call appears in LangSmith
    - Logs start / end to the log file

API Key Flow:
    config.py loads GOOGLE_API_KEY from .env into settings AND os.environ.
    init_chat_model / init_embeddings read os.environ at invocation time.
    This survives LangGraph's internal model re-instantiation at runtime.
"""

import os
import sys

from langsmith import traceable
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.messages import SystemMessage, HumanMessage

from app.graph.state import ReviewState
from app.mcp.github_client import GitHubClient
from app.core.config import settings
from app.core.exceptions import CustomException
from app.core.logger import get_logger

logger = get_logger(__name__)


# ── API key validation ────────────────────────────────────────────────────────
# config.py sets os.environ["GOOGLE_API_KEY"] after loading settings.
# Verify it landed correctly before initialising the LLM.

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
# Uses init_chat_model — provider-agnostic, reads GOOGLE_API_KEY from
# os.environ. Survives LangGraph's internal re-instantiation at runtime.

llm = init_chat_model(
    model="gemini-2.5-flash",
    model_provider="google_genai",
    configurable_fields="any",
)

logger.info("LLM initialised — gemini-2.5-flash via init_chat_model")


# ── ChromaDB long-term memory ─────────────────────────────────────────────────
# Lazy import — only initialised when the node actually runs.
# This avoids ChromaDB startup cost during import / tests.
#
# Embedding stack:
#   init_embeddings("google_genai:models/text-embedding-004")
#       → GoogleGenerativeAIEmbeddings (langchain-google-genai)
#       → google-genai SDK (not deprecated)
#   create_langchain_embedding(embedder)
#       → ChromaDB official bridge — implements full ChromaDB EmbeddingFunction
#         interface (name, is_legacy, default_space, supported_spaces, etc.)
#         so no manual adapter or whack-a-mole with new interface methods.

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

# ── Node 1: fetch_diff_node ───────────────────────────────────────────────────

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


# ── Node 2: analyze_code_node ─────────────────────────────────────────────────

@traceable(name="analyze_code_node", tags=["llm", "analysis"])
def analyze_code_node(state: ReviewState) -> dict:
    """
    Uses Gemini to analyze the PR diff and identify issues / suggestions.
    Injects ChromaDB repo context to make the review codebase-aware.

    Reads  : state["diff"], state["metadata"], state["repo_context"]
    Writes : state["issues"], state["suggestions"], state["repo_context"]
    """
    diff     = state["diff"]
    metadata = state["metadata"]

    logger.info(
        f"[analyze_code_node] Starting analysis — "
        f"PR: '{metadata.get('title', 'unknown')}'"
    )

    # ── Step 1: retrieve repo context from ChromaDB ──────────────────────
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
                "[analyze_code_node] No ChromaDB context found — "
                "proceeding without it"
            )

    except Exception as e:
        # Memory failure must never block the review
        logger.warning(
            f"[analyze_code_node] ChromaDB query failed (non-fatal): {e}"
        )
        repo_context = ""

    # ── Step 2: build prompt ─────────────────────────────────────────────
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

    # ── Step 3: call Gemini ──────────────────────────────────────────────
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ])
        raw_output = response.content
        logger.info("[analyze_code_node] LLM response received")

    except Exception as e:
        logger.error(f"[analyze_code_node] LLM call failed: {e}")
        raise CustomException(str(e), sys)

    # ── Step 4: parse structured output ─────────────────────────────────
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


def _parse_llm_output(raw: str) -> tuple[list[str], list[str]]:
    """
    Parses the structured ISSUES / SUGGESTIONS response from the LLM.
    Returns (issues, suggestions) as lists of strings.
    Falls back gracefully if the LLM doesn't follow the format exactly.
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


# ── Node 3: reflect_node ──────────────────────────────────────────────────────

@traceable(name="reflect_node", tags=["llm", "reflection"])
def reflect_node(state: ReviewState) -> dict:
    """
    Self-reflection pass — asks Gemini to review its own previous analysis
    and add anything it missed. This is what makes the agent 'agentic'.

    Runs up to 2 times (controlled by reflection_count in workflow.py).

    Reads  : state["issues"], state["suggestions"], state["diff"],
             state["reflection_count"]
    Writes : state["issues"], state["suggestions"], state["reflection_count"]
    """
    current_count = state["reflection_count"]
    logger.info(f"[reflect_node] Reflection pass #{current_count + 1} starting")

    existing_issues      = state["issues"]
    existing_suggestions = state["suggestions"]
    diff                 = state["diff"]

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

    # Merge — deduplicate by lowercased content
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


# ── Node 4: verdict_node ──────────────────────────────────────────────────────

@traceable(name="verdict_node", tags=["llm", "verdict"])
def verdict_node(state: ReviewState) -> dict:
    """
    Produces the final APPROVE or REQUEST_CHANGES verdict and formats
    the full review comment that will be posted to GitHub.

    Reads  : state["issues"], state["suggestions"], state["metadata"]
    Writes : state["verdict"], state["summary"]
    """
    issues      = state["issues"]
    suggestions = state["suggestions"]
    metadata    = state["metadata"]

    logger.info(
        f"[verdict_node] Generating final verdict — "
        f"{len(issues)} issue(s), {len(suggestions)} suggestion(s)"
    )

    # Simple deterministic rule: any issues → REQUEST_CHANGES
    verdict = "REQUEST_CHANGES" if issues else "APPROVE"

    verdict_emoji = "✅" if verdict == "APPROVE" else "🔴"

    issues_section = (
        "\n".join(f"- {i}" for i in issues)
        if issues else "_No issues found._"
    )
    suggestions_section = (
        "\n".join(f"- {s}" for s in suggestions)
        if suggestions else "_No suggestions._"
    )

    summary = f"""## {verdict_emoji} AI Code Review

**PR:** {metadata.get('title', 'N/A')}
**Author:** {metadata.get('author', 'N/A')}
**Verdict:** `{verdict}`

---

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