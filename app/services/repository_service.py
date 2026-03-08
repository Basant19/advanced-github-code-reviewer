"""
app/services/repository_service.py

Repository Vector Memory Service
----------------------------------
Manages ChromaDB collections for long-term codebase memory.

Each repository gets its own isolated collection in ChromaDB, keyed by
"owner__repo" (double underscore to avoid path separator conflicts).

Responsibilities:
    1. index_repository()  — crawl GitHub repo files, chunk, embed, store
    2. query_context()     — similarity search → relevant code chunks
    3. delete_repository() — remove all embeddings for a repo

This service is called:
    - By the webhook when a new repo is registered (index_repository)
    - By analyze_code_node in nodes.py (via _get_chroma_collection)
    - By review routes when a repo is removed (delete_repository)

Architecture note:
    ChromaDB uses Google's embedding-001 model (same API key as Gemini).
    Each chunk is stored with metadata: repo, filepath, language, chunk_index.
    This metadata enables filtered queries per repo.

Supported file extensions for indexing:
    Python, JavaScript, TypeScript, Java, Go, Rust, C/C++, Ruby,
    Markdown, YAML, TOML, JSON (config files provide useful context).
"""

import sys
import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction

from app.mcp.github_client import GitHubClient
from app.core.config import settings
from app.core.exceptions import CustomException
from app.core.logger import get_logger

logger = get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

CHROMA_STORE_PATH = "./chroma_store"

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".go", ".rs", ".c", ".cpp", ".h",
    ".rb", ".php", ".cs", ".swift", ".kt",
    ".md", ".yaml", ".yml", ".toml", ".json",
}

# Max characters per chunk — keeps each embedding within token limits
CHUNK_SIZE = 1500

# Max files to index per repo — prevents runaway costs on huge repos
MAX_FILES = 200


# ── ChromaDB client + embedding function (module-level singletons) ────────────

def _get_embedding_function() -> GoogleGenerativeAiEmbeddingFunction:
    """Returns the Google embedding function using GOOGLE_API_KEY."""
    return GoogleGenerativeAiEmbeddingFunction(
        api_key=settings.google_api_key,
        model_name="models/embedding-001",
    )


def _get_chroma_client() -> chromadb.PersistentClient:
    """Returns a persistent ChromaDB client writing to CHROMA_STORE_PATH."""
    os.makedirs(CHROMA_STORE_PATH, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_STORE_PATH)


def _collection_name(owner: str, repo: str) -> str:
    """
    Produces a stable, ChromaDB-safe collection name for a repo.
    ChromaDB requires: 3-63 chars, alphanumeric + hyphens, no leading/trailing hyphens.
    We sanitize by replacing non-alphanumeric chars with hyphens.
    """
    raw  = f"{owner}-{repo}".lower()
    safe = "".join(c if c.isalnum() or c == "-" else "-" for c in raw)
    # Truncate to 63 chars (ChromaDB limit)
    return safe[:63].strip("-") or "default-collection"


def _get_collection(owner: str, repo: str) -> chromadb.Collection:
    """Gets or creates the ChromaDB collection for a given repo."""
    client       = _get_chroma_client()
    embedding_fn = _get_embedding_function()
    name         = _collection_name(owner, repo)

    return client.get_or_create_collection(
        name=name,
        embedding_function=embedding_fn,
        metadata={"owner": owner, "repo": repo},
    )


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """
    Splits file content into overlapping chunks for better retrieval.
    Uses a simple line-aware split to avoid cutting mid-function.

    Args:
        text:       Raw file content
        chunk_size: Max characters per chunk

    Returns:
        List of text chunks, each <= chunk_size characters.
    """
    if not text or not text.strip():
        return []

    lines  = text.splitlines(keepends=True)
    chunks = []
    current = ""

    for line in lines:
        if len(current) + len(line) > chunk_size and current:
            chunks.append(current.strip())
            # 20% overlap: keep last few lines for context continuity
            overlap_lines = current.splitlines()[-5:]
            current = "\n".join(overlap_lines) + "\n"
        current += line

    if current.strip():
        chunks.append(current.strip())

    return chunks


# ── Public API ────────────────────────────────────────────────────────────────

def index_repository(owner: str, repo: str) -> dict:
    """
    Crawls a GitHub repository, chunks all supported files, and stores
    them as embeddings in ChromaDB.

    This function is idempotent — re-indexing clears old embeddings first.

    Args:
        owner: GitHub repository owner login
        repo:  GitHub repository name

    Returns:
        Summary dict: {"files_indexed": int, "chunks_stored": int, "collection": str}

    Raises:
        CustomException on GitHub API failure or ChromaDB error.
    """
    logger.info(f"[index_repository] Starting indexing — {owner}/{repo}")

    try:
        github_client = GitHubClient()
        collection    = _get_collection(owner, repo)

        # Clear existing embeddings to avoid stale data on re-index
        existing = collection.count()
        if existing > 0:
            logger.info(
                f"[index_repository] Clearing {existing} existing chunk(s) "
                f"before re-index"
            )
            all_ids = collection.get()["ids"]
            if all_ids:
                collection.delete(ids=all_ids)

        # Fetch repo file tree from GitHub
        repo_obj   = github_client._get_repo(owner, repo)
        git_tree   = repo_obj.get_git_tree(sha="HEAD", recursive=True)
        all_files  = [
            item for item in git_tree.tree
            if item.type == "blob"
            and Path(item.path).suffix in SUPPORTED_EXTENSIONS
        ]

        if len(all_files) > MAX_FILES:
            logger.warning(
                f"[index_repository] Repo has {len(all_files)} files — "
                f"capping at {MAX_FILES}"
            )
            all_files = all_files[:MAX_FILES]

        documents  : list[str]  = []
        metadatas  : list[dict] = []
        ids        : list[str]  = []
        files_done : int        = 0

        for file_item in all_files:
            filepath  = file_item.path
            extension = Path(filepath).suffix

            try:
                # Fetch raw file content from GitHub
                content_obj = repo_obj.get_contents(filepath)

                # Skip binary files
                if content_obj.encoding == "none":
                    continue

                raw_content = content_obj.decoded_content.decode(
                    "utf-8", errors="ignore"
                )

            except Exception as fetch_err:
                logger.warning(
                    f"[index_repository] Skipping {filepath}: {fetch_err}"
                )
                continue

            chunks = _chunk_text(raw_content)
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{owner}-{repo}-{filepath}-{idx}".replace("/", "_")
                documents.append(chunk)
                metadatas.append({
                    "owner":       owner,
                    "repo":        repo,
                    "filepath":    filepath,
                    "language":    extension.lstrip("."),
                    "chunk_index": idx,
                })
                ids.append(chunk_id)

            files_done += 1

        # Batch upsert into ChromaDB
        if documents:
            # ChromaDB recommends batches of ≤ 100
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                collection.upsert(
                    documents=ids[i : i + batch_size],
                    metadatas=metadatas[i : i + batch_size],
                    ids=ids[i : i + batch_size],
                )
                # Store actual text in documents field
                collection.upsert(
                    documents=documents[i : i + batch_size],
                    metadatas=metadatas[i : i + batch_size],
                    ids=ids[i : i + batch_size],
                )

        result = {
            "files_indexed":  files_done,
            "chunks_stored":  len(documents),
            "collection":     _collection_name(owner, repo),
        }

        logger.info(
            f"[index_repository] Done — "
            f"{files_done} file(s), {len(documents)} chunk(s) stored"
        )
        return result

    except CustomException:
        raise
    except Exception as e:
        logger.error(f"[index_repository] Failed: {e}")
        raise CustomException(str(e), sys)


def query_context(
    owner: str,
    repo:  str,
    query: str,
    n_results: int = 3,
) -> list[str]:
    """
    Performs a similarity search against the repository's ChromaDB collection
    and returns the most relevant code chunks.

    Called by analyze_code_node to inject codebase context into the LLM prompt.

    Args:
        owner:     GitHub repository owner
        repo:      GitHub repository name
        query:     Natural language or code query (typically PR title + diff snippet)
        n_results: Number of chunks to return (default 3)

    Returns:
        List of relevant code chunk strings. Empty list if collection is empty
        or no results found — callers must handle empty gracefully.

    Raises:
        CustomException on ChromaDB error.
    """
    logger.info(
        f"[query_context] Querying context — {owner}/{repo}, "
        f"n_results={n_results}"
    )

    try:
        collection = _get_collection(owner, repo)

        if collection.count() == 0:
            logger.info(
                f"[query_context] Collection empty for {owner}/{repo} "
                f"— returning no context"
            )
            return []

        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, collection.count()),
            where={"repo": repo},
        )

        docs = results.get("documents", [[]])[0]
        logger.info(
            f"[query_context] Retrieved {len(docs)} chunk(s) for {owner}/{repo}"
        )
        return docs

    except Exception as e:
        logger.error(f"[query_context] Failed: {e}")
        raise CustomException(str(e), sys)


def delete_repository(owner: str, repo: str) -> dict:
    """
    Removes all embeddings for a repository from ChromaDB.
    Called when a repository is deregistered from the system.

    Args:
        owner: GitHub repository owner
        repo:  GitHub repository name

    Returns:
        {"deleted": True, "collection": str}

    Raises:
        CustomException on ChromaDB error.
    """
    logger.info(f"[delete_repository] Deleting embeddings — {owner}/{repo}")

    try:
        client = _get_chroma_client()
        name   = _collection_name(owner, repo)

        try:
            client.delete_collection(name=name)
            logger.info(f"[delete_repository] Collection '{name}' deleted")
        except Exception:
            # Collection may not exist — treat as success
            logger.info(
                f"[delete_repository] Collection '{name}' not found "
                f"— nothing to delete"
            )

        return {"deleted": True, "collection": name}

    except Exception as e:
        logger.error(f"[delete_repository] Failed: {e}")
        raise CustomException(str(e), sys)


def get_repository_stats(owner: str, repo: str) -> dict:
    """
    Returns indexing stats for a repository's ChromaDB collection.
    Useful for health checks and the /repositories endpoint.

    Args:
        owner: GitHub repository owner
        repo:  GitHub repository name

    Returns:
        {"collection": str, "chunks": int, "indexed": bool}
    """
    logger.info(f"[get_repository_stats] Fetching stats — {owner}/{repo}")

    try:
        collection = _get_collection(owner, repo)
        count      = collection.count()

        return {
            "collection": _collection_name(owner, repo),
            "chunks":     count,
            "indexed":    count > 0,
        }

    except Exception as e:
        logger.error(f"[get_repository_stats] Failed: {e}")
        raise CustomException(str(e), sys)