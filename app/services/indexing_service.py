"""
app/services/indexing_service.py

Repository Indexing Service — P4
---------------------------------
Orchestrates the clone → chunk → embed → store pipeline for indexing
a GitHub repository into ChromaDB for use by the RAG review pipeline.

Once a repository is indexed, retrieve_context_node queries ChromaDB
using PR diff as the search vector and grade_context_node grades
the relevance before injecting context into analyze_code_node.

Pipeline Overview
-----------------
1. Clone the repository from GitHub using a temp directory
2. Walk all .py files recursively (configurable via FILE_EXTENSIONS)
3. Chunk each file into overlapping token windows
4. Embed each chunk using gemini-embedding-001
5. Store chunks + metadata in ChromaDB "repo_context" collection
6. Clean up temp directory

Chunking Strategy
-----------------
Files are split into chunks of CHUNK_SIZE lines with CHUNK_OVERLAP lines
of overlap between adjacent chunks. This preserves function-level context
across chunk boundaries and prevents relevant code from being split exactly
at a function boundary.

    CHUNK_SIZE    = 50 lines per chunk
    CHUNK_OVERLAP = 10 lines overlap

For a 200-line file this produces approximately 5 chunks.

ChromaDB Document IDs
---------------------
Each chunk is stored with a deterministic ID:
    "{owner}_{repo}_{filepath}_{chunk_index}"

This enables safe re-indexing — calling index_repository() again for the
same repo will produce the same IDs, and ChromaDB upsert semantics ensure
existing chunks are updated rather than duplicated.

Supported File Extensions
-------------------------
FILE_EXTENSIONS controls which file types are indexed.
Default: [".py"] — Python files only (P4).
Extend in P5/P6: [".py", ".js", ".ts", ".go"] for multi-language support.

Rate Limiting
-------------
Embedding calls use a simple per-batch sleep to avoid hitting the
gemini-embedding-001 RPM limit on the free tier.
EMBED_BATCH_SIZE  = 10 chunks per API call
EMBED_BATCH_SLEEP = 1 second between batches
"""

import os
import sys
import time
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from app.core.exceptions import CustomException
from app.core.logger import get_logger

logger = get_logger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────

FILE_EXTENSIONS: List[str] = [".py"]
"""
File extensions to index. Only files matching these extensions are chunked
and embedded. Extend this list in P5/P6 for multi-language support.
"""

CHUNK_SIZE: int = 50
"""
Number of lines per chunk. 50 lines ≈ one function or class method.
Larger chunks have more context but cost more tokens per embedding.
"""

CHUNK_OVERLAP: int = 10
"""
Number of lines of overlap between adjacent chunks.
Prevents relevant code from being split exactly at a chunk boundary.
"""

EMBED_BATCH_SIZE: int = 10
"""
Number of chunks to embed per API call to gemini-embedding-001.
Larger batches are more efficient but risk hitting token limits.
"""

EMBED_BATCH_SLEEP: float = 1.0
"""
Seconds to sleep between embedding batches.
Prevents RPM burst errors on the free tier.
"""

MAX_FILE_SIZE_KB: int = 100
"""
Maximum file size in kilobytes to index.
Files larger than this are skipped to avoid very long embedding calls.
Large files (e.g. generated code, data files) are usually not useful context.
"""

COLLECTION_NAME: str = "repo_context"
"""
ChromaDB collection name. Must match the collection used by nodes.py
retrieve_context_node and memory_write_node.
"""


# ── Data Classes ──────────────────────────────────────────────────────────────

class IndexingResult:
    """
    Result returned by index_repository().

    Attributes
    ----------
    owner : str
        GitHub repository owner.
    repo : str
        Repository name.
    total_files : int
        Number of .py files found in the repository.
    indexed_files : int
        Number of files successfully chunked and embedded.
    skipped_files : int
        Number of files skipped (too large, encoding error, etc).
    total_chunks : int
        Total number of chunks stored in ChromaDB.
    duration_seconds : float
        Total wall-clock time for the full indexing pipeline.
    error : Optional[str]
        Error message if indexing failed, None on success.
    """

    def __init__(
        self,
        owner: str,
        repo: str,
        total_files: int = 0,
        indexed_files: int = 0,
        skipped_files: int = 0,
        total_chunks: int = 0,
        duration_seconds: float = 0.0,
        error: Optional[str] = None,
    ):
        self.owner = owner
        self.repo = repo
        self.total_files = total_files
        self.indexed_files = indexed_files
        self.skipped_files = skipped_files
        self.total_chunks = total_chunks
        self.duration_seconds = duration_seconds
        self.error = error

    @property
    def success(self) -> bool:
        """True if indexing completed without a fatal error."""
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict for API response."""
        return {
            "owner":            self.owner,
            "repo":             self.repo,
            "success":          self.success,
            "total_files":      self.total_files,
            "indexed_files":    self.indexed_files,
            "skipped_files":    self.skipped_files,
            "total_chunks":     self.total_chunks,
            "duration_seconds": round(self.duration_seconds, 2),
            "error":            self.error,
        }

    def __repr__(self) -> str:
        return (
            f"IndexingResult({self.owner}/{self.repo} "
            f"files={self.indexed_files}/{self.total_files} "
            f"chunks={self.total_chunks} "
            f"duration={self.duration_seconds:.1f}s "
            f"success={self.success})"
        )


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_file(
    filepath: str,
    content: str,
    owner: str,
    repo: str,
) -> List[Dict[str, Any]]:
    """
    Split a file's content into overlapping line-based chunks.

    Each chunk is a dict containing:
        - id       : deterministic ChromaDB document ID
        - text     : the chunk text (lines joined with newline)
        - metadata : source filepath, repo, chunk index, line range

    Parameters
    ----------
    filepath : str
        Relative file path within the repository (e.g. "app/utils.py").
    content : str
        Full file content as a string.
    owner : str
        GitHub repository owner — used in document ID.
    repo : str
        Repository name — used in document ID and metadata.

    Returns
    -------
    List[Dict[str, Any]]
        List of chunk dicts, each with id, text, and metadata.
    """
    lines = content.splitlines()
    chunks = []
    chunk_index = 0

    # Sanitize filepath for use in document IDs (no slashes or spaces)
    safe_filepath = filepath.replace("/", "_").replace("\\", "_").replace(" ", "_")

    i = 0
    while i < len(lines):
        chunk_lines = lines[i: i + CHUNK_SIZE]
        chunk_text = "\n".join(chunk_lines)

        if chunk_text.strip():  # skip empty chunks
            doc_id = f"{owner}_{repo}_{safe_filepath}_{chunk_index}"
            start_line = i + 1
            end_line = i + len(chunk_lines)

            chunks.append({
                "id":   doc_id,
                "text": chunk_text,
                "metadata": {
                    "source":     f"{owner}/{repo}",
                    "filepath":   filepath,
                    "repo":       f"{owner}/{repo}",
                    "type":       "code_chunk",
                    "chunk_index": str(chunk_index),
                    "start_line": str(start_line),
                    "end_line":   str(end_line),
                },
            })
            chunk_index += 1

        # Advance by CHUNK_SIZE - CHUNK_OVERLAP for overlap
        i += max(1, CHUNK_SIZE - CHUNK_OVERLAP)

    return chunks


# ── File Discovery ────────────────────────────────────────────────────────────

def _discover_files(repo_path: str) -> List[Tuple[str, str]]:
    """
    Walk repo_path and return (relative_path, absolute_path) for indexable files.

    Filters by FILE_EXTENSIONS and MAX_FILE_SIZE_KB.
    Skips hidden directories (starting with '.') and __pycache__.

    Parameters
    ----------
    repo_path : str
        Absolute path to the cloned repository root.

    Returns
    -------
    List[Tuple[str, str]]
        List of (relative_path, absolute_path) tuples for files to index.
    """
    files = []
    repo_root = Path(repo_path)

    for abs_path in repo_root.rglob("*"):
        # Skip directories
        if not abs_path.is_file():
            continue

        # Skip hidden dirs and __pycache__
        parts = abs_path.relative_to(repo_root).parts
        if any(p.startswith(".") or p == "__pycache__" for p in parts):
            continue

        # Filter by extension
        if abs_path.suffix not in FILE_EXTENSIONS:
            continue

        # Skip oversized files
        size_kb = abs_path.stat().st_size / 1024
        if size_kb > MAX_FILE_SIZE_KB:
            logger.debug(
                "[indexing_service] Skipping large file — "
                "%s (%.1f KB > %d KB limit)",
                abs_path.relative_to(repo_root),
                size_kb,
                MAX_FILE_SIZE_KB,
            )
            continue

        relative_path = str(abs_path.relative_to(repo_root))
        files.append((relative_path, str(abs_path)))

    logger.info(
        "[indexing_service] _discover_files: found %d file(s) in %s",
        len(files), repo_path,
    )
    return files


# ── ChromaDB Helpers ──────────────────────────────────────────────────────────

def _get_chroma_collection_and_embedder():
    """
    Get or create the ChromaDB collection and embedder for indexing.

    Uses the same collection (repo_context) and embedding model
    (gemini-embedding-001) as nodes.py so retrieval and indexing
    are consistent.

    Returns
    -------
    Tuple[collection, embedder] or (None, None) on failure.
    """
    try:
        import chromadb
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.error(
                "[indexing_service] GOOGLE_API_KEY not set — "
                "cannot initialize ChromaDB embedder"
            )
            return None, None

        embedder = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=api_key,
            task_type="RETRIEVAL_DOCUMENT",
        )

        client = chromadb.PersistentClient(path="./chroma_store")

        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            "[indexing_service] ChromaDB ready — "
            "collection=%s existing_count=%d",
            COLLECTION_NAME, collection.count(),
        )

        return collection, embedder

    except Exception as e:
        logger.exception(
            "[indexing_service] ChromaDB init failed — error=%s", str(e),
        )
        return None, None


def _embed_and_store_chunks(
    chunks: List[Dict[str, Any]],
    collection: Any,
    embedder: Any,
    owner: str,
    repo: str,
) -> int:
    """
    Embed a list of chunks and upsert them into ChromaDB.

    Processes chunks in batches of EMBED_BATCH_SIZE with a sleep
    between batches to avoid RPM burst on the free tier.

    Parameters
    ----------
    chunks : List[Dict]
        List of chunk dicts from _chunk_file().
    collection : chromadb.Collection
        Target ChromaDB collection.
    embedder : GoogleGenerativeAIEmbeddings
        Embedding model instance.
    owner : str
        Repository owner — for logging.
    repo : str
        Repository name — for logging.

    Returns
    -------
    int
        Number of chunks successfully stored.
    """
    stored = 0
    total = len(chunks)

    for batch_start in range(0, total, EMBED_BATCH_SIZE):
        batch = chunks[batch_start: batch_start + EMBED_BATCH_SIZE]
        batch_num = batch_start // EMBED_BATCH_SIZE + 1
        total_batches = (total + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE

        logger.info(
            "[indexing_service] Embedding batch %d/%d — "
            "chunks=%d/%d %s/%s",
            batch_num, total_batches,
            batch_start + len(batch), total,
            owner, repo,
        )

        try:
            texts = [c["text"] for c in batch]
            embeddings = embedder.embed_documents(texts)

            collection.upsert(
                ids=[c["id"] for c in batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=[c["metadata"] for c in batch],
            )

            stored += len(batch)

            logger.debug(
                "[indexing_service] Batch %d/%d stored — "
                "batch_size=%d stored_total=%d",
                batch_num, total_batches, len(batch), stored,
            )

        except Exception as e:
            logger.warning(
                "[indexing_service] Batch %d/%d failed — "
                "skipping batch. error=%s",
                batch_num, total_batches, str(e),
            )

        # Rate limit protection between batches
        if batch_start + EMBED_BATCH_SIZE < total:
            logger.debug(
                "[indexing_service] Sleeping %.1fs between batches "
                "(RPM protection)",
                EMBED_BATCH_SLEEP,
            )
            time.sleep(EMBED_BATCH_SLEEP)

    return stored


# ── Core Service ──────────────────────────────────────────────────────────────

async def index_repository(
    owner: str,
    repo: str,
    github_token: Optional[str] = None,
) -> IndexingResult:
    """
    Clone, chunk, embed, and store a GitHub repository in ChromaDB.

    This is the main entry point for the P4 indexing pipeline.
    Called by POST /repos/index (app/api/routes/repos.py).

    After indexing completes, retrieve_context_node will find relevant
    code chunks when reviewing PRs for this repository.

    Pipeline
    --------
    1. Validate inputs
    2. Initialize ChromaDB collection and embedder
    3. Clone the repo to a temp directory using git
    4. Discover all indexable .py files
    5. Read, chunk, and embed each file
    6. Upsert all chunks into ChromaDB
    7. Clean up temp directory
    8. Return IndexingResult

    Parameters
    ----------
    owner : str
        GitHub username or organization name.
    repo : str
        Repository name (without owner prefix).
    github_token : str, optional
        GitHub Personal Access Token for private repositories.
        If not provided, uses GITHUB_TOKEN from environment.
        Public repositories do not require a token.

    Returns
    -------
    IndexingResult
        Contains counts of files indexed, chunks stored, and duration.
        result.success is True if no fatal error occurred.
        result.error contains the error message on failure.

    Notes
    -----
    Calling this function again for the same repo is safe — ChromaDB
    upsert semantics update existing chunks rather than duplicating them.
    Use this to re-index after significant code changes.
    """
    start_time = time.time()
    result = IndexingResult(owner=owner, repo=repo)
    temp_dir = None

    try:
        logger.info(
            "[indexing_service] index_repository: Starting — %s/%s",
            owner, repo,
        )

        # ── Input validation ──────────────────────────────────────────────────
        owner = owner.strip()
        repo = repo.strip()

        if not owner or not repo:
            raise ValueError("owner and repo must not be empty")

        # ── ChromaDB setup ────────────────────────────────────────────────────
        collection, embedder = _get_chroma_collection_and_embedder()

        if not collection or not embedder:
            result.error = (
                "ChromaDB or embedding model unavailable — "
                "check GOOGLE_API_KEY and chromadb installation"
            )
            result.duration_seconds = time.time() - start_time
            logger.error(
                "[indexing_service] index_repository: Failed — %s",
                result.error,
            )
            return result

        # ── Clone repository ──────────────────────────────────────────────────
        token = github_token or os.environ.get("GITHUB_TOKEN", "")

        if token:
            clone_url = (
                f"https://{token}@github.com/{owner}/{repo}.git"
            )
        else:
            clone_url = f"https://github.com/{owner}/{repo}.git"

        temp_dir = tempfile.mkdtemp(prefix=f"indexing_{owner}_{repo}_")

        logger.info(
            "[indexing_service] index_repository: Cloning — "
            "%s/%s → %s",
            owner, repo, temp_dir,
        )

        import subprocess
        clone_result = subprocess.run(
            ["git", "clone", "--depth", "1", clone_url, temp_dir],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if clone_result.returncode != 0:
            error_msg = clone_result.stderr.strip()
            # Mask token in error message for security
            error_msg = error_msg.replace(token, "***") if token else error_msg
            raise RuntimeError(
                f"git clone failed (exit={clone_result.returncode}): {error_msg}"
            )

        logger.info(
            "[indexing_service] index_repository: Clone complete — %s/%s",
            owner, repo,
        )

        # ── Discover files ────────────────────────────────────────────────────
        file_list = _discover_files(temp_dir)
        result.total_files = len(file_list)

        if result.total_files == 0:
            logger.warning(
                "[indexing_service] index_repository: No indexable files found — "
                "%s/%s extensions=%s",
                owner, repo, FILE_EXTENSIONS,
            )
            result.duration_seconds = time.time() - start_time
            return result

        # ── Chunk, embed, store ───────────────────────────────────────────────
        all_chunks: List[Dict[str, Any]] = []

        for relative_path, abs_path in file_list:
            try:
                content = Path(abs_path).read_text(encoding="utf-8", errors="ignore")

                if not content.strip():
                    logger.debug(
                        "[indexing_service] Skipping empty file — %s",
                        relative_path,
                    )
                    result.skipped_files += 1
                    continue

                chunks = _chunk_file(relative_path, content, owner, repo)
                all_chunks.extend(chunks)
                result.indexed_files += 1

                logger.debug(
                    "[indexing_service] Chunked — %s → %d chunk(s)",
                    relative_path, len(chunks),
                )

            except Exception as e:
                logger.warning(
                    "[indexing_service] Failed to read/chunk file — "
                    "%s error=%s",
                    relative_path, str(e),
                )
                result.skipped_files += 1

        logger.info(
            "[indexing_service] index_repository: Chunking complete — "
            "files=%d/%d chunks=%d",
            result.indexed_files, result.total_files, len(all_chunks),
        )

        if not all_chunks:
            logger.warning(
                "[indexing_service] index_repository: No chunks produced — "
                "nothing to embed"
            )
            result.duration_seconds = time.time() - start_time
            return result

        # ── Embed and store ───────────────────────────────────────────────────
        stored = _embed_and_store_chunks(
            all_chunks, collection, embedder, owner, repo
        )
        result.total_chunks = stored

        logger.info(
            "[indexing_service] index_repository: Complete — "
            "%s/%s files=%d/%d chunks=%d duration=%.1fs",
            owner, repo,
            result.indexed_files, result.total_files,
            result.total_chunks,
            time.time() - start_time,
        )

    except Exception as e:
        logger.exception(
            "[indexing_service] index_repository: Fatal error — "
            "%s/%s error=%s",
            owner, repo, str(e),
        )
        result.error = str(e)

    finally:
        # ── Clean up temp directory ───────────────────────────────────────────
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.debug(
                    "[indexing_service] index_repository: Temp dir cleaned — %s",
                    temp_dir,
                )
            except Exception:
                logger.warning(
                    "[indexing_service] index_repository: "
                    "Could not clean temp dir — %s",
                    temp_dir,
                )

        result.duration_seconds = time.time() - start_time

    return result


async def get_index_status(owner: str, repo: str) -> Dict[str, Any]:
    """
    Check whether a repository has been indexed in ChromaDB.

    Queries ChromaDB for documents with source metadata matching
    "{owner}/{repo}" to determine if indexing has been run.

    Parameters
    ----------
    owner : str
        GitHub repository owner.
    repo : str
        Repository name.

    Returns
    -------
    Dict[str, Any]
        {
            "owner": str,
            "repo": str,
            "indexed": bool,      — True if any chunks found
            "chunk_count": int,   — number of chunks in ChromaDB
            "collection_total": int — total chunks across all repos
        }
    """
    try:
        collection, _ = _get_chroma_collection_and_embedder()

        if not collection:
            return {
                "owner":            owner,
                "repo":             repo,
                "indexed":          False,
                "chunk_count":      0,
                "collection_total": 0,
                "error":            "ChromaDB unavailable",
            }

        # Query by metadata filter
        results = collection.get(
            where={"source": f"{owner}/{repo}"},
            include=["metadatas"],
        )

        chunk_count = len(results.get("ids", []))

        logger.info(
            "[indexing_service] get_index_status: %s/%s — "
            "indexed=%s chunk_count=%d",
            owner, repo, chunk_count > 0, chunk_count,
        )

        return {
            "owner":            owner,
            "repo":             repo,
            "indexed":          chunk_count > 0,
            "chunk_count":      chunk_count,
            "collection_total": collection.count(),
        }

    except Exception as e:
        logger.exception(
            "[indexing_service] get_index_status: error — %s/%s error=%s",
            owner, repo, str(e),
        )
        return {
            "owner":   owner,
            "repo":    repo,
            "indexed": False,
            "error":   str(e),
        }


async def clear_repository_index(owner: str, repo: str) -> Dict[str, Any]:
    """
    Remove all indexed chunks for a specific repository from ChromaDB.

    Used when a repository needs to be re-indexed from scratch.
    Deletes all documents where metadata.source == "{owner}/{repo}".

    Parameters
    ----------
    owner : str
        GitHub repository owner.
    repo : str
        Repository name.

    Returns
    -------
    Dict[str, Any]
        {
            "owner": str,
            "repo": str,
            "deleted_count": int,
            "success": bool
        }
    """
    try:
        collection, _ = _get_chroma_collection_and_embedder()

        if not collection:
            return {
                "owner":         owner,
                "repo":          repo,
                "deleted_count": 0,
                "success":       False,
                "error":         "ChromaDB unavailable",
            }

        # Get all IDs for this repo
        results = collection.get(
            where={"source": f"{owner}/{repo}"},
            include=[],
        )

        ids_to_delete = results.get("ids", [])

        if not ids_to_delete:
            logger.info(
                "[indexing_service] clear_repository_index: "
                "Nothing to delete — %s/%s not indexed",
                owner, repo,
            )
            return {
                "owner":         owner,
                "repo":          repo,
                "deleted_count": 0,
                "success":       True,
            }

        collection.delete(ids=ids_to_delete)

        logger.info(
            "[indexing_service] clear_repository_index: "
            "Deleted %d chunk(s) — %s/%s",
            len(ids_to_delete), owner, repo,
        )

        return {
            "owner":         owner,
            "repo":          repo,
            "deleted_count": len(ids_to_delete),
            "success":       True,
        }

    except Exception as e:
        logger.exception(
            "[indexing_service] clear_repository_index: error — "
            "%s/%s error=%s",
            owner, repo, str(e),
        )
        return {
            "owner":         owner,
            "repo":          repo,
            "deleted_count": 0,
            "success":       False,
            "error":         str(e),
        }