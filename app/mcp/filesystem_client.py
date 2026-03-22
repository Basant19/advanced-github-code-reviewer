"""
app/mcp/filesystem_client.py

Filesystem MCP Client — P4 Production Version
----------------------------------------------
Handles reading, chunking, and inspecting files from a locally cloned
GitHub repository for the ChromaDB indexing pipeline.

Responsibility
--------------
This client knows ONLY about filesystem operations:
    - Walking a directory tree to discover indexable files
    - Reading file content safely (encoding-tolerant)
    - Chunking large files into overlapping windows for embedding
    - Extracting file metadata for ChromaDB document metadata fields

It knows NOTHING about:
    - ChromaDB        — handled by indexing_service.py
    - LangGraph state — handled by nodes.py
    - GitHub API      — handled by github_client.py
    - Embedding models — handled by indexing_service.py

Call Chain (P4)
---------------
POST /repos/index
    → indexing_service.index_repository()
        → FilesystemClient.discover_files()    find all .py files
        → FilesystemClient.read_file()         read content
        → FilesystemClient.chunk_file()        split into chunks
        → indexing_service._embed_and_store()  embed + store in ChromaDB

Why This Exists as a Separate Client
-------------------------------------
Consistent with the MCP pattern used by github_client.py and sandbox_client.py
— every external resource has its own client. This makes the indexing pipeline
testable via mock and allows the filesystem strategy to change (e.g. streaming
from GitHub API instead of local clone) without touching indexing_service.py.

Chunking Strategy
-----------------
Files are split into overlapping line-based chunks:
    CHUNK_SIZE    = 50 lines — approximately one function or class method
    CHUNK_OVERLAP = 10 lines — preserves context at chunk boundaries

For a 200-line file this produces ~5 chunks with context overlap.
Each chunk is stored as a separate ChromaDB document.

File Filtering
--------------
discover_files() filters by:
    - File extension (default: .py only)
    - File size (default: max 100 KB)
    - Hidden directories (skipped: .git, .venv, __pycache__, etc.)
    - Empty files (skipped)

All filter thresholds are configurable via constructor parameters.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.core.logger import get_logger

logger = get_logger(__name__)


class FilesystemClient:
    """
    MCP client for local filesystem operations used by the P4 indexing pipeline.

    All methods are synchronous — called from indexing_service.py which
    runs the heavy async work (embedding, ChromaDB upsert) separately.

    Parameters
    ----------
    extensions : Tuple[str, ...]
        File extensions to include when discovering files.
        Default: (".py",) — Python files only.
    max_file_size_kb : int
        Maximum file size in kilobytes to include.
        Files larger than this are skipped. Default: 100 KB.
    chunk_size : int
        Number of lines per chunk. Default: 50.
    chunk_overlap : int
        Number of overlapping lines between adjacent chunks. Default: 10.

    Examples
    --------
    Basic usage:

        client = FilesystemClient()
        files = client.discover_files("/tmp/my_repo")
        for rel_path, abs_path in files:
            content = client.read_file(abs_path)
            if content:
                chunks = client.chunk_file(rel_path, content, "owner", "repo")
                meta = client.get_file_metadata(abs_path, rel_path, "owner", "repo")

    Custom extensions for multi-language support (P5/P6):

        client = FilesystemClient(extensions=(".py", ".js", ".ts"))
    """

    # Directories to skip when walking the repository tree
    SKIP_DIRS = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        "node_modules",
        ".tox",
        "dist",
        "build",
        "*.egg-info",
        ".eggs",
    }

    def __init__(
        self,
        extensions: Tuple[str, ...] = (".py",),
        max_file_size_kb: int = 100,
        chunk_size: int = 50,
        chunk_overlap: int = 10,
    ):
        self.extensions = extensions
        self.max_file_size_kb = max_file_size_kb
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        logger.info(
            "[FilesystemClient] Initialized — "
            "extensions=%s max_file_size_kb=%d "
            "chunk_size=%d chunk_overlap=%d",
            extensions, max_file_size_kb,
            chunk_size, chunk_overlap,
        )

    # ── File Discovery ────────────────────────────────────────────────────────

    def discover_files(self, repo_path: str) -> List[Tuple[str, str]]:
        """
        Walk the repository directory and return indexable files.

        Applies all filters: extension, size, hidden directories, empty files.
        Returns a list of (relative_path, absolute_path) tuples.

        Parameters
        ----------
        repo_path : str
            Absolute path to the root of the cloned repository.

        Returns
        -------
        List[Tuple[str, str]]
            List of (relative_path, absolute_path) tuples for files to index.
            relative_path is relative to repo_path root (e.g. "app/utils.py").
            absolute_path is the full filesystem path.

        Examples
        --------
            files = client.discover_files("/tmp/my_repo")
            # [("app/utils.py", "/tmp/my_repo/app/utils.py"), ...]
        """
        repo_root = Path(repo_path)
        files: List[Tuple[str, str]] = []
        skipped_hidden = 0
        skipped_size = 0
        skipped_empty = 0

        if not repo_root.exists():
            logger.error(
                "[FilesystemClient] discover_files: repo_path does not exist — %s",
                repo_path,
            )
            return []

        for abs_path in repo_root.rglob("*"):
            if not abs_path.is_file():
                continue

            # ── Skip hidden/build directories ─────────────────────────────────
            parts = abs_path.relative_to(repo_root).parts
            should_skip = False
            for part in parts[:-1]:  # check directory parts only
                if part.startswith(".") or part in self.SKIP_DIRS:
                    should_skip = True
                    break
            if should_skip:
                skipped_hidden += 1
                continue

            # ── Filter by extension ───────────────────────────────────────────
            if abs_path.suffix not in self.extensions:
                continue

            # ── Filter by size ────────────────────────────────────────────────
            try:
                size_kb = abs_path.stat().st_size / 1024
            except OSError:
                continue

            if size_kb > self.max_file_size_kb:
                logger.debug(
                    "[FilesystemClient] discover_files: skipping large file — "
                    "%s (%.1f KB > %d KB limit)",
                    abs_path.relative_to(repo_root),
                    size_kb,
                    self.max_file_size_kb,
                )
                skipped_size += 1
                continue

            # ── Skip empty files ──────────────────────────────────────────────
            if abs_path.stat().st_size == 0:
                skipped_empty += 1
                continue

            relative_path = str(abs_path.relative_to(repo_root))
            files.append((relative_path, str(abs_path)))

        logger.info(
            "[FilesystemClient] discover_files: found %d file(s) — "
            "skipped hidden=%d oversized=%d empty=%d | path=%s",
            len(files),
            skipped_hidden,
            skipped_size,
            skipped_empty,
            repo_path,
        )
        return files

    # ── File Reading ──────────────────────────────────────────────────────────

    def read_file(self, abs_path: str) -> Optional[str]:
        """
        Read a file's content as a string.

        Uses UTF-8 encoding with errors="ignore" to handle files with
        non-UTF-8 characters gracefully — important for repos with mixed
        encodings or binary-adjacent files.

        Parameters
        ----------
        abs_path : str
            Absolute filesystem path to the file.

        Returns
        -------
        str or None
            File content as string, or None if the file cannot be read
            or is empty after stripping whitespace.
        """
        try:
            content = Path(abs_path).read_text(encoding="utf-8", errors="ignore")

            if not content.strip():
                logger.debug(
                    "[FilesystemClient] read_file: empty file skipped — %s",
                    abs_path,
                )
                return None

            logger.debug(
                "[FilesystemClient] read_file: read %d chars — %s",
                len(content), abs_path,
            )
            return content

        except OSError as e:
            logger.warning(
                "[FilesystemClient] read_file: OS error reading file — "
                "%s error=%s",
                abs_path, str(e),
            )
            return None

        except Exception as e:
            logger.warning(
                "[FilesystemClient] read_file: unexpected error — "
                "%s error=%s",
                abs_path, str(e),
            )
            return None

    # ── File Chunking ─────────────────────────────────────────────────────────

    def chunk_file(
        self,
        relative_path: str,
        content: str,
        owner: str,
        repo: str,
    ) -> List[Dict]:
        """
        Split a file's content into overlapping line-based chunks for embedding.

        Each chunk is returned as a dict with:
            - id       : deterministic ChromaDB document ID
            - text     : chunk content (lines joined with newline)
            - metadata : source info for ChromaDB filtering and display

        Document ID Format
        ------------------
        "{owner}_{repo}_{safe_filepath}_{chunk_index}"

        The ID is deterministic — re-indexing the same file produces the
        same IDs, so ChromaDB upsert updates existing chunks rather than
        creating duplicates.

        Parameters
        ----------
        relative_path : str
            File path relative to repo root (e.g. "app/utils.py").
            Used in document ID and metadata.
        content : str
            Full file content as a string.
        owner : str
            GitHub repository owner — used in document ID and metadata.
        repo : str
            Repository name — used in document ID and metadata.

        Returns
        -------
        List[Dict]
            List of chunk dicts, each containing:
                {
                    "id":       str,   # unique ChromaDB document ID
                    "text":     str,   # chunk content
                    "metadata": dict   # source, filepath, line range, etc.
                }
            Empty list if content produces no non-empty chunks.

        Examples
        --------
            chunks = client.chunk_file("app/utils.py", content, "owner", "repo")
            # [{"id": "owner_repo_app_utils.py_0", "text": "...", "metadata": {...}}, ...]
        """
        lines = content.splitlines()
        chunks: List[Dict] = []
        chunk_index = 0

        # Sanitize filepath for use in document IDs
        safe_path = (
            relative_path
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
            .replace(".", "_")
        )

        i = 0
        while i < len(lines):
            chunk_lines = lines[i: i + self.chunk_size]
            chunk_text = "\n".join(chunk_lines)

            if chunk_text.strip():
                doc_id = f"{owner}_{repo}_{safe_path}_{chunk_index}"
                start_line = i + 1
                end_line = i + len(chunk_lines)

                chunks.append({
                    "id":   doc_id,
                    "text": chunk_text,
                    "metadata": {
                        "source":      f"{owner}/{repo}",
                        "filepath":    relative_path,
                        "repo":        f"{owner}/{repo}",
                        "type":        "code_chunk",
                        "chunk_index": str(chunk_index),
                        "start_line":  str(start_line),
                        "end_line":    str(end_line),
                        "extension":   Path(relative_path).suffix,
                    },
                })
                chunk_index += 1

            # Advance with overlap
            i += max(1, self.chunk_size - self.chunk_overlap)

        logger.debug(
            "[FilesystemClient] chunk_file: %s → %d chunk(s) "
            "from %d lines",
            relative_path, len(chunks), len(lines),
        )
        return chunks

    # ── File Metadata ─────────────────────────────────────────────────────────

    def get_file_metadata(
        self,
        abs_path: str,
        relative_path: str,
        owner: str,
        repo: str,
    ) -> Dict:
        """
        Return filesystem metadata for a file as a ChromaDB-compatible dict.

        Used as the metadata field when upserting chunks into ChromaDB.
        Enables future filtering by file type, size, recency, or repo.

        All values are strings — ChromaDB metadata values must be
        str, int, float, or bool. datetime is converted to ISO format string.

        Parameters
        ----------
        abs_path : str
            Absolute filesystem path to the file.
        relative_path : str
            Path relative to repo root (e.g. "app/utils.py").
        owner : str
            GitHub repository owner.
        repo : str
            Repository name.

        Returns
        -------
        Dict
            Metadata dict with all values as ChromaDB-compatible types:
            {
                "source":        "{owner}/{repo}",
                "filepath":      relative_path,
                "filename":      "utils.py",
                "extension":     ".py",
                "size_bytes":    "4096",
                "size_kb":       "4.0",
                "repo":          "{owner}/{repo}",
                "type":          "source_file",
                "last_modified": "2026-03-21T12:00:00",
            }

        Notes
        -----
        Returns a minimal dict with available values if stat() fails.
        Never raises — missing metadata is better than a crashed pipeline.
        """
        path = Path(abs_path)
        base_meta = {
            "source":    f"{owner}/{repo}",
            "filepath":  relative_path,
            "filename":  path.name,
            "extension": path.suffix,
            "repo":      f"{owner}/{repo}",
            "type":      "source_file",
        }

        try:
            stat = path.stat()
            size_bytes = stat.st_size
            size_kb = size_bytes / 1024
            last_modified = datetime.fromtimestamp(
                stat.st_mtime
            ).isoformat(timespec="seconds")

            base_meta.update({
                "size_bytes":    str(size_bytes),
                "size_kb":       f"{size_kb:.1f}",
                "last_modified": last_modified,
            })

        except OSError as e:
            logger.warning(
                "[FilesystemClient] get_file_metadata: "
                "stat() failed for %s — error=%s",
                abs_path, str(e),
            )

        return base_meta

    # ── Utility ───────────────────────────────────────────────────────────────

    def count_files(self, repo_path: str) -> Dict[str, int]:
        """
        Count files by extension in a repository without reading content.

        Useful for estimating indexing cost before starting the pipeline.
        Returns a dict mapping extension to count, plus a "total" key.

        Parameters
        ----------
        repo_path : str
            Absolute path to the repository root.

        Returns
        -------
        Dict[str, int]
            {".py": 42, ".js": 18, "total": 60}
        """
        counts: Dict[str, int] = {}
        repo_root = Path(repo_path)

        if not repo_root.exists():
            return {"total": 0}

        for abs_path in repo_root.rglob("*"):
            if not abs_path.is_file():
                continue

            parts = abs_path.relative_to(repo_root).parts
            if any(p.startswith(".") or p in self.SKIP_DIRS for p in parts[:-1]):
                continue

            ext = abs_path.suffix
            counts[ext] = counts.get(ext, 0) + 1

        total = sum(v for k, v in counts.items() if k in self.extensions)
        counts["total_indexable"] = total
        counts["total_all"] = sum(counts.values())

        logger.info(
            "[FilesystemClient] count_files: %s — indexable=%d total=%d",
            repo_path, total, counts["total_all"],
        )
        return counts