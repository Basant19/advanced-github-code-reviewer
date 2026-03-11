"""
app/mcp/filesystem_client.py
=============================
MCP client for local filesystem operations — repo indexing for ChromaDB.

STATUS: STUB — planned for P4 (Corrective RAG + Long-Term Memory)

RESPONSIBILITY (P4)
-------------------
This file will handle reading files from a cloned GitHub repository
and preparing them for embedding into ChromaDB. It is the bridge between
a repository on disk and the vector store indexing pipeline.

It will know NOTHING about:
  - ChromaDB directly       (that is repository_service.py's job)
  - LangGraph state         (that is nodes.py's job)
  - GitHub API              (that is github_client.py's job)

CALL CHAIN (P4)
---------------
  repository_service.index_repository()
    → github_client.clone_or_fetch_repo()   fetch files from GitHub
      → filesystem_client.read_files()      read + chunk file contents
        → repository_service               embed chunks into ChromaDB

PLANNED METHODS (P4)
---------------------
  read_files(repo_path, extensions) → dict[str, str]
      Walk a local repo directory. Return {relative_path: content}
      for all files matching the given extensions (default: .py only).

  chunk_file(content, chunk_size, overlap) → list[str]
      Split a large file into overlapping chunks for embedding.
      Large files (>500 lines) cannot be embedded as a single vector —
      chunking ensures semantic coverage across the full file.

  get_file_metadata(file_path) → dict
      Return filename, extension, size, last_modified for ChromaDB metadata.
      Used by the Corrective RAG grader to filter by file type or recency.

WHY THIS EXISTS AS A SEPARATE CLIENT
--------------------------------------
repository_service.py orchestrates the indexing pipeline but should not
contain raw filesystem I/O. Separating filesystem access into this MCP
client means:
  1. The indexing pipeline is testable — mock this client in tests.
  2. The filesystem access strategy can change (local clone vs GitHub API
     streaming) without touching repository_service.py.
  3. Consistent with the MCP pattern used by github_client.py and
     sandbox_client.py — every external resource has its own client.
"""

from __future__ import annotations

from app.core.logger import get_logger

logger = get_logger(__name__)


class FilesystemClient:
    """
    Stub — will be implemented in P4.

    P4 implementation will read repository files from a local clone
    and prepare them for ChromaDB embedding via repository_service.py.
    """

    def read_files(
        self,
        repo_path: str,
        extensions: tuple[str, ...] = (".py",),
    ) -> dict[str, str]:
        """
        P4: Walk repo_path and return {relative_path: content} for all
        files matching the given extensions.

        Not implemented yet — raises NotImplementedError if called.
        """
        raise NotImplementedError(
            "FilesystemClient.read_files() is planned for P4. "
            "It will read and return repo files for ChromaDB indexing."
        )

    def chunk_file(
        self,
        content: str,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> list[str]:
        """
        P4: Split file content into overlapping chunks for embedding.
        Large files need chunking — a 2000-line file cannot be a single vector.

        Not implemented yet — raises NotImplementedError if called.
        """
        raise NotImplementedError(
            "FilesystemClient.chunk_file() is planned for P4. "
            "It will split large files into overlapping chunks for embedding."
        )

    def get_file_metadata(self, file_path: str) -> dict:
        """
        P4: Return filename, extension, size, last_modified for a file.
        Used as ChromaDB document metadata for Corrective RAG filtering.

        Not implemented yet — raises NotImplementedError if called.
        """
        raise NotImplementedError(
            "FilesystemClient.get_file_metadata() is planned for P4."
        )