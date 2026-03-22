"""
app/api/routes/repos.py

Repository Indexing API Routes — P4
-------------------------------------
Endpoints for triggering and monitoring ChromaDB indexing of GitHub
repositories. Indexed repositories provide codebase context to the
RAG review pipeline in retrieve_context_node and grade_context_node.

Endpoints
---------
POST   /repos/index
    Trigger full indexing pipeline for a GitHub repository.
    Clones the repo, chunks all .py files, embeds with gemini-embedding-001,
    stores in ChromaDB. Returns IndexingResult with counts and duration.

GET    /repos/{owner}/{repo}/status
    Check whether a repository has been indexed in ChromaDB.
    Returns chunk count and collection total — useful for dashboard display.

DELETE /repos/{owner}/{repo}/index
    Remove all indexed chunks for a repository from ChromaDB.
    Used to force a clean re-index after major code changes.

Usage Flow
----------
1. POST /repos/index  {"owner": "Basant19", "repo": "python_tuts"}
2. Wait for response (may take 30-120s depending on repo size)
3. GET  /repos/Basant19/python_tuts/status  → {"indexed": true, "chunk_count": 47}
4. Trigger reviews normally — retrieve_context_node will now find context

Rate Limiting Note
------------------
The indexing pipeline calls gemini-embedding-001 in batches with a 1-second
sleep between batches. Large repositories (500+ files) may take several minutes.
The POST /repos/index endpoint is synchronous — it blocks until complete.
For very large repos, consider running indexing as a background task in P6.

Architecture
------------
Routes → indexing_service.index_repository() → FilesystemClient + ChromaDB
All business logic lives in indexing_service.py — routes handle HTTP only.
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Path
from fastapi import status as http_status
from pydantic import BaseModel, Field, field_validator

from app.core.exceptions import CustomException
from app.core.logger import get_logger
from app.services.indexing_service import (
    index_repository,
    get_index_status,
    clear_repository_index,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/repos", tags=["repos"])


# ── Request / Response Schemas ────────────────────────────────────────────────

class IndexRepoRequest(BaseModel):
    """
    Request body for POST /repos/index.

    owner and repo identify the GitHub repository to index.
    github_token is optional — required only for private repositories.
    Public repositories (like python_tuts) do not require a token.
    """
    owner: str = Field(
        ...,
        min_length=1,
        description="GitHub username or organization name",
        examples=["Basant19"],
    )
    repo: str = Field(
        ...,
        min_length=1,
        description="Repository name without owner prefix",
        examples=["python_tuts"],
    )
    github_token: str = Field(
        default="",
        description=(
            "GitHub Personal Access Token for private repositories. "
            "Leave empty for public repos — GITHUB_TOKEN env var is used "
            "if set."
        ),
    )

    @field_validator("owner", "repo")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        """Strip whitespace to prevent silent GitHub 404 errors."""
        return v.strip()


class IndexRepoResponse(BaseModel):
    """
    Response for POST /repos/index.
    Maps directly from IndexingResult.to_dict().
    """
    owner:            str
    repo:             str
    success:          bool
    total_files:      int
    indexed_files:    int
    skipped_files:    int
    total_chunks:     int
    duration_seconds: float
    error: Optional[str] = ""
    message:          str = ""


class IndexStatusResponse(BaseModel):
    """
    Response for GET /repos/{owner}/{repo}/status.
    """
    owner:            str
    repo:             str
    indexed:          bool
    chunk_count:      int
    collection_total: int
    message:          str = ""


class ClearIndexResponse(BaseModel):
    """
    Response for DELETE /repos/{owner}/{repo}/index.
    """
    owner:         str
    repo:          str
    deleted_count: int
    success:       bool
    message:       str = ""


# ── Helper ────────────────────────────────────────────────────────────────────

def _handle_service_error(e: Exception, context: str) -> None:
    """
    Map service exceptions to appropriate HTTP responses.

    Parameters
    ----------
    e : Exception
        Exception from service layer.
    context : str
        Calling function name for log context.

    Raises
    ------
    HTTPException
        Always raises — maps CustomException to 400/404/500.
    """
    logger.error(
        "[repos_route] Error in %s — type=%s message=%s",
        context, type(e).__name__, str(e),
        exc_info=True,
    )

    if isinstance(e, CustomException):
        msg = str(e).lower()
        if "not found" in msg:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=str(e),
            )
        if "invalid" in msg or "empty" in msg:
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

    raise HTTPException(
        status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="An internal server error occurred during repository indexing.",
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "/index",
    response_model=IndexRepoResponse,
    status_code=http_status.HTTP_200_OK,
    summary="Index a GitHub repository into ChromaDB",
    description=(
        "Clones the repository, chunks all Python files, embeds them with "
        "gemini-embedding-001, and stores them in ChromaDB.\n\n"
        "After indexing, PR reviews for this repository will include relevant "
        "codebase context via the RAG pipeline (retrieve_context_node + "
        "grade_context_node).\n\n"
        "**Note:** This is a synchronous operation. Large repositories may "
        "take 30–120 seconds. Re-indexing the same repo is safe — existing "
        "chunks are updated via ChromaDB upsert semantics."
    ),
)
async def index_repo(
    request: IndexRepoRequest,
) -> IndexRepoResponse:
    """
    Trigger the full clone → chunk → embed → store pipeline.

    Returns an IndexRepoResponse with counts of files indexed, chunks stored,
    and total duration. success=False indicates a fatal error during indexing.

    Call GET /repos/{owner}/{repo}/status after this to verify the result.
    """
    logger.info(
        "[repos_route] Index request — %s/%s",
        request.owner, request.repo,
    )

    try:
        result = await index_repository(
            owner=request.owner,
            repo=request.repo,
            github_token=request.github_token or None,
        )

        logger.info(
            "[repos_route] Indexing complete — %s/%s "
            "success=%s files=%d/%d chunks=%d duration=%.1fs",
            request.owner, request.repo,
            result.success,
            result.indexed_files, result.total_files,
            result.total_chunks,
            result.duration_seconds,
        )

        message = (
            f"Indexed {result.indexed_files}/{result.total_files} file(s) "
            f"into {result.total_chunks} chunk(s) in {result.duration_seconds:.1f}s"
            if result.success
            else f"Indexing failed: {result.error}"
        )

        return IndexRepoResponse(
            **result.to_dict(),
            message=message,
        )

    except Exception as e:
        _handle_service_error(e, "index_repo")


@router.get(
    "/{owner}/{repo}/status",
    response_model=IndexStatusResponse,
    summary="Check repository indexing status",
    description=(
        "Returns whether the repository has been indexed in ChromaDB "
        "and how many chunks are stored. "
        "Use this to verify indexing completed successfully before "
        "triggering reviews."
    ),
)
async def get_repo_status(
    owner: str = Path(..., description="GitHub username or organization"),
    repo: str = Path(..., description="Repository name"),
) -> IndexStatusResponse:
    """
    Check whether a repository is indexed in ChromaDB.

    Returns chunk_count > 0 if indexed. chunk_count == 0 means either
    the repo has not been indexed or indexing produced no chunks.
    """
    logger.info(
        "[repos_route] Status request — %s/%s",
        owner, repo,
    )

    try:
        status = await get_index_status(owner, repo)

        logger.info(
            "[repos_route] Status fetched — %s/%s indexed=%s chunks=%d",
            owner, repo,
            status.get("indexed"),
            status.get("chunk_count", 0),
        )

        message = (
            f"Repository indexed with {status.get('chunk_count', 0)} chunk(s)"
            if status.get("indexed")
            else "Repository not yet indexed — call POST /repos/index first"
        )

        return IndexStatusResponse(
            owner=owner,
            repo=repo,
            indexed=status.get("indexed", False),
            chunk_count=status.get("chunk_count", 0),
            collection_total=status.get("collection_total", 0),
            message=message,
        )

    except Exception as e:
        _handle_service_error(e, "get_repo_status")


@router.delete(
    "/{owner}/{repo}/index",
    response_model=ClearIndexResponse,
    summary="Remove all indexed chunks for a repository",
    description=(
        "Deletes all ChromaDB documents where metadata.source == "
        "'{owner}/{repo}'. "
        "Use this to force a clean re-index after major code changes. "
        "After deleting, call POST /repos/index to re-index."
    ),
)
async def delete_repo_index(
    owner: str = Path(..., description="GitHub username or organization"),
    repo: str = Path(..., description="Repository name"),
) -> ClearIndexResponse:
    """
    Remove all indexed chunks for a repository from ChromaDB.

    Safe to call even if the repository has not been indexed — returns
    deleted_count=0 with success=True in that case.
    """
    logger.info(
        "[repos_route] Clear index request — %s/%s",
        owner, repo,
    )

    try:
        result = await clear_repository_index(owner, repo)

        logger.info(
            "[repos_route] Index cleared — %s/%s deleted=%d success=%s",
            owner, repo,
            result.get("deleted_count", 0),
            result.get("success"),
        )

        deleted = result.get("deleted_count", 0)
        message = (
            f"Deleted {deleted} chunk(s) for {owner}/{repo}"
            if deleted > 0
            else f"No indexed chunks found for {owner}/{repo}"
        )

        return ClearIndexResponse(
            owner=owner,
            repo=repo,
            deleted_count=deleted,
            success=result.get("success", False),
            message=message,
        )

    except Exception as e:
        _handle_service_error(e, "delete_repo_index")