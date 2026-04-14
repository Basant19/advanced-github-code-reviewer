"""
app/mcp/github_client.py

Production-Ready GitHub Client
------------------------------

Features:
✔ Safe GitHub API access via PyGithub
✔ Strong error handling with CustomException
✔ Structured logging with context
✔ Defensive handling of missing/large patches
✔ No silent failures
✔ Clear method contracts

Used by:
- nodes.py (fetch_diff_node)
- review_service.py (posting comments)

ENV REQUIREMENT:
    GITHUB_TOKEN must be set (via config settings.github_token)
"""

import sys
from typing import List, Dict, Optional
import httpx
from github import Github, GithubException
import requests
from app.core.config import settings
from app.core.exceptions import CustomException
from app.core.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

MAX_DIFF_SIZE = 20000  # prevent extremely large LLM input


# ─────────────────────────────────────────────
# CLIENT
# ─────────────────────────────────────────────
class GitHubClient:
    def __init__(self) -> None:
        try:
            if not settings.github_token:
                raise ValueError("Missing GitHub token")
            
            self._token = settings.github_token
            self.client = Github(self._token)
            # We'll use a persistent async client for better performance
            self.async_client = httpx.AsyncClient(
                headers={
                    "Authorization": f"token {self._token}",
                    "Accept": "application/vnd.github.raw",
                },
                timeout=10.0
            )

            logger.info("[GitHubClient] Initialized successfully")
        except Exception as e:
            logger.exception("[GitHubClient] Initialization failed")
            raise CustomException("GitHub client initialization failed", sys) from e


    # ─────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────

    def _get_repo(self, owner: str, repo: str):
        """
        Fetch repository safely.

        Raises:
            CustomException if repo not found / API error
        """
        try:
            full_name = f"{owner}/{repo}"

            logger.debug(
                "[GitHub] Fetching repository",
                extra={"repo": full_name},
            )

            return self.client.get_repo(full_name)

        except GithubException as e:
            logger.error(
                "[GitHub] Repository fetch failed",
                extra={"repo": f"{owner}/{repo}", "status": e.status},
                exc_info=True,
            )
            raise CustomException(
                f"Repository {owner}/{repo} not found", sys
            ) from e

        except Exception as e:
            logger.exception("[GitHub] Unexpected repo error")
            raise CustomException("Unexpected GitHub error", sys) from e

    def _get_pr(self, owner: str, repo: str, pr_number: int):
        """
        Fetch pull request safely.
        """
        repo_obj = self._get_repo(owner, repo)

        try:
            logger.debug(
                "[GitHub] Fetching PR",
                extra={"repo": f"{owner}/{repo}", "pr": pr_number},
            )

            return repo_obj.get_pull(pr_number)

        except GithubException as e:
            logger.error(
                "[GitHub] PR fetch failed",
                extra={
                    "repo": f"{owner}/{repo}",
                    "pr": pr_number,
                    "status": e.status,
                },
                exc_info=True,
            )
            raise CustomException(
                f"PR #{pr_number} not found in {owner}/{repo}", sys
            ) from e

        except Exception as e:
            logger.exception("[GitHub] Unexpected PR error")
            raise CustomException("Unexpected GitHub error", sys) from e

    # ─────────────────────────────────────────
    # PUBLIC METHODS
    # ─────────────────────────────────────────

    def get_pr_metadata(self, owner: str, repo: str, pr_number: int) -> Dict:
            """
            Fetch PR metadata for context.
            """
            logger.info(
                "[GitHub] Fetching PR metadata",
                extra={"repo": f"{owner}/{repo}", "pr": pr_number},
            )

            try:
                # This is the object we use
                pr = self._get_pr(owner, repo, pr_number)

                metadata = {
                    "number": pr.number,
                    "title": pr.title or "",
                    "author": getattr(pr.user, "login", "unknown"),
                    "description": pr.body or "",
                    "base_branch": pr.base.ref,
                    "head_branch": pr.head.ref,
                    # Accessing SHAs via the PyGithub 'pr' object:
                    "base_sha": pr.base.sha,  
                    "head_sha": pr.head.sha,
                    "state": "merged" if pr.merged else pr.state,
                }

                logger.debug(
                    "[GitHub] Metadata fetched",
                    extra={"title": metadata["title"]},
                )

                return metadata

            except Exception as e:
                logger.exception("[GitHub] Metadata fetch failed")
                raise CustomException("Failed to fetch PR metadata", sys) from e

    # ─────────────────────────────────────────

    def get_pr_files(self, owner: str, repo: str, pr_number: int) -> List[Dict]:
        """
        Fetch changed files in PR.

        Returns:
            List[dict]
        """
        logger.info(
            "[GitHub] Fetching PR files",
            extra={"repo": f"{owner}/{repo}", "pr": pr_number},
        )

        try:
            pr = self._get_pr(owner, repo, pr_number)

            files = []
            for f in pr.get_files():
                files.append({
                    "filename": f.filename,
                    "status": f.status,
                    "changes": f.changes,
                    "patch": f.patch or "",
                })

            logger.info(
                "[GitHub] Files fetched",
                extra={"count": len(files)},
            )

            return files

        except Exception as e:
            logger.exception("[GitHub] File fetch failed")
            raise CustomException("Failed to fetch PR files", sys) from e

    # ─────────────────────────────────────────

    def get_pr_diff(self, owner: str, repo: str, pr_number: int) -> str:
        """
        Build unified diff string for LLM.

        Safety:
        - Skips empty patches
        - Truncates large diffs
        """
        logger.info(
            "[GitHub] Building PR diff",
            extra={"repo": f"{owner}/{repo}", "pr": pr_number},
        )

        try:
            files = self.get_pr_files(owner, repo, pr_number)

            diff_parts = []

            for f in files:
                if not f["patch"]:
                    continue

                diff_parts.append(
                    f"--- {f['filename']} ({f['status']})\n{f['patch']}"
                )

            diff = "\n\n".join(diff_parts)

            if not diff:
                logger.warning("[GitHub] Empty diff generated")

            # 🔥 SIZE CONTROL
            if len(diff) > MAX_DIFF_SIZE:
                logger.warning(
                    "[GitHub] Diff too large → truncating",
                    extra={"original_size": len(diff)},
                )
                diff = diff[:MAX_DIFF_SIZE]

            logger.info(
                "[GitHub] Diff ready",
                extra={"size": len(diff)},
            )

            return diff

        except Exception as e:
            logger.exception("[GitHub] Diff build failed")
            raise CustomException("Failed to build PR diff", sys) from e
    #----------------------------------------------------------------------------

    async def get_file_content(self, url: str) -> Optional[str]:
            """
            Fetch raw file content asynchronously. 
            Crucial for not blocking the FastAPI event loop during graph execution.
            """
            try:
                response = await self.async_client.get(url)

                if response.status_code == 200:
                    logger.info("[GitHubClient] File content fetched — %d chars", len(response.text))
                    return response.text

                logger.warning(
                    "[GitHubClient] Content fetch failed — status=%d url=%s",
                    response.status_code, url[:80],
                )
                return None

            except Exception as e:
                logger.warning("[GitHubClient] get_file_content error — %s", str(e))
                return None

    # ─────────────────────────────────────────

    def post_review_comment(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        body: str,
    ) -> None:
        """
        Post review comment to PR.

        Raises:
            CustomException on failure
        """
        logger.info(
            "[GitHub] Posting review comment",
            extra={"repo": f"{owner}/{repo}", "pr": pr_number},
        )

        try:
            pr = self._get_pr(owner, repo, pr_number)

            if not body or not body.strip():
                logger.warning("[GitHub] Empty comment body — skipping")
                return

            pr.create_issue_comment(body)

            logger.info(
                "[GitHub] Comment posted successfully",
                extra={"pr": pr_number},
            )

        except GithubException as e:
            logger.error(
                "[GitHub] Comment failed",
                extra={"status": e.status},
                exc_info=True,
            )
            raise CustomException(
                f"Failed to post review comment on PR #{pr_number}", sys
            ) from e

        except Exception as e:
            logger.exception("[GitHub] Unexpected comment error")
            raise CustomException("Unexpected GitHub error", sys) from e


    async def get_base_file_content(
        self,
        owner: str,
        repo: str,
        filepath: str,
        base_sha: str,
    ) -> Optional[str]:
        """
        Fetch a file's content asynchronously using the existing httpx async_client.
        """
        try:
            url = f"https://api.github.com/repos/{owner}/{repo}/contents/{filepath}"
            params = {"ref": base_sha}

            # Use the persistent async_client instead of requests
            response = await self.async_client.get(url, params=params)

            if response.status_code == 200:
                logger.info(
                    "[GitHubClient] Base file fetched — %s@%s (%d chars)",
                    filepath, base_sha[:8], len(response.text),
                )
                return response.text

            logger.warning(
                "[GitHubClient] Base file fetch failed — %s status=%d",
                filepath, response.status_code,
            )
            return None

        except Exception as e:
            logger.warning(
                "[GitHubClient] get_base_file_content error — %s: %s",
                filepath, str(e),
            )
            return None