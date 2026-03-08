"""
app/mcp/github_client.py

GitHub API Client
------------------
Uses PyGithub (uv add PyGithub) to interact with the GitHub REST API.

Methods the agent needs:
    - get_pr_metadata     → PR title, author, description
    - get_pr_files        → changed files with patches
    - get_pr_diff         → single unified diff string for the LLM
    - post_review_comment → post AI verdict back to the PR

Note on config attribute name:
    config.py defines:  github_token: str = Field(..., alias="GITHUB_TOKEN")
    So the correct access is:  settings.github_token  (lowercase)
    NOT:                       settings.GITHUB_TOKEN
"""

import sys
from github import Github, GithubException

from app.core.config import settings
from app.core.exceptions import CustomException
from app.core.logger import get_logger

logger = get_logger(__name__)


class GitHubClient:

    def __init__(self):
        self.client = Github(settings.github_token)   # ← lowercase, matches config.py
        logger.info("GitHubClient initialized")

    # ── internal helpers ───────────────────────────────────────────────────

    def _get_repo(self, owner: str, repo: str):
        """Return a PyGithub Repository or raise CustomException."""
        try:
            return self.client.get_repo(f"{owner}/{repo}")
        except GithubException as e:
            logger.error(f"Repository not found: {owner}/{repo} — {e.status}")
            raise CustomException(
                f"Repository {owner}/{repo} not found: {e.data}", sys
            )

    def _get_pr(self, owner: str, repo: str, pr_number: int):
        """Return a PyGithub PullRequest or raise CustomException."""
        repo_obj = self._get_repo(owner, repo)
        try:
            return repo_obj.get_pull(pr_number)
        except GithubException as e:
            logger.error(f"PR #{pr_number} not found in {owner}/{repo} — {e.status}")
            raise CustomException(
                f"PR #{pr_number} not found in {owner}/{repo}: {e.data}", sys
            )

    # ── public methods ─────────────────────────────────────────────────────

    def get_pr_metadata(self, owner: str, repo: str, pr_number: int) -> dict:
        """
        Returns basic PR information the agent uses for context.

        Returns:
            {
                "number":      int,
                "title":       str,
                "author":      str,
                "description": str,
                "base_branch": str,
                "head_branch": str,
                "state":       str,   # "open" | "closed" | "merged"
            }
        """
        logger.info(f"Fetching PR metadata: {owner}/{repo}#{pr_number}")

        pr = self._get_pr(owner, repo, pr_number)
        metadata = {
            "number":      pr.number,
            "title":       pr.title,
            "author":      pr.user.login,
            "description": pr.body or "",
            "base_branch": pr.base.ref,
            "head_branch": pr.head.ref,
            "state":       "merged" if pr.merged else pr.state,
        }

        logger.info(
            f"PR metadata fetched — title: '{pr.title}', author: {pr.user.login}"
        )
        return metadata

    def get_pr_files(self, owner: str, repo: str, pr_number: int) -> list[dict]:
        """
        Returns the list of files changed in the PR.

        Returns:
            [
                {
                    "filename": str,
                    "status":   str,   # "added" | "modified" | "removed"
                    "changes":  int,
                    "patch":    str,   # unified diff for this file
                },
                ...
            ]
        """
        logger.info(f"Fetching PR files: {owner}/{repo}#{pr_number}")

        pr = self._get_pr(owner, repo, pr_number)
        result = [
            {
                "filename": f.filename,
                "status":   f.status,
                "changes":  f.changes,
                "patch":    f.patch or "",    # binary files have no patch
            }
            for f in pr.get_files()
        ]

        logger.info(f"Fetched {len(result)} changed file(s) for PR #{pr_number}")
        return result

    def get_pr_diff(self, owner: str, repo: str, pr_number: int) -> str:
        """
        Builds a single unified diff string from all changed files.
        This is passed directly to the LLM for code analysis.

        Returns:
            str — concatenated patches for every changed file
        """
        logger.info(f"Building unified diff: {owner}/{repo}#{pr_number}")

        files = self.get_pr_files(owner, repo, pr_number)
        diff_parts = [
            f"--- {f['filename']} ({f['status']})\n{f['patch']}"
            for f in files
            if f["patch"]
        ]

        diff = "\n\n".join(diff_parts)
        logger.info(
            f"Diff built — {len(diff_parts)} file(s) with patches, "
            f"{len(diff)} chars total"
        )
        return diff

    def post_review_comment(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        body: str,
    ) -> None:
        """
        Posts the AI review as a comment on the GitHub PR.

        Args:
            body: Formatted review text (verdict + issues + suggestions).
        """
        logger.info(f"Posting review comment to {owner}/{repo}#{pr_number}")

        pr = self._get_pr(owner, repo, pr_number)
        try:
            pr.create_issue_comment(body)
            logger.info(f"Review comment posted successfully to PR #{pr_number}")
        except GithubException as e:
            logger.error(f"Failed to post comment on PR #{pr_number} — {e.status}")
            raise CustomException(
                f"Failed to post review comment on PR #{pr_number}: {e.data}", sys
            )