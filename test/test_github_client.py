"""
tests/test_github_client.py

Unit Tests for GitHubClient
-----------------------------
Tests all 4 public methods using mocks — no real GitHub API calls made.

Run with:
    pytest tests/test_github_client.py -v

What is mocked:
    - Github()             → the PyGithub client
    - client.get_repo()    → returns a fake repo object
    - repo.get_pull()      → returns a fake PR object
    - pr.get_files()       → returns fake changed files
    - pr.create_issue_comment() → simulates posting a comment
"""

import sys
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

# ── make sure project root is on sys.path ────────────────────────────────────
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.mcp.github_client import GitHubClient
from app.core.exceptions import CustomException
from github import GithubException


# ──────────────────────────────────────────────────────────────────────────────
# Shared test fixtures
# ──────────────────────────────────────────────────────────────────────────────

def make_fake_pr():
    """Build a realistic fake PyGithub PullRequest object."""
    pr = MagicMock()
    pr.number        = 42
    pr.title         = "Add review workflow"
    pr.body          = "This PR adds the LangGraph review workflow."
    pr.state         = "open"
    pr.merged        = False

    pr.user          = MagicMock()
    pr.user.login    = "dev_user"

    pr.base          = MagicMock()
    pr.base.ref      = "main"

    pr.head          = MagicMock()
    pr.head.ref      = "feature/review-workflow"

    return pr


def make_fake_files():
    """Build a list of fake PyGithub File objects."""
    file1        = MagicMock()
    file1.filename = "app/graph/nodes.py"
    file1.status   = "modified"
    file1.changes  = 15
    file1.patch    = "@@ -1,3 +1,5 @@\n+import sys\n+from app.core.logger import get_logger"

    file2        = MagicMock()
    file2.filename = "app/graph/state.py"
    file2.status   = "added"
    file2.changes  = 8
    file2.patch    = "@@ -0,0 +1,8 @@\n+from typing import TypedDict"

    file3        = MagicMock()          # binary file — no patch
    file3.filename = "assets/logo.png"
    file3.status   = "added"
    file3.changes  = 0
    file3.patch    = None

    return [file1, file2, file3]


@pytest.fixture
def client_and_pr():
    """
    Returns a (GitHubClient, fake_pr) tuple with Github() fully mocked.
    The mock is patched at the github_client module level so settings
    are never touched.
    """
    fake_pr   = make_fake_pr()
    fake_repo = MagicMock()
    fake_repo.get_pull.return_value = fake_pr

    with patch("app.mcp.github_client.Github") as MockGithub:
        MockGithub.return_value.get_repo.return_value = fake_repo
        client = GitHubClient()
        yield client, fake_pr, fake_repo


# ──────────────────────────────────────────────────────────────────────────────
# Test: GitHubClient initialisation
# ──────────────────────────────────────────────────────────────────────────────

class TestGitHubClientInit:

    def test_client_initialises_successfully(self):
        with patch("app.mcp.github_client.Github") as MockGithub:
            client = GitHubClient()
            assert client.client is not None
            MockGithub.assert_called_once()   # Github() was called with the token

    def test_client_uses_github_token_from_settings(self):
        with patch("app.mcp.github_client.Github") as MockGithub:
            with patch("app.mcp.github_client.settings") as mock_settings:
                mock_settings.github_token = "test-token-abc"
                GitHubClient()
                MockGithub.assert_called_once_with("test-token-abc")


# ──────────────────────────────────────────────────────────────────────────────
# Test: get_pr_metadata
# ──────────────────────────────────────────────────────────────────────────────

class TestGetPrMetadata:

    def test_returns_correct_metadata(self, client_and_pr):
        client, fake_pr, _ = client_and_pr

        result = client.get_pr_metadata("owner", "repo", 42)

        assert result["number"]      == 42
        assert result["title"]       == "Add review workflow"
        assert result["author"]      == "dev_user"
        assert result["description"] == "This PR adds the LangGraph review workflow."
        assert result["base_branch"] == "main"
        assert result["head_branch"] == "feature/review-workflow"
        assert result["state"]       == "open"

    def test_state_is_merged_when_pr_is_merged(self, client_and_pr):
        client, fake_pr, _ = client_and_pr
        fake_pr.merged = True

        result = client.get_pr_metadata("owner", "repo", 42)

        assert result["state"] == "merged"

    def test_description_defaults_to_empty_string_when_body_is_none(self, client_and_pr):
        client, fake_pr, _ = client_and_pr
        fake_pr.body = None

        result = client.get_pr_metadata("owner", "repo", 42)

        assert result["description"] == ""

    def test_raises_custom_exception_on_github_error(self):
        with patch("app.mcp.github_client.Github") as MockGithub:
            MockGithub.return_value.get_repo.side_effect = GithubException(
                404, {"message": "Not Found"}, None
            )
            client = GitHubClient()

            with pytest.raises(CustomException):
                client.get_pr_metadata("bad_owner", "bad_repo", 1)


# ──────────────────────────────────────────────────────────────────────────────
# Test: get_pr_files
# ──────────────────────────────────────────────────────────────────────────────

class TestGetPrFiles:

    def test_returns_correct_number_of_files(self, client_and_pr):
        client, fake_pr, _ = client_and_pr
        fake_pr.get_files.return_value = make_fake_files()

        result = client.get_pr_files("owner", "repo", 42)

        assert len(result) == 3

    def test_file_fields_are_correct(self, client_and_pr):
        client, fake_pr, _ = client_and_pr
        fake_pr.get_files.return_value = make_fake_files()

        result = client.get_pr_files("owner", "repo", 42)
        first  = result[0]

        assert first["filename"] == "app/graph/nodes.py"
        assert first["status"]   == "modified"
        assert first["changes"]  == 15
        assert "@@ -1,3" in first["patch"]

    def test_binary_file_patch_is_empty_string(self, client_and_pr):
        client, fake_pr, _ = client_and_pr
        fake_pr.get_files.return_value = make_fake_files()

        result = client.get_pr_files("owner", "repo", 42)
        binary = result[2]   # logo.png

        assert binary["filename"] == "assets/logo.png"
        assert binary["patch"]    == ""

    def test_raises_custom_exception_on_pr_not_found(self):
        fake_repo = MagicMock()
        fake_repo.get_pull.side_effect = GithubException(
            404, {"message": "PR Not Found"}, None
        )
        with patch("app.mcp.github_client.Github") as MockGithub:
            MockGithub.return_value.get_repo.return_value = fake_repo
            client = GitHubClient()

            with pytest.raises(CustomException):
                client.get_pr_files("owner", "repo", 999)


# ──────────────────────────────────────────────────────────────────────────────
# Test: get_pr_diff
# ──────────────────────────────────────────────────────────────────────────────

class TestGetPrDiff:

    def test_diff_contains_all_files_with_patches(self, client_and_pr):
        client, fake_pr, _ = client_and_pr
        fake_pr.get_files.return_value = make_fake_files()

        diff = client.get_pr_diff("owner", "repo", 42)

        # Both files with patches should appear
        assert "app/graph/nodes.py" in diff
        assert "app/graph/state.py" in diff

    def test_diff_excludes_binary_files(self, client_and_pr):
        client, fake_pr, _ = client_and_pr
        fake_pr.get_files.return_value = make_fake_files()

        diff = client.get_pr_diff("owner", "repo", 42)

        # Binary file with no patch should be excluded
        assert "assets/logo.png" not in diff

    def test_diff_is_empty_string_when_no_patches(self, client_and_pr):
        client, fake_pr, _ = client_and_pr

        # All files are binary (no patches)
        binary = MagicMock()
        binary.filename = "image.png"
        binary.status   = "added"
        binary.changes  = 0
        binary.patch    = None
        fake_pr.get_files.return_value = [binary]

        diff = client.get_pr_diff("owner", "repo", 42)

        assert diff == ""

    def test_diff_format_includes_filename_and_status(self, client_and_pr):
        client, fake_pr, _ = client_and_pr
        fake_pr.get_files.return_value = make_fake_files()

        diff = client.get_pr_diff("owner", "repo", 42)

        # Check the header format: "--- filename (status)"
        assert "--- app/graph/nodes.py (modified)" in diff
        assert "--- app/graph/state.py (added)"    in diff


# ──────────────────────────────────────────────────────────────────────────────
# Test: post_review_comment
# ──────────────────────────────────────────────────────────────────────────────

class TestPostReviewComment:

    def test_posts_comment_successfully(self, client_and_pr):
        client, fake_pr, _ = client_and_pr

        # Should not raise
        client.post_review_comment("owner", "repo", 42, "LGTM — no issues found.")

        fake_pr.create_issue_comment.assert_called_once_with(
            "LGTM — no issues found."
        )

    def test_raises_custom_exception_when_post_fails(self, client_and_pr):
        client, fake_pr, _ = client_and_pr
        fake_pr.create_issue_comment.side_effect = GithubException(
            403, {"message": "Forbidden"}, None
        )

        with pytest.raises(CustomException):
            client.post_review_comment("owner", "repo", 42, "Review body")

    def test_comment_body_is_passed_correctly(self, client_and_pr):
        client, fake_pr, _ = client_and_pr
        review_body = "## AI Review\n\n**Verdict:** REQUEST_CHANGES\n\n- Line 12: missing type hint"

        client.post_review_comment("owner", "repo", 42, review_body)

        args, _ = fake_pr.create_issue_comment.call_args
        assert args[0] == review_body