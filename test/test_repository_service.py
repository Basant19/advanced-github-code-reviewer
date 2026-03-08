"""
tests/test_repository_service.py

Unit Tests for Repository Vector Memory Service
-------------------------------------------------
Tests all public functions with mocked ChromaDB and GitHub — no real
API calls or disk writes.

Run with:
    pytest tests/test_repository_service.py -v

What is mocked:
    - _get_chroma_client()      → fake ChromaDB client
    - _get_embedding_function() → fake embedding function
    - GitHubClient              → fake GitHub repo/file responses
    - chromadb.PersistentClient → prevented from writing to disk
"""

import sys
import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.exceptions import CustomException


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_mock_collection(count: int = 0) -> MagicMock:
    """Returns a mock ChromaDB collection."""
    col = MagicMock()
    col.count.return_value = count
    col.get.return_value   = {"ids": [f"id-{i}" for i in range(count)]}
    col.query.return_value = {"documents": [[]]}
    return col


def make_mock_github_file(path: str, content: str = "def foo(): pass\n") -> MagicMock:
    """Returns a mock GitHub ContentFile."""
    f = MagicMock()
    f.path     = path
    f.type     = "blob"
    f.encoding = "base64"
    f.decoded_content = content.encode("utf-8")
    return f


def make_mock_tree_item(path: str, item_type: str = "blob") -> MagicMock:
    """Returns a mock git tree item."""
    item      = MagicMock()
    item.path = path
    item.type = item_type
    return item


# ──────────────────────────────────────────────────────────────────────────────
# Test: _collection_name  (pure function)
# ──────────────────────────────────────────────────────────────────────────────

class TestCollectionName:

    def test_produces_safe_collection_name(self):
        from app.services.repository_service import _collection_name
        name = _collection_name("Basant19", "advanced-github-code-reviewer")
        assert name == "basant19-advanced-github-code-reviewer"

    def test_lowercases_owner_and_repo(self):
        from app.services.repository_service import _collection_name
        name = _collection_name("MyOwner", "MyRepo")
        assert name == "myowner-myrepo"

    def test_replaces_special_chars_with_hyphens(self):
        from app.services.repository_service import _collection_name
        name = _collection_name("owner", "repo_with_underscores")
        assert "_" not in name
        assert "-" in name

    def test_truncates_to_63_chars(self):
        from app.services.repository_service import _collection_name
        long_repo = "a" * 100
        name      = _collection_name("owner", long_repo)
        assert len(name) <= 63

    def test_different_repos_produce_different_names(self):
        from app.services.repository_service import _collection_name
        name1 = _collection_name("owner", "repo-a")
        name2 = _collection_name("owner", "repo-b")
        assert name1 != name2

    def test_same_inputs_produce_same_name(self):
        from app.services.repository_service import _collection_name
        assert (
            _collection_name("Basant19", "my-repo")
            == _collection_name("Basant19", "my-repo")
        )


# ──────────────────────────────────────────────────────────────────────────────
# Test: _chunk_text  (pure function)
# ──────────────────────────────────────────────────────────────────────────────

class TestChunkText:

    def test_short_text_returns_single_chunk(self):
        from app.services.repository_service import _chunk_text
        chunks = _chunk_text("def foo(): pass\n", chunk_size=1500)
        assert len(chunks) == 1
        assert "def foo" in chunks[0]

    def test_empty_text_returns_empty_list(self):
        from app.services.repository_service import _chunk_text
        assert _chunk_text("") == []
        assert _chunk_text("   ") == []
        assert _chunk_text(None) == []

    def test_long_text_is_split_into_multiple_chunks(self):
        from app.services.repository_service import _chunk_text
        # 50 lines of 40 chars each = 2000 chars total
        long_text = ("x" * 39 + "\n") * 50
        chunks    = _chunk_text(long_text, chunk_size=500)
        assert len(chunks) > 1

    def test_each_chunk_is_within_size_limit(self):
        from app.services.repository_service import _chunk_text
        long_text = ("def function_" + "x" * 30 + "(): pass\n") * 30
        chunks    = _chunk_text(long_text, chunk_size=300)
        # Allow slight overage due to overlap lines
        for chunk in chunks:
            assert len(chunk) < 600   # reasonable upper bound

    def test_chunks_contain_original_content(self):
        from app.services.repository_service import _chunk_text
        content = "class MyClass:\n    def method(self):\n        return 42\n"
        chunks  = _chunk_text(content)
        combined = " ".join(chunks)
        assert "MyClass" in combined
        assert "method"  in combined


# ──────────────────────────────────────────────────────────────────────────────
# Test: query_context
# ──────────────────────────────────────────────────────────────────────────────

class TestQueryContext:

    def test_returns_list_of_strings(self):
        from app.services.repository_service import query_context

        col = make_mock_collection(count=5)
        col.query.return_value = {
            "documents": [["def foo(): pass", "class Bar: ..."]]
        }

        with patch("app.services.repository_service._get_collection", return_value=col):
            result = query_context("Basant19", "my-repo", "foo function")

        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)

    def test_returns_correct_number_of_results(self):
        from app.services.repository_service import query_context

        col = make_mock_collection(count=10)
        col.query.return_value = {
            "documents": [["chunk1", "chunk2", "chunk3"]]
        }

        with patch("app.services.repository_service._get_collection", return_value=col):
            result = query_context("owner", "repo", "query", n_results=3)

        assert len(result) == 3

    def test_returns_empty_list_when_collection_is_empty(self):
        from app.services.repository_service import query_context

        col = make_mock_collection(count=0)

        with patch("app.services.repository_service._get_collection", return_value=col):
            result = query_context("owner", "repo", "query")

        assert result == []
        col.query.assert_not_called()   # should short-circuit

    def test_raises_custom_exception_on_chromadb_error(self):
        from app.services.repository_service import query_context

        with patch(
            "app.services.repository_service._get_collection",
            side_effect=Exception("ChromaDB connection failed"),
        ):
            with pytest.raises(CustomException):
                query_context("owner", "repo", "query")

    def test_query_uses_repo_as_where_filter(self):
        from app.services.repository_service import query_context

        col = make_mock_collection(count=5)
        col.query.return_value = {"documents": [["some chunk"]]}

        with patch("app.services.repository_service._get_collection", return_value=col):
            query_context("owner", "my-repo", "find this", n_results=2)

        call_kwargs = col.query.call_args[1]
        assert call_kwargs["where"] == {"repo": "my-repo"}


# ──────────────────────────────────────────────────────────────────────────────
# Test: delete_repository
# ──────────────────────────────────────────────────────────────────────────────

class TestDeleteRepository:

    def test_returns_deleted_true(self):
        from app.services.repository_service import delete_repository

        mock_client = MagicMock()

        with patch("app.services.repository_service._get_chroma_client",
                   return_value=mock_client):
            result = delete_repository("owner", "repo")

        assert result["deleted"] is True

    def test_returns_collection_name(self):
        from app.services.repository_service import delete_repository, _collection_name

        mock_client = MagicMock()

        with patch("app.services.repository_service._get_chroma_client",
                   return_value=mock_client):
            result = delete_repository("Basant19", "my-repo")

        assert result["collection"] == _collection_name("Basant19", "my-repo")

    def test_calls_delete_collection_with_correct_name(self):
        from app.services.repository_service import delete_repository, _collection_name

        mock_client = MagicMock()

        with patch("app.services.repository_service._get_chroma_client",
                   return_value=mock_client):
            delete_repository("Basant19", "my-repo")

        mock_client.delete_collection.assert_called_once_with(
            name=_collection_name("Basant19", "my-repo")
        )

    def test_succeeds_even_if_collection_does_not_exist(self):
        """delete_collection raising means collection not found — should not crash."""
        from app.services.repository_service import delete_repository

        mock_client = MagicMock()
        mock_client.delete_collection.side_effect = Exception("Not found")

        with patch("app.services.repository_service._get_chroma_client",
                   return_value=mock_client):
            result = delete_repository("owner", "repo")   # must NOT raise

        assert result["deleted"] is True

    def test_raises_custom_exception_on_unexpected_error(self):
        from app.services.repository_service import delete_repository

        with patch(
            "app.services.repository_service._get_chroma_client",
            side_effect=Exception("Disk error"),
        ):
            with pytest.raises(CustomException):
                delete_repository("owner", "repo")


# ──────────────────────────────────────────────────────────────────────────────
# Test: get_repository_stats
# ──────────────────────────────────────────────────────────────────────────────

class TestGetRepositoryStats:

    def test_returns_stats_dict_with_required_keys(self):
        from app.services.repository_service import get_repository_stats

        col = make_mock_collection(count=42)

        with patch("app.services.repository_service._get_collection", return_value=col):
            result = get_repository_stats("owner", "repo")

        assert "collection" in result
        assert "chunks"     in result
        assert "indexed"    in result

    def test_indexed_is_true_when_chunks_exist(self):
        from app.services.repository_service import get_repository_stats

        col = make_mock_collection(count=10)

        with patch("app.services.repository_service._get_collection", return_value=col):
            result = get_repository_stats("owner", "repo")

        assert result["indexed"] is True
        assert result["chunks"]  == 10

    def test_indexed_is_false_when_no_chunks(self):
        from app.services.repository_service import get_repository_stats

        col = make_mock_collection(count=0)

        with patch("app.services.repository_service._get_collection", return_value=col):
            result = get_repository_stats("owner", "repo")

        assert result["indexed"] is False
        assert result["chunks"]  == 0

    def test_raises_custom_exception_on_error(self):
        from app.services.repository_service import get_repository_stats

        with patch(
            "app.services.repository_service._get_collection",
            side_effect=Exception("ChromaDB offline"),
        ):
            with pytest.raises(CustomException):
                get_repository_stats("owner", "repo")


# ──────────────────────────────────────────────────────────────────────────────
# Test: index_repository  (most complex — mocks GitHub + ChromaDB)
# ──────────────────────────────────────────────────────────────────────────────

class TestIndexRepository:

    def _make_github_mock(self, files: list[tuple[str, str]]) -> MagicMock:
        """
        Builds a mock GitHubClient whose repo has the given files.
        files: list of (path, content) tuples
        """
        tree_items = [make_mock_tree_item(path) for path, _ in files]

        git_tree       = MagicMock()
        git_tree.tree  = tree_items

        repo_obj = MagicMock()
        repo_obj.get_git_tree.return_value = git_tree

        def fake_get_contents(path):
            for fpath, content in files:
                if fpath == path:
                    return make_mock_github_file(fpath, content)
            raise Exception(f"File not found: {path}")

        repo_obj.get_contents.side_effect = fake_get_contents

        client = MagicMock()
        client._get_repo.return_value = repo_obj
        return client

    def test_returns_summary_dict(self):
        from app.services.repository_service import index_repository

        files     = [("app/main.py", "def main(): pass\n")]
        gh_mock   = self._make_github_mock(files)
        col_mock  = make_mock_collection(count=0)

        with patch("app.services.repository_service.GitHubClient", return_value=gh_mock), \
             patch("app.services.repository_service._get_collection", return_value=col_mock):
            result = index_repository("owner", "repo")

        assert "files_indexed"  in result
        assert "chunks_stored"  in result
        assert "collection"     in result

    def test_indexes_supported_file_types(self):
        from app.services.repository_service import index_repository

        files    = [
            ("app/main.py",    "def foo(): pass\n"),
            ("app/utils.js",   "function bar() {}"),
            ("README.md",      "# My project"),
            ("config.yaml",    "key: value"),
        ]
        gh_mock  = self._make_github_mock(files)
        col_mock = make_mock_collection(count=0)

        with patch("app.services.repository_service.GitHubClient", return_value=gh_mock), \
             patch("app.services.repository_service._get_collection", return_value=col_mock):
            result = index_repository("owner", "repo")

        assert result["files_indexed"] == 4

    def test_skips_unsupported_file_types(self):
        from app.services.repository_service import index_repository

        files    = [
            ("image.png",    "binary data"),
            ("binary.exe",   "binary data"),
            ("app/code.py",  "def foo(): pass\n"),   # only this should be indexed
        ]
        gh_mock  = self._make_github_mock(files)
        col_mock = make_mock_collection(count=0)

        with patch("app.services.repository_service.GitHubClient", return_value=gh_mock), \
             patch("app.services.repository_service._get_collection", return_value=col_mock):
            result = index_repository("owner", "repo")

        assert result["files_indexed"] == 1

    def test_clears_existing_chunks_before_reindex(self):
        from app.services.repository_service import index_repository

        files    = [("app/main.py", "def foo(): pass\n")]
        gh_mock  = self._make_github_mock(files)
        # Collection already has 5 chunks from a previous index run
        col_mock = make_mock_collection(count=5)

        with patch("app.services.repository_service.GitHubClient", return_value=gh_mock), \
             patch("app.services.repository_service._get_collection", return_value=col_mock):
            index_repository("owner", "repo")

        # delete() must have been called to clear stale embeddings
        col_mock.delete.assert_called()

    def test_raises_custom_exception_on_github_error(self):
        from app.services.repository_service import index_repository

        with patch(
            "app.services.repository_service.GitHubClient",
            side_effect=Exception("GitHub API rate limit"),
        ):
            with pytest.raises(CustomException):
                index_repository("owner", "repo")

    def test_skips_binary_files_gracefully(self):
        from app.services.repository_service import index_repository

        # File that claims to be binary (encoding="none")
        binary_file       = MagicMock()
        binary_file.path  = "app/code.py"
        binary_file.type  = "blob"
        binary_file.encoding = "none"

        tree_items    = [make_mock_tree_item("app/code.py")]
        git_tree      = MagicMock()
        git_tree.tree = tree_items

        repo_obj = MagicMock()
        repo_obj.get_git_tree.return_value = git_tree
        repo_obj.get_contents.return_value = binary_file

        gh_mock  = MagicMock()
        gh_mock._get_repo.return_value = repo_obj
        col_mock = make_mock_collection(count=0)

        with patch("app.services.repository_service.GitHubClient", return_value=gh_mock), \
             patch("app.services.repository_service._get_collection", return_value=col_mock):
            result = index_repository("owner", "repo")   # must NOT raise

        assert result["files_indexed"] == 0