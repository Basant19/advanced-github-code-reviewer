"""
test/test_config_and_nodes.py

Smoke Tests — Config + Nodes API Key Loading
---------------------------------------------
Verifies:
    1. config.py loads all required keys from .env
    2. config.py writes keys to os.environ correctly
    3. nodes.py receives GOOGLE_API_KEY from os.environ
    4. LLM initialises without error
    5. ChromaDB collection initialises without error

Run with:
    pytest test/test_config_and_nodes.py -v -s
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Config Tests ──────────────────────────────────────────────────────────────

class TestConfig:

    def test_settings_loads(self):
        from app.core.config import settings
        assert settings is not None

    def test_google_api_key_in_settings(self):
        from app.core.config import settings
        assert settings.google_api_key
        assert len(settings.google_api_key) > 10

    def test_github_token_in_settings(self):
        from app.core.config import settings
        assert settings.github_token
        assert len(settings.github_token) > 10

    def test_langsmith_api_key_in_settings(self):
        from app.core.config import settings
        assert settings.langsmith_api_key
        assert len(settings.langsmith_api_key) > 10

    def test_database_url_in_settings(self):
        from app.core.config import settings
        assert settings.DATABASE_URL
        assert "postgresql" in settings.DATABASE_URL

    def test_google_api_key_in_os_environ(self):
        """config.py must write GOOGLE_API_KEY to os.environ"""
        import app.core.config  # ensure config is loaded
        assert os.environ.get("GOOGLE_API_KEY"), (
            "GOOGLE_API_KEY missing from os.environ — "
            "config.py did not call os.environ['GOOGLE_API_KEY'] = settings.google_api_key"
        )

    def test_langsmith_api_key_in_os_environ(self):
        """config.py must write LANGSMITH_API_KEY to os.environ"""
        import app.core.config
        assert os.environ.get("LANGSMITH_API_KEY"), (
            "LANGSMITH_API_KEY missing from os.environ"
        )

    def test_os_environ_matches_settings(self):
        """os.environ values must match what settings loaded"""
        from app.core.config import settings
        assert os.environ["GOOGLE_API_KEY"] == settings.google_api_key
        assert os.environ["LANGSMITH_API_KEY"] == settings.langsmith_api_key


# ── Nodes Tests ───────────────────────────────────────────────────────────────

class TestNodes:

    def test_nodes_module_imports(self):
        """nodes.py must import without error — LLM initialises at import"""
        import app.graph.nodes as n
        assert n is not None

    def test_llm_is_initialised(self):
        """llm object must exist and not be None"""
        import app.graph.nodes as n
        assert n.llm is not None

    def test_llm_is_callable(self):
        """llm must be invokable (has .invoke method)"""
        import app.graph.nodes as n
        assert hasattr(n.llm, "invoke")

    def test_nodes_received_google_api_key(self):
        """
        nodes.py reads GOOGLE_API_KEY from os.environ at import.
        If it reached this point without raising, the key was present.
        """
        assert os.environ.get("GOOGLE_API_KEY"), (
            "GOOGLE_API_KEY not in os.environ when nodes.py was loaded"
        )
        assert len(os.environ["GOOGLE_API_KEY"]) > 10

    def test_chroma_collection_initialises(self):
        """ChromaDB collection must initialise with text-embedding-004"""
        import app.graph.nodes as n
        try:
            collection = n._get_chroma_collection()
            assert collection is not None
        except Exception as e:
            assert False, f"ChromaDB collection failed to initialise: {e}"

    def test_all_node_functions_exist(self):
        """All 4 node functions must be importable"""
        from app.graph.nodes import (
            fetch_diff_node,
            analyze_code_node,
            reflect_node,
            verdict_node,
        )
        assert all([
            fetch_diff_node,
            analyze_code_node,
            reflect_node,
            verdict_node,
        ])