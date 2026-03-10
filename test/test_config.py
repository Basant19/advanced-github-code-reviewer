"""
test/test_config.py

Tests for app/core/config.py
------------------------------
Verifies that all required settings are loaded correctly from .env.

Run with:
    pytest test/test_config.py -v -s
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path when run directly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import settings


class TestSettingsLoad:

    def test_settings_object_exists(self):
        assert settings is not None

    def test_app_name_is_set(self):
        assert settings.app_name == "Advanced GitHub Code Reviewer"

    def test_google_api_key_loaded(self):
        assert settings.google_api_key
        assert len(settings.google_api_key) > 10

    def test_github_token_loaded(self):
        assert settings.github_token
        assert len(settings.github_token) > 10

    def test_langsmith_api_key_loaded(self):
        assert settings.langsmith_api_key
        assert len(settings.langsmith_api_key) > 10

    def test_database_url_loaded(self):
        assert settings.DATABASE_URL
        assert "postgresql" in settings.DATABASE_URL

    def test_github_webhook_secret_loaded(self):
        assert settings.github_webhook_secret
        assert len(settings.github_webhook_secret) > 0

    def test_tavily_api_key_loaded(self):
        assert settings.tavily_api_key
        assert len(settings.tavily_api_key) > 0

    def test_environment_default(self):
        assert settings.environment in ("development", "production", "testing")

    def test_debug_is_bool(self):
        assert isinstance(settings.debug, bool)