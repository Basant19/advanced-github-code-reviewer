# test_config.py

from app.core.config import settings

print("App Name:", settings.app_name)
print("Google Key Loaded:", bool(settings.google_api_key))
print("GitHub Token Loaded:", bool(settings.github_token))