"""
Application Configuration Module

This module defines the centralized configuration system for the
Advanced GitHub Code Reviewer application.

It uses Pydantic Settings to load environment variables from a `.env`
file and convert them into strongly typed Python attributes.

Responsibilities
---------------
1. Load and validate environment variables.
2. Provide a centralized configuration object used across the project.
3. Ensure required API keys and secrets are present.
4. Support different environments (development, production, testing).

Environment Variables
---------------------
The following variables are expected in the `.env` file:

AI / Model Keys
    GOOGLE_API_KEY
    TAVILY_API_KEY

LangSmith
    LANGSMITH_API_KEY
    LANGSMITH_TRACING
    LANGSMITH_PROJECT

GitHub
    GITHUB_TOKEN
    GITHUB_WEBHOOK_SECRET

Database
    DATABASE_URL

Usage
-----
    from app.core.config import settings

    print(settings.app_name)
    print(settings.DATABASE_URL)

The `settings` object is a singleton and should be imported wherever
configuration is required.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

from app.core.logger import get_logger
from app.core.exceptions import CustomException
import sys


logger = get_logger(__name__)


class Settings(BaseSettings):
    """
    Application Settings

    This class defines all configuration variables used by the application.
    Values are automatically loaded from environment variables or `.env`.

    Attributes
    ----------
    app_name : str
        Name of the application.

    environment : str
        Current environment (development / production / testing).

    debug : bool
        Enables debug mode.

    google_api_key : str
        API key used for Google AI models.

    tavily_api_key : str
        API key for Tavily search tool.

    langsmith_tracing : bool
        Enables LangSmith tracing for LangGraph workflows.

    langsmith_project : str
        LangSmith project name.

    langsmith_api_key : str
        LangSmith authentication key.

    github_token : str
        GitHub personal access token.

    github_webhook_secret : str
        Secret used to verify GitHub webhooks.

    DATABASE_URL : str
        PostgreSQL database connection string.
    """

    # ============================
    # Application
    # ============================

    app_name: str = "Advanced GitHub Code Reviewer"
    environment: str = "development"
    debug: bool = True

    # ============================
    # AI / Model Keys
    # ============================

    google_api_key: str = Field(..., alias="GOOGLE_API_KEY")
    tavily_api_key: str = Field(..., alias="TAVILY_API_KEY")

    # ============================
    # LangSmith
    # ============================

    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")

    langsmith_project: str = Field(
        default="advanced-github-code-reviewer",
        alias="LANGSMITH_PROJECT"
    )

    langsmith_api_key: str = Field(..., alias="LANGSMITH_API_KEY")

    # ============================
    # GitHub
    # ============================

    github_token: str = Field(..., alias="GITHUB_TOKEN")

    github_webhook_secret: str = Field(..., alias="GITHUB_WEBHOOK_SECRET")

    # ============================
    # Database
    # ============================

    DATABASE_URL: str = Field(..., alias="DATABASE_URL")

    # ============================
    # Pydantic Settings Config
    # ============================

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )


try:
    settings = Settings()
    logger.info("Application configuration loaded successfully")

except Exception as e:
    logger.error("Failed to load application configuration")
    raise CustomException(e, sys)