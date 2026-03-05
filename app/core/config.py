from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
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

    # # Database
    DATABASE_URL: str = Field (..., alias="DATABASE_URL")
   
    # ============================
    # Pydantic Settings Config
    # ============================
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )


# Singleton instance
settings = Settings()