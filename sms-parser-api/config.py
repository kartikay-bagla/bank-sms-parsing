"""
Configuration settings loaded from environment variables.
"""

from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # LM Studio endpoint
    lm_studio_url: str = "http://localhost:1234/v1/completions"

    # Actual HTTP API configuration
    actual_api_url: str = "http://localhost:5007"
    actual_api_key: str = "your-actual-http-api-key"

    # Default budget and account IDs
    default_budget_sync_id: str = "your-budget-sync-id"
    default_account_id: str = "your-account-id"

    # Optional API key for protecting this endpoint
    sms_api_key: Optional[str] = None

    # Testing mode: disable Actual Budget import, just return parsed data
    disable_actual_budget: bool = False

    # Database for request logging (SQLite default, supports PostgreSQL)
    # SQLite: sqlite:///./requests.db
    # PostgreSQL: postgresql://user:pass@localhost/dbname
    database_url: str = "sqlite:///./requests.db"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
