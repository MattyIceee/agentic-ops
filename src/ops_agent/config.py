"""Settings for ops-agent, loaded from environment / .env file."""

import functools

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings.

    All fields are read from environment variables or a .env file.
    Required fields (no default) must be present at runtime; they are not
    validated at import time so the module can be imported without a live env.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # llama.cpp / model
    llamacpp_base_url: str = "http://localhost:8080/v1"
    model_alias: str = "qwen3.6-a3b"
    llamacpp_api_key: str = "sk-no-auth"

    # External services (required)
    searxng_url: str = Field(..., description="Base URL of the self-hosted SearXNG instance")
    gitea_base_url: str = Field(..., description="Base URL of the Gitea instance, no trailing slash")
    gitea_token: str = Field(..., description="Gitea personal access token")

    # Git commit identity
    git_author_name: str = "ops-agent"
    git_author_email: str = "ops-agent@homelab.local"

    # Optional: local repo path Graph B uses to learn conventions
    example_repo_path: str | None = None

    # HTTP behaviour
    request_timeout_seconds: int = 120


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings (instantiated once per process)."""
    return Settings()
