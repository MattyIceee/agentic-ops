"""Settings for ops-agent, loaded from environment / .env file."""

import functools
from typing import Literal

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

    # LLM provider — any OpenAI-compatible /v1 endpoint works (llama.cpp, vLLM,
    # Ollama, LM Studio, OpenRouter, Groq, ...). Set LLM_PROVIDER to
    # "openai-compatible" to disable the llama.cpp-specific extensions.
    # The legacy LLAMACPP_* / MODEL_ALIAS variables are used as fallbacks when
    # the corresponding LLM_* variables are unset.
    llm_provider: Literal["llamacpp", "openai-compatible"] = "llamacpp"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None

    # Legacy llama.cpp / model — fallback when the LLM_* variables are unset
    llamacpp_base_url: str = "http://localhost:8080/v1"
    llamacpp_api_key: str = "sk-no-auth"
    model_alias: str = "qwen3.6-a3b"

    # External services (required)
    searxng_url: str = Field(..., description="Base URL of the self-hosted SearXNG instance")

    # GitHub: the API host and the git clone host differ, so they are separate settings.
    github_api_base_url: str = "https://api.github.com/"
    github_base_url: str = "https://github.com"
    github_token: str = Field(..., description="GitHub fine-grained personal access token")

    # Git commit identity
    git_author_name: str = "ops-agent"
    git_author_email: str = "ops-agent@homelab.local"

    # Base directory where repos are cloned and managed.
    # Defaults to a container-friendly path; mount a volume here so cloned
    # repos persist across container restarts (the clone/reuse fast path
    # depends on them surviving). Override via env for local dev on Windows,
    # where a bare POSIX path resolves to the drive root.
    repo_location: str = "/data/repos"

    # HTTP behaviour
    request_timeout_seconds: int = 120

    # Checkpointing (PostgreSQL) — optional
    checkpoint_enabled: bool = True
    checkpoint_db_host: str = "localhost"
    checkpoint_db_port: int = 5432
    checkpoint_db_user: str = "ops_agent"
    checkpoint_db_password: str = "ops_agent_dev_password"
    checkpoint_db_name: str = "ops_agent_checkpoints"

    # Langfuse tracing — optional
    langfuse_enabled: bool = False
    langfuse_secret_key: str | None = None
    langfuse_public_key: str | None = None
    langfuse_base_url: str = "http://localhost:3000"


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings (instantiated once per process)."""
    return Settings()
