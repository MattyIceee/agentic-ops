"""Embedding client factory for ops-agent.

Mirrors llm/personas.py's get_llm(): the same OpenAI-compatible endpoint
(LLM_BASE_URL/LLM_API_KEY) is reused for embeddings, just with a different
model name (EMBEDDING_MODEL).
"""

from langchain_openai import OpenAIEmbeddings

from ops_agent.config import get_settings


def get_embeddings() -> OpenAIEmbeddings:
    """Return an OpenAIEmbeddings client bound to the configured embedding model."""
    settings = get_settings()

    return OpenAIEmbeddings(
        base_url=settings.llm_base_url or settings.llamacpp_base_url,
        api_key=settings.llm_api_key or settings.llamacpp_api_key,
        model=settings.embedding_model,
        timeout=settings.request_timeout_seconds,
    )
