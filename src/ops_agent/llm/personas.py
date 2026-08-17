"""Persona / model client factory for ops-agent.

Personas are params, not server aliases. There is exactly ONE model alias
(configured in Settings.model_alias). A "persona" is a named bundle of
sampling parameters (temperature, top_p, top_k, presence_penalty) plus
an enable_thinking flag that controls Qwen3's extended thinking mode.

Adding a new persona is a one-line addition to PERSONAS — nothing else changes.
"""

from dataclasses import dataclass

from langchain_openai import ChatOpenAI

from ops_agent.config import get_settings


@dataclass(frozen=True)
class PersonaParams:
    """Sampling parameters for a single persona."""

    temperature: float
    top_p: float
    top_k: int
    presence_penalty: float
    enable_thinking: bool


PERSONAS: dict[str, PersonaParams] = {
    "research": PersonaParams(temperature=0.7, top_p=0.95, top_k=40, presence_penalty=0.0, enable_thinking=True),
    "coding":   PersonaParams(temperature=0.25, top_p=0.9, top_k=40, presence_penalty=0.0, enable_thinking=True),
    "extract":  PersonaParams(temperature=0.0, top_p=1.0, top_k=0,  presence_penalty=0.0, enable_thinking=False),
    "reason":   PersonaParams(temperature=0.2, top_p=0.9, top_k=40, presence_penalty=0.0, enable_thinking=False),
    "instruct": PersonaParams(temperature=0.3, top_p=0.9, top_k=40, presence_penalty=0.0, enable_thinking=False),
}


def get_llm(persona: str) -> ChatOpenAI:
    """Return a ChatOpenAI client bound with the named persona's sampling params.

    The endpoint is OpenAI-compatible. Configure it with LLM_PROVIDER,
    LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL; the legacy LLAMACPP_BASE_URL /
    LLAMACPP_API_KEY / MODEL_ALIAS variables are used as fallbacks.

    Provider profiles:

    - ``llamacpp`` (default): temperature and top_p are standard params;
      top_k, presence_penalty, and ``chat_template_kwargs.enable_thinking`` are
      llama.cpp/Qwen3 extensions sent via extra_body.
    - ``openai-compatible``: standard OpenAI params only (temperature, top_p) —
      safe for any /v1 server that rejects unknown body fields.
    """
    settings = get_settings()
    params = PERSONAS[persona]

    kwargs: dict = {
        "base_url": settings.llm_base_url or settings.llamacpp_base_url,
        "model": settings.llm_model or settings.model_alias,
        "api_key": settings.llm_api_key or settings.llamacpp_api_key,
        "timeout": settings.request_timeout_seconds,
        "temperature": params.temperature,
        "top_p": params.top_p,
    }

    if settings.llm_provider == "llamacpp":
        kwargs["extra_body"] = {
            "top_k": params.top_k,
            "presence_penalty": params.presence_penalty,
            "chat_template_kwargs": {"enable_thinking": params.enable_thinking},
        }

    return ChatOpenAI(**kwargs)
