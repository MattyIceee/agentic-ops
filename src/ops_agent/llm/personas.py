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
    "instruct": PersonaParams(temperature=0.3, top_p=0.9, top_k=40, presence_penalty=0.0, enable_thinking=False),
}


def get_llm(persona: str) -> ChatOpenAI:
    """Return a ChatOpenAI client bound with the named persona's sampling params.

    temperature and top_p are standard OpenAI params passed as kwargs.
    top_k, presence_penalty, and chat_template_kwargs are llama.cpp extensions
    sent via extra_body so they reach the model server unmodified.
    """
    settings = get_settings()
    params = PERSONAS[persona]

    return ChatOpenAI(
        base_url=settings.llamacpp_base_url,
        model=settings.model_alias,
        api_key=settings.llamacpp_api_key,
        timeout=settings.request_timeout_seconds,
        temperature=params.temperature,
        top_p=params.top_p,
        extra_body={
            "top_k": params.top_k,
            "presence_penalty": params.presence_penalty,
            "chat_template_kwargs": {"enable_thinking": params.enable_thinking},
        },
    )
