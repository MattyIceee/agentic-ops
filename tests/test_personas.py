"""Tests for llm/personas.py — no network, no model inference."""

import pytest
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from ops_agent.llm.personas import PERSONAS, get_llm


def test_all_personas_present():
    assert set(PERSONAS) == {"research", "coding", "extract", "instruct", "reason"}


def test_reason_params():
    p = PERSONAS["reason"]
    assert p.enable_thinking is False
    assert 0.0 < p.temperature < 0.5


def test_extract_params():
    p = PERSONAS["extract"]
    assert p.enable_thinking is False
    assert p.temperature == 0.0


def test_research_params():
    p = PERSONAS["research"]
    assert p.enable_thinking is True
    assert p.temperature > 0


def test_get_llm_returns_chat_openai():
    llm = get_llm("extract")
    assert isinstance(llm, ChatOpenAI)


def test_get_llm_all_personas_no_error():
    for name in PERSONAS:
        llm = get_llm(name)
        assert isinstance(llm, ChatOpenAI)


def test_llamacpp_provider_sends_extensions():
    llm = get_llm("research")
    assert llm.extra_body == {
        "top_k": PERSONAS["research"].top_k,
        "presence_penalty": PERSONAS["research"].presence_penalty,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def test_openai_compatible_provider_omits_extensions(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai-compatible")
    llm = get_llm("research")
    assert llm.extra_body is None
    assert llm.model_name == "qwen3.6-a3b"


def test_generic_vars_override_legacy(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://ollama.local:11434/v1")
    monkeypatch.setenv("LLM_API_KEY", "ollama-key")
    monkeypatch.setenv("LLM_MODEL", "llama3.2")
    llm = get_llm("extract")
    assert llm.openai_api_base == "http://ollama.local:11434/v1"
    assert llm.openai_api_key.get_secret_value() == "ollama-key"
    assert llm.model_name == "llama3.2"


def test_legacy_vars_used_as_fallback():
    llm = get_llm("extract")
    assert llm.openai_api_base == "http://localhost:8080/v1"
    assert llm.openai_api_key.get_secret_value() == "sk-no-auth"
    assert llm.model_name == "qwen3.6-a3b"


def test_invalid_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "not-a-provider")
    with pytest.raises(ValidationError):
        get_llm("extract")
