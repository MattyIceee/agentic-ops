"""Tests for llm/personas.py — no network, no model inference."""

from langchain_openai import ChatOpenAI

from ops_agent.llm.personas import PERSONAS, get_llm


def test_all_four_personas_present():
    assert set(PERSONAS) == {"research", "coding", "extract", "instruct"}


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
