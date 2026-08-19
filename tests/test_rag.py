"""Tests for ops_agent.rag.quotes and the pick_quote node — no network, no live model."""

from __future__ import annotations

import httpx
import pytest
from langchain_core.documents import Document


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeStore:
    def __init__(self, search_results=None, raise_on_search=None):
        self.added_texts: list[str] = []
        self.added_ids: list[str] = []
        self._search_results = search_results or []
        self._raise_on_search = raise_on_search

    def add_texts(self, texts, ids=None):
        self.added_texts.extend(texts)
        self.added_ids.extend(ids or [])

    def similarity_search(self, query, k=1):
        if self._raise_on_search:
            raise self._raise_on_search
        return self._search_results[:k]


# ─────────────────────────── ingest_script ──────────────────────────────────


def test_ingest_script_chunks_and_upserts(monkeypatch):
    from ops_agent.rag import quotes as rq

    filler = " ".join(f"word{i}" for i in range(120))
    fake_text = "Paragraph one is here.\n\n" + filler + "\n\nParagraph three."
    monkeypatch.setattr(
        httpx, "get",
        lambda url, timeout=None: _FakeResponse({"files": {"script.txt": {"content": fake_text}}}),
    )

    store = _FakeStore()
    monkeypatch.setattr(rq, "get_vector_store", lambda: store)

    count = rq.ingest_script(gist_id="fake-gist-id")

    assert count > 0
    assert len(store.added_texts) == count
    assert len(store.added_ids) == count
    # ids are deterministic sha1 hashes of the chunk text, not empty/duplicated.
    assert len(set(store.added_ids)) == count


def test_ingest_script_is_idempotent(monkeypatch):
    """Re-ingesting the same text produces the same chunk ids (upsert, not duplicate)."""
    from ops_agent.rag import quotes as rq

    fake_text = "Some line.\n\nAnother line entirely."
    monkeypatch.setattr(
        httpx, "get",
        lambda url, timeout=None: _FakeResponse({"files": {"script.txt": {"content": fake_text}}}),
    )

    store = _FakeStore()
    monkeypatch.setattr(rq, "get_vector_store", lambda: store)

    rq.ingest_script(gist_id="fake-gist-id")
    first_ids = list(store.added_ids)

    rq.ingest_script(gist_id="fake-gist-id")
    second_ids = store.added_ids[len(first_ids):]

    assert first_ids == second_ids


def test_ingest_script_raises_on_empty_gist(monkeypatch):
    from ops_agent.rag import quotes as rq

    monkeypatch.setattr(httpx, "get", lambda url, timeout=None: _FakeResponse({"files": {}}))

    with pytest.raises(ValueError):
        rq.ingest_script(gist_id="fake-gist-id")


# ─────────────────────────── get_relevant_quote ─────────────────────────────


def test_get_relevant_quote_returns_none_when_disabled(monkeypatch):
    from ops_agent.rag import quotes as rq

    monkeypatch.setenv("VECTOR_DB_ENABLED", "false")

    def _boom():
        raise AssertionError("vector store must not be touched when disabled")

    monkeypatch.setattr(rq, "get_vector_store", _boom)

    assert rq.get_relevant_quote("some query") is None


def test_get_relevant_quote_returns_top_result(monkeypatch):
    from ops_agent.rag import quotes as rq

    store = _FakeStore(search_results=[Document(page_content="mock quote text")])
    monkeypatch.setattr(rq, "get_vector_store", lambda: store)

    assert rq.get_relevant_quote("some query") == "mock quote text"


def test_get_relevant_quote_returns_none_on_empty_store(monkeypatch):
    from ops_agent.rag import quotes as rq

    store = _FakeStore(search_results=[])
    monkeypatch.setattr(rq, "get_vector_store", lambda: store)

    assert rq.get_relevant_quote("some query") is None


def test_get_relevant_quote_returns_none_on_exception(monkeypatch):
    from ops_agent.rag import quotes as rq

    store = _FakeStore(raise_on_search=RuntimeError("store unreachable"))
    monkeypatch.setattr(rq, "get_vector_store", lambda: store)

    assert rq.get_relevant_quote("some query") is None


# ─────────────────────────── pick_quote node ────────────────────────────────


def test_pick_quote_short_circuits_when_annotations_disabled(monkeypatch):
    from ops_agent.graphs.update_review.nodes import pick_quote as pq

    monkeypatch.setenv("PR_QUOTE_ANNOTATIONS_ENABLED", "false")

    def _boom(_query):
        raise AssertionError("RAG lookup must not run when annotations are disabled")

    monkeypatch.setattr(pq, "get_relevant_quote", _boom)

    out = pq.pick_quote({"dependency": "requests", "verdict": None})
    assert out == {"_quote": None}


def test_pick_quote_returns_quote_when_enabled(monkeypatch):
    from ops_agent.graphs.update_review.nodes import pick_quote as pq

    monkeypatch.setattr(pq, "get_relevant_quote", lambda query: "mock quote text")

    out = pq.pick_quote({"dependency": "requests", "verdict": None})
    assert out == {"_quote": "mock quote text"}


def test_pick_quote_degrades_to_none_on_lookup_failure(monkeypatch):
    from ops_agent.graphs.update_review.nodes import pick_quote as pq

    def _boom(_query):
        raise RuntimeError("offline")

    monkeypatch.setattr(pq, "get_relevant_quote", _boom)

    out = pq.pick_quote({"dependency": "requests", "verdict": None})
    assert out == {"_quote": None}
