"""RAG-backed quote retrieval for PR review comments.

Ingests an arbitrary text source (by default a public gist) into a Chroma
vector store, and retrieves the single most relevant chunk for a query. Used
by the update_review graph's pick_quote node to append a lighthearted quote
to posted PR comments — see ops_agent.config.Settings.pr_quote_annotations_enabled
for the on/off switch.
"""

from __future__ import annotations

import hashlib
import logging
from urllib.parse import urlparse

import httpx
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ops_agent.config import get_settings
from ops_agent.llm.embeddings import get_embeddings

logger = logging.getLogger(__name__)

# The Bee Movie script gist (https://gist.github.com/MattIPv4/045239bc27b16b2bcf7a3a9a4648c08a),
# used as the default quote source for ingest_script().
_DEFAULT_GIST_ID = "045239bc27b16b2bcf7a3a9a4648c08a"

_CHUNK_SIZE = 400
_CHUNK_OVERLAP = 40


def get_vector_store() -> Chroma:
    """Return a Chroma vector store client for the configured collection."""
    settings = get_settings()
    parsed = urlparse(settings.chroma_url)

    import chromadb

    client = chromadb.HttpClient(
        host=parsed.hostname or "localhost",
        port=parsed.port or 8000,
    )
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings(),
        client=client,
    )


def _fetch_gist_text(gist_id: str) -> str:
    """Fetch a public gist's file contents via the GitHub gists API.

    The gists API returns file content inline for files under its size cap,
    so there's no need to resolve a raw-content URL/revision hash separately.
    """
    settings = get_settings()
    resp = httpx.get(
        f"https://api.github.com/gists/{gist_id}",
        timeout=settings.request_timeout_seconds,
    )
    resp.raise_for_status()
    files = resp.json().get("files", {})
    if not files:
        raise ValueError(f"Gist {gist_id} has no files")

    first_file = next(iter(files.values()))
    return first_file["content"]


def ingest_script(gist_id: str = _DEFAULT_GIST_ID) -> int:
    """Fetch, chunk, embed, and upsert a gist's text into the vector store.

    Chunk ids are deterministic (sha1 of chunk text), so re-running is an
    idempotent upsert rather than a duplicate-growing append.

    Returns the number of chunks ingested.
    """
    text = _fetch_gist_text(gist_id)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = [c for c in splitter.split_text(text) if c.strip()]
    if not chunks:
        return 0

    ids = [hashlib.sha1(chunk.encode("utf-8")).hexdigest() for chunk in chunks]

    store = get_vector_store()
    store.add_texts(texts=chunks, ids=ids)
    return len(chunks)


def get_relevant_quote(query: str) -> str | None:
    """Return the single most relevant ingested chunk for a query, or None.

    This is decorative — any failure (store unreachable, empty collection,
    embedding error) must degrade to None rather than propagate, so a review
    is never blocked on the quote lookup.
    """
    settings = get_settings()
    if not settings.vector_db_enabled:
        return None

    try:
        store = get_vector_store()
        results = store.similarity_search(query, k=1)
    except Exception:
        logger.warning("Quote RAG lookup failed", exc_info=True)
        return None

    if not results:
        return None
    return results[0].page_content
