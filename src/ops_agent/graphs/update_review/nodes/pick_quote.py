"""Node: pick_quote - RAG lookup for a lighthearted quote to append to the PR comment."""

from typing import Any

from ops_agent.config import get_settings
from ops_agent.rag.quotes import get_relevant_quote
from ops_agent.state import UpdateReviewState


def pick_quote(state: UpdateReviewState) -> dict[str, Any]:
    """Retrieve a quote relevant to this PR's verdict, if the feature is enabled.

    Purely decorative: any failure here must never sink a review, so this
    degrades to {"_quote": None} rather than raising.
    """
    if not get_settings().pr_quote_annotations_enabled:
        return {"_quote": None}

    verdict = state.get("verdict")
    query = f"{state.get('dependency', '')} {verdict.summary if verdict else ''}".strip()

    try:
        quote = get_relevant_quote(query)
    except Exception:
        quote = None

    return {"_quote": quote}
