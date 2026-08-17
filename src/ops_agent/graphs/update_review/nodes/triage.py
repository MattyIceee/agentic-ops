"""Node: triage - deterministic entry node deciding whether to (re-)review.

Runs on every invocation (including idle cron ticks). Cheap: at most one PR
fetch + one comment list, no LLM. Routes:

* ``full``    - first review, or a new foreign commit landed → full pipeline
* ``comment`` - a new foreign steering comment → classify_comment decides scope
* ``none``    - nothing new since we last acted → END
"""

import logging
from typing import Any

from ops_agent.graphs.interactive import (
    get_bot_login,
    head_commit_sha,
    new_steering_inputs,
)
from ops_agent.state import UpdateReviewState
from ops_agent.tools.github import GitHubClient

logger = logging.getLogger(__name__)


def triage(state: UpdateReviewState) -> dict[str, Any]:
    """Decide whether the PR needs a (re-)review and by which path."""
    # First run: nothing reviewed yet → run the full pipeline. No GitHub calls
    # needed for change detection; ingest_pr will fetch what it needs.
    if state.get("verdict") is None:
        return {"_route": "full", "new_inputs": []}

    owner: str = state["_owner"]  # type: ignore[typeddict-item]
    repo: str = state["_repo"]  # type: ignore[typeddict-item]
    index: int = state["pr_index"]

    bot = get_bot_login()
    client = GitHubClient()
    try:
        pr = client.get_pr(owner, repo, index)
        new_inputs = new_steering_inputs(client, owner, repo, index, bot)
    finally:
        client.close()

    # A new commit means the reviewed artifact itself changed → full re-review.
    head = head_commit_sha(pr)
    if head and head != state.get("thread_id"):
        logger.info("triage: new commit %s (was %s) → full", head, state.get("thread_id"))
        return {"_route": "full", "new_inputs": []}

    # Otherwise look for foreign comments/reviews newer than our last activity.
    if new_inputs:
        logger.info("triage: %d new foreign item(s) → comment", len(new_inputs))
        return {"_route": "comment", "new_inputs": new_inputs}

    logger.debug("triage: no external change → none")
    return {"_route": "none", "new_inputs": []}
