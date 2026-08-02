"""Node: triage - deterministic entry node deciding whether to (re-)scaffold.

Runs on every invocation (including idle cron ticks). Cheap: at most one comment
list on the opened PR, no LLM. Routes:

* ``full``  - no PR opened yet → run the full scaffold pipeline
* ``steer`` - a new foreign comment on the PR steers the implementation →
              re-generate manifests and push to the existing PR
* ``none``  - nothing new since we last acted → END

Steering is watched on the opened PR only (single source for the comment
boundary); commit_and_pr posts an acknowledgment there after acting, which
advances the boundary so the same comment never re-triggers.
"""

import logging
from typing import Any

from ops_agent.graphs.interactive import get_bot_login, new_steering_inputs
from ops_agent.state import ServiceDeployState
from ops_agent.tools.gitea import GiteaClient

logger = logging.getLogger(__name__)


def triage(state: ServiceDeployState) -> dict[str, Any]:
    """Decide whether to run the full pipeline, apply a steer, or stop."""
    # First run: no PR yet → full scaffold. (No steering target to watch.)
    pr_index = state.get("pr_index")
    if not state.get("pr_url") or not pr_index:
        return {"_route": "full", "new_inputs": []}

    owner: str = state.get("_owner", "")  # type: ignore[typeddict-item]
    repo: str = state.get("_repo", "")  # type: ignore[typeddict-item]
    if not owner or not repo:
        return {"_route": "none", "new_inputs": []}

    bot = get_bot_login()
    client = GiteaClient()
    try:
        new_inputs = new_steering_inputs(client, owner, repo, pr_index, bot)
    finally:
        client.close()

    if new_inputs:
        logger.info("triage: %d new steering item(s) on PR #%s → steer", len(new_inputs), pr_index)
        return {
            "_route": "steer",
            "new_inputs": new_inputs,
            # Give the re-generation a fresh self-review budget.
            "retry_count": 0,
            "review_issues": [],
            "review_passed": False,
        }

    logger.debug("triage: no new steering comments or reviews → none")
    return {"_route": "none", "new_inputs": []}
