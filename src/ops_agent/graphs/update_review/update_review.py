"""Graph: Update Review - Renovate PR reviewer.

Linear graph: ingest_pr → research → extract_breaking_changes → assess_risk
              → assemble_verdict → post_review → END

- ingest_pr:               reads PR metadata + diff via GiteaClient (no LLM)
- research:                tool-calling agent (research persona) gathers evidence
- extract_breaking_changes: structured extraction of quote-backed findings
                            (extract persona, no tools) — high-confidence trigger
- assess_risk:             reasoned, non-verbatim judgment of whether the update
                            *could* break (reason persona) — complements, never
                            overrides, the verbatim findings
- assemble_verdict:        pure Python decision logic, incl. a deterministic
                            downgrade / version-scheme check
- post_review:             posts markdown comment via GiteaClient (no LLM)
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from ops_agent.graphs.update_review.nodes.assess_risk import assess_risk
from ops_agent.graphs.update_review.nodes.assemble_verdict import assemble_verdict
from ops_agent.graphs.update_review.nodes.extract_breaking_changes import extract_breaking_changes
from ops_agent.graphs.update_review.nodes.ingest_pr import ingest_pr
from ops_agent.graphs.update_review.nodes.post_review import post_review
from ops_agent.graphs.update_review.nodes.research import research
from ops_agent.state import UpdateReviewState
from ops_agent.tracing import get_langfuse_handler
from ops_agent.types import Verdict

logger = logging.getLogger(__name__)


def build_graph() -> Any:
    """Build and compile the update review graph.

    The graph is linear:
      ingest_pr → research → extract_breaking_changes → assess_risk
        → assemble_verdict → post_review → END

    Compiled with checkpointing support if PostgreSQL is available.
    """
    from ops_agent.checkpointing import get_checkpoint_saver

    graph = StateGraph(UpdateReviewState)

    graph.add_node("ingest_pr", ingest_pr)
    graph.add_node("research", research)
    graph.add_node("extract_breaking_changes", extract_breaking_changes)
    graph.add_node("assess_risk", assess_risk)
    graph.add_node("assemble_verdict", assemble_verdict)
    graph.add_node("post_review", post_review)

    graph.set_entry_point("ingest_pr")
    graph.add_edge("ingest_pr", "research")
    graph.add_edge("research", "extract_breaking_changes")
    graph.add_edge("extract_breaking_changes", "assess_risk")
    graph.add_edge("assess_risk", "assemble_verdict")
    graph.add_edge("assemble_verdict", "post_review")
    graph.add_edge("post_review", END)

    checkpointer = get_checkpoint_saver()
    if checkpointer:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()


def run(pr_index: int, owner: str, repo: str, thread_id: str | None = None) -> Verdict:
    """Run the full update review graph for a single PR.

    Resumes from checkpoint if one exists for the thread_id, otherwise starts fresh.

    Returns the Verdict from the final state.
    """
    trace_name = f"update-review {pr_index}"
    logger.info("Starting %s", trace_name)

    ctx = get_langfuse_handler(
        trace_name=trace_name,
        tags=["graph:update-review"],
        metadata={"owner": owner, "repo": repo, "pr_index": pr_index},
    )

    compiled = build_graph()
    thread_id = thread_id or f"{owner}/{repo}#{pr_index}"

    initial_state: dict[str, Any] = {
        "pr_index": pr_index,
        "_owner": owner,
        "_repo": repo,
        "thread_id": thread_id,
        "dependency": "",
        "current_version": "",
        "new_version": "",
        "diff": "",
        "renovate_rating": None,
        "evidence": [],
        "_findings": [],
        "_risk": None,
        "verdict": None,
        "posted": False,
    }

    callbacks = [ctx.handler] if ctx.handler else []
    config = {
        "callbacks": callbacks,
        "configurable": {"thread_id": thread_id},
    }

    try:
        state = compiled.get_state(config)
        logger.debug("Checkpoint state: values=%s, next=%s", state.values is not None, state.next)

        if state.values and not state.next:
            # Checkpoint exists, no pending nodes -> already finished
            logger.debug("Resuming from completed checkpoint (thread_id=%s)", thread_id)
            final_state = state.values
        elif state.next:
            # Checkpoint exists with pending nodes -> resume
            logger.debug("Resuming from checkpoint (thread_id=%s) at nodes: %s", thread_id, state.next)
            final_state = compiled.invoke(None, config)
        else:
            # No checkpoint -> fresh run
            logger.debug("Starting fresh execution (thread_id=%s)", thread_id)
            final_state = compiled.invoke(initial_state, config)
    except Exception as e:
        logger.error("Graph execution failed: %s", e, exc_info=True)
        raise

    if final_state is None:
        logger.error("No final state returned from graph")
        return None

    return final_state.get("verdict")
