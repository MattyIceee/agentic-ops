"""Graph: Update Review - Renovate PR reviewer.

Interactive, re-drivable graph. A deterministic ``triage`` entry node runs on
every invocation and routes:

  triage ─▶ full     → ingest_pr → research → extract_breaking_changes
          │            → assess_risk → assemble_verdict → pick_quote → post_review → END
          ├▶ comment  → classify_comment ─▶ full  → ingest_pr (as above)
          │                               └▶ light → assess_risk → … → post_review
          └▶ none     → END

- triage:                  deterministic change detection (no LLM). First run or
                           a new foreign commit → full; a new foreign steering
                           comment → classify_comment; nothing new → END.
- classify_comment:        reasoned scope decision — does the comment need fresh
                           research (full) or just a re-judgment (light)?
- ingest_pr:               reads PR metadata + diff via GitHubClient (no LLM)
- research:                tool-calling agent (research persona) gathers evidence
- extract_breaking_changes: structured extraction of quote-backed findings
                            (extract persona, no tools) — high-confidence trigger
- assess_risk:             reasoned, non-verbatim judgment of whether the update
                            *could* break (reason persona) — complements, never
                            overrides, the verbatim findings
- assemble_verdict:        pure Python decision logic, incl. a deterministic
                            downgrade / version-scheme check
- pick_quote:              RAG lookup for an optional quote appended to the
                            comment (no LLM; toggled by
                            pr_quote_annotations_enabled)
- post_review:             posts markdown comment via GitHubClient (no LLM)

Checkpointing preserves prior evidence/verdict across turns, so a comment-driven
re-review resumes from a specific node without re-running upstream work.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from ops_agent.graphs.update_review.nodes.assess_risk import assess_risk
from ops_agent.graphs.update_review.nodes.assemble_verdict import assemble_verdict
from ops_agent.graphs.update_review.nodes.classify_comment import classify_comment
from ops_agent.graphs.update_review.nodes.extract_breaking_changes import extract_breaking_changes
from ops_agent.graphs.update_review.nodes.ingest_pr import ingest_pr
from ops_agent.graphs.update_review.nodes.pick_quote import pick_quote
from ops_agent.graphs.update_review.nodes.post_review import post_review
from ops_agent.graphs.update_review.nodes.research import research
from ops_agent.graphs.update_review.nodes.triage import triage
from ops_agent.state import UpdateReviewState
from ops_agent.tracing import get_langfuse_handler
from ops_agent.types import Verdict

logger = logging.getLogger(__name__)


def _route_after_triage(
    state: UpdateReviewState,
) -> Literal["ingest_pr", "classify_comment", "__end__"]:
    """Dispatch from triage: full pipeline, comment classification, or stop."""
    route = state.get("_route", "none")
    if route == "full":
        return "ingest_pr"
    if route == "comment":
        return "classify_comment"
    return "__end__"


def _route_after_classify(
    state: UpdateReviewState,
) -> Literal["ingest_pr", "assess_risk"]:
    """Full re-research vs light re-judgment for a follow-up comment."""
    return "ingest_pr" if state.get("_route") == "full" else "assess_risk"


def build_graph() -> Any:
    """Build and compile the update review graph.

    Entry is ``triage``; conditional edges jump straight to the needed re-entry
    node so upstream work is not repeated. Compiled with checkpointing support
    if PostgreSQL is available.
    """
    from ops_agent.checkpointing import get_checkpoint_saver

    graph = StateGraph(UpdateReviewState)

    graph.add_node("triage", triage)
    graph.add_node("classify_comment", classify_comment)
    graph.add_node("ingest_pr", ingest_pr)
    graph.add_node("research", research)
    graph.add_node("extract_breaking_changes", extract_breaking_changes)
    graph.add_node("assess_risk", assess_risk)
    graph.add_node("assemble_verdict", assemble_verdict)
    graph.add_node("pick_quote", pick_quote)
    graph.add_node("post_review", post_review)

    graph.set_entry_point("triage")
    graph.add_conditional_edges(
        "triage",
        _route_after_triage,
        {
            "ingest_pr": "ingest_pr",
            "classify_comment": "classify_comment",
            "__end__": END,
        },
    )
    graph.add_conditional_edges(
        "classify_comment",
        _route_after_classify,
        {"ingest_pr": "ingest_pr", "assess_risk": "assess_risk"},
    )
    graph.add_edge("ingest_pr", "research")
    graph.add_edge("research", "extract_breaking_changes")
    graph.add_edge("extract_breaking_changes", "assess_risk")
    graph.add_edge("assess_risk", "assemble_verdict")
    graph.add_edge("assemble_verdict", "pick_quote")
    graph.add_edge("pick_quote", "post_review")
    graph.add_edge("post_review", END)

    checkpointer = get_checkpoint_saver()
    if checkpointer:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()


def run(pr_index: int, owner: str, repo: str, thread_id: str | None = None) -> Verdict:
    """Run (or re-drive) the update review graph for a single PR.

    Three cases, decided from the checkpoint:

    1. No checkpoint            → fresh full run (seed the initial state).
    2. Checkpoint with pending  → native resume ``invoke(None)`` (crash recovery).
       nodes (``state.next``)
    3. Completed checkpoint      → re-invoke with an *increment only* (never the
       (no pending nodes)          full initial state, which would double the
                                    accumulating ``evidence`` channel). This
                                    re-enters at ``triage``, which decides whether
                                    any real work runs.

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
        "_quote": None,
        "posted": False,
        "new_inputs": [],
        "turn": 0,
        "_route": "",
    }

    callbacks = [ctx.handler] if ctx.handler else []
    config = {
        "callbacks": callbacks,
        "configurable": {"thread_id": thread_id},
    }

    try:
        state = compiled.get_state(config)
        logger.debug("Checkpoint state: values=%s, next=%s", state.values is not None, state.next)

        if state.next:
            # Case 2: checkpoint with pending nodes -> native crash-recovery resume.
            logger.debug("Resuming from checkpoint (thread_id=%s) at nodes: %s", thread_id, state.next)
            final_state = compiled.invoke(None, config)
        elif state.values:
            # Case 3: completed checkpoint -> re-drive. Pass ONLY the increment so
            # accumulating channels are not duplicated; triage re-detects change.
            prev_turn = state.values.get("turn", 0) or 0
            logger.debug("Re-driving completed checkpoint (thread_id=%s) turn=%d", thread_id, prev_turn + 1)
            final_state = compiled.invoke({"turn": prev_turn + 1}, config)
        else:
            # Case 1: no checkpoint -> fresh run.
            logger.debug("Starting fresh execution (thread_id=%s)", thread_id)
            final_state = compiled.invoke(initial_state, config)
    except Exception as e:
        logger.error("Graph execution failed: %s", e, exc_info=True)
        raise

    if final_state is None:
        logger.error("No final state returned from graph")
        return None

    return final_state.get("verdict")
