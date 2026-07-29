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

from typing import Any

from langgraph.graph import END, StateGraph

from ops_agent.graphs.update_review.nodes.assess_risk import assess_risk
from ops_agent.graphs.update_review.nodes.assemble_verdict import assemble_verdict
from ops_agent.graphs.update_review.nodes.extract_breaking_changes import extract_breaking_changes
from ops_agent.graphs.update_review.nodes.ingest_pr import ingest_pr
from ops_agent.graphs.update_review.nodes.post_review import post_review
from ops_agent.graphs.update_review.nodes.research import research
from ops_agent.state import UpdateReviewState
from ops_agent.types import Verdict


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

    If thread_id is provided, the graph will resume from checkpoint if it exists.
    Otherwise, a fresh run is performed.

    Returns the Verdict from the final state.
    """
    compiled = build_graph()
    initial_state: dict[str, Any] = {
        "pr_index": pr_index,
        "_owner": owner,
        "_repo": repo,
        "thread_id": thread_id or f"{owner}/{repo}#{pr_index}",
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
    # Use thread_id for checkpoint resumption if available
    if thread_id:
        final_state = compiled.invoke(initial_state, config={"configurable": {"thread_id": thread_id}})
    else:
        final_state = compiled.invoke(initial_state)
    return final_state["verdict"]
