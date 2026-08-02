"""Graph: Service Deploy - Deployment scaffolder.

Edges:
  parse_request → research_service → assess_helm → load_conventions
  → (route_helm) → generate_helmrelease / generate_kustomize
  → self_review → (route_review) → commit_and_pr / retry generate node
  → END

The self_review retry loop runs up to MAX_RETRIES=2 times before
falling through to commit_and_pr regardless of review outcome.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from ops_agent.graphs.service_deploy.nodes.assess_helm import assess_helm
from ops_agent.graphs.service_deploy.nodes.commit_and_pr import commit_and_pr
from ops_agent.graphs.service_deploy.nodes.generate_helmrelease import generate_helmrelease
from ops_agent.graphs.service_deploy.nodes.generate_kustomize import generate_kustomize
from ops_agent.graphs.service_deploy.nodes.load_conventions import load_conventions
from ops_agent.graphs.service_deploy.nodes.parse_request import parse_request
from ops_agent.graphs.service_deploy.nodes.research_service import research_service
from ops_agent.graphs.service_deploy.nodes.self_review import self_review
from ops_agent.state import ServiceDeployState
from ops_agent.tracing import get_langfuse_handler

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


def _route_after_conventions(
    state: ServiceDeployState,
) -> Literal["generate_helmrelease", "generate_kustomize"]:
    """Route to Helm or Kustomize generator based on helm_chart_found."""
    return "generate_helmrelease" if state.get("helm_chart_found") else "generate_kustomize"


def _route_after_review(
    state: ServiceDeployState,
) -> Literal["generate_helmrelease", "generate_kustomize", "commit_and_pr"]:
    """Retry the generate node if review failed and retries remain; else commit."""
    passed = state.get("review_passed", False)
    retries = state.get("retry_count", 0)

    if not passed and retries < MAX_RETRIES:
        return "generate_helmrelease" if state.get("helm_chart_found") else "generate_kustomize"
    return "commit_and_pr"


def build_graph() -> Any:
    """Build and compile the deployment scaffolder graph.

    Topology:
      parse_request → research_service → assess_helm → load_conventions
      → (helm?) → generate_helmrelease / generate_kustomize
      → self_review → (passed? retries?) → commit_and_pr / retry generate
      → END

    Compiled with checkpointing support if PostgreSQL is available.
    """
    from ops_agent.checkpointing import get_checkpoint_saver

    graph = StateGraph(ServiceDeployState)

    graph.add_node("parse_request", parse_request)
    graph.add_node("research_service", research_service)
    graph.add_node("assess_helm", assess_helm)
    graph.add_node("load_conventions", load_conventions)
    graph.add_node("generate_helmrelease", generate_helmrelease)
    graph.add_node("generate_kustomize", generate_kustomize)
    graph.add_node("self_review", self_review)
    graph.add_node("commit_and_pr", commit_and_pr)

    graph.set_entry_point("parse_request")
    graph.add_edge("parse_request", "research_service")
    graph.add_edge("research_service", "assess_helm")
    graph.add_edge("assess_helm", "load_conventions")

    graph.add_conditional_edges(
        "load_conventions",
        _route_after_conventions,
        {
            "generate_helmrelease": "generate_helmrelease",
            "generate_kustomize": "generate_kustomize",
        },
    )

    graph.add_edge("generate_helmrelease", "self_review")
    graph.add_edge("generate_kustomize", "self_review")

    graph.add_conditional_edges(
        "self_review",
        _route_after_review,
        {
            "generate_helmrelease": "generate_helmrelease",
            "generate_kustomize": "generate_kustomize",
            "commit_and_pr": "commit_and_pr",
        },
    )

    graph.add_edge("commit_and_pr", END)

    checkpointer = get_checkpoint_saver()
    if checkpointer:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()


def run(
    prompt: str,
    owner: str = "",
    repo: str = "",
    issue_number: int | None = None,
    id: str | None = None,
    thread_id: str | None = None,
    fresh: bool = False,
) -> str | None:
    """Run the full scaffold graph for a deployment request.

    Resumes from checkpoint if one exists for the thread_id, otherwise starts fresh.

    Args:
        prompt: Deployment request text
        owner: Optional Gitea repo owner
        repo: Optional Gitea repo name
        issue_number: Optional issue number to link
        id: Optional explicit checkpoint ID for resuming
        thread_id: Optional explicit thread ID for checkpointing
        fresh: If True, delete any existing checkpoint for this thread before
            running, forcing a clean execution from the start.

    Returns the PR URL from the final state, or None if not set.
    """
    # Determine the checkpoint thread_id
    if thread_id:
        checkpoint_id = thread_id
    elif issue_number:
        checkpoint_id = f"issue_{issue_number}"
    elif id:
        checkpoint_id = id
    else:
        checkpoint_id = "scaffold_deploy_no_id"

    if fresh:
        from ops_agent.checkpointing import clear_thread

        clear_thread(checkpoint_id)

    trace_name = f"service-deploy {issue_number}" if issue_number else "service-deploy"
    logger.info("Starting %s", trace_name)

    tags = ["graph:service-deploy"]
    if issue_number:
        tags.append("from-issue")

    ctx = get_langfuse_handler(
        trace_name=trace_name,
        tags=tags,
        metadata={
            "issue_number": issue_number,
            "owner": owner,
            "repo": repo,
            "request": prompt[:200],
        },
    )

    compiled = build_graph()

    initial_state: dict[str, Any] = {
        "request": prompt,
        "thread_id": checkpoint_id,
        "provided_links": [],
        "spec": {},
        "service_evidence": [],
        "helm_chart_found": False,
        "helm_chart_ref": None,
        "conventions": "",
        "manifests": {},
        "review_passed": False,
        "review_issues": [],
        "retry_count": 0,
        "pr_url": None,
        "_owner": owner,
        "_repo": repo,
        "issue_number": issue_number,
    }

    callbacks = [ctx.handler] if ctx.handler else []
    config = {
        "callbacks": callbacks,
        "configurable": {"thread_id": checkpoint_id},
    }

    try:
        state = compiled.get_state(config)
        logger.debug("Checkpoint state: values=%s, next=%s", state.values is not None, state.next)

        if state.values and not state.next:
            # Checkpoint exists, no pending nodes -> already finished
            logger.debug("Resuming from completed checkpoint (thread_id=%s)", checkpoint_id)
            final_state = state.values
        elif state.next:
            # Checkpoint exists with pending nodes -> resume
            logger.debug("Resuming from checkpoint (thread_id=%s) at nodes: %s", checkpoint_id, state.next)
            final_state = compiled.invoke(None, config)
        else:
            # No checkpoint -> fresh run
            logger.debug("Starting fresh execution (thread_id=%s)", checkpoint_id)
            final_state = compiled.invoke(initial_state, config)
    except Exception as e:
        logger.error("Graph execution failed: %s", e, exc_info=True)
        raise

    if final_state is None:
        logger.error("No final state returned from graph")
        return None

    return final_state.get("pr_url")
