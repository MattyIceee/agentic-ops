"""LangGraph state containers for both graphs.

Pydantic models (EvidenceItem, Finding, RiskAssessment, Verdict) are in ops_agent.types.
TypedDicts below define LangGraph state with list-accumulating reducers.
"""

import operator
from typing import Annotated

from typing_extensions import TypedDict

from ops_agent.types import EvidenceItem, Finding, RiskAssessment, Verdict


class UpdateReviewState(TypedDict):
    """State for Graph A: Renovate PR reviewer."""

    pr_index: int
    _owner: str
    _repo: str
    thread_id: str  # Checkpoint key: based on last commit ID from PR
    dependency: str
    current_version: str
    new_version: str
    diff: str
    renovate_rating: str | None
    evidence: Annotated[list[EvidenceItem], operator.add]
    # Intermediate channels — must be declared, or LangGraph silently drops
    # them between nodes (undeclared keys do not propagate).
    _findings: list[Finding]
    _risk: RiskAssessment | None
    verdict: Verdict | None
    posted: bool


class ServiceDeployState(TypedDict):
    """State for Graph B: deployment scaffolder."""

    request: str
    thread_id: str  # Checkpoint key: based on issue_number or explicit id parameter
    provided_links: list[str]
    spec: dict
    service_evidence: Annotated[list[EvidenceItem], operator.add]
    helm_chart_found: bool
    helm_chart_ref: str | None
    conventions: str
    existing_namespaces: list[str]  # Namespaces discovered in the target repo
    manifests: dict[str, str]
    review_passed: bool
    review_issues: list[str]
    retry_count: int
    pr_url: str | None
    _owner: str  # Internal: Gitea repo owner (optional)
    _repo: str  # Internal: Gitea repo name (optional)
    issue_number: int | None  # Internal: issue number if provided
