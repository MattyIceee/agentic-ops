"""Shared state types for both LangGraph graphs.

Pydantic models (EvidenceItem, Finding, Verdict) are used for structured LLM output.
TypedDicts are used as LangGraph state containers with list-accumulating reducers.
"""

import operator
from typing import Annotated, Literal

from pydantic import BaseModel
from typing_extensions import TypedDict


class EvidenceItem(BaseModel):
    """A piece of evidence gathered from an external source."""

    source: str
    url: str | None
    text: str


class Finding(BaseModel):
    """A single risk claim grounded in a verbatim quote from evidence.

    ``category`` distinguishes a documented breaking change from a *regression*
    signal — evidence that the new version is older, stale, deprecated, yanked,
    or otherwise a downgrade relative to the current or latest release.
    """

    claim: str
    source: str
    quote: str
    category: Literal["breaking", "regression"] = "breaking"


class Findings(BaseModel):
    """Container for a list of Findings.

    Structured-output backends (``with_structured_output``) require a schema
    class — a bare ``list[Finding]`` generic is rejected — so extraction targets
    this wrapper and reads ``.findings``.
    """

    findings: list[Finding] = []


class RiskAssessment(BaseModel):
    """A reasoned (non-verbatim) judgment about whether an update could break.

    Unlike a :class:`Finding`, this is allowed to *infer* — it weighs the
    gathered evidence and expresses a level of confidence. It complements, and
    never overrides, verbatim-grounded findings.
    """

    could_break: bool = False
    risk_level: Literal["none", "low", "medium", "high"] = "none"
    rationale: str = ""
    signals: list[str] = []


class Verdict(BaseModel):
    """Final decision on a Renovate PR."""

    decision: Literal["clear", "breaking", "regression", "needs_human"]
    findings: list[Finding]
    summary: str
    risk: RiskAssessment | None = None


class RenovateReviewState(TypedDict):
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


class DeployScaffoldState(TypedDict):
    """State for Graph B: deployment scaffolder."""

    request: str
    thread_id: str  # Checkpoint key: based on issue_number or explicit id parameter
    provided_links: list[str]
    spec: dict
    service_evidence: Annotated[list[EvidenceItem], operator.add]
    helm_chart_found: bool
    helm_chart_ref: str | None
    conventions: str
    manifests: dict[str, str]
    review_passed: bool
    review_issues: list[str]
    retry_count: int
    pr_url: str | None
    _owner: str  # Internal: Gitea repo owner (optional)
    _repo: str  # Internal: Gitea repo name (optional)
    issue_number: int | None  # Internal: issue number if provided
