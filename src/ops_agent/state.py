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
    """A single breaking-change claim grounded in a verbatim quote from evidence."""

    claim: str
    source: str
    quote: str


class Findings(BaseModel):
    """Container for a list of Findings.

    Structured-output backends (``with_structured_output``) require a schema
    class — a bare ``list[Finding]`` generic is rejected — so extraction targets
    this wrapper and reads ``.findings``.
    """

    findings: list[Finding] = []


class Verdict(BaseModel):
    """Final decision on a Renovate PR."""

    decision: Literal["clear", "breaking", "needs_human"]
    findings: list[Finding]
    summary: str


class RenovateReviewState(TypedDict):
    """State for Graph A: Renovate PR reviewer."""

    pr_index: int
    dependency: str
    current_version: str
    new_version: str
    diff: str
    renovate_rating: str | None
    evidence: Annotated[list[EvidenceItem], operator.add]
    verdict: Verdict | None
    posted: bool


class DeployScaffoldState(TypedDict):
    """State for Graph B: deployment scaffolder."""

    request: str
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
