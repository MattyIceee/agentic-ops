"""Finding, risk assessment, and verdict models."""

from typing import Literal

from pydantic import BaseModel


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
