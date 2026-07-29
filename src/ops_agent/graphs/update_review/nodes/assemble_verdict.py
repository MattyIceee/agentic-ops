"""Node: assemble_verdict - pure Python decision logic."""

import re
from typing import Any

from ops_agent.state import UpdateReviewState
from ops_agent.types import Finding, RiskAssessment, Verdict

# A leading component resembling a 4-digit year signals calendar versioning.
_CALVER_MIN_YEAR = 2000
_CALVER_MAX_YEAR = 2100


def _parse_version_components(version: str) -> list[int] | None:
    """Return the leading run of integer components of a version, or None.

    Stops at the first non-numeric component (e.g. ``ubu2604``, ``rc1``) so that
    ``10.11.11ubu2604-ls43`` parses to ``[10, 11, 11]``.
    """
    if not version:
        return None
    stripped = version.strip().lstrip("vV")
    nums: list[int] = []
    for part in re.split(r"[.\-_+]", stripped):
        m = re.match(r"\d+", part)
        if not m:
            break  # first non-numeric component ends the comparable prefix
        nums.append(int(m.group(0)))
    return nums or None


def _looks_like_calver(components: list[int]) -> bool:
    return bool(components) and _CALVER_MIN_YEAR <= components[0] <= _CALVER_MAX_YEAR


def _analyze_version_direction(current: str, new: str) -> str | None:
    """Deterministically flag a suspicious version *direction*.

    Returns ``"downgrade"`` (new numerically precedes current, same scheme),
    ``"scheme_change"`` (calendar vs semantic versioning mismatch — the trap that
    fools Renovate, e.g. ``2021.12.16`` sorting above ``10.11.11``), or ``None``.
    """
    cur = _parse_version_components(current)
    nxt = _parse_version_components(new)
    if not cur or not nxt:
        return None
    if _looks_like_calver(cur) != _looks_like_calver(nxt):
        return "scheme_change"
    if nxt < cur:
        return "downgrade"
    return None


def _regression_finding(direction: str, current: str, new: str) -> Finding:
    if direction == "downgrade":
        claim = (
            f"Version downgrade: {current} → {new} moves to an OLDER release."
        )
    else:  # scheme_change
        claim = (
            f"Version scheme change: {current} → {new}. The versioning scheme "
            "changes (calendar vs semantic), which Renovate frequently mis-orders "
            "— this is likely a downgrade to a stale release, not an upgrade."
        )
    return Finding(
        claim=claim,
        source="version-analysis",
        quote=f"{current} → {new}",
        category="regression",
    )


def assemble_verdict(state: UpdateReviewState) -> dict[str, Any]:
    """Build a Verdict from findings, deterministic checks, and reasoned risk.

    Precedence (highest wins):
      1. breaking   — a verbatim-grounded breaking change
      2. regression — a downgrade / stale target (deterministic or verbatim)
      3. needs_human — no evidence, OR the reasoned layer judges it could break
      4. clear      — evidence gathered and nothing above triggered
    """
    dep = state["dependency"]
    old_v = state.get("current_version", "")
    new_v = state.get("new_version", "")
    findings: list[Finding] = list(state.get("_findings", []) or [])
    evidence = state.get("evidence", [])
    risk: RiskAssessment | None = state.get("_risk")

    # Deterministic backstop: catch downgrades / scheme swaps the LLM may miss.
    direction = _analyze_version_direction(old_v, new_v)
    if direction and not any(f.category == "regression" for f in findings):
        findings.append(_regression_finding(direction, old_v, new_v))

    breaking = [f for f in findings if f.category == "breaking"]
    regression = [f for f in findings if f.category == "regression"]
    reasoned_risk = bool(risk and risk.could_break and risk.risk_level in ("medium", "high"))

    if breaking:
        decision = "breaking"
        summary = f"Found {len(breaking)} breaking change(s) in {dep} {old_v} → {new_v}."
    elif regression:
        decision = "regression"
        summary = (
            f"{dep} {old_v} → {new_v} looks like a downgrade or a move to a stale "
            "release. This is probably not a real upgrade — do not merge without review."
        )
    elif reasoned_risk:
        decision = "needs_human"
        summary = (
            f"No breaking change was proven for {dep} {old_v} → {new_v}, but the "
            f"evidence suggests {risk.risk_level} risk that it could break. "  # type: ignore[union-attr]
            "Manual review recommended."
        )
    elif not evidence:
        decision = "needs_human"
        summary = (
            f"No evidence could be gathered for {dep} {old_v} → {new_v}. "
            "Manual review required."
        )
    else:
        decision = "clear"
        summary = (
            f"No breaking changes found in {dep} {old_v} → {new_v} "
            f"based on {len(evidence)} evidence source(s)."
        )

    verdict = Verdict(
        decision=decision,  # type: ignore[arg-type]
        findings=findings,
        summary=summary,
        risk=risk,
    )
    return {"verdict": verdict}
