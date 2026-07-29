"""Node: post_review - posts verdict as Gitea PR comment."""

from typing import Any

from ops_agent.state import UpdateReviewState
from ops_agent.tools.gitea import GiteaClient
from ops_agent.types import Finding, RiskAssessment, Verdict

_RATING_LINE = "\n\n*Renovate Merge Confidence: {rating}*"


def _render_findings(findings: list[Finding], category: str) -> list[str]:
    lines: list[str] = []
    for f in findings:
        if f.category != category:
            continue
        lines += [
            f"- **{f.claim}**",
            f"  - Source: `{f.source}`",
            f"  - Quote: > {f.quote}",
        ]
    return lines


def _render_risk(risk: RiskAssessment | None) -> list[str]:
    """Render the reasoned (non-verbatim) risk assessment, if it has an opinion."""
    if not risk or (not risk.could_break and risk.risk_level in ("none", "low")):
        return []
    lines = [
        "",
        f"### Reasoned Risk — {risk.risk_level}",
        "",
        f"_This is an inferred judgment, not a verbatim finding._ {risk.rationale}".strip(),
    ]
    if risk.signals:
        lines += ["", "Signals:"] + [f"- {s}" for s in risk.signals]
    return lines


def _render_evidence(state: UpdateReviewState) -> list[str]:
    evidence = state.get("evidence", [])
    if not evidence:
        return []
    lines = ["", "### Release Notes", ""]
    for e in evidence:
        label = f"[{e.source}]({e.url})" if e.url else e.source
        lines += ["<details>", f"<summary>{label}</summary>", "", e.text, "", "</details>"]
    return lines


def _build_comment(state: UpdateReviewState) -> str:
    verdict: Verdict = state["verdict"]  # type: ignore[assignment]
    dep = state["dependency"]
    old_v = state.get("current_version", "")
    new_v = state.get("new_version", "")
    rating = state.get("renovate_rating")

    if verdict.decision == "breaking":
        lines = [
            f"## ❌ Breaking Changes Detected — {dep} {old_v} → {new_v}",
            "",
            verdict.summary,
            "",
            "### Findings",
            *_render_findings(verdict.findings, "breaking"),
        ]
        regression = _render_findings(verdict.findings, "regression")
        if regression:
            lines += ["", "### Version Concerns", *regression]
        lines += _render_risk(verdict.risk)
        lines += ["", "Please review before merging."]
    elif verdict.decision == "regression":
        lines = [
            f"## ⚠️ Possible Downgrade / Stale Version — {dep} {old_v} → {new_v}",
            "",
            verdict.summary,
            "",
            "### Version Concerns",
            *_render_findings(verdict.findings, "regression"),
        ]
        lines += _render_risk(verdict.risk)
        lines += _render_evidence(state)
        lines += [
            "",
            "Do NOT merge without confirming this is an intentional version change.",
        ]
    elif verdict.decision == "needs_human":
        lines = [
            f"## ⚠️ Needs Human Review — {dep} {old_v} → {new_v}",
            "",
            verdict.summary,
        ]
        lines += _render_risk(verdict.risk)
        lines += _render_evidence(state)
        lines += ["", "Manual review is required before merging."]
    else:
        lines = [
            f"## ✅ No Breaking Changes — {dep} {old_v} → {new_v}",
            "",
            verdict.summary,
        ]
        lines += _render_risk(verdict.risk)
        lines += _render_evidence(state)

    comment = "\n".join(lines)
    if rating:
        comment += _RATING_LINE.format(rating=rating)

    comment += "\n\n---\n*Posted by ops-agent*"
    return comment


def post_review(state: UpdateReviewState) -> dict[str, Any]:
    """Post the verdict as a Gitea PR comment."""
    import sys

    owner: str = state["_owner"]  # type: ignore[typeddict-item]
    repo: str = state["_repo"]  # type: ignore[typeddict-item]
    index: int = state["pr_index"]

    comment = _build_comment(state)

    client = GiteaClient()
    try:
        client.post_issue_comment(owner, repo, index, comment)
        return {"posted": True}
    except Exception as exc:
        print(
            f"\nWarning: could not post Gitea comment on {owner}/{repo}#{index}: {exc}",
            file=sys.stderr,
        )
        print("\n--- Verdict (not posted) ---\n", file=sys.stderr)
        print(comment, file=sys.stderr)
        return {"posted": False}
    finally:
        client.close()
