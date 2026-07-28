"""Graph A: Renovate PR reviewer.

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

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

from ops_agent.llm.personas import get_llm
from ops_agent.state import (
    EvidenceItem,
    Finding,
    Findings,
    RenovateReviewState,
    RiskAssessment,
    Verdict,
)
from ops_agent.tools.fetch import get_fetch_tool
from ops_agent.tools.gitea import GiteaClient
from ops_agent.tools.search import get_search_tool

# ---------------------------------------------------------------------------
# Node 1: ingest_pr
# ---------------------------------------------------------------------------

_MERGE_CONFIDENCE_RE = re.compile(
    r"merge\s+confidence[^\n]*?:\s*([^\n]+)", re.IGNORECASE
)

# A version token following a `:` (docker tag) or `=`/`:` assignment, e.g.
#   image = "lscr.io/linuxserver/jellyfin:10.11.11"
#   tag: "10.11.11"
_DIFF_VERSION_RE = re.compile(r"""[:=]\s*["']?v?([0-9][0-9A-Za-z.\-_+]*)""")


def _extract_current_version_from_diff(diff: str, new_version: str) -> str:
    """Best-effort recovery of the *current* version from a Renovate diff.

    Renovate PR titles for Docker tags often read "... to vX" with no "from",
    so the outgoing version only exists in the diff's removed (``-``) lines.
    We pair the removed line with the added (``+``) line carrying ``new_version``
    and pull the version token off the removed line.
    """
    if not diff:
        return ""

    removed: list[str] = []
    added: list[str] = []
    for line in diff.splitlines():
        if line.startswith(("---", "+++")):
            continue  # file headers, not content
        if line.startswith("-"):
            removed.append(line[1:])
        elif line.startswith("+"):
            added.append(line[1:])

    # Prefer the removed line whose added counterpart mentions new_version.
    for rem, add in zip(removed, added):
        if new_version and new_version in add:
            m = _DIFF_VERSION_RE.search(rem)
            if m:
                return m.group(1)

    # Fallback: first version-looking token on any removed line that isn't the
    # new version itself.
    for rem in removed:
        m = _DIFF_VERSION_RE.search(rem)
        if m and m.group(1) != new_version:
            return m.group(1)

    return ""


def ingest_pr(state: RenovateReviewState) -> dict[str, Any]:
    """Read PR metadata and diff from Gitea; parse dependency + versions."""
    owner: str = state["_owner"]  # type: ignore[typeddict-item]
    repo: str = state["_repo"]  # type: ignore[typeddict-item]
    index: int = state["pr_index"]

    client = GiteaClient()
    try:
        pr = client.get_pr(owner, repo, index)
        diff = client.get_pr_diff(owner, repo, index)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch PR {owner}/{repo}#{index} from Gitea: {exc}"
        ) from exc
    finally:
        client.close()

    title: str = pr.get("title", "")
    body: str = pr.get("body", "") or ""

    # Renovate PR titles look like: "chore(deps): update dependency foo to v1.2.3"
    # We extract dependency name + versions with a best-effort regex.
    dep_match = re.search(
        r"update\s+(?:dependency\s+)?([^\s]+)\s+(?:from\s+v?(\S+)\s+)?to\s+v?(\S+)",
        title,
        re.IGNORECASE,
    )
    dependency = dep_match.group(1) if dep_match else title
    current_version = dep_match.group(2) if (dep_match and dep_match.group(2)) else ""
    new_version = dep_match.group(3) if dep_match else ""

    # The title frequently lacks the outgoing version ("... to vX"); recover it
    # from the diff so downstream downgrade/scheme checks have both endpoints.
    if not current_version:
        current_version = _extract_current_version_from_diff(diff, new_version)

    # Pull Renovate's Merge Confidence rating from the PR body if present.
    mc_match = _MERGE_CONFIDENCE_RE.search(body)
    renovate_rating = mc_match.group(1).strip() if mc_match else None

    return {
        "dependency": dependency,
        "current_version": current_version,
        "new_version": new_version,
        "diff": diff,
        "renovate_rating": renovate_rating,
    }


# ---------------------------------------------------------------------------
# Node 2: research
# ---------------------------------------------------------------------------

_RESEARCH_SYSTEM = """\
You are a dependency-update researcher. Your job is to gather evidence about \
whether a version bump is safe — covering both breaking changes AND whether the \
target version is a sensible one to move to.

Use the web_search and fetch_url tools to find:
- The changelog or release notes for the new version
- Any migration guides
- GitHub/GitLab issues mentioning breaking changes for this bump
- The RELEASE DATE of the new version and of the current version
- The LATEST / recommended version available, and whether the new version is it
- Whether the new tag is deprecated, yanked, "unstable"/"nightly", or no longer \
  listed as supported

Watch for regressions: an update that moves to an OLDER, stale, or unmaintained \
release (for example because the versioning scheme changed, like a calendar date \
tag sorting above a semver tag) is a serious problem, not an upgrade.

For every piece of evidence you find, record its source URL and relevant text. \
Do NOT speculate — only report what you actually retrieved.
"""


def research(state: RenovateReviewState) -> dict[str, Any]:
    """Tool-calling agent that gathers changelog/release-note evidence."""
    dep = state["dependency"]
    old_v = state.get("current_version") or "unknown"
    new_v = state.get("new_version") or "unknown"

    llm = get_llm("research")
    tools = [get_search_tool(), get_fetch_tool()]

    agent = create_react_agent(llm, tools, prompt=_RESEARCH_SYSTEM)

    user_msg = (
        f"Research the version bump: {dep} {old_v} → {new_v}.\n"
        f"Diff preview (first 2000 chars):\n{state.get('diff', '')[:2000]}\n\n"
        "Find and retrieve changelogs, release notes, and migration docs. "
        "Return a JSON list of objects with keys: source, url, text."
    )

    result = agent.invoke({"messages": [HumanMessage(content=user_msg)]})
    last_msg = result["messages"][-1]
    output: str = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    # Parse evidence items from agent output (best-effort JSON extraction).
    evidence: list[EvidenceItem] = []
    try:
        import json

        # Look for a JSON array anywhere in the output.
        json_match = re.search(r"\[.*\]", output, re.DOTALL)
        if json_match:
            items = json.loads(json_match.group(0))
            for item in items:
                if isinstance(item, dict) and "text" in item:
                    evidence.append(
                        EvidenceItem(
                            source=item.get("source", "web"),
                            url=item.get("url"),
                            text=str(item["text"]),
                        )
                    )
    except Exception:
        # If parsing fails, store the raw output as a single evidence item.
        if output.strip():
            evidence.append(
                EvidenceItem(source="agent_output", url=None, text=output[:4000])
            )

    return {"evidence": evidence}


# ---------------------------------------------------------------------------
# Node 3: extract_breaking_changes
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM = """\
You are a precise risk extractor. You receive evidence snippets and must \
identify concrete, quote-backed problems with a version update.

Emit two kinds of findings, set via the `category` field:
- category="breaking": a documented breaking change — a removed/renamed API or \
  config, a required migration, a changed default, dropped platform support, etc.
- category="regression": evidence that the NEW version is OLDER, stale, \
  deprecated, yanked, unmaintained, or otherwise a downgrade relative to the \
  current version or the latest release (e.g. a release dated years before the \
  current one, a tag not listed as supported, an "unstable"/"nightly" build).

Rules:
- Only emit a Finding when you have a verbatim quote from the evidence that \
  supports the claim.
- The quote field MUST be copied word-for-word from the evidence text.
- If you have no quote, do not emit the finding.
- Never fabricate or infer — ground every claim in the provided text.
- Do not include confidence scores.
"""


def extract_breaking_changes(state: RenovateReviewState) -> dict[str, Any]:
    """Structured extraction of breaking changes from gathered evidence."""
    evidence = state.get("evidence", [])
    if not evidence:
        return {}

    evidence_block = "\n\n---\n\n".join(
        f"[{e.source}] {e.url or ''}\n{e.text}" for e in evidence
    )

    llm = get_llm("extract")
    structured_llm = llm.with_structured_output(Findings)

    messages = [
        SystemMessage(content=_EXTRACT_SYSTEM),
        HumanMessage(
            content=(
                f"Dependency: {state['dependency']} "
                f"{state.get('current_version', '')} → {state.get('new_version', '')}\n\n"
                f"Evidence:\n{evidence_block}\n\n"
                "Return a list of findings. Each must have claim, source, and quote."
            )
        ),
    ]

    result: Findings = structured_llm.invoke(messages)  # type: ignore[assignment]
    findings: list[Finding] = result.findings if result else []
    # Filter out any findings that somehow have an empty quote (defensive).
    findings = [f for f in (findings or []) if f.quote.strip()]

    return {"_findings": findings}


# ---------------------------------------------------------------------------
# Node 4: assess_risk  (reasoned judgment — allowed to infer)
# ---------------------------------------------------------------------------

_RISK_SYSTEM = """\
You are a cautious dependency-update risk assessor. You are given the evidence \
gathered about a version bump plus any verbatim breaking-change findings that \
were already extracted.

Reason about whether this update COULD introduce breaking changes or operational \
risk, even when no single sentence proves it. Weigh signals such as: a major \
version jump, removed or renamed config/APIs, changed default behavior, database \
or schema migrations, dropped platform support, a target release that is much \
older or unmaintained than the current one, or release notes that are ambiguous \
or missing.

This is a judgment call, not a verbatim extraction — you MAY infer, and you \
SHOULD express your confidence. Be honest about uncertainty and prefer flagging \
for human review when genuinely unsure. Do NOT invent specific breaking changes \
that the evidence does not support.

Return: could_break (bool), risk_level (none|low|medium|high), a short rationale, \
and a list of the concrete signals you based the judgment on.
"""


def assess_risk(state: RenovateReviewState) -> dict[str, Any]:
    """Reasoned, non-verbatim risk judgment layered on top of the evidence.

    Complements :func:`extract_breaking_changes`: that node only speaks when it
    has a smoking-gun quote; this one weighs the whole picture and may infer.
    """
    evidence = state.get("evidence", [])
    if not evidence:
        return {}  # nothing to reason over — assemble_verdict handles this

    findings: list[Finding] = state.get("_findings", []) or []
    evidence_block = "\n\n---\n\n".join(
        f"[{e.source}] {e.url or ''}\n{e.text}" for e in evidence
    )
    findings_block = (
        "\n".join(f"- [{f.category}] {f.claim}" for f in findings)
        if findings
        else "(none extracted)"
    )

    llm = get_llm("reason")
    structured_llm = llm.with_structured_output(RiskAssessment)

    messages = [
        SystemMessage(content=_RISK_SYSTEM),
        HumanMessage(
            content=(
                f"Dependency: {state['dependency']} "
                f"{state.get('current_version', '')} → {state.get('new_version', '')}\n\n"
                f"Verbatim findings already extracted:\n{findings_block}\n\n"
                f"Evidence:\n{evidence_block}\n\n"
                "Give your reasoned risk assessment."
            )
        ),
    ]

    try:
        risk: RiskAssessment = structured_llm.invoke(messages)  # type: ignore[assignment]
    except Exception:
        # Never let the reasoning layer sink the run; degrade to "no opinion".
        return {}

    return {"_risk": risk}


# ---------------------------------------------------------------------------
# Node 5: assemble_verdict  (pure Python, no LLM)
# ---------------------------------------------------------------------------

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


def assemble_verdict(state: RenovateReviewState) -> dict[str, Any]:
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


# ---------------------------------------------------------------------------
# Node 6: post_review
# ---------------------------------------------------------------------------

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


def _render_evidence(state: RenovateReviewState) -> list[str]:
    evidence = state.get("evidence", [])
    if not evidence:
        return []
    lines = ["", "### Release Notes", ""]
    for e in evidence:
        label = f"[{e.source}]({e.url})" if e.url else e.source
        lines += ["<details>", f"<summary>{label}</summary>", "", e.text, "", "</details>"]
    return lines


def _build_comment(state: RenovateReviewState) -> str:
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


def post_review(state: RenovateReviewState) -> dict[str, Any]:
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


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_renovate_graph() -> Any:
    """Build and compile the Renovate PR review graph.

    The graph is linear:
      ingest_pr → research → extract_breaking_changes → assess_risk
        → assemble_verdict → post_review → END
    """
    graph = StateGraph(RenovateReviewState)

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

    return graph.compile()


# ---------------------------------------------------------------------------
# Public run helper
# ---------------------------------------------------------------------------


def run(pr_index: int, owner: str, repo: str) -> Verdict:
    """Run the full Renovate review graph for a single PR.

    Returns the Verdict from the final state.
    """
    compiled = build_renovate_graph()
    initial_state: dict[str, Any] = {
        "pr_index": pr_index,
        "_owner": owner,
        "_repo": repo,
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
    final_state = compiled.invoke(initial_state)
    return final_state["verdict"]
