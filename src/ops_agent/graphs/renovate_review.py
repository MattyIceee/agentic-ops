"""Graph A: Renovate PR reviewer.

Linear graph: ingest_pr → research → extract_breaking_changes → assemble_verdict → post_review → END

- ingest_pr:               reads PR metadata + diff via GiteaClient (no LLM)
- research:                tool-calling agent (research persona) gathers evidence
- extract_breaking_changes: structured extraction (extract persona, no tools)
- assemble_verdict:        pure Python decision logic
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


def ingest_pr(state: RenovateReviewState) -> dict[str, Any]:
    """Read PR metadata and diff from Gitea; parse dependency + versions."""
    owner: str = state["_owner"]  # type: ignore[typeddict-item]
    repo: str = state["_repo"]  # type: ignore[typeddict-item]
    index: int = state["pr_index"]

    client = GiteaClient()
    try:
        pr = client.get_pr(owner, repo, index)
        diff = client.get_pr_diff(owner, repo, index)
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
whether a version bump introduces breaking changes.

Use the web_search and fetch_url tools to find:
- The changelog or release notes for the new version
- Any migration guides
- GitHub/GitLab issues mentioning breaking changes for this bump

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
You are a precise breaking-change extractor. You receive evidence snippets and \
must identify any breaking changes.

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

    # Store findings in a transient key; assemble_verdict reads it.
    return {"_findings": findings}  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Node 4: assemble_verdict  (pure Python, no LLM)
# ---------------------------------------------------------------------------


def assemble_verdict(state: RenovateReviewState) -> dict[str, Any]:
    """Build a Verdict from findings and evidence — no LLM involved."""
    findings: list[Finding] = state.get("_findings", [])  # type: ignore[assignment]
    evidence = state.get("evidence", [])

    if findings:
        decision = "breaking"
        summary = (
            f"Found {len(findings)} breaking change(s) in "
            f"{state['dependency']} {state.get('current_version', '')} → "
            f"{state.get('new_version', '')}."
        )
    elif not evidence:
        decision = "needs_human"
        summary = (
            f"No evidence could be gathered for {state['dependency']} "
            f"{state.get('current_version', '')} → {state.get('new_version', '')}. "
            "Manual review required."
        )
    else:
        decision = "clear"
        summary = (
            f"No breaking changes found in {state['dependency']} "
            f"{state.get('current_version', '')} → {state.get('new_version', '')} "
            f"based on {len(evidence)} evidence source(s)."
        )

    verdict = Verdict(decision=decision, findings=findings, summary=summary)  # type: ignore[arg-type]
    return {"verdict": verdict}


# ---------------------------------------------------------------------------
# Node 5: post_review
# ---------------------------------------------------------------------------

_RATING_LINE = "\n\n*Renovate Merge Confidence: {rating}*"


def _build_comment(state: RenovateReviewState) -> str:
    verdict: Verdict = state["verdict"]  # type: ignore[assignment]
    dep = state["dependency"]
    old_v = state.get("current_version", "")
    new_v = state.get("new_version", "")
    rating = state.get("renovate_rating")

    if verdict.decision == "breaking":
        lines = [
            f"## :warning: Breaking Changes Detected — {dep} {old_v} → {new_v}",
            "",
            verdict.summary,
            "",
            "### Findings",
        ]
        for f in verdict.findings:
            lines += [
                f"- **{f.claim}**",
                f"  - Source: `{f.source}`",
                f"  - Quote: > {f.quote}",
            ]
        lines += ["", "Please review before merging."]
    elif verdict.decision == "needs_human":
        lines = [
            f"## :question: Insufficient Evidence — {dep} {old_v} → {new_v}",
            "",
            verdict.summary,
            "",
            "Could not gather enough evidence to assess breaking changes. "
            "Manual review is required before merging.",
        ]
    else:
        lines = [
            f"## :white_check_mark: No Breaking Changes — {dep} {old_v} → {new_v}",
            "",
            verdict.summary,
        ]

    comment = "\n".join(lines)
    if rating:
        comment += _RATING_LINE.format(rating=rating)

    comment += "\n\n---\n*Posted by ops-agent*"
    return comment


def post_review(state: RenovateReviewState) -> dict[str, Any]:
    """Post the verdict as a Gitea PR comment."""
    owner: str = state["_owner"]  # type: ignore[typeddict-item]
    repo: str = state["_repo"]  # type: ignore[typeddict-item]
    index: int = state["pr_index"]

    comment = _build_comment(state)

    client = GiteaClient()
    try:
        client.post_issue_comment(owner, repo, index, comment)
    finally:
        client.close()

    return {"posted": True}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_renovate_graph() -> Any:
    """Build and compile the Renovate PR review graph.

    The graph is linear:
      ingest_pr → research → extract_breaking_changes → assemble_verdict → post_review → END
    """
    graph = StateGraph(RenovateReviewState)

    graph.add_node("ingest_pr", ingest_pr)
    graph.add_node("research", research)
    graph.add_node("extract_breaking_changes", extract_breaking_changes)
    graph.add_node("assemble_verdict", assemble_verdict)
    graph.add_node("post_review", post_review)

    graph.set_entry_point("ingest_pr")
    graph.add_edge("ingest_pr", "research")
    graph.add_edge("research", "extract_breaking_changes")
    graph.add_edge("extract_breaking_changes", "assemble_verdict")
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
        "verdict": None,
        "posted": False,
    }
    final_state = compiled.invoke(initial_state)
    return final_state["verdict"]
