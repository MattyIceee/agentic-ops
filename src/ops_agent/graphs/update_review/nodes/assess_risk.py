"""Node: assess_risk - reasoned risk judgment layered on evidence."""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ops_agent.llm.personas import get_llm
from ops_agent.state import UpdateReviewState
from ops_agent.types import Finding, RiskAssessment

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


def assess_risk(state: UpdateReviewState) -> dict[str, Any]:
    """Reasoned, non-verbatim risk judgment layered on top of the evidence.

    Complements extract_breaking_changes: that node only speaks when it
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
