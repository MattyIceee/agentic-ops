"""Node: extract_breaking_changes - structured extraction of quote-backed findings."""

import logging
from typing import Any

from ops_agent.llm.personas import get_llm
from ops_agent.prompting import build_agent_messages
from ops_agent.state import UpdateReviewState
from ops_agent.types import Finding, Findings

logger = logging.getLogger(__name__)

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


def extract_breaking_changes(state: UpdateReviewState) -> dict[str, Any]:
    """Structured extraction of breaking changes from gathered evidence.

    On failure, logs error and returns empty findings to allow downstream nodes to proceed.
    """
    try:
        evidence = state.get("evidence", [])
        if not evidence:
            return {}

        evidence_block = "\n\n---\n\n".join(
            f"[{e.source}] {e.url or ''}\n{e.text}" for e in evidence
        )

        llm = get_llm("extract")
        structured_llm = llm.with_structured_output(Findings)

        messages = build_agent_messages(
            system=_EXTRACT_SYSTEM,
            untrusted_blocks=[("evidence", evidence_block)],
            trusted_tail=(
                f"Dependency: {state['dependency']} "
                f"{state.get('current_version', '')} → {state.get('new_version', '')}\n\n"
                "Return a list of findings. Each must have claim, source, and quote."
            ),
        )

        result: Findings = structured_llm.invoke(messages)  # type: ignore[assignment]
        findings: list[Finding] = result.findings if result else []
        # Filter out any findings that somehow have an empty quote (defensive).
        findings = [f for f in (findings or []) if f.quote.strip()]

        return {"_findings": findings}
    except Exception as exc:
        logger.warning("extract_breaking_changes failed for %s: %s", state["dependency"], exc)
        return {}
