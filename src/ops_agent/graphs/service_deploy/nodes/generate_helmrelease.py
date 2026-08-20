"""Node: generate_helmrelease - generates Flux HelmRelease + values.yaml."""

import json
import logging
import re
from typing import Any

from ops_agent.llm.personas import get_llm
from ops_agent.prompting import build_agent_messages
from ops_agent.state import ServiceDeployState
from ops_agent.types import EvidenceItem

logger = logging.getLogger(__name__)

_HELMRELEASE_SYSTEM = """\
You are a Flux HelmRelease generator. Generate Kubernetes manifests to deploy a
service using a Flux HelmRelease and a values.yaml override file.

Follow the conventions provided. Use the service evidence for image tag, ports,
env variables, and volume details. Deploy into the namespace given in the spec's
"namespace" field — set it on every namespaced resource. Never use "default".
Return ONLY valid YAML — no prose.

Output a JSON object mapping filename → file content:
{
  "helmrelease.yaml": "...",
  "values.yaml": "..."
}
"""


def _format_evidence(evidence: list[EvidenceItem]) -> str:
    if not evidence:
        return "(no evidence gathered)"
    return "\n\n---\n\n".join(
        f"[{e.source}] {e.url or ''}\n{e.text[:1000]}" for e in evidence
    )


def _extract_manifests(llm_output: str) -> dict[str, str]:
    """Parse a JSON object mapping filename→YAML content from LLM output."""
    try:
        json_match = re.search(r"\{.*\}", llm_output, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            if isinstance(data, dict) and all(isinstance(v, str) for v in data.values()):
                return data
    except Exception:
        logger.debug("no JSON object found in LLM output; storing raw output")
    return {"output.yaml": llm_output}


def generate_helmrelease(state: ServiceDeployState) -> dict[str, Any]:
    """Generate Flux HelmRelease + values.yaml manifests."""
    spec = state.get("spec", {})
    review_issues = state.get("review_issues", [])
    issue_block = "\n".join(f"- {i}" for i in review_issues) if review_issues else ""
    steering_text = "\n".join(
        f"@{c.get('author', '?')}: {c.get('text', '')}" for c in state.get("new_inputs", [])
    )

    llm = get_llm("coding")
    messages = build_agent_messages(
        system=_HELMRELEASE_SYSTEM,
        untrusted_blocks=[
            ("service_spec", json.dumps(spec, indent=2)),
            ("helm_chart", state.get("helm_chart_ref") or "unknown"),
            ("conventions", state.get("conventions", "")),
            ("service_evidence", _format_evidence(state.get("service_evidence", []))),
            ("previous_review_issues", issue_block),
            ("pr_steering", steering_text),
        ],
        trusted_tail="Generate the HelmRelease and values.yaml.",
    )

    response = llm.invoke(messages)
    content: str = response.content if hasattr(response, "content") else str(response)
    manifests = _extract_manifests(content)

    retry_increment = 1 if review_issues else 0
    return {
        "manifests": manifests,
        "retry_count": state.get("retry_count", 0) + retry_increment,
        "review_issues": [],
    }
