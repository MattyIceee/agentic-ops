"""Node: self_review - checks generated manifests against conventions."""

import json
import logging
from typing import Any

from pydantic import BaseModel

from ops_agent.llm.personas import get_llm
from ops_agent.prompting import build_agent_messages
from ops_agent.state import ServiceDeployState

logger = logging.getLogger(__name__)

_SELF_REVIEW_SYSTEM = """\
You are a Kubernetes manifest reviewer. Check the generated manifests against
the provided conventions and this checklist:

- All required fields are present (apiVersion, kind, metadata.name, etc.)
- Namespace is set and matches the spec
- Container image reference is not a placeholder ("unknown", "TODO", etc.)
- Port numbers in the Service selector match those in the Deployment
- Labels and selectors are consistent between Deployment and Service
- No raw secret values in env (use secretKeyRef or separate Secret objects)
- Follows the repository conventions

Return ONLY a JSON object: {"passed": true/false, "issues": ["issue 1", ...]}
If there are no issues, return {"passed": true, "issues": []}
"""


class ReviewResult(BaseModel):
    """Outcome of the self-review step."""

    passed: bool
    issues: list[str]


def self_review(state: ServiceDeployState) -> dict[str, Any]:
    """Check generated manifests against conventions; structured output.

    On failure, logs error and returns negative review to allow retries.
    """
    try:
        manifests = state.get("manifests", {})
        if not manifests:
            return {"review_passed": False, "review_issues": ["No manifests were generated."]}

        manifest_block = "\n\n".join(
            f"### {fname}\n```yaml\n{content}\n```" for fname, content in manifests.items()
        )

        llm = get_llm("extract")
        structured_llm = llm.with_structured_output(ReviewResult)

        messages = build_agent_messages(
            system=_SELF_REVIEW_SYSTEM,
            untrusted_blocks=[
                ("spec", json.dumps(state.get("spec", {}), indent=2)),
                ("conventions", state.get("conventions", "")),
                ("manifests", manifest_block),
            ],
            trusted_tail=(
                "Review the generated manifests and report passed/issues as a JSON "
                'object: {"passed": true/false, "issues": [...]}.'
            ),
        )

        result: ReviewResult = structured_llm.invoke(messages)  # type: ignore[assignment]
        return {"review_passed": result.passed, "review_issues": result.issues}
    except Exception as exc:
        logger.error("self_review failed: %s", exc)
        return {"review_passed": False, "review_issues": [f"Review failed: {exc}"]}
