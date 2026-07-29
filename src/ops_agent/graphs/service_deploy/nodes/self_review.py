"""Node: self_review - checks generated manifests against conventions."""

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from ops_agent.llm.personas import get_llm
from ops_agent.state import ServiceDeployState

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
    """Check generated manifests against conventions; structured output."""
    manifests = state.get("manifests", {})
    if not manifests:
        return {"review_passed": False, "review_issues": ["No manifests were generated."]}

    manifest_block = "\n\n".join(
        f"### {fname}\n```yaml\n{content}\n```" for fname, content in manifests.items()
    )

    llm = get_llm("extract")
    structured_llm = llm.with_structured_output(ReviewResult)

    messages = [
        SystemMessage(content=_SELF_REVIEW_SYSTEM),
        HumanMessage(
            content=(
                f"Service spec: {json.dumps(state.get('spec', {}), indent=2)}\n\n"
                f"Conventions:\n{state.get('conventions', '')}\n\n"
                f"Generated manifests:\n{manifest_block}"
            )
        ),
    ]

    result: ReviewResult = structured_llm.invoke(messages)  # type: ignore[assignment]
    return {"review_passed": result.passed, "review_issues": result.issues}
