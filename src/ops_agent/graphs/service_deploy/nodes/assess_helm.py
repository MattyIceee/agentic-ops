"""Node: assess_helm - determines if upstream Helm chart exists."""

import json
import re
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from ops_agent.llm.personas import get_llm
from ops_agent.prompting import untrusted_data_instruction
from ops_agent.state import ServiceDeployState
from ops_agent.tools.fetch import get_fetch_tool
from ops_agent.tools.search import get_search_tool
from ops_agent.types import EvidenceItem

_ASSESS_HELM_SYSTEM = """\
You are a Helm chart assessor. Determine whether a well-maintained upstream Helm
chart exists for the given service.

Use web_search and fetch_url to:
1. Search Artifact Hub (artifacthub.io) for the service name.
2. Check the project's own repository for a charts/ directory.
3. Check Bitnami, bjw-s, or other well-known chart repositories.

Return a JSON object:
{
  "helm_chart_found": true or false,
  "helm_chart_ref": "chart reference string or null",
  "source_url": "url where found or null",
  "evidence": [{"source": "...", "url": "...", "text": "..."}]
}
"""


def assess_helm(state: ServiceDeployState) -> dict[str, Any]:
    """Tool-calling agent that determines whether an upstream Helm chart exists.

    On failure, logs error and assumes no Helm chart to allow downstream nodes to proceed.
    """
    try:
        spec = state.get("spec", {})
        name = spec.get("name", state["request"][:60])

        llm = get_llm("research")
        tools = [get_search_tool(), get_fetch_tool()]
        agent = create_agent(
            llm,
            tools,
            system_prompt=f"{_ASSESS_HELM_SYSTEM}\n\n{untrusted_data_instruction()}",
        )

        user_msg = (
            f"Assess whether a well-maintained upstream Helm chart exists for: {name}\n"
            "Return the JSON object described in your instructions."
        )

        result = agent.invoke({"messages": [HumanMessage(content=user_msg)]})
        output: str = result["messages"][-1].content

        helm_found = False
        helm_ref: str | None = None
        extra_evidence: list[EvidenceItem] = []

        try:
            json_match = re.search(r"\{.*\}", output, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                helm_found = bool(data.get("helm_chart_found", False))
                helm_ref = data.get("helm_chart_ref") or None
                for item in data.get("evidence", []):
                    if isinstance(item, dict) and "text" in item:
                        extra_evidence.append(
                            EvidenceItem(
                                source=item.get("source", "helm-research"),
                                url=item.get("url"),
                                text=str(item["text"]),
                            )
                        )
        except Exception:
            if output.strip():
                extra_evidence.append(
                    EvidenceItem(source="helm_research_output", url=None, text=output[:2000])
                )

        return {
            "helm_chart_found": helm_found,
            "helm_chart_ref": helm_ref,
            "service_evidence": extra_evidence,
        }
    except Exception as exc:
        import sys
        print(f"Warning: assess_helm failed: {exc}", file=sys.stderr)
        return {
            "helm_chart_found": False,
            "helm_chart_ref": None,
            "service_evidence": [],
        }
