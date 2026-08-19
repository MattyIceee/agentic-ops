"""Node: research_service - gathers service deployment evidence."""

import json
import logging
import re
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from ops_agent.llm.personas import get_llm
from ops_agent.state import ServiceDeployState
from ops_agent.tools.fetch import get_fetch_tool
from ops_agent.tools.search import get_search_tool
from ops_agent.types import EvidenceItem

logger = logging.getLogger(__name__)

_RESEARCH_SERVICE_SYSTEM = """\
You are a deployment researcher. Given a service name and any links provided by
the user, find the Docker image, exposed ports, required environment variables,
volumes, and any compose/install documentation.

Use web_search and fetch_url to:
1. Follow provided links to find installation docs, docker-compose examples, etc.
2. Locate the official container image on Docker Hub or the project's registry.
3. Find any example docker-compose.yml or Kubernetes deployment documentation.

Return a JSON list of objects with keys: source, url, text.
Do NOT speculate — only report what you actually retrieved.
"""


def research_service(state: ServiceDeployState) -> dict[str, Any]:
    """Tool-calling agent that gathers service deployment evidence.

    On failure, logs error and returns empty evidence to allow downstream nodes to proceed.
    """
    try:
        spec = state.get("spec", {})
        name = spec.get("name", state["request"][:60])
        links = state.get("provided_links", [])

        llm = get_llm("research")
        tools = [get_search_tool(), get_fetch_tool()]
        agent = create_agent(llm, tools, system_prompt=_RESEARCH_SERVICE_SYSTEM)

        link_block = "\n".join(f"- {url}" for url in links) if links else "(none provided)"
        user_msg = (
            f"Service to deploy: {name}\n"
            f"Provided links:\n{link_block}\n\n"
            "Find: docker image name, exposed ports, required env vars, volumes, "
            "and docker-compose or install documentation. "
            "Return a JSON list of objects with keys: source, url, text."
        )

        result = agent.invoke({"messages": [HumanMessage(content=user_msg)]})
        output: str = result["messages"][-1].content

        evidence: list[EvidenceItem] = []
        try:
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
            if output.strip():
                evidence.append(EvidenceItem(source="agent_output", url=None, text=output[:4000]))

        return {"service_evidence": evidence}
    except Exception as exc:
        logger.warning("research_service failed: %s", exc)
        return {"service_evidence": []}
