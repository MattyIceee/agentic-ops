"""Node: parse_request - parse user request into structured spec."""

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from ops_agent.llm.personas import get_llm
from ops_agent.state import ServiceDeployState

logger = logging.getLogger(__name__)

_PARSE_SYSTEM = """\
You are a deployment spec parser. Given a user request describing a service to
deploy, extract a structured specification and any URLs the user provided.

Return a ServiceSpec with:
- name: service name (lowercase, hyphen-separated, no spaces)
- image: docker image reference if explicitly mentioned, else null
- namespace: kubernetes namespace (default: "default")
- ports: list of integer port numbers the service exposes
- env: dict of environment variable names to values or descriptions
- volumes: list of volume mount paths or descriptions
- provided_links: list of any URLs mentioned in the request
"""


class ServiceSpec(BaseModel):
    """Parsed deployment specification from user request."""

    name: str
    image: str | None = None
    namespace: str = "default"
    ports: list[int] = []
    env: dict[str, str] = {}
    volumes: list[str] = []
    provided_links: list[str] = []


def parse_request(state: ServiceDeployState) -> dict[str, Any]:
    """Parse the user request into a structured spec dict and extract links.

    On failure, logs error and returns empty spec to allow downstream nodes to proceed.
    """
    try:
        llm = get_llm("instruct")
        structured_llm = llm.with_structured_output(ServiceSpec)

        messages = [
            SystemMessage(content=_PARSE_SYSTEM),
            HumanMessage(content=f"User request:\n{state['request']}"),
        ]

        result: ServiceSpec = structured_llm.invoke(messages)  # type: ignore[assignment]
        return {
            "spec": result.model_dump(exclude={"provided_links"}),
            "provided_links": result.provided_links,
        }
    except Exception as exc:
        logger.warning("parse_request failed: %s", exc)
        return {
            "spec": {"name": "unknown", "namespace": "default", "ports": [], "env": {}, "volumes": []},
            "provided_links": [],
        }
