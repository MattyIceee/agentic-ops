"""Node: decide_namespace - choose the target namespace for the deployment.

Namespace selection needs information that isn't available at parse time — namely
the namespaces that already exist in the target repo. This node runs after
load_conventions and makes the decision explicit and auditable:

  1. If the user explicitly named a namespace, honor it.
  2. Otherwise, pick an existing namespace the service fits, or propose a new one
     named after the service/domain following the repo's conventions.

It never falls through to "default" — a deterministic guardrail rewrites any
empty/"default"/system namespace to the service name.
"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from ops_agent.llm.personas import get_llm
from ops_agent.state import ServiceDeployState

logger = logging.getLogger(__name__)

# Namespaces we refuse to auto-assign a new app into.
_SYSTEM_NAMESPACES = {"default", "kube-system", "kube-public", "kube-node-lease", "flux-system"}

_DECIDE_SYSTEM = """\
You are a Kubernetes namespace planner. Choose the single namespace a new service
should be deployed into, given the repository's existing namespaces and conventions.

Rules:
- Prefer an EXISTING namespace when the service logically belongs to an established
  group (e.g. shared "media", "monitoring", "databases" stacks).
- Otherwise propose a NEW namespace name following the repo's naming conventions —
  typically the service name or its logical domain, lowercase and hyphen-separated.
- NEVER choose "default" or a cluster system namespace (kube-system, flux-system, etc.).
- Set created_new=true only when the chosen namespace is not in the existing list.

Return a NamespaceDecision object.
"""


class NamespaceDecision(BaseModel):
    """LLM decision about which namespace to deploy into."""

    namespace: str
    created_new: bool
    reasoning: str


def _sanitize(name: str | None) -> str | None:
    """Return a usable namespace or None if it's empty/default/system."""
    if not name:
        return None
    cleaned = name.strip().lower()
    if not cleaned or cleaned in _SYSTEM_NAMESPACES:
        return None
    return cleaned


def decide_namespace(state: ServiceDeployState) -> dict[str, Any]:
    """Resolve spec.namespace, choosing from existing repo namespaces when unset.

    On any failure, falls back deterministically to the service name so the
    deployment never lands in "default".
    """
    spec = dict(state.get("spec", {}))
    service_name = spec.get("name") or "app"
    existing = state.get("existing_namespaces", []) or []

    # 1. Honor an explicit, non-system user choice.
    user_ns = _sanitize(spec.get("namespace"))
    if user_ns:
        spec["namespace"] = user_ns
        logger.info("Using user-specified namespace: %s", user_ns)
        return {"spec": spec}

    # 2. Ask the model to pick an existing namespace or propose a new one.
    chosen: str | None = None
    try:
        llm = get_llm("reason").with_structured_output(NamespaceDecision)
        existing_block = "\n".join(f"- {ns}" for ns in existing) if existing else "(none found)"
        messages = [
            SystemMessage(content=_DECIDE_SYSTEM),
            HumanMessage(
                content=(
                    f"Service name: {service_name}\n\n"
                    f"Existing namespaces in the repo:\n{existing_block}\n\n"
                    f"Repository conventions:\n{state.get('conventions', '')}\n\n"
                    "Choose the target namespace."
                )
            ),
        ]
        decision: NamespaceDecision = llm.invoke(messages)  # type: ignore[assignment]
        chosen = _sanitize(decision.namespace)
        if chosen:
            logger.info(
                "decide_namespace chose %r (created_new=%s): %s",
                chosen,
                decision.created_new,
                decision.reasoning,
            )
    except Exception as exc:
        logger.warning("decide_namespace LLM step failed: %s", exc)

    # 3. Deterministic guardrail — never land in default.
    if not chosen:
        chosen = _sanitize(service_name) or "apps"
        logger.info("decide_namespace falling back to service-derived namespace: %s", chosen)

    spec["namespace"] = chosen
    return {"spec": spec}
