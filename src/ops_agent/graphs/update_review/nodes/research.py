"""Node: research - gathers evidence about dependency updates."""

import json
import logging
import re
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from ops_agent.llm.personas import get_llm
from ops_agent.state import UpdateReviewState
from ops_agent.tools.fetch import get_fetch_tool
from ops_agent.tools.search import get_search_tool
from ops_agent.types import EvidenceItem

logger = logging.getLogger(__name__)

_RESEARCH_SYSTEM = """\
You are a dependency-update researcher. Your job is to gather evidence about \
whether a version bump is safe — covering both breaking changes AND whether the \
target version is a sensible one to move to.

Use the web_search and fetch_url tools to find:
- The changelog or release notes for the new version
- Any migration guides
- GitHub/GitLab issues mentioning breaking changes for this bump
- The RELEASE DATE of the new version and of the current version
- The LATEST / recommended version available, and whether the new version is it
- Whether the new tag is deprecated, yanked, "unstable"/"nightly", or no longer \
  listed as supported

Watch for regressions: an update that moves to an OLDER, stale, or unmaintained \
release (for example because the versioning scheme changed, like a calendar date \
tag sorting above a semver tag) is a serious problem, not an upgrade.

For every piece of evidence you find, record its source URL and relevant text. \
Do NOT speculate — only report what you actually retrieved.
"""


def research(state: UpdateReviewState) -> dict[str, Any]:
    """Tool-calling agent that gathers changelog/release-note evidence.

    On failure, logs error and returns empty evidence to allow downstream nodes to proceed.
    """
    dep = state["dependency"]
    old_v = state.get("current_version") or "unknown"
    new_v = state.get("new_version") or "unknown"

    try:
        llm = get_llm("research")
        tools = [get_search_tool(), get_fetch_tool()]

        agent = create_agent(llm, tools, system_prompt=_RESEARCH_SYSTEM)

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
    except Exception as exc:
        logger.warning("research node failed for %s %s → %s: %s", dep, old_v, new_v, exc)
        return {"evidence": []}
