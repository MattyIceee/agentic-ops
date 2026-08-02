"""Node: classify_comment - decides how much of the graph a comment re-drives.

Only reached when triage found a new foreign comment. Reasoned (agentic) call:
does answering this comment require re-gathering evidence (``full`` → research),
or can the existing evidence be re-judged in light of the comment
(``light`` → assess_risk → assemble_verdict)?
"""

import logging
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from ops_agent.llm.personas import get_llm
from ops_agent.state import UpdateReviewState

logger = logging.getLogger(__name__)

_CLASSIFY_SYSTEM = """\
You triage follow-up comments on an automated dependency-update review. A prior \
review already gathered evidence and produced a verdict. Decide the smallest \
amount of work needed to address the new comment(s):

- "light": the comment questions, challenges, or asks you to reconsider the \
  existing verdict, and the evidence already gathered is sufficient to answer. \
  Re-judge without new research.
- "full": the comment introduces a new concern, points to a new source, or asks \
  about something the current evidence does not cover. Fresh research is needed.

Prefer "light" when the existing evidence can plausibly answer the comment.
Return scope ("light" or "full") and a one-line reason.
"""


class CommentScope(BaseModel):
    """Re-drive scope decision for a follow-up comment."""

    scope: Literal["light", "full"] = "light"
    reason: str = ""


def _comment_block(state: UpdateReviewState) -> str:
    inputs = state.get("new_inputs", []) or []
    if not inputs:
        return "(no comment text)"
    return "\n\n".join(f"@{c.get('author', '?')}: {c.get('text', '')}" for c in inputs)


def classify_comment(state: UpdateReviewState) -> dict[str, Any]:
    """Choose 'light' vs 'full' re-drive for the new comment(s)."""
    verdict = state.get("verdict")
    summary = verdict.summary if verdict else "(no prior verdict)"

    try:
        llm = get_llm("reason").with_structured_output(CommentScope)
        messages = [
            SystemMessage(content=_CLASSIFY_SYSTEM),
            HumanMessage(
                content=(
                    f"Dependency: {state['dependency']} "
                    f"{state.get('current_version', '')} → {state.get('new_version', '')}\n\n"
                    f"Prior verdict summary: {summary}\n\n"
                    f"New comment(s):\n{_comment_block(state)}\n\n"
                    "Decide the re-drive scope."
                )
            ),
        ]
        result: CommentScope = llm.invoke(messages)  # type: ignore[assignment]
        logger.info("classify_comment: %s — %s", result.scope, result.reason)
        return {"_route": result.scope}
    except Exception as exc:
        # Degrade safely to the cheaper path rather than sinking the run.
        logger.warning("classify_comment failed (%s); defaulting to light", exc)
        return {"_route": "light"}
