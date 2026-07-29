"""Langfuse tracing integration for ops-agent graphs.

Provides a simple wrapper to conditionally enable distributed tracing without
adding guard logic to every graph file. When tracing is disabled (default),
the handler is None and trace operations are no-ops.
"""

import logging
import uuid
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TraceContext:
    """Container for trace ID and handler. Handler is None when tracing disabled."""

    trace_id: str
    handler: Any  # langfuse.callback.CallbackHandler | None


def get_langfuse_handler(
    trace_name: str,
    tags: list[str] | None = None,
    metadata: dict | None = None,
) -> TraceContext:
    """Return a TraceContext with an initialized Langfuse handler, or None if disabled.

    Args:
        trace_name: Human-readable name for the trace (e.g., "update-review/owner/repo#42")
        tags: List of tags for filtering in Langfuse UI (optional)
        metadata: Initial metadata dict (optional)

    Returns:
        TraceContext with trace_id and handler (handler is None if tracing disabled)
    """
    from ops_agent.config import get_settings

    settings = get_settings()
    if not settings.langfuse_enabled:
        return TraceContext(trace_id=trace_name, handler=None)

    from langfuse.langchain import CallbackHandler

    trace_id = str(uuid.uuid4())
    handler = CallbackHandler()
    logger.debug("Initialized Langfuse trace: %s (id=%s)", trace_name, trace_id)
    return TraceContext(trace_id=trace_id, handler=handler)


def flush_langfuse() -> None:
    """Flush buffered trace events. Call before CLI process exits.

    This is necessary for short-lived processes where the SDK's background
    flush may not complete before the process terminates.
    """
    from ops_agent.config import get_settings

    if not get_settings().langfuse_enabled:
        return

    try:
        from langfuse import Langfuse

        Langfuse().flush()
        logger.debug("Flushed Langfuse traces")
    except Exception as exc:
        logger.warning("Failed to flush Langfuse traces: %s", exc)
