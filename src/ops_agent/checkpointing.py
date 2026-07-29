"""Checkpointing infrastructure for LangGraph using PostgreSQL."""

import functools
import os

from langgraph.checkpoint.postgres import PostgresSaver


@functools.lru_cache(maxsize=1)
def get_checkpoint_saver() -> PostgresSaver:
    """Get or create a PostgreSQL checkpoint saver.

    Connection details are read from environment variables:
    - CHECKPOINT_DB_HOST: postgres host (default: localhost)
    - CHECKPOINT_DB_PORT: postgres port (default: 5432)
    - CHECKPOINT_DB_USER: postgres user (default: ops_agent)
    - CHECKPOINT_DB_PASSWORD: postgres password (default: ops_agent_dev_password)
    - CHECKPOINT_DB_NAME: postgres database (default: ops_agent_checkpoints)

    Returns None if checkpointing is disabled via CHECKPOINT_ENABLED=false.
    """
    if os.getenv("CHECKPOINT_ENABLED", "true").lower() == "false":
        return None  # type: ignore[return-value]

    conn_str = (
        f"postgresql://{os.getenv('CHECKPOINT_DB_USER', 'ops_agent')}:"
        f"{os.getenv('CHECKPOINT_DB_PASSWORD', 'ops_agent_dev_password')}@"
        f"{os.getenv('CHECKPOINT_DB_HOST', 'localhost')}:"
        f"{os.getenv('CHECKPOINT_DB_PORT', '5432')}/"
        f"{os.getenv('CHECKPOINT_DB_NAME', 'ops_agent_checkpoints')}"
    )
    return PostgresSaver(conn_str)
