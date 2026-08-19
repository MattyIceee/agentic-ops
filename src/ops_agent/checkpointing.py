"""Checkpointing infrastructure for LangGraph using PostgreSQL."""

import functools
import logging
import os

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

logger = logging.getLogger(__name__)

ALLOWED_MSGPACK_MODULES = [
    ("ops_agent.types.findings", "Finding"),
    ("ops_agent.types.findings", "RiskAssessment"),
    ("ops_agent.types.findings", "Verdict"),
    ("ops_agent.types.evidence", "EvidenceItem"),
]


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
        logger.debug("Checkpointing disabled via CHECKPOINT_ENABLED=false")
        return None  # type: ignore[return-value]

    conn = psycopg.connect(
        host=os.getenv("CHECKPOINT_DB_HOST", "localhost"),
        port=int(os.getenv("CHECKPOINT_DB_PORT", "5432")),
        user=os.getenv("CHECKPOINT_DB_USER", "ops_agent"),
        password=os.getenv("CHECKPOINT_DB_PASSWORD", "ops_agent_dev_password"),
        dbname=os.getenv("CHECKPOINT_DB_NAME", "ops_agent_checkpoints"),
        autocommit=True,
    )
    logger.debug("Connected to checkpoint database")
    serde = JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_MSGPACK_MODULES)
    saver = PostgresSaver(conn, serde=serde)
    saver.setup()
    logger.debug("PostgreSQL checkpoint saver initialized")
    return saver


def clear_thread(thread_id: str) -> bool:
    """Delete all checkpoints for a thread so the next run starts fresh.

    Returns False if checkpointing is disabled (nothing to clear).
    """
    saver = get_checkpoint_saver()
    if saver is None:
        return False
    saver.delete_thread(thread_id)
    logger.info("Cleared checkpoint thread %s", thread_id)
    return True
