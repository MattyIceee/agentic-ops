"""Shared pytest fixtures for ops-agent tests."""

import pytest


@pytest.fixture(autouse=True)
def _test_settings(monkeypatch):
    """Provide required env vars and clear the settings lru_cache around each test.

    Settings fields with defaults (llamacpp_base_url, model_alias, etc.) stay at
    their defaults. Only the two required fields need env vars.
    """
    from ops_agent.config import get_settings

    monkeypatch.setenv("SEARXNG_URL", "http://searxng.test")
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    # Tests are hermetic: never try to connect to PostgreSQL. The checkpoint
    # tests override this with an in-memory saver explicitly.
    monkeypatch.setenv("CHECKPOINT_ENABLED", "false")
    # Trusted-reviewer gating (layer 7) is enabled for steering tests; these
    # logins are used as trusted reviewers across the graph tests.
    monkeypatch.setenv("TRUSTED_GITHUB_LOGINS", "human,mattyice")
    monkeypatch.setenv("STEERING_TRUSTED_ONLY", "true")

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
