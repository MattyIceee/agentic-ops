"""Tests for tools: search, fetch, github.

GitHubClient tests mock the githubkit transport so no network is required.
"""

from __future__ import annotations

# ─────────────────────────── search / fetch tools ──────────────────────────

def test_get_search_tool_is_invocable():
    from ops_agent.tools.search import get_search_tool

    tool = get_search_tool()
    assert tool.name == "web_search"
    assert hasattr(tool, "invoke")


def test_get_fetch_tool_is_invocable():
    from ops_agent.tools.fetch import get_fetch_tool

    tool = get_fetch_tool()
    assert hasattr(tool, "invoke")


def test_fetch_domain_allowed_defaults_true():
    from ops_agent.tools.fetch import _domain_allowed

    assert _domain_allowed("https://example.com/x") is True
    # Non-http schemes are always refused regardless of allow-list.
    assert _domain_allowed("file:///etc/passwd") is False
    assert _domain_allowed("data:text/plain,hi") is False
    assert _domain_allowed("") is False


def test_fetch_domain_allowed_respects_allowlist(monkeypatch):
    from ops_agent.config import get_settings
    from ops_agent.tools.fetch import _domain_allowed

    monkeypatch.setenv("FETCH_ALLOWED_DOMAINS", "github.com,*.github.io")
    get_settings.cache_clear()
    try:
        assert _domain_allowed("https://github.com/owner/repo") is True
        assert _domain_allowed("https://docs.github.io/guide") is True
        assert _domain_allowed("https://sub.docs.github.io/x") is True
        assert _domain_allowed("https://example.com/") is False
    finally:
        get_settings.cache_clear()


# ─────────────────────── GitHubClient endpoint test ────────────────────────

def test_github_client_get_pr_calls_correct_endpoint(monkeypatch):
    """GitHubClient.get_pr must hit the right GitHub endpoint with auth."""
    import httpx

    import ops_agent.tools.github as gh_mod

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization", "")
        return httpx.Response(
            200,
            json={"number": 7, "title": "bump requests"},
            request=request,
        )

    real_github = gh_mod.GitHub

    def patched_github(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_github(*args, **kwargs)

    monkeypatch.setattr(gh_mod, "GitHub", patched_github)

    client = gh_mod.GitHubClient()
    result = client.get_pr("myorg", "myrepo", 7)

    assert captured["url"].endswith("/repos/myorg/myrepo/pulls/7")
    assert "test-token" in captured["auth"]
    assert result["number"] == 7
    assert result["title"] == "bump requests"


def test_head_commit_sha_ignores_github_commits_int():
    """GitHub returns `commits` as an int count; only `head.sha` is a real SHA.

    Regression: the Gitea-era code did `pr["commits"][-1]["id"]`, which raises
    TypeError on GitHub because an int is truthy but not subscriptable.
    """
    from ops_agent.graphs.interactive import head_commit_sha

    assert head_commit_sha({"commits": 3, "head": {"sha": "abc123"}}) == "abc123"
    assert head_commit_sha({}) is None
    assert head_commit_sha({"head": {}}) is None
