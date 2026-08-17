"""Tests for tools: git_ops, search, fetch, github.

git_ops tests use a real temporary git repo via tmp_path.
GitHubClient tests mock the githubkit transport so no network is required.
"""

from __future__ import annotations

import git
import pytest


# ─────────────────────────── helpers ───────────────────────────────────────

def _init_repo(path):
    """Create a git repo at *path* with a single initial commit so HEAD exists."""
    repo = git.Repo.init(str(path))
    repo.config_writer().set_value("user", "name", "Tester").release()
    repo.config_writer().set_value("user", "email", "tester@example.com").release()
    (path / "README.md").write_text("initial content")
    repo.index.add(["README.md"])
    repo.index.commit("initial commit")
    return repo


# ─────────────────────────── git_ops tests ─────────────────────────────────

def test_commit_all_creates_commit(tmp_path):
    from ops_agent.tools.git_ops import commit_all

    repo = _init_repo(tmp_path)
    (tmp_path / "new_file.txt").write_text("hello world")

    result = commit_all.invoke({"repo_path": str(tmp_path), "message": "add new_file"})

    assert "add new_file" in result or result.startswith("Committed")
    assert len(list(repo.iter_commits())) == 2


def test_commit_all_nothing_to_commit(tmp_path):
    from ops_agent.tools.git_ops import commit_all

    _init_repo(tmp_path)
    result = commit_all.invoke({"repo_path": str(tmp_path), "message": "empty"})

    assert "Nothing to commit" in result


def test_read_diff_shows_change(tmp_path):
    from ops_agent.tools.git_ops import read_diff

    repo = _init_repo(tmp_path)
    (tmp_path / "README.md").write_text("changed content")
    repo.index.add(["README.md"])
    repo.index.commit("update README")

    result = read_diff.invoke({"repo_path": str(tmp_path)})

    # The diff must mention the changed file and include removed/added lines
    assert "README" in result
    assert "-initial" in result or "+changed" in result


def test_create_branch_succeeds(tmp_path):
    from ops_agent.tools.git_ops import create_branch

    _init_repo(tmp_path)
    result = create_branch.invoke({"repo_path": str(tmp_path), "branch": "feature/test"})

    assert "feature/test" in result


def test_push_branch_missing_remote_returns_error_string(tmp_path):
    from ops_agent.tools.git_ops import push_branch

    _init_repo(tmp_path)
    result = push_branch.invoke({"repo_path": str(tmp_path), "branch": "main"})

    # No 'origin' remote — must return an error string, not raise
    assert isinstance(result, str)
    assert any(word in result.lower() for word in ("remote", "error", "origin"))


def test_get_git_tools_returns_four_tools():
    from ops_agent.tools.git_ops import get_git_tools

    tools = get_git_tools()
    assert len(tools) == 4


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


def test_get_github_tools_returns_expected_tools():
    from ops_agent.tools.github import get_github_tools

    tools = get_github_tools()
    assert {t.name for t in tools} == {
        "github_get_pr",
        "github_post_comment",
        "github_create_pr",
        "github_approve_pr",
    }


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
