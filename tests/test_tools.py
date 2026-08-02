"""Tests for tools: git_ops, search, fetch, gitea.

git_ops tests use a real temporary git repo via tmp_path.
GiteaClient tests mock httpx so no network is required.
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


def test_get_gitea_tools_returns_expected_tools():
    from ops_agent.tools.gitea import get_gitea_tools

    tools = get_gitea_tools()
    assert {t.name for t in tools} == {
        "gitea_get_pr",
        "gitea_post_comment",
        "gitea_create_pr",
        "gitea_approve_pr",
    }


# ─────────────────────────── GiteaClient mock test ─────────────────────────

def test_gitea_client_get_pr_url_and_auth(monkeypatch):
    """GiteaClient.get_pr must call the correct Gitea endpoint with auth header."""
    from unittest.mock import MagicMock

    import httpx

    captured: dict = {}

    def fake_get(self_client, url, **kwargs):
        captured["url"] = url
        captured["auth_header"] = str(self_client.headers.get("authorization", ""))
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"number": 7, "title": "bump requests"}
        return resp

    monkeypatch.setattr(httpx.Client, "get", fake_get)

    from ops_agent.tools.gitea import GiteaClient

    client = GiteaClient()
    result = client.get_pr("myorg", "myrepo", 7)

    assert captured["url"].endswith("/repos/myorg/myrepo/pulls/7")
    assert "test-token" in captured["auth_header"]
    assert result["number"] == 7
    assert result["title"] == "bump requests"
