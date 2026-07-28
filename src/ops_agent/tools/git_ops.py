"""Local git operations implemented with GitPython, exposed as LangChain tools."""

from __future__ import annotations

import git
from langchain_core.tools import tool

from ops_agent.config import get_settings


@tool
def read_diff(repo_path: str, ref_a: str = "HEAD~1", ref_b: str = "HEAD") -> str:
    """Return the unified diff between two refs in the given local git repo."""
    try:
        repo = git.Repo(repo_path)
        diff = repo.git.diff(ref_a, ref_b)
        return diff if diff else "No differences between refs."
    except git.GitCommandError as exc:
        return f"git diff error: {exc}"
    except git.InvalidGitRepositoryError:
        return f"Not a git repository: {repo_path}"
    except Exception as exc:  # noqa: BLE001
        return f"Unexpected error reading diff: {exc}"


@tool
def create_branch(repo_path: str, branch: str) -> str:
    """Create and checkout a new branch in the given local git repo."""
    try:
        repo = git.Repo(repo_path)
        new_branch = repo.create_head(branch)
        new_branch.checkout()
        return f"Created and checked out branch '{branch}'."
    except git.GitCommandError as exc:
        return f"git branch error: {exc}"
    except git.InvalidGitRepositoryError:
        return f"Not a git repository: {repo_path}"
    except Exception as exc:  # noqa: BLE001
        return f"Unexpected error creating branch: {exc}"


@tool
def commit_all(repo_path: str, message: str) -> str:
    """Stage all changes and commit with the configured author identity."""
    try:
        settings = get_settings()
        repo = git.Repo(repo_path)
        repo.git.add(A=True)
        if not repo.index.diff("HEAD") and not repo.untracked_files:
            return "Nothing to commit; working tree is clean."
        author = git.Actor(settings.git_author_name, settings.git_author_email)
        commit = repo.index.commit(message, author=author, committer=author)
        return f"Committed {commit.hexsha[:7]}: {message}"
    except git.GitCommandError as exc:
        return f"git commit error: {exc}"
    except git.InvalidGitRepositoryError:
        return f"Not a git repository: {repo_path}"
    except Exception as exc:  # noqa: BLE001
        return f"Unexpected error committing: {exc}"


@tool
def push_branch(repo_path: str, branch: str, remote: str = "origin") -> str:
    """Push the named branch to the given remote."""
    try:
        repo = git.Repo(repo_path)
        push_infos = repo.remotes[remote].push(refspec=f"{branch}:{branch}")
        results = []
        for info in push_infos:
            if info.flags & git.remote.PushInfo.ERROR:
                results.append(f"Push error: {info.summary}")
            else:
                results.append(f"Pushed '{branch}' to '{remote}': {info.summary}")
        return "\n".join(results) if results else f"Pushed '{branch}' to '{remote}'."
    except IndexError:
        return f"Remote '{remote}' not found in repository."
    except git.GitCommandError as exc:
        return f"git push error: {exc}"
    except git.InvalidGitRepositoryError:
        return f"Not a git repository: {repo_path}"
    except Exception as exc:  # noqa: BLE001
        return f"Unexpected error pushing: {exc}"


def get_git_tools() -> list:
    """Return all git LangChain tools as a list."""
    return [read_diff, create_branch, commit_all, push_branch]
