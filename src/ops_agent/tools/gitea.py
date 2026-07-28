"""Gitea REST client and LangChain tools.

The raw GiteaClient is usable directly by graph nodes. LangChain tool wrappers
are exposed for operations the agent needs to call from within an AgentExecutor.
"""

from __future__ import annotations

import base64
from typing import Any

import httpx
from langchain_core.tools import tool
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from ops_agent.config import get_settings

# ---------------------------------------------------------------------------
# Tenacity retry predicate
# ---------------------------------------------------------------------------

def _is_retryable(exc: BaseException) -> bool:
    """Retry on 5xx responses or transport-level timeouts/connect errors."""
    if isinstance(exc, httpx.TimeoutException | httpx.ConnectError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return False


_retry = retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)


# ---------------------------------------------------------------------------
# GiteaClient
# ---------------------------------------------------------------------------

class GiteaClient:
    """Thin synchronous wrapper around the Gitea REST API (v1).

    Uses Gitea's own endpoints — not the GitHub API, which differs in subtle ways.
    All methods raise httpx.HTTPStatusError on non-retryable 4xx responses.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._base = settings.gitea_base_url.rstrip("/") + "/api/v1"
        self._client = httpx.Client(
            headers={
                "Authorization": f"token {settings.gitea_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=settings.request_timeout_seconds,
            follow_redirects=True,
        )

    def _get(self, path: str, **params: Any) -> httpx.Response:
        resp = self._client.get(f"{self._base}{path}", params=params or None)
        resp.raise_for_status()
        return resp

    def _post(self, path: str, json: dict[str, Any]) -> httpx.Response:
        resp = self._client.post(f"{self._base}{path}", json=json)
        resp.raise_for_status()
        return resp

    @_retry
    def get_pr(self, owner: str, repo: str, index: int) -> dict:
        """Fetch PR metadata. Returns the Gitea pull-request object as a dict."""
        return self._get(f"/repos/{owner}/{repo}/pulls/{index}").json()

    @_retry
    def get_pr_diff(self, owner: str, repo: str, index: int) -> str:
        """Fetch the unified diff for a pull request."""
        resp = self._client.get(
            f"{self._base}/repos/{owner}/{repo}/pulls/{index}.diff",
            headers={**self._client.headers, "Accept": "text/plain"},
        )
        resp.raise_for_status()
        return resp.text

    @_retry
    def post_issue_comment(self, owner: str, repo: str, index: int, body: str) -> dict:
        """Post a comment on a PR (which is also an issue in Gitea)."""
        return self._post(
            f"/repos/{owner}/{repo}/issues/{index}/comments",
            json={"body": body},
        ).json()

    @_retry
    def create_pr(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str,
        head: str,
        base: str,
    ) -> dict:
        """Open a new pull request. Returns the created PR object."""
        return self._post(
            f"/repos/{owner}/{repo}/pulls",
            json={"title": title, "body": body, "head": head, "base": base},
        ).json()

    @_retry
    def get_file(self, owner: str, repo: str, path: str, ref: str = "main") -> str:
        """Return decoded file content from the repository at a given ref."""
        data = self._get(f"/repos/{owner}/{repo}/contents/{path}", ref=ref).json()
        return base64.b64decode(data["content"]).decode()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "GiteaClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# LangChain tools
# ---------------------------------------------------------------------------

@tool
def gitea_get_pr(owner: str, repo: str, index: int) -> str:
    """Fetch metadata for a Gitea pull request.

    Args:
        owner: Repository owner (user or org name).
        repo: Repository name.
        index: PR number.

    Returns:
        JSON-serialisable string of the PR object.
    """
    client = GiteaClient()
    try:
        result = client.get_pr(owner, repo, index)
        import json
        return json.dumps(result, indent=2)
    except Exception as exc:
        return f"Error fetching PR: {exc}"
    finally:
        client.close()


@tool
def gitea_post_comment(owner: str, repo: str, index: int, body: str) -> str:
    """Post a review comment on a Gitea pull request.

    Args:
        owner: Repository owner.
        repo: Repository name.
        index: PR number.
        body: Markdown comment body.

    Returns:
        Confirmation string with the comment ID, or an error message.
    """
    client = GiteaClient()
    try:
        result = client.post_issue_comment(owner, repo, index, body)
        return f"Comment posted (id={result.get('id', '?')})"
    except Exception as exc:
        return f"Error posting comment: {exc}"
    finally:
        client.close()


@tool
def gitea_create_pr(
    owner: str,
    repo: str,
    title: str,
    body: str,
    head: str,
    base: str,
) -> str:
    """Open a new pull request on Gitea.

    Args:
        owner: Repository owner.
        repo: Repository name.
        title: PR title.
        body: PR description (Markdown).
        head: Source branch name.
        base: Target branch name (e.g. "main").

    Returns:
        HTML URL of the created PR, or an error message.
    """
    client = GiteaClient()
    try:
        result = client.create_pr(owner, repo, title, body, head, base)
        return result.get("html_url", str(result))
    except Exception as exc:
        return f"Error creating PR: {exc}"
    finally:
        client.close()


def get_gitea_tools() -> list:
    """Return the list of LangChain Gitea tools."""
    return [gitea_get_pr, gitea_post_comment, gitea_create_pr]
