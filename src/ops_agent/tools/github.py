"""GitHub REST client and LangChain tools.

The raw GitHubClient is usable directly by graph nodes. LangChain tool wrappers
are exposed for operations the agent needs to call from within an AgentExecutor.
"""

from __future__ import annotations

import base64
import json
from typing import Self

from githubkit import GitHub
from githubkit.exception import RequestFailed
from langchain_core.tools import tool

from ops_agent.config import get_settings

# GitHub's default page size is 30. Ask for the maximum instead so ordinary PRs
# never silently truncate their comment/review history. Beyond 100 items real
# pagination is still required — see docs/GITHUB_MIGRATION.md.
_PER_PAGE = 100


class GitHubClient:
    """Thin synchronous wrapper around the GitHub REST API.

    Every method returns plain dicts / lists / strings, never githubkit's pydantic
    models, so graph nodes and the duck-typed test fakes stay decoupled from the SDK.

    No retry logic here on purpose: githubkit retries rate-limited requests once
    (honouring GitHub's suggested delay) and 5xx responses three times, by default.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._gh = GitHub(
            settings.github_token,
            base_url=settings.github_api_base_url,
            timeout=settings.request_timeout_seconds,
            user_agent="ops-agent/1.0",
        )

    def get_pr(self, owner: str, repo: str, index: int) -> dict:
        """Fetch PR metadata. Returns the GitHub pull-request object as a dict."""
        return self._gh.rest.pulls.get(owner, repo, index).json()

    def get_pr_diff(self, owner: str, repo: str, index: int) -> str:
        """Fetch the unified diff for a pull request.

        GitHub refuses to render diffs for very large PRs (~300 files / 20k lines)
        with a 406. Treat that as an empty diff rather than failing the whole run —
        downstream already handles "" (see ingest_pr._extract_current_version_from_diff).
        """
        try:
            resp = self._gh.rest.pulls.get(
                owner,
                repo,
                index,
                headers={"Accept": "application/vnd.github.diff"},
            )
        except RequestFailed as exc:
            if exc.response.status_code == 406:
                return ""
            raise
        return resp.text

    def post_issue_comment(self, owner: str, repo: str, index: int, body: str) -> dict:
        """Post a comment on a PR (which is also an issue on GitHub)."""
        return self._gh.rest.issues.create_comment(owner, repo, index, body=body).json()

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
        return self._gh.rest.pulls.create(
            owner, repo, title=title, body=body, head=head, base=base
        ).json()

    def get_file(self, owner: str, repo: str, path: str, ref: str = "main") -> str:
        """Return decoded file content from the repository at a given ref."""
        data = self._gh.rest.repos.get_content(owner, repo, path, ref=ref).json()
        return base64.b64decode(data["content"]).decode()

    def get_issue(self, owner: str, repo: str, index: int) -> dict:
        """Fetch issue metadata. Returns the GitHub issue object as a dict."""
        return self._gh.rest.issues.get(owner, repo, index).json()

    def list_issue_comments(self, owner: str, repo: str, index: int) -> list[dict]:
        """List comments on a PR or issue (GitHub treats PRs as issues).

        Returns comment objects in chronological order; each carries a ``user``
        object (with ``login``) and a ``created_at`` timestamp.
        """
        return self._gh.rest.issues.list_comments(
            owner, repo, index, per_page=_PER_PAGE
        ).json()

    def list_pr_reviews(self, owner: str, repo: str, index: int) -> list[dict]:
        """List reviews on a pull request.

        Change requests submitted via the review UI ("Request changes") do NOT
        appear in ``list_issue_comments``; they live here. Each review carries a
        ``user``, a ``state`` (APPROVED/CHANGES_REQUESTED/COMMENTED/DISMISSED/PENDING),
        a ``body``, and a ``submitted_at`` timestamp.
        """
        return self._gh.rest.pulls.list_reviews(
            owner, repo, index, per_page=_PER_PAGE
        ).json()

    def list_pr_review_comments(
        self, owner: str, repo: str, index: int, review_id: int
    ) -> list[dict]:
        """List the line comments attached to a single PR review."""
        return self._gh.rest.pulls.list_comments_for_review(
            owner, repo, index, review_id, per_page=_PER_PAGE
        ).json()

    def get_authenticated_user(self) -> dict:
        """Return the account the current token authenticates as (``/user``).

        Used to identify our own comments/reviews so re-drive triage never
        re-triggers on activity we produced ourselves.
        """
        return self._gh.rest.users.get_authenticated().json()

    def approve_pr(self, owner: str, repo: str, index: int, body: str = "LGTM") -> dict:
        """Approve a pull request with a review. Returns the created review object.

        Note the event verb is "APPROVE" — Gitea used "APPROVED". GitHub also
        rejects (422) approving a PR opened by the token's own account; callers
        are expected to tolerate that failure.
        """
        return self._gh.rest.pulls.create_review(
            owner, repo, index, event="APPROVE", body=body
        ).json()

    def close(self) -> None:
        """No-op: githubkit 0.13.7 owns the HTTP client lifecycle.

        `GitHub` exposes no close()/aclose() (verified in core.py). Each request
        runs through the get_sync_client() context manager, which creates and
        closes an httpx.Client per call, so there is nothing to clean up here.
        Kept as a method so the GiteaClient interface (and callers' finally
        blocks / context managers) is unchanged.
        """

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# LangChain tools
# ---------------------------------------------------------------------------

@tool
def github_get_pr(owner: str, repo: str, index: int) -> str:
    """Fetch metadata for a GitHub pull request.

    Args:
        owner: Repository owner (user or org name).
        repo: Repository name.
        index: PR number.

    Returns:
        JSON-serialisable string of the PR object.
    """
    client = GitHubClient()
    try:
        return json.dumps(client.get_pr(owner, repo, index), indent=2)
    except Exception as exc:  # noqa: BLE001 - tools return error strings, never raise
        return f"Error fetching PR: {exc}"
    finally:
        client.close()


@tool
def github_post_comment(owner: str, repo: str, index: int, body: str) -> str:
    """Post a review comment on a GitHub pull request.

    Args:
        owner: Repository owner.
        repo: Repository name.
        index: PR number.
        body: Markdown comment body.

    Returns:
        Confirmation string with the comment ID, or an error message.
    """
    client = GitHubClient()
    try:
        result = client.post_issue_comment(owner, repo, index, body)
        return f"Comment posted (id={result.get('id', '?')})"
    except Exception as exc:  # noqa: BLE001 - tools return error strings, never raise
        return f"Error posting comment: {exc}"
    finally:
        client.close()


@tool
def github_create_pr(
    owner: str,
    repo: str,
    title: str,
    body: str,
    head: str,
    base: str,
) -> str:
    """Open a new pull request on GitHub.

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
    client = GitHubClient()
    try:
        result = client.create_pr(owner, repo, title, body, head, base)
        return result.get("html_url", str(result))
    except Exception as exc:  # noqa: BLE001 - tools return error strings, never raise
        return f"Error creating PR: {exc}"
    finally:
        client.close()


@tool
def github_approve_pr(owner: str, repo: str, index: int, body: str = "LGTM") -> str:
    """Approve a pull request on GitHub.

    Args:
        owner: Repository owner.
        repo: Repository name.
        index: PR number.
        body: Optional review body text (default: "LGTM").

    Returns:
        Confirmation string with the review ID, or an error message.
    """
    client = GitHubClient()
    try:
        result = client.approve_pr(owner, repo, index, body)
        return f"PR approved (review_id={result.get('id', '?')})"
    except Exception as exc:  # noqa: BLE001 - tools return error strings, never raise
        return f"Error approving PR: {exc}"
    finally:
        client.close()


def get_github_tools() -> list:
    """Return the list of LangChain GitHub tools."""
    return [github_get_pr, github_post_comment, github_create_pr, github_approve_pr]
