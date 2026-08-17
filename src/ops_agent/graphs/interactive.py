"""Shared helpers for interactive, re-drivable graphs.

Both graphs finish a run (post a review / open a PR) and are then re-invoked
later on the same checkpoint thread by a cron poll. A deterministic ``triage``
entry node checks the PR for NEW external input — activity authored by someone
*other than us* — and decides whether, and where, to re-drive the graph.
LangGraph's checkpointer preserves all prior state, so a re-drive resumes from a
specific node without re-running upstream work.

External steering can arrive two ways on a GitHub PR, and we watch BOTH:

* issue/PR conversation comments  (``/issues/{index}/comments``)
* pull-request reviews            (``/pulls/{index}/reviews``) — this is where a
  "Request changes" review body and its line comments live; they do NOT show up
  in the comments endpoint.

"New" needs no separate watermark store; the boundary is derived live: any
foreign activity newer than our own most recent activity (comment or review).
Timestamps are parsed to timezone-aware datetimes — raw ISO strings do not sort
correctly across offsets. Our identity comes from GitHub's authenticated-user
endpoint, so we never re-trigger on our own activity.
"""

from __future__ import annotations

import functools
import logging
from datetime import datetime, timezone
from typing import Any

from ops_agent.tools.github import GitHubClient

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def get_bot_login() -> str | None:
    """Return the login of the account our GitHub token belongs to (cached).

    Returns None if identity can't be resolved; callers then skip author
    filtering (the timestamp boundary still limits what counts as new).
    """
    try:
        client = GitHubClient()
        try:
            return client.get_authenticated_user().get("login")
        finally:
            client.close()
    except Exception as exc:  # pragma: no cover - network/1st-run only
        logger.warning("could not resolve bot identity from GitHub /user: %s", exc)
        return None


def _parse_ts(raw: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp (offset or 'Z') to an aware datetime."""
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _activity(kind: str, ref: Any, author: str | None, text: str, ts_raw: str, state: str = "") -> dict:
    """Normalize a comment or review into a common activity dict.

    ``_ts`` is the parsed datetime used for boundary comparison; it is stripped
    before the dict is stored in ``new_inputs`` (which must stay serializable).
    """
    return {
        "kind": kind,
        "ref": str(ref),
        "author": author or "",
        "text": text or "",
        "created_at": ts_raw or "",
        "state": state,
        "_ts": _parse_ts(ts_raw),
    }


def _normalize_comments(comments: list[dict]) -> list[dict]:
    return [
        _activity("comment", c.get("id", ""), (c.get("user") or {}).get("login"),
                  c.get("body", ""), c.get("created_at", ""))
        for c in comments
    ]


def _normalize_reviews(client: GitHubClient, owner: str, repo: str, index: int, reviews: list[dict]) -> list[dict]:
    acts: list[dict] = []
    for rv in reviews:
        text = rv.get("body", "") or ""
        # A review may carry its steer only in line comments (empty body).
        if not text.strip():
            try:
                line = client.list_pr_review_comments(owner, repo, index, rv.get("id"))
            except Exception:
                line = []
            text = "\n".join(lc.get("body", "") for lc in line if lc.get("body"))
        acts.append(
            _activity("review", rv.get("id", ""), (rv.get("user") or {}).get("login"),
                      text, rv.get("submitted_at", ""), rv.get("state", ""))
        )
    return acts


def _bot_boundary(activities: list[dict], bot_login: str | None) -> datetime | None:
    """Latest timestamp among our own activities, or None if we have none yet."""
    ours = [
        a["_ts"] for a in activities
        if bot_login is not None and a["author"] == bot_login and a["_ts"]
    ]
    return max(ours) if ours else None


def _foreign_since(activities: list[dict], bot_login: str | None, boundary: datetime | None) -> list[dict]:
    """Foreign activities newer than the boundary that carry actual text.

    Bare approvals / empty activities are skipped so they never re-trigger a run.
    """
    out: list[dict] = []
    for a in activities:
        if bot_login is not None and a["author"] == bot_login:
            continue
        if boundary and a["_ts"] and a["_ts"] <= boundary:
            continue
        if not a["text"].strip():
            continue
        out.append({k: v for k, v in a.items() if k != "_ts"})
    return out


def new_steering_inputs(
    client: GitHubClient, owner: str, repo: str, index: int, bot_login: str | None
) -> list[dict]:
    """All foreign, unhandled steering input on a PR — comments AND reviews.

    Returns plain dicts (kind/ref/author/text/created_at/state) suitable for the
    ``new_inputs`` state channel, ordered as fetched.
    """
    comments = client.list_issue_comments(owner, repo, index)
    try:
        reviews = client.list_pr_reviews(owner, repo, index)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("could not list PR reviews for %s/%s#%s: %s", owner, repo, index, exc)
        reviews = []

    activities = _normalize_comments(comments) + _normalize_reviews(client, owner, repo, index, reviews)
    boundary = _bot_boundary(activities, bot_login)
    return _foreign_since(activities, bot_login, boundary)


def head_commit_sha(pr: dict) -> str | None:
    """Latest commit SHA on a PR.

    Reads ``head.sha``. Do NOT use ``pr["commits"]`` — on GitHub that field is an
    integer count, not a list, so indexing it raises TypeError.
    """
    return (pr.get("head") or {}).get("sha")


def steer_block(new_inputs: list[dict]) -> str:
    """Render steering comment(s)/review(s) as a prompt block, or '' if none."""
    if not new_inputs:
        return ""
    joined = "\n".join(f"@{c.get('author', '?')}: {c.get('text', '')}" for c in new_inputs)
    return (
        "A reviewer left steering feedback on the PR — adjust the deployment "
        f"to satisfy the request(s):\n{joined}\n\n"
    )
