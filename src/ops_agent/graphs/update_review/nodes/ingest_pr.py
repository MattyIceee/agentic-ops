"""Node: ingest_pr - reads PR metadata and diff from GitHub."""

import re
from typing import Any

from ops_agent.graphs.interactive import head_commit_sha
from ops_agent.state import UpdateReviewState
from ops_agent.tools.github import GitHubClient

_MERGE_CONFIDENCE_RE = re.compile(
    r"merge\s+confidence[^\n]*?:\s*([^\n]+)", re.IGNORECASE
)

# A version token following a `:` (docker tag) or `=`/`:` assignment, e.g.
#   image = "lscr.io/linuxserver/jellyfin:10.11.11"
#   tag: "10.11.11"
_DIFF_VERSION_RE = re.compile(r"""[:=]\s*["']?v?([0-9][0-9A-Za-z.\-_+]*)""")


def _extract_current_version_from_diff(diff: str, new_version: str) -> str:
    """Best-effort recovery of the *current* version from a Renovate diff.

    Renovate PR titles for Docker tags often read "... to vX" with no "from",
    so the outgoing version only exists in the diff's removed (``-``) lines.
    We pair the removed line with the added (``+``) line carrying ``new_version``
    and pull the version token off the removed line.
    """
    if not diff:
        return ""

    removed: list[str] = []
    added: list[str] = []
    for line in diff.splitlines():
        if line.startswith(("---", "+++")):
            continue  # file headers, not content
        if line.startswith("-"):
            removed.append(line[1:])
        elif line.startswith("+"):
            added.append(line[1:])

    # Prefer the removed line whose added counterpart mentions new_version.
    for rem, add in zip(removed, added):
        if new_version and new_version in add:
            m = _DIFF_VERSION_RE.search(rem)
            if m:
                return m.group(1)

    # Fallback: first version-looking token on any removed line that isn't the
    # new version itself.
    for rem in removed:
        m = _DIFF_VERSION_RE.search(rem)
        if m and m.group(1) != new_version:
            return m.group(1)

    return ""


def ingest_pr(state: UpdateReviewState) -> dict[str, Any]:
    """Read PR metadata and diff from GitHub; parse dependency + versions."""
    owner: str = state["_owner"]  # type: ignore[typeddict-item]
    repo: str = state["_repo"]  # type: ignore[typeddict-item]
    index: int = state["pr_index"]

    client = GitHubClient()
    try:
        pr = client.get_pr(owner, repo, index)
        diff = client.get_pr_diff(owner, repo, index)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch PR {owner}/{repo}#{index} from GitHub: {exc}"
        ) from exc
    finally:
        client.close()

    # Head commit SHA doubles as the checkpoint key; fall back to the PR ref if
    # GitHub did not return a head (deleted fork branch, etc).
    last_commit_id = head_commit_sha(pr) or f"{owner}/{repo}#{index}"

    title: str = pr.get("title", "")
    body: str = pr.get("body", "") or ""

    # Renovate PR titles look like: "chore(deps): update dependency foo to v1.2.3"
    # Docker-tag titles insert "docker tag" between the name and "to":
    # "chore(deps): update postgres docker tag to v15". We extract dependency
    # name + versions with a best-effort regex.
    dep_match = re.search(
        r"update\s+(?:dependency\s+)?(.+?)(?:\s+docker\s+tag)?"
        r"\s+(?:from\s+v?(\S+)\s+)?to\s+v?(\S+)",
        title,
        re.IGNORECASE,
    )
    dependency = dep_match.group(1) if dep_match else title
    current_version = dep_match.group(2) if (dep_match and dep_match.group(2)) else ""
    new_version = dep_match.group(3) if dep_match else ""

    # The title frequently lacks the outgoing version ("... to vX"); recover it
    # from the diff so downstream downgrade/scheme checks have both endpoints.
    if not current_version:
        current_version = _extract_current_version_from_diff(diff, new_version)

    # Pull Renovate's Merge Confidence rating from the PR body if present.
    mc_match = _MERGE_CONFIDENCE_RE.search(body)
    renovate_rating = mc_match.group(1).strip() if mc_match else None

    return {
        "dependency": dependency,
        "current_version": current_version,
        "new_version": new_version,
        "diff": diff,
        "renovate_rating": renovate_rating,
        "thread_id": last_commit_id,
    }
