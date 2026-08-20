"""Fetch tool: GET a URL and return readable text for use by LangChain agents."""

from urllib.parse import urlparse

import html2text
import httpx
from langchain_core.tools import tool

from ops_agent.config import get_settings

_USER_AGENT = "ops-agent/1.0 (homelab automation; +https://homelab.local)"
_MAX_CHARS = 8_000


def _domain_allowed(url: str) -> bool:
    """Enforce the trust boundary on outbound fetches (layer 5).

    Only http/https schemes are ever allowed (blocks file://, data://, etc.).
    An optional ``FETCH_ALLOWED_DOMAINS`` allow-list restricts which hosts may be
    retrieved; when it is empty (the default) any http(s) host is permitted so
    research recall is not reduced. Empty = best-effort scheme guard only.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    if not parsed.hostname:
        return False

    allowed = get_settings().fetch_allowed_domains
    if not allowed:
        return True

    host = (parsed.hostname or "").lower()
    for entry in allowed:
        domain = entry.strip().lower().lstrip("*.")
        if host == domain or host.endswith("." + domain):
            return True
        if domain.startswith("*.") and host.endswith("." + domain[2:]):
            return True
    return False


def _fetch(url: str) -> str:
    """GET *url* and return text content, truncated to _MAX_CHARS."""
    settings = get_settings()
    try:
        if not _domain_allowed(url):
            return f"Refused to fetch {url}: scheme or domain not allowed"
        with httpx.Client(
            follow_redirects=True,
            timeout=settings.request_timeout_seconds,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            response = client.get(url)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return f"HTTP error {exc.response.status_code} fetching {url}"
    except httpx.RequestError as exc:
        return f"Request error fetching {url}: {exc}"

    content_type = response.headers.get("content-type", "")

    if "html" in content_type:
        converter = html2text.HTML2Text()
        converter.ignore_links = False
        converter.ignore_images = True
        text = converter.handle(response.text)
    else:
        text = response.text

    if len(text) > _MAX_CHARS:
        text = text[:_MAX_CHARS] + "\n\n[… truncated]"

    return text


@tool
def fetch_url(url: str) -> str:
    """Fetch the content of a URL and return it as plain text or markdown.

    HTML pages are converted to markdown. JSON/text/markdown responses are
    returned as-is. Output is truncated to ~8000 characters. Errors are
    returned as descriptive strings rather than raised exceptions. Only
    http/https URLs may be fetched (an allow-list may further restrict hosts).
    """
    return _fetch(url)


def get_fetch_tool():
    """Return the fetch_url LangChain tool."""
    return fetch_url
