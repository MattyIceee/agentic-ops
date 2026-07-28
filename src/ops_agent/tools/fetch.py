"""Fetch tool: GET a URL and return readable text for use by LangChain agents."""

import html2text
import httpx
from langchain_core.tools import tool

from ops_agent.config import get_settings

_USER_AGENT = "ops-agent/1.0 (homelab automation; +https://homelab.local)"
_MAX_CHARS = 8_000


def _fetch(url: str) -> str:
    """GET *url* and return text content, truncated to _MAX_CHARS."""
    settings = get_settings()
    try:
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
    returned as descriptive strings rather than raised exceptions.
    """
    return _fetch(url)


def get_fetch_tool():
    """Return the fetch_url LangChain tool."""
    return fetch_url
