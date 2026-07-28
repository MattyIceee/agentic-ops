"""SearXNG web search tool.

Wraps SearxSearchWrapper in a lazy factory so missing config never breaks imports.
"""

from __future__ import annotations

from langchain_community.utilities import SearxSearchWrapper
from langchain_core.tools import Tool


def get_search_tool(num_results: int = 5) -> Tool:
    """Return a LangChain Tool that queries the configured SearXNG instance.

    Config is read lazily so this can be imported without a live .env.
    """
    from ops_agent.config import get_settings

    settings = get_settings()
    wrapper = SearxSearchWrapper(searx_host=settings.searxng_url)

    def _search(query: str) -> str:
        """Search SearXNG and return formatted snippets."""
        try:
            results = wrapper.results(query, num_results=num_results)
        except Exception as exc:
            return f"Search error: {exc}"

        if not results:
            return "No results found."

        lines: list[str] = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "").strip()
            url = r.get("link", r.get("url", "")).strip()
            snippet = r.get("snippet", "").strip()
            lines.append(f"{i}. {title}\n   {url}\n   {snippet}")

        return "\n\n".join(lines)

    return Tool(
        name="web_search",
        func=_search,
        description=(
            "Search the web via SearXNG. Input is a plain-text query string. "
            "Returns up to 5 results with title, URL, and snippet."
        ),
    )
