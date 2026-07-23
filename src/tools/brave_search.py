"""
Brave Search - privacy-focused web search via the Brave Search API.

Brave Search does not track users or queries and maintains its own
independent search index (not a Google/Bing reskin).  The API returns
structured results enriched with FAQ answers, infoboxes, and extra
deep-result snippets when available.

Architecture::

    ┌──────────┐        ┌─────────────────┐        ┌──────────────┐
    │  Agent   │──GET─→ │  Brave API      │──idx─→ │  Brave index │
    │          │        │  /v1/web/search  │        │  (independent│
    │          │←─json  │  /v1/news/search │←─────  │   from Google│
    └──────────┘        └─────────────────┘        │   and Bing)  │
                                                    └──────────────┘

    Implemented via direct HTTP (``requests.get``) because the
    ``brave-search`` Python package (v0.2.0) has incompatible
    dependency constraints (httpx<0.26, numpy<2.0, tenacity<9.0).
    No extra package needed beyond what Cogtrix already uses.

Response extras (when available):
    - FAQ answers to related questions
    - Infobox (knowledge-panel-style summaries)
    - Extra deep-result snippets per hit

Configuration:
    Environment variable: ``BRAVE_API_KEY``
    Config file:          ``services.brave.api_key``
                          (legacy: ``brave.api_key`` at top level)
    Free tier:            2 000 queries / month

The tool is automatically removed from the agent if the API key is
not configured — the agent will never see it without a valid key.
"""

import logging
import os
from typing import Any

import requests
from pydantic import BaseModel, Field

from src.tools.error_sanitizer import sanitize_search_error as _sanitize_search_error

log = logging.getLogger("cogtrix")

# -- Module-level configuration ------------------------------------------------

_brave_config: dict[str, Any] = {}

BRAVE_API_BASE = "https://api.search.brave.com/res/v1"


def configure_brave(config: dict[str, Any]) -> None:
    """
    Set runtime configuration.  Called from ``cogtrix.py`` during startup.

    Expected keys:
        api_key  - Brave Search API key (or read from BRAVE_API_KEY env var)
    """
    global _brave_config
    # Atomic reference swap — safe for concurrent readers without a lock
    _brave_config = {**_brave_config, **config}


def _get_api_key() -> str | None:
    """Resolve API key from config or environment."""
    return _brave_config.get("api_key") or os.getenv("BRAVE_API_KEY")


def is_configured() -> bool:
    """Return True if the tool has the required API key."""
    return bool(_get_api_key())


# -- Input schemas -------------------------------------------------------------


class BraveSearchInput(BaseModel):
    """Input schema for Brave web search."""

    query: str = Field(description="The search query")
    count: int = Field(
        default=5,
        description="Number of results to return (1-20)",
    )
    search_type: str = Field(
        default="web",
        description="Search type: 'web' for general search, 'news' for news articles",
    )
    freshness: str = Field(
        default="",
        description=(
            "Time filter: '' (none), 'pd' (past day), 'pw' (past week), "
            "'pm' (past month), 'py' (past year)"
        ),
    )


# -- Tool functions ------------------------------------------------------------


def brave_search(
    query: str,
    count: int = 5,
    search_type: str = "web",
    freshness: str = "",
) -> str:
    """
    Search the web using Brave Search - a privacy-focused search engine.

    Brave Search does not track users or queries and provides high-quality
    results with descriptions.  Supports web and news search with optional
    time filtering.

    Args:
        query: The search query.
        count: Number of results (1-20).
        search_type: 'web' or 'news'.
        freshness: Time filter (empty, 'pd', 'pw', 'pm', 'py').

    Returns:
        Formatted search results.
    """
    api_key = _get_api_key()
    if not api_key:
        return (
            "Error: Brave API key not configured. "
            "Set BRAVE_API_KEY environment variable or add "
            '"services": {"brave": {"api_key": "..."}} to .cogtrix.json'
        )

    if not query.strip():
        return "Error: Empty search query"

    count = max(1, min(count, 20))
    if search_type not in ("web", "news"):
        search_type = "web"
    valid_freshness = {"", "pd", "pw", "pm", "py"}
    if freshness not in valid_freshness:
        freshness = ""

    endpoint = f"{BRAVE_API_BASE}/{'news' if search_type == 'news' else 'web'}/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }
    params: dict[str, Any] = {
        "q": query,
        "count": count,
    }
    if freshness:
        params["freshness"] = freshness

    try:
        response = requests.get(endpoint, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.HTTPError as e:
        status = getattr(e.response, "status_code", "unknown")
        return f"Error: Brave Search API returned HTTP {status}: {_sanitize_search_error(e)}"
    except requests.exceptions.RequestException as e:
        return f"Error performing Brave search: {_sanitize_search_error(e)}"
    except Exception as e:
        return f"Error performing Brave search: {_sanitize_search_error(e)}"

    # -- Format results --------------------------------------------------------
    output: list[str] = [f"Brave {search_type} search results for: {query}\n"]

    if search_type == "news":
        results = data.get("results", [])
    else:
        results = data.get("web", {}).get("results", [])

    if not results:
        output.append("No results found.")
        return "\n".join(output)

    for i, result in enumerate(results[:count], 1):
        title = result.get("title", "No title")
        url = result.get("url", "No URL")
        description = result.get("description", "")

        output.append(f"{i}. {title}")
        output.append(f"   URL: {url}")

        # Age / date info
        age = result.get("age", "")
        if age:
            output.append(f"   Age: {age}")

        if description:
            if len(description) > 1000:
                description = description[:1000] + "..."
            output.append(f"   {description}")

        # Extra snippet from deep results if available
        extra_snippets = result.get("extra_snippets", [])
        if extra_snippets:
            for snippet in extra_snippets[:2]:
                if len(snippet) > 500:
                    snippet = snippet[:500] + "..."
                output.append(f"   >> {snippet}")

        output.append("")

    # Include FAQ if present
    faq = data.get("faq", {}).get("results", [])
    if faq:
        output.append("**Related Questions:**")
        for item in faq[:3]:
            q = item.get("question", "")
            a = item.get("answer", "")
            if q:
                output.append(f"  Q: {q}")
            if a:
                if len(a) > 300:
                    a = a[:300] + "..."
                output.append(f"  A: {a}")
            output.append("")

    # Include infobox if present
    infobox = data.get("infobox", {})
    if infobox and infobox.get("results"):
        for box in infobox["results"][:1]:
            box_title = box.get("title", "")
            box_desc = box.get("long_desc", "") or box.get("description", "")
            if box_title:
                output.append(f"**Infobox: {box_title}**")
            if box_desc:
                if len(box_desc) > 1000:
                    box_desc = box_desc[:1000] + "..."
                output.append(f"  {box_desc}")
            output.append("")

    return "\n".join(output)


# -- Tool registration --------------------------------------------------------

TOOL_CONFIG = {
    "name": "brave_search",
    "description": (
        "Search the web using Brave Search, a privacy-focused search engine "
        "with its own independent index (not a Google/Bing reskin). Brave "
        "does not track users or queries.\n"
        "\n"
        "Supports:\n"
        "- Web search (general queries) and news search\n"
        "- Time filtering: 'pd' (past day), 'pw' (past week), "
        "'pm' (past month), 'py' (past year)\n"
        "- Rich extras: FAQ answers, infoboxes, deep-result snippets\n"
        "\n"
        "USE THIS TOOL WHEN:\n"
        "- You want results from an independent index (not Google/Bing)\n"
        "- Privacy matters and you want a tracker-free search\n"
        "- You need recent news with time filtering\n"
        "- Other search tools are unavailable or returning poor results\n"
        "\n"
        "Output includes: title, URL, description, age, extra snippets, "
        "plus FAQ answers and infobox data when the API provides them."
    ),
    "input_schema": BraveSearchInput,
    "function": brave_search,
    "requires_confirmation": False,
}

__all__ = [
    "brave_search",
    "configure_brave",
    "is_configured",
    "BraveSearchInput",
    "TOOL_CONFIG",
]
