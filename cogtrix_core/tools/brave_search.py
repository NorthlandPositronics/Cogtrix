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

from cogtrix_core.tools._web_search_aggregator import ProviderResult
from cogtrix_core.tools.error_sanitizer import sanitize_search_error as _sanitize_search_error

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


async def _search_async(
    query: str, num_results: int = 5, region: str | None = None
) -> list[ProviderResult]:
    """Async Brave search returning ProviderResult list (ADR-0056 PR-B).

    Wraps the sync ``requests.get`` call in ``asyncio.to_thread``.
    Raises on missing API key / HTTP errors so the stage-1 aggregator
    can catch and sanitise. ``region`` is accepted for cross-provider
    symmetry and ignored (Brave uses request-level locale, not a
    region code).
    """
    import asyncio

    del region

    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("Brave API key not configured")
    if not query.strip():
        return []

    num_results = max(1, min(num_results, 20))

    def _sync_call() -> dict[str, Any]:
        response = requests.get(
            f"{BRAVE_API_BASE}/web/search",
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": api_key,
            },
            params={"q": query, "count": num_results},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    data = await asyncio.to_thread(_sync_call)
    raw_results = data.get("web", {}).get("results", []) or []
    return [
        ProviderResult(
            provider="brave",
            url=r.get("url") or "",
            title=r.get("title") or "",
            snippet=r.get("description") or "",
            # Brave's "age" is a non-ISO relative string ("3 days ago" etc.);
            # leave as None for v1 — normalising deferred per ADR-0056.
            published_date=None,
        )
        for r in raw_results
        if r.get("url")
    ]


# -- Tool registration --------------------------------------------------------
#
# PR-G removed brave_search from the agent catalogue. The sync function
# and the async _search_async stay in this module so web_search can
# reach Brave via _resolve_providers().
TOOL_CONFIGS: list[dict[str, Any]] = []

__all__ = [
    "brave_search",
    "configure_brave",
    "is_configured",
    "BraveSearchInput",
    "TOOL_CONFIGS",
]
