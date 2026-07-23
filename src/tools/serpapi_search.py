"""
SerpAPI Search - structured Google/Bing search results.

SerpAPI scrapes and parses results from Google, Bing, and other major
search engines, returning clean structured data.  It is mature, widely
used in AI agent pipelines, and provides the richest structured output
of all search tools (answer boxes, knowledge graphs, People Also Ask,
rich snippets, etc.).

Architecture::

    ┌──────────┐        ┌─────────────────┐        ┌──────────────┐
    │  Agent   │──q──→  │  SerpAPI proxy  │──q──→  │  Google /    │
    │          │        │  (scrape+parse) │        │  Bing / etc  │
    │          │←─json  │                 │←─html  │              │
    └──────────┘        └─────────────────┘        └──────────────┘

    SerpAPI acts as a structured proxy: it executes real search engine
    queries, parses the messy HTML responses, and returns clean JSON
    with organic results, knowledge graph, answer boxes, related
    questions ("People Also Ask"), rich snippets, and more.

Supported search types (via ``tbm`` parameter):
    ''     — regular web search
    'nws'  — Google News
    'isch' — Google Images
    'shop' — Google Shopping

Configuration:
    Environment variable: ``SERPAPI_API_KEY``
    Config file:          ``services.serpapi.api_key``
                          (legacy: ``serpapi.api_key`` at top level)
    Free tier:            100 searches / month
    Package:              ``google-search-results`` (PyPI)

The tool is automatically removed from the agent if the
``google-search-results`` package is not installed or the API key
is not configured.
"""

import logging
import os
from typing import Any

from pydantic import BaseModel, Field

from src.tools._web_search_aggregator import ProviderResult
from src.tools.error_sanitizer import sanitize_search_error as _sanitize_search_error

log = logging.getLogger("cogtrix")

# -- Optional import -----------------------------------------------------------

try:
    from serpapi import GoogleSearch  # type: ignore[import-untyped]

    SERPAPI_AVAILABLE = True
except ImportError:
    GoogleSearch = None  # type: ignore[misc, assignment]
    SERPAPI_AVAILABLE = False

# -- Module-level configuration ------------------------------------------------

_serpapi_config: dict[str, Any] = {}


def configure_serpapi(config: dict[str, Any]) -> None:
    """
    Set runtime configuration.  Called from ``cogtrix.py`` during startup.

    Expected keys:
        api_key  - SerpAPI key (or read from SERPAPI_API_KEY env var)
    """
    global _serpapi_config
    # Atomic reference swap — safe for concurrent readers without a lock
    _serpapi_config = {**_serpapi_config, **config}


def _get_api_key() -> str | None:
    """Resolve API key from config or environment."""
    return _serpapi_config.get("api_key") or os.getenv("SERPAPI_API_KEY")


def is_configured() -> bool:
    """Return True if the tool has the required API key and SDK."""
    return SERPAPI_AVAILABLE and bool(_get_api_key())


# -- Input schemas -------------------------------------------------------------


class SerpAPISearchInput(BaseModel):
    """Input schema for SerpAPI web search."""

    query: str = Field(description="The search query")
    engine: str = Field(
        default="google",
        description="Search engine: 'google' or 'bing'",
    )
    num_results: int = Field(
        default=10,
        description="Number of results to return (1-20)",
    )
    search_type: str = Field(
        default="",
        description=("Type of search: '' (web), 'nws' (news), 'isch' (images), 'shop' (shopping)"),
    )
    time_period: str = Field(
        default="",
        description=(
            "Time filter: '' (any time), 'qdr:d' (past day), "
            "'qdr:w' (past week), 'qdr:m' (past month), 'qdr:y' (past year)"
        ),
    )


# -- Tool functions ------------------------------------------------------------


def serpapi_search(
    query: str,
    engine: str = "google",
    num_results: int = 10,
    search_type: str = "",
    time_period: str = "",
) -> str:
    """
    Search the web using SerpAPI - structured Google/Bing search results.

    SerpAPI scrapes real search engine results and returns clean structured
    data including organic results, knowledge graph, answer boxes, related
    questions, and more.

    Args:
        query: The search query.
        engine: Search engine ('google' or 'bing').
        num_results: Number of results (1-20).
        search_type: '' (web), 'nws' (news), 'isch' (images), 'shop' (shopping).
        time_period: Time filter (empty, 'qdr:d', 'qdr:w', 'qdr:m', 'qdr:y').

    Returns:
        Formatted search results.
    """
    if not SERPAPI_AVAILABLE:
        return "Error: google-search-results is not installed. Run: uv add google-search-results"

    api_key = _get_api_key()
    if not api_key:
        return (
            "Error: SerpAPI key not configured. "
            "Set SERPAPI_API_KEY environment variable or add "
            '"services": {"serpapi": {"api_key": "..."}} to .cogtrix.json'
        )

    if not query.strip():
        return "Error: Empty search query"

    num_results = max(1, min(num_results, 20))
    if engine not in ("google", "bing"):
        engine = "google"
    valid_types = {"", "nws", "isch", "shop"}
    if search_type not in valid_types:
        search_type = ""
    valid_periods = {"", "qdr:d", "qdr:w", "qdr:m", "qdr:y"}
    if time_period not in valid_periods:
        time_period = ""

    params: dict[str, Any] = {
        "q": query,
        "engine": engine,
        "num": num_results,
        "api_key": api_key,
    }
    if search_type:
        params["tbm"] = search_type
    if time_period:
        params["tbs"] = time_period

    try:
        search = GoogleSearch(params)
        data = search.get_dict()
    except Exception as e:
        return f"Error performing SerpAPI search: {_sanitize_search_error(e)}"

    if "error" in data:
        return f"Error from SerpAPI: {data['error']}"

    # -- Format results --------------------------------------------------------
    output: list[str] = [f"SerpAPI ({engine}) search results for: {query}\n"]

    # Answer box (direct answer)
    answer_box = data.get("answer_box", {})
    if answer_box:
        answer = answer_box.get("answer") or answer_box.get("snippet", "")
        title = answer_box.get("title", "")
        if answer:
            if title:
                output.append(f"**Direct Answer ({title}):** {answer}\n")
            else:
                output.append(f"**Direct Answer:** {answer}\n")

    # Knowledge graph
    knowledge = data.get("knowledge_graph", {})
    if knowledge:
        kg_title = knowledge.get("title", "")
        kg_type = knowledge.get("type", "")
        kg_desc = knowledge.get("description", "")
        if kg_title:
            label = f"{kg_title} ({kg_type})" if kg_type else kg_title
            output.append(f"**Knowledge Graph: {label}**")
        if kg_desc:
            if len(kg_desc) > 1000:
                kg_desc = kg_desc[:1000] + "..."
            output.append(f"  {kg_desc}")
            output.append("")

    # Organic results
    organic = data.get("organic_results", [])
    if not organic:
        if not answer_box and not knowledge:
            output.append("No results found.")
        return "\n".join(output)

    for i, result in enumerate(organic[:num_results], 1):
        title = result.get("title", "No title")
        link = result.get("link", "No URL")
        snippet = result.get("snippet", "")

        output.append(f"{i}. {title}")
        output.append(f"   URL: {link}")

        date = result.get("date", "")
        if date:
            output.append(f"   Date: {date}")

        if snippet:
            if len(snippet) > 1000:
                snippet = snippet[:1000] + "..."
            output.append(f"   {snippet}")

        # Rich snippet data
        rich = result.get("rich_snippet", {})
        if rich:
            top = rich.get("top", {})
            extensions = top.get("extensions", [])
            if extensions:
                output.append(f"   Info: {', '.join(str(e) for e in extensions[:5])}")

        output.append("")

    # Related questions (People Also Ask)
    related = data.get("related_questions", [])
    if related:
        output.append("**People Also Ask:**")
        for item in related[:3]:
            q = item.get("question", "")
            snippet = item.get("snippet", "")
            if q:
                output.append(f"  Q: {q}")
            if snippet:
                if len(snippet) > 300:
                    snippet = snippet[:300] + "..."
                output.append(f"  A: {snippet}")
            output.append("")

    return "\n".join(output)


async def _search_async(
    query: str, num_results: int = 5, region: str | None = None
) -> list[ProviderResult]:
    """Async SerpAPI search returning ProviderResult list (ADR-0056 PR-B).

    Wraps the sync ``GoogleSearch(params).get_dict()`` call in
    ``asyncio.to_thread``. Raises on missing API key / API errors.
    ``region`` is accepted for cross-provider symmetry; SerpAPI uses
    ``gl`` (geo-location) for region targeting which we don't surface
    here — the v1 fan-out passes a uniform query, not per-provider
    locale params.
    """
    import asyncio

    del region

    if not SERPAPI_AVAILABLE:
        raise RuntimeError("SerpAPI SDK not installed")
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("SerpAPI key not configured")
    if not query.strip():
        return []

    num_results = max(1, min(num_results, 20))

    def _sync_call() -> dict[str, Any]:
        return GoogleSearch(
            {
                "q": query,
                "engine": "google",
                "num": num_results,
                "api_key": api_key,
            }
        ).get_dict()

    data = await asyncio.to_thread(_sync_call)
    if "error" in data:
        raise RuntimeError(f"SerpAPI error: {data['error']}")

    organic = data.get("organic_results", []) or []
    return [
        ProviderResult(
            provider="serpapi",
            url=r.get("link") or "",
            title=r.get("title") or "",
            snippet=r.get("snippet") or "",
            # SerpAPI's "date" is a free-form string ("3 hours ago",
            # "Mar 15, 2026"); leaving unparsed for v1.
            published_date=None,
        )
        for r in organic[:num_results]
        if r.get("link")
    ]


# -- Tool registration --------------------------------------------------------
#
# PR-G removed serpapi_search from the agent catalogue. The sync
# function and async _search_async stay in this module so web_search
# can reach SerpAPI via _resolve_providers().
TOOL_CONFIGS: list[dict[str, Any]] = []


__all__ = [
    "serpapi_search",
    "configure_serpapi",
    "is_configured",
    "SerpAPISearchInput",
    "TOOL_CONFIGS",
]
