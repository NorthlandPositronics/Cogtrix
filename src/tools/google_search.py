"""
Google Search - official Google Custom Search JSON API.

Uses Google's own search index via the Programmable Search Engine
(formerly Custom Search Engine) API.  This is the real Google Search,
not a third-party scraper - results come directly from Google's index
with the same quality and freshness as google.com.

Architecture::

    ┌──────────┐        ┌─────────────────┐        ┌──────────────┐
    │  Agent   │──GET─→ │  Google CSE API │──q──→  │  Google      │
    │          │        │  /customsearch/ │        │  search      │
    │          │←─json  │  v1             │←─────  │  index       │
    └──────────┘        └─────────────────┘        └──────────────┘

    Implemented via direct HTTP (``requests.get``) to the Google
    Custom Search JSON API.  No extra package needed beyond what
    Cogtrix already uses.

Requirements:
    1. A Google API key (from Google Cloud Console)
    2. A Programmable Search Engine ID (cx) configured to search
       the entire web (create at https://programmablesearchengine.google.com)

Configuration:
    Environment variables: ``GOOGLE_API_KEY``, ``GOOGLE_CSE_ID``
    Config file:           ``services.google.api_key``, ``services.google.cse_id``
                           (legacy: ``google.api_key`` / ``google.cse_id`` at top level)
    Free tier:             100 queries / day (10 000/day with billing)

The tool is automatically removed from the agent if the API key or
CSE ID is not configured — both are required for the tool to appear.
"""

import logging
import os
from typing import Any

import requests
from pydantic import BaseModel, Field

from src.tools.error_sanitizer import sanitize_search_error as _sanitize_search_error

log = logging.getLogger("cogtrix")

# -- Module-level configuration ------------------------------------------------

_google_config: dict[str, Any] = {}

GOOGLE_CSE_API = "https://customsearch.googleapis.com/customsearch/v1"


def configure_google_search(config: dict[str, Any]) -> None:
    """
    Set runtime configuration.  Called from ``cogtrix.py`` during startup.

    Expected keys:
        api_key  - Google API key (or read from GOOGLE_API_KEY env var)
        cse_id   - Programmable Search Engine ID (or GOOGLE_CSE_ID env var)
    """
    global _google_config
    # Atomic reference swap — safe for concurrent readers without a lock
    _google_config = {**_google_config, **config}


def _get_api_key() -> str | None:
    """Resolve Google API key from config or environment."""
    return _google_config.get("api_key") or os.getenv("GOOGLE_API_KEY")


def _get_cse_id() -> str | None:
    """Resolve Custom Search Engine ID from config or environment."""
    return _google_config.get("cse_id") or os.getenv("GOOGLE_CSE_ID")


def is_configured() -> bool:
    """Return True if the tool has the required API key and CSE ID."""
    return bool(_get_api_key()) and bool(_get_cse_id())


# -- Input schemas -------------------------------------------------------------


class GoogleSearchInput(BaseModel):
    """Input schema for Google web search."""

    query: str = Field(description="The search query")
    num_results: int = Field(
        default=10,
        description="Number of results to return (1-10, API maximum is 10 per page)",
    )
    date_restrict: str = Field(
        default="",
        description=(
            "Restrict results by date: '' (any time), 'd7' (past 7 days), "
            "'w2' (past 2 weeks), 'm1' (past month), 'm6' (past 6 months), "
            "'y1' (past year).  Format: d[N], w[N], m[N], or y[N]"
        ),
    )
    language: str = Field(
        default="",
        description=(
            "Restrict results to a language, e.g. 'lang_en' (English), "
            "'lang_de' (German), 'lang_fr' (French).  Empty = any language."
        ),
    )
    safe_search: str = Field(
        default="off",
        description="Safe search: 'off' (default) or 'active'",
    )


# -- Tool functions ------------------------------------------------------------


def google_search(
    query: str,
    num_results: int = 10,
    date_restrict: str = "",
    language: str = "",
    safe_search: str = "off",
) -> str:
    """
    Search the web using the official Google Custom Search JSON API.

    Returns real Google search results with titles, URLs, snippets,
    and optional rich data (knowledge graph, metatags, page maps).
    This is the same Google index used by google.com.

    Requires a Google API key and a Programmable Search Engine ID (cx)
    configured to search the entire web.

    Args:
        query: The search query.
        num_results: Number of results (1-10).
        date_restrict: Date filter (e.g. 'd7', 'w2', 'm1', 'y1').
        language: Language restriction (e.g. 'lang_en').
        safe_search: 'off' or 'active'.

    Returns:
        Formatted search results from Google.
    """
    api_key = _get_api_key()
    if not api_key:
        return (
            "Error: Google API key not configured. "
            "Set GOOGLE_API_KEY environment variable or add "
            '"services": {"google": {"api_key": "..."}} to .cogtrix.json'
        )

    cse_id = _get_cse_id()
    if not cse_id:
        return (
            "Error: Google Custom Search Engine ID not configured. "
            "Set GOOGLE_CSE_ID environment variable or add "
            '"services": {"google": {"cse_id": "..."}} to .cogtrix.json. '
            "Create one at https://programmablesearchengine.google.com"
        )

    if not query.strip():
        return "Error: Empty search query"

    num_results = max(1, min(num_results, 10))
    if safe_search not in ("off", "active"):
        safe_search = "off"

    params: dict[str, Any] = {
        "key": api_key,
        "cx": cse_id,
        "q": query,
        "num": num_results,
        "safe": safe_search,
    }
    if date_restrict:
        params["dateRestrict"] = date_restrict
    if language:
        params["lr"] = language

    try:
        response = requests.get(GOOGLE_CSE_API, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.HTTPError as e:
        status = getattr(e.response, "status_code", "unknown")
        body = ""
        try:
            err_data = e.response.json()
            body = err_data.get("error", {}).get("message", "")
        except (ValueError, AttributeError, KeyError):
            body = ""
        detail = f": {body}" if body else ""
        return f"Error: Google Search API returned HTTP {status}{detail}"
    except requests.exceptions.RequestException as e:
        return f"Error performing Google search: {_sanitize_search_error(e)}"
    except Exception as e:
        return f"Error performing Google search: {_sanitize_search_error(e)}"

    # -- Format results --------------------------------------------------------
    output: list[str] = [f"Google search results for: {query}\n"]

    # Search information
    search_info = data.get("searchInformation", {})
    total = search_info.get("formattedTotalResults", "")
    search_time = search_info.get("formattedSearchTime", "")
    if total:
        output.append(f"About {total} results ({search_time}s)\n")

    # Spelling suggestion
    spelling = data.get("spelling", {})
    if spelling.get("correctedQuery"):
        output.append(f"Did you mean: {spelling['correctedQuery']}\n")

    # Items (organic results)
    items = data.get("items", [])
    if not items:
        output.append("No results found.")
        return "\n".join(output)

    for i, item in enumerate(items, 1):
        title = item.get("title", "No title")
        link = item.get("link", "No URL")
        snippet = item.get("snippet", "")
        display_link = item.get("displayLink", "")

        output.append(f"{i}. {title}")
        output.append(f"   URL: {link}")
        if display_link:
            output.append(f"   Site: {display_link}")

        if snippet:
            # Google snippets can contain newlines; normalise them
            snippet = " ".join(snippet.split())
            if len(snippet) > 1000:
                snippet = snippet[:1000] + "..."
            output.append(f"   {snippet}")

        # Metatags (often contain description, date, author)
        pagemap = item.get("pagemap", {})
        metatags = pagemap.get("metatags", [{}])
        if metatags:
            meta = metatags[0] if isinstance(metatags, list) else metatags
            og_desc = meta.get("og:description", "")
            article_date = (
                meta.get("article:published_time", "")
                or meta.get("datePublished", "")
                or meta.get("date", "")
            )
            if article_date:
                output.append(f"   Published: {article_date[:10]}")
            if og_desc and og_desc != snippet:
                og_desc = " ".join(og_desc.split())
                if len(og_desc) > 500:
                    og_desc = og_desc[:500] + "..."
                output.append(f"   Meta: {og_desc}")

        output.append("")

    return "\n".join(output)


# -- Tool registration --------------------------------------------------------

TOOL_CONFIG = {
    "name": "google_search",
    "description": (
        "Search the web using the official Google Custom Search API. "
        "This is real Google Search - results come directly from Google's "
        "index with the same quality and freshness as google.com.\n"
        "\n"
        "Supports:\n"
        "- Full Google web search with up to 10 results per query\n"
        "- Date filtering: 'd7' (7 days), 'w2' (2 weeks), 'm1' (month), "
        "'y1' (year)\n"
        "- Language restriction (e.g. 'lang_en', 'lang_de')\n"
        "- Safe search toggle\n"
        "\n"
        "Rich data returned (when available):\n"
        "- Organic results with titles, URLs, and snippets\n"
        "- Spelling suggestions\n"
        "- Published dates and meta descriptions from page metadata\n"
        "- Total result count and search time\n"
        "\n"
        "USE THIS TOOL WHEN:\n"
        "- You need the highest-quality, most comprehensive web results\n"
        "- You want results from Google's index specifically\n"
        "- You need date-filtered search (e.g. recent results only)\n"
        "- Other search tools return insufficient or irrelevant results\n"
        "\n"
        "Requires a Google API key and a Programmable Search Engine ID. "
        "Free tier: 100 queries/day."
    ),
    "input_schema": GoogleSearchInput,
    "function": google_search,
    "requires_confirmation": False,
}

__all__ = [
    "google_search",
    "configure_google_search",
    "is_configured",
    "GoogleSearchInput",
    "TOOL_CONFIG",
]
