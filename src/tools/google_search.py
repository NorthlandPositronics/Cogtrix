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

from src.tools._web_search_aggregator import ProviderResult
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


async def _search_async(
    query: str, num_results: int = 5, region: str | None = None
) -> list[ProviderResult]:
    """Async Google CSE search returning ProviderResult list (ADR-0056 PR-B).

    Wraps the sync ``requests.get`` against the Custom Search JSON API.
    Best-effort date extraction from pagemap metatags
    (``article:published_time`` / ``datePublished`` / ``date``).
    Raises on missing API key / CSE ID / HTTP errors.
    """
    import asyncio

    del region  # Google CSE uses 'lr' (language), not region — out of scope here

    api_key = _get_api_key()
    cse_id = _get_cse_id()
    if not api_key:
        raise RuntimeError("Google API key not configured")
    if not cse_id:
        raise RuntimeError("Google Custom Search Engine ID not configured")
    if not query.strip():
        return []

    num_results = max(1, min(num_results, 10))

    def _sync_call() -> dict[str, Any]:
        response = requests.get(
            GOOGLE_CSE_API,
            params={
                "key": api_key,
                "cx": cse_id,
                "q": query,
                "num": num_results,
                "safe": "off",
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    data = await asyncio.to_thread(_sync_call)
    items = data.get("items", []) or []

    results: list[ProviderResult] = []
    for item in items:
        link = item.get("link") or ""
        if not link:
            continue
        pagemap = item.get("pagemap", {}) or {}
        metatags = pagemap.get("metatags", []) or []
        meta = metatags[0] if metatags else {}
        published = (
            meta.get("article:published_time")
            or meta.get("datePublished")
            or meta.get("date")
            or None
        )
        results.append(
            ProviderResult(
                provider="google",
                url=link,
                title=item.get("title") or "",
                snippet=" ".join((item.get("snippet") or "").split()),
                # The date may include time component; keep first 10 chars
                # (YYYY-MM-DD) when present.
                published_date=published[:10] if isinstance(published, str) and published else None,
            )
        )
    return results


# -- Tool registration --------------------------------------------------------
#
# PR-G removed google_search from the agent catalogue. The sync
# function and async _search_async stay in this module so web_search
# can reach Google via _resolve_providers().
TOOL_CONFIGS: list[dict[str, Any]] = []


__all__ = [
    "google_search",
    "configure_google_search",
    "is_configured",
    "GoogleSearchInput",
    "TOOL_CONFIGS",
]
