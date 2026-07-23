"""
DuckDuckGo Web Search — free, keyless web and news search.

DuckDuckGo is a privacy-focused search engine that aggregates results
from multiple sources.  No API key is required, making this the default
fallback search tool that is always available.

Architecture::

    ┌──────────┐        ┌─────────────────┐        ┌──────────────┐
    │  Agent   │──q──→  │  DDGS library   │──q──→  │  DuckDuckGo  │
    │          │        │  (scrape)       │        │  search      │
    │          │←─────  │  text │ news    │←─html  │  engine      │
    └──────────┘  list  └─────────────────┘        └──────────────┘

Two tools are exposed:
    search_web   — General web search with snippets.
    search_news  — Recent news articles with source and date.

Configuration:
    No API key required.
    Package: ``duckduckgo-search`` (or ``ddgs``) on PyPI.
"""

import logging
import os
import threading
from collections.abc import Generator
from contextlib import contextmanager

from pydantic import BaseModel, Field

# Try to import ddgs (formerly duckduckgo_search)
try:
    from ddgs import DDGS

    DDGS_AVAILABLE = True
except ImportError:
    # Fallback to old package name for compatibility
    try:
        from duckduckgo_search import DDGS  # type: ignore[assignment]

        DDGS_AVAILABLE = True
    except ImportError:
        DDGS = None  # type: ignore[misc, assignment]
        DDGS_AVAILABLE = False


# ── Suppress noisy warnings from primp / ddgs native code ────────────
# The primp Rust HTTP client writes "gzip response with content-length of 0"
# directly to stderr, which pollutes the spinner / terminal output.

# 1. Mute the Python-level loggers for the ddgs / primp stack.
for _logger_name in ("ddgs", "ddgs.http_client", "ddgs.http_client2", "primp"):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)


_stderr_lock = threading.Lock()


@contextmanager
def _suppress_native_stderr() -> Generator[None]:
    """Redirect fd 2 (stderr) to /dev/null for the duration of the block.

    This catches output from native (Rust/C) libraries that bypass
    Python's ``sys.stderr`` and write directly to file descriptor 2.

    The lock is held only during the dup/dup2 calls, not during the search,
    so concurrent searches are not serialized.
    """
    try:
        with _stderr_lock:
            saved_fd = os.dup(2)
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull_fd, 2)
            os.close(devnull_fd)
    except OSError:
        yield
        return
    try:
        yield
    finally:
        with _stderr_lock:
            os.dup2(saved_fd, 2)
            os.close(saved_fd)


class WebSearchInput(BaseModel):
    """Input schema for web search."""

    query: str = Field(description="The search query")
    num_results: int = Field(default=5, description="Number of results to return (max 10)")
    region: str = Field(
        default="wt-wt",
        description="Region for search (e.g., 'us-en', 'uk-en', 'wt-wt' for worldwide)",
    )


class NewsSearchInput(BaseModel):
    """Input schema for news search."""

    query: str = Field(description="The news search query")
    num_results: int = Field(default=5, description="Number of results to return (max 10)")
    timelimit: str | None = Field(
        default="w",
        description="Time limit: 'd' (day), 'w' (week), 'm' (month)",
    )


def search_web(query: str, num_results: int = 5, region: str = "wt-wt") -> str:
    """
    Search the web using DuckDuckGo.

    Args:
        query: The search query
        num_results: Number of results to return (max 10)
        region: Region for search results

    Returns:
        Search results with titles, URLs, and snippets
    """
    if not DDGS_AVAILABLE:
        return (
            "Error: DuckDuckGo search not available. "
            "Install it with: pip install duckduckgo-search"
        )

    if not query.strip():
        return "Error: Empty search query"

    num_results = min(max(1, num_results), 10)  # Clamp between 1 and 10

    try:
        with _suppress_native_stderr(), DDGS() as ddgs:
            results = list(ddgs.text(query, region=region, max_results=num_results))

        if not results:
            return f"No results found for: {query}"

        output = [f"Search results for: {query}\n"]
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("href", result.get("link", "No URL"))
            snippet = result.get("body", result.get("snippet", "No description"))

            # Truncate long snippets — keep enough for the LLM to get
            # the key facts without having to fill gaps from memory.
            if len(snippet) > 500:
                snippet = snippet[:500] + "..."

            output.append(f"{i}. {title}")
            output.append(f"   URL: {url}")
            output.append(f"   {snippet}")
            output.append("")

        return "\n".join(output)

    except Exception as e:
        return f"Error searching: {e}"


def search_news(query: str, num_results: int = 5, timelimit: str | None = "w") -> str:
    """
    Search for news using DuckDuckGo.

    Args:
        query: The news search query
        num_results: Number of results to return (max 10)
        timelimit: Time limit - 'd' (day), 'w' (week), 'm' (month)

    Returns:
        News results with titles, URLs, dates, and snippets
    """
    if not DDGS_AVAILABLE:
        return (
            "Error: DuckDuckGo search not available. "
            "Install it with: pip install duckduckgo-search"
        )

    if not query.strip():
        return "Error: Empty search query"

    num_results = min(max(1, num_results), 10)  # Clamp between 1 and 10

    try:
        with _suppress_native_stderr(), DDGS() as ddgs:
            results = list(ddgs.news(query, timelimit=timelimit, max_results=num_results))

        if not results:
            return f"No news found for: {query}"

        output = [f"News results for: {query}\n"]
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("url", result.get("link", "No URL"))
            date = result.get("date", "Unknown date")
            source = result.get("source", "Unknown source")
            snippet = result.get("body", result.get("snippet", "No description"))

            # Truncate long snippets
            if len(snippet) > 500:
                snippet = snippet[:500] + "..."

            output.append(f"{i}. {title}")
            output.append(f"   Source: {source} | Date: {date}")
            output.append(f"   URL: {url}")
            output.append(f"   {snippet}")
            output.append("")

        return "\n".join(output)

    except Exception as e:
        return f"Error searching news: {e}"


# Tool configurations for registry
TOOL_CONFIGS = [
    {
        "name": "search_web",
        "description": (
            "Search the web using DuckDuckGo — a free, privacy-focused "
            "search engine. No API key required.\n"
            "\n"
            "Returns titles, URLs, and text snippets (up to 500 chars each). "
            "Supports regional search via the 'region' parameter.\n"
            "\n"
            "USE THIS TOOL WHEN:\n"
            "- You need a quick, free web lookup\n"
            "- Other search tools are unavailable or have returned errors\n"
            "- No API key is configured for premium search providers"
        ),
        "input_schema": WebSearchInput,
        "requires_confirmation": False,
        "function": search_web,
    },
    {
        "name": "search_news",
        "description": (
            "Search for recent news articles using DuckDuckGo. No API key "
            "required.\n"
            "\n"
            "Returns news articles with titles, sources, publication dates, "
            "and snippets. Supports time filtering: 'd' (past day), "
            "'w' (past week), 'm' (past month).\n"
            "\n"
            "USE THIS TOOL WHEN:\n"
            "- You need recent news on a topic\n"
            "- Other news search tools are unavailable\n"
            "- You want a quick news check without needing an API key"
        ),
        "input_schema": NewsSearchInput,
        "requires_confirmation": False,
        "function": search_news,
    },
]

# Default single tool config (for backwards compatibility)
TOOL_CONFIG = TOOL_CONFIGS[0]

__all__ = [
    "search_web",
    "search_news",
    "WebSearchInput",
    "NewsSearchInput",
    "TOOL_CONFIG",
    "TOOL_CONFIGS",
]
