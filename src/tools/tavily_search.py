"""
Tavily Search — AI-optimised web search with content extraction.

Tavily is designed specifically for AI agents.  Unlike traditional search
engines that return short snippets, Tavily crawls pages and extracts their
full text content, providing the LLM with real data to reason about.

Architecture::

    ┌──────────┐        ┌─────────────────┐        ┌──────────────┐
    │  Agent   │──q──→  │  Tavily API     │──q──→  │  Web crawl   │
    │          │        │  (search_depth)  │        │  + extract   │
    │          │←─────  │  basic │advanced │←─────  │  + summarise │
    └──────────┘  json  └─────────────────┘  html  └──────────────┘

    search_depth="basic"     → fast, snippet-level results
    search_depth="advanced"  → deep crawl with full content extraction

Two tools are exposed:
    tavily_search   — Search the web and get extracted page content.
    tavily_extract  — Extract content from specific URLs (up to 20).

Configuration:
    Environment variable: ``TAVILY_API_KEY``
    Config file:          ``services.tavily.api_key``
                          (legacy: ``tavily.api_key`` at top level)
    Free tier:            1 000 searches / month

The tool is automatically removed from the agent if the ``tavily-python``
package is not installed or the API key is not configured.
"""

import logging
import os
from typing import Any

from pydantic import BaseModel, Field

log = logging.getLogger("cogtrix")

# ── Optional import ──────────────────────────────────────────────────────

try:
    from tavily import TavilyClient  # type: ignore[import-untyped]

    TAVILY_AVAILABLE = True
except ImportError:
    TavilyClient = None  # type: ignore[misc, assignment]
    TAVILY_AVAILABLE = False

# ── Module-level configuration ───────────────────────────────────────────

_tavily_config: dict[str, Any] = {}


def configure_tavily(config: dict[str, Any]) -> None:
    """
    Set runtime configuration.  Called from ``cogtrix.py`` during startup.

    Expected keys:
        api_key  – Tavily API key (or read from TAVILY_API_KEY env var)
    """
    global _tavily_config
    # Atomic reference swap — safe for concurrent readers without a lock
    _tavily_config = {**_tavily_config, **config}


def _get_api_key() -> str | None:
    """Resolve API key from config or environment."""
    return _tavily_config.get("api_key") or os.getenv("TAVILY_API_KEY")


def is_configured() -> bool:
    """Return True if the tool has the required API key and SDK."""
    return TAVILY_AVAILABLE and bool(_get_api_key())


def _get_client() -> Any:
    """Create a TavilyClient with the configured API key."""
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError(
            "Tavily API key not configured. "
            "Set TAVILY_API_KEY environment variable or add "
            '"services": {"tavily": {"api_key": "tvly-..."}} to .cogtrix.json'
        )
    return TavilyClient(api_key=api_key)


# ── Input schemas ────────────────────────────────────────────────────────


class TavilySearchInput(BaseModel):
    """Input schema for Tavily web search."""

    query: str = Field(description="The search query")
    search_depth: str = Field(
        default="advanced",
        description=(
            "Search depth: 'basic' for fast results with snippets, "
            "'advanced' for deeper crawling with full content extraction "
            "(recommended for factual research)"
        ),
    )
    max_results: int = Field(
        default=5,
        description="Number of results to return (1-10)",
    )
    include_answer: bool = Field(
        default=True,
        description=(
            "Include a short AI-generated answer summary " "synthesised from the search results"
        ),
    )
    topic: str = Field(
        default="general",
        description="Search topic: 'general' or 'news'",
    )


class TavilyExtractInput(BaseModel):
    """Input schema for Tavily URL content extraction."""

    urls: list[str] = Field(
        description="List of URLs to extract content from (max 20)",
    )


# ── Tool functions ───────────────────────────────────────────────────────


def tavily_search(
    query: str,
    search_depth: str = "advanced",
    max_results: int = 5,
    include_answer: bool = True,
    topic: str = "general",
) -> str:
    """
    Search the web using Tavily — an AI-optimised search engine that
    returns extracted page content, not just short snippets.

    Use this tool for factual research where accuracy matters.  The
    'advanced' depth crawls pages and extracts their full text, giving
    the LLM real data to work with instead of 500-char snippets.

    Args:
        query: The search query.
        search_depth: 'basic' or 'advanced' (default: advanced).
        max_results: Number of results (1-10).
        include_answer: Whether to include an AI-generated summary.
        topic: 'general' or 'news'.

    Returns:
        Formatted search results with extracted content.
    """
    if not TAVILY_AVAILABLE:
        return (
            "Error: tavily-python is not installed. " "Install it with: pip install tavily-python"
        )

    if not query.strip():
        return "Error: Empty search query"

    # Clamp parameters
    max_results = max(1, min(max_results, 10))
    if search_depth not in ("basic", "advanced"):
        search_depth = "advanced"
    if topic not in ("general", "news"):
        topic = "general"

    try:
        client = _get_client()
        response = client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results,
            include_answer=include_answer,
            include_raw_content=False,  # raw_content is huge; content is enough
            topic=topic,
        )
    except RuntimeError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error performing Tavily search: {e}"

    # ── Format the response ──────────────────────────────────────
    output: list[str] = [f"Tavily search results for: {query}\n"]

    # AI-generated answer (short synthesis)
    answer = response.get("answer")
    if answer:
        output.append(f"**AI Summary:** {answer}\n")

    results = response.get("results", [])
    if not results:
        output.append("No results found.")
        return "\n".join(output)

    for i, result in enumerate(results, 1):
        title = result.get("title", "No title")
        url = result.get("url", "No URL")
        content = result.get("content", "")
        score = result.get("score", 0)

        output.append(f"{i}. {title}")
        output.append(f"   URL: {url}")
        if score:
            output.append(f"   Relevance: {score:.2f}")

        # Tavily 'content' is already extracted text, much richer
        # than DuckDuckGo snippets.  Truncate only very long entries.
        if content:
            if len(content) > 2000:
                content = content[:2000] + "..."
            output.append(f"   {content}")
        output.append("")

    return "\n".join(output)


def tavily_extract(urls: list[str]) -> str:
    """
    Extract clean text content from one or more web pages using Tavily.

    This is useful when you have specific URLs (e.g. from a previous
    search) and need their full content for analysis.  Tavily handles
    JavaScript-rendered pages and returns clean, readable text.

    Args:
        urls: List of URLs to extract content from (max 20).

    Returns:
        Extracted text content from each URL.
    """
    if not TAVILY_AVAILABLE:
        return (
            "Error: tavily-python is not installed. " "Install it with: pip install tavily-python"
        )

    if not urls:
        return "Error: No URLs provided"

    # Clamp to API limit
    urls = urls[:20]

    try:
        client = _get_client()
        response = client.extract(urls=urls)
    except RuntimeError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error extracting content: {e}"

    # ── Format the response ──────────────────────────────────────
    output: list[str] = []

    results = response.get("results", [])
    failed = response.get("failed_results", [])

    for result in results:
        url = result.get("url", "unknown")
        raw_content = result.get("raw_content", "")

        output.append(f"## {url}\n")
        if raw_content:
            # Truncate very long pages
            if len(raw_content) > 8000:
                raw_content = raw_content[:8000] + "\n\n... (truncated)"
            output.append(raw_content)
        else:
            output.append("(no content extracted)")
        output.append("")

    if failed:
        output.append("**Failed to extract:**")
        for fail in failed:
            url = fail.get("url", "unknown") if isinstance(fail, dict) else str(fail)
            output.append(f"  - {url}")

    if not results and not failed:
        output.append("No content could be extracted from the provided URLs.")

    return "\n".join(output)


# ── Tool registration ───────────────────────────────────────────────────

TOOL_CONFIGS = [
    {
        "name": "tavily_search",
        "description": (
            "Search the web using Tavily, an AI-optimised search engine "
            "designed for AI agents. Unlike basic search engines, Tavily "
            "crawls pages and extracts their full text content, and can "
            "provide an AI-generated answer summary synthesised from the "
            "results.\n"
            "\n"
            "Two search depths are available:\n"
            "- 'basic'    — fast, returns snippets (similar to DuckDuckGo)\n"
            "- 'advanced' — deep crawl, returns extracted page text "
            "(default, recommended for factual research)\n"
            "\n"
            "PREFER THIS TOOL over search_web when:\n"
            "- You need accurate, detailed factual information\n"
            "- You need to verify specific claims or numbers\n"
            "- Search snippets alone are too short or ambiguous\n"
            "- The query requires understanding full page content\n"
            "\n"
            "Use search_web (DuckDuckGo) as a fallback when Tavily is "
            "unavailable or for quick, low-stakes lookups.\n"
            "\n"
            "Output includes: AI summary (optional), title, URL, relevance "
            "score, and extracted page content per result.\n"
            "\n"
            "If results for one query are insufficient, issue a follow-up "
            "search with a different angle or more specific terms before "
            "drawing conclusions."
        ),
        "input_schema": TavilySearchInput,
        "function": tavily_search,
        "requires_confirmation": False,
    },
    {
        "name": "tavily_extract",
        "description": (
            "Extract clean text content from one or more web pages using "
            "Tavily. Handles JavaScript-rendered pages automatically and "
            "returns readable text.\n"
            "\n"
            "USE THIS TOOL WHEN:\n"
            "- You have specific URLs (e.g. from a previous search) and "
            "need their full content for analysis\n"
            "- You need to read a full article, documentation page, or "
            "blog post\n"
            "- The page is JavaScript-heavy and http_get returns raw HTML\n"
            "\n"
            "Accepts up to 20 URLs per call.  Returns extracted text per "
            "URL, plus a list of any URLs that failed extraction."
        ),
        "input_schema": TavilyExtractInput,
        "function": tavily_extract,
        "requires_confirmation": False,
    },
]

TOOL_CONFIG = TOOL_CONFIGS[0]

__all__ = [
    "tavily_search",
    "tavily_extract",
    "configure_tavily",
    "is_configured",
    "TavilySearchInput",
    "TavilyExtractInput",
    "TOOL_CONFIG",
    "TOOL_CONFIGS",
]
