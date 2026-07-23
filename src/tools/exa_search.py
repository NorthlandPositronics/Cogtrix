"""
Exa Search - AI-native semantic web search with content extraction.

Exa uses neural embeddings to understand the *meaning* of queries, not
just keywords.  This makes it particularly effective for conceptual,
exploratory, or ambiguous searches where traditional keyword engines
struggle.

Architecture::

    ┌──────────┐        ┌─────────────────┐        ┌──────────────┐
    │  Agent   │──q──→  │   Exa API       │──q──→  │  Neural      │
    │          │        │                 │        │  index       │
    │          │←─────  │  search │similar│←─────  │  + extract   │
    └──────────┘  json  │  contents      │  emb   └──────────────┘
                        └─────────────────┘

    search_type="auto"     → Exa picks the best strategy automatically
    search_type="neural"   → semantic/meaning-based ranking
    search_type="fast"     → quick, lower-latency search
    search_type="deep"     → thorough, higher-quality search
    search_type="instant"  → fastest, cached results

Three tools are exposed:
    exa_search       - Semantic web search with optional content extraction.
    exa_find_similar - Find pages similar to a given URL.
    exa_get_contents - Extract text from specific URLs.

Configuration:
    Environment variable: ``EXA_API_KEY``
    Config file:          ``services.exa.api_key``
                          (legacy: ``exa.api_key`` at top level)
    Free tier:            1 000 searches / month

The tool is automatically removed from the agent if the ``exa-py``
package is not installed or the API key is not configured.
"""

import logging
import os
from typing import Any

from pydantic import BaseModel, Field

from src.tools.error_sanitizer import sanitize_error as _sanitize_error

log = logging.getLogger("cogtrix")

# -- Optional import -----------------------------------------------------------

try:
    from exa_py import Exa  # type: ignore[import-untyped]

    EXA_AVAILABLE = True
except ImportError:
    Exa = None  # type: ignore[misc, assignment]
    EXA_AVAILABLE = False

# -- Module-level configuration ------------------------------------------------

_exa_config: dict[str, Any] = {}


def configure_exa(config: dict[str, Any]) -> None:
    """
    Set runtime configuration.  Called from ``cogtrix.py`` during startup.

    Expected keys:
        api_key  - Exa API key (or read from EXA_API_KEY env var)
    """
    global _exa_config
    # Atomic reference swap — safe for concurrent readers without a lock
    _exa_config = {**_exa_config, **config}


def _get_api_key() -> str | None:
    """Resolve API key from config or environment."""
    return _exa_config.get("api_key") or os.getenv("EXA_API_KEY")


def is_configured() -> bool:
    """Return True if the tool has the required API key and SDK."""
    return EXA_AVAILABLE and bool(_get_api_key())


def _get_client() -> Any:
    """Create an Exa client with the configured API key."""
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError(
            "Exa API key not configured. "
            "Set EXA_API_KEY environment variable or add "
            '"services": {"exa": {"api_key": "..."}} to .cogtrix.json'
        )
    return Exa(api_key=api_key)


# -- Input schemas -------------------------------------------------------------


class ExaSearchInput(BaseModel):
    """Input schema for Exa web search."""

    query: str = Field(description="The search query (can be a natural-language question)")
    num_results: int = Field(
        default=5,
        description="Number of results to return (1-10)",
    )
    include_text: bool = Field(
        default=True,
        description="Include extracted page text in results (recommended for accuracy)",
    )
    search_type: str = Field(
        default="auto",
        description=(
            "Search type: 'auto' (let Exa decide), 'neural' (semantic), "
            "'fast' (quick), 'deep' (thorough), or 'instant' (cached)"
        ),
    )


class ExaFindSimilarInput(BaseModel):
    """Input schema for Exa find-similar search."""

    url: str = Field(description="URL of the page to find similar pages for")
    num_results: int = Field(
        default=5,
        description="Number of similar results to return (1-10)",
    )
    include_text: bool = Field(
        default=True,
        description="Include extracted page text in results",
    )


class ExaGetContentsInput(BaseModel):
    """Input schema for Exa content extraction."""

    urls: list[str] = Field(
        description="List of URLs to extract content from",
    )


# -- Helpers -------------------------------------------------------------------


def _exa_error(operation: str, exc: Exception) -> str:
    """Return a user-facing error message, with a fallback hint on credits errors."""
    err = str(exc)
    if any(
        marker in err
        for marker in ("402", "credit", "quota", "payment", "Payment", "Credit", "Quota")
    ):
        return (
            "Error: Exa API credits exhausted or payment required. "
            "Use search_web or search_news as a free fallback instead."
        )
    return f"Error performing {operation}: {_sanitize_error(exc)}"


# -- Tool functions ------------------------------------------------------------


def exa_search(
    query: str,
    num_results: int = 5,
    include_text: bool = True,
    search_type: str = "auto",
) -> str:
    """
    Search the web using Exa - an AI-native semantic search engine.

    Exa understands the *meaning* of queries, not just keywords.  With
    ``include_text=True`` (default) it also returns extracted page text,
    giving the LLM real content to work with.

    Args:
        query: Natural-language search query.
        num_results: Number of results (1-10).
        include_text: Whether to include page text.
        search_type: 'auto', 'neural', 'fast', 'deep', or 'instant'.

    Returns:
        Formatted search results with content.
    """
    if not EXA_AVAILABLE:
        return "Error: exa-py is not installed. Run: uv add exa-py"

    if not query.strip():
        return "Error: Empty search query"

    num_results = max(1, min(num_results, 10))
    _valid_types = ("auto", "neural", "fast", "deep", "instant")
    if search_type not in _valid_types:
        search_type = "auto"

    try:
        client = _get_client()
        kwargs: dict[str, Any] = {
            "query": query,
            "num_results": num_results,
            "type": search_type,
        }
        if include_text:
            kwargs["contents"] = {"text": True}

        results = client.search(**kwargs)
    except RuntimeError as e:
        return f"Error: {e}"
    except Exception as e:
        return _exa_error("Exa search", e)

    return _format_search_results(query, results)


def exa_find_similar(
    url: str,
    num_results: int = 5,
    include_text: bool = True,
) -> str:
    """
    Find web pages similar to a given URL using Exa.

    Useful for discovering related articles, alternative sources, or
    competing products/services.

    Args:
        url: URL of the reference page.
        num_results: Number of similar results (1-10).
        include_text: Whether to include page text.

    Returns:
        Formatted list of similar pages with content.
    """
    if not EXA_AVAILABLE:
        return "Error: exa-py is not installed. Run: uv add exa-py"

    if not url.strip():
        return "Error: Empty URL"

    num_results = max(1, min(num_results, 10))

    try:
        client = _get_client()
        kwargs: dict[str, Any] = {
            "url": url,
            "num_results": num_results,
        }
        if include_text:
            kwargs["contents"] = {"text": True}

        results = client.find_similar(**kwargs)
    except RuntimeError as e:
        return f"Error: {e}"
    except Exception as e:
        return _exa_error("Exa find_similar", e)

    return _format_search_results(f"pages similar to {url}", results)


def exa_get_contents(urls: list[str]) -> str:
    """
    Extract clean text content from web pages using Exa.

    Give it one or more URLs and it returns readable text.
    Useful for reading full articles or documentation pages.

    Args:
        urls: List of URLs to extract content from.

    Returns:
        Extracted text content from each URL.
    """
    if not EXA_AVAILABLE:
        return "Error: exa-py is not installed. Run: uv add exa-py"

    if not urls:
        return "Error: No URLs provided"

    urls = urls[:20]

    try:
        client = _get_client()
        results = client.get_contents(urls, text=True)
    except RuntimeError as e:
        return f"Error: {e}"
    except Exception as e:
        return _exa_error("Exa get_contents", e)

    output: list[str] = []
    result_list = getattr(results, "results", [])
    for result in result_list:
        r_url = getattr(result, "url", "unknown")
        text = getattr(result, "text", "")

        output.append(f"## {r_url}\n")
        if text:
            if len(text) > 8000:
                text = text[:8000] + "\n\n... (truncated)"
            output.append(text)
        else:
            output.append("(no content extracted)")
        output.append("")

    if not result_list:
        output.append("No content could be extracted from the provided URLs.")

    return "\n".join(output)


# -- Helpers -------------------------------------------------------------------


def _format_search_results(query_desc: str, results: Any) -> str:
    """Format Exa search results into a readable string."""
    output: list[str] = [f"Exa search results for: {query_desc}\n"]

    result_list = getattr(results, "results", [])
    if not result_list:
        output.append("No results found.")
        return "\n".join(output)

    for i, result in enumerate(result_list, 1):
        title = getattr(result, "title", "No title")
        url = getattr(result, "url", "No URL")
        score = getattr(result, "score", None)
        text = getattr(result, "text", "")

        output.append(f"{i}. {title}")
        output.append(f"   URL: {url}")
        if score is not None:
            output.append(f"   Relevance: {score:.4f}")

        if text:
            if len(text) > 2000:
                text = text[:2000] + "..."
            output.append(f"   {text}")
        output.append("")

    return "\n".join(output)


# -- Tool registration --------------------------------------------------------

TOOL_CONFIGS = [
    {
        "name": "exa_search",
        "description": (
            "Semantic web search using Exa's neural embeddings. "
            "Use search_type='neural' for conceptual queries, 'deep' for thorough results. "
            "Set include_text=True to get extracted page content."
        ),
        "input_schema": ExaSearchInput,
        "function": exa_search,
        "requires_confirmation": False,
    },
    {
        "name": "exa_find_similar",
        "description": (
            "Find web pages similar to a given URL using Exa's neural embeddings. "
            "Returns pages with similar content and meaning."
        ),
        "input_schema": ExaFindSimilarInput,
        "function": exa_find_similar,
        "requires_confirmation": False,
    },
    {
        "name": "exa_get_contents",
        "description": (
            "Extract clean text content from one or more URLs. "
            "Accepts up to 20 URLs per call, truncated at 8000 chars per page."
        ),
        "input_schema": ExaGetContentsInput,
        "function": exa_get_contents,
        "requires_confirmation": False,
    },
]

TOOL_CONFIG = TOOL_CONFIGS[0]

__all__ = [
    "exa_search",
    "exa_find_similar",
    "exa_get_contents",
    "configure_exa",
    "is_configured",
    "ExaSearchInput",
    "ExaFindSimilarInput",
    "ExaGetContentsInput",
    "TOOL_CONFIG",
    "TOOL_CONFIGS",
]
