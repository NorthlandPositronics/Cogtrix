"""
SearXNG — privacy-respecting self-hosted metasearch engine.

SearXNG aggregates results from 70+ search engines without tracking users.
It exposes a JSON REST API at ``/search?q=...&format=json``.

Configuration:
    Environment variable: ``SEARXNG_URL``   (e.g. http://localhost:8888)
    Config file:          ``services.searxng.url``

The tool is hidden when no URL is configured (``is_configured()`` returns False).
"""

import logging
import os
from typing import Any

import httpx
from pydantic import BaseModel, Field

log = logging.getLogger("cogtrix")

_searxng_config: dict[str, Any] = {}


def configure_searxng(config: dict[str, Any]) -> None:
    """Set runtime configuration. Called from configure.py during startup."""
    global _searxng_config
    _searxng_config = {**_searxng_config, **config}


def _get_url() -> str | None:
    return _searxng_config.get("url") or os.getenv("SEARXNG_URL")


def is_configured() -> bool:
    return bool(_get_url())


class SearXNGSearchInput(BaseModel):
    query: str = Field(description="The search query")
    num_results: int = Field(default=5, description="Number of results to return (1-10)")
    language: str = Field(default="en", description="Language code for results (e.g. 'en', 'de')")


def searxng_search(query: str, num_results: int = 5, language: str = "en") -> str:
    """
    Search the web using a self-hosted SearXNG instance.

    SearXNG aggregates results from multiple search engines without tracking.
    Requires a running SearXNG instance configured via SEARXNG_URL or
    services.searxng.url in .cogtrix.json.

    Args:
        query: Search query string.
        num_results: Number of results to return (1-10).
        language: Language code for results.

    Returns:
        Formatted search results.
    """
    base_url = _get_url()
    if not base_url:
        return (
            "Error: SearXNG URL not configured. "
            "Set SEARXNG_URL environment variable or add "
            '"services": {"searxng": {"url": "http://localhost:8888"}} to .cogtrix.json'
        )

    if not query.strip():
        return "Error: Empty search query"

    num_results = max(1, min(num_results, 10))
    base_url = base_url.rstrip("/")

    try:
        resp = httpx.get(
            f"{base_url}/search",
            params={"q": query, "format": "json", "language": language},
            timeout=10.0,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        return f"Error: SearXNG returned HTTP {e.response.status_code}"
    except httpx.RequestError as e:
        return f"Error: Could not connect to SearXNG at {base_url}: {e}"
    except Exception as e:
        return f"Error: {e}"

    results = data.get("results", [])[:num_results]
    if not results:
        return f"No results found for: {query}"

    output = [f"SearXNG results for: {query}\n"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        url = r.get("url", "")
        content = r.get("content", "")
        output.append(f"{i}. {title}")
        if url:
            output.append(f"   URL: {url}")
        if content:
            if len(content) > 500:
                content = content[:500] + "..."
            output.append(f"   {content}")
        output.append("")

    return "\n".join(output)


TOOL_CONFIGS = [
    {
        "name": "searxng_search",
        "description": (
            "Search the web using a self-hosted SearXNG instance. "
            "Privacy-respecting metasearch across 70+ engines. "
            "Only available when SEARXNG_URL is configured."
        ),
        "input_schema": SearXNGSearchInput,
        "function": searxng_search,
        "requires_confirmation": False,
    }
]

TOOL_CONFIG = TOOL_CONFIGS[0]

__all__ = [
    "searxng_search",
    "configure_searxng",
    "is_configured",
    "SearXNGSearchInput",
    "TOOL_CONFIG",
    "TOOL_CONFIGS",
]
