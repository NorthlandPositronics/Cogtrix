"""Tests for the ``_search_async`` functions added to each provider
module in ADR-0056 PR-B.

Each provider exposes ``async def _search_async(query, num_results,
region) -> list[ProviderResult]``. These tests pin the contract:
provider tag, URL/title/snippet extraction, behaviour on missing
config / empty query / zero results.

All HTTP / SDK calls are mocked. No network.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools._web_search_aggregator import ProviderResult


class _FakeProc:
    """Minimal asyncio.subprocess.Process stand-in for the DDG
    subprocess sandbox path."""

    def __init__(self, stdout: bytes, returncode: int = 0) -> None:
        self._stdout = stdout
        self.returncode = returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, b""

    def kill(self) -> None:  # pragma: no cover - tests don't exercise timeout path
        pass

    async def wait(self) -> int:  # pragma: no cover - same
        return self.returncode


# ── DuckDuckGo ────────────────────────────────────────────────────────


class TestDdgAsync:
    @pytest.mark.asyncio
    async def test_returns_provider_results(self) -> None:
        """``_search_async`` parses the DDG worker's JSON output and
        maps DDG's raw fields (``href`` / ``link`` / ``body`` /
        ``snippet``) into ProviderResult.

        Note: DDG runs in a subprocess to contain primp heap aborts
        (Bug D fix). We mock the subprocess boundary; the field-mapping
        contract is the same as the pre-sandbox in-process code path.
        """
        from src.tools import web_search as mod

        proc = _FakeProc(
            stdout=json.dumps(
                {
                    "results": [
                        {
                            "title": "Hello",
                            "href": "https://example.com/hello",
                            "body": "world",
                        },
                        {
                            "title": "Foo",
                            "link": "https://example.com/foo",
                            "snippet": "bar",
                        },
                    ]
                }
            ).encode()
        )

        with (
            patch.object(mod, "DDGS_AVAILABLE", True),
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)),
        ):
            results = await mod._search_async("anything")

        assert len(results) == 2
        assert results[0] == ProviderResult(
            provider="ddg",
            url="https://example.com/hello",
            title="Hello",
            snippet="world",
            published_date=None,
        )
        assert results[1].url == "https://example.com/foo"
        assert results[1].snippet == "bar"

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty_list(self) -> None:
        from src.tools import web_search as mod

        with patch.object(mod, "DDGS_AVAILABLE", True):
            assert await mod._search_async("   ") == []

    @pytest.mark.asyncio
    async def test_missing_package_raises(self) -> None:
        """When the DDG scraper deps (curl_cffi → ``_ddg``) are
        unavailable, ``_search_async`` must raise a clear RuntimeError
        so the aggregator records this as a per-provider failure."""
        from src.tools import web_search as mod

        with patch.object(mod, "DDG_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="curl_cffi"):
                await mod._search_async("query")


# ── Tavily ────────────────────────────────────────────────────────────


class TestTavilyAsync:
    @pytest.mark.asyncio
    async def test_returns_provider_results(self) -> None:
        from src.tools import tavily_search as mod

        fake_client = MagicMock()
        fake_client.search.return_value = {
            "results": [
                {
                    "title": "Doc",
                    "url": "https://docs.example.com/x",
                    "content": "Lorem ipsum",
                    "score": 0.9,
                }
            ],
        }
        with (
            patch.object(mod, "TAVILY_AVAILABLE", True),
            patch.object(mod, "_get_client", return_value=fake_client),
        ):
            results = await mod._search_async("query")

        assert len(results) == 1
        assert results[0].provider == "tavily"
        assert results[0].url == "https://docs.example.com/x"
        assert results[0].snippet == "Lorem ipsum"

    @pytest.mark.asyncio
    async def test_missing_sdk_raises(self) -> None:
        from src.tools import tavily_search as mod

        with patch.object(mod, "TAVILY_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="Tavily SDK"):
                await mod._search_async("query")

    @pytest.mark.asyncio
    async def test_missing_api_key_raises(self) -> None:
        from src.tools import tavily_search as mod

        with (
            patch.object(mod, "TAVILY_AVAILABLE", True),
            patch.object(mod, "_get_client", side_effect=RuntimeError("no key")),
        ):
            with pytest.raises(RuntimeError):
                await mod._search_async("query")


# ── Brave ─────────────────────────────────────────────────────────────


class TestBraveAsync:
    @pytest.mark.asyncio
    async def test_returns_provider_results(self) -> None:
        from src.tools import brave_search as mod

        fake_response = MagicMock()
        fake_response.json.return_value = {
            "web": {
                "results": [
                    {
                        "title": "Brave result",
                        "url": "https://example.com/b",
                        "description": "A description",
                    }
                ]
            }
        }
        fake_response.raise_for_status.return_value = None
        with (
            patch.object(mod, "_get_api_key", return_value="fake-key"),
            patch.object(mod.requests, "get", return_value=fake_response),
        ):
            results = await mod._search_async("query")
        assert len(results) == 1
        assert results[0].provider == "brave"
        assert results[0].url == "https://example.com/b"
        assert results[0].snippet == "A description"

    @pytest.mark.asyncio
    async def test_missing_api_key_raises(self) -> None:
        from src.tools import brave_search as mod

        with patch.object(mod, "_get_api_key", return_value=None):
            with pytest.raises(RuntimeError, match="Brave"):
                await mod._search_async("query")


# ── Google CSE ────────────────────────────────────────────────────────


class TestGoogleAsync:
    @pytest.mark.asyncio
    async def test_returns_results_with_date_extraction(self) -> None:
        from src.tools import google_search as mod

        fake_response = MagicMock()
        fake_response.json.return_value = {
            "items": [
                {
                    "title": "Article",
                    "link": "https://news.example.com/x",
                    "snippet": "Snippet text\nwith newline",
                    "pagemap": {"metatags": [{"article:published_time": "2026-03-15T08:00:00Z"}]},
                }
            ]
        }
        fake_response.raise_for_status.return_value = None
        with (
            patch.object(mod, "_get_api_key", return_value="key"),
            patch.object(mod, "_get_cse_id", return_value="cse"),
            patch.object(mod.requests, "get", return_value=fake_response),
        ):
            results = await mod._search_async("query")

        assert len(results) == 1
        assert results[0].provider == "google"
        assert results[0].url == "https://news.example.com/x"
        assert results[0].snippet == "Snippet text with newline"  # newline collapsed
        assert results[0].published_date == "2026-03-15"  # first 10 chars

    @pytest.mark.asyncio
    async def test_missing_key_or_cse_raises(self) -> None:
        from src.tools import google_search as mod

        with patch.object(mod, "_get_api_key", return_value=None):
            with pytest.raises(RuntimeError, match="Google API key"):
                await mod._search_async("query")

        with (
            patch.object(mod, "_get_api_key", return_value="key"),
            patch.object(mod, "_get_cse_id", return_value=None),
        ):
            with pytest.raises(RuntimeError, match="Custom Search"):
                await mod._search_async("query")


# ── Exa ───────────────────────────────────────────────────────────────


class TestExaAsync:
    @pytest.mark.asyncio
    async def test_returns_provider_results(self) -> None:
        from src.tools import exa_search as mod

        class FakeExaResult:
            def __init__(self, url: str, title: str, text: str) -> None:
                self.url = url
                self.title = title
                self.text = text

        fake_response = MagicMock()
        fake_response.results = [
            FakeExaResult("https://example.com/a", "Title A", "Text A"),
        ]
        fake_client = MagicMock()
        fake_client.search.return_value = fake_response

        with (
            patch.object(mod, "EXA_AVAILABLE", True),
            patch.object(mod, "_get_client", return_value=fake_client),
        ):
            results = await mod._search_async("query")

        assert len(results) == 1
        assert results[0].provider == "exa"
        assert results[0].url == "https://example.com/a"
        assert results[0].snippet == "Text A"

    @pytest.mark.asyncio
    async def test_missing_sdk_raises(self) -> None:
        from src.tools import exa_search as mod

        with patch.object(mod, "EXA_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="Exa SDK"):
                await mod._search_async("query")


# ── SerpAPI ───────────────────────────────────────────────────────────


class TestSerpApiAsync:
    @pytest.mark.asyncio
    async def test_returns_provider_results(self) -> None:
        from src.tools import serpapi_search as mod

        fake_search = MagicMock()
        fake_search.get_dict.return_value = {
            "organic_results": [
                {
                    "title": "Org result",
                    "link": "https://example.com/s",
                    "snippet": "Snippet",
                }
            ]
        }
        with (
            patch.object(mod, "SERPAPI_AVAILABLE", True),
            patch.object(mod, "_get_api_key", return_value="key"),
            patch.object(mod, "GoogleSearch", return_value=fake_search),
        ):
            results = await mod._search_async("query")

        assert len(results) == 1
        assert results[0].provider == "serpapi"
        assert results[0].url == "https://example.com/s"

    @pytest.mark.asyncio
    async def test_api_error_in_response_raises(self) -> None:
        from src.tools import serpapi_search as mod

        fake_search = MagicMock()
        fake_search.get_dict.return_value = {"error": "Invalid API key"}
        with (
            patch.object(mod, "SERPAPI_AVAILABLE", True),
            patch.object(mod, "_get_api_key", return_value="key"),
            patch.object(mod, "GoogleSearch", return_value=fake_search),
        ):
            with pytest.raises(RuntimeError, match="SerpAPI error"):
                await mod._search_async("query")

    @pytest.mark.asyncio
    async def test_missing_sdk_raises(self) -> None:
        from src.tools import serpapi_search as mod

        with patch.object(mod, "SERPAPI_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="SerpAPI SDK"):
                await mod._search_async("query")


# ── SearXNG ───────────────────────────────────────────────────────────


class TestSearXNGAsync:
    @pytest.mark.asyncio
    async def test_returns_provider_results(self) -> None:
        import httpx

        from src.tools import searxng_search as mod

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Searx result",
                            "url": "https://example.com/sx",
                            "content": "Snippet",
                        }
                    ]
                },
            )

        # Patch AsyncClient to use a MockTransport.
        real_async_client = httpx.AsyncClient

        def fake_async_client(**_kwargs: Any) -> httpx.AsyncClient:
            return real_async_client(transport=httpx.MockTransport(handler))

        with (
            patch.object(mod, "_get_url", return_value="https://searx.example.com"),
            patch.object(mod.httpx, "AsyncClient", fake_async_client),
        ):
            results = await mod._search_async("query")

        assert len(results) == 1
        assert results[0].provider == "searxng"
        assert results[0].url == "https://example.com/sx"

    @pytest.mark.asyncio
    async def test_missing_url_raises(self) -> None:
        from src.tools import searxng_search as mod

        with patch.object(mod, "_get_url", return_value=None):
            with pytest.raises(RuntimeError, match="SearXNG URL"):
                await mod._search_async("query")


# ── Cross-cutting: ProviderResult uniformity ──────────────────────────


class TestProviderResultUniformity:
    """Every ``_search_async`` returns the same canonical shape."""

    @pytest.mark.asyncio
    async def test_all_providers_emit_correct_provider_tag(self) -> None:
        """Sanity check: each provider tags its results with the right
        short name. This is what the stage-1 aggregator uses for the
        consensus-count signal and the Coverage block."""

        from src.tools import (
            brave_search,
            exa_search,
            google_search,
            searxng_search,
            serpapi_search,
            tavily_search,
        )
        from src.tools import (
            web_search as ddg,
        )

        # Each block patches the provider's deps, calls _search_async with
        # a single-result happy-path mock, and records the resulting tag.

        # DDG — routes through subprocess + curl_cffi. The subprocess
        # worker emits canned JSON on stdout; we mock the boundary
        # ``asyncio.create_subprocess_exec`` rather than reach into the
        # subprocess internals.
        ddg_proc = _FakeProc(
            stdout=json.dumps(
                {"results": [{"title": "t", "href": "https://example.com/", "body": "b"}]}
            ).encode()
        )

        # Tavily
        tav_client = MagicMock()
        tav_client.search.return_value = {
            "results": [{"title": "t", "url": "https://example.com/", "content": "c"}]
        }

        # Brave
        brave_resp = MagicMock()
        brave_resp.json.return_value = {
            "web": {"results": [{"title": "t", "url": "https://example.com/", "description": "d"}]}
        }
        brave_resp.raise_for_status.return_value = None

        # Google
        goog_resp = MagicMock()
        goog_resp.json.return_value = {
            "items": [{"title": "t", "link": "https://example.com/", "snippet": "s", "pagemap": {}}]
        }
        goog_resp.raise_for_status.return_value = None

        # Exa
        class FakeExaResult:
            url = "https://example.com/"
            title = "t"
            text = "x"

        exa_resp = MagicMock()
        exa_resp.results = [FakeExaResult()]
        exa_client = MagicMock()
        exa_client.search.return_value = exa_resp

        # SerpAPI
        serp_search = MagicMock()
        serp_search.get_dict.return_value = {
            "organic_results": [{"title": "t", "link": "https://example.com/", "snippet": "s"}]
        }

        # SearXNG via MockTransport
        import httpx

        def searx_handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"results": [{"title": "t", "url": "https://example.com/", "content": "c"}]},
            )

        real_searx_client = httpx.AsyncClient

        def fake_searx_client(**_kw: Any) -> httpx.AsyncClient:
            return real_searx_client(transport=httpx.MockTransport(searx_handler))

        provider_tags: dict[str, str] = {}

        with (
            patch.object(ddg, "DDG_AVAILABLE", True),
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=ddg_proc)),
        ):
            r = await ddg._search_async("q")
            provider_tags["ddg"] = r[0].provider

        with (
            patch.object(tavily_search, "TAVILY_AVAILABLE", True),
            patch.object(tavily_search, "_get_client", return_value=tav_client),
        ):
            r = await tavily_search._search_async("q")
            provider_tags["tavily"] = r[0].provider

        with (
            patch.object(brave_search, "_get_api_key", return_value="k"),
            patch.object(brave_search.requests, "get", return_value=brave_resp),
        ):
            r = await brave_search._search_async("q")
            provider_tags["brave"] = r[0].provider

        with (
            patch.object(google_search, "_get_api_key", return_value="k"),
            patch.object(google_search, "_get_cse_id", return_value="cse"),
            patch.object(google_search.requests, "get", return_value=goog_resp),
        ):
            r = await google_search._search_async("q")
            provider_tags["google"] = r[0].provider

        with (
            patch.object(exa_search, "EXA_AVAILABLE", True),
            patch.object(exa_search, "_get_client", return_value=exa_client),
        ):
            r = await exa_search._search_async("q")
            provider_tags["exa"] = r[0].provider

        with (
            patch.object(serpapi_search, "SERPAPI_AVAILABLE", True),
            patch.object(serpapi_search, "_get_api_key", return_value="k"),
            patch.object(serpapi_search, "GoogleSearch", return_value=serp_search),
        ):
            r = await serpapi_search._search_async("q")
            provider_tags["serpapi"] = r[0].provider

        with (
            patch.object(searxng_search, "_get_url", return_value="https://searx.example.com"),
            patch.object(searxng_search.httpx, "AsyncClient", fake_searx_client),
        ):
            r = await searxng_search._search_async("q")
            provider_tags["searxng"] = r[0].provider

        assert provider_tags == {
            "ddg": "ddg",
            "tavily": "tavily",
            "brave": "brave",
            "google": "google",
            "exa": "exa",
            "serpapi": "serpapi",
            "searxng": "searxng",
        }
