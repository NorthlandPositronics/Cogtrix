"""Tests for cogtrix_core/tools/_web_search_fetcher.py — stage 3 fetch fan-out
(ADR-0056 PR-C)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from cogtrix_core.tools._http_fetch import FetchResult
from cogtrix_core.tools._web_search_aggregator import RankedResult
from cogtrix_core.tools._web_search_domain_class import DomainClass
from cogtrix_core.tools._web_search_fetcher import fetch_top_k


def _rank(url: str) -> RankedResult:
    """Build a minimal RankedResult for fetcher tests."""
    return RankedResult(
        canonical_url=url,
        title="T",
        snippet="S",
        published_date=None,
        domain_class=DomainClass.UNKNOWN,
        score=1.0,
        providers=("ddg",),
    )


def _ok_result(url: str, body: bytes = b"<html>ok</html>") -> FetchResult:
    return FetchResult(
        url=url,
        status_code=200,
        content=body,
        encoding="utf-8",
        content_type="text/html",
        elapsed_ms=10,
        truncated=False,
        error=None,
    )


def _err_result(url: str, error: str) -> FetchResult:
    return FetchResult(
        url=url,
        status_code=None,
        content=None,
        encoding=None,
        content_type=None,
        elapsed_ms=10,
        truncated=False,
        error=error,
    )


class TestFetchTopK:
    @pytest.mark.asyncio
    async def test_empty_input(self) -> None:
        assert await fetch_top_k([]) == []

    @pytest.mark.asyncio
    async def test_single_success(self) -> None:
        rank = _rank("https://example.com/a")
        with patch(
            "cogtrix_core.tools._web_search_fetcher.fetch_async",
            AsyncMock(return_value=_ok_result("https://example.com/a")),
        ):
            outcomes = await fetch_top_k([rank])
        assert len(outcomes) == 1
        assert outcomes[0].status == "fetched"
        assert outcomes[0].error is None
        assert outcomes[0].fetch_result is not None
        assert outcomes[0].fetch_result.content == b"<html>ok</html>"

    @pytest.mark.asyncio
    async def test_truncated_marked_fetched_with_warning(self) -> None:
        rank = _rank("https://example.com/a")
        big = FetchResult(
            url="https://example.com/a",
            status_code=200,
            content=b"x" * 1000,
            encoding="utf-8",
            content_type="text/html",
            elapsed_ms=10,
            truncated=True,
            error=None,
        )
        with patch(
            "cogtrix_core.tools._web_search_fetcher.fetch_async", AsyncMock(return_value=big)
        ):
            outcomes = await fetch_top_k([rank])
        assert outcomes[0].status == "fetched-with-warning"

    @pytest.mark.asyncio
    async def test_empty_body_marked_with_warning(self) -> None:
        rank = _rank("https://example.com/a")
        empty = FetchResult(
            url="https://example.com/a",
            status_code=200,
            content=b"",
            encoding="utf-8",
            content_type="text/html",
            elapsed_ms=10,
            truncated=False,
            error=None,
        )
        with patch(
            "cogtrix_core.tools._web_search_fetcher.fetch_async", AsyncMock(return_value=empty)
        ):
            outcomes = await fetch_top_k([rank])
        assert outcomes[0].status == "fetched-with-warning"
        assert outcomes[0].error == "empty-body"

    @pytest.mark.asyncio
    async def test_robots_blocked_becomes_snippet_only(self) -> None:
        rank = _rank("https://example.com/a")
        with patch(
            "cogtrix_core.tools._web_search_fetcher.fetch_async",
            AsyncMock(return_value=_err_result("https://example.com/a", "blocked-robots")),
        ):
            outcomes = await fetch_top_k([rank])
        assert outcomes[0].status == "snippet-only"
        assert outcomes[0].error == "blocked-robots"
        assert outcomes[0].fetch_result is None

    @pytest.mark.asyncio
    async def test_ssl_error_becomes_snippet_only(self) -> None:
        rank = _rank("https://example.com/a")
        with patch(
            "cogtrix_core.tools._web_search_fetcher.fetch_async",
            AsyncMock(return_value=_err_result("https://example.com/a", "ssl-error")),
        ):
            outcomes = await fetch_top_k([rank])
        assert outcomes[0].status == "snippet-only"
        assert outcomes[0].error == "ssl-error"

    @pytest.mark.asyncio
    async def test_4xx_status_becomes_snippet_only(self) -> None:
        rank = _rank("https://example.com/a")
        not_found = FetchResult(
            url="https://example.com/a",
            status_code=404,
            content=b"Not Found",
            encoding="utf-8",
            content_type="text/html",
            elapsed_ms=10,
            truncated=False,
            error="http-status",
        )
        with patch(
            "cogtrix_core.tools._web_search_fetcher.fetch_async",
            AsyncMock(return_value=not_found),
        ):
            outcomes = await fetch_top_k([rank])
        assert outcomes[0].status == "snippet-only"
        assert outcomes[0].error == "http-status"

    @pytest.mark.asyncio
    async def test_mixed_outcomes_preserve_order(self) -> None:
        ranks = [
            _rank("https://example.com/a"),
            _rank("https://example.com/b"),
            _rank("https://example.com/c"),
        ]

        # a → fetched, b → snippet-only (blocked), c → fetched
        async def fake_fetch(url: str, **_kw: Any) -> FetchResult:
            if url == "https://example.com/b":
                return _err_result(url, "blocked-robots")
            return _ok_result(url, b"<html>" + url.encode() + b"</html>")

        with patch("cogtrix_core.tools._web_search_fetcher.fetch_async", new=fake_fetch):
            outcomes = await fetch_top_k(ranks)
        # Order preserved positionally.
        assert outcomes[0].ranked.canonical_url == "https://example.com/a"
        assert outcomes[0].status == "fetched"
        assert outcomes[1].ranked.canonical_url == "https://example.com/b"
        assert outcomes[1].status == "snippet-only"
        assert outcomes[2].ranked.canonical_url == "https://example.com/c"
        assert outcomes[2].status == "fetched"

    @pytest.mark.asyncio
    async def test_fetch_raises_caught(self) -> None:
        """An exception bubbling out of fetch_async is caught per-URL."""
        rank = _rank("https://example.com/a")
        with patch(
            "cogtrix_core.tools._web_search_fetcher.fetch_async",
            AsyncMock(side_effect=RuntimeError("kaboom")),
        ):
            outcomes = await fetch_top_k([rank])
        assert outcomes[0].status == "snippet-only"
        assert outcomes[0].error == "RuntimeError"

    @pytest.mark.asyncio
    async def test_default_client_closed(self) -> None:
        """When no client is passed, fetch_top_k owns and closes one."""
        rank = _rank("https://example.com/a")
        real_async_client = httpx.AsyncClient

        clients: list[httpx.AsyncClient] = []

        def fake_async_client(**_kwargs: Any) -> httpx.AsyncClient:
            c = real_async_client(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
            clients.append(c)
            return c

        with (
            patch(
                "cogtrix_core.tools._web_search_fetcher.httpx.AsyncClient", new=fake_async_client
            ),
            patch(
                "cogtrix_core.tools._web_search_fetcher.fetch_async",
                AsyncMock(return_value=_ok_result("https://example.com/a")),
            ),
        ):
            await fetch_top_k([rank])

        assert len(clients) == 1
        # is_closed becomes True after aclose().
        assert clients[0].is_closed


class TestParallelDnsAtFetcherLayer:
    """Bug-A regression at the ``fetch_top_k`` integration layer.

    The lower-level ``fetch_async`` test in ``test_http_fetch.py``
    covers the underlying ``asyncio.to_thread`` offload of
    ``_validate_url``. This test pins the same property one layer up
    — ``fetch_top_k`` orchestrating K parallel ``fetch_async`` calls
    via ``asyncio.gather`` — so a regression that broke parallelism
    in the orchestration (e.g., a missed ``await``, a global lock
    accidentally introduced into ``_one``) is also caught.

    Scenario from the cogtrix31.log incident: 6 distinct-host URLs,
    DNS resolution takes ~0.5s each. Without parallelism the event
    loop serialises and 6 × 0.5s = 3s — well over the typical
    expected wall time for a parallelised stage-3 fetch.
    """

    @pytest.mark.asyncio
    async def test_six_hosts_parallel_dns_under_fetch_top_k(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import time as _time

        dns_call_starts: list[float] = []

        def slow_getaddrinfo(*_args, **_kwargs):
            dns_call_starts.append(_time.monotonic())
            _time.sleep(0.5)
            return [(None, None, None, "", ("93.184.216.34", 0))]

        monkeypatch.setattr("cogtrix_core.tools._http_safety.socket.getaddrinfo", slow_getaddrinfo)

        # We patch fetch_async to a fast no-op so the assertion is
        # purely about the DNS-stage parallelism inside fetch_top_k's
        # gather over _one(). fetch_async is called once per ranked
        # input; its body (which would also call _validate_url) is
        # replaced. To still exercise _validate_url-equivalent DNS
        # work, we go a layer up: don't mock fetch_async but rely on
        # the real one with a mock httpx client that completes
        # instantly. That way the only blocking work is the DNS
        # lookup inside fetch_async's _validate_url call.
        from cogtrix_core.tools._web_search_fetcher import fetch_top_k as _fetch_top_k

        ranks = [_rank(f"https://host{i}.example.com/path") for i in range(6)]

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/robots.txt":
                return httpx.Response(404)
            return httpx.Response(
                200, content=b"<html>ok</html>", headers={"Content-Type": "text/html"}
            )

        from cogtrix_core.tools._http_fetch import USER_AGENT

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            follow_redirects=False,
            headers={"User-Agent": USER_AGENT},
        )

        t0 = _time.monotonic()
        outcomes = await _fetch_top_k(ranks, client=client)
        elapsed = _time.monotonic() - t0
        await client.aclose()

        # All six should succeed because the mock transport returns 200.
        assert all(o.status == "fetched" for o in outcomes), [o.status for o in outcomes]
        # All six DNS calls happened (one per distinct host, no cache).
        assert len(dns_call_starts) >= 6
        # Parallel-execution ceiling: under 1.5s. Six sequential 0.5s
        # blocking lookups would be ≥3s; that's the regression we're
        # guarding.
        assert elapsed < 1.5, (
            f"fetch_top_k appears to be serialising on DNS (elapsed {elapsed:.2f}s); "
            f"expected concurrent execution via asyncio.to_thread offload"
        )


class TestSlowUrlDoesNotPoisonBatch:
    """Bug F regression: a single slow URL must not blow away successful
    siblings. The pre-fix shape wrapped the whole ``asyncio.gather`` in
    ``wait_for(timeout=deadline_s)``; when one URL was slow, the outer
    TimeoutError fired and the except clause replaced every outcome
    with ``status='skipped, error=timeout'`` — including ones that had
    already produced a result. Real-world impact: 0/N fetched whenever
    the slowest URL in a batch was slow enough."""

    @pytest.mark.asyncio
    async def test_slow_sibling_does_not_overwrite_fast_successes(self) -> None:
        ranks = [
            _rank("https://fast-a.example.com/x"),
            _rank("https://slow.example.com/x"),
            _rank("https://fast-b.example.com/x"),
        ]

        async def fake_fetch(url: str, **_kw: Any) -> FetchResult:
            if "slow" in url:
                # Sleep past the per-task deadline so this URL becomes
                # "snippet-only timeout" — but the fast siblings should
                # have already completed and their results retained.
                await asyncio.sleep(3.0)
                return _ok_result(url)
            return _ok_result(url, b"<html>" + url.encode() + b"</html>")

        with patch("cogtrix_core.tools._web_search_fetcher.fetch_async", new=fake_fetch):
            # deadline_s=1.0 → per-task hard deadline=2.0s. The slow
            # sibling's 3s sleep blows that, fast siblings finish at ~0s.
            outcomes = await fetch_top_k(ranks, deadline_s=1.0)

        # Fast siblings retained their successful fetch.
        assert outcomes[0].status == "fetched"
        assert outcomes[0].fetch_result is not None
        assert b"fast-a" in outcomes[0].fetch_result.content
        assert outcomes[2].status == "fetched"
        assert outcomes[2].fetch_result is not None
        assert b"fast-b" in outcomes[2].fetch_result.content
        # Slow sibling became snippet-only individually.
        assert outcomes[1].status == "snippet-only"
        assert outcomes[1].error == "timeout"

    @pytest.mark.asyncio
    async def test_all_slow_still_returns_individual_timeouts(self) -> None:
        """When every URL is slow, each becomes snippet-only/timeout —
        not ``status='skipped'`` as the pre-fix code emitted. The
        positional ordering is also preserved."""
        ranks = [_rank(f"https://slow-{i}.example.com/x") for i in range(3)]

        async def fake_fetch(url: str, **_kw: Any) -> FetchResult:
            await asyncio.sleep(3.0)
            return _ok_result(url)

        with patch("cogtrix_core.tools._web_search_fetcher.fetch_async", new=fake_fetch):
            outcomes = await fetch_top_k(ranks, deadline_s=0.5)

        assert len(outcomes) == 3
        for i, outcome in enumerate(outcomes):
            assert outcome.status == "snippet-only", (
                f"outcome[{i}] should be snippet-only after per-task timeout, "
                f"got {outcome.status}"
            )
            assert outcome.error == "timeout"
