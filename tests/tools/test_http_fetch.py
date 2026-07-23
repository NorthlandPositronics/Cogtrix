"""Tests for src/tools/_http_fetch.py — async fetch primitive
(ADR-0056 PR-A2 stage 3).

Uses httpx.MockTransport to intercept HTTP calls without network. Each
test sets up a handler returning canned responses for the URLs it
cares about.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import httpx
import pytest

from src.tools import _http_fetch
from src.tools._http_fetch import USER_AGENT, FetchResult, fetch_async


def _public_dns_mock() -> Callable[..., list[tuple]]:
    """Make ``socket.getaddrinfo`` return a public IP so _validate_url passes."""

    def fake(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return [(None, None, None, "", ("93.184.216.34", 0))]

    return fake


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts with a fresh robots cache + host-spacing state."""
    _http_fetch._clear_robots_cache()
    _http_fetch._reset_host_state()
    # All tests use mocked DNS so we never hit real DNS.
    monkeypatch.setattr("src.tools._http_safety.socket.getaddrinfo", _public_dns_mock())


def _make_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    """Build an AsyncClient bound to a MockTransport with the given handler."""
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        follow_redirects=False,
        headers={"User-Agent": USER_AGENT},
    )


def _ok(body: bytes = b"<html>hi</html>", status: int = 200) -> Callable:
    """Handler returning *body* for any URL except /robots.txt (which 404s)."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(status, content=body, headers={"Content-Type": "text/html"})

    return handler


# ── Validation ────────────────────────────────────────────────────────


class TestValidation:
    @pytest.mark.asyncio
    async def test_invalid_url_returns_validation_failed(self) -> None:
        result = await fetch_async("not-a-url")
        assert result.error == "validation-failed"
        assert result.status_code is None

    @pytest.mark.asyncio
    async def test_blocked_scheme(self) -> None:
        result = await fetch_async("file:///etc/passwd")
        assert result.error == "validation-failed"

    @pytest.mark.asyncio
    async def test_blocked_internal_host(self) -> None:
        result = await fetch_async("http://localhost/path")
        assert result.error == "validation-failed"


# ── Happy path ────────────────────────────────────────────────────────


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_simple_get_returns_content(self) -> None:
        client = _make_client(_ok(b"<html>hi</html>"))
        result = await fetch_async("https://example.com/page", client=client)
        await client.aclose()
        assert result.error is None
        assert result.status_code == 200
        assert result.content == b"<html>hi</html>"
        assert result.content_type == "text/html"
        assert result.truncated is False

    @pytest.mark.asyncio
    async def test_http_400_marks_error_status(self) -> None:
        client = _make_client(_ok(b"not found", status=404))
        result = await fetch_async("https://example.com/missing", client=client)
        await client.aclose()
        assert result.error == "http-status"
        assert result.status_code == 404
        assert result.content == b"not found"


# ── robots.txt ────────────────────────────────────────────────────────


class TestRobotsTxt:
    @pytest.mark.asyncio
    async def test_blocked_by_robots_returns_blocked_robots(self) -> None:
        # robotparser matches the bot name in the User-Agent (the part
        # before the first ``/``) against the ``User-agent:`` directive.
        # We present as Chrome (``Mozilla/5.0 ... Chrome/N.0.0.0 Safari/...``),
        # so a wildcard ``*`` rule is what cleanly verifies that explicit
        # robots.txt denials are still honoured.
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/robots.txt":
                return httpx.Response(
                    200,
                    content=b"User-agent: *\nDisallow: /\n",
                )
            return httpx.Response(200, content=b"should-not-be-fetched")

        client = _make_client(handler)
        result = await fetch_async("https://example.com/secret", client=client)
        await client.aclose()
        assert result.error == "blocked-robots"
        assert result.content is None

    @pytest.mark.asyncio
    async def test_allowed_by_robots_proceeds(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/robots.txt":
                return httpx.Response(200, content=b"User-agent: *\nAllow: /\n")
            return httpx.Response(200, content=b"<html>ok</html>")

        client = _make_client(handler)
        result = await fetch_async("https://example.com/page", client=client)
        await client.aclose()
        assert result.error is None
        assert result.content == b"<html>ok</html>"

    @pytest.mark.asyncio
    async def test_robots_fetch_failure_fails_open(self) -> None:
        """Robots.txt 5xx / network error → treat as allowed."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/robots.txt":
                raise httpx.ConnectError("boom")
            return httpx.Response(200, content=b"ok")

        client = _make_client(handler)
        result = await fetch_async("https://example.com/page", client=client)
        await client.aclose()
        assert result.error is None
        assert result.content == b"ok"

    @pytest.mark.asyncio
    async def test_robots_cache_avoids_second_fetch(self) -> None:
        robots_fetches = [0]

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/robots.txt":
                robots_fetches[0] += 1
                return httpx.Response(200, content=b"User-agent: *\nAllow: /\n")
            return httpx.Response(200, content=b"ok")

        client = _make_client(handler)
        await fetch_async("https://example.com/a", client=client)
        await fetch_async("https://example.com/b", client=client)
        await client.aclose()
        assert robots_fetches[0] == 1


# ── Redirects ─────────────────────────────────────────────────────────


class TestRedirects:
    @pytest.mark.asyncio
    async def test_same_domain_redirect_followed(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/robots.txt":
                return httpx.Response(404)
            if request.url.path == "/start":
                return httpx.Response(302, headers={"Location": "/dest"})
            if request.url.path == "/dest":
                return httpx.Response(200, content=b"final")
            return httpx.Response(404)

        client = _make_client(handler)
        result = await fetch_async("https://example.com/start", client=client)
        await client.aclose()
        assert result.error is None
        assert result.content == b"final"

    @pytest.mark.asyncio
    async def test_cross_domain_redirect_refused(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/robots.txt":
                return httpx.Response(404)
            # With DNS-pinning the URL host is the IP; use the Host header
            # to identify the original hostname.
            if request.headers.get("Host") == "example.com":
                return httpx.Response(
                    302, headers={"Location": "https://attacker.example.org/landing"}
                )
            return httpx.Response(200, content=b"should-not-reach")

        client = _make_client(handler)
        result = await fetch_async("https://example.com/start", client=client)
        await client.aclose()
        assert result.error == "cross-domain-redirect"
        assert result.content is None

    @pytest.mark.asyncio
    async def test_too_many_redirects(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/robots.txt":
                return httpx.Response(404)
            # Always redirect to the next hop on same domain.
            n = int(request.url.path.lstrip("/") or "0")
            return httpx.Response(302, headers={"Location": f"/{n + 1}"})

        client = _make_client(handler)
        result = await fetch_async("https://example.com/0", client=client)
        await client.aclose()
        assert result.error == "too-many-redirects"


# ── HTTP 429 retry ────────────────────────────────────────────────────


class TestHttp429:
    @pytest.mark.asyncio
    async def test_429_then_200_retries_and_succeeds(self) -> None:
        attempts = [0]

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/robots.txt":
                return httpx.Response(404)
            attempts[0] += 1
            if attempts[0] == 1:
                return httpx.Response(429, headers={"Retry-After": "0"})
            return httpx.Response(200, content=b"second-try-ok")

        client = _make_client(handler)
        result = await fetch_async("https://example.com/", client=client, deadline_s=5.0)
        await client.aclose()
        assert result.error is None
        assert result.content == b"second-try-ok"
        assert attempts[0] == 2

    @pytest.mark.asyncio
    async def test_429_persistent_returns_http_status(self) -> None:
        """If 429 persists after the single retry, we surface it as http-status."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/robots.txt":
                return httpx.Response(404)
            return httpx.Response(429, headers={"Retry-After": "0"})

        client = _make_client(handler)
        result = await fetch_async("https://example.com/", client=client, deadline_s=5.0)
        await client.aclose()
        assert result.status_code == 429
        # Single retry happened; the second 429 is consumed as the final response.
        assert result.error == "http-status"


# ── Size cap ──────────────────────────────────────────────────────────


class TestSizeCap:
    @pytest.mark.asyncio
    async def test_oversized_body_truncated_at_cap(self) -> None:
        big = b"a" * (_http_fetch._RESPONSE_SIZE_CAP + 1024)
        client = _make_client(_ok(big))
        result = await fetch_async("https://example.com/big", client=client)
        await client.aclose()
        assert result.truncated is True
        assert result.content is not None
        assert len(result.content) <= _http_fetch._RESPONSE_SIZE_CAP

    @pytest.mark.asyncio
    async def test_under_cap_not_truncated(self) -> None:
        small = b"a" * 1024
        client = _make_client(_ok(small))
        result = await fetch_async("https://example.com/small", client=client)
        await client.aclose()
        assert result.truncated is False
        assert result.content == small


# ── Per-host spacing ─────────────────────────────────────────────────


class TestPerHostSpacing:
    @pytest.mark.asyncio
    async def test_back_to_back_same_host_spaced(self) -> None:
        """Two fetches to the same host must space by ≥ _PER_HOST_MIN_SPACING_S."""
        import time

        client = _make_client(_ok(b"ok"))
        t0 = time.monotonic()
        await fetch_async("https://example.com/a", client=client)
        await fetch_async("https://example.com/b", client=client)
        elapsed = time.monotonic() - t0
        await client.aclose()
        assert elapsed >= _http_fetch._PER_HOST_MIN_SPACING_S * 0.9  # generous


# ── SSL / connection errors ──────────────────────────────────────────


class TestConnectionErrors:
    @pytest.mark.asyncio
    async def test_ssl_error_mapped(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/robots.txt":
                return httpx.Response(404)
            raise httpx.ConnectError("SSL: CERTIFICATE_VERIFY_FAILED")

        client = _make_client(handler)
        result = await fetch_async("https://example.com/", client=client)
        await client.aclose()
        assert result.error == "ssl-error"

    @pytest.mark.asyncio
    async def test_generic_connect_error_mapped(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/robots.txt":
                return httpx.Response(404)
            raise httpx.ConnectError("Connection refused")

        client = _make_client(handler)
        result = await fetch_async("https://example.com/", client=client)
        await client.aclose()
        assert result.error == "connect-error"

    @pytest.mark.asyncio
    async def test_timeout_mapped(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/robots.txt":
                return httpx.Response(404)
            raise httpx.TimeoutException("slow")

        client = _make_client(handler)
        result = await fetch_async("https://example.com/", client=client)
        await client.aclose()
        assert result.error == "timeout"


# ── DNS-pinning (TOCTOU elimination) ──────────────────────────────────


class TestDNSPinning:
    """Verify that the resolved IP is pinned into the httpx request.

    With DNS-pinning the TCP connection targets the pre-validated IP
    while the HTTP ``Host`` header (and TLS SNI) keep the original
    hostname.  These tests assert the observable request properties.
    """

    @pytest.mark.asyncio
    async def test_pinned_request_uses_ip_literal_url(self) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/robots.txt":
                return httpx.Response(404)
            captured.append(request)
            return httpx.Response(200, content=b"ok")

        client = _make_client(handler)
        result = await fetch_async("https://example.com/page", client=client)
        await client.aclose()
        assert result.error is None
        assert len(captured) == 1
        # The mocked DNS resolver always returns 93.184.216.34
        assert captured[0].url.host == "93.184.216.34"

    @pytest.mark.asyncio
    async def test_pinned_request_preserves_host_header(self) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/robots.txt":
                return httpx.Response(404)
            captured.append(request)
            return httpx.Response(200, content=b"ok")

        client = _make_client(handler)
        result = await fetch_async("https://example.com/page", client=client)
        await client.aclose()
        assert result.error is None
        assert len(captured) == 1
        assert captured[0].headers.get("Host") == "example.com"

    @pytest.mark.asyncio
    async def test_pinned_request_includes_sni_extension(self) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/robots.txt":
                return httpx.Response(404)
            captured.append(request)
            return httpx.Response(200, content=b"ok")

        client = _make_client(handler)
        result = await fetch_async("https://example.com/page", client=client)
        await client.aclose()
        assert result.error is None
        assert len(captured) == 1
        assert captured[0].extensions.get("sni_hostname") == "example.com"

    @pytest.mark.asyncio
    async def test_redirect_updates_pin(self) -> None:
        """After a same-domain redirect the new target's IP is pinned."""
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/robots.txt":
                return httpx.Response(404)
            captured.append(request)
            if request.url.path == "/start":
                return httpx.Response(302, headers={"Location": "/dest"})
            return httpx.Response(200, content=b"final")

        client = _make_client(handler)
        result = await fetch_async("https://example.com/start", client=client)
        await client.aclose()
        assert result.error is None
        # Two requests: original + redirect. Both use the same mocked IP.
        assert len(captured) == 2
        assert all(r.url.host == "93.184.216.34" for r in captured)
        assert captured[0].headers.get("Host") == "example.com"
        assert captured[1].headers.get("Host") == "example.com"


# ── Default client ───────────────────────────────────────────────────


class TestDefaultClient:
    @pytest.mark.asyncio
    async def test_no_client_creates_and_closes_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When ``client`` is None, fetch_async builds + closes its own.

        Replace ``httpx.AsyncClient`` inside the module with a builder
        that returns a MockTransport-backed client. The fetch completes
        without error → lifecycle works.
        """
        # Bind the real class BEFORE monkeypatching — otherwise the fake
        # recurses into the patched symbol forever.
        real_async_client = httpx.AsyncClient

        def fake_async_client(**_kwargs):  # type: ignore[no-untyped-def]
            return real_async_client(
                transport=httpx.MockTransport(_ok(b"default")),
                follow_redirects=False,
                headers={"User-Agent": USER_AGENT},
            )

        monkeypatch.setattr(_http_fetch.httpx, "AsyncClient", fake_async_client)

        result = await fetch_async("https://example.com/")
        assert result.error is None
        assert result.content == b"default"


# ── Robots-fetch pinning + helper unit tests (follow-up to #1668) ─────


class TestRobotsTxtPinning:
    """The robots.txt probe shares its host with the main fetch and
    must be DNS-pinned the same way — otherwise an attacker could
    rebind DNS for the robots round-trip alone to probe internal
    endpoints. The merged PR #1673 left this gap; this verifies it
    is closed."""

    @pytest.mark.asyncio
    async def test_robots_request_uses_pinned_ip(self) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/robots.txt":
                captured["url_host"] = request.url.host
                captured["host_header"] = request.headers.get("Host", "")
                captured["sni"] = request.extensions.get("sni_hostname")
                return httpx.Response(200, content=b"User-agent: *\nAllow: /\n")
            return httpx.Response(200, content=b"page")

        client = _make_client(handler)
        await fetch_async("https://example.com/page", client=client)
        await client.aclose()

        # The fixture's mock DNS returns 93.184.216.34 for any host.
        assert captured["url_host"] == "93.184.216.34"
        assert captured["host_header"] == "example.com"
        assert captured["sni"] == "example.com"


class TestBuildPinnedUrl:
    """Direct unit tests for ``_build_pinned_url``. PR #1673 added the
    function but only exercised it through end-to-end fetch tests; this
    class pins each rewriting rule."""

    def test_ipv4_rewrite(self) -> None:
        from src.tools._http_fetch import _build_pinned_url

        assert (
            _build_pinned_url("https://example.com/path?q=1", "1.2.3.4")
            == "https://1.2.3.4/path?q=1"
        )

    def test_ipv6_brackets_in_url(self) -> None:
        from src.tools._http_fetch import _build_pinned_url

        assert (
            _build_pinned_url("https://example.com/", "2606:4700::1111")
            == "https://[2606:4700::1111]/"
        )

    def test_explicit_port_preserved(self) -> None:
        from src.tools._http_fetch import _build_pinned_url

        assert (
            _build_pinned_url("https://example.com:8443/x", "10.0.0.1") == "https://10.0.0.1:8443/x"
        )

    def test_query_preserved(self) -> None:
        from src.tools._http_fetch import _build_pinned_url

        assert (
            _build_pinned_url("https://example.com/x?a=1&b=2", "1.2.3.4")
            == "https://1.2.3.4/x?a=1&b=2"
        )

    def test_default_port_not_emitted(self) -> None:
        """When the URL has no explicit port, the pinned URL also has
        none — we don't fabricate ``:443`` or ``:80``."""
        from src.tools._http_fetch import _build_pinned_url

        assert _build_pinned_url("https://example.com/x", "1.2.3.4") == "https://1.2.3.4/x"
        assert _build_pinned_url("http://example.com/x", "1.2.3.4") == "http://1.2.3.4/x"


class TestParallelDnsDoesNotSerialise:
    """Regression guard for the blocking-DNS bug observed in
    cogtrix31.log: socket.getaddrinfo() is sync and used to block
    the event loop, serialising the fetcher's asyncio.gather. With
    asyncio.to_thread offload, six concurrent fetches whose DNS each
    takes ~0.5s must complete in well under 3s, not 3+s."""

    @pytest.mark.asyncio
    async def test_six_fetches_parallel_under_outer_deadline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import time as _time

        slow_dns_calls: list[float] = []

        def slow_getaddrinfo(*_args, **_kwargs):
            slow_dns_calls.append(_time.monotonic())
            _time.sleep(0.5)  # simulate slow DNS resolution
            return [(None, None, None, "", ("93.184.216.34", 0))]

        monkeypatch.setattr("src.tools._http_safety.socket.getaddrinfo", slow_getaddrinfo)

        # Six parallel fetches against six distinct hostnames. If DNS
        # blocks the event loop, they serialise: 6 × 0.5s = 3s minimum
        # (likely exceeds the 6s outer deadline on a real CI runner).
        # With to_thread offload they run concurrently → ~0.5s wall.
        urls = [f"https://host{i}.example.com/path" for i in range(6)]
        client = _make_client(_ok(b"ok"))
        t0 = _time.monotonic()
        results = await asyncio.gather(*(fetch_async(u, client=client) for u in urls))
        elapsed = _time.monotonic() - t0
        await client.aclose()

        assert all(r.error is None for r in results), [r.error for r in results]
        # Six DNS calls happened (six unique hosts, no caching).
        assert len(slow_dns_calls) >= 6
        # Strict parallel ceiling: 1s allows for two sequential
        # DNS-resolution waves under thread-pool contention but is
        # far below the 3s the bug would produce.
        assert elapsed < 1.5, (
            f"DNS appears to be serialising the event loop (elapsed {elapsed:.2f}s); "
            f"expected concurrent execution under asyncio.to_thread"
        )


class TestLocksSurviveLoopRecreation:
    """Regression guard for Bug C (cogtrix35.log incident).

    Module-level ``asyncio.Lock`` instances bind to the first event
    loop that touches them. The CLI / assistant paths reach
    ``web_search`` through ``asyncio.run`` (one fresh loop per
    invocation), so a module-level lock survives from the first call
    and then raises "bound to a different event loop" for every
    subsequent call — making ``fetch_async`` fail-closed forever
    after the first web_search.

    The fix moved ``_robots_lock`` to ``threading.Lock`` and made
    ``_host_locks`` loop-aware via ``_get_host_lock``. These tests
    pin that behaviour."""

    def test_fetch_async_works_across_separate_asyncio_run_calls(self) -> None:
        """Run ``fetch_async`` under two distinct ``asyncio.run`` calls
        in sequence. Pre-fix the second call raised
        "RuntimeError: <asyncio.locks.Lock ...> is bound to a different
        event loop". Post-fix both calls succeed."""
        import asyncio as _asyncio

        async def _do_fetch() -> FetchResult:
            client = _make_client(_ok(b"hello"))
            try:
                return await fetch_async("https://example.com/x", client=client)
            finally:
                await client.aclose()

        # First asyncio.run — establishes the lock binding under the
        # buggy code path.
        r1 = _asyncio.run(_do_fetch())
        # Second asyncio.run with a fresh loop — pre-fix this raised
        # the stale-loop RuntimeError; post-fix it succeeds.
        r2 = _asyncio.run(_do_fetch())

        assert r1.error is None
        assert r2.error is None
        assert r1.content == b"hello"
        assert r2.content == b"hello"

    def test_host_lock_rebuilds_when_loop_changes(self) -> None:
        """The internal helper that returns the per-host lock must
        return a *different* lock instance when called from a
        different event loop — otherwise the lock is still bound to
        the prior loop.

        We hold strong refs to both locks so CPython can't reuse the
        same address for the second allocation (which would mask the
        bug by making the id comparison pass for stale-loop locks too).
        """
        import asyncio as _asyncio

        from src.tools._http_fetch import _get_host_lock

        captured: list[_asyncio.Lock] = []

        async def _capture_lock() -> _asyncio.Lock:
            return _get_host_lock("example.com")

        # Pin both Lock objects via the list before asserting, so id()
        # can't lie via address reuse.
        captured.append(_asyncio.run(_capture_lock()))
        captured.append(_asyncio.run(_capture_lock()))

        assert captured[0] is not captured[1], (
            "Same lock object was returned across two asyncio.run loops — "
            "Bug C would re-emerge: the lock from loop 1 is still bound to "
            "loop 1 and unusable in loop 2."
        )

    @pytest.mark.asyncio
    async def test_host_lock_stable_within_same_loop(self) -> None:
        """Within a single event loop, ``_get_host_lock`` must return
        the same lock for the same host so per-host rate-limit
        serialisation actually serialises."""
        from src.tools._http_fetch import _get_host_lock

        lock_a1 = _get_host_lock("example.com")
        lock_a2 = _get_host_lock("example.com")
        lock_b = _get_host_lock("other.example.com")

        assert lock_a1 is lock_a2
        assert lock_a1 is not lock_b
