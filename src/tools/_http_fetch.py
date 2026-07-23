"""Async HTTP fetch primitive for the web_search tool (ADR-0056 stage 3).

Built on httpx.AsyncClient. Shares URL-validation / IP-block /
header-sanitisation with the sync ``http_get`` tool via
``src/tools/_http_safety.py``.

Adds the policies that the stage-3 fetcher needs but ``http_get``
intentionally does not enforce:

* robots.txt awareness (per-domain cache, 24h TTL)
* per-host concurrency limit (1 request in flight per host)
* per-host minimum spacing (250ms token bucket)
* size caps (5MB pre-decompression, 50MB post)
* same-domain redirect only, 3-hop limit
* HTTP 429 single retry honouring ``Retry-After`` (capped 2s)
* SSL errors fail fast (no insecure fallback)

SSRF safety note: this primitive validates the URL up-front via
``_http_safety._validate_url`` and DNS-pins the resolved IP into the
httpx request (IP literal URL + ``Host`` header + ``sni_hostname``
extension). This eliminates the TOCTOU window between validation and
connect that the sync ``http_get`` path addresses with urllib3
monkey-patching (see ``src/tools/http_request.py``).

Concurrency note: ``_validate_url`` is a sync function that calls
``socket.getaddrinfo`` (blocking). We offload it via
``asyncio.to_thread`` so concurrent fetches under ``asyncio.gather``
actually run in parallel — without this the event loop serialises on
DNS lookups and the fetcher's 6 s outer deadline expires before any
HTTP requests start.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import typing
import urllib.robotparser
from collections import defaultdict
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx

from src.tools._http_safety import _validate_url
from src.tools._web_search_domain_class import _EXTRACT

log = logging.getLogger("cogtrix")

# Policy constants — see ADR-0056 "Stage 3: FETCH FAN-OUT" section.
_DEFAULT_DEADLINE_S = 6.0
_MAX_REDIRECTS = 3

# Response-body size cap.  Applied to the *decompressed* stream via
# ``aiter_bytes``, which means a gzip-bomb that decompresses past this
# threshold also gets cut off here — we never materialise more than
# this many bytes in memory.  Single cap; the ADR's earlier two-tier
# (5MB pre / 50MB post) is unnecessary now that we stream.
_RESPONSE_SIZE_CAP = 5 * 1024 * 1024  # 5 MB

_PER_HOST_MIN_SPACING_S = 0.250  # 250 ms token-bucket spacing
_HTTP_429_RETRY_CAP_S = 2.0
_ROBOTS_CACHE_TTL_S = 24 * 3600  # 24 hours
_ROBOTS_FETCH_TIMEOUT_S = 3.0

# Slow-loris guard inside ``_stream_with_cap`` (forge audit H7, 2026-05-23).
# ``_SLOW_LORIS_PROBATION_S`` gives the server a chance to send a meaningful
# burst before we measure throughput — useful for sites that buffer the
# whole HTML before flushing. ``_SLOW_LORIS_MIN_BPS`` is the floor below
# which we treat the stream as adversarial (a healthy server on a modern
# link clears 100 KB/s easily; a hostile server holding our slot trickles
# bytes well below 1 KB/s).
_SLOW_LORIS_PROBATION_S = 2.0
_SLOW_LORIS_MIN_BPS = 1024.0

# Browser fingerprint. We present as a recent stable Chrome on Windows so
# that anti-bot heuristics (User-Agent + Client-Hints + Sec-Fetch-*) treat
# us as a normal navigation, not a scraper. Major-version-pinning matches
# what Chrome itself does post-2022 (UA-reduction): the minor.build.patch
# fields are frozen to 0.0.0 in the User-Agent string.
_CHROME_MAJOR = "138"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    f"Chrome/{_CHROME_MAJOR}.0.0.0 Safari/537.36"
)

# Top-level-navigation header set sent on every fetch. Order matches
# Chrome's actual emission order to minimise fingerprint divergence;
# httpx preserves insertion order on the wire. ``Accept-Encoding`` is
# left unset so httpx advertises only the codecs it can actually decode
# (gzip+deflate, plus zstd because zstandard is installed; brotli is
# not bundled). Claiming ``br`` here would risk an undecodable response.
BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": USER_AGENT,
    "sec-ch-ua": (
        f'"Not.A/Brand";v="99", '
        f'"Google Chrome";v="{_CHROME_MAJOR}", '
        f'"Chromium";v="{_CHROME_MAJOR}"'
    ),
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "Upgrade-Insecure-Requests": "1",
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8,"
        "application/signed-exchange;v=b3;q=0.7"
    ),
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-User": "?1",
    "Sec-Fetch-Dest": "document",
    "Accept-Language": "en-US,en;q=0.9",
    "Priority": "u=0, i",
}


@dataclass(frozen=True)
class FetchResult:
    """Outcome of a single ``fetch_async`` call.

    ``error`` is ``None`` on success. On failure, ``error`` is a short
    machine-readable category ("blocked-robots", "rate-limited",
    "ssl-error", "timeout", "cross-domain-redirect", "size-cap",
    "validation-failed", "http-status") and ``status_code`` may be
    populated where the failure was an HTTP-level rejection.
    """

    url: str
    status_code: int | None
    content: bytes | None
    encoding: str | None
    content_type: str | None
    elapsed_ms: int
    truncated: bool
    error: str | None
    error_detail: str | None = None


# ── Robots.txt cache ──────────────────────────────────────────────────

_robots_cache: dict[str, tuple[float, urllib.robotparser.RobotFileParser | None]] = {}
# ``threading.Lock`` (not ``asyncio.Lock``) is deliberate: the two
# critical sections that hold this lock are pure dict get/set with no
# ``await`` inside, so there's no benefit to async lock semantics.
# More importantly, ``asyncio.Lock`` instantiated at module-load time
# binds to whichever event loop touches it first; the CLI / assistant
# paths reach ``web_search`` through ``asyncio.run`` (one loop per
# call), so a module-level ``asyncio.Lock`` survives the loop that
# created it and then raises "bound to a different event loop" for
# every subsequent call (Bug C from the cogtrix35.log incident).
_robots_lock = threading.Lock()


async def _is_allowed_by_robots(
    client: httpx.AsyncClient, url: str, *, resolved_ip: str | None = None
) -> bool:
    """Return True if our User-Agent is allowed to fetch *url*.

    The robots.txt URL shares its host with *url* so the same
    ``resolved_ip`` from the caller's ``_validate_url`` applies — the
    robots probe is DNS-pinned the same way the main fetch is (issue
    #1668 follow-up). Without this, an attacker could still rebind
    DNS for the robots round-trip to probe internal endpoints.

    Fail-open: if robots.txt fetch errors (DNS failure, timeout, 5xx),
    we treat the URL as allowed. Polite-bot principle says respect
    explicit denials but don't block on infrastructure flakiness.
    """
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return True
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    cache_key = parsed.netloc.lower()
    now = time.monotonic()

    with _robots_lock:
        entry = _robots_cache.get(cache_key)
        if entry is not None:
            cached_at, parser = entry
            if now - cached_at < _ROBOTS_CACHE_TTL_S:
                if parser is None:  # cached fetch-failure → fail-open
                    return True
                return parser.can_fetch(USER_AGENT, url)

    # Cache miss or stale — fetch + parse outside the lock to avoid
    # serialising every fetch behind one slow robots.txt.
    parser: urllib.robotparser.RobotFileParser | None = None
    try:
        original_host = parsed.hostname or ""
        if resolved_ip:
            request_url = _build_pinned_url(robots_url, resolved_ip)
            headers: dict[str, str] = {**BROWSER_HEADERS, "Host": original_host}
            extensions: dict[str, typing.Any] = {"sni_hostname": original_host}
        else:
            request_url = robots_url
            headers = dict(BROWSER_HEADERS)
            extensions = {}
        response = await client.get(
            request_url,
            timeout=_ROBOTS_FETCH_TIMEOUT_S,
            headers=headers,
            extensions=extensions,
        )
        if response.status_code == 200:
            parser = urllib.robotparser.RobotFileParser()
            parser.parse(response.text.splitlines())
        # Any other status → fail-open (parser stays None)
    except (TimeoutError, httpx.HTTPError) as exc:
        log.debug("robots.txt fetch failed for %s: %s — failing open", robots_url, exc)
        parser = None

    with _robots_lock:
        _robots_cache[cache_key] = (now, parser)

    if parser is None:
        return True
    return parser.can_fetch(USER_AGENT, url)


def _clear_robots_cache() -> None:
    """Test helper — drops every cached robots entry."""
    _robots_cache.clear()


# ── Per-host rate limiting ────────────────────────────────────────────

# ``_host_last_request`` is a plain dict — only read/written from
# within ``_acquire_host_slot`` while the per-host asyncio.Lock is
# held, so no extra synchronisation needed.
_host_last_request: dict[str, float] = defaultdict(float)

# ``_host_locks`` MUST be loop-aware: ``asyncio.Lock`` instances
# bind to the first loop that touches them. The CLI's
# ``asyncio.run`` per-call pattern means each web_search call gets a
# fresh loop; a module-level ``defaultdict(asyncio.Lock)`` survives
# from the first loop and then raises "bound to a different event
# loop" for every subsequent call (Bug C from cogtrix35.log).
#
# Fix: keep the per-host dict but tagged with the loop id that owns
# it. When the running loop changes (new CLI call), rebuild. The
# meta-lock is sync because the swap is a brief dict assignment.
_host_locks: dict[str, asyncio.Lock] = {}
# Hold the loop *object* (not its id) so that a freed-and-reallocated
# loop at the same address can't masquerade as "still the same loop"
# (CPython's allocator readily reuses ids after garbage collection).
# A strong reference is fine: one dangling reference to a closed loop
# is negligible memory and gets replaced on the next asyncio.run.
_host_locks_loop: asyncio.AbstractEventLoop | None = None
_host_locks_meta_lock = threading.Lock()


def _get_host_lock(host: str) -> asyncio.Lock:
    """Return the per-host asyncio.Lock bound to the *current* loop.

    Rebuilds the host-lock dict when the running loop changes so a
    stale lock from a previous ``asyncio.run`` invocation can never
    leak into the new loop. Callers must already be inside a running
    event loop.
    """
    global _host_locks, _host_locks_loop
    loop = asyncio.get_running_loop()
    with _host_locks_meta_lock:
        if _host_locks_loop is not loop:
            _host_locks = {}
            _host_last_request.clear()
            _host_locks_loop = loop
        lock = _host_locks.get(host)
        if lock is None:
            lock = asyncio.Lock()
            _host_locks[host] = lock
        return lock


async def _acquire_host_slot(host: str, deadline_at: float) -> asyncio.Lock | None:
    """Wait for the per-host rate-limit slot.

    Returns the acquired lock on success (caller must pass it back to
    ``_release_host_slot``), or ``None`` if the deadline expired before
    the slot became available.

    Two ownership pitfalls the previous implementation had
    (forge audit H7 + B4, 2026-05-23):

    1. **Acquire-cancel race in ``asyncio.wait_for(lock.acquire(), ...)``**:
       ``lock.acquire()`` can complete at the same instant ``wait_for``
       fires its timeout cancellation. The caller catches ``TimeoutError``
       and returns ``None``, but the lock is *held* — subsequent fetches
       to that host wait forever. Fix: ``asyncio.timeout()`` context
       manager (Python 3.11+) handles the race more cleanly, AND we
       defensively check + release if the lock somehow ended up held.
    2. **BaseException handler released someone-else's lock**:
       ``except BaseException: if lock.locked(): release()`` is wrong
       because ``asyncio.Lock`` doesn't track ownership — ``locked()``
       only tells us *some* coroutine holds it, not that *we* do. Under
       same-host concurrency this could release another fetch's hold.
       Fix: explicit ``we_own_lock`` flag, only released by us if True.
    """
    lock = _get_host_lock(host)

    # Acquire via asyncio.timeout() context manager. Python 3.11+ guarantees
    # cleaner CancelledError handling on the inner await, reducing the
    # chance of the acquire-cancel race. The defensive cleanup below covers
    # the residual window.
    we_own_lock = False
    try:
        async with asyncio.timeout(max(0.01, deadline_at - time.monotonic())):
            await lock.acquire()
            we_own_lock = True
    except TimeoutError:
        # If somehow the acquire completed before the cancellation took
        # effect, release the orphaned hold — leaving it would deadlock
        # every future fetch to this host.
        if we_own_lock:
            try:
                lock.release()
            except RuntimeError:
                # Lock wasn't actually held by us; ``we_own_lock`` was True
                # but ``lock.acquire()`` raised after setting it. Safe to
                # ignore.
                pass
        return None

    # From here on, WE own the lock. The ``we_own_lock`` flag tracks that
    # so the exception handler below only releases what we acquired —
    # critical under same-host concurrency.
    try:
        last = _host_last_request[host]
        elapsed = time.monotonic() - last
        if elapsed < _PER_HOST_MIN_SPACING_S:
            wait = _PER_HOST_MIN_SPACING_S - elapsed
            remaining = deadline_at - time.monotonic()
            if wait >= remaining:
                lock.release()
                we_own_lock = False
                return None
            await asyncio.sleep(wait)
        _host_last_request[host] = time.monotonic()
        # Caller now takes ownership via the returned lock object — we no
        # longer release it on exception below.
        we_own_lock = False
        return lock
    except BaseException:
        if we_own_lock:
            try:
                lock.release()
            except RuntimeError:
                pass
        raise


def _release_host_slot(lock: asyncio.Lock | None) -> None:
    """Release a host-slot lock acquired via ``_acquire_host_slot``.

    No-op when ``lock`` is ``None`` (acquire timed out) or already
    released. The caller MUST pass back the exact lock object returned
    by ``_acquire_host_slot`` so we release the right hold under
    same-host concurrency.
    """
    if lock is not None and lock.locked():
        lock.release()


def _reset_host_state() -> None:
    """Test helper — drops every per-host spacing record."""
    global _host_locks_loop
    _host_last_request.clear()
    _host_locks.clear()
    _host_locks_loop = None


# ── Public entry point ────────────────────────────────────────────────


async def fetch_async(
    url: str,
    *,
    deadline_s: float = _DEFAULT_DEADLINE_S,
    client: httpx.AsyncClient | None = None,
    extra_headers: dict[str, str] | None = None,
) -> FetchResult:
    """Fetch a single URL with all stage-3 policies enforced.

    Parameters
    ----------
    url
        The URL to fetch. Must pass SSRF validation up front.
    deadline_s
        Hard wall-clock budget for the entire fetch (including robots.txt
        lookup, rate-limit wait, retries). Default 6s per ADR-0056.
    client
        Optional pre-configured ``httpx.AsyncClient``. Tests pass a
        mocked client; production code passes a real one from the
        fetcher's shared pool.
    extra_headers
        Headers merged on top of the default User-Agent. ``Host`` and
        ``X-Forwarded-*`` are stripped via _http_safety on the call
        path that uses ``_parse_headers``; this entry takes a dict so
        callers must sanitise themselves.
    """
    started_at = time.monotonic()
    deadline_at = started_at + deadline_s

    def _now_ms() -> int:
        return int((time.monotonic() - started_at) * 1000)

    # _validate_url is sync and does a blocking socket.getaddrinfo() —
    # offload to a thread so concurrent fetches (asyncio.gather across
    # K URLs) don't serialise on the event loop. Without this, each
    # validation blocks the loop for the duration of its DNS lookup,
    # which exhausts the fetcher's 6s outer deadline before the
    # actual HTTP requests start.
    is_valid, error_msg, resolved_ip = await asyncio.to_thread(_validate_url, url)
    if not is_valid:
        return FetchResult(
            url=url,
            status_code=None,
            content=None,
            encoding=None,
            content_type=None,
            elapsed_ms=_now_ms(),
            truncated=False,
            error="validation-failed",
            error_detail=error_msg,
        )

    owns_client = client is None
    if client is None:
        # HTTP/1.1 only — the h2 extra adds a dep without a material
        # latency win for stage-3 fetches.
        client = httpx.AsyncClient(
            follow_redirects=False,
            headers=BROWSER_HEADERS,
        )

    try:
        try:
            allowed = await _is_allowed_by_robots(client, url, resolved_ip=resolved_ip)
        except Exception as exc:  # noqa: BLE001
            log.debug("robots check threw %s; failing open", exc)
            allowed = True
        if not allowed:
            return FetchResult(
                url=url,
                status_code=None,
                content=None,
                encoding=None,
                content_type=None,
                elapsed_ms=_now_ms(),
                truncated=False,
                error="blocked-robots",
                error_detail=None,
            )

        return await _fetch_with_redirects(
            client=client,
            url=url,
            resolved_ip=resolved_ip,
            started_at=started_at,
            deadline_at=deadline_at,
            extra_headers=extra_headers or {},
        )
    finally:
        if owns_client:
            await client.aclose()


async def _fetch_with_redirects(
    *,
    client: httpx.AsyncClient,
    url: str,
    resolved_ip: str | None,
    started_at: float,
    deadline_at: float,
    extra_headers: dict[str, str],
) -> FetchResult:
    """Manual redirect loop. Same-domain only, max 3 hops, HTTP 429 retry."""
    original_registered_domain = _registered_domain(url)
    current_url = url
    hops = 0
    retry_429_used = False

    while True:
        if time.monotonic() >= deadline_at:
            return FetchResult(
                url=current_url,
                status_code=None,
                content=None,
                encoding=None,
                content_type=None,
                elapsed_ms=int((time.monotonic() - started_at) * 1000),
                truncated=False,
                error="timeout",
                error_detail="deadline exceeded",
            )

        host = urlparse(current_url).netloc.lower()
        host_slot = await _acquire_host_slot(host, deadline_at)
        if host_slot is None:
            return FetchResult(
                url=current_url,
                status_code=None,
                content=None,
                encoding=None,
                content_type=None,
                elapsed_ms=int((time.monotonic() - started_at) * 1000),
                truncated=False,
                error="rate-limited",
                error_detail="per-host slot timed out",
            )

        try:
            remaining = max(0.01, deadline_at - time.monotonic())
            try:
                response = await _stream_with_cap(
                    client=client,
                    url=current_url,
                    resolved_ip=resolved_ip,
                    timeout=remaining,
                    extra_headers=extra_headers,
                )
            except httpx.ConnectError as exc:
                if "ssl" in str(exc).lower() or "certificate" in str(exc).lower():
                    return _error_result(current_url, started_at, "ssl-error", str(exc))
                return _error_result(current_url, started_at, "connect-error", str(exc))
            except (TimeoutError, httpx.TimeoutException) as exc:
                return _error_result(current_url, started_at, "timeout", str(exc))
            except httpx.HTTPError as exc:
                return _error_result(current_url, started_at, "http-error", str(exc))
        finally:
            # Pass the lock object back so we release the right hold under
            # same-host concurrency (forge audit H7, 2026-05-23).
            _release_host_slot(host_slot)

        # HTTP 429 — single retry honouring Retry-After (capped 2s).
        if response.status_code == 429 and not retry_429_used:
            retry_429_used = True
            retry_after = response.headers.get("Retry-After", "")
            try:
                wait_s = float(retry_after) if retry_after else 1.0
            except ValueError:
                wait_s = 1.0
            wait_s = min(wait_s, _HTTP_429_RETRY_CAP_S)
            if time.monotonic() + wait_s >= deadline_at:
                return _error_result(
                    current_url, started_at, "rate-limited", "Retry-After exceeds budget"
                )
            await asyncio.sleep(wait_s)
            continue

        # Redirect handling. httpx with follow_redirects=False returns
        # the 3xx response with the Location header intact.
        if response.status_code in (301, 302, 303, 307, 308):
            location = response.headers.get("Location", "")
            if not location:
                return _consume_response(response, current_url, started_at)
            redirect_url = urljoin(current_url, location)
            hops += 1
            if hops > _MAX_REDIRECTS:
                return _error_result(
                    redirect_url, started_at, "too-many-redirects", f">{_MAX_REDIRECTS} hops"
                )
            if _registered_domain(redirect_url) != original_registered_domain:
                return _error_result(
                    redirect_url,
                    started_at,
                    "cross-domain-redirect",
                    f"refused redirect from {original_registered_domain}",
                )
            # Re-validate target before next iteration. to_thread again —
            # the redirect target's getaddrinfo also blocks.
            is_valid, error_msg, redirect_ip = await asyncio.to_thread(_validate_url, redirect_url)
            if not is_valid:
                return _error_result(redirect_url, started_at, "validation-failed", error_msg)
            current_url = redirect_url
            resolved_ip = redirect_ip
            continue

        return _consume_response(response, current_url, started_at)


@dataclass
class _StreamedResponse:
    """Internal representation of a streamed-and-size-capped HTTP response."""

    status_code: int
    headers: httpx.Headers
    content: bytes
    encoding: str | None
    truncated: bool


def _build_pinned_url(url: str, resolved_ip: str) -> str:
    """Rewrite *url* so the hostname is replaced by *resolved_ip*.

    Preserves scheme, port, path, query and fragment. IPv6 literals are
    wrapped in brackets so they form a valid netloc.
    """
    parsed = urlparse(url)
    port_part = f":{parsed.port}" if parsed.port else ""
    if ":" in resolved_ip:
        netloc = f"[{resolved_ip}]{port_part}"
    else:
        netloc = f"{resolved_ip}{port_part}"
    return parsed._replace(netloc=netloc).geturl()


async def _stream_with_cap(
    *,
    client: httpx.AsyncClient,
    url: str,
    resolved_ip: str | None,
    timeout: float,
    extra_headers: dict[str, str],
) -> _StreamedResponse:
    """Stream a GET response, capping the body at ``_RESPONSE_SIZE_CAP``.

    When *resolved_ip* is provided the connection is DNS-pinned: the
    TCP layer connects to the pre-validated IP while the HTTP ``Host``
    header (and TLS SNI, for HTTPS) keep the original hostname so
    certificate validation and virtual-host routing still work.

    httpx auto-decompresses gzip / brotli; ``aiter_bytes`` yields the
    decompressed stream. We break as soon as we've accumulated more
    than the cap, which means a gzip-bomb that would decompress past
    5 MB never makes it past the cap. ``Content-Length`` is not used
    as a pre-check because servers can lie / chunked encoding lacks it.
    """
    original_host = urlparse(url).hostname or ""
    if resolved_ip:
        request_url = _build_pinned_url(url, resolved_ip)
        headers = {**extra_headers, "Host": original_host}
        extensions: dict[str, typing.Any] = {"sni_hostname": original_host}
    else:
        request_url = url
        headers = extra_headers
        extensions = {}

    async with client.stream(
        "GET", request_url, timeout=timeout, headers=headers, extensions=extensions
    ) as resp:
        chunks: list[bytes] = []
        total = 0
        truncated = False
        # Slow-loris guard (forge audit H7, 2026-05-23). A server that
        # streams a few bytes per second can hold our per-host slot for
        # the entire fetch deadline (6 s) without ever finishing, which
        # an attacker who controls one of the top-K result domains can
        # weaponise to degrade the whole fan-out. After a short
        # probationary window (``_SLOW_LORIS_PROBATION_S``) we abort if
        # the byte-rate is under ``_SLOW_LORIS_MIN_BPS`` — fast enough
        # for healthy modern servers, slow enough that a real shaky
        # connection still gets a chance.
        stream_started_at = time.monotonic()
        try:
            async for chunk in resp.aiter_bytes(chunk_size=8192):
                if not chunk:
                    continue
                chunks.append(chunk)
                total += len(chunk)
                if total > _RESPONSE_SIZE_CAP:
                    truncated = True
                    break
                elapsed = time.monotonic() - stream_started_at
                if elapsed >= _SLOW_LORIS_PROBATION_S and (total / elapsed) < _SLOW_LORIS_MIN_BPS:
                    raise httpx.ReadTimeout(
                        f"slow-loris: {total} bytes in {elapsed:.1f}s "
                        f"(< {_SLOW_LORIS_MIN_BPS} B/s)"
                    )
        except httpx.HTTPError:
            raise
        body = b"".join(chunks)[:_RESPONSE_SIZE_CAP]
        return _StreamedResponse(
            status_code=resp.status_code,
            headers=resp.headers,
            content=body,
            encoding=resp.encoding,
            truncated=truncated,
        )


def _consume_response(response: _StreamedResponse, url: str, started_at: float) -> FetchResult:
    """Build a FetchResult from an already-streamed response."""
    error: str | None = None
    error_detail: str | None = None
    if response.status_code >= 400:
        error = "http-status"
        error_detail = f"HTTP {response.status_code}"

    return FetchResult(
        url=url,
        status_code=response.status_code,
        content=response.content,
        encoding=response.encoding,
        content_type=response.headers.get("Content-Type"),
        elapsed_ms=int((time.monotonic() - started_at) * 1000),
        truncated=response.truncated,
        error=error,
        error_detail=error_detail,
    )


def _error_result(url: str, started_at: float, category: str, detail: str) -> FetchResult:
    return FetchResult(
        url=url,
        status_code=None,
        content=None,
        encoding=None,
        content_type=None,
        elapsed_ms=int((time.monotonic() - started_at) * 1000),
        truncated=False,
        error=category,
        error_detail=detail,
    )


def _registered_domain(url: str) -> str:
    """Same-origin key for the same-domain redirect check.

    For normal hosts this is the registered domain (so ``example.com`` and
    ``www.example.com`` are treated as same-site). Suffix-less or IP-literal
    hosts have no registered domain, so fall back to the full ``host:port``
    authority — a port change on an IP/bare host is then treated as a
    cross-origin redirect (fail closed) instead of collapsing distinct
    targets to the bare host (#2136 F6).
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").casefold()
    extracted = _EXTRACT(host)
    if not extracted.domain or not extracted.suffix:
        try:
            port = parsed.port
        except ValueError:
            port = None
        return f"{host}:{port}" if port is not None else host
    return f"{extracted.domain}.{extracted.suffix}"


__all__ = [
    "BROWSER_HEADERS",
    "FetchResult",
    "USER_AGENT",
    "fetch_async",
]
