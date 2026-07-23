"""Shared rate-limit configuration for the Cogtrix API."""

from __future__ import annotations

import ipaddress
import logging
import re
import threading
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

from fastapi import HTTPException, Request, status
from limits import RateLimitItemPerSecond
from limits.storage import MemoryStorage, Storage, storage_from_string
from limits.strategies import MovingWindowRateLimiter
from slowapi import Limiter
from slowapi.util import get_remote_address

log = logging.getLogger("cogtrix.api.rate_limit")

_lock = threading.Lock()
_trusted_proxy_networks: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = ()

# ── Per-route sliding-window backend (#1879 Slice B) ─────────────────
#
# The per-route limiter delegates to the ``limits`` library's
# ``MovingWindowRateLimiter`` over a pluggable :class:`Storage` backend.
# The default is :class:`MemoryStorage` (per-process, correct for
# single-node deployments). Calling :func:`configure_rate_limit_backend`
# with a ``redis://...`` URL at app startup swaps the storage for
# :class:`limits.storage.redis.RedisStorage` so a single sliding window
# is shared across replicas.
#
# Legacy compat: the ``_hit_counters`` dict + ``_evict_stale_counters``
# helper that the pre-Slice-B code used are kept as no-op stubs to
# preserve the public surface for any external tests that imported them.
# They no longer participate in enforcement.
_storage: Storage = MemoryStorage()
_strategy: MovingWindowRateLimiter = MovingWindowRateLimiter(_storage)
_backend_label: str = "memory://"
_backend_lock = threading.Lock()

# Legacy in-memory counter dict (retained as a stub for back-compat —
# enforcement now goes through ``_strategy``). External tests that
# imported ``_hit_counters`` continue to work; the dict is simply
# never read by the live path.
_hit_counters: dict[tuple[str, str], list[datetime]] = defaultdict(list)
_counters_lock = threading.Lock()
# Set to True in test fixtures to bypass per-route limits without affecting
# global SlowAPI middleware. Never set this in production code.
_per_route_disabled: bool = False
# Retained for symbolic back-compat — the new ``limits`` backend handles
# its own bucket eviction (memory) or has unbounded server-side keys
# (redis). The constants and ``_last_counters_cleanup`` are no longer
# read by the enforcement path.
_last_counters_cleanup: datetime | None = None
_COUNTERS_CLEANUP_INTERVAL_SEC = 300
_COUNTERS_MAX_AGE_SEC = 3600


def configure_rate_limit_backend(*, redis_url: str | None) -> None:
    """Install the rate-limit storage backend (#1879 Slice B + follow-up).

    Installs storage for BOTH the per-route limiter and the SlowAPI
    global limiter — when ``redis_url`` is set, both paths share a
    single counter across replicas; when unset, both use independent
    per-process in-memory backends.

    Called at app startup by ``cogtrix_core/api/app.py`` after settings load.
    Idempotent — safe to call multiple times. Each call:

    1. Atomically replaces the ``MovingWindowRateLimiter`` strategy
       used by per-route ``_enforce_per_route`` (under
       ``_backend_lock``).
    2. Rebuilds the module-level SlowAPI :class:`Limiter` with the
       configured ``storage_uri``. **The caller must then reassign
       ``app.state.limiter`` to the rebuilt module-level instance** —
       ``SlowAPIMiddleware.dispatch`` reads ``app.state.limiter`` on
       every request, so the swap takes effect on the next request.

    Args:
        redis_url: A ``redis://``, ``rediss://``, or
            ``redis+sentinel://`` URL to share the counter across
            replicas. Pass ``None`` (or an empty string) to use the
            default per-process :class:`MemoryStorage`. Requires the
            ``cogtrix[redis]`` install extra when a URL is supplied;
            ``import redis`` happens lazily inside ``limits`` so the
            ``redis`` package is only needed when actually opting in.
    """
    global _storage, _strategy, _backend_label, limiter
    url = (redis_url or "").strip()
    if url:
        new_storage = storage_from_string(url)
        new_label = _redact_redis_url(url)
    else:
        new_storage = MemoryStorage()
        new_label = "memory://"
    with _backend_lock:
        _storage = new_storage
        _strategy = MovingWindowRateLimiter(_storage)
        _backend_label = new_label
        # Rebuild the SlowAPI global limiter so its blunt guard
        # (120/minute default) also benefits from the shared backend
        # under horizontal scaling. The default_limits string survives
        # the rebuild; ``COGTRIX_RATE_LIMIT_DEFAULT`` env-var operators
        # can already tune the per-route default via Slice A, but the
        # SlowAPI global keeps its module-load default ("120/minute")
        # here so this follow-up doesn't quietly retune the blunt
        # guard. Future operators who want to drive the global
        # default from config can opt in via a separate explicit knob.
        limiter_kwargs: dict[str, Any] = {
            "key_func": rate_limit_key,
            "default_limits": ["120/minute"],
        }
        if url:
            limiter_kwargs["storage_uri"] = url
        limiter = Limiter(**limiter_kwargs)
    log.info("Rate-limit backend installed: %s", new_label)


def _redact_redis_url(url: str) -> str:
    """Return *url* with any inline password masked for safe logging.

    ``redis://user:secret@host:6379/0`` → ``redis://user:***@host:6379/0``
    ``redis://:secret@host:6379`` → ``redis://:***@host:6379``
    Falls back to the scheme + host when the URL cannot be parsed.
    """
    try:
        from urllib.parse import urlparse, urlunparse

        parsed = urlparse(url)
        if parsed.password is None:
            return url
        # ``netloc`` rebuild with masked password.
        userinfo = parsed.username or ""
        netloc = f"{userinfo}:***@{parsed.hostname or ''}" + (
            f":{parsed.port}" if parsed.port else ""
        )
        return urlunparse(parsed._replace(netloc=netloc))
    except Exception:
        # Belt-and-braces — never crash the startup log.
        return url.split("@", 1)[-1] if "@" in url else url


def current_backend_label() -> str:
    """Return a human-readable label for the active rate-limit backend.

    Used by startup banners and ``/api/v1/health`` introspection. Always
    safe to log — :func:`_redact_redis_url` masks any inline password.
    """
    with _backend_lock:
        return _backend_label


def _parse_trusted_proxy_cidrs(
    raw: str | Iterable[str] | None,
) -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    if raw is None:
        return ()

    if isinstance(raw, str):
        values = [value.strip() for value in raw.split(",")]
    else:
        values = [str(value).strip() for value in raw]

    networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    for value in values:
        if not value:
            continue
        try:
            networks.append(ipaddress.ip_network(value, strict=False))
        except ValueError as exc:
            raise ValueError(f"Invalid trusted proxy CIDR: {value}") from exc
    return tuple(networks)


def configure_trusted_proxy_cidrs(raw: str | Iterable[str] | None) -> None:
    """Configure the CIDR allowlist for trusted reverse proxies."""
    networks = _parse_trusted_proxy_cidrs(raw)
    with _lock:
        global _trusted_proxy_networks
        _trusted_proxy_networks = networks

    if networks:
        log.info("Trusted proxy allowlist configured with %d network(s)", len(networks))
    else:
        log.info("Trusted proxy allowlist disabled")


def _is_trusted_proxy(remote_addr: str, networks: tuple[Any, ...]) -> bool:
    try:
        address = ipaddress.ip_address(remote_addr)
    except ValueError:
        return False
    return any(address in network for network in networks)


def _is_valid_ip(value: str) -> bool:
    """Return True iff ``value`` parses as a valid IPv4 or IPv6 address."""
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return False
    return True


def _client_key(request: Request) -> str:
    """Return the client IP key for rate limiting, honouring trusted proxy headers.

    Walks the ``X-Forwarded-For`` chain RIGHT-TO-LEFT starting at the socket
    peer (forge audit C6, 2026-05-23). The previous implementation took
    ``forwarded_for.split(",", 1)[0]`` — the LEFTMOST entry — which is
    client-supplied and trivially spoofable. With trusted proxies
    configured, an attacker sending ``X-Forwarded-For: 1.2.3.4, <real>``
    would get rate-limited as ``1.2.3.4`` (a value they pick fresh on every
    request), defeating per-IP limits entirely.

    Correct algorithm (matches nginx ``real_ip_from`` + ``real_ip_header``):

    1. Effective chain from us back to the originator is
       ``[socket_peer, *reversed(XFF)]``.
    2. Walk that chain left-to-right (= right-to-left from XFF's
       perspective); skip each entry that's a trusted proxy.
    3. Return the first untrusted hop — that's the real client.
    4. If every hop is trusted (chain is exhausted), fall back to the
       leftmost XFF entry (best-effort) or the socket peer.
    """
    remote_addr = get_remote_address(request)

    with _lock:
        networks = _trusted_proxy_networks

    if not networks:
        return remote_addr

    forwarded_for = request.headers.get("x-forwarded-for", "")
    xff_entries = [e.strip() for e in forwarded_for.split(",") if e.strip()]

    # The TCP peer (``remote_addr``) is the last hop into us — i.e. the
    # rightmost entry of the full client→...→us chain. XFF lists the hops
    # BEFORE the socket peer, in original order. Reverse XFF and prepend the
    # socket peer to get the chain ordered from "closest to us" outward.
    chain: list[str] = [remote_addr, *reversed(xff_entries)]

    for hop in chain:
        # Reject malformed values outright (forge audit B6, second-order
        # to C6). ``_is_trusted_proxy`` returns False for un-parseable
        # strings, which previously meant garbage XFF entries like
        # ``"junk-AAA, junk-BBB, ..."`` were each accepted as a fresh
        # rate-limit bucket — letting an attacker pick unlimited unique
        # keys per request and defeat per-IP limits entirely.
        if not _is_valid_ip(hop):
            continue
        if not _is_trusted_proxy(hop, networks):
            return hop

    # Every hop was either trusted or malformed — chain exhausted without
    # finding a real client. Fall back to the leftmost VALID XFF entry if
    # one exists (the client's own claim — untrusted but our best signal),
    # else the socket peer. ``remote_addr`` from ``get_remote_address`` is
    # the parsed socket peer and is itself a valid IP (or ``"unknown"`` —
    # treated as a fixed single bucket, which is the safe failure mode).
    for entry in xff_entries:
        if _is_valid_ip(entry):
            return entry
    return remote_addr


# Global SlowAPI middleware default (120 req/min blunt guard at module
# load — rebuilt at startup by :func:`configure_rate_limit_backend`).
#
# Why this works even though SlowAPIMiddleware was registered before
# startup (#1879 follow-up, correcting the rationale documented in
# PR #1881): the middleware's ``dispatch`` reads ``app.state.limiter``
# **per request**, not at construction time. So if startup rebuilds
# this module-level ``limiter`` with a Redis storage_uri AND reassigns
# ``app.state.limiter`` to the rebuilt instance, the middleware uses
# the new limiter on the next request. No middleware recreation
# required.
#
# Note: the SlowAPI ``Limiter.__init__`` calls ``storage_from_string``
# immediately (slowapi.Limiter:154 in the installed version), so the
# storage is bound at construction. That's why ``configure_rate_limit_backend``
# REBUILDS the ``Limiter`` rather than mutating ``_storage`` after the
# fact — the internal ``_limiter`` strategy caches its storage reference
# and won't notice a post-hoc attribute swap.
rate_limit_key = _client_key
limiter = Limiter(key_func=rate_limit_key, default_limits=["120/minute"])


# ── Config-driven per-route limits (#1879 Slice A) ────────────────────
#
# Populated at app startup by :func:`configure_rate_limits` from
# ``Config.api.rate_limits`` (with ``COGTRIX_RATE_LIMIT_<NAME>`` env-var
# overrides). Routes wired with :func:`per_route_rate_limit_for` look up
# their (max_calls, window_seconds) at request time, so live config
# reloads (e.g. ``configure_rate_limits`` called again during tests)
# take effect immediately on the next request without rebuilding any
# route table.
#
# A route name that is not present in ``_route_limits`` falls back to
# ``_default_limit_spec`` (the ``api.rate_limits["default"]`` value).
_RATE_LIMIT_SPEC_RE = re.compile(
    r"^\s*(\d+)\s*/\s*(second|minute|hour|day|s|m|h|d)s?\s*$",
    re.IGNORECASE,
)
_RATE_LIMIT_UNIT_SECONDS: dict[str, int] = {
    "s": 1,
    "second": 1,
    "m": 60,
    "minute": 60,
    "h": 3600,
    "hour": 3600,
    "d": 86400,
    "day": 86400,
}
_route_limits: dict[str, tuple[int, int]] = {}
_default_limit_spec: str = "120/minute"
_route_limits_lock = threading.Lock()


def parse_rate_limit_spec(spec: str) -> tuple[int, int]:
    """Parse a SlowAPI-style ``"<N>/<window>"`` spec to ``(max_calls, window_seconds)``.

    Windows accepted (case-insensitive, optional trailing ``s``):

    * ``second`` / ``s``
    * ``minute`` / ``m``
    * ``hour``   / ``h``
    * ``day``    / ``d``

    Raises:
        ValueError: when *spec* does not parse.
    """
    match = _RATE_LIMIT_SPEC_RE.match(spec)
    if match is None:
        raise ValueError(
            f"invalid rate-limit spec {spec!r}; expected '<N>/<window>' where "
            f"window is one of second/minute/hour/day (or s/m/h/d)"
        )
    count = int(match.group(1))
    unit_seconds = _RATE_LIMIT_UNIT_SECONDS[match.group(2).lower()]
    return (count, unit_seconds)


def configure_rate_limits(*, default: str, per_route: Mapping[str, str]) -> None:
    """Install the per-route rate-limit table consulted by
    :func:`per_route_rate_limit_for`.

    Args:
        default: Fallback ``"<N>/<window>"`` spec applied when a route name
            is not present in *per_route*. Also used to reset the SlowAPI
            middleware's ``default_limits`` so the global blunt guard
            tracks the configured value.
        per_route: Mapping of route name → spec string. Each value is
            validated by :func:`parse_rate_limit_spec`; any invalid entry
            raises ``ValueError`` and the existing table is left
            unchanged so partial application can't silently corrupt
            production limits.
    """
    parse_rate_limit_spec(default)  # validate eagerly
    parsed: dict[str, tuple[int, int]] = {
        name: parse_rate_limit_spec(spec) for name, spec in per_route.items()
    }
    with _route_limits_lock:
        global _route_limits, _default_limit_spec
        _route_limits = parsed
        _default_limit_spec = default
    # NOTE: We intentionally do NOT mutate ``limiter._default_limits``
    # to track the configured ``default``. SlowAPI's middleware path
    # constructs limit objects from the value at startup and feeds them
    # through ``sync_check_limits``; replacing the list mid-flight broke
    # the middleware in earlier iterations (TypeError surfacing from
    # the slowapi exception handler). The blunt SlowAPI default
    # (``"120/minute"`` at module import) stays in place as a fixed
    # global guard; the configured ``default`` is consulted by
    # :func:`per_route_rate_limit_for` for the per-route fallback path
    # only. Making the global guard config-driven requires deeper
    # surgery (e.g. Redis-backed Limiter from #1879 Slice B) and is out
    # of scope for Slice A.


def reset_rate_limits() -> None:
    """Clear all per-route sliding-window counters.  Called at app startup.

    For :class:`MemoryStorage` this empties the in-process buckets so a
    fresh process starts at zero. For Redis (or any other shared
    backend) we deliberately do NOT wipe the storage — that would nuke
    counters belonging to other replicas / prior runs that are still
    legitimately in their sliding window. Operators who want a hard
    Redis reset should flush the DB out of band.
    """
    global _last_counters_cleanup
    with _counters_lock:
        _hit_counters.clear()
        _last_counters_cleanup = None
    with _backend_lock:
        storage = _storage
    if isinstance(storage, MemoryStorage):
        try:
            storage.reset()
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("MemoryStorage reset raised %s — ignoring", exc)
    limiter.reset()


def _evict_stale_counters(now: datetime) -> int:
    """No-op shim retained for back-compat. The active backend
    (``limits`` ``MemoryStorage`` / ``RedisStorage``) handles bucket
    eviction internally; the legacy ``_hit_counters`` dict is never
    populated by the new enforcement path.
    """
    return 0


def _enforce_per_route(request: Request, max_calls: int, window_seconds: int) -> None:
    """Sliding-window enforcement shared by :func:`per_route_rate_limit`
    and :func:`per_route_rate_limit_for`. Raises 429 when the configured
    cap is hit; no return value on success.

    Delegates to ``limits.MovingWindowRateLimiter`` over the backend
    installed by :func:`configure_rate_limit_backend` (defaults to
    in-process :class:`MemoryStorage`). Keying preserves the prior
    behaviour (forge audit H5, 2026-05-23): ``(route_template, client_ip)``
    where ``route_template`` is the matched route's ``path`` (e.g.
    ``/sessions/{session_id}``), falling back to the raw URL path if no
    route was matched yet.
    """
    client_key = _client_key(request)
    route = request.scope.get("route")
    route_key = getattr(route, "path", None) or request.url.path
    # ``MovingWindowRateLimiter.hit(item, *identifiers)`` returns True
    # when allowed, False when blocked. The identifier tuple becomes the
    # bucket namespace — keeping ``route_key`` first ensures a single
    # client hitting many routes does NOT collapse into one bucket.
    item = RateLimitItemPerSecond(max_calls, multiples=window_seconds)
    with _backend_lock:
        strategy = _strategy
    try:
        allowed = strategy.hit(item, "per_route", route_key, client_key)
    except Exception as exc:
        # Storage backend (e.g. transient Redis hiccup) — fail OPEN so
        # we don't 503 the API on every request when Redis blips. The
        # log entry is operator-visible.
        log.warning("Rate-limit backend hit() raised %s — allowing request", exc)
        return
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"code": "RATE_LIMIT_EXCEEDED", "message": "Too many requests."},
        )


def per_route_rate_limit(max_calls: int, window_seconds: int = 60):
    """Return a FastAPI ``Depends``-compatible callable that enforces a per-client rate limit.

    Uses in-memory sliding-window counters keyed by ``(client_ip, route_path)``.
    Works correctly with FastAPI nested routers, where ``SlowAPIMiddleware`` cannot
    resolve per-route limits because ``scope["endpoint"]`` is not set when middleware runs.

    The limit is captured at the time of dependency construction. Prefer
    :func:`per_route_rate_limit_for` when the value should follow
    ``Config.api.rate_limits`` reloads at runtime.
    """

    def _dep(request: Request) -> None:
        if _per_route_disabled:
            return
        _enforce_per_route(request, max_calls, window_seconds)

    return _dep


def per_route_rate_limit_for(name: str):
    """Return a FastAPI ``Depends``-compatible callable that enforces the
    per-route rate limit configured for *name* (#1879 Slice A).

    The ``(max_calls, window_seconds)`` pair is resolved at request time
    from the table installed by :func:`configure_rate_limits` — i.e.
    ``Config.api.rate_limits[<name>]`` with the
    ``COGTRIX_RATE_LIMIT_<NAME>`` env-var override applied at startup.
    Unknown names fall back to the configured ``default`` spec.

    Route name conventions used today:

    * ``auth_register`` — ``POST /api/v1/auth/register``
    * ``auth_login``    — ``POST /api/v1/auth/login``
    * ``auth_refresh``  — ``POST /api/v1/auth/refresh``
    * ``saml_acs``      — ``POST /api/v1/auth/saml/acs``

    New names need only a matching entry in ``Config.api.rate_limits``
    (or the corresponding env-var override) — no other code changes.
    """

    def _dep(request: Request) -> None:
        if _per_route_disabled:
            return
        with _route_limits_lock:
            spec = _route_limits.get(name)
            default_spec = _default_limit_spec
        if spec is None:
            try:
                spec = parse_rate_limit_spec(default_spec)
            except ValueError as exc:  # pragma: no cover - validated at configure time
                log.error(
                    "Default rate-limit spec %r is invalid, allowing request: %s",
                    default_spec,
                    exc,
                )
                return
        max_calls, window_seconds = spec
        _enforce_per_route(request, max_calls, window_seconds)

    return _dep
