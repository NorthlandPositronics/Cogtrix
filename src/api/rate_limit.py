"""Shared rate-limit configuration for the Cogtrix API."""

from __future__ import annotations

import ipaddress
import logging
import threading
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import HTTPException, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address

log = logging.getLogger("cogtrix.api.rate_limit")

_lock = threading.Lock()
_trusted_proxy_networks: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = ()

# In-memory sliding-window counters for per-route rate limiting.
# Key: (client_ip, route_path) → list of hit timestamps.
_hit_counters: dict[tuple[str, str], list[datetime]] = defaultdict(list)
_counters_lock = threading.Lock()
# Set to True in test fixtures to bypass per-route limits without affecting
# global SlowAPI middleware. Never set this in production code.
_per_route_disabled: bool = False
# Periodic eviction of stale counter keys to bound memory in long-running processes.
_last_counters_cleanup: datetime | None = None
_COUNTERS_CLEANUP_INTERVAL_SEC = 300  # 5 minutes
_COUNTERS_MAX_AGE_SEC = 3600  # evict keys idle for >1 hour


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


def _client_key(request: Request) -> str:
    """Return the client IP key for rate limiting, honouring trusted proxy headers."""
    remote_addr = get_remote_address(request)

    with _lock:
        networks = _trusted_proxy_networks

    if networks and _is_trusted_proxy(remote_addr, networks):
        forwarded_for = request.headers.get("x-forwarded-for", "")
        if forwarded_for:
            client_addr = forwarded_for.split(",", 1)[0].strip()
            if client_addr:
                return client_addr

    return remote_addr


# Preserved for the global SlowAPI middleware default (120 req/min blunt guard).
rate_limit_key = _client_key
limiter = Limiter(key_func=rate_limit_key, default_limits=["120/minute"])


def reset_rate_limits() -> None:
    """Clear all per-route sliding-window counters.  Called at app startup."""
    global _last_counters_cleanup
    with _counters_lock:
        _hit_counters.clear()
        _last_counters_cleanup = None
    limiter.reset()


def _evict_stale_counters(now: datetime) -> int:
    """Remove counter keys whose most recent hit is older than _COUNTERS_MAX_AGE_SEC.

    Must be called while holding _counters_lock.  Returns the number of keys removed.
    """
    cutoff = now - timedelta(seconds=_COUNTERS_MAX_AGE_SEC)
    stale = [k for k, v in _hit_counters.items() if not any(t > cutoff for t in v)]
    for k in stale:
        del _hit_counters[k]
    return len(stale)


def per_route_rate_limit(max_calls: int, window_seconds: int = 60):
    """Return a FastAPI ``Depends``-compatible callable that enforces a per-client rate limit.

    Uses in-memory sliding-window counters keyed by ``(client_ip, route_path)``.
    Works correctly with FastAPI nested routers, where ``SlowAPIMiddleware`` cannot
    resolve per-route limits because ``scope["endpoint"]`` is not set when middleware runs.
    """

    def _dep(request: Request) -> None:
        if _per_route_disabled:
            return
        client_key = _client_key(request)
        key = (client_key, request.url.path)
        now = datetime.now(UTC)
        cutoff = now - timedelta(seconds=window_seconds)
        with _counters_lock:
            # Periodic eviction of keys with no recent hits to bound memory.
            global _last_counters_cleanup
            if (
                _last_counters_cleanup is None
                or (now - _last_counters_cleanup).total_seconds() >= _COUNTERS_CLEANUP_INTERVAL_SEC
            ):
                removed = _evict_stale_counters(now)
                _last_counters_cleanup = now
                if removed:
                    log.debug("Evicted %d stale rate-limit counter keys", removed)

            hits = [t for t in _hit_counters[key] if t > cutoff]
            if len(hits) >= max_calls:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={"code": "RATE_LIMIT_EXCEEDED", "message": "Too many requests."},
                )
            hits.append(now)
            _hit_counters[key] = hits

    return _dep
