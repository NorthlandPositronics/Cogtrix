"""Redis-backed session presence store for horizontal scaling.

Tracks {session_id → last_activity_timestamp} in Redis with automatic TTL.
Falls back gracefully if Redis is unavailable or not configured.

Each running Cogtrix API instance keeps the full ``ApiSession`` object (live
LLM, asyncio Queue, memory manager) in its own process memory — those cannot
be serialized.  What goes into Redis is *only* the presence/activity record:
    "cogtrix:session:{session_id}"  →  str(last_activity_unix_timestamp)

This is enough for:
- Correct idle eviction across the fleet (any instance can see when a session
  was last used, even if that use happened on a different instance).
- Session existence checks before hitting the DB.
- Graceful session hand-off (instance B warms from DB when not in local cache).
"""

from __future__ import annotations

import logging

log = logging.getLogger("cogtrix.api.redis_sessions")

# ---------------------------------------------------------------------------
# Optional redis dependency
# ---------------------------------------------------------------------------

try:
    import redis.asyncio as aioredis  # type: ignore[import-untyped]

    _HAS_REDIS = True
except ImportError:  # pragma: no cover
    _HAS_REDIS = False
    aioredis = None  # type: ignore[assignment]

# Key prefix for all session presence keys.
_KEY_PREFIX = "cogtrix:session:"

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_store: SessionPresenceStore | None = None


# ---------------------------------------------------------------------------
# Public configuration API
# ---------------------------------------------------------------------------


def configure_redis(redis_url: str, ttl_seconds: int = 7200) -> None:
    """Initialize the module-level presence store.

    Call once at startup from the application lifespan.  Subsequent calls
    update the LLM factory reference (model switches) but do NOT recreate
    the store or the underlying Redis connection.
    """
    global _store
    if _store is None:
        _store = SessionPresenceStore(redis_url=redis_url, ttl_seconds=ttl_seconds)


def get_store() -> SessionPresenceStore | None:
    """Return the configured store, or None if Redis not configured."""
    return _store


# ---------------------------------------------------------------------------
# SessionPresenceStore
# ---------------------------------------------------------------------------


class SessionPresenceStore:
    """Redis-backed store for session last_activity timestamps.

    All methods are async and safe to call when Redis is not configured
    or unavailable — they become no-ops or return sensible defaults.

    Thread/concurrency safety: uses a single ``redis.asyncio`` client which
    is internally thread-safe and coroutine-safe.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        ttl_seconds: int = 7200,
    ) -> None:
        """
        Args:
            redis_url: Redis connection URL, e.g. ``"redis://localhost:6379/0"``
                or ``"rediss://..."`` for TLS.  ``None`` = disabled (all methods
                become no-ops).
            ttl_seconds: Key TTL in Redis — sessions not touched within this
                window are automatically expired by Redis (default 2 hours).
        """
        self._redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self._client: aioredis.Redis | None = None  # type: ignore[name-defined]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open Redis connection.  No-op if ``redis_url`` is None or
        ``redis`` package is not installed.  Logs a warning on failure.
        """
        if not _HAS_REDIS or not self._redis_url:
            return
        try:
            assert aioredis is not None
            client = aioredis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
            )
            # Verify connectivity with a lightweight PING.
            await client.ping()
            self._client = client
        except Exception as exc:
            log.warning("Redis connect failed (%s): %s", self._redis_url, exc)
            self._client = None

    async def disconnect(self) -> None:
        """Close Redis connection gracefully."""
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception as exc:  # pragma: no cover
                log.debug("Redis disconnect error: %s", exc)
            finally:
                self._client = None

    # ------------------------------------------------------------------
    # Presence operations
    # ------------------------------------------------------------------

    async def touch(self, session_id: str, timestamp: float) -> None:
        """Record or refresh ``last_activity`` for a session.

        Key:   ``cogtrix:session:{session_id}``
        Value: ``str(timestamp)``
        TTL:   ``self.ttl_seconds`` (reset on every touch)

        No-op when Redis is unavailable; connection errors are caught and
        logged at DEBUG to avoid spamming logs on transient failures.
        """
        if self._client is None:
            return
        key = _KEY_PREFIX + session_id
        try:
            await self._client.set(key, str(timestamp), ex=self.ttl_seconds)
        except Exception as exc:
            log.debug("Redis touch failed for session %s: %s", session_id, exc)

    async def get_last_activity(self, session_id: str) -> float | None:
        """Return ``last_activity`` timestamp, or ``None`` if the session
        key is absent or Redis is unavailable.
        """
        if self._client is None:
            return None
        key = _KEY_PREFIX + session_id
        try:
            raw = await self._client.get(key)
            if raw is None:
                return None
            return float(raw)
        except Exception as exc:
            log.debug("Redis get_last_activity failed for session %s: %s", session_id, exc)
            return None

    async def remove(self, session_id: str) -> None:
        """Delete session key from Redis.  No-op if unavailable."""
        if self._client is None:
            return
        key = _KEY_PREFIX + session_id
        try:
            await self._client.delete(key)
        except Exception as exc:
            log.debug("Redis remove failed for session %s: %s", session_id, exc)

    async def list_active(self) -> list[str]:
        """Return all session IDs currently tracked in Redis.

        Uses ``SCAN`` (not ``KEYS``) to avoid blocking the Redis server.
        Returns ``[]`` when Redis is unavailable.
        """
        if self._client is None:
            return []
        prefix = _KEY_PREFIX
        session_ids: list[str] = []
        try:
            async for key in self._client.scan_iter(match=f"{prefix}*"):
                # Strip the prefix to return bare session IDs.
                session_ids.append(key[len(prefix) :])
        except Exception as exc:
            log.debug("Redis list_active failed: %s", exc)
            return []
        return session_ids

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """True if a Redis connection is open (post-``connect()``)."""
        return self._client is not None
