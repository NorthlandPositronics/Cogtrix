"""In-memory SAML assertion nonce cache with TTL expiry for replay protection."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

log = logging.getLogger("cogtrix.api.saml.nonce_cache")


@dataclass
class NonceEntry:
    """A single nonce entry in the cache."""

    assertion_id: str
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default=0.0)

    def __post_init__(self) -> None:
        """Set default expiry if not provided."""
        if self.expires_at == 0.0:
            # Default to 10 minutes from creation
            self.expires_at = self.created_at + 600.0

    @property
    def is_expired(self) -> bool:
        """Check if this nonce entry has expired."""
        return time.time() > self.expires_at


class SAMLNonceCache:
    """Thread-safe in-memory cache for SAML assertion IDs with TTL expiry."""

    def __init__(self, default_ttl_seconds: int = 600):
        """
        Initialize the nonce cache.

        Args:
            default_ttl_seconds: Default time-to-live for each nonce entry in seconds.
        """
        self._default_ttl = default_ttl_seconds
        self._cache: dict[str, NonceEntry] = {}
        self._lock = threading.RLock()

    def add(self, assertion_id: str, ttl_seconds: int | None = None) -> bool:
        """
        Add a new assertion ID to the cache.

        Args:
            assertion_id: The SAML assertion ID to cache.
            ttl_seconds: Optional custom TTL; uses default if not provided.

        Returns:
            True if added successfully, False if already present.
        """
        with self._lock:
            if assertion_id in self._cache and not self._cache[assertion_id].is_expired:
                return False

            ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
            entry = NonceEntry(assertion_id=assertion_id, expires_at=time.time() + ttl)
            self._cache[assertion_id] = entry

            # Periodic sweep: clean up expired entries when cache grows too large.
            # This prevents unbounded memory growth in long-running deployments.
            # Sweep threshold of 1000 entries balances cleanup frequency vs overhead.
            if len(self._cache) > 1000:
                expired_count = self.cleanup_expired()
                log.debug(
                    "SAMLNonceCache: sweep triggered at size %d, removed %d expired entries",
                    len(self._cache),
                    expired_count,
                )

            return True

    def contains(self, assertion_id: str) -> bool:
        """
        Check if an assertion ID exists and is not expired.

        Args:
            assertion_id: The SAML assertion ID to check.

        Returns:
            True if the assertion ID exists and is not expired, False otherwise.
        """
        with self._lock:
            entry = self._cache.get(assertion_id)
            if entry is None:
                return False
            if entry.is_expired:
                del self._cache[assertion_id]
                return False
            return True

    def remove(self, assertion_id: str) -> bool:
        """
        Remove an assertion ID from the cache.

        Args:
            assertion_id: The SAML assertion ID to remove.

        Returns:
            True if removed, False if not found.
        """
        with self._lock:
            if assertion_id in self._cache:
                del self._cache[assertion_id]
                return True
            return False

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from the cache.

        Returns:
            Number of entries removed.
        """
        with self._lock:
            expired_ids = [aid for aid, entry in self._cache.items() if entry.is_expired]
            for aid in expired_ids:
                del self._cache[aid]
            return len(expired_ids)

    def clear(self) -> None:
        """Remove all entries from the cache."""
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        """Return the number of non-expired entries in the cache."""
        with self._lock:
            return sum(1 for entry in self._cache.values() if not entry.is_expired)
