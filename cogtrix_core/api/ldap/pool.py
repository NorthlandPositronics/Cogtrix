"""LDAP/AD connection pool (Enterprise Phase 2 — task 2.1.3).

Manages a pool of reusable ``ldap3.Connection`` objects so that sync and
search operations do not pay the bind cost on every call.  The pool is
thread-safe and supports configurable size, borrow timeout, and connection
lifetime.

Usage::

    from cogtrix_core.api.ldap.config import LDAPConfig
    from cogtrix_core.api.ldap.pool import LDAPConnectionPool

    pool = LDAPConnectionPool(config)
    with pool.borrow() as conn:
        conn.search(search_base, search_filter, ...)
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ldap3 import Connection  # type: ignore[import]

    from cogtrix_core.api.ldap.config import LDAPConfig

log = logging.getLogger("cogtrix.api.ldap")


@dataclass
class _PooledConnection:
    """Wrapper that tracks when a connection was created."""

    conn: Connection
    created_at: float


class LDAPConnectionPool:
    """Thread-safe pool of bound LDAP connections.

    Args:
        config:       LDAP configuration used to create every connection.
        pool_size:    Maximum number of connections to keep open.
        borrow_timeout: Seconds to wait for a free connection before raising.
        max_lifetime: Seconds after which a connection is discarded and
                      replaced on the next borrow.
    """

    def __init__(
        self,
        config: LDAPConfig,
        *,
        pool_size: int = 5,
        borrow_timeout: float = 10.0,
        max_lifetime: float = 300.0,
    ) -> None:
        self._config = config
        self._pool_size = pool_size
        self._borrow_timeout = borrow_timeout
        self._max_lifetime = max_lifetime

        self._available: queue.Queue[_PooledConnection] = queue.Queue()
        self._in_use: set[int] = set()
        self._lock = threading.Lock()
        self._closed = False

        # Pre-warm the pool with a single connection so the first borrow
        # is fast; remaining connections are created lazily.
        self._warm(1)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Drain the pool and unbind every connection. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True

        while not self._available.empty():
            try:
                wrapped = self._available.get_nowait()
                self._safe_unbind(wrapped.conn)
            except queue.Empty:
                break

        log.info("LDAP connection pool closed")

    # ------------------------------------------------------------------
    # Borrow / release
    # ------------------------------------------------------------------

    @contextmanager
    def borrow(self):
        """Context manager that yields a bound ``ldap3.Connection``.

        The connection is returned to the pool on successful exit.
        If the caller raises, the connection is discarded.

        Raises:
            RuntimeError: when the pool is closed or exhausted.
        """
        wrapped = self._acquire()
        conn_id = id(wrapped.conn)
        try:
            with self._lock:
                self._in_use.add(conn_id)
            yield wrapped.conn
        except Exception:
            # Discard on error — the connection state is suspect.
            self._safe_unbind(wrapped.conn)
            with self._lock:
                self._in_use.discard(conn_id)
            raise
        else:
            # Return to pool after validating health.
            if self._is_healthy(wrapped):
                self._available.put_nowait(wrapped)
            else:
                self._safe_unbind(wrapped.conn)
                # Back-fill so the pool doesn't shrink permanently.
                self._warm(1)
            with self._lock:
                self._in_use.discard(conn_id)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _acquire(self) -> _PooledConnection:
        """Return an available connection or create a new one."""
        if self._closed:
            raise RuntimeError("LDAPConnectionPool is closed")

        # Fast path — connection already available.
        try:
            return self._available.get_nowait()
        except queue.Empty:
            pass

        # Slow path — create a fresh connection if under pool limit.
        with self._lock:
            current_size = self._available.qsize() + len(self._in_use)
            if current_size < self._pool_size:
                return self._create_connection()

        # Pool at capacity — block until a connection is returned.
        try:
            return self._available.get(timeout=self._borrow_timeout)
        except queue.Empty as exc:
            raise RuntimeError(
                f"LDAP connection pool exhausted (size={self._pool_size}, "
                f"timeout={self._borrow_timeout}s)"
            ) from exc

    def _create_connection(self) -> _PooledConnection:
        """Build a new bound ``ldap3.Connection``."""
        from ldap3 import Connection, Server, Tls  # type: ignore[import]

        cfg = self._config
        tls = None
        if cfg.use_ssl:
            import ssl as _ssl

            import certifi

            validate = _ssl.CERT_NONE if cfg.ldap_tls_skip_verify else _ssl.CERT_REQUIRED
            ca_certs = None if cfg.ldap_tls_skip_verify else certifi.where()
            tls = Tls(
                validate=validate,
                ca_certs_file=ca_certs,
                version=_ssl.PROTOCOL_TLS_CLIENT,
            )

        server = Server(
            cfg.server_url,
            use_ssl=cfg.use_ssl,
            tls=tls,
            get_info=None,
        )
        conn = Connection(
            server,
            user=cfg.bind_dn,
            password=cfg.bind_password,
            auto_bind=True,
        )
        if not conn.bind():
            raise RuntimeError(f"LDAP bind failed: {conn.result}")

        log.debug("LDAP pool: created new connection to %s", cfg.server_url)
        return _PooledConnection(conn=conn, created_at=time.monotonic())

    def _is_healthy(self, wrapped: _PooledConnection) -> bool:
        """Return True when the connection is still usable."""
        # Connection lifetime exceeded — force rotation.
        if time.monotonic() - wrapped.created_at > self._max_lifetime:
            log.debug("LDAP pool: connection exceeded max lifetime, discarding")
            return False

        # ldap3 exposes a closed flag on the socket level.
        try:
            if wrapped.conn.closed:
                log.debug("LDAP pool: connection is closed, discarding")
                return False
        except Exception as exc:
            log.debug("LDAP pool: health check error: %s", exc)
            return False

        return True

    def _safe_unbind(self, conn: Connection) -> None:
        """Unbind without raising."""
        try:
            if not conn.closed:
                conn.unbind()
        except Exception as exc:
            log.debug("LDAP pool: unbind error (ignored): %s", exc)

    def _warm(self, n: int) -> None:
        """Pre-create *n* connections if the pool is under capacity."""
        with self._lock:
            for _ in range(n):
                current_size = self._available.qsize() + len(self._in_use)
                if current_size >= self._pool_size:
                    break
                try:
                    wrapped = self._create_connection()
                    self._available.put_nowait(wrapped)
                except Exception as exc:
                    log.warning("LDAP pool: warm-up failed: %s", exc)
                    break

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Current number of open connections (available + in-use)."""
        with self._lock:
            return self._available.qsize() + len(self._in_use)

    @property
    def available(self) -> int:
        """Number of connections ready to borrow."""
        return self._available.qsize()

    @property
    def in_use(self) -> int:
        """Number of connections currently borrowed."""
        with self._lock:
            return len(self._in_use)

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"size={self.size} available={self.available} in_use={self.in_use}>"
        )


# ---------------------------------------------------------------------------
# Global pool (singleton per process, recreated when config changes)
# ---------------------------------------------------------------------------

_pool_lock = threading.Lock()
_pool_instance: LDAPConnectionPool | None = None
_pool_config_hash: int | None = None


def get_pool(config: LDAPConfig) -> LDAPConnectionPool:
    """Return the global pool for *config*, recreating it when config changes."""
    global _pool_instance, _pool_config_hash

    new_hash = hash(
        (
            config.server_url,
            config.bind_dn,
            config.bind_password,
            config.use_ssl,
            config.ldap_tls_skip_verify,
        )
    )

    with _pool_lock:
        if _pool_instance is not None and _pool_config_hash == new_hash:
            return _pool_instance

        # Config changed — drain old pool first.
        if _pool_instance is not None:
            _pool_instance.close()

        _pool_instance = LDAPConnectionPool(config)
        _pool_config_hash = new_hash
        log.info("LDAP connection pool initialised for %s", config.server_url)
        return _pool_instance


def close_pool() -> None:
    """Close the global pool, if any.  Safe to call repeatedly."""
    global _pool_instance, _pool_config_hash
    with _pool_lock:
        if _pool_instance is not None:
            _pool_instance.close()
            _pool_instance = None
            _pool_config_hash = None
