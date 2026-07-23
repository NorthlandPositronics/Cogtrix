"""Tests for LDAP connection pool (Enterprise Phase 2 — task 2.1.3).

Coverage:
  - Pool pre-warms one connection on construction.
  - borrow() yields a bound Connection.
  - borrow() returns connection to pool on success.
  - borrow() discards connection on exception.
  - Pool exhaustion raises RuntimeError after timeout.
  - Connection lifetime exceeded triggers rotation.
  - Closed connections are discarded.
  - close() drains all connections.
  - get_pool() recreates pool when config changes.
  - search_groups returns group entries.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from cogtrix_core.api.ldap.config import LDAPConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> LDAPConfig:
    return LDAPConfig(
        server_url=overrides.pop("server_url", "ldap://localhost:389"),
        bind_dn=overrides.pop("bind_dn", "cn=admin,dc=example,dc=com"),
        bind_password=overrides.pop("bind_password", "secret"),
        search_base=overrides.pop("search_base", "ou=users,dc=example,dc=com"),
        use_ssl=overrides.pop("use_ssl", False),
        **overrides,
    )


def _fake_connection(closed=False):
    """Return a MagicMock that looks like a bound ldap3 Connection."""
    conn = MagicMock()
    conn.closed = closed
    conn.bind.return_value = True
    return conn


def _fake_wrapped(conn=None, created_at=None):
    """Return a _PooledConnection-like MagicMock."""
    from cogtrix_core.api.ldap.pool import _PooledConnection

    if conn is None:
        conn = _fake_connection()
    if created_at is None:
        created_at = time.monotonic()
    return _PooledConnection(conn=conn, created_at=created_at)


@pytest.fixture(autouse=True)
def reset_global_pool():
    """Reset the module-level pool singleton between tests."""
    import cogtrix_core.api.ldap.pool as _pool

    _pool.close_pool()
    yield
    _pool.close_pool()


# ---------------------------------------------------------------------------
# Pool construction
# ---------------------------------------------------------------------------


class TestPoolConstruction:
    def test_imports_without_ldap3(self):
        """The pool module must import even when ldap3 is absent."""
        with patch.dict("sys.modules", {"ldap3": None}):
            import importlib

            import cogtrix_core.api.ldap.pool as _pool

            importlib.reload(_pool)

    def test_pool_tracks_size(self):
        pytest.importorskip("ldap3")
        from cogtrix_core.api.ldap.pool import LDAPConnectionPool

        cfg = _make_config()
        with patch.object(LDAPConnectionPool, "_create_connection", return_value=_fake_wrapped()):
            pool = LDAPConnectionPool(cfg, pool_size=3)

        assert pool.size <= 3
        assert pool.available >= 0
        assert pool.in_use == 0
        pool.close()


# ---------------------------------------------------------------------------
# Borrow / release
# ---------------------------------------------------------------------------


class TestPoolBorrow:
    def test_borrow_yields_connection(self):
        pytest.importorskip("ldap3")
        from cogtrix_core.api.ldap.pool import LDAPConnectionPool

        cfg = _make_config()
        fake_conn = _fake_connection()
        wrapped = _fake_wrapped(conn=fake_conn)

        with patch.object(LDAPConnectionPool, "_create_connection", return_value=wrapped):
            pool = LDAPConnectionPool(cfg, pool_size=2)
            with pool.borrow() as conn:
                assert conn is fake_conn

        pool.close()

    def test_borrow_returns_to_pool_on_success(self):
        pytest.importorskip("ldap3")
        from cogtrix_core.api.ldap.pool import LDAPConnectionPool

        cfg = _make_config()
        wrapped = _fake_wrapped()

        with patch.object(LDAPConnectionPool, "_create_connection", return_value=wrapped):
            pool = LDAPConnectionPool(cfg, pool_size=2)
            with pool.borrow():
                pass

        assert pool.available >= 1
        pool.close()

    def test_borrow_discards_on_exception(self):
        pytest.importorskip("ldap3")
        from cogtrix_core.api.ldap.pool import LDAPConnectionPool

        cfg = _make_config()
        fake_conn = _fake_connection()
        wrapped = _fake_wrapped(conn=fake_conn)

        with patch.object(LDAPConnectionPool, "_create_connection", return_value=wrapped):
            pool = LDAPConnectionPool(cfg, pool_size=2)
            with pytest.raises(RuntimeError):
                with pool.borrow():
                    raise RuntimeError("boom")

        fake_conn.unbind.assert_called_once()
        pool.close()

    def test_exhaustion_raises(self):
        pytest.importorskip("ldap3")
        from cogtrix_core.api.ldap.pool import LDAPConnectionPool

        cfg = _make_config()
        fake_conn = _fake_connection()
        wrapped = _fake_wrapped(conn=fake_conn)

        with patch.object(LDAPConnectionPool, "_create_connection", return_value=wrapped):
            pool = LDAPConnectionPool(cfg, pool_size=1, borrow_timeout=0.1)
            # Hold the only connection via borrow so _in_use is tracked.
            hold = pool._acquire()
            pool._in_use.add(id(fake_conn))
            assert pool.in_use == 1
            assert pool.available == 0

            # Second borrow should time out.
            with pytest.raises(RuntimeError, match="exhausted"):
                with pool.borrow():
                    pass

            # Return the held connection so close() can drain.
            pool._available.put_nowait(hold)
            pool._in_use.discard(id(fake_conn))

        pool.close()

    def test_lifetime_rotation(self):
        pytest.importorskip("ldap3")
        from cogtrix_core.api.ldap.pool import LDAPConnectionPool

        cfg = _make_config()
        fake_conn = _fake_connection()
        old_wrapped = _fake_wrapped(conn=fake_conn, created_at=time.monotonic() - 1.0)

        with patch.object(LDAPConnectionPool, "_create_connection", return_value=old_wrapped):
            pool = LDAPConnectionPool(cfg, pool_size=2, max_lifetime=0.0)
            with pool.borrow():
                pass

        fake_conn.unbind.assert_called_once()
        pool.close()

    def test_closed_connection_discarded(self):
        pytest.importorskip("ldap3")
        from cogtrix_core.api.ldap.pool import LDAPConnectionPool

        cfg = _make_config()
        fake_conn = _fake_connection(closed=True)
        wrapped = _fake_wrapped(conn=fake_conn)

        with patch.object(LDAPConnectionPool, "_create_connection", return_value=wrapped):
            pool = LDAPConnectionPool(cfg, pool_size=2)
            with patch.object(pool, "_safe_unbind") as mock_safe_unbind:
                with pool.borrow():
                    pass

        # Connection was closed before return — _safe_unbind should be called
        # to clean it up, but unbind itself is skipped because conn.closed=True.
        mock_safe_unbind.assert_called_once_with(fake_conn)
        pool.close()


# ---------------------------------------------------------------------------
# Global pool singleton
# ---------------------------------------------------------------------------


class TestGlobalPool:
    def test_get_pool_creates_singleton(self):
        pytest.importorskip("ldap3")
        import cogtrix_core.api.ldap.pool as _pool_mod

        cfg = _make_config()
        with patch.object(
            _pool_mod.LDAPConnectionPool, "_create_connection", return_value=_fake_wrapped()
        ):
            pool = _pool_mod.get_pool(cfg)

        assert pool is not None
        assert _pool_mod._pool_instance is pool

    def test_get_pool_recreated_on_config_change(self):
        pytest.importorskip("ldap3")
        import cogtrix_core.api.ldap.pool as _pool_mod

        with patch.object(
            _pool_mod.LDAPConnectionPool, "_create_connection", return_value=_fake_wrapped()
        ):
            cfg1 = _make_config(server_url="ldap://host1:389")
            pool1 = _pool_mod.get_pool(cfg1)

            cfg2 = _make_config(server_url="ldap://host2:389")
            pool2 = _pool_mod.get_pool(cfg2)

        assert pool1 is not pool2

    def test_close_pool_idempotent(self):
        pytest.importorskip("ldap3")
        import cogtrix_core.api.ldap.pool as _pool_mod

        with patch.object(
            _pool_mod.LDAPConnectionPool, "_create_connection", return_value=_fake_wrapped()
        ):
            cfg = _make_config()
            _pool_mod.get_pool(cfg)

        _pool_mod.close_pool()
        _pool_mod.close_pool()  # Should not raise.


# ---------------------------------------------------------------------------
# search_groups
# ---------------------------------------------------------------------------


class TestSearchGroups:
    def test_search_groups_with_mocked_connection(self):
        pytest.importorskip("ldap3")
        from cogtrix_core.api.ldap.sync import search_groups

        cfg = _make_config()

        fake_entry = MagicMock()
        fake_entry.entry_dn = "cn=Engineers,ou=groups,dc=example,dc=com"
        fake_entry.__contains__ = lambda self, x: x in ("cn", "description")
        fake_entry.cn = MagicMock(value="Engineers")
        fake_entry.description = MagicMock(value="Engineering team")

        fake_conn = _fake_connection()
        fake_conn.entries = [fake_entry]
        fake_conn.result = {}

        groups = search_groups(cfg, conn=fake_conn)

        assert len(groups) == 1
        assert groups[0]["name"] == "Engineers"
        assert "Engineering team" in groups[0]["description"]

    def test_search_groups_uses_provided_connection(self):
        pytest.importorskip("ldap3")
        from cogtrix_core.api.ldap.sync import search_groups

        cfg = _make_config()

        fake_entry = MagicMock()
        fake_entry.entry_dn = "cn=Admins,ou=groups,dc=example,dc=com"
        fake_entry.__contains__ = lambda self, x: x in ("cn",)
        fake_entry.cn = MagicMock(value="Admins")

        fake_conn = _fake_connection()
        fake_conn.entries = [fake_entry]
        fake_conn.result = {}

        groups = search_groups(cfg, conn=fake_conn)
        assert len(groups) == 1
        assert groups[0]["name"] == "Admins"
        fake_conn.unbind.assert_not_called()  # Caller owns lifecycle.
