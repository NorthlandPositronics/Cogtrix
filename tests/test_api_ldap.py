"""Tests for LDAP/AD sync (Enterprise Phase 1 — task 1.2.3).

ldap3 is an optional dependency. Tests that require an actual LDAP
connection are skipped when ldap3 is not installed.

Tests that do NOT require ldap3 (config, 503 routes, sync result dataclass)
run unconditionally.
"""

from __future__ import annotations

import asyncio
import os
import ssl
import uuid
from unittest.mock import patch

import pytest

_TEST_JWT_SECRET = "testsecret_mustbe32chars_minimum00"
os.environ.setdefault("COGTRIX_JWT_SECRET", _TEST_JWT_SECRET)
os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from src.api.auth import create_access_token  # noqa: E402
from src.api.db import models as _models  # noqa: E402, F401
from src.api.db.engine import Base, get_db  # noqa: E402
from src.api.ldap.config import (  # noqa: E402
    LDAPConfig,
    configure_ldap,
    get_ldap_config,
    is_ldap_configured,
)
from src.api.ldap.sync import LDAPSyncResult, _fetch_ldap_users  # noqa: E402


def _uid() -> str:
    return str(uuid.uuid4())


def _admin_header(user_id: str) -> dict:
    with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
        token = create_access_token(user_id=user_id, role="admin")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def reset_ldap_config():
    import src.api.ldap.config as _cfg

    _cfg._ldap_config = None
    yield
    _cfg._ldap_config = None


def _make_config(**overrides) -> LDAPConfig:
    return LDAPConfig(
        server_url=overrides.pop("server_url", "ldap://localhost:389"),
        bind_dn=overrides.pop("bind_dn", "cn=admin,dc=example,dc=com"),
        bind_password=overrides.pop("bind_password", "secret"),
        search_base=overrides.pop("search_base", "ou=users,dc=example,dc=com"),
        **overrides,
    )


# ---------------------------------------------------------------------------
# LDAPConfig unit tests
# ---------------------------------------------------------------------------


class TestLDAPConfig:
    def test_configure_and_get(self):
        cfg = _make_config()
        assert not is_ldap_configured()
        configure_ldap(cfg)
        assert is_ldap_configured()
        assert get_ldap_config() is cfg

    def test_configure_replaces_prior(self):
        cfg1 = _make_config(server_url="ldap://host1:389")
        cfg2 = _make_config(server_url="ldap://host2:389")
        configure_ldap(cfg1)
        configure_ldap(cfg2)
        assert get_ldap_config() is cfg2

    def test_not_configured_by_default(self):
        assert get_ldap_config() is None
        assert not is_ldap_configured()

    def test_default_search_filter(self):
        cfg = _make_config()
        assert cfg.search_filter == "(objectClass=person)"

    def test_default_attribute_map_has_required_keys(self):
        cfg = _make_config()
        assert "username" in cfg.attribute_map
        assert "email" in cfg.attribute_map

    def test_default_role_is_user(self):
        cfg = _make_config()
        assert cfg.default_role == "user"

    def test_page_size_default(self):
        cfg = _make_config()
        assert cfg.page_size == 200

    def test_custom_filter(self):
        cfg = _make_config(search_filter="(&(objectClass=person)(department=Engineering))")
        assert "Engineering" in cfg.search_filter

    def test_default_tls_verifies_certificates(self):
        cfg = _make_config()
        assert cfg.ldap_tls_skip_verify is False

    def test_tls_skip_verify_can_be_enabled_for_dev(self):
        cfg = _make_config(ldap_tls_skip_verify=True)
        assert cfg.ldap_tls_skip_verify is True


# ---------------------------------------------------------------------------
# LDAPSyncResult unit tests
# ---------------------------------------------------------------------------


class TestLDAPSyncResult:
    def test_total_processed(self):
        r = LDAPSyncResult(added=3, updated=2, skipped=1)
        assert r.total_processed == 6

    def test_success_when_no_errors(self):
        r = LDAPSyncResult(added=5)
        assert r.success is True

    def test_failure_when_errors_present(self):
        r = LDAPSyncResult(errors=["something failed"])
        assert r.success is False

    def test_default_all_zero(self):
        r = LDAPSyncResult()
        assert r.added == 0
        assert r.updated == 0
        assert r.skipped == 0
        assert r.errors == []


# ---------------------------------------------------------------------------
# Routes — 503 when not configured
# ---------------------------------------------------------------------------


@pytest.fixture()
def ldap_client():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)

    admin_id = _uid()

    async def _seed():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_seed())

    from src.api.app import create_app

    with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
        app = create_app()

        async def _override():
            async with factory() as session:
                try:
                    yield session
                    await session.commit()
                except Exception:
                    await session.rollback()
                    raise

        app.dependency_overrides[get_db] = _override
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, admin_id

    asyncio.run(engine.dispose())


class TestLDAPRoutes:
    def test_status_not_configured(self, ldap_client):
        client, admin_id = ldap_client
        r = client.get("/api/v1/ldap/status", headers=_admin_header(admin_id))
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["configured"] is False
        assert data["server_url"] is None

    def test_status_configured(self, ldap_client):
        client, admin_id = ldap_client
        configure_ldap(_make_config())
        r = client.get("/api/v1/ldap/status", headers=_admin_header(admin_id))
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["configured"] is True
        assert data["server_url"] == "ldap://localhost:389"

    def test_sync_503_when_not_configured(self, ldap_client):
        client, admin_id = ldap_client
        r = client.post("/api/v1/ldap/sync", headers=_admin_header(admin_id))
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "LDAP_NOT_CONFIGURED"

    def test_status_requires_admin(self, ldap_client):
        client, _ = ldap_client
        with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
            non_admin_token = create_access_token(user_id=_uid(), role="user")
        r = client.get(
            "/api/v1/ldap/status",
            headers={"Authorization": f"Bearer {non_admin_token}"},
        )
        assert r.status_code == 403

    def test_sync_requires_admin(self, ldap_client):
        client, _ = ldap_client
        configure_ldap(_make_config())
        with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
            non_admin_token = create_access_token(user_id=_uid(), role="user")
        r = client.post(
            "/api/v1/ldap/sync",
            headers={"Authorization": f"Bearer {non_admin_token}"},
        )
        assert r.status_code == 403


# ---------------------------------------------------------------------------
# Sync logic — with mocked ldap3
# ---------------------------------------------------------------------------


class TestLDAPSyncWithMock:
    """Tests that mock ldap3's Connection to avoid needing a real LDAP server."""

    def test_fetch_ldap_users_builds_verified_tls_by_default(self):
        from unittest.mock import MagicMock, patch

        cfg = _make_config()
        cfg.use_ssl = True

        server_ctor = MagicMock()
        conn_instance = MagicMock()
        conn_instance.bind.return_value = True
        conn_instance.entries = []
        conn_instance.result = {}
        conn_ctor = MagicMock(return_value=conn_instance)
        tls_ctor = MagicMock(return_value="tls-object")

        with (
            patch(
                "src.api.ldap.sync._require_ldap3",
                return_value=(server_ctor, conn_ctor, tls_ctor, object(), object()),
            ),
            patch("src.api.ldap.sync.certifi.where", return_value="/etc/ssl/certs/ca.pem"),
        ):
            users = _fetch_ldap_users(cfg)

        assert users == []
        tls_ctor.assert_called_once_with(
            validate=ssl.CERT_REQUIRED,
            ca_certs_file="/etc/ssl/certs/ca.pem",
            version=ssl.PROTOCOL_TLS_CLIENT,
        )
        server_ctor.assert_called_once_with(
            cfg.server_url,
            use_ssl=True,
            tls="tls-object",
            get_info=None,
        )

    def test_fetch_ldap_users_can_skip_cert_validation_in_dev(self):
        from unittest.mock import MagicMock, patch

        cfg = _make_config(ldap_tls_skip_verify=True)
        cfg.use_ssl = True

        server_ctor = MagicMock()
        conn_instance = MagicMock()
        conn_instance.bind.return_value = True
        conn_instance.entries = []
        conn_instance.result = {}
        conn_ctor = MagicMock(return_value=conn_instance)
        tls_ctor = MagicMock(return_value="tls-object")

        with patch(
            "src.api.ldap.sync._require_ldap3",
            return_value=(server_ctor, conn_ctor, tls_ctor, object(), object()),
        ):
            users = _fetch_ldap_users(cfg)

        assert users == []
        tls_ctor.assert_called_once_with(
            validate=ssl.CERT_NONE,
            ca_certs_file=None,
            version=ssl.PROTOCOL_TLS_CLIENT,
        )
        server_ctor.assert_called_once_with(
            cfg.server_url,
            use_ssl=True,
            tls="tls-object",
            get_info=None,
        )

    def test_sync_provisions_new_users(self):
        """Sync creates new Cogtrix users for LDAP entries."""
        pytest.importorskip("ldap3")

        from unittest.mock import MagicMock, patch

        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            echo=False,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        factory = async_sessionmaker(engine, expire_on_commit=False)

        async def _run():
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            from src.api.db.repositories.users import UserRepository
            from src.api.ldap.sync import sync_users

            cfg = _make_config()
            cfg.use_ssl = False

            # Fake LDAP entries.
            fake_entry_1 = MagicMock()
            fake_entry_1.__contains__ = lambda self, x: True
            fake_entry_1.__getitem__ = lambda self, x: MagicMock(
                value="alice" if x == "sAMAccountName" else "alice@example.com"
            )

            fake_entry_2 = MagicMock()
            fake_entry_2.__contains__ = lambda self, x: True
            fake_entry_2.__getitem__ = lambda self, x: MagicMock(
                value="bob" if x == "sAMAccountName" else "bob@example.com"
            )

            mock_conn = MagicMock()
            mock_conn.bind.return_value = True
            mock_conn.entries = [fake_entry_1, fake_entry_2]
            mock_conn.result = {}

            with patch("src.api.ldap.sync._fetch_ldap_users") as mock_fetch:
                mock_fetch.return_value = [
                    {"username": "alice", "email": "alice@example.com"},
                    {"username": "bob", "email": "bob@example.com"},
                ]

                async with factory() as db:
                    result = await sync_users(cfg, db)

            assert result.added == 2
            assert result.updated == 0
            assert result.success is True

            async with factory() as db:
                user_repo = UserRepository(db)
                alice = await user_repo.get_by_username("alice")
                assert alice is not None
                assert alice.email == "alice@example.com"

        asyncio.run(_run())
        asyncio.run(engine.dispose())

    def test_sync_updates_existing_user_email(self):
        """Sync updates email when it differs from LDAP."""
        pytest.importorskip("ldap3")

        from unittest.mock import patch

        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            echo=False,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        factory = async_sessionmaker(engine, expire_on_commit=False)

        async def _run():
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            from src.api.db.repositories.users import UserRepository
            from src.api.ldap.sync import sync_users

            cfg = _make_config()
            cfg.org_id = _uid()

            # Pre-create alice with an old email.
            async with factory() as db:
                repo = UserRepository(db)
                await repo.create(
                    user_id=_uid(),
                    username="alice",
                    email="old-alice@example.com",
                    password_hash="h",
                    org_id=cfg.org_id,
                )
                await db.commit()

            with patch("src.api.ldap.sync._fetch_ldap_users") as mock_fetch:
                mock_fetch.return_value = [
                    {"username": "alice", "email": "new-alice@example.com"},
                ]
                async with factory() as db:
                    result = await sync_users(cfg, db)

            assert result.updated == 1
            assert result.added == 0

            async with factory() as db:
                repo = UserRepository(db)
                alice = await repo.get_by_username("alice")
                assert alice.email == "new-alice@example.com"

        asyncio.run(_run())
        asyncio.run(engine.dispose())
