"""Tests for JIT user provisioning (Enterprise Phase 1 — task 1.2.5).

Covers:
  - JITConfig: domain allowlist, is_domain_allowed, configure/get/is_enabled
  - provision_jit_user: new user creation, existing user, domain denied,
    capacity exceeded, auto-team assignment, username deduplication
  - GET /api/v1/jit/status
  - POST /api/v1/jit/test
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

pytest.importorskip("fastapi")

_TEST_JWT_SECRET = "testsecret_mustbe32chars_minimum00"

from unittest.mock import patch  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker  # noqa: E402

from cogtrix_core.api.auth import create_access_token  # noqa: E402
from cogtrix_core.api.db.engine import get_db  # noqa: E402
from cogtrix_core.api.db.repositories.organization import OrganizationRepository  # noqa: E402
from cogtrix_core.api.db.repositories.teams import TeamRepository  # noqa: E402
from cogtrix_core.api.db.repositories.users import UserRepository  # noqa: E402
from cogtrix_core.api.jit.config import (  # noqa: E402
    JITConfig,
    configure_jit,
    get_jit_config,
    is_jit_enabled,
)


def _uid() -> str:
    return str(uuid.uuid4())


def _admin_header(user_id: str) -> dict:
    with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
        token = create_access_token(user_id=user_id, role="admin")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def reset_jit():
    import cogtrix_core.api.jit.config as _cfg

    _cfg._jit_config = None
    yield
    _cfg._jit_config = None


# ---------------------------------------------------------------------------
# JITConfig unit tests
# ---------------------------------------------------------------------------


class TestJITConfig:
    def test_configure_and_get(self):
        cfg = JITConfig(enabled=True, allowed_domains=["company.com"])
        assert not is_jit_enabled()
        configure_jit(cfg)
        assert is_jit_enabled()
        assert get_jit_config() is cfg

    def test_disabled_config_is_not_enabled(self):
        configure_jit(JITConfig(enabled=False))
        assert not is_jit_enabled()

    def test_is_domain_allowed_empty_list_accepts_all(self):
        cfg = JITConfig(enabled=True)
        assert cfg.is_domain_allowed("anyone@anydomain.com")

    def test_is_domain_allowed_specific_domain(self):
        cfg = JITConfig(enabled=True, allowed_domains=["company.com"])
        assert cfg.is_domain_allowed("alice@company.com")
        assert not cfg.is_domain_allowed("alice@other.com")

    def test_is_domain_allowed_case_insensitive(self):
        cfg = JITConfig(enabled=True, allowed_domains=["Company.com"])
        assert cfg.is_domain_allowed("alice@COMPANY.COM")

    def test_is_domain_allowed_invalid_email_returns_false(self):
        cfg = JITConfig(enabled=True, allowed_domains=["company.com"])
        assert not cfg.is_domain_allowed("not-an-email")

    def test_multiple_allowed_domains(self):
        cfg = JITConfig(enabled=True, allowed_domains=["company.com", "partner.org"])
        assert cfg.is_domain_allowed("alice@company.com")
        assert cfg.is_domain_allowed("bob@partner.org")
        assert not cfg.is_domain_allowed("eve@evil.com")


# ---------------------------------------------------------------------------
# provision_jit_user
# ---------------------------------------------------------------------------


class TestProvisionJITUser:
    def test_creates_new_user(self, sf):
        from cogtrix_core.api.jit.provisioning import provision_jit_user

        cfg = JITConfig(enabled=True)

        async def _run():
            async with sf() as db:
                user, token = await provision_jit_user(
                    email="new@example.com", username="new", config=cfg, db=db
                )
            assert user.email == "new@example.com"
            assert user.username == "new"
            assert token  # non-empty JWT

        asyncio.run(_run())

    def test_returns_existing_user(self, sf):
        from cogtrix_core.api.jit.provisioning import provision_jit_user

        cfg = JITConfig(enabled=True)
        user_id = _uid()

        async def _run():
            async with sf() as db:
                repo = UserRepository(db)
                await repo.create(
                    user_id=user_id,
                    username="existing",
                    email="existing@example.com",
                    password_hash="h",
                )
                await db.commit()

            async with sf() as db:
                user, token = await provision_jit_user(
                    email="existing@example.com",
                    username="existing",
                    config=cfg,
                    db=db,
                )
            assert user.id == user_id

        asyncio.run(_run())

    def test_domain_denied_raises_403(self, sf):
        from fastapi import HTTPException

        from cogtrix_core.api.jit.provisioning import provision_jit_user

        cfg = JITConfig(enabled=True, allowed_domains=["company.com"])

        async def _run():
            async with sf() as db:
                with pytest.raises(HTTPException) as exc_info:
                    await provision_jit_user(
                        email="hacker@evil.com", username="hacker", config=cfg, db=db
                    )
            assert exc_info.value.status_code == 403
            assert exc_info.value.detail["code"] == "JIT_DOMAIN_DENIED"

        asyncio.run(_run())

    def test_capacity_exceeded_raises_403(self, sf):
        from fastapi import HTTPException

        from cogtrix_core.api.jit.provisioning import provision_jit_user

        cfg = JITConfig(enabled=True, max_users=1)

        async def _run():
            # Create one user (filling the quota)
            async with sf() as db:
                org_repo = OrganizationRepository(db)
                default_org = await org_repo.ensure_default_org()
                await db.commit()
                user_repo = UserRepository(db)
                await user_repo.create(
                    user_id=_uid(),
                    username="first",
                    email="first@example.com",
                    password_hash="h",
                    org_id=default_org.id,
                )
                await db.commit()

            # Second user should be rejected
            async with sf() as db:
                with pytest.raises(HTTPException) as exc_info:
                    await provision_jit_user(
                        email="second@example.com", username="second", config=cfg, db=db
                    )
            assert exc_info.value.status_code == 403
            assert exc_info.value.detail["code"] == "JIT_CAPACITY_EXCEEDED"

        asyncio.run(_run())

    def test_disabled_raises_503(self, sf):
        from fastapi import HTTPException

        from cogtrix_core.api.jit.provisioning import provision_jit_user

        cfg = JITConfig(enabled=False)

        async def _run():
            async with sf() as db:
                with pytest.raises(HTTPException) as exc_info:
                    await provision_jit_user(
                        email="alice@example.com", username="alice", config=cfg, db=db
                    )
            assert exc_info.value.status_code == 503

        asyncio.run(_run())

    def test_auto_team_assignment(self, sf):
        from cogtrix_core.api.jit.provisioning import provision_jit_user

        async def _run():
            org_id = _uid()
            team_id = _uid()

            async with sf() as db:
                org_repo = OrganizationRepository(db)
                team_repo = TeamRepository(db)
                await org_repo.create(org_id=org_id, name="JIT Org", slug="jit-org")
                await team_repo.create(team_id=team_id, org_id=org_id, name="Auto Team")
                await db.commit()

            cfg = JITConfig(enabled=True, org_id=org_id, auto_team_id=team_id)

            async with sf() as db:
                user, _ = await provision_jit_user(
                    email="auto@example.com", username="auto", config=cfg, db=db
                )

            async with sf() as db:
                team_repo = TeamRepository(db)
                count = await team_repo.count_members(team_id)
            assert count == 1

        asyncio.run(_run())

    def test_auto_team_cross_org_raises_500(self, sf):
        """Auto-team must belong to the user's org — cross-org is a security violation."""
        from fastapi import HTTPException

        from cogtrix_core.api.jit.provisioning import provision_jit_user

        async def _run():
            org_a_id = _uid()
            org_b_id = _uid()
            team_b_id = _uid()

            async with sf() as db:
                org_repo = OrganizationRepository(db)
                team_repo = TeamRepository(db)
                await org_repo.create(org_id=org_a_id, name="JIT Org A", slug="jit-org-a")
                await org_repo.create(org_id=org_b_id, name="JIT Org B", slug="jit-org-b")
                await team_repo.create(team_id=team_b_id, org_id=org_b_id, name="Team B")
                await db.commit()

            # Config points to org A, but auto_team_id belongs to org B.
            cfg = JITConfig(enabled=True, org_id=org_a_id, auto_team_id=team_b_id)

            with pytest.raises(HTTPException) as exc_info:
                async with sf() as db:
                    await provision_jit_user(
                        email="cross@example.com", username="cross", config=cfg, db=db
                    )

            assert exc_info.value.status_code == 500
            assert exc_info.value.detail["code"] == "JIT_TEAM_ORG_MISMATCH"

            # Verify user was NOT added to the cross-org team.
            async with sf() as db:
                team_repo = TeamRepository(db)
                count = await team_repo.count_members(team_b_id)
            assert count == 0

        asyncio.run(_run())

    def test_username_deduplication(self, sf):
        from cogtrix_core.api.jit.provisioning import provision_jit_user

        cfg = JITConfig(enabled=True)

        async def _run():
            async with sf() as db:
                repo = UserRepository(db)
                await repo.create(
                    user_id=_uid(),
                    username="alice",
                    email="alice.one@example.com",
                    password_hash="h",
                )
                await db.commit()

            async with sf() as db:
                user, _ = await provision_jit_user(
                    email="alice.two@example.com", username="alice", config=cfg, db=db
                )
            # Username should be deduplicated with a suffix
            assert user.username != "alice"
            assert "alice" in user.username

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# JIT routes
# ---------------------------------------------------------------------------


@pytest.fixture()
def jit_client(engine):
    factory = async_sessionmaker(engine, expire_on_commit=False)
    admin_id = _uid()

    from cogtrix_core.api.app import create_app

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

    app.dependency_overrides.clear()


class TestJITRoutes:
    def test_status_not_configured(self, jit_client):
        client, admin_id = jit_client
        r = client.get("/api/v1/jit/status", headers=_admin_header(admin_id))
        assert r.status_code == 200
        assert r.json()["data"]["configured"] is False
        assert r.json()["data"]["enabled"] is False

    def test_status_configured(self, jit_client):
        client, admin_id = jit_client
        configure_jit(JITConfig(enabled=True, allowed_domains=["company.com"]))
        r = client.get("/api/v1/jit/status", headers=_admin_header(admin_id))
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["configured"] is True
        assert data["enabled"] is True
        assert "company.com" in data["allowed_domains"]

    def test_test_allowed_email(self, jit_client):
        client, admin_id = jit_client
        configure_jit(JITConfig(enabled=True, allowed_domains=["company.com"]))
        r = client.post(
            "/api/v1/jit/test",
            json={"email": "alice@company.com"},
            headers=_admin_header(admin_id),
        )
        assert r.status_code == 200
        assert r.json()["data"]["allowed"] is True

    def test_test_denied_email(self, jit_client):
        client, admin_id = jit_client
        configure_jit(JITConfig(enabled=True, allowed_domains=["company.com"]))
        r = client.post(
            "/api/v1/jit/test",
            json={"email": "alice@evil.com"},
            headers=_admin_header(admin_id),
        )
        assert r.status_code == 200
        assert r.json()["data"]["allowed"] is False

    def test_test_when_not_configured(self, jit_client):
        client, admin_id = jit_client
        r = client.post(
            "/api/v1/jit/test",
            json={"email": "alice@company.com"},
            headers=_admin_header(admin_id),
        )
        assert r.status_code == 200
        assert r.json()["data"]["allowed"] is False

    def test_requires_admin(self, jit_client):
        client, _ = jit_client
        with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
            non_admin = create_access_token(user_id=_uid(), role="user")
        r = client.get("/api/v1/jit/status", headers={"Authorization": f"Bearer {non_admin}"})
        assert r.status_code == 403
