"""Regression tests for #1136 — assistant admin enumeration org-scoping (ADR-0055 Phase 1).

Verifies that:
- Superadmins can access all assistant admin enumeration endpoints.
- Regular admins receive 403 ORG_SCOPING_NOT_AVAILABLE on endpoints that
  lack org metadata in Phase 1.
- Non-admin users are blocked by the existing require_admin dependency.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

pytest.importorskip("fastapi")

_TEST_JWT_SECRET = "testsecret_mustbe32chars_minimum00"
os.environ.setdefault("COGTRIX_JWT_SECRET", _TEST_JWT_SECRET)
os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("COGTRIX_ENABLE_ORG_SCOPING", "true")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from src.api.db.engine import Base  # noqa: E402
from src.api.db.models import Organization, User  # noqa: E402


def _uid() -> str:
    return str(uuid.uuid4())


@pytest.fixture()
def engine():
    eng = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    async def _setup():
        async with eng.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_setup())
    yield eng
    asyncio.run(eng.dispose())


@pytest.fixture()
def sf(engine):
    return async_sessionmaker(engine, expire_on_commit=False)


class TestAssistantAdminOrgScoping:
    """End-to-end org-scoping regression for #1136."""

    def _make_app(self, org_id: str | None, admin_id: str, role: str = "admin"):
        from fastapi import FastAPI

        from src.api.auth import TokenData, get_admin_org, get_current_user
        from src.api.db.engine import get_db
        from src.api.routes import assistant as assistant_module

        app = FastAPI()

        async def _override_db():
            async with self._sf() as session:
                yield session

        def _make_current_user():
            return TokenData(
                user_id=admin_id, role=role, raw_claims={"sub": admin_id, "role": role}
            )

        async def _override_admin_org():
            if role == "superadmin":
                return None
            # Look up from DB
            async with self._sf() as session:
                from src.api.db.repositories.users import UserRepository

                repo = UserRepository(session)
                user = await repo.get_by_id(admin_id)
                return user.org_id if user else None

        app.dependency_overrides[get_db] = _override_db
        app.dependency_overrides[get_current_user] = _make_current_user
        app.dependency_overrides[get_admin_org] = _override_admin_org
        app.include_router(assistant_module.router)
        return app

    def _seed_db(self, sf, admin_org_id: str | None, admin_role: str = "admin"):
        """Create org and admin user in the DB."""

        async def _run():
            async with sf() as session:
                org_id = admin_org_id or _uid()
                org = Organization(id=org_id, name="Test Org", slug="test-org")
                session.add(org)
                await session.commit()

                self._admin_id = _uid()
                user = User(
                    id=self._admin_id,
                    username="test_admin",
                    email="admin@example.com",
                    password_hash="x",
                    role=admin_role,
                    org_id=org_id if admin_role != "superadmin" else None,
                )
                session.add(user)
                await session.commit()

        asyncio.run(_run())

    def test_superadmin_can_list_chats(self, sf):
        self._sf = sf
        self._seed_db(sf, None, "superadmin")
        app = self._make_app(None, self._admin_id, "superadmin")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/assistant/chats")
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data

    def test_regular_admin_list_chats_returns_403(self, sf):
        self._sf = sf
        org_id = _uid()
        self._seed_db(sf, org_id, "admin")
        app = self._make_app(org_id, self._admin_id, "admin")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/assistant/chats")
        assert resp.status_code == 403
        detail = resp.json()["detail"]
        assert detail["code"] == "ORG_SCOPING_NOT_AVAILABLE"

    def test_superadmin_can_list_scheduled(self, sf):
        self._sf = sf
        self._seed_db(sf, None, "superadmin")
        app = self._make_app(None, self._admin_id, "superadmin")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/assistant/scheduled")
        assert resp.status_code == 200

    def test_regular_admin_list_scheduled_returns_403(self, sf):
        self._sf = sf
        org_id = _uid()
        self._seed_db(sf, org_id, "admin")
        app = self._make_app(org_id, self._admin_id, "admin")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/assistant/scheduled")
        assert resp.status_code == 403
        assert resp.json()["detail"]["code"] == "ORG_SCOPING_NOT_AVAILABLE"

    def test_superadmin_can_list_deferred(self, sf):
        self._sf = sf
        self._seed_db(sf, None, "superadmin")
        app = self._make_app(None, self._admin_id, "superadmin")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/assistant/deferred")
        assert resp.status_code == 200

    def test_regular_admin_list_deferred_returns_403(self, sf):
        self._sf = sf
        org_id = _uid()
        self._seed_db(sf, org_id, "admin")
        app = self._make_app(org_id, self._admin_id, "admin")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/assistant/deferred")
        assert resp.status_code == 403
        assert resp.json()["detail"]["code"] == "ORG_SCOPING_NOT_AVAILABLE"

    def test_superadmin_can_list_contacts(self, sf):
        self._sf = sf
        self._seed_db(sf, None, "superadmin")
        app = self._make_app(None, self._admin_id, "superadmin")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/assistant/contacts")
        assert resp.status_code == 200

    def test_regular_admin_list_contacts_returns_403(self, sf):
        self._sf = sf
        org_id = _uid()
        self._seed_db(sf, org_id, "admin")
        app = self._make_app(org_id, self._admin_id, "admin")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/assistant/contacts")
        assert resp.status_code == 403
        assert resp.json()["detail"]["code"] == "ORG_SCOPING_NOT_AVAILABLE"

    def test_superadmin_can_list_guardrails(self, sf):
        self._sf = sf
        self._seed_db(sf, None, "superadmin")
        app = self._make_app(None, self._admin_id, "superadmin")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/assistant/guardrails")
        assert resp.status_code == 200

    def test_regular_admin_list_guardrails_returns_403(self, sf):
        self._sf = sf
        org_id = _uid()
        self._seed_db(sf, org_id, "admin")
        app = self._make_app(org_id, self._admin_id, "admin")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/assistant/guardrails")
        assert resp.status_code == 403
        assert resp.json()["detail"]["code"] == "ORG_SCOPING_NOT_AVAILABLE"

    def test_superadmin_can_list_knowledge(self, sf):
        self._sf = sf
        self._seed_db(sf, None, "superadmin")
        app = self._make_app(None, self._admin_id, "superadmin")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/assistant/knowledge")
        assert resp.status_code == 200

    def test_regular_admin_list_knowledge_returns_403(self, sf):
        self._sf = sf
        org_id = _uid()
        self._seed_db(sf, org_id, "admin")
        app = self._make_app(org_id, self._admin_id, "admin")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/assistant/knowledge")
        assert resp.status_code == 403
        assert resp.json()["detail"]["code"] == "ORG_SCOPING_NOT_AVAILABLE"

    def test_superadmin_can_search_knowledge(self, sf):
        self._sf = sf
        self._seed_db(sf, None, "superadmin")
        app = self._make_app(None, self._admin_id, "superadmin")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/assistant/knowledge/search", json={"query": "test", "top_k": 5})
        assert resp.status_code == 200

    def test_regular_admin_search_knowledge_returns_403(self, sf):
        self._sf = sf
        org_id = _uid()
        self._seed_db(sf, org_id, "admin")
        app = self._make_app(org_id, self._admin_id, "admin")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/assistant/knowledge/search", json={"query": "test", "top_k": 5})
        assert resp.status_code == 403
        assert resp.json()["detail"]["code"] == "ORG_SCOPING_NOT_AVAILABLE"

    def test_superadmin_can_list_campaigns(self, sf):
        self._sf = sf
        self._seed_db(sf, None, "superadmin")
        app = self._make_app(None, self._admin_id, "superadmin")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/assistant/campaigns")
        # 409 means the org-scoping gate passed (superadmin) but the campaign
        # manager is not available in the test environment. That is sufficient
        # to prove the endpoint is reachable for superadmins.
        assert resp.status_code in (200, 409)

    def test_regular_admin_list_campaigns_returns_403(self, sf):
        self._sf = sf
        org_id = _uid()
        self._seed_db(sf, org_id, "admin")
        app = self._make_app(org_id, self._admin_id, "admin")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/assistant/campaigns")
        assert resp.status_code == 403
        assert resp.json()["detail"]["code"] == "ORG_SCOPING_NOT_AVAILABLE"

    def test_non_admin_user_blocked_by_require_admin(self, sf):
        """Non-admin users should still be blocked by require_admin before org scoping runs."""
        self._sf = sf
        org_id = _uid()

        async def _run():
            async with sf() as session:
                org = Organization(id=org_id, name="Test Org", slug="test-org")
                session.add(org)
                await session.commit()

                user_id = _uid()
                user = User(
                    id=user_id,
                    username="regular_user",
                    email="user@example.com",
                    password_hash="x",
                    role="user",
                    org_id=org_id,
                )
                session.add(user)
                await session.commit()

                app = self._make_app(org_id, user_id, "user")
                client = TestClient(app, raise_server_exceptions=False)
                resp = client.get("/assistant/chats")
                assert resp.status_code == 403
                assert resp.json()["detail"]["code"] == "FORBIDDEN"

        asyncio.run(_run())


class TestGetAdminOrgReal:
    """Tests that exercise the real get_admin_org dependency (no overrides)."""

    def test_superadmin_returns_none(self, sf):
        async def _run():
            from src.api.auth import TokenData, get_admin_org

            async with sf() as session:
                user_id = _uid()
                current_user = TokenData(
                    user_id=user_id,
                    role="superadmin",
                    raw_claims={"sub": user_id, "role": "superadmin"},
                )
                result = await get_admin_org(current_user, session)
                assert result is None

        asyncio.run(_run())

    def test_regular_admin_with_org_returns_org_id(self, sf):
        async def _run():
            from src.api.auth import TokenData, get_admin_org

            async with sf() as session:
                org_id = _uid()
                org = Organization(id=org_id, name="Test Org", slug="test-org")
                session.add(org)
                await session.commit()

                user_id = _uid()
                user = User(
                    id=user_id,
                    username="test_admin",
                    email="admin@example.com",
                    password_hash="x",
                    role="admin",
                    org_id=org_id,
                )
                session.add(user)
                await session.commit()

                current_user = TokenData(
                    user_id=user_id, role="admin", raw_claims={"sub": user_id, "role": "admin"}
                )
                result = await get_admin_org(current_user, session)
                assert result == org_id

        asyncio.run(_run())

    def test_regular_admin_with_null_org_id_returns_403(self, sf):
        async def _run():
            from src.api.auth import TokenData, get_admin_org

            async with sf() as session:
                org_id = _uid()
                org = Organization(id=org_id, name="Test Org", slug="test-org")
                session.add(org)
                await session.commit()

                user_id = _uid()
                user = User(
                    id=user_id,
                    username="test_admin",
                    email="admin@example.com",
                    password_hash="x",
                    role="admin",
                    org_id=None,
                )
                session.add(user)
                await session.commit()

                current_user = TokenData(
                    user_id=user_id, role="admin", raw_claims={"sub": user_id, "role": "admin"}
                )
                from fastapi import HTTPException

                with pytest.raises(HTTPException) as exc_info:
                    await get_admin_org(current_user, session)
                assert exc_info.value.status_code == 403
                assert exc_info.value.detail["code"] == "ORG_NOT_ASSIGNED"

        asyncio.run(_run())

    def test_non_admin_returns_403(self, sf):
        async def _run():
            from src.api.auth import TokenData, get_admin_org

            async with sf() as session:
                user_id = _uid()
                current_user = TokenData(
                    user_id=user_id, role="user", raw_claims={"sub": user_id, "role": "user"}
                )
                from fastapi import HTTPException

                with pytest.raises(HTTPException) as exc_info:
                    await get_admin_org(current_user, session)
                assert exc_info.value.status_code == 403
                assert exc_info.value.detail["code"] == "FORBIDDEN"

        asyncio.run(_run())
