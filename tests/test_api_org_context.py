"""Tests for org context dependencies (Enterprise Phase 1 — task 1.1.3).

Covers:
  - get_org_context: returns OrgContext(org_id=None) for unassigned users
  - get_org_context: returns populated OrgContext for org-assigned users
  - require_org_context: passes through when org is set
  - require_org_context: raises 403 ORG_REQUIRED when no org assigned
  - OrgContext.has_org property
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

from unittest.mock import patch  # noqa: E402

from fastapi import Depends  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from src.api.auth import create_access_token  # noqa: E402
from src.api.db.engine import Base, get_db  # noqa: E402
from src.api.db.repositories.organization import OrganizationRepository  # noqa: E402
from src.api.db.repositories.users import UserRepository  # noqa: E402
from src.api.org_context import OrgContext, get_org_context, require_org_context  # noqa: E402

# ---------------------------------------------------------------------------
# App fixture with test routes
# ---------------------------------------------------------------------------


def _make_test_app(factory):
    """Create a minimal FastAPI app with org-context test routes."""
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI()

    async def _override_db():
        async with factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db] = _override_db

    @app.get("/optional-org")
    async def optional_org(ctx: OrgContext = Depends(get_org_context)):  # noqa: B008
        return JSONResponse(
            {
                "user_id": ctx.user_id,
                "org_id": ctx.org_id,
                "has_org": ctx.has_org,
            }
        )

    @app.get("/required-org")
    async def required_org(ctx: OrgContext = Depends(require_org_context)):  # noqa: B008
        return JSONResponse({"org_id": ctx.org_id})

    return app


@pytest.fixture()
def client_and_ids():
    """Return (TestClient, user_id, org_id) with a user already seeded in DB."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)

    user_id = str(uuid.uuid4())
    org_id = str(uuid.uuid4())

    async def _seed():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with factory() as session:
            org_repo = OrganizationRepository(session)
            user_repo = UserRepository(session)
            await org_repo.create(org_id=org_id, name="Test Org", slug="test-org")
            await user_repo.create(
                user_id=user_id,
                username="testuser",
                email="testuser@example.com",
                password_hash="hash",
                org_id=org_id,
            )
            await session.commit()

    asyncio.run(_seed())

    with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
        app = _make_test_app(factory)
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c, user_id, org_id

    asyncio.run(engine.dispose())


@pytest.fixture()
def client_no_org():
    """Return (TestClient, user_id) with a user NOT assigned to any org."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)

    user_id = str(uuid.uuid4())

    async def _seed():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with factory() as session:
            user_repo = UserRepository(session)
            await user_repo.create(
                user_id=user_id,
                username="noorguser",
                email="noorg@example.com",
                password_hash="hash",
            )
            await session.commit()

    asyncio.run(_seed())

    with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
        app = _make_test_app(factory)
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c, user_id

    asyncio.run(engine.dispose())


def _auth_header(user_id: str) -> dict:
    with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
        token = create_access_token(user_id=user_id, role="user")
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# OrgContext unit tests
# ---------------------------------------------------------------------------


class TestOrgContext:
    def test_has_org_true_when_org_id_set(self):
        ctx = OrgContext(user_id="u1", org_id="o1")
        assert ctx.has_org is True

    def test_has_org_false_when_org_id_none(self):
        ctx = OrgContext(user_id="u1", org_id=None)
        assert ctx.has_org is False

    def test_defaults(self):
        ctx = OrgContext(user_id="u1")
        assert ctx.org_id is None
        assert ctx.org is None
        assert ctx.has_org is False


# ---------------------------------------------------------------------------
# get_org_context dependency
# ---------------------------------------------------------------------------


class TestGetOrgContext:
    def test_returns_org_id_for_assigned_user(self, client_and_ids):
        client, user_id, org_id = client_and_ids
        r = client.get("/optional-org", headers=_auth_header(user_id))
        assert r.status_code == 200
        data = r.json()
        assert data["org_id"] == org_id
        assert data["has_org"] is True
        assert data["user_id"] == user_id

    def test_returns_none_for_unassigned_user(self, client_no_org):
        client, user_id = client_no_org
        r = client.get("/optional-org", headers=_auth_header(user_id))
        assert r.status_code == 200
        data = r.json()
        assert data["org_id"] is None
        assert data["has_org"] is False

    def test_requires_auth(self, client_and_ids):
        client, _, _ = client_and_ids
        r = client.get("/optional-org")
        assert r.status_code == 401

    def test_returns_none_for_unknown_user_id(self, client_and_ids):
        """JWT for a user not in DB → OrgContext with no org."""
        client, _, _ = client_and_ids
        unknown_id = str(uuid.uuid4())
        r = client.get("/optional-org", headers=_auth_header(unknown_id))
        assert r.status_code == 200
        assert r.json()["org_id"] is None


# ---------------------------------------------------------------------------
# require_org_context dependency
# ---------------------------------------------------------------------------


class TestRequireOrgContext:
    def test_passes_through_for_assigned_user(self, client_and_ids):
        client, user_id, org_id = client_and_ids
        r = client.get("/required-org", headers=_auth_header(user_id))
        assert r.status_code == 200
        assert r.json()["org_id"] == org_id

    def test_raises_403_for_unassigned_user(self, client_no_org):
        client, user_id = client_no_org
        r = client.get("/required-org", headers=_auth_header(user_id))
        assert r.status_code == 403
        detail = r.json()
        assert detail["detail"]["code"] == "ORG_REQUIRED"

    def test_requires_auth(self, client_and_ids):
        client, _, _ = client_and_ids
        r = client.get("/required-org")
        assert r.status_code == 401
