"""Tests for organizations member role endpoint.

Covers:
    PUT /api/v1/organizations/{org_id}/members/{user_id}/role

Auth permutations: unauthenticated, non-admin, admin.
Edge cases: self-demotion guard, cross-org 404, audit logging.
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

import jwt as jose_jwt  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from cogtrix_core.api.db.engine import Base, get_db  # noqa: E402
from cogtrix_core.api.db.models import Organization  # noqa: E402
from cogtrix_core.api.db.repositories.users import UserRepository  # noqa: E402

_VALID_PASSWORD = "TestPass1!"


def _uid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


def _seed_user(sf, username, email, role, org_id):
    async def _run():
        async with sf() as db:
            repo = UserRepository(db)
            from cogtrix_core.api.auth import hash_password

            user = await repo.create(
                user_id=_uid(),
                username=username,
                email=email,
                password_hash=hash_password(_VALID_PASSWORD),
                role=role,
                org_id=org_id,
            )
            await db.commit()
            await db.refresh(user)
            return user

    return asyncio.run(_run())


def _seed_org(sf, name, slug):
    async def _run():
        async with sf() as db:
            org = Organization(name=name, slug=slug)
            db.add(org)
            await db.commit()
            await db.refresh(org)
            return org

    return asyncio.run(_run())


def _make_token(user_id, role="admin"):
    return jose_jwt.encode(
        {"sub": user_id, "role": role, "exp": 9999999999},
        _TEST_JWT_SECRET,
        algorithm="HS256",
    )


# ---------------------------------------------------------------------------
# App & Client
# ---------------------------------------------------------------------------


def _make_app(sf, user_id, org_id, role="admin", *, bypass_auth=False):
    """Create a FastAPI app with configurable auth override.

    Set *bypass_auth=True* to skip auth overrides (real token validation will run).
    Set *role='user'* to test non-admin rejection.
    """
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse

    from cogtrix_core.api.auth import TokenData, get_current_user
    from cogtrix_core.api.org_context import OrgContext, get_org_context
    from cogtrix_core.api.routes import organizations as org_module
    from cogtrix_core.api.schemas.common import APIError, APIResponse

    app = FastAPI()

    # Register Cogtrix error envelope handler so HTTPException responses
    # use the standard {"error": {"code": "...", "message": "..."}} shape.
    async def _http_error_handler(request: Request, exc: Exception):
        http_exc: HTTPException = exc  # type: ignore[assignment]
        detail = http_exc.detail
        if isinstance(detail, dict):
            code = detail.get("code", "INTERNAL_ERROR")
            message = detail.get("message", str(detail))
        else:
            code = "INTERNAL_ERROR"
            message = str(detail) if detail else str(http_exc.status_code)
        envelope = APIResponse(data=None, error=APIError(code=code, message=message))
        return JSONResponse(
            status_code=http_exc.status_code,
            content=envelope.model_dump(mode="json"),
        )

    app.add_exception_handler(HTTPException, _http_error_handler)  # type: ignore[arg-type]

    async def _override_db():
        async with sf() as session:
            yield session

    app.dependency_overrides[get_db] = _override_db

    if not bypass_auth:

        def _override_current_user():
            return TokenData(
                user_id=user_id,
                role=role,
                raw_claims={"sub": user_id, "role": role},
            )

        def _override_org_context():
            return OrgContext(
                user_id=user_id,
                role=role,
                org_id=org_id,
                org=None,
            )

        app.dependency_overrides[get_current_user] = _override_current_user
        app.dependency_overrides[get_org_context] = _override_org_context

    app.include_router(org_module.router, prefix="/api/v1")
    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestUpdateMemberRole:
    def test_admin_can_update_member_role(self, sf):
        org = _seed_org(sf, "Test Org", "test-org")
        admin = _seed_user(sf, "admin_user", "admin@test.org", "admin", org.id)
        member = _seed_user(sf, "reg_user", "reg@test.org", "user", org.id)

        app = _make_app(sf, admin.id, org.id)
        with TestClient(app, raise_server_exceptions=False) as client:
            token = _make_token(admin.id, "admin")
            resp = client.put(
                f"/api/v1/organizations/{org.id}/members/{member.id}/role",
                headers={"Authorization": f"Bearer {token}"},
                json={"role": "admin"},
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["error"] is None
        assert body["data"]["role"] == "admin"
        assert body["data"]["id"] == member.id

    def test_role_update_persists_in_db(self, sf):
        org = _seed_org(sf, "Test Org", "test-org")
        admin = _seed_user(sf, "admin_user", "admin@test.org", "admin", org.id)
        member = _seed_user(sf, "reg_user", "reg@test.org", "user", org.id)

        app = _make_app(sf, admin.id, org.id)
        with TestClient(app, raise_server_exceptions=False) as client:
            token = _make_token(admin.id, "admin")
            resp = client.put(
                f"/api/v1/organizations/{org.id}/members/{member.id}/role",
                headers={"Authorization": f"Bearer {token}"},
                json={"role": "admin"},
            )
        assert resp.status_code == 200

        # Verify in DB
        async def _check():
            async with sf() as db:
                repo = UserRepository(db)
                user = await repo.get_by_id(member.id)
                assert user is not None
                assert user.role == "admin"

        asyncio.run(_check())

    def test_unauthenticated_returns_401(self, sf):
        org = _seed_org(sf, "Test Org", "test-org")
        admin = _seed_user(sf, "admin_user", "admin@test.org", "admin", org.id)
        member = _seed_user(sf, "reg_user", "reg@test.org", "user", org.id)

        app = _make_app(sf, admin.id, org.id, bypass_auth=True)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.put(
                f"/api/v1/organizations/{org.id}/members/{member.id}/role",
                json={"role": "admin"},
            )
        assert resp.status_code == 401

    def test_non_admin_returns_403(self, sf):
        org = _seed_org(sf, "Test Org", "test-org")
        member = _seed_user(sf, "reg_user", "reg@test.org", "user", org.id)

        app = _make_app(sf, member.id, org.id, role="user")
        with TestClient(app, raise_server_exceptions=False) as client:
            token = _make_token(member.id, "user")
            resp = client.put(
                f"/api/v1/organizations/{org.id}/members/{member.id}/role",
                headers={"Authorization": f"Bearer {token}"},
                json={"role": "admin"},
            )
        assert resp.status_code == 403, resp.text
        assert resp.json()["error"]["code"] == "FORBIDDEN"

    def test_user_not_in_org_returns_404(self, sf):
        org = _seed_org(sf, "Test Org", "test-org")
        other_org = _seed_org(sf, "Other Org", "other-org")
        admin = _seed_user(sf, "admin_user", "admin@test.org", "admin", org.id)
        outsider = _seed_user(sf, "outsider", "out@test.org", "user", other_org.id)

        app = _make_app(sf, admin.id, org.id)
        with TestClient(app, raise_server_exceptions=False) as client:
            token = _make_token(admin.id, "admin")
            resp = client.put(
                f"/api/v1/organizations/{org.id}/members/{outsider.id}/role",
                headers={"Authorization": f"Bearer {token}"},
                json={"role": "admin"},
            )
        assert resp.status_code == 404, resp.text
        assert resp.json()["error"]["code"] == "NOT_FOUND"

    def test_self_demotion_returns_400(self, sf):
        org = _seed_org(sf, "Test Org", "test-org")
        admin = _seed_user(sf, "admin_user", "admin@test.org", "admin", org.id)

        app = _make_app(sf, admin.id, org.id)
        with TestClient(app, raise_server_exceptions=False) as client:
            token = _make_token(admin.id, "admin")
            resp = client.put(
                f"/api/v1/organizations/{org.id}/members/{admin.id}/role",
                headers={"Authorization": f"Bearer {token}"},
                json={"role": "user"},
            )
        assert resp.status_code == 400, resp.text
        assert resp.json()["error"]["code"] == "BAD_REQUEST"

    def test_invalid_role_returns_422(self, sf):
        org = _seed_org(sf, "Test Org", "test-org")
        admin = _seed_user(sf, "admin_user", "admin@test.org", "admin", org.id)
        member = _seed_user(sf, "reg_user", "reg@test.org", "user", org.id)

        app = _make_app(sf, admin.id, org.id)
        with TestClient(app, raise_server_exceptions=False) as client:
            token = _make_token(admin.id, "admin")
            resp = client.put(
                f"/api/v1/organizations/{org.id}/members/{member.id}/role",
                headers={"Authorization": f"Bearer {token}"},
                json={"role": "superadmin"},
            )
        assert resp.status_code == 422, resp.text

    def test_missing_role_returns_200_with_noop(self, sf):
        org = _seed_org(sf, "Test Org", "test-org")
        admin = _seed_user(sf, "admin_user", "admin@test.org", "admin", org.id)
        member = _seed_user(sf, "reg_user", "reg@test.org", "user", org.id)

        app = _make_app(sf, admin.id, org.id)
        with TestClient(app, raise_server_exceptions=False) as client:
            token = _make_token(admin.id, "admin")
            resp = client.put(
                f"/api/v1/organizations/{org.id}/members/{member.id}/role",
                headers={"Authorization": f"Bearer {token}"},
                json={},
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["data"]["role"] == "user"

    def test_demote_user_to_user_role(self, sf):
        org = _seed_org(sf, "Test Org", "test-org")
        admin = _seed_user(sf, "admin_user", "admin@test.org", "admin", org.id)
        member = _seed_user(sf, "admin2", "admin2@test.org", "admin", org.id)

        app = _make_app(sf, admin.id, org.id)
        with TestClient(app, raise_server_exceptions=False) as client:
            token = _make_token(admin.id, "admin")
            resp = client.put(
                f"/api/v1/organizations/{org.id}/members/{member.id}/role",
                headers={"Authorization": f"Bearer {token}"},
                json={"role": "user"},
            )
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"]["role"] == "user"
