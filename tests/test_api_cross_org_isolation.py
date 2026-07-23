"""Tests for cross-org isolation (Enterprise Phase 1 — task 1.1.4).

Verifies that users cannot access resources belonging to a different
organization via the assert_same_org guard and the updated OrgContext.

Covers:
  - assert_same_org: passes when org IDs match
  - assert_same_org: raises 403 CROSS_ORG_ACCESS when org IDs differ
  - assert_same_org: unassigned caller can only access unscoped resources
  - assert_same_org: admin bypass (default True)
  - assert_same_org: admin bypass disabled (admin_bypass=False)
  - OrgContext.is_admin property
  - OrgContext.role field propagated from JWT
  - Integration: user from org A cannot retrieve org B's resources
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

pytest.importorskip("fastapi")

_TEST_JWT_SECRET = "testsecret_mustbe32chars_minimum00"

from unittest.mock import patch  # noqa: E402

from fastapi import Depends, HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker  # noqa: E402

from src.api.auth import create_access_token  # noqa: E402
from src.api.db.engine import get_db  # noqa: E402
from src.api.db.repositories.organization import OrganizationRepository  # noqa: E402
from src.api.db.repositories.users import UserRepository  # noqa: E402
from src.api.org_context import (  # noqa: E402
    OrgContext,
    assert_same_org,
    get_org_context,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uid() -> str:
    return str(uuid.uuid4())


def _auth_header(user_id: str, role: str = "user") -> dict:
    with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
        token = create_access_token(user_id=user_id, role=role)
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# assert_same_org unit tests (no DB needed)
# ---------------------------------------------------------------------------


class TestAssertSameOrg:
    def test_same_org_passes(self):
        ctx = OrgContext(user_id="u1", org_id="org-a")
        assert_same_org(ctx, "org-a")  # must not raise

    def test_different_org_raises_403(self):
        ctx = OrgContext(user_id="u1", org_id="org-a")
        with pytest.raises(HTTPException) as exc_info:
            assert_same_org(ctx, "org-b")
        assert exc_info.value.status_code == 403
        assert exc_info.value.detail["code"] == "CROSS_ORG_ACCESS"

    def test_unassigned_caller_can_access_unscoped_resource(self):
        ctx = OrgContext(user_id="u1", org_id=None)
        assert_same_org(ctx, None)  # must not raise

    def test_unassigned_caller_cannot_access_org_resource(self):
        ctx = OrgContext(user_id="u1", org_id=None)
        with pytest.raises(HTTPException) as exc_info:
            assert_same_org(ctx, "org-a")
        assert exc_info.value.status_code == 403

    def test_org_caller_cannot_access_unscoped_resource(self):
        ctx = OrgContext(user_id="u1", org_id="org-a")
        with pytest.raises(HTTPException) as exc_info:
            assert_same_org(ctx, None)
        assert exc_info.value.status_code == 403

    def test_admin_bypass_default_true(self):
        ctx = OrgContext(user_id="u1", role="admin", org_id="org-a")
        assert_same_org(ctx, "org-b")  # admin bypasses check — must not raise

    def test_admin_bypass_false_enforces_check(self):
        ctx = OrgContext(user_id="u1", role="admin", org_id="org-a")
        with pytest.raises(HTTPException) as exc_info:
            assert_same_org(ctx, "org-b", admin_bypass=False)
        assert exc_info.value.status_code == 403

    def test_admin_same_org_always_passes(self):
        ctx = OrgContext(user_id="u1", role="admin", org_id="org-a")
        assert_same_org(ctx, "org-a", admin_bypass=False)  # same org — no raise

    def test_non_admin_same_org_passes(self):
        ctx = OrgContext(user_id="u1", role="user", org_id="org-a")
        assert_same_org(ctx, "org-a")  # must not raise


# ---------------------------------------------------------------------------
# OrgContext properties
# ---------------------------------------------------------------------------


class TestOrgContextProperties:
    def test_is_admin_true_for_admin_role(self):
        ctx = OrgContext(user_id="u1", role="admin", org_id="org-a")
        assert ctx.is_admin is True

    def test_is_admin_false_for_user_role(self):
        ctx = OrgContext(user_id="u1", role="user", org_id="org-a")
        assert ctx.is_admin is False

    def test_role_defaults_to_user(self):
        ctx = OrgContext(user_id="u1")
        assert ctx.role == "user"
        assert ctx.is_admin is False


# ---------------------------------------------------------------------------
# Integration: two-org isolation via TestClient
# ---------------------------------------------------------------------------


@pytest.fixture()
def two_org_setup(engine):
    """Seed two orgs (A and B) each with one user. Return TestClient + IDs."""
    factory = async_sessionmaker(engine, expire_on_commit=False)

    org_a_id = _uid()
    org_b_id = _uid()
    user_a_id = _uid()
    user_b_id = _uid()
    user_admin_id = _uid()

    async def _seed():
        async with factory() as session:
            org_repo = OrganizationRepository(session)
            user_repo = UserRepository(session)
            await org_repo.create(org_id=org_a_id, name="Org A", slug="org-a")
            await org_repo.create(org_id=org_b_id, name="Org B", slug="org-b")
            await user_repo.create(
                user_id=user_a_id,
                username="user_a",
                email="a@example.com",
                password_hash="h",
                org_id=org_a_id,
            )
            await user_repo.create(
                user_id=user_b_id,
                username="user_b",
                email="b@example.com",
                password_hash="h",
                org_id=org_b_id,
            )
            await user_repo.create(
                user_id=user_admin_id,
                username="admin_user",
                email="admin@example.com",
                password_hash="h",
                org_id=org_a_id,
                role="admin",
            )
            await session.commit()

    asyncio.run(_seed())

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

    # Simulate a route that returns a "resource" belonging to org_a_id.
    # The route uses assert_same_org to block cross-org access.
    @app.get("/resource/{resource_org_id}")
    async def get_resource(  # noqa: B008
        resource_org_id: str,
        ctx: OrgContext = Depends(get_org_context),  # noqa: B008
    ):
        assert_same_org(ctx, resource_org_id)
        return JSONResponse({"org_id": resource_org_id, "caller_org": ctx.org_id})

    with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, org_a_id, org_b_id, user_a_id, user_b_id, user_admin_id

    app.dependency_overrides.clear()


class TestCrossOrgIsolationIntegration:
    def test_user_a_can_access_own_org_resource(self, two_org_setup):
        client, org_a, org_b, user_a, user_b, admin = two_org_setup
        r = client.get(f"/resource/{org_a}", headers=_auth_header(user_a))
        assert r.status_code == 200
        assert r.json()["org_id"] == org_a

    def test_user_a_cannot_access_org_b_resource(self, two_org_setup):
        client, org_a, org_b, user_a, user_b, admin = two_org_setup
        r = client.get(f"/resource/{org_b}", headers=_auth_header(user_a))
        assert r.status_code == 403
        assert r.json()["detail"]["code"] == "CROSS_ORG_ACCESS"

    def test_user_b_cannot_access_org_a_resource(self, two_org_setup):
        client, org_a, org_b, user_a, user_b, admin = two_org_setup
        r = client.get(f"/resource/{org_a}", headers=_auth_header(user_b))
        assert r.status_code == 403

    def test_admin_can_access_org_b_resource_by_default(self, two_org_setup):
        client, org_a, org_b, user_a, user_b, admin = two_org_setup
        r = client.get(f"/resource/{org_b}", headers=_auth_header(admin, role="admin"))
        assert r.status_code == 200

    def test_unauthenticated_request_rejected(self, two_org_setup):
        client, org_a, org_b, user_a, user_b, admin = two_org_setup
        r = client.get(f"/resource/{org_a}")
        assert r.status_code == 401
