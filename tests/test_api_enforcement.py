"""Tests for plan enforcement (Enterprise Phase 1 — task 1.4.4)."""

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
from cogtrix_core.api.db.repositories.plans import PlanRepository  # noqa: E402
from cogtrix_core.api.db.repositories.users import UserRepository  # noqa: E402
from cogtrix_core.api.plan_enforcement import (  # noqa: E402
    PlanLimitSnapshot,
    get_plan_limit_snapshot,
)


def _uid() -> str:
    return str(uuid.uuid4())


def _auth(user_id: str, role: str = "admin") -> dict:
    with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
        token = create_access_token(user_id=user_id, role=role)
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# PlanLimitSnapshot unit tests
# ---------------------------------------------------------------------------


class TestPlanLimitSnapshot:
    def test_within_limit_unlimited(self):
        snap = PlanLimitSnapshot("free", 0, 0, 0, 0, 5, 3, 1000)
        assert snap.within_limit(0, 999999) is True

    def test_within_limit_at_cap(self):
        snap = PlanLimitSnapshot("pro", 10, 5, 50000, 20, 10, 3, 1000)
        assert snap.within_limit(10, 10) is False

    def test_within_limit_below_cap(self):
        snap = PlanLimitSnapshot("pro", 10, 5, 50000, 20, 9, 3, 1000)
        assert snap.within_limit(10, 9) is True

    def test_can_add_user_unlimited(self):
        snap = PlanLimitSnapshot("enterprise", 0, 0, 0, 0, 500, 20, 10000)
        assert snap.can_add_user is True

    def test_can_add_user_at_cap(self):
        snap = PlanLimitSnapshot("pro", 10, 5, 50000, 20, 10, 3, 1000)
        assert snap.can_add_user is False

    def test_can_add_workspace_at_cap(self):
        snap = PlanLimitSnapshot("pro", 10, 5, 50000, 20, 5, 5, 1000)
        assert snap.can_add_workspace is False

    def test_can_make_api_call_at_cap(self):
        snap = PlanLimitSnapshot("pro", 10, 5, 50000, 20, 5, 3, 50000)
        assert snap.can_make_api_call is False


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def enf_setup(engine):
    factory = async_sessionmaker(engine, expire_on_commit=False)
    org_id = _uid()
    admin_id = _uid()
    plan_id = _uid()

    async def _seed():
        async with factory() as db:
            org_repo = OrganizationRepository(db)
            user_repo = UserRepository(db)
            plan_repo = PlanRepository(db)
            # Create a plan with tight limits for testing.
            plan = await plan_repo.create(
                plan_id=plan_id,
                name="Test Plan",
                slug="test-plan",
                limits={"max_users": 3, "max_workspaces": 2, "max_api_calls_per_month": 100},
            )
            await db.commit()
            org = await org_repo.create(org_id=org_id, name="Enf Org", slug="enf-org")
            org.plan_id = plan.id
            org.plan = plan.slug
            await db.commit()
            await user_repo.create(
                user_id=admin_id,
                username="admin",
                email="admin@example.com",
                password_hash="h",
                role="admin",
                org_id=org_id,
            )
            await db.commit()

    asyncio.run(_seed())

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
            yield client, org_id, admin_id, plan_id, factory

    app.dependency_overrides.clear()


class TestEnforcementStatus:
    def test_returns_snapshot(self, enf_setup):
        client, _, admin_id, __, ___ = enf_setup
        r = client.get("/api/v1/enforcement/status", headers=_auth(admin_id))
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["plan"] == "test-plan"
        assert data["limits"]["users"] == 3
        assert data["limits"]["workspaces"] == 2
        assert data["limits"]["api_calls_per_month"] == 100
        assert data["usage"]["users"] == 1  # admin seeded
        assert data["headroom"]["can_add_user"] is True
        assert data["headroom"]["can_add_workspace"] is True

    def test_requires_auth(self, enf_setup):
        client, *_ = enf_setup
        r = client.get("/api/v1/enforcement/status")
        assert r.status_code == 401


class TestGetPlanLimitSnapshot:
    def test_no_plan_returns_unlimited(self, enf_setup):
        _, org_id, _, __, factory = enf_setup

        async def _run():
            async with factory() as db:
                # Create an org with no plan.
                org_repo = OrganizationRepository(db)
                no_plan_org_id = _uid()
                await org_repo.create(org_id=no_plan_org_id, name="No Plan Org", slug="no-plan-org")
                await db.commit()
                snap = await get_plan_limit_snapshot(no_plan_org_id, db)
                assert snap.max_users == 0
                assert snap.max_workspaces == 0
                assert snap.can_add_user is True
                assert snap.can_add_workspace is True

        asyncio.run(_run())

    def test_at_user_cap_blocks_add(self, enf_setup):
        _, org_id, admin_id, __, factory = enf_setup

        async def _run():
            async with factory() as db:
                user_repo = UserRepository(db)
                # Add 2 more users to hit the cap of 3.
                for i in range(2):
                    await user_repo.create(
                        user_id=_uid(),
                        username=f"extra{i}",
                        email=f"extra{i}@example.com",
                        password_hash="h",
                        org_id=org_id,
                    )
                await db.commit()
                snap = await get_plan_limit_snapshot(org_id, db)
                assert snap.current_users == 3
                assert snap.can_add_user is False

        asyncio.run(_run())
