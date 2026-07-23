"""Tests for Plan model and API (Enterprise Phase 1 — task 1.4.1)."""

from __future__ import annotations

import asyncio
import os
import pathlib
import subprocess
import uuid

import pytest

pytest.importorskip("fastapi")

_TEST_JWT_SECRET = "testsecret_mustbe32chars_minimum00"

from unittest.mock import patch  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker  # noqa: E402

from src.api.auth import create_access_token  # noqa: E402
from src.api.db.engine import get_db  # noqa: E402
from src.api.db.repositories.organization import OrganizationRepository  # noqa: E402
from src.api.db.repositories.plans import PlanRepository  # noqa: E402
from src.api.db.repositories.users import UserRepository  # noqa: E402
from src.api.schemas.plan import PlanLimits  # noqa: E402

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent


def _uid() -> str:
    return str(uuid.uuid4())


def _admin_header(user_id: str) -> dict:
    with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
        token = create_access_token(user_id=user_id, role="admin")
    return {"Authorization": f"Bearer {token}"}


def _user_header(user_id: str) -> dict:
    with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
        token = create_access_token(user_id=user_id, role="user")
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# PlanLimits unit tests
# ---------------------------------------------------------------------------


class TestPlanLimits:
    def test_defaults_are_zero(self):
        lim = PlanLimits()
        assert lim.max_users == 0
        assert lim.max_workspaces == 0

    def test_custom_values(self):
        lim = PlanLimits(max_users=10, max_api_calls_per_month=50000)
        assert lim.max_users == 10
        assert lim.max_api_calls_per_month == 50000


# ---------------------------------------------------------------------------
# PlanRepository unit tests
# ---------------------------------------------------------------------------


class TestPlanRepository:
    def test_create_and_get(self, sf):
        async def _run():
            async with sf() as db:
                repo = PlanRepository(db)
                p = await repo.create(
                    plan_id=_uid(),
                    name="Pro",
                    slug="pro",
                    price_monthly_cents=2900,
                    limits={"max_users": 10},
                )
                await db.commit()
                found = await repo.get_by_id(p.id)
                assert found is not None
                assert found.slug == "pro"
                assert found.price_monthly_cents == 2900

        asyncio.run(_run())

    def test_get_by_slug(self, sf):
        async def _run():
            async with sf() as db:
                repo = PlanRepository(db)
                await repo.create(plan_id=_uid(), name="Free", slug="free")
                await db.commit()
                found = await repo.get_by_slug("free")
                assert found is not None

        asyncio.run(_run())

    def test_list_all(self, sf):
        async def _run():
            async with sf() as db:
                repo = PlanRepository(db)
                await repo.create(plan_id=_uid(), name="A", slug="plan-a")
                await repo.create(plan_id=_uid(), name="B", slug="plan-b")
                await db.commit()
                plans = await repo.list_all()
                assert len(plans) == 2

        asyncio.run(_run())

    def test_update_price(self, sf):
        async def _run():
            async with sf() as db:
                repo = PlanRepository(db)
                pid = _uid()
                await repo.create(plan_id=pid, name="P", slug="plan-p", price_monthly_cents=100)
                await db.commit()
                updated = await repo.update(pid, price_monthly_cents=200)
                await db.commit()
                assert updated.price_monthly_cents == 200

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def plan_setup(engine):
    factory = async_sessionmaker(engine, expire_on_commit=False)
    admin_id = _uid()
    user_id = _uid()
    org_id = _uid()

    async def _seed():
        async with factory() as db:
            org_repo = OrganizationRepository(db)
            user_repo = UserRepository(db)
            await org_repo.create(org_id=org_id, name="Plans Org", slug="plans-org")
            await user_repo.create(
                user_id=admin_id,
                username="admin",
                email="admin@example.com",
                password_hash="h",
                role="admin",
                org_id=org_id,
            )
            await user_repo.create(
                user_id=user_id,
                username="normaluser",
                email="user@example.com",
                password_hash="h",
                org_id=org_id,
            )
            await db.commit()

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
            yield client, admin_id, user_id, org_id

    app.dependency_overrides.clear()


class TestPlanAPI:
    def test_list_plans_empty(self, plan_setup):
        client, _, user_id, __ = plan_setup
        r = client.get("/api/v1/plans", headers=_user_header(user_id))
        assert r.status_code == 200
        assert isinstance(r.json()["data"], list)

    def test_create_plan_admin(self, plan_setup):
        client, admin_id, _, __ = plan_setup
        r = client.post(
            "/api/v1/plans",
            json={"name": "Pro", "slug": "pro-test", "price_monthly_cents": 2900},
            headers=_admin_header(admin_id),
        )
        assert r.status_code == 201
        assert r.json()["data"]["slug"] == "pro-test"

    def test_create_plan_non_admin_returns_403(self, plan_setup):
        client, _, user_id, __ = plan_setup
        r = client.post(
            "/api/v1/plans",
            json={"name": "X", "slug": "x-test", "price_monthly_cents": 0},
            headers=_user_header(user_id),
        )
        assert r.status_code == 403

    def test_create_duplicate_slug_returns_409(self, plan_setup):
        client, admin_id, _, __ = plan_setup
        client.post(
            "/api/v1/plans",
            json={"name": "Dup", "slug": "dup-plan"},
            headers=_admin_header(admin_id),
        )
        r = client.post(
            "/api/v1/plans",
            json={"name": "Dup2", "slug": "dup-plan"},
            headers=_admin_header(admin_id),
        )
        assert r.status_code == 409

    def test_get_plan(self, plan_setup):
        client, admin_id, user_id, __ = plan_setup
        r = client.post(
            "/api/v1/plans",
            json={"name": "Getable", "slug": "getable"},
            headers=_admin_header(admin_id),
        )
        plan_id = r.json()["data"]["id"]
        r = client.get(f"/api/v1/plans/{plan_id}", headers=_user_header(user_id))
        assert r.status_code == 200
        assert r.json()["data"]["id"] == plan_id

    def test_update_plan(self, plan_setup):
        client, admin_id, _, __ = plan_setup
        r = client.post(
            "/api/v1/plans",
            json={"name": "Updateable", "slug": "updateable"},
            headers=_admin_header(admin_id),
        )
        plan_id = r.json()["data"]["id"]
        r = client.patch(
            f"/api/v1/plans/{plan_id}",
            json={"price_monthly_cents": 999},
            headers=_admin_header(admin_id),
        )
        assert r.status_code == 200
        assert r.json()["data"]["price_monthly_cents"] == 999

    def test_deactivate_plan(self, plan_setup):
        client, admin_id, user_id, __ = plan_setup
        r = client.post(
            "/api/v1/plans",
            json={"name": "Deactivatable", "slug": "deactivatable"},
            headers=_admin_header(admin_id),
        )
        plan_id = r.json()["data"]["id"]
        r = client.delete(f"/api/v1/plans/{plan_id}", headers=_admin_header(admin_id))
        assert r.status_code == 200
        assert r.json()["data"]["is_active"] is False

    def test_assign_plan_to_org(self, plan_setup):
        client, admin_id, _, org_id = plan_setup
        r = client.post(
            "/api/v1/plans",
            json={"name": "Assignable", "slug": "assignable"},
            headers=_admin_header(admin_id),
        )
        plan_id = r.json()["data"]["id"]
        r = client.patch(
            f"/api/v1/organizations/{org_id}/plan",
            json={"plan_id": plan_id},
            headers=_admin_header(admin_id),
        )
        assert r.status_code == 200
        assert r.json()["data"]["plan_id"] == plan_id

    def test_assign_plan_cross_org_denied(self, plan_setup):
        client, admin_id, _, org_id = plan_setup
        r = client.post(
            "/api/v1/plans",
            json={"name": "Assignable", "slug": "assignable"},
            headers=_admin_header(admin_id),
        )
        plan_id = r.json()["data"]["id"]
        # Try to assign plan to a different org (non-existent, but should be denied before lookup)
        r = client.patch(
            f"/api/v1/organizations/{_uid()}/plan",
            json={"plan_id": plan_id},
            headers=_admin_header(admin_id),
        )
        assert r.status_code == 403
        # Response structure uses 'error' key for errors
        resp_json = r.json()
        assert resp_json.get("error") is not None
        assert resp_json["error"].get("code") == "CROSS_ORG_ACCESS"


# ---------------------------------------------------------------------------
# Migration round-trip
# ---------------------------------------------------------------------------


class TestMigration0008:
    def test_upgrade_and_downgrade(self):
        db_path = _PROJECT_ROOT / "data" / "api" / "cogtrix_0008_test.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["COGTRIX_DB_URL"] = f"sqlite+aiosqlite:///{db_path}"
        result = subprocess.run(
            ["uv", "run", "alembic", "upgrade", "head"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(_PROJECT_ROOT),
        )
        assert result.returncode == 0, f"upgrade failed:\n{result.stderr}"
        # Downgrade to 0007 (removes plans table and plan_id from orgs)
        result = subprocess.run(
            ["uv", "run", "alembic", "downgrade", "0007"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(_PROJECT_ROOT),
        )
        assert result.returncode == 0, f"downgrade failed:\n{result.stderr}"
        db_path.unlink(missing_ok=True)
