"""Tests for usage metering (Enterprise Phase 1 — task 1.4.2)."""

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
from src.api.db.repositories.usage import (  # noqa: E402
    EVENT_API_CALL,
    EVENT_SESSION_CREATED,
    UsageRepository,
)
from src.api.db.repositories.users import UserRepository  # noqa: E402

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
# UsageRepository unit tests
# ---------------------------------------------------------------------------


class TestUsageRepository:
    def test_record_and_count(self, sf):
        org_id = _uid()

        async def _run():
            async with sf() as db:
                org_repo = OrganizationRepository(db)
                await org_repo.create(org_id=org_id, name="Usage Org", slug="usage-org")
                repo = UsageRepository(db)
                await repo.record(org_id=org_id, event_type=EVENT_API_CALL, quantity=5)
                await repo.record(org_id=org_id, event_type=EVENT_API_CALL, quantity=3)
                await repo.record(org_id=org_id, event_type=EVENT_SESSION_CREATED)
                await db.commit()

            async with sf() as db:
                from datetime import UTC, datetime

                now = datetime.now(UTC)
                repo = UsageRepository(db)
                total = await repo.count_for_period(org_id, EVENT_API_CALL, now.year, now.month)
                assert total == 8

        asyncio.run(_run())

    def test_get_monthly_totals(self, sf):
        org_id = _uid()

        async def _run():
            async with sf() as db:
                org_repo = OrganizationRepository(db)
                await org_repo.create(org_id=org_id, name="Totals Org", slug="totals-org")
                repo = UsageRepository(db)
                await repo.record(org_id=org_id, event_type=EVENT_API_CALL, quantity=10)
                await repo.record(org_id=org_id, event_type=EVENT_SESSION_CREATED, quantity=2)
                await db.commit()

            async with sf() as db:
                from datetime import UTC, datetime

                now = datetime.now(UTC)
                repo = UsageRepository(db)
                totals = await repo.get_monthly_totals(org_id, now.year, now.month)
                assert totals[EVENT_API_CALL] == 10
                assert totals[EVENT_SESSION_CREATED] == 2

        asyncio.run(_run())

    def test_get_recent_records_filtered(self, sf):
        org_id = _uid()

        async def _run():
            async with sf() as db:
                org_repo = OrganizationRepository(db)
                await org_repo.create(org_id=org_id, name="Filter Org", slug="filter-org")
                repo = UsageRepository(db)
                await repo.record(org_id=org_id, event_type=EVENT_API_CALL)
                await repo.record(org_id=org_id, event_type=EVENT_SESSION_CREATED)
                await db.commit()

            async with sf() as db:
                repo = UsageRepository(db)
                api_records = await repo.get_recent_records(org_id, event_type=EVENT_API_CALL)
                assert len(api_records) == 1
                assert api_records[0].event_type == EVENT_API_CALL

        asyncio.run(_run())

    def test_empty_org_returns_zero(self, sf):
        async def _run():
            async with sf() as db:
                repo = UsageRepository(db)
                total = await repo.count_for_period(_uid(), EVENT_API_CALL, 2026, 1)
                assert total == 0

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def usage_setup(engine):
    factory = async_sessionmaker(engine, expire_on_commit=False)
    org_id = _uid()
    admin_id = _uid()
    user_id = _uid()

    async def _seed():
        async with factory() as db:
            org_repo = OrganizationRepository(db)
            user_repo = UserRepository(db)
            await org_repo.create(org_id=org_id, name="API Usage Org", slug="api-usage-org")
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
            yield client, org_id, admin_id, user_id

    app.dependency_overrides.clear()


class TestUsageAPI:
    def test_summary_empty(self, usage_setup):
        client, _, admin_id, __ = usage_setup
        r = client.get("/api/v1/usage/summary", headers=_admin_header(admin_id))
        assert r.status_code == 200
        assert r.json()["data"]["totals"] == {}

    def test_record_and_summary(self, usage_setup):
        client, _, admin_id, __ = usage_setup
        # Record via API
        r = client.post(
            "/api/v1/usage/record",
            json={"event_type": "api_call", "quantity": 7},
            headers=_admin_header(admin_id),
        )
        assert r.status_code == 201
        assert r.json()["data"]["quantity"] == 7

        # Verify in summary
        r = client.get("/api/v1/usage/summary", headers=_admin_header(admin_id))
        assert r.json()["data"]["totals"].get("api_call") == 7

    def test_records_admin_only(self, usage_setup):
        client, _, admin_id, user_id = usage_setup
        r = client.get("/api/v1/usage/records", headers=_user_header(user_id))
        assert r.status_code == 403

    def test_record_missing_event_type(self, usage_setup):
        client, _, admin_id, __ = usage_setup
        r = client.post(
            "/api/v1/usage/record",
            json={"quantity": 1},
            headers=_admin_header(admin_id),
        )
        assert r.status_code == 422

    def test_summary_accessible_by_regular_user(self, usage_setup):
        client, _, __, user_id = usage_setup
        r = client.get("/api/v1/usage/summary", headers=_user_header(user_id))
        assert r.status_code == 200


class TestMigration0009:
    def test_upgrade_and_downgrade(self):
        db_path = _PROJECT_ROOT / "data" / "api" / "cogtrix_0009_test.db"
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
        result = subprocess.run(
            ["uv", "run", "alembic", "downgrade", "0008"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(_PROJECT_ROOT),
        )
        assert result.returncode == 0, f"downgrade failed:\n{result.stderr}"
        db_path.unlink(missing_ok=True)
