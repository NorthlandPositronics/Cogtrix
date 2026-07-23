"""Tests for admin endpoints.

Coverage:
  - GET /api/v1/admin/orgs returns paginated org list for admin users.
  - GET /api/v1/admin/orgs rejects non-admin users with 403.
  - Filter by name (substring), status, and plan.
  - Cursor pagination works correctly.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker  # noqa: E402

from src.api.app import create_app  # noqa: E402
from src.api.db.models import Organization  # noqa: E402
from src.api.db.repositories.organization import OrganizationRepository  # noqa: E402
from src.api.db.repositories.sessions import SessionRepository  # noqa: E402
from src.api.db.repositories.usage import UsageRepository  # noqa: E402
from src.api.db.repositories.users import UserRepository  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(engine):
    """Yield a TestClient with the async engine attached to app state."""
    import time
    from datetime import UTC, datetime

    # Create a fresh app with the test engine
    _app = create_app()

    from src.api.db import get_db

    async def _override_get_db():
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            yield session

    _app.dependency_overrides[get_db] = _override_get_db
    # Initialize startup_time and started_at for uptime calculations
    _app.state.startup_time = time.monotonic()
    _app.state.started_at = datetime.now(UTC)
    try:
        yield TestClient(_app)
    finally:
        _app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def _admin_token() -> str:
    """Return a valid admin JWT for test requests."""
    from src.api.auth import create_access_token

    return create_access_token(
        user_id=str(uuid.uuid4()),
        role="admin",
    )


def _superadmin_token() -> str:
    """Return a valid superadmin JWT for test requests."""
    from src.api.auth import create_access_token

    return create_access_token(
        user_id=str(uuid.uuid4()),
        role="superadmin",
    )


def _user_token() -> str:
    """Return a valid non-admin JWT for test requests."""
    from src.api.auth import create_access_token

    return create_access_token(
        user_id=str(uuid.uuid4()),
        role="user",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAdminOrgListAuth:
    def test_non_admin_gets_403(self, client):
        response = client.get(
            "/api/v1/admin/orgs",
            headers={"Authorization": f"Bearer {_user_token()}"},
        )
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"

    def test_admin_without_superadmin_gets_403(self, client):
        """Admin role alone (without superadmin) is denied."""
        response = client.get(
            "/api/v1/admin/orgs",
            headers={"Authorization": f"Bearer {_admin_token()}"},
        )
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"

    def test_superadmin_gets_200(self, client):
        response = client.get(
            "/api/v1/admin/orgs",
            headers={"Authorization": f"Bearer {_superadmin_token()}"},
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert "items" in data
        assert "next_cursor" in data
        assert "has_more" in data
        assert "total" in data


class TestAdminOrgListPagination:
    def test_pagination_limits(self, client):
        response = client.get(
            "/api/v1/admin/orgs?limit=5",
            headers={"Authorization": f"Bearer {_superadmin_token()}"},
        )
        assert response.status_code == 200

    def test_pagination_limit_too_high(self, client):
        response = client.get(
            "/api/v1/admin/orgs?limit=101",
            headers={"Authorization": f"Bearer {_superadmin_token()}"},
        )
        assert response.status_code == 422

    def test_cursor_pagination(self, client, sf):
        async def _seed():
            async with sf() as session:
                repo = OrganizationRepository(session)
                for i in range(5):
                    await repo.create(
                        org_id=str(uuid.uuid4()),
                        name=f"Org {i}",
                        slug=f"org-{i}",
                        plan="free",
                    )
                await session.commit()

        asyncio.run(_seed())

        # First page with limit=2
        r1 = client.get(
            "/api/v1/admin/orgs?limit=2",
            headers={"Authorization": f"Bearer {_superadmin_token()}"},
        )
        assert r1.status_code == 200
        data1 = r1.json()["data"]
        assert len(data1["items"]) == 2
        assert data1["has_more"] is True
        assert data1["next_cursor"] is not None

        # Second page
        r2 = client.get(
            f"/api/v1/admin/orgs?limit=2&cursor={data1['next_cursor']}",
            headers={"Authorization": f"Bearer {_superadmin_token()}"},
        )
        assert r2.status_code == 200
        data2 = r2.json()["data"]
        assert len(data2["items"]) == 2
        assert data2["has_more"] is True

        # Third page
        r3 = client.get(
            f"/api/v1/admin/orgs?limit=2&cursor={data2['next_cursor']}",
            headers={"Authorization": f"Bearer {_superadmin_token()}"},
        )
        assert r3.status_code == 200
        data3 = r3.json()["data"]
        assert len(data3["items"]) == 1
        assert data3["has_more"] is False
        assert data3["next_cursor"] is None

    def test_invalid_cursor_returns_400(self, client):
        response = client.get(
            "/api/v1/admin/orgs?cursor=not-valid-base64!!!",
            headers={"Authorization": f"Bearer {_superadmin_token()}"},
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "INVALID_CURSOR"


class TestAdminOrgListFilters:
    def test_filter_by_name(self, client, sf):
        async def _seed():
            async with sf() as session:
                repo = OrganizationRepository(session)
                await repo.create(
                    org_id=str(uuid.uuid4()), name="Alpha Corp", slug="alpha", plan="free"
                )
                await repo.create(
                    org_id=str(uuid.uuid4()), name="Beta Inc", slug="beta", plan="pro"
                )
                await session.commit()

        asyncio.run(_seed())

        response = client.get(
            "/api/v1/admin/orgs?name=Alpha",
            headers={"Authorization": f"Bearer {_superadmin_token()}"},
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert len(data["items"]) == 1
        assert data["items"][0]["name"] == "Alpha Corp"

    def test_filter_by_status(self, client, sf):
        async def _seed():
            async with sf() as session:
                repo = OrganizationRepository(session)
                o1 = await repo.create(
                    org_id=str(uuid.uuid4()), name="Active Org", slug="active-org", plan="free"
                )
                o1.status = "active"
                o2 = await repo.create(
                    org_id=str(uuid.uuid4()), name="Inactive Org", slug="inactive-org", plan="free"
                )
                o2.status = "inactive"
                await session.commit()

        asyncio.run(_seed())

        response = client.get(
            "/api/v1/admin/orgs?status=inactive",
            headers={"Authorization": f"Bearer {_superadmin_token()}"},
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert len(data["items"]) == 1
        assert data["items"][0]["name"] == "Inactive Org"

    def test_filter_by_plan(self, client, sf):
        async def _seed():
            async with sf() as session:
                repo = OrganizationRepository(session)
                await repo.create(
                    org_id=str(uuid.uuid4()), name="Free Org", slug="free-org", plan="free"
                )
                await repo.create(
                    org_id=str(uuid.uuid4()), name="Pro Org", slug="pro-org", plan="pro"
                )
                await session.commit()

        asyncio.run(_seed())

        response = client.get(
            "/api/v1/admin/orgs?plan=pro",
            headers={"Authorization": f"Bearer {_superadmin_token()}"},
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert len(data["items"]) == 1
        assert data["items"][0]["name"] == "Pro Org"

    def test_filter_combined(self, client, sf):
        """Multiple filters applied together use AND logic."""

        async def _seed():
            async with sf() as session:
                repo = OrganizationRepository(session)
                o1 = await repo.create(
                    org_id=str(uuid.uuid4()),
                    name="Alpha Corp",
                    slug="alpha",
                    plan="free",
                )
                o1.status = "active"
                o2 = await repo.create(
                    org_id=str(uuid.uuid4()),
                    name="Alpha Pro",
                    slug="alpha-pro",
                    plan="pro",
                )
                o2.status = "active"
                o3 = await repo.create(
                    org_id=str(uuid.uuid4()),
                    name="Beta Corp",
                    slug="beta",
                    plan="free",
                )
                o3.status = "inactive"
                await session.commit()

        asyncio.run(_seed())

        response = client.get(
            "/api/v1/admin/orgs?name=Alpha&status=active&plan=free",
            headers={"Authorization": f"Bearer {_superadmin_token()}"},
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert len(data["items"]) == 1
        assert data["items"][0]["name"] == "Alpha Corp"

    def test_filter_created_after(self, client, sf):
        from datetime import UTC, datetime, timedelta

        async def _seed():
            async with sf() as session:
                # Create two orgs directly (bypassing default created_at)
                old = Organization(
                    id=str(uuid.uuid4()),
                    name="Old Org",
                    slug="old-org",
                    plan="free",
                    created_at=datetime.now(UTC) - timedelta(days=30),
                )
                new = Organization(
                    id=str(uuid.uuid4()),
                    name="New Org",
                    slug="new-org",
                    plan="free",
                )
                session.add_all([old, new])
                await session.commit()

        asyncio.run(_seed())

        cutoff = (datetime.now(UTC) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")

        response = client.get(
            f"/api/v1/admin/orgs?created_after={cutoff}",
            headers={"Authorization": f"Bearer {_superadmin_token()}"},
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert len(data["items"]) >= 1
        names = [item["name"] for item in data["items"]]
        assert "New Org" in names
        assert "Old Org" not in names

    def test_member_count_computed(self, client, sf):
        async def _seed():
            async with sf() as session:
                from src.api.auth import hash_password

                org_repo = OrganizationRepository(session)
                user_repo = UserRepository(session)
                org = await org_repo.create(
                    org_id=str(uuid.uuid4()),
                    name="Member Test Org",
                    slug="member-test",
                    plan="free",
                )
                await session.commit()

                for i in range(3):
                    await user_repo.create(
                        user_id=str(uuid.uuid4()),
                        username=f"user{i}",
                        email=f"user{i}@example.com",
                        password_hash=hash_password("secret"),
                        role="user",
                        org_id=org.id,
                    )
                await session.commit()

        asyncio.run(_seed())

        response = client.get(
            "/api/v1/admin/orgs?name=Member Test",
            headers={"Authorization": f"Bearer {_superadmin_token()}"},
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert len(data["items"]) == 1
        assert data["items"][0]["member_count"] == 3


class TestAdminStatsAuth:
    def test_non_admin_gets_403(self, client):
        response = client.get(
            "/api/v1/admin/stats",
            headers={"Authorization": f"Bearer {_user_token()}"},
        )
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"

    def test_admin_gets_200(self, client):
        response = client.get(
            "/api/v1/admin/stats",
            headers={"Authorization": f"Bearer {_admin_token()}"},
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert "total_orgs" in data
        assert "active_sessions" in data
        assert "total_users" in data
        assert "mcp_server_count" in data


class TestAdminStatsCounts:
    def test_stats_match_seeded_data(self, client, sf):
        from src.api.auth import hash_password

        async def _seed():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                user_repo = UserRepository(session)
                session_repo = SessionRepository(session)

                # Seed 2 orgs
                org1 = await org_repo.create(
                    org_id=str(uuid.uuid4()),
                    name="Stats Org 1",
                    slug="stats-org-1",
                    plan="free",
                )
                org2 = await org_repo.create(
                    org_id=str(uuid.uuid4()),
                    name="Stats Org 2",
                    slug="stats-org-2",
                    plan="pro",
                )
                await session.commit()

                # Seed 4 users (2 per org)
                for i in range(2):
                    await user_repo.create(
                        user_id=str(uuid.uuid4()),
                        username=f"user{i}",
                        email=f"user{i}@example.com",
                        password_hash=hash_password("secret"),
                        role="user",
                        org_id=org1.id,
                    )
                for i in range(2, 4):
                    await user_repo.create(
                        user_id=str(uuid.uuid4()),
                        username=f"user{i}",
                        email=f"user{i}@example.com",
                        password_hash=hash_password("secret"),
                        role="user",
                        org_id=org2.id,
                    )
                await session.commit()

                # Seed 3 active sessions for the first user
                first_user = await user_repo.get_by_username("user0")
                assert first_user is not None
                for i in range(3):
                    await session_repo.create(
                        user_id=first_user.id,
                        name=f"Session {i}",
                    )
                await session.commit()

        asyncio.run(_seed())

        response = client.get(
            "/api/v1/admin/stats",
            headers={"Authorization": f"Bearer {_admin_token()}"},
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["total_orgs"] == 2
        assert data["total_users"] == 4
        assert data["active_sessions"] == 3


class TestAdminOrgUsageAuth:
    def test_non_admin_gets_403(self, client):
        org_id = str(uuid.uuid4())
        response = client.get(
            f"/api/v1/admin/orgs/{org_id}/usage",
            headers={"Authorization": f"Bearer {_user_token()}"},
        )
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"


class TestAdminOrgUsage:
    def test_usage_empty_org(self, client):
        org_id = str(uuid.uuid4())
        response = client.get(
            f"/api/v1/admin/orgs/{org_id}/usage",
            headers={"Authorization": f"Bearer {_admin_token()}"},
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["org_id"] == org_id
        assert data["total_api_calls"] == 0
        assert data["total_sessions"] == 0

    def test_usage_with_records(self, client, sf):
        async def _seed():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                usage_repo = UsageRepository(session)
                org = await org_repo.create(
                    org_id=str(uuid.uuid4()),
                    name="Usage Test Org",
                    slug="usage-test",
                    plan="free",
                )
                await session.commit()

                # Seed usage records
                await usage_repo.record(
                    org_id=org.id,
                    event_type="api_call",
                    quantity=5,
                )
                await usage_repo.record(
                    org_id=org.id,
                    event_type="session_created",
                    quantity=2,
                )
                await session.commit()

        asyncio.run(_seed())

        async def _fetch_org_id():
            async with sf() as session:
                repo = OrganizationRepository(session)
                org = await repo.get_by_slug("usage-test")
                return org.id

        org_id = asyncio.run(_fetch_org_id())

        response = client.get(
            f"/api/v1/admin/orgs/{org_id}/usage",
            headers={"Authorization": f"Bearer {_admin_token()}"},
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["org_id"] == org_id
        assert data["total_api_calls"] == 5
        assert data["total_sessions"] == 2
        assert data["total_users_provisioned"] == 0

    def test_usage_date_filter(self, client, sf):
        from datetime import UTC, datetime, timedelta

        async def _seed():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                usage_repo = UsageRepository(session)
                org = await org_repo.create(
                    org_id=str(uuid.uuid4()),
                    name="Date Filter Org",
                    slug="date-filter",
                    plan="free",
                )
                await session.commit()

                # Old record
                await usage_repo.record(
                    org_id=org.id,
                    event_type="api_call",
                    quantity=1,
                    at=datetime.now(UTC) - timedelta(days=10),
                )
                # Recent record
                await usage_repo.record(
                    org_id=org.id,
                    event_type="api_call",
                    quantity=7,
                    at=datetime.now(UTC) - timedelta(days=1),
                )
                await session.commit()
                return org.id

        org_id = asyncio.run(_seed())

        from_param = (datetime.now(UTC) - timedelta(days=5)).strftime("%Y-%m-%d")
        to_param = datetime.now(UTC).strftime("%Y-%m-%d")

        response = client.get(
            f"/api/v1/admin/orgs/{org_id}/usage?from={from_param}&to={to_param}",
            headers={"Authorization": f"Bearer {_admin_token()}"},
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["total_api_calls"] == 7


class TestAdminOrgAudit:
    def test_non_admin_gets_403(self, client):
        org_id = str(uuid.uuid4())
        response = client.get(
            f"/api/v1/admin/orgs/{org_id}/audit",
            headers={"Authorization": f"Bearer {_user_token()}"},
        )
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"

    def test_audit_stub(self, client):
        org_id = str(uuid.uuid4())
        response = client.get(
            f"/api/v1/admin/orgs/{org_id}/audit",
            headers={"Authorization": f"Bearer {_admin_token()}"},
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["entries"] == []
        assert "not yet implemented" in data["note"]


# Autouse fixture to clear system stats cache before each test to prevent interference
@pytest.fixture(autouse=True)
def _clear_system_cache():
    """Clear the system stats cache before each test."""
    from src.api.routes.admin import _cache

    _cache.set(None)
    _cache._data = None
    _cache._timestamp = None


class TestAdminSystemStatsAuth:
    """Tests for GET /api/v1/admin/system authentication."""

    def test_non_admin_gets_403(self, client):
        """Non-admin users should get 403 FORBIDDEN."""
        response = client.get(
            "/api/v1/admin/system",
            headers={"Authorization": f"Bearer {_user_token()}"},
        )
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"

    def test_admin_gets_403(self, client):
        """Regular admin users should get 403 FORBIDDEN (superadmin only)."""
        response = client.get(
            "/api/v1/admin/system",
            headers={"Authorization": f"Bearer {_admin_token()}"},
        )
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"

    def test_superadmin_gets_200(self, client):
        """Superadmin users should get 200 OK."""
        # Create a superadmin token
        from src.api.auth import create_access_token

        superadmin_token = create_access_token(
            user_id=str(uuid.uuid4()),
            role="superadmin",
        )
        response = client.get(
            "/api/v1/admin/system",
            headers={"Authorization": f"Bearer {superadmin_token}"},
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert "total_orgs" in data
        assert "total_users" in data
        assert "active_sessions" in data
        assert "estimated_token_usage_24h" in data
        assert "api_requests_24h" in data
        assert "error_rate_24h" in data
        assert "db_pool_status" in data
        assert "db_pool_size" in data
        assert "db_pool_max" in data
        assert "redis_connected" in data
        assert "uptime_s" in data
        assert "version" in data
        assert "started_at" in data

    def test_no_token_gets_401(self, client):
        """Requests without token should get 401 UNAUTHORIZED."""
        response = client.get("/api/v1/admin/system")
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "UNAUTHORIZED"


class TestAdminSystemStatsValues:
    """Tests for GET /api/v1/admin/system data."""

    def test_system_stats_counts(self, client, sf):
        """Verify statistics match seeded data - uses shared session factory (sf)."""
        from src.api.auth import hash_password

        async def _seed():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                user_repo = UserRepository(session)
                session_repo = SessionRepository(session)
                usage_repo = UsageRepository(session)

                # Seed 2 orgs
                await org_repo.create(
                    org_id=str(uuid.uuid4()),
                    name="Stats Org 1",
                    slug="stats-org-1",
                    plan="free",
                )
                await org_repo.create(
                    org_id=str(uuid.uuid4()),
                    name="Stats Org 2",
                    slug="stats-org-2",
                    plan="pro",
                )
                await session.commit()

                # Seed 4 users (no org for these users)
                for i in range(4):
                    await user_repo.create(
                        user_id=str(uuid.uuid4()),
                        username=f"user{i}",
                        email=f"user{i}@example.com",
                        password_hash=hash_password("secret"),
                        role="user",
                        org_id=None,
                    )
                await session.commit()

                # Seed 3 active sessions
                first_user = await user_repo.get_by_username("user0")
                assert first_user is not None
                for i in range(3):
                    await session_repo.create(
                        user_id=first_user.id,
                        name=f"Session {i}",
                    )
                await session.commit()

                # Seed some API usage records for the last 24h
                from datetime import UTC, datetime, timedelta

                now = datetime.now(UTC)
                # Record API usage at various times within the last 24 hours
                await usage_repo.record(
                    org_id=str(uuid.uuid4()),
                    event_type="api_call",
                    quantity=5,
                    at=now - timedelta(hours=1),
                )
                await usage_repo.record(
                    org_id=str(uuid.uuid4()),
                    event_type="api_call",
                    quantity=3,
                    at=now - timedelta(hours=2),
                )
                await session.commit()

        asyncio.run(_seed())

        # Create superadmin token
        from src.api.auth import create_access_token

        superadmin_token = create_access_token(
            user_id=str(uuid.uuid4()),
            role="superadmin",
        )

        response = client.get(
            "/api/v1/admin/system",
            headers={"Authorization": f"Bearer {superadmin_token}"},
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["total_orgs"] == 2
        assert data["total_users"] == 4
        assert data["active_sessions"] == 3

    def test_system_stats_cached(self, client, sf):
        """Verify response is served from cache after first fetch."""
        from src.api.auth import create_access_token

        async def _seed():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                await org_repo.create(
                    org_id=str(uuid.uuid4()),
                    name="Cache Test Org",
                    slug="cache-test",
                    plan="free",
                )
                await session.commit()

                # Seed some orgs for more accurate testing
                for i in range(2):
                    await org_repo.create(
                        org_id=str(uuid.uuid4()),
                        name=f"User Org {i}",
                        slug=f"user-org-{i}",
                        plan="free",
                    )
                await session.commit()

        asyncio.run(_seed())

        # Create superadmin token
        superadmin_token = create_access_token(
            user_id=str(uuid.uuid4()),
            role="superadmin",
        )

        # First request - fetches fresh data
        response1 = client.get(
            "/api/v1/admin/system",
            headers={"Authorization": f"Bearer {superadmin_token}"},
        )
        assert response1.status_code == 200
        data1 = response1.json()["data"]

        # Second request - should be cached
        response2 = client.get(
            "/api/v1/admin/system",
            headers={"Authorization": f"Bearer {superadmin_token}"},
        )
        assert response2.status_code == 200
        data2 = response2.json()["data"]

        # Both responses should have same values for non-uptime fields
        assert data1["total_orgs"] == data2["total_orgs"]
        assert data1["total_users"] == data2["total_users"]
        assert data1["active_sessions"] == data2["active_sessions"]


class TestAdminPoolMaxFormula:
    """Regression tests for #1112 — pool max formula must use configured size."""

    def test_db_pool_max_uses_configured_size_not_checked_out(self, client, engine):
        """QueuePool.size() returns checked-out count; db_pool_max must use configured size."""
        from src.api.auth import create_access_token

        real_pool = engine.pool

        class FakeQueuePool:
            def __init__(self):
                self.maxoverflow = 5
                self._pool = type("_pool", (), {"maxsize": 10})()

            def size(self):
                return 3  # simulate 3 checked-out connections

            def connect(self, *args, **kwargs):
                return real_pool.connect(*args, **kwargs)

            def __getattr__(self, name):
                return getattr(real_pool, name)

        engine.pool = FakeQueuePool()
        try:
            superadmin_token = create_access_token(
                user_id=str(uuid.uuid4()),
                role="superadmin",
            )
            response = client.get(
                "/api/v1/admin/system",
                headers={"Authorization": f"Bearer {superadmin_token}"},
            )
            assert response.status_code == 200
            data = response.json()["data"]
            # Configured max = 10 (pool_size) + 5 (maxoverflow) = 15
            # Bug would return 3 (checked-out) + 5 = 8
            assert data["db_pool_max"] == 15
        finally:
            engine.pool = real_pool


# ---------------------------------------------------------------------------
# Impersonation auth
# ---------------------------------------------------------------------------


class TestImpersonationAuth:
    def test_non_superadmin_gets_403(self, client):
        org_id = str(uuid.uuid4())
        response = client.post(
            f"/api/v1/admin/orgs/{org_id}/impersonate",
            json={"user_id": str(uuid.uuid4()), "reason": "test"},
            headers={"Authorization": f"Bearer {_admin_token()}"},
        )
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"

    def test_user_gets_403(self, client):
        org_id = str(uuid.uuid4())
        response = client.post(
            f"/api/v1/admin/orgs/{org_id}/impersonate",
            json={"user_id": str(uuid.uuid4()), "reason": "test"},
            headers={"Authorization": f"Bearer {_user_token()}"},
        )
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"


# ---------------------------------------------------------------------------
# Impersonation lifecycle
# ---------------------------------------------------------------------------


class TestImpersonationLifecycle:
    def test_start_impersonation_success(self, client, sf):
        from src.api.auth import hash_password

        async def _seed():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                user_repo = UserRepository(session)
                org = await org_repo.create(
                    org_id=str(uuid.uuid4()),
                    name="Imp Org",
                    slug="imp-org",
                    plan="free",
                )
                superadmin = await user_repo.create(
                    user_id=str(uuid.uuid4()),
                    username="super",
                    email="super@example.com",
                    password_hash=hash_password("secret"),
                    role="superadmin",
                )
                member = await user_repo.create(
                    user_id=str(uuid.uuid4()),
                    username="member",
                    email="member@example.com",
                    password_hash=hash_password("secret"),
                    role="user",
                    org_id=org.id,
                )
                await session.commit()
                return org.id, superadmin.id, member.id

        org_id, superadmin_id, member_id = asyncio.run(_seed())

        from src.api.auth import create_access_token

        token = create_access_token(superadmin_id, "superadmin")
        response = client.post(
            f"/api/v1/admin/orgs/{org_id}/impersonate",
            json={"user_id": member_id, "reason": "debugging"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["impersonated_user_id"] == member_id
        assert data["org_id"] == org_id
        assert "impersonation_token" in data
        assert "expires_at" in data

        # Verify the returned token is a valid JWT that decodes correctly.
        import jwt as _jwt

        from src.api.auth import _get_jwt_secret

        decoded = _jwt.decode(
            data["impersonation_token"],
            _get_jwt_secret(),
            algorithms=["HS256"],
        )
        assert decoded["sub"] == member_id
        assert decoded["role"] == "user"
        assert decoded["impersonated_by"] == superadmin_id
        assert "impersonation_session_id" in decoded

    def test_cannot_chain_impersonation(self, client, sf):
        from src.api.auth import hash_password

        async def _seed():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                user_repo = UserRepository(session)
                org = await org_repo.create(
                    org_id=str(uuid.uuid4()),
                    name="Imp Org 2",
                    slug="imp-org-2",
                    plan="free",
                )
                superadmin = await user_repo.create(
                    user_id=str(uuid.uuid4()),
                    username="super2",
                    email="super2@example.com",
                    password_hash=hash_password("secret"),
                    role="superadmin",
                )
                member = await user_repo.create(
                    user_id=str(uuid.uuid4()),
                    username="member2",
                    email="member2@example.com",
                    password_hash=hash_password("secret"),
                    role="user",
                    org_id=org.id,
                )
                await session.commit()
                return org.id, superadmin.id, member.id

        org_id, superadmin_id, member_id = asyncio.run(_seed())

        from src.api.auth import create_access_token

        token = create_access_token(superadmin_id, "superadmin")
        # First impersonation
        r1 = client.post(
            f"/api/v1/admin/orgs/{org_id}/impersonate",
            json={"user_id": member_id, "reason": "first"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r1.status_code == 200

        # Second impersonation should fail
        r2 = client.post(
            f"/api/v1/admin/orgs/{org_id}/impersonate",
            json={"user_id": member_id, "reason": "second"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r2.status_code == 409
        assert r2.json()["error"]["code"] == "ALREADY_IMPERSONATING"

    def test_impersonate_missing_org(self, client):
        org_id = str(uuid.uuid4())
        response = client.post(
            f"/api/v1/admin/orgs/{org_id}/impersonate",
            json={"user_id": str(uuid.uuid4()), "reason": "test"},
            headers={"Authorization": f"Bearer {_superadmin_token()}"},
        )
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "NOT_FOUND"

    def test_impersonate_missing_user(self, client, sf):
        async def _seed():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                org = await org_repo.create(
                    org_id=str(uuid.uuid4()),
                    name="Imp Org 3",
                    slug="imp-org-3",
                    plan="free",
                )
                await session.commit()
                return org.id

        org_id = asyncio.run(_seed())
        response = client.post(
            f"/api/v1/admin/orgs/{org_id}/impersonate",
            json={"user_id": str(uuid.uuid4()), "reason": "test"},
            headers={"Authorization": f"Bearer {_superadmin_token()}"},
        )
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "NOT_FOUND"

    def test_stop_impersonation_by_superadmin(self, client, sf):
        from src.api.auth import hash_password

        async def _seed():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                user_repo = UserRepository(session)
                org = await org_repo.create(
                    org_id=str(uuid.uuid4()),
                    name="Imp Org 4",
                    slug="imp-org-4",
                    plan="free",
                )
                superadmin = await user_repo.create(
                    user_id=str(uuid.uuid4()),
                    username="super4",
                    email="super4@example.com",
                    password_hash=hash_password("secret"),
                    role="superadmin",
                )
                member = await user_repo.create(
                    user_id=str(uuid.uuid4()),
                    username="member4",
                    email="member4@example.com",
                    password_hash=hash_password("secret"),
                    role="user",
                    org_id=org.id,
                )
                await session.commit()
                return org.id, superadmin.id, member.id

        org_id, superadmin_id, member_id = asyncio.run(_seed())

        from src.api.auth import create_access_token

        token = create_access_token(superadmin_id, "superadmin")
        r1 = client.post(
            f"/api/v1/admin/orgs/{org_id}/impersonate",
            json={"user_id": member_id, "reason": "test"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r1.status_code == 200

        r2 = client.delete(
            "/api/v1/admin/impersonate",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r2.status_code == 200
        assert r2.json()["data"]["status"] == "ended"

    def test_stop_impersonation_by_token(self, client, sf):
        from src.api.auth import hash_password

        async def _seed():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                user_repo = UserRepository(session)
                org = await org_repo.create(
                    org_id=str(uuid.uuid4()),
                    name="Imp Org 5",
                    slug="imp-org-5",
                    plan="free",
                )
                superadmin = await user_repo.create(
                    user_id=str(uuid.uuid4()),
                    username="super5",
                    email="super5@example.com",
                    password_hash=hash_password("secret"),
                    role="superadmin",
                )
                member = await user_repo.create(
                    user_id=str(uuid.uuid4()),
                    username="member5",
                    email="member5@example.com",
                    password_hash=hash_password("secret"),
                    role="user",
                    org_id=org.id,
                )
                await session.commit()
                return org.id, superadmin.id, member.id

        org_id, superadmin_id, member_id = asyncio.run(_seed())

        from src.api.auth import create_access_token

        super_token = create_access_token(superadmin_id, "superadmin")
        r1 = client.post(
            f"/api/v1/admin/orgs/{org_id}/impersonate",
            json={"user_id": member_id, "reason": "test"},
            headers={"Authorization": f"Bearer {super_token}"},
        )
        assert r1.status_code == 200
        imp_token = r1.json()["data"]["impersonation_token"]

        r2 = client.delete(
            "/api/v1/admin/impersonate",
            headers={"Authorization": f"Bearer {imp_token}"},
        )
        assert r2.status_code == 200
        assert r2.json()["data"]["status"] == "ended"


# ---------------------------------------------------------------------------
# Impersonation token validation
# ---------------------------------------------------------------------------


class TestImpersonationToken:
    def test_impersonation_token_rejected_after_stop(self, client, sf):
        from src.api.auth import hash_password

        async def _seed():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                user_repo = UserRepository(session)
                org = await org_repo.create(
                    org_id=str(uuid.uuid4()),
                    name="Imp Org 6",
                    slug="imp-org-6",
                    plan="free",
                )
                superadmin = await user_repo.create(
                    user_id=str(uuid.uuid4()),
                    username="super6",
                    email="super6@example.com",
                    password_hash=hash_password("secret"),
                    role="superadmin",
                )
                member = await user_repo.create(
                    user_id=str(uuid.uuid4()),
                    username="member6",
                    email="member6@example.com",
                    password_hash=hash_password("secret"),
                    role="user",
                    org_id=org.id,
                )
                await session.commit()
                return org.id, superadmin.id, member.id

        org_id, superadmin_id, member_id = asyncio.run(_seed())

        from src.api.auth import create_access_token

        super_token = create_access_token(superadmin_id, "superadmin")
        r1 = client.post(
            f"/api/v1/admin/orgs/{org_id}/impersonate",
            json={"user_id": member_id, "reason": "test"},
            headers={"Authorization": f"Bearer {super_token}"},
        )
        assert r1.status_code == 200
        imp_token = r1.json()["data"]["impersonation_token"]

        # Stop the session
        r2 = client.delete(
            "/api/v1/admin/impersonate",
            headers={"Authorization": f"Bearer {super_token}"},
        )
        assert r2.status_code == 200

        # Try to use the token — should fail
        r3 = client.get(
            "/api/v1/admin/stats",
            headers={"Authorization": f"Bearer {imp_token}"},
        )
        assert r3.status_code == 401


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------


class TestImpersonationAudit:
    def test_audit_entries_created(self, client, sf):
        from src.api.auth import hash_password

        async def _seed():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                user_repo = UserRepository(session)
                org = await org_repo.create(
                    org_id=str(uuid.uuid4()),
                    name="Audit Org",
                    slug="audit-org",
                    plan="free",
                )
                superadmin = await user_repo.create(
                    user_id=str(uuid.uuid4()),
                    username="superaudit",
                    email="superaudit@example.com",
                    password_hash=hash_password("secret"),
                    role="superadmin",
                )
                member = await user_repo.create(
                    user_id=str(uuid.uuid4()),
                    username="memberaudit",
                    email="memberaudit@example.com",
                    password_hash=hash_password("secret"),
                    role="user",
                    org_id=org.id,
                )
                await session.commit()
                return org.id, superadmin.id, member.id

        org_id, superadmin_id, member_id = asyncio.run(_seed())

        from src.api.auth import create_access_token

        super_token = create_access_token(superadmin_id, "superadmin")
        r1 = client.post(
            f"/api/v1/admin/orgs/{org_id}/impersonate",
            json={"user_id": member_id, "reason": "audit test"},
            headers={"Authorization": f"Bearer {super_token}"},
        )
        assert r1.status_code == 200

        r2 = client.delete(
            "/api/v1/admin/impersonate",
            headers={"Authorization": f"Bearer {super_token}"},
        )
        assert r2.status_code == 200

        r3 = client.get(
            f"/api/v1/admin/orgs/{org_id}/audit",
            headers={"Authorization": f"Bearer {super_token}"},
        )
        assert r3.status_code == 200
        entries = r3.json()["data"]["entries"]
        actions = [e["action"] for e in entries]
        assert "impersonation.start" in actions
        assert "impersonation.end" in actions

        start_entry = next(e for e in entries if e["action"] == "impersonation.start")
        assert start_entry["actor_id"] == superadmin_id
        assert start_entry["impersonated_by"] is None

        end_entry = next(e for e in entries if e["action"] == "impersonation.end")
        assert end_entry["actor_id"] == superadmin_id
        assert end_entry["impersonated_by"] is None
