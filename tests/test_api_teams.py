"""Tests for Team management API (Enterprise Phase 1 — task 1.2.4).

Covers:
  - TeamRepository CRUD and membership operations
  - GET/POST/PATCH/DELETE /api/v1/teams
  - GET/POST/DELETE /api/v1/teams/{id}/members
  - Org-scope isolation, 404s, 409 conflicts
  - Alembic migration 0006 round-trip
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import subprocess
import uuid
from dataclasses import dataclass

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

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent


def _uid() -> str:
    return str(uuid.uuid4())


def _admin_header(user_id: str) -> dict:
    with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
        token = create_access_token(user_id=user_id, role="admin")
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# TeamRepository unit tests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TeamSetup:
    client: TestClient
    org_id: str
    admin_id: str
    member_id: str
    null_admin_id: str

    def __iter__(self):
        yield self.client
        yield self.org_id
        yield self.admin_id
        yield self.member_id


class TestTeamRepository:
    def _seed(self, sf) -> tuple[str, str]:
        """Return (org_id, user_id) after seeding one org and one user."""
        org_id = _uid()
        user_id = _uid()

        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                user_repo = UserRepository(session)
                await org_repo.create(org_id=org_id, name="Repo Org", slug="repo-org")
                await user_repo.create(
                    user_id=user_id,
                    username="repouser",
                    email="repouser@example.com",
                    password_hash="h",
                    org_id=org_id,
                )
                await session.commit()

        asyncio.run(_run())
        return org_id, user_id

    def test_create_and_get(self, sf):
        org_id, _ = self._seed(sf)

        async def _run():
            async with sf() as session:
                repo = TeamRepository(session)
                team = await repo.create(
                    team_id=_uid(), org_id=org_id, name="Engineering", description="Dev team"
                )
                await session.commit()
                found = await repo.get_by_id(team.id)
                assert found is not None
                assert found.name == "Engineering"
                assert found.org_id == org_id

        asyncio.run(_run())

    def test_get_missing_returns_none(self, sf):
        async def _run():
            async with sf() as session:
                repo = TeamRepository(session)
                assert await repo.get_by_id(_uid()) is None

        asyncio.run(_run())

    def test_list_by_org(self, sf):
        org_id, _ = self._seed(sf)

        async def _run():
            async with sf() as session:
                repo = TeamRepository(session)
                await repo.create(team_id=_uid(), org_id=org_id, name="Team A")
                await repo.create(team_id=_uid(), org_id=org_id, name="Team B")
                await session.commit()
                teams = await repo.list_by_org(org_id)
                assert len(teams) == 2

        asyncio.run(_run())

    def test_update_name(self, sf):
        org_id, _ = self._seed(sf)

        async def _run():
            async with sf() as session:
                repo = TeamRepository(session)
                tid = _uid()
                await repo.create(team_id=tid, org_id=org_id, name="Old Name")
                await session.commit()
                updated = await repo.update(tid, name="New Name")
                await session.commit()
                assert updated is not None
                assert updated.name == "New Name"

        asyncio.run(_run())

    def test_delete(self, sf):
        org_id, _ = self._seed(sf)

        async def _run():
            async with sf() as session:
                repo = TeamRepository(session)
                tid = _uid()
                await repo.create(team_id=tid, org_id=org_id, name="To Delete")
                await session.commit()
                deleted = await repo.delete(tid)
                await session.commit()
                assert deleted is True
                assert await repo.get_by_id(tid) is None

        asyncio.run(_run())

    def test_add_and_remove_member(self, sf):
        org_id, user_id = self._seed(sf)

        async def _run():
            async with sf() as session:
                repo = TeamRepository(session)
                tid = _uid()
                await repo.create(team_id=tid, org_id=org_id, name="Membership Team")
                await session.commit()
                await repo.add_member(membership_id=_uid(), team_id=tid, user_id=user_id)
                await session.commit()
                members = await repo.list_members(tid)
                assert len(members) == 1
                assert members[0].id == user_id
                count = await repo.count_members(tid)
                assert count == 1
                removed = await repo.remove_member(tid, user_id)
                await session.commit()
                assert removed is True
                assert await repo.count_members(tid) == 0

        asyncio.run(_run())

    def test_duplicate_membership_raises(self, sf):
        from sqlalchemy.exc import IntegrityError

        org_id, user_id = self._seed(sf)

        async def _run():
            async with sf() as session:
                repo = TeamRepository(session)
                tid = _uid()
                await repo.create(team_id=tid, org_id=org_id, name="Dup Team")
                await repo.add_member(membership_id=_uid(), team_id=tid, user_id=user_id)
                await session.commit()

            with pytest.raises(IntegrityError):
                async with sf() as session:
                    repo = TeamRepository(session)
                    await repo.add_member(membership_id=_uid(), team_id=tid, user_id=user_id)
                    await session.commit()

        asyncio.run(_run())

    def test_db_constraint_blocks_duplicate_team_name(self, sf):
        """DB-level unique constraint on (org_id, name) catches races."""
        from sqlalchemy.exc import IntegrityError

        org_id, _ = self._seed(sf)

        async def _run():
            async with sf() as session:
                repo = TeamRepository(session)
                await repo.create(team_id=_uid(), org_id=org_id, name="Race Team")
                await session.commit()

            with pytest.raises(IntegrityError):
                async with sf() as session:
                    repo = TeamRepository(session)
                    await repo.create(team_id=_uid(), org_id=org_id, name="Race Team")
                    await session.commit()

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def team_setup(engine):
    """Return (TestClient, org_id, admin_id, member_id)."""
    factory = async_sessionmaker(engine, expire_on_commit=False)

    org_id = _uid()
    admin_id = _uid()
    member_id = _uid()
    null_admin_id = _uid()

    async def _seed():
        async with factory() as session:
            org_repo = OrganizationRepository(session)
            user_repo = UserRepository(session)
            await org_repo.create(org_id=org_id, name="API Org", slug="api-org")
            await user_repo.create(
                user_id=admin_id,
                username="admin",
                email="admin@example.com",
                password_hash="h",
                role="admin",
                org_id=org_id,
            )
            await user_repo.create(
                user_id=member_id,
                username="member",
                email="member@example.com",
                password_hash="h",
                role="user",
                org_id=org_id,
            )
            await user_repo.create(
                user_id=null_admin_id,
                username="nulladmin",
                email="nulladmin@example.com",
                password_hash="h",
                role="admin",
                org_id=None,
            )
            await session.commit()

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
            yield _TeamSetup(client, org_id, admin_id, member_id, null_admin_id)

    app.dependency_overrides.clear()


class TestTeamAPI:
    def test_list_teams_empty(self, team_setup):
        client, _, admin_id, __ = team_setup
        r = client.get("/api/v1/teams", headers=_admin_header(admin_id))
        assert r.status_code == 200
        assert r.json()["data"] == []

    def test_create_team(self, team_setup):
        client, _, admin_id, __ = team_setup
        r = client.post(
            "/api/v1/teams",
            json={"name": "Engineering", "description": "Dev team"},
            headers=_admin_header(admin_id),
        )
        assert r.status_code == 201
        data = r.json()["data"]
        assert data["name"] == "Engineering"
        assert data["id"] is not None

    def test_create_duplicate_team_returns_409(self, team_setup):
        client, _, admin_id, __ = team_setup
        client.post("/api/v1/teams", json={"name": "Dup Team"}, headers=_admin_header(admin_id))
        r = client.post("/api/v1/teams", json={"name": "Dup Team"}, headers=_admin_header(admin_id))
        assert r.status_code == 409

    def test_create_team_db_integrity_error_returns_409(self, team_setup):
        """If DB constraint fires despite app check, route returns 409."""
        client, _, admin_id, __ = team_setup
        client.post("/api/v1/teams", json={"name": "DB Race"}, headers=_admin_header(admin_id))

        with patch(
            "cogtrix_core.api.routes.teams.TeamRepository.get_by_name_and_org", return_value=None
        ):
            r = client.post(
                "/api/v1/teams", json={"name": "DB Race"}, headers=_admin_header(admin_id)
            )
            assert r.status_code == 409
            assert "already exists" in r.json()["error"]["message"]

    def test_get_team(self, team_setup):
        client, _, admin_id, __ = team_setup
        create_r = client.post(
            "/api/v1/teams", json={"name": "Get Team"}, headers=_admin_header(admin_id)
        )
        team_id = create_r.json()["data"]["id"]
        r = client.get(f"/api/v1/teams/{team_id}", headers=_admin_header(admin_id))
        assert r.status_code == 200
        assert r.json()["data"]["id"] == team_id

    def test_get_missing_team_returns_404(self, team_setup):
        client, _, admin_id, __ = team_setup
        r = client.get(f"/api/v1/teams/{_uid()}", headers=_admin_header(admin_id))
        assert r.status_code == 404

    def test_update_team(self, team_setup):
        client, _, admin_id, __ = team_setup
        r = client.post("/api/v1/teams", json={"name": "Old Name"}, headers=_admin_header(admin_id))
        team_id = r.json()["data"]["id"]
        r = client.patch(
            f"/api/v1/teams/{team_id}",
            json={"name": "New Name"},
            headers=_admin_header(admin_id),
        )
        assert r.status_code == 200
        assert r.json()["data"]["name"] == "New Name"

    def test_delete_team(self, team_setup):
        client, _, admin_id, __ = team_setup
        r = client.post(
            "/api/v1/teams", json={"name": "Delete Me"}, headers=_admin_header(admin_id)
        )
        team_id = r.json()["data"]["id"]
        r = client.delete(f"/api/v1/teams/{team_id}", headers=_admin_header(admin_id))
        assert r.status_code == 200
        r = client.get(f"/api/v1/teams/{team_id}", headers=_admin_header(admin_id))
        assert r.status_code == 404


class TestTeamMembersAPI:
    def _create_team(self, client, admin_id, name="Test Team") -> str:
        r = client.post("/api/v1/teams", json={"name": name}, headers=_admin_header(admin_id))
        return r.json()["data"]["id"]

    def test_list_members_empty(self, team_setup):
        client, _, admin_id, __ = team_setup
        tid = self._create_team(client, admin_id)
        r = client.get(f"/api/v1/teams/{tid}/members", headers=_admin_header(admin_id))
        assert r.status_code == 200
        assert r.json()["data"] == []

    def test_add_member(self, team_setup):
        client, _, admin_id, member_id = team_setup
        tid = self._create_team(client, admin_id)
        r = client.post(
            f"/api/v1/teams/{tid}/members",
            json={"user_id": member_id, "role": "member"},
            headers=_admin_header(admin_id),
        )
        assert r.status_code == 201
        data = r.json()["data"]
        assert data["user_id"] == member_id
        assert data["role"] == "member"

    def test_add_duplicate_member_returns_409(self, team_setup):
        client, _, admin_id, member_id = team_setup
        tid = self._create_team(client, admin_id, "Dup Member Team")
        client.post(
            f"/api/v1/teams/{tid}/members",
            json={"user_id": member_id},
            headers=_admin_header(admin_id),
        )
        r = client.post(
            f"/api/v1/teams/{tid}/members",
            json={"user_id": member_id},
            headers=_admin_header(admin_id),
        )
        assert r.status_code == 409

    def test_remove_member(self, team_setup):
        client, _, admin_id, member_id = team_setup
        tid = self._create_team(client, admin_id, "Remove Team")
        client.post(
            f"/api/v1/teams/{tid}/members",
            json={"user_id": member_id},
            headers=_admin_header(admin_id),
        )
        r = client.delete(
            f"/api/v1/teams/{tid}/members/{member_id}", headers=_admin_header(admin_id)
        )
        assert r.status_code == 200

    def test_remove_nonmember_returns_404(self, team_setup):
        client, _, admin_id, __ = team_setup
        tid = self._create_team(client, admin_id, "404 Team")
        r = client.delete(f"/api/v1/teams/{tid}/members/{_uid()}", headers=_admin_header(admin_id))
        assert r.status_code == 404

    def test_member_count_in_team_response(self, team_setup):
        client, _, admin_id, member_id = team_setup
        tid = self._create_team(client, admin_id, "Count Team")
        client.post(
            f"/api/v1/teams/{tid}/members",
            json={"user_id": member_id},
            headers=_admin_header(admin_id),
        )
        r = client.get(f"/api/v1/teams/{tid}", headers=_admin_header(admin_id))
        assert r.json()["data"]["member_count"] == 1

    @pytest.mark.parametrize("suffix", ["", "/members"])
    def test_null_org_admin_cannot_read_team_scoped_endpoints(self, team_setup, suffix):
        client, _, admin_id, __ = team_setup
        tid = self._create_team(client, admin_id, "Null Org Team")
        r = client.get(
            f"/api/v1/teams/{tid}{suffix}",
            headers=_admin_header(team_setup.null_admin_id),
        )
        assert r.status_code == 403
        assert r.json()["error"]["code"] == "ORG_REQUIRED"


# ---------------------------------------------------------------------------
# Migration round-trip
# ---------------------------------------------------------------------------


class TestMigration0006:
    def test_upgrade_and_downgrade(self):
        db_path = _PROJECT_ROOT / "data" / "api" / "cogtrix_0006_test.db"
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

        # Downgrade to 0005 (removes teams/team_memberships added by 0006)
        result = subprocess.run(
            ["uv", "run", "alembic", "downgrade", "0005"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(_PROJECT_ROOT),
        )
        assert result.returncode == 0, f"downgrade failed:\n{result.stderr}"
        db_path.unlink(missing_ok=True)
