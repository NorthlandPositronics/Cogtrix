"""Tests for Workspace management API (Enterprise Phase 1 — task 1.3.1)."""

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
from cogtrix_core.api.db.repositories.users import UserRepository  # noqa: E402

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent


def _uid() -> str:
    return str(uuid.uuid4())


def _admin_header(user_id: str) -> dict:
    with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
        token = create_access_token(user_id=user_id, role="admin")
    return {"Authorization": f"Bearer {token}"}


@dataclass(frozen=True)
class _WorkspaceSetup:
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


@pytest.fixture()
def ws_setup(engine):
    factory = async_sessionmaker(engine, expire_on_commit=False)
    org_id = _uid()
    admin_id = _uid()
    member_id = _uid()
    null_admin_id = _uid()

    async def _seed():
        async with factory() as session:
            org_repo = OrganizationRepository(session)
            user_repo = UserRepository(session)
            await org_repo.create(org_id=org_id, name="WS Org", slug="ws-org")
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
            yield _WorkspaceSetup(client, org_id, admin_id, member_id, null_admin_id)

    app.dependency_overrides.clear()


class TestWorkspacesCRUD:
    def test_list_empty(self, ws_setup):
        client, _, admin_id, __ = ws_setup
        r = client.get("/api/v1/workspaces", headers=_admin_header(admin_id))
        assert r.status_code == 200
        assert r.json()["data"] == []

    def test_create(self, ws_setup):
        client, _, admin_id, __ = ws_setup
        r = client.post(
            "/api/v1/workspaces",
            json={"name": "Engineering", "description": "Dev workspace"},
            headers=_admin_header(admin_id),
        )
        assert r.status_code == 201
        data = r.json()["data"]
        assert data["name"] == "Engineering"
        assert data["is_active"] is True

    def test_create_duplicate_returns_409(self, ws_setup):
        client, _, admin_id, __ = ws_setup
        client.post("/api/v1/workspaces", json={"name": "Dup"}, headers=_admin_header(admin_id))
        r = client.post("/api/v1/workspaces", json={"name": "Dup"}, headers=_admin_header(admin_id))
        assert r.status_code == 409

    def test_get(self, ws_setup):
        client, _, admin_id, __ = ws_setup
        r = client.post(
            "/api/v1/workspaces", json={"name": "Get WS"}, headers=_admin_header(admin_id)
        )
        ws_id = r.json()["data"]["id"]
        r = client.get(f"/api/v1/workspaces/{ws_id}", headers=_admin_header(admin_id))
        assert r.status_code == 200
        assert r.json()["data"]["id"] == ws_id

    def test_get_missing_returns_404(self, ws_setup):
        client, _, admin_id, __ = ws_setup
        r = client.get(f"/api/v1/workspaces/{_uid()}", headers=_admin_header(admin_id))
        assert r.status_code == 404

    def test_update(self, ws_setup):
        client, _, admin_id, __ = ws_setup
        r = client.post("/api/v1/workspaces", json={"name": "Old"}, headers=_admin_header(admin_id))
        ws_id = r.json()["data"]["id"]
        r = client.patch(
            f"/api/v1/workspaces/{ws_id}",
            json={"name": "New", "is_active": False},
            headers=_admin_header(admin_id),
        )
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["name"] == "New"
        assert data["is_active"] is False

    def test_delete(self, ws_setup):
        client, _, admin_id, __ = ws_setup
        r = client.post("/api/v1/workspaces", json={"name": "Del"}, headers=_admin_header(admin_id))
        ws_id = r.json()["data"]["id"]
        assert (
            client.delete(
                f"/api/v1/workspaces/{ws_id}", headers=_admin_header(admin_id)
            ).status_code
            == 200
        )
        assert (
            client.get(f"/api/v1/workspaces/{ws_id}", headers=_admin_header(admin_id)).status_code
            == 404
        )


class TestWorkspaceMembers:
    def _create_ws(self, client, admin_id, name="Test WS") -> str:
        r = client.post("/api/v1/workspaces", json={"name": name}, headers=_admin_header(admin_id))
        return r.json()["data"]["id"]

    def test_add_and_list_member(self, ws_setup):
        client, _, admin_id, member_id = ws_setup
        ws_id = self._create_ws(client, admin_id)
        r = client.post(
            f"/api/v1/workspaces/{ws_id}/members",
            json={"user_id": member_id, "role": "member"},
            headers=_admin_header(admin_id),
        )
        assert r.status_code == 201
        r = client.get(f"/api/v1/workspaces/{ws_id}/members", headers=_admin_header(admin_id))
        assert len(r.json()["data"]) == 1

    def test_add_duplicate_returns_409(self, ws_setup):
        client, _, admin_id, member_id = ws_setup
        ws_id = self._create_ws(client, admin_id, "Dup Member WS")
        client.post(
            f"/api/v1/workspaces/{ws_id}/members",
            json={"user_id": member_id},
            headers=_admin_header(admin_id),
        )
        r = client.post(
            f"/api/v1/workspaces/{ws_id}/members",
            json={"user_id": member_id},
            headers=_admin_header(admin_id),
        )
        assert r.status_code == 409

    def test_remove_member(self, ws_setup):
        client, _, admin_id, member_id = ws_setup
        ws_id = self._create_ws(client, admin_id, "Remove WS")
        client.post(
            f"/api/v1/workspaces/{ws_id}/members",
            json={"user_id": member_id},
            headers=_admin_header(admin_id),
        )
        r = client.delete(
            f"/api/v1/workspaces/{ws_id}/members/{member_id}",
            headers=_admin_header(admin_id),
        )
        assert r.status_code == 200

    def test_member_count_in_response(self, ws_setup):
        client, _, admin_id, member_id = ws_setup
        ws_id = self._create_ws(client, admin_id, "Count WS")
        client.post(
            f"/api/v1/workspaces/{ws_id}/members",
            json={"user_id": member_id},
            headers=_admin_header(admin_id),
        )
        r = client.get(f"/api/v1/workspaces/{ws_id}", headers=_admin_header(admin_id))
        assert r.json()["data"]["member_count"] == 1

    @pytest.mark.parametrize("suffix", ["", "/members", "/config"])
    def test_null_org_admin_cannot_read_workspace_scoped_endpoints(self, ws_setup, suffix):
        client, _, admin_id, __ = ws_setup
        ws_id = self._create_ws(client, admin_id, "Null Org WS")
        r = client.get(
            f"/api/v1/workspaces/{ws_id}{suffix}",
            headers=_admin_header(ws_setup.null_admin_id),
        )
        assert r.status_code == 403
        assert r.json()["error"]["code"] == "ORG_REQUIRED"


class TestMigration0007:
    def test_upgrade_and_downgrade(self):
        db_path = _PROJECT_ROOT / "data" / "api" / "cogtrix_0007_test.db"
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
        # Downgrade to 0006 (removes workspace tables added by 0007)
        result = subprocess.run(
            ["uv", "run", "alembic", "downgrade", "0006"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(_PROJECT_ROOT),
        )
        assert result.returncode == 0, f"downgrade failed:\n{result.stderr}"
        db_path.unlink(missing_ok=True)
