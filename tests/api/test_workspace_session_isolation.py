"""Workspace isolation regression tests for session and message endpoints (issue #960).

Verifies that:
- Users can only create sessions in workspaces they are members of
- Removed workspace members lose access to existing workspace sessions
- Admins retain access to workspace sessions without explicit membership
- Session responses include the workspace_id field
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker  # noqa: E402

from src.api.app import app  # noqa: E402
from src.api.auth import create_access_token  # noqa: E402
from src.api.db.engine import get_db  # noqa: E402
from src.api.db.models import (  # noqa: E402
    ApiSessionRecord,
    Organization,
    User,
    Workspace,
    WorkspaceMembership,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uid() -> str:
    return str(uuid.uuid4())


def _auth_header(user_id: str, role: str = "user") -> dict[str, str]:
    token = create_access_token(user_id=user_id, role=role)
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Fixture: seeded DB with org, workspace, members, and sessions
# ---------------------------------------------------------------------------


@pytest.fixture()
def seeded_client(engine):
    """Return a TestClient with org, workspace, two users (one member, one removed), admin, and sessions."""
    factory = async_sessionmaker(engine, expire_on_commit=False)

    org_id = _uid()
    ws_id = _uid()
    member_user_id = _uid()
    removed_user_id = _uid()
    admin_user_id = _uid()
    other_user_id = _uid()
    member_session_id = _uid()
    removed_session_id = _uid()

    async def _seed():
        async with factory() as session:
            # ── Org ────────────────────────────────────────────────────
            session.add(Organization(id=org_id, name="Test Org", slug="test-org"))

            # ── Workspace ──────────────────────────────────────────────
            session.add(
                Workspace(
                    id=ws_id,
                    org_id=org_id,
                    name="Test Workspace",
                    is_active=True,
                )
            )

            # ── Users ───────────────────────────────────────────────────
            session.add_all(
                [
                    User(
                        id=member_user_id,
                        username="member",
                        email="member@test.com",
                        password_hash="h",
                        org_id=org_id,
                    ),
                    User(
                        id=removed_user_id,
                        username="removed",
                        email="removed@test.com",
                        password_hash="h",
                        org_id=org_id,
                    ),
                    User(
                        id=admin_user_id,
                        username="admin",
                        email="admin@test.com",
                        password_hash="h",
                        role="admin",
                        org_id=org_id,
                    ),
                    User(
                        id=other_user_id,
                        username="other",
                        email="other@test.com",
                        password_hash="h",
                        org_id=org_id,
                    ),
                ]
            )

            # ── Workspace memberships ───────────────────────────────────
            # member_user is active member
            session.add(
                WorkspaceMembership(
                    id=_uid(),
                    workspace_id=ws_id,
                    user_id=member_user_id,
                    role="member",
                )
            )
            # removed_user was a member but membership is omitted (simulating removal)
            # other_user has no membership

            # ── Sessions ────────────────────────────────────────────────
            session.add_all(
                [
                    ApiSessionRecord(
                        id=member_session_id,
                        user_id=member_user_id,
                        name="Member Session",
                        state="idle",
                        workspace_id=ws_id,
                    ),
                    ApiSessionRecord(
                        id=removed_session_id,
                        user_id=removed_user_id,
                        name="Removed User Session",
                        state="idle",
                        workspace_id=ws_id,
                    ),
                ]
            )

            await session.commit()

    asyncio.run(_seed())

    async def _override_get_db():
        async with factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db] = _override_get_db
    try:
        from unittest.mock import MagicMock
        from unittest.mock import patch as _patch

        with _patch("src.api.session_bridge._build_llm", return_value=MagicMock()):
            with TestClient(app, raise_server_exceptions=False) as client:
                yield (
                    client,
                    org_id,
                    ws_id,
                    member_user_id,
                    removed_user_id,
                    admin_user_id,
                    other_user_id,
                    member_session_id,
                    removed_session_id,
                )
    finally:
        app.dependency_overrides.clear()


# ===========================================================================
# Session creation workspace isolation
# ===========================================================================


class TestSessionCreationWorkspaceIsolation:
    """Verify workspace membership is enforced at session creation time."""

    def test_member_can_create_session_in_workspace(self, seeded_client):
        client, _, ws_id, member_id, *_ = seeded_client
        r = client.post(
            "/api/v1/sessions",
            json={"name": "New Workspace Session", "workspace_id": ws_id},
            headers=_auth_header(member_id),
        )
        assert r.status_code == 201, f"Expected 201, got {r.status_code}: {r.json()}"
        data = r.json()["data"]
        assert data["workspace_id"] == ws_id

    def test_non_member_cannot_create_session_in_workspace(self, seeded_client):
        client, _, ws_id, _, _, _, other_id, *_ = seeded_client
        r = client.post(
            "/api/v1/sessions",
            json={"name": "Hijack Session", "workspace_id": ws_id},
            headers=_auth_header(other_id),
        )
        assert r.status_code == 403, f"Expected 403, got {r.status_code}: {r.json()}"

    def test_admin_can_create_session_without_membership(self, seeded_client):
        client, _, ws_id, _, _, admin_id, *_ = seeded_client
        r = client.post(
            "/api/v1/sessions",
            json={"name": "Admin Session", "workspace_id": ws_id},
            headers=_auth_header(admin_id, role="admin"),
        )
        assert r.status_code == 201, f"Expected 201, got {r.status_code}: {r.json()}"

    def test_create_session_with_nonexistent_workspace(self, seeded_client):
        client, _, _, member_id, *_ = seeded_client
        r = client.post(
            "/api/v1/sessions",
            json={"name": "Bad Session", "workspace_id": _uid()},
            headers=_auth_header(member_id),
        )
        assert r.status_code == 404, f"Expected 404, got {r.status_code}: {r.json()}"

    def test_create_session_without_workspace_is_personal(self, seeded_client):
        client, _, _, member_id, *_ = seeded_client
        r = client.post(
            "/api/v1/sessions",
            json={"name": "Personal Session"},
            headers=_auth_header(member_id),
        )
        assert r.status_code == 201, f"Expected 201, got {r.status_code}: {r.json()}"
        assert r.json()["data"]["workspace_id"] is None


# ===========================================================================
# Session access workspace isolation (removed member)
# ===========================================================================


class TestSessionAccessWorkspaceIsolation:
    """Verify removed workspace members lose access to existing sessions."""

    def test_removed_member_cannot_get_workspace_session(self, seeded_client):
        client, _, _, _, removed_id, _, _, _, removed_session_id = seeded_client
        r = client.get(
            f"/api/v1/sessions/{removed_session_id}",
            headers=_auth_header(removed_id),
        )
        assert r.status_code == 403, f"Expected 403, got {r.status_code}: {r.json()}"

    def test_removed_member_cannot_patch_workspace_session(self, seeded_client):
        client, _, _, _, removed_id, _, _, _, removed_session_id = seeded_client
        r = client.patch(
            f"/api/v1/sessions/{removed_session_id}",
            json={"name": "Renamed by ex-member"},
            headers=_auth_header(removed_id),
        )
        assert r.status_code == 403, f"Expected 403, got {r.status_code}: {r.json()}"

    def test_removed_member_cannot_delete_workspace_session(self, seeded_client):
        client, _, _, _, removed_id, _, _, _, removed_session_id = seeded_client
        r = client.delete(
            f"/api/v1/sessions/{removed_session_id}",
            headers=_auth_header(removed_id),
        )
        assert r.status_code == 403, f"Expected 403, got {r.status_code}: {r.json()}"

    def test_removed_member_cannot_restore_workspace_session(self, seeded_client):
        client, _, _, _, removed_id, _, _, _, removed_session_id = seeded_client
        r = client.post(
            f"/api/v1/sessions/{removed_session_id}/restore",
            headers=_auth_header(removed_id),
        )
        assert r.status_code == 403, f"Expected 403, got {r.status_code}: {r.json()}"

    def test_current_member_can_still_access_workspace_session(self, seeded_client):
        client, _, _, member_id, _, _, _, member_session_id, _ = seeded_client
        r = client.get(
            f"/api/v1/sessions/{member_session_id}",
            headers=_auth_header(member_id),
        )
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.json()}"

    def test_admin_can_access_workspace_session_without_membership(self, seeded_client):
        client, _, _, _, _, admin_id, _, member_session_id, _ = seeded_client
        r = client.get(
            f"/api/v1/sessions/{member_session_id}",
            headers=_auth_header(admin_id, role="admin"),
        )
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.json()}"


# ===========================================================================
# Message endpoint workspace isolation
# ===========================================================================


class TestMessageWorkspaceIsolation:
    """Verify message endpoints enforce workspace membership."""

    def test_removed_member_cannot_send_message(self, seeded_client):
        client, _, _, _, removed_id, _, _, _, removed_session_id = seeded_client
        r = client.post(
            f"/api/v1/sessions/{removed_session_id}/messages",
            json={"content": "hello from ex-member"},
            headers=_auth_header(removed_id),
        )
        assert r.status_code == 403, f"Expected 403, got {r.status_code}: {r.json()}"

    def test_removed_member_cannot_list_messages(self, seeded_client):
        client, _, _, _, removed_id, _, _, _, removed_session_id = seeded_client
        r = client.get(
            f"/api/v1/sessions/{removed_session_id}/messages",
            headers=_auth_header(removed_id),
        )
        assert r.status_code == 403, f"Expected 403, got {r.status_code}: {r.json()}"

    def test_removed_member_cannot_clear_history(self, seeded_client):
        client, _, _, _, removed_id, _, _, _, removed_session_id = seeded_client
        r = client.delete(
            f"/api/v1/sessions/{removed_session_id}/messages",
            headers=_auth_header(removed_id),
        )
        assert r.status_code == 403, f"Expected 403, got {r.status_code}: {r.json()}"

    def test_current_member_can_send_message(self, seeded_client):
        client, _, _, member_id, _, _, _, member_session_id, _ = seeded_client
        r = client.post(
            f"/api/v1/sessions/{member_session_id}/messages",
            json={"content": "hello from member"},
            headers=_auth_header(member_id),
        )
        # 202 accepted (async) or 200 (sync) — both indicate permission granted
        assert r.status_code in (200, 202), f"Unexpected {r.status_code}: {r.json()}"

    def test_admin_can_send_message_without_membership(self, seeded_client):
        client, _, _, _, _, admin_id, _, member_session_id, _ = seeded_client
        r = client.post(
            f"/api/v1/sessions/{member_session_id}/messages",
            json={"content": "hello from admin"},
            headers=_auth_header(admin_id, role="admin"),
        )
        assert r.status_code in (200, 202), f"Unexpected {r.status_code}: {r.json()}"


# ===========================================================================
# Session list workspace isolation
# ===========================================================================


class TestSessionListWorkspaceIsolation:
    """Verify list endpoints do not leak workspace sessions to removed members."""

    def test_removed_member_still_sees_workspace_session_in_list(self, seeded_client):
        """List is user-scoped; ownership is preserved even after workspace removal.

        The isolation boundary is on ACCESS (GET /sessions/{id}, messages, etc.)
        not on listing. A removed member can see they own the session but
        cannot open it.
        """
        client, _, _, _, removed_id, _, _, _, removed_session_id = seeded_client
        r = client.get("/api/v1/sessions", headers=_auth_header(removed_id))
        assert r.status_code == 200, f"List failed: {r.json()}"
        items = r.json()["data"]["items"]
        session_ids = {item["id"] for item in items}
        assert (
            removed_session_id in session_ids
        ), "Removed member should still see own session in list"

    def test_current_member_sees_own_workspace_session_in_list(self, seeded_client):
        client, _, _, member_id, _, _, _, member_session_id, _ = seeded_client
        r = client.get("/api/v1/sessions", headers=_auth_header(member_id))
        assert r.status_code == 200, f"List failed: {r.json()}"
        items = r.json()["data"]["items"]
        session_ids = {item["id"] for item in items}
        assert member_session_id in session_ids, "Current member should see own workspace session"
