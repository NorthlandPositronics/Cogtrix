"""Cross-org isolation integration tests on real API endpoints (issue #740).

Verifies that authenticated users cannot access or modify resources
belonging to users in a different organization through the actual Cogtrix
API routes (sessions, messages, memory, tools, API keys, workspaces).

Previously, cross-org isolation was only tested on a synthetic toy route.
This module adds integration tests against every resource endpoint using
the real FastAPI application.

Coverage:
    - Sessions:      GET/PATCH/DELETE cross-org access denied (403)
    - Messages:      GET list cross-org access denied (403)
    - Memory:        GET cross-org access denied (403)
    - Session tools: GET cross-org access denied (403)
    - API keys:      GET list / DELETE revoke cross-user access denied
    - Workspaces:    GET cross-org access returns 404 (org-scoped)
    - Admin bypass:  admin can access resources across orgs
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker  # noqa: E402

from src.api.app import app  # noqa: E402
from src.api.auth import _hash_api_key, create_access_token  # noqa: E402
from src.api.db.engine import get_db  # noqa: E402
from src.api.db.models import ApiSessionRecord, Workspace  # noqa: E402
from src.api.db.repositories.api_keys import ApiKeyRepository  # noqa: E402
from src.api.db.repositories.organization import OrganizationRepository  # noqa: E402
from src.api.db.repositories.users import UserRepository  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_API_KEY_PREFIX = "cxt_"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uid() -> str:
    return str(uuid.uuid4())


def _auth_header(user_id: str, role: str = "user") -> dict[str, str]:
    """Return an Authorization header with a JWT for the given user."""
    token = create_access_token(user_id=user_id, role=role)
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Shared fixture: seeded DB with two orgs, two users, two sessions
# ---------------------------------------------------------------------------


@pytest.fixture()
def seeded_client(engine):
    """Return a TestClient with two orgs, two users, and two sessions seeded.

    Returns:
        tuple of (client, org_a_id, org_b_id, user_a_id, user_b_id,
                  session_a_id, session_b_id, admin_a_id)
    """
    factory = async_sessionmaker(engine, expire_on_commit=False)

    org_a_id = _uid()
    org_b_id = _uid()
    user_a_id = _uid()
    user_b_id = _uid()
    admin_a_id = _uid()
    session_a_id = _uid()
    session_b_id = _uid()

    async def _seed():
        async with factory() as session:
            # ── Organizations ──────────────────────────────────────────
            org_repo = OrganizationRepository(session)
            await org_repo.create(org_id=org_a_id, name="Org A", slug="org-a")
            await org_repo.create(org_id=org_b_id, name="Org B", slug="org-b")

            # ── Users ───────────────────────────────────────────────────
            user_repo = UserRepository(session)
            await user_repo.create(
                user_id=user_a_id,
                username="user_a",
                email="a@test.com",
                password_hash="h",
                org_id=org_a_id,
            )
            await user_repo.create(
                user_id=user_b_id,
                username="user_b",
                email="b@test.com",
                password_hash="h",
                org_id=org_b_id,
            )
            await user_repo.create(
                user_id=admin_a_id,
                username="admin_a",
                email="admin_a@test.com",
                password_hash="h",
                role="admin",
                org_id=org_a_id,
            )

            # ── Sessions (direct DB insert — avoids registry dependency) ─
            session.add_all(
                [
                    ApiSessionRecord(
                        id=session_a_id,
                        user_id=user_a_id,
                        name="Session A",
                        state="idle",
                    ),
                    ApiSessionRecord(
                        id=session_b_id,
                        user_id=user_b_id,
                        name="Session B",
                        state="idle",
                    ),
                ]
            )

            # ── API keys for user A ─────────────────────────────────────
            key_repo = ApiKeyRepository(session)
            raw_key_a = _API_KEY_PREFIX + "a" + "x" * 30
            key_hash_a = _hash_api_key(raw_key_a)
            await key_repo.create(
                key_id=_uid(),
                user_id=user_a_id,
                key_hash=key_hash_a,
                key_prefix=raw_key_a[:12],
                label="Key A",
            )

            await session.commit()

    asyncio.run(_seed())

    # Override get_db dependency
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
        # Patch _build_llm so session warm succeeds in CI (no LLM configured)
        from unittest.mock import MagicMock
        from unittest.mock import patch as _patch

        with _patch("src.api.session_bridge._build_llm", return_value=MagicMock()):
            with TestClient(app, raise_server_exceptions=False) as client:
                yield (
                    client,
                    org_a_id,
                    org_b_id,
                    user_a_id,
                    user_b_id,
                    session_a_id,
                    session_b_id,
                    admin_a_id,
                )
    finally:
        app.dependency_overrides.clear()


# ===========================================================================
# Session cross-org isolation
# ===========================================================================


class TestSessionCrossOrgIsolation:
    """Verify that a user from org B cannot access org A user's sessions."""

    def test_user_a_can_get_own_session(self, seeded_client):
        client, _, _, ua, _, sa, _, _ = seeded_client
        r = client.get(f"/api/v1/sessions/{sa}", headers=_auth_header(ua))
        assert r.status_code == 200, f"GET own session failed: {r.json()}"

    def test_user_b_cannot_get_user_a_session(self, seeded_client):
        client, _, _, _, ub, sa, _, _ = seeded_client
        r = client.get(f"/api/v1/sessions/{sa}", headers=_auth_header(ub))
        assert r.status_code == 403, f"Expected 403, got {r.status_code}: {r.json()}"

    def test_user_b_cannot_patch_user_a_session(self, seeded_client):
        client, _, _, _, ub, sa, _, _ = seeded_client
        r = client.patch(
            f"/api/v1/sessions/{sa}",
            json={"name": "hijacked"},
            headers=_auth_header(ub),
        )
        assert r.status_code == 403, f"Expected 403, got {r.status_code}: {r.json()}"

    def test_user_b_cannot_delete_user_a_session(self, seeded_client):
        client, _, _, _, ub, sa, _, _ = seeded_client
        r = client.delete(f"/api/v1/sessions/{sa}", headers=_auth_header(ub))
        assert r.status_code == 403, f"Expected 403, got {r.status_code}: {r.json()}"

    def test_user_b_cannot_restore_user_a_session(self, seeded_client):
        client, _, _, _, ub, sa, _, _ = seeded_client
        r = client.post(f"/api/v1/sessions/{sa}/restore", headers=_auth_header(ub))
        assert r.status_code == 403, f"Expected 403, got {r.status_code}: {r.json()}"

    def test_user_a_can_list_own_sessions(self, seeded_client):
        client, _, _, ua, _, _, _, _ = seeded_client
        r = client.get("/api/v1/sessions", headers=_auth_header(ua))
        assert r.status_code == 200, f"List sessions failed: {r.json()}"
        items = r.json()["data"]["items"]
        assert len(items) >= 1, "user_a should see at least own session"

    def test_user_b_only_sees_own_sessions_in_list(self, seeded_client):
        client, _, _, _, ub, sa, sb, _ = seeded_client
        r = client.get("/api/v1/sessions", headers=_auth_header(ub))
        assert r.status_code == 200, f"List sessions failed: {r.json()}"
        items = r.json()["data"]["items"]
        session_ids = {item["id"] for item in items}
        assert sb in session_ids, "user_b should see own session"
        assert sa not in session_ids, "user_b should NOT see user_a's session"


# ===========================================================================
# Messages cross-org isolation
# ===========================================================================


class TestMessagesCrossOrgIsolation:
    """Verify that a user from org B cannot read org A's session messages."""

    def test_user_b_cannot_list_messages_for_user_a_session(self, seeded_client):
        client, _, _, _, ub, sa, _, _ = seeded_client
        r = client.get(f"/api/v1/sessions/{sa}/messages", headers=_auth_header(ub))
        assert r.status_code == 403, f"Expected 403, got {r.status_code}: {r.json()}"

    def test_user_b_cannot_clear_history_for_user_a_session(self, seeded_client):
        client, _, _, _, ub, sa, _, _ = seeded_client
        r = client.delete(f"/api/v1/sessions/{sa}/messages", headers=_auth_header(ub))
        assert r.status_code == 403, f"Expected 403, got {r.status_code}: {r.json()}"

    def test_user_b_cannot_send_message_to_user_a_session(self, seeded_client):
        client, _, _, _, ub, sa, _, _ = seeded_client
        r = client.post(
            f"/api/v1/sessions/{sa}/messages",
            json={"content": "hello from org B"},
            headers=_auth_header(ub),
        )
        # verify_session_owner raises 403 before any registry/turn logic
        assert r.status_code == 403, f"Expected 403, got {r.status_code}: {r.json()}"


# ===========================================================================
# Memory cross-org isolation
# ===========================================================================


class TestMemoryCrossOrgIsolation:
    """Verify that a user from org B cannot access org A's session memory."""

    def test_user_b_cannot_get_memory_for_user_a_session(self, seeded_client):
        client, _, _, _, ub, sa, _, _ = seeded_client
        r = client.get(f"/api/v1/sessions/{sa}/memory", headers=_auth_header(ub))
        assert r.status_code == 403, f"Expected 403, got {r.status_code}: {r.json()}"

    def test_user_b_cannot_clear_memory_for_user_a_session(self, seeded_client):
        client, _, _, _, ub, sa, _, _ = seeded_client
        r = client.delete(f"/api/v1/sessions/{sa}/memory", headers=_auth_header(ub))
        assert r.status_code == 403, f"Expected 403, got {r.status_code}: {r.json()}"

    def test_user_b_cannot_switch_memory_mode_for_user_a_session(self, seeded_client):
        client, _, _, _, ub, sa, _, _ = seeded_client
        r = client.patch(
            f"/api/v1/sessions/{sa}/memory",
            json={"mode": "code"},
            headers=_auth_header(ub),
        )
        assert r.status_code == 403, f"Expected 403, got {r.status_code}: {r.json()}"


# ===========================================================================
# Session tools cross-org isolation
# ===========================================================================


class TestSessionToolsCrossOrgIsolation:
    """Verify that a user from org B cannot access org A's session tools."""

    def test_user_b_cannot_get_session_tools_for_user_a_session(self, seeded_client):
        client, _, _, _, ub, sa, _, _ = seeded_client
        r = client.get(f"/api/v1/sessions/{sa}/tools", headers=_auth_header(ub))
        assert r.status_code == 403, f"Expected 403, got {r.status_code}: {r.json()}"

    def test_user_b_cannot_patch_session_tools_for_user_a_session(self, seeded_client):
        client, _, _, _, ub, sa, _, _ = seeded_client
        r = client.patch(
            f"/api/v1/sessions/{sa}/tools",
            json={"load": ["search_file"]},
            headers=_auth_header(ub),
        )
        assert r.status_code == 403, f"Expected 403, got {r.status_code}: {r.json()}"


# ===========================================================================
# API keys cross-user isolation
# ===========================================================================


class TestApiKeysCrossUserIsolation:
    """Verify that a user cannot access another user's API keys."""

    def test_user_b_cannot_list_user_a_api_keys(self, seeded_client):
        client, _, _, _, ub, _, _, _ = seeded_client
        r = client.get("/api/v1/auth/api-keys", headers=_auth_header(ub))
        assert r.status_code == 200, f"List API keys failed: {r.json()}"
        items = r.json()["data"]["items"]
        # user_b should see their own keys (none), not user_a's
        assert all(
            item.get("label") != "Key A" for item in items
        ), "user_b should not see user_a's API keys"

    def test_user_b_cannot_revoke_user_a_api_key(self, seeded_client):
        client, _, _, ua, ub, _, _, _ = seeded_client
        # Get user_a's key id
        r = client.get("/api/v1/auth/api-keys", headers=_auth_header(ua))
        assert r.status_code == 200
        items = r.json()["data"]["items"]
        assert len(items) >= 1, "user_a should have at least one API key"
        key_a_id = items[0]["id"]

        # Try to revoke user_a's key as user_b — should get 403 (FORBIDDEN)
        r = client.delete(
            f"/api/v1/auth/api-keys/{key_a_id}",
            headers=_auth_header(ub),
        )
        assert r.status_code == 403, f"Expected 403, got {r.status_code}: {r.json()}"


# ===========================================================================
# Workspaces cross-org isolation (org-scoped, admin-only)
# ===========================================================================


class TestWorkspacesCrossOrgIsolation:
    """Verify org-scoped isolation for workspace endpoints (admin-only)."""

    def test_admin_from_org_a_cannot_access_org_b_workspace(self, seeded_client, engine):
        """Admin from org A gets 404/403 when trying to GET org B's workspace."""
        client, _, org_b, _, _, _, _, admin_a = seeded_client

        workspace_b_id = _uid()
        factory = async_sessionmaker(engine, expire_on_commit=False)

        # Insert a workspace belonging to org B
        async def _seed_ws():
            async with factory() as session:
                ws = Workspace(
                    id=workspace_b_id,
                    org_id=org_b,
                    name="Workspace B",
                    description="Org B workspace",
                    is_active=True,
                )
                session.add(ws)
                await session.commit()

        asyncio.run(_seed_ws())

        # Admin from org A tries to access org B's workspace
        r = client.get(
            f"/api/v1/workspaces/{workspace_b_id}",
            headers=_auth_header(admin_a, role="admin"),
        )
        # Workspace route uses _get_ws_in_org which checks org scoping
        # and returns 404 (not 403) for wrong org
        assert r.status_code in (403, 404), f"Expected 403 or 404, got {r.status_code}: {r.json()}"


# ===========================================================================
# Admin bypass: admin can access any org's resources
# ===========================================================================


class TestAdminBypass:
    """Verify that admin users can access resources across orgs."""

    def test_admin_a_can_access_user_b_session(self, seeded_client):
        client, _, _, _, _, _, sb, admin_a = seeded_client
        r = client.get(f"/api/v1/sessions/{sb}", headers=_auth_header(admin_a, role="admin"))
        assert r.status_code == 200, f"Admin should bypass: got {r.status_code}: {r.json()}"

    def test_admin_a_can_list_all_sessions(self, seeded_client):
        client, _, _, _, _, _, _, admin_a = seeded_client
        r = client.get("/api/v1/sessions", headers=_auth_header(admin_a, role="admin"))
        assert r.status_code == 200, f"Admin list sessions failed: {r.json()}"
        items = r.json()["data"]["items"]
        assert len(items) >= 2, "Admin should see sessions from all users"

    def test_admin_a_can_access_user_b_session_messages(self, seeded_client):
        client, _, _, _, _, _, sb, admin_a = seeded_client
        r = client.get(
            f"/api/v1/sessions/{sb}/messages",
            headers=_auth_header(admin_a, role="admin"),
        )
        assert r.status_code == 200, f"Admin should bypass: got {r.status_code}: {r.json()}"

    def test_admin_a_can_access_user_b_session_memory(self, seeded_client):
        client, _, _, _, _, _, sb, admin_a = seeded_client
        r = client.get(
            f"/api/v1/sessions/{sb}/memory",
            headers=_auth_header(admin_a, role="admin"),
        )
        assert r.status_code == 200, f"Admin should bypass: got {r.status_code}: {r.json()}"


# ===========================================================================
# Unauthenticated access
# ===========================================================================


class TestUnauthenticatedAccess:
    """Verify that unauthenticated requests are rejected on real endpoints."""

    def test_unauthenticated_get_session_returns_401(self, seeded_client):
        client, _, _, _, _, sa, _, _ = seeded_client
        r = client.get(f"/api/v1/sessions/{sa}")
        assert r.status_code == 401, f"Expected 401, got {r.status_code}"

    def test_unauthenticated_get_messages_returns_401(self, seeded_client):
        client, _, _, _, _, sa, _, _ = seeded_client
        r = client.get(f"/api/v1/sessions/{sa}/messages")
        assert r.status_code == 401, f"Expected 401, got {r.status_code}"

    def test_unauthenticated_get_memory_returns_401(self, seeded_client):
        client, _, _, _, _, sa, _, _ = seeded_client
        r = client.get(f"/api/v1/sessions/{sa}/memory")
        assert r.status_code == 401, f"Expected 401, got {r.status_code}"

    def test_unauthenticated_get_api_keys_returns_401(self, seeded_client):
        client, _, _, _, _, _, _, _ = seeded_client
        r = client.get("/api/v1/auth/api-keys")
        assert r.status_code == 401, f"Expected 401, got {r.status_code}"
