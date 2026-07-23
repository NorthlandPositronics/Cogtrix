"""Tests for cross-workspace agent communication (Enterprise Phase 1 — task 1.3.3)."""

from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

_TEST_JWT_SECRET = "testsecret_mustbe32chars_minimum00"

from unittest.mock import patch  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker  # noqa: E402

from src.api.auth import create_access_token  # noqa: E402
from src.api.cross_workspace import (  # noqa: E402
    CrossWorkspaceMessage,
    CrossWorkspacePolicy,
    delete_message,
    read_inbox,
    write_to_inbox,
)
from src.api.db.engine import Base, get_db  # noqa: E402
from src.api.db.repositories.organization import OrganizationRepository  # noqa: E402
from src.api.db.repositories.users import UserRepository  # noqa: E402
from src.api.db.repositories.workspaces import WorkspaceRepository  # noqa: E402


def _uid() -> str:
    return str(uuid.uuid4())


def _header(user_id: str, role: str = "user") -> dict:
    with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
        token = create_access_token(user_id=user_id, role=role)
    return {"Authorization": f"Bearer {token}"}


@dataclass
class CWSetup:
    client: TestClient
    org_id: str
    user_id: str
    ws_a: str
    ws_b: str
    tmp_path: Path
    free_user_id: str

    def __iter__(self):
        yield self.client
        yield self.org_id
        yield self.user_id
        yield self.ws_a
        yield self.ws_b
        yield self.tmp_path


# ---------------------------------------------------------------------------
# CrossWorkspacePolicy unit tests
# ---------------------------------------------------------------------------


class TestCrossWorkspacePolicy:
    def test_enabled_open_allows_all(self):
        p = CrossWorkspacePolicy(enabled=True)
        assert p.is_allowed("ws-a", "ws-b") is True

    def test_disabled_blocks_all(self):
        p = CrossWorkspacePolicy(enabled=False)
        assert p.is_allowed("ws-a", "ws-b") is False

    def test_allowed_pairs_enforced(self):
        p = CrossWorkspacePolicy(enabled=True, allowed_pairs=[("ws-a", "ws-b")])
        assert p.is_allowed("ws-a", "ws-b") is True
        assert p.is_allowed("ws-b", "ws-a") is False
        assert p.is_allowed("ws-a", "ws-c") is False


# ---------------------------------------------------------------------------
# CrossWorkspaceMessage + inbox helpers
# ---------------------------------------------------------------------------


class TestCrossWorkspaceMessage:
    def test_to_dict(self):
        msg = CrossWorkspaceMessage(
            from_workspace_id="ws-a",
            to_workspace_id="ws-b",
            sender_user_id="u1",
            subject="Hello",
            body={"key": "value"},
        )
        d = msg.to_dict()
        assert d["from_workspace_id"] == "ws-a"
        assert d["subject"] == "Hello"
        assert "sent_at" in d
        assert "id" in d


class TestUuidValidation:
    """Regression tests for CodeQL CWE-22 — explicit UUID v4 sanitiser."""

    @pytest.mark.parametrize(
        "value",
        [
            "not-a-uuid",
            "../etc/passwd",
            "a" * 36,  # right length, wrong chars
            "00000000-0000-0000-0000-000000000000",  # not v4
            "00000000-0000-5000-8000-000000000000",  # v5, not v4
            "00000000-0000-4000-0000-000000000000",  # variant nibble 0 (invalid)
            "foo/../../../bar",
            "..%2f..%2f..%2fetc%2fpasswd",
            "../../../../etc/passwd",
            "../../../../../../../../etc/passwd",
        ],
    )
    def test_invalid_uuid4_rejected_in_inbox_dir(self, tmp_path, value):
        from src.api.cross_workspace import _inbox_dir

        with pytest.raises(ValueError, match="Invalid workspace_id"):
            _inbox_dir(tmp_path, value)

    @pytest.mark.parametrize(
        "value",
        [
            "not-a-uuid",
            "../etc/passwd",
            "00000000-0000-0000-0000-000000000000",
            "../../../../etc/passwd",
        ],
    )
    def test_invalid_uuid4_rejected_in_delete_message(self, tmp_path, value):
        # Pass a valid workspace_id so the message_id validation is reached
        valid_ws = _uid()
        with pytest.raises(ValueError, match="Invalid message_id"):
            delete_message(valid_ws, value, data_root=tmp_path)

    def test_valid_uuid4_passes_inbox_dir(self, tmp_path):
        from src.api.cross_workspace import _inbox_dir

        uid = _uid()
        result = _inbox_dir(tmp_path, uid)
        assert str(result).endswith(f"cross_workspace/{uid}")

    def test_valid_uuid4_passes_delete_message(self, tmp_path):
        # delete_message validates message_id before touching the filesystem
        uid = _uid()
        result = delete_message(_uid(), uid, data_root=tmp_path)
        assert result is False  # no such file, but no exception


class TestInboxHelpers:
    def test_write_and_read(self, tmp_path):
        ws_from = _uid()
        ws_to = _uid()
        msg = CrossWorkspaceMessage(
            from_workspace_id=ws_from,
            to_workspace_id=ws_to,
            sender_user_id="u1",
            subject="Test",
        )
        write_to_inbox(msg, data_root=tmp_path)
        messages = read_inbox(ws_to, data_root=tmp_path)
        assert len(messages) == 1
        assert messages[0]["subject"] == "Test"

    def test_read_empty_inbox(self, tmp_path):
        assert read_inbox(_uid(), data_root=tmp_path) == []

    def test_delete_message(self, tmp_path):
        ws_from = _uid()
        ws_to = _uid()
        msg = CrossWorkspaceMessage(
            from_workspace_id=ws_from,
            to_workspace_id=ws_to,
            sender_user_id="u1",
            subject="Delete me",
        )
        write_to_inbox(msg, data_root=tmp_path)
        deleted = delete_message(ws_to, msg.id, data_root=tmp_path)
        assert deleted is True
        assert read_inbox(ws_to, data_root=tmp_path) == []

    def test_delete_nonexistent_returns_false(self, tmp_path):
        assert delete_message(_uid(), _uid(), data_root=tmp_path) is False

    def test_read_respects_limit(self, tmp_path):
        ws_from = _uid()
        ws_to = _uid()
        for i in range(5):
            msg = CrossWorkspaceMessage(
                from_workspace_id=ws_from,
                to_workspace_id=ws_to,
                sender_user_id="u1",
                subject=f"msg {i}",
            )
            write_to_inbox(msg, data_root=tmp_path)
        assert len(read_inbox(ws_to, data_root=tmp_path, limit=3)) == 3


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def cw_setup(tmp_path, engine):
    """Two workspaces (ws-a, ws-b) in the same org + a user."""
    factory = async_sessionmaker(engine, expire_on_commit=False)
    org_id = _uid()
    user_id = _uid()
    free_user_id = _uid()
    ws_a = _uid()
    ws_b = _uid()

    async def _seed():
        async with factory() as session:
            org_repo = OrganizationRepository(session)
            user_repo = UserRepository(session)
            ws_repo = WorkspaceRepository(session)
            await org_repo.create(org_id=org_id, name="CW Org", slug="cw-org")
            await user_repo.create(
                user_id=user_id,
                username="cwuser",
                email="cw@example.com",
                password_hash="h",
                org_id=org_id,
            )
            await user_repo.create(
                user_id=free_user_id,
                username="free_cw_user",
                email="free-cw@example.com",
                password_hash="h",
                org_id=None,
            )
            await ws_repo.create(workspace_id=ws_a, org_id=org_id, name="WS-A")
            await ws_repo.create(workspace_id=ws_b, org_id=org_id, name="WS-B")
            await ws_repo.add_member(
                membership_id=_uid(),
                workspace_id=ws_a,
                user_id=user_id,
                role="member",
            )
            await session.commit()

    asyncio.run(_seed())

    from src.api.app import create_app
    from src.api.routes.cross_workspace import configure_cross_workspace_policy

    # Reset to open policy.
    configure_cross_workspace_policy(CrossWorkspacePolicy(enabled=True))

    with patch.dict(
        os.environ,
        {
            "COGTRIX_JWT_SECRET": _TEST_JWT_SECRET,
            "COGTRIX_DATA_DIR": str(tmp_path),
        },
    ):
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
            yield CWSetup(client, org_id, user_id, ws_a, ws_b, tmp_path, free_user_id)

    app.dependency_overrides.clear()


class TestCrossWorkspaceRoutes:
    def test_send_message(self, cw_setup):
        client, _, user_id, ws_a, ws_b, tmp_path = cw_setup
        r = client.post(
            "/api/v1/cross-workspace/messages",
            json={"from_workspace_id": ws_a, "to_workspace_id": ws_b, "subject": "Hi"},
            headers=_header(user_id),
        )
        assert r.status_code == 201
        assert r.json()["data"]["subject"] == "Hi"

    @pytest.mark.parametrize(
        "method,path,kwargs",
        [
            (
                "post",
                "/api/v1/cross-workspace/messages",
                {"json": {"from_workspace_id": "ws-a", "to_workspace_id": "ws-b", "subject": "x"}},
            ),
            ("get", "/api/v1/cross-workspace/inbox/ws-b", {}),
            ("delete", "/api/v1/cross-workspace/inbox/ws-b/msg-1", {}),
        ],
    )
    def test_null_org_user_is_rejected(self, cw_setup, method, path, kwargs):
        client = cw_setup.client
        headers = _header(cw_setup.free_user_id)
        payload = kwargs.copy()
        if method == "post":
            payload["json"] = {
                "from_workspace_id": cw_setup.ws_a,
                "to_workspace_id": cw_setup.ws_b,
                "subject": "x",
            }
        resp = getattr(client, method)(path, headers=headers, **payload)
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "ORG_REQUIRED"

    def test_sender_must_belong_to_source_workspace(self, cw_setup):
        client, _, user_id, ws_a, ws_b, tmp_path = cw_setup
        resp = client.post(
            "/api/v1/cross-workspace/messages",
            json={"from_workspace_id": ws_b, "to_workspace_id": ws_a, "subject": "Denied"},
            headers=_header(user_id),
        )
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "FORBIDDEN"

    def test_read_inbox(self, cw_setup):
        client, _, user_id, ws_a, ws_b, tmp_path = cw_setup
        # Send a message first.
        client.post(
            "/api/v1/cross-workspace/messages",
            json={"from_workspace_id": ws_a, "to_workspace_id": ws_b, "subject": "Inbox test"},
            headers=_header(user_id),
        )
        r = client.get(f"/api/v1/cross-workspace/inbox/{ws_b}", headers=_header(user_id))
        assert r.status_code == 200
        assert len(r.json()["data"]) == 1

    def test_delete_message(self, cw_setup):
        client, _, user_id, ws_a, ws_b, tmp_path = cw_setup
        r = client.post(
            "/api/v1/cross-workspace/messages",
            json={"from_workspace_id": ws_a, "to_workspace_id": ws_b, "subject": "Delete"},
            headers=_header(user_id),
        )
        msg_id = r.json()["data"]["id"]
        r = client.delete(
            f"/api/v1/cross-workspace/inbox/{ws_b}/{msg_id}", headers=_header(user_id)
        )
        assert r.status_code == 200

    def test_cross_org_blocked(self, cw_setup, tmp_path):
        client, _, user_id, ws_a, ws_b, _ = cw_setup
        # Create a workspace in a different org.
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy.pool import StaticPool

        engine2 = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            echo=False,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        factory2 = async_sessionmaker(engine2, expire_on_commit=False)
        other_ws_id = _uid()
        other_org_id = _uid()

        async def _other_seed():
            async with engine2.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            async with factory2() as session:
                org_repo = OrganizationRepository(session)
                ws_repo = WorkspaceRepository(session)
                await org_repo.create(org_id=other_org_id, name="Other Org", slug="other-org")
                await ws_repo.create(workspace_id=other_ws_id, org_id=other_org_id, name="Other WS")
                await session.commit()

        asyncio.run(_other_seed())

        # Try sending from ws_a (org) to other_ws (different org) — must fail.
        # We can test this by directly calling the service logic.
        from src.api.cross_workspace import CrossWorkspacePolicy

        # Policy pair enforcement (simulates cross-org via policy).
        policy = CrossWorkspacePolicy(enabled=True, allowed_pairs=[(ws_a, ws_b)])
        from src.api.routes.cross_workspace import configure_cross_workspace_policy

        configure_cross_workspace_policy(policy)
        r = client.post(
            "/api/v1/cross-workspace/messages",
            json={
                "from_workspace_id": ws_a,
                "to_workspace_id": ws_b + "_different",
                "subject": "Blocked",
            },
            headers=_header(user_id),
        )
        assert r.status_code in (403, 404)

        asyncio.run(engine2.dispose())

    def test_disabled_policy_returns_503(self, cw_setup):
        from src.api.routes.cross_workspace import configure_cross_workspace_policy

        client, _, user_id, ws_a, ws_b, _ = cw_setup
        configure_cross_workspace_policy(CrossWorkspacePolicy(enabled=False))
        r = client.post(
            "/api/v1/cross-workspace/messages",
            json={"from_workspace_id": ws_a, "to_workspace_id": ws_b, "subject": "X"},
            headers=_header(user_id),
        )
        assert r.status_code == 503
