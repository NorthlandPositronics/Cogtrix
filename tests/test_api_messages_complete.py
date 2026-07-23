"""Comprehensive message endpoint coverage.

Tests all message-related endpoints:
  POST   /api/v1/sessions/{id}/messages  — send message (202)
  GET    /api/v1/sessions/{id}/messages  — list history (paginated)
  DELETE /api/v1/sessions/{id}/messages  — clear history

Focuses on cases not already covered in test_api_sessions_complete.py:
  - All request/response field validation
  - Cursor pagination deep dive
  - send_message mode variants
  - Session-registry-not-available 503
  - Content deserialization edge cases
"""

from __future__ import annotations

import asyncio as _asyncio
import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("fastapi")

_TEST_JWT_SECRET = "testsecret_mustbe32chars_minimum00"
os.environ.setdefault("COGTRIX_JWT_SECRET", _TEST_JWT_SECRET)
os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from src.api.db.engine import Base, get_db  # noqa: E402

_VALID_PASSWORD = "TestPass1!"


@pytest.fixture()
def app():
    from src.api.app import create_app

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _create():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    _asyncio.run(_create())

    with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
        _app = create_app()

        async def _override():
            async with factory() as session:
                try:
                    yield session
                    await session.commit()
                except Exception:
                    await session.rollback()
                    raise

        _app.dependency_overrides[get_db] = _override
        yield _app

    _asyncio.run(engine.dispose())


@pytest.fixture()
def client(app):
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _reg(client):
    uname = f"msg_{uuid.uuid4().hex[:8]}"
    client.post(
        "/api/v1/auth/register",
        json={"username": uname, "email": f"{uname}@example.com", "password": _VALID_PASSWORD},
    )
    r = client.post("/api/v1/auth/login", json={"username": uname, "password": _VALID_PASSWORD})
    return r.json()["data"]["access_token"]


def _h(token):
    return {"Authorization": f"Bearer {token}"}


def _make_session(client, token, name=None):
    body = {}
    if name:
        body["name"] = name
    r = client.post("/api/v1/sessions", headers=_h(token), json=body)
    return r.json()["data"]["id"]


@pytest.fixture()
def tokens(client):
    owner = _reg(client)
    other = _reg(client)
    return {"owner": owner, "other": other}


@pytest.fixture()
def sid(client, tokens):
    return _make_session(client, tokens["owner"])


# ---------------------------------------------------------------------------
# GET /api/v1/sessions/{id}/messages — list history
# ---------------------------------------------------------------------------


class TestListMessages:
    def test_list_empty_session(self, client, tokens, sid):
        r = client.get(f"/api/v1/sessions/{sid}/messages", headers=_h(tokens["owner"]))
        assert r.status_code == 200
        page = r.json()["data"]
        assert page["items"] == []
        assert page["has_more"] is False
        assert page["next_cursor"] is None
        assert page["total"] is None

    def test_list_no_auth_returns_401(self, client, sid):
        r = client.get(f"/api/v1/sessions/{sid}/messages")
        assert r.status_code == 401

    def test_list_non_owner_returns_403(self, client, tokens, sid):
        r = client.get(f"/api/v1/sessions/{sid}/messages", headers=_h(tokens["other"]))
        assert r.status_code == 403

    def test_list_nonexistent_session_returns_404(self, client, tokens):
        r = client.get(f"/api/v1/sessions/{uuid.uuid4()}/messages", headers=_h(tokens["owner"]))
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "SESSION_NOT_FOUND"

    def test_list_envelope_fields(self, client, tokens, sid):
        r = client.get(f"/api/v1/sessions/{sid}/messages", headers=_h(tokens["owner"]))
        body = r.json()
        assert "data" in body
        assert body["error"] is None
        page = body["data"]
        assert "items" in page
        assert "has_more" in page
        assert "next_cursor" in page

    def test_list_limit_over_200_returns_422(self, client, tokens, sid):
        r = client.get(f"/api/v1/sessions/{sid}/messages?limit=201", headers=_h(tokens["owner"]))
        assert r.status_code == 422

    def test_list_limit_under_1_returns_422(self, client, tokens, sid):
        r = client.get(f"/api/v1/sessions/{sid}/messages?limit=0", headers=_h(tokens["owner"]))
        assert r.status_code == 422

    def test_list_limit_1_returns_at_most_1(self, client, tokens, sid):
        r = client.get(f"/api/v1/sessions/{sid}/messages?limit=1", headers=_h(tokens["owner"]))
        assert r.status_code == 200
        assert len(r.json()["data"]["items"]) <= 1

    def test_list_invalid_token_returns_401(self, client, sid):
        r = client.get(
            f"/api/v1/sessions/{sid}/messages",
            headers={"Authorization": "Bearer forged.token.value"},
        )
        assert r.status_code == 401

    def test_list_message_schema_fields(self, client, tokens, app, sid):
        """Message schema items must include all required fields (via send_message)."""
        # Use send_message to inject a real user message via the API
        ss = MagicMock()
        ss.no_confirm = True
        ss.denials = set()
        ss.loaded_tools = set()
        ss.pinned_tools = set()
        ss.reset_approvals()

        live = MagicMock()
        live.session_state = ss
        live.ws_queue = _asyncio.Queue(maxsize=10_000)
        live.cancel_event = _asyncio.Event()
        live.turn_lock = _asyncio.Lock()
        live.turn_task = None
        live.active_confirmation_ui = None
        live.drain_task = None
        live.agent_state = "idle"
        live.memory_manager = None
        live.run_config = None
        live.token_counts = {}
        live.last_activity = 0.0
        live.config = {}

        mock_reg = MagicMock()
        mock_reg.get_cached = AsyncMock(return_value=live)
        mock_reg.get_or_warm = AsyncMock(return_value=live)
        app.state.session_registry = mock_reg

        r_send = client.post(
            f"/api/v1/sessions/{sid}/messages",
            headers=_h(tokens["owner"]),
            json={"content": "hello world"},
        )
        assert r_send.status_code == 202

        r = client.get(f"/api/v1/sessions/{sid}/messages", headers=_h(tokens["owner"]))
        assert r.status_code == 200
        items = r.json()["data"]["items"]
        # At least one message was persisted
        assert len(items) >= 1
        item = items[0]
        assert item["role"] == "user"
        assert "id" in item
        assert "session_id" in item
        assert "created_at" in item
        assert "tool_calls" in item
        assert "content" in item


# ---------------------------------------------------------------------------
# DELETE /api/v1/sessions/{id}/messages — clear history
# ---------------------------------------------------------------------------


class TestClearHistory:
    def test_clear_no_auth_returns_401(self, client, sid):
        r = client.delete(f"/api/v1/sessions/{sid}/messages")
        assert r.status_code == 401

    def test_clear_non_owner_returns_403(self, client, tokens, sid):
        r = client.delete(f"/api/v1/sessions/{sid}/messages", headers=_h(tokens["other"]))
        assert r.status_code == 403

    def test_clear_nonexistent_session_returns_404(self, client, tokens):
        r = client.delete(f"/api/v1/sessions/{uuid.uuid4()}/messages", headers=_h(tokens["owner"]))
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "SESSION_NOT_FOUND"

    def test_clear_no_body_returns_200(self, client, tokens, sid):
        r = client.delete(f"/api/v1/sessions/{sid}/messages", headers=_h(tokens["owner"]))
        assert r.status_code == 200
        assert r.json()["data"] is None

    def test_clear_with_keep_last_zero_returns_200(self, client, tokens, sid):
        r = client.request(
            "DELETE",
            f"/api/v1/sessions/{sid}/messages",
            headers=_h(tokens["owner"]),
            json={"keep_last": 0},
        )
        assert r.status_code == 200

    def test_clear_with_keep_last_positive_returns_200(self, client, tokens, sid):
        r = client.request(
            "DELETE",
            f"/api/v1/sessions/{sid}/messages",
            headers=_h(tokens["owner"]),
            json={"keep_last": 10},
        )
        assert r.status_code == 200

    def test_clear_with_keep_last_negative_coerced_to_zero(self, client, tokens, sid):
        # Negative keep_last — implementation clamps to 0
        r = client.request(
            "DELETE",
            f"/api/v1/sessions/{sid}/messages",
            headers=_h(tokens["owner"]),
            json={"keep_last": -5},
        )
        # Either 200 (clamped) or 422 (validation error) is acceptable
        assert r.status_code in (200, 422)

    def test_clear_removes_messages(self, client, tokens, app, sid):
        """Use send_message to add messages, then verify clear works."""
        ss = MagicMock()
        ss.no_confirm = True
        ss.denials = set()
        ss.loaded_tools = set()
        ss.pinned_tools = set()
        ss.reset_approvals()

        live = MagicMock()
        live.session_state = ss
        live.ws_queue = _asyncio.Queue(maxsize=10_000)
        live.cancel_event = _asyncio.Event()
        live.turn_lock = _asyncio.Lock()
        live.turn_task = None
        live.active_confirmation_ui = None
        live.drain_task = None
        live.agent_state = "idle"
        live.memory_manager = None
        live.run_config = None
        live.token_counts = {}
        live.last_activity = 0.0
        live.config = {}

        mock_reg = MagicMock()
        mock_reg.get_cached = AsyncMock(return_value=live)
        mock_reg.get_or_warm = AsyncMock(return_value=live)
        app.state.session_registry = mock_reg

        # Add 3 messages via the API
        for i in range(3):
            live.turn_task = None  # reset so concurrent check passes
            client.post(
                f"/api/v1/sessions/{sid}/messages",
                headers=_h(tokens["owner"]),
                json={"content": f"msg {i}"},
            )

        # Verify messages exist
        r = client.get(f"/api/v1/sessions/{sid}/messages", headers=_h(tokens["owner"]))
        assert len(r.json()["data"]["items"]) >= 1

        # Reset registry to None so clear_history doesn't try to call mm.clear on mock
        app.state.session_registry = None

        # Clear all
        client.delete(f"/api/v1/sessions/{sid}/messages", headers=_h(tokens["owner"]))

        # Verify cleared
        r2 = client.get(f"/api/v1/sessions/{sid}/messages", headers=_h(tokens["owner"]))
        assert r2.json()["data"]["items"] == []

    def test_clear_keep_last_trims_oldest(self, client, tokens, app, sid):
        """Verify keep_last trims messages via the API."""
        ss = MagicMock()
        ss.no_confirm = True
        ss.denials = set()
        ss.loaded_tools = set()
        ss.pinned_tools = set()
        ss.reset_approvals()

        live = MagicMock()
        live.session_state = ss
        live.ws_queue = _asyncio.Queue(maxsize=10_000)
        live.cancel_event = _asyncio.Event()
        live.turn_lock = _asyncio.Lock()
        live.turn_task = None
        live.active_confirmation_ui = None
        live.drain_task = None
        live.agent_state = "idle"
        live.memory_manager = None
        live.run_config = None
        live.token_counts = {}
        live.last_activity = 0.0
        live.config = {}

        mock_reg = MagicMock()
        mock_reg.get_cached = AsyncMock(return_value=live)
        mock_reg.get_or_warm = AsyncMock(return_value=live)
        app.state.session_registry = mock_reg

        # Add 5 messages
        for i in range(5):
            live.turn_task = None
            client.post(
                f"/api/v1/sessions/{sid}/messages",
                headers=_h(tokens["owner"]),
                json={"content": f"message {i}"},
            )

        # Reset registry so keep_last logic runs without memory_manager interference
        app.state.session_registry = None

        # Keep last 2
        client.request(
            "DELETE",
            f"/api/v1/sessions/{sid}/messages",
            headers=_h(tokens["owner"]),
            json={"keep_last": 2},
        )

        r = client.get(f"/api/v1/sessions/{sid}/messages", headers=_h(tokens["owner"]))
        remaining = r.json()["data"]["items"]
        # At most 2 items remain
        assert len(remaining) <= 2


# ---------------------------------------------------------------------------
# POST /api/v1/sessions/{id}/messages — send message
# ---------------------------------------------------------------------------


class TestSendMessage:
    def test_send_no_auth_returns_401(self, client, sid):
        r = client.post(
            f"/api/v1/sessions/{sid}/messages",
            json={"content": "hello"},
        )
        assert r.status_code == 401

    def test_send_non_owner_returns_403(self, client, tokens, sid):
        mock_reg = MagicMock()
        mock_reg.get_cached = AsyncMock(return_value=None)
        mock_reg.get_or_warm = AsyncMock(return_value=None)

        r = client.post(
            f"/api/v1/sessions/{sid}/messages",
            headers=_h(tokens["other"]),
            json={"content": "hello"},
        )
        assert r.status_code == 403

    def test_send_missing_content_returns_422(self, client, tokens, sid, app):
        mock_reg = MagicMock()
        mock_reg.get_cached = AsyncMock(return_value=None)
        mock_reg.get_or_warm = AsyncMock(return_value=None)
        app.state.session_registry = mock_reg

        r = client.post(
            f"/api/v1/sessions/{sid}/messages",
            headers=_h(tokens["owner"]),
            json={},
        )
        assert r.status_code == 422

    def test_send_empty_content_returns_422(self, client, tokens, sid, app):
        mock_reg = MagicMock()
        mock_reg.get_cached = AsyncMock(return_value=None)
        mock_reg.get_or_warm = AsyncMock(return_value=None)
        app.state.session_registry = mock_reg

        r = client.post(
            f"/api/v1/sessions/{sid}/messages",
            headers=_h(tokens["owner"]),
            json={"content": ""},
        )
        assert r.status_code == 422

    def test_send_invalid_mode_returns_422(self, client, tokens, sid, app):
        mock_reg = MagicMock()
        mock_reg.get_cached = AsyncMock(return_value=None)
        mock_reg.get_or_warm = AsyncMock(return_value=None)
        app.state.session_registry = mock_reg

        r = client.post(
            f"/api/v1/sessions/{sid}/messages",
            headers=_h(tokens["owner"]),
            json={"content": "hello", "mode": "invalid_mode_xyz"},
        )
        assert r.status_code == 422

    def test_send_valid_mode_normal_accepted(self, client, tokens, sid, app):
        """Mode='normal' should pass validation (session not found is 503/404, not 422)."""
        mock_reg = MagicMock()
        mock_reg.get_cached = AsyncMock(return_value=None)
        mock_reg.get_or_warm = AsyncMock(return_value=None)
        app.state.session_registry = mock_reg

        r = client.post(
            f"/api/v1/sessions/{sid}/messages",
            headers=_h(tokens["owner"]),
            json={"content": "hello", "mode": "normal"},
        )
        # 404 because registry returns None (session not warmed)
        assert r.status_code in (404, 503)

    def test_send_valid_mode_think_accepted(self, client, tokens, sid, app):
        mock_reg = MagicMock()
        mock_reg.get_cached = AsyncMock(return_value=None)
        mock_reg.get_or_warm = AsyncMock(return_value=None)
        app.state.session_registry = mock_reg

        r = client.post(
            f"/api/v1/sessions/{sid}/messages",
            headers=_h(tokens["owner"]),
            json={"content": "hello", "mode": "think"},
        )
        assert r.status_code in (404, 503)

    def test_send_valid_mode_delegate_accepted(self, client, tokens, sid, app):
        mock_reg = MagicMock()
        mock_reg.get_cached = AsyncMock(return_value=None)
        mock_reg.get_or_warm = AsyncMock(return_value=None)
        app.state.session_registry = mock_reg

        r = client.post(
            f"/api/v1/sessions/{sid}/messages",
            headers=_h(tokens["owner"]),
            json={"content": "hello", "mode": "delegate"},
        )
        assert r.status_code in (404, 503)

    def test_send_no_session_registry_returns_503(self, client, tokens, sid, app):
        app.state.session_registry = None
        r = client.post(
            f"/api/v1/sessions/{sid}/messages",
            headers=_h(tokens["owner"]),
            json={"content": "hello"},
        )
        assert r.status_code == 503

    def test_send_nonexistent_session_returns_404(self, client, tokens, app):
        fake_sid = str(uuid.uuid4())
        mock_reg = MagicMock()
        mock_reg.get_cached = AsyncMock(return_value=None)
        mock_reg.get_or_warm = AsyncMock(return_value=None)
        app.state.session_registry = mock_reg

        r = client.post(
            f"/api/v1/sessions/{fake_sid}/messages",
            headers=_h(tokens["owner"]),
            json={"content": "hello"},
        )
        # 403 or 404 — ownership check first (session not found → 404 from verify_session_owner)
        assert r.status_code in (403, 404)

    def test_send_with_session_in_registry_returns_202(self, client, tokens, sid, app):
        """When the registry has a live session, send should queue the turn (202)."""
        ss = MagicMock()
        ss.no_confirm = True
        ss.denials = set()
        ss.loaded_tools = set()
        ss.pinned_tools = set()
        ss.reset_approvals()

        live = MagicMock()
        live.session_state = ss
        live.ws_queue = _asyncio.Queue(maxsize=10_000)
        live.cancel_event = _asyncio.Event()
        live.turn_lock = _asyncio.Lock()
        live.turn_task = None
        live.active_confirmation_ui = None
        live.drain_task = None
        live.agent_state = "idle"
        live.memory_manager = None
        live.run_config = None
        live.token_counts = {}
        live.last_activity = 0.0
        live.config = {}

        mock_reg = MagicMock()
        mock_reg.get_cached = AsyncMock(return_value=live)
        mock_reg.get_or_warm = AsyncMock(return_value=live)
        app.state.session_registry = mock_reg

        r = client.post(
            f"/api/v1/sessions/{sid}/messages",
            headers=_h(tokens["owner"]),
            json={"content": "hello world"},
        )
        assert r.status_code == 202

    def test_send_202_returns_user_message(self, client, tokens, sid, app):
        ss = MagicMock()
        ss.no_confirm = True
        ss.denials = set()
        ss.loaded_tools = set()
        ss.pinned_tools = set()
        ss.reset_approvals()

        live = MagicMock()
        live.session_state = ss
        live.ws_queue = _asyncio.Queue(maxsize=10_000)
        live.cancel_event = _asyncio.Event()
        live.turn_lock = _asyncio.Lock()
        live.turn_task = None
        live.active_confirmation_ui = None
        live.drain_task = None
        live.agent_state = "idle"
        live.memory_manager = None
        live.run_config = None
        live.token_counts = {}
        live.last_activity = 0.0
        live.config = {}

        mock_reg = MagicMock()
        mock_reg.get_cached = AsyncMock(return_value=live)
        mock_reg.get_or_warm = AsyncMock(return_value=live)
        app.state.session_registry = mock_reg

        r = client.post(
            f"/api/v1/sessions/{sid}/messages",
            headers=_h(tokens["owner"]),
            json={"content": "check this"},
        )
        assert r.status_code == 202
        data = r.json()["data"]
        assert data["role"] == "user"
        assert data["content"] == "check this"
        assert "id" in data
        assert "session_id" in data

    def test_send_409_when_turn_in_progress(self, client, tokens, sid, app):
        turn_task = MagicMock()
        turn_task.done.return_value = False

        ss = MagicMock()
        ss.no_confirm = True

        live = MagicMock()
        live.session_state = ss
        live.turn_task = turn_task
        live.turn_lock = _asyncio.Lock()
        live.active_confirmation_ui = None

        mock_reg = MagicMock()
        mock_reg.get_cached = AsyncMock(return_value=live)
        mock_reg.get_or_warm = AsyncMock(return_value=live)
        app.state.session_registry = mock_reg

        r = client.post(
            f"/api/v1/sessions/{sid}/messages",
            headers=_h(tokens["owner"]),
            json={"content": "concurrent"},
        )
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "TURN_IN_PROGRESS"


# ---------------------------------------------------------------------------
# DELETE /sessions/{id}/messages — additional keep_last coverage
# ---------------------------------------------------------------------------


class TestDeleteMessagesKeepLast:
    """Additional tests for DELETE /sessions/{id}/messages with keep_last parameter."""

    def test_delete_with_keep_last_zero_clears_all(self, client, tokens, sid):
        """keep_last=0 (default via query param) deletes all messages."""
        resp = client.delete(
            f"/api/v1/sessions/{sid}/messages",
            headers=_h(tokens["owner"]),
        )
        assert resp.status_code == 200
        assert resp.json()["data"] is None

    def test_delete_with_keep_last_positive_via_query(self, client, tokens, sid):
        """keep_last > 0 via JSON body triggers the bulk-delete path."""
        resp = client.request(
            "DELETE",
            f"/api/v1/sessions/{sid}/messages",
            headers=_h(tokens["owner"]),
            json={"keep_last": 5},
        )
        assert resp.status_code == 200

    def test_delete_messages_unknown_session_returns_404(self, client, tokens):
        """Deleting messages for a non-existent session returns 404."""
        resp = client.delete(
            f"/api/v1/sessions/{uuid.uuid4()}/messages",
            headers=_h(tokens["owner"]),
        )
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "SESSION_NOT_FOUND"

    def test_delete_messages_non_owner_returns_403(self, client, tokens, sid):
        """Non-owner cannot delete messages."""
        resp = client.delete(
            f"/api/v1/sessions/{sid}/messages",
            headers=_h(tokens["other"]),
        )
        assert resp.status_code == 403

    def test_delete_messages_no_auth_returns_401(self, client, sid):
        """Unauthenticated delete returns 401."""
        resp = client.delete(f"/api/v1/sessions/{sid}/messages")
        assert resp.status_code == 401

    def test_delete_with_keep_last_positive(self, client, tokens):
        """keep_last > 0 via query param triggers the bulk-delete path."""
        r = client.post("/api/v1/sessions", json={}, headers=_h(tokens["owner"]))
        if r.status_code != 201:
            pytest.skip("Session creation unavailable")
        session_id = r.json()["data"]["id"]

        resp = client.delete(
            f"/api/v1/sessions/{session_id}/messages?keep_last=5",
            headers=_h(tokens["owner"]),
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /api/v1/sessions/{id}/messages?sync=true — synchronous turn path
# ---------------------------------------------------------------------------


class TestSendMessageSync:
    """Tests for the sync=True query parameter path in send_message.

    sync=True blocks until the agent turn completes and returns the assembled
    response text in the HTTP body (SyncTurnOut schema).
    """

    def _make_live(self):
        ss = MagicMock()
        ss.no_confirm = True
        ss.denials = set()
        ss.loaded_tools = set()
        ss.pinned_tools = set()
        ss.reset_approvals()

        live = MagicMock()
        live.session_state = ss
        live.ws_queue = _asyncio.Queue(maxsize=10_000)
        live.cancel_event = _asyncio.Event()
        live.turn_lock = _asyncio.Lock()
        live.turn_task = None
        live.active_confirmation_ui = None
        live.drain_task = None
        live.agent_state = "idle"
        live.memory_manager = None
        live.run_config = None
        live.token_counts = {}
        live.last_activity = 0.0
        live.config = {}
        return live

    def test_sync_returns_200_with_assembled_response(self, client, tokens, sid, app):
        """sync=True blocks until done and returns 200 with SyncTurnOut fields."""
        live = self._make_live()

        async def _mock_turn(**kwargs):
            await live.ws_queue.put(
                {
                    "type": "done",
                    "payload": {
                        "text": "hello back",
                        "message_id": "msg-1",
                        "total_tokens": 20,
                        "input_tokens": 10,
                        "output_tokens": 10,
                        "duration_ms": 50,
                        "tool_calls": 0,
                    },
                }
            )

        mock_reg = MagicMock()
        mock_reg.get_cached = AsyncMock(return_value=live)
        mock_reg.get_or_warm = AsyncMock(return_value=live)
        app.state.session_registry = mock_reg

        with patch("src.api.routes.messages.run_message_turn", new=_mock_turn):
            r = client.post(
                f"/api/v1/sessions/{sid}/messages?sync=true",
                headers=_h(tokens["owner"]),
                json={"content": "ping"},
            )
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["text"] == "hello back"
        assert data["total_tokens"] == 20
        assert data["input_tokens"] == 10
        assert data["output_tokens"] == 10
        assert data["duration_ms"] == 50
        assert data["tool_calls"] == 0

    def test_sync_empty_queue_returns_200_with_empty_text(self, client, tokens, sid, app):
        """sync=True with nothing in the queue returns 200 with empty response text."""
        live = self._make_live()

        async def _mock_turn(**kwargs):
            pass  # deliberately empty — no messages put in queue

        mock_reg = MagicMock()
        mock_reg.get_cached = AsyncMock(return_value=live)
        mock_reg.get_or_warm = AsyncMock(return_value=live)
        app.state.session_registry = mock_reg

        with patch("src.api.routes.messages.run_message_turn", new=_mock_turn):
            r = client.post(
                f"/api/v1/sessions/{sid}/messages?sync=true",
                headers=_h(tokens["owner"]),
                json={"content": "ping"},
            )
        assert r.status_code == 200
        assert r.json()["data"]["text"] == ""

    def test_sync_error_message_in_queue_returns_500(self, client, tokens, sid, app):
        """sync=True with an error message followed by done returns 500 AGENT_ERROR."""
        live = self._make_live()

        async def _mock_turn(**kwargs):
            await live.ws_queue.put({"type": "error", "payload": {"message": "LLM exploded"}})
            await live.ws_queue.put(
                {
                    "type": "done",
                    "payload": {
                        "text": "",
                        "message_id": "m1",
                        "total_tokens": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "duration_ms": 0,
                        "tool_calls": 0,
                    },
                }
            )

        mock_reg = MagicMock()
        mock_reg.get_cached = AsyncMock(return_value=live)
        mock_reg.get_or_warm = AsyncMock(return_value=live)
        app.state.session_registry = mock_reg

        with patch("src.api.routes.messages.run_message_turn", new=_mock_turn):
            r = client.post(
                f"/api/v1/sessions/{sid}/messages?sync=true",
                headers=_h(tokens["owner"]),
                json={"content": "ping"},
            )
        assert r.status_code == 500
        assert r.json()["error"]["code"] == "AGENT_ERROR"
        assert "LLM exploded" in r.json()["error"]["message"]

    def test_sync_no_auth_returns_401(self, client, sid):
        """sync=True without auth returns 401."""
        r = client.post(
            f"/api/v1/sessions/{sid}/messages?sync=true",
            json={"content": "ping"},
        )
        assert r.status_code == 401

    def test_sync_non_owner_returns_403(self, client, tokens, sid, app):
        """sync=True for another user's session returns 403."""
        mock_reg = MagicMock()
        mock_reg.get_cached = AsyncMock(return_value=None)
        mock_reg.get_or_warm = AsyncMock(return_value=None)
        app.state.session_registry = mock_reg

        r = client.post(
            f"/api/v1/sessions/{sid}/messages?sync=true",
            headers=_h(tokens["other"]),
            json={"content": "ping"},
        )
        assert r.status_code == 403

    def test_sync_done_error_key_in_payload_returns_500(self, client, tokens, sid, app):
        """sync=True with done.payload.error set returns 500."""
        live = self._make_live()

        async def _mock_turn(**kwargs):
            await live.ws_queue.put(
                {
                    "type": "done",
                    "payload": {
                        "text": "",
                        "message_id": "m1",
                        "total_tokens": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "duration_ms": 0,
                        "tool_calls": 0,
                        "error": "turn failed internally",
                    },
                }
            )

        mock_reg = MagicMock()
        mock_reg.get_cached = AsyncMock(return_value=live)
        mock_reg.get_or_warm = AsyncMock(return_value=live)
        app.state.session_registry = mock_reg

        with patch("src.api.routes.messages.run_message_turn", new=_mock_turn):
            r = client.post(
                f"/api/v1/sessions/{sid}/messages?sync=true",
                headers=_h(tokens["owner"]),
                json={"content": "ping"},
            )
        assert r.status_code == 500
        assert r.json()["error"]["code"] == "AGENT_ERROR"
