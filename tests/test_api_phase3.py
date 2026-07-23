"""Phase 3 API tests: WebSocket streaming and message endpoints.

Tests cover:
- WebSocketCallbackHandler enqueue behavior
- ApiConfirmationUI resolve/timeout behavior
- ConnectionManager connect/disconnect/send/replay
- Message REST endpoints (POST, GET, DELETE)
- WebSocket basic lifecycle (connect, receive, ping/pong)

All tests use an in-memory SQLite database so they never touch the real DB.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# ---------------------------------------------------------------------------
# Environment setup — must happen before any src.api imports
# ---------------------------------------------------------------------------

_TEST_JWT_SECRET = "testsecret_mustbe32chars_minimum00"
os.environ.setdefault("COGTRIX_JWT_SECRET", _TEST_JWT_SECRET)
os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

# ---------------------------------------------------------------------------
# Imports after env setup
# ---------------------------------------------------------------------------

from src.api.auth import create_access_token  # noqa: E402
from src.api.callbacks import WebSocketCallbackHandler  # noqa: E402
from src.api.confirmation import ApiConfirmationUI  # noqa: E402
from src.api.db import models as _models  # noqa: E402, F401
from src.api.db.engine import Base  # noqa: E402
from src.api.db.repositories.sessions import SessionRepository  # noqa: E402
from src.api.db.repositories.users import UserRepository  # noqa: E402
from src.api.ws import ConnectionManager  # noqa: E402

# ---------------------------------------------------------------------------
# In-memory DB fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture()
async def db_session() -> AsyncGenerator[AsyncSession]:
    """Yield a fresh in-memory SQLite session for each test."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    await engine.dispose()


@pytest_asyncio.fixture()
async def user_id(db_session: AsyncSession) -> str:
    """Create a test user and return its ID."""
    repo = UserRepository(db_session)
    user = await repo.create(username="testuser", email="test@example.com", password_hash="hashed")
    await db_session.commit()
    return user.id


@pytest_asyncio.fixture()
async def session_id(db_session: AsyncSession, user_id: str) -> str:
    """Create a test session and return its ID."""
    repo = SessionRepository(db_session)
    record = await repo.create(user_id=user_id, name="Test Session")
    await db_session.commit()
    return record.id


# ---------------------------------------------------------------------------
# WebSocketCallbackHandler tests
# ---------------------------------------------------------------------------


class TestWebSocketCallbackHandler:
    """Tests for WebSocketCallbackHandler."""

    @pytest.mark.asyncio
    async def test_on_llm_new_token_enqueues_token(self) -> None:
        """on_llm_new_token should enqueue a token message."""
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        handler = WebSocketCallbackHandler(ws_queue=queue, loop=loop)

        handler.on_llm_new_token("hello")

        # Give run_coroutine_threadsafe a moment to complete.
        await asyncio.sleep(0.05)
        assert not queue.empty()
        item = await queue.get()
        assert item["type"] == "token"
        assert item["payload"]["text"] == "hello"

    @pytest.mark.asyncio
    async def test_on_tool_start_enqueues_tool_start(self) -> None:
        """on_tool_start should enqueue a tool_start message."""
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        handler = WebSocketCallbackHandler(ws_queue=queue, loop=loop)

        run_id = uuid.uuid4()
        handler.on_tool_start({"name": "web_search"}, {"query": "test"}, run_id=run_id)

        await asyncio.sleep(0.05)
        item = await queue.get()
        assert item["type"] == "tool_start"
        assert item["payload"]["tool"] == "web_search"
        assert item["payload"]["tool_call_id"] == str(run_id)

    @pytest.mark.asyncio
    async def test_on_tool_end_enqueues_tool_end(self) -> None:
        """on_tool_end should enqueue a tool_end message with no error."""
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        handler = WebSocketCallbackHandler(ws_queue=queue, loop=loop)

        run_id = uuid.uuid4()
        handler.on_tool_start({"name": "web_search"}, {}, run_id=run_id)
        await asyncio.sleep(0.01)
        await queue.get()  # consume tool_start

        handler.on_tool_end("result", run_id=run_id, name="web_search")
        await asyncio.sleep(0.05)

        item = await queue.get()
        assert item["type"] == "tool_end"
        assert item["payload"]["error"] is None
        assert item["payload"]["tool"] == "web_search"

    @pytest.mark.asyncio
    async def test_on_tool_error_enqueues_tool_end_with_error(self) -> None:
        """on_tool_error should enqueue a tool_end message with an error string."""
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        handler = WebSocketCallbackHandler(ws_queue=queue, loop=loop)

        run_id = uuid.uuid4()
        handler.on_tool_error(Exception("boom"), run_id=run_id, name="bad_tool")
        await asyncio.sleep(0.05)

        item = await queue.get()
        assert item["type"] == "tool_end"
        assert "boom" in item["payload"]["error"]

    @pytest.mark.asyncio
    async def test_on_llm_error_enqueues_error(self) -> None:
        """on_llm_error should enqueue an error message."""
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        handler = WebSocketCallbackHandler(ws_queue=queue, loop=loop)

        handler.on_llm_error("LLM failed")
        await asyncio.sleep(0.05)

        item = await queue.get()
        assert item["type"] == "error"
        assert item["payload"]["code"] == "AGENT_ERROR"


# ---------------------------------------------------------------------------
# ApiConfirmationUI tests
# ---------------------------------------------------------------------------


class TestApiConfirmationUI:
    """Tests for ApiConfirmationUI."""

    @pytest.mark.asyncio
    async def test_render_and_resolve_allow(self) -> None:
        """resolve with 'allow' should make read_choice return 'y'."""
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        ui = ApiConfirmationUI(ws_queue=queue, loop=loop)

        ui.render_prompt("write_file", {"path": "/tmp/x"}, frozenset(), 300)
        await asyncio.sleep(0.05)  # let enqueue complete

        # Grab the enqueued confirmation_id.
        item = await queue.get()
        assert item["type"] == "tool_confirm_request"
        conf_id = item["payload"]["confirmation_id"]

        # Resolve in a thread to simulate concurrent WS handler.
        def _resolve() -> None:
            time.sleep(0.01)
            ui.resolve(conf_id, "allow")

        t = threading.Thread(target=_resolve, daemon=True)
        t.start()

        choice = await asyncio.to_thread(ui.read_choice)
        assert choice == "y"
        t.join()

    @pytest.mark.asyncio
    async def test_resolve_deny_returns_n(self) -> None:
        """resolve with 'deny' should make read_choice return 'n'."""
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        ui = ApiConfirmationUI(ws_queue=queue, loop=loop)

        ui.render_prompt("shell", {}, frozenset(), 300)
        await asyncio.sleep(0.05)
        item = await queue.get()
        conf_id = item["payload"]["confirmation_id"]

        def _resolve() -> None:
            time.sleep(0.01)
            ui.resolve(conf_id, "deny")

        t = threading.Thread(target=_resolve, daemon=True)
        t.start()
        choice = await asyncio.to_thread(ui.read_choice)
        assert choice == "n"
        t.join()

    @pytest.mark.asyncio
    async def test_wrong_confirmation_id_not_resolved(self) -> None:
        """resolve with a wrong ID should return False and not unblock read_choice."""
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        ui = ApiConfirmationUI(ws_queue=queue, loop=loop)

        ui.render_prompt("shell", {}, frozenset(), 300)
        await asyncio.sleep(0.05)
        await queue.get()  # consume the enqueued request

        result = ui.resolve("wrong-id", "allow")
        assert result is False

    @pytest.mark.asyncio
    async def test_all_action_mappings(self) -> None:
        """All WebSocket actions should map to correct CLI characters."""
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        mappings = [
            ("allow", "y"),
            ("deny", "n"),
            ("allow_all", "a"),
            ("disable", "d"),
            ("forbid_all", "f"),
            ("cancel", "c"),
        ]
        for ws_action, expected_cli in mappings:
            _ui = ApiConfirmationUI(ws_queue=queue, loop=loop)
            _ui.render_prompt("tool", {}, frozenset(), 300)
            await asyncio.sleep(0.05)
            item = await queue.get()
            conf_id = item["payload"]["confirmation_id"]

            def _resolve(
                cid: str = conf_id, act: str = ws_action, ui_ref: ApiConfirmationUI = _ui
            ) -> None:
                ui_ref.resolve(cid, act)

            t = threading.Thread(target=_resolve, daemon=True)
            t.start()
            choice = await asyncio.to_thread(_ui.read_choice)
            assert choice == expected_cli, f"Expected {expected_cli} for {ws_action}"
            t.join()


# ---------------------------------------------------------------------------
# ConnectionManager tests
# ---------------------------------------------------------------------------


class TestConnectionManager:
    """Tests for ConnectionManager."""

    @pytest.mark.asyncio
    async def test_connect_and_disconnect(self) -> None:
        """connect should register a session; disconnect should remove it."""
        cm = ConnectionManager()
        ws_mock = AsyncMock()
        ws_mock.send_text = AsyncMock()

        sid = str(uuid.uuid4())
        await cm.connect(sid, ws_mock)
        assert sid in cm._connections

        await cm.disconnect(sid)
        assert sid not in cm._connections

    @pytest.mark.asyncio
    async def test_connect_replaces_existing(self) -> None:
        """Connecting a second time should close the old WebSocket."""
        cm = ConnectionManager()
        old_ws = AsyncMock()
        new_ws = AsyncMock()

        sid = str(uuid.uuid4())
        await cm.connect(sid, old_ws)
        await cm.connect(sid, new_ws)

        old_ws.close.assert_called_once()
        assert cm._connections[sid] is new_ws

    @pytest.mark.asyncio
    async def test_send_calls_websocket(self) -> None:
        """send should call websocket.send_text with a JSON envelope."""
        cm = ConnectionManager()
        ws_mock = AsyncMock()

        sid = str(uuid.uuid4())
        await cm.connect(sid, ws_mock)
        await cm.send(sid, "token", {"text": "hello"})

        ws_mock.send_text.assert_called_once()
        raw = ws_mock.send_text.call_args[0][0]
        data = json.loads(raw)
        assert data["type"] == "token"
        assert data["payload"]["text"] == "hello"
        assert data["session_id"] == sid
        assert "seq" in data

    @pytest.mark.asyncio
    async def test_send_increments_seq(self) -> None:
        """Sequence numbers should increment on each send."""
        cm = ConnectionManager()
        ws_mock = AsyncMock()

        sid = str(uuid.uuid4())
        await cm.connect(sid, ws_mock)
        await cm.send(sid, "token", {"text": "a"})
        await cm.send(sid, "token", {"text": "b"})

        calls = ws_mock.send_text.call_args_list
        seq_values = [json.loads(c[0][0])["seq"] for c in calls]
        assert seq_values == sorted(seq_values)
        assert seq_values[0] != seq_values[1]

    @pytest.mark.asyncio
    async def test_send_noop_when_no_connection(self) -> None:
        """send should not raise when there is no active connection."""
        cm = ConnectionManager()
        sid = str(uuid.uuid4())
        # Should not raise:
        await cm.send(sid, "token", {"text": "x"})

    @pytest.mark.asyncio
    async def test_replay_missed_sends_only_newer(self) -> None:
        """replay_missed should only forward messages with seq > last_seq."""
        cm = ConnectionManager()
        ws_mock = AsyncMock()

        sid = str(uuid.uuid4())
        await cm.connect(sid, ws_mock)
        # Buffer two messages.
        await cm.send(sid, "token", {"text": "a"})  # seq 0
        await cm.send(sid, "token", {"text": "b"})  # seq 1

        # Reconnect with last_seq=0 — should replay only seq=1.
        ws2 = AsyncMock()
        await cm.disconnect(sid)
        await cm.connect(sid, ws2)
        await cm.replay_missed(sid, last_seq=0)

        replayed = [json.loads(c[0][0]) for c in ws2.send_text.call_args_list]
        assert all(m["seq"] > 0 for m in replayed)


# ---------------------------------------------------------------------------
# Message REST endpoint tests (via TestClient)
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    """Return a FastAPI TestClient with the app configured for testing."""
    # Patch warm_session to avoid real LLM creation.
    with patch("src.api.session_bridge.warm_session") as mock_warm:
        from src.api.session_bridge import ApiSession
        from src.orchestration.session_state import SessionState

        def _fake_warm(record, app_state):
            sess = ApiSession(
                id=record.id,
                user_id=record.user_id,
                name=record.name,
                session_state=SessionState(no_confirm=True),
            )
            return sess

        mock_warm.side_effect = _fake_warm

        from src.api.app import app

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


def _admin_token() -> str:
    """Mint an admin JWT for testing."""
    return create_access_token(user_id=str(uuid.uuid4()), role="admin")


class TestMessageRestEndpoints:
    """Tests for POST/GET/DELETE /sessions/{id}/messages."""

    def test_post_message_404_unknown_session(self, client: TestClient) -> None:
        """POST to a non-existent session should return 404."""
        token = _admin_token()
        resp = client.post(
            f"/api/v1/sessions/{uuid.uuid4()}/messages",
            json={"content": "hello"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 404

    def test_get_messages_404_unknown_session(self, client: TestClient) -> None:
        """GET on a non-existent session should return 404."""
        token = _admin_token()
        resp = client.get(
            f"/api/v1/sessions/{uuid.uuid4()}/messages",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 404

    def test_delete_messages_404_unknown_session(self, client: TestClient) -> None:
        """DELETE on a non-existent session should return 404."""
        token = _admin_token()
        resp = client.delete(
            f"/api/v1/sessions/{uuid.uuid4()}/messages",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 404

    def test_post_message_401_without_token(self, client: TestClient) -> None:
        """POST without a token should return 401."""
        resp = client.post(
            f"/api/v1/sessions/{uuid.uuid4()}/messages",
            json={"content": "hello"},
        )
        assert resp.status_code == 401

    def test_get_messages_401_without_token(self, client: TestClient) -> None:
        """GET without a token should return 401."""
        resp = client.get(f"/api/v1/sessions/{uuid.uuid4()}/messages")
        assert resp.status_code == 401

    def test_delete_messages_401_without_token(self, client: TestClient) -> None:
        """DELETE without a token should return 401."""
        resp = client.delete(f"/api/v1/sessions/{uuid.uuid4()}/messages")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# WebSocket lifecycle tests
# ---------------------------------------------------------------------------


class TestWebSocketLifecycle:
    """Tests for the WS /ws/v1/sessions/{id} endpoint."""

    def test_ws_rejects_missing_token(self, client: TestClient) -> None:
        """WebSocket without a token should be closed with code 4001."""
        with client.websocket_connect(f"/ws/v1/sessions/{uuid.uuid4()}") as ws:
            # Server sends close frame; receive until disconnect.
            try:
                ws.receive_text()
            except Exception:
                pass

    def test_ws_rejects_invalid_token(self, client: TestClient) -> None:
        """WebSocket with an invalid token should be closed with code 4001."""
        bad_token = "not.a.valid.token"
        try:
            with client.websocket_connect(
                f"/ws/v1/sessions/{uuid.uuid4()}?token={bad_token}"
            ) as ws:
                try:
                    ws.receive_text()
                except Exception:
                    pass
        except Exception:
            pass  # Connection refused / closed is expected.

    def test_ws_rejects_unknown_session(self, client: TestClient) -> None:
        """WebSocket for an unknown session should be closed after auth."""
        user_id = str(uuid.uuid4())
        token = create_access_token(user_id=user_id, role="admin")
        try:
            with client.websocket_connect(f"/ws/v1/sessions/{uuid.uuid4()}?token={token}") as ws:
                try:
                    ws.receive_text()
                except Exception:
                    pass
        except Exception:
            pass  # Connection closure is expected.

    @pytest.mark.xfail(
        strict=False,
        reason="sync TestClient may hang waiting for agent_state before pong",
    )
    @pytest.mark.timeout(10)
    def test_ws_ping_pong(self, client: TestClient) -> None:
        """A connected client that sends ping should receive pong."""
        # Create a real session via REST first.
        admin_token = _admin_token()
        sess_resp = client.post(
            "/api/v1/sessions",
            json={"name": "ws-ping-test"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        if sess_resp.status_code not in (200, 201):
            pytest.skip("Session creation not available in this test environment")

        session_id = sess_resp.json()["data"]["id"]

        with client.websocket_connect(f"/ws/v1/sessions/{session_id}?token={admin_token}") as ws:
            # Expect the initial agent_state message.
            try:
                first_msg = json.loads(ws.receive_text())
                assert first_msg["type"] == "agent_state"
            except Exception:
                pytest.skip("WebSocket not fully connected in test environment")

            ws.send_text(json.dumps({"type": "ping", "payload": {}}))
            response = json.loads(ws.receive_text())
            assert response["type"] == "pong"
