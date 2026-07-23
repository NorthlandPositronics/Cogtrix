"""Phase 6 API tests: Assistant Mode Dashboard endpoints.

Tests cover:
- GET /assistant/status when service not running (stopped)
- GET /assistant/status with mocked running service
- POST /assistant/stop when not running (409)
- GET /assistant/chats with mocked sessions
- GET /assistant/chats/{key}/messages with mocked session
- GET /assistant/scheduled with mocked scheduler
- PATCH /assistant/scheduled/{id} edit
- DELETE /assistant/scheduled/{id} cancel
- GET /assistant/deferred with mocked deferral manager
- DELETE /assistant/deferred/{key} cancel
- GET /assistant/contacts with mocked phonebook
- GET /assistant/guardrails admin only
- DELETE /assistant/guardrails/blacklist/{chat_id} admin only
- GET /assistant/knowledge list
- POST /assistant/knowledge/search
- DELETE /assistant/knowledge/{fact_id} admin only

State injection strategy: set app.state.* after TestClient startup.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi")

# ---------------------------------------------------------------------------
# Environment setup — must happen before any src.api imports
# ---------------------------------------------------------------------------

os.environ.setdefault("COGTRIX_JWT_SECRET", "testsecret_mustbe32chars_minimum00")
os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

# ---------------------------------------------------------------------------
# Imports after env setup
# ---------------------------------------------------------------------------

from fastapi.testclient import TestClient  # noqa: E402

from src.api.auth import create_access_token  # noqa: E402

# ---------------------------------------------------------------------------
# Token factories
# ---------------------------------------------------------------------------


def _admin_token() -> str:
    return create_access_token(user_id=str(uuid.uuid4()), role="admin")


def _user_token() -> str:
    return create_access_token(user_id=str(uuid.uuid4()), role="user")


def _admin_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_admin_token()}"}


def _user_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_user_token()}"}


# ---------------------------------------------------------------------------
# Mock data builders
# ---------------------------------------------------------------------------


@dataclass
class _MockFact:
    entity: str
    fact: str
    source_session: str
    timestamp: float
    fact_hash: str


@dataclass
class _MockScheduledMessage:
    id: str
    channel: str
    chat_id: str
    text: str
    send_at: float
    created_at: float
    recipient: str | None = None
    status: str = "pending"
    attempts: int = 0
    max_attempts: int = 3


@dataclass
class _MockDeferredRecord:
    id: str
    channel: str
    chat_id: str
    fire_at: float
    created_at: float
    pending_messages: list[dict[str, Any]] = field(default_factory=list)
    deferral_depth: int = 0
    status: str = "pending"


def _make_mock_scheduler(msgs: list[_MockScheduledMessage] | None = None) -> MagicMock:
    scheduler = MagicMock()
    queue: dict[str, Any] = {}
    if msgs:
        for m in msgs:
            queue[m.id] = m
    scheduler._queue = queue
    scheduler._lock = threading.Lock()

    def _get_pending(recipient=None, chat_id=None, include_all=False):
        result = []
        for m in queue.values():
            if not include_all and m.status != "pending":
                continue
            if chat_id and m.chat_id != chat_id:
                continue
            result.append(m)
        return result

    scheduler.get_pending.side_effect = _get_pending

    def _edit_message(msg_id, new_text=None, new_send_at=None):
        m = queue.get(msg_id)
        if m is None or m.status != "pending":
            return False
        if new_text is not None:
            m.text = new_text
        if new_send_at is not None:
            m.send_at = new_send_at
        return True

    scheduler.edit_message.side_effect = _edit_message

    def _cancel_message(msg_id):
        m = queue.get(msg_id)
        if m is None or m.status != "pending":
            return False
        m.status = "cancelled"
        return True

    scheduler.cancel_message.side_effect = _cancel_message
    return scheduler


def _make_mock_deferral_mgr(records: dict[str, Any] | None = None) -> MagicMock:
    dmgr = MagicMock()
    recs = records or {}
    dmgr._records = recs
    dmgr._lock = threading.Lock()
    dmgr.max_depth = 3

    def _cancel(session_key):
        rec = recs.get(session_key)
        if rec is None or rec.status not in ("pending", "firing"):
            return False
        rec.status = "cancelled"
        return True

    dmgr.cancel.side_effect = _cancel
    return dmgr


def _make_mock_knowledge_store(facts: list[_MockFact] | None = None) -> MagicMock:
    ks = MagicMock()
    f_list = list(facts or [])
    ks._facts = f_list
    ks._lock = threading.Lock()
    ks._fact_hashes = {f.fact_hash for f in f_list}

    def _recall(query, k=5):
        if not f_list:
            return None
        return "\n".join(f"- {f.entity}: {f.fact}" for f in f_list[:k])

    ks.recall.side_effect = _recall
    ks.save.return_value = None
    return ks


def _make_mock_guardrails(blacklisted: list[str] | None = None) -> MagicMock:
    g = MagicMock()
    vt = MagicMock()
    violations: dict[str, deque] = {}
    now = time.monotonic()
    for chat_id in blacklisted or []:
        # 3 violations within the window to exceed max_violations=2
        violations[chat_id] = deque([now - 10, now - 5, now - 1])
    vt._violations = violations
    vt._lock = threading.Lock()
    vt._max_violations = 2
    vt._window_seconds = 1800.0
    vt.save.return_value = None
    g._violation_tracker = vt
    return g


def _make_mock_session_mgr(sessions: list[tuple[str, str, str]] | None = None) -> MagicMock:
    """Build a mock ChatSessionManager. sessions: list of (session_key, channel, chat_id)."""
    sm = MagicMock()
    sdict: dict[str, MagicMock] = {}
    for key, channel, chat_id in sessions or []:
        sess = MagicMock()
        sess.session_key = key
        sess.channel = channel
        sess.chat_id = chat_id
        sess.last_activity = time.monotonic()
        sess.lock = threading.Lock()
        mm = MagicMock()
        mm.get_messages.return_value = []
        mm.mode = "conversation"
        sess.memory_manager = mm
        sdict[key] = sess
    sm._sessions = sdict
    return sm


def _make_mock_service(
    channels: list[str] | None = None,
    scheduler: Any = None,
    deferral_mgr: Any = None,
    knowledge_store: Any = None,
    session_mgr: Any = None,
    guardrails: Any = None,
) -> MagicMock:
    svc = MagicMock()
    svc._started_at = datetime.now(UTC)

    chs = []
    for name in channels or []:
        ch = MagicMock()
        ch.name = name
        ch.is_ready.return_value = True
        chs.append(ch)
    svc._channels = chs

    svc._scheduler = scheduler or _make_mock_scheduler()
    svc._deferral_mgr = deferral_mgr
    svc._knowledge_store = knowledge_store
    svc._session_mgr = session_mgr or _make_mock_session_mgr()
    svc._poller = MagicMock()
    svc._executor = MagicMock()
    svc._config = MagicMock()
    svc._config.services = {}

    if guardrails is not None:
        handler = MagicMock()
        handler._guardrails = guardrails
        svc._handler = handler
    else:
        svc._handler = MagicMock()
        svc._handler._guardrails = None

    return svc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    from src.api.app import app

    with TestClient(app) as c:
        app.state.assistant_service = None
        yield c


@pytest.fixture()
def client_with_service():
    from src.api.app import app

    with TestClient(app) as c:
        svc = _make_mock_service(channels=["whatsapp", "telegram"])
        app.state.assistant_service = svc
        yield c, svc
        app.state.assistant_service = None


# ---------------------------------------------------------------------------
# Service lifecycle tests
# ---------------------------------------------------------------------------


class TestAssistantStatus:
    def test_status_when_stopped(self, client):
        resp = client.get("/api/v1/assistant/status", headers=_user_headers())
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["status"] == "stopped"
        assert body["data"]["channels"] == []

    def test_status_when_running(self, client_with_service):
        client, svc = client_with_service
        resp = client.get("/api/v1/assistant/status", headers=_user_headers())
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["status"] == "running"
        assert len(body["data"]["channels"]) == 2

    def test_status_requires_auth(self, client):
        resp = client.get("/api/v1/assistant/status")
        assert resp.status_code == 401

    def test_stop_when_not_running_returns_409(self, client):
        resp = client.post("/api/v1/assistant/stop", headers=_admin_headers())
        assert resp.status_code == 409
        body = resp.json()
        assert body["error"]["code"] == "ASSISTANT_NOT_RUNNING"

    def test_stop_requires_admin(self, client):
        resp = client.post("/api/v1/assistant/stop", headers=_user_headers())
        assert resp.status_code == 403

    def test_stop_when_running(self, client_with_service):
        client, svc = client_with_service
        resp = client.post("/api/v1/assistant/stop", headers=_admin_headers())
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["status"] == "stopped"
        from src.api.app import app

        assert app.state.assistant_service is None

    def test_start_requires_admin(self, client):
        resp = client.post(
            "/api/v1/assistant/start",
            json={"force_restart": False},
            headers=_user_headers(),
        )
        assert resp.status_code == 403

    def test_start_already_running_no_force(self, client_with_service):
        client, svc = client_with_service
        resp = client.post(
            "/api/v1/assistant/start",
            json={"force_restart": False},
            headers=_admin_headers(),
        )
        assert resp.status_code == 409
        body = resp.json()
        assert body["error"]["code"] == "ASSISTANT_ALREADY_RUNNING"


# ---------------------------------------------------------------------------
# Chat session tests
# ---------------------------------------------------------------------------


class TestAssistantChats:
    def test_list_chats_service_not_running(self, client):
        resp = client.get("/api/v1/assistant/chats", headers=_user_headers())
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["items"] == []
        assert body["data"]["has_more"] is False

    def test_list_chats_with_sessions(self):
        from src.api.app import app

        sessions = [
            ("whatsapp::+1234", "whatsapp", "+1234@c.us"),
            ("telegram::100", "telegram", "100"),
        ]
        sm = _make_mock_session_mgr(sessions)
        svc = _make_mock_service(channels=["whatsapp"], session_mgr=sm)

        with TestClient(app) as c:
            app.state.assistant_service = svc
            resp = c.get("/api/v1/assistant/chats", headers=_user_headers())
            app.state.assistant_service = None

        assert resp.status_code == 200
        body = resp.json()
        items = body["data"]["items"]
        assert len(items) == 2
        keys = {item["session_key"] for item in items}
        assert "whatsapp::+1234" in keys
        assert "telegram::100" in keys

    def test_list_chats_filter_by_channel(self):
        from src.api.app import app

        sessions = [
            ("whatsapp::+1234", "whatsapp", "+1234@c.us"),
            ("telegram::100", "telegram", "100"),
        ]
        sm = _make_mock_session_mgr(sessions)
        svc = _make_mock_service(channels=["whatsapp"], session_mgr=sm)

        with TestClient(app) as c:
            app.state.assistant_service = svc
            resp = c.get("/api/v1/assistant/chats?channel=whatsapp", headers=_user_headers())
            app.state.assistant_service = None

        assert resp.status_code == 200
        items = resp.json()["data"]["items"]
        assert len(items) == 1
        assert items[0]["channel"] == "whatsapp"

    def test_get_chat_messages_not_found(self):
        from src.api.app import app

        svc = _make_mock_service()

        with TestClient(app) as c:
            app.state.assistant_service = svc
            resp = c.get(
                "/api/v1/assistant/chats/nonexistent::key/messages",
                headers=_user_headers(),
            )
            app.state.assistant_service = None

        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "NOT_FOUND"

    def test_get_chat_messages_service_not_running(self, client):
        resp = client.get(
            "/api/v1/assistant/chats/whatsapp::+1234/messages",
            headers=_user_headers(),
        )
        assert resp.status_code == 409

    def test_get_chat_messages_returns_history(self):
        from unittest.mock import MagicMock

        from src.api.app import app

        sessions = [("wa::123", "whatsapp", "123")]
        sm = _make_mock_session_mgr(sessions)
        # Add mock messages
        mock_msg = MagicMock()
        mock_msg.content = "Hello!"
        sm._sessions["wa::123"].memory_manager.get_messages.return_value = [mock_msg]

        svc = _make_mock_service(session_mgr=sm)

        with TestClient(app) as c:
            app.state.assistant_service = svc
            resp = c.get(
                "/api/v1/assistant/chats/wa::123/messages",
                headers=_user_headers(),
            )
            app.state.assistant_service = None

        assert resp.status_code == 200
        items = resp.json()["data"]["items"]
        assert len(items) == 1
        assert items[0]["content"] == "Hello!"


# ---------------------------------------------------------------------------
# Scheduled message tests
# ---------------------------------------------------------------------------


class TestScheduledMessages:
    def _make_msg(self) -> _MockScheduledMessage:
        return _MockScheduledMessage(
            id=str(uuid.uuid4()),
            channel="whatsapp",
            chat_id="+123@c.us",
            text="Hello!",
            send_at=time.time() + 3600,
            created_at=time.time(),
            recipient="+123",
            status="pending",
        )

    def test_list_scheduled_no_service(self, client):
        resp = client.get("/api/v1/assistant/scheduled", headers=_user_headers())
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["items"] == []

    def test_list_scheduled_with_messages(self):
        from src.api.app import app

        msg = self._make_msg()
        scheduler = _make_mock_scheduler([msg])
        svc = _make_mock_service(scheduler=scheduler)

        with TestClient(app) as c:
            app.state.assistant_service = svc
            resp = c.get("/api/v1/assistant/scheduled", headers=_user_headers())
            app.state.assistant_service = None

        assert resp.status_code == 200
        items = resp.json()["data"]["items"]
        assert len(items) == 1
        assert items[0]["id"] == msg.id
        assert items[0]["status"] == "pending"
        assert items[0]["text"] == "Hello!"

    def test_list_scheduled_filter_by_channel(self):
        from src.api.app import app

        msg1 = self._make_msg()
        msg2 = self._make_msg()
        msg2.channel = "telegram"
        scheduler = _make_mock_scheduler([msg1, msg2])
        svc = _make_mock_service(scheduler=scheduler)

        with TestClient(app) as c:
            app.state.assistant_service = svc
            resp = c.get("/api/v1/assistant/scheduled?channel=telegram", headers=_user_headers())
            app.state.assistant_service = None

        assert resp.status_code == 200
        items = resp.json()["data"]["items"]
        assert len(items) == 1
        assert items[0]["channel"] == "telegram"

    def test_edit_scheduled_not_running(self, client):
        resp = client.patch(
            f"/api/v1/assistant/scheduled/{uuid.uuid4()}",
            json={"text": "Updated"},
            headers=_user_headers(),
        )
        assert resp.status_code == 409

    def test_edit_scheduled_not_found(self):
        from src.api.app import app

        scheduler = _make_mock_scheduler()
        svc = _make_mock_service(scheduler=scheduler)

        with TestClient(app) as c:
            app.state.assistant_service = svc
            resp = c.patch(
                f"/api/v1/assistant/scheduled/{uuid.uuid4()}",
                json={"text": "Updated"},
                headers=_user_headers(),
            )
            app.state.assistant_service = None

        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "SCHEDULED_MSG_NOT_FOUND"

    def test_edit_scheduled_success(self):
        from src.api.app import app

        msg = self._make_msg()
        scheduler = _make_mock_scheduler([msg])
        svc = _make_mock_service(scheduler=scheduler)

        with TestClient(app) as c:
            app.state.assistant_service = svc
            resp = c.patch(
                f"/api/v1/assistant/scheduled/{msg.id}",
                json={"text": "Updated text"},
                headers=_user_headers(),
            )
            app.state.assistant_service = None

        assert resp.status_code == 200
        assert resp.json()["data"]["text"] == "Updated text"

    def test_cancel_scheduled_not_running(self, client):
        resp = client.delete(
            f"/api/v1/assistant/scheduled/{uuid.uuid4()}",
            headers=_user_headers(),
        )
        assert resp.status_code == 409

    def test_cancel_scheduled_not_found(self):
        from src.api.app import app

        scheduler = _make_mock_scheduler()
        svc = _make_mock_service(scheduler=scheduler)

        with TestClient(app) as c:
            app.state.assistant_service = svc
            resp = c.delete(
                f"/api/v1/assistant/scheduled/{uuid.uuid4()}",
                headers=_user_headers(),
            )
            app.state.assistant_service = None

        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "SCHEDULED_MSG_NOT_FOUND"

    def test_cancel_scheduled_success(self):
        from src.api.app import app

        msg = self._make_msg()
        scheduler = _make_mock_scheduler([msg])
        svc = _make_mock_service(scheduler=scheduler)

        with TestClient(app) as c:
            app.state.assistant_service = svc
            resp = c.delete(
                f"/api/v1/assistant/scheduled/{msg.id}",
                headers=_user_headers(),
            )
            app.state.assistant_service = None

        assert resp.status_code == 200
        assert resp.json()["data"] is None
        assert msg.status == "cancelled"


# ---------------------------------------------------------------------------
# Deferred record tests
# ---------------------------------------------------------------------------


class TestDeferredRecords:
    def _make_record(self, session_key: str) -> _MockDeferredRecord:
        return _MockDeferredRecord(
            id=str(uuid.uuid4()),
            channel="whatsapp",
            chat_id="+123",
            fire_at=time.time() + 600,
            created_at=time.time(),
            pending_messages=[{"text": "hello", "chat_id": "+123"}],
            deferral_depth=1,
            status="pending",
        )

    def test_list_deferred_no_service(self, client):
        resp = client.get("/api/v1/assistant/deferred", headers=_user_headers())
        assert resp.status_code == 200
        assert resp.json()["data"] == []

    def test_list_deferred_with_records(self):
        from src.api.app import app

        key = "whatsapp::+123"
        record = self._make_record(key)
        dmgr = _make_mock_deferral_mgr({key: record})
        svc = _make_mock_service(deferral_mgr=dmgr)

        with TestClient(app) as c:
            app.state.assistant_service = svc
            resp = c.get("/api/v1/assistant/deferred", headers=_user_headers())
            app.state.assistant_service = None

        assert resp.status_code == 200
        items = resp.json()["data"]
        assert len(items) == 1
        assert items[0]["session_key"] == key
        assert items[0]["status"] == "pending"
        assert items[0]["depth"] == 1

    def test_cancel_deferred_not_running(self, client):
        resp = client.delete(
            "/api/v1/assistant/deferred/whatsapp::+123",
            headers=_user_headers(),
        )
        assert resp.status_code == 409

    def test_cancel_deferred_not_found(self):
        from src.api.app import app

        dmgr = _make_mock_deferral_mgr({})
        svc = _make_mock_service(deferral_mgr=dmgr)

        with TestClient(app) as c:
            app.state.assistant_service = svc
            resp = c.delete(
                "/api/v1/assistant/deferred/nonexistent::key",
                headers=_user_headers(),
            )
            app.state.assistant_service = None

        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "DEFERRED_MSG_NOT_FOUND"

    def test_cancel_deferred_success(self):
        from src.api.app import app

        key = "whatsapp::+123"
        record = self._make_record(key)
        dmgr = _make_mock_deferral_mgr({key: record})
        svc = _make_mock_service(deferral_mgr=dmgr)

        with TestClient(app) as c:
            app.state.assistant_service = svc
            resp = c.delete(
                f"/api/v1/assistant/deferred/{key}",
                headers=_user_headers(),
            )
            app.state.assistant_service = None

        assert resp.status_code == 200
        assert resp.json()["data"] is None
        assert record.status == "cancelled"

    def test_list_deferred_no_deferral_mgr(self):
        from src.api.app import app

        svc = _make_mock_service(deferral_mgr=None)
        svc._deferral_mgr = None

        with TestClient(app) as c:
            app.state.assistant_service = svc
            resp = c.get("/api/v1/assistant/deferred", headers=_user_headers())
            app.state.assistant_service = None

        assert resp.status_code == 200
        assert resp.json()["data"] == []


# ---------------------------------------------------------------------------
# Contacts tests
# ---------------------------------------------------------------------------


class TestContacts:
    def test_list_contacts_no_service(self, client):
        resp = client.get("/api/v1/assistant/contacts", headers=_user_headers())
        assert resp.status_code == 200
        assert resp.json()["data"] == []

    def test_list_contacts_with_phonebook(self):
        from src.api.app import app

        svc = _make_mock_service()
        svc._config.services = {
            "whatsapp": {
                "phonebook": {
                    "Alice": "+1234567890",
                    "Bob": "+9876543210",
                }
            }
        }

        with TestClient(app) as c:
            app.state.assistant_service = svc
            resp = c.get("/api/v1/assistant/contacts", headers=_user_headers())
            app.state.assistant_service = None

        assert resp.status_code == 200
        items = resp.json()["data"]
        assert len(items) == 2
        names = {item["name"] for item in items}
        assert "alice" in names
        assert "bob" in names

    def test_list_contacts_requires_auth(self, client):
        resp = client.get("/api/v1/assistant/contacts")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Guardrails tests
# ---------------------------------------------------------------------------


class TestGuardrails:
    def test_get_guardrails_requires_admin(self, client):
        resp = client.get("/api/v1/assistant/guardrails", headers=_user_headers())
        assert resp.status_code == 403

    def test_get_guardrails_no_service(self, client):
        resp = client.get("/api/v1/assistant/guardrails", headers=_admin_headers())
        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["blacklisted_chats"] == []
        assert body["total_violations"] == 0

    def test_get_guardrails_with_blacklisted(self):
        from src.api.app import app

        g = _make_mock_guardrails(blacklisted=["chat1", "chat2"])
        svc = _make_mock_service(guardrails=g)

        with TestClient(app) as c:
            app.state.assistant_service = svc
            resp = c.get("/api/v1/assistant/guardrails", headers=_admin_headers())
            app.state.assistant_service = None

        assert resp.status_code == 200
        body = resp.json()["data"]
        assert set(body["blacklisted_chats"]) == {"chat1", "chat2"}
        assert body["total_violations"] >= 2

    def test_remove_from_blacklist_requires_admin(self, client):
        resp = client.delete(
            "/api/v1/assistant/guardrails/blacklist/chat1",
            headers=_user_headers(),
        )
        assert resp.status_code == 403

    def test_remove_from_blacklist_service_not_running(self, client):
        resp = client.delete(
            "/api/v1/assistant/guardrails/blacklist/chat1",
            headers=_admin_headers(),
        )
        assert resp.status_code == 409

    def test_remove_from_blacklist_not_found(self):
        from src.api.app import app

        g = _make_mock_guardrails(blacklisted=[])
        svc = _make_mock_service(guardrails=g)

        with TestClient(app) as c:
            app.state.assistant_service = svc
            resp = c.delete(
                "/api/v1/assistant/guardrails/blacklist/nonexistent",
                headers=_admin_headers(),
            )
            app.state.assistant_service = None

        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "NOT_FOUND"

    def test_remove_from_blacklist_success(self):
        from src.api.app import app

        g = _make_mock_guardrails(blacklisted=["bad_chat"])
        svc = _make_mock_service(guardrails=g)

        with TestClient(app) as c:
            app.state.assistant_service = svc
            resp = c.delete(
                "/api/v1/assistant/guardrails/blacklist/bad_chat",
                headers=_admin_headers(),
            )
            app.state.assistant_service = None

        assert resp.status_code == 200
        assert resp.json()["data"] is None
        # Verify it was removed
        assert "bad_chat" not in g._violation_tracker._violations


# ---------------------------------------------------------------------------
# Knowledge store tests
# ---------------------------------------------------------------------------


class TestKnowledge:
    def _make_fact(self, entity: str, fact: str) -> _MockFact:
        import hashlib

        h = hashlib.sha256(f"{entity.lower()}::{fact.lower()}".encode()).hexdigest()[:16]
        return _MockFact(
            entity=entity,
            fact=fact,
            source_session="wa::+123",
            timestamp=time.time(),
            fact_hash=h,
        )

    def test_list_knowledge_no_service(self, client):
        resp = client.get("/api/v1/assistant/knowledge", headers=_user_headers())
        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["items"] == []

    def test_list_knowledge_no_store(self):
        from src.api.app import app

        svc = _make_mock_service(knowledge_store=None)
        svc._knowledge_store = None

        with TestClient(app) as c:
            app.state.assistant_service = svc
            resp = c.get("/api/v1/assistant/knowledge", headers=_user_headers())
            app.state.assistant_service = None

        assert resp.status_code == 200
        assert resp.json()["data"]["items"] == []

    def test_list_knowledge_with_facts(self):
        from src.api.app import app

        facts = [
            self._make_fact("Alice", "Is a veterinarian"),
            self._make_fact("Bob", "Lives in Paris"),
        ]
        ks = _make_mock_knowledge_store(facts)
        svc = _make_mock_service(knowledge_store=ks)

        with TestClient(app) as c:
            app.state.assistant_service = svc
            resp = c.get("/api/v1/assistant/knowledge", headers=_user_headers())
            app.state.assistant_service = None

        assert resp.status_code == 200
        items = resp.json()["data"]["items"]
        assert len(items) == 2
        texts = {item["text"] for item in items}
        assert any("Alice" in t for t in texts)
        assert any("Bob" in t for t in texts)

    def test_list_knowledge_filter_by_source_chat(self):
        from src.api.app import app

        fact1 = self._make_fact("Alice", "Loves cats")
        fact1.source_session = "wa::chat111"
        fact2 = self._make_fact("Bob", "Hates mondays")
        fact2.source_session = "wa::chat222"
        ks = _make_mock_knowledge_store([fact1, fact2])
        svc = _make_mock_service(knowledge_store=ks)

        with TestClient(app) as c:
            app.state.assistant_service = svc
            resp = c.get(
                "/api/v1/assistant/knowledge?source_chat=wa::chat111",
                headers=_user_headers(),
            )
            app.state.assistant_service = None

        assert resp.status_code == 200
        items = resp.json()["data"]["items"]
        assert len(items) == 1
        assert "Alice" in items[0]["text"]

    def test_search_knowledge_no_service(self, client):
        resp = client.post(
            "/api/v1/assistant/knowledge/search",
            json={"query": "Alice", "top_k": 5},
            headers=_user_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["data"] == []

    def test_search_knowledge_returns_results(self):
        from src.api.app import app

        facts = [
            self._make_fact("Alice", "Is a veterinarian in Portland"),
        ]
        ks = _make_mock_knowledge_store(facts)
        svc = _make_mock_service(knowledge_store=ks)

        with TestClient(app) as c:
            app.state.assistant_service = svc
            resp = c.post(
                "/api/v1/assistant/knowledge/search",
                json={"query": "veterinarian Alice", "top_k": 5},
                headers=_user_headers(),
            )
            app.state.assistant_service = None

        assert resp.status_code == 200
        items = resp.json()["data"]
        assert len(items) >= 1
        assert all(item["relevance_score"] is not None for item in items)

    def test_search_knowledge_empty_store(self):
        from src.api.app import app

        ks = _make_mock_knowledge_store([])
        svc = _make_mock_service(knowledge_store=ks)

        with TestClient(app) as c:
            app.state.assistant_service = svc
            resp = c.post(
                "/api/v1/assistant/knowledge/search",
                json={"query": "test query", "top_k": 5},
                headers=_user_headers(),
            )
            app.state.assistant_service = None

        assert resp.status_code == 200
        assert resp.json()["data"] == []

    def test_delete_fact_requires_admin(self, client):
        resp = client.delete(
            "/api/v1/assistant/knowledge/somehash",
            headers=_user_headers(),
        )
        assert resp.status_code == 403

    def test_delete_fact_service_not_running(self, client):
        resp = client.delete(
            "/api/v1/assistant/knowledge/somehash",
            headers=_admin_headers(),
        )
        assert resp.status_code == 409

    def test_delete_fact_not_found(self):
        from src.api.app import app

        ks = _make_mock_knowledge_store([])
        svc = _make_mock_service(knowledge_store=ks)

        with TestClient(app) as c:
            app.state.assistant_service = svc
            resp = c.delete(
                "/api/v1/assistant/knowledge/nonexistenthash",
                headers=_admin_headers(),
            )
            app.state.assistant_service = None

        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "FACT_NOT_FOUND"

    def test_delete_fact_success(self):
        from src.api.app import app

        fact = self._make_fact("Alice", "Is a vet")
        ks = _make_mock_knowledge_store([fact])
        svc = _make_mock_service(knowledge_store=ks)

        with TestClient(app) as c:
            app.state.assistant_service = svc
            resp = c.delete(
                f"/api/v1/assistant/knowledge/{fact.fact_hash}",
                headers=_admin_headers(),
            )
            app.state.assistant_service = None

        assert resp.status_code == 200
        assert resp.json()["data"] is None
        assert fact.fact_hash not in ks._fact_hashes
