"""Tests for /api/v1/tasks — background task queue endpoints.

Coverage:
    POST   /api/v1/tasks              — submit task (202 / 503)
    GET    /api/v1/tasks              — list tasks  (200 / 400 / 503)
    GET    /api/v1/tasks/{task_id}    — get task    (200 / 404 / 503)
    DELETE /api/v1/tasks/{task_id}    — cancel task (200 / 404 / 409 / 503)
    GET    /api/v1/tasks/{task_id}/log — task log   (200 / 404 / 503)
    Auth                              — 401 without token
"""

from __future__ import annotations

import asyncio as _asyncio
import os
import uuid
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")

_TEST_JWT_SECRET = "testsecret_mustbe32chars_minimum00"
os.environ.setdefault("COGTRIX_JWT_SECRET", _TEST_JWT_SECRET)
os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import src.tasks.queue as _queue_mod  # noqa: E402
from src.api.db.engine import Base, get_db  # noqa: E402

# ---------------------------------------------------------------------------
# App / DB fixture
# ---------------------------------------------------------------------------


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


@pytest.fixture(autouse=True)
def _reset_queue():
    original = _queue_mod._queue
    yield
    _queue_mod._queue = original


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

_VALID_PASSWORD = "TestPass1!"


def _register_and_login(client) -> str:
    """Register a user and return an access token."""
    name = f"u_{uuid.uuid4().hex[:8]}"
    r = client.post(
        "/api/v1/auth/register",
        json={"username": name, "email": f"{name}@ex.com", "password": _VALID_PASSWORD},
    )
    assert r.status_code == 201, r.text
    return r.json()["data"]["access_token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Queue fixture — initialise a real in-memory SQLite queue
# ---------------------------------------------------------------------------


@pytest.fixture()
def queue(tmp_path):
    from src.tasks.queue import init_task_queue

    q = init_task_queue(tmp_path / "tasks.db", tmp_path / "logs")
    yield q
    if q._executor is not None:
        q.stop()


# ---------------------------------------------------------------------------
# 503 — queue not available
# ---------------------------------------------------------------------------


class TestQueueUnavailable:
    def test_post_returns_503_when_no_queue(self, client):
        _queue_mod._queue = None
        token = _register_and_login(client)
        r = client.post(
            "/api/v1/tasks",
            json={"agent_name": "myagent", "prompt": "do something"},
            headers=_auth(token),
        )
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "TASK_QUEUE_UNAVAILABLE"

    def test_get_list_returns_503_when_no_queue(self, client):
        _queue_mod._queue = None
        token = _register_and_login(client)
        r = client.get("/api/v1/tasks", headers=_auth(token))
        assert r.status_code == 503

    def test_get_by_id_returns_503_when_no_queue(self, client):
        _queue_mod._queue = None
        token = _register_and_login(client)
        r = client.get("/api/v1/tasks/fakeid", headers=_auth(token))
        assert r.status_code == 503

    def test_delete_returns_503_when_no_queue(self, client):
        _queue_mod._queue = None
        token = _register_and_login(client)
        r = client.delete("/api/v1/tasks/fakeid", headers=_auth(token))
        assert r.status_code == 503

    def test_log_returns_503_when_no_queue(self, client):
        _queue_mod._queue = None
        token = _register_and_login(client)
        r = client.get("/api/v1/tasks/fakeid/log", headers=_auth(token))
        assert r.status_code == 503


# ---------------------------------------------------------------------------
# 401 — no auth
# ---------------------------------------------------------------------------


class TestNoAuth:
    def test_post_returns_401(self, client, queue):
        r = client.post("/api/v1/tasks", json={"agent_name": "a", "prompt": "p"})
        assert r.status_code == 401

    def test_get_list_returns_401(self, client, queue):
        r = client.get("/api/v1/tasks")
        assert r.status_code == 401

    def test_get_by_id_returns_401(self, client, queue):
        r = client.get("/api/v1/tasks/fakeid")
        assert r.status_code == 401

    def test_delete_returns_401(self, client, queue):
        r = client.delete("/api/v1/tasks/fakeid")
        assert r.status_code == 401

    def test_log_returns_401(self, client, queue):
        r = client.get("/api/v1/tasks/fakeid/log")
        assert r.status_code == 401


# ---------------------------------------------------------------------------
# POST /api/v1/tasks
# ---------------------------------------------------------------------------


class TestCreateTask:
    def test_submit_returns_202(self, client, queue):
        token = _register_and_login(client)
        r = client.post(
            "/api/v1/tasks",
            json={"agent_name": "researcher", "prompt": "Summarise arXiv papers"},
            headers=_auth(token),
        )
        assert r.status_code == 202

    def test_submit_returns_task_record(self, client, queue):
        token = _register_and_login(client)
        r = client.post(
            "/api/v1/tasks",
            json={"agent_name": "coder", "prompt": "refactor module"},
            headers=_auth(token),
        )
        data = r.json()["data"]
        assert data["agent_name"] == "coder"
        assert data["prompt"] == "refactor module"
        assert data["status"] == "PENDING"
        assert "task_id" in data

    def test_submit_validates_empty_agent_name(self, client, queue):
        token = _register_and_login(client)
        r = client.post(
            "/api/v1/tasks",
            json={"agent_name": "", "prompt": "some task"},
            headers=_auth(token),
        )
        assert r.status_code == 422

    def test_submit_validates_empty_prompt(self, client, queue):
        token = _register_and_login(client)
        r = client.post(
            "/api/v1/tasks",
            json={"agent_name": "agent", "prompt": ""},
            headers=_auth(token),
        )
        assert r.status_code == 422

    def test_submit_task_appears_in_list(self, client, queue):
        token = _register_and_login(client)
        client.post(
            "/api/v1/tasks",
            json={"agent_name": "myagent", "prompt": "find me"},
            headers=_auth(token),
        )
        r = client.get("/api/v1/tasks", headers=_auth(token))
        items = r.json()["data"]
        assert any(t["agent_name"] == "myagent" for t in items)


# ---------------------------------------------------------------------------
# GET /api/v1/tasks
# ---------------------------------------------------------------------------


class TestListTasks:
    def test_list_empty_returns_empty_list(self, client, queue):
        token = _register_and_login(client)
        r = client.get("/api/v1/tasks", headers=_auth(token))
        assert r.status_code == 200
        assert r.json()["data"] == []

    def test_list_returns_submitted_tasks(self, client, queue):
        token = _register_and_login(client)
        client.post(
            "/api/v1/tasks",
            json={"agent_name": "a1", "prompt": "task one"},
            headers=_auth(token),
        )
        client.post(
            "/api/v1/tasks",
            json={"agent_name": "a2", "prompt": "task two"},
            headers=_auth(token),
        )
        r = client.get("/api/v1/tasks", headers=_auth(token))
        assert r.status_code == 200
        items = r.json()["data"]
        assert len(items) == 2

    def test_list_status_filter_pending(self, client, queue):
        token = _register_and_login(client)
        r1 = client.post(
            "/api/v1/tasks",
            json={"agent_name": "a", "prompt": "keep"},
            headers=_auth(token),
        )
        task_id = r1.json()["data"]["task_id"]
        # Cancel it directly via the queue
        queue.cancel(task_id)

        client.post(
            "/api/v1/tasks",
            json={"agent_name": "b", "prompt": "pending task"},
            headers=_auth(token),
        )

        r = client.get("/api/v1/tasks?status=pending", headers=_auth(token))
        assert r.status_code == 200
        items = r.json()["data"]
        assert all(t["status"] == "PENDING" for t in items)
        assert len(items) == 1

    def test_list_invalid_status_returns_400(self, client, queue):
        token = _register_and_login(client)
        r = client.get("/api/v1/tasks?status=NOTASTATUS", headers=_auth(token))
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "INVALID_STATUS"

    def test_list_limit_param(self, client, queue):
        token = _register_and_login(client)
        for i in range(5):
            client.post(
                "/api/v1/tasks",
                json={"agent_name": "a", "prompt": f"task {i}"},
                headers=_auth(token),
            )
        r = client.get("/api/v1/tasks?limit=2", headers=_auth(token))
        assert r.status_code == 200
        assert len(r.json()["data"]) == 2


# ---------------------------------------------------------------------------
# GET /api/v1/tasks/{task_id}
# ---------------------------------------------------------------------------


class TestGetTask:
    def test_get_existing_task(self, client, queue):
        token = _register_and_login(client)
        r = client.post(
            "/api/v1/tasks",
            json={"agent_name": "tester", "prompt": "run all tests"},
            headers=_auth(token),
        )
        task_id = r.json()["data"]["task_id"]

        r2 = client.get(f"/api/v1/tasks/{task_id}", headers=_auth(token))
        assert r2.status_code == 200
        assert r2.json()["data"]["task_id"] == task_id
        assert r2.json()["data"]["agent_name"] == "tester"

    def test_get_unknown_task_returns_404(self, client, queue):
        token = _register_and_login(client)
        r = client.get(f"/api/v1/tasks/{uuid.uuid4()}", headers=_auth(token))
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "TASK_NOT_FOUND"


# ---------------------------------------------------------------------------
# DELETE /api/v1/tasks/{task_id}
# ---------------------------------------------------------------------------


class TestCancelTask:
    def test_cancel_pending_task_returns_200(self, client, queue):
        token = _register_and_login(client)
        r = client.post(
            "/api/v1/tasks",
            json={"agent_name": "a", "prompt": "cancel me"},
            headers=_auth(token),
        )
        task_id = r.json()["data"]["task_id"]

        r2 = client.delete(f"/api/v1/tasks/{task_id}", headers=_auth(token))
        assert r2.status_code == 200
        assert r2.json()["error"] is None

    def test_cancel_updates_status_to_cancelled(self, client, queue):
        token = _register_and_login(client)
        r = client.post(
            "/api/v1/tasks",
            json={"agent_name": "a", "prompt": "cancel me"},
            headers=_auth(token),
        )
        task_id = r.json()["data"]["task_id"]
        client.delete(f"/api/v1/tasks/{task_id}", headers=_auth(token))

        r2 = client.get(f"/api/v1/tasks/{task_id}", headers=_auth(token))
        assert r2.json()["data"]["status"] == "CANCELLED"

    def test_cancel_unknown_task_returns_404(self, client, queue):
        token = _register_and_login(client)
        r = client.delete(f"/api/v1/tasks/{uuid.uuid4()}", headers=_auth(token))
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "TASK_NOT_FOUND"

    def test_cancel_already_cancelled_returns_409(self, client, queue):
        token = _register_and_login(client)
        r = client.post(
            "/api/v1/tasks",
            json={"agent_name": "a", "prompt": "cancel twice"},
            headers=_auth(token),
        )
        task_id = r.json()["data"]["task_id"]
        client.delete(f"/api/v1/tasks/{task_id}", headers=_auth(token))
        r2 = client.delete(f"/api/v1/tasks/{task_id}", headers=_auth(token))
        assert r2.status_code == 409
        assert r2.json()["error"]["code"] == "TASK_NOT_CANCELLABLE"


# ---------------------------------------------------------------------------
# GET /api/v1/tasks/{task_id}/log
# ---------------------------------------------------------------------------


class TestTaskLog:
    def test_log_returns_plaintext(self, client, queue):
        token = _register_and_login(client)
        r = client.post(
            "/api/v1/tasks",
            json={"agent_name": "logger", "prompt": "log test"},
            headers=_auth(token),
        )
        task_id = r.json()["data"]["task_id"]
        r2 = client.get(f"/api/v1/tasks/{task_id}/log", headers=_auth(token))
        assert r2.status_code == 200
        assert "text/plain" in r2.headers.get("content-type", "")

    def test_log_empty_when_no_file(self, client, queue):
        token = _register_and_login(client)
        r = client.post(
            "/api/v1/tasks",
            json={"agent_name": "a", "prompt": "no log yet"},
            headers=_auth(token),
        )
        task_id = r.json()["data"]["task_id"]
        r2 = client.get(f"/api/v1/tasks/{task_id}/log", headers=_auth(token))
        assert r2.status_code == 200
        assert r2.text == ""

    def test_log_returns_file_content(self, client, queue):
        token = _register_and_login(client)
        r = client.post(
            "/api/v1/tasks",
            json={"agent_name": "a", "prompt": "write log"},
            headers=_auth(token),
        )
        task_id = r.json()["data"]["task_id"]
        # Write content to the log file directly
        queue._append_log(task_id, "test log line")
        r2 = client.get(f"/api/v1/tasks/{task_id}/log", headers=_auth(token))
        assert r2.status_code == 200
        assert "test log line" in r2.text

    def test_log_unknown_task_returns_404(self, client, queue):
        token = _register_and_login(client)
        r = client.get(f"/api/v1/tasks/{uuid.uuid4()}/log", headers=_auth(token))
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "TASK_NOT_FOUND"
