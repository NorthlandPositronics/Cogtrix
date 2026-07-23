"""Phase 2 API tests: session bridge, session repositories, and session endpoints.

All tests use an in-memory SQLite database so they never touch the real DB.
The session registry is tested with lightweight mocks so they don't require
a real LLM provider.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

# ---------------------------------------------------------------------------
# Environment setup — must happen before any src.api imports
# ---------------------------------------------------------------------------

_TEST_JWT_SECRET = "testsecret_mustbe32chars_minimum00"
os.environ.setdefault("COGTRIX_JWT_SECRET", _TEST_JWT_SECRET)
os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

# ---------------------------------------------------------------------------
# Imports after env setup
# ---------------------------------------------------------------------------

from cogtrix_core.api.auth import create_access_token  # noqa: E402
from cogtrix_core.api.db import models as _models  # noqa: E402, F401
from cogtrix_core.api.db.engine import Base  # noqa: E402
from cogtrix_core.api.db.repositories.messages import MessageRepository  # noqa: E402
from cogtrix_core.api.db.repositories.sessions import SessionRepository  # noqa: E402
from cogtrix_core.api.db.repositories.users import UserRepository  # noqa: E402

# ---------------------------------------------------------------------------
# In-memory DB fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture()
async def db_session() -> AsyncGenerator[AsyncSession]:
    """Yield a fresh in-memory SQLite session for each test."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    await engine.dispose()


@pytest_asyncio.fixture()
async def user_id(db_session: AsyncSession) -> str:
    """Create a test user and return their UUID."""
    repo = UserRepository(db_session)
    uid = str(uuid.uuid4())
    await repo.create(
        user_id=uid,
        username="testuser",
        email="testuser@example.com",
        password_hash="hashed",
        role="user",
    )
    await db_session.commit()
    return uid


@pytest_asyncio.fixture()
async def admin_id(db_session: AsyncSession) -> str:
    """Create an admin user and return their UUID."""
    repo = UserRepository(db_session)
    uid = str(uuid.uuid4())
    await repo.create(
        user_id=uid,
        username="adminuser",
        email="admin@example.com",
        password_hash="hashed",
        role="admin",
    )
    await db_session.commit()
    return uid


# ---------------------------------------------------------------------------
# SessionRepository tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_create_and_retrieve(db_session: AsyncSession, user_id: str) -> None:
    """SessionRepository.create + get_by_id round-trips correctly."""
    repo = SessionRepository(db_session)
    record = await repo.create(
        user_id=user_id,
        name="Test session",
        config_json=json.dumps({"provider": "openai", "model": "gpt-4.1-mini"}),
    )
    await db_session.commit()

    fetched = await repo.get_by_id(record.id)
    assert fetched is not None
    assert fetched.name == "Test session"
    assert fetched.user_id == user_id
    assert fetched.state == "idle"
    assert fetched.archived_at is None


@pytest.mark.asyncio
async def test_session_list_by_user(db_session: AsyncSession, user_id: str) -> None:
    """list_by_user returns sessions ordered newest first."""
    repo = SessionRepository(db_session)
    for i in range(3):
        await repo.create(user_id=user_id, name=f"Session {i}")
    await db_session.commit()

    rows = await repo.list_by_user(user_id, limit=10)
    assert len(rows) == 3


@pytest.mark.asyncio
async def test_session_list_excludes_archived(db_session: AsyncSession, user_id: str) -> None:
    """Archived sessions are excluded by default."""
    repo = SessionRepository(db_session)
    r1 = await repo.create(user_id=user_id, name="Active")
    r2 = await repo.create(user_id=user_id, name="Archived")
    await db_session.commit()

    await repo.archive(r2.id)
    await db_session.commit()

    rows = await repo.list_by_user(user_id, limit=10)
    assert len(rows) == 1
    assert rows[0].id == r1.id

    rows_with_archived = await repo.list_by_user(user_id, limit=10, include_archived=True)
    assert len(rows_with_archived) == 2


@pytest.mark.asyncio
async def test_session_update(db_session: AsyncSession, user_id: str) -> None:
    """update() modifies the specified columns."""
    repo = SessionRepository(db_session)
    record = await repo.create(user_id=user_id, name="Original")
    await db_session.commit()

    updated = await repo.update(record.id, name="Renamed", state="thinking")
    await db_session.commit()

    assert updated is not None
    assert updated.name == "Renamed"
    assert updated.state == "thinking"


@pytest.mark.asyncio
async def test_session_archive(db_session: AsyncSession, user_id: str) -> None:
    """archive() sets archived_at timestamp."""
    repo = SessionRepository(db_session)
    record = await repo.create(user_id=user_id, name="To archive")
    await db_session.commit()

    assert record.archived_at is None
    await repo.archive(record.id)
    await db_session.commit()

    fetched = await repo.get_by_id(record.id)
    assert fetched is not None
    assert fetched.archived_at is not None


@pytest.mark.asyncio
async def test_session_count_by_user(db_session: AsyncSession, user_id: str) -> None:
    """count_by_user returns correct active count (excluding archived)."""
    repo = SessionRepository(db_session)
    assert await repo.count_by_user(user_id) == 0

    r1 = await repo.create(user_id=user_id, name="S1")
    await repo.create(user_id=user_id, name="S2")
    await db_session.commit()
    assert await repo.count_by_user(user_id) == 2

    await repo.archive(r1.id)
    await db_session.commit()
    assert await repo.count_by_user(user_id) == 1


@pytest.mark.asyncio
async def test_session_cursor_pagination(db_session: AsyncSession, user_id: str) -> None:
    """Cursor pagination returns correct pages without repeating items."""
    repo = SessionRepository(db_session)
    for i in range(5):
        await repo.create(user_id=user_id, name=f"S{i}")
    await db_session.commit()

    page1 = await repo.list_by_user(user_id, limit=3)
    assert len(page1) == 4  # limit+1 to detect more
    has_more = len(page1) > 3
    assert has_more
    cursor_id = page1[2].id  # last item on page 1

    page2 = await repo.list_by_user(user_id, after_id=cursor_id, limit=3)
    assert len(page2) <= 3

    # No overlap between pages
    page1_ids = {r.id for r in page1[:3]}
    page2_ids = {r.id for r in page2[:3]}
    assert page1_ids.isdisjoint(page2_ids)


# ---------------------------------------------------------------------------
# MessageRepository tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_message_create_and_list(db_session: AsyncSession, user_id: str) -> None:
    """MessageRepository.create + list_by_session works correctly."""
    session_repo = SessionRepository(db_session)
    record = await session_repo.create(user_id=user_id, name="Chat")
    await db_session.commit()

    msg_repo = MessageRepository(db_session)
    m1 = await msg_repo.create(
        session_id=record.id,
        role="user",
        content_json=json.dumps("Hello"),
    )
    m2 = await msg_repo.create(
        session_id=record.id,
        role="assistant",
        content_json=json.dumps("Hi there!"),
    )
    await db_session.commit()

    msgs = await msg_repo.list_by_session(record.id, limit=10)
    assert len(msgs) == 2
    assert msgs[0].id == m1.id
    assert msgs[1].id == m2.id


@pytest.mark.asyncio
async def test_message_delete_by_session(db_session: AsyncSession, user_id: str) -> None:
    """delete_by_session removes all messages and returns count."""
    session_repo = SessionRepository(db_session)
    record = await session_repo.create(user_id=user_id, name="Chat")
    await db_session.commit()

    msg_repo = MessageRepository(db_session)
    for i in range(4):
        await msg_repo.create(
            session_id=record.id,
            role="user",
            content_json=json.dumps(f"msg {i}"),
        )
    await db_session.commit()

    deleted = await msg_repo.delete_by_session(record.id)
    await db_session.commit()
    assert deleted == 4

    remaining = await msg_repo.list_by_session(record.id, limit=10)
    assert len(remaining) == 0


@pytest.mark.asyncio
async def test_message_cursor_pagination(db_session: AsyncSession, user_id: str) -> None:
    """Message cursor pagination works correctly."""
    session_repo = SessionRepository(db_session)
    record = await session_repo.create(user_id=user_id, name="Paged")
    await db_session.commit()

    msg_repo = MessageRepository(db_session)
    for i in range(5):
        await msg_repo.create(
            session_id=record.id,
            role="user",
            content_json=json.dumps(f"msg {i}"),
        )
    await db_session.commit()

    page1 = await msg_repo.list_by_session(record.id, limit=3)
    assert len(page1) == 4  # limit+1
    cursor_id = page1[2].id

    page2 = await msg_repo.list_by_session(record.id, after_id=cursor_id, limit=3)
    assert len(page2) == 2  # last 2 messages

    # No overlap
    p1_ids = {m.id for m in page1[:3]}
    p2_ids = {m.id for m in page2}
    assert p1_ids.isdisjoint(p2_ids)


# ---------------------------------------------------------------------------
# HTTP endpoint tests
# ---------------------------------------------------------------------------
#
# We override get_db and mock warm_session so tests don't need a real LLM.
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture()
async def http_db_engine():
    """Create a dedicated in-memory engine for HTTP endpoint tests."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture()
async def http_client(http_db_engine):
    """Return a TestClient wired to the in-memory DB and mocked session_bridge."""
    from cogtrix_core.api.app import create_app
    from cogtrix_core.api.db.engine import get_db

    session_factory = async_sessionmaker(http_db_engine, expire_on_commit=False)

    async def override_get_db() -> AsyncGenerator[AsyncSession]:
        async with session_factory() as session:
            yield session

    # Create a minimal mock for the session registry
    mock_registry = MagicMock()
    mock_registry.get_cached.return_value = None
    mock_registry.put = AsyncMock()
    mock_registry.remove = AsyncMock()
    mock_registry.start_eviction_loop = MagicMock()
    mock_registry.stop_eviction_loop = AsyncMock()

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    app.state.session_registry = mock_registry
    app.state.config = None
    app.state.tool_registry = None

    # Patch warm_session at the route module level so the top-level import is intercepted
    with patch(
        "cogtrix_core.api.routes.sessions.warm_session", new_callable=AsyncMock
    ) as mock_warm:

        async def _fake_warm(record, app_state):
            from cogtrix_core.api.session_bridge import ApiSession

            return ApiSession(
                id=record.id,
                user_id=record.user_id,
                name=record.name,
                config={},
                agent_state="idle",
            )

        mock_warm.side_effect = _fake_warm

        with TestClient(app, raise_server_exceptions=True) as client:
            yield client, session_factory


def _make_token(user_id: str, role: str = "user") -> str:
    return create_access_token(user_id, role)


async def _create_db_user(session_factory, user_id: str, role: str = "user") -> None:
    """Helper: insert a user into the test DB."""
    async with session_factory() as db:
        repo = UserRepository(db)
        await repo.create(
            user_id=user_id,
            username=f"user_{user_id[:8]}",
            email=f"{user_id[:8]}@test.com",
            password_hash="h",
            role=role,
        )
        await db.commit()


# ---------------------------------------------------------------------------
# POST /sessions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_session_success(http_client) -> None:
    client, session_factory = http_client
    uid = str(uuid.uuid4())
    await _create_db_user(session_factory, uid)
    token = _make_token(uid)

    resp = client.post(
        "/api/v1/sessions",
        json={"name": "My session", "config": {"provider": "openai"}},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["data"]["name"] == "My session"
    assert data["data"]["state"] == "idle"
    assert data["data"]["owner_id"] == uid
    assert "id" in data["data"]


@pytest.mark.asyncio
async def test_create_session_requires_auth(http_client) -> None:
    client, _ = http_client
    resp = client.post("/api/v1/sessions", json={"name": "No auth"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_create_session_default_name(http_client) -> None:
    client, session_factory = http_client
    uid = str(uuid.uuid4())
    await _create_db_user(session_factory, uid)
    token = _make_token(uid)

    resp = client.post(
        "/api/v1/sessions",
        json={},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 201
    assert resp.json()["data"]["name"].startswith("Session ")


# ---------------------------------------------------------------------------
# GET /sessions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_sessions_empty(http_client) -> None:
    client, session_factory = http_client
    uid = str(uuid.uuid4())
    await _create_db_user(session_factory, uid)
    token = _make_token(uid)

    resp = client.get("/api/v1/sessions", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["items"] == []
    assert body["data"]["has_more"] is False


@pytest.mark.asyncio
async def test_list_sessions_returns_own_only(http_client) -> None:
    client, session_factory = http_client
    uid1 = str(uuid.uuid4())
    uid2 = str(uuid.uuid4())
    await _create_db_user(session_factory, uid1)
    await _create_db_user(session_factory, uid2)

    # User 1 creates 2 sessions; user 2 creates 1
    token1 = _make_token(uid1)
    token2 = _make_token(uid2)

    for i in range(2):
        client.post(
            "/api/v1/sessions",
            json={"name": f"User1 session {i}"},
            headers={"Authorization": f"Bearer {token1}"},
        )
    client.post(
        "/api/v1/sessions",
        json={"name": "User2 session"},
        headers={"Authorization": f"Bearer {token2}"},
    )

    resp = client.get("/api/v1/sessions", headers={"Authorization": f"Bearer {token1}"})
    assert resp.status_code == 200
    items = resp.json()["data"]["items"]
    assert len(items) == 2
    assert all(s["owner_id"] == uid1 for s in items)


@pytest.mark.asyncio
async def test_list_sessions_admin_sees_all(http_client) -> None:
    client, session_factory = http_client
    uid = str(uuid.uuid4())
    admin = str(uuid.uuid4())
    await _create_db_user(session_factory, uid, role="user")
    await _create_db_user(session_factory, admin, role="admin")

    user_token = _make_token(uid)
    admin_token = _make_token(admin, role="admin")

    client.post(
        "/api/v1/sessions",
        json={"name": "User session"},
        headers={"Authorization": f"Bearer {user_token}"},
    )
    client.post(
        "/api/v1/sessions",
        json={"name": "Admin session"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    # Admin should see both sessions
    resp = client.get("/api/v1/sessions", headers={"Authorization": f"Bearer {admin_token}"})
    assert resp.status_code == 200
    assert len(resp.json()["data"]["items"]) == 2


@pytest.mark.asyncio
async def test_list_sessions_pagination(http_client) -> None:
    client, session_factory = http_client
    uid = str(uuid.uuid4())
    await _create_db_user(session_factory, uid)
    token = _make_token(uid)

    for i in range(5):
        client.post(
            "/api/v1/sessions",
            json={"name": f"S{i}"},
            headers={"Authorization": f"Bearer {token}"},
        )

    resp = client.get("/api/v1/sessions?limit=3", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    body = resp.json()["data"]
    assert len(body["items"]) == 3
    assert body["has_more"] is True
    assert body["next_cursor"] is not None

    # Follow next cursor
    cursor = body["next_cursor"]
    resp2 = client.get(
        f"/api/v1/sessions?limit=3&cursor={cursor}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp2.status_code == 200
    body2 = resp2.json()["data"]
    assert len(body2["items"]) == 2
    assert body2["has_more"] is False

    # No overlap
    ids1 = {s["id"] for s in body["items"]}
    ids2 = {s["id"] for s in body2["items"]}
    assert ids1.isdisjoint(ids2)


# ---------------------------------------------------------------------------
# GET /sessions/{id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_session_success(http_client) -> None:
    client, session_factory = http_client
    uid = str(uuid.uuid4())
    await _create_db_user(session_factory, uid)
    token = _make_token(uid)

    create_resp = client.post(
        "/api/v1/sessions",
        json={"name": "Detail test"},
        headers={"Authorization": f"Bearer {token}"},
    )
    session_id = create_resp.json()["data"]["id"]

    resp = client.get(
        f"/api/v1/sessions/{session_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["data"]["id"] == session_id


@pytest.mark.asyncio
async def test_get_session_not_found(http_client) -> None:
    client, session_factory = http_client
    uid = str(uuid.uuid4())
    await _create_db_user(session_factory, uid)
    token = _make_token(uid)

    resp = client.get(
        "/api/v1/sessions/nonexistent-id",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "SESSION_NOT_FOUND"


@pytest.mark.asyncio
async def test_get_session_ownership_enforced(http_client) -> None:
    """A regular user cannot access another user's session."""
    client, session_factory = http_client
    uid1 = str(uuid.uuid4())
    uid2 = str(uuid.uuid4())
    await _create_db_user(session_factory, uid1)
    await _create_db_user(session_factory, uid2)

    token1 = _make_token(uid1)
    token2 = _make_token(uid2)

    create_resp = client.post(
        "/api/v1/sessions",
        json={"name": "Private"},
        headers={"Authorization": f"Bearer {token1}"},
    )
    session_id = create_resp.json()["data"]["id"]

    # User 2 should be forbidden
    resp = client.get(
        f"/api/v1/sessions/{session_id}",
        headers={"Authorization": f"Bearer {token2}"},
    )
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "FORBIDDEN"


@pytest.mark.asyncio
async def test_get_session_admin_bypass(http_client) -> None:
    """Admin can access any session."""
    client, session_factory = http_client
    uid = str(uuid.uuid4())
    admin = str(uuid.uuid4())
    await _create_db_user(session_factory, uid, role="user")
    await _create_db_user(session_factory, admin, role="admin")

    user_token = _make_token(uid)
    admin_token = _make_token(admin, role="admin")

    create_resp = client.post(
        "/api/v1/sessions",
        json={"name": "User's session"},
        headers={"Authorization": f"Bearer {user_token}"},
    )
    session_id = create_resp.json()["data"]["id"]

    resp = client.get(
        f"/api/v1/sessions/{session_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# PATCH /sessions/{id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_patch_session_name(http_client) -> None:
    client, session_factory = http_client
    uid = str(uuid.uuid4())
    await _create_db_user(session_factory, uid)
    token = _make_token(uid)

    create_resp = client.post(
        "/api/v1/sessions",
        json={"name": "Original"},
        headers={"Authorization": f"Bearer {token}"},
    )
    session_id = create_resp.json()["data"]["id"]

    patch_resp = client.patch(
        f"/api/v1/sessions/{session_id}",
        json={"name": "Renamed"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["data"]["name"] == "Renamed"


@pytest.mark.asyncio
async def test_patch_session_config(http_client) -> None:
    client, session_factory = http_client
    uid = str(uuid.uuid4())
    await _create_db_user(session_factory, uid)
    token = _make_token(uid)

    create_resp = client.post(
        "/api/v1/sessions",
        json={"name": "Session", "config": {"model": "gpt-4o", "max_steps": 10}},
        headers={"Authorization": f"Bearer {token}"},
    )
    session_id = create_resp.json()["data"]["id"]

    patch_resp = client.patch(
        f"/api/v1/sessions/{session_id}",
        json={"config": {"max_steps": 25}},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert patch_resp.status_code == 200
    cfg = patch_resp.json()["data"]["config"]
    assert cfg["max_steps"] == 25
    assert cfg.get("model") == "gpt-4o"  # preserved from original config


@pytest.mark.asyncio
async def test_patch_session_ownership_enforced(http_client) -> None:
    client, session_factory = http_client
    uid1 = str(uuid.uuid4())
    uid2 = str(uuid.uuid4())
    await _create_db_user(session_factory, uid1)
    await _create_db_user(session_factory, uid2)

    token1 = _make_token(uid1)
    token2 = _make_token(uid2)

    create_resp = client.post(
        "/api/v1/sessions",
        json={"name": "Protected"},
        headers={"Authorization": f"Bearer {token1}"},
    )
    session_id = create_resp.json()["data"]["id"]

    resp = client.patch(
        f"/api/v1/sessions/{session_id}",
        json={"name": "Hacked"},
        headers={"Authorization": f"Bearer {token2}"},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_patch_session_not_found(http_client) -> None:
    client, session_factory = http_client
    uid = str(uuid.uuid4())
    await _create_db_user(session_factory, uid)
    token = _make_token(uid)

    resp = client.patch(
        "/api/v1/sessions/does-not-exist",
        json={"name": "X"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /sessions/{id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_session_success(http_client) -> None:
    client, session_factory = http_client
    uid = str(uuid.uuid4())
    await _create_db_user(session_factory, uid)
    token = _make_token(uid)

    create_resp = client.post(
        "/api/v1/sessions",
        json={"name": "To delete"},
        headers={"Authorization": f"Bearer {token}"},
    )
    session_id = create_resp.json()["data"]["id"]

    del_resp = client.delete(
        f"/api/v1/sessions/{session_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert del_resp.status_code == 200
    assert del_resp.json()["data"] is None

    # Session should now be excluded from list
    list_resp = client.get("/api/v1/sessions", headers={"Authorization": f"Bearer {token}"})
    items = list_resp.json()["data"]["items"]
    assert not any(s["id"] == session_id for s in items)


@pytest.mark.asyncio
async def test_delete_session_appears_in_archived(http_client) -> None:
    client, session_factory = http_client
    uid = str(uuid.uuid4())
    await _create_db_user(session_factory, uid)
    token = _make_token(uid)

    create_resp = client.post(
        "/api/v1/sessions",
        json={"name": "Archived session"},
        headers={"Authorization": f"Bearer {token}"},
    )
    session_id = create_resp.json()["data"]["id"]

    client.delete(
        f"/api/v1/sessions/{session_id}",
        headers={"Authorization": f"Bearer {token}"},
    )

    # With include_archived=true it should appear
    list_resp = client.get(
        "/api/v1/sessions?include_archived=true",
        headers={"Authorization": f"Bearer {token}"},
    )
    ids = [s["id"] for s in list_resp.json()["data"]["items"]]
    assert session_id in ids


@pytest.mark.asyncio
async def test_delete_session_ownership_enforced(http_client) -> None:
    client, session_factory = http_client
    uid1 = str(uuid.uuid4())
    uid2 = str(uuid.uuid4())
    await _create_db_user(session_factory, uid1)
    await _create_db_user(session_factory, uid2)

    token1 = _make_token(uid1)
    token2 = _make_token(uid2)

    create_resp = client.post(
        "/api/v1/sessions",
        json={"name": "Protected"},
        headers={"Authorization": f"Bearer {token1}"},
    )
    session_id = create_resp.json()["data"]["id"]

    resp = client.delete(
        f"/api/v1/sessions/{session_id}",
        headers={"Authorization": f"Bearer {token2}"},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_delete_session_not_found(http_client) -> None:
    client, session_factory = http_client
    uid = str(uuid.uuid4())
    await _create_db_user(session_factory, uid)
    token = _make_token(uid)

    resp = client.delete(
        "/api/v1/sessions/nonexistent",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Session bridge warm / evict lifecycle tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_bridge_warm(db_session: AsyncSession, user_id: str) -> None:
    """warm_session builds a minimal ApiSession from a DB record."""
    from cogtrix_core.api.session_bridge import ApiSession, warm_session

    repo = SessionRepository(db_session)
    record = await repo.create(
        user_id=user_id,
        name="Bridge test",
        config_json=json.dumps({"memory_mode": "conversation"}),
    )
    await db_session.commit()

    # Mock app_state without a real LLM
    app_state = MagicMock()
    app_state.config = None
    app_state.tool_registry = None

    mock_llm = MagicMock()
    with patch("cogtrix_core.api.session_bridge._build_llm", return_value=mock_llm):
        session = await warm_session(record, app_state)

    assert isinstance(session, ApiSession)
    assert session.id == record.id
    assert session.user_id == user_id
    assert session.name == "Bridge test"
    assert session.agent_state == "idle"
    assert session._lock is not None
    assert session.cancel_event is not None
    assert session.ws_queue is not None


@pytest.mark.asyncio
async def test_session_registry_get_or_warm(db_session: AsyncSession, user_id: str) -> None:
    """ApiSessionRegistry.get_or_warm loads from DB and caches the result."""
    from cogtrix_core.api.session_bridge import ApiSessionRegistry

    repo = SessionRepository(db_session)
    record = await repo.create(user_id=user_id, name="Reg test")
    await db_session.commit()

    app_state = MagicMock()
    app_state.config = None
    app_state.tool_registry = None

    registry = ApiSessionRegistry(app_state)

    mock_llm = MagicMock()
    with patch("cogtrix_core.api.session_bridge._build_llm", return_value=mock_llm):
        sess1 = await registry.get_or_warm(record.id, db_session)
        sess2 = await registry.get_or_warm(record.id, db_session)

    assert sess1 is not None
    assert sess2 is sess1  # cached


@pytest.mark.asyncio
async def test_session_registry_get_or_warm_missing(db_session: AsyncSession) -> None:
    """get_or_warm returns None for a non-existent session ID."""
    from cogtrix_core.api.session_bridge import ApiSessionRegistry

    app_state = MagicMock()
    registry = ApiSessionRegistry(app_state)

    result = await registry.get_or_warm("non-existent-id", db_session)
    assert result is None


@pytest.mark.asyncio
async def test_session_registry_evict_idle(db_session: AsyncSession, user_id: str) -> None:
    """evict_idle removes sessions that have been idle longer than max_age."""
    import time

    from cogtrix_core.api.session_bridge import ApiSession, ApiSessionRegistry

    app_state = MagicMock()
    registry = ApiSessionRegistry(app_state)

    # Insert a session that has been idle for a long time
    old_session = ApiSession(
        id=str(uuid.uuid4()),
        user_id=user_id,
        name="Old",
    )
    old_session.last_activity = time.time() - 9999

    new_session = ApiSession(
        id=str(uuid.uuid4()),
        user_id=user_id,
        name="New",
    )

    async with registry._lock:
        registry._sessions[old_session.id] = old_session
        registry._sessions[new_session.id] = new_session

    evicted = await registry.evict_idle(max_age_seconds=100)
    assert evicted == 1
    assert old_session.id not in registry._sessions
    assert new_session.id in registry._sessions


@pytest.mark.asyncio
async def test_session_registry_remove(db_session: AsyncSession, user_id: str) -> None:
    """remove() saves memory and evicts from registry."""
    from cogtrix_core.api.session_bridge import ApiSession, ApiSessionRegistry

    app_state = MagicMock()
    registry = ApiSessionRegistry(app_state)

    sess = ApiSession(id=str(uuid.uuid4()), user_id=user_id, name="To remove")
    mock_mm = MagicMock()
    sess.memory_manager = mock_mm

    async with registry._lock:
        registry._sessions[sess.id] = sess

    await registry.remove(sess.id)
    assert sess.id not in registry._sessions
    mock_mm.save.assert_called_once()
