"""Phase 1 API tests: database layer, JWT auth, auth endpoints, health endpoints,
and app factory smoke tests.

All tests use an in-memory SQLite database via a test fixture so they never
touch the real data/api/cogtrix.db.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

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
# Imports after env is set
# ---------------------------------------------------------------------------

from src.api.auth import (  # noqa: E402
    _decode_jwt,
    create_access_token,
)
from src.api.db import models as _models  # noqa: E402, F401
from src.api.db.engine import Base  # noqa: E402
from src.api.db.repositories.api_keys import ApiKeyRepository  # noqa: E402
from src.api.db.repositories.tokens import RefreshTokenRepository  # noqa: E402
from src.api.db.repositories.users import UserRepository  # noqa: E402

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


# ---------------------------------------------------------------------------
# DB model tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_user_create_and_retrieve(db_session: AsyncSession) -> None:
    """UserRepository.create + get_by_id round-trips correctly."""
    repo = UserRepository(db_session)
    uid = str(uuid.uuid4())
    _user = await repo.create(
        user_id=uid,
        username="alice",
        email="alice@example.com",
        password_hash="hashed",
        role="admin",
    )
    await db_session.commit()

    fetched = await repo.get_by_id(uid)
    assert fetched is not None
    assert fetched.username == "alice"
    assert fetched.role == "admin"


@pytest.mark.asyncio
async def test_user_get_by_username_and_email(db_session: AsyncSession) -> None:
    """get_by_username and get_by_email work correctly."""
    repo = UserRepository(db_session)
    uid = str(uuid.uuid4())
    await repo.create(
        user_id=uid,
        username="bob",
        email="BOB@example.com",
        password_hash="hashed",
    )
    await db_session.commit()

    assert await repo.get_by_username("bob") is not None
    assert await repo.get_by_username("nonexistent") is None
    assert await repo.get_by_email("bob@example.com") is not None  # case-insensitive
    assert await repo.get_by_email("unknown@example.com") is None


@pytest.mark.asyncio
async def test_create_with_role_election(db_session: AsyncSession) -> None:
    """First user gets admin role, subsequent users get user role."""
    repo = UserRepository(db_session)
    first = await repo.create_with_role_election(
        user_id=str(uuid.uuid4()),
        username="first",
        email="first@x.com",
        password_hash="h",
    )
    await db_session.commit()
    assert first.role == "admin"

    second = await repo.create_with_role_election(
        user_id=str(uuid.uuid4()),
        username="second",
        email="second@x.com",
        password_hash="h",
    )
    await db_session.commit()
    assert second.role == "user"


@pytest.mark.asyncio
async def test_refresh_token_create_and_revoke(db_session: AsyncSession) -> None:
    """RefreshTokenRepository create, get_by_hash, revoke work correctly."""
    # Create a user first
    user_repo = UserRepository(db_session)
    user = await user_repo.create(
        user_id=str(uuid.uuid4()),
        username="charlie",
        email="charlie@example.com",
        password_hash="h",
    )
    await db_session.commit()

    raw = secrets.token_urlsafe(48)
    token_hash = hashlib.sha256(raw.encode()).hexdigest()
    tok_id = str(uuid.uuid4())

    repo = RefreshTokenRepository(db_session)
    _token = await repo.create(
        token_id=tok_id,
        user_id=user.id,
        token_hash=token_hash,
        expires_at=datetime.now(UTC) + timedelta(days=30),
    )
    await db_session.commit()

    fetched = await repo.get_by_hash(token_hash)
    assert fetched is not None
    assert fetched.revoked is False

    await repo.revoke(tok_id)
    await db_session.commit()

    fetched_after = await repo.get_by_hash(token_hash)
    assert fetched_after is not None
    assert fetched_after.revoked is True


@pytest.mark.asyncio
async def test_api_key_create_and_list(db_session: AsyncSession) -> None:
    """ApiKeyRepository create and list_for_user work."""
    user_repo = UserRepository(db_session)
    user = await user_repo.create(
        user_id=str(uuid.uuid4()),
        username="dave",
        email="dave@example.com",
        password_hash="h",
    )
    await db_session.commit()

    repo = ApiKeyRepository(db_session)
    raw_key = "cgx_live_" + secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    _key = await repo.create(
        key_id=str(uuid.uuid4()),
        user_id=user.id,
        key_hash=key_hash,
        key_prefix=raw_key[:12],
        label="test key",
    )
    await db_session.commit()

    keys = await repo.list_for_user(user.id)
    assert len(keys) == 1
    assert keys[0].label == "test key"


# ---------------------------------------------------------------------------
# JWT encode/decode tests
# ---------------------------------------------------------------------------


def test_create_and_decode_access_token() -> None:
    """create_access_token produces a token that _decode_jwt can verify."""
    with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
        token = create_access_token("user-123", "admin")
        claims = _decode_jwt(token)
        assert claims["sub"] == "user-123"
        assert claims["role"] == "admin"


# ---------------------------------------------------------------------------
# FastAPI app fixture with isolated in-memory DB
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_app():
    """Return the FastAPI app with DB patched to use in-memory SQLite."""
    import asyncio

    from src.api.db.engine import Base as _Base

    # Build a dedicated in-memory engine for the test app
    test_engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    test_session_factory = async_sessionmaker(test_engine, expire_on_commit=False)

    # Synchronously create tables
    async def _create():
        async with test_engine.begin() as conn:
            await conn.run_sync(_Base.metadata.create_all)

    asyncio.get_event_loop().run_until_complete(_create())

    # Import after env vars are set
    with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
        from src.api.app import create_app
        from src.api.db.engine import get_db

        app = create_app()

        # Override get_db to use the test session
        async def _override_get_db():
            async with test_session_factory() as session:
                try:
                    yield session
                    await session.commit()
                except Exception:
                    await session.rollback()
                    raise

        app.dependency_overrides[get_db] = _override_get_db

        yield app

    # Cleanup
    asyncio.get_event_loop().run_until_complete(test_engine.dispose())


@pytest.fixture()
def client(test_app):
    """Return a synchronous TestClient backed by the test app."""
    # Bypass lifespan so we don't need JWT secret validated at startup
    with TestClient(test_app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# Health endpoint tests
# ---------------------------------------------------------------------------


def test_liveness(client: TestClient) -> None:
    """GET /api/v1/health returns 200 with status ok."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["data"]["status"] == "ok"
    assert body["error"] is None


def test_readiness_without_tool_registry(client: TestClient) -> None:
    """GET /api/v1/health/ready returns a ReadinessOut with component list."""
    response = client.get("/api/v1/health/ready")
    assert response.status_code in (200, 503)
    body = response.json()
    assert "ready" in body["data"]
    assert isinstance(body["data"]["components"], list)


# ---------------------------------------------------------------------------
# Auth endpoint tests
# ---------------------------------------------------------------------------


def test_register_creates_first_admin(client: TestClient) -> None:
    """POST /auth/register: first user gets admin role, returns token pair."""
    resp = client.post(
        "/api/v1/auth/register",
        json={"username": "admin1", "email": "admin1@test.com", "password": "Password1!"},
    )
    assert resp.status_code == 201
    data = resp.json()["data"]
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"
    assert data["expires_in"] == 3600


def test_register_second_user_gets_user_role(client: TestClient) -> None:
    """POST /auth/register: subsequent users get 'user' role."""
    # First user (admin)
    client.post(
        "/api/v1/auth/register",
        json={"username": "first", "email": "first@test.com", "password": "Password1!"},
    )
    # Second user
    resp = client.post(
        "/api/v1/auth/register",
        json={"username": "second", "email": "second@test.com", "password": "Password2!"},
    )
    assert resp.status_code == 201
    # Decode the token to verify the role
    token = resp.json()["data"]["access_token"]
    with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
        claims = _decode_jwt(token)
    assert claims["role"] == "user"


def test_register_duplicate_username_409(client: TestClient) -> None:
    """POST /auth/register: duplicate username returns 409."""
    client.post(
        "/api/v1/auth/register",
        json={"username": "dup", "email": "dup@test.com", "password": "Password1!"},
    )
    resp = client.post(
        "/api/v1/auth/register",
        json={"username": "dup", "email": "other@test.com", "password": "Password1!"},
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "VALIDATION_ERROR"


def test_register_duplicate_email_409(client: TestClient) -> None:
    """POST /auth/register: duplicate email returns 409."""
    client.post(
        "/api/v1/auth/register",
        json={"username": "user1", "email": "shared@test.com", "password": "Password1!"},
    )
    resp = client.post(
        "/api/v1/auth/register",
        json={"username": "user2", "email": "shared@test.com", "password": "Password2!"},
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "VALIDATION_ERROR"


def test_login_valid_credentials(client: TestClient) -> None:
    """POST /auth/login: valid creds return token pair."""
    client.post(
        "/api/v1/auth/register",
        json={"username": "loginuser", "email": "login@test.com", "password": "MyPass123!"},
    )
    resp = client.post(
        "/api/v1/auth/login",
        json={"username": "loginuser", "password": "MyPass123!"},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert "access_token" in data
    assert "refresh_token" in data


def test_get_me_authenticated(client: TestClient) -> None:
    """GET /auth/me: returns user profile for valid token."""
    register_resp = client.post(
        "/api/v1/auth/register",
        json={"username": "meuser", "email": "me@test.com", "password": "Password1!"},
    )
    token = register_resp.json()["data"]["access_token"]

    resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["username"] == "meuser"
    assert data["email"] == "me@test.com"


def test_get_me_no_token_401(client: TestClient) -> None:
    """GET /auth/me: missing token returns 401."""
    resp = client.get("/api/v1/auth/me")
    assert resp.status_code == 401


def test_refresh_token_rotation(client: TestClient) -> None:
    """POST /auth/refresh: returns new token pair and old refresh token is rotated."""
    reg = client.post(
        "/api/v1/auth/register",
        json={"username": "refresher", "email": "refresh@test.com", "password": "Password1!"},
    )
    old_refresh = reg.json()["data"]["refresh_token"]

    resp = client.post("/api/v1/auth/refresh", json={"refresh_token": old_refresh})
    assert resp.status_code == 200
    new_data = resp.json()["data"]
    assert "access_token" in new_data
    assert "refresh_token" in new_data
    # Old token is now revoked
    resp2 = client.post("/api/v1/auth/refresh", json={"refresh_token": old_refresh})
    assert resp2.status_code == 401


def test_logout_revokes_refresh_tokens(client: TestClient) -> None:
    """POST /auth/logout: revokes all refresh tokens for user."""
    reg = client.post(
        "/api/v1/auth/register",
        json={"username": "logouter", "email": "logout@test.com", "password": "Password1!"},
    )
    data = reg.json()["data"]
    access = data["access_token"]
    refresh = data["refresh_token"]

    resp = client.post(
        "/api/v1/auth/logout",
        headers={"Authorization": f"Bearer {access}"},
    )
    assert resp.status_code == 200

    # Refresh should now fail
    resp2 = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh})
    assert resp2.status_code == 401


def test_create_and_list_api_keys(client: TestClient) -> None:
    """POST + GET /auth/api-keys: create key, then list it."""
    reg = client.post(
        "/api/v1/auth/register",
        json={"username": "keyuser", "email": "key@test.com", "password": "Password1!"},
    )
    token = reg.json()["data"]["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    create_resp = client.post(
        "/api/v1/auth/api-keys",
        json={"label": "my key", "expires_in_days": 30},
        headers=headers,
    )
    assert create_resp.status_code == 201
    key_data = create_resp.json()["data"]
    assert key_data["key"].startswith("cgx_live_")
    assert key_data["key_prefix"] == key_data["key"][:12]

    list_resp = client.get("/api/v1/auth/api-keys", headers=headers)
    assert list_resp.status_code == 200
    items = list_resp.json()["data"]["items"]
    assert len(items) == 1
    assert items[0]["label"] == "my key"
    assert items[0]["key"] is None  # full key not returned in list


def test_revoke_api_key(client: TestClient) -> None:
    """DELETE /auth/api-keys/{id}: revokes the key, removing it from list."""
    reg = client.post(
        "/api/v1/auth/register",
        json={"username": "revoker", "email": "revoke@test.com", "password": "Password1!"},
    )
    token = reg.json()["data"]["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    create_resp = client.post(
        "/api/v1/auth/api-keys",
        json={"label": "to revoke"},
        headers=headers,
    )
    key_id = create_resp.json()["data"]["id"]

    del_resp = client.delete(f"/api/v1/auth/api-keys/{key_id}", headers=headers)
    assert del_resp.status_code == 200

    # Key no longer appears in list
    list_resp = client.get("/api/v1/auth/api-keys", headers=headers)
    assert list_resp.json()["data"]["items"] == []


def test_revoke_other_users_key_403(client: TestClient) -> None:
    """DELETE /auth/api-keys/{id}: cannot revoke another user's key."""
    # User A creates a key
    reg_a = client.post(
        "/api/v1/auth/register",
        json={"username": "usera", "email": "usera@test.com", "password": "Password1!"},
    )
    token_a = reg_a.json()["data"]["access_token"]
    create_resp = client.post(
        "/api/v1/auth/api-keys",
        json={"label": "user a key"},
        headers={"Authorization": f"Bearer {token_a}"},
    )
    key_id = create_resp.json()["data"]["id"]

    # User B tries to revoke it
    reg_b = client.post(
        "/api/v1/auth/register",
        json={"username": "userb", "email": "userb@test.com", "password": "Password2!"},
    )
    token_b = reg_b.json()["data"]["access_token"]
    del_resp = client.delete(
        f"/api/v1/auth/api-keys/{key_id}",
        headers={"Authorization": f"Bearer {token_b}"},
    )
    assert del_resp.status_code == 403
    assert del_resp.json()["error"]["code"] == "FORBIDDEN"


def test_revoke_nonexistent_key_404(client: TestClient) -> None:
    """DELETE /auth/api-keys/{id}: 404 for unknown key."""
    reg = client.post(
        "/api/v1/auth/register",
        json={"username": "notfound", "email": "nf@test.com", "password": "Password1!"},
    )
    token = reg.json()["data"]["access_token"]
    resp = client.delete(
        f"/api/v1/auth/api-keys/{uuid.uuid4()}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "NOT_FOUND"


# ---------------------------------------------------------------------------
# App factory smoke test
# ---------------------------------------------------------------------------


def test_app_factory_creates_app() -> None:
    """create_app() returns a FastAPI app with correct metadata."""
    with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
        from src.api.app import create_app

        app = create_app()
    assert app.title == "Cogtrix API"
    assert app.version == "1.1.0"


def test_openapi_schema_available() -> None:
    """OpenAPI schema endpoint is accessible."""
    with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
        from src.api.app import create_app

        app = create_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        resp = c.get("/api/v1/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert schema["info"]["title"] == "Cogtrix API"
