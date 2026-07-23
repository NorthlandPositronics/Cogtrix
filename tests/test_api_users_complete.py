"""Comprehensive users endpoint coverage.

Covers every endpoint in cogtrix_core/api/routes/users.py:
    GET    /api/v1/users                — list all users (admin)
    POST   /api/v1/users               — create user (admin)
    PATCH  /api/v1/users/{user_id}     — update role (admin)
    DELETE /api/v1/users/{user_id}     — delete user (admin)

All auth permutations: unauthenticated, non-admin, admin.
All error codes: UNAUTHORIZED, FORBIDDEN, NOT_FOUND, BAD_REQUEST.
All validation: username/email/password/role field rules.
"""

from __future__ import annotations

import asyncio as _asyncio
import os
import uuid

import pytest

pytest.importorskip("fastapi")

_TEST_JWT_SECRET = "testsecret_mustbe32chars_minimum00"
os.environ.setdefault("COGTRIX_JWT_SECRET", _TEST_JWT_SECRET)
os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

from unittest.mock import patch  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from cogtrix_core.api.db.engine import Base, get_db  # noqa: E402

_VALID_PASSWORD = "TestPass1!"  # lowercase + uppercase + digit + special


# ---------------------------------------------------------------------------
# Fixtures — function-scope so each test gets a clean DB
# ---------------------------------------------------------------------------


@pytest.fixture()
def app():
    """FastAPI app backed by a fresh in-memory SQLite DB per test."""
    from cogtrix_core.api.app import create_app

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _setup():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    loop = _asyncio.new_event_loop()
    _asyncio.set_event_loop(loop)
    loop.run_until_complete(_setup())

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

    loop.run_until_complete(engine.dispose())
    loop.close()


@pytest.fixture()
def client(app):
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _register(client, username=None, email=None, password=_VALID_PASSWORD):
    if username is None:
        username = f"u_{uuid.uuid4().hex[:8]}"
    if email is None:
        email = f"{username}@ex.com"
    r = client.post(
        "/api/v1/auth/register",
        json={"username": username, "email": email, "password": password},
    )
    assert r.status_code == 201, f"register failed: {r.text}"
    return r


def _create_user_as_admin(client, admin_headers, username=None, role="user"):
    if username is None:
        username = f"u_{uuid.uuid4().hex[:8]}"
    r = client.post(
        "/api/v1/users",
        headers=admin_headers,
        json={
            "username": username,
            "email": f"{username}@ex.com",
            "password": _VALID_PASSWORD,
            "role": role,
        },
    )
    return r


@pytest.fixture()
def admin_client(client):
    """Returns (client, admin_headers, admin_user_id, user_headers, user_user_id)."""
    # First registered user becomes admin
    admin_r = _register(client)
    admin_token = admin_r.json()["data"]["access_token"]
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    # Decode admin user id
    import jwt as jose_jwt

    claims = jose_jwt.decode(admin_token, options={"verify_signature": False})
    admin_user_id = claims["sub"]

    # Create a regular user via admin endpoint
    uname = f"usr_{uuid.uuid4().hex[:6]}"
    user_r = _create_user_as_admin(client, admin_headers, username=uname, role="user")
    user_id = user_r.json()["data"]["id"]

    # Login as that user to get their token
    login_r = client.post(
        "/api/v1/auth/login",
        json={"username": uname, "password": _VALID_PASSWORD},
    )
    user_token = login_r.json()["data"]["access_token"]
    user_headers = {"Authorization": f"Bearer {user_token}"}

    return {
        "client": client,
        "admin_headers": admin_headers,
        "admin_user_id": admin_user_id,
        "user_headers": user_headers,
        "user_user_id": user_id,
    }


# ---------------------------------------------------------------------------
# List Users
# ---------------------------------------------------------------------------


class TestListUsers:
    def test_admin_can_list_users(self, client, admin_client):
        r = client.get("/api/v1/users", headers=admin_client["admin_headers"])
        assert r.status_code == 200
        body = r.json()
        assert body["error"] is None
        assert isinstance(body["data"], list)

    def test_list_users_returns_expected_fields(self, client, admin_client):
        r = client.get("/api/v1/users", headers=admin_client["admin_headers"])
        assert r.status_code == 200
        for user in r.json()["data"]:
            assert "id" in user
            assert "username" in user
            assert "email" in user
            assert "role" in user
            assert "created_at" in user

    def test_non_admin_returns_403(self, client, admin_client):
        r = client.get("/api/v1/users", headers=admin_client["user_headers"])
        assert r.status_code == 403
        assert r.json()["error"]["code"] == "FORBIDDEN"

    def test_no_auth_returns_401(self, client):
        r = client.get("/api/v1/users")
        assert r.status_code == 401

    def test_list_includes_created_users(self, client, admin_client):
        uname = f"listed_{uuid.uuid4().hex[:6]}"
        _create_user_as_admin(client, admin_client["admin_headers"], username=uname)
        r = client.get("/api/v1/users", headers=admin_client["admin_headers"])
        usernames = [u["username"] for u in r.json()["data"]]
        assert uname in usernames


# ---------------------------------------------------------------------------
# Create User
# ---------------------------------------------------------------------------


class TestCreateUser:
    def test_admin_can_create_user(self, client, admin_client):
        uname = f"new_{uuid.uuid4().hex[:6]}"
        r = _create_user_as_admin(client, admin_client["admin_headers"], username=uname)
        assert r.status_code == 201
        body = r.json()
        assert body["error"] is None
        assert body["data"]["username"] == uname
        assert body["data"]["role"] == "user"

    def test_admin_can_create_admin_user(self, client, admin_client):
        uname = f"adm2_{uuid.uuid4().hex[:6]}"
        r = _create_user_as_admin(
            client, admin_client["admin_headers"], username=uname, role="admin"
        )
        assert r.status_code == 201
        assert r.json()["data"]["role"] == "admin"

    def test_non_admin_returns_403(self, client, admin_client):
        r = client.post(
            "/api/v1/users",
            headers=admin_client["user_headers"],
            json={
                "username": "badattempt",
                "email": "bad@ex.com",
                "password": _VALID_PASSWORD,
                "role": "user",
            },
        )
        assert r.status_code == 403

    def test_no_auth_returns_401(self, client):
        r = client.post(
            "/api/v1/users",
            json={
                "username": "nope",
                "email": "n@ex.com",
                "password": _VALID_PASSWORD,
                "role": "user",
            },
        )
        assert r.status_code == 401

    def test_duplicate_username_returns_409(self, client, admin_client):
        uname = f"dupusr_{uuid.uuid4().hex[:6]}"
        _create_user_as_admin(client, admin_client["admin_headers"], username=uname)
        r = client.post(
            "/api/v1/users",
            headers=admin_client["admin_headers"],
            json={
                "username": uname,
                "email": f"other_{uname}@ex.com",
                "password": _VALID_PASSWORD,
                "role": "user",
            },
        )
        assert r.status_code == 409

    def test_duplicate_email_returns_409(self, client, admin_client):
        email = f"shared_{uuid.uuid4().hex[:6]}@ex.com"
        uname1 = f"u1_{uuid.uuid4().hex[:6]}"
        uname2 = f"u2_{uuid.uuid4().hex[:6]}"
        client.post(
            "/api/v1/users",
            headers=admin_client["admin_headers"],
            json={
                "username": uname1,
                "email": email,
                "password": _VALID_PASSWORD,
                "role": "user",
            },
        )
        r = client.post(
            "/api/v1/users",
            headers=admin_client["admin_headers"],
            json={
                "username": uname2,
                "email": email,
                "password": _VALID_PASSWORD,
                "role": "user",
            },
        )
        assert r.status_code == 409

    def test_invalid_role_returns_422(self, client, admin_client):
        uname = f"ir_{uuid.uuid4().hex[:6]}"
        r = client.post(
            "/api/v1/users",
            headers=admin_client["admin_headers"],
            json={
                "username": uname,
                "email": f"{uname}@ex.com",
                "password": _VALID_PASSWORD,
                "role": "superuser",
            },
        )
        assert r.status_code == 422

    def test_short_password_returns_422(self, client, admin_client):
        uname = f"sp_{uuid.uuid4().hex[:6]}"
        r = client.post(
            "/api/v1/users",
            headers=admin_client["admin_headers"],
            json={
                "username": uname,
                "email": f"{uname}@ex.com",
                "password": "Sh0rt!",
                "role": "user",
            },
        )
        assert r.status_code == 422

    def test_invalid_email_returns_422(self, client, admin_client):
        uname = f"iu_{uuid.uuid4().hex[:6]}"
        r = client.post(
            "/api/v1/users",
            headers=admin_client["admin_headers"],
            json={
                "username": uname,
                "email": "not-an-email",
                "password": _VALID_PASSWORD,
                "role": "user",
            },
        )
        assert r.status_code == 422

    def test_missing_required_field_returns_422(self, client, admin_client):
        r = client.post(
            "/api/v1/users",
            headers=admin_client["admin_headers"],
            json={"username": "missingpw", "email": "m@ex.com", "role": "user"},
        )
        assert r.status_code == 422

    def test_created_user_has_all_fields(self, client, admin_client):
        uname = f"full_{uuid.uuid4().hex[:6]}"
        r = _create_user_as_admin(client, admin_client["admin_headers"], username=uname)
        assert r.status_code == 201
        data = r.json()["data"]
        assert "id" in data
        assert "username" in data
        assert "email" in data
        assert "role" in data
        assert "created_at" in data


# ---------------------------------------------------------------------------
# Update User
# ---------------------------------------------------------------------------


class TestUpdateUser:
    @pytest.fixture()
    def target_user_id(self, client, admin_client):
        uname = f"tgt_{uuid.uuid4().hex[:6]}"
        r = _create_user_as_admin(client, admin_client["admin_headers"], username=uname)
        assert r.status_code == 201
        return r.json()["data"]["id"]

    def test_admin_can_promote_user(self, client, admin_client, target_user_id):
        r = client.patch(
            f"/api/v1/users/{target_user_id}",
            headers=admin_client["admin_headers"],
            json={"role": "admin"},
        )
        assert r.status_code == 200
        assert r.json()["data"]["role"] == "admin"

    def test_admin_can_demote_other_user(self, client, admin_client, target_user_id):
        client.patch(
            f"/api/v1/users/{target_user_id}",
            headers=admin_client["admin_headers"],
            json={"role": "admin"},
        )
        r = client.patch(
            f"/api/v1/users/{target_user_id}",
            headers=admin_client["admin_headers"],
            json={"role": "user"},
        )
        assert r.status_code == 200
        assert r.json()["data"]["role"] == "user"

    def test_non_admin_returns_403(self, client, admin_client, target_user_id):
        r = client.patch(
            f"/api/v1/users/{target_user_id}",
            headers=admin_client["user_headers"],
            json={"role": "admin"},
        )
        assert r.status_code == 403

    def test_no_auth_returns_401(self, client, target_user_id):
        r = client.patch(
            f"/api/v1/users/{target_user_id}",
            json={"role": "admin"},
        )
        assert r.status_code == 401

    def test_not_found_returns_404(self, client, admin_client):
        r = client.patch(
            f"/api/v1/users/{uuid.uuid4()}",
            headers=admin_client["admin_headers"],
            json={"role": "user"},
        )
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "NOT_FOUND"

    def test_no_role_field_returns_user_unchanged(self, client, admin_client, target_user_id):
        r = client.patch(
            f"/api/v1/users/{target_user_id}",
            headers=admin_client["admin_headers"],
            json={},
        )
        assert r.status_code == 200
        assert "role" in r.json()["data"]

    def test_invalid_role_value_returns_422(self, client, admin_client, target_user_id):
        r = client.patch(
            f"/api/v1/users/{target_user_id}",
            headers=admin_client["admin_headers"],
            json={"role": "superadmin"},
        )
        assert r.status_code == 422

    def test_admin_cannot_demote_self(self, client, admin_client):
        admin_user_id = admin_client["admin_user_id"]
        r = client.patch(
            f"/api/v1/users/{admin_user_id}",
            headers=admin_client["admin_headers"],
            json={"role": "user"},
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "BAD_REQUEST"


# ---------------------------------------------------------------------------
# Delete User
# ---------------------------------------------------------------------------


class TestDeleteUser:
    @pytest.fixture()
    def deletable_user_id(self, client, admin_client):
        uname = f"del_{uuid.uuid4().hex[:6]}"
        r = _create_user_as_admin(client, admin_client["admin_headers"], username=uname)
        assert r.status_code == 201
        return r.json()["data"]["id"]

    def test_admin_can_delete_user(self, client, admin_client, deletable_user_id):
        r = client.delete(
            f"/api/v1/users/{deletable_user_id}",
            headers=admin_client["admin_headers"],
        )
        assert r.status_code == 200
        assert r.json()["data"] is None

    def test_deleted_user_not_in_list(self, client, admin_client, deletable_user_id):
        client.delete(
            f"/api/v1/users/{deletable_user_id}",
            headers=admin_client["admin_headers"],
        )
        r = client.get("/api/v1/users", headers=admin_client["admin_headers"])
        ids = [u["id"] for u in r.json()["data"]]
        assert deletable_user_id not in ids

    def test_non_admin_returns_403(self, client, admin_client, deletable_user_id):
        r = client.delete(
            f"/api/v1/users/{deletable_user_id}",
            headers=admin_client["user_headers"],
        )
        assert r.status_code == 403

    def test_no_auth_returns_401(self, client, deletable_user_id):
        r = client.delete(f"/api/v1/users/{deletable_user_id}")
        assert r.status_code == 401

    def test_nonexistent_user_returns_404(self, client, admin_client):
        r = client.delete(
            f"/api/v1/users/{uuid.uuid4()}",
            headers=admin_client["admin_headers"],
        )
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "NOT_FOUND"

    def test_admin_cannot_delete_self(self, client, admin_client):
        admin_user_id = admin_client["admin_user_id"]
        r = client.delete(
            f"/api/v1/users/{admin_user_id}",
            headers=admin_client["admin_headers"],
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "BAD_REQUEST"

    def test_double_delete_returns_404(self, client, admin_client, deletable_user_id):
        client.delete(
            f"/api/v1/users/{deletable_user_id}",
            headers=admin_client["admin_headers"],
        )
        r = client.delete(
            f"/api/v1/users/{deletable_user_id}",
            headers=admin_client["admin_headers"],
        )
        assert r.status_code == 404
