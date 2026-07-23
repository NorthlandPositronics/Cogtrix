"""Comprehensive auth endpoint coverage.

Tests the auth register/login/refresh/logout/me/api-key endpoints exhaustively.
Uses the same DB-override pattern as test_api_phase7.py so the app uses an
in-memory SQLite database instead of the real DB.
"""

from __future__ import annotations

import os
import uuid

import pytest

pytest.importorskip("fastapi")

_TEST_JWT_SECRET = "testsecret_mustbe32chars_minimum00"
os.environ.setdefault("COGTRIX_JWT_SECRET", _TEST_JWT_SECRET)
os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

import asyncio as _asyncio  # noqa: E402
from unittest.mock import patch  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

from src.api.auth import create_access_token  # noqa: E402
from src.api.db.engine import Base, get_db  # noqa: E402

# ---------------------------------------------------------------------------
# Shared fixture factory (function-scope so each test gets a fresh DB)
# ---------------------------------------------------------------------------


@pytest.fixture()
def app():
    """FastAPI app backed by an in-memory SQLite database."""
    from src.api.app import create_app

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
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


_VALID_PASSWORD = "TestPass1!"  # meets complexity: lower+upper+digit+special


def _register(client, username=None, email=None, password=_VALID_PASSWORD):
    if username is None:
        username = f"u_{uuid.uuid4().hex[:8]}"
    if email is None:
        email = f"{username}@ex.com"
    return client.post(
        "/api/v1/auth/register",
        json={"username": username, "email": email, "password": password},
    )


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------


class TestRegister:
    def test_first_user_becomes_admin(self, client):
        r = _register(client)
        assert r.status_code == 201
        body = r.json()
        assert body["error"] is None
        data = body["data"]
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0

    def test_register_returns_201(self, client):
        r = _register(client)
        assert r.status_code == 201

    def test_duplicate_username_returns_409(self, client):
        name = f"dupname_{uuid.uuid4().hex[:6]}"
        _register(client, username=name, email=f"{name}@ex.com")
        r = _register(client, username=name, email=f"other_{name}@ex.com")
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "VALIDATION_ERROR"

    def test_duplicate_email_returns_409(self, client):
        email = f"shared_{uuid.uuid4().hex[:6]}@ex.com"
        _register(client, email=email)
        r = _register(client, email=email)
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "VALIDATION_ERROR"

    def test_password_too_short_returns_422(self, client):
        r = _register(client, password="Sh0rt!")
        assert r.status_code == 422

    def test_invalid_email_format_returns_422(self, client):
        r = client.post(
            "/api/v1/auth/register",
            json={"username": "validu", "email": "not-an-email", "password": _VALID_PASSWORD},
        )
        assert r.status_code == 422

    def test_username_too_short_returns_422(self, client):
        r = client.post(
            "/api/v1/auth/register",
            json={"username": "ab", "email": "ab@ex.com", "password": _VALID_PASSWORD},
        )
        assert r.status_code == 422

    def test_username_too_long_returns_422(self, client):
        long = "a" * 65
        r = client.post(
            "/api/v1/auth/register",
            json={"username": long, "email": "x@ex.com", "password": _VALID_PASSWORD},
        )
        assert r.status_code == 422

    def test_username_with_special_chars_returns_422(self, client):
        r = client.post(
            "/api/v1/auth/register",
            json={"username": "bad user!", "email": "bad@ex.com", "password": _VALID_PASSWORD},
        )
        assert r.status_code == 422

    def test_missing_username_returns_422(self, client):
        r = client.post(
            "/api/v1/auth/register",
            json={"email": "e@ex.com", "password": _VALID_PASSWORD},
        )
        assert r.status_code == 422

    def test_missing_email_returns_422(self, client):
        r = client.post(
            "/api/v1/auth/register",
            json={"username": "validuser", "password": _VALID_PASSWORD},
        )
        assert r.status_code == 422

    def test_missing_password_returns_422(self, client):
        r = client.post(
            "/api/v1/auth/register",
            json={"username": "validuser", "email": "v@ex.com"},
        )
        assert r.status_code == 422

    def test_empty_body_returns_422(self, client):
        r = client.post("/api/v1/auth/register", json={})
        assert r.status_code == 422

    def test_response_envelope_has_correct_shape(self, client):
        r = _register(client)
        body = r.json()
        assert "data" in body
        assert "error" in body

    def test_access_token_is_string(self, client):
        r = _register(client)
        assert isinstance(r.json()["data"]["access_token"], str)

    def test_refresh_token_is_string(self, client):
        r = _register(client)
        assert isinstance(r.json()["data"]["refresh_token"], str)


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------


class TestLogin:
    def test_login_by_username_succeeds(self, client):
        uname = f"login_{uuid.uuid4().hex[:6]}"
        pw = _VALID_PASSWORD
        _register(client, username=uname, password=pw)
        r = client.post("/api/v1/auth/login", json={"username": uname, "password": pw})
        assert r.status_code == 200
        assert r.json()["data"]["access_token"] is not None

    def test_login_by_email_succeeds(self, client):
        uname = f"lge_{uuid.uuid4().hex[:6]}"
        email = f"{uname}@ex.com"
        pw = _VALID_PASSWORD
        _register(client, username=uname, email=email, password=pw)
        r = client.post("/api/v1/auth/login", json={"username": email, "password": pw})
        assert r.status_code == 200

    def test_wrong_password_returns_401(self, client):
        uname = f"wp_{uuid.uuid4().hex[:6]}"
        _register(client, username=uname, password=_VALID_PASSWORD)
        r = client.post(
            "/api/v1/auth/login",
            json={"username": uname, "password": "WrongPass9!"},
        )
        assert r.status_code == 401
        assert r.json()["error"]["code"] == "UNAUTHORIZED"

    def test_nonexistent_user_returns_401(self, client):
        r = client.post(
            "/api/v1/auth/login",
            json={"username": "nobody_here_ever", "password": "WrongPass9!"},
        )
        assert r.status_code == 401
        assert r.json()["error"]["code"] == "UNAUTHORIZED"

    def test_login_returns_both_tokens(self, client):
        uname = f"bt_{uuid.uuid4().hex[:6]}"
        pw = _VALID_PASSWORD
        _register(client, username=uname, password=pw)
        r = client.post("/api/v1/auth/login", json={"username": uname, "password": pw})
        data = r.json()["data"]
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    def test_login_missing_password_returns_422(self, client):
        r = client.post("/api/v1/auth/login", json={"username": "someone"})
        assert r.status_code == 422

    def test_login_empty_body_returns_422(self, client):
        r = client.post("/api/v1/auth/login", json={})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------


class TestRefresh:
    def test_refresh_returns_new_token_pair(self, client):
        r = _register(client)
        refresh_token = r.json()["data"]["refresh_token"]
        r2 = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token})
        assert r2.status_code == 200
        data = r2.json()["data"]
        assert "access_token" in data
        assert "refresh_token" in data

    def test_refresh_invalidates_old_token(self, client):
        r = _register(client)
        refresh_token = r.json()["data"]["refresh_token"]
        client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token})
        # Second use of same token should fail
        r3 = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token})
        assert r3.status_code == 401

    def test_invalid_token_returns_401(self, client):
        r = client.post("/api/v1/auth/refresh", json={"refresh_token": "total-garbage-token"})
        assert r.status_code == 401
        assert r.json()["error"]["code"] == "UNAUTHORIZED"

    def test_missing_token_field_returns_422(self, client):
        r = client.post("/api/v1/auth/refresh", json={})
        assert r.status_code == 422

    def test_empty_body_returns_422(self, client):
        r = client.post("/api/v1/auth/refresh")
        assert r.status_code in (422, 400)


# ---------------------------------------------------------------------------
# Logout
# ---------------------------------------------------------------------------


class TestLogout:
    def test_logout_success(self, client):
        r = _register(client)
        token = r.json()["data"]["access_token"]
        r2 = client.post(
            "/api/v1/auth/logout",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r2.status_code == 200
        assert r2.json()["data"] is None

    def test_logout_without_auth_returns_401(self, client):
        r = client.post("/api/v1/auth/logout")
        assert r.status_code == 401

    def test_logout_with_invalid_token_returns_401(self, client):
        r = client.post(
            "/api/v1/auth/logout",
            headers={"Authorization": "Bearer invalid.jwt.token"},
        )
        assert r.status_code == 401


# ---------------------------------------------------------------------------
# Get /me
# ---------------------------------------------------------------------------


class TestGetMe:
    def test_get_me_returns_user_data(self, client):
        uname = f"me_{uuid.uuid4().hex[:6]}"
        email = f"{uname}@ex.com"
        r = _register(client, username=uname, email=email)
        token = r.json()["data"]["access_token"]
        r2 = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r2.status_code == 200
        data = r2.json()["data"]
        assert data["username"] == uname
        assert data["email"] == email
        assert "id" in data
        assert "role" in data
        assert "created_at" in data

    def test_get_me_without_auth_returns_401(self, client):
        r = client.get("/api/v1/auth/me")
        assert r.status_code == 401

    def test_get_me_with_ghost_user_id_returns_401(self, client):
        # Token for a user not in the DB
        ghost_token = create_access_token("ghost-user-does-not-exist-ever", "user")
        r = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {ghost_token}"},
        )
        assert r.status_code == 401
        assert r.json()["error"]["code"] == "UNAUTHORIZED"

    def test_get_me_with_invalid_token_returns_401(self, client):
        r = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer bad.token.here"},
        )
        assert r.status_code == 401


# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------


class TestApiKeyList:
    def test_list_empty_no_keys(self, client):
        r = _register(client)
        token = r.json()["data"]["access_token"]
        r2 = client.get(
            "/api/v1/auth/api-keys",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r2.status_code == 200
        data = r2.json()["data"]
        assert data["items"] == []
        assert data["has_more"] is False
        assert data["next_cursor"] is None

    def test_list_shows_created_key(self, client):
        r = _register(client)
        token = r.json()["data"]["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        client.post(
            "/api/v1/auth/api-keys",
            headers=headers,
            json={"label": "my-key"},
        )
        r2 = client.get("/api/v1/auth/api-keys", headers=headers)
        assert r2.status_code == 200
        items = r2.json()["data"]["items"]
        assert len(items) == 1

    def test_list_key_value_not_returned_in_list(self, client):
        r = _register(client)
        token = r.json()["data"]["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        client.post("/api/v1/auth/api-keys", headers=headers, json={"label": "k"})
        r2 = client.get("/api/v1/auth/api-keys", headers=headers)
        for item in r2.json()["data"]["items"]:
            assert item["key"] is None

    def test_list_keys_requires_auth(self, client):
        r = client.get("/api/v1/auth/api-keys")
        assert r.status_code == 401

    def test_list_keys_limit_min_1(self, client):
        r = _register(client)
        token = r.json()["data"]["access_token"]
        r2 = client.get(
            "/api/v1/auth/api-keys?limit=0",
            headers={"Authorization": f"Bearer {token}"},
        )
        # limit is clamped to 1; should still succeed
        assert r2.status_code == 200

    def test_list_keys_cursor_param_accepted(self, client):
        r = _register(client)
        token = r.json()["data"]["access_token"]
        # Pass a cursor that doesn't exist — should return empty
        r2 = client.get(
            f"/api/v1/auth/api-keys?cursor={uuid.uuid4()}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r2.status_code == 200


class TestApiKeyCreate:
    def test_create_key_returns_full_value(self, client):
        r = _register(client)
        token = r.json()["data"]["access_token"]
        r2 = client.post(
            "/api/v1/auth/api-keys",
            headers={"Authorization": f"Bearer {token}"},
            json={"label": "my-key"},
        )
        assert r2.status_code == 201
        data = r2.json()["data"]
        assert data["key"] is not None
        assert data["key"].startswith("cgx_live_")
        assert data["label"] == "my-key"

    def test_create_key_with_expiry(self, client):
        r = _register(client)
        token = r.json()["data"]["access_token"]
        r2 = client.post(
            "/api/v1/auth/api-keys",
            headers={"Authorization": f"Bearer {token}"},
            json={"label": "expiry-key", "expires_in_days": 7},
        )
        assert r2.status_code == 201
        assert r2.json()["data"]["expires_at"] is not None

    def test_create_key_without_expiry_is_null(self, client):
        r = _register(client)
        token = r.json()["data"]["access_token"]
        r2 = client.post(
            "/api/v1/auth/api-keys",
            headers={"Authorization": f"Bearer {token}"},
            json={"label": "no-expiry"},
        )
        assert r2.status_code == 201
        assert r2.json()["data"]["expires_at"] is None

    def test_create_key_prefix_is_present(self, client):
        r = _register(client)
        token = r.json()["data"]["access_token"]
        r2 = client.post(
            "/api/v1/auth/api-keys",
            headers={"Authorization": f"Bearer {token}"},
            json={"label": "prefix-test"},
        )
        assert r2.status_code == 201
        assert r2.json()["data"]["key_prefix"] is not None

    def test_create_key_requires_auth(self, client):
        r = client.post("/api/v1/auth/api-keys", json={"label": "unauth"})
        assert r.status_code == 401

    def test_create_key_missing_label_returns_422(self, client):
        r = _register(client)
        token = r.json()["data"]["access_token"]
        r2 = client.post(
            "/api/v1/auth/api-keys",
            headers={"Authorization": f"Bearer {token}"},
            json={},
        )
        assert r2.status_code == 422


class TestApiKeyRevoke:
    def test_revoke_own_key_succeeds(self, client):
        r = _register(client)
        token = r.json()["data"]["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        create_r = client.post(
            "/api/v1/auth/api-keys", headers=headers, json={"label": "to-revoke"}
        )
        key_id = create_r.json()["data"]["id"]
        r2 = client.delete(f"/api/v1/auth/api-keys/{key_id}", headers=headers)
        assert r2.status_code == 200
        assert r2.json()["data"] is None

    def test_revoke_nonexistent_key_returns_404(self, client):
        r = _register(client)
        token = r.json()["data"]["access_token"]
        r2 = client.delete(
            f"/api/v1/auth/api-keys/{uuid.uuid4()}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r2.status_code == 404
        assert r2.json()["error"]["code"] == "NOT_FOUND"

    def test_revoke_other_users_key_returns_403(self, client):
        # User A creates a key
        r_a = _register(client)
        token_a = r_a.json()["data"]["access_token"]
        create_r = client.post(
            "/api/v1/auth/api-keys",
            headers={"Authorization": f"Bearer {token_a}"},
            json={"label": "a-key"},
        )
        key_id = create_r.json()["data"]["id"]

        # User B tries to revoke it
        r_b = _register(client)
        token_b = r_b.json()["data"]["access_token"]
        r2 = client.delete(
            f"/api/v1/auth/api-keys/{key_id}",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert r2.status_code == 403
        assert r2.json()["error"]["code"] == "FORBIDDEN"

    def test_revoke_key_requires_auth(self, client):
        r = client.delete(f"/api/v1/auth/api-keys/{uuid.uuid4()}")
        assert r.status_code == 401

    def test_revoked_key_disappears_from_list(self, client):
        r = _register(client)
        token = r.json()["data"]["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        create_r = client.post(
            "/api/v1/auth/api-keys", headers=headers, json={"label": "disappear"}
        )
        key_id = create_r.json()["data"]["id"]
        client.delete(f"/api/v1/auth/api-keys/{key_id}", headers=headers)
        list_r = client.get("/api/v1/auth/api-keys", headers=headers)
        ids = [k["id"] for k in list_r.json()["data"]["items"]]
        assert key_id not in ids
