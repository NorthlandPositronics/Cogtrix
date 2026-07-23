"""Comprehensive API authentication and authorization security tests.

Covers:
- TestTokenExpiry: expired, malformed, wrong-secret, and missing tokens on GET /auth/me
- TestRefreshTokenSecurity: valid/expired/invalid/revoked/missing refresh tokens; access token reuse
- TestApiKeyAuth: API key creation, direct validate_api_key coverage, revocation, ownership
- TestLogoutSecurity: logout requires auth; revokes tokens; revoked token rejected
- TestOwnershipEnforcement: non-owner gets 403 on all session-scoped endpoints; admin bypass
- TestAdminOnlyEndpoints: non-admin user gets 403 on admin-gated REST endpoints
"""

from __future__ import annotations

import asyncio
import os
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")

import jwt  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import (  # noqa: E402
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

# ---------------------------------------------------------------------------
# Environment — must be set before importing any src.api modules
# ---------------------------------------------------------------------------

_TEST_JWT_SECRET = "testsecret_mustbe32chars_minimum00"
_WRONG_JWT_SECRET = "wrongsecret_mustbe32chars_minimum0"
os.environ.setdefault("COGTRIX_JWT_SECRET", _TEST_JWT_SECRET)
os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

# ---------------------------------------------------------------------------
# Imports after env is set
# ---------------------------------------------------------------------------

from src.api.auth import validate_api_key  # noqa: E402
from src.api.db import models as _models  # noqa: E402, F401
from src.api.db.engine import Base  # noqa: E402
from src.api.db.models import ApiKey  # noqa: E402
from src.api.db.repositories.api_keys import ApiKeyRepository  # noqa: E402
from src.api.db.repositories.users import UserRepository  # noqa: E402

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_app():
    """FastAPI app backed by an isolated in-memory SQLite database."""
    test_engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    test_session_factory = async_sessionmaker(test_engine, expire_on_commit=False)

    async def _create():
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_create())

    with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
        from src.api.app import create_app
        from src.api.db.engine import get_db

        app = create_app()
        app.state.test_session_factory = test_session_factory

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

    loop.run_until_complete(test_engine.dispose())
    loop.close()


@pytest.fixture()
def client(test_app):
    """Synchronous TestClient backed by the test app."""
    with TestClient(test_app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# Helper: register a user and return (access_token, refresh_token)
# ---------------------------------------------------------------------------


def _register(client: TestClient, username: str, email: str, password: str = "Password1!"):
    resp = client.post(
        "/api/v1/auth/register",
        json={"username": username, "email": email, "password": password},
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()["data"]
    return data["access_token"], data["refresh_token"]


def _login(client: TestClient, username: str, password: str = "Password1!"):
    resp = client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": password},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    return data["access_token"], data["refresh_token"]


# ---------------------------------------------------------------------------
# Helper: mint an expired access JWT using the correct secret
# ---------------------------------------------------------------------------


def _expired_access_token(user_id: str, role: str = "user") -> str:
    with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
        now = datetime.now(UTC)
        claims = {
            "sub": user_id,
            "role": role,
            "iat": now - timedelta(hours=2),
            "exp": now - timedelta(hours=1),
        }
        return jwt.encode(claims, _TEST_JWT_SECRET, algorithm="HS256")


# ---------------------------------------------------------------------------
# 1. TestTokenExpiry
# ---------------------------------------------------------------------------


class TestTokenExpiry:
    """GET /api/v1/auth/me rejects invalid / expired tokens with 401."""

    def test_expired_access_token_returns_401(self, client: TestClient) -> None:
        token = _expired_access_token("some-user-id")
        resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] in ("TOKEN_EXPIRED", "UNAUTHORIZED")

    def test_malformed_token_returns_401(self, client: TestClient) -> None:
        resp = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer this.is.not.a.jwt"})
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "UNAUTHORIZED"

    def test_wrong_secret_token_returns_401(self, client: TestClient) -> None:
        claims = {
            "sub": str(uuid.uuid4()),
            "role": "user",
            "iat": datetime.now(UTC),
            "exp": datetime.now(UTC) + timedelta(hours=1),
        }
        bad_token = jwt.encode(claims, _WRONG_JWT_SECRET, algorithm="HS256")
        resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {bad_token}"})
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "UNAUTHORIZED"

    def test_no_authorization_header_returns_401(self, client: TestClient) -> None:
        resp = client.get("/api/v1/auth/me")
        assert resp.status_code == 401

    def test_empty_bearer_token_returns_401(self, client: TestClient) -> None:
        resp = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer "})
        assert resp.status_code == 401


class TestJwtSecretSnapshot:
    """JWT auth should use the startup snapshot, not the live environment."""

    def test_secret_remains_valid_after_env_is_cleared(self, client: TestClient) -> None:
        from src.api.auth import _decode_jwt, create_access_token

        user_id = str(uuid.uuid4())
        with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": ""}, clear=False):
            token = create_access_token(user_id=user_id, role="user")

        claims = _decode_jwt(token)
        assert claims["sub"] == user_id
        assert claims["role"] == "user"


# ---------------------------------------------------------------------------
# 2. TestRefreshTokenSecurity
# ---------------------------------------------------------------------------


class TestRefreshTokenSecurity:
    """POST /api/v1/auth/refresh validates refresh tokens strictly."""

    def test_valid_refresh_token_returns_new_pair(self, client: TestClient) -> None:
        _, refresh = _register(client, "rf_valid", "rf_valid@test.com")
        resp = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "access_token" in data
        assert "refresh_token" in data

    def test_expired_refresh_token_returns_401(self, client: TestClient) -> None:
        """A refresh token not found in the DB (simulating expired/deleted) returns 401."""
        _register(client, "rf_expired", "rf_expired@test.com")
        # A freshly generated random token does not exist in the DB → 401.
        raw_nonexistent = secrets.token_urlsafe(48)
        resp = client.post("/api/v1/auth/refresh", json={"refresh_token": raw_nonexistent})
        assert resp.status_code == 401

    def test_invalid_random_refresh_token_returns_401(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": secrets.token_urlsafe(48)},
        )
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "UNAUTHORIZED"

    def test_revoked_refresh_token_returns_401(self, client: TestClient) -> None:
        _, refresh = _register(client, "rf_revoked", "rf_revoked@test.com")
        # First use rotates (revokes) the old token.
        client.post("/api/v1/auth/refresh", json={"refresh_token": refresh})
        # Second use of the now-revoked token must fail.
        resp = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh})
        assert resp.status_code == 401

    def test_missing_body_returns_422(self, client: TestClient) -> None:
        resp = client.post("/api/v1/auth/refresh", json={})
        assert resp.status_code == 422

    def test_access_token_used_as_refresh_token_returns_401(self, client: TestClient) -> None:
        access, _ = _register(client, "rf_access_as_refresh", "rf_aar@test.com")
        # The access token is a JWT, not a hashed DB token — must fail.
        resp = client.post("/api/v1/auth/refresh", json={"refresh_token": access})
        assert resp.status_code == 401


class TestAuthRateLimits:
    """Auth-sensitive endpoints should be rate-limited more aggressively."""

    def test_login_is_rate_limited_after_five_attempts(self, client: TestClient) -> None:
        payload = {"username": "missing-user", "password": "Password1!"}
        for _ in range(5):
            resp = client.post("/api/v1/auth/login", json=payload)
            assert resp.status_code == 401, resp.text

        resp = client.post("/api/v1/auth/login", json=payload)
        assert resp.status_code == 429, resp.text

    def test_refresh_is_rate_limited_after_five_requests(self, client: TestClient) -> None:
        _, refresh = _register(client, "rl_refresh", "rl_refresh@test.com")

        for _ in range(5):
            resp = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh})
            assert resp.status_code == 200, resp.text
            refresh = resp.json()["data"]["refresh_token"]

        resp = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh})
        assert resp.status_code == 429, resp.text

    def test_register_is_rate_limited_after_three_attempts(self, client: TestClient) -> None:
        for i in range(3):
            resp = client.post(
                "/api/v1/auth/register",
                json={
                    "username": f"rl_reg{i}",
                    "email": f"rl_reg{i}@test.com",
                    "password": "Password1!",
                },
            )
            assert resp.status_code == 201, resp.text

        resp = client.post(
            "/api/v1/auth/register",
            json={
                "username": "rl_reg3",
                "email": "rl_reg3@test.com",
                "password": "Password1!",
            },
        )
        assert resp.status_code == 429, resp.text


# ---------------------------------------------------------------------------
# 3. TestApiKeyAuth
# ---------------------------------------------------------------------------


class TestApiKeyAuth:
    """API key creation, validate_api_key coverage, revocation, and ownership."""

    def test_create_api_key_returns_full_key_once(self, client: TestClient) -> None:
        access, _ = _register(client, "ak_creator", "ak_creator@test.com")
        resp = client.post(
            "/api/v1/auth/api-keys",
            json={"label": "ci-key"},
            headers={"Authorization": f"Bearer {access}"},
        )
        assert resp.status_code == 201
        data = resp.json()["data"]
        assert data["key"].startswith("cgx_live_")
        assert data["key_prefix"] == data["key"][:12]

    def test_api_key_used_as_bearer_returns_profile(self, client: TestClient) -> None:
        """A live API key should authenticate through get_current_user."""
        access, _ = _register(client, "ak_bearer", "ak_bearer@test.com")
        create_resp = client.post(
            "/api/v1/auth/api-keys",
            json={"label": "bad-bearer"},
            headers={"Authorization": f"Bearer {access}"},
        )
        raw_key = create_resp.json()["data"]["key"]
        resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {raw_key}"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["username"] == "ak_bearer"
        assert data["email"] == "ak_bearer@test.com"
        assert data["role"] == "admin"

    def test_invalid_prefix_api_key_as_bearer_returns_401(self, client: TestClient) -> None:
        bad_key = "invalid_prefix_" + secrets.token_urlsafe(32)
        resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {bad_key}"})
        assert resp.status_code == 401

    def test_nonexistent_api_key_validate_raises_401(self, test_app) -> None:
        """validate_api_key raises HTTPException 401 for a key not in the DB."""
        import asyncio as _asyncio

        from fastapi import HTTPException

        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            echo=False,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )

        async def _run():
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            session_factory = async_sessionmaker(engine, expire_on_commit=False)
            async with session_factory() as db:
                with pytest.raises(HTTPException) as exc_info:
                    await validate_api_key("cgx_live_" + secrets.token_urlsafe(32), db)
                assert exc_info.value.status_code == 401
            await engine.dispose()

        _asyncio.run(_run())

    def test_api_key_updates_last_used(self, test_app) -> None:
        """validate_api_key updates last_used_at on the key record."""
        import asyncio as _asyncio

        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            echo=False,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )

        async def _run():
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            session_factory = async_sessionmaker(engine, expire_on_commit=False)
            async with session_factory() as db:
                # Create a user.
                user_repo = UserRepository(db)
                user = await user_repo.create(
                    user_id=str(uuid.uuid4()),
                    username="ak_lastused",
                    email="ak_lastused@test.com",
                    password_hash="x",
                    role="user",
                )
                await db.commit()

                # Create an API key.
                raw_key = "cgx_live_" + secrets.token_urlsafe(32)
                from src.api.auth import _hash_api_key

                key_hash = _hash_api_key(raw_key)
                key_repo = ApiKeyRepository(db)
                key_record = await key_repo.create(
                    key_id=str(uuid.uuid4()),
                    user_id=user.id,
                    key_hash=key_hash,
                    key_prefix=raw_key[:12],
                    label="test",
                )
                await db.commit()

                assert key_record.last_used_at is None

                # Call validate_api_key — should update last_used_at.
                await validate_api_key(raw_key, db)
                await db.commit()

                updated = await key_repo.get_by_id(key_record.id)
                assert updated is not None
                assert updated.last_used_at is not None

            await engine.dispose()

        _asyncio.run(_run())

    def test_api_key_debounce_skips_immediate_revalidation(self, test_app) -> None:
        """validate_api_key debounces last_used_at writes within 60 s."""
        import asyncio as _asyncio

        from src.api.auth import _API_KEY_LAST_USED

        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            echo=False,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )

        async def _run():
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            session_factory = async_sessionmaker(engine, expire_on_commit=False)
            async with session_factory() as db:
                user_repo = UserRepository(db)
                user = await user_repo.create(
                    user_id=str(uuid.uuid4()),
                    username="ak_debounce",
                    email="ak_debounce@test.com",
                    password_hash="x",
                    role="user",
                )
                await db.commit()

                raw_key = "cgx_live_" + secrets.token_urlsafe(32)
                from src.api.auth import _hash_api_key

                key_hash = _hash_api_key(raw_key)
                key_repo = ApiKeyRepository(db)
                key_record = await key_repo.create(
                    key_id=str(uuid.uuid4()),
                    user_id=user.id,
                    key_hash=key_hash,
                    key_prefix=raw_key[:12],
                    label="test",
                )
                await db.commit()

                # First call updates last_used_at.
                await validate_api_key(raw_key, db)
                await db.commit()
                first = await key_repo.get_by_id(key_record.id)
                assert first is not None
                assert first.last_used_at is not None
                first_ts = first.last_used_at

                # Reset last_used_at in DB to detect whether second call writes.
                from sqlalchemy import update

                await db.execute(
                    update(ApiKey).where(ApiKey.id == key_record.id).values(last_used_at=None)
                )
                await db.commit()

                # Second call within debounce window should NOT write.
                await validate_api_key(raw_key, db)
                await db.commit()
                second = await key_repo.get_by_id(key_record.id)
                assert second is not None
                assert second.last_used_at is None

                # Clear debounce cache and call again — should write.
                _API_KEY_LAST_USED.pop(key_record.id, None)
                await validate_api_key(raw_key, db)
                await db.commit()
                third = await key_repo.get_by_id(key_record.id)
                assert third is not None
                assert third.last_used_at is not None
                assert third.last_used_at > first_ts

            await engine.dispose()

        _asyncio.run(_run())

    def test_validate_api_key_rejects_inactive_user(self, client: TestClient) -> None:
        import asyncio as _asyncio

        from fastapi import HTTPException

        async def _run():
            factory = client.app.state.test_session_factory
            async with factory() as db:
                user_repo = UserRepository(db)
                user = await user_repo.create(
                    user_id=str(uuid.uuid4()),
                    username="ak_inactive",
                    email="ak_inactive@test.com",
                    password_hash="x",
                    role="user",
                )
                await db.commit()

                raw_key = "cgx_live_" + secrets.token_urlsafe(32)
                from src.api.auth import _hash_api_key

                key_hash = _hash_api_key(raw_key)
                key_repo = ApiKeyRepository(db)
                await key_repo.create(
                    key_id=str(uuid.uuid4()),
                    user_id=user.id,
                    key_hash=key_hash,
                    key_prefix=raw_key[:12],
                    label="inactive",
                )
                await db.commit()

                await user_repo.set_active(user.id, False)
                await db.commit()

                with pytest.raises(HTTPException) as exc_info:
                    await validate_api_key(raw_key, db)
                assert exc_info.value.status_code == 401

        _asyncio.run(_run())

    def test_inactive_user_jwt_is_rejected(self, client: TestClient) -> None:
        import asyncio as _asyncio

        async def _run() -> tuple[str, str]:
            factory = client.app.state.test_session_factory
            async with factory() as db:
                user_repo = UserRepository(db)
                user = await user_repo.create(
                    user_id=str(uuid.uuid4()),
                    username="jwt_inactive",
                    email="jwt_inactive@test.com",
                    password_hash="x",
                    role="user",
                )
                await db.commit()
                await user_repo.set_active(user.id, False)
                await db.commit()
                return user.id, user.role

        user_id, role = _asyncio.run(_run())

        from src.api.auth import create_access_token

        with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
            token = create_access_token(user_id, role)

        resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401

    def test_admin_can_revoke_any_users_key(self, client: TestClient) -> None:
        # Admin is the first registered user.
        admin_access, _ = _register(client, "ak_admin", "ak_admin@test.com")
        # Regular user creates a key.
        user_access, _ = _register(client, "ak_user_for_admin", "ak_u4a@test.com")
        create_resp = client.post(
            "/api/v1/auth/api-keys",
            json={"label": "user-key"},
            headers={"Authorization": f"Bearer {user_access}"},
        )
        key_id = create_resp.json()["data"]["id"]

        # Admin revokes it.
        resp = client.delete(
            f"/api/v1/auth/api-keys/{key_id}",
            headers={"Authorization": f"Bearer {admin_access}"},
        )
        assert resp.status_code == 200

    def test_non_admin_cannot_revoke_another_users_key(self, client: TestClient) -> None:
        # Admin registers first; then two regular users.
        _register(client, "ak_admin2", "ak_admin2@test.com")
        user_a_access, _ = _register(client, "ak_ua", "ak_ua@test.com")
        user_b_access, _ = _register(client, "ak_ub", "ak_ub@test.com")

        # User A creates a key.
        create_resp = client.post(
            "/api/v1/auth/api-keys",
            json={"label": "a-key"},
            headers={"Authorization": f"Bearer {user_a_access}"},
        )
        key_id = create_resp.json()["data"]["id"]

        # User B (non-admin) tries to revoke User A's key.
        resp = client.delete(
            f"/api/v1/auth/api-keys/{key_id}",
            headers={"Authorization": f"Bearer {user_b_access}"},
        )
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "FORBIDDEN"


# ---------------------------------------------------------------------------
# 4. TestLogoutSecurity
# ---------------------------------------------------------------------------


class TestLogoutSecurity:
    """POST /api/v1/auth/logout security boundaries."""

    def test_logout_without_auth_returns_401(self, client: TestClient) -> None:
        resp = client.post("/api/v1/auth/logout")
        assert resp.status_code == 401

    def test_logout_revokes_only_specified_refresh_token(self, client: TestClient) -> None:
        access1, refresh1 = _register(client, "lo_revoker", "lo_revoker@test.com")
        access2, refresh2 = _login(client, "lo_revoker")

        resp = client.post(
            "/api/v1/auth/logout",
            json={"refresh_token": refresh1},
            headers={"Authorization": f"Bearer {access1}"},
        )
        assert resp.status_code == 200
        revoked_resp = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh1})
        assert revoked_resp.status_code == 401

        still_valid_resp = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh2})
        assert still_valid_resp.status_code == 200

    def test_revoked_refresh_token_rejected_after_logout(self, client: TestClient) -> None:
        """Verify the revoked token returns UNAUTHORIZED (not 200 or 422)."""
        access, refresh = _register(client, "lo_postlogout", "lo_postlogout@test.com")
        client.post(
            "/api/v1/auth/logout",
            json={"refresh_token": refresh},
            headers={"Authorization": f"Bearer {access}"},
        )
        resp = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh})
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] in ("UNAUTHORIZED", "TOKEN_EXPIRED")

    def test_logout_all_requires_password_and_revokes_all_tokens(self, client: TestClient) -> None:
        access1, refresh1 = _register(client, "lo_logout_all", "lo_logout_all@test.com")
        access2, refresh2 = _login(client, "lo_logout_all")

        wrong = client.post(
            "/api/v1/auth/logout-all",
            json={"password": "wrong-pass"},
            headers={"Authorization": f"Bearer {access1}"},
        )
        assert wrong.status_code == 401
        assert wrong.json()["error"]["code"] == "UNAUTHORIZED"

        resp = client.post(
            "/api/v1/auth/logout-all",
            json={"password": "Password1!"},
            headers={"Authorization": f"Bearer {access2}"},
        )
        assert resp.status_code == 200

        resp1 = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh1})
        resp2 = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh2})
        assert resp1.status_code == 401
        assert resp2.status_code == 401


# ---------------------------------------------------------------------------
# 5. TestOwnershipEnforcement
# ---------------------------------------------------------------------------


class TestOwnershipEnforcement:
    """verify_session_owner blocks non-owners; admin bypass works."""

    def _create_session(self, client: TestClient, access_token: str) -> str:
        """Create a session and return its ID."""
        resp = client.post(
            "/api/v1/sessions",
            json={"name": "test session"},
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert resp.status_code == 201, resp.text
        return resp.json()["data"]["id"]

    def test_get_session_as_non_owner_returns_403(self, client: TestClient) -> None:
        admin_access, _ = _register(client, "oe_admin", "oe_admin@test.com")
        user_b_access, _ = _register(client, "oe_userb1", "oe_userb1@test.com")
        session_id = self._create_session(client, admin_access)

        resp = client.get(
            f"/api/v1/sessions/{session_id}",
            headers={"Authorization": f"Bearer {user_b_access}"},
        )
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "FORBIDDEN"

    def test_patch_session_as_non_owner_returns_403(self, client: TestClient) -> None:
        admin_access, _ = _register(client, "oe_admin2", "oe_admin2@test.com")
        user_b_access, _ = _register(client, "oe_userb2", "oe_userb2@test.com")
        session_id = self._create_session(client, admin_access)

        resp = client.patch(
            f"/api/v1/sessions/{session_id}",
            json={"name": "hijacked"},
            headers={"Authorization": f"Bearer {user_b_access}"},
        )
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "FORBIDDEN"

    def test_delete_session_as_non_owner_returns_403(self, client: TestClient) -> None:
        admin_access, _ = _register(client, "oe_admin3", "oe_admin3@test.com")
        user_b_access, _ = _register(client, "oe_userb3", "oe_userb3@test.com")
        session_id = self._create_session(client, admin_access)

        resp = client.delete(
            f"/api/v1/sessions/{session_id}",
            headers={"Authorization": f"Bearer {user_b_access}"},
        )
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "FORBIDDEN"

    def test_get_session_messages_as_non_owner_returns_403(self, client: TestClient) -> None:
        admin_access, _ = _register(client, "oe_admin4", "oe_admin4@test.com")
        user_b_access, _ = _register(client, "oe_userb4", "oe_userb4@test.com")
        session_id = self._create_session(client, admin_access)

        resp = client.get(
            f"/api/v1/sessions/{session_id}/messages",
            headers={"Authorization": f"Bearer {user_b_access}"},
        )
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "FORBIDDEN"

    def test_post_session_message_as_non_owner_returns_403_or_401(self, client: TestClient) -> None:
        admin_access, _ = _register(client, "oe_admin5", "oe_admin5@test.com")
        user_b_access, _ = _register(client, "oe_userb5", "oe_userb5@test.com")
        session_id = self._create_session(client, admin_access)

        resp = client.post(
            f"/api/v1/sessions/{session_id}/messages",
            json={"content": "hello"},
            headers={"Authorization": f"Bearer {user_b_access}"},
        )
        assert resp.status_code in (403, 401)

    def test_get_session_memory_as_non_owner_returns_403(self, client: TestClient) -> None:
        admin_access, _ = _register(client, "oe_admin6", "oe_admin6@test.com")
        user_b_access, _ = _register(client, "oe_userb6", "oe_userb6@test.com")
        session_id = self._create_session(client, admin_access)

        resp = client.get(
            f"/api/v1/sessions/{session_id}/memory",
            headers={"Authorization": f"Bearer {user_b_access}"},
        )
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "FORBIDDEN"

    def test_admin_can_access_any_session(self, client: TestClient) -> None:
        """Admin (first user) bypasses ownership check and sees other users' sessions."""
        admin_access, _ = _register(client, "oe_admin7", "oe_admin7@test.com")
        user_access, _ = _register(client, "oe_owner7", "oe_owner7@test.com")
        session_id = self._create_session(client, user_access)

        resp = client.get(
            f"/api/v1/sessions/{session_id}",
            headers={"Authorization": f"Bearer {admin_access}"},
        )
        # Admin should get 200 with the session data
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["id"] == session_id


# ---------------------------------------------------------------------------
# 6. TestAdminOnlyEndpoints
# ---------------------------------------------------------------------------


class TestAdminOnlyEndpoints:
    """Non-admin users receive 403 on endpoints protected by require_admin."""

    def _admin_and_user_tokens(self, client: TestClient, suffix: str):
        """Register an admin (first) and a regular user; return both access tokens."""
        admin_access, _ = _register(client, f"aoe_admin_{suffix}", f"aoe_admin_{suffix}@test.com")
        user_access, _ = _register(client, f"aoe_user_{suffix}", f"aoe_user_{suffix}@test.com")
        return admin_access, user_access

    def test_patch_config_non_admin_returns_403(self, client: TestClient) -> None:
        _, user_access = self._admin_and_user_tokens(client, "pc")
        resp = client.patch(
            "/api/v1/config",
            json={"debug": False},
            headers={"Authorization": f"Bearer {user_access}"},
        )
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "FORBIDDEN"

    def test_patch_config_admin_is_not_403(self, client: TestClient) -> None:
        admin_access, _ = self._admin_and_user_tokens(client, "pca")
        resp = client.patch(
            "/api/v1/config",
            json={"debug": False},
            headers={"Authorization": f"Bearer {admin_access}"},
        )
        # Admin should get 200 with the updated config snapshot
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "debug" in data

    def test_reload_config_non_admin_returns_403(self, client: TestClient) -> None:
        _, user_access = self._admin_and_user_tokens(client, "rc")
        resp = client.post(
            "/api/v1/config/reload",
            headers={"Authorization": f"Bearer {user_access}"},
        )
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "FORBIDDEN"

    def test_reload_config_admin_is_not_403(self, client: TestClient) -> None:
        admin_access, _ = self._admin_and_user_tokens(client, "rca")
        resp = client.post(
            "/api/v1/config/reload",
            headers={"Authorization": f"Bearer {admin_access}"},
        )
        # Admin should get 200 with reload confirmation
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["reloaded"] is True

    def test_system_debug_non_admin_returns_403(self, client: TestClient) -> None:
        _, user_access = self._admin_and_user_tokens(client, "sd")
        resp = client.post(
            "/api/v1/system/debug",
            json={"debug": False},
            headers={"Authorization": f"Bearer {user_access}"},
        )
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "FORBIDDEN"

    def test_rag_upload_non_admin_returns_403(self, client: TestClient) -> None:
        _, user_access = self._admin_and_user_tokens(client, "ru")
        resp = client.post(
            "/api/v1/rag/documents",
            files={"file": ("test.txt", b"hello world", "text/plain")},
            headers={"Authorization": f"Bearer {user_access}"},
        )
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "FORBIDDEN"

    def test_rag_delete_non_admin_returns_403(self, client: TestClient) -> None:
        _, user_access = self._admin_and_user_tokens(client, "rd")
        fake_doc_id = str(uuid.uuid4())
        resp = client.delete(
            f"/api/v1/rag/documents/{fake_doc_id}",
            headers={"Authorization": f"Bearer {user_access}"},
        )
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "FORBIDDEN"
