"""Regression tests for #2065 — self-service and admin password change.

Covers the two new endpoints, exercised as direct handler calls with mocked
repositories (mirrors the unit-test style of test_auth_refresh_rotation.py):

  - POST /api/v1/auth/change-password (change_password)
      * correct current password -> hash updated, all sessions revoked, commit
      * wrong current password   -> 401, no update, no revocation
      * missing user record      -> 401 (constant-time path)
  - POST /api/v1/users/{id}/reset-password (reset_user_password, admin)
      * existing same-org user   -> hash updated, target sessions revoked
      * unknown / cross-org user -> 404, no update
  - Schema complexity validation rejects weak new passwords.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

import cogtrix_core.api.routes.auth as auth_routes
import cogtrix_core.api.routes.users as users_routes
from cogtrix_core.api.auth import hash_password
from cogtrix_core.api.schemas.auth import ChangePasswordRequest
from cogtrix_core.api.schemas.user import PasswordResetRequest

_OLD = "OldPass1!"
_NEW = "NewPass2@"


def _user(user_id: str = "u1", org_id: str = "org1", password: str = _OLD):
    return SimpleNamespace(
        id=user_id,
        org_id=org_id,
        username="alice",
        email="alice@example.com",
        role="user",
        password_hash=hash_password(password),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _mock_db():
    db = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    return db


# ---------------------------------------------------------------------------
# Self-service change-password
# ---------------------------------------------------------------------------


class TestChangePassword:
    @pytest.mark.asyncio
    async def test_correct_current_password_updates_and_revokes(self):
        user = _user()
        user_repo = MagicMock()
        user_repo.get_by_id = AsyncMock(return_value=user)
        user_repo.update_password = AsyncMock(return_value=user)
        token_repo = MagicMock()
        token_repo.revoke_all_for_user = AsyncMock()
        db = _mock_db()

        with (
            patch.object(auth_routes, "UserRepository", return_value=user_repo),
            patch.object(auth_routes, "RefreshTokenRepository", return_value=token_repo),
        ):
            body = ChangePasswordRequest(current_password=_OLD, new_password=_NEW)
            resp = await auth_routes.change_password(
                body, current_user=SimpleNamespace(user_id="u1"), db=db
            )

        assert resp.data is None
        user_repo.update_password.assert_awaited_once()
        # The stored hash must be a fresh bcrypt hash of the new password, not the old.
        new_hash = user_repo.update_password.await_args.args[1]
        assert new_hash != user.password_hash
        token_repo.revoke_all_for_user.assert_awaited_once_with("u1")
        db.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_wrong_current_password_rejected(self):
        user = _user()
        user_repo = MagicMock()
        user_repo.get_by_id = AsyncMock(return_value=user)
        user_repo.update_password = AsyncMock()
        token_repo = MagicMock()
        token_repo.revoke_all_for_user = AsyncMock()
        db = _mock_db()

        with (
            patch.object(auth_routes, "UserRepository", return_value=user_repo),
            patch.object(auth_routes, "RefreshTokenRepository", return_value=token_repo),
        ):
            body = ChangePasswordRequest(current_password="WrongPass9!", new_password=_NEW)
            with pytest.raises(HTTPException) as exc:
                await auth_routes.change_password(
                    body, current_user=SimpleNamespace(user_id="u1"), db=db
                )

        assert exc.value.status_code == 401
        user_repo.update_password.assert_not_awaited()
        token_repo.revoke_all_for_user.assert_not_awaited()
        db.commit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_user_rejected(self):
        user_repo = MagicMock()
        user_repo.get_by_id = AsyncMock(return_value=None)
        user_repo.update_password = AsyncMock()
        db = _mock_db()

        with (
            patch.object(auth_routes, "UserRepository", return_value=user_repo),
            patch.object(auth_routes, "RefreshTokenRepository", return_value=MagicMock()),
        ):
            body = ChangePasswordRequest(current_password=_OLD, new_password=_NEW)
            with pytest.raises(HTTPException) as exc:
                await auth_routes.change_password(
                    body, current_user=SimpleNamespace(user_id="ghost"), db=db
                )

        assert exc.value.status_code == 401
        user_repo.update_password.assert_not_awaited()

    def test_weak_new_password_rejected_by_schema(self):
        with pytest.raises(ValidationError):
            ChangePasswordRequest(current_password=_OLD, new_password="weak")


# ---------------------------------------------------------------------------
# Admin reset-password
# ---------------------------------------------------------------------------


class TestAdminResetPassword:
    @pytest.mark.asyncio
    async def test_same_org_user_reset_and_revoked(self):
        user = _user(user_id="target", org_id="org1")
        repo = MagicMock()
        repo.get_by_id = AsyncMock(return_value=user)
        repo.update_password = AsyncMock(return_value=user)
        token_repo = MagicMock()
        token_repo.revoke_all_for_user = AsyncMock()
        db = _mock_db()
        ctx = SimpleNamespace(org_id="org1")

        with (
            patch.object(users_routes, "UserRepository", return_value=repo),
            patch.object(users_routes, "RefreshTokenRepository", return_value=token_repo),
        ):
            body = PasswordResetRequest(new_password=_NEW)
            resp = await users_routes.reset_user_password(
                "target",
                body,
                ctx=ctx,
                db=db,
                current_user=SimpleNamespace(user_id="admin"),
            )

        assert resp.data.id == "target"
        repo.update_password.assert_awaited_once()
        new_hash = repo.update_password.await_args.args[1]
        assert new_hash != user.password_hash
        token_repo.revoke_all_for_user.assert_awaited_once_with("target")
        db.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cross_org_user_not_found(self):
        user = _user(user_id="target", org_id="other-org")
        repo = MagicMock()
        repo.get_by_id = AsyncMock(return_value=user)
        repo.update_password = AsyncMock()
        token_repo = MagicMock()
        token_repo.revoke_all_for_user = AsyncMock()
        db = _mock_db()
        ctx = SimpleNamespace(org_id="org1")

        with (
            patch.object(users_routes, "UserRepository", return_value=repo),
            patch.object(users_routes, "RefreshTokenRepository", return_value=token_repo),
        ):
            body = PasswordResetRequest(new_password=_NEW)
            with pytest.raises(HTTPException) as exc:
                await users_routes.reset_user_password(
                    "target",
                    body,
                    ctx=ctx,
                    db=db,
                    current_user=SimpleNamespace(user_id="admin"),
                )

        assert exc.value.status_code == 404
        repo.update_password.assert_not_awaited()
        token_repo.revoke_all_for_user.assert_not_awaited()
