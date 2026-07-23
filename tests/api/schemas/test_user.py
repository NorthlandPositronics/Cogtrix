"""Tests for cogtrix_core/api/schemas/user.py — admin user management schemas.

Note: this module's UserOut/Create/Update schemas are the org-admin user-
management surface (POST/PATCH /api/v1/users/{id}); the auth.py UserOut
schema is the self-profile surface (GET /api/v1/auth/me).  Both exist
deliberately — different fields and different validators.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from cogtrix_core.api.schemas.user import UserCreateRequest, UserOut, UserUpdateRequest

# ---------------------------------------------------------------------------
# UserOut — public profile, ensure_utc on created_at
# ---------------------------------------------------------------------------


class TestUserOut:
    def test_valid_with_utc(self) -> None:
        u = UserOut(
            id="3f2504e0-4f89-11d3-9a0c-0305e82c3301",
            username="alice",
            email="alice@example.com",
            role="user",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        assert u.role == "user"

    def test_naive_datetime_gets_utc(self) -> None:
        u = UserOut(
            id="x",
            username="alice",
            email="alice@example.com",
            role="user",
            created_at=datetime(2026, 1, 1),  # naive
        )
        assert u.created_at.tzinfo is UTC

    def test_admin_role_accepted(self) -> None:
        u = UserOut(
            id="x",
            username="root",
            email="root@example.com",
            role="admin",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        assert u.role == "admin"

    def test_missing_required_field(self) -> None:
        with pytest.raises(ValidationError):
            UserOut(  # type: ignore[call-arg]
                username="alice",
                email="alice@example.com",
                role="user",
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            )


# ---------------------------------------------------------------------------
# UserCreateRequest — username pattern, password complexity, role enum
# ---------------------------------------------------------------------------


class TestUserCreateRequestValid:
    def test_valid_minimal(self) -> None:
        r = UserCreateRequest(username="alice", email="alice@example.com", password="P@ssw0rd")
        assert r.role == "user"  # default

    def test_valid_admin(self) -> None:
        r = UserCreateRequest(
            username="alice",
            email="alice@example.com",
            password="P@ssw0rd",
            role="admin",
        )
        assert r.role == "admin"

    def test_username_at_min_length_3(self) -> None:
        r = UserCreateRequest(username="abc", email="abc@example.com", password="P@ssw0rd")
        assert r.username == "abc"

    def test_username_at_max_length_64(self) -> None:
        r = UserCreateRequest(username="a" * 64, email="a@example.com", password="P@ssw0rd")
        assert len(r.username) == 64

    def test_password_with_all_four_classes(self) -> None:
        # Reuses validate_password_complexity; pattern coverage is in
        # test_validators.py — here just ensure delegation is wired.
        r = UserCreateRequest(username="alice", email="a@example.com", password="Aa1!cdef")
        assert r.password == "Aa1!cdef"


class TestUserCreateRequestInvalid:
    def test_username_too_short(self) -> None:
        with pytest.raises(ValidationError, match="at least 3"):
            UserCreateRequest(username="ab", email="a@example.com", password="P@ssw0rd")

    def test_username_too_long(self) -> None:
        with pytest.raises(ValidationError, match="at most 64"):
            UserCreateRequest(username="a" * 65, email="a@example.com", password="P@ssw0rd")

    def test_username_with_special_char_rejected(self) -> None:
        with pytest.raises(ValidationError, match="pattern"):
            UserCreateRequest(username="alice!", email="a@example.com", password="P@ssw0rd")

    def test_invalid_email(self) -> None:
        with pytest.raises(ValidationError):
            UserCreateRequest(username="alice", email="not-email", password="P@ssw0rd")

    def test_password_missing_uppercase(self) -> None:
        with pytest.raises(ValidationError, match="uppercase"):
            UserCreateRequest(username="alice", email="a@example.com", password="p@ssw0rd")

    def test_password_too_short(self) -> None:
        with pytest.raises(ValidationError, match="at least 8"):
            UserCreateRequest(username="alice", email="a@example.com", password="A1b@")

    def test_password_too_long(self) -> None:
        with pytest.raises(ValidationError, match="at most 128"):
            UserCreateRequest(
                username="alice",
                email="a@example.com",
                password="A1b@" + ("x" * 125),
            )

    def test_role_invalid_value_rejected(self) -> None:
        with pytest.raises(ValidationError, match="pattern"):
            UserCreateRequest(
                username="alice",
                email="a@example.com",
                password="P@ssw0rd",
                role="superadmin",
            )

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            UserCreateRequest(email="a@example.com", password="P@ssw0rd")  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            UserCreateRequest(username="alice", password="P@ssw0rd")  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            UserCreateRequest(username="alice", email="a@example.com")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# UserUpdateRequest — only role today, must match (user|admin)
# ---------------------------------------------------------------------------


class TestUserUpdateRequest:
    def test_empty(self) -> None:
        u = UserUpdateRequest()
        assert u.role is None

    def test_promote_to_admin(self) -> None:
        u = UserUpdateRequest(role="admin")
        assert u.role == "admin"

    def test_demote_to_user(self) -> None:
        u = UserUpdateRequest(role="user")
        assert u.role == "user"

    def test_invalid_role_rejected(self) -> None:
        with pytest.raises(ValidationError, match="pattern"):
            UserUpdateRequest(role="owner")

    def test_role_none_passes(self) -> None:
        """None bypasses the pattern check — required for partial updates."""
        assert UserUpdateRequest(role=None).role is None
