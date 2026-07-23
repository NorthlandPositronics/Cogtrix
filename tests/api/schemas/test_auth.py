"""Tests for src/api/schemas/auth.py — registration, login, tokens, API keys."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.api.schemas.auth import (
    APIKeyCreateRequest,
    APIKeyOut,
    LoginRequest,
    LogoutAllRequest,
    LogoutRequest,
    RefreshRequest,
    RegisterRequest,
    TokenPair,
    UserOut,
    UserRole,
)

# ---------------------------------------------------------------------------
# UserRole — constant exposure
# ---------------------------------------------------------------------------


class TestUserRole:
    """The UserRole helper exposes role constants as string values."""

    def test_admin_constant(self) -> None:
        assert UserRole.ADMIN == "admin"

    def test_user_constant(self) -> None:
        assert UserRole.USER == "user"


# ---------------------------------------------------------------------------
# RegisterRequest — username, email, password complexity
# ---------------------------------------------------------------------------


class TestRegisterRequestValid:
    """Valid registration payloads construct without error."""

    def test_valid_minimal(self) -> None:
        req = RegisterRequest(username="alice", email="alice@example.com", password="P@ssw0rd")
        assert req.username == "alice"
        assert req.email == "alice@example.com"
        assert req.password == "P@ssw0rd"

    def test_username_with_hyphen_and_underscore(self) -> None:
        req = RegisterRequest(username="alice_bob-123", email="ab@example.com", password="P@ssw0rd")
        assert req.username == "alice_bob-123"

    def test_username_at_min_length_3(self) -> None:
        req = RegisterRequest(username="abc", email="abc@example.com", password="P@ssw0rd")
        assert req.username == "abc"

    def test_username_at_max_length_64(self) -> None:
        username = "a" * 64
        req = RegisterRequest(username=username, email="a@example.com", password="P@ssw0rd")
        assert len(req.username) == 64

    def test_password_at_min_length_8(self) -> None:
        # Exactly 8 chars with all 4 character classes.
        req = RegisterRequest(username="alice", email="a@example.com", password="A1b@cdef")
        assert req.password == "A1b@cdef"


class TestRegisterRequestInvalid:
    """Registration payloads with invalid fields raise ValidationError."""

    def test_username_too_short(self) -> None:
        with pytest.raises(ValidationError, match="at least 3"):
            RegisterRequest(username="ab", email="a@example.com", password="P@ssw0rd")

    def test_username_too_long(self) -> None:
        with pytest.raises(ValidationError, match="at most 64"):
            RegisterRequest(username="a" * 65, email="a@example.com", password="P@ssw0rd")

    def test_username_with_space_rejected(self) -> None:
        with pytest.raises(ValidationError, match="pattern"):
            RegisterRequest(username="alice bob", email="a@example.com", password="P@ssw0rd")

    def test_username_with_special_char_rejected(self) -> None:
        with pytest.raises(ValidationError, match="pattern"):
            RegisterRequest(username="alice!", email="a@example.com", password="P@ssw0rd")

    def test_username_empty_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RegisterRequest(username="", email="a@example.com", password="P@ssw0rd")

    def test_invalid_email_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RegisterRequest(username="alice", email="not-an-email", password="P@ssw0rd")

    def test_password_missing_uppercase_rejected(self) -> None:
        with pytest.raises(ValidationError, match="uppercase"):
            RegisterRequest(username="alice", email="a@example.com", password="p@ssw0rd")

    def test_password_missing_lowercase_rejected(self) -> None:
        with pytest.raises(ValidationError, match="lowercase"):
            RegisterRequest(username="alice", email="a@example.com", password="P@SSW0RD")

    def test_password_missing_digit_rejected(self) -> None:
        with pytest.raises(ValidationError, match="digit"):
            RegisterRequest(username="alice", email="a@example.com", password="P@ssword")

    def test_password_missing_special_rejected(self) -> None:
        with pytest.raises(ValidationError, match="special"):
            RegisterRequest(username="alice", email="a@example.com", password="Passw0rd")

    def test_password_too_short(self) -> None:
        with pytest.raises(ValidationError, match="at least 8"):
            RegisterRequest(username="alice", email="a@example.com", password="A1b@")

    def test_password_too_long(self) -> None:
        with pytest.raises(ValidationError, match="at most 128"):
            RegisterRequest(
                username="alice",
                email="a@example.com",
                password="A1b@" + ("x" * 125),
            )

    def test_missing_required_username(self) -> None:
        with pytest.raises(ValidationError):
            RegisterRequest(email="a@example.com", password="P@ssw0rd")  # type: ignore[call-arg]

    def test_missing_required_email(self) -> None:
        with pytest.raises(ValidationError):
            RegisterRequest(username="alice", password="P@ssw0rd")  # type: ignore[call-arg]

    def test_missing_required_password(self) -> None:
        with pytest.raises(ValidationError):
            RegisterRequest(username="alice", email="a@example.com")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# LoginRequest — no complexity guard (server-side credential check)
# ---------------------------------------------------------------------------


class TestLoginRequest:
    """Login schema accepts any string credentials — complexity is irrelevant
    server-side because the check is auth, not validation."""

    def test_valid_username_password(self) -> None:
        req = LoginRequest(username="alice", password="anything")
        assert req.username == "alice"
        assert req.password == "anything"

    def test_login_with_email_as_username(self) -> None:
        req = LoginRequest(username="alice@example.com", password="x")
        assert req.username == "alice@example.com"

    def test_missing_username_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LoginRequest(password="x")  # type: ignore[call-arg]

    def test_missing_password_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LoginRequest(username="alice")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# TokenPair — bearer-only literal, all four fields required
# ---------------------------------------------------------------------------


class TestTokenPair:
    def test_valid(self) -> None:
        tp = TokenPair(
            access_token="eyJ.abc.def",
            refresh_token="refresh-opaque-token",
            expires_in=3600,
        )
        assert tp.access_token == "eyJ.abc.def"
        assert tp.refresh_token == "refresh-opaque-token"
        # token_type defaults to literal "bearer".
        assert tp.token_type == "bearer"
        assert tp.expires_in == 3600

    def test_token_type_literal_only_accepts_bearer(self) -> None:
        # Non-"bearer" literal is rejected.
        with pytest.raises(ValidationError):
            TokenPair(
                access_token="a",
                refresh_token="r",
                token_type="basic",  # type: ignore[arg-type]
                expires_in=3600,
            )

    def test_missing_required_fields(self) -> None:
        # access_token / refresh_token / expires_in are all required.
        with pytest.raises(ValidationError):
            TokenPair(refresh_token="r", expires_in=3600)  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            TokenPair(access_token="a", expires_in=3600)  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            TokenPair(access_token="a", refresh_token="r")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# RefreshRequest / LogoutRequest / LogoutAllRequest
# ---------------------------------------------------------------------------


class TestRefreshAndLogout:
    def test_refresh_request_valid(self) -> None:
        assert RefreshRequest(refresh_token="r").refresh_token == "r"

    def test_refresh_request_missing_token(self) -> None:
        with pytest.raises(ValidationError):
            RefreshRequest()  # type: ignore[call-arg]

    def test_logout_request_valid(self) -> None:
        assert LogoutRequest(refresh_token="r").refresh_token == "r"

    def test_logout_request_missing_token(self) -> None:
        with pytest.raises(ValidationError):
            LogoutRequest()  # type: ignore[call-arg]

    def test_logout_all_request_valid(self) -> None:
        assert LogoutAllRequest(password="anything").password == "anything"

    def test_logout_all_request_missing_password(self) -> None:
        with pytest.raises(ValidationError):
            LogoutAllRequest()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# UserOut — public profile with ensure_utc on created_at
# ---------------------------------------------------------------------------


class TestUserOut:
    def test_valid_with_utc_datetime(self) -> None:
        u = UserOut(
            id="3f2504e0-4f89-11d3-9a0c-0305e82c3301",
            username="alice",
            email="alice@example.com",
            role="user",
            created_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
        )
        assert u.id == "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
        assert u.role == "user"
        assert u.created_at.tzinfo is UTC

    def test_naive_datetime_gets_utc_tzinfo(self) -> None:
        """ensure_utc validator attaches UTC tzinfo when missing (SQLite path)."""
        u = UserOut(
            id="x",
            username="alice",
            email="alice@example.com",
            role="user",
            created_at=datetime(2026, 1, 1, 12, 0, 0),  # naive
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

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            UserOut(  # type: ignore[call-arg]
                username="alice",
                email="alice@example.com",
                role="user",
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            )


# ---------------------------------------------------------------------------
# APIKeyCreateRequest — label cap + optional expiry range
# ---------------------------------------------------------------------------


class TestAPIKeyCreateRequest:
    def test_valid_minimal(self) -> None:
        r = APIKeyCreateRequest(label="CI pipeline key")
        assert r.label == "CI pipeline key"
        assert r.expires_in_days is None  # default

    def test_valid_with_expiry(self) -> None:
        r = APIKeyCreateRequest(label="x", expires_in_days=90)
        assert r.expires_in_days == 90

    def test_label_at_max_length_128(self) -> None:
        r = APIKeyCreateRequest(label="x" * 128)
        assert len(r.label) == 128

    def test_label_too_long_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at most 128"):
            APIKeyCreateRequest(label="x" * 129)

    def test_expiry_below_min_rejected(self) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            APIKeyCreateRequest(label="x", expires_in_days=0)

    def test_expiry_above_max_rejected(self) -> None:
        with pytest.raises(ValidationError, match="less than or equal to 3650"):
            APIKeyCreateRequest(label="x", expires_in_days=3651)

    def test_expiry_at_min_1_accepted(self) -> None:
        assert APIKeyCreateRequest(label="x", expires_in_days=1).expires_in_days == 1

    def test_expiry_at_max_3650_accepted(self) -> None:
        assert APIKeyCreateRequest(label="x", expires_in_days=3650).expires_in_days == 3650

    def test_missing_required_label(self) -> None:
        with pytest.raises(ValidationError):
            APIKeyCreateRequest()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# APIKeyOut — create-only key, masked prefix, ensure_utc on three datetimes
# ---------------------------------------------------------------------------


class TestAPIKeyOut:
    def test_create_response_includes_raw_key(self) -> None:
        out = APIKeyOut(
            id="abc",
            label="CI",
            key="cgx_live_abcdef1234567890",
            key_prefix="cgx_live_abc",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        assert out.key == "cgx_live_abcdef1234567890"
        assert out.key_prefix == "cgx_live_abc"

    def test_list_response_omits_raw_key(self) -> None:
        """Subsequent reads return key=None — pinned by the docstring contract
        that the raw key is returned ONLY on creation."""
        out = APIKeyOut(
            id="abc",
            label="CI",
            key_prefix="cgx_live_abc",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        assert out.key is None

    def test_naive_datetime_gets_utc_tzinfo(self) -> None:
        """ensure_utc applied to created_at, expires_at, last_used_at — the
        three datetime fields that come from SQLite without tzinfo."""
        out = APIKeyOut(
            id="abc",
            label="CI",
            key_prefix="cgx_live_abc",
            created_at=datetime(2026, 1, 1),  # naive
            expires_at=datetime(2026, 6, 1),  # naive
            last_used_at=datetime(2026, 3, 1),  # naive
        )
        assert out.created_at.tzinfo is UTC
        assert out.expires_at is not None and out.expires_at.tzinfo is UTC
        assert out.last_used_at is not None and out.last_used_at.tzinfo is UTC

    def test_optional_datetimes_default_none(self) -> None:
        out = APIKeyOut(
            id="abc",
            label="CI",
            key_prefix="cgx_live_abc",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        assert out.expires_at is None
        assert out.last_used_at is None

    def test_missing_required_fields(self) -> None:
        # id, label, key_prefix, created_at are required.
        with pytest.raises(ValidationError):
            APIKeyOut(  # type: ignore[call-arg]
                label="CI",
                key_prefix="cgx_live_abc",
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        with pytest.raises(ValidationError):
            APIKeyOut(  # type: ignore[call-arg]
                id="abc",
                key_prefix="cgx_live_abc",
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            )

    def test_explicit_offset_preserved(self) -> None:
        """If a non-UTC tzinfo is supplied, ensure_utc does not overwrite it
        — only naive datetimes get UTC attached."""
        explicit_tz = UTC
        out = APIKeyOut(
            id="abc",
            label="CI",
            key_prefix="cgx_live_abc",
            created_at=datetime(2026, 1, 1, tzinfo=explicit_tz),
        )
        # tzinfo is preserved (UTC stays UTC).
        assert out.created_at.utcoffset() == datetime(2026, 1, 1, tzinfo=UTC).utcoffset()
