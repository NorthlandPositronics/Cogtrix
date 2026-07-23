"""Authentication schemas — registration, login, token refresh, API keys."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr, Field, field_validator

from src.api.schemas.common import ensure_utc
from src.api.schemas.validators import validate_password_complexity

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class UserRole(str):
    """Role constants for role-based access control."""

    ADMIN = "admin"
    USER = "user"


# ---------------------------------------------------------------------------
# Registration & Login
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    """Request body for POST /api/v1/auth/register."""

    username: str = Field(
        ...,
        min_length=3,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Unique username (alphanumeric, underscores, hyphens; 3–64 chars).",
        examples=["alice"],
    )
    email: EmailStr = Field(
        ...,
        description="Email address used for account recovery.",
        examples=["alice@example.com"],
    )
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Password (8–128 chars; must include lowercase, uppercase, digit, and special character).",
        examples=["s3cur3P@ss"],
    )

    _password_complexity = field_validator("password")(validate_password_complexity)


class LoginRequest(BaseModel):
    """Request body for POST /api/v1/auth/login."""

    username: str = Field(
        ...,
        description="Username or email address.",
        examples=["alice"],
    )
    password: str = Field(
        ...,
        description="Account password.",
        examples=["s3cur3P@ss"],
    )


class TokenPair(BaseModel):
    """Access + refresh token pair returned on login or refresh.

    ``access_token`` is a short-lived JWT (default 1 hour).
    ``refresh_token`` is a long-lived opaque token (default 30 days).
    The frontend should store both and use ``refresh_token`` to silently
    renew the access token when it receives a TOKEN_EXPIRED error.
    """

    access_token: str = Field(
        ...,
        description="Short-lived JWT for API authentication (attach as Bearer token).",
    )
    refresh_token: str = Field(
        ...,
        description="Long-lived opaque token for silent access token renewal.",
    )
    token_type: Literal["bearer"] = Field(
        default="bearer",
        description="Always 'bearer'.",
    )
    expires_in: int = Field(
        ...,
        description="Seconds until the access token expires.",
        examples=[3600],
    )


class RefreshRequest(BaseModel):
    """Request body for POST /api/v1/auth/refresh."""

    refresh_token: str = Field(
        ...,
        description="Refresh token obtained from a previous login or refresh.",
    )


class LogoutRequest(BaseModel):
    """Request body for POST /api/v1/auth/logout."""

    refresh_token: str = Field(
        ...,
        description="Refresh token for the session to revoke.",
    )


class LogoutAllRequest(BaseModel):
    """Request body for POST /api/v1/auth/logout-all."""

    password: str = Field(
        ...,
        description="Current account password used to confirm full sign-out.",
    )


# ---------------------------------------------------------------------------
# Current user
# ---------------------------------------------------------------------------


class UserOut(BaseModel):
    """Public user profile returned from GET /api/v1/auth/me."""

    id: str = Field(
        ...,
        description="UUID v4 uniquely identifying this user.",
        examples=["3f2504e0-4f89-11d3-9a0c-0305e82c3301"],
    )
    username: str = Field(
        ...,
        description="Display username.",
        examples=["alice"],
    )
    email: str = Field(
        ...,
        description="Registered email address.",
        examples=["alice@example.com"],
    )
    role: str = Field(
        ...,
        description="Role controlling access level: 'admin' or 'user'.",
        examples=["user"],
    )
    created_at: datetime = Field(
        ...,
        description="UTC timestamp of account creation.",
    )

    _ensure_utc = field_validator("created_at", mode="before")(ensure_utc)


# ---------------------------------------------------------------------------
# API keys (programmatic access)
# ---------------------------------------------------------------------------


class APIKeyCreateRequest(BaseModel):
    """Request body for POST /api/v1/auth/api-keys."""

    label: str = Field(
        ...,
        max_length=128,
        description="Human-readable label to identify this key.",
        examples=["CI pipeline key"],
    )
    expires_in_days: int | None = Field(
        default=None,
        ge=1,
        le=3650,
        description="Optional expiry in days from creation; null means no expiry.",
        examples=[90],
    )


class APIKeyOut(BaseModel):
    """API key response — the raw key is returned ONLY on creation.

    Store the ``key`` value immediately; it cannot be retrieved again.
    Subsequent GET responses show a masked ``key_prefix`` only.
    """

    id: str = Field(
        ...,
        description="UUID v4 identifying this API key.",
    )
    label: str = Field(
        ...,
        description="Human-readable label.",
    )
    key: str | None = Field(
        default=None,
        description="Full API key — present ONLY on creation response; null on all other responses.",
        examples=["cgx_live_XXXXX"],
    )
    key_prefix: str = Field(
        ...,
        description="First 12 characters of the key for identification (always present).",
        examples=["cgx_live_XXX"],
    )
    created_at: datetime = Field(
        ...,
        description="UTC creation timestamp.",
    )
    expires_at: datetime | None = Field(
        default=None,
        description="UTC expiry timestamp; null if key never expires.",
    )
    last_used_at: datetime | None = Field(
        default=None,
        description="UTC timestamp of the last authenticated request using this key.",
    )

    _ensure_utc = field_validator("created_at", "expires_at", "last_used_at", mode="before")(
        ensure_utc
    )
