"""User management schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field, field_validator

from cogtrix_core.api.schemas.common import ensure_utc
from cogtrix_core.api.schemas.validators import validate_password_complexity


class UserOut(BaseModel):
    """A registered user account."""

    id: str = Field(
        ...,
        description="UUID v4 of the user.",
        examples=["3f2504e0-4f89-11d3-9a0c-0305e82c3301"],
    )
    username: str = Field(
        ...,
        description="Unique username.",
        examples=["alice"],
    )
    email: str = Field(
        ...,
        description="User email address.",
        examples=["alice@example.com"],
    )
    role: str = Field(
        ...,
        description="User role: 'admin' or 'user'.",
        examples=["user"],
    )
    created_at: datetime = Field(
        ...,
        description="UTC timestamp when the account was created.",
    )

    _ensure_utc = field_validator("created_at", mode="before")(ensure_utc)


class UserCreateRequest(BaseModel):
    """Request body for POST /api/v1/users."""

    username: str = Field(
        ...,
        min_length=3,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Unique username (letters, digits, underscore, hyphen).",
        examples=["alice"],
    )
    email: EmailStr = Field(
        ...,
        description="User email address.",
        examples=["alice@example.com"],
    )
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Initial password (8–128 chars; must include lowercase, uppercase, digit, and special character).",
    )

    _password_complexity = field_validator("password")(validate_password_complexity)
    role: str = Field(
        default="user",
        pattern=r"^(user|admin)$",
        description="User role: 'admin' or 'user'.",
        examples=["user"],
    )


class UserUpdateRequest(BaseModel):
    """Request body for PATCH /api/v1/users/{user_id}.

    All fields are optional; only supplied fields are updated.
    """

    role: str | None = Field(
        default=None,
        pattern=r"^(user|admin)$",
        description="New role for the user: 'admin' or 'user'.",
        examples=["admin"],
    )


class PasswordResetRequest(BaseModel):
    """Request body for POST /api/v1/users/{user_id}/reset-password (admin, #2065)."""

    new_password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="New password for the user (8–128 chars; lowercase, uppercase, digit and special required).",
    )

    _new_password_complexity = field_validator("new_password")(validate_password_complexity)
