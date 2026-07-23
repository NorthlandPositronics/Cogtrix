"""JWT authentication and API key validation.

All endpoints (except /api/v1/health, /api/v1/health/ready, and the auth
registration/login endpoints) require a valid bearer token in the
``Authorization: Bearer <jwt>`` header.

WebSocket endpoints accept the token either in the Authorization header or
as the ``token`` query parameter (browsers cannot set custom WS headers).

JWT claims:
    sub  — user UUID v4
    exp  — expiry UNIX timestamp
    iat  — issued-at UNIX timestamp
    role — 'admin' or 'user'

The JWT secret is read exclusively from the environment variable
``COGTRIX_JWT_SECRET`` — never hardcoded.  The variable is accessed via
``src.config`` to participate in the hierarchical config system.

Error codes:
    UNAUTHORIZED   — token missing, malformed, or signature invalid
    TOKEN_EXPIRED  — token is valid but has expired (frontend should refresh)
    FORBIDDEN      — token valid but role insufficient for this endpoint
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Query, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger("cogtrix.api.auth")

# ---------------------------------------------------------------------------
# Security scheme (used by FastAPI /docs and OpenAPI schema)
# ---------------------------------------------------------------------------

_bearer_scheme = HTTPBearer(auto_error=False)

# ---------------------------------------------------------------------------
# bcrypt password hashing helpers
# ---------------------------------------------------------------------------

_BCRYPT_ROUNDS = 12


def hash_password(password: str) -> str:
    """Hash a password with bcrypt, returning the encoded hash string."""
    pw_bytes = password.encode("utf-8")[:72]  # bcrypt truncates at 72 bytes
    salt = bcrypt.gensalt(rounds=_BCRYPT_ROUNDS)
    return bcrypt.hashpw(pw_bytes, salt).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against a bcrypt hash."""
    pw_bytes = password.encode("utf-8")[:72]
    hash_bytes = password_hash.encode("utf-8")
    return bcrypt.checkpw(pw_bytes, hash_bytes)


# ---------------------------------------------------------------------------
# JWT constants
# ---------------------------------------------------------------------------

_ALGORITHM = "HS256"
_ACCESS_TOKEN_EXPIRE_SECONDS = 3600  # 1 hour
_REFRESH_TOKEN_EXPIRE_DAYS = 30


def _get_jwt_secret() -> str:
    """Return the JWT signing secret from env, raising RuntimeError if missing."""
    secret = os.environ.get("COGTRIX_JWT_SECRET", "")
    if len(secret) < 32:
        raise RuntimeError(
            "COGTRIX_JWT_SECRET must be set to at least 32 characters. "
            'Generate one with: python -c "import secrets; print(secrets.token_hex(32))"'
        )
    return secret


# ---------------------------------------------------------------------------
# Token data model
# ---------------------------------------------------------------------------


class TokenData:
    """Decoded, validated JWT claims attached to the request.

    Available on every authenticated endpoint via the ``current_user``
    FastAPI dependency.
    """

    def __init__(self, user_id: str, role: str, raw_claims: dict[str, Any]) -> None:
        self.user_id = user_id
        """UUID v4 of the authenticated user (``sub`` claim)."""
        self.role = role
        """Role string: 'admin' or 'user'."""
        self.raw_claims = raw_claims
        """Full decoded JWT payload for advanced use."""

    @property
    def is_admin(self) -> bool:
        """True when the user holds the 'admin' role."""
        return self.role == "admin"


# ---------------------------------------------------------------------------
# JWT helpers (used by auth routes)
# ---------------------------------------------------------------------------


def create_access_token(user_id: str, role: str) -> str:
    """Mint a new HS256 access JWT with a 1-hour expiry."""
    now = datetime.now(UTC)
    claims = {
        "sub": user_id,
        "role": role,
        "iat": now,
        "exp": now + timedelta(seconds=_ACCESS_TOKEN_EXPIRE_SECONDS),
    }
    return jwt.encode(claims, _get_jwt_secret(), algorithm=_ALGORITHM)


# ---------------------------------------------------------------------------
# JWT validation
# ---------------------------------------------------------------------------


def _decode_jwt(token: str) -> dict[str, Any]:
    """Decode and verify a JWT, returning its claims.

    Uses ``PyJWT`` with the HS256 algorithm.  The secret is loaded
    from the COGTRIX_JWT_SECRET environment variable.

    Raises:
        HTTPException 401 UNAUTHORIZED — token is malformed or signature invalid.
        HTTPException 401 TOKEN_EXPIRED — token has a valid signature but is expired.
    """
    try:
        claims: dict[str, Any] = jwt.decode(token, _get_jwt_secret(), algorithms=[_ALGORITHM])
        return claims
    except ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "TOKEN_EXPIRED",
                "message": "The JWT has expired; refresh the token and retry.",
            },
        ) from exc
    except InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "UNAUTHORIZED", "message": "Missing or invalid bearer token."},
        ) from exc


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    token: str | None = Query(
        default=None, description="JWT for WebSocket connections that cannot set headers."
    ),
) -> TokenData:
    """FastAPI dependency: validate the bearer token and return decoded claims.

    Accepts the token from:
    1. ``Authorization: Bearer <jwt>`` header (preferred).
    2. ``?token=<jwt>`` query parameter (WebSocket fallback).

    Raises:
        HTTPException 401 UNAUTHORIZED — no token provided or invalid signature.
        HTTPException 401 TOKEN_EXPIRED — valid signature but token expired.
    """
    raw_token: str | None = None
    if credentials is not None:
        raw_token = credentials.credentials
    elif token is not None:
        raw_token = token

    if raw_token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "UNAUTHORIZED", "message": "Missing or invalid bearer token."},
        )

    claims = _decode_jwt(raw_token)
    user_id: str = claims.get("sub", "")
    role: str = claims.get("role", "user")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "UNAUTHORIZED", "message": "Missing or invalid bearer token."},
        )
    return TokenData(user_id=user_id, role=role, raw_claims=claims)


async def require_admin(current_user: TokenData = Depends(get_current_user)) -> TokenData:
    """FastAPI dependency: require admin role.

    Raises:
        HTTPException 403 FORBIDDEN — authenticated user is not an admin.
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "FORBIDDEN",
                "message": "Authenticated user lacks permission for this action.",
            },
        )
    return current_user


async def get_current_user_optional(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    token: str | None = Query(default=None),
) -> TokenData | None:
    """FastAPI dependency: return decoded claims or None if no token is present.

    Used on endpoints that behave differently for authenticated vs. anonymous
    callers.

    Re-raises TOKEN_EXPIRED and UNAUTHORIZED so callers cannot exploit an
    expired token to obtain anonymous-level access on protected endpoints.
    """
    raw_token: str | None = None
    if credentials is not None:
        raw_token = credentials.credentials
    elif token is not None:
        raw_token = token

    if raw_token is None:
        return None

    # A token was supplied — validate it strictly.  Do not silently degrade
    # an invalid/expired credential to anonymous access.
    return await get_current_user(request, credentials, token)


# ---------------------------------------------------------------------------
# API key validation (programmatic access alternative to JWT)
# ---------------------------------------------------------------------------


async def validate_api_key(api_key: str, db: AsyncSession) -> TokenData:
    """Look up an API key and return the associated user's TokenData.

    API keys are stored hashed in the database.  The raw key is only
    available at creation time.

    Args:
        api_key: The raw API key string.
        db: The caller's database session (from ``Depends(get_db)``).

    Raises:
        HTTPException 401 UNAUTHORIZED — key not found or revoked.
        HTTPException 401 TOKEN_EXPIRED — key has passed its expires_at timestamp.
    """
    import hashlib

    from src.api.db.repositories.api_keys import ApiKeyRepository
    from src.api.db.repositories.users import UserRepository

    key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    repo = ApiKeyRepository(db)
    key_record = await repo.get_by_hash(key_hash)

    if key_record is None or key_record.revoked:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "UNAUTHORIZED", "message": "Missing or invalid bearer token."},
        )

    if key_record.expires_at is not None:
        now = datetime.now(UTC)
        expires = key_record.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=UTC)
        if now > expires:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "UNAUTHORIZED", "message": "Missing or invalid bearer token."},
            )

    user_repo = UserRepository(db)
    user = await user_repo.get_by_id(key_record.user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "UNAUTHORIZED", "message": "Missing or invalid bearer token."},
        )

    # Update last_used_at only after confirming the user exists.
    # Explicit commit ensures the timestamp persists even for read-only endpoints
    # where the route handler never calls db.commit().
    await repo.update_last_used(key_record.id, datetime.now(UTC))
    await db.commit()

    return TokenData(
        user_id=user.id,
        role=user.role,
        raw_claims={"sub": user.id, "role": user.role},
    )


# ---------------------------------------------------------------------------
# Session ownership guard
# ---------------------------------------------------------------------------


async def verify_session_owner(session_id: str, current_user: TokenData, db: AsyncSession) -> None:
    """Ensure the current user owns the given session.

    Admins may access any session.  Regular users may only access their own.

    Args:
        session_id: UUID v4 of the session to check.
        current_user: Decoded JWT claims from the request.
        db: The caller's database session (from ``Depends(get_db)``).

    Raises:
        HTTPException 404 SESSION_NOT_FOUND — session does not exist.
        HTTPException 403 FORBIDDEN — session belongs to a different user.
    """
    if current_user.is_admin:
        return

    from sqlalchemy import select

    from src.api.db.models import ApiSessionRecord

    result = await db.execute(select(ApiSessionRecord).where(ApiSessionRecord.id == session_id))
    record = result.scalar_one_or_none()

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "SESSION_NOT_FOUND",
                "message": "The requested session does not exist.",
            },
        )

    if record.user_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "FORBIDDEN",
                "message": "Authenticated user lacks permission for this action.",
            },
        )
