"""JWT authentication and API key validation.

All endpoints (except /api/v1/health, /api/v1/health/ready, and the auth
registration/login endpoints) require a valid bearer token in the
``Authorization: Bearer <jwt>`` header. API keys with the ``cgx_live_``
prefix are also accepted on the same bearer channel and are validated via
``validate_api_key()``.

WebSocket endpoints accept the token either in the Authorization header or
as the ``token`` query parameter (browsers cannot set custom WS headers).

JWT claims:
    sub  — user UUID v4
    exp  — expiry UNIX timestamp
    iat  — issued-at UNIX timestamp
    role — 'admin' or 'user'

The JWT secret is read exclusively from the environment variable
``COGTRIX_JWT_SECRET`` — never hardcoded.  API startup snapshots the value
into an in-process cache so token operations do not re-read the environment
on every request.

Error codes:
    UNAUTHORIZED   — token missing, malformed, or signature invalid
    TOKEN_EXPIRED  — token is valid but has expired (frontend should refresh)
    FORBIDDEN      — token valid but role insufficient for this endpoint
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Query, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.db import get_db

log = logging.getLogger("cogtrix.api.auth")

# In-process debounce for API key last_used_at writes.
# Maps key_id -> monotonic timestamp of the last DB write.
_API_KEY_LAST_USED: dict[str, float] = {}
_API_KEY_DEBOUNCE_SECONDS = 60.0
_API_KEY_LOCK = asyncio.Lock()


def _cleanup_stale_api_key_entries() -> None:
    """Remove entries older than twice the debounce interval."""
    cutoff = time.monotonic() - (_API_KEY_DEBOUNCE_SECONDS * 2)
    stale = [k for k, v in _API_KEY_LAST_USED.items() if v < cutoff]
    for k in stale:
        del _API_KEY_LAST_USED[k]


def _hash_api_key(token: str) -> str:
    """HMAC-SHA256 hash for API key lookup — delegates to isolated module."""
    from src.api._key_hash import hash_api_key

    return hash_api_key(token)


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
_JWT_SECRET: str | None = None


def configure_jwt_secret(secret: str) -> None:
    """Snapshot the JWT signing secret for this process."""
    if len(secret) < 32:
        raise RuntimeError(
            "COGTRIX_JWT_SECRET must be set to at least 32 characters. "
            'Generate one with: python -c "import secrets; print(secrets.token_hex(32))"'
        )
    global _JWT_SECRET
    _JWT_SECRET = secret


def _get_jwt_secret() -> str:
    """Return the cached JWT signing secret, lazily initializing it once."""
    global _JWT_SECRET
    if _JWT_SECRET is None:
        configure_jwt_secret(os.environ.get("COGTRIX_JWT_SECRET", ""))
    if _JWT_SECRET is None:
        raise RuntimeError("JWT secret not configured")
    return _JWT_SECRET


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
        """Role string: 'admin', 'superadmin', or 'user'."""
        self.raw_claims = raw_claims
        """Full decoded JWT payload for advanced use."""

    @property
    def is_admin(self) -> bool:
        """True when the user holds the 'admin' or 'superadmin' role."""
        return self.role in ("admin", "superadmin")

    @property
    def is_superadmin(self) -> bool:
        """True when the user holds the 'superadmin' role."""
        return self.role == "superadmin"

    @property
    def is_impersonating(self) -> bool:
        """True when the request is carrying an impersonation token."""
        return bool(self.raw_claims.get("impersonated_by"))


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


def create_impersonation_token(
    impersonated_user_id: str,
    impersonated_role: str,
    superadmin_id: str,
    session_id: str,
    duration_minutes: int = 30,
) -> str:
    """Mint an impersonation JWT.

    The token encodes the impersonated user as ``sub`` and includes
    ``impersonated_by`` / ``impersonation_session_id`` claims so the
    auth layer can validate the active session on every request.
    """
    now = datetime.now(UTC)
    claims = {
        "sub": impersonated_user_id,
        "role": impersonated_role,
        "impersonated_by": superadmin_id,
        "impersonation_session_id": session_id,
        "iat": now,
        "exp": now + timedelta(minutes=duration_minutes),
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
    db: AsyncSession = Depends(get_db),
) -> TokenData:
    """FastAPI dependency: validate the bearer token and return decoded claims.

    Accepts the token from:
    1. ``Authorization: Bearer <jwt>`` header (preferred).
    2. ``?token=<jwt>`` query parameter (WebSocket fallback).

    API keys with the ``cgx_live_`` prefix are validated through the API key
    repository path before JWT decoding.

    Falls back to OIDC validation when a validator is configured and the local
    JWT check raises UNAUTHORIZED (invalid token).  TOKEN_EXPIRED is never
    retried via OIDC — it propagates immediately so the frontend can refresh.

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

    if raw_token.startswith("cgx_live_"):
        current = await validate_api_key(raw_token, db)
        await _reject_inactive_user(current.user_id, db)
        return current

    try:
        claims = _decode_jwt(raw_token)
    except HTTPException as local_exc:
        detail = local_exc.detail
        code = detail.get("code") if isinstance(detail, dict) else None
        if code == "TOKEN_EXPIRED":
            raise
        # UNAUTHORIZED: try OIDC fallback if configured.
        from src.api.oidc import get_validator  # lazy to avoid circular at import time

        validator = get_validator()
        if validator is None:
            raise
        try:
            # validator.validate() uses urllib (blocking I/O) — run in a
            # thread pool to avoid blocking the async event loop.
            import asyncio as _asyncio

            oidc_claims = await _asyncio.to_thread(validator.validate, raw_token)
        except Exception:
            raise local_exc from None
        oidc_role = validator.map_role(oidc_claims)
        oidc_user_id = str(oidc_claims.get("sub", ""))
        if not oidc_user_id:
            raise local_exc from None
        current = TokenData(user_id=oidc_user_id, role=oidc_role, raw_claims=oidc_claims)
        await _reject_inactive_user(current.user_id, db)
        return current

    user_id: str = claims.get("sub", "")
    role: str = claims.get("role", "user")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "UNAUTHORIZED", "message": "Missing or invalid bearer token."},
        )

    # Handle impersonation tokens
    impersonation_session_id: str | None = claims.get("impersonation_session_id")
    if impersonation_session_id is not None:
        from sqlalchemy import select

        from src.api.db.models import ImpersonationSession

        result = await db.execute(
            select(ImpersonationSession).where(ImpersonationSession.id == impersonation_session_id)
        )
        imp_session = result.scalar_one_or_none()
        now = datetime.now(UTC)
        if imp_session is None or (imp_session.ended_at is not None):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "code": "UNAUTHORIZED",
                    "message": "Impersonation session has ended or does not exist.",
                },
            )
        expires_at = imp_session.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        if expires_at < now:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "code": "TOKEN_EXPIRED",
                    "message": "Impersonation session has expired; re-authenticate.",
                },
            )

    current = TokenData(user_id=user_id, role=role, raw_claims=claims)
    await _reject_inactive_user(current.user_id, db)
    return current


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


async def require_superadmin(current_user: TokenData = Depends(get_current_user)) -> TokenData:
    """FastAPI dependency: require superadmin role.

    Raises:
        HTTPException 403 FORBIDDEN — authenticated user is not a superadmin.
    """
    if not current_user.is_superadmin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "FORBIDDEN",
                "message": "Authenticated user lacks permission for this action.",
            },
        )
    return current_user


async def get_admin_org(
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> str | None:
    """FastAPI dependency: return the admin's org_id, or None for superadmins.

    Used by admin enumeration endpoints that need org-scoping.  Superadmins
    (role == 'superadmin') receive ``None`` so they can see data across all
    organizations.  Regular admins receive their ``org_id`` from the user
    record so the endpoint can filter (or reject when org metadata is not
    yet available).

    When ``enable_org_scoping`` is False (default), all admins receive ``None``
    to preserve backward compatibility until Phase 2 rollout.

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
    # Feature flag: disable org scoping by default for backward compatibility
    if os.getenv("COGTRIX_ENABLE_ORG_SCOPING", "").lower() not in ("true", "1", "yes"):
        return None
    if current_user.is_superadmin:
        return None
    from src.api.db.repositories.users import UserRepository

    repo = UserRepository(db)
    user = await repo.get_by_id(current_user.user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "FORBIDDEN",
                "message": "Authenticated user lacks permission for this action.",
            },
        )
    if user.org_id is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "ORG_NOT_ASSIGNED",
                "message": "Admin account is not assigned to an organization.",
            },
        )
    return user.org_id


async def get_current_user_optional(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    token: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
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
    return await get_current_user(request, credentials, token, db)


async def _reject_inactive_user(user_id: str, db: AsyncSession) -> None:
    """Reject authentication for a user record that has been deactivated."""
    from src.api.db.repositories.users import UserRepository

    repo = UserRepository(db)
    user = await repo.get_by_id(user_id)
    if user is not None and not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "UNAUTHORIZED", "message": "Missing or invalid bearer token."},
        )


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
    from src.api.db.repositories.api_keys import ApiKeyRepository
    from src.api.db.repositories.users import UserRepository

    key_hash = _hash_api_key(api_key)

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
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "UNAUTHORIZED", "message": "Missing or invalid bearer token."},
        )

    # Update last_used_at only after confirming the user exists.
    # Debounce in-process to avoid write amplification at high QPS.
    async with _API_KEY_LOCK:
        now_mono = time.monotonic()
        last_written = _API_KEY_LAST_USED.get(key_record.id, 0)
        if now_mono - last_written >= _API_KEY_DEBOUNCE_SECONDS:
            await repo.update_last_used(key_record.id, datetime.now(UTC))
            await db.commit()
            _API_KEY_LAST_USED[key_record.id] = now_mono
            _cleanup_stale_api_key_entries()

    return TokenData(
        user_id=user.id,
        role=user.role,
        raw_claims={"sub": user.id, "role": user.role},
    )


# ---------------------------------------------------------------------------
# Session ownership guard
# ---------------------------------------------------------------------------


async def verify_session_owner(
    session_id: str,
    current_user: TokenData,
    db: AsyncSession,
    *,
    admin_bypass: bool = True,
) -> None:
    """Ensure the current user owns the given session.

    Admins may access any session when *admin_bypass* is ``True`` (default).
    Regular users may only access their own.

    Args:
        session_id: UUID v4 of the session to check.
        current_user: Decoded JWT claims from the request.
        db: The caller's database session (from ``Depends(get_db)``).
        admin_bypass: When ``True``, skip the check for admin callers.

    Raises:
        HTTPException 404 SESSION_NOT_FOUND — session does not exist.
        HTTPException 403 FORBIDDEN — session belongs to a different user.
    """
    if admin_bypass and current_user.is_admin:
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
