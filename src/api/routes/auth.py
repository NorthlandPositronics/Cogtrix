"""Authentication endpoints — registration, login, token refresh, API key management.

Endpoints:
    POST /api/v1/auth/register          — create a new user account
    POST /api/v1/auth/login             — authenticate and receive a token pair
    POST /api/v1/auth/refresh           — silently renew an access token
    POST /api/v1/auth/logout            — invalidate the current refresh token
    GET  /api/v1/auth/me                — retrieve the current user's profile
    GET  /api/v1/auth/api-keys          — list the current user's API keys
    POST /api/v1/auth/api-keys          — create a new API key
    DELETE /api/v1/auth/api-keys/{id}   — revoke an API key
"""

from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import (
    _ACCESS_TOKEN_EXPIRE_SECONDS,
    _REFRESH_TOKEN_EXPIRE_DAYS,
    TokenData,
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)
from src.api.db import get_db
from src.api.db.repositories.api_keys import ApiKeyRepository
from src.api.db.repositories.tokens import RefreshTokenRepository
from src.api.db.repositories.users import UserRepository
from src.api.pagination import decode_cursor, encode_cursor
from src.api.schemas.auth import (
    APIKeyCreateRequest,
    APIKeyOut,
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenPair,
    UserOut,
)
from src.api.schemas.common import APIResponse, CursorPage

router = APIRouter(prefix="/auth", tags=["Authentication"])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_API_KEY_PREFIX = "cgx_live_"


async def _create_token_pair(
    user_id: str,
    role: str,
    db: AsyncSession,
) -> TokenPair:
    """Create an access + refresh token pair, persist the refresh token hash, return TokenPair."""
    access_token = create_access_token(user_id, role)
    raw_refresh = secrets.token_urlsafe(48)
    refresh_hash = hashlib.sha256(raw_refresh.encode()).hexdigest()

    refresh_repo = RefreshTokenRepository(db)
    await refresh_repo.create(
        token_id=str(uuid.uuid4()),
        user_id=user_id,
        token_hash=refresh_hash,
        expires_at=datetime.now(UTC) + timedelta(days=_REFRESH_TOKEN_EXPIRE_DAYS),
    )

    return TokenPair(
        access_token=access_token,
        refresh_token=raw_refresh,
        token_type="bearer",
        expires_in=_ACCESS_TOKEN_EXPIRE_SECONDS,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/register",
    summary="Register a new user account",
    description=(
        "Create a new user account. "
        "Returns a token pair so the user can immediately start making authenticated requests. "
        "The first registered user is automatically granted the 'admin' role."
    ),
    response_model=APIResponse[TokenPair],
    status_code=201,
    responses={
        201: {"description": "Account created; token pair returned."},
        409: {"description": "Username or email already exists (VALIDATION_ERROR)."},
        422: {"description": "Request body validation failed (VALIDATION_ERROR)."},
    },
)
async def register(
    body: RegisterRequest,
    db: AsyncSession = Depends(get_db),
) -> APIResponse[TokenPair]:
    """Create a new user account and return an access + refresh token pair.

    Auth: none (open endpoint).
    Error codes:
        VALIDATION_ERROR — username/email already taken or body invalid.
    """
    user_repo = UserRepository(db)

    # Check for duplicates
    existing_by_username = await user_repo.get_by_username(body.username)
    if existing_by_username is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "VALIDATION_ERROR", "message": "Username already taken."},
        )
    existing_by_email = await user_repo.get_by_email(body.email)
    if existing_by_email is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "VALIDATION_ERROR", "message": "Email already registered."},
        )

    # Atomic role election: first registered user gets admin role.
    # Uses INSERT…SELECT so the count check and insert are a single statement,
    # preventing a race where two concurrent registrations both get admin.
    # The unique-constraint check above reduces DB load but is not sufficient
    # on its own — a concurrent registration that slips through is caught here
    # by the database unique constraint, producing an IntegrityError.
    password_hash = hash_password(body.password)
    try:
        user = await user_repo.create_with_role_election(
            user_id=str(uuid.uuid4()),
            username=body.username,
            email=body.email,
            password_hash=password_hash,
        )
    except IntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "VALIDATION_ERROR",
                "message": "Username or email already taken.",
            },
        ) from exc

    token_pair = await _create_token_pair(user.id, user.role, db)
    await db.commit()
    return APIResponse(data=token_pair)


@router.post(
    "/login",
    summary="Authenticate and receive tokens",
    description=(
        "Authenticate with username + password. "
        "Returns a short-lived access token (1 hour) and a long-lived refresh token (30 days). "
        "Store both; use the refresh token to silently renew the access token."
    ),
    response_model=APIResponse[TokenPair],
    responses={
        200: {"description": "Authenticated; token pair returned."},
        401: {"description": "Invalid credentials (UNAUTHORIZED)."},
    },
)
async def login(
    body: LoginRequest,
    db: AsyncSession = Depends(get_db),
) -> APIResponse[TokenPair]:
    """Authenticate with username/email + password and return a token pair.

    Auth: none.
    Error codes:
        UNAUTHORIZED — credentials are incorrect.
    """
    user_repo = UserRepository(db)

    # Support both username and email login
    user = await user_repo.get_by_username(body.username)
    if user is None:
        user = await user_repo.get_by_email(body.username)

    _invalid = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"code": "UNAUTHORIZED", "message": "Invalid username or password."},
    )

    if user is None:
        raise _invalid

    if not verify_password(body.password, user.password_hash):
        raise _invalid

    token_pair = await _create_token_pair(user.id, user.role, db)
    await db.commit()
    return APIResponse(data=token_pair)


@router.post(
    "/refresh",
    summary="Renew access token using refresh token",
    description=(
        "Exchange a valid refresh token for a new access + refresh token pair. "
        "The old refresh token is invalidated on success (rotation). "
        "Call this endpoint when the frontend receives TOKEN_EXPIRED on any other endpoint."
    ),
    response_model=APIResponse[TokenPair],
    responses={
        200: {"description": "New token pair returned."},
        401: {"description": "Refresh token invalid or expired (TOKEN_EXPIRED)."},
    },
)
async def refresh(
    body: RefreshRequest,
    db: AsyncSession = Depends(get_db),
) -> APIResponse[TokenPair]:
    """Rotate the refresh token and issue a new access + refresh token pair.

    Auth: none (the refresh_token in the body is the credential).
    Error codes:
        TOKEN_EXPIRED  — the refresh token has expired; user must log in again.
        UNAUTHORIZED   — the refresh token is invalid or has been revoked.
    """
    token_repo = RefreshTokenRepository(db)
    token_hash = hashlib.sha256(body.refresh_token.encode()).hexdigest()
    token_record = await token_repo.get_by_hash(token_hash)

    if token_record is None or token_record.revoked:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "UNAUTHORIZED", "message": "Missing or invalid bearer token."},
        )

    now = datetime.now(UTC)
    expires = token_record.expires_at
    if expires.tzinfo is None:

        expires = expires.replace(tzinfo=UTC)
    if now > expires:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "TOKEN_EXPIRED",
                "message": "The JWT has expired; refresh the token and retry.",
            },
        )

    # Revoke old token
    await token_repo.revoke(token_record.id)

    # Get user for role
    user_repo = UserRepository(db)
    user = await user_repo.get_by_id(token_record.user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "UNAUTHORIZED", "message": "Missing or invalid bearer token."},
        )

    token_pair = await _create_token_pair(user.id, user.role, db)
    await db.commit()
    return APIResponse(data=token_pair)


@router.post(
    "/logout",
    summary="Invalidate the current session",
    description="Revoke the refresh token associated with the current session. The access token remains valid until it expires naturally.",
    response_model=APIResponse[None],
    responses={
        200: {"description": "Refresh token revoked."},
        401: {"description": "Not authenticated (UNAUTHORIZED)."},
    },
)
async def logout(
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[None]:
    """Revoke the current refresh token (server-side session invalidation).

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED.
    """
    token_repo = RefreshTokenRepository(db)
    await token_repo.revoke_all_for_user(current_user.user_id)
    await db.commit()
    return APIResponse(data=None)


@router.get(
    "/me",
    summary="Get current user profile",
    description="Return the profile of the currently authenticated user.",
    response_model=APIResponse[UserOut],
    responses={
        200: {"description": "User profile returned."},
        401: {"description": "Not authenticated (UNAUTHORIZED or TOKEN_EXPIRED)."},
    },
)
async def get_me(
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[UserOut]:
    """Return the profile of the currently authenticated user.

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED.
    """
    user_repo = UserRepository(db)
    user = await user_repo.get_by_id(current_user.user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "UNAUTHORIZED", "message": "Missing or invalid bearer token."},
        )
    return APIResponse(
        data=UserOut(
            id=user.id,
            username=user.username,
            email=user.email,
            role=user.role,
            created_at=user.created_at,
        )
    )


@router.get(
    "/api-keys",
    summary="List API keys",
    description="List all API keys owned by the current user. Key values are never returned — only the prefix.",
    response_model=APIResponse[CursorPage[APIKeyOut]],
    responses={
        200: {"description": "API key list returned."},
        401: {"description": "Not authenticated."},
    },
)
async def list_api_keys(
    cursor: str | None = None,
    limit: int = 20,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[CursorPage[APIKeyOut]]:
    """List the current user's API keys (paginated).

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, INVALID_CURSOR.
    """
    limit = max(1, min(limit, 100))

    after_id: str | None = None
    if cursor is not None:
        try:
            after_id = decode_cursor(cursor)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "INVALID_CURSOR", "message": str(exc)},
            ) from exc

    key_repo = ApiKeyRepository(db)
    rows = await key_repo.list_for_user(current_user.user_id, after_id=after_id, limit=limit)

    has_more = len(rows) > limit
    items = rows[:limit]
    next_cursor = encode_cursor(items[-1].id) if has_more and items else None

    return APIResponse(
        data=CursorPage(
            items=[
                APIKeyOut(
                    id=k.id,
                    label=k.label,
                    key=None,
                    key_prefix=k.key_prefix,
                    created_at=k.created_at,
                    expires_at=k.expires_at,
                    last_used_at=k.last_used_at,
                )
                for k in items
            ],
            next_cursor=next_cursor,
            has_more=has_more,
            total=None,
        )
    )


@router.post(
    "/api-keys",
    summary="Create an API key",
    description=(
        "Create a new API key for programmatic access. "
        "The full key value is returned ONCE in this response and cannot be retrieved again. "
        "Store it immediately."
    ),
    response_model=APIResponse[APIKeyOut],
    status_code=201,
    responses={
        201: {"description": "API key created; full key value in response."},
        401: {"description": "Not authenticated."},
    },
)
async def create_api_key(
    body: APIKeyCreateRequest,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[APIKeyOut]:
    """Create a new API key and return the full key value (one-time).

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED.
    """
    from src.api.auth import _hash_api_key

    raw_key = _API_KEY_PREFIX + secrets.token_urlsafe(32)
    key_hash = _hash_api_key(raw_key)
    key_prefix = raw_key[:12]

    expires_at = None
    if body.expires_in_days is not None:
        expires_at = datetime.now(UTC) + timedelta(days=body.expires_in_days)

    key_repo = ApiKeyRepository(db)
    key_record = await key_repo.create(
        key_id=str(uuid.uuid4()),
        user_id=current_user.user_id,
        key_hash=key_hash,
        key_prefix=key_prefix,
        label=body.label,
        expires_at=expires_at,
    )

    await db.commit()
    return APIResponse(
        data=APIKeyOut(
            id=key_record.id,
            label=key_record.label,
            key=raw_key,
            key_prefix=key_prefix,
            created_at=key_record.created_at,
            expires_at=key_record.expires_at,
            last_used_at=key_record.last_used_at,
        )
    )


@router.delete(
    "/api-keys/{key_id}",
    summary="Revoke an API key",
    description="Permanently revoke an API key. Requests authenticated with this key will immediately receive 401.",
    response_model=APIResponse[None],
    responses={
        200: {"description": "API key revoked."},
        401: {"description": "Not authenticated."},
        404: {"description": "Key not found (NOT_FOUND)."},
    },
)
async def revoke_api_key(
    key_id: str,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[None]:
    """Permanently revoke an API key.

    Auth: bearer token required. Users may only revoke their own keys; admins may revoke any.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, NOT_FOUND, FORBIDDEN.
    """
    key_repo = ApiKeyRepository(db)
    key_record = await key_repo.get_by_id(key_id)

    if key_record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "The requested resource does not exist."},
        )

    if not current_user.is_admin and key_record.user_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "FORBIDDEN",
                "message": "Authenticated user lacks permission for this action.",
            },
        )

    await key_repo.revoke(key_id)
    await db.commit()
    return APIResponse(data=None)
