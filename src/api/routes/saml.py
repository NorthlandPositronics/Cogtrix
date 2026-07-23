"""SAML 2.0 SP routes (Enterprise Phase 1 — task 1.2.1).

Endpoints:
    GET  /api/v1/saml/metadata          — SP metadata XML
    GET  /api/v1/saml/sso               — initiate SP-initiated SSO
    POST /api/v1/saml/acs               — Assertion Consumer Service

Requires the ``[saml]`` optional extra.  When not installed, all endpoints
return ``503 SAML_NOT_INSTALLED``.

Org binding:
    If ``SAMLConfig.org_id`` is set, all SAML-provisioned users are assigned
    to that organization.  If not set, users are assigned to the default org
    (slug='default') via ``ensure_default_org()``.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import RedirectResponse, Response
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.db.engine import get_db
from src.api.rate_limit import per_route_rate_limit
from src.api.saml.config import SAMLConfig, get_saml_config

log = logging.getLogger("cogtrix.api.saml")

router = APIRouter(prefix="/saml", tags=["SAML SSO"])


def _saml_not_configured() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"code": "SAML_NOT_CONFIGURED", "message": "SAML SSO is not configured."},
    )


def _saml_not_installed() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "code": "SAML_NOT_INSTALLED",
            "message": (
                "SAML SSO requires the [saml] optional extra. "
                "Install with: pip install cogtrix[saml]"
            ),
        },
    )


def _build_request_data(
    request: Request,
    config: SAMLConfig,
    post_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the request context dict expected by python3-saml.

    http_host, server_port, and https are derived from the configured
    sp_acs_url rather than from the incoming request headers — preventing
    Host header injection attacks where an attacker supplies a spoofed
    Host header to manipulate python3-saml's URL construction.
    """
    parsed = urlparse(config.sp_acs_url)
    is_https = parsed.scheme == "https"
    # netloc includes port when non-standard (e.g. "example.com:8443")
    http_host = parsed.netloc or parsed.hostname or ""
    port = parsed.port or (443 if is_https else 80)
    return {
        "http_host": http_host,
        "script_name": request.url.path,
        "server_port": str(port),
        "get_data": dict(request.query_params),
        "post_data": post_data or {},
        "https": "on" if is_https else "off",
    }


# ---------------------------------------------------------------------------
# SP Metadata
# ---------------------------------------------------------------------------


@router.get(
    "/metadata",
    summary="SAML SP metadata",
    description="Returns the Service Provider metadata XML for IdP registration.",
    response_class=Response,
)
async def saml_metadata() -> Response:
    """Return SP metadata XML for IdP configuration."""
    config = get_saml_config()
    if config is None:
        raise _saml_not_configured()
    try:
        from src.api.saml.provider import get_metadata_xml

        xml = get_metadata_xml(config)
        return Response(content=xml, media_type="application/xml")
    except ImportError:
        raise _saml_not_installed() from None
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "SAML_METADATA_ERROR", "message": str(exc)},
        ) from exc


# ---------------------------------------------------------------------------
# SP-initiated SSO
# ---------------------------------------------------------------------------


@router.get(
    "/sso",
    summary="Initiate SAML SSO",
    description="Redirects the user to the IdP for authentication.",
    response_class=RedirectResponse,
    status_code=302,
)
async def saml_sso(request: Request) -> RedirectResponse:
    """Build a SAMLRequest and redirect to the IdP SSO URL."""
    config = get_saml_config()
    if config is None:
        raise _saml_not_configured()
    try:
        from src.api.saml.provider import get_sso_redirect_url

        redirect_url = get_sso_redirect_url(config, _build_request_data(request, config))
        return RedirectResponse(url=redirect_url, status_code=302)
    except ImportError:
        raise _saml_not_installed() from None


# ---------------------------------------------------------------------------
# Assertion Consumer Service
# ---------------------------------------------------------------------------


@router.post(
    "/acs",
    summary="SAML Assertion Consumer Service",
    description=(
        "Receives the SAMLResponse POST from the IdP, validates it, "
        "provisions or retrieves the user, and issues a Cogtrix JWT."
    ),
)
async def saml_acs(
    request: Request,
    SAMLResponse: str = Form(..., description="Base64-encoded SAMLResponse from the IdP."),
    db: AsyncSession = Depends(get_db),
    _rl: None = Depends(per_route_rate_limit(5, 60)),
) -> dict[str, str]:
    """Process IdP SAMLResponse, provision the user, and return a JWT."""
    config = get_saml_config()
    if config is None:
        raise _saml_not_configured()

    try:
        from src.api.saml.provider import process_saml_response
    except ImportError:
        raise _saml_not_installed() from None

    post_data = {"SAMLResponse": SAMLResponse}
    if relay_state := (await request.form()).get("RelayState"):
        post_data["RelayState"] = str(relay_state)

    try:
        assertion = process_saml_response(config, _build_request_data(request, config, post_data))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "SAML_INVALID_RESPONSE", "message": str(exc)},
        ) from exc

    user, token = await _provision_user(db, assertion, config)
    return {"access_token": token, "token_type": "bearer"}  # nosec B105


async def _provision_user(
    db: AsyncSession,
    assertion: Any,
    config: Any,
) -> tuple[Any, str]:
    """Find or create a user from a SAML assertion and issue a JWT."""
    from src.api.auth import create_access_token, hash_password
    from src.api.db.repositories.organization import OrganizationRepository  # noqa: F811
    from src.api.db.repositories.users import UserRepository  # noqa: F811

    user_repo = UserRepository(db)
    org_repo = OrganizationRepository(db)

    # Resolve org.
    if config.org_id:
        org_id: str | None = config.org_id
    else:
        default_org = await org_repo.ensure_default_org()
        await db.commit()
        org_id = default_org.id

    # Find existing user scoped to this org, or check for conflicts.
    existing = await user_repo.get_by_email(assertion.email, org_id=org_id)
    if existing is not None:
        token = create_access_token(user_id=existing.id, role=existing.role)
        return existing, token

    conflict = await user_repo.get_by_email(assertion.email)
    if conflict is not None:
        if conflict.org_id is not None:
            # User belongs to a specific different org — cross-org takeover attempt.
            # Return 422 (not 409) with an opaque message to prevent enumeration.
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={"code": "USER_ACCOUNT_CONFLICT", "message": "User account conflict."},
            )
        # User exists but has no org — assign to this org.
        await user_repo.assign_org(conflict.id, org_id)
        await db.commit()
        token = create_access_token(user_id=conflict.id, role=conflict.role)
        return conflict, token

    # Resolve role from SAML attribute mapping, falling back to default_role.
    attr_map = config.attribute_map
    role_attr = attr_map.get("role")
    if role_attr:
        role = assertion.attributes.get(role_attr, [None])[0]
        if role and isinstance(role, str) and role.strip():
            role = role.strip().lower()
        else:
            role = None
    else:
        role = None
    provision_role = role if role else config.default_role

    # Provision a new user.
    import uuid

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            new_user = await user_repo.create(
                user_id=str(uuid.uuid4()),
                username=_unique_username(assertion.username),
                email=assertion.email,
                password_hash=hash_password(str(uuid.uuid4())),  # random unusable password
                role=provision_role,
                org_id=org_id,
            )
            await db.commit()
            token = create_access_token(user_id=new_user.id, role=new_user.role)
            return new_user, token
        except IntegrityError as exc:
            await db.rollback()
            if attempt == max_attempts - 1:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail={
                        "code": "VALIDATION_ERROR",
                        "message": "Username already taken.",
                    },
                ) from exc


def _unique_username(base: str) -> str:
    """Append a short random suffix to ensure username uniqueness."""
    import uuid

    suffix = uuid.uuid4().hex[:6]
    return f"{base[:58]}_{suffix}"
