"""SCIM 2.0 resource schemas (RFC 7643).

Pydantic models for SCIM request and response payloads.
All models serialise to/from the SCIM wire format.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# SCIM URNs
# ---------------------------------------------------------------------------

SCHEMA_USER = "urn:ietf:params:scim:schemas:core:2.0:User"
SCHEMA_LIST_RESPONSE = "urn:ietf:params:scim:api:messages:2.0:ListResponse"
SCHEMA_ERROR = "urn:ietf:params:scim:api:messages:2.0:Error"
SCHEMA_SERVICE_PROVIDER_CONFIG = "urn:ietf:params:scim:schemas:core:2.0:ServiceProviderConfig"
SCHEMA_PATCH_OP = "urn:ietf:params:scim:api:messages:2.0:PatchOp"

# ---------------------------------------------------------------------------
# Sub-resources
# ---------------------------------------------------------------------------


class SCIMName(BaseModel):
    formatted: str | None = None
    givenName: str | None = None
    familyName: str | None = None


class SCIMEmail(BaseModel):
    value: str
    primary: bool = False
    type: str | None = None


class SCIMMeta(BaseModel):
    resourceType: str
    created: datetime | None = None
    lastModified: datetime | None = None
    location: str | None = None


# ---------------------------------------------------------------------------
# User resource
# ---------------------------------------------------------------------------


class SCIMUser(BaseModel):
    """SCIM 2.0 User resource (RFC 7643 §4.1)."""

    schemas: list[str] = Field(default_factory=lambda: [SCHEMA_USER])
    id: str | None = None
    externalId: str | None = None
    userName: str
    name: SCIMName | None = None
    displayName: str | None = None
    emails: list[SCIMEmail] = Field(default_factory=list)
    active: bool = True
    meta: SCIMMeta | None = None

    model_config = ConfigDict(populate_by_name=True)


class SCIMUserCreate(BaseModel):
    """Request body for POST /scim/v2/Users."""

    schemas: list[str] = Field(default_factory=lambda: [SCHEMA_USER])
    userName: str
    name: SCIMName | None = None
    displayName: str | None = None
    emails: list[SCIMEmail] = Field(default_factory=list)
    active: bool = True
    externalId: str | None = None
    password: str | None = None


class SCIMUserReplace(SCIMUserCreate):
    """Request body for PUT /scim/v2/Users/{id} — full replacement."""


# ---------------------------------------------------------------------------
# PATCH Operation
# ---------------------------------------------------------------------------


class SCIMPatchOp(BaseModel):
    op: Literal["add", "replace", "remove"]
    path: str | None = None
    value: Any = None


class SCIMPatch(BaseModel):
    schemas: list[str] = Field(default_factory=lambda: [SCHEMA_PATCH_OP])
    Operations: list[SCIMPatchOp]


# ---------------------------------------------------------------------------
# List response
# ---------------------------------------------------------------------------


class SCIMListResponse(BaseModel):
    schemas: list[str] = Field(default_factory=lambda: [SCHEMA_LIST_RESPONSE])
    totalResults: int
    startIndex: int = 1
    itemsPerPage: int
    Resources: list[SCIMUser] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class SCIMError(BaseModel):
    schemas: list[str] = Field(default_factory=lambda: [SCHEMA_ERROR])
    status: str
    detail: str | None = None
    scimType: str | None = None


# ---------------------------------------------------------------------------
# ServiceProviderConfig
# ---------------------------------------------------------------------------


class _Supported(BaseModel):
    supported: bool


class _BulkConfig(BaseModel):
    supported: bool = False
    maxOperations: int = 0
    maxPayloadSize: int = 0


class _FilterConfig(BaseModel):
    supported: bool = True
    maxResults: int = 200


class _AuthScheme(BaseModel):
    type: str
    name: str
    description: str
    specUri: str | None = None
    documentationUri: str | None = None
    primary: bool = False


class SCIMServiceProviderConfig(BaseModel):
    schemas: list[str] = Field(default_factory=lambda: [SCHEMA_SERVICE_PROVIDER_CONFIG])
    patch: _Supported = Field(default_factory=lambda: _Supported(supported=True))
    bulk: _BulkConfig = Field(default_factory=_BulkConfig)
    filter: _FilterConfig = Field(default_factory=_FilterConfig)
    changePassword: _Supported = Field(default_factory=lambda: _Supported(supported=False))
    sort: _Supported = Field(default_factory=lambda: _Supported(supported=False))
    etag: _Supported = Field(default_factory=lambda: _Supported(supported=False))
    authenticationSchemes: list[_AuthScheme] = Field(
        default_factory=lambda: [
            _AuthScheme(
                type="oauthbearertoken",
                name="OAuth Bearer Token",
                description="Authentication scheme using the OAuth Bearer Token standard.",
                specUri="http://www.rfc-editor.org/info/rfc6750",
                primary=True,
            )
        ]
    )
    meta: SCIMMeta = Field(default_factory=lambda: SCIMMeta(resourceType="ServiceProviderConfig"))
