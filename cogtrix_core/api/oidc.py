"""OIDC/SSO token validation for the Cogtrix API.

Validates RS256/ES256 ID tokens issued by an external identity provider
(Keycloak, Auth0, Okta, Azure AD, Google, etc.).

Flow:
1. Discover JWKS URI via ``<issuer>/.well-known/openid-configuration`` (cached 1 h).
2. Fetch the JWKS and cache the key set (TTL 5 min).
3. Extract ``kid`` from the unverified token header, find the matching key.
4. Decode and verify: signature, expiry, issuer, audience.
5. Map the role claim (default ``roles``) to Cogtrix role strings; fall back to
   ``oidc_default_role`` when the claim is absent.

All network I/O uses ``urllib.request.urlopen`` with a 5-second socket timeout.
Call sites must invoke ``validate()`` via ``asyncio.to_thread`` so the event loop
is never blocked.

Thread safety: JWKS cache and discovery cache each have an independent
``threading.Lock``.  Neither lock is held during network I/O — concurrent cache
misses may result in duplicate fetches, but the outcome is always correct.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

import jwt
from jwt.exceptions import InvalidTokenError

log = logging.getLogger("cogtrix.api.oidc")

_JWKS_TTL = 300  # 5 minutes
_DISCOVERY_TTL = 3600  # 1 hour
_HTTP_TIMEOUT = 5  # seconds
_SECURE_SCHEME = "https"
_INSECURE_SCHEME = "http"


@dataclass
class OIDCConfig:
    """Configuration for OIDC token validation."""

    issuer: str
    audience: str
    jwks_uri: str | None = field(default=None)
    allow_insecure_oidc: bool = field(default=False)
    production_mode: bool = field(default=True)
    role_claim: str = field(default="roles")
    default_role: str = field(default="user")


class OIDCValidator:
    """Validates OIDC ID tokens against a remote JWKS endpoint.

    Instances should be created via ``configure_oidc()`` and accessed via
    ``get_validator()``.
    """

    def __init__(self, config: OIDCConfig) -> None:
        self._config = config
        self._jwks_lock = threading.Lock()
        self._discovery_lock = threading.Lock()

        # (PyJWKSet, fetched_at_monotonic)
        self._jwks_cache: tuple[jwt.PyJWKSet, float] | None = None

        # (jwks_uri_string, fetched_at_monotonic)
        self._discovery_cache: tuple[str, float] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, token: str) -> dict[str, Any]:
        """Decode and verify an OIDC ID token.

        Returns the decoded claims dict on success.

        Raises:
            jwt.ExpiredSignatureError — token is valid but expired.
            jwt.InvalidTokenError    — invalid signature, audience, issuer, etc.
            RuntimeError             — JWKS fetch or discovery failure.
        """
        header = jwt.get_unverified_header(token)
        kid: str | None = header.get("kid")

        signing_key = self._get_signing_key(kid)
        claims: dict[str, Any] = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256", "ES256"],
            audience=self._config.audience,
            issuer=self._config.issuer,
            options={"verify_exp": True},
        )
        return claims

    def map_role(self, claims: dict[str, Any]) -> str:
        """Map OIDC claims to a Cogtrix role string (``'admin'`` or ``'user'``).

        Reads ``config.role_claim`` from the token payload.  Accepts either a
        list of role strings or a single string.  Returns ``config.default_role``
        when the claim is absent or empty.
        """
        raw = claims.get(self._config.role_claim)
        if raw is None:
            return self._config.default_role
        if isinstance(raw, list):
            roles = [str(r).lower() for r in raw]
        else:
            roles = [str(raw).lower()]
        if "admin" in roles:
            return "admin"
        return roles[0] if roles else self._config.default_role

    # ------------------------------------------------------------------
    # JWKS resolution
    # ------------------------------------------------------------------

    def _get_signing_key(self, kid: str | None) -> jwt.PyJWK:
        """Return the signing key matching ``kid``, refreshing the cache if needed."""
        jwks_set = self._load_jwks()

        if kid is None:
            raise InvalidTokenError("kid header required")

        for k in jwks_set.keys:
            if k.key_id == kid:
                return k
        # kid not in cache — may be stale; force one refresh
        log.debug("kid %r not in cached JWKS — forcing refresh", kid)
        jwks_set = self._load_jwks(force=True)
        for k in jwks_set.keys:
            if k.key_id == kid:
                return k
        raise InvalidTokenError(f"No key with kid={kid!r} found in JWKS")

    def _load_jwks(self, *, force: bool = False) -> jwt.PyJWKSet:
        """Return the cached JWKS set, fetching from the network when expired."""
        now = time.monotonic()
        # Fast path: cache hit (lock held only for the read, not the fetch)
        if not force:
            with self._jwks_lock:
                if self._jwks_cache is not None and (now - self._jwks_cache[1]) < _JWKS_TTL:
                    return self._jwks_cache[0]

        # Cache miss or forced refresh — fetch without holding the lock
        jwks_uri = self._resolve_jwks_uri()
        jwks_set = self._fetch_jwks(jwks_uri)

        # Store the result
        with self._jwks_lock:
            self._jwks_cache = (jwks_set, time.monotonic())
        return jwks_set

    # ------------------------------------------------------------------
    # JWKS fetch
    # ------------------------------------------------------------------

    def _fetch_jwks(self, jwks_uri: str) -> jwt.PyJWKSet:
        """Fetch and parse a JWKS document from ``jwks_uri``."""
        self._validate_oidc_uri_scheme(jwks_uri, context="JWKS URI")
        log.debug("Fetching JWKS from %s", jwks_uri)
        try:
            with urlopen(jwks_uri, timeout=_HTTP_TIMEOUT) as resp:  # nosec B310
                body = resp.read()
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch JWKS from {jwks_uri}: {exc}") from exc
        try:
            data = json.loads(body)
            return jwt.PyJWKSet.from_dict(data)
        except Exception as exc:
            raise RuntimeError(f"Invalid JWKS response from {jwks_uri}: {exc}") from exc

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _resolve_jwks_uri(self) -> str:
        """Return the JWKS URI, discovering it from openid-configuration if needed."""
        if self._config.jwks_uri:
            return self._config.jwks_uri

        now = time.monotonic()
        with self._discovery_lock:
            if (
                self._discovery_cache is not None
                and (now - self._discovery_cache[1]) < _DISCOVERY_TTL
            ):
                return self._discovery_cache[0]

        # Fetch discovery document without holding the lock
        discovered = self.discover_jwks_uri(self._config.issuer)

        with self._discovery_lock:
            self._discovery_cache = (discovered, time.monotonic())
        return discovered

    def discover_jwks_uri(self, issuer: str) -> str:
        """Fetch ``<issuer>/.well-known/openid-configuration`` and return ``jwks_uri``.

        Raises ``RuntimeError`` on network failure or a malformed document.
        """
        well_known = issuer.rstrip("/") + "/.well-known/openid-configuration"
        self._validate_oidc_uri_scheme(issuer, context="OIDC issuer")
        log.debug("Discovering OIDC configuration from %s", well_known)
        try:
            with urlopen(well_known, timeout=_HTTP_TIMEOUT) as resp:  # nosec B310
                body = resp.read()
        except Exception as exc:
            raise RuntimeError(f"OIDC discovery failed for {issuer}: {exc}") from exc
        try:
            doc = json.loads(body)
        except Exception as exc:
            raise RuntimeError(f"Invalid OIDC discovery document from {issuer}: {exc}") from exc
        jwks_uri = doc.get("jwks_uri")
        if not jwks_uri:
            raise RuntimeError(f"OIDC discovery document missing 'jwks_uri' for {issuer}")
        return str(jwks_uri)

    def _validate_oidc_uri_scheme(self, uri: str, *, context: str) -> None:
        """Reject insecure or unsupported URL schemes for OIDC endpoints."""
        scheme = urlparse(uri).scheme.lower()
        if scheme == _SECURE_SCHEME:
            return
        if (
            scheme == _INSECURE_SCHEME
            and self._config.allow_insecure_oidc
            and not self._config.production_mode
        ):
            log.warning(
                "%s uses insecure HTTP because allow_insecure_oidc is enabled: %s", context, uri
            )
            return
        raise ValueError(f"{context} uses disallowed scheme '{scheme}': {uri}")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_validator: OIDCValidator | None = None
_validator_lock = threading.Lock()


def configure_oidc(config: OIDCConfig) -> None:
    """Install a new ``OIDCValidator`` from ``config``.

    Called during application startup when ``oidc_enabled`` is set in the
    Cogtrix config.  Thread-safe.
    """
    global _validator
    with _validator_lock:
        _validator = OIDCValidator(config)
    log.info("OIDC validator configured (issuer=%s)", config.issuer)


def get_validator() -> OIDCValidator | None:
    """Return the active ``OIDCValidator``, or ``None`` if OIDC is not configured."""
    return _validator
