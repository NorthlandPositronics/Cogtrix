"""SAML 2.0 SP provider — wraps python3-saml for Cogtrix (Enterprise Phase 1 — task 1.2.1).

All methods that call into ``python3-saml`` raise ``ImportError`` (with a clear
message) when the ``[saml]`` extra is not installed.  The routes module catches
this and returns ``503 SAML_NOT_INSTALLED``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from cogtrix_core.api.saml.config import SAMLConfig

log = logging.getLogger("cogtrix.api.saml")


@dataclass
class SAMLAssertion:
    """Parsed, validated SAML 2.0 assertion from an IdP response.

    Attributes:
        name_id:    NameID from the assertion (typically the user's email).
        email:      Mapped email attribute, falls back to name_id.
        username:   Mapped username attribute, falls back to email local-part.
        attributes: Raw IdP attribute dict.
        session_index: SAML session index for SLO support.
        assertion_id: SAML assertion ID for replay protection (ID attribute).
    """

    name_id: str
    email: str
    username: str
    attributes: dict[str, list[str]]
    session_index: str | None = None
    assertion_id: str | None = None


def _require_saml2() -> Any:
    """Import and return the OneLogin_Saml2_Auth class, raising ImportError if absent."""
    try:
        from onelogin.saml2.auth import OneLogin_Saml2_Auth  # type: ignore[import]

        return OneLogin_Saml2_Auth
    except ImportError as exc:
        raise ImportError(
            "The [saml] optional extra is required for SAML 2.0 support. "
            "Install with: pip install cogtrix[saml]\n"
            "System prerequisites: libxmlsec1-dev libxml2-dev pkg-config"
        ) from exc


def build_saml_auth(
    config: SAMLConfig,
    request_data: dict[str, Any],
) -> Any:
    """Construct a ``OneLogin_Saml2_Auth`` instance for the given request.

    Args:
        config:       Active SAML SP configuration.
        request_data: Dict with keys ``http_host``, ``script_name``,
                      ``server_port``, ``get_data``, ``post_data``,
                      ``https`` (bool) — matches python3-saml's expected format.

    Returns:
        A ``OneLogin_Saml2_Auth`` instance ready for SSO or ACS operations.
    """
    OneLogin_Saml2_Auth = _require_saml2()
    return OneLogin_Saml2_Auth(request_data, old_settings=config.to_python3_saml_settings())


def get_metadata_xml(config: SAMLConfig) -> str:
    """Generate and return SP metadata XML.

    Raises:
        ImportError: when python3-saml is not installed.
        ValueError:  when the generated metadata fails validation.
    """
    try:
        from onelogin.saml2.settings import OneLogin_Saml2_Settings  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The [saml] optional extra is required. Install with: pip install cogtrix[saml]"
        ) from exc

    settings = OneLogin_Saml2_Settings(
        settings=config.to_python3_saml_settings(), sp_validation_only=True
    )
    metadata = settings.get_sp_metadata()
    errors = settings.validate_metadata(metadata)
    if errors:
        raise ValueError(f"Generated SAML metadata failed validation: {errors}")
    return metadata


def get_sso_redirect_url(config: SAMLConfig, request_data: dict[str, Any]) -> str:
    """Build the SSO redirect URL to send the user to the IdP.

    Args:
        config:       Active SAML SP configuration.
        request_data: Request context dict (see ``build_saml_auth``).

    Returns:
        The IdP SSO URL with the encoded SAMLRequest parameter.
    """
    auth = build_saml_auth(config, request_data)
    return auth.login()


def process_saml_response(
    config: SAMLConfig,
    request_data: dict[str, Any],
) -> SAMLAssertion:
    """Validate the SAMLResponse POST and return a parsed assertion.

    Args:
        config:       Active SAML SP configuration.
        request_data: Request context dict including ``post_data`` with
                      the raw ``SAMLResponse`` field.

    Returns:
        A ``SAMLAssertion`` with user identity and attribute data.

    Raises:
        ValueError:  when the SAML response is invalid, expired, or the
                     signature cannot be verified.
    """
    auth = build_saml_auth(config, request_data)
    auth.process_response()

    errors = auth.get_errors()
    if errors:
        last_error = auth.get_last_error_reason()
        log.warning("SAML ACS validation failed: %s — %s", errors, last_error)
        raise ValueError(f"Invalid SAML response: {errors} — {last_error}")

    if not auth.is_authenticated():
        raise ValueError("SAML authentication failed: user is not authenticated")

    name_id: str = auth.get_nameid() or ""
    assertion_id: str | None = auth.get_assertion_id()
    attributes: dict[str, list[str]] = auth.get_attributes()
    session_index: str | None = auth.get_session_index()

    # Map IdP attributes to Cogtrix fields using the configured attribute_map.
    attr_map = config.attribute_map
    email = _first_attr(attributes, attr_map.get("email", "")) or name_id
    username_raw = _first_attr(attributes, attr_map.get("username", "")) or email
    # Sanitize username: take the local part if it looks like an email.
    username = username_raw.split("@")[0] if "@" in username_raw else username_raw

    return SAMLAssertion(
        name_id=name_id,
        email=email,
        username=username,
        attributes=attributes,
        session_index=session_index,
        assertion_id=assertion_id,
    )


def _first_attr(attributes: dict[str, list[str]], key: str) -> str:
    """Return the first value for *key* in *attributes*, or empty string."""
    values = attributes.get(key, [])
    return values[0] if values else ""
