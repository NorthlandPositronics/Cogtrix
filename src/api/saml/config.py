"""SAML 2.0 Service Provider configuration (Enterprise Phase 1 — task 1.2.1).

Configuration is provided via the ``services.saml`` section in ``.cogtrix.yaml``
or via the ``SAMLConfig`` dataclass directly.

Requires the ``[saml]`` optional extra::

    pip install cogtrix[saml]

System prerequisites::

    apt-get install libxmlsec1-dev libxml2-dev pkg-config
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("cogtrix.api.saml")

_lock = threading.Lock()
_saml_config: SAMLConfig | None = None


@dataclass
class SAMLIdPConfig:
    """Identity Provider settings for a single SAML IdP."""

    entity_id: str
    """IdP entityID URI."""
    sso_url: str
    """IdP Single Sign-On URL (HTTP-Redirect or HTTP-POST binding)."""
    certificate: str
    """IdP signing certificate (PEM, without headers)."""
    slo_url: str | None = field(default=None)
    """Optional IdP Single Logout URL."""


@dataclass
class SAMLConfig:
    """SAML 2.0 Service Provider configuration.

    Attributes:
        sp_entity_id:   SP entityID URI (typically the metadata URL).
        sp_acs_url:     Assertion Consumer Service URL for SAMLResponse POST.
        sp_certificate: SP signing certificate PEM (without headers). Optional.
        sp_private_key: SP private key PEM (without headers). Optional.
        idp:            Identity Provider settings.
        name_id_format: NameID format requested of the IdP.
        attribute_map:  Map IdP attribute names → Cogtrix field names.
                        Keys: ``"email"``, ``"username"``, ``"role"``.
        default_role:   Role assigned when ``attribute_map["role"]`` is absent.
        org_id:         Organization all SAML-authenticated users are assigned to.
        scim_base_url:  Trusted base URL used by SCIM responses. When set, SCIM
                        `meta.location` values are derived from this URL instead
                        of request headers.
    """

    sp_entity_id: str
    sp_acs_url: str
    idp: SAMLIdPConfig
    sp_certificate: str = field(default="")
    sp_private_key: str = field(default="")
    name_id_format: str = field(default="urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress")
    attribute_map: dict[str, str] = field(
        default_factory=lambda: {
            "email": "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress",
            "username": "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name",
        }
    )
    default_role: str = field(default="user")
    org_id: str | None = field(default=None)
    scim_base_url: str | None = field(default=None)

    def to_python3_saml_settings(self) -> dict[str, Any]:
        """Return the settings dict expected by ``python3-saml``."""
        settings: dict[str, Any] = {
            "strict": True,
            "debug": False,
            # Require signed assertions and messages; reject weak algorithms.
            # python3-saml defaults wantAssertionsSigned/wantMessagesSigned to
            # False — strict:True alone does NOT enforce signatures.
            "security": {
                "wantAssertionsSigned": True,
                "wantMessagesSigned": True,
                "signatureAlgorithm": "http://www.w3.org/2001/04/xmldsig-more#rsa-sha256",
                "digestAlgorithm": "http://www.w3.org/2001/04/xmlenc#sha256",
                "rejectDeprecatedAlgorithm": True,
                "wantNameId": True,
            },
            "sp": {
                "entityId": self.sp_entity_id,
                "assertionConsumerService": {
                    "url": self.sp_acs_url,
                    "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST",
                },
                "NameIDFormat": self.name_id_format,
            },
            "idp": {
                "entityId": self.idp.entity_id,
                "singleSignOnService": {
                    "url": self.idp.sso_url,
                    "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect",
                },
                "x509cert": self.idp.certificate,
            },
        }
        if self.sp_certificate and self.sp_private_key:
            settings["sp"]["x509cert"] = self.sp_certificate
            settings["sp"]["privateKey"] = self.sp_private_key
        if self.idp.slo_url:
            settings["idp"]["singleLogoutService"] = {
                "url": self.idp.slo_url,
                "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect",
            }
        return settings


def configure_saml(config: SAMLConfig) -> None:
    """Register the SAML SP configuration. Thread-safe; replaces any prior config."""
    global _saml_config
    with _lock:
        _saml_config = config
    log.info("SAML SP configured: entity_id=%s acs=%s", config.sp_entity_id, config.sp_acs_url)


def get_saml_config() -> SAMLConfig | None:
    """Return the active SAML SP configuration, or ``None`` if not configured."""
    with _lock:
        return _saml_config


def is_saml_configured() -> bool:
    """Return True when a SAML SP configuration has been registered."""
    with _lock:
        return _saml_config is not None
