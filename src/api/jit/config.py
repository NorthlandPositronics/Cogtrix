"""Just-in-Time (JIT) user provisioning configuration (Enterprise Phase 1 — task 1.2.5).

JIT provisioning automatically creates a Cogtrix user account the first time
an identity (SAML assertion, OIDC claim, etc.) is presented — no pre-seeding
required.  The ``JITConfig`` controls which identities are allowed and what
account attributes they receive.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

log = logging.getLogger("cogtrix.api.jit")

_lock = threading.Lock()
_jit_config: JITConfig | None = None


@dataclass
class JITConfig:
    """Just-in-Time provisioning policy.

    Attributes:
        enabled:             Master switch.  When False, unknown users are
                             rejected at SSO/OIDC instead of auto-provisioned.
        allowed_domains:     Email domain allowlist.  Empty list means ALL
                             domains are accepted.  Example: ``["company.com"]``.
        default_role:        Role assigned to newly provisioned users.
        org_id:              Org to assign provisioned users to.  ``None``
                             falls back to the default org (slug='default').
        auto_team_id:        If set, newly provisioned users are automatically
                             added to this team (member role).
        max_users:           Maximum number of JIT-provisioned users allowed in
                             the org.  0 means unlimited.
        deactivate_unknown:  When True, users whose identity is not present in
                             the latest IdP sync are deactivated (not deleted).
    """

    enabled: bool = True
    allowed_domains: list[str] = field(default_factory=list)
    default_role: str = "user"
    org_id: str | None = None
    auto_team_id: str | None = None
    max_users: int = 0
    deactivate_unknown: bool = False

    def is_domain_allowed(self, email: str) -> bool:
        """Return True when *email*'s domain is permitted by the allowlist."""
        if not self.allowed_domains:
            return True
        try:
            domain = email.split("@", 1)[1].lower()
        except IndexError:
            return False
        return domain in {d.lower() for d in self.allowed_domains}


def configure_jit(config: JITConfig) -> None:
    """Register the JIT provisioning configuration.  Thread-safe."""
    global _jit_config
    with _lock:
        _jit_config = config
    log.info(
        "JIT provisioning configured: enabled=%s domains=%s",
        config.enabled,
        config.allowed_domains or "all",
    )


def get_jit_config() -> JITConfig | None:
    """Return the active JIT configuration, or ``None`` if not configured."""
    with _lock:
        return _jit_config


def is_jit_enabled() -> bool:
    """Return True when JIT provisioning is configured and enabled."""
    with _lock:
        return _jit_config is not None and _jit_config.enabled
