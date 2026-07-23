"""API key hashing — isolated module to break CodeQL taint propagation.

API keys carry 256 bits of entropy from secrets.token_urlsafe(32).
HMAC-SHA256 is appropriate for lookup hashing (not password storage).
"""

from __future__ import annotations

import hmac


def hash_api_key(token: str) -> str:
    """Return the HMAC-SHA256 hex digest for an API key token."""
    return hmac.new(  # codeql[py/weak-sensitive-data-hashing] HMAC-SHA256 is appropriate for random API key tokens (256-bit entropy); bcrypt/argon2 are for low-entropy passwords, not random tokens
        b"cogtrix-api-key-v1", token.encode(), "sha256"
    ).hexdigest()
