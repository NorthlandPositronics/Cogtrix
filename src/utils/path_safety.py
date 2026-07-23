"""Path safety utilities for Cogtrix.

This module provides utilities for safely converting arbitrary strings
into filesystem-safe path components.
"""

import re

_SESSION_ID_MAX_LEN = 200  # Match src/memory/manager.py: keep consistent


def _sanitize_session_id(session_id: str) -> str:
    """Sanitize a session ID for safe use as a filesystem path component.

    Uses percent-encoding for non-safe characters to ensure bijectivity
    (distinct session IDs always produce distinct sanitized IDs).

    Args:
        session_id: The raw session ID string to sanitize.

    Returns:
        A filesystem-safe string that uniquely represents the original session ID.
    """
    if not session_id:
        return "default"

    # Encode anything that isn't alphanumeric, dot, hyphen, or underscore
    sanitized = re.sub(
        r"[^a-zA-Z0-9._-]",
        lambda m: f"%{ord(m.group()):02X}",
        session_id,
    )

    # Still prevent directory traversal via sequences like "..".
    sanitized = sanitized.replace("..", "%2E%2E")

    if len(sanitized) > _SESSION_ID_MAX_LEN:
        sanitized = sanitized[:_SESSION_ID_MAX_LEN]
        # Don't split a percent-encoded triplet (e.g. %2E)
        sanitized = re.sub(r"%[0-9A-Fa-f]?$", "", sanitized)

    if not sanitized:
        return "default"

    return sanitized
