"""Shared Pydantic field validators for API schemas."""

from __future__ import annotations

import re


def validate_password_complexity(v: str) -> str:
    """Enforce password complexity: lowercase, uppercase, digit, special char."""
    if not re.search(r"[a-z]", v):
        raise ValueError("Password must contain at least one lowercase letter.")
    if not re.search(r"[A-Z]", v):
        raise ValueError("Password must contain at least one uppercase letter.")
    if not re.search(r"[0-9]", v):
        raise ValueError("Password must contain at least one digit.")
    if not re.search(r"[^a-zA-Z0-9]", v):
        raise ValueError("Password must contain at least one special character.")
    return v
