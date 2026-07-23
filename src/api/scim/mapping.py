"""Mapping between Cogtrix User ORM model and SCIM 2.0 User resource."""

from __future__ import annotations

import re
from typing import Any

from src.api.db.models import User
from src.api.scim.schemas import SCIMEmail, SCIMMeta, SCIMName, SCIMUser

_SCIM_BASE_PATH = "/scim/v2/Users"


def user_to_scim(user: User, base_url: str = "") -> SCIMUser:
    """Convert a Cogtrix ``User`` ORM object to a ``SCIMUser`` resource."""
    location = f"{base_url.rstrip('/')}{_SCIM_BASE_PATH}/{user.id}"
    emails = [SCIMEmail(value=user.email, primary=True, type="work")]
    name_parts = user.username.split(".", 1)
    given = name_parts[0].capitalize() if name_parts else user.username
    family = name_parts[1].capitalize() if len(name_parts) > 1 else ""
    active = user.is_active if getattr(user, "is_active", None) is not None else True

    return SCIMUser(
        id=user.id,
        userName=user.username,
        name=SCIMName(
            formatted=user.username,
            givenName=given,
            familyName=family,
        ),
        displayName=user.username,
        emails=emails,
        active=active,
        meta=SCIMMeta(
            resourceType="User",
            created=user.created_at,
            lastModified=user.created_at,
            location=location,
        ),
    )


# ---------------------------------------------------------------------------
# Simple SCIM filter parser
# ---------------------------------------------------------------------------

_FILTER_RE = re.compile(
    r'(?P<attr>[\w.]+)\s+(?P<op>eq|ne|co|sw|ew|pr|gt|ge|lt|le)\s+"?(?P<value>[^"]*)"?',
    re.IGNORECASE,
)


def parse_scim_filter(filter_str: str | None) -> dict[str, Any] | list[dict[str, Any]] | None:
    """Parse a SCIM filter string into a query dict or list of dicts.

    Supports simple filters:
    - ``userName eq "alice"``
    - ``emails.value eq "alice@example.com"``
    - ``active eq "true"``

    And compound ``and`` filters:
    - ``userName eq "alice" and active eq true``

    Returns:
        - ``None`` if the filter string is absent or cannot be parsed.
        - A single dict with keys ``attr``, ``op``, ``value`` for simple filters.
        - A list of dicts for compound ``and`` filters.
    """
    if not filter_str:
        return None

    stripped = filter_str.strip()

    # Handle compound "and" filters (case-insensitive, not inside quotes)
    if " and " in stripped.lower():
        clauses = _split_and_clauses(stripped)
        parsed_clauses = []
        for clause in clauses:
            m = _FILTER_RE.fullmatch(clause)
            if m is None:
                return None
            parsed_clauses.append(
                {
                    "attr": m.group("attr").lower(),
                    "op": m.group("op").lower(),
                    "value": m.group("value"),
                }
            )
        return parsed_clauses

    m = _FILTER_RE.fullmatch(stripped)
    if m is None:
        return None
    return {
        "attr": m.group("attr").lower(),
        "op": m.group("op").lower(),
        "value": m.group("value"),
    }


def _split_and_clauses(filter_str: str) -> list[str]:
    """Split a compound filter on unquoted ' and ' operators.

    Preserves ' and ' inside quoted values.
    """
    clauses: list[str] = []
    current = ""
    in_quotes = False
    i = 0
    lower_filter = filter_str.lower()
    while i < len(filter_str):
        ch = filter_str[i]
        if ch == '"':
            in_quotes = not in_quotes
            current += ch
            i += 1
            continue
        if not in_quotes and lower_filter[i : i + 5] == " and ":
            clauses.append(current.strip())
            current = ""
            i += 5
            continue
        current += ch
        i += 1
    if current.strip():
        clauses.append(current.strip())
    return clauses
