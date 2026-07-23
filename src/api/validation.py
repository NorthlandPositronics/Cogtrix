"""Validation error translation layer.

Transforms Pydantic's internal error format into a flat, field-keyed structure
with stable error codes and human-readable messages.  This module is the sole
translation boundary between Pydantic and the public API contract — no Pydantic
error internals leak past ``translate_validation_errors()``.

Registry keying: ``FIELD_MESSAGES`` uses ``(field_name, pydantic_type)`` tuples.
If two schemas share a field name with different constraints (e.g. ``top_k`` in
``RAGSearchRequest`` vs ``KnowledgeSearchRequest``), the fallback builder extracts
actual limits from ``err["ctx"]`` and produces a correct, schema-specific message.
Escalate to a three-part key ``(schema_class, field_name, pydantic_type)`` only if
the fallback message is unacceptable for a specific collision — that case does not
currently exist.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Stable error code mapping
# ---------------------------------------------------------------------------

_CODE_MAP: dict[str, str] = {
    # Length constraints
    "string_too_short": "TOO_SHORT",
    "too_short": "TOO_SHORT",
    "string_too_long": "TOO_LONG",
    "too_long": "TOO_LONG",
    # Format constraints
    "string_pattern_mismatch": "INVALID_FORMAT",
    # Value constraints
    "value_error": "INVALID_VALUE",
    # Required fields
    "missing": "REQUIRED",
    # Range constraints
    "greater_than_equal": "OUT_OF_RANGE",
    "less_than_equal": "OUT_OF_RANGE",
    "greater_than": "OUT_OF_RANGE",
    "less_than": "OUT_OF_RANGE",
    # Enum / Literal
    "literal_error": "INVALID_CHOICE",
    "enum": "INVALID_CHOICE",
    # JSON parsing
    "json_invalid": "INVALID_JSON",
    "json_type": "INVALID_JSON",
    # Type mismatches
    "int_parsing": "TYPE_MISMATCH",
    "int_type": "TYPE_MISMATCH",
    "float_parsing": "TYPE_MISMATCH",
    "float_type": "TYPE_MISMATCH",
    "string_type": "TYPE_MISMATCH",
    "bool_type": "TYPE_MISMATCH",
    "bool_parsing": "TYPE_MISMATCH",
    "list_type": "TYPE_MISMATCH",
    "dict_type": "TYPE_MISMATCH",
}

# ---------------------------------------------------------------------------
# Human-readable message overrides
# ---------------------------------------------------------------------------

FIELD_MESSAGES: dict[tuple[str, str], str] = {
    # Auth — RegisterRequest
    ("username", "string_pattern_mismatch"): (
        "Username may only contain letters, digits, hyphens, and underscores."
    ),
    ("username", "string_too_short"): "Username must be at least 3 characters.",
    ("username", "string_too_long"): "Username must be at most 64 characters.",
    ("password", "string_too_short"): "Password must be at least 8 characters.",
    ("password", "string_too_long"): "Password must be at most 128 characters.",
    ("email", "value_error"): "Please enter a valid email address.",
    # Message — SendMessageRequest
    ("content", "string_too_short"): "Message cannot be empty.",
    ("content", "string_too_long"): "Message exceeds the 65,536 character limit.",
    # RAG / Knowledge — search requests
    ("query", "string_too_short"): "Search query cannot be empty.",
    ("query", "string_too_long"): "Search query exceeds the maximum length.",
    # Session
    ("name", "string_too_long"): "Name must be at most 256 characters.",
    ("system_prompt", "string_too_long"): "System prompt exceeds the 32,768 character limit.",
    # Assistant
    ("label", "string_too_long"): "Label must be at most 128 characters.",
    ("text", "string_too_long"): "Text must be at most 4,096 characters.",
}

# Structural prefixes in Pydantic ``loc`` tuples that should be stripped.
_LOC_PREFIXES = frozenset({"body", "query", "path", "header"})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_field_path(loc: list | tuple) -> list[str]:
    """Extract the meaningful field path from a Pydantic ``loc`` tuple.

    Strips structural prefixes (``body``, ``query``, etc.) and returns the
    remaining segments as strings.  Integer indices (from list items) are
    converted to strings.
    """
    parts: list[str] = []
    for segment in loc:
        s = str(segment)
        if s in _LOC_PREFIXES and not parts:
            continue
        parts.append(s)
    return parts or ["_root"]


def _humanize_name(name: str) -> str:
    """Convert a snake_case field name to a human-readable label."""
    return name.replace("_", " ").capitalize()


def _build_fallback_message(field_name: str, err: dict) -> str:
    """Build a reasonable fallback message when no registry entry exists."""
    label = _humanize_name(field_name)
    err_type: str = err.get("type", "")
    ctx: dict = err.get("ctx", {})

    # Length constraints — extract actual limits
    if err_type == "string_too_short":
        limit = ctx.get("min_length", "")
        if limit:
            return f"{label} must be at least {limit} characters."
        return f"{label} is too short."

    if err_type == "string_too_long":
        limit = ctx.get("max_length", "")
        if limit:
            return f"{label} must be at most {limit:,} characters."
        return f"{label} is too long."

    # Pattern — don't expose the regex
    if err_type == "string_pattern_mismatch":
        return f"{label} has an invalid format."

    # Range constraints
    if err_type == "greater_than_equal":
        limit = ctx.get("ge", "")
        return f"{label} must be at least {limit}." if limit != "" else f"{label} is too small."

    if err_type == "less_than_equal":
        limit = ctx.get("le", "")
        return f"{label} must be at most {limit}." if limit != "" else f"{label} is too large."

    if err_type == "greater_than":
        limit = ctx.get("gt", "")
        return f"{label} must be greater than {limit}." if limit != "" else f"{label} is too small."

    if err_type == "less_than":
        limit = ctx.get("lt", "")
        return f"{label} must be less than {limit}." if limit != "" else f"{label} is too large."

    # Literal / enum — list allowed values
    if err_type == "literal_error":
        expected = ctx.get("expected", "")
        if expected:
            return f"Must be one of: {expected}."
        return f"{label} is not a valid choice."

    # Missing / required
    if err_type == "missing":
        return f"{label} is required."

    # Type mismatch
    if err_type in (
        "int_parsing",
        "int_type",
        "float_parsing",
        "float_type",
    ):
        return f"{label} must be a number."
    if err_type in ("bool_type", "bool_parsing"):
        return f"{label} must be true or false."
    if err_type == "list_type":
        return f"{label} must be a list."
    if err_type in ("string_type",):
        return f"{label} must be a string."

    # Value error (e.g. email validation)
    if err_type == "value_error":
        msg = err.get("msg", "")
        if msg:
            # Humanize: "Value is not a valid email address" → "Email is not..."
            for prefix in ("Value ", "String ", "Input "):
                if msg.startswith(prefix):
                    return f"{label} {msg[len(prefix):]}"
        return f"{label} is invalid."

    # Generic fallback — humanize the raw message
    msg = err.get("msg", "Validation failed.")
    for prefix in ("Value ", "String ", "Input "):
        if msg.startswith(prefix):
            return f"{label} {msg[len(prefix):]}"
    return msg


def _set_nested(target: dict, path: list[str], value: dict) -> None:
    """Insert ``value`` into ``target`` at the nested ``path``.

    Single-segment: ``target["username"] = [value, ...]``
    Multi-segment:  ``target["config"]["max_steps"] = [value, ...]``

    Appends to existing lists if the leaf already exists.
    """
    for i, key in enumerate(path):
        if i == len(path) - 1:
            # Leaf — append to list
            if key not in target:
                target[key] = []
            if isinstance(target[key], list):
                target[key].append(value)
        else:
            # Intermediate — ensure dict exists
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
            target = target[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def translate_validation_errors(errors: list[dict[str, Any]]) -> dict[str, Any]:
    """Translate Pydantic validation errors to the public API format.

    Returns a dict ``{"fields": {...}}`` ready for ``APIError.details``.
    """
    fields: dict = {}

    for err in errors:
        loc = err.get("loc", ())
        err_type: str = err.get("type", "")

        path = _extract_field_path(loc)
        field_name = path[-1]

        # Stable error code
        code = _CODE_MAP.get(err_type, "INVALID")

        # Human-readable message: registry first, then fallback
        message = FIELD_MESSAGES.get((field_name, err_type))
        if message is None:
            message = _build_fallback_message(field_name, err)

        _set_nested(fields, path, {"code": code, "message": message})

    return {"fields": fields}
