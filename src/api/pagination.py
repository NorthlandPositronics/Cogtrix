"""Shared cursor-based pagination helpers for the Cogtrix API."""

from __future__ import annotations

import base64


def encode_cursor(value: str) -> str:
    """Encode a string value as an opaque base64url cursor."""
    return base64.urlsafe_b64encode(value.encode()).decode()


def decode_cursor(cursor: str) -> str:
    """Decode a base64url cursor back to the original value.

    Restores missing base64 padding before decoding so that unpadded cursors
    (produced by some clients that strip trailing ``=``) are accepted.

    Raises ValueError on malformed input.
    """
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        return base64.urlsafe_b64decode(padded.encode()).decode()
    except Exception as exc:
        raise ValueError(f"Malformed cursor: {exc}") from exc


def paginate_list(
    items: list,
    cursor: str | None,
    limit: int,
) -> tuple[list, str | None, bool]:
    """Paginate an in-memory list using cursor-based pagination.

    cursor is the raw (decoded) start-after value, matched by name attribute or
    string equality. Returns (page_items, next_cursor_encoded, has_more).
    """
    limit = max(1, min(limit, 500))
    start = 0
    if cursor is not None:
        for i, item in enumerate(items):
            key = item if isinstance(item, str) else getattr(item, "name", str(i))
            if key == cursor:
                start = i + 1
                break
    page = items[start : start + limit]
    has_more = (start + limit) < len(items)
    next_cursor: str | None = None
    if has_more and page:
        last = page[-1]
        last_key = last if isinstance(last, str) else getattr(last, "name", "")
        next_cursor = encode_cursor(last_key)
    return page, next_cursor, has_more
