"""Pure helpers for the Prometheus metrics endpoint.

Kept separate from ``metrics.py`` so they can be imported without
pulling in FastAPI, which keeps unit tests lightweight.
"""

from __future__ import annotations

import re

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_NUMERIC_ID_RE = re.compile(r"^\d+$")


def _normalize_path(path: str) -> str:
    """Normalize a request path for use as a Prometheus label.

    Replaces UUID and numeric path segments with ``{id}`` to prevent
    unbounded label cardinality.  Static path segments are preserved.
    """
    parts = path.split("/")
    out: list[str] = []
    for part in parts:
        if _UUID_RE.match(part) or _NUMERIC_ID_RE.match(part):
            out.append("{id}")
        else:
            out.append(part)
    return "/".join(out)
