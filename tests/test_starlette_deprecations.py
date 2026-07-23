"""Guard against re-introducing deprecated Starlette status constants.

Starlette aligned its status constants with RFC 9110.  The old names are
still present as int aliases (so existing imports do not break), but they
emit ``DeprecationWarning`` on access.  This module fails CI if any
production code under ``src/`` references the deprecated spellings.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# (deprecated, replacement) — used for both detection and error messaging.
_RENAMES: tuple[tuple[str, str], ...] = (
    ("HTTP_422_UNPROCESSABLE_ENTITY", "HTTP_422_UNPROCESSABLE_CONTENT"),
    ("HTTP_413_REQUEST_ENTITY_TOO_LARGE", "HTTP_413_CONTENT_TOO_LARGE"),
)

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src"


@pytest.mark.parametrize("deprecated,replacement", _RENAMES)
def test_no_deprecated_starlette_constants(deprecated: str, replacement: str) -> None:
    """No file under ``src/`` may reference the deprecated constant name."""
    pattern = re.compile(rf"\b{re.escape(deprecated)}\b")
    offenders: list[str] = []
    for py_file in _SRC_ROOT.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                offenders.append(f"{py_file.relative_to(_SRC_ROOT.parent)}:{lineno}")
    assert (
        not offenders
    ), f"Use {replacement} instead of {deprecated}. Offenders:\n  " + "\n  ".join(offenders)
