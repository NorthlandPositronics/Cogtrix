"""Canonical error strings for path-policy violations across file tools.

Issue #1928 / #1919 (Finding 3): ``cogtrix_core/tools/file_ops.py`` and
``cogtrix_core/tools/generate_tests.py`` historically emitted at least three
distinct string shapes for the *same* logical failure — a path falling
outside the permitted area:

  * ``"Write path must be within the working directory"`` (file_ops.py)
  * ``"Path must be within the working directory"`` (file_ops.py +
    generate_tests.py)
  * ``"Error: Path outside allowed write paths: <path>"`` (legacy)

In ``.agent-test-1918/test1-gas`` the agent tried ``write_file(path=
"Code.gs")`` and ``write_file(path="cogtrix_core/api/EchoApi.gs")`` and got two
different error strings for what — from the agent's perspective — is
the same problem: the path is not under a writable area, pivot to
``/tmp``. The string drift made it harder for the model to recognise
the repeated mistake as a single class.

This module centralises the canonical strings. Each tool calls back
to the constants here so the dispatcher / agent see a *consistent*
surface regardless of which file tool produced the error.

What's NOT here: OS-level ``PermissionError`` outcomes from
``open()`` etc. (``"Error: Permission denied: <path>"``). Those are
genuinely a different class — the path was permitted by Cogtrix's
policy, but the OS denied the operation (file mode, mount option,
container UID drift, ...). Keeping them distinct lets the agent
disambiguate "wrong path" from "right path but no OS permission".
"""

from __future__ import annotations

#: Canonical message for a path outside the permitted *write* area.
#: Includes the offending path so the agent can see exactly what was
#: rejected — the prior wording ("Write path must be within the
#: working directory") gave no path context, which made it harder for
#: the recovery prompt to point at a sensible alternative.
ERR_WRITE_OUTSIDE_PERMITTED_AREA = (
    "Error: Path '{path}' is outside the permitted write area "
    "(working directory + COGTRIX_ALLOWED_WRITE_PATHS). "
    "Write under /tmp or one of the explicitly allowed write paths."
)

#: Canonical message for a path outside the permitted *read* area.
#: Same shape as the write variant — operator can extend the allowed
#: set via ``COGTRIX_ALLOWED_READ_PATHS``.
ERR_READ_OUTSIDE_PERMITTED_AREA = (
    "Error: Path '{path}' is outside the permitted read area "
    "(working directory + app install dir + COGTRIX_ALLOWED_READ_PATHS)."
)

#: Canonical message for a parent-directory-traversal attempt
#: (``../`` segments that escape the working directory).  Kept
#: distinct from the outside-permitted-area class because traversal
#: indicates intent (whether by the model or by an injected path),
#: while a plain outside-area path is often just an honest mistake.
ERR_PATH_TRAVERSAL = "Error: Path traversal not allowed: '{path}'"


def format_write_outside_error(path: object) -> str:
    """Return the canonical 'outside write area' error message for *path*."""
    return ERR_WRITE_OUTSIDE_PERMITTED_AREA.format(path=path)


def format_read_outside_error(path: object) -> str:
    """Return the canonical 'outside read area' error message for *path*."""
    return ERR_READ_OUTSIDE_PERMITTED_AREA.format(path=path)


def format_traversal_error(path: object) -> str:
    """Return the canonical 'path traversal not allowed' error message for *path*."""
    return ERR_PATH_TRAVERSAL.format(path=path)


def is_path_policy_error(message: str) -> bool:
    """Return True when *message* matches one of this module's canonical
    path-policy error shapes.

    Useful for downstream consumers (dispatcher diagnostics, test
    harness) that want to classify a tool error as
    'agent-recoverable path choice' vs 'environmental issue' without
    parsing the prose.  Anchors on the canonical prefixes — refactors
    inside the messages don't break the classifier as long as the
    fixed prefix tokens are preserved.
    """
    if not isinstance(message, str):
        return False
    return (
        "is outside the permitted write area" in message
        or "is outside the permitted read area" in message
        or "Path traversal not allowed" in message
    )


__all__ = [
    "ERR_WRITE_OUTSIDE_PERMITTED_AREA",
    "ERR_READ_OUTSIDE_PERMITTED_AREA",
    "ERR_PATH_TRAVERSAL",
    "format_write_outside_error",
    "format_read_outside_error",
    "format_traversal_error",
    "is_path_policy_error",
]
