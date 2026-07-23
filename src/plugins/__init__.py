"""Cogtrix plugin system — public API.

Third-party packages that provide tools should import ``hookimpl`` from here,
implement ``cogtrix_tools()``, and declare an entry-point in the
``cogtrix.tools`` group.  See ``docs/TOOLS_AUTHORING.md`` for a full guide.
"""

from __future__ import annotations

from src.plugins.spec import hookimpl  # noqa: F401

__all__ = ["hookimpl"]
