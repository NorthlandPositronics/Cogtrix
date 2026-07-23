"""Database package — exports engine, Base, and get_db dependency.

``Base`` and ``get_db`` are imported eagerly (cheap — class definition and
function reference).  ``engine`` and ``AsyncSessionLocal`` are resolved
lazily via PEP 562 ``__getattr__`` so that importing this package does not
trigger config read, parent-dir ``mkdir``, or engine construction.
"""

from typing import Any

from cogtrix_core.api.db.engine import Base, get_db

# Eagerly-importable names; ``engine`` and ``AsyncSessionLocal`` are
# resolved lazily via ``__getattr__`` below.  Both remain accessible as
# ``from cogtrix_core.api.db import engine`` / ``from cogtrix_core.api.db import
# AsyncSessionLocal`` — only ``from cogtrix_core.api.db import *`` is affected.
__all__ = ["Base", "get_db"]


def __getattr__(name: str) -> Any:
    if name in ("engine", "AsyncSessionLocal"):
        from cogtrix_core.api.db import engine as _engine_mod

        return getattr(_engine_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
