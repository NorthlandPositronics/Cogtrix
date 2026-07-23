"""Session-scoped mutable state for a single Cogtrix session.

Consolidates the 7 module-level globals that were previously scattered
across cogtrix.py into a single dataclass, enabling proper session
isolation and eliminating the need for ``global`` declarations.

.. note::
    SessionState has been moved to ``src/common/types.py`` to break the
    bidirectional dependency between ``src/agent/`` and ``src/orchestration/``.
    This module re-exports the class for backward compatibility.
"""

from src.common.types import SessionState

__all__ = ["SessionState"]
