"""Session-constant agent execution configuration dataclass.

.. note::
    AgentRunConfig has been moved to ``cogtrix_core/common/types.py`` to break the
    bidirectional dependency between ``cogtrix_core/agent/`` and ``cogtrix_core/orchestration/``.
    This module re-exports the class for backward compatibility.
"""

from cogtrix_core.common.types import AgentRunConfig, ExecutionSettings

__all__ = ["AgentRunConfig", "ExecutionSettings"]
