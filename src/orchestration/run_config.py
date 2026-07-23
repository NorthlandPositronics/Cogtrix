"""Session-constant agent execution configuration dataclass.

.. note::
    AgentRunConfig has been moved to ``src/common/types.py`` to break the
    bidirectional dependency between ``src/agent/`` and ``src/orchestration/``.
    This module re-exports the class for backward compatibility.
"""

from src.common.types import AgentRunConfig, ExecutionSettings

__all__ = ["AgentRunConfig", "ExecutionSettings"]
