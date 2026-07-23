"""
Safety layer utilities for wrapping tools that require human confirmation.
Based on the provided SafeTool/CreateSafeTool concept.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from src.logging_config import get_logger

# LangChain imports — always available at type-check time so Pyright sees
# StructuredTool as a real class rather than ``None``.
if TYPE_CHECKING:
    from langchain_core.tools import StructuredTool as _StructuredToolType  # noqa: F401

try:
    from langchain_core.tools import StructuredTool
except ImportError:  # pragma: no cover
    StructuredTool = None  # type: ignore[misc, assignment]


class SafeTool(StructuredTool):  # type: ignore[misc]
    """StructuredTool with an extra flag indicating human confirmation is needed."""

    requires_confirmation: bool = False


def create_safe_tool(
    func: Callable,
    name: str,
    description: str,
    confirm: bool = False,
    args_schema=None,
) -> SafeTool:
    """
    Factory to create a tool that might require user approval.

    Args:
        func: Callable implementing the tool logic.
        name: Tool name exposed to the agent.
        description: Human/LLM-readable description.
        confirm: Whether the tool requires explicit user confirmation.
        args_schema: Optional Pydantic BaseModel schema for tool arguments.
    """
    log = get_logger()

    def implementation(*args, **kwargs):
        # Execution is handled by the agent; this wrapper only tags metadata.
        return func(*args, **kwargs)

    tool = SafeTool.from_function(
        func=implementation,
        name=name,
        description=description,
        args_schema=args_schema,
    )

    # Tag custom metadata for the executor to check.
    tool.metadata = {"requires_confirmation": confirm}  # type: ignore[attr-defined]
    tool.requires_confirmation = confirm  # type: ignore[attr-defined]

    if confirm:
        log.debug(f"Created safe tool: {name} (requires confirmation)")

    return tool  # type: ignore[return-value]
