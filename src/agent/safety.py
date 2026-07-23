"""
Safety layer utilities for wrapping tools that require human confirmation.
Based on the provided SafeTool/CreateSafeTool concept.
"""

from __future__ import annotations

import enum
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

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


@runtime_checkable
class ConfirmationUI(Protocol):
    """Abstract interface for tool confirmation rendering.

    CLI implementations use Rich panels; tests use stubs; assistant mode
    passes None (which means silent deny for unrecognized tools).
    """

    def render_prompt(
        self, tool_name: str, tool_input: dict, last_keys: frozenset[str], preview_limit: int
    ) -> None: ...

    def read_choice(self) -> str: ...

    def show_message(self, message: str, style: str) -> None: ...

    def pause_spinner(self) -> None: ...

    def resume_spinner(self) -> None: ...


class ConfirmationResult(enum.Enum):
    APPROVED_ONCE = "approved_once"
    APPROVED_SESSION = "approved_session"
    DENIED_ONCE = "denied_once"
    DENIED_DISABLE = "denied_disable"
    DENIED_ALL = "denied_all"
    CANCELLED = "cancelled"


class UserCancelledRun(Exception):
    """Raised when the user cancels the agent workflow from a tool prompt."""


LAST_KEYS: frozenset[str] = frozenset({"content", "body", "text", "code", "data"})

_confirmation_lock = threading.Lock()


def run_confirmation_prompt(
    tool_name: str,
    tool_input: dict,
    ui: ConfirmationUI,
    last_keys: frozenset[str] = LAST_KEYS,
    preview_limit: int = 300,
) -> ConfirmationResult:
    """Display a confirmation prompt via *ui* and return the user's decision."""
    ui.render_prompt(tool_name, tool_input, last_keys, preview_limit)
    choice = ui.read_choice().strip().lower()
    if choice in ("a", "all"):
        return ConfirmationResult.APPROVED_SESSION
    elif choice in ("y", "yes"):
        return ConfirmationResult.APPROVED_ONCE
    elif choice in ("f", "forbid-all"):
        return ConfirmationResult.DENIED_ALL
    elif choice in ("d", "disable"):
        return ConfirmationResult.DENIED_DISABLE
    elif choice in ("c", "cancel"):
        return ConfirmationResult.CANCELLED
    else:
        return ConfirmationResult.DENIED_ONCE


def create_safe_tool_wrapper(
    tool: Any,
    tool_name: str,
    registry: Any,
    approvals: set[str],
    session_state: Any | None = None,
    ui: ConfirmationUI | None = None,
) -> Any:
    """Wrap a tool to intercept execution and prompt for confirmation if needed.

    Returns a new tool that wraps the original.
    """
    try:
        from langchain_core.tools import StructuredTool as _ST
    except ImportError:
        return tool

    from src.orchestration.session_state import SessionState

    ss = session_state if session_state is not None else SessionState()
    original_func = tool.func if hasattr(tool, "func") else tool._run

    def safe_wrapper(*args, **kwargs):
        if registry.requires_confirmation(tool_name):
            if not ss.no_confirm:
                try:
                    if ui is not None:
                        ui.pause_spinner()
                    with _confirmation_lock:
                        if ss.deny_all or tool_name in ss.denials:
                            return "User denied execution"
                        if tool_name in approvals:
                            pass
                        else:
                            if ui is None:
                                return "User denied execution"

                            if kwargs:
                                tool_input = kwargs
                            else:
                                tool_input = args[0] if args else {}

                            result = run_confirmation_prompt(tool_name, tool_input, ui)

                            if result == ConfirmationResult.APPROVED_SESSION:
                                approvals.add(tool_name)
                                ui.show_message(
                                    f"✓ Approved '{tool_name}' for this session", "green"
                                )
                            elif result == ConfirmationResult.APPROVED_ONCE:
                                pass
                            elif result == ConfirmationResult.DENIED_ALL:
                                ss.deny_all = True
                                ui.show_message("✗ All tool requests will be forbidden", "red")
                                return "User denied execution"
                            elif result == ConfirmationResult.DENIED_DISABLE:
                                ss.denials.add(tool_name)
                                ui.show_message(
                                    f"✗ Tool '{tool_name}' disabled for this session", "red"
                                )
                                return "User denied execution"
                            elif result == ConfirmationResult.CANCELLED:
                                raise UserCancelledRun()
                            else:
                                ui.show_message("✗ Execution denied by user", "red")
                                return "User denied execution"
                finally:
                    if ui is not None:
                        ui.resume_spinner()

        try:
            return original_func(*args, **kwargs)
        except Exception as e:
            return f"Tool execution error: {e}"

    return _ST.from_function(
        func=safe_wrapper,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema if hasattr(tool, "args_schema") else None,
    )
