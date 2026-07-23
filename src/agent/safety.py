"""
Safety layer utilities for wrapping tools that require human confirmation.
Based on the provided SafeTool/CreateSafeTool concept.
"""

from __future__ import annotations

import difflib
import enum
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from src.logging_config import get_logger

log = get_logger()

try:
    from src.ui.spinner import _spinner as _activity_spinner
except Exception:
    _activity_spinner = None  # type: ignore[assignment]

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
        log.debug("Created safe tool: %s (requires confirmation)", name)

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

    def show_diff_preview(self, path: str, diff_lines: list[str]) -> None: ...

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

_DIFF_TOOL_NAMES = frozenset({"write_file", "patch_file"})
_GIT_WRITE_TOOLS = _DIFF_TOOL_NAMES  # tools eligible for git-native auto-commit


def _compute_file_diff(tool_name: str, tool_input: dict) -> tuple[str, list[str]] | None:
    """Compute unified diff for write_file or patch_file calls.

    Returns (file_path, diff_lines) or None if diff cannot be computed.
    diff_lines uses lineterm="" so each line has no trailing newline.
    """
    try:
        if tool_name == "write_file":
            path_str = tool_input.get("path", "")
            new_content = tool_input.get("content", "")
            if not path_str:
                return None
            path = Path(path_str)
            if not path.is_absolute():
                path = Path.cwd() / path_str
            old_content = (
                path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
            )
            old_lines = old_content.splitlines(keepends=True)
            new_lines = new_content.splitlines(keepends=True)
            diff = list(
                difflib.unified_diff(
                    old_lines,
                    new_lines,
                    fromfile=f"a/{path_str}",
                    tofile=f"b/{path_str}",
                    lineterm="",
                )
            )
            return (path_str, diff) if diff else None

        elif tool_name == "patch_file":
            path_str = tool_input.get("path", "")
            old_str = tool_input.get("old_str", "")
            new_str = tool_input.get("new_str", "")
            if not path_str:
                return None
            path = Path(path_str)
            if not path.is_absolute():
                path = Path.cwd() / path_str
            if not path.exists():
                return None
            old_content = path.read_text(encoding="utf-8", errors="replace")
            if old_content.count(old_str) != 1:
                return None  # ambiguous or missing — let patch_file handle the error
            new_content = old_content.replace(old_str, new_str, 1)
            old_lines = old_content.splitlines(keepends=True)
            new_lines = new_content.splitlines(keepends=True)
            diff = list(
                difflib.unified_diff(
                    old_lines,
                    new_lines,
                    fromfile=f"a/{path_str}",
                    tofile=f"b/{path_str}",
                    lineterm="",
                )
            )
            return (path_str, diff) if diff else None
    except Exception:
        return None
    return None


def run_confirmation_prompt(
    tool_name: str,
    tool_input: dict,
    ui: ConfirmationUI,
    last_keys: frozenset[str] = LAST_KEYS,
    preview_limit: int = 300,
) -> ConfirmationResult:
    """Display a confirmation prompt via *ui* and return the user's decision."""
    ui.render_prompt(tool_name, tool_input, last_keys, preview_limit)
    while True:
        choice = ui.read_choice().strip().lower()
        if choice in ("a", "all"):
            return ConfirmationResult.APPROVED_SESSION
        elif choice in ("y", "yes", ""):
            return ConfirmationResult.APPROVED_ONCE
        elif choice in ("n", "no"):
            return ConfirmationResult.DENIED_ONCE
        elif choice in ("f", "forbid-all", "forbid"):
            return ConfirmationResult.DENIED_ALL
        elif choice in ("d", "disable"):
            return ConfirmationResult.DENIED_DISABLE
        elif choice in ("c", "cancel"):
            return ConfirmationResult.CANCELLED
        else:
            ui.show_message(
                f"Invalid choice '{choice}'. Enter y/n/a/d/f/c.",
                "yellow",
            )


def _git_auto_commit(tool_name: str, tool_input: dict) -> None:
    """Stage and commit the written file when git-native mode is active.

    Fails silently if git is unavailable or the path is not in a repo.
    """
    try:
        from src.tools.git_tools import git_add, git_commit

        path_str = tool_input.get("path", "")
        if not path_str:
            return
        add_result = git_add([path_str])
        if "error" in add_result.lower() or "fatal" in add_result.lower():
            log.debug("git_auto_commit: git add failed: %s", add_result)
            return
        commit_result = git_commit(f"cogtrix: {tool_name} {path_str}")
        if "nothing to commit" in commit_result.lower():
            return
        log.debug("git_auto_commit: committed %s", path_str)
        if _activity_spinner is not None:
            try:
                _activity_spinner.set_context(f"git: committed {path_str}")
            except Exception:
                pass
    except Exception as e:
        log.debug("git_auto_commit: skipped (%s)", e)


def create_safe_tool_wrapper(
    tool: Any,
    tool_name: str,
    registry: Any,
    approvals: set[str],
    session_state: Any | None = None,
    ui: ConfirmationUI | None = None,
    git_native: bool = False,
    tool_trust: dict[str, str] | None = None,
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
    original_func = getattr(tool, "func", None) or getattr(tool, "invoke", None) or tool._run

    def safe_wrapper(*args, **kwargs):
        _trust = (tool_trust or {}).get(tool_name, "ask")
        if _trust == "deny":
            return "User denied execution"
        if registry.requires_confirmation(tool_name) and _trust != "always":
            if not ss.no_confirm:
                try:
                    if ui is not None:
                        try:
                            ui.pause_spinner()
                        except Exception:
                            pass
                    with _confirmation_lock:
                        if ss.deny_all or tool_name in ss.denials:
                            return "User denied execution"
                        if tool_name in approvals:
                            pass
                        else:
                            if ui is None:
                                log.warning(
                                    "Tool '%s' requires confirmation but no UI is available"
                                    " — denying silently",
                                    tool_name,
                                )
                                return "User denied execution"

                            if kwargs:
                                tool_input = kwargs
                            else:
                                tool_input = args[0] if args else {}

                            # Show diff preview for file-write tools
                            if tool_name in _DIFF_TOOL_NAMES:
                                diff_result = _compute_file_diff(tool_name, tool_input)
                                if diff_result is not None:
                                    _diff_path, _diff_lines = diff_result
                                    ui.show_diff_preview(_diff_path, _diff_lines)

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
                        try:
                            ui.resume_spinner()
                        except Exception:
                            pass

        try:
            if _activity_spinner is not None:
                _activity_spinner.set_context(tool_name)
            _result = original_func(*args, **kwargs)
            if (
                git_native
                and tool_name in _GIT_WRITE_TOOLS
                and isinstance(_result, str)
                and not _result.startswith("Tool execution error")
            ):
                _tool_input_for_git: dict = {}
                try:
                    if kwargs:
                        _tool_input_for_git = kwargs
                    elif args:
                        _tool_input_for_git = args[0] if isinstance(args[0], dict) else {}
                except Exception:
                    pass
                _git_auto_commit(tool_name, _tool_input_for_git)
            return _result
        except Exception as e:
            log.warning("Tool %s execution error: %s", tool_name, e, exc_info=True)
            return f"Tool execution error ({type(e).__name__}): {e}"
        finally:
            if _activity_spinner is not None:
                _activity_spinner.clear_context()

    return _ST.from_function(
        func=safe_wrapper,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema if hasattr(tool, "args_schema") else None,
    )
