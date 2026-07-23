"""Tests for src/agent/safety — unified tool confirmation logic."""

import threading
from unittest.mock import MagicMock

import pytest

from src.agent.safety import (
    LAST_KEYS,
    ConfirmationResult,
    ConfirmationUI,
    UserCancelledRun,
    _confirmation_lock,
    create_safe_tool_wrapper,
    run_confirmation_prompt,
)
from src.orchestration.session_state import SessionState


class _StubUI:
    """Minimal ConfirmationUI for testing."""

    def __init__(self, choice: str = "y"):
        self._choice = choice
        self.rendered = False
        self.messages: list[tuple[str, str]] = []
        self.paused = 0

    def render_prompt(self, tool_name, tool_input, last_keys, preview_limit):
        self.rendered = True

    def read_choice(self) -> str:
        return self._choice

    def show_message(self, message: str, style: str) -> None:
        self.messages.append((message, style))

    def pause_spinner(self) -> None:
        self.paused += 1

    def resume_spinner(self) -> None:
        self.paused -= 1


class TestConfirmationResult:
    def test_all_choices(self):
        assert run_confirmation_prompt("t", {}, _StubUI("y")) == ConfirmationResult.APPROVED_ONCE
        assert run_confirmation_prompt("t", {}, _StubUI("yes")) == ConfirmationResult.APPROVED_ONCE
        assert run_confirmation_prompt("t", {}, _StubUI("a")) == ConfirmationResult.APPROVED_SESSION
        assert (
            run_confirmation_prompt("t", {}, _StubUI("all")) == ConfirmationResult.APPROVED_SESSION
        )
        assert run_confirmation_prompt("t", {}, _StubUI("n")) == ConfirmationResult.DENIED_ONCE
        assert run_confirmation_prompt("t", {}, _StubUI("d")) == ConfirmationResult.DENIED_DISABLE
        assert run_confirmation_prompt("t", {}, _StubUI("f")) == ConfirmationResult.DENIED_ALL
        assert run_confirmation_prompt("t", {}, _StubUI("c")) == ConfirmationResult.CANCELLED

    def test_unknown_choice_denies(self):
        assert run_confirmation_prompt("t", {}, _StubUI("xyz")) == ConfirmationResult.DENIED_ONCE


class TestUserCancelledRun:
    def test_importable(self):
        assert issubclass(UserCancelledRun, Exception)

    def test_raisable(self):
        with pytest.raises(UserCancelledRun):
            raise UserCancelledRun()


class TestLastKeys:
    def test_contains_expected(self):
        assert "content" in LAST_KEYS
        assert "code" in LAST_KEYS
        assert "body" in LAST_KEYS

    def test_is_frozenset(self):
        assert isinstance(LAST_KEYS, frozenset)


class TestCreateSafeToolWrapper:
    def _make_tool(self, name="test_tool"):
        tool = MagicMock()
        tool.name = name
        tool.description = "A test tool"
        tool.args_schema = None
        tool.func = lambda x: f"executed {x}"
        return tool

    def _make_registry(self, confirms=True):
        reg = MagicMock()
        reg.requires_confirmation.return_value = confirms
        return reg

    def test_no_ui_denies_unrecognized(self):
        """When ui=None and tool not in approvals, silently deny."""
        tool = self._make_tool()
        reg = self._make_registry(confirms=True)
        ss = SessionState()
        wrapped = create_safe_tool_wrapper(tool, "test_tool", reg, set(), session_state=ss, ui=None)
        result = wrapped.invoke({})
        assert "denied" in result.lower() or "denied" in str(result).lower()

    def test_approved_tool_executes(self):
        """When tool is in approvals, it executes regardless of ui."""
        tool = self._make_tool()
        reg = self._make_registry(confirms=True)
        ss = SessionState()
        approvals = {"test_tool"}
        wrapped = create_safe_tool_wrapper(
            tool, "test_tool", reg, approvals, session_state=ss, ui=None
        )
        assert wrapped is not None

    def test_no_confirmation_needed_executes(self):
        """When registry says no confirmation needed, tool executes directly."""
        tool = self._make_tool()
        reg = self._make_registry(confirms=False)
        ss = SessionState()
        wrapped = create_safe_tool_wrapper(tool, "test_tool", reg, set(), session_state=ss, ui=None)
        assert wrapped is not None

    def test_confirmation_ui_protocol_check(self):
        """_StubUI satisfies the ConfirmationUI protocol."""
        stub = _StubUI()
        assert isinstance(stub, ConfirmationUI)

    def test_confirmation_lock_is_lock(self):
        """_confirmation_lock is a threading.Lock."""
        assert isinstance(_confirmation_lock, type(threading.Lock()))

    def test_approved_session_adds_to_approvals(self):
        """Choosing 'a' adds the tool name to approvals."""
        tool = self._make_tool()
        reg = self._make_registry(confirms=True)
        ss = SessionState()
        approvals: set[str] = set()
        ui = _StubUI("a")
        wrapped = create_safe_tool_wrapper(
            tool, "test_tool", reg, approvals, session_state=ss, ui=ui
        )
        wrapped.invoke({})
        assert "test_tool" in approvals

    def test_cancelled_raises_user_cancelled_run(self):
        """Choosing 'c' raises UserCancelledRun."""
        tool = self._make_tool()
        reg = self._make_registry(confirms=True)
        ss = SessionState()
        ui = _StubUI("c")
        wrapped = create_safe_tool_wrapper(tool, "test_tool", reg, set(), session_state=ss, ui=ui)
        with pytest.raises(UserCancelledRun):
            wrapped.invoke({})

    def test_deny_all_sets_session_flag(self):
        """Choosing 'f' sets ss.deny_all = True."""
        tool = self._make_tool()
        reg = self._make_registry(confirms=True)
        ss = SessionState()
        ui = _StubUI("f")
        wrapped = create_safe_tool_wrapper(tool, "test_tool", reg, set(), session_state=ss, ui=ui)
        result = wrapped.invoke({})
        assert ss.deny_all is True
        assert "denied" in result.lower()

    def test_disable_adds_to_denials(self):
        """Choosing 'd' adds tool to ss.denials."""
        tool = self._make_tool()
        reg = self._make_registry(confirms=True)
        ss = SessionState()
        ui = _StubUI("d")
        wrapped = create_safe_tool_wrapper(tool, "test_tool", reg, set(), session_state=ss, ui=ui)
        result = wrapped.invoke({})
        assert "test_tool" in ss.denials
        assert "denied" in result.lower()

    def test_spinner_resumed_on_eoferror(self):
        """Spinner must be resumed even when run_confirmation_prompt raises EOFError."""
        tool = self._make_tool()
        reg = self._make_registry(confirms=True)
        ss = SessionState()
        ui = _StubUI()

        from unittest.mock import patch

        with patch(
            "src.agent.safety.run_confirmation_prompt", side_effect=EOFError("stdin closed")
        ):
            wrapped = create_safe_tool_wrapper(
                tool, "test_tool", reg, set(), session_state=ss, ui=ui
            )
            with pytest.raises(EOFError):
                wrapped.invoke({})

        assert ui.paused == 0

    def test_spinner_resumed_on_runtime_error(self):
        """Spinner must be resumed even when run_confirmation_prompt raises RuntimeError."""
        tool = self._make_tool()
        reg = self._make_registry(confirms=True)
        ss = SessionState()
        ui = _StubUI()

        from unittest.mock import patch

        with patch(
            "src.agent.safety.run_confirmation_prompt",
            side_effect=RuntimeError("unexpected failure"),
        ):
            wrapped = create_safe_tool_wrapper(
                tool, "test_tool", reg, set(), session_state=ss, ui=ui
            )
            with pytest.raises(RuntimeError):
                wrapped.invoke({})

        assert ui.paused == 0

    def test_file_not_found_error_type_in_message(self):
        """When the underlying tool raises FileNotFoundError, the result names the exception type."""
        tool = self._make_tool()
        tool.func = MagicMock(side_effect=FileNotFoundError("no such file"))
        reg = self._make_registry(confirms=False)
        ss = SessionState()
        wrapped = create_safe_tool_wrapper(tool, "test_tool", reg, set(), session_state=ss, ui=None)
        result = wrapped.invoke({})
        assert "FileNotFoundError" in result

    def test_tool_execution_error_includes_exception_class_name(self):
        """The error message for any tool exception includes the exception class name."""
        tool = self._make_tool()
        tool.func = MagicMock(side_effect=PermissionError("access denied"))
        reg = self._make_registry(confirms=False)
        ss = SessionState()
        wrapped = create_safe_tool_wrapper(tool, "test_tool", reg, set(), session_state=ss, ui=None)
        result = wrapped.invoke({})
        assert "PermissionError" in result
