"""Tests for src/agent/safety — unified tool confirmation logic."""

import threading
from unittest.mock import MagicMock

import pytest

from src.agent.safety import (
    ConfirmationResult,
    ConfirmationUI,
    UserCancelledRun,
    _confirmation_lock,
    create_safe_tool_wrapper,
    run_confirmation_prompt,
)
from src.orchestration.session_state import SessionState


class _StubUI:
    """Minimal ConfirmationUI for testing.

    Pass a single string for a constant response, or a list of strings to
    return each value in sequence (last entry repeated when exhausted).
    """

    def __init__(self, choice: str | list[str] = "y"):
        self._choices = [choice] if isinstance(choice, str) else list(choice)
        self._index = 0
        self.rendered = False
        self.messages: list[tuple[str, str]] = []
        self.paused = 0

    def render_prompt(self, tool_name, tool_input, last_keys, preview_limit):
        self.rendered = True

    def read_choice(self) -> str:
        val = self._choices[self._index]
        if self._index < len(self._choices) - 1:
            self._index += 1
        return val

    def show_message(self, message: str, style: str) -> None:
        self.messages.append((message, style))

    def show_diff_preview(self, path: str, diff_lines: list[str]) -> None:
        pass

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

    def test_unknown_choice_reprompts_then_denies(self):
        # "xyz" is invalid → show_message is called, then "n" is accepted
        ui = _StubUI(["xyz", "n"])
        result = run_confirmation_prompt("t", {}, ui)
        assert result == ConfirmationResult.DENIED_ONCE
        assert any("Invalid choice" in msg for msg, _ in ui.messages)

    def test_empty_string_approves_once(self):
        assert run_confirmation_prompt("t", {}, _StubUI("")) == ConfirmationResult.APPROVED_ONCE


class TestUserCancelledRun:
    def test_importable(self):
        assert issubclass(UserCancelledRun, Exception)

    def test_raisable(self):
        with pytest.raises(UserCancelledRun):
            raise UserCancelledRun()


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

    # ── Bug #1704 regressions: tool-spam blocking after no-UI denial ──

    def test_no_ui_denial_disables_tool_for_session(self):
        """After the first no-UI silent denial, the tool must be added
        to session_state.denials so subsequent invocations short-circuit
        instead of looping. Bug #1704: write_file got called 7+ times
        on B01 because the deny path didn't pin the denial."""
        tool = self._make_tool(name="write_file")
        reg = self._make_registry(confirms=True)
        ss = SessionState()
        assert not ss.is_denied("write_file"), "pre-condition"

        wrapped = create_safe_tool_wrapper(
            tool, "write_file", reg, set(), session_state=ss, ui=None
        )
        wrapped.invoke({})

        # First denial must have pinned the deny.
        assert ss.is_denied("write_file"), (
            "expected the first no-UI silent denial to mark the tool as denied "
            "for the session — without that, the agent loops on retries"
        )

    def test_no_ui_denial_returns_explicit_do_not_retry_message(self):
        """The returned tool result must explicitly tell the model not
        to retry and to respond inline. Without that nudge the model
        keeps trying with different args/paths."""
        tool = self._make_tool(name="write_file")
        reg = self._make_registry(confirms=True)
        ss = SessionState()
        wrapped = create_safe_tool_wrapper(
            tool, "write_file", reg, set(), session_state=ss, ui=None
        )
        result = wrapped.invoke({})
        text = result if isinstance(result, str) else str(result)
        lower = text.lower()
        # Strong stop signal — at least one of "do not retry" / "disabled"
        # / "respond inline" must appear.
        assert any(
            phrase in lower for phrase in ("do not retry", "disabled for", "respond inline")
        ), f"weak no-UI denial message: {text!r}"

    def test_second_invocation_short_circuits_after_first_denial(self):
        """Once a tool is in session_state.denials (by the first call),
        the second invocation returns the "User denied execution" path
        without re-entering the confirmation branch."""
        tool_func_calls = []

        def fake_func(*args, **kwargs):
            tool_func_calls.append((args, kwargs))
            return "executed"

        tool = self._make_tool(name="write_file")
        tool.func = fake_func
        reg = self._make_registry(confirms=True)
        ss = SessionState()
        wrapped = create_safe_tool_wrapper(
            tool, "write_file", reg, set(), session_state=ss, ui=None
        )

        # First call — gets pinned-denied.
        wrapped.invoke({})
        # Second call — must NOT invoke the underlying func because the
        # tool is now in the denials set.
        result2 = wrapped.invoke({})
        assert "denied" in str(result2).lower()
        assert tool_func_calls == [], "underlying tool func was invoked despite the prior denial"

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

    def test_tool_trust_deny_blocks_execution(self):
        """tool_trust={"tool": "deny"} blocks execution regardless of registry or approvals.

        Regression test for issue #1000: dynamic tool loading paths (process_tools.py,
        sessions.py) were not propagating tool_trust to create_safe_tool_wrapper, causing
        tool_trust: deny to be silently ignored for on-demand tools.
        """
        tool_func = MagicMock(return_value="executed result")
        tool = MagicMock()
        tool.name = "test_tool"
        tool.description = "A test tool"
        tool.args_schema = None
        tool.func = tool_func
        reg = self._make_registry(confirms=True)  # Would normally prompt
        ss = SessionState()
        approvals: set[str] = set()  # Not pre-approved
        wrapped = create_safe_tool_wrapper(
            tool,
            "test_tool",
            reg,
            approvals,
            session_state=ss,
            ui=None,
            tool_trust={"test_tool": "deny"},
        )
        result = wrapped.invoke({})
        assert result == "User denied execution"
        # Tool func must NOT have been called
        tool_func.assert_not_called()
        # Denied tool must NOT be added to approvals
        assert "test_tool" not in approvals

    def test_tool_trust_always_skips_confirmation(self):
        """tool_trust={"tool": "always"} auto-approves without prompting.

        Regression test for issue #1000: ensures tool_trust: always works for dynamically
        loaded tools, preventing unwanted confirmation prompts for trusted tools.
        """
        tool_func = MagicMock(return_value="executed result")
        tool = MagicMock()
        tool.name = "test_tool"
        tool.description = "A test tool"
        tool.args_schema = None
        tool.func = tool_func
        reg = self._make_registry(confirms=True)  # Would normally prompt
        ss = SessionState()
        approvals: set[str] = set()
        wrapped = create_safe_tool_wrapper(
            tool,
            "test_tool",
            reg,
            approvals,
            session_state=ss,
            ui=None,  # No UI — if trust: always is not respected, this would deny silently
            tool_trust={"test_tool": "always"},
        )
        # With tool_trust: always, tool should execute even with no UI and no approvals
        result = wrapped.invoke({})
        # The wrapped tool executes and returns the func result (not a denial message)
        assert result == "executed result"
        # Tool func WAS called
        tool_func.assert_called_once()

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

    def test_user_cancelled_run_propagates_rather_than_being_swallowed(self):
        """Regression #1193: UserCancelledRun must not be caught by broad except Exception."""
        tool = self._make_tool()
        reg = self._make_registry(confirms=False)  # No confirmation needed, tool executes directly
        ss = SessionState()

        # Simulate the tool raising UserCancelledRun directly
        def raise_cancel():
            raise UserCancelledRun("user pressed 'c'")

        tool.func = raise_cancel

        wrapped = create_safe_tool_wrapper(tool, "test_tool", reg, set(), session_state=ss, ui=None)
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

    def test_file_not_found_returns_sanitized_message(self):
        """When the underlying tool raises FileNotFoundError, no exception class name is exposed."""
        tool = self._make_tool()
        tool.func = MagicMock(side_effect=FileNotFoundError("no such file"))
        reg = self._make_registry(confirms=False)
        ss = SessionState()
        wrapped = create_safe_tool_wrapper(tool, "test_tool", reg, set(), session_state=ss, ui=None)
        result = wrapped.invoke({})
        # Sanitized: no exception class name, no raw error content
        assert "FileNotFoundError" not in result
        assert "no such file" not in result
        assert "Tool execution error:" in result

    def test_permission_error_returns_sanitized_message(self):
        """When the underlying tool raises PermissionError, no exception class name is exposed."""
        tool = self._make_tool()
        tool.func = MagicMock(side_effect=PermissionError("access denied"))
        reg = self._make_registry(confirms=False)
        ss = SessionState()
        wrapped = create_safe_tool_wrapper(tool, "test_tool", reg, set(), session_state=ss, ui=None)
        result = wrapped.invoke({})
        # Sanitized: no exception class name, no raw error content
        assert "PermissionError" not in result
        assert "access denied" not in result
        assert "Tool execution error:" in result

    def test_error_message_hides_path_and_class_name(self):
        """When the underlying tool raises with an absolute path, no path or class name is exposed."""
        tool = self._make_tool()
        tool.func = MagicMock(side_effect=FileNotFoundError("No such file: /etc/passwd"))
        reg = self._make_registry(confirms=False)
        ss = SessionState()
        wrapped = create_safe_tool_wrapper(tool, "test_tool", reg, set(), session_state=ss, ui=None)
        result = wrapped.invoke({})
        # Sanitized: no absolute path, no exception class name, no raw error content
        assert "/etc/passwd" not in result
        assert "FileNotFoundError" not in result
        assert "No such file" not in result
        assert "Tool execution error:" in result


class TestErrorSanitization:
    """Regression tests for error sanitization in create_safe_tool_wrapper (issue #1424).

    Verifies that exception class names, library internals, filesystem paths,
    and raw error content are never exposed to the LLM in tool error messages.
    """

    def _make_tool(self):
        from pydantic import BaseModel

        from src.tools.weather import get_weather

        tool = MagicMock()
        tool.name = "test_tool"
        tool.func = get_weather
        tool.input_schema = BaseModel
        tool.args_schema = BaseModel  # must be BaseModel subclass, not MagicMock
        tool.requires_confirmation = False
        return tool

    def _make_registry(self, confirms: bool = False):
        from src.registry import ToolRegistry

        reg = ToolRegistry()
        return reg

    def _invoke(self, side_effect):
        tool = self._make_tool()
        tool.func = MagicMock(side_effect=side_effect)
        reg = self._make_registry()
        ss = SessionState()
        wrapped = create_safe_tool_wrapper(tool, "test_tool", reg, set(), session_state=ss, ui=None)
        return wrapped.invoke({})

    def test_os_error_errno_not_exposed(self):
        """OSError with errno should not expose the errno number or class name."""
        exc = OSError(28, "No space left on device")
        result = self._invoke(exc)
        assert "OSError" not in result
        assert "28" not in result  # errno
        assert "No space left on device" not in result
        assert "Tool execution error:" in result

    def test_subprocess_called_process_error_hides_output(self):
        """CalledProcessError should not expose command output or class name."""
        import subprocess

        exc = subprocess.CalledProcessError(
            1, "rm -rf /", stderr="rm: cannot remove '/': Permission denied"
        )
        result = self._invoke(exc)
        assert "CalledProcessError" not in result
        assert "Permission denied" not in result
        assert "rm: cannot remove" not in result
        assert "exit code 1" in result  # return code is safe to expose

    def test_runtime_error_sanitized(self):
        """RuntimeError should not expose the class name or message."""
        exc = RuntimeError("Session state corrupted: missing required key 'agent_id'")
        result = self._invoke(exc)
        assert "RuntimeError" not in result
        assert "Session state corrupted" not in result
        assert "missing required key" not in result
        assert "Tool execution error:" in result

    def test_generic_exception_no_class_name(self):
        """A bare Exception should not expose class name or message."""
        exc = Exception("Internal error: database connection pool exhausted (max_connections=10)")
        result = self._invoke(exc)
        assert "Exception" not in result
        assert "connection pool exhausted" not in result
        assert "max_connections" not in result
        assert "Tool execution error:" in result

    def test_no_internal_paths_in_messages(self):
        """Error messages must not contain filesystem paths."""
        import pathlib

        test_path = str(pathlib.Path("/home/user/.config/cogtrix/secrets.json"))
        exc = FileNotFoundError(f"Config file not found at {test_path}")
        result = self._invoke(exc)
        assert test_path not in result
        assert "/home/user/.config" not in result
        assert "secrets.json" not in result
        assert "FileNotFoundError" not in result
        assert "Config file not found" not in result  # raw message content hidden


class TestComputeDiff:
    """Tests for _compute_file_diff() — the diff preview helper."""

    def test_write_file_new_file_returns_diff(self, tmp_path, monkeypatch):
        from src.agent.safety import _compute_file_diff

        monkeypatch.chdir(tmp_path)
        p = tmp_path / "new.txt"
        result = _compute_file_diff("write_file", {"path": str(p), "content": "hello\n"})
        assert result is not None
        path_str, diff_lines = result
        assert path_str == str(p)
        assert any("+hello" in line for line in diff_lines)

    def test_write_file_no_change_returns_none(self, tmp_path, monkeypatch):
        from src.agent.safety import _compute_file_diff

        monkeypatch.chdir(tmp_path)
        p = tmp_path / "existing.txt"
        p.write_text("hello\n")
        result = _compute_file_diff("write_file", {"path": str(p), "content": "hello\n"})
        assert result is None

    def test_write_file_existing_file_returns_diff(self, tmp_path, monkeypatch):
        from src.agent.safety import _compute_file_diff

        monkeypatch.chdir(tmp_path)
        p = tmp_path / "existing.txt"
        p.write_text("old content\n")
        result = _compute_file_diff("write_file", {"path": str(p), "content": "new content\n"})
        assert result is not None
        _, diff_lines = result
        assert any("-old content" in line for line in diff_lines)
        assert any("+new content" in line for line in diff_lines)

    def test_write_file_missing_path_returns_none(self):
        from src.agent.safety import _compute_file_diff

        result = _compute_file_diff("write_file", {"path": "", "content": "x"})
        assert result is None

    def test_patch_file_not_exists_returns_none(self, tmp_path, monkeypatch):
        from src.agent.safety import _compute_file_diff

        monkeypatch.chdir(tmp_path)
        p = tmp_path / "missing.txt"
        result = _compute_file_diff("patch_file", {"path": str(p), "old_str": "x", "new_str": "y"})
        assert result is None

    def test_patch_file_ambiguous_returns_none(self, tmp_path, monkeypatch):
        from src.agent.safety import _compute_file_diff

        monkeypatch.chdir(tmp_path)
        p = tmp_path / "f.txt"
        p.write_text("x\nx\n")
        result = _compute_file_diff("patch_file", {"path": str(p), "old_str": "x", "new_str": "y"})
        assert result is None

    def test_patch_file_returns_diff(self, tmp_path, monkeypatch):
        from src.agent.safety import _compute_file_diff

        monkeypatch.chdir(tmp_path)
        p = tmp_path / "f.txt"
        p.write_text("hello world\n")
        result = _compute_file_diff(
            "patch_file", {"path": str(p), "old_str": "world", "new_str": "earth"}
        )
        assert result is not None
        _, diff_lines = result
        assert any("-hello world" in line for line in diff_lines)

    def test_unknown_tool_returns_none(self):
        from src.agent.safety import _compute_file_diff

        result = _compute_file_diff("read_file", {"path": "/tmp/x"})
        assert result is None

    def test_patch_file_missing_path_returns_none(self):
        from src.agent.safety import _compute_file_diff

        result = _compute_file_diff("patch_file", {"path": "", "old_str": "x", "new_str": "y"})
        assert result is None

    def test_write_file_path_traversal_returns_none(self):
        from src.agent.safety import _compute_file_diff

        result = _compute_file_diff("write_file", {"path": "/etc/passwd", "content": "pwned\n"})
        assert result is None

    def test_patch_file_path_traversal_returns_none(self):
        from src.agent.safety import _compute_file_diff

        result = _compute_file_diff(
            "patch_file",
            {"path": "/etc/passwd", "old_str": "root", "new_str": "toor"},
        )
        assert result is None

    def test_write_file_dotdot_traversal_returns_none(self, tmp_path):
        from src.agent.safety import _compute_file_diff

        result = _compute_file_diff(
            "write_file",
            {"path": str(tmp_path / ".." / ".." / "secret.txt"), "content": "x"},
        )
        assert result is None

    def test_patch_file_dotdot_traversal_returns_none(self, tmp_path):
        from src.agent.safety import _compute_file_diff

        result = _compute_file_diff(
            "patch_file",
            {
                "path": str(tmp_path / ".." / ".." / "secret.txt"),
                "old_str": "a",
                "new_str": "b",
            },
        )
        assert result is None


class TestRunConfirmationPromptExtended:
    def _make_ui(self, choices):
        if isinstance(choices, str):
            choices = [choices]
        return _StubUI(choices)

    def test_forbid_all_returns_denied_all(self):
        ui = self._make_ui("f")
        result = run_confirmation_prompt("shell", {}, ui)
        assert result == ConfirmationResult.DENIED_ALL

    def test_forbid_long_form_returns_denied_all(self):
        ui = self._make_ui("forbid-all")
        result = run_confirmation_prompt("shell", {}, ui)
        assert result == ConfirmationResult.DENIED_ALL

    def test_cancel_returns_cancelled(self):
        ui = self._make_ui("c")
        result = run_confirmation_prompt("shell", {}, ui)
        assert result == ConfirmationResult.CANCELLED

    def test_disable_returns_denied_disable(self):
        ui = self._make_ui("d")
        result = run_confirmation_prompt("shell", {}, ui)
        assert result == ConfirmationResult.DENIED_DISABLE

    def test_none_choice_denies_once(self):
        """read_choice() returning None must yield DENIED_ONCE without crashing."""

        class _NoneUI:
            def render_prompt(self, tool_name, tool_input, last_keys, preview_limit):
                pass

            def read_choice(self):
                return None

            def show_message(self, message, style):
                pass

            def show_diff_preview(self, path, diff_lines):
                pass

            def pause_spinner(self):
                pass

            def resume_spinner(self):
                pass

        result = run_confirmation_prompt("my_tool", {}, _NoneUI())
        assert (
            result == ConfirmationResult.DENIED_ONCE
        ), "None from read_choice() must return DENIED_ONCE"

    def test_none_choice_logs_warning(self, caplog):
        """read_choice() returning None must log a WARNING mentioning the tool name."""
        import logging

        class _NoneUI:
            def render_prompt(self, tool_name, tool_input, last_keys, preview_limit):
                pass

            def read_choice(self):
                return None

            def show_message(self, message, style):
                pass

            def show_diff_preview(self, path, diff_lines):
                pass

            def pause_spinner(self):
                pass

            def resume_spinner(self):
                pass

        with caplog.at_level(logging.WARNING, logger="cogtrix"):
            run_confirmation_prompt("dangerous_tool", {}, _NoneUI())

        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warning_records, "Expected at least one WARNING log record when read_choice is None"
        combined = " ".join(r.getMessage() for r in warning_records)
        assert (
            "dangerous_tool" in combined
        ), f"WARNING must mention the tool name 'dangerous_tool'; got: {combined!r}"


class TestConfirmationLockModel:
    """Validate the simplified single-lock confirmation model.

    With the current design a single ``_confirmation_lock`` is held for the
    entire check-prompt-commit window.  These tests verify that the lock is
    correctly shared (not per-thread) and that the approvals set is protected
    against concurrent races.
    """

    def _make_tool(self, name="test_tool"):
        tool = MagicMock()
        tool.name = name
        tool.description = "A test tool"
        tool.args_schema = None
        tool.func = lambda *args, **kwargs: "executed"
        return tool

    def _make_registry(self, confirms=True):
        reg = MagicMock()
        reg.requires_confirmation.return_value = confirms
        return reg

    def test_lock_is_shared_module_singleton(self):
        """Same lock object is visible from all call sites."""
        assert isinstance(_confirmation_lock, type(threading.Lock()))
        # Read again — must be the same object (singleton)
        from src.agent import safety as _safety_mod

        assert _safety_mod._confirmation_lock is _confirmation_lock

    def test_lock_protects_approvals_mutation(self):
        """Concurrent mutation of the approvals set under the lock is safe."""
        tool_a = self._make_tool("tool_a")
        tool_b = self._make_tool("tool_b")
        reg = self._make_registry(confirms=True)
        ss = SessionState()
        approvals: set[str] = set()
        ui_a = _StubUI("a")
        ui_b = _StubUI("a")

        wrapped_a = create_safe_tool_wrapper(
            tool_a, "tool_a", reg, approvals, session_state=ss, ui=ui_a
        )
        wrapped_b = create_safe_tool_wrapper(
            tool_b, "tool_b", reg, approvals, session_state=ss, ui=ui_b
        )

        errors: list[Exception] = []

        def _run_a():
            try:
                wrapped_a.invoke({})
            except Exception as exc:
                errors.append(exc)

        def _run_b():
            try:
                wrapped_b.invoke({})
            except Exception as exc:
                errors.append(exc)

        t_a = threading.Thread(target=_run_a)
        t_b = threading.Thread(target=_run_b)
        t_a.start()
        t_b.start()
        t_a.join()
        t_b.join()

        assert not errors, f"Unexpected exceptions: {errors}"
        # Both tools were approved — approvals set must contain both
        assert "tool_a" in approvals
        assert "tool_b" in approvals

    def test_second_thread_sees_existing_approval(self):
        """Thread B sees an approval added by thread A and skips the prompt."""
        from unittest.mock import patch

        tool = self._make_tool("test_tool")
        reg = self._make_registry(confirms=True)
        ss = SessionState()
        approvals: set[str] = set()

        prompt_count = 0

        def _counting_prompt(name, inp, ui):
            nonlocal prompt_count
            prompt_count += 1
            return run_confirmation_prompt(name, inp, ui)

        wrapped = create_safe_tool_wrapper(
            tool, "test_tool", reg, approvals, session_state=ss, ui=_StubUI("a")
        )

        def _run():
            with patch("src.agent.safety.run_confirmation_prompt", _counting_prompt):
                wrapped.invoke({})

        # Thread A runs to completion
        _run()
        assert prompt_count == 1
        assert "test_tool" in approvals

        # Thread B starts after A has fully added to approvals
        t2 = threading.Thread(target=_run)
        t2.start()
        t2.join()

        # With the shared lock, B sees the existing approval and skips prompting
        assert prompt_count == 1, f"expected 1 prompt, got {prompt_count}"
