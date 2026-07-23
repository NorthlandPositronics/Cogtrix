"""Regression tests for --silent mode (issue #119).

Tests cover:
- CLI argument parsing: --silent flag, positional PROMPT, NO_COLOR env setup
- run_single_prompt: deny_all_tools sets session.deny_all after reset
- Spinner: _tty_output_enabled respects NO_COLOR
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Argument parser tests
# ---------------------------------------------------------------------------


class TestSilentArgParsing:
    def _parse(self, argv: list[str]):

        import src.cli.args as args_mod

        with patch.object(sys, "argv", ["cogtrix.py"] + argv):
            return args_mod.parse_arguments()

    def test_silent_flag_long(self):
        args = self._parse(["--silent", "--prompt", "hello"])
        assert args.silent is True

    def test_silent_flag_short(self):
        args = self._parse(["-S", "--prompt", "hello"])
        assert args.silent is True

    def test_silent_default_false(self):
        args = self._parse(["--prompt", "hello"])
        assert args.silent is False

    def test_positional_prompt_sets_silent(self):
        """A positional PROMPT without --silent should auto-set silent=True."""
        args = self._parse(["my prompt text"])
        assert args.inline_prompt == "my prompt text"
        assert args.silent is True

    def test_positional_prompt_with_silent(self):
        args = self._parse(["--silent", "my prompt text"])
        assert args.inline_prompt == "my prompt text"
        assert args.silent is True

    def test_silent_without_prompt_accepted_by_parser(self):
        """Parser must not reject --silent with no prompt (prompt can come from stdin)."""
        args = self._parse(["--silent"])
        assert args.silent is True
        assert not args.prompt
        assert not getattr(args, "inline_prompt", None)

    def test_silent_compatible_with_no_confirm(self):
        args = self._parse(["--silent", "-y", "--prompt", "hi"])
        assert args.silent is True
        assert args.no_confirm is True

    def test_silent_compatible_with_output(self):
        args = self._parse(["--silent", "--output", "/tmp/out.txt", "--prompt", "hi"])
        assert args.silent is True
        assert args.output == "/tmp/out.txt"

    def test_silent_compatible_with_model(self):
        args = self._parse(["--silent", "-m", "gemini", "--prompt", "hi"])
        assert args.silent is True
        assert args.model == "gemini"


# ---------------------------------------------------------------------------
# Spinner NO_COLOR tests
# ---------------------------------------------------------------------------


class TestSpinnerNoColor:
    def test_tty_output_disabled_when_no_color_set(self):
        from src.ui.spinner import ActivityIndicator

        with patch.dict(os.environ, {"NO_COLOR": "1"}, clear=False):
            assert ActivityIndicator._tty_output_enabled() is False

    def test_tty_output_enabled_without_no_color_on_tty(self):
        from src.ui.spinner import ActivityIndicator

        env = {k: v for k, v in os.environ.items() if k not in ("NO_COLOR", "FORCE_COLOR")}
        with patch.dict(os.environ, env, clear=True):
            with patch.object(sys.stdout, "isatty", return_value=True):
                assert ActivityIndicator._tty_output_enabled() is True

    def test_no_color_overrides_force_color(self):
        """NO_COLOR must win even if FORCE_COLOR is also set."""
        from src.ui.spinner import ActivityIndicator

        with patch.dict(os.environ, {"NO_COLOR": "1", "FORCE_COLOR": "1"}, clear=False):
            assert ActivityIndicator._tty_output_enabled() is False

    def test_force_color_enables_on_non_tty(self):
        from src.ui.spinner import ActivityIndicator

        env = {k: v for k, v in os.environ.items() if k not in ("NO_COLOR", "FORCE_COLOR")}
        env["FORCE_COLOR"] = "1"
        with patch.dict(os.environ, env, clear=True):
            with patch.object(sys.stdout, "isatty", return_value=False):
                assert ActivityIndicator._tty_output_enabled() is True


# ---------------------------------------------------------------------------
# run_single_prompt: deny_all_tools
# ---------------------------------------------------------------------------


class TestRunSinglePromptDenyAllTools:
    """deny_all_tools=True must set _session.deny_all after reset_for_new_prompt."""

    def _import_cogtrix(self):

        import cogtrix

        return cogtrix

    def _make_mock_agent_result(self):
        return "The answer is 42."

    def test_deny_all_set_when_deny_all_tools_true(self):
        """_session.deny_all must be True during the agent run when deny_all_tools=True."""
        import cogtrix

        # Track deny_all value at the time run_agent is called
        captured = {}

        def fake_run_agent(*args, **kwargs):
            captured["deny_all"] = cogtrix._session.deny_all
            return "response"

        mm = MagicMock()
        mm.prepare_context.return_value = MagicMock(
            messages=[], context_prefix="", context_messages_count=0
        )
        mm.update = MagicMock()
        mm.save = MagicMock()

        registry = MagicMock()

        with patch.object(cogtrix, "run_agent", side_effect=fake_run_agent):
            with patch.object(cogtrix, "_spinner") as mock_spinner:
                mock_spinner.start = MagicMock()
                mock_spinner.stop = MagicMock()
                with patch.object(cogtrix, "log_user_message"):
                    with patch.object(cogtrix, "log_agent_response"):
                        with patch.object(cogtrix, "_is_valid_response", return_value=False):
                            with patch.object(cogtrix, "user_wants_deep_think", return_value=False):
                                with patch.object(
                                    cogtrix, "user_wants_delegation", return_value=False
                                ):
                                    with patch.object(
                                        cogtrix, "prompt_requests_action", return_value=False
                                    ):
                                        cogtrix.run_single_prompt(
                                            prompt_text="hello",
                                            memory_manager=mm,
                                            registry=registry,
                                            approvals=set(),
                                            deny_all_tools=True,
                                        )

        assert (
            captured.get("deny_all") is True
        ), "deny_all_tools=True must set _session.deny_all=True before run_agent"

    def test_deny_all_not_set_when_deny_all_tools_false(self):
        """_session.deny_all must remain False when deny_all_tools=False."""
        import cogtrix

        captured = {}

        def fake_run_agent(*args, **kwargs):
            captured["deny_all"] = cogtrix._session.deny_all
            return "response"

        mm = MagicMock()
        mm.prepare_context.return_value = MagicMock(
            messages=[], context_prefix="", context_messages_count=0
        )
        mm.update = MagicMock()
        mm.save = MagicMock()

        registry = MagicMock()

        with patch.object(cogtrix, "run_agent", side_effect=fake_run_agent):
            with patch.object(cogtrix, "_spinner") as mock_spinner:
                mock_spinner.start = MagicMock()
                mock_spinner.stop = MagicMock()
                with patch.object(cogtrix, "log_user_message"):
                    with patch.object(cogtrix, "log_agent_response"):
                        with patch.object(cogtrix, "_is_valid_response", return_value=False):
                            with patch.object(cogtrix, "user_wants_deep_think", return_value=False):
                                with patch.object(
                                    cogtrix, "user_wants_delegation", return_value=False
                                ):
                                    with patch.object(
                                        cogtrix, "prompt_requests_action", return_value=False
                                    ):
                                        cogtrix.run_single_prompt(
                                            prompt_text="hello",
                                            memory_manager=mm,
                                            registry=registry,
                                            approvals=set(),
                                            deny_all_tools=False,
                                        )

        assert (
            captured.get("deny_all") is False
        ), "deny_all_tools=False must leave _session.deny_all unchanged (False)"


# ---------------------------------------------------------------------------
# Source-level checks for the --silent integration
# ---------------------------------------------------------------------------


class TestSilentModeSourceInvariants:
    def test_run_single_prompt_has_deny_all_tools_param(self):
        import inspect

        import cogtrix

        sig = inspect.signature(cogtrix.run_single_prompt)
        assert (
            "deny_all_tools" in sig.parameters
        ), "run_single_prompt must accept deny_all_tools parameter"
        assert sig.parameters["deny_all_tools"].default is False

    def test_main_sets_no_color_for_silent(self):
        import inspect

        import cogtrix

        src = inspect.getsource(cogtrix.main)
        assert "NO_COLOR" in src, "main() must set NO_COLOR env var when --silent is active"
        assert "silent" in src

    def test_main_reads_stdin_for_silent_prompt(self):
        import inspect

        import cogtrix

        src = inspect.getsource(cogtrix.main)
        assert (
            "stdin.read" in src or "stdin.isatty" in src
        ), "main() must handle stdin prompt source for --silent mode"

    def test_main_passes_deny_all_tools_to_run_single_prompt(self):
        import inspect

        import cogtrix

        src = inspect.getsource(cogtrix.main)
        assert (
            "deny_all_tools" in src
        ), "main() must pass deny_all_tools to run_single_prompt for silent mode"
