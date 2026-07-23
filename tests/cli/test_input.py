"""Tests for src/cli/input.py — REPL input handling, history, and inline shell."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# run_inline_shell
# ---------------------------------------------------------------------------


class TestRunInlineShell:
    def test_empty_command_prints_usage(self, capsys):
        from src.cli.input import run_inline_shell

        run_inline_shell("")
        captured = capsys.readouterr()
        assert "Usage:" in captured.out

    def test_simple_command_without_metacharacters(self):
        from src.cli.input import run_inline_shell

        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("hello\n", "")
        mock_proc.returncode = 0

        with patch("subprocess.Popen", return_value=mock_proc) as popen_mock:
            run_inline_shell("echo hello")

        popen_mock.assert_called_once()
        args, kwargs = popen_mock.call_args
        assert kwargs.get("shell") is not True
        assert args[0] == ["echo", "hello"]

    def test_metacharacter_command_uses_shell_true(self):
        from src.cli.input import run_inline_shell

        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("out\n", "")
        mock_proc.returncode = 0

        with patch("subprocess.Popen", return_value=mock_proc) as popen_mock:
            run_inline_shell("ls | grep foo")

        popen_mock.assert_called_once()
        args, kwargs = popen_mock.call_args
        assert kwargs.get("shell") is True
        assert args[0] == "ls | grep foo"

    def test_nonexistent_command_file_not_found(self, capsys):
        from src.cli.input import run_inline_shell

        with patch(
            "subprocess.Popen",
            side_effect=FileNotFoundError("No such file"),
        ):
            run_inline_shell("nonexistent_command_xyz")

        captured = capsys.readouterr()
        assert "Command not found" in captured.out
        assert "nonexistent_command_xyz" in captured.out

    def test_command_timeout(self, capsys):
        from src.cli.input import run_inline_shell

        mock_proc = MagicMock()
        mock_proc.communicate.side_effect = subprocess.TimeoutExpired("cmd", 30)

        with patch("subprocess.Popen", return_value=mock_proc):
            run_inline_shell("sleep 60")

        captured = capsys.readouterr()
        assert "timed out" in captured.out.lower() or "timeout" in captured.out.lower()

    def test_output_truncation_when_too_large(self):
        from src.cli.input import run_inline_shell

        large_output = "x" * 600_000
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (large_output, "")
        mock_proc.returncode = 0

        with patch("subprocess.Popen", return_value=mock_proc):
            run_inline_shell("cat bigfile")

    def test_stderr_appended_to_stdout(self):
        from src.cli.input import run_inline_shell

        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("stdout\n", "stderr\n")
        mock_proc.returncode = 0

        with patch("subprocess.Popen", return_value=mock_proc):
            run_inline_shell("cmd")

    def test_nonzero_exit_code_prints_code(self, capsys):
        from src.cli.input import run_inline_shell

        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("", "")
        mock_proc.returncode = 1

        with patch("subprocess.Popen", return_value=mock_proc):
            run_inline_shell("false")

        captured = capsys.readouterr()
        assert "exit code: 1" in captured.out

    def test_crlf_stripped_from_command(self):
        from src.cli.input import run_inline_shell

        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("", "")
        mock_proc.returncode = 0

        with patch("subprocess.Popen", return_value=mock_proc) as popen_mock:
            run_inline_shell("echo hello\r")

        args, kwargs = popen_mock.call_args
        assert "\r" not in args[0]


# ---------------------------------------------------------------------------
# History management
# ---------------------------------------------------------------------------


class TestLoadInputHistory:
    def test_no_history_file_does_not_crash(self):
        from src.cli import input as input_mod

        mock_rl = MagicMock()
        mock_rl.read_history_file.side_effect = FileNotFoundError("no file")

        with patch.object(input_mod, "readline", mock_rl):
            input_mod.load_input_history()

    def test_existing_history_file_loaded(self, tmp_path):
        from src.cli import input as input_mod

        history_file = tmp_path / ".input_history"
        history_file.write_text("line1\nline2\n")

        mock_rl = MagicMock()
        mock_rl.read_history_file = MagicMock()

        with patch.object(input_mod, "readline", mock_rl):
            with patch.object(input_mod, "_history_file", return_value=history_file):
                input_mod.load_input_history()

        mock_rl.read_history_file.assert_called_once_with(str(history_file))
        mock_rl.set_history_length.assert_called_once()

    def test_oserror_disables_history(self, capsys):
        from src.cli import input as input_mod

        input_mod._history_disabled = False
        mock_rl = MagicMock()
        mock_rl.read_history_file.side_effect = OSError("permission denied")

        mock_path = MagicMock()
        mock_path.exists.return_value = True

        with patch.object(input_mod, "readline", mock_rl):
            with patch.object(input_mod, "_history_file", return_value=mock_path):
                input_mod.load_input_history()

        assert input_mod._history_disabled is True
        captured = capsys.readouterr()
        assert "Could not load" in captured.out


class TestSaveInputHistory:
    def test_creates_directory_and_writes_file(self, tmp_path):
        from src.cli import input as input_mod

        history_dir = tmp_path / "history"
        history_file = history_dir / ".input_history"

        mock_rl = MagicMock()

        with patch.object(input_mod, "readline", mock_rl):
            with patch.object(input_mod, "_history_dir", return_value=history_dir):
                with patch.object(input_mod, "_history_file", return_value=history_file):
                    input_mod._history_disabled = False
                    input_mod.save_input_history()

        assert history_dir.exists()
        mock_rl.write_history_file.assert_called_once_with(str(history_file))

    def test_oserror_disables_history(self, capsys):
        from src.cli import input as input_mod

        mock_rl = MagicMock()
        mock_rl.write_history_file.side_effect = OSError("disk full")

        with patch.object(input_mod, "readline", mock_rl):
            input_mod._history_disabled = False
            input_mod.save_input_history()

        assert input_mod._history_disabled is True
        captured = capsys.readouterr()
        assert "Could not save" in captured.out

    def test_skips_when_history_disabled(self):
        from src.cli import input as input_mod

        mock_rl = MagicMock()

        with patch.object(input_mod, "readline", mock_rl):
            input_mod._history_disabled = True
            input_mod.save_input_history()

        mock_rl.write_history_file.assert_not_called()

    def test_skips_when_readline_unavailable(self):
        from src.cli import input as input_mod

        with patch.object(input_mod, "readline", None):
            input_mod._history_disabled = False
            input_mod.save_input_history()


# ---------------------------------------------------------------------------
# read_multiline
# ---------------------------------------------------------------------------


class TestReadMultiline:
    def test_normal_input_terminated_by_triple_quote(self):
        from src.cli.input import read_multiline

        inputs = ["line1", "line2", '"""']
        with patch("builtins.input", side_effect=inputs):
            result = read_multiline()

        assert result == "line1\nline2"

    def test_ctrl_c_returns_empty_string(self):
        from src.cli.input import read_multiline

        with patch("builtins.input", side_effect=KeyboardInterrupt):
            result = read_multiline()

        assert result == ""

    def test_eof_finishes_input(self):
        from src.cli.input import read_multiline

        inputs = ["line1", EOFError]
        with patch("builtins.input", side_effect=inputs):
            result = read_multiline()

        assert result == "line1"

    def test_first_line_included(self):
        from src.cli.input import read_multiline

        inputs = ['"""']
        with patch("builtins.input", side_effect=inputs):
            result = read_multiline("prefill")

        assert result == "prefill"

    def test_empty_input_with_just_delimiter(self):
        from src.cli.input import read_multiline

        with patch("builtins.input", return_value='"""'):
            result = read_multiline()

        assert result == ""


# ---------------------------------------------------------------------------
# prefill_next_input
# ---------------------------------------------------------------------------


class TestPrefillNextInput:
    def test_sets_startup_hook(self):
        from src.cli import input as input_mod

        mock_rl = MagicMock()

        with patch.object(input_mod, "readline", mock_rl):
            input_mod.prefill_next_input("hello world")

        mock_rl.set_startup_hook.assert_called_once()
        hook = mock_rl.set_startup_hook.call_args[0][0]
        assert hook is not None

    def test_hook_inserts_text_and_clears(self):
        from src.cli import input as input_mod

        mock_rl = MagicMock()

        with patch.object(input_mod, "readline", mock_rl):
            input_mod.prefill_next_input("test text")
            hook = mock_rl.set_startup_hook.call_args[0][0]
            hook()

        mock_rl.insert_text.assert_called_once_with("test text")
        mock_rl.set_startup_hook.assert_called_with(None)

    def test_noop_when_readline_unavailable(self):
        from src.cli import input as input_mod

        with patch.object(input_mod, "readline", None):
            input_mod.prefill_next_input("text")


# ---------------------------------------------------------------------------
# _completer
# ---------------------------------------------------------------------------


class TestCompleter:
    def test_completes_slash_commands(self):
        from src.cli import input as input_mod

        mock_rl = MagicMock()
        mock_rl.get_line_buffer.return_value = "/com"

        with patch.object(input_mod, "readline", mock_rl):
            input_mod.set_slash_commands(["/compact", "/compact-aggressive"])
            result = input_mod._completer("/com", 0)

        assert result == "/compact"

    def test_completes_second_slash_command(self):
        from src.cli import input as input_mod

        mock_rl = MagicMock()
        mock_rl.get_line_buffer.return_value = "/com"

        with patch.object(input_mod, "readline", mock_rl):
            input_mod.set_slash_commands(["/compact", "/compact-aggressive"])
            result0 = input_mod._completer("/com", 0)
            result1 = input_mod._completer("/com", 1)

        assert result0 == "/compact"
        assert result1 == "/compact-aggressive"

    def test_no_match_returns_none(self):
        from src.cli import input as input_mod

        mock_rl = MagicMock()
        mock_rl.get_line_buffer.return_value = "/zzz"

        with patch.object(input_mod, "readline", mock_rl):
            input_mod.set_slash_commands(["/compact"])
            result = input_mod._completer("/zzz", 0)

        assert result is None

    def test_completes_at_file_paths(self, tmp_path, monkeypatch):
        from src.cli import input as input_mod

        monkeypatch.chdir(tmp_path)
        (tmp_path / "foo.txt").write_text("content")

        mock_rl = MagicMock()
        mock_rl.get_line_buffer.return_value = "@fo"

        with patch.object(input_mod, "readline", mock_rl):
            result = input_mod._completer("@fo", 0)

        assert result == "@foo.txt"

    def test_at_path_no_match_returns_none(self, tmp_path, monkeypatch):
        from src.cli import input as input_mod

        monkeypatch.chdir(tmp_path)

        mock_rl = MagicMock()
        mock_rl.get_line_buffer.return_value = "@nonexistent"

        with patch.object(input_mod, "readline", mock_rl):
            result = input_mod._completer("@nonexistent", 0)

        assert result is None

    def test_exception_swallowed_gracefully(self):
        from src.cli import input as input_mod

        mock_rl = MagicMock()
        mock_rl.get_line_buffer.side_effect = RuntimeError("boom")

        with patch.object(input_mod, "readline", mock_rl):
            result = input_mod._completer("", 0)

        assert result is None

    def test_noop_when_readline_unavailable(self):
        from src.cli import input as input_mod

        with patch.object(input_mod, "readline", None):
            result = input_mod._completer("", 0)

        assert result is None


# ---------------------------------------------------------------------------
# setup_readline_completion
# ---------------------------------------------------------------------------


class TestSetupReadlineCompletion:
    def test_registers_completer(self):
        from src.cli import input as input_mod

        mock_rl = MagicMock()

        with patch.object(input_mod, "readline", mock_rl):
            input_mod.setup_readline_completion()

        mock_rl.set_completer.assert_called_once()
        mock_rl.parse_and_bind.assert_called_once_with("tab: complete")

    def test_noop_when_readline_unavailable(self):
        from src.cli import input as input_mod

        with patch.object(input_mod, "readline", None):
            input_mod.setup_readline_completion()


# ---------------------------------------------------------------------------
# _history_dir / _history_file
# ---------------------------------------------------------------------------


class TestHistoryPaths:
    def test_history_dir_uses_env_var(self, monkeypatch):
        from src.cli import input as input_mod

        monkeypatch.setenv("COGTRIX_DATA_DIR", "/tmp/cogtrix")
        result = input_mod._history_dir()
        assert result == Path("/tmp/cogtrix/history")

    def test_history_dir_fallback(self, monkeypatch):
        from src.cli import input as input_mod

        monkeypatch.delenv("COGTRIX_DATA_DIR", raising=False)
        result = input_mod._history_dir()
        assert result == Path("data/history")

    def test_history_file(self):
        from src.cli import input as input_mod

        with patch.object(input_mod, "_history_dir", return_value=Path("/tmp/history")):
            result = input_mod._history_file()
            assert result == Path("/tmp/history/.input_history")


# ---------------------------------------------------------------------------
# set_slash_commands
# ---------------------------------------------------------------------------


class TestSetSlashCommands:
    def test_sets_commands(self):
        from src.cli import input as input_mod

        input_mod.set_slash_commands(["/foo", "/bar"])
        assert input_mod._slash_commands == ["/bar", "/foo"]

    def test_overwrites_existing(self):
        from src.cli import input as input_mod

        input_mod._slash_commands = ["/old"]
        input_mod.set_slash_commands(["/new"])
        assert input_mod._slash_commands == ["/new"]
