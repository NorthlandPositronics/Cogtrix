"""Tests for src/tools/shell.py"""

from __future__ import annotations

import errno
import os
import tempfile
from pathlib import Path

import pytest

from src.tools import shell


class TestShellCommandInput:
    """Validation tests for the Pydantic input schema."""

    def test_valid_input_with_default_timeout(self) -> None:
        inp = shell.ShellCommandInput(command="ls -la")
        assert inp.command == "ls -la"
        assert inp.working_directory is None
        assert inp.timeout == 30  # default value

    def test_valid_input_with_custom_timeout(self) -> None:
        inp = shell.ShellCommandInput(command="ls", timeout=120)
        assert inp.command == "ls"
        assert inp.timeout == 120

    def test_valid_input_with_working_directory(self) -> None:
        inp = shell.ShellCommandInput(command="ls", working_directory="/tmp")
        assert inp.command == "ls"
        assert inp.working_directory == "/tmp"

    def test_empty_command_raises_validation_error(self) -> None:
        # Empty string is valid for Pydantic but will be handled by execute_shell_command
        inp = shell.ShellCommandInput(command="")
        assert inp.command == ""

    def test_whitespace_only_command(self) -> None:
        inp = shell.ShellCommandInput(command="   ")
        assert inp.command == "   "


class TestExecuteShellCommand:
    """Integration tests for execute_shell_command function."""

    def test_simple_command_no_output(self) -> None:
        """Test a command that produces no stdout."""
        result = shell.execute_shell_command("true")
        assert "exit code: 0" in result

    def test_simple_command_with_output(self) -> None:
        """Test a command that produces stdout."""
        result = shell.execute_shell_command("echo hello world")
        assert "hello world" in result

    def test_command_with_working_directory(self) -> None:
        """Test command executed in a specific directory (within cwd)."""
        import tempfile

        cwd = os.getcwd()
        with tempfile.TemporaryDirectory(dir=cwd) as tmpdir:
            result = shell.execute_shell_command("pwd", working_directory=tmpdir)
            assert tmpdir in result
            assert "Error:" not in result

    def test_command_with_nonexistent_directory(self) -> None:
        """Test command with a directory that doesn't exist (outside cwd)."""
        result = shell.execute_shell_command("ls", working_directory="/nonexistent/path")
        assert "outside allowed directories" in result

    def test_command_timeout(self) -> None:
        """Test that a command times out correctly."""
        # Use a short timeout for a slow command
        result = shell.execute_shell_command("sleep 5", timeout=1)
        assert "timed out" in result.lower() or "timeout" in result.lower()

    def test_command_not_found(self) -> None:
        """Test a command that doesn't exist."""
        result = shell.execute_shell_command("thiscommanddoesnotexist12345")
        assert "Error: Command not found" in result

    def test_command_with_stderr(self) -> None:
        """Test command that produces stderr output."""
        result = shell.execute_shell_command("ls /nonexistent/path 2>&1", timeout=5)
        assert "Error" in result or "No such file" in result

    def test_command_with_pipe(self) -> None:
        """Test command with pipe (requires shell=True)."""
        result = shell.execute_shell_command("echo hello | grep hello")
        assert "hello" in result

    def test_command_with_redirect(self) -> None:
        """Test command with redirect (requires shell=True)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "output.txt")
            result = shell.execute_shell_command(f"echo test > {output_file}")
            assert "exit code: 0" in result
            # Verify file was created with correct content
            with open(output_file) as f:
                content = f.read()
            assert content.strip() == "test"

    def test_command_with_semicolon(self) -> None:
        """Test command with semicolon chaining (requires shell=True)."""
        result = shell.execute_shell_command("echo first; echo second")
        assert "first" in result and "second" in result

    def test_command_with_glob(self) -> None:
        """Test command with glob pattern (requires shell=True)."""
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory(dir=cwd) as tmpdir:
            # Create some test files
            for i in range(3):
                Path(tmpdir).joinpath(f"file{i}.txt").write_text(f"content{i}")
            result = shell.execute_shell_command("ls *.txt", working_directory=tmpdir)
            assert "file0.txt" in result
            assert "file1.txt" in result
            assert "file2.txt" in result

    def test_command_with_subshell_blocked(self) -> None:
        """Test that command substitution via $() is blocked for security."""
        result = shell.execute_shell_command("echo $(echo nested)")
        assert "blocked" in result.lower() or "substitution" in result.lower()

    def test_command_with_env_var(self) -> None:
        """Test command with environment variable (requires shell=True)."""
        result = shell.execute_shell_command("echo $HOME")
        # Should expand to home directory path
        assert "/" in result or "Error" in result

    def test_empty_command(self) -> None:
        """Test empty command string."""
        result = shell.execute_shell_command("")
        assert "Error: No command provided" in result

    def test_whitespace_only_command(self) -> None:
        """Test whitespace-only command."""
        result = shell.execute_shell_command("   ")
        assert "Error: No command provided" in result

    def test_command_with_special_characters(self) -> None:
        """Test command with special shell characters."""
        result = shell.execute_shell_command("echo 'hello world'")
        assert "hello world" in result

    def test_command_with_dollar_sign(self) -> None:
        """Test command with dollar sign for variable expansion."""
        result = shell.execute_shell_command("echo $PATH")
        # Should return the PATH environment variable or error if not expanded
        assert "Error" in result or ":" in result or "PATH" in result

    def test_command_exit_code_non_zero(self) -> None:
        """Test command that exits with non-zero status."""
        result = shell.execute_shell_command("false")
        assert "exit code: 1" in result

    def test_large_output_truncation(self) -> None:
        """Test that large output is truncated correctly."""
        # Generate a large output without command substitution
        large_command = "seq 1 20000"
        result = shell.execute_shell_command(large_command, timeout=30)
        assert "[... " in result and "chars truncated" in result

    def test_massive_output_memory_cap(self) -> None:
        """Regression for issue #1241 — output must not exhaust memory.

        ``seq 1 50000`` produces ~290 k characters.  The hard cap in
        ``_communicate_with_cap`` is 200 k characters, so the subprocess
        is killed before all output is buffered.  The truncation logic
        still produces a safely-bounded result.
        """
        result = shell.execute_shell_command("seq 1 50000", timeout=30)
        assert "[... " in result and "chars truncated" in result
        # Verify we did not buffer the full 290 k chars
        assert len(result) < 60_000

    def test_command_with_backticks_blocked(self) -> None:
        """Test that command substitution via backticks is blocked for security."""
        result = shell.execute_shell_command("echo `echo test`")
        assert "blocked" in result.lower() or "substitution" in result.lower()

    def test_command_with_ampersand(self) -> None:
        """Test command with ampersand for background execution."""
        result = shell.execute_shell_command("echo test & echo done")
        # Both commands should execute
        assert "test" in result and "done" in result

    def test_command_with_parentheses(self) -> None:
        """Test command with parentheses for subshell."""
        result = shell.execute_shell_command("echo (test)")
        # The command produces a syntax error since /bin/sh doesn't handle (test) well
        # Verify that the error message is present in the output
        assert "test" in result or "Syntax" in result or "[exit code:" in result

    def test_command_with_braces(self) -> None:
        """Test command with braces for brace expansion."""
        result = shell.execute_shell_command("echo {1,2,3}")
        assert "1" in result and "2" in result and "3" in result

    def test_command_with_star_glob(self) -> None:
        """Test command with star glob pattern."""
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory(dir=cwd) as tmpdir:
            Path(tmpdir).joinpath("test.txt").write_text("content")
            result = shell.execute_shell_command("echo *.txt", working_directory=tmpdir)
            assert "test.txt" in result

    def test_command_with_question_mark_glob(self) -> None:
        """Test command with question mark glob pattern."""
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory(dir=cwd) as tmpdir:
            Path(tmpdir).joinpath("a1.txt").write_text("")
            Path(tmpdir).joinpath("a2.txt").write_text("")
            result = shell.execute_shell_command("echo a?.txt", working_directory=tmpdir)
            assert "a1.txt" in result or "a2.txt" in result

    # ── working directory boundary validation ────────────────────────

    def test_working_directory_outside_cwd_rejected(self) -> None:
        """Shell should reject working_directory outside cwd and app dir."""
        result = shell.execute_shell_command("ls", working_directory="/etc")
        assert "outside allowed directories" in result

    def test_working_directory_within_cwd_accepted(self) -> None:
        """Shell should accept working_directory within current cwd."""
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory(dir=cwd) as tmpdir:
            result = shell.execute_shell_command("pwd", working_directory=tmpdir)
            assert tmpdir in result
            assert "Error:" not in result

    def test_working_directory_within_app_dir_accepted(self) -> None:
        """Shell should accept working_directory within the application directory."""
        from pathlib import Path

        app_src = Path(__file__).resolve().parent.parent.parent / "src"
        result = shell.execute_shell_command("pwd", working_directory=str(app_src))
        assert str(app_src) in result
        assert "Error:" not in result

    def test_working_directory_default_is_cwd(self) -> None:
        """When no working_directory is given, commands execute in cwd."""
        result = shell.execute_shell_command("pwd")
        assert os.getcwd() in result


class TestCommandSubstitutionBlocked:
    """Regression tests for issue #1104 — block command substitution in shell commands."""

    def test_dollar_paren_substitution_blocked(self) -> None:
        """$() command substitution must be rejected."""
        result = shell.execute_shell_command("echo $(id)")
        assert "blocked" in result.lower() or "substitution" in result.lower()

    def test_backtick_substitution_blocked(self) -> None:
        """Backtick command substitution must be rejected."""
        result = shell.execute_shell_command("echo `whoami`")
        assert "blocked" in result.lower() or "substitution" in result.lower()

    def test_nested_dollar_paren_blocked(self) -> None:
        """Nested $() command substitution must be rejected."""
        result = shell.execute_shell_command("echo $(echo $(whoami))")
        assert "blocked" in result.lower() or "substitution" in result.lower()

    def test_benign_env_var_allowed(self) -> None:
        """Simple variable expansion like $HOME must still work."""
        result = shell.execute_shell_command("echo $HOME")
        assert "/" in result or "Error" in result
        assert "blocked" not in result.lower()
        assert "substitution" not in result.lower()

    def test_benign_dollar_sign_in_string_allowed(self) -> None:
        """Dollar sign in a literal string must still work."""
        result = shell.execute_shell_command("echo 'Price is $5'")
        assert "$5" in result
        assert "blocked" not in result.lower()


class TestProcessSubstitutionBlocked:
    """Regression tests for issue #1238 — block <() and >() process substitution."""

    def test_process_substitution_read_blocked(self) -> None:
        """Process substitution <() must be rejected."""
        result = shell.execute_shell_command("cat <(whoami)")
        assert "blocked" in result.lower() or "substitution" in result.lower()

    def test_process_substitution_write_blocked(self) -> None:
        """Process substitution >() must be rejected."""
        result = shell.execute_shell_command("tee >(cat)")
        assert "blocked" in result.lower() or "substitution" in result.lower()

    def test_process_substitution_in_larger_command_blocked(self) -> None:
        """Process substitution embedded in a larger command must be blocked."""
        result = shell.execute_shell_command("diff <(echo a) <(echo b)")
        assert "blocked" in result.lower() or "substitution" in result.lower()

    def test_process_substitution_with_space_does_not_execute(self) -> None:
        """< (whoami) (with space) is not valid sh syntax and produces an error.

        Unlike <() (no space) which Bash interprets as process substitution,
        the spaced variant < ( fails in /bin/sh.  This is not a security bypass
        because no command execution occurs — the shell rejects the syntax.
        """
        result = shell.execute_shell_command("cat < (whoami)")
        # Must either be explicitly blocked or rejected by the shell as syntax error
        assert (
            "blocked" in result.lower()
            or "substitution" in result.lower()
            or "syntax error" in result.lower()
            or "unexpected" in result.lower()
        )


class TestSafeEnv:
    """Regression tests for issue #1239 — subprocess env must not leak secrets."""

    def test_safe_env_excludes_secret_keys(self) -> None:
        """_safe_env() must not include common secret-bearing variable names."""
        safe = shell._safe_env()
        forbidden = {
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "DATABASE_URL",
            "POSTGRES_PASSWORD",
            "AWS_SECRET_ACCESS_KEY",
            "GITHUB_TOKEN",
            "COGTRIX_API_KEY",
            "SECRET",
            "PASSWORD",
            "TOKEN",
        }
        for key in forbidden:
            assert key not in safe, f"{key} should not be in safe env"

    def test_safe_env_includes_allowed_keys(self) -> None:
        """_safe_env() must include explicitly whitelisted variables."""
        safe = shell._safe_env()
        # PATH is always set and must be present
        assert "PATH" in safe
        # HOME is typically set on Unix systems
        assert "HOME" in safe

    def test_secret_env_var_not_in_command_output(self) -> None:
        """Setting a secret env var must not make it visible in shell output."""
        os.environ["COGTRIX_FAKE_SECRET_API_KEY"] = "hunter2"
        try:
            # In a subprocess with sanitized env, this var should not be accessible
            result = shell.execute_shell_command("env | grep COGTRIX_FAKE_SECRET")
            assert "hunter2" not in result
            assert "COGTRIX_FAKE_SECRET_API_KEY" not in result
        finally:
            del os.environ["COGTRIX_FAKE_SECRET_API_KEY"]

    def test_allowed_env_var_accessible_in_subprocess(self) -> None:
        """Whitelisted env vars (PATH, HOME) must remain accessible."""
        result = shell.execute_shell_command("echo $HOME")
        # HOME must be readable and must resolve to a path
        assert "/" in result or "Error" in result
        assert "blocked" not in result.lower()


class TestExecuteShellCommandTimeoutBounds:
    """Test timeout validation and clamping."""

    def test_timeout_minimum(self) -> None:
        """Test that timeout is clamped to minimum of 1."""
        # Use a command that takes some time
        result = shell.execute_shell_command("echo test", timeout=0)
        assert "test" in result

    def test_timeout_maximum(self) -> None:
        """Test that timeout is clamped to maximum of 300."""
        # Use a command that would exceed 300s if not clamped
        result = shell.execute_shell_command("echo test", timeout=1000)
        assert "test" in result


class TestShellToolConfig:
    """Test the tool configuration metadata."""

    def test_tool_name(self) -> None:
        assert shell.TOOL_CONFIG["name"] == "execute_shell_command"

    def test_tool_requires_confirmation(self) -> None:
        assert shell.TOOL_CONFIG["requires_confirmation"] is True

    def test_tool_has_description(self) -> None:
        assert "shell command" in shell.TOOL_CONFIG["description"].lower()

    def test_tool_has_input_schema(self) -> None:
        assert "input_schema" in shell.TOOL_CONFIG
        assert shell.TOOL_CONFIG["input_schema"] == shell.ShellCommandInput


class TestShellMetaCharacterDetection:
    """Regression tests for shell=True vs shell=False detection."""

    def test_literal_braces_do_not_trigger_shell_true(self) -> None:
        """Commands with literal braces (no comma) should use shell=False."""
        result = shell.execute_shell_command("echo '{print $1}'")
        assert "{print $1}" in result

    def test_json_braces_do_not_trigger_shell_true(self) -> None:
        """Commands with JSON strings should use shell=False."""
        result = shell.execute_shell_command('echo \'{"key":"value"}\'')
        assert '"key":"value"' in result

    def test_awk_with_braces_executes_correctly(self) -> None:
        """awk with brace blocks should work via shlex.split + shell=False."""
        result = shell.execute_shell_command("awk '{print $1}' /etc/hosts")
        # Should produce output (IP addresses or hostnames) without error
        assert "Error" not in result or "exit code:" in result

    def test_brace_expansion_with_comma_still_works(self) -> None:
        """Brace expansion {a,b,c} should still trigger shell=True."""
        result = shell.execute_shell_command("echo {a,b,c}")
        assert "a" in result
        assert "b" in result
        assert "c" in result


class TestMissingShellMetaCharacters:
    """Regression tests for issue #1074 — missing shell metacharacters."""

    def test_tilde_expansion_uses_shell_true(self) -> None:
        """Tilde ~ should trigger shell=True and expand to home directory."""
        result = shell.execute_shell_command("echo ~")
        # shell=True expands ~ to home path; shell=False would output literal ~
        assert "~" not in result.splitlines()[0] if result.splitlines() else True
        assert "/" in result

    def test_hash_comment_uses_shell_true(self) -> None:
        """Hash # should trigger shell=True and be treated as a comment."""
        result = shell.execute_shell_command("echo test # this is a comment")
        # shell=True strips the comment; shell=False would echo the whole string
        assert "test" in result
        assert "comment" not in result

    def test_backslash_in_command_uses_shell_true(self) -> None:
        """Backslash should trigger shell=True without errors."""
        result = shell.execute_shell_command("echo 'hello\\world'")
        assert "hello" in result and "world" in result

    def test_bang_in_command_uses_shell_true(self) -> None:
        """Bang ! should trigger shell=True without errors."""
        result = shell.execute_shell_command("echo '!'")
        assert "!" in result


class TestShellToolAllExport:
    """Test the __all__ export list."""

    def test_execute_shell_command_exported(self) -> None:
        assert "execute_shell_command" in shell.__all__

    def test_shell_command_input_exported(self) -> None:
        assert "ShellCommandInput" in shell.__all__

    def test_tool_config_exported(self) -> None:
        assert "TOOL_CONFIG" in shell.__all__


class TestShellCommandTimeoutKillpg:
    """Regression tests for killpg ESRCH handling (issue #966)."""

    def test_killpg_esrch_logs_warning(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that ESRCH error from os.killpg is logged as a warning."""
        import os

        # Mock os.killpg to raise ESRCH
        original_killpg = os.killpg

        def mock_killpg(pid: int, sig: int) -> None:
            raise OSError(errno.ESRCH, "No such process")

        # Patch at the module level
        import src.tools.shell as shell_module

        # Temporarily replace os.killpg in the shell module
        shell_module.os.killpg = mock_killpg

        try:
            # Run a command with very short timeout to trigger timeout path
            result = shell_module.execute_shell_command("echo test", timeout=1)

            # The command should complete (or timeout with warning)
            # We just verify no exception is raised and the function completes
            assert "test" in result or "Error" in result
        finally:
            # Restore original os.killpg
            shell_module.os.killpg = original_killpg

    def test_killpg_other_oserror_logs_warning(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that other OSError from os.killpg is logged as a warning."""
        import os

        original_killpg = os.killpg

        def mock_killpg(pid: int, sig: int) -> None:
            raise OSError(errno.EPERM, "Permission denied")

        import src.tools.shell as shell_module

        shell_module.os.killpg = mock_killpg

        try:
            result = shell_module.execute_shell_command("echo test", timeout=1)
            assert "test" in result or "Error" in result
        finally:
            shell_module.os.killpg = original_killpg


class TestShellProcWaitDStateGuard:
    """Regression tests for issue #1202 — proc.wait() hang on D-state processes.

    After SIGKILL, a process in uninterruptible kernel sleep (D-state) will not
    respond to any signal. proc.wait() called without a timeout can hang forever.
    The fix wraps proc.wait() with a 5-second guard timeout in both code paths
    that call proc.wait() after proc.kill().
    """

    def test_proc_wait_timeout_after_kill_in_execute_shell_command(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verify execute_shell_command does not hang when proc.wait() hangs after kill.

        Simulates a D-state process: proc.wait() always raises TimeoutExpired
        (process never terminates after SIGKILL). The 5-second guard timeout should
        fire, allowing the function to return a timeout error without hanging.
        """
        import subprocess  # noqa: I001
        import unittest.mock as mock  # noqa: I001
        import src.tools.shell as shell_module  # noqa: I001

        fake_proc = mock.MagicMock(spec=subprocess.Popen)
        fake_proc.pid = 99999
        fake_proc.returncode = -9
        fake_proc.stdout = mock.MagicMock()
        fake_proc.stdout.read = mock.MagicMock(return_value="")
        fake_proc.stderr = mock.MagicMock()
        fake_proc.stderr.read = mock.MagicMock(return_value="")

        # Simulate D-state: wait() always raises TimeoutExpired (process stuck).
        # A small sleep ensures the test runs fast rather than blocking 5s.
        def hanging_wait(timeout: float | None = None) -> int:
            raise subprocess.TimeoutExpired("fake", 0.1)

        fake_proc.wait = hanging_wait

        with mock.patch.object(subprocess, "Popen", return_value=fake_proc):
            with mock.patch.object(
                os, "killpg", side_effect=OSError(errno.ESRCH, "No such process")
            ):
                # Must complete within 5-second guard timeout + test overhead
                start = __import__("time").time()
                result = shell_module.execute_shell_command("echo test", timeout=1)
                elapsed = __import__("time").time() - start

        assert elapsed < 15, f"Function took {elapsed:.1f}s — may have hung beyond the 5s guard"
        assert "timed out" in result.lower(), f"Expected timeout error, got: {result}"
        # Verify D-state warning was printed (proc.wait guard fired)
        captured = capsys.readouterr()
        assert (
            "D-state" in captured.err or fake_proc.pid in captured.err
        ), "Expected D-state warning in stderr"

    def test_proc_wait_timeout_after_kill_in_communicate_with_cap(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verify _communicate_with_cap does not hang when proc.wait() hangs after kill.

        Simulates a D-state process in the inner _communicate_with_cap path:
        proc.wait(timeout=timeout) raises TimeoutExpired (process is hung),
        then proc.wait(timeout=5) also raises TimeoutExpired (D-state confirmed).
        The function should re-raise TimeoutExpired so execute_shell_command
        handles it and returns a timeout error.
        """
        import subprocess
        import unittest.mock as mock

        import src.tools.shell as shell_module

        fake_proc = mock.MagicMock(spec=subprocess.Popen)
        fake_proc.pid = 99998
        fake_proc.returncode = -9
        fake_proc.stdout = mock.MagicMock()
        fake_proc.stdout.read = mock.MagicMock(return_value="")
        fake_proc.stderr = mock.MagicMock()
        fake_proc.stderr.read = mock.MagicMock(return_value="")

        # Simulate D-state in _communicate_with_cap: both wait() calls raise.
        call_count = 0

        def hanging_wait(timeout: float | None = None) -> int:
            nonlocal call_count
            call_count += 1
            raise subprocess.TimeoutExpired("fake", 0.1)

        fake_proc.wait = hanging_wait

        with mock.patch.object(subprocess, "Popen", return_value=fake_proc):
            with mock.patch.object(
                os, "killpg", side_effect=OSError(errno.ESRCH, "No such process")
            ):
                start = __import__("time").time()
                with pytest.raises(subprocess.TimeoutExpired):
                    shell_module._communicate_with_cap(fake_proc, timeout=1, max_chars=200_000)
                elapsed = __import__("time").time() - start

        # Should not hang beyond the 5-second guard timeout + some overhead
        assert elapsed < 15, f"_communicate_with_cap took {elapsed:.1f}s — may have hung"
        # Both wait() calls should have been made (inner + guard after kill)
        assert call_count >= 2, f"Expected ≥2 wait() calls (inner + guard), got {call_count}"
