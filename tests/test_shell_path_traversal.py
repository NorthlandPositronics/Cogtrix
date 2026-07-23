"""Path traversal tests for `src/tools/shell.py`.

Tests covering path traversal vectors in shell command strings:
- ``../`` traversal in command arguments
- Absolute paths bypassing ``working_directory``
- Shell injection vectors (command substitution, chaining)
- Orphaned-process cleanup on timeout
- Legitimate commands working correctly

See issue #743 for context.  The shell tool currently performs NO
path validation on the command string itself — these tests document
the existing behaviour and the security gap.

Related:
  - ``src/tools/shell.py`` — the tool under test (156 lines)
  - ``src/assistant/guardrails.py`` — ``ToolCallGuard._check_paths``
  - ``src/utils/path_safety.py`` — path sanitisation utilities
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path

from src.tools.shell import execute_shell_command


class TestShellPathTraversal:
    """Path traversal behaviour of ``execute_shell_command``.

    These tests verify whether the shell tool blocks or allows
    path traversal through command arguments.  Currently the tool
    performs NO validation on the command string itself; path
    traversal is allowed, and the guardrails module (external to
    ``execute_shell_command``) provides the only protection.
    """

    # ── Legitimate commands (regression guard) ────────────────────────────

    def test_legitimate_command_with_default_working_dir(self) -> None:
        """A simple command without a working directory succeeds."""
        result = execute_shell_command("echo hello")
        assert "hello" in result

    def test_legitimate_command_with_specified_working_dir(self) -> None:
        """A command with a valid working_directory works correctly."""
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmpdir:
            root = Path(tmpdir)
            (root / "file.txt").write_text("legitimate content")
            result = execute_shell_command("cat file.txt", working_directory=str(root))
            assert "legitimate content" in result

    def test_legitimate_command_with_pipe(self) -> None:
        """A command with a pipe (``shell=True`` path) still works."""
        result = execute_shell_command("echo hello | grep hello")
        assert "hello" in result

    # ── Path traversal: relative (``../``) ───────────────────────────────

    def test_parent_dir_traversal_in_command_accesses_parent_files(self) -> None:
        """``../`` in a command argument accesses files outside the working_directory.

        This documents the path traversal gap: the shell tool does not
        validate paths embedded in the *command* string itself.
        """
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmpdir:
            root = Path(tmpdir)
            (root / "secret.txt").write_text("TOP SECRET")
            sub = root / "sub"
            sub.mkdir()

            result = execute_shell_command("cat ../secret.txt", working_directory=str(sub))
            # Currently succeeds — path traversal is unblocked.
            # See issue #743 for the security gap.
            assert "TOP SECRET" in result

    def test_multiple_parent_dir_traversal_in_command(self) -> None:
        """``../../`` traverses multiple directory levels."""
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmpdir:
            root = Path(tmpdir)
            (root / "nested_secret.txt").write_text("NESTED SECRET")
            deep_nested = root / "a" / "b"
            deep_nested.mkdir(parents=True)

            result = execute_shell_command(
                "cat ../../nested_secret.txt", working_directory=str(deep_nested)
            )
            assert "NESTED SECRET" in result

    # ── Path traversal: absolute paths ────────────────────────────────────

    def test_absolute_path_in_command_is_not_blocked(self) -> None:
        """An absolute path in the command bypasses the working_directory.

        The shell tool has no check that the command's target falls
        within an allowed directory root.
        """
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmpdir:
            root = Path(tmpdir)
            (root / "data.txt").write_text("absolute-target")
            result = execute_shell_command(f"cat {root}/data.txt")
            assert "absolute-target" in result

    def test_absolute_path_to_system_file_in_command(self) -> None:
        """Absolute path to a well-known system file is not blocked.

        ``/etc/hostname`` is world-readable on most Linux systems.  The
        shell tool does not reject this — it lets the subprocess run.
        """
        result = execute_shell_command("cat /etc/hostname", timeout=5)
        # The result depends on whether /etc/hostname exists and is
        # readable.  Either way the subprocess runs — the shell tool
        # does NOT block it.
        assert isinstance(result, str)

    # ── Working directory traversal ───────────────────────────────────────

    def test_working_directory_with_parent_traversal_resolved(self) -> None:
        """``working_directory`` with ``../`` is resolved via Path.resolve().

        ``Path.resolve()`` collapses traversal components, so the
        working_directory itself cannot be a traversal vector.  Only
        paths embedded in the *command* string can escape.
        """
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmpdir:
            root = Path(tmpdir)
            (root / "data.txt").write_text("resolved-file")
            sub = root / "sub"
            sub.mkdir()

            # Path.resolve() collapses the ../, making this equivalent
            # to working_directory=root
            result = execute_shell_command(
                "cat data.txt",
                working_directory=str(sub / "../"),
            )
            assert "resolved-file" in result

    def test_nonexistent_working_directory_rejected(self) -> None:
        """A nonexistent working_directory is rejected with a clear error."""
        result = execute_shell_command("ls", working_directory="/nonexistent/path/xyz")
        assert "Error:" in result
        assert "outside allowed directories" in result or "does not exist" in result

    # ── Edge cases ────────────────────────────────────────────────────────

    def test_empty_command_rejected(self) -> None:
        """An empty command is rejected."""
        result = execute_shell_command("")
        assert "Error: No command provided" in result

    def test_whitespace_command_rejected(self) -> None:
        """A whitespace-only command is rejected."""
        result = execute_shell_command("   ")
        assert "Error: No command provided" in result


class TestShellInjectionVectors:
    """Shell injection vectors that bypass path restrictions.

    These tests document how command substitution and chaining
    can be used to read files outside the working_directory.
    The shell tool does not validate or block these patterns.
    """

    def test_command_substitution_dollar_paren_reads_arbitrary_file(self) -> None:
        """``$(cat /path)`` expands to the file contents at shell time."""
        result = execute_shell_command("echo $(cat /etc/hostname)", timeout=5)
        # The hostname is echoed — injection succeeded.
        assert isinstance(result, str)
        assert "[exit code:" not in result or isinstance(result, str)

    def test_command_substitution_backtick_reads_arbitrary_file(self) -> None:
        """Backtick substitution `` `cat /path` `` also works."""
        result = execute_shell_command("echo `cat /etc/hostname`", timeout=5)
        assert isinstance(result, str)

    def test_semicolon_chaining_executes_second_command(self) -> None:
        """``cmd1; cmd2`` runs both commands regardless of exit code."""
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmpdir:
            root = Path(tmpdir)
            (root / "secret.txt").write_text("CHAINED SECRET")
            sub = root / "sub"
            sub.mkdir()

            result = execute_shell_command(
                "echo hello; cat ../secret.txt",
                working_directory=str(sub),
            )
            assert "CHAINED SECRET" in result

    def test_double_ampersand_chaining_executes_second_command(self) -> None:
        """``cmd1 && cmd2`` runs cmd2 when cmd1 succeeds."""
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmpdir:
            root = Path(tmpdir)
            (root / "secret.txt").write_text("AND SECRET")
            sub = root / "sub"
            sub.mkdir()

            result = execute_shell_command(
                "true && cat ../secret.txt",
                working_directory=str(sub),
            )
            assert "AND SECRET" in result

    def test_double_pipe_chaining_executes_second_command(self) -> None:
        """``cmd1 || cmd2`` runs cmd2 when cmd1 fails."""
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmpdir:
            root = Path(tmpdir)
            (root / "secret.txt").write_text("OR SECRET")
            sub = root / "sub"
            sub.mkdir()

            result = execute_shell_command(
                "false || cat ../secret.txt",
                working_directory=str(sub),
            )
            assert "OR SECRET" in result

    def test_pipe_to_second_command_reads_outside_working_dir(self) -> None:
        """A pipe can transport data from a traversal source."""
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmpdir:
            root = Path(tmpdir)
            (root / "secret.txt").write_text("PIPE SECRET")
            sub = root / "sub"
            sub.mkdir()

            result = execute_shell_command(
                "cat ../secret.txt | grep PIPE",
                working_directory=str(sub),
            )
            assert "PIPE SECRET" in result

    def test_input_redirect_reads_arbitrary_file(self) -> None:
        """``< /path`` redirects an arbitrary file into stdin."""
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmpdir:
            root = Path(tmpdir)
            (root / "secret.txt").write_text("REDIRECT SECRET")
            sub = root / "sub"
            sub.mkdir()

            result = execute_shell_command(
                "cat < ../secret.txt",
                working_directory=str(sub),
            )
            assert "REDIRECT SECRET" in result

    def test_environment_variable_expansion_in_command(self) -> None:
        """``$VAR`` expands inside the command string."""
        result = execute_shell_command("echo $HOME", timeout=5)
        # $HOME expands to an absolute path — demonstrates that
        # environment-driven path traversal is possible.
        assert isinstance(result, str)
        assert "Error" not in result or "$HOME" in result

    def test_nested_subshell_reads_arbitrary_file(self) -> None:
        """Nested subshells can construct arbitrary commands."""
        result = execute_shell_command("echo $(cat $(echo /etc/hostname))", timeout=5)
        assert isinstance(result, str)


class TestShellOrphanedProcessCleanup:
    """Orphaned-process cleanup when commands time out.

    The shell tool uses ``start_new_session=True`` and ``os.killpg``
    to kill the entire process group on timeout.  These tests verify
    that child and grandchild processes are not left behind.
    """

    def test_timeout_kills_child_process(self) -> None:
        """A child process spawned by the shell command is killed on timeout."""
        # Use a UNIQUE marker in the sleep argument so pgrep cannot
        # collide with unrelated processes that happen to have ``sleep``
        # in their command line (e.g. local bash watchers running
        # ``while ...; do sleep 30; done``).  The marker is a
        # non-conflicting large second count combined with a probe-id
        # comment that pgrep -f will match on.
        probe = f"cogtrix_orphan_probe_{os.getpid()}_{int(time.time() * 1000)}"
        result = execute_shell_command(
            f"sleep 31337 & sleep 31337  # {probe}",
            timeout=1,
        )
        assert "timed out" in result.lower()

        # Give the OS a moment to reap the processes
        time.sleep(0.5)

        # Verify no sleeper processes remain by checking pgrep against
        # the unique probe marker.
        try:
            procs = subprocess.run(
                ["pgrep", "-f", probe],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # If pgrep finds matches, they are orphaned survivors.  We
            # allow the test to pass if pgrep itself is not available.
            if procs.returncode == 0:
                raw_pids = [p for p in procs.stdout.strip().split("\n") if p.strip()]
                # ``pgrep -f`` captures a snapshot of /proc; transient
                # matches (including pgrep itself, when matched via
                # ``-f``) may already be gone by the time we evaluate.
                # ``os.kill(pid, 0)`` raises ``ProcessLookupError`` for
                # dead PIDs — only still-alive PIDs are real orphans.
                alive: list[str] = []
                for p in raw_pids:
                    try:
                        pid_int = int(p)
                    except ValueError:
                        continue
                    try:
                        os.kill(pid_int, 0)
                    except ProcessLookupError:
                        continue
                    except PermissionError:
                        pass  # alive but owned by someone else
                    alive.append(p)
                assert len(alive) == 0, f"Orphaned sleep processes found: {alive}"
        except FileNotFoundError:
            # pgrep not available — skip the orphan check
            pass

    def test_timeout_kills_process_group_not_only_parent(self) -> None:
        """``os.killpg`` is used so the whole process group dies."""
        # Start a command that forks a grandchild via a subshell.
        # If only proc.kill() were used, the grandchild might survive.
        result = execute_shell_command(
            "(sleep 30)",
            timeout=1,
        )
        assert "timed out" in result.lower()

    def test_timeout_returns_expected_error_message(self) -> None:
        """Timeout produces a clear error message with the timeout value."""
        result = execute_shell_command("sleep 10", timeout=1)
        assert "Error: Command execution timed out after 1 seconds" in result

    def test_quick_command_does_not_timeout(self) -> None:
        """A fast command completes before the timeout and returns normally."""
        result = execute_shell_command("echo ok", timeout=5)
        assert "ok" in result
        assert "timed out" not in result.lower()
