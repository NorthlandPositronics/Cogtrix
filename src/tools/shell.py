"""
Shell execution tool - executes shell commands with safety confirmation.
Enhanced with working directory and configurable timeout options.
"""

import errno
import os
import re
import shlex
import signal
import subprocess  # nosec B404
import sys
import threading
from pathlib import Path

from pydantic import BaseModel, Field

from src.tools.error_sanitizer import sanitize_shell_error as _sanitize_shell_error

# Application install directory — allows read/write access similar to file_ops.
_APP_DIR: Path = Path(__file__).resolve().parent.parent.parent

# Extra allowed directories, populated from file_ops configuration at import time
# and kept in sync via the setter functions.
_extra_allowed_dirs: list[Path] = []

# Whitelisted env vars passed to subprocess — excludes all secret-bearing vars.
# Issue #1239.
_ALLOWED_ENV_KEYS = frozenset({"PATH", "HOME", "TERM", "LANG", "LC_ALL", "PWD", "USER"})


def _safe_env() -> dict[str, str]:
    """Return a sanitized environment dict for subprocess execution.

    Only whitelisted, non-secret variables are included.  This prevents API keys,
    database credentials, and other secrets inherited from the parent process
    from leaking into shell command output or being accessible to the child.
    """
    return {k: v for k, v in os.environ.items() if k in _ALLOWED_ENV_KEYS}


def _resolve_allowed_dirs() -> list[Path]:
    """Return the current list of allowed root directories for shell operations."""
    cwd = Path.cwd()
    dirs: list[Path] = [cwd, _APP_DIR]
    # Import locally to avoid circular dependency at module level
    from src.tools.file_ops import _extra_read_dirs, _extra_write_dirs  # noqa: PLC0415

    try:
        dirs.extend(_extra_write_dirs)
    except Exception:
        pass
    try:
        dirs.extend(_extra_read_dirs)
    except Exception:
        pass
    dirs.extend(_extra_allowed_dirs)
    return dirs


def _validate_working_directory(path: str) -> tuple[bool, str, Path | None]:
    """Validate that a working directory path is within allowed boundaries.

    Mirrors the directory containment logic from file_ops._validate_path
    but without file-specific checks (no symlink or ".." traversal blocking
    since shell operations naturally span the filesystem).
    """
    try:
        resolved = Path(path).resolve()
    except (OSError, PermissionError, RuntimeError):
        return False, f"Cannot resolve path: {path}", None

    allowed = _resolve_allowed_dirs()
    for root in allowed:
        try:
            resolved.relative_to(root)
            return True, "", resolved
        except ValueError:
            continue

    return (
        False,
        (
            f"Working directory '{path}' is outside allowed directories. "
            f"Shell operations are restricted to the current working directory "
            f"and application directory."
        ),
        None,
    )


def _communicate_with_cap(
    proc: subprocess.Popen,
    timeout: int,
    max_chars: int,
) -> tuple[str, str]:
    """Read stdout/stderr with a hard character cap to avoid memory exhaustion.

    Uses background threads to drain both pipes concurrently so the subprocess
    never deadlocks because one buffer is full.  Once *max_chars* have been
    accumulated the process group is killed and the remaining data is discarded.
    If the subprocess does not finish within *timeout* seconds,
    ``subprocess.TimeoutExpired`` is raised (mirroring ``Popen.communicate``).
    """
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    total = 0
    cap_hit = False
    lock = threading.Lock()

    def _drain(pipe: subprocess.PIPE, chunks: list[str]) -> None:  # type: ignore[type-arg]
        nonlocal total, cap_hit
        while True:
            chunk = pipe.read(4096)
            if not chunk:
                break
            with lock:
                if not cap_hit:
                    chunks.append(chunk)
                    total += len(chunk)
                    if total > max_chars:
                        cap_hit = True
                        # Stop the producer — kill the whole process group.
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except OSError:
                            try:
                                proc.kill()
                            except OSError:
                                pass
                # After the cap is hit we keep reading (but discard) so the
                # pipe does not deadlock.

    t_out = threading.Thread(target=_drain, args=(proc.stdout, stdout_chunks))
    t_err = threading.Thread(target=_drain, args=(proc.stderr, stderr_chunks))
    t_out.start()
    t_err.start()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Replicate communicate() behaviour — kill and re-raise.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError as exc:
            if exc.errno != errno.ESRCH:
                print(
                    f"Warning: os.killpg failed for process group {proc.pid}: {exc}",
                    file=sys.stderr,
                )
            proc.kill()
        proc.wait()
        raise

    t_out.join(timeout=5)
    t_err.join(timeout=5)

    return "".join(stdout_chunks), "".join(stderr_chunks)


class ShellCommandInput(BaseModel):
    """Input schema for shell command execution."""

    command: str = Field(
        description="The shell command to execute (e.g., 'ls -la', 'pwd', 'cat file.txt')",
    )
    working_directory: str | None = Field(
        default=None,
        description="Directory to execute the command in (default: current directory)",
    )
    timeout: int = Field(
        default=30,
        description="Command timeout in seconds (default: 30, max: 300)",
    )


def execute_shell_command(
    command: str,
    working_directory: str | None = None,
    timeout: int = 30,
) -> str:
    """
    Execute a shell command and return its output.

    WARNING: This tool can execute arbitrary shell commands. It requires
    user confirmation before execution (handled by the safety layer).

    Args:
        command: The shell command to execute
        working_directory: Directory to execute the command in (default: current directory)
        timeout: Command timeout in seconds (default: 30, max: 300)

    Returns:
        Command output (stdout) or error message (stderr)
    """
    if not command or not command.strip():
        return "Error: No command provided."

    # Block command-substitution syntax that can embed arbitrary code execution
    # (issue #1104). Variable expansion ($VAR) is still allowed.
    if "$(" in command:
        return (
            "Error: Command substitution via $() is blocked for security. "
            "Use a safe alternative or split the command into separate steps."
        )
    if "`" in command:
        return (
            "Error: Command substitution via backticks is blocked for security. "
            "Use a safe alternative or split the command into separate steps."
        )
    if "<(" in command or ">(" in command:
        return (
            "Error: Command substitution via <() or >() process substitution is blocked for security. "
            "Use a safe alternative or split the command into separate steps."
        )

    # Validate and clamp timeout
    timeout = min(max(1, timeout), 300)

    # Validate working directory
    cwd = None
    if working_directory:
        is_valid, error, resolved = _validate_working_directory(working_directory)
        if not is_valid:
            return f"Error: {error}"
        if resolved is None:
            return "Error: Could not resolve working directory"
        if not resolved.is_dir():
            return f"Error: Working directory does not exist: {resolved}"
        cwd = str(resolved)

    try:
        # Detect shell metacharacters that require a real shell to interpret
        # (pipes, redirects, chaining, subshells, globs, env vars, etc.)
        _shell_meta = {"|", ">", "<", "&", ";", "`", "$", "(", ")", "*", "?", "~", "\\", "!", "#"}
        needs_shell = any(ch in command for ch in _shell_meta) or bool(
            re.search(r"\{[^}]*,[^}]*\}", command)
        )

        if needs_shell:
            # Use shell=True so pipes, redirects, etc. work correctly.
            # Safety is enforced by the confirmation prompt (requires_confirmation=True).
            proc = subprocess.Popen(  # nosec B602
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd,
                shell=True,  # nosec B602
                start_new_session=True,
                env=_safe_env(),  # nosec B605 — _safe_env strips secrets
            )
        else:
            # Simple command — use shlex.split for cleaner execution
            try:
                cmd_parts = shlex.split(command)
            except ValueError:
                return "Error: Malformed command — unbalanced quotes or unsupported shell syntax."
            proc = subprocess.Popen(  # nosec B603
                cmd_parts,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd,
                start_new_session=True,
                env=_safe_env(),
            )

        _SAFETY_CAP = 50_000
        _HARD_CAP = _SAFETY_CAP * 4  # 200 k chars — enough for truncation logic

        try:
            stdout, stderr = _communicate_with_cap(proc, timeout, _HARD_CAP)
        except subprocess.TimeoutExpired:
            # Kill the entire process group so grandchild processes are cleaned up
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError as e:
                # ESRCH means the process group doesn't exist anymore
                if e.errno == errno.ESRCH:
                    print(
                        f"Warning: Process group {proc.pid} no longer exists (ESRCH). "
                        "Grandchild processes may still be running.",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"Warning: os.killpg failed for process group {proc.pid}: {e}",
                        file=sys.stderr,
                    )
                # Fall back to killing just the main process
                proc.kill()
            proc.wait()
            return f"Error: Command execution timed out after {timeout} seconds"

        # Combine stdout and stderr
        output = stdout
        if stderr:
            output += f"\n[stderr]\n{stderr}"

        # Include exit code if non-zero
        if proc.returncode != 0:
            output += f"\n[exit code: {proc.returncode}]"

        if len(output) > _SAFETY_CAP:
            half = _SAFETY_CAP // 2
            output = (
                output[:half]
                + f"\n\n[... {len(output) - _SAFETY_CAP:,} chars truncated ...]\n\n"
                + output[-half:]
            )

        if output.strip():
            return output
        if proc.returncode != 0:
            return f"Command failed with no output (exit code: {proc.returncode})"
        return f"Command executed successfully (exit code: {proc.returncode})"

    except FileNotFoundError:
        cmd_name = command.split()[0] if command.split() else command
        return f"Error: Command not found: {cmd_name}"
    except PermissionError:
        return "Error: Permission denied executing command"
    except Exception as e:  # noqa: BLE001
        return f"Error executing command: {_sanitize_shell_error(e)}"


# Tool metadata for registry
TOOL_CONFIG = {
    "name": "execute_shell_command",
    "description": (
        "Execute a shell command on the system. Use this to run terminal commands like "
        "'ls', 'pwd', 'cat file.txt', 'git status', etc. "
        "Set timeout appropriately for the command: quick commands ~10s, "
        "downloads/builds/installs 120–300s. Default is 30s — commands that "
        "exceed it are killed. Do NOT retry a timed-out command with the same "
        "timeout; increase it instead."
    ),
    "input_schema": ShellCommandInput,
    "requires_confirmation": True,  # Flagged as sensitive
}

__all__ = ["execute_shell_command", "ShellCommandInput", "TOOL_CONFIG"]
