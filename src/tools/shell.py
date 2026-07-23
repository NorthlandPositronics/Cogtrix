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
from pathlib import Path

from pydantic import BaseModel, Field

# Application install directory — allows read/write access similar to file_ops.
_APP_DIR: Path = Path(__file__).resolve().parent.parent.parent

# Extra allowed directories, populated from file_ops configuration at import time
# and kept in sync via the setter functions.
_extra_allowed_dirs: list[Path] = []


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
            )

        try:
            stdout, stderr = proc.communicate(timeout=timeout)
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

        _SAFETY_CAP = 50_000
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
        return f"Error executing command ({type(e).__name__}): {e}"


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
