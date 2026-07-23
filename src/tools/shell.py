"""
Shell execution tool - executes shell commands with safety confirmation.
Enhanced with working directory and configurable timeout options.
"""

import os
import shlex
import signal
import subprocess  # nosec B404
from pathlib import Path

from pydantic import BaseModel, Field


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

    # Validate and clamp timeout
    timeout = min(max(1, timeout), 300)

    # Validate working directory
    cwd = None
    if working_directory:
        working_directory = str(Path(working_directory).resolve())
        if not os.path.isdir(working_directory):
            return f"Error: Working directory does not exist: {working_directory}"
        cwd = working_directory

    try:
        # Detect shell metacharacters that require a real shell to interpret
        # (pipes, redirects, chaining, subshells, globs, env vars, etc.)
        _shell_meta = {"|", ">", "<", "&", ";", "`", "$", "(", ")", "*", "?", "{", "}"}
        needs_shell = any(ch in command for ch in _shell_meta)

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
            cmd_parts = shlex.split(command)
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
            except OSError:
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
    except Exception as e:
        return f"Error executing command: {str(e)}"


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
