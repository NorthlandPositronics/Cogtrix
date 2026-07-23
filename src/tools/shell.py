"""
Shell execution tool - executes shell commands with safety confirmation.
Enhanced with working directory and configurable timeout options.
"""

import os
import shlex
import subprocess  # nosec B404
from pathlib import Path

from pydantic import BaseModel, Field


class ShellCommandInput(BaseModel):
    """Input schema for shell command execution."""

    cmd: str = Field(
        description="The shell command to execute (e.g., 'ls -la', 'pwd', 'cat file.txt')"
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
    cmd: str,
    working_directory: str | None = None,
    timeout: int = 30,
) -> str:
    """
    Execute a shell command and return its output.

    WARNING: This tool can execute arbitrary shell commands. It requires
    user confirmation before execution (handled by the safety layer).

    Args:
        cmd: The shell command to execute
        working_directory: Directory to execute the command in (default: current directory)
        timeout: Command timeout in seconds (default: 30, max: 300)

    Returns:
        Command output (stdout) or error message (stderr)
    """
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
        needs_shell = any(ch in cmd for ch in _shell_meta)

        if needs_shell:
            # Use shell=True so pipes, redirects, etc. work correctly.
            # Safety is enforced by the confirmation prompt (requires_confirmation=True).
            result = subprocess.run(  # nosec B602
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                cwd=cwd,
                shell=True,  # nosec B602
            )
        else:
            # Simple command — use shlex.split for cleaner execution
            cmd_parts = shlex.split(cmd)
            result = subprocess.run(  # nosec B603
                cmd_parts,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,  # Don't raise on non-zero exit
                cwd=cwd,
            )

        # Combine stdout and stderr
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"

        # Include exit code if non-zero
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"

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
        if result.returncode != 0:
            return f"Command failed with no output (exit code: {result.returncode})"
        return f"Command executed successfully (exit code: {result.returncode})"

    except subprocess.TimeoutExpired:
        return f"Error: Command execution timed out after {timeout} seconds"
    except FileNotFoundError:
        cmd_name = cmd.split()[0] if cmd.split() else cmd
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
        "Supports custom working directory and timeout."
    ),
    "input_schema": ShellCommandInput,
    "requires_confirmation": True,  # Flagged as sensitive
}

__all__ = ["execute_shell_command", "ShellCommandInput", "TOOL_CONFIG"]
