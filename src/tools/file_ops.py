"""
File operations tool - Read, write, and manage files.
Write operations require user confirmation for safety.
"""

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field


class ReadFileInput(BaseModel):
    """Input schema for reading files."""

    path: str = Field(description="Path to the file to read")
    encoding: str = Field(default="utf-8", description="File encoding (default: utf-8)")
    max_lines: int | None = Field(
        default=None,
        description="Maximum number of lines to read (None for all)",
    )


class WriteFileInput(BaseModel):
    """Input schema for writing files."""

    path: str = Field(description="Path to the file to write")
    content: str = Field(description="Content to write to the file")
    encoding: str = Field(default="utf-8", description="File encoding (default: utf-8)")


class AppendFileInput(BaseModel):
    """Input schema for appending to files."""

    path: str = Field(description="Path to the file to append to")
    content: str = Field(description="Content to append to the file")
    encoding: str = Field(default="utf-8", description="File encoding (default: utf-8)")


class ListDirectoryInput(BaseModel):
    """Input schema for listing directory contents."""

    path: str = Field(default=".", description="Path to the directory to list")
    pattern: str = Field(default="*", description="Glob pattern to filter files (e.g., '*.py')")
    show_hidden: bool = Field(default=False, description="Whether to show hidden files")


class FileInfoInput(BaseModel):
    """Input schema for getting file information."""

    path: str = Field(description="Path to the file or directory")


def _validate_path(path: str) -> tuple[bool, str, Path | None]:
    """
    Validate a file path for safety.

    Returns:
        Tuple of (is_valid, error_message, resolved_path)
    """
    try:
        p = Path(path).resolve()

        # Check for path traversal attempts
        if ".." in path:
            # Reject any path with ".." that resolves outside cwd
            cwd = Path.cwd().resolve()
            try:
                p.relative_to(cwd)
            except ValueError:
                return False, "Path traversal not allowed", None

        return True, "", p
    except Exception as e:
        return False, f"Invalid path: {e}", None


def read_file(path: str, encoding: str = "utf-8", max_lines: int | None = None) -> str:
    """
    Read the contents of a file.

    Args:
        path: Path to the file to read
        encoding: File encoding (default: utf-8)
        max_lines: Maximum number of lines to read (None for all)

    Returns:
        File contents or error message
    """
    is_valid, error, resolved = _validate_path(path)
    if not is_valid:
        return f"Error: {error}"

    if resolved is None:
        return "Error: Could not resolve path"

    if not resolved.exists():
        return f"Error: File not found: {path}"

    if not resolved.is_file():
        return f"Error: Not a file: {path}"

    try:
        with open(resolved, encoding=encoding) as f:
            if max_lines is not None:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        lines.append(f"\n... (truncated after {max_lines} lines)")
                        break
                    lines.append(line)
                return "".join(lines)
            else:
                content = f.read()
                # Warn if file is very large
                if len(content) > 100000:
                    return (
                        f"[File is {len(content)} characters]\n\n"
                        f"{content[:50000]}\n\n... (truncated, file too large)"
                    )
                return content
    except UnicodeDecodeError:
        return f"Error: Could not decode file with encoding '{encoding}'. Try a different encoding."
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(path: str, content: str, encoding: str = "utf-8") -> str:
    """
    Write content to a file. Creates the file if it doesn't exist.
    WARNING: This will overwrite existing files.

    Args:
        path: Path to the file to write
        content: Content to write
        encoding: File encoding (default: utf-8)

    Returns:
        Success or error message
    """
    is_valid, error, resolved = _validate_path(path)
    if not is_valid:
        return f"Error: {error}"

    if resolved is None:
        return "Error: Could not resolve path"

    try:
        # Create parent directories if they don't exist
        resolved.parent.mkdir(parents=True, exist_ok=True)

        with open(resolved, "w", encoding=encoding) as f:
            f.write(content)

        return f"Successfully wrote {len(content)} characters to {path}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def append_file(path: str, content: str, encoding: str = "utf-8") -> str:
    """
    Append content to a file. Creates the file if it doesn't exist.

    Args:
        path: Path to the file to append to
        content: Content to append
        encoding: File encoding (default: utf-8)

    Returns:
        Success or error message
    """
    is_valid, error, resolved = _validate_path(path)
    if not is_valid:
        return f"Error: {error}"

    if resolved is None:
        return "Error: Could not resolve path"

    try:
        # Create parent directories if they don't exist
        resolved.parent.mkdir(parents=True, exist_ok=True)

        with open(resolved, "a", encoding=encoding) as f:
            f.write(content)

        return f"Successfully appended {len(content)} characters to {path}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error appending to file: {e}"


def list_directory(path: str = ".", pattern: str = "*", show_hidden: bool = False) -> str:
    """
    List contents of a directory.

    Args:
        path: Path to the directory
        pattern: Glob pattern to filter files (e.g., '*.py')
        show_hidden: Whether to show hidden files (starting with .)

    Returns:
        Formatted directory listing or error message
    """
    is_valid, error, resolved = _validate_path(path)
    if not is_valid:
        return f"Error: {error}"

    if resolved is None:
        return "Error: Could not resolve path"

    if not resolved.exists():
        return f"Error: Directory not found: {path}"

    if not resolved.is_dir():
        return f"Error: Not a directory: {path}"

    try:
        entries = []
        for item in sorted(resolved.glob(pattern)):
            name = item.name

            # Skip hidden files unless requested
            if not show_hidden and name.startswith("."):
                continue

            if item.is_dir():
                entries.append(f"[DIR]  {name}/")
            else:
                size = item.stat().st_size
                if size < 1024:
                    size_str = f"{size} B"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size / (1024 * 1024):.1f} MB"
                entries.append(f"[FILE] {name} ({size_str})")

        if not entries:
            return f"Directory '{path}' is empty or no files match pattern '{pattern}'"

        return f"Contents of {path}:\n" + "\n".join(entries)
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error listing directory: {e}"


def file_info(path: str) -> str:
    """
    Get detailed information about a file or directory.

    Args:
        path: Path to the file or directory

    Returns:
        Formatted file information or error message
    """
    is_valid, error, resolved = _validate_path(path)
    if not is_valid:
        return f"Error: {error}"

    if resolved is None:
        return "Error: Could not resolve path"

    if not resolved.exists():
        return f"Error: Path not found: {path}"

    try:
        stat = resolved.stat()

        info = []
        info.append(f"Path: {resolved}")
        info.append(f"Type: {'Directory' if resolved.is_dir() else 'File'}")
        info.append(f"Size: {stat.st_size} bytes")

        # Format times (tz-aware to avoid DeprecationWarning on Python 3.12+)
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
        atime = datetime.fromtimestamp(stat.st_atime, tz=UTC)
        ctime = datetime.fromtimestamp(stat.st_ctime, tz=UTC)

        info.append(f"Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        info.append(f"Accessed: {atime.strftime('%Y-%m-%d %H:%M:%S')}")
        info.append(f"Created: {ctime.strftime('%Y-%m-%d %H:%M:%S')}")

        # Permissions (Unix-style)
        mode = stat.st_mode
        info.append(f"Permissions: {oct(mode)[-3:]}")

        # File-specific info
        if resolved.is_file():
            info.append(f"Extension: {resolved.suffix or '(none)'}")

            # Try to count lines for text files
            try:
                with open(resolved, encoding="utf-8") as f:
                    lines = sum(1 for _ in f)
                info.append(f"Lines: {lines}")
            except Exception:  # noqa: BLE001  # nosec B110
                pass

        return "\n".join(info)
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error getting file info: {e}"


# Tool configurations for registry
TOOL_CONFIGS = [
    {
        "name": "read_file",
        "description": (
            "Read the contents of a file. "
            "Use this to view file contents, configuration files, source code, etc."
        ),
        "input_schema": ReadFileInput,
        "requires_confirmation": False,
        "function": read_file,
    },
    {
        "name": "write_file",
        "description": (
            "Write content to a file. Creates the file if it doesn't exist. "
            "WARNING: This will overwrite existing files."
        ),
        "input_schema": WriteFileInput,
        "requires_confirmation": True,  # Requires confirmation for safety
        "function": write_file,
    },
    {
        "name": "append_file",
        "description": (
            "Append content to the end of a file. Creates the file if it doesn't exist."
        ),
        "input_schema": AppendFileInput,
        "requires_confirmation": True,  # Requires confirmation for safety
        "function": append_file,
    },
    {
        "name": "list_directory",
        "description": (
            "List the contents of a directory. Shows files and subdirectories with sizes."
        ),
        "input_schema": ListDirectoryInput,
        "requires_confirmation": False,
        "function": list_directory,
    },
    {
        "name": "file_info",
        "description": (
            "Get detailed information about a file or directory "
            "(size, dates, permissions, etc.)."
        ),
        "input_schema": FileInfoInput,
        "requires_confirmation": False,
        "function": file_info,
    },
]

# Default single tool config (for backwards compatibility)
TOOL_CONFIG = TOOL_CONFIGS[0]

__all__ = [
    "read_file",
    "write_file",
    "append_file",
    "list_directory",
    "file_info",
    "ReadFileInput",
    "WriteFileInput",
    "AppendFileInput",
    "ListDirectoryInput",
    "FileInfoInput",
    "TOOL_CONFIG",
    "TOOL_CONFIGS",
]
