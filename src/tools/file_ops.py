"""
File operations tool - Read, write, and manage files.
Write operations require user confirmation for safety.
"""

import collections
import os
import threading
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

# Application install directory — allows read access to project docs/source
# even when the working directory is set elsewhere (e.g., Docker with -w /tmp).
_APP_DIR: Path = Path(__file__).resolve().parent.parent.parent

_extra_write_dirs: list[Path] = []
_extra_read_dirs: list[Path] = []

# Wire COGTRIX_ALLOWED_WRITE_PATHS env var at import time.
# Comma-separated list of directories where write operations are allowed
# in addition to cwd.  Typically set in Dockerfile for container deployments
# where cwd (/app) is read-only but /tmp is writable.
_env_write_paths = os.environ.get("COGTRIX_ALLOWED_WRITE_PATHS", "")
if _env_write_paths:
    _extra_write_dirs = [
        Path(d.strip()).resolve() for d in _env_write_paths.split(",") if d.strip()
    ]


class _RefLock:
    """A threading.Lock paired with a reference count.

    The LRU registry increments ref_count while holding _append_lock_guard.
    __exit__ decrements ref_count after releasing the file lock.
    An entry is only evictable when ref_count == 0.
    """

    __slots__ = ("_lock", "_ref_lock", "ref_count")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ref_lock = threading.Lock()
        self.ref_count: int = 0

    def __enter__(self) -> "_RefLock":
        self._lock.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        self._lock.release()
        with self._ref_lock:
            self.ref_count -= 1


_append_lock_guard = threading.Lock()
_append_locks: collections.OrderedDict[str, _RefLock] = collections.OrderedDict()
_APPEND_LOCK_MAX = 256


def _get_append_lock(path: str) -> _RefLock:
    with _append_lock_guard:
        if path in _append_locks:
            _append_locks.move_to_end(path)
            ref_lock = _append_locks[path]
        else:
            ref_lock = _RefLock()
            _append_locks[path] = ref_lock
        with ref_lock._ref_lock:
            ref_lock.ref_count += 1
        n_seen = 0
        n_to_scan = len(_append_locks)
        while len(_append_locks) >= _APPEND_LOCK_MAX and n_seen < n_to_scan:
            key, lock_ref = next(iter(_append_locks.items()))
            with lock_ref._ref_lock:
                if lock_ref.ref_count > 0:
                    _append_locks.move_to_end(key)
                    n_seen += 1
                    continue
                del _append_locks[key]
        return ref_lock


def set_allowed_write_dirs(dirs: list[str] | None) -> None:
    """Configure additional directories where file write operations are allowed."""
    global _extra_write_dirs
    _extra_write_dirs = [Path(d).resolve() for d in (dirs or [])]


def set_allowed_read_dirs(dirs: list[str] | None) -> None:
    """Configure additional directories where file read operations are allowed."""
    global _extra_read_dirs
    _extra_read_dirs = [Path(d).resolve() for d in (dirs or [])]


class ReadFileInput(BaseModel):
    """Input schema for reading files."""

    path: str | dict = Field(description="Path to the file to read", alias="file_path")

    model_config = {"populate_by_name": True}  # accept both "path" and "file_path"
    encoding: str = Field(default="utf-8", description="File encoding (default: utf-8)")
    start_line: int = Field(
        default=0,
        ge=0,
        description="0-based line number to start reading from (default: 0)",
    )
    max_lines: int | None = Field(
        default=None,
        description="Maximum number of lines to read from start_line (None for all)",
    )


class WriteFileInput(BaseModel):
    """Input schema for writing files."""

    path: str | dict = Field(description="Path to the file to write")
    content: str = Field(description="Content to write to the file")
    encoding: str = Field(default="utf-8", description="File encoding (default: utf-8)")


class AppendFileInput(BaseModel):
    """Input schema for appending to files."""

    path: str | dict = Field(description="Path to the file to append to")
    content: str = Field(description="Content to append to the file")
    encoding: str = Field(default="utf-8", description="File encoding (default: utf-8)")


class ListDirectoryInput(BaseModel):
    """Input schema for listing directory contents."""

    path: str | dict = Field(default=".", description="Path to the directory to list")
    pattern: str = Field(default="*", description="Glob pattern to filter files (e.g., '*.py')")
    show_hidden: bool = Field(default=False, description="Whether to show hidden files")


class FileInfoInput(BaseModel):
    """Input schema for getting file information."""

    path: str | dict = Field(description="Path to the file or directory")


def _validate_path(path: str, is_write: bool = False) -> tuple[bool, str, Path | None]:
    """
    Validate a file path for safety.

    Returns:
        Tuple of (is_valid, error_message, resolved_path)
    """
    try:
        p = Path(path).resolve()
        cwd = Path.cwd().resolve()

        # Check for path traversal attempts
        if ".." in path:
            in_allowed = False
            try:
                p.relative_to(cwd)
                in_allowed = True
            except ValueError:
                pass
            if not in_allowed:
                try:
                    p.relative_to(_APP_DIR)
                    in_allowed = True
                except ValueError:
                    pass
            if not in_allowed:
                for extra_dir in _extra_write_dirs:
                    try:
                        p.relative_to(extra_dir)
                        in_allowed = True
                    except ValueError:
                        pass
            if not in_allowed:
                return False, "Path traversal not allowed", None

        # Paths must resolve within an allowed root directory.
        # Writes are restricted to cwd; reads also allow the app install directory
        # (so Docker users with -w /tmp can still read project docs at /app).
        in_cwd = False
        try:
            p.relative_to(cwd)
            in_cwd = True
        except ValueError:
            pass

        if not in_cwd:
            if is_write:
                for extra_dir in _extra_write_dirs:
                    try:
                        p.relative_to(extra_dir)
                        return True, "", p
                    except ValueError:
                        pass
                return False, "Write path must be within the working directory", None
            # Reads: allow app dir, extra write dirs, and extra read dirs
            in_allowed_read = False
            try:
                p.relative_to(_APP_DIR)
                in_allowed_read = True
            except ValueError:
                pass
            if not in_allowed_read:
                for extra_dir in _extra_write_dirs:
                    try:
                        p.relative_to(extra_dir)
                        in_allowed_read = True
                    except ValueError:
                        pass
            if not in_allowed_read:
                for extra_dir in _extra_read_dirs:
                    try:
                        p.relative_to(extra_dir)
                        in_allowed_read = True
                    except ValueError:
                        pass
            if not in_allowed_read:
                return False, "Path must be within the working directory", None

        return True, "", p
    except Exception as e:
        return False, f"Invalid path: {e}", None


def read_file(
    path: str,
    encoding: str = "utf-8",
    start_line: int = 0,
    max_lines: int | None = None,
) -> str:
    """
    Read the contents of a file.

    Args:
        path: Path to the file to read
        encoding: File encoding (default: utf-8)
        start_line: 0-based line number to start reading from (default: 0)
        max_lines: Maximum number of lines to read from start_line (None for all)

    Returns:
        File contents or error message
    """
    # Handle alternative parameter name from agent (e.g., absolute_path instead of path)
    if isinstance(path, dict):
        # Agent passed extra kwargs - extract actual path
        if "absolute_path" in path:
            path = path["absolute_path"]
        elif "file_path" in path:
            path = path["file_path"]
        elif "path" in path:
            path = path["path"]
        else:
            return "Error: Invalid arguments for read_file"

    is_valid, error, resolved = _validate_path(path)
    if not is_valid:
        return f"Error: {error}"

    if resolved is None:
        return "Error: Could not resolve path"

    if start_line < 0:
        return "Error: start_line must be >= 0"

    try:
        file_size = resolved.stat().st_size
        _MAX_READ_BYTES = 100 * 1024 * 1024  # 100 MB
        if file_size > _MAX_READ_BYTES:
            return (
                f"Error: File too large ({file_size / (1024 * 1024):.1f} MB). "
                f"Maximum readable size is {_MAX_READ_BYTES // (1024 * 1024)} MB."
            )

        with open(resolved, encoding=encoding) as f:
            all_lines = f.readlines()
        total_lines = len(all_lines)
        total_chars = sum(len(ln) for ln in all_lines)

        if start_line > 0 or max_lines is not None:
            if start_line >= total_lines:
                return (
                    f"Error: start_line {start_line} is beyond end of file "
                    f"({total_lines} lines)"
                )
            end = min(start_line + max_lines, total_lines) if max_lines is not None else total_lines
            selected = all_lines[start_line:end]
            header = (
                f"[File: {total_lines:,} lines, {total_chars:,} chars — "
                f"showing lines {start_line}-{end - 1} "
                f"({len(selected)} lines)]\n"
            )
            return header + "".join(selected)
        else:
            content = "".join(all_lines)
            _SAFETY_CAP = 512_000
            if len(content) > _SAFETY_CAP:
                half = _SAFETY_CAP // 2
                return (
                    f"[File: {total_lines:,} lines, {total_chars:,} chars "
                    f"— showing first and last {half:,} chars. "
                    f"Use start_line/max_lines to page through.]\n\n"
                    f"{content[:half]}\n\n"
                    f"[... {len(content) - _SAFETY_CAP:,} chars "
                    f"omitted ...]\n\n"
                    f"{content[-half:]}"
                )
            return content
    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except IsADirectoryError:
        return f"Error: Not a file: {path}"
    except UnicodeDecodeError:
        return f"Error: Could not decode file with encoding '{encoding}'. Try a different encoding."
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except OSError as e:
        return f"Error reading file: {e}"
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
    is_valid, error, resolved = _validate_path(path, is_write=True)
    if not is_valid:
        return f"Error: {error}"

    if resolved is None:
        return "Error: Could not resolve path"

    try:
        # Create parent directories if they don't exist
        resolved.parent.mkdir(parents=True, exist_ok=True)

        from src.utils.atomic_write import atomic_write_json

        with atomic_write_json(resolved, encoding=encoding) as f:
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
    is_valid, error, resolved = _validate_path(path, is_write=True)
    if not is_valid:
        return f"Error: {error}"

    if resolved is None:
        return "Error: Could not resolve path"

    try:
        # Create parent directories if they don't exist
        resolved.parent.mkdir(parents=True, exist_ok=True)

        lock = _get_append_lock(str(resolved))
        with lock:
            with open(resolved, "a", encoding=encoding) as f:
                f.write(content)

        return f"Successfully appended {len(content)} characters to {path}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error appending to file: {e}"


def list_directory(
    path: str | dict = ".",
    pattern: str = "*",
    show_hidden: bool | dict = False,
) -> str:
    """
    List contents of a directory.

    Args:
        path: Path to the directory
        pattern: Glob pattern to filter files (e.g., '*.py')
        show_hidden: Whether to show hidden files (starting with .)

    Returns:
        Formatted directory listing or error message
    """
    # Handle alternative parameter name from agent (e.g., absolute_path instead of path)
    if isinstance(path, dict):
        if "absolute_path" in path:
            path = path["absolute_path"]
        elif "path" in path:
            path = path["path"]
        else:
            return "Error: Invalid arguments for list_directory"
    # Handle show_hidden being passed as dict (shouldn't happen but be defensive)
    if isinstance(show_hidden, dict):
        show_hidden = show_hidden.get("show_hidden", False)

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
            if not item.resolve().is_relative_to(resolved):
                continue
            name = item.name

            # Skip hidden files unless requested
            if not show_hidden and name.startswith("."):
                continue

            if item.is_dir():
                entries.append(f"[DIR]  {name}/")
            else:
                size = item.lstat().st_size
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


class PatchFileInput(BaseModel):
    path: str | dict = Field(..., description="Path to the file to patch.")
    old_str: str = Field(..., description="Exact string to find in the file (must be unique).")
    new_str: str = Field(..., description="Replacement string.")


def patch_file(path: str, old_str: str, new_str: str) -> str:
    """
    Surgically replace an exact string in a file.

    Reads the file, finds ``old_str``, replaces it with ``new_str``, and
    writes the result back atomically.  Fails clearly when ``old_str`` is
    not found or appears more than once — add more surrounding context to
    resolve ambiguity.

    Args:
        path:    Path to the file to edit (must be inside an allowed write root).
        old_str: Exact string to search for.  Must occur exactly once.
        new_str: Replacement string (may be empty to delete the match).

    Returns:
        Success message or a descriptive error string.
    """
    is_valid, error, resolved = _validate_path(path, is_write=True)
    if not is_valid:
        return f"Error: {error}"

    if resolved is None:
        return "Error: Could not resolve path"

    if not resolved.exists():
        return f"Error: File not found: {path}"

    if not resolved.is_file():
        return f"Error: Not a file: {path}"

    try:
        content = resolved.read_text(encoding="utf-8")
    except PermissionError:
        return f"Error: Permission denied reading: {path}"
    except Exception as e:
        return f"Error reading file: {e}"

    count = content.count(old_str)
    if count == 0:
        return f"Error: old_str not found in {path}. " "Check for exact whitespace and indentation."
    if count > 1:
        return (
            f"Error: old_str found {count} times in {path} — ambiguous. "
            "Add more surrounding context to make the match unique."
        )

    new_content = content.replace(old_str, new_str, 1)

    try:
        from src.utils.atomic_write import atomic_write_json

        with atomic_write_json(resolved, encoding="utf-8") as f:
            f.write(new_content)
    except PermissionError:
        return f"Error: Permission denied writing: {path}"
    except Exception as e:
        return f"Error writing file: {e}"

    old_lines = content.count("\n") + 1
    new_lines = new_content.count("\n") + 1
    delta = new_lines - old_lines
    sign = "+" if delta >= 0 else ""
    return (
        f"Patched {path}: {old_lines} → {new_lines} lines ({sign}{delta}), "
        f"{len(content)} → {len(new_content)} chars."
    )


def file_info(path: str | dict) -> str:
    """
    Get detailed information about a file or directory.

    Args:
        path: Path to the file or directory

    Returns:
        Formatted file information or error message
    """
    # Handle alternative parameter name from agent (e.g., absolute_path instead of path)
    if isinstance(path, dict):
        if "absolute_path" in path:
            path = path["absolute_path"]
        elif "path" in path:
            path = path["path"]
        else:
            return "Error: Invalid arguments for file_info"

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
            "Large files are automatically truncated. "
            "Use start_line and max_lines to page through large files "
            "(e.g. start_line=0, max_lines=200 for the first 200 lines; "
            "start_line=200, max_lines=200 for the next page)."
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
        "name": "patch_file",
        "description": (
            "Surgically replace an exact string in a file. "
            "Finds old_str in the file (must appear exactly once) and replaces it with new_str. "
            "Fails if old_str is not found or is ambiguous — add more surrounding context. "
            "Prefer this over write_file for targeted edits to existing files."
        ),
        "input_schema": PatchFileInput,
        "requires_confirmation": True,
        "function": patch_file,
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
    "patch_file",
    "append_file",
    "list_directory",
    "file_info",
    "set_allowed_write_dirs",
    "set_allowed_read_dirs",
    "ReadFileInput",
    "WriteFileInput",
    "PatchFileInput",
    "AppendFileInput",
    "ListDirectoryInput",
    "FileInfoInput",
    "TOOL_CONFIG",
    "TOOL_CONFIGS",
]
