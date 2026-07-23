"""Git operations tools.

Provides read-only inspection (status, diff, log) and write operations
(add, commit, create-branch, checkout) for git repositories.

Read-only tools run without confirmation.  Write tools require confirmation.
All commands run in ``os.getcwd()`` via ``subprocess`` with list args (no
shell interpolation) so tool argument values cannot inject shell commands.
"""

from __future__ import annotations

import os
import re as _re
import subprocess
from typing import Any

from pydantic import BaseModel, Field

from cogtrix_core.tools.delegate import register_tool_categories

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_GIT = "git"
_TIMEOUT = 60  # seconds for any single git command


def _run_git(*args: str, cwd: str | None = None) -> str:
    """Run a git subcommand and return combined stdout+stderr as a string."""
    cmd = [_GIT, *args]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            cwd=cwd or os.getcwd(),
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += result.stderr
        if result.returncode != 0 and not output.strip():
            output = f"git exited with code {result.returncode}"
        return output.rstrip() or "(no output)"
    except FileNotFoundError:
        return "Error: git is not installed or not on PATH."
    except subprocess.TimeoutExpired:
        return f"Error: git command timed out after {_TIMEOUT} s."
    except Exception as exc:
        return f"Error running git: {exc}"


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


class GitStatusInput(BaseModel):
    pass  # no parameters needed


class GitDiffInput(BaseModel):
    path: str = Field(
        default="",
        description="Specific file or directory to diff. Leave empty to diff all changes.",
    )
    staged: bool = Field(
        default=False,
        description="Show staged (cached) diff instead of unstaged working-tree diff.",
    )


class GitLogInput(BaseModel):
    max_count: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of commits to show (1–100).",
    )
    branch: str = Field(
        default="",
        description="Branch or ref to show history for. Defaults to current branch.",
    )


class GitAddInput(BaseModel):
    paths: list[str] = Field(
        ...,
        min_length=1,
        description="One or more file paths to stage. Use ['.'] to stage all changes.",
    )


class GitCommitInput(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Commit message.",
    )


class GitCreateBranchInput(BaseModel):
    name: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="New branch name.",
    )
    base: str = Field(
        default="HEAD",
        description="Starting point for the new branch (branch name, tag, or commit hash).",
    )


class GitCheckoutInput(BaseModel):
    ref: str = Field(
        ...,
        min_length=1,
        description=(
            "Branch name, commit hash, or file path to restore. "
            "To restore a file to HEAD: pass the file path. "
            "To switch branches: pass the branch name."
        ),
    )


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


def git_status() -> str:
    """Show the working tree status (staged, unstaged, and untracked files)."""
    return _run_git("status", "--short", "--branch")


def git_diff(path: str = "", staged: bool = False) -> str:
    """Show changes between the working tree and the index (or staged changes)."""
    args: list[str] = ["diff"]
    if staged:
        args.append("--cached")
    args.extend(["--stat", "--patch"])
    if path:
        args.extend(["--", path])
    return _run_git(*args)


_SAFE_REF_RE = _re.compile(r"^[a-zA-Z0-9_./@#\-][a-zA-Z0-9_./@#\-]*$")


def _validate_ref(ref: str) -> str | None:
    """Reject refs that could inject shell metacharacters or option flags."""
    if not ref:
        return "Error: ref name must not be empty"
    if ref.startswith("-"):
        return f"Error: invalid ref name — must not start with '-': {ref}"
    # Block shell metacharacters and whitespace that could escape subprocess quoting
    if not _SAFE_REF_RE.match(ref):
        bad_chars = "".join(sorted({c for c in ref if not _SAFE_REF_RE.match(c)}))
        return (
            f"Error: invalid ref name '{ref}' — contains unsafe characters: {bad_chars!r}. "
            "Only alphanumerics, '.', '/', '@', '#', '_', '-' are allowed."
        )
    return None


def git_log(max_count: int = 10, branch: str = "") -> str:
    """Show recent commits with author, date, and subject."""
    args = [
        "log",
        f"--max-count={max_count}",
        "--pretty=format:%h %ad %an — %s",
        "--date=short",
    ]
    if branch:
        if err := _validate_ref(branch):
            return err
        args.append(branch)
    return _run_git(*args)


def git_add(paths: list[str]) -> str:
    """Stage the specified files for the next commit."""
    return _run_git("add", "--", *paths)


def git_commit(message: str) -> str:
    """Create a commit with all staged changes."""
    return _run_git("commit", "--message", message)


def git_create_branch(name: str, base: str = "HEAD") -> str:
    """Create a new branch from the given base and switch to it."""
    if err := _validate_ref(name):
        return err
    if err := _validate_ref(base):
        return err
    return _run_git("checkout", "-b", name, base)


def git_checkout(ref: str) -> str:
    """Switch to a branch or restore a file to its last committed state."""
    if err := _validate_ref(ref):
        return err
    return _run_git("checkout", ref)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOL_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "git_status",
        "description": (
            "Show the working tree status: staged changes, unstaged changes, "
            "and untracked files. Safe — read-only."
        ),
        "input_schema": GitStatusInput,
        "requires_confirmation": False,
        "function": git_status,
        "category": "readonly",
    },
    {
        "name": "git_diff",
        "description": (
            "Show a unified diff of changes. Pass staged=true to see staged changes; "
            "pass a file path to narrow the diff to one file. Safe — read-only."
        ),
        "input_schema": GitDiffInput,
        "requires_confirmation": False,
        "function": git_diff,
        "category": "readonly",
    },
    {
        "name": "git_log",
        "description": (
            "Show recent commit history with hash, date, author, and subject. "
            "max_count controls how many commits to show (default 10). Safe — read-only."
        ),
        "input_schema": GitLogInput,
        "requires_confirmation": False,
        "function": git_log,
        "category": "readonly",
    },
    {
        "name": "git_add",
        "description": (
            "Stage one or more files for the next commit. "
            "Pass paths=['.'] to stage all changes in the working tree."
        ),
        "input_schema": GitAddInput,
        "requires_confirmation": True,
        "function": git_add,
        "category": "mutation",
    },
    {
        "name": "git_commit",
        "description": "Create a commit from all currently staged changes with the given message.",
        "input_schema": GitCommitInput,
        "requires_confirmation": True,
        "function": git_commit,
        "category": "mutation",
    },
    {
        "name": "git_create_branch",
        "description": (
            "Create a new branch from a given base (default HEAD) and switch to it. "
            "Useful for isolating a fix attempt before committing."
        ),
        "input_schema": GitCreateBranchInput,
        "requires_confirmation": True,
        "function": git_create_branch,
        "category": "mutation",
    },
    {
        "name": "git_checkout",
        "description": (
            "Switch to a branch or restore a file to its last committed state. "
            "Pass a branch name to switch branches; pass a file path to discard "
            "local changes to that file."
        ),
        "input_schema": GitCheckoutInput,
        "requires_confirmation": True,
        "function": git_checkout,
        "category": "mutation",
    },
]

register_tool_categories(
    {
        "git_status": "readonly",
        "git_diff": "readonly",
        "git_log": "readonly",
        "git_add": "mutation",
        "git_commit": "mutation",
        "git_create_branch": "mutation",
        "git_checkout": "mutation",
    }
)

__all__ = [
    "git_status",
    "git_diff",
    "git_log",
    "git_add",
    "git_commit",
    "git_create_branch",
    "git_checkout",
    "GitStatusInput",
    "GitDiffInput",
    "GitLogInput",
    "GitAddInput",
    "GitCommitInput",
    "GitCreateBranchInput",
    "GitCheckoutInput",
    "TOOL_CONFIGS",
]
