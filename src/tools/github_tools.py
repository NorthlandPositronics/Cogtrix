"""GitHub integration tools via the gh CLI.

Tools:
    gh_create_issue  — create a GitHub issue
    gh_comment_issue — add a comment to an issue or PR
    gh_list_prs      — list pull requests
    gh_get_file      — get file contents from a repository

Configuration (optional):
    services:
      github:
        default_repo: "owner/repo"   # used when repo not specified in tool call

    TOOL_SETUP(config) is called automatically by ToolRegistry after loading.
    The module is silently skipped when the gh CLI is not installed.
"""

from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from src.tools.delegate import register_tool_categories
from src.tools.error_sanitizer import sanitize_error, sanitize_file_error

if TYPE_CHECKING:
    from pydantic import BaseModel, Field
else:
    try:
        from pydantic import BaseModel, Field
    except ImportError:  # pragma: no cover
        BaseModel = object  # type: ignore[assignment,misc]
        Field = lambda *a, **kw: None  # type: ignore[assignment]  # noqa: E731

if TYPE_CHECKING:
    from src.config import Config

# ── Module-level state (set by TOOL_SETUP) ─────────────────────────────────────

_default_repo: str = ""

# ── Constants ─────────────────────────────────────────────────────────────────

_REPO_RE = re.compile(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$")
_REF_RE = re.compile(r"^[a-zA-Z0-9_./-]+$")
_FILE_TRUNCATE = 10_000

# Timeout defaults (seconds) — prevents indefinite hangs when gh CLI hangs.
_GH_READ_TIMEOUT = 30  # gh pr list, gh api (read operations)
_GH_WRITE_TIMEOUT = 60  # gh issue create, gh issue comment (write operations)
_GH_TIMEOUT_CAP = 120  # hard cap for any caller-supplied override

# ── Validation helpers ─────────────────────────────────────────────────────────


def _validate_repo(repo: str) -> str | None:
    """Return an error message if *repo* has an invalid format, else None."""
    if repo and not _REPO_RE.match(repo):
        return f"invalid repo format {repo!r}: expected 'owner/repo' (alphanumeric, _, ., -)"
    return None


def _validate_path(path: str) -> str | None:
    """Return an error message if *path* is unsafe for gh_get_file, else None."""
    if not path:
        return "path must not be empty"
    if path.startswith("/"):
        return "path must not start with '/'"
    if ".." in PurePosixPath(path).parts:
        return "path must not contain '..' components"
    return None


def _resolve_repo(repo: str) -> tuple[str, str | None]:
    """Return ``(resolved_repo, error_or_None)``.

    Falls back to *_default_repo* when *repo* is empty.
    """
    r = repo.strip() or _default_repo
    if not r:
        return "", "repo not specified and no default_repo configured"
    err = _validate_repo(r)
    if err:
        return "", err
    return r, None


def _sanitize(text: str) -> str:
    """Strip null bytes from *text*."""
    return text.replace("\x00", "")


def _classify_gh_error(stderr: str) -> str:
    """
    Return a category hint for a failed gh command based on stderr content.

    The LLM uses this to iterate remediation efficiently rather than blind retry.
    """
    s = stderr.lower()
    if (
        "bad credentials" in s
        or "requires authentication" in s
        or "gh auth login" in s
        or "authentication required" in s
    ):
        return "authentication"
    if "rate limit" in s or "403" in s or "abuse" in s:
        return "rate-limit"
    if "404" in s or "not found" in s:
        return "not-found"
    if "connection" in s or "timeout" in s or "network" in s or "could not resolve" in s:
        return "network"
    return "unknown"


def _run_gh(cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
    """Run *cmd* via subprocess with the given *timeout* (seconds)."""
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


# ── Configuration ──────────────────────────────────────────────────────────────


def configure_github_tools(cfg: dict) -> None:
    """Apply GitHub tool configuration from the ``services.github`` dict."""
    global _default_repo
    _default_repo = (cfg.get("default_repo") or "").strip()


def TOOL_SETUP(config: Config) -> None:
    """Called automatically by ToolRegistry after this module is loaded."""
    svc = getattr(config, "services", {}) or {}
    gh_cfg = svc.get("github", {}) or {}
    configure_github_tools(gh_cfg)


def is_configured() -> bool:
    """Return True only when the gh CLI is available on PATH."""
    return shutil.which("gh") is not None


# ── Input schemas ──────────────────────────────────────────────────────────────


class GhCreateIssueInput(BaseModel):
    title: str = Field(..., description="Issue title.")
    body: str = Field(default="", description="Issue body / description (markdown supported).")
    repo: str = Field(
        default="",
        description=(
            "Repository in 'owner/repo' format. Uses the configured default_repo when empty."
        ),
    )
    labels: str = Field(
        default="",
        description="Comma-separated label names to apply (e.g. 'bug,enhancement').",
    )


class GhCommentIssueInput(BaseModel):
    issue_number: int = Field(..., description="Issue or PR number to comment on.")
    body: str = Field(..., description="Comment text (markdown supported).")
    repo: str = Field(
        default="",
        description=(
            "Repository in 'owner/repo' format. Uses the configured default_repo when empty."
        ),
    )


class GhListPrsInput(BaseModel):
    repo: str = Field(
        default="",
        description=(
            "Repository in 'owner/repo' format. Uses the configured default_repo when empty."
        ),
    )
    state: str = Field(
        default="open",
        description="PR state to list: 'open', 'closed', or 'merged'.",
    )
    limit: int = Field(default=10, description="Maximum number of PRs to return (1–100).")


class GhGetFileInput(BaseModel):
    path: str = Field(
        ...,
        description="File path within the repository (e.g. 'src/main.py').",
    )
    repo: str = Field(
        default="",
        description=(
            "Repository in 'owner/repo' format. Uses the configured default_repo when empty."
        ),
    )
    ref: str = Field(
        default="",
        description="Branch, tag, or commit SHA. Defaults to the repository's default branch.",
    )


# ── Tool functions ─────────────────────────────────────────────────────────────


def gh_create_issue(title: str, body: str = "", repo: str = "", labels: str = "") -> str:
    """Create a GitHub issue via the gh CLI."""
    repo, err = _resolve_repo(repo)
    if err:
        return f"Error: {err}"

    title = _sanitize(title)
    body = _sanitize(body)

    cmd = [
        "gh",
        "issue",
        "create",
        "--repo",
        repo,
        "--title",
        title,
        "--body",
        body,
        "--json",
        "number,url",
    ]
    if labels.strip():
        cmd += ["--label", labels.strip()]

    try:
        result = _run_gh(cmd, _GH_WRITE_TIMEOUT)
    except subprocess.TimeoutExpired:
        return f"Error: gh command timed out after {_GH_WRITE_TIMEOUT} seconds"
    if result.returncode != 0:
        category = _classify_gh_error(result.stderr)
        return f"Error: GitHub command failed ({category})"

    try:
        data = json.loads(result.stdout)
        number = data["number"]
        url = data["url"]
        return f"Issue created: #{number} — {title}\n{url}"
    except (json.JSONDecodeError, KeyError) as exc:
        return f"Error parsing GitHub response: {sanitize_error(exc)}"


def gh_comment_issue(issue_number: int, body: str, repo: str = "") -> str:
    """Add a comment to a GitHub issue or pull request via the gh CLI."""
    repo, err = _resolve_repo(repo)
    if err:
        return f"Error: {err}"

    body = _sanitize(body)

    cmd = [
        "gh",
        "issue",
        "comment",
        str(issue_number),
        "--repo",
        repo,
        "--body",
        body,
    ]
    try:
        result = _run_gh(cmd, _GH_WRITE_TIMEOUT)
    except subprocess.TimeoutExpired:
        return f"Error: gh command timed out after {_GH_WRITE_TIMEOUT} seconds"
    if result.returncode != 0:
        category = _classify_gh_error(result.stderr)
        return f"Error: GitHub command failed ({category})"

    return f"Comment added to #{issue_number}"


def gh_list_prs(repo: str = "", state: str = "open", limit: int = 10) -> str:
    """List pull requests in a GitHub repository via the gh CLI."""
    repo, err = _resolve_repo(repo)
    if err:
        return f"Error: {err}"

    limit = max(1, min(100, int(limit)))
    state = state.lower()
    if state not in ("open", "closed", "merged"):
        state = "open"

    cmd = [
        "gh",
        "pr",
        "list",
        "--repo",
        repo,
        "--state",
        state,
        "--limit",
        str(limit),
        "--json",
        "number,title,author,state",
    ]
    try:
        result = _run_gh(cmd, _GH_READ_TIMEOUT)
    except subprocess.TimeoutExpired:
        return f"Error: gh command timed out after {_GH_READ_TIMEOUT} seconds"
    if result.returncode != 0:
        category = _classify_gh_error(result.stderr)
        return f"Error: GitHub command failed ({category})"

    try:
        prs = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return f"Error parsing GitHub response: {sanitize_error(exc)}"

    if not prs:
        return "No pull requests found."

    lines = []
    for pr in prs:
        num = pr.get("number", "?")
        pr_title = pr.get("title", "")
        author = (pr.get("author") or {}).get("login", "")
        pr_state = pr.get("state", "").lower()
        lines.append(f"#{num} | {pr_title} | {author} | {pr_state}")

    return "\n".join(lines)


def gh_get_file(path: str, repo: str = "", ref: str = "") -> str:
    """Get the contents of a file from a GitHub repository via the gh CLI."""
    path_err = _validate_path(path)
    if path_err:
        return f"Error: {path_err}"

    repo, err = _resolve_repo(repo)
    if err:
        return f"Error: {err}"

    if ref and not _REF_RE.match(ref):
        return f"Error: invalid ref format {ref!r}"

    endpoint = f"repos/{repo}/contents/{path}"
    cmd = ["gh", "api", endpoint]
    if ref:
        cmd += ["--raw-field", f"ref={ref}"]

    try:
        result = _run_gh(cmd, _GH_READ_TIMEOUT)
    except subprocess.TimeoutExpired:
        return f"Error: gh command timed out after {_GH_READ_TIMEOUT} seconds"
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "404" in stderr or "not found" in stderr.lower():
            return f"Error: file not found: {path}"
        category = _classify_gh_error(stderr)
        return f"Error: GitHub command failed ({category})"

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return f"Error parsing GitHub response: {sanitize_error(exc)}"

    if data.get("type") != "file":
        return f"Error: {path!r} is not a file"

    encoding = data.get("encoding", "")
    content_raw = data.get("content", "")

    if encoding == "base64":
        content_clean = content_raw.replace("\n", "").replace("\r", "")
        try:
            content_bytes = base64.b64decode(content_clean)
            try:
                content_str = content_bytes.decode("utf-8")
            except UnicodeDecodeError:
                content_str = repr(content_bytes)
        except Exception as exc:
            return f"Error decoding file content: {sanitize_file_error(exc)}"
    else:
        content_str = content_raw

    if len(content_str) > _FILE_TRUNCATE:
        omitted = len(content_str) - _FILE_TRUNCATE
        content_str = content_str[:_FILE_TRUNCATE] + f"\n[truncated — {omitted} chars omitted]"

    return content_str


# ── Tool registry entries ──────────────────────────────────────────────────────

TOOL_CONFIGS = [
    {
        "name": "gh_create_issue",
        "description": (
            "Create a GitHub issue via the gh CLI. "
            "Provide a title, optional body, the repo as 'owner/repo', "
            "and optional comma-separated label names."
        ),
        "input_schema": GhCreateIssueInput,
        "requires_confirmation": True,
        "function": gh_create_issue,
        "category": "mutation",
    },
    {
        "name": "gh_comment_issue",
        "description": (
            "Add a comment to a GitHub issue or pull request via the gh CLI. "
            "Provide the issue/PR number, comment body, and repo ('owner/repo')."
        ),
        "input_schema": GhCommentIssueInput,
        "requires_confirmation": True,
        "function": gh_comment_issue,
        "category": "mutation",
    },
    {
        "name": "gh_list_prs",
        "description": (
            "List pull requests in a GitHub repository via the gh CLI. "
            "Returns each PR as: #number | title | author | state. "
            "Filter by state: 'open' (default), 'closed', or 'merged'."
        ),
        "input_schema": GhListPrsInput,
        "requires_confirmation": False,
        "function": gh_list_prs,
        "category": "readonly",
    },
    {
        "name": "gh_get_file",
        "description": (
            "Get the contents of a file from a GitHub repository via the gh CLI. "
            "Provide the file path within the repo and optionally a branch, tag, or commit ref. "
            "Output is truncated to 10,000 characters for large files."
        ),
        "input_schema": GhGetFileInput,
        "requires_confirmation": False,
        "function": gh_get_file,
        "category": "readonly",
    },
]

TOOL_CONFIG = TOOL_CONFIGS[0]


register_tool_categories(
    {
        "gh_create_issue": "mutation",
        "gh_comment_issue": "mutation",
        "gh_list_prs": "readonly",
        "gh_get_file": "readonly",
    }
)

__all__ = [
    "TOOL_SETUP",
    "configure_github_tools",
    "is_configured",
    "gh_create_issue",
    "gh_comment_issue",
    "gh_list_prs",
    "gh_get_file",
    "GhCreateIssueInput",
    "GhCommentIssueInput",
    "GhListPrsInput",
    "GhGetFileInput",
    "TOOL_CONFIG",
    "TOOL_CONFIGS",
]
