"""Self-improvement tool — automated bug detection, LLM patch, and optional commit.

Tools:
    self_improve  — run ruff/bandit, patch findings with LLM, verify with pytest,
                    and optionally auto-commit verified fixes.

Configuration:
    TOOL_SETUP(config) is called automatically by ToolRegistry after this module
    is loaded.  It stores the Config reference used to build the LLM on demand.
    No changes to configure.py are required.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import logging
import re
import subprocess  # nosec B404
from pathlib import Path
from typing import TYPE_CHECKING

try:
    from langchain_core.messages import HumanMessage as _HumanMessage
except ImportError:  # pragma: no cover
    _HumanMessage = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from pydantic import BaseModel, Field
else:
    try:
        from pydantic import BaseModel, Field
    except ImportError:  # pragma: no cover
        BaseModel = object  # type: ignore[assignment,misc]
        Field = lambda *a, **kw: None  # type: ignore[assignment]  # noqa: E731

from cogtrix_core.concurrency import invoke_with_timeout
from cogtrix_core.logging_config import _scrub_secrets
from cogtrix_core.providers import create_chat_model_from_configs
from cogtrix_core.tools.delegate import register_tool_categories
from cogtrix_core.tools.error_sanitizer import sanitize_error

if TYPE_CHECKING:
    from cogtrix_core.config import Config

log = logging.getLogger("cogtrix.tools.self_improve")

# ── Module-level state (set by TOOL_SETUP) ────────────────────────────────────

_config: Config | None = None

# Pre-compiled fence regex for stripping markdown code blocks from LLM output
_FENCE_RE = re.compile(r"^```(?:python)?\s*\n?(.*?)```\s*$", re.DOTALL)

# Timeout for LLM invoke calls during patch generation (seconds)
_SELF_IMPROVE_LLM_TIMEOUT_SECONDS = 120


# ── Configuration ─────────────────────────────────────────────────────────────


def TOOL_SETUP(config: Config) -> None:
    """Called automatically by ToolRegistry after loading this module."""
    global _config
    _config = config


def is_configured() -> bool:
    """Return True when a Config has been wired via TOOL_SETUP."""
    return _config is not None


# ── Internal data types ────────────────────────────────────────────────────────


@dataclasses.dataclass
class _Finding:
    linter: str  # "ruff" or "bandit"
    file: str  # path string as returned by the linter
    line: int
    code: str
    message: str


# ── Path helpers ───────────────────────────────────────────────────────────────


def _validate_target(target: str) -> tuple[bool, str, Path]:
    """Ensure *target* resolves within cwd. Returns (ok, error, resolved)."""
    try:
        resolved = Path(target).resolve()
        cwd = Path.cwd().resolve()
        resolved.relative_to(cwd)
        return True, "", resolved
    except ValueError:
        return (
            False,
            f"target {target!r} is outside the working directory (path traversal not allowed)",
            Path(),
        )
    except Exception as exc:
        return False, str(exc), Path()


def _safe_patch_target(filepath: str) -> bool:
    """Return True only if *filepath* is under cogtrix_core/ or is cogtrix.py in cwd."""
    try:
        p = Path(filepath).resolve()
        cwd = Path.cwd().resolve()
        src_dir = (cwd / "src").resolve()
        cogtrix_py = (cwd / "cogtrix.py").resolve()
        try:
            p.relative_to(src_dir)
            return True
        except ValueError:
            pass
        return p == cogtrix_py
    except Exception:
        return False


def _derive_test_path(source_file: str) -> str:
    """Derive a test path from a source file.

    Returns ``tests/test_<stem>.py`` when that file exists, otherwise ``tests/``.
    """
    stem = Path(source_file).stem
    candidate = Path(f"tests/test_{stem}.py")
    if candidate.exists():
        return str(candidate)
    return "tests/"


# ── Subprocess helper ──────────────────────────────────────────────────────────


def _run(cmd: list[str], timeout: int = 60) -> tuple[int, str, str]:
    """Run *cmd* as a subprocess; return (returncode, stdout, stderr).

    Never raises — timeout and OS errors are captured into stderr.
    """
    try:
        result = subprocess.run(  # nosec B603
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout}s"
    except Exception as exc:
        return -1, "", str(exc)


# ── Linter output parsers ──────────────────────────────────────────────────────


def _parse_ruff(stdout: str, cap: int) -> list[_Finding]:
    """Parse ``ruff --output-format json`` output into _Finding objects."""
    findings: list[_Finding] = []
    if not stdout.strip():
        return findings
    try:
        items = json.loads(stdout)
    except json.JSONDecodeError:
        return findings
    for item in items:
        if len(findings) >= cap:
            break
        try:
            filepath = item["filename"]
            line = int(item["location"]["row"])
            code = item.get("code") or item.get("rule_code") or ""
            message = item.get("message", "")
            findings.append(_Finding("ruff", filepath, line, code, message))
        except (KeyError, TypeError, ValueError):
            continue
    return findings


def _parse_bandit(stdout: str, cap: int) -> list[_Finding]:
    """Parse ``bandit -f json`` output; include only HIGH + MEDIUM severity."""
    findings: list[_Finding] = []
    if not stdout.strip():
        return findings
    try:
        data = json.loads(stdout)
        results = data.get("results", [])
    except json.JSONDecodeError:
        return findings
    for item in results:
        if len(findings) >= cap:
            break
        severity = (item.get("issue_severity") or "").upper()
        if severity not in ("HIGH", "MEDIUM"):
            continue
        try:
            filepath = item["filename"]
            line = int(item["line_number"])
            code = item.get("test_id", "")
            message = item.get("issue_text", "")
            findings.append(_Finding("bandit", filepath, line, code, message))
        except (KeyError, TypeError, ValueError):
            continue
    return findings


def _deduplicate(findings: list[_Finding]) -> list[_Finding]:
    """Remove entries with duplicate (file, line) keys, keeping first occurrence."""
    seen: set[tuple[str, int]] = set()
    out: list[_Finding] = []
    for f in findings:
        key = (f.file, f.line)
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


# ── LLM patch helpers ──────────────────────────────────────────────────────────

_PATCH_PROMPT = """\
Fix the following {linter} issue in this Python file.
Issue: {code} at line {line}: {message}
File: {filepath}

```python
{file_content}
```

Return ONLY the complete corrected file content, no explanation."""


def _build_patch_prompt(finding: _Finding, file_content: str) -> str:
    return _PATCH_PROMPT.format(
        linter=finding.linter,
        code=finding.code,
        line=finding.line,
        message=finding.message,
        filepath=finding.file,
        file_content=file_content,
    )


def _extract_code(text: str) -> str:
    """Strip markdown code fences from LLM output if present."""
    text = text.strip()
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).rstrip()
    return text


def _is_valid_python(code: str) -> bool:
    """Return True iff *code* parses as valid Python (via ast.parse)."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


# ── Input schema ───────────────────────────────────────────────────────────────


class SelfImproveInput(BaseModel):
    target: str = Field(
        default="cogtrix_core/",
        description=(
            "Directory or file to scan for issues (must be within the working directory)."
        ),
    )
    max_fixes: int = Field(
        default=3,
        description="Maximum number of issues to attempt to fix in a single run.",
    )
    auto_commit: bool = Field(
        default=False,
        description=(
            "When True, commit all verified fixes in a single git commit. "
            "Requires confirmation before committing."
        ),
    )
    dry_run: bool = Field(
        default=False,
        description="When True, detect issues and return them without patching.",
    )


# ── Tool function ──────────────────────────────────────────────────────────────


def self_improve(
    target: str = "cogtrix_core/",
    max_fixes: int = 3,
    auto_commit: bool = False,
    dry_run: bool = False,
) -> str:
    """Run a lightweight automated quality cycle.

    1. **Detect** — run ruff and bandit on *target*, collect up to *max_fixes* issues.
    2. **Patch** — for each issue, ask the LLM to fix the file and validate the result.
    3. **Verify** — run pytest; revert any patch that breaks tests.
    4. **Commit** — if *auto_commit* is True and fixes survived, create a git commit.

    *dry_run=True* returns only the detected findings without modifying any files.
    """
    if _config is None:
        return "Error: self_improve is not configured. Run `cogtrix.py --setup` to configure it."

    # ── 1. Validate target path ────────────────────────────────────────────────
    ok, err, _ = _validate_target(target)
    if not ok:
        return f"Error: {err}"

    # ── 2. Run linters ─────────────────────────────────────────────────────────
    _ruff_rc, ruff_out, _ruff_err = _run(
        ["uv", "run", "ruff", "check", target, "--output-format", "json"],
        timeout=60,
    )
    _bandit_rc, bandit_out, _bandit_err = _run(
        ["uv", "run", "bandit", "-r", target, "-f", "json", "-ll"],
        timeout=60,
    )

    ruff_findings = _parse_ruff(ruff_out, max_fixes)
    bandit_findings = _parse_bandit(bandit_out, max_fixes)

    # Surface linter failures that produced no parseable findings.
    warnings: list[str] = []
    if _ruff_rc != 0 and not ruff_findings:
        warnings.append(f"Warning: ruff exited with code {_ruff_rc}. stderr: {_ruff_err[:200]}")
    if _bandit_rc != 0 and not bandit_findings:
        warnings.append(
            f"Warning: bandit exited with code {_bandit_rc}. stderr: {_bandit_err[:200]}"
        )

    combined = _deduplicate(ruff_findings + bandit_findings)[:max_fixes]

    if not combined:
        if warnings:
            return "\n".join(warnings)
        return f"No issues found in {target}. Nothing to improve."

    # ── 3. Dry-run: report findings and exit ───────────────────────────────────
    if dry_run:
        lines = [f"Dry-run findings in {target} ({len(combined)} issue(s)):"]
        for f in combined:
            lines.append(f"  [{f.linter}] {f.file}:{f.line} [{f.code}] {f.message}")
        if warnings:
            lines.extend([""] + warnings)
        return "\n".join(lines)

    # ── 4. Patch phase ─────────────────────────────────────────────────────────
    assert _HumanMessage is not None
    try:
        llm = create_chat_model_from_configs(*_config.resolve_llm_config())
    except Exception as exc:
        return f"Error: failed to create LLM: {sanitize_error(exc)}"

    fixed: list[_Finding] = []
    reverted: list[_Finding] = []
    skipped: list[tuple[_Finding, str]] = []  # (finding, reason)
    patched_files: set[str] = set()

    for finding in combined:
        filepath = finding.file

        # Security: only patch files within cogtrix_core/ or cogtrix.py
        if not _safe_patch_target(filepath):
            skipped.append((finding, "file outside allowed patch targets"))
            continue

        # Read original content
        try:
            original_content = Path(filepath).read_text(encoding="utf-8")
        except OSError as exc:
            skipped.append((finding, f"cannot read file: {exc}"))
            continue

        # Security: redact secrets before sending to LLM
        scrubbed_content = _scrub_secrets(original_content)

        # Build LLM prompt and call
        prompt = _build_patch_prompt(finding, scrubbed_content)
        try:
            response = invoke_with_timeout(
                llm.invoke,
                [_HumanMessage(content=prompt)],
                timeout=_SELF_IMPROVE_LLM_TIMEOUT_SECONDS,
            )
            raw = getattr(response, "content", str(response))
        except TimeoutError:
            skipped.append((finding, "LLM call timed out"))
            continue
        except Exception as exc:
            skipped.append((finding, f"LLM call failed: {exc}"))
            continue

        patched_code = _extract_code(str(raw))

        # Validate: must parse as valid Python
        if not _is_valid_python(patched_code):
            log.warning(
                "self_improve: LLM returned invalid Python for %s:%d [%s]; skipping",
                filepath,
                finding.line,
                finding.code,
            )
            skipped.append((finding, "LLM returned invalid Python (ast.parse failed)"))
            continue

        # Write patch
        try:
            Path(filepath).write_text(patched_code, encoding="utf-8")
        except OSError as exc:
            skipped.append((finding, f"cannot write patch: {exc}"))
            continue

        # ── 5. Verify phase ────────────────────────────────────────────────────
        test_path = _derive_test_path(filepath)
        pytest_rc, _pytest_out, _pytest_err = _run(
            ["uv", "run", "pytest", test_path, "-q", "--tb=short", "-x"],
            timeout=120,
        )

        if pytest_rc != 0:
            # Revert
            try:
                Path(filepath).write_text(original_content, encoding="utf-8")
            except OSError as exc:
                log.error(
                    "self_improve: failed to revert %s after test failure: %s",
                    filepath,
                    exc,
                )
            reverted.append(finding)
        else:
            fixed.append(finding)
            patched_files.add(filepath)

    # ── 6. Commit phase ────────────────────────────────────────────────────────
    committed = False
    commit_err: str = ""
    if auto_commit and fixed:
        files_list = sorted(patched_files)
        git_add_rc, _, git_add_err = _run(["git", "add"] + files_list, timeout=30)
        if git_add_rc == 0:
            patch_summary = ", ".join(f"{f.file}:{f.line} [{f.code}]" for f in fixed)
            commit_msg = (
                f"fix: auto-patch {len(fixed)} issue(s) via self_improve\n\n"
                f"Patched: {patch_summary}"
            )
            git_commit_rc, _, git_commit_err = _run(["git", "commit", "-m", commit_msg], timeout=30)
            if git_commit_rc == 0:
                committed = True
            else:
                commit_err = git_commit_err.strip()
        else:
            commit_err = git_add_err.strip()

    # ── 7. Build summary ───────────────────────────────────────────────────────
    total_found = len(combined)
    lines: list[str] = [
        "Self-improve cycle complete:",
        f"  Found:    {total_found} issue(s)",
        f"  Patched:  {len(fixed)} (verified)",
        f"  Reverted: {len(reverted)} (tests failed)",
        f"  Skipped:  {len(skipped)} (LLM or file errors)",
    ]
    if warnings:
        lines.extend([""] + warnings)
    if auto_commit:
        if committed:
            lines.append(f"  Committed: {len(patched_files)} file(s)")
        elif fixed:
            lines.append(f"  Commit failed: {commit_err or 'unknown error'}")
        else:
            lines.append("  Committed: 0 (no verified fixes)")

    if fixed:
        lines.append("\nFixed:")
        for f in fixed:
            lines.append(f"  - {f.file}:{f.line} [{f.code}] {f.message}")

    if reverted:
        lines.append("\nReverted (tests broke):")
        for f in reverted:
            lines.append(f"  - {f.file}:{f.line} [{f.code}] {f.message}")

    if skipped:
        lines.append("\nSkipped:")
        for f, reason in skipped:
            lines.append(f"  - {f.file}:{f.line} [{f.code}] {reason}")

    return "\n".join(lines)


# ── Tool registry entries ──────────────────────────────────────────────────────

TOOL_CONFIGS = [
    {
        "name": "self_improve",
        "description": (
            "Run an automated quality cycle on the codebase: detect issues with ruff "
            "and bandit, apply LLM-generated patches, verify each fix with pytest, "
            "and optionally commit all passing fixes. "
            "Use dry_run=True to preview findings without modifying any files. "
            "Set max_fixes to control how many issues to attempt per run."
        ),
        "input_schema": SelfImproveInput,
        "requires_confirmation": True,
        "function": self_improve,
        "category": "mutation",
    },
]

register_tool_categories({"self_improve": "mutation"})

TOOL_CONFIG = TOOL_CONFIGS[0]

__all__ = [
    "TOOL_SETUP",
    "is_configured",
    "self_improve",
    "SelfImproveInput",
    "TOOL_CONFIG",
    "TOOL_CONFIGS",
]
