"""Test generation tool — generate pytest test cases from a Python source file.

Tools:
    generate_tests  — read a source file and write an LLM-generated pytest suite

Configuration:
    TOOL_SETUP(config) is called automatically by ToolRegistry after this module
    is loaded.  It stores the Config reference used to build the LLM on demand.
"""

from __future__ import annotations

import re
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

from cogtrix_core.agent.safety import UserCancelledRun
from cogtrix_core.concurrency import invoke_with_timeout
from cogtrix_core.logging_config import _scrub_secrets
from cogtrix_core.providers import create_chat_model_from_configs
from cogtrix_core.tools.delegate import register_tool_categories
from cogtrix_core.tools.error_sanitizer import sanitize_error, sanitize_file_error

if TYPE_CHECKING:
    from cogtrix_core.config import Config

# ── Module-level state (set by TOOL_SETUP) ────────────────────────────────────

_config: Config | None = None

_APP_DIR: Path = Path(__file__).resolve().parent.parent.parent

# Timeout for LLM invoke calls during test generation (seconds)
_GENERATE_TESTS_LLM_TIMEOUT_SECONDS = 120


# ── Configuration ─────────────────────────────────────────────────────────────


def TOOL_SETUP(config: Config) -> None:
    """Called automatically by ToolRegistry after loading this module."""
    global _config
    _config = config


def is_configured() -> bool:
    """Return True when a Config has been wired via TOOL_SETUP."""
    return _config is not None


# ── Path helpers ──────────────────────────────────────────────────────────────

# #1928: route all path-policy error strings through the canonical
# ``cogtrix_core/tools/_path_policy.py`` helpers so every file tool emits the
# same shape for the same logical class of failure.  Import here
# (after the conditional pydantic block above) — the ``noqa: E402``
# marker on each line is required because ruff sees these as
# module-level imports after non-import statements.
from cogtrix_core.tools._path_policy import (  # noqa: E402
    format_read_outside_error as _format_outside,  # noqa: E402
)
from cogtrix_core.tools._path_policy import (  # noqa: E402
    format_traversal_error as _format_traversal,  # noqa: E402
)
from cogtrix_core.tools._path_policy import (  # noqa: E402
    format_write_outside_error as _format_write_outside,  # noqa: E402
)


def _resolve_source_path(path: str) -> tuple[bool, str, Path | None]:
    """Resolve *path* for reading.  Reads are allowed within cwd or _APP_DIR."""
    try:
        p = Path(path).resolve()
        cwd = Path.cwd().resolve()
        if ".." in path:
            try:
                p.relative_to(cwd)
            except ValueError:
                try:
                    p.relative_to(_APP_DIR)
                except ValueError:
                    return False, _format_traversal(path), None
        try:
            p.relative_to(cwd)
            return True, "", p
        except ValueError:
            pass
        try:
            p.relative_to(_APP_DIR)
            return True, "", p
        except ValueError:
            pass
        return False, _format_outside(path), None
    except Exception as exc:
        return False, str(exc), None


def _resolve_output_path(path: str) -> tuple[bool, str, Path | None]:
    """Resolve *path* for writing.  Writes are restricted to cwd."""
    try:
        p = Path(path).resolve()
        cwd = Path.cwd().resolve()
        if ".." in path:
            try:
                p.relative_to(cwd)
            except ValueError:
                return False, _format_traversal(path), None
        try:
            p.relative_to(cwd)
            return True, "", p
        except ValueError:
            return False, _format_write_outside(path), None
    except Exception as exc:
        return False, str(exc), None


def _derive_output_file(source_file: str) -> str:
    """Derive a tests/ path from a source path.

    Examples:
        cogtrix_core/foo/bar.py          → tests/test_bar.py
        cogtrix_core/tools/email_tools.py → tests/test_email_tools.py
        my_module.py            → tests/test_my_module.py
    """
    stem = Path(source_file).stem
    return f"tests/test_{stem}.py"


# ── LLM prompt builder ────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = """\
You are a test engineer. Generate comprehensive pytest tests for the following Python code.

Requirements:
- Use pytest fixtures and parametrize where appropriate
- Mock all external dependencies (I/O, network, LLM calls)
- Cover happy path, error paths, and edge cases
- Each test must be independent (no shared state)
- Do not test private functions (prefixed with _)
{focus_line}
Source file: {source_file}

```python
{source_content}
```

Respond with ONLY the complete test file content, no explanation."""


def _build_prompt(source_file: str, source_content: str, focus: str) -> str:
    focus_line = f"- Focus on: {focus}\n" if focus.strip() else ""
    return _PROMPT_TEMPLATE.format(
        focus_line=focus_line,
        source_file=source_file,
        source_content=source_content,
    )


# ── Code extraction ───────────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"^```(?:python)?\s*\n?(.*?)```\s*$", re.DOTALL)


def _extract_code(text: str) -> str:
    """Strip markdown code fences if present; otherwise return text as-is."""
    text = text.strip()
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).rstrip()
    return text


# ── Input schema ──────────────────────────────────────────────────────────────


class GenerateTestsInput(BaseModel):
    source_file: str = Field(
        ...,
        description=(
            "Path to the Python source file to generate tests for "
            "(relative to cwd or absolute within the project)."
        ),
    )
    output_file: str = Field(
        default="",
        description=(
            "Where to write the generated test file. "
            "Defaults to tests/test_<stem>.py derived from source_file."
        ),
    )
    focus: str = Field(
        default="",
        description=(
            "Optional hint to the LLM about what to focus on "
            "(e.g. 'error paths', 'concurrency', 'edge cases')."
        ),
    )
    style: str = Field(
        default="pytest",
        description="Test style. Only 'pytest' is supported.",
    )


# ── Tool function ─────────────────────────────────────────────────────────────


def generate_tests(
    source_file: str,
    output_file: str = "",
    focus: str = "",
    style: str = "pytest",
) -> str:
    """Read a Python source file and generate a pytest test suite using the LLM.

    Writes the generated tests to *output_file* (or a derived path when empty).
    Returns an error string if the source file cannot be read, the output file
    already exists, or the LLM call fails.
    """
    if _config is None:
        return "Error: generate_tests is not configured. Run `cogtrix.py --setup` to configure it."

    if style != "pytest":
        return f"Error: unsupported style {style!r} — only 'pytest' is supported"

    # ── Resolve source path ────────────────────────────────────────────────
    ok, err, src_path = _resolve_source_path(source_file)
    if not ok:
        return f"Error: {err}"
    assert src_path is not None

    if not src_path.exists():
        return f"Error: source file not found: {source_file}"
    if not src_path.is_file():
        return f"Error: {source_file} is not a file"

    try:
        source_content = src_path.read_text(encoding="utf-8")
    except OSError as exc:
        return f"Error reading {source_file}: {sanitize_file_error(exc)}"

    # Security: redact secrets before sending to LLM
    source_content = _scrub_secrets(source_content)

    # ── Resolve output path ────────────────────────────────────────────────
    out_str = output_file.strip() or _derive_output_file(source_file)
    ok, err, out_path = _resolve_output_path(out_str)
    if not ok:
        return f"Error: {err}"
    assert out_path is not None

    if out_path.exists():
        return (
            f"Error: output file already exists: {out_str} — "
            "delete it or specify a different output_file to proceed"
        )

    # ── Call LLM ──────────────────────────────────────────────────────────
    try:
        assert _HumanMessage is not None
        llm = create_chat_model_from_configs(*_config.resolve_llm_config())
        prompt = _build_prompt(source_file, source_content, focus)
        try:
            response = invoke_with_timeout(
                llm.invoke,
                [_HumanMessage(content=prompt)],
                timeout=_GENERATE_TESTS_LLM_TIMEOUT_SECONDS,
            )
            raw = getattr(response, "content", str(response))
        except TimeoutError:
            return "Error: LLM call timed out"
    except UserCancelledRun:
        raise
    except Exception as exc:
        return f"Error: LLM call failed: {sanitize_error(exc)}"

    # ── Extract code and write ─────────────────────────────────────────────
    code = _extract_code(str(raw))

    try:
        from cogtrix_core.utils.atomic_write import atomic_write_json

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with atomic_write_json(out_path) as f:
            f.write(code)
    except OSError as exc:
        return f"Error writing {out_str}: {sanitize_file_error(exc)}"

    line_count = code.count("\n") + 1
    return f"Tests written to {out_str} ({line_count} lines)"


# ── Tool registry entries ─────────────────────────────────────────────────────

TOOL_CONFIGS = [
    {
        "name": "generate_tests",
        "description": (
            "Read a Python source file and generate a comprehensive pytest test suite "
            "using the current LLM. Writes the tests to tests/test_<name>.py by default. "
            "Use the 'focus' parameter to guide the LLM (e.g. 'error paths', 'edge cases'). "
            "Will not overwrite an existing file."
        ),
        "input_schema": GenerateTestsInput,
        "requires_confirmation": True,
        "function": generate_tests,
        "category": "mutation",
    },
]

register_tool_categories({"generate_tests": "mutation"})

TOOL_CONFIG = TOOL_CONFIGS[0]

__all__ = [
    "TOOL_SETUP",
    "is_configured",
    "generate_tests",
    "GenerateTestsInput",
    "TOOL_CONFIG",
    "TOOL_CONFIGS",
]
