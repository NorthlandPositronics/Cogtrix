"""Import-time warning regression check.

Each Cogtrix entry point (the CLI ``cogtrix`` module and the FastAPI
``src.api.app`` module) is imported in a fresh subprocess and its stderr
is asserted free of any ``Warning:`` line.  Catches dep-upgrade
regressions that would otherwise leak deprecation noise into container
startup logs — the exact mode that surfaced the langgraph/langchain-core
``allowed_objects`` warning in v0.2.6.

Why a stderr grep instead of ``-W error``: ``langchain_core/__init__.py``
runs ``surface_langchain_deprecation_warnings()`` at import time, which
prepends a ``default`` filter for ``LangChainPendingDeprecationWarning``
after the interpreter has already processed ``-W``.  That filter wins
over ``-W error``, so a ``-W``-based gate would silently pass on the
exact regression we want to catch.  Reading the real stderr the user
would see is the only honest signal.

When a new warning legitimately needs to be silenced, add a narrow
``warnings.filterwarnings`` block in the corresponding entry point (with
an upstream-tracking comment) and the test goes green again.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Each entry point gets its own subprocess so failure messages are scoped
# to the offending import path.
_ENTRY_POINTS = [
    "import cogtrix",
    "import cogtrix_core.api.app",
]


@pytest.mark.parametrize("import_stmt", _ENTRY_POINTS)
def test_no_warnings_on_import(import_stmt: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", import_stmt],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=_REPO_ROOT,
    )

    assert result.returncode == 0, f"Importing '{import_stmt}' failed:\n{result.stderr}"

    warning_lines = [line for line in result.stderr.splitlines() if "Warning" in line]
    assert not warning_lines, (
        f"Importing '{import_stmt}' emitted unsuppressed warning(s).  Either "
        f"narrow-suppress at the entry point (with an upstream-tracking "
        f"comment) or upgrade the dependency that emits it.\n\n"
        f"Offending stderr lines:\n  " + "\n  ".join(warning_lines)
    )
