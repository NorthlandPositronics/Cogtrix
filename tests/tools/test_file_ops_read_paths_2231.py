"""Regression test for #2231 — COGTRIX_ALLOWED_READ_PATHS must be honored on
the API path, symmetric with the write allowlist.

Before the fix, read paths were wired ONLY by ``configure_file_read_dirs``
(called from the CLI), never at import — so ``COGTRIX_ALLOWED_READ_PATHS`` was a
silent no-op for API-served (TUI / WebUI) sessions, while the write var worked
everywhere (it is wired at import). The fix wires read paths at import too, and
adds the #2060-style guard to ``configure_file_read_dirs`` so a config without
``allowed_read_paths`` doesn't wipe the env-wired dirs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cogtrix_core.tools.file_ops as fo
from cogtrix_core.tools.configure import configure_file_read_dirs

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_env_read_paths_wired_at_import(tmp_path: Path) -> None:
    """COGTRIX_ALLOWED_READ_PATHS populates _extra_read_dirs at import (#2231).

    Runs in a fresh subprocess so the import-time wiring re-executes with the env
    set — this is the path that the API session uses (no configure call).
    """
    readable = tmp_path / "readable"
    readable.mkdir()
    code = (
        "from pathlib import Path\n"
        "from cogtrix_core.tools import file_ops\n"
        f"assert Path({str(readable)!r}).resolve() in file_ops._extra_read_dirs, "
        "file_ops._extra_read_dirs\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_REPO_ROOT),
        env={
            "COGTRIX_ALLOWED_READ_PATHS": str(readable),
            "PYTHONPATH": str(_REPO_ROOT),
            "PATH": "/usr/bin:/bin",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "OK" in result.stdout


def test_config_without_read_paths_preserves_env_wired_dirs() -> None:
    """#2231/#2060 guard: empty/None config must NOT wipe env-wired read dirs."""
    orig = list(fo._extra_read_dirs)
    try:
        fo.set_allowed_read_dirs(["/tmp/cogtrix-2231-env"])
        seeded = list(fo._extra_read_dirs)
        assert seeded

        configure_file_read_dirs(SimpleNamespace(allowed_read_paths=[]))
        assert fo._extra_read_dirs == seeded, "empty config must not wipe env-wired read dirs"

        configure_file_read_dirs(SimpleNamespace(allowed_read_paths=None))
        assert fo._extra_read_dirs == seeded, "None config must not wipe env-wired read dirs"
    finally:
        fo._extra_read_dirs = orig


def test_config_with_read_paths_replaces() -> None:
    orig = list(fo._extra_read_dirs)
    try:
        fo.set_allowed_read_dirs(["/tmp/cogtrix-2231-env"])
        configure_file_read_dirs(SimpleNamespace(allowed_read_paths=["/tmp/cogtrix-2231-cfg"]))
        joined = " ".join(str(p) for p in fo._extra_read_dirs)
        assert "cogtrix-2231-cfg" in joined
        assert "cogtrix-2231-env" not in joined
    finally:
        fo._extra_read_dirs = orig


def test_set_allowed_read_dirs_still_clears_on_empty() -> None:
    """The low-level setter retains reset semantics (fixtures rely on it)."""
    orig = list(fo._extra_read_dirs)
    try:
        fo.set_allowed_read_dirs(["/tmp/cogtrix-2231-env"])
        assert fo._extra_read_dirs
        fo.set_allowed_read_dirs(None)
        assert fo._extra_read_dirs == []
        fo.set_allowed_read_dirs(["/tmp/cogtrix-2231-env"])
        fo.set_allowed_read_dirs([])
        assert fo._extra_read_dirs == []
    finally:
        fo._extra_read_dirs = orig
