"""Regression test for #2060 — config without allowed_write_paths must not wipe
env-wired write dirs.

``COGTRIX_ALLOWED_WRITE_PATHS`` is wired into ``_extra_write_dirs`` at import.
``configure_file_ops_tool`` used to call ``set_allowed_write_dirs`` with the
config value unconditionally, so a config file *without* ``allowed_write_paths``
(an empty list) silently revoked the env-configured write dirs. The guard now
lives in ``configure_file_ops_tool``: it only applies a non-empty config value.

The low-level ``set_allowed_write_dirs`` keeps its clear-on-empty semantics —
it is the canonical reset used by test fixtures and ``--allow-write-path``.
"""

from __future__ import annotations

from types import SimpleNamespace

import cogtrix_core.tools.file_ops as fo
from cogtrix_core.tools.configure import configure_file_ops_tool


def test_config_without_write_paths_preserves_env_wired_dirs() -> None:
    orig = list(fo._extra_write_dirs)
    try:
        # Simulate dirs wired from COGTRIX_ALLOWED_WRITE_PATHS at import.
        fo.set_allowed_write_dirs(["/tmp/cogtrix-2060-env"])
        seeded = list(fo._extra_write_dirs)
        assert seeded

        # Config file without an allowed_write_paths key -> empty list / None.
        configure_file_ops_tool(SimpleNamespace(allowed_write_paths=[]))
        assert fo._extra_write_dirs == seeded, "empty config must not wipe env-wired dirs"

        configure_file_ops_tool(SimpleNamespace(allowed_write_paths=None))
        assert fo._extra_write_dirs == seeded, "None config must not wipe env-wired dirs"
    finally:
        fo._extra_write_dirs = orig


def test_config_with_write_paths_replaces() -> None:
    orig = list(fo._extra_write_dirs)
    try:
        fo.set_allowed_write_dirs(["/tmp/cogtrix-2060-env"])
        configure_file_ops_tool(SimpleNamespace(allowed_write_paths=["/tmp/cogtrix-2060-cfg"]))
        joined = " ".join(str(p) for p in fo._extra_write_dirs)
        assert "cogtrix-2060-cfg" in joined
        assert "cogtrix-2060-env" not in joined
    finally:
        fo._extra_write_dirs = orig


def test_set_allowed_write_dirs_still_clears_on_empty() -> None:
    """The low-level setter retains reset semantics (fixtures rely on it)."""
    orig = list(fo._extra_write_dirs)
    try:
        fo.set_allowed_write_dirs(["/tmp/cogtrix-2060-env"])
        assert fo._extra_write_dirs
        fo.set_allowed_write_dirs(None)
        assert fo._extra_write_dirs == []
        fo.set_allowed_write_dirs(["/tmp/cogtrix-2060-env"])
        fo.set_allowed_write_dirs([])
        assert fo._extra_write_dirs == []
    finally:
        fo._extra_write_dirs = orig
