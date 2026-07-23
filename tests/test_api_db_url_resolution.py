"""Tests that the API database URL respects data_dir from config.

Regression for the bug where data_dir: /data/cogtrix in .cogtrix.yaml was
ignored by engine.py, causing the SQLite DB to be created in ./data/api/
instead of /data/cogtrix/api/.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch


def _reimport_engine(env: dict[str, str]) -> ModuleType:
    """Import (or re-import) engine.py with the given environment variables.

    Any key whose value in *env* is the empty string is removed from the
    environment (used to mask variables like COGTRIX_DB_URL that the pytest
    runner may have set).
    """
    # Remove any cached module so _resolve_default_db_url() runs fresh
    for key in list(sys.modules):
        if "api.db.engine" in key or key == "src.api.db.engine":
            del sys.modules[key]

    # Keys to mask (remove) from the real environment during re-import
    mask = {k for k, v in env.items() if v == ""}
    clean_env = {k: v for k, v in env.items() if v != ""}

    # If COGTRIX_DB_URL is not explicitly requested, mask it so the pytest
    # runner's own value (often :memory:) does not override the test.
    if "COGTRIX_DB_URL" not in env:
        mask.add("COGTRIX_DB_URL")

    with patch.dict("os.environ", clean_env, clear=False):
        for key in mask:
            os.environ.pop(key, None)
        # Patch load_config to avoid needing a real config file
        mock_cfg = MagicMock()
        mock_cfg.data_dir = env.get("_MOCK_DATA_DIR", "data")
        with patch("src.config.load_config", return_value=mock_cfg):
            mod = importlib.import_module("src.api.db.engine")
            # URL resolution is lazy (PEP 562); force it now while the
            # load_config patch and env-var patches are still active.
            mod._get_db_url()
    return mod


def _extract_db_path(db_url: str) -> str:
    """Return the filesystem path portion of a sqlite+aiosqlite:/// URL."""
    return db_url.split("///", 1)[-1]


class TestDbUrlResolution:
    def setup_method(self, _: Any) -> None:
        # Snapshot the engine module(s) so teardown can restore the original
        # Base class (with model registrations) after each reimport test.
        self._saved_engine_modules = {
            k: v for k, v in sys.modules.items() if "api.db.engine" in k or k == "src.api.db.engine"
        }

    def teardown_method(self, _: Any) -> None:
        # Remove the freshly-imported module then restore the original so that
        # the original Base (with models registered) stays in sys.modules for
        # subsequent tests.
        for key in list(sys.modules):
            if "api.db.engine" in key or key == "src.api.db.engine":
                del sys.modules[key]
        sys.modules.update(self._saved_engine_modules)

    def test_explicit_cogtrix_db_url_wins(self, tmp_path: Path) -> None:
        """COGTRIX_DB_URL takes full precedence over everything."""
        explicit = f"sqlite+aiosqlite:///{tmp_path}/explicit.db"
        mod = _reimport_engine(
            {
                "COGTRIX_DB_URL": explicit,
                "COGTRIX_DATA_DIR": "/should/be/ignored",
                "_MOCK_DATA_DIR": "/also/ignored",
            }
        )
        assert str(mod._get_db_url()) == explicit

    def test_cogtrix_data_dir_env_var_used_when_no_db_url(self, tmp_path: Path) -> None:
        """COGTRIX_DATA_DIR relocates the SQLite file without setting COGTRIX_DB_URL."""
        mod = _reimport_engine({"COGTRIX_DATA_DIR": str(tmp_path), "_MOCK_DATA_DIR": str(tmp_path)})
        db_path = _extract_db_path(mod._get_db_url())
        assert db_path.startswith(
            str(tmp_path)
        ), f"DB path {db_path!r} should be under COGTRIX_DATA_DIR={tmp_path}"
        assert db_path.endswith("api/cogtrix.db")

    def test_config_data_dir_used_when_no_env_vars(self, tmp_path: Path) -> None:
        """data_dir from the config file is used when no env vars are set."""
        import os

        saved_db_url = os.environ.pop("COGTRIX_DB_URL", None)
        saved_data_dir = os.environ.pop("COGTRIX_DATA_DIR", None)
        try:
            mod = _reimport_engine({"_MOCK_DATA_DIR": str(tmp_path)})
            db_path = _extract_db_path(mod._get_db_url())
            assert db_path.startswith(
                str(tmp_path)
            ), f"DB path {db_path!r} should be derived from config.data_dir={tmp_path}"
            assert db_path.endswith("api/cogtrix.db")
        finally:
            if saved_db_url is not None:
                os.environ["COGTRIX_DB_URL"] = saved_db_url
            if saved_data_dir is not None:
                os.environ["COGTRIX_DATA_DIR"] = saved_data_dir

    def test_fallback_to_builtin_default_when_config_fails(self) -> None:
        """Falls back to ./data/api/cogtrix.db when config cannot be loaded."""
        import os

        saved_db_url = os.environ.pop("COGTRIX_DB_URL", None)
        saved_data_dir = os.environ.pop("COGTRIX_DATA_DIR", None)
        try:
            for key in list(sys.modules):
                if "api.db.engine" in key or key == "src.api.db.engine":
                    del sys.modules[key]

            with patch("src.config.load_config", side_effect=RuntimeError("no config")):
                mod = importlib.import_module("src.api.db.engine")
                # Force lazy resolution while the patch is still active.
                mod._get_db_url()

            db_path = _extract_db_path(mod._get_db_url())
            assert "data/api/cogtrix.db" in db_path
        finally:
            if saved_db_url is not None:
                os.environ["COGTRIX_DB_URL"] = saved_db_url
            if saved_data_dir is not None:
                os.environ["COGTRIX_DATA_DIR"] = saved_data_dir

    def test_cogtrix_data_dir_takes_priority_over_config_file(self, tmp_path: Path) -> None:
        """COGTRIX_DATA_DIR env var wins over data_dir in the config file."""
        import os

        env_data_dir = str(tmp_path / "from_env")
        config_data_dir = str(tmp_path / "from_config")
        saved_db_url = os.environ.pop("COGTRIX_DB_URL", None)
        try:
            mod = _reimport_engine(
                {
                    "COGTRIX_DATA_DIR": env_data_dir,
                    "_MOCK_DATA_DIR": config_data_dir,
                }
            )
            db_path = _extract_db_path(mod._get_db_url())
            assert db_path.startswith(
                env_data_dir
            ), f"COGTRIX_DATA_DIR should win over config data_dir; got {db_path!r}"
        finally:
            if saved_db_url is not None:
                os.environ["COGTRIX_DB_URL"] = saved_db_url


import os  # noqa: E402  (imported late to keep test body clean)
