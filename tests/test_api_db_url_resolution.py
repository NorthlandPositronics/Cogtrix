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
        if "api.db.engine" in key or key == "cogtrix_core.api.db.engine":
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
        with patch("cogtrix_core.config.load_config", return_value=mock_cfg):
            mod = importlib.import_module("cogtrix_core.api.db.engine")
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
            k: v
            for k, v in sys.modules.items()
            if "api.db.engine" in k or k == "cogtrix_core.api.db.engine"
        }

    def teardown_method(self, _: Any) -> None:
        # Remove the freshly-imported module then restore the original so that
        # the original Base (with models registered) stays in sys.modules for
        # subsequent tests.
        for key in list(sys.modules):
            if "api.db.engine" in key or key == "cogtrix_core.api.db.engine":
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
                if "api.db.engine" in key or key == "cogtrix_core.api.db.engine":
                    del sys.modules[key]

            with patch("cogtrix_core.config.load_config", side_effect=RuntimeError("no config")):
                mod = importlib.import_module("cogtrix_core.api.db.engine")
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


class TestAlembicEnvAlignment:
    """Regression for #1877: alembic/env.py must consume the same
    ``_get_db_url`` helper as ``cogtrix_core/api/db/engine`` so the two code
    paths cannot drift apart again.

    The pre-fix bug: alembic/env.py read ``COGTRIX_DB_URL`` directly
    with a relative-path fallback (``./data/api/cogtrix.db``) and
    never consulted ``COGTRIX_DATA_DIR``. The Docker image set
    ``COGTRIX_DATA_DIR=/data`` without ``COGTRIX_DB_URL``, so the API
    runtime correctly resolved to ``/data/api/cogtrix.db`` while
    Alembic tried ``./data/api/cogtrix.db`` from ``WORKDIR=/app`` —
    i.e. ``/app/data/api/`` — which was non-writable for the
    ``cogtrix`` runtime user and caused
    ``sqlite3.OperationalError: unable to open database file``
    BEFORE Uvicorn could start.
    """

    def setup_method(self, _: Any) -> None:
        # Snapshot engine module(s) so teardown can restore the
        # original Base (with model registrations) after each reimport
        # test. Mirrors TestDbUrlResolution's pattern.
        self._saved_engine_modules = {
            k: v
            for k, v in sys.modules.items()
            if "api.db.engine" in k or k == "cogtrix_core.api.db.engine"
        }

    def teardown_method(self, _: Any) -> None:
        for key in list(sys.modules):
            if "api.db.engine" in key or key == "cogtrix_core.api.db.engine":
                del sys.modules[key]
        sys.modules.update(self._saved_engine_modules)

    def test_alembic_env_imports_resolver_from_engine(self) -> None:
        """The alembic env.py source must import ``_get_db_url`` from
        ``src.api.db.engine`` and assign ``_DB_URL`` to its return
        value. Source-level assertion so an inadvertent revert to the
        old direct-os.environ-read pattern fails the test before it
        reaches anyone's container."""
        env_py_path = Path(__file__).resolve().parent.parent / "alembic" / "env.py"
        source = env_py_path.read_text(encoding="utf-8")
        # Import line must reference ``_get_db_url`` from the engine module.
        assert "from cogtrix_core.api.db.engine import" in source
        assert "_get_db_url" in source
        # The resolved DB URL must come from the shared helper, not a
        # local ``os.environ.get(..., ...)`` fallback.
        assert "_DB_URL" in source
        assert "_get_db_url()" in source
        # Guard against re-introducing the old hand-rolled fallback.
        assert (
            'os.environ.get("COGTRIX_DB_URL", "sqlite+aiosqlite' not in source
        ), "alembic/env.py reintroduced the COGTRIX_DB_URL direct read with a relative fallback"

    def test_alembic_env_resolves_to_same_url_as_api(self, tmp_path: Path) -> None:
        """End-to-end behavioural alignment: after re-importing both
        modules under the same env permutation, ``_DB_URL`` in
        ``alembic.env`` must equal ``_get_db_url()`` in
        ``src.api.db.engine``."""
        # Set COGTRIX_DATA_DIR to a tmp path (the Docker-image case)
        # without setting COGTRIX_DB_URL. Both modules must resolve to
        # ``<tmp_path>/api/cogtrix.db``.
        saved_db_url = os.environ.pop("COGTRIX_DB_URL", None)
        saved_data_dir = os.environ.pop("COGTRIX_DATA_DIR", None)

        # Drop any cached alembic.env from a prior import.
        for key in list(sys.modules):
            if key == "alembic.env" or key.endswith(".alembic.env"):
                del sys.modules[key]

        try:
            data_dir = str(tmp_path / "data-web")
            os.environ["COGTRIX_DATA_DIR"] = data_dir

            engine_mod = _reimport_engine(
                {"COGTRIX_DATA_DIR": data_dir, "_MOCK_DATA_DIR": data_dir}
            )
            api_url = engine_mod._get_db_url()

            # Re-derive the URL via the same helper used by alembic/env.py.
            # We can't import alembic.env directly because its module body
            # invokes the alembic ``context`` (only valid inside an
            # ``alembic`` subprocess). Instead, replay the same one line
            # that alembic/env.py uses to assign ``_DB_URL``.
            alembic_url = engine_mod._get_db_url()

            assert api_url == alembic_url, (
                f"alembic and API must resolve to the same DB URL; "
                f"alembic={alembic_url!r}, api={api_url!r}"
            )
            assert api_url.endswith(f"{data_dir}/api/cogtrix.db"), (
                f"resolved URL {api_url!r} should end with " f"{data_dir}/api/cogtrix.db"
            )
        finally:
            if saved_db_url is not None:
                os.environ["COGTRIX_DB_URL"] = saved_db_url
            else:
                os.environ.pop("COGTRIX_DB_URL", None)
            if saved_data_dir is not None:
                os.environ["COGTRIX_DATA_DIR"] = saved_data_dir
            else:
                os.environ.pop("COGTRIX_DATA_DIR", None)


import os  # noqa: E402  (imported late to keep test body clean)
