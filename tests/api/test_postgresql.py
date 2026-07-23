"""Tests for PostgreSQL production-readiness: engine, migrations, and dependencies.

Covers:
  - validate_connection() happy path and error formatting
  - _sanitize_url() password redaction
  - _connect_args_for() dialect selection (SQLite vs PG)
  - Migration 0002 upgrade/downgrade behaviour per dialect
  - pyproject.toml postgresql optional dependency presence
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest  # noqa: E402


def _mock_connect_ok() -> MagicMock:
    """Return a mock that acts as a successful async context manager for engine.connect()."""
    mock_conn = AsyncMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _mock_connect_fail(exc: Exception) -> MagicMock:
    """Return a mock context manager whose __aenter__ raises *exc*."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(side_effect=exc)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


_PROJECT_ROOT = Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# _sanitize_url
# ---------------------------------------------------------------------------


class TestSanitizeUrl:
    def test_redacts_password_from_postgresql_url(self):
        from cogtrix_core.api.db.engine import _sanitize_url

        url = "postgresql+asyncpg://user:s3cr3t@db.example.com:5432/cogtrix"
        sanitized = _sanitize_url(url)
        assert "s3cr3t" not in sanitized
        assert sanitized == "postgresql+asyncpg://user:***@db.example.com:5432/cogtrix"

    def test_leaves_sqlite_url_unchanged(self):
        from cogtrix_core.api.db.engine import _sanitize_url

        url = "sqlite+aiosqlite:///./data/api/cogtrix.db"
        assert _sanitize_url(url) == url

    def test_handles_url_without_password(self):
        from cogtrix_core.api.db.engine import _sanitize_url

        url = "postgresql+asyncpg://localhost/cogtrix"
        # No password → URL unchanged (no :***@ substitution needed)
        result = _sanitize_url(url)
        assert "localhost" in result
        assert "***" not in result

    def test_redacts_only_password_segment(self):
        from cogtrix_core.api.db.engine import _sanitize_url

        url = "postgresql+asyncpg://admin:pa$$word@10.0.0.1:5432/db"
        result = _sanitize_url(url)
        assert "pa$$word" not in result
        assert "admin" in result
        assert "10.0.0.1" in result


# ---------------------------------------------------------------------------
# validate_connection
# ---------------------------------------------------------------------------


class TestValidateConnection:
    @pytest.mark.asyncio
    async def test_succeeds_on_healthy_connection(self):
        """validate_connection() completes without error when SELECT 1 works."""
        from cogtrix_core.api.db.engine import validate_connection

        cm = _mock_connect_ok()
        mock_conn = cm.__aenter__.return_value
        mock_engine = MagicMock()
        mock_engine.connect.return_value = cm

        with patch("cogtrix_core.api.db.engine._get_engine", return_value=mock_engine):
            await validate_connection()

        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_raises_runtime_error_on_failure(self):
        """validate_connection() converts any connection error to RuntimeError."""
        from cogtrix_core.api.db.engine import validate_connection

        cm = _mock_connect_fail(OSError("connection refused"))
        mock_engine = MagicMock()
        mock_engine.connect.return_value = cm

        with patch("cogtrix_core.api.db.engine._get_engine", return_value=mock_engine):
            with pytest.raises(RuntimeError):
                await validate_connection()

    @pytest.mark.asyncio
    async def test_postgresql_error_message_is_actionable(self):
        """PG connection failure message contains operator-useful hints."""
        from cogtrix_core.api.db.engine import validate_connection

        cm = _mock_connect_fail(OSError("connection refused"))
        mock_engine = MagicMock()
        mock_engine.connect.return_value = cm

        with patch("cogtrix_core.api.db.engine._get_engine", return_value=mock_engine):
            with patch(
                "cogtrix_core.api.db.engine._get_db_url",
                return_value="postgresql+asyncpg://user:secret@localhost/cogtrix",
            ):
                with pytest.raises(RuntimeError) as exc_info:
                    await validate_connection()

        msg = str(exc_info.value)
        assert "Check:" in msg
        assert "asyncpg" in msg
        assert "COGTRIX_DB_URL" in msg

    @pytest.mark.asyncio
    async def test_sqlite_error_message_is_actionable(self):
        """SQLite connection failure message contains operator-useful hints."""
        from cogtrix_core.api.db.engine import validate_connection

        cm = _mock_connect_fail(OSError("no such file or directory"))
        mock_engine = MagicMock()
        mock_engine.connect.return_value = cm

        with patch("cogtrix_core.api.db.engine._get_engine", return_value=mock_engine):
            with patch(
                "cogtrix_core.api.db.engine._get_db_url",
                return_value="sqlite+aiosqlite:///./data/api/cogtrix.db",
            ):
                with pytest.raises(RuntimeError) as exc_info:
                    await validate_connection()

        msg = str(exc_info.value)
        assert "Check:" in msg
        assert "SQLite" in msg

    @pytest.mark.asyncio
    async def test_original_exception_chained(self):
        """RuntimeError must chain the original exception (raise ... from exc)."""
        from cogtrix_core.api.db.engine import validate_connection

        original = ConnectionRefusedError("port 5432 not open")
        cm = _mock_connect_fail(original)
        mock_engine = MagicMock()
        mock_engine.connect.return_value = cm

        with patch("cogtrix_core.api.db.engine._get_engine", return_value=mock_engine):
            with pytest.raises(RuntimeError) as exc_info:
                await validate_connection()

        assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# _connect_args dialect selection
# ---------------------------------------------------------------------------


class TestConnectArgs:
    def test_sqlite_url_sets_check_same_thread(self):
        from cogtrix_core.api.db.engine import _connect_args_for

        assert _connect_args_for("sqlite+aiosqlite:///./data/api/cogtrix.db") == {
            "check_same_thread": False
        }

    def test_postgresql_url_has_empty_connect_args(self):
        from cogtrix_core.api.db.engine import _connect_args_for

        assert _connect_args_for("postgresql+asyncpg://user:pw@localhost:5432/cogtrix") == {}


# ---------------------------------------------------------------------------
# Migration 0002
# ---------------------------------------------------------------------------


def _load_migration_0002():
    """Load alembic/versions/0002_pg_compat.py by file path.

    Module names starting with a digit are not valid Python identifiers so
    the regular import machinery cannot find them; we use spec_from_file_location
    (the same mechanism alembic itself uses internally).
    """
    import importlib.util

    path = _PROJECT_ROOT / "alembic" / "versions" / "0002_pg_compat.py"
    spec = importlib.util.spec_from_file_location("_migration_0002", str(path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


class TestMigration0002:
    def _make_bind(self, dialect_name: str) -> MagicMock:
        bind = MagicMock()
        bind.dialect.name = dialect_name
        return bind

    def test_upgrade_calls_batch_alter_on_sqlite(self):
        """On SQLite both Boolean columns get server_default corrected via batch mode."""
        m = _load_migration_0002()
        bind = self._make_bind("sqlite")
        batch_ctx = MagicMock()
        batch_ctx.__enter__ = MagicMock(return_value=batch_ctx)
        batch_ctx.__exit__ = MagicMock(return_value=False)
        with patch.object(m.op, "get_bind", return_value=bind):
            with patch.object(m.op, "batch_alter_table", return_value=batch_ctx) as mock_batch:
                m.upgrade()

        assert mock_batch.call_count == 2
        tables_patched = {c.args[0] for c in mock_batch.call_args_list}
        assert "refresh_tokens" in tables_patched
        assert "api_keys" in tables_patched
        assert batch_ctx.alter_column.call_count == 2

    def test_upgrade_skips_alter_column_on_postgresql(self):
        """On PostgreSQL upgrade() is a no-op — no ALTER TABLE emitted."""
        m = _load_migration_0002()
        bind = self._make_bind("postgresql")
        with patch.object(m.op, "get_bind", return_value=bind):
            with patch.object(m.op, "alter_column") as mock_alter:
                m.upgrade()

        mock_alter.assert_not_called()

    def test_downgrade_is_noop(self):
        """downgrade() must not raise and must not call any DDL."""
        m = _load_migration_0002()
        with patch.object(m.op, "alter_column") as mock_alter:
            m.downgrade()  # must not raise

        mock_alter.assert_not_called()


# ---------------------------------------------------------------------------
# pyproject.toml optional dependency checks
# ---------------------------------------------------------------------------


class TestPyprojectDependencies:
    def _load_toml(self) -> dict:
        path = _PROJECT_ROOT / "pyproject.toml"
        return tomllib.loads(path.read_text(encoding="utf-8"))

    def test_postgresql_extra_lists_asyncpg(self):
        """[project.optional-dependencies.postgresql] must contain asyncpg."""
        data = self._load_toml()
        pg_extras: list[str] = (
            data.get("project", {}).get("optional-dependencies", {}).get("postgresql", [])
        )
        assert any(
            "asyncpg" in dep for dep in pg_extras
        ), f"asyncpg not found in postgresql extra: {pg_extras}"

    def test_asyncpg_not_in_main_dependencies(self):
        """asyncpg must remain optional — not in [project.dependencies]."""
        data = self._load_toml()
        main_deps: list[str] = data.get("project", {}).get("dependencies", [])
        assert not any(
            "asyncpg" in dep for dep in main_deps
        ), "asyncpg was added to main dependencies — it must stay optional"
