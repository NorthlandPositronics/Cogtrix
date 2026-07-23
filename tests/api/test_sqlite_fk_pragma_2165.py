"""Regression test for #2165 — the SQLite engine enables PRAGMA foreign_keys=ON.

SQLite leaves foreign-key enforcement OFF by default, so without a per-connection
`PRAGMA foreign_keys=ON` the 21 `ondelete` CASCADE / SET NULL clauses in
models.py silently never fire. `_build_engine` now registers a `connect`
listener for SQLite URLs.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa

from src.api.db import engine as eng


@pytest.mark.asyncio
async def test_sqlite_engine_sets_foreign_keys_pragma(tmp_path, monkeypatch):
    db_path = tmp_path / "fk.db"
    monkeypatch.setattr(eng, "_get_db_url", lambda: f"sqlite+aiosqlite:///{db_path}")
    engine = eng._build_engine()
    try:
        async with engine.connect() as conn:
            enabled = (await conn.execute(sa.text("PRAGMA foreign_keys"))).scalar()
        assert enabled == 1, "PRAGMA foreign_keys must be ON for SQLite (#2165)"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_sqlite_on_delete_cascade_actually_fires(tmp_path, monkeypatch):
    """End-to-end proof that ON DELETE CASCADE is enforced once the pragma is on."""
    db_path = tmp_path / "fk_cascade.db"
    monkeypatch.setattr(eng, "_get_db_url", lambda: f"sqlite+aiosqlite:///{db_path}")
    engine = eng._build_engine()
    try:
        async with engine.begin() as conn:
            await conn.execute(sa.text("CREATE TABLE parent (id INTEGER PRIMARY KEY)"))
            await conn.execute(
                sa.text(
                    "CREATE TABLE child (id INTEGER PRIMARY KEY, "
                    "pid INTEGER REFERENCES parent(id) ON DELETE CASCADE)"
                )
            )
            await conn.execute(sa.text("INSERT INTO parent (id) VALUES (1)"))
            await conn.execute(sa.text("INSERT INTO child (id, pid) VALUES (10, 1)"))

        async with engine.begin() as conn:
            await conn.execute(sa.text("DELETE FROM parent WHERE id = 1"))

        async with engine.connect() as conn:
            remaining = (await conn.execute(sa.text("SELECT COUNT(*) FROM child"))).scalar()
        assert remaining == 0, "ON DELETE CASCADE must remove child rows on SQLite (#2165)"
    finally:
        await engine.dispose()
