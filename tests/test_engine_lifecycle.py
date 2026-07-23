"""Regression tests for the async DB engine lifecycle.

These tests guard against a class of bugs where the module-level
``_engine`` / ``_session_factory`` cache in ``src.api.db.engine``
survives lifespan shutdown.  When that happens, a new ``TestClient``
(which spins up a fresh asyncio event loop via anyio's portal) reuses
a disposed engine whose aiosqlite worker threads are still bound to
the previous, now-closed loop.  Pending thread callbacks then fire on
the closed loop and Python emits
``PytestUnhandledThreadExceptionWarning: RuntimeError: Event loop is
closed``.

The production fix lives in ``src.api.app.lifespan``: after
``engine.dispose()``, the module-level cache is reset to ``None`` so
the next lifespan builds a fresh engine bound to its own loop.

Why behavioural-only assertions: ``tests/test_api_db_url_resolution.py``
deletes and re-imports ``src.api.db.engine`` mid-suite to test
``data_dir`` precedence.  Although it restores ``sys.modules`` on
teardown, the ``validate_connection`` function captured by ``app.py``
at import time still binds to the ORIGINAL module's closure, while
the lifespan's runtime ``import cogtrix_core.api.db.engine as _db_engine_mod``
resolves to whichever instance is currently in ``sys.modules``.  The
two references can diverge after the re-import class runs, which
makes direct ``_engine is None`` assertions flaky in the full suite.
These tests instead verify the user-facing guarantee: consecutive
``TestClient`` cycles serve requests without leaking
``PytestUnhandledThreadExceptionWarning``.
"""

from __future__ import annotations

import asyncio
import os
import warnings

import pytest

pytest.importorskip("fastapi")

_TEST_JWT_SECRET = "testsecret_mustbe32chars_minimum00"
os.environ.setdefault("COGTRIX_JWT_SECRET", _TEST_JWT_SECRET)
os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

from fastapi.testclient import TestClient  # noqa: E402

from cogtrix_core.api.app import create_app  # noqa: E402


def test_consecutive_test_clients_work() -> None:
    """Two back-to-back TestClient cycles must both serve requests.

    This proves the second cycle gets a functional engine — either a
    fresh one (after the lifespan cache reset) or a reusable one.
    Verified via the database-touching readiness endpoint
    (``/api/v1/health/ready``).  What matters is the request does not
    crash with "Event loop is closed".
    """
    for _ in range(2):
        app = create_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.get("/api/v1/health/ready")
            assert r.status_code in (200, 503), (
                f"second TestClient cycle failed: status={r.status_code}, " f"body={r.text}"
            )


def test_data_does_not_leak_between_test_clients() -> None:
    """Each ``TestClient`` cycle must see a fresh in-memory database.

    Regression for the parent-package-attribute drift bug
    (commit 4620f9e): when the lifespan resolved the engine module via
    ``import cogtrix_core.api.db.engine`` instead of ``sys.modules``, a prior
    re-import test (``test_api_db_url_resolution``) could leave the
    parent package's ``engine`` attribute pointing at an orphaned
    module.  Startup then created tables on that orphan while
    ``get_db`` (bound at package import time) queried the original
    module's engine, breaking data isolation between TestClient
    cycles.

    This test asserts the user-facing guarantee: data uploaded inside
    one ``with TestClient(...)`` block must NOT appear in a fresh
    ``with TestClient(...)`` block immediately afterwards.  Sufficient
    to catch any future regression where the lifespan engine reset
    silently skips, the wrong engine is used, or ``:memory:`` SQLite
    is shared across clients.
    """
    import uuid
    from pathlib import Path
    from unittest.mock import AsyncMock, patch

    from cogtrix_core.api.auth import create_access_token

    admin_token = create_access_token(user_id=str(uuid.uuid4()), role="admin")

    # ── Cycle 1: upload a document ─────────────────────────────────────
    app_1 = create_app()
    with TestClient(app_1, raise_server_exceptions=True) as client_1:
        with (
            patch(
                "cogtrix_core.api.routes.rag.ingest_document_task",
                new=AsyncMock(return_value=None),
            ),
            patch("cogtrix_core.api.routes.rag._get_uploads_dir", return_value=Path("/tmp")),
            patch("cogtrix_core.api.tasks.rag._get_uploads_dir", return_value=Path("/tmp")),
        ):
            upload = client_1.post(
                "/api/v1/rag/documents",
                files={"file": ("leak-probe.txt", b"x", "text/plain")},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert upload.status_code == 202, upload.text

    # ── Cycle 2: brand-new TestClient, list must be empty ──────────────
    app_2 = create_app()
    admin_token_2 = create_access_token(user_id=str(uuid.uuid4()), role="admin")
    with TestClient(app_2, raise_server_exceptions=True) as client_2:
        resp = client_2.get(
            "/api/v1/rag/documents",
            headers={"Authorization": f"Bearer {admin_token_2}"},
        )
        assert resp.status_code == 200, resp.text
        page = resp.json()["data"]
        assert page["items"] == [], (
            "data leaked across TestClient cycles — engine cache was not "
            f"reset on lifespan shutdown.  Got: {page['items']}"
        )


def test_no_event_loop_closed_warning_across_clients() -> None:
    """Three consecutive TestClient cycles must not leak
    ``PytestUnhandledThreadExceptionWarning`` or ``RuntimeError: Event
    loop is closed`` from aiosqlite worker threads.

    The block re-enables ``PytestUnhandledThreadExceptionWarning`` as
    an error in case a global filter is masking it elsewhere — that
    way a regression here will fail loudly even if the project-wide
    ``filterwarnings`` config is later relaxed.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", pytest.PytestUnhandledThreadExceptionWarning)
        for _ in range(3):
            app = create_app()
            with TestClient(app, raise_server_exceptions=False) as client:
                r = client.get("/api/v1/health/ready")
                assert r.status_code in (200, 503)
        # Give any in-flight aiosqlite worker callbacks a tick to
        # surface before catch_warnings exits.
        asyncio.run(asyncio.sleep(0))
