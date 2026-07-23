"""Regression tests for #1882.

Two distinct problems exercised here:

1. ``POST /api/v1/sessions`` accepts a missing request body / empty JSON
   ``{}`` and creates a session with defaults. Pre-#1882 the route
   signature required the body to be present, even though every field
   on :class:`SessionCreateRequest` has a default; the validation
   handler responded with a confusingly-shaped 422 pointing at
   ``_root``.

2. :func:`src.api.validation._humanize_name` renders the synthetic
   ``_root`` sentinel as ``"Request body"`` so root-level validation
   errors read ``"Request body is required."`` instead of the prior
   ``" root is required."`` (with leading whitespace).
"""

from __future__ import annotations

import asyncio as _asyncio
import os
import uuid
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")

_TEST_JWT_SECRET = "testsecret_mustbe32chars_minimum00"
os.environ.setdefault("COGTRIX_JWT_SECRET", _TEST_JWT_SECRET)
os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from cogtrix_core.api.db.engine import Base, get_db  # noqa: E402

_VALID_PASSWORD = "TestPass1!"


# ---------------------------------------------------------------------------
# Fixtures (mirror test_api_sessions_complete.py — proven scaffold)
# ---------------------------------------------------------------------------


@pytest.fixture()
def app():
    from cogtrix_core.api.app import create_app

    loop = _asyncio.new_event_loop()
    _asyncio.set_event_loop(loop)

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _setup():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    loop.run_until_complete(_setup())

    with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
        _app = create_app()

        async def _override():
            async with factory() as session:
                try:
                    yield session
                    await session.commit()
                except Exception:
                    await session.rollback()
                    raise

        _app.dependency_overrides[get_db] = _override
        yield _app

    loop.run_until_complete(engine.dispose())
    loop.close()


@pytest.fixture()
def client(app):
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _register(client):
    username = f"u_{uuid.uuid4().hex[:8]}"
    r = client.post(
        "/api/v1/auth/register",
        json={
            "username": username,
            "email": f"{username}@ex.com",
            "password": _VALID_PASSWORD,
        },
    )
    assert r.status_code == 201, f"register failed: {r.text}"
    return r.json()["data"]["access_token"]


def _h(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# POST /api/v1/sessions — body shapes
# ---------------------------------------------------------------------------


class TestCreateSessionBodyShapes:
    """Every field on ``SessionCreateRequest`` has a default. The route
    must accept all three body shapes uniformly: absent, ``{}``, and
    ``{"name": "X"}``."""

    def test_no_body_creates_session_with_defaults(self, client) -> None:
        token = _register(client)
        r = client.post("/api/v1/sessions", headers=_h(token))
        assert r.status_code == 201, r.text
        body = r.json()["data"]
        # Auto-generated name is non-empty and not a literal placeholder.
        assert isinstance(body.get("name"), str) and body["name"]
        # All-defaults SessionConfig yields an empty / default
        # config_json — the server stores the dict; here we just check
        # the round-trip didn't 422.
        assert "id" in body

    def test_empty_json_object_creates_session(self, client) -> None:
        token = _register(client)
        r = client.post("/api/v1/sessions", headers=_h(token), json={})
        assert r.status_code == 201, r.text

    def test_explicit_name_overrides_default(self, client) -> None:
        token = _register(client)
        name = f"explicit_{uuid.uuid4().hex[:6]}"
        r = client.post("/api/v1/sessions", headers=_h(token), json={"name": name})
        assert r.status_code == 201, r.text
        assert r.json()["data"]["name"] == name

    def test_no_body_requires_auth(self, client) -> None:
        """The empty-body acceptance MUST NOT bypass the auth gate."""
        r = client.post("/api/v1/sessions")
        assert r.status_code == 401, r.text


# ---------------------------------------------------------------------------
# Validation handler rendering — request-root errors
# ---------------------------------------------------------------------------


class TestRequestBodyValidationRendering:
    """The synthetic ``_root`` sentinel emitted by
    :func:`_extract_field_path` when Pydantic reports a root-level
    error must humanise to ``"Request body"`` so the resulting message
    reads cleanly (no leading whitespace, no internal token leak)."""

    def test_humanize_name_special_cases_root(self) -> None:
        from cogtrix_core.api.validation import _humanize_name

        assert _humanize_name("_root") == "Request body"

    def test_humanize_name_unchanged_for_regular_fields(self) -> None:
        from cogtrix_core.api.validation import _humanize_name

        # The fix must not alter unrelated field-name humanisation.
        assert _humanize_name("workspace_id") == "Workspace id"
        assert _humanize_name("system_prompt") == "System prompt"

    def test_build_fallback_message_for_root_missing(self) -> None:
        from cogtrix_core.api.validation import _build_fallback_message

        msg = _build_fallback_message("_root", {"type": "missing"})
        assert msg == "Request body is required."
        # Defensive: no leading whitespace, no '_root' leakage to API
        # consumers, no double space.
        assert not msg.startswith(" ")
        assert "_root" not in msg
        assert "  " not in msg

    def test_extract_field_path_falls_back_to_root_sentinel(self) -> None:
        from cogtrix_core.api.validation import _extract_field_path

        # An empty loc tuple (or one stripped of all prefixes) must
        # surface as the ``_root`` sentinel for the special-cased
        # downstream renderer to pick up.
        assert _extract_field_path([]) == ["_root"]
        assert _extract_field_path(["body"]) == ["_root"]
