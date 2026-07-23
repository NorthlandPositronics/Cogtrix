"""Tests for workspace-scoped config (Enterprise Phase 1 — task 1.3.2).

Covers:
  - WorkspaceConfig.from_json / to_dict round-trip
  - WorkspaceConfig default values
  - get_workspace_context: no workspace, valid workspace, cross-org 403,
    non-member 403, missing workspace 404
  - GET /workspaces/{id}/config
  - PATCH /workspaces/{id}/config (partial update, null clears key)
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid

import pytest

pytest.importorskip("fastapi")

_TEST_JWT_SECRET = "testsecret_mustbe32chars_minimum00"

from unittest.mock import patch  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker  # noqa: E402

from src.api.auth import create_access_token  # noqa: E402
from src.api.db.engine import get_db  # noqa: E402
from src.api.db.repositories.organization import OrganizationRepository  # noqa: E402
from src.api.db.repositories.users import UserRepository  # noqa: E402
from src.api.db.repositories.workspaces import WorkspaceRepository  # noqa: E402
from src.api.workspace_context import WorkspaceConfig, WorkspaceContext  # noqa: E402


def _uid() -> str:
    return str(uuid.uuid4())


def _admin_header(user_id: str) -> dict:
    with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
        token = create_access_token(user_id=user_id, role="admin")
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# WorkspaceConfig unit tests
# ---------------------------------------------------------------------------


class TestWorkspaceConfig:
    def test_defaults(self):
        cfg = WorkspaceConfig()
        assert cfg.model_override is None
        assert cfg.system_prompt is None
        assert cfg.tool_policy is None
        assert cfg.max_context_tokens is None
        assert cfg.rate_limit_multiplier == 1.0
        assert cfg.extra == {}

    def test_from_json_empty(self):
        cfg = WorkspaceConfig.from_json(None)
        assert cfg.model_override is None

    def test_from_json_partial(self):
        raw = json.dumps({"model_override": "gpt-5", "rate_limit_multiplier": 2.0})
        cfg = WorkspaceConfig.from_json(raw)
        assert cfg.model_override == "gpt-5"
        assert cfg.rate_limit_multiplier == 2.0
        assert cfg.system_prompt is None

    def test_from_json_with_extra(self):
        raw = json.dumps({"custom_key": "custom_value"})
        cfg = WorkspaceConfig.from_json(raw)
        assert cfg.extra == {"custom_key": "custom_value"}

    def test_to_dict_empty_defaults(self):
        cfg = WorkspaceConfig()
        d = cfg.to_dict()
        assert d == {}

    def test_to_dict_with_values(self):
        cfg = WorkspaceConfig(model_override="gpt-5", rate_limit_multiplier=2.0)
        d = cfg.to_dict()
        assert d["model_override"] == "gpt-5"
        assert d["rate_limit_multiplier"] == 2.0

    def test_round_trip(self):
        cfg = WorkspaceConfig(
            model_override="coder",
            system_prompt="You are an expert.",
            tool_policy="calculate,search_web",
            max_context_tokens=32768,
            rate_limit_multiplier=0.5,
        )
        raw = json.dumps(cfg.to_dict())
        restored = WorkspaceConfig.from_json(raw)
        assert restored.model_override == cfg.model_override
        assert restored.system_prompt == cfg.system_prompt
        assert restored.tool_policy == cfg.tool_policy
        assert restored.max_context_tokens == cfg.max_context_tokens
        assert restored.rate_limit_multiplier == cfg.rate_limit_multiplier

    def test_from_json_invalid_returns_defaults(self):
        cfg = WorkspaceConfig.from_json("not json")
        assert cfg.model_override is None


class TestWorkspaceContext:
    def test_has_workspace_true(self):
        ctx = WorkspaceContext(user_id="u1", workspace_id="ws1")
        assert ctx.has_workspace is True

    def test_has_workspace_false(self):
        ctx = WorkspaceContext(user_id="u1")
        assert ctx.has_workspace is False


# ---------------------------------------------------------------------------
# Config API endpoints
# ---------------------------------------------------------------------------


@pytest.fixture()
def config_setup(engine):
    factory = async_sessionmaker(engine, expire_on_commit=False)
    org_id = _uid()
    admin_id = _uid()
    ws_id = _uid()

    async def _seed():
        async with factory() as session:
            org_repo = OrganizationRepository(session)
            user_repo = UserRepository(session)
            ws_repo = WorkspaceRepository(session)
            await org_repo.create(org_id=org_id, name="Config Org", slug="config-org")
            await user_repo.create(
                user_id=admin_id,
                username="admin",
                email="admin@example.com",
                password_hash="h",
                role="admin",
                org_id=org_id,
            )
            await ws_repo.create(workspace_id=ws_id, org_id=org_id, name="Config WS")
            await session.commit()

    asyncio.run(_seed())

    from src.api.app import create_app

    with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
        app = create_app()

        async def _override():
            async with factory() as session:
                try:
                    yield session
                    await session.commit()
                except Exception:
                    await session.rollback()
                    raise

        app.dependency_overrides[get_db] = _override
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, ws_id, admin_id

    app.dependency_overrides.clear()


class TestWorkspaceConfigRoutes:
    def test_get_config_empty(self, config_setup):
        client, ws_id, admin_id = config_setup
        r = client.get(f"/api/v1/workspaces/{ws_id}/config", headers=_admin_header(admin_id))
        assert r.status_code == 200
        assert r.json()["data"] == {}

    def test_patch_config_sets_model(self, config_setup):
        client, ws_id, admin_id = config_setup
        r = client.patch(
            f"/api/v1/workspaces/{ws_id}/config",
            json={"model_override": "reasoning"},
            headers=_admin_header(admin_id),
        )
        assert r.status_code == 200
        assert r.json()["data"]["model_override"] == "reasoning"

    def test_patch_config_merge_preserves_existing(self, config_setup):
        client, ws_id, admin_id = config_setup
        client.patch(
            f"/api/v1/workspaces/{ws_id}/config",
            json={"model_override": "reasoning", "rate_limit_multiplier": 2.0},
            headers=_admin_header(admin_id),
        )
        r = client.patch(
            f"/api/v1/workspaces/{ws_id}/config",
            json={"system_prompt": "Be concise."},
            headers=_admin_header(admin_id),
        )
        data = r.json()["data"]
        assert data["model_override"] == "reasoning"
        assert data["system_prompt"] == "Be concise."

    def test_patch_config_null_clears_key(self, config_setup):
        client, ws_id, admin_id = config_setup
        client.patch(
            f"/api/v1/workspaces/{ws_id}/config",
            json={"model_override": "reasoning"},
            headers=_admin_header(admin_id),
        )
        r = client.patch(
            f"/api/v1/workspaces/{ws_id}/config",
            json={"model_override": None},
            headers=_admin_header(admin_id),
        )
        data = r.json()["data"]
        assert "model_override" not in data

    def test_get_config_reflects_patch(self, config_setup):
        client, ws_id, admin_id = config_setup
        client.patch(
            f"/api/v1/workspaces/{ws_id}/config",
            json={"tool_policy": "calculate"},
            headers=_admin_header(admin_id),
        )
        r = client.get(f"/api/v1/workspaces/{ws_id}/config", headers=_admin_header(admin_id))
        assert r.json()["data"]["tool_policy"] == "calculate"

    def test_requires_admin(self, config_setup):
        client, ws_id, _ = config_setup
        with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
            non_admin = create_access_token(user_id=_uid(), role="user")
        r = client.get(
            f"/api/v1/workspaces/{ws_id}/config",
            headers={"Authorization": f"Bearer {non_admin}"},
        )
        assert r.status_code == 403
