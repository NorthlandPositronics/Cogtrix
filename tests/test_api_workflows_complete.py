"""Comprehensive workflow API endpoint coverage.

Covers all endpoints in src/api/routes/workflows.py:
    GET    /api/v1/assistant/workflows                           — list
    POST   /api/v1/assistant/workflows                           — create (admin)
    GET    /api/v1/assistant/workflows/bindings                  — list bindings
    GET    /api/v1/assistant/workflows/{id}                      — get
    PUT    /api/v1/assistant/workflows/{id}                      — update (admin)
    DELETE /api/v1/assistant/workflows/{id}                      — delete (admin)
    POST   /api/v1/assistant/workflows/{id}/documents            — upload doc (admin)
    GET    /api/v1/assistant/workflows/{id}/documents            — list docs (admin)
    DELETE /api/v1/assistant/workflows/{id}/documents/{doc_id}   — delete doc (admin)
    PUT    /api/v1/assistant/workflows/{id}/bind                 — bind (admin)
    DELETE /api/v1/assistant/workflows/{id}/bind                 — unbind (admin)

All auth permutations: unauthenticated, non-admin, admin.
Error codes: UNAUTHORIZED, FORBIDDEN, NOT_FOUND, VALIDATION_ERROR,
             SERVICE_UNAVAILABLE, WORKFLOW_ALREADY_EXISTS, WORKFLOW_NOT_FOUND.
"""

from __future__ import annotations

import asyncio as _asyncio
import os
import uuid

import pytest

pytest.importorskip("fastapi")

_TEST_JWT_SECRET = "testsecret_mustbe32chars_minimum00"
os.environ.setdefault("COGTRIX_JWT_SECRET", _TEST_JWT_SECRET)
os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

from unittest.mock import patch  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from src.api.db.engine import Base, get_db  # noqa: E402

_VALID_PASSWORD = "TestPass1!"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def app():
    from src.api.app import create_app

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

    _asyncio.run(_setup())

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

    _asyncio.run(engine.dispose())


@pytest.fixture()
def client(app, tmp_path):
    from src.assistant.workflows import WorkflowRegistry

    with TestClient(app, raise_server_exceptions=False) as c:
        # Override with a fresh registry AFTER the lifespan startup runs
        registry = WorkflowRegistry(data_dir=str(tmp_path))
        app.state.workflow_registry = registry
        yield c


def _register(client, username=None, password=_VALID_PASSWORD):
    if username is None:
        username = f"u_{uuid.uuid4().hex[:8]}"
    r = client.post(
        "/api/v1/auth/register",
        json={"username": username, "email": f"{username}@ex.com", "password": password},
    )
    assert r.status_code == 201, f"register failed: {r.text}"
    return r.json()["data"]["access_token"]


def _h(token):
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def tokens(client):
    """Admin token (first registered) and regular user token."""
    admin_token = _register(client)
    user_token = _register(client)
    return {"admin": admin_token, "user": user_token}


_WF_BASE = {
    "id": "test-wf",
    "name": "Test Workflow",
    "description": "A test workflow",
    "tool_policy": {"excluded_tools": [], "additional_approved_tools": []},
    "auto_detect": {"enabled": False, "keywords": [], "patterns": [], "min_confidence": 1},
}

_BASE_URL = "/api/v1/assistant/workflows"


def _create_wf(client, admin_token, wf_id=None, name=None):
    body = dict(_WF_BASE)
    if wf_id:
        body["id"] = wf_id
    if name:
        body["name"] = name
    return client.post(_BASE_URL, headers=_h(admin_token), json=body)


# ---------------------------------------------------------------------------
# Service unavailable (no registry)
# ---------------------------------------------------------------------------


class TestNoRegistry:
    def test_list_503_when_no_registry(self, client, tokens, app):
        app.state.workflow_registry = None
        r = client.get(_BASE_URL, headers=_h(tokens["admin"]))
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "SERVICE_UNAVAILABLE"


# ---------------------------------------------------------------------------
# Workflow ID validation
# ---------------------------------------------------------------------------


class TestWorkflowIdValidation:
    def test_invalid_id_leading_hyphen_returns_error(self, client, tokens):
        # create_workflow uses registry.create_workflow which raises ValueError → 409
        # get/update/delete calls _validate_wf_id → 400
        r = client.post(
            _BASE_URL,
            headers=_h(tokens["admin"]),
            json={**_WF_BASE, "id": "-bad-id"},
        )
        assert r.status_code in (400, 409)

    def test_get_invalid_id_via_get_returns_400(self, client, tokens):
        r = client.get(f"{_BASE_URL}/-bad-id", headers=_h(tokens["admin"]))
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "VALIDATION_ERROR"

    def test_invalid_id_with_slash_400(self, client, tokens):
        r = client.get(
            f"{_BASE_URL}/../secret",
            headers=_h(tokens["admin"]),
        )
        # URL path traversal attempt — FastAPI returns 404 or 400
        assert r.status_code in (400, 404, 422)


# ---------------------------------------------------------------------------
# List workflows
# ---------------------------------------------------------------------------


class TestListWorkflows:
    def test_list_empty(self, client, tokens):
        r = client.get(_BASE_URL, headers=_h(tokens["admin"]))
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["items"] == []
        assert data["total"] == 0

    def test_list_shows_created_workflow(self, client, tokens):
        _create_wf(client, tokens["admin"])
        r = client.get(_BASE_URL, headers=_h(tokens["admin"]))
        assert r.status_code == 200
        items = r.json()["data"]["items"]
        assert len(items) == 1
        assert items[0]["id"] == "test-wf"
        assert items[0]["name"] == "Test Workflow"

    def test_list_non_admin_user_can_list(self, client, tokens):
        r = client.get(_BASE_URL, headers=_h(tokens["user"]))
        assert r.status_code == 200

    def test_list_no_auth_returns_401(self, client):
        r = client.get(_BASE_URL)
        assert r.status_code == 401


# ---------------------------------------------------------------------------
# Create workflow
# ---------------------------------------------------------------------------


class TestCreateWorkflow:
    def test_admin_can_create(self, client, tokens):
        r = _create_wf(client, tokens["admin"])
        assert r.status_code == 201
        data = r.json()["data"]
        assert data["id"] == "test-wf"
        assert data["name"] == "Test Workflow"
        assert "tool_policy" in data
        assert "auto_detect" in data
        assert "created_at" in data

    def test_duplicate_id_returns_409(self, client, tokens):
        _create_wf(client, tokens["admin"])
        r = _create_wf(client, tokens["admin"])
        assert r.status_code == 409
        # Registry raises ValueError → 409 CONFLICT
        assert r.json()["error"]["code"] == "CONFLICT"

    def test_non_admin_returns_403(self, client, tokens):
        r = _create_wf(client, tokens["user"])
        assert r.status_code == 403
        assert r.json()["error"]["code"] == "FORBIDDEN"

    def test_no_auth_returns_401(self, client):
        r = client.post(_BASE_URL, json=_WF_BASE)
        assert r.status_code == 401

    def test_missing_id_returns_422(self, client, tokens):
        body = {k: v for k, v in _WF_BASE.items() if k != "id"}
        r = client.post(_BASE_URL, headers=_h(tokens["admin"]), json=body)
        assert r.status_code == 422

    def test_missing_name_returns_422(self, client, tokens):
        body = {k: v for k, v in _WF_BASE.items() if k != "name"}
        r = client.post(_BASE_URL, headers=_h(tokens["admin"]), json=body)
        assert r.status_code == 422

    def test_invalid_id_pattern_returns_409_or_400(self, client, tokens):
        # create_workflow does not call _validate_wf_id; invalid ID hits registry
        # which raises ValueError → 409 CONFLICT
        body = {**_WF_BASE, "id": "-invalid"}
        r = client.post(_BASE_URL, headers=_h(tokens["admin"]), json=body)
        # Either 400 (future: pre-validation) or 409 (current: registry ValueError)
        assert r.status_code in (400, 409)

    def test_create_with_system_prompt(self, client, tokens):
        body = {**_WF_BASE, "id": "sp-wf", "system_prompt": "You are a helper."}
        r = client.post(_BASE_URL, headers=_h(tokens["admin"]), json=body)
        assert r.status_code == 201
        assert r.json()["data"]["system_prompt"] == "You are a helper."

    def test_create_with_tool_policy(self, client, tokens):
        body = {
            **_WF_BASE,
            "id": "tp-wf",
            "tool_policy": {
                "excluded_tools": ["shell"],
                "additional_approved_tools": ["web_search"],
            },
        }
        r = client.post(_BASE_URL, headers=_h(tokens["admin"]), json=body)
        assert r.status_code == 201
        tp = r.json()["data"]["tool_policy"]
        assert "shell" in tp["excluded_tools"]
        assert "web_search" in tp["additional_approved_tools"]


# ---------------------------------------------------------------------------
# Get workflow
# ---------------------------------------------------------------------------


class TestGetWorkflow:
    def test_get_existing(self, client, tokens):
        _create_wf(client, tokens["admin"])
        r = client.get(f"{_BASE_URL}/test-wf", headers=_h(tokens["admin"]))
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["id"] == "test-wf"

    def test_get_nonexistent_returns_404(self, client, tokens):
        r = client.get(f"{_BASE_URL}/does-not-exist", headers=_h(tokens["admin"]))
        assert r.status_code == 404
        assert r.json()["error"]["code"] in ("WORKFLOW_NOT_FOUND", "NOT_FOUND")

    def test_get_non_admin_can_read(self, client, tokens):
        _create_wf(client, tokens["admin"])
        r = client.get(f"{_BASE_URL}/test-wf", headers=_h(tokens["user"]))
        assert r.status_code == 200

    def test_get_no_auth_returns_401(self, client):
        r = client.get(f"{_BASE_URL}/test-wf")
        assert r.status_code == 401

    def test_get_invalid_id_returns_400(self, client, tokens):
        r = client.get(f"{_BASE_URL}/-invalid", headers=_h(tokens["admin"]))
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Update workflow
# ---------------------------------------------------------------------------


class TestUpdateWorkflow:
    def test_admin_can_update_name(self, client, tokens):
        _create_wf(client, tokens["admin"])
        r = client.put(
            f"{_BASE_URL}/test-wf",
            headers=_h(tokens["admin"]),
            json={"name": "Updated Name"},
        )
        assert r.status_code == 200
        assert r.json()["data"]["name"] == "Updated Name"

    def test_admin_can_update_description(self, client, tokens):
        _create_wf(client, tokens["admin"])
        r = client.put(
            f"{_BASE_URL}/test-wf",
            headers=_h(tokens["admin"]),
            json={"description": "New description"},
        )
        assert r.status_code == 200
        assert r.json()["data"]["description"] == "New description"

    def test_admin_can_update_system_prompt(self, client, tokens):
        _create_wf(client, tokens["admin"])
        r = client.put(
            f"{_BASE_URL}/test-wf",
            headers=_h(tokens["admin"]),
            json={"system_prompt": "You are helpful."},
        )
        assert r.status_code == 200
        assert r.json()["data"]["system_prompt"] == "You are helpful."

    def test_update_nonexistent_returns_404(self, client, tokens):
        r = client.put(
            f"{_BASE_URL}/no-such-wf",
            headers=_h(tokens["admin"]),
            json={"name": "Update"},
        )
        assert r.status_code == 404

    def test_non_admin_returns_403(self, client, tokens):
        _create_wf(client, tokens["admin"])
        r = client.put(
            f"{_BASE_URL}/test-wf",
            headers=_h(tokens["user"]),
            json={"name": "Hijack"},
        )
        assert r.status_code == 403

    def test_no_auth_returns_401(self, client):
        r = client.put(f"{_BASE_URL}/test-wf", json={"name": "X"})
        assert r.status_code == 401

    def test_update_invalid_id_returns_400(self, client, tokens):
        r = client.put(
            f"{_BASE_URL}/-invalid",
            headers=_h(tokens["admin"]),
            json={"name": "X"},
        )
        assert r.status_code == 400

    def test_empty_body_does_not_crash(self, client, tokens):
        _create_wf(client, tokens["admin"])
        r = client.put(
            f"{_BASE_URL}/test-wf",
            headers=_h(tokens["admin"]),
            json={},
        )
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Delete workflow
# ---------------------------------------------------------------------------


class TestDeleteWorkflow:
    def test_admin_can_delete(self, client, tokens):
        _create_wf(client, tokens["admin"])
        r = client.delete(f"{_BASE_URL}/test-wf", headers=_h(tokens["admin"]))
        assert r.status_code == 200
        assert r.json()["data"] is None

    def test_deleted_workflow_not_in_list(self, client, tokens):
        _create_wf(client, tokens["admin"])
        client.delete(f"{_BASE_URL}/test-wf", headers=_h(tokens["admin"]))
        r = client.get(_BASE_URL, headers=_h(tokens["admin"]))
        ids = [w["id"] for w in r.json()["data"]["items"]]
        assert "test-wf" not in ids

    def test_delete_nonexistent_returns_404(self, client, tokens):
        r = client.delete(f"{_BASE_URL}/no-such-wf", headers=_h(tokens["admin"]))
        assert r.status_code == 404

    def test_non_admin_returns_403(self, client, tokens):
        _create_wf(client, tokens["admin"])
        r = client.delete(f"{_BASE_URL}/test-wf", headers=_h(tokens["user"]))
        assert r.status_code == 403

    def test_no_auth_returns_401(self, client):
        r = client.delete(f"{_BASE_URL}/test-wf")
        assert r.status_code == 401

    def test_delete_invalid_id_returns_400(self, client, tokens):
        r = client.delete(f"{_BASE_URL}/-bad", headers=_h(tokens["admin"]))
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Workflow bindings  (PUT/DELETE /bindings/{session_key:path})
# ---------------------------------------------------------------------------


class TestWorkflowBindings:
    def test_list_bindings_empty(self, client, tokens):
        r = client.get(f"{_BASE_URL}/bindings", headers=_h(tokens["admin"]))
        assert r.status_code == 200
        assert r.json()["data"] == []

    def test_list_bindings_no_auth_returns_401(self, client):
        r = client.get(f"{_BASE_URL}/bindings")
        assert r.status_code == 401

    def test_bind_workflow(self, client, tokens):
        _create_wf(client, tokens["admin"])
        r = client.put(
            f"{_BASE_URL}/bindings/whatsapp::test-chat-123",
            headers=_h(tokens["admin"]),
            json={"workflow_id": "test-wf"},
        )
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["workflow_id"] == "test-wf"
        assert data["session_key"] == "whatsapp::test-chat-123"

    def test_bind_nonexistent_workflow_returns_404(self, client, tokens):
        r = client.put(
            f"{_BASE_URL}/bindings/wa::123",
            headers=_h(tokens["admin"]),
            json={"workflow_id": "no-such"},
        )
        assert r.status_code == 404

    def test_bind_non_admin_returns_403(self, client, tokens):
        _create_wf(client, tokens["admin"])
        r = client.put(
            f"{_BASE_URL}/bindings/wa::123",
            headers=_h(tokens["user"]),
            json={"workflow_id": "test-wf"},
        )
        assert r.status_code == 403

    def test_bind_no_auth_returns_401(self, client):
        r = client.put(
            f"{_BASE_URL}/bindings/wa::123",
            json={"workflow_id": "test-wf"},
        )
        assert r.status_code == 401

    def test_unbind_workflow(self, client, tokens):
        _create_wf(client, tokens["admin"])
        client.put(
            f"{_BASE_URL}/bindings/wa::unbind-chat",
            headers=_h(tokens["admin"]),
            json={"workflow_id": "test-wf"},
        )
        r = client.delete(
            f"{_BASE_URL}/bindings/wa::unbind-chat",
            headers=_h(tokens["admin"]),
        )
        assert r.status_code == 200

    def test_bind_missing_workflow_id_returns_422(self, client, tokens):
        _create_wf(client, tokens["admin"])
        r = client.put(
            f"{_BASE_URL}/bindings/wa::123",
            headers=_h(tokens["admin"]),
            json={},
        )
        assert r.status_code == 422

    def test_bind_shows_in_bindings_list(self, client, tokens):
        _create_wf(client, tokens["admin"])
        client.put(
            f"{_BASE_URL}/bindings/wa::chat-for-list",
            headers=_h(tokens["admin"]),
            json={"workflow_id": "test-wf"},
        )
        r = client.get(f"{_BASE_URL}/bindings", headers=_h(tokens["admin"]))
        assert r.status_code == 200
        session_keys = [b["session_key"] for b in r.json()["data"]]
        assert "wa::chat-for-list" in session_keys

    def test_unbind_non_admin_returns_403(self, client, tokens):
        _create_wf(client, tokens["admin"])
        r = client.delete(
            f"{_BASE_URL}/bindings/wa::123",
            headers=_h(tokens["user"]),
        )
        assert r.status_code == 403

    def test_unbind_no_auth_returns_401(self, client):
        r = client.delete(f"{_BASE_URL}/bindings/wa::123")
        assert r.status_code == 401


# ---------------------------------------------------------------------------
# Workflow documents
# ---------------------------------------------------------------------------


class TestWorkflowDocuments:
    def test_list_docs_empty(self, client, tokens):
        _create_wf(client, tokens["admin"])
        r = client.get(
            f"{_BASE_URL}/test-wf/documents",
            headers=_h(tokens["admin"]),
        )
        assert r.status_code == 200
        # list_workflow_documents returns APIResponse[list] not paginated
        data = r.json()["data"]
        assert isinstance(data, list)
        assert data == []

    def test_list_docs_no_auth_returns_401(self, client):
        r = client.get(f"{_BASE_URL}/test-wf/documents")
        assert r.status_code == 401

    def test_list_docs_nonexistent_workflow_returns_404(self, client, tokens):
        r = client.get(
            f"{_BASE_URL}/no-such-wf/documents",
            headers=_h(tokens["admin"]),
        )
        assert r.status_code == 404

    def test_upload_doc_invalid_extension_returns_415(self, client, tokens):
        _create_wf(client, tokens["admin"])
        r = client.post(
            f"{_BASE_URL}/test-wf/documents",
            headers=_h(tokens["admin"]),
            files={"file": ("malware.exe", b"binary", "application/octet-stream")},
        )
        assert r.status_code == 415
        assert r.json()["error"]["code"] == "VALIDATION_ERROR"

    def test_upload_doc_non_admin_returns_403(self, client, tokens):
        _create_wf(client, tokens["admin"])
        r = client.post(
            f"{_BASE_URL}/test-wf/documents",
            headers=_h(tokens["user"]),
            files={"file": ("doc.txt", b"hello world", "text/plain")},
        )
        assert r.status_code == 403

    def test_upload_doc_no_auth_returns_401(self, client):
        r = client.post(
            f"{_BASE_URL}/test-wf/documents",
            files={"file": ("doc.txt", b"hello", "text/plain")},
        )
        assert r.status_code == 401

    def test_upload_and_list_txt_doc(self, client, tokens):
        _create_wf(client, tokens["admin"])
        r = client.post(
            f"{_BASE_URL}/test-wf/documents",
            headers=_h(tokens["admin"]),
            files={"file": ("guide.txt", b"This is a guide.", "text/plain")},
        )
        assert r.status_code == 202  # upload returns 202 Accepted
        data = r.json()["data"]
        assert data["filename"] == "guide.txt"
        assert "doc_id" in data

        list_r = client.get(
            f"{_BASE_URL}/test-wf/documents",
            headers=_h(tokens["admin"]),
        )
        assert list_r.status_code == 200
        # Returns plain list, not paginated
        items = list_r.json()["data"]
        filenames = [d["filename"] for d in items]
        assert "guide.txt" in filenames

    def test_delete_doc(self, client, tokens):
        _create_wf(client, tokens["admin"])
        upload_r = client.post(
            f"{_BASE_URL}/test-wf/documents",
            headers=_h(tokens["admin"]),
            files={"file": ("to-delete.txt", b"content", "text/plain")},
        )
        assert upload_r.status_code == 202  # upload returns 202
        doc_id = upload_r.json()["data"]["doc_id"]

        r = client.delete(
            f"{_BASE_URL}/test-wf/documents/{doc_id}",
            headers=_h(tokens["admin"]),
        )
        assert r.status_code == 200

    def test_delete_nonexistent_doc_returns_404(self, client, tokens):
        _create_wf(client, tokens["admin"])
        r = client.delete(
            f"{_BASE_URL}/test-wf/documents/{uuid.uuid4()}",
            headers=_h(tokens["admin"]),
        )
        assert r.status_code == 404

    def test_delete_doc_non_admin_returns_403(self, client, tokens):
        _create_wf(client, tokens["admin"])
        r = client.delete(
            f"{_BASE_URL}/test-wf/documents/{uuid.uuid4()}",
            headers=_h(tokens["user"]),
        )
        assert r.status_code == 403

    def test_delete_doc_no_auth_returns_401(self, client):
        r = client.delete(f"{_BASE_URL}/test-wf/documents/{uuid.uuid4()}")
        assert r.status_code == 401
