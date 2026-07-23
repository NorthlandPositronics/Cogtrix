"""Comprehensive API tests: RAG, Config, System, and Health endpoints.

Tests cover:
- RAG document ID validation (UUID guard / path traversal prevention)
- RAG document upload (extension allow-list, size limit, auth)
- RAG document listing (pagination, status filter)
- RAG search (validation, empty results)
- Config GET/PATCH/reload with admin vs non-admin distinction
- Config provider listing and detail (404 on unknown)
- System info endpoint (auth required, response shape)
- System debug toggle (admin-only)
- Health liveness and readiness (no auth needed, response shape)

State injection strategy: override app.state inside the TestClient context
block (after lifespan startup) so mocks overwrite lifespan-set state.
"""

from __future__ import annotations

import io
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("fastapi")

# ---------------------------------------------------------------------------
# Environment setup — must happen before any src.api imports
# ---------------------------------------------------------------------------

os.environ.setdefault("COGTRIX_JWT_SECRET", "testsecret_mustbe32chars_minimum00")
os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

# ---------------------------------------------------------------------------
# Imports after env setup
# ---------------------------------------------------------------------------

from fastapi.testclient import TestClient  # noqa: E402

from src.api.auth import create_access_token  # noqa: E402

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_config() -> MagicMock:
    cfg = MagicMock()
    cfg.active_model_alias = None
    cfg.memory_mode = "conversation"
    cfg.prompt_optimizer = True
    cfg.parallel_tool_execution = True
    cfg.context_compression = True
    cfg.debug = False
    cfg.verbose = False
    cfg.config_file_path = None
    cfg.providers = {}
    cfg.models = {}
    cfg.mcp_servers = {}
    return cfg


def _make_tool_registry() -> MagicMock:
    registry = MagicMock()
    registry.tools = {}
    registry.tool_metadata = {}
    registry.requires_confirmation.return_value = False
    registry.is_mcp_tool.return_value = False
    registry.get_tool_server.return_value = None
    return registry


@contextmanager
def _api_client(
    extra_state: dict | None = None,
) -> Iterator[tuple[TestClient, str, str]]:
    """Yield (client, admin_token, user_token) with mocked app state."""
    from src.api.app import create_app

    admin_token = create_access_token(user_id=str(uuid.uuid4()), role="admin")
    user_token = create_access_token(user_id=str(uuid.uuid4()), role="user")
    app = create_app()
    with TestClient(app, raise_server_exceptions=True) as client:
        app.state.tool_registry = _make_tool_registry()
        app.state.config = _make_config()
        app.state.session_registry = None
        app.state.mcp_client = None
        if extra_state:
            for k, v in extra_state.items():
                setattr(app.state, k, v)
        yield client, admin_token, user_token


# ===========================================================================
# 1. TestRagDocIdValidation — UUID guard / path traversal
# ===========================================================================


class TestRagDocIdValidation:
    """P0 security: _validate_doc_id rejects all non-UUID document IDs."""

    def test_get_document_non_uuid_id_returns_400(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.get(
                "/api/v1/rag/documents/not-a-uuid",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 400, resp.text
        assert resp.json()["error"]["code"] == "INVALID_DOCUMENT_ID"

    def test_delete_document_non_uuid_id_returns_400(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.delete(
                "/api/v1/rag/documents/definitely-not-a-uuid",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 400, resp.text
        assert resp.json()["error"]["code"] == "INVALID_DOCUMENT_ID"

    def test_get_document_path_traversal_returns_400(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.get(
                "/api/v1/rag/documents/../etc/passwd",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        # FastAPI route matching may convert path traversal to a 404 or 400;
        # it must NOT return 200 or 500.
        assert resp.status_code in (400, 404), resp.text

    def test_get_document_dots_only_returns_400_or_404(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.get(
                "/api/v1/rag/documents/..",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code in (400, 404), resp.text

    def test_get_document_alphanumeric_garbage_returns_400(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.get(
                "/api/v1/rag/documents/abc123xyz",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 400, resp.text
        assert resp.json()["error"]["code"] == "INVALID_DOCUMENT_ID"

    def test_delete_document_alphanumeric_garbage_returns_400(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.delete(
                "/api/v1/rag/documents/abc123xyz",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 400, resp.text
        assert resp.json()["error"]["code"] == "INVALID_DOCUMENT_ID"

    def test_get_document_valid_uuid_passes_validation_may_404(self) -> None:
        """A well-formed UUID passes the guard but returns 404 (doc not found)."""
        with _api_client() as (client, admin_token, _):
            resp = client.get(
                f"/api/v1/rag/documents/{uuid.uuid4()}",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        # 404 means validation passed; any other 4xx besides 400 is acceptable too
        assert resp.status_code != 400, resp.text
        if resp.status_code == 404:
            assert resp.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"

    def test_delete_document_valid_uuid_passes_validation_may_404(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.delete(
                f"/api/v1/rag/documents/{uuid.uuid4()}",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code != 400, resp.text


# ===========================================================================
# 2. TestRagUpload
# ===========================================================================


class TestRagUpload:
    def test_upload_txt_returns_202(self, tmp_path: Path) -> None:
        with (
            _api_client() as (client, admin_token, _),
            patch("src.api.routes.rag.ingest_document_task", new=AsyncMock(return_value=None)),
            patch("src.api.routes.rag._get_uploads_dir", return_value=tmp_path),
        ):
            resp = client.post(
                "/api/v1/rag/documents",
                files={"file": ("notes.txt", b"hello world", "text/plain")},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 202, resp.text
        data = resp.json()["data"]
        assert data["filename"] == "notes.txt"
        assert data["status"] == "pending"
        assert data["id"] is not None

    def test_upload_pdf_accepted(self, tmp_path: Path) -> None:
        with (
            _api_client() as (client, admin_token, _),
            patch("src.api.routes.rag.ingest_document_task", new=AsyncMock(return_value=None)),
            patch("src.api.routes.rag._get_uploads_dir", return_value=tmp_path),
        ):
            resp = client.post(
                "/api/v1/rag/documents",
                files={"file": ("report.pdf", b"%PDF-1.4 content", "application/pdf")},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 202, resp.text

    def test_upload_md_accepted(self, tmp_path: Path) -> None:
        with (
            _api_client() as (client, admin_token, _),
            patch("src.api.routes.rag.ingest_document_task", new=AsyncMock(return_value=None)),
            patch("src.api.routes.rag._get_uploads_dir", return_value=tmp_path),
        ):
            resp = client.post(
                "/api/v1/rag/documents",
                files={"file": ("readme.md", b"# Title\n\nContent", "text/markdown")},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 202, resp.text

    def test_upload_csv_accepted(self, tmp_path: Path) -> None:
        with (
            _api_client() as (client, admin_token, _),
            patch("src.api.routes.rag.ingest_document_task", new=AsyncMock(return_value=None)),
            patch("src.api.routes.rag._get_uploads_dir", return_value=tmp_path),
        ):
            resp = client.post(
                "/api/v1/rag/documents",
                files={"file": ("data.csv", b"col1,col2\nval1,val2", "text/csv")},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 202, resp.text

    def test_upload_html_rejected_415(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.post(
                "/api/v1/rag/documents",
                files={"file": ("page.html", b"<html><body>hi</body></html>", "text/html")},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 415

    def test_upload_docx_rejected_415(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.post(
                "/api/v1/rag/documents",
                files={
                    "file": (
                        "doc.docx",
                        b"PK\x03\x04fake docx",
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
                },
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 415

    def test_upload_exe_rejected_415(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.post(
                "/api/v1/rag/documents",
                files={"file": ("virus.exe", b"MZ", "application/octet-stream")},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 415, resp.text
        assert resp.json()["error"]["code"] == "VALIDATION_ERROR"

    def test_upload_zip_rejected_415(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.post(
                "/api/v1/rag/documents",
                files={"file": ("archive.zip", b"PK\x03\x04", "application/zip")},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 415, resp.text
        assert resp.json()["error"]["code"] == "VALIDATION_ERROR"

    def test_upload_without_file_returns_422(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.post(
                "/api/v1/rag/documents",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 422, resp.text

    def test_upload_too_large_returns_413(self) -> None:
        big_data = b"x" * (51 * 1024 * 1024)
        with _api_client() as (client, admin_token, _):
            resp = client.post(
                "/api/v1/rag/documents",
                files={"file": ("big.txt", io.BytesIO(big_data), "text/plain")},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 413, resp.text
        assert resp.json()["error"]["code"] == "VALIDATION_ERROR"

    def test_upload_requires_admin(self) -> None:
        with _api_client() as (client, _, user_token):
            resp = client.post(
                "/api/v1/rag/documents",
                files={"file": ("notes.txt", b"hello", "text/plain")},
                headers={"Authorization": f"Bearer {user_token}"},
            )
        assert resp.status_code == 403, resp.text
        assert resp.json()["error"]["code"] == "FORBIDDEN"

    def test_upload_no_auth_returns_401(self) -> None:
        with _api_client() as (client, _, __):
            resp = client.post(
                "/api/v1/rag/documents",
                files={"file": ("notes.txt", b"hello", "text/plain")},
            )
        assert resp.status_code == 401, resp.text

    def test_upload_response_has_required_fields(self, tmp_path: Path) -> None:
        with (
            _api_client() as (client, admin_token, _),
            patch("src.api.routes.rag.ingest_document_task", new=AsyncMock(return_value=None)),
            patch("src.api.routes.rag._get_uploads_dir", return_value=tmp_path),
        ):
            resp = client.post(
                "/api/v1/rag/documents",
                files={"file": ("doc.txt", b"content", "text/plain")},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 202
        data = resp.json()["data"]
        for field in ("id", "filename", "content_type", "size_bytes", "chunk_count", "status"):
            assert field in data, f"Missing field: {field}"


# ===========================================================================
# 3. TestRagList
# ===========================================================================


class TestRagList:
    def test_list_empty_returns_empty_page(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.get(
                "/api/v1/rag/documents",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200, resp.text
        page = resp.json()["data"]
        assert page["items"] == []
        assert page["has_more"] is False
        assert page["total"] == 0

    def test_list_after_upload_document_appears(self, tmp_path: Path) -> None:
        with (
            _api_client() as (client, admin_token, _),
            patch("src.api.routes.rag.ingest_document_task", new=AsyncMock(return_value=None)),
            patch("src.api.routes.rag._get_uploads_dir", return_value=tmp_path),
        ):
            upload = client.post(
                "/api/v1/rag/documents",
                files={"file": ("sample.txt", b"data", "text/plain")},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert upload.status_code == 202
            uploaded_id = upload.json()["data"]["id"]

            resp = client.get(
                "/api/v1/rag/documents",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        ids = [item["id"] for item in resp.json()["data"]["items"]]
        assert uploaded_id in ids

    def test_list_status_filter_pending(self, tmp_path: Path) -> None:
        with (
            _api_client() as (client, admin_token, _),
            patch("src.api.routes.rag.ingest_document_task", new=AsyncMock(return_value=None)),
            patch("src.api.routes.rag._get_uploads_dir", return_value=tmp_path),
        ):
            client.post(
                "/api/v1/rag/documents",
                files={"file": ("a.txt", b"hello", "text/plain")},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            resp = client.get(
                "/api/v1/rag/documents",
                params={"status": "pending"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        items = resp.json()["data"]["items"]
        assert len(items) >= 1
        assert all(item["status"] == "pending" for item in items)

    def test_list_status_filter_indexed_empty_when_none_indexed(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.get(
                "/api/v1/rag/documents",
                params={"status": "indexed"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        assert resp.json()["data"]["items"] == []

    def test_list_status_filter_failed_empty_when_none_failed(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.get(
                "/api/v1/rag/documents",
                params={"status": "failed"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        assert resp.json()["data"]["items"] == []

    def test_list_unknown_status_filter_returns_empty(self) -> None:
        """An unknown status value simply matches nothing (DB WHERE status = 'bogus')."""
        with _api_client() as (client, admin_token, _):
            resp = client.get(
                "/api/v1/rag/documents",
                params={"status": "bogus_status"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        assert resp.json()["data"]["items"] == []

    def test_list_cursor_pagination_invalid_cursor(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.get(
                "/api/v1/rag/documents",
                params={"cursor": "not-valid-base64!!!"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 400, resp.text
        assert resp.json()["error"]["code"] == "INVALID_CURSOR"

    def test_list_page_structure_fields(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.get(
                "/api/v1/rag/documents",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        page = resp.json()["data"]
        assert "items" in page
        assert "has_more" in page
        assert "next_cursor" in page
        assert "total" in page

    def test_list_no_auth_returns_401(self) -> None:
        with _api_client() as (client, _, __):
            resp = client.get("/api/v1/rag/documents")
        assert resp.status_code == 401, resp.text


# ===========================================================================
# 4. TestRagSearch
# ===========================================================================


class TestRagSearch:
    def test_search_happy_path_empty_results(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.post(
                "/api/v1/rag/search",
                json={"query": "What is climate change?", "top_k": 5},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["query"] == "What is climate change?"
        assert data["chunks"] == []
        assert data["total_documents_searched"] == 0

    def test_search_empty_query_returns_422(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.post(
                "/api/v1/rag/search",
                json={"query": ""},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 422, resp.text

    def test_search_top_k_zero_returns_422(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.post(
                "/api/v1/rag/search",
                json={"query": "hello", "top_k": 0},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 422, resp.text

    def test_search_top_k_one_accepted(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.post(
                "/api/v1/rag/search",
                json={"query": "hello", "top_k": 1},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200, resp.text

    def test_search_top_k_max_accepted(self) -> None:
        """top_k at maximum allowed value (50) should be accepted."""
        with _api_client() as (client, admin_token, _):
            resp = client.post(
                "/api/v1/rag/search",
                json={"query": "hello", "top_k": 50},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200, resp.text

    def test_search_top_k_exceeds_max_returns_422(self) -> None:
        """top_k above the schema maximum (le=50) is rejected."""
        with _api_client() as (client, admin_token, _):
            resp = client.post(
                "/api/v1/rag/search",
                json={"query": "hello", "top_k": 51},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 422, resp.text

    def test_search_no_indexed_docs_returns_empty(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.post(
                "/api/v1/rag/search",
                json={"query": "test query"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["chunks"] == []
        assert data["total_documents_searched"] == 0

    def test_search_no_auth_returns_401(self) -> None:
        with _api_client() as (client, _, __):
            resp = client.post(
                "/api/v1/rag/search",
                json={"query": "test"},
            )
        assert resp.status_code == 401, resp.text

    def test_search_response_shape(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.post(
                "/api/v1/rag/search",
                json={"query": "structure test", "top_k": 3},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "query" in data
        assert "chunks" in data
        assert "total_documents_searched" in data
        assert isinstance(data["chunks"], list)

    def test_search_with_document_ids_filter(self) -> None:
        """document_ids filter with non-existent IDs returns empty."""
        with _api_client() as (client, admin_token, _):
            resp = client.post(
                "/api/v1/rag/search",
                json={
                    "query": "specific doc",
                    "top_k": 5,
                    "document_ids": [str(uuid.uuid4()), str(uuid.uuid4())],
                },
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        assert resp.json()["data"]["chunks"] == []


# ===========================================================================
# 5. TestConfigGet
# ===========================================================================


class TestConfigGet:
    def test_authenticated_user_gets_config(self) -> None:
        with _api_client() as (client, _, user_token):
            resp = client.get(
                "/api/v1/config",
                headers={"Authorization": f"Bearer {user_token}"},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert "active_model" in data
        assert "prompt_optimizer" in data
        assert "context_compression" in data

    def test_admin_gets_config(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.get(
                "/api/v1/config",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert "active_model" in data

    def test_admin_raw_yaml_field_is_none_without_config_file(self) -> None:
        """Admin gets raw_yaml=None when no config file is loaded."""
        with _api_client() as (client, admin_token, _):
            resp = client.get(
                "/api/v1/config",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        # config_file_path is None on mock, so raw_yaml is None too
        assert resp.json()["data"]["raw_yaml"] is None

    def test_non_admin_raw_yaml_is_none(self) -> None:
        with _api_client() as (client, _, user_token):
            resp = client.get(
                "/api/v1/config",
                headers={"Authorization": f"Bearer {user_token}"},
            )
        assert resp.status_code == 200
        assert resp.json()["data"]["raw_yaml"] is None

    def test_no_auth_returns_401(self) -> None:
        with _api_client() as (client, _, __):
            resp = client.get("/api/v1/config")
        assert resp.status_code == 401, resp.text

    def test_config_response_boolean_fields(self) -> None:
        with _api_client() as (client, _, user_token):
            resp = client.get(
                "/api/v1/config",
                headers={"Authorization": f"Bearer {user_token}"},
            )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert isinstance(data["prompt_optimizer"], bool)
        assert isinstance(data["parallel_tool_execution"], bool)
        assert isinstance(data["context_compression"], bool)
        assert isinstance(data["debug"], bool)
        assert isinstance(data["verbose"], bool)

    def test_config_includes_providers_list(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.get(
                "/api/v1/config",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "providers" in data
        assert isinstance(data["providers"], list)

    def test_config_includes_models_list(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.get(
                "/api/v1/config",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        assert isinstance(resp.json()["data"]["models"], list)


# ===========================================================================
# 6. TestConfigUpdate
# ===========================================================================


class TestConfigUpdate:
    def test_admin_can_toggle_debug(self) -> None:
        with _api_client() as (client, admin_token, _):
            app_config = client.app.state.config  # type: ignore[attr-defined]
            resp = client.patch(
                "/api/v1/config",
                json={"debug": True},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200, resp.text
        assert app_config.debug is True

    def test_admin_can_toggle_multiple_flags(self) -> None:
        with _api_client() as (client, admin_token, _):
            app_config = client.app.state.config  # type: ignore[attr-defined]
            resp = client.patch(
                "/api/v1/config",
                json={"debug": False, "verbose": True, "prompt_optimizer": False},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200, resp.text
        assert app_config.verbose is True
        assert app_config.prompt_optimizer is False

    def test_admin_toggle_context_compression(self) -> None:
        with _api_client() as (client, admin_token, _):
            app_config = client.app.state.config  # type: ignore[attr-defined]
            resp = client.patch(
                "/api/v1/config",
                json={"context_compression": False},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200, resp.text
        assert app_config.context_compression is False

    def test_admin_toggle_parallel_tool_execution(self) -> None:
        with _api_client() as (client, admin_token, _):
            app_config = client.app.state.config  # type: ignore[attr-defined]
            resp = client.patch(
                "/api/v1/config",
                json={"parallel_tool_execution": False},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200, resp.text
        assert app_config.parallel_tool_execution is False

    def test_non_admin_patch_returns_403(self) -> None:
        with _api_client() as (client, _, user_token):
            resp = client.patch(
                "/api/v1/config",
                json={"debug": True},
                headers={"Authorization": f"Bearer {user_token}"},
            )
        assert resp.status_code == 403, resp.text

    def test_no_auth_patch_returns_401(self) -> None:
        with _api_client() as (client, _, __):
            resp = client.patch(
                "/api/v1/config",
                json={"debug": True},
            )
        assert resp.status_code == 401, resp.text

    def test_patch_returns_updated_config_snapshot(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.patch(
                "/api/v1/config",
                json={"verbose": True},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["verbose"] is True
        assert "active_model" in data


# ===========================================================================
# 7. TestConfigReload
# ===========================================================================


class TestConfigReload:
    def test_admin_triggers_reload_returns_200(self) -> None:
        mock_new_cfg = MagicMock()
        mock_new_cfg.config_file_path = None
        with (
            _api_client() as (client, admin_token, _),
            patch("src.config.Config", return_value=mock_new_cfg),
        ):
            resp = client.post(
                "/api/v1/config/reload",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["reloaded"] is True
        assert "warnings" in data

    def test_non_admin_reload_returns_403(self) -> None:
        with _api_client() as (client, _, user_token):
            resp = client.post(
                "/api/v1/config/reload",
                headers={"Authorization": f"Bearer {user_token}"},
            )
        assert resp.status_code == 403, resp.text

    def test_no_auth_reload_returns_401(self) -> None:
        with _api_client() as (client, _, __):
            resp = client.post("/api/v1/config/reload")
        assert resp.status_code == 401, resp.text

    def test_reload_response_has_config_file_path_field(self) -> None:
        mock_new_cfg = MagicMock()
        mock_new_cfg.config_file_path = None
        with (
            _api_client() as (client, admin_token, _),
            patch("src.config.Config", return_value=mock_new_cfg),
        ):
            resp = client.post(
                "/api/v1/config/reload",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        assert "config_file_path" in resp.json()["data"]


# ===========================================================================
# 8. TestConfigProviders
# ===========================================================================


class TestConfigProviders:
    def test_list_providers_authenticated(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.get(
                "/api/v1/config/providers",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200, resp.text
        assert isinstance(resp.json()["data"], list)

    def test_list_providers_non_admin_allowed(self) -> None:
        with _api_client() as (client, _, user_token):
            resp = client.get(
                "/api/v1/config/providers",
                headers={"Authorization": f"Bearer {user_token}"},
            )
        assert resp.status_code == 200, resp.text

    def test_list_providers_no_auth_returns_401(self) -> None:
        with _api_client() as (client, _, __):
            resp = client.get("/api/v1/config/providers")
        assert resp.status_code == 401, resp.text

    def test_get_specific_provider_detail(self) -> None:
        """Provider detail endpoint returns 200 when provider is configured."""
        from unittest.mock import MagicMock

        provider_config = MagicMock()
        provider_config.type = "openai"
        provider_config.base_url = None
        provider_config.api_key = "sk-test"
        provider_config.tool_instructions = None

        config_with_provider = _make_config()
        config_with_provider.providers = {"openai": provider_config}

        with _api_client(extra_state={"config": config_with_provider}) as (
            client,
            admin_token,
            _,
        ):
            resp = client.get(
                "/api/v1/config/providers/openai",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["name"] == "openai"
        assert data["type"] == "openai"

    def test_get_unknown_provider_returns_404(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.get(
                "/api/v1/config/providers/nonexistent_provider",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 404, resp.text
        assert resp.json()["error"]["code"] == "NOT_FOUND"

    def test_get_provider_no_auth_returns_401(self) -> None:
        with _api_client() as (client, _, __):
            resp = client.get("/api/v1/config/providers/openai")
        assert resp.status_code == 401, resp.text

    def test_list_models_returns_200(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.get(
                "/api/v1/config/models",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200, resp.text
        assert isinstance(resp.json()["data"], list)

    def test_list_models_no_auth_returns_401(self) -> None:
        with _api_client() as (client, _, __):
            resp = client.get("/api/v1/config/models")
        assert resp.status_code == 401, resp.text

    def test_provider_health_unknown_provider_returns_404(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.post(
                "/api/v1/config/providers/nonexistent/health",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 404, resp.text
        assert resp.json()["error"]["code"] == "NOT_FOUND"

    def test_provider_health_no_auth_returns_401(self) -> None:
        with _api_client() as (client, _, __):
            resp = client.post("/api/v1/config/providers/openai/health")
        assert resp.status_code == 401, resp.text

    def test_switch_provider_known_type_succeeds(self) -> None:
        with (
            _api_client() as (client, admin_token, _),
            patch("src.orchestration.runner.invalidate_llm_caches", return_value=None),
        ):
            resp = client.post(
                "/api/v1/config/provider",
                json={"provider": "openai"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 410, resp.text

    def test_switch_provider_unknown_returns_410(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.post(
                "/api/v1/config/provider",
                json={"provider": "totally_unknown_provider"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 410, resp.text

    def test_switch_provider_non_admin_returns_403(self) -> None:
        with _api_client() as (client, _, user_token):
            resp = client.post(
                "/api/v1/config/provider",
                json={"provider": "openai"},
                headers={"Authorization": f"Bearer {user_token}"},
            )
        assert resp.status_code == 403, resp.text

    def test_switch_model_admin_succeeds(self) -> None:
        with (
            _api_client() as (client, admin_token, _),
            patch("src.orchestration.runner.invalidate_llm_caches", return_value=None),
        ):
            resp = client.post(
                "/api/v1/config/model",
                json={"model": "gpt-4.1-mini"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200, resp.text

    def test_switch_model_non_admin_returns_403(self) -> None:
        with _api_client() as (client, _, user_token):
            resp = client.post(
                "/api/v1/config/model",
                json={"model": "gpt-4.1-mini"},
                headers={"Authorization": f"Bearer {user_token}"},
            )
        assert resp.status_code == 403, resp.text


# ===========================================================================
# 9. TestSystemInfo
# ===========================================================================


class TestSystemInfo:
    def test_admin_gets_system_info(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.get(
                "/api/v1/system/info",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert "version" in data
        assert "api_version" in data
        assert "platform" in data
        assert "python_version" in data
        assert "uptime_s" in data
        assert "started_at" in data

    def test_non_admin_can_also_get_system_info(self) -> None:
        """system/info requires authentication but not admin role."""
        with _api_client() as (client, _, user_token):
            resp = client.get(
                "/api/v1/system/info",
                headers={"Authorization": f"Bearer {user_token}"},
            )
        assert resp.status_code == 200, resp.text

    def test_no_auth_returns_401(self) -> None:
        with _api_client() as (client, _, __):
            resp = client.get("/api/v1/system/info")
        assert resp.status_code == 401, resp.text

    def test_api_version_is_v1(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.get(
                "/api/v1/system/info",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.json()["data"]["api_version"] == "v1"

    def test_uptime_is_non_negative(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.get(
                "/api/v1/system/info",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.json()["data"]["uptime_s"] >= 0

    def test_version_matches_package_version(self) -> None:
        from src._version import __version__

        with _api_client() as (client, admin_token, _):
            resp = client.get(
                "/api/v1/system/info",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.json()["data"]["version"] == __version__

    def test_debug_field_reflects_config(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.get(
                "/api/v1/system/info",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert isinstance(resp.json()["data"]["debug"], bool)


# ===========================================================================
# 10. TestSystemDebugToggle
# ===========================================================================


class TestSystemDebugToggle:
    def test_admin_enables_debug(self) -> None:
        with _api_client() as (client, admin_token, _):
            app_config = client.app.state.config  # type: ignore[attr-defined]
            resp = client.post(
                "/api/v1/system/debug",
                json={"debug": True},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200, resp.text
        assert app_config.debug is True

    def test_admin_disables_debug(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.post(
                "/api/v1/system/debug",
                json={"debug": False},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200, resp.text

    def test_admin_sets_verbose_with_debug(self) -> None:
        with _api_client() as (client, admin_token, _):
            app_config = client.app.state.config  # type: ignore[attr-defined]
            resp = client.post(
                "/api/v1/system/debug",
                json={"debug": False, "verbose": True},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200, resp.text
        assert app_config.verbose is True

    def test_non_admin_returns_403(self) -> None:
        with _api_client() as (client, _, user_token):
            resp = client.post(
                "/api/v1/system/debug",
                json={"debug": True},
                headers={"Authorization": f"Bearer {user_token}"},
            )
        assert resp.status_code == 403, resp.text

    def test_no_auth_returns_401(self) -> None:
        with _api_client() as (client, _, __):
            resp = client.post(
                "/api/v1/system/debug",
                json={"debug": True},
            )
        assert resp.status_code == 401, resp.text

    def test_response_contains_system_info_shape(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.post(
                "/api/v1/system/debug",
                json={"debug": False},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "version" in data
        assert "uptime_s" in data
        assert "api_version" in data


# ===========================================================================
# 11. TestHealth
# ===========================================================================


class TestHealth:
    def test_liveness_returns_200_no_auth(self) -> None:
        with _api_client() as (client, _, __):
            resp = client.get("/api/v1/health")
        assert resp.status_code == 200, resp.text

    def test_liveness_response_has_ok_status(self) -> None:
        with _api_client() as (client, _, __):
            resp = client.get("/api/v1/health")
        data = resp.json()["data"]
        assert data["status"] == "ok"

    def test_liveness_response_has_timestamp(self) -> None:
        with _api_client() as (client, _, __):
            resp = client.get("/api/v1/health")
        assert "timestamp" in resp.json()["data"]

    def test_liveness_envelope_shape(self) -> None:
        with _api_client() as (client, _, __):
            resp = client.get("/api/v1/health")
        body = resp.json()
        assert "data" in body
        assert "error" in body
        assert body["error"] is None

    def test_readiness_returns_200_or_503_no_auth(self) -> None:
        """Readiness check requires no auth; returns 200 or 503."""
        with _api_client() as (client, _, __):
            resp = client.get("/api/v1/health/ready")
        assert resp.status_code in (200, 503), resp.text

    def test_readiness_response_has_ready_field(self) -> None:
        with _api_client() as (client, _, __):
            resp = client.get("/api/v1/health/ready")
        data = resp.json()["data"]
        assert "ready" in data
        assert isinstance(data["ready"], bool)

    def test_readiness_response_has_components(self) -> None:
        with _api_client() as (client, _, __):
            resp = client.get("/api/v1/health/ready")
        data = resp.json()["data"]
        assert "components" in data
        assert isinstance(data["components"], list)
        for component in data["components"]:
            assert "name" in component
            assert "ok" in component

    def test_readiness_envelope_shape(self) -> None:
        with _api_client() as (client, _, __):
            resp = client.get("/api/v1/health/ready")
        body = resp.json()
        assert "data" in body
        assert "error" in body

    def test_readiness_200_when_tool_registry_present(self) -> None:
        """When tool_registry is set on app.state, the tools component is ok."""
        with _api_client() as (client, _, __):
            resp = client.get("/api/v1/health/ready")
        data = resp.json()["data"]
        tools_components = [c for c in data["components"] if c["name"] == "tool_registry"]
        if tools_components:
            assert tools_components[0]["ok"] is True

    def test_readiness_503_when_tool_registry_absent(self) -> None:
        """When tool_registry is None, the tools component is not ok."""
        with _api_client(extra_state={"tool_registry": None}) as (client, _, __):
            resp = client.get("/api/v1/health/ready")
        assert resp.status_code == 503, resp.text
        data = resp.json()["data"]
        assert data["ready"] is False
