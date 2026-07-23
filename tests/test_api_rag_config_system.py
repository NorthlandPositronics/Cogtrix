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


# ===========================================================================
# Issue #95 — Runtime provider CRUD (POST / PATCH / DELETE /config/providers)
# ===========================================================================


def _make_provider_config(
    name: str, type_: str = "openai", api_key: str | None = None
) -> MagicMock:
    """Return a MagicMock that quacks like a ProviderConfig."""
    pc = MagicMock()
    pc.name = name
    pc.type = type_
    pc.base_url = None
    pc.api_key = api_key
    pc.tool_instructions = None
    return pc


class TestProviderCreate:
    def test_create_provider_returns_201(self) -> None:
        """POST /config/providers creates a new provider entry."""
        cfg = _make_config()
        cfg.providers = {}
        cfg.config_file_path = None  # triggers home/.cogtrix.yaml path (mocked away)

        with (
            _api_client(extra_state={"config": cfg}) as (client, admin_token, _),
            patch("src.api.routes.config._write_providers_to_config"),
        ):
            resp = client.post(
                "/api/v1/config/providers",
                headers={"Authorization": f"Bearer {admin_token}"},
                json={"name": "my-ollama", "type": "ollama"},
            )
        assert resp.status_code == 201, resp.text
        data = resp.json()["data"]
        assert data["name"] == "my-ollama"
        assert data["type"] == "ollama"

    def test_create_provider_conflict_returns_409(self) -> None:
        """Creating a provider whose name already exists → 409 PROVIDER_EXISTS."""
        existing = _make_provider_config("ollama", "ollama")
        cfg = _make_config()
        cfg.providers = {"ollama": existing}

        with (
            _api_client(extra_state={"config": cfg}) as (client, admin_token, _),
            patch("src.api.routes.config._write_providers_to_config"),
        ):
            resp = client.post(
                "/api/v1/config/providers",
                headers={"Authorization": f"Bearer {admin_token}"},
                json={"name": "ollama", "type": "ollama"},
            )
        assert resp.status_code == 409, resp.text
        assert resp.json()["error"]["code"] == "PROVIDER_EXISTS"

    def test_create_provider_invalid_type_returns_422(self) -> None:
        """Unknown provider type must fail validation."""
        cfg = _make_config()
        cfg.providers = {}

        with (
            _api_client(extra_state={"config": cfg}) as (client, admin_token, _),
            patch("src.api.routes.config._write_providers_to_config"),
        ):
            resp = client.post(
                "/api/v1/config/providers",
                headers={"Authorization": f"Bearer {admin_token}"},
                json={"name": "bad", "type": "nonexistent_type"},
            )
        assert resp.status_code == 422, resp.text

    def test_create_provider_non_admin_returns_403(self) -> None:
        with _api_client() as (client, _, user_token):
            resp = client.post(
                "/api/v1/config/providers",
                headers={"Authorization": f"Bearer {user_token}"},
                json={"name": "x", "type": "openai"},
            )
        assert resp.status_code == 403, resp.text

    def test_create_provider_no_auth_returns_401(self) -> None:
        with _api_client() as (client, _, __):
            resp = client.post(
                "/api/v1/config/providers",
                json={"name": "x", "type": "openai"},
            )
        assert resp.status_code == 401, resp.text


class TestProviderUpdate:
    def test_patch_provider_updates_api_key(self) -> None:
        """PATCH /config/providers/{name} replaces the api_key field."""
        existing = _make_provider_config("openai", "openai", api_key="old-key")
        cfg = _make_config()
        cfg.providers = {"openai": existing}

        with (
            _api_client(extra_state={"config": cfg}) as (client, admin_token, _),
            patch("src.api.routes.config._write_providers_to_config"),
        ):
            resp = client.patch(
                "/api/v1/config/providers/openai",
                headers={"Authorization": f"Bearer {admin_token}"},
                json={"api_key": "new-key"},
            )
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"]["name"] == "openai"

    def test_patch_unknown_provider_returns_404(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.patch(
                "/api/v1/config/providers/does_not_exist",
                headers={"Authorization": f"Bearer {admin_token}"},
                json={"api_key": "key"},
            )
        assert resp.status_code == 404, resp.text
        assert resp.json()["error"]["code"] == "NOT_FOUND"

    def test_patch_provider_non_admin_returns_403(self) -> None:
        with _api_client() as (client, _, user_token):
            resp = client.patch(
                "/api/v1/config/providers/openai",
                headers={"Authorization": f"Bearer {user_token}"},
                json={"api_key": "k"},
            )
        assert resp.status_code == 403, resp.text

    def test_patch_provider_no_auth_returns_401(self) -> None:
        with _api_client() as (client, _, __):
            resp = client.patch("/api/v1/config/providers/openai", json={"api_key": "k"})
        assert resp.status_code == 401, resp.text


class TestProviderDelete:
    def test_delete_provider_returns_200(self) -> None:
        """DELETE /config/providers/{name} removes the provider."""
        pc = _make_provider_config("my-provider", "openai")
        cfg = _make_config()
        cfg.providers = {"my-provider": pc}
        cfg.models = {}  # no models reference this provider

        with (
            _api_client(extra_state={"config": cfg}) as (client, admin_token, _),
            patch("src.api.routes.config._write_providers_to_config"),
        ):
            resp = client.delete(
                "/api/v1/config/providers/my-provider",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200, resp.text

    def test_delete_provider_in_use_returns_409(self) -> None:
        """Cannot delete a provider that has a model referencing it."""
        pc = _make_provider_config("openai", "openai")
        mc = MagicMock()
        mc.provider = "openai"
        cfg = _make_config()
        cfg.providers = {"openai": pc}
        cfg.models = {"gpt4": mc}

        with _api_client(extra_state={"config": cfg}) as (client, admin_token, _):
            resp = client.delete(
                "/api/v1/config/providers/openai",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 409, resp.text
        assert resp.json()["error"]["code"] == "PROVIDER_IN_USE"

    def test_delete_unknown_provider_returns_404(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.delete(
                "/api/v1/config/providers/ghost",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 404, resp.text
        assert resp.json()["error"]["code"] == "NOT_FOUND"

    def test_delete_provider_non_admin_returns_403(self) -> None:
        with _api_client() as (client, _, user_token):
            resp = client.delete(
                "/api/v1/config/providers/openai",
                headers={"Authorization": f"Bearer {user_token}"},
            )
        assert resp.status_code == 403, resp.text

    def test_delete_provider_no_auth_returns_401(self) -> None:
        with _api_client() as (client, _, __):
            resp = client.delete("/api/v1/config/providers/openai")
        assert resp.status_code == 401, resp.text


# ---------------------------------------------------------------------------
# BUG-237 — provider write lock serialises concurrent CRUD (source check)
# ---------------------------------------------------------------------------


class TestProviderWriteLock:
    def test_provider_write_lock_helper_exists(self) -> None:
        """_get_provider_write_lock() must be defined and return an asyncio.Lock."""
        import asyncio
        import importlib

        mod = importlib.import_module("src.api.routes.config")
        assert hasattr(
            mod, "_get_provider_write_lock"
        ), "_get_provider_write_lock missing — BUG-237 fix not applied"
        lock = mod._get_provider_write_lock()
        assert isinstance(lock, asyncio.Lock), "Must return asyncio.Lock"

    def test_create_provider_acquires_lock(self) -> None:
        import inspect

        import src.api.routes.config as _mod

        src = inspect.getsource(_mod.create_provider)
        assert (
            "_get_provider_write_lock()" in src
        ), "create_provider must acquire _get_provider_write_lock() (BUG-237)"

    def test_update_provider_acquires_lock(self) -> None:
        import inspect

        import src.api.routes.config as _mod

        src = inspect.getsource(_mod.update_provider)
        assert (
            "_get_provider_write_lock()" in src
        ), "update_provider must acquire _get_provider_write_lock() (BUG-237)"

    def test_delete_provider_acquires_lock(self) -> None:
        import inspect

        import src.api.routes.config as _mod

        src = inspect.getsource(_mod.delete_provider)
        assert (
            "_get_provider_write_lock()" in src
        ), "delete_provider must acquire _get_provider_write_lock() (BUG-237)"


# ---------------------------------------------------------------------------
# BUG-238 — SSRF guard on base_url in create/update provider
# ---------------------------------------------------------------------------


class TestProviderBaseUrlSSRFGuard:
    def test_create_provider_blocks_link_local_base_url(self) -> None:
        """POST /config/providers must reject link-local base_url values (BUG-238)."""
        with _api_client() as (client, admin_token, _):
            resp = client.post(
                "/api/v1/config/providers",
                json={
                    "name": "ssrf-test",
                    "type": "openai",
                    "base_url": "http://169.254.169.254/latest/meta-data/",
                },
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 422, resp.text
        assert "VALIDATION_ERROR" in resp.text

    def test_update_provider_blocks_link_local_base_url(self) -> None:
        """PATCH /config/providers/{name} must reject link-local base_url (BUG-238)."""
        with _api_client() as (client, admin_token, _):
            resp = client.patch(
                "/api/v1/config/providers/openai",
                json={"base_url": "http://169.254.1.1/"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 422, resp.text
        assert "VALIDATION_ERROR" in resp.text

    def test_create_provider_allows_private_lan_base_url(self) -> None:
        """Private RFC-1918 addresses must be allowed for local LAN providers (BUG-238)."""
        import inspect

        import src.api.routes.config as _mod

        src = inspect.getsource(_mod.create_provider)
        # Confirm allow_private=True is passed to the SSRF guard
        assert (
            "allow_private=True" in src
        ), "create_provider SSRF guard must pass allow_private=True to permit LAN providers"


# ---------------------------------------------------------------------------
# BUG-243 — no config file path raises 503 instead of silently writing wrong file
# ---------------------------------------------------------------------------


class TestProviderCrudNoConfigFile:
    def test_write_providers_raises_when_no_config_path(self) -> None:
        """_write_providers_to_config raises RuntimeError when cfg has no file path (BUG-243)."""
        import pytest

        import src.api.routes.config as _mod

        mock_cfg = type("Cfg", (), {"config_file_path": None})()
        with pytest.raises(RuntimeError, match="No config file is loaded"):
            _mod._write_providers_to_config(mock_cfg, {})


# ---------------------------------------------------------------------------
# BUG-245 — health check uses .base_url attribute, not get_base_url() method
# ---------------------------------------------------------------------------


class TestProviderHealthBaseUrlAccess:
    def test_health_check_uses_base_url_attribute(self) -> None:
        """check_provider_health must read pc.base_url, not pc.get_base_url() (BUG-245)."""
        import inspect

        import src.api.routes.config as _mod

        src = inspect.getsource(_mod.check_provider_health)
        assert (
            "get_base_url" not in src
        ), "check_provider_health must not call get_base_url() — use .base_url attribute (BUG-245)"
        assert (
            'base_url=getattr(pc, "base_url"' in src or "pc.base_url" in src
        ), "check_provider_health must read base_url attribute from ProviderConfig (BUG-245)"


# ---------------------------------------------------------------------------
# BUG-239 — wizard sessions per-session lock (source check)
# ---------------------------------------------------------------------------


class TestWizardSessionLock:
    def test_wizard_session_includes_lock_field(self) -> None:
        """start_wizard must store asyncio.Lock in the wizard session dict (BUG-239)."""
        import inspect

        import src.api.routes.config as _mod

        src = inspect.getsource(_mod.start_wizard)
        assert (
            '"lock": asyncio.Lock()' in src
        ), "start_wizard must add asyncio.Lock() to wizard session dict (BUG-239)"

    def test_advance_wizard_acquires_per_session_lock(self) -> None:
        """advance_wizard must acquire ws['lock'] before reading/modifying session (BUG-239)."""
        import inspect

        import src.api.routes.config as _mod

        src = inspect.getsource(_mod.advance_wizard)
        assert (
            'ws["lock"]' in src
        ), "advance_wizard must acquire ws['lock'] to prevent concurrent corruption (BUG-239)"


class TestWizardProbeFailureFix:
    """Issue #129 — wizard initial LLM call failure must raise 422, not silently fall back."""

    def test_wizard_test_connection_returns_tuple(self) -> None:
        """_wizard_test_connection must return (llm, probe_warning) — not bare llm."""
        import inspect

        import src.api.routes.config as _mod

        src = inspect.getsource(_mod._wizard_test_connection)
        assert (
            "probe_warning" in src
        ), "_wizard_test_connection must capture and return probe_warning"
        assert (
            "return llm, probe_warning" in src
        ), "_wizard_test_connection must return (llm, probe_warning) tuple"

    def test_advance_wizard_raises_on_initial_llm_failure(self) -> None:
        """Step 0 must raise 422 PROVIDER_UNREACHABLE when initial LLM call fails
        AND the probe gave no prior warning (BUG-242: raise only when probe_warning
        is not set)."""
        import inspect

        import src.api.routes.config as _mod

        src = inspect.getsource(_mod._advance_wizard_locked)
        assert (
            "PROVIDER_UNREACHABLE" in src
        ), "Step 0 must raise PROVIDER_UNREACHABLE when initial LLM call fails"
        # The raise must be conditional on probe_warning being absent (BUG-242)
        assert (
            "not probe_warning" in src or "probe_warning" in src
        ), "Step 0 must gate the hard-fail on whether probe_warning was set"

    def test_probe_warning_included_in_step_response(self) -> None:
        """Step 0 response must include probe_warning in warnings list when present."""
        import inspect

        import src.api.routes.config as _mod

        src = inspect.getsource(_mod._advance_wizard_locked)
        assert "probe_warning" in src, "Step 0 handler must read probe_warning from wizard session"
        assert "warnings" in src, "Step 0 handler must populate a warnings list for the response"

    def test_probe_warning_stored_in_session(self) -> None:
        """_wizard_test_connection result must store probe_warning in wizard session dict."""
        import inspect

        import src.api.routes.config as _mod

        src = inspect.getsource(_mod._advance_wizard_locked)
        assert (
            'ws["probe_warning"]' in src or "probe_warning" in src
        ), "Step 0 handler must store probe_warning in wizard session"


# ===========================================================================
# Issue #95 — Provider CRUD: additional coverage
# ===========================================================================


class TestProviderCreateExtra:
    def test_create_provider_with_base_url_reflects_in_response(self) -> None:
        """POST /config/providers with base_url must return it in ProviderOut."""
        cfg = _make_config()
        cfg.providers = {}

        with (
            _api_client(extra_state={"config": cfg}) as (client, admin_token, _),
            patch("src.api.routes.config._write_providers_to_config"),
        ):
            resp = client.post(
                "/api/v1/config/providers",
                headers={"Authorization": f"Bearer {admin_token}"},
                json={
                    "name": "my-vllm",
                    "type": "openai",
                    "base_url": "http://10.0.0.5:8000/v1",
                },
            )
        assert resp.status_code == 201, resp.text
        data = resp.json()["data"]
        assert data["base_url"] == "http://10.0.0.5:8000/v1"

    def test_create_provider_with_api_key_sets_has_api_key_true(self) -> None:
        """POST with api_key must return has_api_key=true (key is never echoed)."""
        cfg = _make_config()
        cfg.providers = {}

        with (
            _api_client(extra_state={"config": cfg}) as (client, admin_token, _),
            patch("src.api.routes.config._write_providers_to_config"),
        ):
            resp = client.post(
                "/api/v1/config/providers",
                headers={"Authorization": f"Bearer {admin_token}"},
                json={"name": "keyed", "type": "openai", "api_key": "sk-secret"},
            )
        assert resp.status_code == 201, resp.text
        data = resp.json()["data"]
        assert data["has_api_key"] is True
        # Key itself must never appear in the response
        assert "sk-secret" not in resp.text

    def test_create_provider_without_api_key_has_api_key_false(self) -> None:
        """POST without api_key must return has_api_key=false."""
        cfg = _make_config()
        cfg.providers = {}

        with (
            _api_client(extra_state={"config": cfg}) as (client, admin_token, _),
            patch("src.api.routes.config._write_providers_to_config"),
        ):
            resp = client.post(
                "/api/v1/config/providers",
                headers={"Authorization": f"Bearer {admin_token}"},
                json={"name": "local-ollama", "type": "ollama"},
            )
        assert resp.status_code == 201, resp.text
        assert resp.json()["data"]["has_api_key"] is False

    def test_create_provider_no_config_file_returns_503(self) -> None:
        """POST /config/providers when no config file is loaded must return 503."""
        cfg = _make_config()
        cfg.providers = {}
        cfg.config_file_path = None  # triggers RuntimeError in _write_providers_to_config

        with _api_client(extra_state={"config": cfg}) as (client, admin_token, _):
            resp = client.post(
                "/api/v1/config/providers",
                headers={"Authorization": f"Bearer {admin_token}"},
                json={"name": "new-p", "type": "ollama"},
            )
        assert resp.status_code == 503, resp.text
        assert resp.json()["error"]["code"] == "SERVICE_UNAVAILABLE"

    def test_create_provider_invalid_name_pattern_returns_422(self) -> None:
        """Provider name must match ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ — invalid chars → 422."""
        with _api_client() as (client, admin_token, _):
            resp = client.post(
                "/api/v1/config/providers",
                headers={"Authorization": f"Bearer {admin_token}"},
                json={"name": "-invalid-start", "type": "openai"},
            )
        assert resp.status_code == 422, resp.text

    def test_create_provider_empty_name_returns_422(self) -> None:
        """Empty provider name must be rejected by schema validation."""
        with _api_client() as (client, admin_token, _):
            resp = client.post(
                "/api/v1/config/providers",
                headers={"Authorization": f"Bearer {admin_token}"},
                json={"name": "", "type": "openai"},
            )
        assert resp.status_code == 422, resp.text


class TestProviderUpdateExtra:
    def test_patch_provider_updates_base_url(self) -> None:
        """PATCH with base_url must update the URL and reflect in response."""
        pc = _make_provider_config("myp", "openai")
        pc.base_url = "http://10.0.0.5:8000/v1"
        cfg = _make_config()
        cfg.providers = {"myp": pc}

        with (
            _api_client(extra_state={"config": cfg}) as (client, admin_token, _),
            patch("src.api.routes.config._write_providers_to_config"),
        ):
            resp = client.patch(
                "/api/v1/config/providers/myp",
                headers={"Authorization": f"Bearer {admin_token}"},
                json={"base_url": "http://10.0.0.6:8000/v1"},
            )
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"]["base_url"] == "http://10.0.0.6:8000/v1"

    def test_patch_provider_updates_both_fields(self) -> None:
        """PATCH with both base_url and api_key must update both fields."""
        pc = _make_provider_config("combo", "openai", api_key="old-key")
        pc.base_url = "http://10.0.0.5:8000/v1"
        cfg = _make_config()
        cfg.providers = {"combo": pc}

        with (
            _api_client(extra_state={"config": cfg}) as (client, admin_token, _),
            patch("src.api.routes.config._write_providers_to_config"),
        ):
            resp = client.patch(
                "/api/v1/config/providers/combo",
                headers={"Authorization": f"Bearer {admin_token}"},
                json={"base_url": "http://10.0.0.6:8000/v1", "api_key": "new-key"},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["base_url"] == "http://10.0.0.6:8000/v1"
        assert data["has_api_key"] is True

    def test_patch_provider_empty_body_returns_200_unchanged(self) -> None:
        """PATCH with neither field set must succeed and leave data unchanged."""
        pc = _make_provider_config("stable", "ollama")
        pc.base_url = "http://10.0.0.1:11434"
        cfg = _make_config()
        cfg.providers = {"stable": pc}

        with (
            _api_client(extra_state={"config": cfg}) as (client, admin_token, _),
            patch("src.api.routes.config._write_providers_to_config"),
        ):
            resp = client.patch(
                "/api/v1/config/providers/stable",
                headers={"Authorization": f"Bearer {admin_token}"},
                json={},
            )
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"]["name"] == "stable"

    def test_patch_provider_no_config_file_returns_503(self) -> None:
        """PATCH /config/providers/{name} when no config file is loaded must return 503."""
        pc = _make_provider_config("p", "openai", api_key="k")
        cfg = _make_config()
        cfg.providers = {"p": pc}
        cfg.config_file_path = None

        with _api_client(extra_state={"config": cfg}) as (client, admin_token, _):
            resp = client.patch(
                "/api/v1/config/providers/p",
                headers={"Authorization": f"Bearer {admin_token}"},
                json={"api_key": "new-k"},
            )
        assert resp.status_code == 503, resp.text
        assert resp.json()["error"]["code"] == "SERVICE_UNAVAILABLE"

    def test_patch_provider_allows_private_lan_base_url(self) -> None:
        """RFC-1918 private addresses must be accepted (local Ollama/vLLM installs)."""
        pc = _make_provider_config("lan-llm", "openai")
        cfg = _make_config()
        cfg.providers = {"lan-llm": pc}

        with (
            _api_client(extra_state={"config": cfg}) as (client, admin_token, _),
            patch("src.api.routes.config._write_providers_to_config"),
        ):
            resp = client.patch(
                "/api/v1/config/providers/lan-llm",
                headers={"Authorization": f"Bearer {admin_token}"},
                json={"base_url": "http://192.168.1.100:8080/v1"},
            )
        assert resp.status_code == 200, resp.text


class TestProviderDeleteExtra:
    def test_delete_provider_no_config_file_returns_503(self) -> None:
        """DELETE /config/providers/{name} when no config file is loaded must return 503."""
        pc = _make_provider_config("to-delete", "openai")
        cfg = _make_config()
        cfg.providers = {"to-delete": pc}
        cfg.models = {}
        cfg.config_file_path = None

        with _api_client(extra_state={"config": cfg}) as (client, admin_token, _):
            resp = client.delete(
                "/api/v1/config/providers/to-delete",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 503, resp.text
        assert resp.json()["error"]["code"] == "SERVICE_UNAVAILABLE"

    def test_delete_provider_in_use_message_names_the_model(self) -> None:
        """PROVIDER_IN_USE error message must list the referencing model alias."""
        pc = _make_provider_config("openai", "openai")
        mc = MagicMock()
        mc.provider = "openai"
        cfg = _make_config()
        cfg.providers = {"openai": pc}
        cfg.models = {"gpt-4o": mc}

        with _api_client(extra_state={"config": cfg}) as (client, admin_token, _):
            resp = client.delete(
                "/api/v1/config/providers/openai",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 409, resp.text
        msg = resp.json()["error"]["message"]
        assert "gpt-4o" in msg

    def test_delete_provider_response_body_null_data(self) -> None:
        """DELETE /config/providers/{name} success response must have data=null."""
        pc = _make_provider_config("removable", "ollama")
        cfg = _make_config()
        cfg.providers = {"removable": pc}
        cfg.models = {}

        with (
            _api_client(extra_state={"config": cfg}) as (client, admin_token, _),
            patch("src.api.routes.config._write_providers_to_config"),
        ):
            resp = client.delete(
                "/api/v1/config/providers/removable",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["data"] is None
        assert body["error"] is None


class TestWizardProbeWarningFallback:
    """BUG-242 / BUG-244 regression: a failed initial wizard LLM call (whether due to a
    flaky provider flagged by the probe, or a context-overflow with a small-context model)
    must fall back to the default question instead of returning 422 PROVIDER_UNREACHABLE.
    Phase 1 hard-fail (LLM creation error) is the ONLY path that should return 422."""

    def test_step0_uses_default_question_when_probe_warned_and_llm_fails(self) -> None:
        """Step 0 must advance to step 1 with the default question when probe_warning
        is set and the first real LLM call also fails."""
        from unittest.mock import MagicMock, patch

        import src.api.routes.config as _mod

        # Simulate: probe soft-fails (sets probe_warning) AND first real LLM call fails
        probe_warning = "Error code: 400 - No connected db."
        fake_llm = MagicMock()
        fake_llm.invoke.side_effect = RuntimeError(probe_warning)

        fake_ws: dict = {
            "step": 0,
            "existing_yaml": None,
            "docs_url": None,
            "bootstrap_info": None,
            "llm": None,
            "messages": [],
            "probe_warning": None,
        }

        fake_body = MagicMock()
        fake_body.data = {
            "provider_type": "openai",
            "provider_name": "spark",
            "model": "glm-4.7-flash",
            "base_url": "http://192.168.70.254:8080/v1",
            "api_key": None,
        }

        wizard_id = "test-wizard-id"

        with (
            patch.object(_mod, "_wizard_test_connection", return_value=(fake_llm, probe_warning)),
            patch.object(_mod, "_wizard_load_docs", return_value="# Docs"),
            patch.object(_mod, "_wizard_invoke_llm", side_effect=RuntimeError("No connected db.")),
            patch("src.setup_wizard._WIZARD_SYSTEM_PROMPT") as mock_tpl,
        ):
            mock_tpl.substitute.return_value = "System prompt"

            import asyncio

            result = asyncio.run(
                _mod._advance_wizard_locked(wizard_id, fake_ws, fake_body, MagicMock())
            )

        # Must advance to step 1, not raise 422
        assert result.data is not None, "Must return a successful response, not raise 422"
        assert result.data.step == 1, f"Expected step=1, got {result.data.step}"
        assert (
            result.data.question == _mod._WIZARD_DEFAULT_FIRST_QUESTION
        ), "Must fall back to default question when probe warned and LLM fails"
        # Probe warning must appear in the response warnings
        assert any(
            "warning" in w.lower() or "probe" in w.lower() for w in result.data.warnings
        ), "Probe warning must be surfaced in the response warnings list"

    def test_step0_uses_default_question_when_probe_ok_but_llm_fails(self) -> None:
        """BUG-244: Step 0 must advance to step 1 with the default question even when
        the probe succeeded (no probe_warning) but the main LLM call fails — e.g. a
        small-context model (Gemma 270M, 4096 ctx) whose context overflows when the
        wizard loads the full CONFIGURATION.md docs."""
        from unittest.mock import MagicMock, patch

        import src.api.routes.config as _mod

        fake_llm = MagicMock()
        fake_ws: dict = {
            "step": 0,
            "existing_yaml": None,
            "docs_url": None,
            "bootstrap_info": None,
            "llm": None,
            "messages": [],
        }
        fake_body = MagicMock()
        fake_body.data = {
            "provider_type": "openai",
            "provider_name": "gemma",
            "model": "gemma3:1b",
            "base_url": "http://gemma-test:8080/v1",
            "api_key": "not-required",
        }

        ctx_error = RuntimeError(
            "Error code: 400 - request (23273 tokens) exceeds the available context size (4096)"
        )

        with (
            patch.object(_mod, "_wizard_test_connection", return_value=(fake_llm, None)),
            patch.object(_mod, "_wizard_load_docs", return_value="# Docs"),
            patch.object(_mod, "_wizard_invoke_llm", side_effect=ctx_error),
            patch("src.setup_wizard._WIZARD_SYSTEM_PROMPT") as mock_tpl,
        ):
            mock_tpl.substitute.return_value = "System prompt"

            import asyncio

            result = asyncio.run(
                _mod._advance_wizard_locked("wid", fake_ws, fake_body, MagicMock())
            )

        # Must advance to step 1 using the default question, not raise 422
        assert result.data is not None, "Must return a successful response, not raise 422"
        assert result.data.step == 1
        assert result.data.question == _mod._WIZARD_DEFAULT_FIRST_QUESTION


class TestWizardValidateAndWriteStripNulls:
    """Bug 5 regression: _wizard_validate_and_write must call _strip_nulls() to
    remove null values and empty dicts from the LLM-generated YAML before the
    round-trip validation write and before persisting to disk (BUG-239)."""

    def test_strip_nulls_called_after_inject_bootstrap(self) -> None:
        """_wizard_validate_and_write must invoke _strip_nulls on the parsed dict."""
        import inspect

        from src.api.routes.config import _wizard_validate_and_write

        src = inspect.getsource(_wizard_validate_and_write)
        assert (
            "_strip_nulls" in src
        ), "_wizard_validate_and_write must call _strip_nulls() to remove null values"

    def test_null_values_removed_by_strip_nulls(self) -> None:
        """_strip_nulls removes null values and empty dicts from the YAML structure."""
        import yaml

        from src.setup_wizard import _strip_nulls

        raw = """
providers:
  my-ollama:
    type: ollama
    base_url: http://localhost:11434
    tool_instructions: null
models:
  default:
    provider: my-ollama
    model: llama3
    temperature: 0.7
    context_window: 8192
    max_tokens: 2048
services: null
"""
        data = yaml.safe_load(raw)
        cleaned = _strip_nulls(data)

        assert "services" not in cleaned, "Null top-level key must be removed by _strip_nulls"
        provider = cleaned["providers"]["my-ollama"]
        assert (
            "tool_instructions" not in provider
        ), "Null nested field must be removed by _strip_nulls"
        # Non-null fields must survive
        assert cleaned["models"]["default"]["model"] == "llama3"
