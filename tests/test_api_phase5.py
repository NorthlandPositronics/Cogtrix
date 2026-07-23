"""Phase 5 API tests: RAG document endpoints.

Tests cover:
- File upload with valid and invalid file types
- File upload exceeds size limit
- Document CRUD (create, read, delete)
- List documents with status filter
- Cursor pagination on document listing
- Search with empty results (no indexed docs)
- Admin-only enforcement on upload and delete
- Auth required on all endpoints

State injection strategy:
    Same pattern as Phase 4 — override app.state inside the TestClient context
    block so mocks win over lifespan-set state.
"""

from __future__ import annotations

import io
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

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
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def _api_client(
    extra_state: dict | None = None,
) -> Iterator[tuple[TestClient, str, str]]:
    """Yield (client, admin_token, user_token)."""
    from src.api.app import create_app

    admin_token = create_access_token(user_id=str(uuid.uuid4()), role="admin")
    user_token = create_access_token(user_id=str(uuid.uuid4()), role="user")
    app = create_app()
    with TestClient(app, raise_server_exceptions=True) as client:
        if extra_state:
            for k, v in extra_state.items():
                setattr(app.state, k, v)
        yield client, admin_token, user_token


def _fake_ingest_task(*args, **kwargs):
    """No-op replacement for the background ingest task."""
    return None


# ---------------------------------------------------------------------------
# Upload / create document tests
# ---------------------------------------------------------------------------


class TestDocumentUpload:
    def test_upload_valid_txt(self, tmp_path: Path) -> None:
        """Upload a .txt file — should return 202 with status=pending."""
        with (
            _api_client() as (client, admin_token, _),
            patch(
                "src.api.routes.rag.ingest_document_task",
                new=AsyncMock(return_value=None),
            ),
            patch("src.api.routes.rag._UPLOADS_DIR", tmp_path),
            patch("src.api.tasks.rag._UPLOADS_DIR", tmp_path),
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
        assert data["chunk_count"] == 0
        assert data["id"] is not None

    def test_upload_valid_pdf(self, tmp_path: Path) -> None:
        """PDF upload is accepted."""
        with (
            _api_client() as (client, admin_token, _),
            patch(
                "src.api.routes.rag.ingest_document_task",
                new=AsyncMock(return_value=None),
            ),
            patch("src.api.routes.rag._UPLOADS_DIR", tmp_path),
        ):
            resp = client.post(
                "/api/v1/rag/documents",
                files={"file": ("report.pdf", b"%PDF-1.4 content", "application/pdf")},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 202, resp.text
        assert resp.json()["data"]["filename"] == "report.pdf"

    def test_upload_valid_md(self, tmp_path: Path) -> None:
        """Markdown upload is accepted."""
        with (
            _api_client() as (client, admin_token, _),
            patch(
                "src.api.routes.rag.ingest_document_task",
                new=AsyncMock(return_value=None),
            ),
            patch("src.api.routes.rag._UPLOADS_DIR", tmp_path),
        ):
            resp = client.post(
                "/api/v1/rag/documents",
                files={"file": ("readme.md", b"# Title\n\nContent", "text/markdown")},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 202, resp.text

    def test_upload_invalid_extension_rejected(self, tmp_path: Path) -> None:
        """Uploading an unsupported file type (e.g. .exe) returns 415."""
        with _api_client() as (client, admin_token, _):
            resp = client.post(
                "/api/v1/rag/documents",
                files={"file": ("virus.exe", b"MZ", "application/octet-stream")},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 415, resp.text
        assert resp.json()["error"]["code"] == "VALIDATION_ERROR"

    def test_upload_zip_rejected(self, tmp_path: Path) -> None:
        """ZIP files are not in the allow-list."""
        with _api_client() as (client, admin_token, _):
            resp = client.post(
                "/api/v1/rag/documents",
                files={"file": ("archive.zip", b"PK\x03\x04", "application/zip")},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 415
        assert resp.json()["error"]["code"] == "VALIDATION_ERROR"

    def test_upload_too_large(self, tmp_path: Path) -> None:
        """Files larger than 50 MB are rejected with 413."""
        big_data = b"x" * (51 * 1024 * 1024)
        with _api_client() as (client, admin_token, _):
            resp = client.post(
                "/api/v1/rag/documents",
                files={"file": ("big.txt", io.BytesIO(big_data), "text/plain")},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 413, resp.text
        assert resp.json()["error"]["code"] == "VALIDATION_ERROR"

    def test_upload_requires_admin(self, tmp_path: Path) -> None:
        """Regular users cannot upload documents."""
        with _api_client() as (client, _, user_token):
            resp = client.post(
                "/api/v1/rag/documents",
                files={"file": ("notes.txt", b"hello", "text/plain")},
                headers={"Authorization": f"Bearer {user_token}"},
            )
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "FORBIDDEN"

    def test_upload_requires_auth(self) -> None:
        """Unauthenticated upload returns 401."""
        with _api_client() as (client, _, __):
            resp = client.post(
                "/api/v1/rag/documents",
                files={"file": ("notes.txt", b"hello", "text/plain")},
            )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Get / list documents
# ---------------------------------------------------------------------------


class TestDocumentListing:
    def test_get_document_returns_detail(self, tmp_path: Path) -> None:
        """GET /rag/documents/{id} returns the correct document (same app instance)."""
        with (
            _api_client() as (client, admin_token, _),
            patch(
                "src.api.routes.rag.ingest_document_task",
                new=AsyncMock(return_value=None),
            ),
            patch("src.api.routes.rag._UPLOADS_DIR", tmp_path),
        ):
            seed_resp = client.post(
                "/api/v1/rag/documents",
                files={"file": ("doc.txt", b"content", "text/plain")},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert seed_resp.status_code == 202
            doc_id = seed_resp.json()["data"]["id"]

            resp = client.get(
                f"/api/v1/rag/documents/{doc_id}",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["id"] == doc_id
        assert data["filename"] == "doc.txt"

    def test_get_document_not_found(self) -> None:
        """GET /rag/documents/{id} with unknown id returns 404."""
        with _api_client() as (client, admin_token, _):
            resp = client.get(
                f"/api/v1/rag/documents/{uuid.uuid4()}",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"

    def test_get_document_requires_auth(self) -> None:
        with _api_client() as (client, _, __):
            resp = client.get(f"/api/v1/rag/documents/{uuid.uuid4()}")
        assert resp.status_code == 401

    def test_list_documents_empty(self) -> None:
        """With no documents, list returns an empty page."""
        with _api_client() as (client, admin_token, _):
            resp = client.get(
                "/api/v1/rag/documents",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        page = resp.json()["data"]
        assert page["items"] == []
        assert page["has_more"] is False

    def test_list_documents_returns_uploaded(self, tmp_path: Path) -> None:
        """After upload, the document appears in the list (same app instance)."""
        with (
            _api_client() as (client, admin_token, _),
            patch(
                "src.api.routes.rag.ingest_document_task",
                new=AsyncMock(return_value=None),
            ),
            patch("src.api.routes.rag._UPLOADS_DIR", tmp_path),
        ):
            upload_resp = client.post(
                "/api/v1/rag/documents",
                files={"file": ("doc.txt", b"content", "text/plain")},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert upload_resp.status_code == 202

            resp = client.get(
                "/api/v1/rag/documents",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        page = resp.json()["data"]
        assert len(page["items"]) >= 1

    def test_list_documents_requires_auth(self) -> None:
        with _api_client() as (client, _, __):
            resp = client.get("/api/v1/rag/documents")
        assert resp.status_code == 401

    def test_list_documents_status_filter_no_match(self) -> None:
        """Filtering by status=indexed on a fresh (empty) DB returns nothing."""
        with _api_client() as (client, admin_token, _):
            resp = client.get(
                "/api/v1/rag/documents",
                params={"status": "indexed"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        page = resp.json()["data"]
        assert page["items"] == []

    def test_list_documents_status_filter_pending(self, tmp_path: Path) -> None:
        """Filtering by status=pending returns only pending documents."""
        with (
            _api_client() as (client, admin_token, _),
            patch(
                "src.api.routes.rag.ingest_document_task",
                new=AsyncMock(return_value=None),
            ),
            patch("src.api.routes.rag._UPLOADS_DIR", tmp_path),
        ):
            client.post(
                "/api/v1/rag/documents",
                files={"file": ("doc.txt", b"hello", "text/plain")},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            resp = client.get(
                "/api/v1/rag/documents",
                params={"status": "pending"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        page = resp.json()["data"]
        assert len(page["items"]) >= 1
        assert all(item["status"] == "pending" for item in page["items"])

    def test_list_documents_invalid_cursor(self) -> None:
        """Malformed cursor returns 400 INVALID_CURSOR."""
        with _api_client() as (client, admin_token, _):
            resp = client.get(
                "/api/v1/rag/documents",
                params={"cursor": "not-valid-base64!!!"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "INVALID_CURSOR"

    def test_list_documents_total_count(self, tmp_path: Path) -> None:
        """Total count in the response matches the number of documents created."""
        with (
            _api_client() as (client, admin_token, _),
            patch(
                "src.api.routes.rag.ingest_document_task",
                new=AsyncMock(return_value=None),
            ),
            patch("src.api.routes.rag._UPLOADS_DIR", tmp_path),
        ):
            client.post(
                "/api/v1/rag/documents",
                files={"file": ("doc.txt", b"content", "text/plain")},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            resp = client.get(
                "/api/v1/rag/documents",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        page = resp.json()["data"]
        assert page["total"] >= 1


# ---------------------------------------------------------------------------
# Delete document
# ---------------------------------------------------------------------------


class TestDocumentDelete:
    def test_delete_existing_document(self, tmp_path: Path) -> None:
        """DELETE /rag/documents/{id} removes the document (same app instance)."""
        with (
            _api_client() as (client, admin_token, _),
            patch(
                "src.api.routes.rag.ingest_document_task",
                new=AsyncMock(return_value=None),
            ),
            patch("src.api.routes.rag._UPLOADS_DIR", tmp_path),
        ):
            seed_resp = client.post(
                "/api/v1/rag/documents",
                files={"file": ("del.txt", b"bye", "text/plain")},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert seed_resp.status_code == 202
            doc_id = seed_resp.json()["data"]["id"]

            del_resp = client.delete(
                f"/api/v1/rag/documents/{doc_id}",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert del_resp.status_code == 200, del_resp.text
        assert del_resp.json()["data"] is None

    def test_delete_nonexistent_returns_404(self) -> None:
        with _api_client() as (client, admin_token, _):
            resp = client.delete(
                f"/api/v1/rag/documents/{uuid.uuid4()}",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"

    def test_delete_requires_admin(self, tmp_path: Path) -> None:
        """Non-admin user cannot delete a document."""
        # Seed the doc as admin, then try to delete as user — all in one app instance
        # so both the upload and the deny-check share the same in-memory DB.
        with (
            _api_client() as (client, admin_token, user_token),
            patch(
                "src.api.routes.rag.ingest_document_task",
                new=AsyncMock(return_value=None),
            ),
            patch("src.api.routes.rag._UPLOADS_DIR", tmp_path),
        ):
            seed_resp = client.post(
                "/api/v1/rag/documents",
                files={"file": ("del.txt", b"bye", "text/plain")},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert seed_resp.status_code == 202
            doc_id = seed_resp.json()["data"]["id"]

            resp = client.delete(
                f"/api/v1/rag/documents/{doc_id}",
                headers={"Authorization": f"Bearer {user_token}"},
            )
        assert resp.status_code == 403

    def test_delete_requires_auth(self) -> None:
        with _api_client() as (client, _, __):
            resp = client.delete(f"/api/v1/rag/documents/{uuid.uuid4()}")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Search endpoint
# ---------------------------------------------------------------------------


class TestRagSearch:
    def test_search_empty_when_no_indexed_docs(self) -> None:
        """With no indexed documents, search returns empty chunks."""
        with _api_client() as (client, admin_token, _):
            resp = client.post(
                "/api/v1/rag/search",
                json={"query": "What is the meaning of life?", "top_k": 5},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["query"] == "What is the meaning of life?"
        assert data["chunks"] == []
        assert data["total_documents_searched"] == 0

    def test_search_requires_auth(self) -> None:
        """Unauthenticated search returns 401."""
        with _api_client() as (client, _, __):
            resp = client.post(
                "/api/v1/rag/search",
                json={"query": "test query"},
            )
        assert resp.status_code == 401

    def test_search_query_too_short_fails(self) -> None:
        """Empty query is rejected by schema validation (min_length=1)."""
        with _api_client() as (client, admin_token, _):
            resp = client.post(
                "/api/v1/rag/search",
                json={"query": ""},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 422

    def test_search_top_k_validation(self) -> None:
        """top_k=0 is rejected by schema validation (ge=1)."""
        with _api_client() as (client, admin_token, _):
            resp = client.post(
                "/api/v1/rag/search",
                json={"query": "test", "top_k": 0},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 422

    def test_search_with_document_ids_filter_empty(self) -> None:
        """Filtering by specific (nonexistent) document IDs returns empty."""
        with _api_client() as (client, admin_token, _):
            resp = client.post(
                "/api/v1/rag/search",
                json={
                    "query": "climate change",
                    "top_k": 3,
                    "document_ids": [str(uuid.uuid4())],
                },
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["chunks"] == []
        assert data["total_documents_searched"] == 0

    def test_search_response_structure(self) -> None:
        """Search response always includes query, chunks, total_documents_searched."""
        with _api_client() as (client, admin_token, _):
            resp = client.post(
                "/api/v1/rag/search",
                json={"query": "testing structure"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "query" in data
        assert "chunks" in data
        assert "total_documents_searched" in data


# ---------------------------------------------------------------------------
# Content-type derivation
# ---------------------------------------------------------------------------


class TestDocumentContentType:
    def test_content_type_pdf(self, tmp_path: Path) -> None:
        with (
            _api_client() as (client, admin_token, _),
            patch(
                "src.api.routes.rag.ingest_document_task",
                new=AsyncMock(return_value=None),
            ),
            patch("src.api.routes.rag._UPLOADS_DIR", tmp_path),
        ):
            resp = client.post(
                "/api/v1/rag/documents",
                files={"file": ("paper.pdf", b"%PDF content", "application/pdf")},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 202
        data = resp.json()["data"]
        assert data["content_type"] == "application/pdf"

    def test_content_type_txt(self, tmp_path: Path) -> None:
        with (
            _api_client() as (client, admin_token, _),
            patch(
                "src.api.routes.rag.ingest_document_task",
                new=AsyncMock(return_value=None),
            ),
            patch("src.api.routes.rag._UPLOADS_DIR", tmp_path),
        ):
            resp = client.post(
                "/api/v1/rag/documents",
                files={"file": ("notes.txt", b"text content", "text/plain")},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 202
        assert resp.json()["data"]["content_type"] == "text/plain"
