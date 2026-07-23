"""Comprehensive RAG endpoint coverage.

Tests all 5 RAG endpoints:
  POST   /api/v1/rag/documents          — upload (admin only)
  GET    /api/v1/rag/documents          — list (auth required)
  GET    /api/v1/rag/documents/{id}     — get one (auth required)
  DELETE /api/v1/rag/documents/{id}     — delete (admin only)
  POST   /api/v1/rag/search             — semantic search (auth required)
"""

from __future__ import annotations

import io
import os
import uuid

import pytest

pytest.importorskip("fastapi")

_TEST_JWT_SECRET = "testsecret_mustbe32chars_minimum00"
os.environ.setdefault("COGTRIX_JWT_SECRET", _TEST_JWT_SECRET)
os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

import asyncio as _asyncio  # noqa: E402
from unittest.mock import AsyncMock, patch  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from src.api.auth import create_access_token  # noqa: E402
from src.api.db.engine import Base, get_db  # noqa: E402
from src.api.db.models import RagDocument  # noqa: E402
from src.api.db.repositories.organization import OrganizationRepository  # noqa: E402
from src.api.db.repositories.users import UserRepository  # noqa: E402
from src.api.pagination import encode_cursor  # noqa: E402

_VALID_PASSWORD = "TestPass1!"


@pytest.fixture()
def app(tmp_path):
    from src.api.app import create_app

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _create():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    loop = _asyncio.new_event_loop()
    _asyncio.set_event_loop(loop)
    loop.run_until_complete(_create())

    async def _noop_ingest(doc_id, file_path):
        """No-op ingest task — avoids real Ollama/embedding calls in tests."""

    with (
        patch.dict(
            os.environ,
            {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET, "COGTRIX_DATA_DIR": str(tmp_path)},
        ),
        patch("src.api.routes.rag.ingest_document_task", _noop_ingest),
    ):
        _app = create_app()
        _app.state.db_factory = factory

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


def _register_and_login(client, role_hint="first"):
    uname = f"rag_{uuid.uuid4().hex[:8]}"
    email = f"{uname}@test.example"
    client.post(
        "/api/v1/auth/register",
        json={"username": uname, "email": email, "password": _VALID_PASSWORD},
    )
    r = client.post("/api/v1/auth/login", json={"username": uname, "password": _VALID_PASSWORD})
    token = r.json()["data"]["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _seed_org_user(
    app,
    *,
    org_name: str,
    org_slug: str,
    username: str,
    email: str,
    role: str = "user",
) -> dict[str, str]:
    """Create an organization-scoped user and return auth headers for it."""
    org_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())

    async def _seed() -> None:
        async with app.state.db_factory() as session:
            org_repo = OrganizationRepository(session)
            user_repo = UserRepository(session)
            await org_repo.create(org_id=org_id, name=org_name, slug=org_slug)
            await user_repo.create(
                user_id=user_id,
                username=username,
                email=email,
                password_hash="hash",
                role=role,
                org_id=org_id,
            )
            await session.commit()

    _asyncio.run(_seed())
    token = create_access_token(user_id=user_id, role=role)
    return {"Authorization": f"Bearer {token}"}


def _mark_document_indexed(app, document_id: str) -> None:
    """Flip a RAG document to indexed so search can see it."""

    async def _seed() -> None:
        async with app.state.db_factory() as session:
            doc = await session.get(RagDocument, document_id)
            assert doc is not None
            doc.status = "indexed"
            await session.commit()

    _asyncio.run(_seed())


@pytest.fixture()
def admin_headers(client):
    return _register_and_login(client, "first")


@pytest.fixture()
def user_headers(client, admin_headers):
    uname = f"ru_{uuid.uuid4().hex[:6]}"
    client.post(
        "/api/v1/auth/register",
        json={"username": uname, "email": f"{uname}@example.com", "password": _VALID_PASSWORD},
    )
    r = client.post("/api/v1/auth/login", json={"username": uname, "password": _VALID_PASSWORD})
    token = r.json()["data"]["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _upload_txt(client, headers, content=b"Hello world document.", filename="test.txt"):
    return client.post(
        "/api/v1/rag/documents",
        headers=headers,
        files={"file": (filename, io.BytesIO(content), "text/plain")},
    )


# ---------------------------------------------------------------------------
# POST /api/v1/rag/documents — upload
# ---------------------------------------------------------------------------


class TestUploadDocument:
    def test_upload_requires_auth(self, client):
        r = client.post(
            "/api/v1/rag/documents",
            files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        assert r.status_code == 401

    def test_upload_requires_admin(self, client, user_headers):
        r = client.post(
            "/api/v1/rag/documents",
            headers=user_headers,
            files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        assert r.status_code == 403

    def test_upload_txt_returns_202(self, client, admin_headers):
        r = _upload_txt(client, admin_headers)
        assert r.status_code == 202

    def test_upload_returns_document_out(self, client, admin_headers):
        r = _upload_txt(client, admin_headers)
        data = r.json()["data"]
        assert "id" in data
        assert data["filename"] == "test.txt"
        assert data["status"] == "pending"
        assert data["chunk_count"] == 0

    def test_upload_invalid_extension_returns_415(self, client, admin_headers):
        r = client.post(
            "/api/v1/rag/documents",
            headers=admin_headers,
            files={"file": ("test.exe", io.BytesIO(b"binary"), "application/octet-stream")},
        )
        assert r.status_code == 415

    def test_upload_docx_extension_returns_415(self, client, admin_headers):
        # .docx is not in the allowed list (only .pdf, .txt, .md, .markdown, .csv)
        r = client.post(
            "/api/v1/rag/documents",
            headers=admin_headers,
            files={"file": ("test.docx", io.BytesIO(b"binary"), "application/octet-stream")},
        )
        assert r.status_code == 415

    def test_upload_md_extension_accepted(self, client, admin_headers):
        r = client.post(
            "/api/v1/rag/documents",
            headers=admin_headers,
            files={"file": ("readme.md", io.BytesIO(b"# Hello"), "text/markdown")},
        )
        assert r.status_code == 202

    def test_upload_csv_extension_accepted(self, client, admin_headers):
        r = client.post(
            "/api/v1/rag/documents",
            headers=admin_headers,
            files={"file": ("data.csv", io.BytesIO(b"a,b\n1,2"), "text/csv")},
        )
        assert r.status_code == 202

    def test_upload_markdown_extension_accepted(self, client, admin_headers):
        r = client.post(
            "/api/v1/rag/documents",
            headers=admin_headers,
            files={"file": ("notes.markdown", io.BytesIO(b"## Notes"), "text/markdown")},
        )
        assert r.status_code == 202

    def test_upload_invalid_extension_error_code(self, client, admin_headers):
        r = client.post(
            "/api/v1/rag/documents",
            headers=admin_headers,
            files={"file": ("hack.sh", io.BytesIO(b"#!/bin/bash"), "text/plain")},
        )
        assert r.status_code == 415
        assert r.json()["error"]["code"] == "VALIDATION_ERROR"

    def test_upload_stores_id_in_data(self, client, admin_headers):
        r = _upload_txt(client, admin_headers)
        doc_id = r.json()["data"]["id"]
        assert len(doc_id) == 36  # UUID v4 length

    def test_upload_no_body_returns_422(self, client, admin_headers):
        r = client.post("/api/v1/rag/documents", headers=admin_headers)
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/v1/rag/documents — list
# ---------------------------------------------------------------------------


class TestListDocuments:
    def test_list_requires_auth(self, client):
        r = client.get("/api/v1/rag/documents")
        assert r.status_code == 401

    def test_list_returns_200_with_auth(self, client, user_headers):
        r = client.get("/api/v1/rag/documents", headers=user_headers)
        assert r.status_code == 200

    def test_list_empty_initially(self, client, user_headers):
        r = client.get("/api/v1/rag/documents", headers=user_headers)
        page = r.json()["data"]
        assert page["items"] == []
        assert page["total"] == 0

    def test_list_shows_uploaded_document(self, client, admin_headers, user_headers):
        _upload_txt(client, admin_headers)
        r = client.get("/api/v1/rag/documents", headers=user_headers)
        page = r.json()["data"]
        assert page["total"] == 1
        assert len(page["items"]) == 1

    def test_list_paginated_envelope(self, client, user_headers):
        r = client.get("/api/v1/rag/documents", headers=user_headers)
        page = r.json()["data"]
        assert "items" in page
        assert "has_more" in page
        assert "next_cursor" in page
        assert "total" in page

    def test_list_limit_param(self, client, admin_headers, user_headers):
        for _ in range(3):
            _upload_txt(client, admin_headers)
        r = client.get("/api/v1/rag/documents?limit=2", headers=user_headers)
        page = r.json()["data"]
        assert len(page["items"]) == 2
        assert page["has_more"] is True
        assert page["next_cursor"] is not None

    def test_list_cursor_pagination(self, client, admin_headers, user_headers):
        for i in range(3):
            _upload_txt(client, admin_headers, filename=f"file{i}.txt")
        r1 = client.get("/api/v1/rag/documents?limit=2", headers=user_headers)
        cursor = r1.json()["data"]["next_cursor"]
        r2 = client.get(f"/api/v1/rag/documents?limit=2&cursor={cursor}", headers=user_headers)
        assert r2.status_code == 200
        page2 = r2.json()["data"]
        assert len(page2["items"]) >= 1

    def test_list_status_filter(self, client, admin_headers, user_headers):
        _upload_txt(client, admin_headers)
        r = client.get("/api/v1/rag/documents?status=pending", headers=user_headers)
        assert r.status_code == 200
        page = r.json()["data"]
        # All uploaded docs start as pending
        assert page["total"] >= 1

    def test_list_status_filter_nonexistent(self, client, admin_headers, user_headers):
        _upload_txt(client, admin_headers)
        r = client.get("/api/v1/rag/documents?status=indexed", headers=user_headers)
        assert r.status_code == 200
        assert r.json()["data"]["total"] == 0

    def test_list_invalid_cursor_returns_400(self, client, user_headers):
        r = client.get("/api/v1/rag/documents?cursor=notvalidbase64!!!", headers=user_headers)
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "INVALID_CURSOR"

    def test_list_stale_cursor_returns_400(self, client, admin_headers, user_headers):
        _upload_txt(client, admin_headers, filename="cursor-target.txt")
        stale_cursor = encode_cursor(str(uuid.uuid4()))
        r = client.get(f"/api/v1/rag/documents?cursor={stale_cursor}", headers=user_headers)
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "INVALID_CURSOR"


# ---------------------------------------------------------------------------
# GET /api/v1/rag/documents/{id} — get one
# ---------------------------------------------------------------------------


class TestGetDocument:
    def test_get_requires_auth(self, client, admin_headers):
        r_up = _upload_txt(client, admin_headers)
        doc_id = r_up.json()["data"]["id"]
        r = client.get(f"/api/v1/rag/documents/{doc_id}")
        assert r.status_code == 401

    def test_get_returns_200(self, client, admin_headers, user_headers):
        r_up = _upload_txt(client, admin_headers)
        doc_id = r_up.json()["data"]["id"]
        r = client.get(f"/api/v1/rag/documents/{doc_id}", headers=user_headers)
        assert r.status_code == 200

    def test_get_returns_correct_doc(self, client, admin_headers, user_headers):
        r_up = _upload_txt(client, admin_headers, filename="unique.txt")
        doc_id = r_up.json()["data"]["id"]
        r = client.get(f"/api/v1/rag/documents/{doc_id}", headers=user_headers)
        data = r.json()["data"]
        assert data["id"] == doc_id
        assert data["filename"] == "unique.txt"

    def test_get_nonexistent_returns_404(self, client, user_headers):
        fake_id = str(uuid.uuid4())
        r = client.get(f"/api/v1/rag/documents/{fake_id}", headers=user_headers)
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"

    def test_get_invalid_id_format_returns_400(self, client, user_headers):
        r = client.get("/api/v1/rag/documents/not-a-uuid", headers=user_headers)
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "INVALID_DOCUMENT_ID"

    def test_get_path_traversal_attempt_returns_400(self, client, user_headers):
        r = client.get("/api/v1/rag/documents/../../etc/passwd", headers=user_headers)
        # Either 400 (invalid ID) or 404 (not found)
        assert r.status_code in (400, 404)

    def test_get_document_has_all_fields(self, client, admin_headers, user_headers):
        r_up = _upload_txt(client, admin_headers)
        doc_id = r_up.json()["data"]["id"]
        r = client.get(f"/api/v1/rag/documents/{doc_id}", headers=user_headers)
        data = r.json()["data"]
        required_fields = [
            "id",
            "filename",
            "content_type",
            "size_bytes",
            "chunk_count",
            "status",
            "ingested_at",
            "created_at",
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"


# ---------------------------------------------------------------------------
# DELETE /api/v1/rag/documents/{id} — delete
# ---------------------------------------------------------------------------


class TestDeleteDocument:
    def test_delete_requires_auth(self, client, admin_headers):
        r_up = _upload_txt(client, admin_headers)
        doc_id = r_up.json()["data"]["id"]
        r = client.delete(f"/api/v1/rag/documents/{doc_id}")
        assert r.status_code == 401

    def test_delete_requires_admin(self, client, admin_headers, user_headers):
        r_up = _upload_txt(client, admin_headers)
        doc_id = r_up.json()["data"]["id"]
        r = client.delete(f"/api/v1/rag/documents/{doc_id}", headers=user_headers)
        assert r.status_code == 403

    def test_delete_returns_200(self, client, admin_headers):
        r_up = _upload_txt(client, admin_headers)
        doc_id = r_up.json()["data"]["id"]
        r = client.delete(f"/api/v1/rag/documents/{doc_id}", headers=admin_headers)
        assert r.status_code == 200

    def test_delete_removes_document(self, client, admin_headers, user_headers):
        r_up = _upload_txt(client, admin_headers)
        doc_id = r_up.json()["data"]["id"]
        client.delete(f"/api/v1/rag/documents/{doc_id}", headers=admin_headers)
        r = client.get(f"/api/v1/rag/documents/{doc_id}", headers=user_headers)
        assert r.status_code == 404

    def test_delete_nonexistent_returns_404(self, client, admin_headers):
        fake_id = str(uuid.uuid4())
        r = client.delete(f"/api/v1/rag/documents/{fake_id}", headers=admin_headers)
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"

    def test_delete_invalid_id_returns_400(self, client, admin_headers):
        r = client.delete("/api/v1/rag/documents/not-uuid-at-all", headers=admin_headers)
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "INVALID_DOCUMENT_ID"

    def test_delete_returns_null_data(self, client, admin_headers):
        r_up = _upload_txt(client, admin_headers)
        doc_id = r_up.json()["data"]["id"]
        r = client.delete(f"/api/v1/rag/documents/{doc_id}", headers=admin_headers)
        assert r.json()["data"] is None


# ---------------------------------------------------------------------------
# POST /api/v1/rag/search — semantic search
# ---------------------------------------------------------------------------


class TestSearchRAG:
    def test_search_requires_auth(self, client):
        r = client.post("/api/v1/rag/search", json={"query": "hello"})
        assert r.status_code == 401

    def test_search_returns_200_with_auth(self, client, user_headers):
        r = client.post("/api/v1/rag/search", json={"query": "hello"}, headers=user_headers)
        assert r.status_code == 200

    def test_search_empty_index_returns_empty_chunks(self, client, user_headers):
        r = client.post("/api/v1/rag/search", json={"query": "hello"}, headers=user_headers)
        data = r.json()["data"]
        assert data["query"] == "hello"
        assert data["chunks"] == []
        assert data["total_documents_searched"] == 0

    def test_search_response_envelope(self, client, user_headers):
        r = client.post("/api/v1/rag/search", json={"query": "test query"}, headers=user_headers)
        body = r.json()
        assert "data" in body
        assert body["error"] is None

    def test_search_response_fields(self, client, user_headers):
        r = client.post("/api/v1/rag/search", json={"query": "test"}, headers=user_headers)
        data = r.json()["data"]
        assert "query" in data
        assert "chunks" in data
        assert "total_documents_searched" in data
        assert isinstance(data["chunks"], list)

    def test_search_missing_query_returns_422(self, client, user_headers):
        r = client.post("/api/v1/rag/search", json={}, headers=user_headers)
        assert r.status_code == 422

    def test_search_empty_query_returns_422(self, client, user_headers):
        r = client.post("/api/v1/rag/search", json={"query": ""}, headers=user_headers)
        assert r.status_code == 422

    def test_search_top_k_param(self, client, user_headers):
        r = client.post(
            "/api/v1/rag/search",
            json={"query": "hello", "top_k": 3},
            headers=user_headers,
        )
        assert r.status_code == 200

    def test_search_top_k_too_large_returns_422(self, client, user_headers):
        r = client.post(
            "/api/v1/rag/search",
            json={"query": "hello", "top_k": 100},
            headers=user_headers,
        )
        assert r.status_code == 422

    def test_search_top_k_zero_returns_422(self, client, user_headers):
        r = client.post(
            "/api/v1/rag/search",
            json={"query": "hello", "top_k": 0},
            headers=user_headers,
        )
        assert r.status_code == 422

    def test_search_document_ids_filter(self, client, user_headers):
        fake_id = str(uuid.uuid4())
        r = client.post(
            "/api/v1/rag/search",
            json={"query": "hello", "document_ids": [fake_id]},
            headers=user_headers,
        )
        assert r.status_code == 200
        assert r.json()["data"]["chunks"] == []

    def test_search_reflects_query_in_response(self, client, user_headers):
        query = "unique query text 42"
        r = client.post("/api/v1/rag/search", json={"query": query}, headers=user_headers)
        assert r.json()["data"]["query"] == query

    def test_search_invalid_token_returns_401(self, client):
        r = client.post(
            "/api/v1/rag/search",
            json={"query": "hello"},
            headers={"Authorization": "Bearer bad.token"},
        )
        assert r.status_code == 401


# ---------------------------------------------------------------------------
# Organization scoping
# ---------------------------------------------------------------------------


class TestRagOrgScoping:
    def test_rag_routes_are_scoped_to_org(self, app, client, tmp_path):
        with (
            patch("src.api.routes.rag.ingest_document_task", new=AsyncMock(return_value=None)),
            patch("src.api.routes.rag._get_uploads_dir", return_value=tmp_path),
            patch("src.api.tasks.rag._get_uploads_dir", return_value=tmp_path),
        ):
            org_a_headers = _seed_org_user(
                app,
                org_name="Org A",
                org_slug="org-a",
                username="orga-admin",
                email="orga-admin@example.com",
                role="admin",
            )
            org_b_headers = _seed_org_user(
                app,
                org_name="Org B",
                org_slug="org-b",
                username="orgb-admin",
                email="orgb-admin@example.com",
                role="admin",
            )

            upload = _upload_txt(client, org_a_headers, filename="tenant-a.txt")
            assert upload.status_code == 202, upload.text
            doc_id = upload.json()["data"]["id"]
            _mark_document_indexed(app, doc_id)

            list_a = client.get("/api/v1/rag/documents", headers=org_a_headers)
            assert list_a.status_code == 200
            assert any(item["id"] == doc_id for item in list_a.json()["data"]["items"])

            list_b = client.get("/api/v1/rag/documents", headers=org_b_headers)
            assert list_b.status_code == 200
            assert all(item["id"] != doc_id for item in list_b.json()["data"]["items"])

            get_b = client.get(f"/api/v1/rag/documents/{doc_id}", headers=org_b_headers)
            assert get_b.status_code == 404
            assert get_b.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"

            search_b = client.post(
                "/api/v1/rag/search", json={"query": "tenant-a"}, headers=org_b_headers
            )
            assert search_b.status_code == 200
            assert search_b.json()["data"]["total_documents_searched"] == 0

            delete_b = client.delete(f"/api/v1/rag/documents/{doc_id}", headers=org_b_headers)
            assert delete_b.status_code == 404
            assert delete_b.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"
