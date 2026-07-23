"""Tests for RBAC middleware — ``require(permission)`` dependency (issue #596)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from cogtrix_core.api.auth import TokenData, get_current_user  # noqa: E402
from cogtrix_core.auth.middleware import require  # noqa: E402
from cogtrix_core.auth.permissions import Permission  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_token_data(role: str, user_id: str = "test-user-id") -> TokenData:
    return TokenData(
        user_id=user_id,
        role=role,
        raw_claims={"sub": user_id, "role": role},
    )


def _make_client(role: str = "member", user_id: str = "test-user-id") -> TestClient:
    app = FastAPI()
    router = APIRouter()

    @router.get("/sessions")
    async def list_sessions(
        current_user: TokenData = Depends(require(Permission.SESSIONS_READ)),
    ) -> dict:
        return {"ok": True, "user_id": current_user.user_id}

    @router.post("/sessions")
    async def create_session(
        current_user: TokenData = Depends(require(Permission.SESSIONS_CREATE)),
    ) -> dict:
        return {"ok": True, "user_id": current_user.user_id}

    @router.get("/admin-only")
    async def admin_only(
        current_user: TokenData = Depends(require(Permission.USERS_MANAGE)),
    ) -> dict:
        return {"ok": True}

    app.include_router(router)

    def _override_current_user() -> TokenData:
        return _make_token_data(role=role, user_id=user_id)

    app.dependency_overrides[get_current_user] = _override_current_user
    return TestClient(app)


# ---------------------------------------------------------------------------
# Success paths
# ---------------------------------------------------------------------------


class TestRequireSuccess:
    def test_superadmin_can_access_any_permission(self):
        client = _make_client(role="superadmin")
        r = client.get("/admin-only")
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_admin_can_access_non_billing_delete(self):
        client = _make_client(role="admin")
        r = client.get("/sessions")
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_member_can_read_sessions(self):
        client = _make_client(role="member")
        r = client.get("/sessions")
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_member_can_create_sessions(self):
        client = _make_client(role="member")
        r = client.post("/sessions")
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_viewer_can_read_sessions(self):
        client = _make_client(role="viewer")
        r = client.get("/sessions")
        assert r.status_code == 200

    def test_returns_user_id_on_success(self):
        client = _make_client(role="member", user_id="u-123")
        r = client.get("/sessions")
        assert r.json()["user_id"] == "u-123"


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


class TestRequireFailure:
    def test_no_auth_returns_401(self):
        app = FastAPI()
        router = APIRouter()

        @router.get("/sessions")
        async def list_sessions(
            current_user: TokenData = Depends(require(Permission.SESSIONS_READ)),
        ) -> dict:
            return {"ok": True}

        app.include_router(router)
        client = TestClient(app)
        r = client.get("/sessions")
        assert r.status_code == 401

    def test_viewer_cannot_create_sessions(self):
        client = _make_client(role="viewer")
        r = client.post("/sessions")
        assert r.status_code == 403

    def test_member_cannot_manage_users(self):
        client = _make_client(role="member")
        r = client.get("/admin-only")
        assert r.status_code == 403

    def test_readonly_cannot_create_sessions(self):
        client = _make_client(role="readonly")
        r = client.post("/sessions")
        assert r.status_code == 403

    def test_unknown_role_is_denied(self):
        client = _make_client(role="bogus")
        r = client.get("/sessions")
        assert r.status_code == 403

    def test_403_includes_permission_name(self):
        client = _make_client(role="viewer")
        r = client.post("/sessions")
        assert r.status_code == 403
        body = r.json()
        assert body["detail"]["code"] == "FORBIDDEN"
        assert Permission.SESSIONS_CREATE in body["detail"]["message"]

    def test_admin_denied_billing_delete(self):
        from fastapi import APIRouter as AR

        app = FastAPI()
        router = AR()

        @router.get("/billing")
        async def billing(
            current_user: TokenData = Depends(require(Permission.BILLING_DELETE)),
        ) -> dict:
            return {"ok": True}

        app.include_router(router)

        def _override_current_user() -> TokenData:
            return _make_token_data(role="admin")

        app.dependency_overrides[get_current_user] = _override_current_user
        c = TestClient(app)
        r = c.get("/billing")
        assert r.status_code == 403
        assert Permission.BILLING_DELETE in r.json()["detail"]["message"]


# ---------------------------------------------------------------------------
# Performance sanity
# ---------------------------------------------------------------------------


class TestRequirePerformance:
    def test_permission_check_is_fast(self):
        import time

        from cogtrix_core.auth.permissions import has_permission

        start = time.perf_counter()
        for _ in range(100_000):
            assert has_permission("member", Permission.SESSIONS_READ) is True
        elapsed = time.perf_counter() - start
        avg_us = (elapsed / 100_000) * 1_000_000
        assert avg_us < 100, f"avg {avg_us:.1f} µs exceeds 100 µs budget"
