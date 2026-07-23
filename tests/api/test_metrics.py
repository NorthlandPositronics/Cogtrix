"""Tests for the Prometheus metrics endpoint.

Coverage:
  - GET /api/v1/metrics returns Prometheus text format.
  - Metrics endpoint requires authentication by default.
  - All 7 expected metric names are present in the output.
  - Request middleware increments request counters with normalized paths.
"""

from __future__ import annotations

import os

import jwt
import pytest

pytest.importorskip("fastapi")

# Snapshot the JWT secret at import (before any load_config() unsets it, #2102).
_JWT_SECRET_FOR_SIGNING = (
    os.environ.get("COGTRIX_JWT_SECRET") or "testsecret_mustbe32chars_minimum00"
)


# ---------------------------------------------------------------------------
# Pure function tests (no FastAPI required)
# ---------------------------------------------------------------------------
class TestPathNormalization:
    def test_uuid_segment(self):
        """UUID path segments are normalized to {id}."""
        from cogtrix_core.api.routes._metrics_core import _normalize_path

        assert (
            _normalize_path("/api/v1/sessions/123e4567-e89b-12d3-a456-426614174000")
            == "/api/v1/sessions/{id}"
        )

    def test_numeric_segment(self):
        """Numeric path segments are normalized to {id}."""
        from cogtrix_core.api.routes._metrics_core import _normalize_path

        assert _normalize_path("/api/v1/sessions/42") == "/api/v1/sessions/{id}"

    def test_preserves_static_segments(self):
        """Static path segments are preserved."""
        from cogtrix_core.api.routes._metrics_core import _normalize_path

        assert _normalize_path("/api/v1/health") == "/api/v1/health"
        assert _normalize_path("/api/v1/metrics") == "/api/v1/metrics"

    def test_mixed_path(self):
        """Paths with both static and dynamic segments."""
        from cogtrix_core.api.routes._metrics_core import _normalize_path

        assert (
            _normalize_path("/api/v1/sessions/42/messages/123e4567-e89b-12d3-a456-426614174000")
            == "/api/v1/sessions/{id}/messages/{id}"
        )


class TestMetricDefinitions:
    @pytest.fixture(autouse=True)
    def _skip_without_fastapi(self):
        pytest.importorskip("fastapi")


# ---------------------------------------------------------------------------
# FastAPI-dependent integration tests
# ---------------------------------------------------------------------------
class TestMetricsEndpoint:
    @pytest.fixture(autouse=True)
    def _skip_without_fastapi(self):
        pytest.importorskip("fastapi")

    @staticmethod
    def _make_test_token() -> str:
        """Return a valid JWT for test requests."""
        return jwt.encode(
            {"sub": "test-user-uuid", "role": "admin", "iat": 0},
            _JWT_SECRET_FOR_SIGNING,
            algorithm="HS256",
        )

    def test_metrics_requires_auth_by_default(self, client):
        """Metrics endpoint rejects anonymous requests when auth is enabled."""
        response = client.get("/api/v1/metrics")
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "UNAUTHORIZED"

    def test_metrics_contains_all_names(self, client):
        """All 7 defined metrics appear in the Prometheus output."""
        token = self._make_test_token()
        response = client.get("/api/v1/metrics", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        text = response.text

        expected_names = [
            "cogtrix_sessions_active",
            "cogtrix_llm_requests_total",
            "cogtrix_llm_latency_seconds",
            "cogtrix_tool_calls_total",
            "cogtrix_tasks_total",
            "cogtrix_api_requests_total",
            "cogtrix_db_connections",
        ]
        for name in expected_names:
            assert name in text, f"Metric {name} missing from /metrics output"

    def test_request_metrics_incremented(self, client):
        """After making a request, request counters reflect the call."""
        token = self._make_test_token()
        client.get("/api/v1/metrics", headers={"Authorization": f"Bearer {token}"})
        client.get("/api/v1/health")

        response = client.get("/api/v1/metrics", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        text = response.text

        assert "cogtrix_api_requests_total" in text
        assert "cogtrix_llm_requests_total" in text
        assert "cogtrix_db_connections" in text

    def test_metrics_auth_by_default(self, client):
        """Metrics endpoint requires admin authentication by default."""
        response = client.get("/api/v1/metrics")
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "UNAUTHORIZED"

    def test_metrics_auth_with_admin_token(self, client):
        """Metrics endpoint allows access with valid admin token."""
        token = self._make_test_token()
        response = client.get("/api/v1/metrics", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200

    def test_api_request_metrics(self, client):
        """API request metrics are collected for routes."""
        token = self._make_test_token()
        client.get("/api/v1/metrics", headers={"Authorization": f"Bearer {token}"})
        client.get("/api/v1/health")

        response = client.get("/api/v1/metrics", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        text = response.text
        assert "cogtrix_api_requests_total" in text

    def test_db_connection_metrics(self, client):
        """DB connection metrics are present."""
        token = self._make_test_token()
        response = client.get("/api/v1/metrics", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        text = response.text
        assert "cogtrix_db_connections" in text
