"""Tests for per-user resource quota enforcement (Task M5.5).

Covers:
- UsageTracker sliding-window and daily-token counters
- QuotaEnforcer raises 429 at each limit type
- Config parsing of `quotas:` YAML section
- GET /api/v1/users/me/quota endpoint
- POST /api/v1/sessions enforces concurrent session limit
- POST /api/v1/sessions/{id}/messages enforces rate and token-budget limits
"""

from __future__ import annotations

import os
import time
import uuid
from collections import deque
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi")

_TEST_JWT_SECRET = "testsecret_mustbe32chars_minimum00"
os.environ.setdefault("COGTRIX_JWT_SECRET", _TEST_JWT_SECRET)
os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

import asyncio as _asyncio  # noqa: E402

from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from cogtrix_core.api.db.engine import Base, get_db  # noqa: E402
from cogtrix_core.api.quota import (  # noqa: E402
    QuotaConfig,
    QuotaEnforcer,
    UsageTracker,
    _quota_config_from_app_config,
    get_enforcer,
    get_tracker,
    get_user_quota_status,
)

# ---------------------------------------------------------------------------
# Shared app fixture
# ---------------------------------------------------------------------------

_VALID_PASSWORD = "TestPass1!"


@pytest.fixture()
def app():
    """FastAPI app with in-memory SQLite."""
    from cogtrix_core.api.app import create_app

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)

    loop = _asyncio.new_event_loop()
    _asyncio.set_event_loop(loop)
    loop.run_until_complete(_create_tables(engine))

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


async def _create_tables(engine):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@pytest.fixture()
def client(app):
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _register_and_login(client):
    """Register a user and return (user_id, access_token)."""
    username = f"u_{uuid.uuid4().hex[:8]}"
    email = f"{username}@ex.com"
    r = client.post(
        "/api/v1/auth/register",
        json={"username": username, "email": email, "password": _VALID_PASSWORD},
    )
    assert r.status_code == 201, r.text
    data = r.json()["data"]
    token = data["access_token"]
    # Decode user_id from the profile endpoint
    me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    user_id = me.json()["data"]["id"]
    return user_id, token


# ---------------------------------------------------------------------------
# UsageTracker unit tests
# ---------------------------------------------------------------------------


class TestUsageTracker:
    def test_record_and_count_requests(self):
        t = UsageTracker()
        uid = "user-1"
        assert t.get_requests_in_window(uid) == 0
        t.record_request(uid)
        t.record_request(uid)
        assert t.get_requests_in_window(uid) == 2

    def test_sliding_window_evicts_old_timestamps(self):
        t = UsageTracker()
        uid = "user-2"
        # Inject an old timestamp directly
        old_ts = time.time() - 7200  # 2 hours ago
        t._requests[uid] = deque([old_ts])
        # Window is 1 hour; old entry must be evicted
        assert t.get_requests_in_window(uid, window_seconds=3600) == 0

    def test_record_tokens_accumulates_today(self):
        t = UsageTracker()
        uid = "user-3"
        t.record_tokens(uid, 100)
        t.record_tokens(uid, 200)
        assert t.get_daily_tokens(uid) == 300

    def test_daily_tokens_reset_on_new_day(self):
        t = UsageTracker()
        uid = "user-4"
        # Simulate a record from "yesterday"
        t._daily_tokens[uid] = ("1990-01-01", 9999)
        # get_daily_tokens should treat stale date as 0
        assert t.get_daily_tokens(uid) == 0

    def test_record_tokens_replaces_stale_date(self):
        t = UsageTracker()
        uid = "user-5"
        t._daily_tokens[uid] = ("1990-01-01", 9999)
        t.record_tokens(uid, 50)
        assert t.get_daily_tokens(uid) == 50

    def test_separate_users_isolated(self):
        t = UsageTracker()
        t.record_request("alice")
        t.record_request("alice")
        t.record_request("bob")
        assert t.get_requests_in_window("alice") == 2
        assert t.get_requests_in_window("bob") == 1
        assert t.get_requests_in_window("charlie") == 0

    def test_cleanup_stale_entries_removes_old_users(self):
        """Test that cleanup_stale_entries removes entries for inactive users."""
        t = UsageTracker()
        # Create some active and inactive users
        t.record_request("active-user")
        # Simulate old entries by setting timestamps from long ago
        old_ts = time.time() - 8 * 24 * 60 * 60  # 8 days ago (beyond 7-day threshold)
        t._requests["old-user-1"] = deque([old_ts])
        t._requests["old-user-2"] = deque([old_ts])
        # One user with both old request and old token entry
        t._requests["old-user-3"] = deque([old_ts])
        t._daily_tokens["old-user-3"] = ("1990-01-01", 9999)
        # Add an entry that's just the token (no requests)
        t._daily_tokens["old-token-user"] = ("1990-01-01", 500)

        # Before cleanup: count all entries
        before_requests = len(t._requests)
        before_tokens = len(t._daily_tokens)

        # Cleanup entries older than 7 days
        removed = t.cleanup_stale_entries(max_idle_days=7)

        # Verify stale entries were removed
        assert "old-user-1" not in t._requests
        assert "old-user-2" not in t._requests
        assert "old-user-3" not in t._requests
        assert "old-user-3" not in t._daily_tokens
        assert "old-token-user" not in t._daily_tokens
        # Active user should still be there
        assert "active-user" in t._requests

        # Verify return value
        assert removed == before_requests + before_tokens - (
            len(t._requests) + len(t._daily_tokens)
        )

    def test_cleanup_stale_entries_keeps_recent_users(self):
        """Test that cleanup_stale_entries keeps entries for recent users."""
        t = UsageTracker()
        # Create users with recent activity
        t.record_request("recent-user-1")
        t.record_request("recent-user-2")
        t.record_tokens("recent-user-2", 100)

        # Cleanup with a very large threshold (should remove nothing)
        removed = t.cleanup_stale_entries(max_idle_days=365)

        # All recent users should still be there
        assert "recent-user-1" in t._requests
        assert "recent-user-2" in t._requests
        assert "recent-user-2" in t._daily_tokens
        assert removed == 0


# ---------------------------------------------------------------------------
# QuotaEnforcer unit tests
# ---------------------------------------------------------------------------


class TestQuotaEnforcer:
    # ── request rate ──────────────────────────────────────────────

    def test_rate_unlimited_always_passes(self):
        cfg = QuotaConfig(requests_per_hour=None)
        t = UsageTracker()
        e = QuotaEnforcer(cfg, t)
        for _ in range(1000):
            e.check_request_rate("u")  # must not raise

    def test_rate_under_limit_passes(self):
        cfg = QuotaConfig(requests_per_hour=10)
        t = UsageTracker()
        for _ in range(9):
            t.record_request("u")
        QuotaEnforcer(cfg, t).check_request_rate("u")

    def test_rate_at_limit_raises_429(self):
        cfg = QuotaConfig(requests_per_hour=5)
        t = UsageTracker()
        for _ in range(5):
            t.record_request("u")
        with pytest.raises(HTTPException) as exc_info:
            QuotaEnforcer(cfg, t).check_request_rate("u")
        assert exc_info.value.status_code == 429
        assert exc_info.value.detail["code"] == "RATE_LIMIT_EXCEEDED"

    def test_rate_over_limit_raises_429(self):
        cfg = QuotaConfig(requests_per_hour=3)
        t = UsageTracker()
        for _ in range(10):
            t.record_request("u")
        with pytest.raises(HTTPException) as exc_info:
            QuotaEnforcer(cfg, t).check_request_rate("u")
        assert exc_info.value.status_code == 429
        assert exc_info.value.detail["code"] == "RATE_LIMIT_EXCEEDED"

    # ── token budget ──────────────────────────────────────────────

    def test_token_budget_unlimited_passes(self):
        cfg = QuotaConfig(token_budget=None)
        t = UsageTracker()
        t.record_tokens("u", 10_000_000)
        QuotaEnforcer(cfg, t).check_token_budget("u")

    def test_token_budget_under_limit_passes(self):
        cfg = QuotaConfig(token_budget=1000)
        t = UsageTracker()
        t.record_tokens("u", 999)
        QuotaEnforcer(cfg, t).check_token_budget("u")

    def test_token_budget_at_limit_raises_429(self):
        cfg = QuotaConfig(token_budget=500)
        t = UsageTracker()
        t.record_tokens("u", 500)
        with pytest.raises(HTTPException) as exc_info:
            QuotaEnforcer(cfg, t).check_token_budget("u")
        assert exc_info.value.status_code == 429
        assert exc_info.value.detail["code"] == "TOKEN_BUDGET_EXCEEDED"

    # ── concurrent sessions ────────────────────────────────────────

    def test_session_limit_unlimited_passes(self):
        cfg = QuotaConfig(max_concurrent_sessions=None)
        t = UsageTracker()
        QuotaEnforcer(cfg, t).check_concurrent_sessions("u", 10_000)

    def test_session_limit_under_limit_passes(self):
        cfg = QuotaConfig(max_concurrent_sessions=5)
        t = UsageTracker()
        QuotaEnforcer(cfg, t).check_concurrent_sessions("u", 4)

    def test_session_limit_at_limit_raises_429(self):
        cfg = QuotaConfig(max_concurrent_sessions=3)
        t = UsageTracker()
        with pytest.raises(HTTPException) as exc_info:
            QuotaEnforcer(cfg, t).check_concurrent_sessions("u", 3)
        assert exc_info.value.status_code == 429
        assert exc_info.value.detail["code"] == "SESSION_LIMIT_EXCEEDED"


# ---------------------------------------------------------------------------
# get_enforcer singleton
# ---------------------------------------------------------------------------


class TestGetEnforcer:
    def test_get_enforcer_returns_enforcer(self):
        e = get_enforcer(QuotaConfig())
        assert isinstance(e, QuotaEnforcer)

    def test_get_enforcer_reconfigures_on_new_config(self):
        e1 = get_enforcer(QuotaConfig(requests_per_hour=10))
        e2 = get_enforcer(QuotaConfig(requests_per_hour=20))
        assert e1 is not e2

    def test_get_tracker_is_stable(self):
        t1 = get_tracker()
        t2 = get_tracker()
        assert t1 is t2


# ---------------------------------------------------------------------------
# _quota_config_from_app_config
# ---------------------------------------------------------------------------


class TestQuotaConfigFromAppConfig:
    def test_reads_all_fields(self):
        fake = MagicMock()
        fake.quota_token_budget_per_day = 50000
        fake.quota_requests_per_hour = 100
        fake.quota_max_concurrent_sessions = 5
        qcfg = _quota_config_from_app_config(fake)
        assert qcfg.token_budget == 50000
        assert qcfg.requests_per_hour == 100
        assert qcfg.max_concurrent_sessions == 5

    def test_missing_attrs_default_to_none(self):
        fake = object()  # no quota attrs
        qcfg = _quota_config_from_app_config(fake)
        assert qcfg.token_budget is None
        assert qcfg.requests_per_hour is None
        assert qcfg.max_concurrent_sessions is None


# ---------------------------------------------------------------------------
# get_user_quota_status
# ---------------------------------------------------------------------------


class TestGetUserQuotaStatus:
    def test_returns_expected_structure(self):
        t = get_tracker()
        uid = f"qs-{uuid.uuid4().hex[:8]}"
        t.record_tokens(uid, 42)
        t.record_request(uid)

        qcfg = QuotaConfig(token_budget=1000, requests_per_hour=60, max_concurrent_sessions=10)
        result = get_user_quota_status(uid, qcfg)
        assert result["limits"]["token_budget_per_day"] == 1000
        assert result["limits"]["requests_per_hour"] == 60
        assert result["limits"]["max_concurrent_sessions"] == 10
        assert result["usage"]["tokens_used_today"] == 42
        assert result["usage"]["requests_last_hour"] == 1


# ---------------------------------------------------------------------------
# Config parsing — quotas: YAML section
# ---------------------------------------------------------------------------


class TestConfigQuotasParsing:
    def _load(self, yaml_text: str, tmp_path):
        """Write yaml_text to a temp file and load via _apply_config_file."""
        from cogtrix_core.config import Config, _apply_config_file

        cfg_file = tmp_path / "test_config.yaml"
        cfg_file.write_text(yaml_text)
        cfg = Config()
        _apply_config_file(cfg, cfg_file)
        return cfg

    def test_parses_all_quota_fields(self, tmp_path):
        cfg = self._load(
            "quotas:\n"
            "  token_budget_per_day: 50000\n"
            "  requests_per_hour: 100\n"
            "  max_concurrent_sessions: 5\n",
            tmp_path,
        )
        assert cfg.quota_token_budget_per_day == 50000
        assert cfg.quota_requests_per_hour == 100
        assert cfg.quota_max_concurrent_sessions == 5

    def test_invalid_zero_value_ignored(self, tmp_path):
        cfg = self._load(
            "quotas:\n"
            "  token_budget_per_day: 0\n"
            "  requests_per_hour: 0\n"
            "  max_concurrent_sessions: 0\n",
            tmp_path,
        )
        assert cfg.quota_token_budget_per_day is None
        assert cfg.quota_requests_per_hour is None
        assert cfg.quota_max_concurrent_sessions is None

    def test_negative_value_ignored(self, tmp_path):
        cfg = self._load("quotas:\n  requests_per_hour: -5\n", tmp_path)
        assert cfg.quota_requests_per_hour is None

    def test_non_numeric_value_ignored_not_crash(self, tmp_path):
        # Regression #2203: a non-numeric quota value must not raise an
        # unhandled ValueError out of load_config (which aborted startup for
        # every entrypoint). It is tolerated like every other numeric config
        # key — warn-and-skip, leaving the default.
        cfg = self._load(
            "quotas:\n"
            "  token_budget_per_day: abc\n"
            "  requests_per_hour: not-a-number\n"
            "  max_concurrent_sessions: []\n",
            tmp_path,
        )
        assert cfg.quota_token_budget_per_day is None
        assert cfg.quota_requests_per_hour is None
        assert cfg.quota_max_concurrent_sessions is None

    def test_valid_value_with_one_invalid_sibling(self, tmp_path):
        # A bad value in one field must not discard a valid sibling.
        cfg = self._load(
            "quotas:\n  token_budget_per_day: abc\n  requests_per_hour: 100\n",
            tmp_path,
        )
        assert cfg.quota_token_budget_per_day is None
        assert cfg.quota_requests_per_hour == 100

    def test_absent_quotas_section_leaves_none(self, tmp_path):
        cfg = self._load("verbosity: 0\n", tmp_path)
        assert cfg.quota_token_budget_per_day is None
        assert cfg.quota_requests_per_hour is None
        assert cfg.quota_max_concurrent_sessions is None


# ---------------------------------------------------------------------------
# GET /api/v1/users/me/quota — HTTP endpoint
# ---------------------------------------------------------------------------


class TestGetMyQuotaEndpoint:
    def test_requires_auth(self, client):
        r = client.get("/api/v1/users/me/quota")
        assert r.status_code == 401

    def test_returns_200_with_structure(self, client):
        _, token = _register_and_login(client)
        r = client.get(
            "/api/v1/users/me/quota",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["error"] is None
        data = body["data"]
        assert "limits" in data
        assert "usage" in data
        limits = data["limits"]
        assert "token_budget_per_day" in limits
        assert "requests_per_hour" in limits
        assert "max_concurrent_sessions" in limits
        usage = data["usage"]
        assert "tokens_used_today" in usage
        assert "requests_last_hour" in usage

    def test_all_limits_none_when_unconfigured(self, client):
        _, token = _register_and_login(client)
        r = client.get(
            "/api/v1/users/me/quota",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        limits = r.json()["data"]["limits"]
        assert limits["token_budget_per_day"] is None
        assert limits["requests_per_hour"] is None
        assert limits["max_concurrent_sessions"] is None


# ---------------------------------------------------------------------------
# POST /api/v1/sessions — concurrent session quota
# ---------------------------------------------------------------------------


class TestSessionCreationQuota:
    def test_session_creation_blocked_at_limit(self, client, app):
        """When max_concurrent_sessions=1, a second create returns 429."""
        _, token = _register_and_login(client)
        headers = {"Authorization": f"Bearer {token}"}

        # Inject a quota config via app.state.config mock
        mock_cfg = MagicMock()
        mock_cfg.quota_max_concurrent_sessions = 1
        mock_cfg.quota_token_budget_per_day = None
        mock_cfg.quota_requests_per_hour = None

        with patch.object(app.state, "config", mock_cfg):
            # First session — should succeed (count=0 < 1)
            r1 = client.post(
                "/api/v1/sessions",
                json={"name": f"s1-{uuid.uuid4().hex[:6]}"},
                headers=headers,
            )
            assert r1.status_code == 201, r1.text

            # Second session — count=1 >= limit=1, should be blocked
            r2 = client.post(
                "/api/v1/sessions",
                json={"name": f"s2-{uuid.uuid4().hex[:6]}"},
                headers=headers,
            )
            assert r2.status_code == 429, r2.text
            assert r2.json()["error"]["code"] == "SESSION_LIMIT_EXCEEDED"

    def test_session_creation_allowed_below_limit(self, client, app):
        """When max_concurrent_sessions=5, creating 2 sessions is fine."""
        _, token = _register_and_login(client)
        headers = {"Authorization": f"Bearer {token}"}

        mock_cfg = MagicMock()
        mock_cfg.quota_max_concurrent_sessions = 5
        mock_cfg.quota_token_budget_per_day = None
        mock_cfg.quota_requests_per_hour = None

        with patch.object(app.state, "config", mock_cfg):
            for i in range(2):
                r = client.post(
                    "/api/v1/sessions",
                    json={"name": f"ok-{i}-{uuid.uuid4().hex[:6]}"},
                    headers=headers,
                )
                assert r.status_code == 201, r.text


# ---------------------------------------------------------------------------
# POST /api/v1/sessions/{id}/messages — rate limit quota
# ---------------------------------------------------------------------------


class TestMessageRateQuota:
    def test_rate_limit_blocks_message(self, client, app):
        """When requests_per_hour=0-already-filled, sending a message returns 429."""
        user_id, token = _register_and_login(client)
        headers = {"Authorization": f"Bearer {token}"}

        mock_cfg = MagicMock()
        mock_cfg.quota_requests_per_hour = 1
        mock_cfg.quota_token_budget_per_day = None
        mock_cfg.quota_max_concurrent_sessions = None

        # Pre-fill the tracker so the user is already at the limit
        tracker = get_tracker()
        tracker.record_request(user_id)

        # Create a session first (no quota on sessions in this mock)
        no_quota_cfg = MagicMock()
        no_quota_cfg.quota_max_concurrent_sessions = None
        no_quota_cfg.quota_token_budget_per_day = None
        no_quota_cfg.quota_requests_per_hour = None

        with patch.object(app.state, "config", no_quota_cfg):
            rc = client.post(
                "/api/v1/sessions",
                json={"name": f"rl-{uuid.uuid4().hex[:6]}"},
                headers=headers,
            )
            assert rc.status_code == 201, rc.text
            session_id = rc.json()["data"]["id"]

        with patch.object(app.state, "config", mock_cfg):
            r = client.post(
                f"/api/v1/sessions/{session_id}/messages",
                json={"content": "hello"},
                headers=headers,
            )
            assert r.status_code == 429, r.text
            assert r.json()["error"]["code"] == "RATE_LIMIT_EXCEEDED"

    def test_token_budget_blocks_message(self, client, app):
        """When token_budget=100 and user already spent 100, sending returns 429."""
        user_id, token = _register_and_login(client)
        headers = {"Authorization": f"Bearer {token}"}

        mock_cfg = MagicMock()
        mock_cfg.quota_requests_per_hour = None
        mock_cfg.quota_token_budget_per_day = 100
        mock_cfg.quota_max_concurrent_sessions = None

        # Pre-fill the token tracker
        tracker = get_tracker()
        tracker.record_tokens(user_id, 100)

        no_quota_cfg = MagicMock()
        no_quota_cfg.quota_max_concurrent_sessions = None
        no_quota_cfg.quota_token_budget_per_day = None
        no_quota_cfg.quota_requests_per_hour = None

        with patch.object(app.state, "config", no_quota_cfg):
            rc = client.post(
                "/api/v1/sessions",
                json={"name": f"tb-{uuid.uuid4().hex[:6]}"},
                headers=headers,
            )
            assert rc.status_code == 201, rc.text
            session_id = rc.json()["data"]["id"]

        with patch.object(app.state, "config", mock_cfg):
            r = client.post(
                f"/api/v1/sessions/{session_id}/messages",
                json={"content": "hello"},
                headers=headers,
            )
            assert r.status_code == 429, r.text
            assert r.json()["error"]["code"] == "TOKEN_BUDGET_EXCEEDED"
