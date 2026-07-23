"""Tests for src/api/schemas/system.py — info, health, readiness, stats."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.api.schemas.system import (
    DebugToggleRequest,
    HealthOut,
    ReadinessComponentStatus,
    ReadinessOut,
    SystemInfoOut,
    SystemStats,
    TurnStatsOut,
)

# ---------------------------------------------------------------------------
# SystemInfoOut — verbosity bounded 0-3, all observability fields required
# ---------------------------------------------------------------------------


class TestSystemInfoOut:
    def _kwargs(self, **overrides) -> dict:
        base = {
            "version": "0.2.6",
            "api_version": "v1",
            "platform": "Linux 6.14",
            "python_version": "3.13.12",
            "debug": False,
            "verbose": False,
            "verbosity": 0,
            "uptime_s": 3600.0,
            "started_at": datetime(2026, 1, 1, tzinfo=UTC),
        }
        base.update(overrides)
        return base

    def test_valid_minimal(self) -> None:
        info = SystemInfoOut(**self._kwargs())
        assert info.commit is None
        assert info.verbosity == 0

    def test_valid_with_commit(self) -> None:
        info = SystemInfoOut(**self._kwargs(commit="abc1234"))
        assert info.commit == "abc1234"

    def test_verbosity_all_valid_levels(self) -> None:
        for level in (0, 1, 2, 3):
            info = SystemInfoOut(**self._kwargs(verbosity=level))
            assert info.verbosity == level

    def test_verbosity_below_min_rejected(self) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            SystemInfoOut(**self._kwargs(verbosity=-1))

    def test_verbosity_above_max_rejected(self) -> None:
        with pytest.raises(ValidationError, match="less than or equal to 3"):
            SystemInfoOut(**self._kwargs(verbosity=4))

    def test_missing_required_field(self) -> None:
        kw = self._kwargs()
        kw.pop("started_at")
        with pytest.raises(ValidationError):
            SystemInfoOut(**kw)


# ---------------------------------------------------------------------------
# HealthOut — status literal "ok", timestamp required
# ---------------------------------------------------------------------------


class TestHealthOut:
    def test_valid_with_default_status(self) -> None:
        h = HealthOut(timestamp=datetime(2026, 1, 1, tzinfo=UTC))
        assert h.status == "ok"

    def test_status_must_be_ok(self) -> None:
        with pytest.raises(ValidationError):
            HealthOut(
                status="degraded",  # type: ignore[arg-type]
                timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            )

    def test_missing_timestamp(self) -> None:
        with pytest.raises(ValidationError):
            HealthOut()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# ReadinessComponentStatus
# ---------------------------------------------------------------------------


class TestReadinessComponentStatus:
    def test_valid_ok(self) -> None:
        c = ReadinessComponentStatus(name="llm_provider", ok=True, latency_ms=42)
        assert c.detail is None

    def test_valid_failing(self) -> None:
        c = ReadinessComponentStatus(
            name="redis", ok=False, latency_ms=None, detail="connection refused"
        )
        assert c.ok is False
        assert c.detail == "connection refused"

    def test_optional_latency_and_detail(self) -> None:
        c = ReadinessComponentStatus(name="x", ok=True)
        assert c.latency_ms is None
        assert c.detail is None

    def test_missing_required_field(self) -> None:
        with pytest.raises(ValidationError):
            ReadinessComponentStatus(ok=True)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# ReadinessOut — components list required (even if empty)
# ---------------------------------------------------------------------------


class TestReadinessOut:
    def test_valid_ready_no_components(self) -> None:
        r = ReadinessOut(ready=True, components=[])
        assert r.ready is True
        assert r.components == []

    def test_valid_with_components(self) -> None:
        r = ReadinessOut(
            ready=False,
            components=[
                ReadinessComponentStatus(name="db", ok=True),
                ReadinessComponentStatus(name="redis", ok=False, detail="timeout"),
            ],
        )
        assert len(r.components) == 2
        assert r.components[1].ok is False

    def test_components_field_is_required(self) -> None:
        with pytest.raises(ValidationError):
            ReadinessOut(ready=True)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# DebugToggleRequest — verbosity bounds, all-optional
# ---------------------------------------------------------------------------


class TestDebugToggleRequest:
    def test_empty(self) -> None:
        r = DebugToggleRequest()
        assert r.debug is None
        assert r.verbose is None
        assert r.verbosity is None

    def test_partial(self) -> None:
        r = DebugToggleRequest(debug=True)
        assert r.debug is True
        assert r.verbose is None

    def test_verbosity_at_min_0(self) -> None:
        assert DebugToggleRequest(verbosity=0).verbosity == 0

    def test_verbosity_at_max_3(self) -> None:
        assert DebugToggleRequest(verbosity=3).verbosity == 3

    def test_verbosity_below_min_rejected(self) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            DebugToggleRequest(verbosity=-1)

    def test_verbosity_above_max_rejected(self) -> None:
        with pytest.raises(ValidationError, match="less than or equal to 3"):
            DebugToggleRequest(verbosity=4)


# ---------------------------------------------------------------------------
# TurnStatsOut
# ---------------------------------------------------------------------------


class TestTurnStatsOut:
    def test_valid(self) -> None:
        t = TurnStatsOut(
            session_id="s",
            message_id="m",
            input_tokens=320,
            output_tokens=88,
            duration_ms=4200,
            tool_calls=3,
        )
        assert t.tool_calls == 3

    def test_zero_values_accepted(self) -> None:
        t = TurnStatsOut(
            session_id="s",
            message_id="m",
            input_tokens=0,
            output_tokens=0,
            duration_ms=0,
            tool_calls=0,
        )
        assert t.tool_calls == 0

    def test_missing_required(self) -> None:
        with pytest.raises(ValidationError):
            TurnStatsOut(  # type: ignore[call-arg]
                session_id="s",
                message_id="m",
                input_tokens=0,
                output_tokens=0,
                duration_ms=0,
            )


# ---------------------------------------------------------------------------
# SystemStats — error_rate bounded 0.0-1.0, several required fields
# ---------------------------------------------------------------------------


class TestSystemStats:
    def _kwargs(self, **overrides) -> dict:
        base = {
            "total_orgs": 10,
            "total_users": 100,
            "active_sessions": 25,
            "estimated_token_usage_24h": 1_000_000,
            "api_requests_24h": 5000,
            "db_pool_status": "healthy",
            "db_pool_size": 10,
            "db_pool_max": 10,
            "redis_connected": True,
            "uptime_s": 7200.0,
            "version": "0.2.6",
            "started_at": datetime(2026, 1, 1, tzinfo=UTC),
        }
        base.update(overrides)
        return base

    def test_valid_minimal(self) -> None:
        s = SystemStats(**self._kwargs())
        assert s.error_rate_24h is None
        assert s.redis_latency_ms is None

    def test_valid_with_optional_metrics(self) -> None:
        s = SystemStats(**self._kwargs(error_rate_24h=0.023, redis_latency_ms=5))
        assert s.error_rate_24h == 0.023
        assert s.redis_latency_ms == 5

    def test_error_rate_at_min_0(self) -> None:
        s = SystemStats(**self._kwargs(error_rate_24h=0.0))
        assert s.error_rate_24h == 0.0

    def test_error_rate_at_max_1(self) -> None:
        s = SystemStats(**self._kwargs(error_rate_24h=1.0))
        assert s.error_rate_24h == 1.0

    def test_error_rate_below_min_rejected(self) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            SystemStats(**self._kwargs(error_rate_24h=-0.01))

    def test_error_rate_above_max_rejected(self) -> None:
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            SystemStats(**self._kwargs(error_rate_24h=1.01))

    def test_missing_required_field(self) -> None:
        kw = self._kwargs()
        kw.pop("redis_connected")
        with pytest.raises(ValidationError):
            SystemStats(**kw)
