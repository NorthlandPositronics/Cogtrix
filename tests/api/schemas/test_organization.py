"""Tests for cogtrix_core/api/schemas/organization.py — tenant org models."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from cogtrix_core.api.schemas.organization import (
    AdminStats,
    AuditLogEntryOut,
    ImpersonateRequest,
    ImpersonateResponse,
    OrganizationCreate,
    OrganizationOut,
    OrganizationUpdate,
    OrgAuditLog,
    OrgSummary,
    OrgUsage,
)

# ---------------------------------------------------------------------------
# OrganizationOut — settings JSON parsing + ensure_utc on both timestamps
# ---------------------------------------------------------------------------


class TestOrganizationOut:
    def test_valid_full(self) -> None:
        org = OrganizationOut(
            id="abc",
            name="Acme",
            slug="acme",
            plan="enterprise",
            settings={"theme": "dark"},
            is_active=True,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            updated_at=datetime(2026, 1, 2, tzinfo=UTC),
        )
        assert org.name == "Acme"
        assert org.settings == {"theme": "dark"}

    def test_settings_json_string_is_parsed(self) -> None:
        """settings comes from the DB as a JSON string; the validator parses
        it into a dict before model construction."""
        org = OrganizationOut(
            id="abc",
            name="Acme",
            slug="acme",
            plan="free",
            settings='{"theme":"dark","retention_days":30}',  # type: ignore[arg-type]
            is_active=True,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            updated_at=datetime(2026, 1, 2, tzinfo=UTC),
        )
        assert org.settings == {"theme": "dark", "retention_days": 30}

    def test_settings_empty_string_becomes_none(self) -> None:
        """Empty string from DB becomes None — distinguishes 'unset' from
        '{}' (explicit empty object)."""
        org = OrganizationOut(
            id="abc",
            name="Acme",
            slug="acme",
            plan="free",
            settings="",  # type: ignore[arg-type]
            is_active=True,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            updated_at=datetime(2026, 1, 2, tzinfo=UTC),
        )
        assert org.settings is None

    def test_settings_none_passes_through(self) -> None:
        org = OrganizationOut(
            id="abc",
            name="Acme",
            slug="acme",
            plan="free",
            settings=None,
            is_active=True,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            updated_at=datetime(2026, 1, 2, tzinfo=UTC),
        )
        assert org.settings is None

    def test_settings_invalid_json_raises(self) -> None:
        with pytest.raises(ValidationError):
            OrganizationOut(
                id="abc",
                name="Acme",
                slug="acme",
                plan="free",
                settings="not-json",  # type: ignore[arg-type]
                is_active=True,
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
                updated_at=datetime(2026, 1, 2, tzinfo=UTC),
            )

    def test_naive_datetimes_get_utc(self) -> None:
        org = OrganizationOut(
            id="abc",
            name="Acme",
            slug="acme",
            plan="free",
            is_active=True,
            created_at=datetime(2026, 1, 1),  # naive
            updated_at=datetime(2026, 1, 2),  # naive
        )
        assert org.created_at.tzinfo is UTC
        assert org.updated_at.tzinfo is UTC

    def test_missing_required_field(self) -> None:
        with pytest.raises(ValidationError):
            OrganizationOut(  # type: ignore[call-arg]
                id="abc",
                slug="acme",
                plan="free",
                is_active=True,
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
                updated_at=datetime(2026, 1, 2, tzinfo=UTC),
            )


# ---------------------------------------------------------------------------
# OrganizationCreate — slug pattern, plan enum, length bounds
# ---------------------------------------------------------------------------


class TestOrganizationCreate:
    def test_valid_minimal(self) -> None:
        # Plan defaults to "free".
        req = OrganizationCreate(name="Acme", slug="acme")
        assert req.name == "Acme"
        assert req.plan == "free"

    def test_all_valid_plans_accepted(self) -> None:
        for plan in ("free", "pro", "team", "enterprise"):
            req = OrganizationCreate(name="A", slug="a", plan=plan)
            assert req.plan == plan

    def test_invalid_plan_rejected(self) -> None:
        with pytest.raises(ValidationError, match="plan must be one of"):
            OrganizationCreate(name="A", slug="a", plan="ultra-platinum")

    def test_slug_pattern_lowercase_hyphens_digits(self) -> None:
        # Valid slugs: lowercase, digits, hyphens (non-leading, non-trailing).
        for slug in ("acme", "acme-corp", "acme-corp-2", "a1b2"):
            assert OrganizationCreate(name="A", slug=slug).slug == slug

    def test_slug_uppercase_rejected(self) -> None:
        with pytest.raises(ValidationError, match="pattern"):
            OrganizationCreate(name="A", slug="Acme")

    def test_slug_underscore_rejected(self) -> None:
        with pytest.raises(ValidationError, match="pattern"):
            OrganizationCreate(name="A", slug="acme_corp")

    def test_slug_leading_hyphen_rejected(self) -> None:
        with pytest.raises(ValidationError, match="pattern"):
            OrganizationCreate(name="A", slug="-acme")

    def test_slug_trailing_hyphen_rejected(self) -> None:
        with pytest.raises(ValidationError, match="pattern"):
            OrganizationCreate(name="A", slug="acme-")

    def test_slug_double_hyphen_rejected(self) -> None:
        with pytest.raises(ValidationError, match="pattern"):
            OrganizationCreate(name="A", slug="acme--corp")

    def test_slug_empty_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OrganizationCreate(name="A", slug="")

    def test_slug_too_long_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at most 64"):
            OrganizationCreate(name="A", slug="a" * 65)

    def test_name_too_long_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at most 256"):
            OrganizationCreate(name="x" * 257, slug="a")

    def test_name_empty_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OrganizationCreate(name="", slug="a")


# ---------------------------------------------------------------------------
# OrganizationUpdate — all-optional with plan validation
# ---------------------------------------------------------------------------


class TestOrganizationUpdate:
    def test_all_fields_optional(self) -> None:
        upd = OrganizationUpdate()
        assert upd.name is None
        assert upd.plan is None
        assert upd.settings is None
        assert upd.is_active is None

    def test_valid_partial_update(self) -> None:
        upd = OrganizationUpdate(plan="enterprise", is_active=False)
        assert upd.plan == "enterprise"
        assert upd.is_active is False

    def test_plan_none_passes_validator(self) -> None:
        """None plan bypasses the enum check — required for partial updates."""
        assert OrganizationUpdate(plan=None).plan is None

    def test_invalid_plan_rejected(self) -> None:
        with pytest.raises(ValidationError, match="plan must be one of"):
            OrganizationUpdate(plan="bogus")

    def test_name_too_short_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OrganizationUpdate(name="")

    def test_name_too_long_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at most 256"):
            OrganizationUpdate(name="x" * 257)


# ---------------------------------------------------------------------------
# OrgSummary / AdminStats / OrgUsage / OrgAuditLog
# ---------------------------------------------------------------------------


class TestSimpleOrgModels:
    def test_org_summary_valid(self) -> None:
        s = OrgSummary(
            id="x",
            name="Acme",
            slug="acme",
            status="active",
            plan="free",
            member_count=5,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        assert s.member_count == 5

    def test_org_summary_naive_datetime_gets_utc(self) -> None:
        s = OrgSummary(
            id="x",
            name="Acme",
            slug="acme",
            status="active",
            plan="free",
            member_count=0,
            created_at=datetime(2026, 1, 1),  # naive
        )
        assert s.created_at.tzinfo is UTC

    def test_admin_stats_valid(self) -> None:
        st = AdminStats(total_orgs=10, active_sessions=100, total_users=42, mcp_server_count=3)
        assert st.total_orgs == 10
        assert st.mcp_server_count == 3

    def test_admin_stats_missing_required(self) -> None:
        with pytest.raises(ValidationError):
            AdminStats(total_orgs=10)  # type: ignore[call-arg]

    def test_org_usage_all_defaults(self) -> None:
        # Positional ``Field(0, ...)`` defaults are runtime-valid Pydantic
        # but pyright doesn't recognise them as defaults — these tests
        # verify the actual runtime behaviour.
        u = OrgUsage(org_id="x")  # type: ignore[call-arg]
        assert u.total_api_calls == 0
        assert u.total_sessions == 0
        assert u.total_users_provisioned == 0
        assert u.total_storage_kb == 0
        assert u.total_workspaces == 0
        assert u.from_date is None
        assert u.to_date is None

    def test_org_usage_populated(self) -> None:
        u = OrgUsage(  # type: ignore[call-arg]
            org_id="x",
            from_date="2026-01-01",
            to_date="2026-01-31",
            total_api_calls=1234,
        )
        assert u.from_date == "2026-01-01"
        assert u.total_api_calls == 1234

    def test_org_audit_log_defaults(self) -> None:
        # ``note: str = Field("", ...)`` is a runtime default for pydantic
        # but pyright treats it as required.
        log = OrgAuditLog()  # type: ignore[call-arg]
        assert log.entries == []
        assert log.note == ""

    def test_org_audit_log_populated(self) -> None:
        entries = [{"action": "login", "user_id": "u1"}]
        log = OrgAuditLog(entries=entries, note="partial coverage")
        assert log.entries == entries
        assert log.note == "partial coverage"


# ---------------------------------------------------------------------------
# Impersonation
# ---------------------------------------------------------------------------


class TestImpersonateRequest:
    def test_valid_with_defaults(self) -> None:
        req = ImpersonateRequest(user_id="u", reason="customer support call")
        assert req.duration_minutes == 30  # default

    def test_duration_at_min_1(self) -> None:
        assert ImpersonateRequest(user_id="u", reason="r", duration_minutes=1).duration_minutes == 1

    def test_duration_at_max_120(self) -> None:
        assert (
            ImpersonateRequest(user_id="u", reason="r", duration_minutes=120).duration_minutes
            == 120
        )

    def test_duration_below_min_rejected(self) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            ImpersonateRequest(user_id="u", reason="r", duration_minutes=0)

    def test_duration_above_max_rejected(self) -> None:
        with pytest.raises(ValidationError, match="less than or equal to 120"):
            ImpersonateRequest(user_id="u", reason="r", duration_minutes=121)

    def test_reason_empty_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ImpersonateRequest(user_id="u", reason="")

    def test_reason_too_long_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at most 512"):
            ImpersonateRequest(user_id="u", reason="x" * 513)

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            ImpersonateRequest(reason="r")  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            ImpersonateRequest(user_id="u")  # type: ignore[call-arg]


class TestImpersonateResponse:
    def test_valid(self) -> None:
        r = ImpersonateResponse(
            impersonation_token="jwt.token.here",
            expires_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            impersonated_user_id="u",
            org_id="o",
        )
        assert r.impersonation_token == "jwt.token.here"

    def test_naive_expires_at_gets_utc(self) -> None:
        r = ImpersonateResponse(
            impersonation_token="jwt",
            expires_at=datetime(2026, 1, 1, 12, 0),  # naive
            impersonated_user_id="u",
            org_id="o",
        )
        assert r.expires_at.tzinfo is UTC


# ---------------------------------------------------------------------------
# AuditLogEntryOut
# ---------------------------------------------------------------------------


class TestAuditLogEntryOut:
    def test_valid_minimal(self) -> None:
        e = AuditLogEntryOut(
            id="e1",
            actor_id="u1",
            action="login",
            resource_type="session",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        assert e.impersonated_by is None
        assert e.resource_id is None
        assert e.details is None

    def test_valid_full_with_impersonation(self) -> None:
        e = AuditLogEntryOut(
            id="e1",
            actor_id="u1",
            impersonated_by="admin1",
            action="impersonation.start",
            resource_type="user",
            resource_id="u2",
            details={"reason": "support"},
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        assert e.impersonated_by == "admin1"
        assert e.details == {"reason": "support"}

    def test_naive_datetime_gets_utc(self) -> None:
        e = AuditLogEntryOut(
            id="e1",
            actor_id="u1",
            action="x",
            resource_type="y",
            created_at=datetime(2026, 1, 1),  # naive
        )
        assert e.created_at.tzinfo is UTC

    def test_missing_required_field(self) -> None:
        with pytest.raises(ValidationError):
            AuditLogEntryOut(  # type: ignore[call-arg]
                id="e1",
                actor_id="u1",
                action="x",
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
