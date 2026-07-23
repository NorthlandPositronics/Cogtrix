"""Tests for cogtrix_core/api/schemas/plan.py — subscription plans."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from cogtrix_core.api.schemas.plan import PlanCreate, PlanLimits, PlanOut, PlanUpdate

# ---------------------------------------------------------------------------
# PlanLimits — quantitative limits, 0 == unlimited
# ---------------------------------------------------------------------------


class TestPlanLimits:
    def test_default_all_zero(self) -> None:
        lim = PlanLimits()
        assert lim.max_users == 0
        assert lim.max_workspaces == 0
        assert lim.max_api_calls_per_month == 0
        assert lim.max_storage_gb == 0

    def test_explicit_values(self) -> None:
        lim = PlanLimits(
            max_users=100,
            max_workspaces=20,
            max_api_calls_per_month=1_000_000,
            max_storage_gb=500,
        )
        assert lim.max_users == 100
        assert lim.max_storage_gb == 500

    def test_partial_values(self) -> None:
        lim = PlanLimits(max_users=10)
        assert lim.max_users == 10
        # Other fields default to 0 (unlimited).
        assert lim.max_storage_gb == 0


# ---------------------------------------------------------------------------
# PlanOut — limits as PlanLimits OR JSON string OR dict
# ---------------------------------------------------------------------------


class TestPlanOut:
    def _base_kwargs(self) -> dict:
        return {
            "id": "p1",
            "name": "Pro",
            "slug": "pro",
            "price_monthly_cents": 1500,
            "price_annual_cents": 15000,
            "limits": PlanLimits(max_users=10),
            "is_active": True,
            "is_public": True,
            "created_at": datetime(2026, 1, 1, tzinfo=UTC),
        }

    def test_valid_with_plan_limits_instance(self) -> None:
        out = PlanOut(**self._base_kwargs())
        assert out.limits.max_users == 10
        assert out.description is None

    def test_limits_from_dict(self) -> None:
        kw = self._base_kwargs()
        kw["limits"] = {"max_users": 5, "max_storage_gb": 100}
        out = PlanOut(**kw)
        assert out.limits.max_users == 5
        assert out.limits.max_storage_gb == 100

    def test_limits_from_json_string(self) -> None:
        """SQLite-stored JSON string is parsed by the validator."""
        kw = self._base_kwargs()
        kw["limits"] = '{"max_users":7,"max_workspaces":3}'
        out = PlanOut(**kw)
        assert out.limits.max_users == 7
        assert out.limits.max_workspaces == 3

    def test_limits_invalid_json_falls_back_to_empty(self) -> None:
        """Validator swallows JSON-decode errors and yields empty PlanLimits.
        Pinning the silent-fallback contract so a stricter future version
        regresses visibly."""
        kw = self._base_kwargs()
        kw["limits"] = "not-json{"
        out = PlanOut(**kw)
        assert out.limits.max_users == 0  # default
        assert out.limits.max_storage_gb == 0

    def test_limits_unknown_type_falls_back_to_empty(self) -> None:
        kw = self._base_kwargs()
        kw["limits"] = 42  # neither PlanLimits, str, nor dict
        out = PlanOut(**kw)
        assert out.limits.max_users == 0

    def test_naive_created_at_gets_utc(self) -> None:
        kw = self._base_kwargs()
        kw["created_at"] = datetime(2026, 1, 1)  # naive
        out = PlanOut(**kw)
        assert out.created_at.tzinfo is UTC

    def test_missing_required_field(self) -> None:
        kw = self._base_kwargs()
        kw.pop("price_monthly_cents")
        with pytest.raises(ValidationError):
            PlanOut(**kw)


# ---------------------------------------------------------------------------
# PlanCreate — slug pattern + price bounds
# ---------------------------------------------------------------------------


class TestPlanCreate:
    def test_valid_minimal(self) -> None:
        p = PlanCreate(name="Pro", slug="pro")
        assert p.description is None
        assert p.price_monthly_cents == 0  # default
        assert p.is_public is True

    def test_valid_full(self) -> None:
        p = PlanCreate(
            name="Pro",
            slug="pro",
            description="The pro plan",
            price_monthly_cents=1500,
            price_annual_cents=15000,
            limits=PlanLimits(max_users=10),
            is_public=False,
        )
        assert p.description == "The pro plan"
        assert p.is_public is False

    def test_name_too_long_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at most 64"):
            PlanCreate(name="x" * 65, slug="x")

    def test_name_empty_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PlanCreate(name="", slug="x")

    def test_slug_pattern_valid_cases(self) -> None:
        for slug in ("a", "pro", "pro-1", "pro-tier-2"):
            assert PlanCreate(name="x", slug=slug).slug == slug

    def test_slug_uppercase_rejected(self) -> None:
        with pytest.raises(ValidationError, match="pattern"):
            PlanCreate(name="x", slug="Pro")

    def test_slug_too_long_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at most 32"):
            PlanCreate(name="x", slug="a" * 33)

    def test_negative_price_rejected(self) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            PlanCreate(name="x", slug="x", price_monthly_cents=-1)

    def test_negative_annual_price_rejected(self) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            PlanCreate(name="x", slug="x", price_annual_cents=-1)

    def test_default_factory_limits(self) -> None:
        p = PlanCreate(name="x", slug="x")
        # default_factory=PlanLimits gives a fresh instance with all-zero fields.
        assert p.limits.max_users == 0


# ---------------------------------------------------------------------------
# PlanUpdate — all-optional
# ---------------------------------------------------------------------------


class TestPlanUpdate:
    def test_empty(self) -> None:
        u = PlanUpdate()
        assert u.name is None
        assert u.description is None
        assert u.price_monthly_cents is None
        assert u.price_annual_cents is None
        assert u.limits is None
        assert u.is_active is None
        assert u.is_public is None

    def test_partial(self) -> None:
        u = PlanUpdate(price_monthly_cents=2000, is_active=False)
        assert u.price_monthly_cents == 2000
        assert u.is_active is False

    def test_name_too_long_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at most 64"):
            PlanUpdate(name="x" * 65)

    def test_name_empty_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PlanUpdate(name="")

    def test_negative_price_rejected(self) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            PlanUpdate(price_monthly_cents=-1)
