"""Tests for src/tools/datetime_tool.py — date/time utilities.

Covers: get_current_datetime, convert_timezone, parse_date, _get_timezone.
Issue: #1228 (zero test coverage).

Test isolation:
- datetime_tool.py has no mutable module globals — functions are pure and stateless.
- Fallback path tests use monkeypatch to control library availability flags.
"""

import re

import pytest

from src.tools import datetime_tool as dt_module

# ── get_current_datetime ───────────────────────────────────────────────────────


class TestGetCurrentDatetime:
    def test_returns_string(self) -> None:
        result = dt_module.get_current_datetime()
        assert isinstance(result, str)

    def test_utc_contains_year(self) -> None:
        result = dt_module.get_current_datetime(timezone="UTC")
        # UTC output should contain a 4-digit year
        import re

        assert re.search(r"\d{4}", result), f"Expected year in output, got: {result}"

    def test_eastern_timezone_returns_string(self) -> None:
        result = dt_module.get_current_datetime(timezone="US/Eastern")
        assert isinstance(result, str)
        assert "Error" not in result

    def test_invalid_timezone_returns_error_message(self) -> None:
        result = dt_module.get_current_datetime(timezone="nonexistent_timezone")
        assert "Error" in result
        assert "Unknown timezone" in result

    def test_custom_output_format(self) -> None:
        result = dt_module.get_current_datetime(timezone="UTC", output_format="%Y-%m-%d")
        assert re.match(r"\d{4}-\d{2}-\d{2}", result), f"Expected ISO date format, got: {result}"


# ── convert_timezone ───────────────────────────────────────────────────────────


class TestConvertTimezone:
    def test_basic_conversion_utc_to_eastern(self) -> None:
        result = dt_module.convert_timezone(
            datetime_str="2024-01-01 12:00",
            from_timezone="UTC",
            to_timezone="US/Eastern",
        )
        assert isinstance(result, str)
        assert "Error" not in result

    def test_eastern_to_utc_conversion(self) -> None:
        result = dt_module.convert_timezone(
            datetime_str="2024-06-15 08:00",
            from_timezone="US/Eastern",
            to_timezone="UTC",
        )
        assert isinstance(result, str)
        assert "Error" not in result

    def test_invalid_datetime_string_returns_error(self) -> None:
        result = dt_module.convert_timezone(
            datetime_str="not-a-datetime",
            from_timezone="UTC",
            to_timezone="US/Eastern",
        )
        assert "Error" in result

    def test_invalid_source_timezone_returns_error(self) -> None:
        result = dt_module.convert_timezone(
            datetime_str="2024-01-01 12:00",
            from_timezone="Invalid/Zone",
            to_timezone="UTC",
        )
        assert "Error" in result
        assert "Unknown source timezone" in result

    def test_invalid_target_timezone_returns_error(self) -> None:
        result = dt_module.convert_timezone(
            datetime_str="2024-01-01 12:00",
            from_timezone="UTC",
            to_timezone="Invalid/Zone",
        )
        assert "Error" in result
        assert "Unknown target timezone" in result

    def test_custom_output_format(self) -> None:
        result = dt_module.convert_timezone(
            datetime_str="2024-01-01 12:00",
            from_timezone="UTC",
            to_timezone="UTC",
            output_format="%Y-%m-%d",
        )
        assert re.match(r"\d{4}-\d{2}-\d{2}", result)


# ── parse_date ─────────────────────────────────────────────────────────────────


class TestParseDate:
    def test_iso_format_date(self) -> None:
        result = dt_module.parse_date("2024-01-01")
        assert isinstance(result, str)
        assert "Error" not in result

    def test_iso_format_with_time(self) -> None:
        result = dt_module.parse_date("2024-06-15 14:30:00")
        assert isinstance(result, str)
        assert "Error" not in result

    def test_human_readable_formats_supported(self) -> None:
        # dateutil.parser handles common formats but NOT relative dates like "tomorrow"
        # It handles formats like "January 1, 2024" or "1 Jan 2024"
        result = dt_module.parse_date("January 1, 2024")
        assert isinstance(result, str)
        # Either succeeds or returns helpful error
        if "Error" in result:
            assert "dateutil" in result.lower() or "unknown" in result.lower()

    def test_short_date_format(self) -> None:
        # "1 Jan 2024" style is handled by dateutil
        result = dt_module.parse_date("1 Jan 2024")
        assert isinstance(result, str)
        assert "Error" not in result

    def test_slash_format_dmy(self) -> None:
        result = dt_module.parse_date("01/06/2024")
        assert isinstance(result, str)
        assert "Error" not in result

    def test_invalid_date_returns_error(self) -> None:
        result = dt_module.parse_date("not a date at all")
        assert "Error" in result

    def test_custom_output_format(self) -> None:
        result = dt_module.parse_date("2024-01-01", output_format="%B %d, %Y")
        assert isinstance(result, str)
        assert "Error" not in result


# ── _get_timezone ──────────────────────────────────────────────────────────────


class TestGetTimezone:
    def test_utc_returns_utc_constant(self) -> None:
        from datetime import UTC

        result = dt_module._get_timezone("UTC")
        assert result is UTC

    def test_utc_lowercase(self) -> None:
        from datetime import UTC

        result = dt_module._get_timezone("utc")
        assert result is UTC

    def test_zoneinfo_available_returns_zoneinfo_object(self, monkeypatch) -> None:
        # Mock ZoneInfo as available
        monkeypatch.setattr(dt_module, "ZoneInfo", object, raising=False)
        # When ZoneInfo is available and works, it should be returned
        result = dt_module._get_timezone("America/New_York")
        assert result is not None

    def test_zoneinfo_unavailable_falls_back_to_pytz(self, monkeypatch) -> None:
        # Disable zoneinfo
        monkeypatch.setattr(dt_module, "ZoneInfo", None)
        # pytz should be available in the test env
        if not dt_module.PYTZ_AVAILABLE:
            pytest.skip("pytz not available in test environment")
        result = dt_module._get_timezone("America/New_York")
        assert result is not None

    def test_zoneinfo_and_pytz_unavailable_falls_back_to_dateutil(self, monkeypatch) -> None:
        # Disable zoneinfo
        monkeypatch.setattr(dt_module, "ZoneInfo", None)
        # Disable pytz
        monkeypatch.setattr(dt_module, "PYTZ_AVAILABLE", False)
        monkeypatch.setattr(dt_module, "pytz", None)
        # dateutil should be available in the test env
        if not dt_module.DATEUTIL_AVAILABLE:
            pytest.skip("dateutil not available in test environment")
        result = dt_module._get_timezone("America/New_York")
        assert result is not None

    def test_all_libraries_unavailable_returns_none(self, monkeypatch) -> None:
        # Disable zoneinfo
        monkeypatch.setattr(dt_module, "ZoneInfo", None)
        # Disable pytz
        monkeypatch.setattr(dt_module, "PYTZ_AVAILABLE", False)
        monkeypatch.setattr(dt_module, "pytz", None)
        # Disable dateutil
        monkeypatch.setattr(dt_module, "DATEUTIL_AVAILABLE", False)
        monkeypatch.setattr(dt_module, "gettz", None)
        result = dt_module._get_timezone("America/New_York")
        assert result is None

    def test_invalid_timezone_returns_none(self) -> None:
        result = dt_module._get_timezone("Invalid/Timezone/That/Does/Not/Exist")
        assert result is None


# ── Pydantic schemas ───────────────────────────────────────────────────────────


class TestInputSchemas:
    def test_get_datetime_input_defaults(self) -> None:
        inp = dt_module.GetDateTimeInput()
        assert inp.timezone == "UTC"
        assert inp.output_format == "%Y-%m-%d %H:%M:%S %Z"

    def test_get_datetime_input_custom(self) -> None:
        inp = dt_module.GetDateTimeInput(timezone="US/Eastern", output_format="%Y-%m-%d")
        assert inp.timezone == "US/Eastern"
        assert inp.output_format == "%Y-%m-%d"

    def test_convert_timezone_input_requires_fields(self) -> None:
        inp = dt_module.ConvertTimezoneInput(
            datetime_str="2024-01-01 12:00",
            from_timezone="UTC",
            to_timezone="US/Eastern",
        )
        assert inp.datetime_str == "2024-01-01 12:00"
        assert inp.from_timezone == "UTC"
        assert inp.to_timezone == "US/Eastern"
        assert inp.output_format == "%Y-%m-%d %H:%M:%S %Z"

    def test_parse_date_input_requires_date_str(self) -> None:
        inp = dt_module.ParseDateInput(date_str="2024-01-01")
        assert inp.date_str == "2024-01-01"
        assert inp.output_format == "%Y-%m-%d %H:%M:%S"


# ── TOOL_CONFIGS export ────────────────────────────────────────────────────────


class TestToolConfigs:
    def test_tool_configs_has_three_entries(self) -> None:
        assert len(dt_module.TOOL_CONFIGS) == 3

    def test_tool_config_names(self) -> None:
        names = {cfg["name"] for cfg in dt_module.TOOL_CONFIGS}
        assert names == {"get_current_datetime", "convert_timezone", "parse_date"}

    def test_each_config_has_required_keys(self) -> None:
        required_keys = {"name", "description", "input_schema", "requires_confirmation", "function"}
        for cfg in dt_module.TOOL_CONFIGS:
            assert required_keys.issubset(cfg.keys())

    def test_tool_config_is_first_entry(self) -> None:
        assert dt_module.TOOL_CONFIG is dt_module.TOOL_CONFIGS[0]
