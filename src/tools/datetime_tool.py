"""
Date and time utilities tool.
Provides current date/time, timezone conversions, and date parsing.
"""

from datetime import UTC, datetime

from pydantic import BaseModel, Field

# Try to import zoneinfo (Python 3.9+) or fall back to pytz
try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None  # type: ignore[misc, assignment]

try:
    import pytz  # type: ignore[import-untyped]

    PYTZ_AVAILABLE = True
except ImportError:
    pytz = None  # type: ignore[assignment]
    PYTZ_AVAILABLE = False

try:
    from dateutil import parser as date_parser  # type: ignore[import-untyped]
    from dateutil.tz import gettz  # type: ignore[import-untyped]

    DATEUTIL_AVAILABLE = True
except ImportError:
    date_parser = None  # type: ignore[assignment]
    gettz = None  # type: ignore[assignment]
    DATEUTIL_AVAILABLE = False


class GetDateTimeInput(BaseModel):
    """Input schema for getting current date/time."""

    timezone: str = Field(
        default="UTC",
        description="Timezone name (e.g., 'UTC', 'US/Eastern', 'Europe/London', 'Asia/Tokyo')",
    )
    output_format: str = Field(
        default="%Y-%m-%d %H:%M:%S %Z",
        description="Output format (strftime format string)",
    )


class ConvertTimezoneInput(BaseModel):
    """Input schema for timezone conversion."""

    datetime_str: str = Field(description="The datetime string to convert")
    from_timezone: str = Field(description="Source timezone (e.g., 'UTC', 'US/Eastern')")
    to_timezone: str = Field(description="Target timezone (e.g., 'Europe/London')")
    output_format: str = Field(
        default="%Y-%m-%d %H:%M:%S %Z",
        description="Output format (strftime format string)",
    )


class ParseDateInput(BaseModel):
    """Input schema for parsing dates."""

    date_str: str = Field(description="The date string to parse (various formats supported)")
    output_format: str = Field(
        default="%Y-%m-%d %H:%M:%S",
        description="Output format (strftime format string)",
    )


def _get_timezone(tz_name: str):
    """Get timezone object by name."""
    if tz_name.upper() == "UTC":
        return UTC

    # Try zoneinfo first (Python 3.9+)
    if ZoneInfo is not None:
        try:
            return ZoneInfo(tz_name)
        except Exception:  # noqa: BLE001 — fallback chain; try next tz library  # nosec B110
            pass

    # Fall back to pytz
    if PYTZ_AVAILABLE and pytz is not None:
        try:
            return pytz.timezone(tz_name)
        except Exception:  # noqa: BLE001 — fallback chain; try next tz library  # nosec B110
            pass

    # Fall back to dateutil
    if DATEUTIL_AVAILABLE and gettz is not None:
        tz = gettz(tz_name)
        if tz is not None:
            return tz

    return None


def get_current_datetime(timezone: str = "UTC", output_format: str = "%Y-%m-%d %H:%M:%S %Z") -> str:
    """
    Get the current date and time in the specified timezone.

    Args:
        timezone: Timezone name (e.g., 'UTC', 'US/Eastern', 'Europe/London')
        output_format: Output format using strftime format codes

    Returns:
        Formatted current date/time string
    """
    try:
        tz = _get_timezone(timezone)
        if tz is None:
            return (
                f"Error: Unknown timezone '{timezone}'. "
                "Use formats like 'UTC', 'US/Eastern', 'Europe/London'."
            )

        now = datetime.now(tz)
        return now.strftime(output_format)

    except Exception as e:
        return f"Error getting current datetime: {e}"


def convert_timezone(
    datetime_str: str,
    from_timezone: str,
    to_timezone: str,
    output_format: str = "%Y-%m-%d %H:%M:%S %Z",
) -> str:
    """
    Convert a datetime from one timezone to another.

    Args:
        datetime_str: The datetime string to convert
        from_timezone: Source timezone
        to_timezone: Target timezone
        output_format: Output format using strftime format codes

    Returns:
        Converted datetime string
    """
    try:
        from_tz = _get_timezone(from_timezone)
        to_tz = _get_timezone(to_timezone)

        if from_tz is None:
            return f"Error: Unknown source timezone '{from_timezone}'"
        if to_tz is None:
            return f"Error: Unknown target timezone '{to_timezone}'"

        # Parse the datetime string
        if DATEUTIL_AVAILABLE and date_parser is not None:
            dt = date_parser.parse(datetime_str)
        else:
            # Try common formats
            for fmt in [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%Y-%m-%d",
                "%d/%m/%Y %H:%M:%S",
                "%m/%d/%Y %H:%M:%S",
            ]:
                try:
                    dt = datetime.strptime(datetime_str, fmt)
                    break
                except ValueError:
                    continue
            else:
                return f"Error: Could not parse datetime '{datetime_str}'"

        # Localize to source timezone if naive
        if dt.tzinfo is None:
            if PYTZ_AVAILABLE and hasattr(from_tz, "localize"):
                dt = from_tz.localize(dt)
            else:
                dt = dt.replace(tzinfo=from_tz)

        # Convert to target timezone
        converted = dt.astimezone(to_tz)
        return converted.strftime(output_format)

    except Exception as e:
        return f"Error converting timezone: {e}"


def parse_date(date_str: str, output_format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Parse a date string in various formats and return it in a standardized format.

    Args:
        date_str: The date string to parse (supports many formats)
        output_format: Output format using strftime format codes

    Returns:
        Parsed and formatted date string
    """
    try:
        if DATEUTIL_AVAILABLE and date_parser is not None:
            dt = date_parser.parse(date_str)
            return dt.strftime(output_format)
        else:
            # Try common formats without dateutil
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%Y-%m-%d",
                "%d/%m/%Y %H:%M:%S",
                "%d/%m/%Y",
                "%m/%d/%Y %H:%M:%S",
                "%m/%d/%Y",
                "%B %d, %Y",
                "%b %d, %Y",
                "%d %B %Y",
                "%d %b %Y",
            ]
            for fmt in formats:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.strftime(output_format)
                except ValueError:
                    continue

            return (
                f"Error: Could not parse date '{date_str}'. "
                "Install python-dateutil for better parsing."
            )

    except Exception as e:
        return f"Error parsing date: {e}"


# Export individual tool configs for the registry
TOOL_CONFIGS = [
    {
        "name": "get_current_datetime",
        "description": (
            "Get the current date and time in a specified timezone. "
            "Useful for checking the current time in different locations."
        ),
        "input_schema": GetDateTimeInput,
        "requires_confirmation": False,
        "function": get_current_datetime,
    },
    {
        "name": "convert_timezone",
        "description": (
            "Convert a datetime from one timezone to another. "
            "Useful for scheduling across timezones."
        ),
        "input_schema": ConvertTimezoneInput,
        "requires_confirmation": False,
        "function": convert_timezone,
    },
    {
        "name": "parse_date",
        "description": (
            "Parse a date string in various formats and return it in a standardized format."
        ),
        "input_schema": ParseDateInput,
        "requires_confirmation": False,
        "function": parse_date,
    },
]

# Default single tool config (for backwards compatibility)
TOOL_CONFIG = TOOL_CONFIGS[0]

__all__ = [
    "get_current_datetime",
    "convert_timezone",
    "parse_date",
    "GetDateTimeInput",
    "ConvertTimezoneInput",
    "ParseDateInput",
    "TOOL_CONFIG",
    "TOOL_CONFIGS",
]
