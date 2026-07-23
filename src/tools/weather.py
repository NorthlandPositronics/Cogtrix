"""
Weather tool — get current weather information.

Uses OpenWeatherMap API.  Automatically removed from the agent's
toolbox when no API key is configured.

Configuration:
    Environment variable: ``OPENWEATHER_API_KEY``
    Config file:          ``services.openweather.api_key``
                          (legacy: ``openweather.api_key`` at top level)
    Free tier:            60 calls / minute, 1 000 000 calls / month

TOOL_SETUP(config) is called automatically by ToolRegistry after this module
is imported.  It captures the OpenWeather API key from the app ``Config``
(which ``_apply_env_vars`` populates before the env var is unset) into
module-level state so the key is available after ``OPENWEATHER_API_KEY`` has
been removed from ``os.environ`` (#2223 phase 2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.config import Config

# Try to import requests
try:
    import requests  # type: ignore[import-untyped]

    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None  # type: ignore[assignment]
    REQUESTS_AVAILABLE = False

from src.tools.error_sanitizer import sanitize_error as _sanitize_error

OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

# ── Module-level state (set by TOOL_SETUP) ────────────────────────────────────
_weather_config: dict[str, str] = {}


def TOOL_SETUP(config: Config) -> None:
    """Called automatically by ToolRegistry after loading this module."""
    key = getattr(config, "openweather_api_key", None)
    if key:
        _weather_config["api_key"] = key


def _get_api_key() -> str | None:
    """Get OpenWeather API key: injected config first, then the cached config."""
    # Config injection (populated by TOOL_SETUP before env var is unset)
    key = _weather_config.get("api_key")
    if key:
        return key

    # #2101: fall back to the process-wide resolved config — the env is read
    # once and the key survives the #2223/#2102 unset via the secret cache, so no
    # per-call os.environ re-read is needed.
    try:
        from src.config import get_cached_config

        return get_cached_config().openweather_api_key
    except Exception:  # noqa: BLE001 — never let config failure break the tool
        return None


class WeatherQueryInput(BaseModel):
    """Input schema for weather queries."""

    location: str = Field(
        description=(
            "The city or location to get weather for (e.g., 'New York', 'London', 'Tokyo, JP')"
        )
    )
    units: str = Field(
        default="metric",
        description="Temperature units: 'metric' (Celsius), 'imperial' (Fahrenheit), or 'kelvin'",
    )


def get_weather(location: str, units: str = "metric") -> str:
    """
    Get current weather information for a location.

    Uses OpenWeatherMap API if configured via environment variable or .cogtrix.json.
    Otherwise returns a helpful message about how to enable real weather data.

    Args:
        location: The city or location name (e.g., 'London', 'New York, US')
        units: Temperature units - 'metric' (Celsius), 'imperial' (Fahrenheit), 'kelvin'

    Returns:
        Weather information string
    """
    if not location.strip():
        return "Error: Please provide a location"

    # Normalize units
    units = units.lower()
    if units in ("celsius", "c"):
        units = "metric"
    elif units in ("fahrenheit", "f"):
        units = "imperial"
    elif units not in ("metric", "imperial", "kelvin"):
        units = "metric"

    # Get API key from environment or config
    api_key = _get_api_key()

    # Check if API is available
    if not api_key:
        return (
            f"Weather API not configured.\n\n"
            f"To enable real weather data:\n"
            f"1. Get a free API key from https://openweathermap.org/api\n"
            f"2. Set the environment variable: OPENWEATHER_API_KEY=your_key\n"
            f"   Or add to your .cogtrix.json config file.\n\n"
            f"Requested location: {location}"
        )

    if not REQUESTS_AVAILABLE:
        return "Error: requests library not available. Run: uv add requests"

    try:
        # Make API request
        params = {
            "q": location,
            "appid": api_key,
            "units": units,
        }

        response = requests.get(OPENWEATHER_BASE_URL, params=params, timeout=10)

        if response.status_code == 401:
            return "Error: Invalid API key. Please check your OPENWEATHER_API_KEY."

        if response.status_code == 404:
            return f"Error: Location not found: {location}"

        if response.status_code != 200:
            return f"Error: API returned status {response.status_code}"

        data = response.json()

        # Extract weather information
        main = data.get("main", {})
        weather = data.get("weather", [{}])[0]
        wind = data.get("wind", {})
        clouds = data.get("clouds", {})
        sys_info = data.get("sys", {})

        # Determine temperature unit symbol
        unit_symbol = "°C" if units == "metric" else "°F" if units == "imperial" else "K"
        speed_unit = "m/s" if units == "metric" else "mph" if units == "imperial" else "m/s"

        # Build response
        city_name = data.get("name", location)
        country = sys_info.get("country", "")
        location_str = f"{city_name}, {country}" if country else city_name

        result = [
            f"Weather in {location_str}:",
            "",
            f"Condition: {weather.get('description', 'Unknown').capitalize()}",
            f"Temperature: {main.get('temp', 'N/A')}{unit_symbol}",
            f"Feels like: {main.get('feels_like', 'N/A')}{unit_symbol}",
            f"Humidity: {main.get('humidity', 'N/A')}%",
            f"Pressure: {main.get('pressure', 'N/A')} hPa",
            f"Wind: {wind.get('speed', 'N/A')} {speed_unit}",
            f"Cloudiness: {clouds.get('all', 'N/A')}%",
        ]

        # Add visibility if available
        visibility = data.get("visibility")
        if visibility:
            result.append(f"Visibility: {visibility / 1000:.1f} km")

        return "\n".join(result)

    except requests.exceptions.Timeout:
        return "Error: Weather API request timed out"
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to weather API"
    except requests.exceptions.RequestException as e:
        return f"Error: Weather API request failed: {_sanitize_error(e)}"
    except Exception as e:
        return f"Error getting weather: {_sanitize_error(e)}"


# Tool metadata for registry
TOOL_CONFIG = {
    "name": "get_weather",
    "description": (
        "Get current weather information for a specified location. "
        "Returns temperature, humidity, wind, and conditions. "
        "Requires OPENWEATHER_API_KEY environment variable for real data."
    ),
    "input_schema": WeatherQueryInput,
    "requires_confirmation": False,
}


def is_configured() -> bool:
    """Return True if the tool has the required API key."""
    return REQUESTS_AVAILABLE and bool(_get_api_key())


__all__ = ["get_weather", "WeatherQueryInput", "TOOL_CONFIG", "TOOL_SETUP", "is_configured"]
