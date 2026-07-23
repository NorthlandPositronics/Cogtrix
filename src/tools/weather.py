"""
Weather tool — get current weather information.

Uses OpenWeatherMap API.  Automatically removed from the agent's
toolbox when no API key is configured.

Configuration:
    Environment variable: ``OPENWEATHER_API_KEY``
    Config file:          ``services.openweather.api_key``
                          (legacy: ``openweather.api_key`` at top level)
    Free tier:            60 calls / minute, 1 000 000 calls / month
"""

import os

from pydantic import BaseModel, Field

# Try to import requests
try:
    import requests  # type: ignore[import-untyped]

    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None  # type: ignore[assignment]
    REQUESTS_AVAILABLE = False


OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"


def _get_api_key() -> str | None:
    """Get OpenWeather API key from environment or config."""
    # First check environment variable
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if api_key:
        return api_key

    # Try to load from config
    try:
        from src.config import load_config

        config = load_config()
        if config.openweather_api_key:
            return config.openweather_api_key
    except ImportError:
        pass

    return None


class WeatherQueryInput(BaseModel):
    """Input schema for weather queries."""

    location: str = Field(
        description=(
            "The city or location to get weather for " "(e.g., 'New York', 'London', 'Tokyo, JP')"
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
        return f"Error: Weather API request failed - {e}"
    except Exception as e:
        return f"Error getting weather: {e}"


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


__all__ = ["get_weather", "WeatherQueryInput", "TOOL_CONFIG", "is_configured"]
