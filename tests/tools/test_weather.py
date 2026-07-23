"""Tests for the weather tool.

The weather tool integrates with the OpenWeatherMap API. Incorrect error
handling or parsing would produce garbled weather data fed to the LLM.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.tools.weather import (
    TOOL_CONFIG,
    WeatherQueryInput,
    _get_api_key,
    get_weather,
    is_configured,
)


class TestGetApiKey:
    """Tests for _get_api_key()."""

    def test_env_var_takes_priority(self):
        """Environment variable OPENWEATHER_API_KEY is checked first."""
        with patch.dict("os.environ", {"OPENWEATHER_API_KEY": "env-key-123"}, clear=False):
            assert _get_api_key() == "env-key-123"

    def test_returns_none_when_not_set(self):
        """Returns None when no API key is configured."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("src.config.load_config", side_effect=ImportError):
                assert _get_api_key() is None

    def test_config_fallback(self):
        """Falls back to config file when env var is not set."""
        mock_config = MagicMock()
        mock_config.openweather_api_key = "config-key-456"

        with patch.dict("os.environ", {}, clear=True):
            with patch("src.config.load_config", return_value=mock_config):
                assert _get_api_key() == "config-key-456"

    def test_injected_config_takes_priority(self):
        """The TOOL_SETUP-injected key wins over the resolved config (#2101: the
        env-over-config-file precedence now lives in config resolution, not in a
        separate os.getenv step inside the tool)."""
        from src.tools.weather import _weather_config

        mock_config = MagicMock()
        mock_config.openweather_api_key = "cached-key"

        _weather_config["api_key"] = "injected-key"
        try:
            with patch("src.config.get_cached_config", return_value=mock_config):
                assert _get_api_key() == "injected-key"
        finally:
            _weather_config.pop("api_key", None)

    def test_falls_back_to_cached_config(self):
        """With no injected key, the tool returns the cached config's value."""
        from src.tools.weather import _weather_config

        mock_config = MagicMock()
        mock_config.openweather_api_key = "cached-key"

        _weather_config.pop("api_key", None)
        with patch("src.config.get_cached_config", return_value=mock_config):
            assert _get_api_key() == "cached-key"


class TestGetWeatherNoKey:
    """Tests for get_weather() when API key is not configured."""

    def test_returns_setup_instructions_when_no_key(self):
        """When no API key, returns helpful setup instructions."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("src.config.load_config", side_effect=ImportError):
                result = get_weather("London")

        assert "Weather API not configured" in result
        assert "openweathermap.org/api" in result
        assert "OPENWEATHER_API_KEY" in result
        assert "London" in result

    def test_empty_location_returns_error(self):
        """Empty location string returns an error message."""
        result = get_weather("   ")
        assert "Error: Please provide a location" in result

    def test_empty_location_with_whitespace(self):
        """Whitespace-only location is treated as empty."""
        result = get_weather("  \t\n  ")
        assert "Error: Please provide a location" in result


class TestGetWeatherHappyPath:
    """Tests for get_weather() with mocked successful API responses."""

    def _make_response(self, status_code: int = 200, json_data: dict | None = None):
        """Helper to create a mock requests.Response."""
        response = MagicMock()
        response.status_code = status_code
        response.json.return_value = json_data or {}
        return response

    def _default_weather_json(self) -> dict:
        """Return a realistic OpenWeatherMap API response."""
        return {
            "name": "London",
            "sys": {"country": "GB"},
            "main": {
                "temp": 15.5,
                "feels_like": 14.2,
                "humidity": 72,
                "pressure": 1013,
            },
            "weather": [{"description": "overcast clouds"}],
            "wind": {"speed": 3.5},
            "clouds": {"all": 85},
            "visibility": 10000,
        }

    def test_happy_path_metric(self):
        """Successful API call returns structured weather data (metric)."""
        with patch.dict("os.environ", {"OPENWEATHER_API_KEY": "test-key"}, clear=False):
            response = self._make_response(200, self._default_weather_json())
            with patch("src.tools.weather.requests.get", return_value=response):
                result = get_weather("London", units="metric")

        assert "Weather in London, GB:" in result
        assert "Condition: Overcast clouds" in result
        assert "Temperature: 15.5°C" in result
        assert "Feels like: 14.2°C" in result
        assert "Humidity: 72%" in result
        assert "Pressure: 1013 hPa" in result
        assert "Wind: 3.5 m/s" in result
        assert "Cloudiness: 85%" in result
        assert "Visibility: 10.0 km" in result

    def test_happy_path_imperial(self):
        """Successful API call with imperial units."""
        data = self._default_weather_json()
        with patch.dict("os.environ", {"OPENWEATHER_API_KEY": "test-key"}, clear=False):
            response = self._make_response(200, data)
            with patch("src.tools.weather.requests.get", return_value=response):
                result = get_weather("New York", units="imperial")

        assert "°F" in result
        assert "mph" in result

    def test_happy_path_kelvin(self):
        """Successful API call with kelvin units."""
        data = self._default_weather_json()
        with patch.dict("os.environ", {"OPENWEATHER_API_KEY": "test-key"}, clear=False):
            response = self._make_response(200, data)
            with patch("src.tools.weather.requests.get", return_value=response):
                result = get_weather("Tokyo", units="kelvin")

        assert "Temperature: 15.5K" in result
        # Kelvin uses m/s for wind
        assert "Wind: 3.5 m/s" in result

    def test_location_without_country(self):
        """Response without country info still formats correctly."""
        data = self._default_weather_json()
        data["sys"] = {}
        with patch.dict("os.environ", {"OPENWEATHER_API_KEY": "test-key"}, clear=False):
            response = self._make_response(200, data)
            with patch("src.tools.weather.requests.get", return_value=response):
                result = get_weather("London")

        assert "Weather in London:" in result
        assert ", GB" not in result

    def test_no_visibility_field(self):
        """When visibility is missing, it is omitted from output."""
        data = self._default_weather_json()
        del data["visibility"]
        with patch.dict("os.environ", {"OPENWEATHER_API_KEY": "test-key"}, clear=False):
            response = self._make_response(200, data)
            with patch("src.tools.weather.requests.get", return_value=response):
                result = get_weather("London")

        assert "Visibility:" not in result

    def test_missing_weather_description(self):
        """When weather description is missing, 'Unknown' is used."""
        data = self._default_weather_json()
        data["weather"] = [{}]
        with patch.dict("os.environ", {"OPENWEATHER_API_KEY": "test-key"}, clear=False):
            response = self._make_response(200, data)
            with patch("src.tools.weather.requests.get", return_value=response):
                result = get_weather("London")

        assert "Condition: Unknown" in result


class TestGetWeatherErrorResponses:
    """Tests for get_weather() with API error responses."""

    def _make_response(self, status_code: int = 200):
        response = MagicMock()
        response.status_code = status_code
        response.json.return_value = {}
        return response

    def test_401_invalid_api_key(self):
        """401 status returns invalid API key message."""
        with patch.dict("os.environ", {"OPENWEATHER_API_KEY": "bad-key"}, clear=False):
            response = self._make_response(401)
            with patch("src.tools.weather.requests.get", return_value=response):
                result = get_weather("London")

        assert "Invalid API key" in result

    def test_404_location_not_found(self):
        """404 status returns location not found message."""
        with patch.dict("os.environ", {"OPENWEATHER_API_KEY": "test-key"}, clear=False):
            response = self._make_response(404)
            with patch("src.tools.weather.requests.get", return_value=response):
                result = get_weather("NonExistentCity12345")

        assert "Location not found" in result
        assert "NonExistentCity12345" in result

    def test_429_rate_limited(self):
        """429 status returns generic API error with status code."""
        with patch.dict("os.environ", {"OPENWEATHER_API_KEY": "test-key"}, clear=False):
            response = self._make_response(429)
            with patch("src.tools.weather.requests.get", return_value=response):
                result = get_weather("London")

        assert "API returned status 429" in result

    def test_500_server_error(self):
        """500 status returns generic API error."""
        with patch.dict("os.environ", {"OPENWEATHER_API_KEY": "test-key"}, clear=False):
            response = self._make_response(500)
            with patch("src.tools.weather.requests.get", return_value=response):
                result = get_weather("London")

        assert "API returned status 500" in result


class TestGetWeatherRequestExceptions:
    """Tests for get_weather() network exception handling."""

    def test_timeout(self):
        """Timeout exception returns a user-friendly message."""
        import requests

        with patch.dict("os.environ", {"OPENWEATHER_API_KEY": "test-key"}, clear=False):
            with patch(
                "src.tools.weather.requests.get",
                side_effect=requests.exceptions.Timeout,
            ):
                result = get_weather("London")

        assert "timed out" in result

    def test_connection_error(self):
        """ConnectionError returns a user-friendly message."""
        import requests

        with patch.dict("os.environ", {"OPENWEATHER_API_KEY": "test-key"}, clear=False):
            with patch(
                "src.tools.weather.requests.get",
                side_effect=requests.exceptions.ConnectionError,
            ):
                result = get_weather("London")

        assert "Could not connect" in result

    def test_generic_request_exception(self):
        """Generic RequestException is sanitized to a safe message."""
        import requests

        with patch.dict("os.environ", {"OPENWEATHER_API_KEY": "test-key"}, clear=False):
            with patch(
                "src.tools.weather.requests.get",
                side_effect=requests.exceptions.RequestException("DNS failure"),
            ):
                result = get_weather("London")

        assert "Weather API request failed" in result
        assert "Request failed" in result

    def test_unexpected_exception(self):
        """Unexpected exceptions are caught and returned as sanitized error string."""
        with patch.dict("os.environ", {"OPENWEATHER_API_KEY": "test-key"}, clear=False):
            with patch(
                "src.tools.weather.requests.get",
                side_effect=RuntimeError("unexpected"),
            ):
                result = get_weather("London")

        assert "Error getting weather" in result
        assert "Operation failed" in result


class TestGetWeatherUnitNormalization:
    """Tests for unit string normalization."""

    def test_celsius_alias(self):
        """'celsius' and 'c' are normalized to 'metric'."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "name": "Paris",
            "sys": {},
            "main": {"temp": 20},
            "weather": [{"description": "clear"}],
            "wind": {},
            "clouds": {},
        }

        with patch.dict("os.environ", {"OPENWEATHER_API_KEY": "test-key"}, clear=False):
            with patch("src.tools.weather.requests.get", return_value=mock_response):
                result_c = get_weather("Paris", units="c")
                result_celsius = get_weather("Paris", units="celsius")

        assert "°C" in result_c
        assert "°C" in result_celsius

    def test_fahrenheit_alias(self):
        """'fahrenheit' and 'f' are normalized to 'imperial'."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "name": "NYC",
            "sys": {},
            "main": {"temp": 70},
            "weather": [{"description": "sunny"}],
            "wind": {},
            "clouds": {},
        }

        with patch.dict("os.environ", {"OPENWEATHER_API_KEY": "test-key"}, clear=False):
            with patch("src.tools.weather.requests.get", return_value=mock_response):
                result_f = get_weather("NYC", units="f")
                result_fahrenheit = get_weather("NYC", units="fahrenheit")

        assert "°F" in result_f
        assert "°F" in result_fahrenheit

    def test_invalid_unit_defaults_to_metric(self):
        """Invalid unit strings default to metric."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "name": "Berlin",
            "sys": {},
            "main": {"temp": 10},
            "weather": [{"description": "cloudy"}],
            "wind": {},
            "clouds": {},
        }

        with patch.dict("os.environ", {"OPENWEATHER_API_KEY": "test-key"}, clear=False):
            with patch("src.tools.weather.requests.get", return_value=mock_response):
                result = get_weather("Berlin", units="invalid_unit")

        assert "°C" in result


class TestIsConfigured:
    """Tests for is_configured()."""

    def test_false_when_no_api_key(self):
        """is_configured() returns False when API key is missing."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("src.config.load_config", side_effect=ImportError):
                assert is_configured() is False

    def test_true_when_api_key_set(self):
        """is_configured() returns True when API key is present."""
        with patch.dict("os.environ", {"OPENWEATHER_API_KEY": "test-key"}, clear=False):
            assert is_configured() is True

    def test_false_when_requests_not_available(self):
        """is_configured() returns False when requests library is missing."""
        from src.tools import weather as weather_mod

        original = weather_mod.REQUESTS_AVAILABLE
        try:
            weather_mod.REQUESTS_AVAILABLE = False
            with patch.dict("os.environ", {"OPENWEATHER_API_KEY": "test-key"}, clear=False):
                assert is_configured() is False
        finally:
            weather_mod.REQUESTS_AVAILABLE = original


class TestWeatherQueryInput:
    """Tests for the Pydantic input schema."""

    def test_default_units(self):
        """Default units is metric."""
        schema = WeatherQueryInput(location="London")
        assert schema.location == "London"
        assert schema.units == "metric"

    def test_custom_units(self):
        """Custom units can be set."""
        schema = WeatherQueryInput(location="NYC", units="imperial")
        assert schema.units == "imperial"

    def test_location_required(self):
        """Location is a required field."""
        with pytest.raises(ValueError):
            WeatherQueryInput.model_validate({})


class TestToolConfig:
    """Tests for the tool metadata registry entry."""

    def test_tool_config_structure(self):
        """TOOL_CONFIG has the expected fields."""
        assert TOOL_CONFIG["name"] == "get_weather"
        assert TOOL_CONFIG["requires_confirmation"] is False
        assert TOOL_CONFIG["input_schema"] == WeatherQueryInput
        assert "weather" in TOOL_CONFIG["description"].lower()
        assert "temperature" in TOOL_CONFIG["description"].lower()
