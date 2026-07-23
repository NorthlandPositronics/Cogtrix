"""Tests for the Google Search tool."""

from unittest.mock import MagicMock, patch

import requests


class TestGoogleSearch:
    """Unit tests for google_search()."""

    def test_google_search_missing_api_key(self):
        from src.tools.google_search import google_search

        with patch("src.tools.google_search._get_api_key", return_value=None):
            result = google_search("python")

        assert "Error" in result
        assert "Google API key" in result

    def test_google_search_missing_cse_id(self):
        from src.tools.google_search import google_search

        with (
            patch("src.tools.google_search._get_api_key", return_value="test-key"),
            patch("src.tools.google_search._get_cse_id", return_value=None),
        ):
            result = google_search("python")

        assert "Error" in result
        assert "Custom Search Engine ID" in result

    def test_google_search_empty_query_returns_error(self):
        from src.tools.google_search import google_search

        with (
            patch("src.tools.google_search._get_api_key", return_value="test-key"),
            patch("src.tools.google_search._get_cse_id", return_value="test-cx"),
        ):
            result = google_search("   ")

        assert "Error" in result
        assert "Empty" in result

    def test_google_search_returns_results(self):
        from src.tools.google_search import google_search

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "searchInformation": {
                "formattedTotalResults": "1,234",
                "formattedSearchTime": "0.45",
            },
            "items": [
                {
                    "title": "Python.org",
                    "link": "https://python.org",
                    "snippet": "Python programming language",
                    "displayLink": "python.org",
                    "pagemap": {
                        "metatags": [
                            {
                                "og:description": "Official Python site",
                                "article:published_time": "2024-01-01T00:00:00Z",
                            }
                        ]
                    },
                },
                {
                    "title": "PyPI",
                    "link": "https://pypi.org",
                    "snippet": "Python package index",
                },
            ],
        }
        mock_response.raise_for_status = MagicMock()

        with (
            patch("src.tools.google_search._get_api_key", return_value="test-key"),
            patch("src.tools.google_search._get_cse_id", return_value="test-cx"),
            patch("requests.get", return_value=mock_response),
        ):
            result = google_search("python", num_results=2)

        assert "Google search results for: python" in result
        assert "Python.org" in result
        assert "https://python.org" in result
        assert "Python programming language" in result
        assert "About 1,234 results" in result
        assert "Official Python site" in result
        assert "2024-01-01" in result

    def test_google_search_spelling_suggestion(self):
        from src.tools.google_search import google_search

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "searchInformation": {},
            "spelling": {"correctedQuery": "python"},
            "items": [],
        }
        mock_response.raise_for_status = MagicMock()

        with (
            patch("src.tools.google_search._get_api_key", return_value="test-key"),
            patch("src.tools.google_search._get_cse_id", return_value="test-cx"),
            patch("requests.get", return_value=mock_response),
        ):
            result = google_search("pthon")

        assert "Did you mean: python" in result

    def test_google_search_no_results(self):
        from src.tools.google_search import google_search

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "searchInformation": {},
            "items": [],
        }
        mock_response.raise_for_status = MagicMock()

        with (
            patch("src.tools.google_search._get_api_key", return_value="test-key"),
            patch("src.tools.google_search._get_cse_id", return_value="test-cx"),
            patch("requests.get", return_value=mock_response),
        ):
            result = google_search("xyzzy_nothing_matches")

        assert "No results found" in result

    def test_google_search_http_error(self):
        from src.tools.google_search import google_search

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.json.return_value = {"error": {"message": "Quota exceeded"}}
        http_error = requests.exceptions.HTTPError("429", response=mock_response)

        with (
            patch("src.tools.google_search._get_api_key", return_value="test-key"),
            patch("src.tools.google_search._get_cse_id", return_value="test-cx"),
            patch("requests.get", side_effect=http_error),
        ):
            result = google_search("test")

        assert "Error" in result
        assert "429" in result
        assert "Quota exceeded" in result

    def test_google_search_connection_error(self):
        from src.tools.google_search import google_search

        with (
            patch("src.tools.google_search._get_api_key", return_value="test-key"),
            patch("src.tools.google_search._get_cse_id", return_value="test-cx"),
            patch("requests.get", side_effect=requests.exceptions.ConnectionError("timeout")),
        ):
            result = google_search("test")

        assert "Error" in result

    def test_google_search_num_results_passed_to_api(self):
        from src.tools.google_search import google_search

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "searchInformation": {},
            "items": [],
        }
        mock_response.raise_for_status = MagicMock()

        with (
            patch("src.tools.google_search._get_api_key", return_value="test-key"),
            patch("src.tools.google_search._get_cse_id", return_value="test-cx"),
            patch("requests.get", return_value=mock_response) as mock_get,
        ):
            google_search("test", num_results=50)

        # Should be clamped to 10 in the API params
        call_kwargs = mock_get.call_args[1]
        assert "params" in call_kwargs
        assert call_kwargs["params"]["num"] == 10

    def test_google_search_invalid_safe_search_defaults(self):
        from src.tools.google_search import google_search

        mock_response = MagicMock()
        mock_response.json.return_value = {"searchInformation": {}, "items": []}
        mock_response.raise_for_status = MagicMock()

        with (
            patch("src.tools.google_search._get_api_key", return_value="test-key"),
            patch("src.tools.google_search._get_cse_id", return_value="test-cx"),
            patch("requests.get", return_value=mock_response) as mock_get,
        ):
            google_search("test", safe_search="invalid")

        # Verify the request was made with 'off' as safe search
        call_kwargs = mock_get.call_args[1]
        assert "params" in call_kwargs
        assert call_kwargs["params"]["safe"] == "off"


class TestGoogleSearchConfigure:
    """Unit tests for configuration helpers."""

    def test_is_configured_false_when_no_key(self):
        from src.tools.google_search import is_configured

        with (
            patch("src.tools.google_search._get_api_key", return_value=None),
            patch("src.tools.google_search._get_cse_id", return_value="test-cx"),
        ):
            assert is_configured() is False

    def test_is_configured_false_when_no_cse_id(self):
        from src.tools.google_search import is_configured

        with (
            patch("src.tools.google_search._get_api_key", return_value="test-key"),
            patch("src.tools.google_search._get_cse_id", return_value=None),
        ):
            assert is_configured() is False

    def test_is_configured_true_when_both_set(self):
        from src.tools.google_search import is_configured

        with (
            patch("src.tools.google_search._get_api_key", return_value="test-key"),
            patch("src.tools.google_search._get_cse_id", return_value="test-cx"),
        ):
            assert is_configured() is True

    def test_configure_google_search_sets_values(self):
        from src.tools.google_search import _get_api_key, _get_cse_id, configure_google_search

        configure_google_search({"api_key": "my-key", "cse_id": "my-cx"})
        assert _get_api_key() == "my-key"
        assert _get_cse_id() == "my-cx"
        # Reset
        configure_google_search({"api_key": "", "cse_id": ""})


class TestGoogleSearchInput:
    """Unit tests for the Pydantic input schema."""

    def test_google_search_input_defaults(self):
        from src.tools.google_search import GoogleSearchInput

        schema = GoogleSearchInput(query="test")
        assert schema.query == "test"
        assert schema.num_results == 10
        assert schema.date_restrict == ""
        assert schema.language == ""
        assert schema.safe_search == "off"

    def test_google_search_input_custom_values(self):
        from src.tools.google_search import GoogleSearchInput

        schema = GoogleSearchInput(
            query="test",
            num_results=5,
            date_restrict="d7",
            language="lang_en",
            safe_search="active",
        )
        assert schema.num_results == 5
        assert schema.date_restrict == "d7"
        assert schema.language == "lang_en"
        assert schema.safe_search == "active"
