"""Tests for the Brave Search tool."""

from unittest.mock import MagicMock, patch

import requests


class TestBraveSearch:
    """Unit tests for brave_search()."""

    def test_brave_search_not_configured_returns_error(self):
        from cogtrix_core.tools.brave_search import brave_search

        with patch("cogtrix_core.tools.brave_search._get_api_key", return_value=None):
            result = brave_search("python")

        assert "Error" in result
        assert "Brave API key" in result

    def test_brave_search_empty_query_returns_error(self):
        from cogtrix_core.tools.brave_search import brave_search

        with patch("cogtrix_core.tools.brave_search._get_api_key", return_value="test-key"):
            result = brave_search("   ")

        assert "Error" in result
        assert "Empty" in result

    def test_brave_search_returns_web_results(self):
        from cogtrix_core.tools.brave_search import brave_search

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {
                        "title": "Python.org",
                        "url": "https://python.org",
                        "description": "Python programming language",
                    },
                    {
                        "title": "PyPI",
                        "url": "https://pypi.org",
                        "description": "Python package index",
                    },
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()

        with (
            patch("cogtrix_core.tools.brave_search._get_api_key", return_value="test-key"),
            patch("requests.get", return_value=mock_response),
        ):
            result = brave_search("python", count=2)

        assert "Brave web search results for: python" in result
        assert "Python.org" in result
        assert "https://python.org" in result
        assert "Python programming language" in result
        assert "PyPI" in result

    def test_brave_search_returns_news_results(self):
        from cogtrix_core.tools.brave_search import brave_search

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "Python News",
                    "url": "https://news.example.com/python",
                    "description": "Latest Python news",
                    "age": "2 hours ago",
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with (
            patch("cogtrix_core.tools.brave_search._get_api_key", return_value="test-key"),
            patch("requests.get", return_value=mock_response),
        ):
            result = brave_search("python", count=1, search_type="news")

        assert "Brave news search results for: python" in result
        assert "Python News" in result
        assert "2 hours ago" in result

    def test_brave_search_no_results(self):
        from cogtrix_core.tools.brave_search import brave_search

        mock_response = MagicMock()
        mock_response.json.return_value = {"web": {"results": []}}
        mock_response.raise_for_status = MagicMock()

        with (
            patch("cogtrix_core.tools.brave_search._get_api_key", return_value="test-key"),
            patch("requests.get", return_value=mock_response),
        ):
            result = brave_search("xyzzy_nothing_matches")

        assert "No results found" in result

    def test_brave_search_http_error(self):
        from cogtrix_core.tools.brave_search import brave_search

        mock_response = MagicMock()
        mock_response.status_code = 403
        http_error = requests.exceptions.HTTPError("403 Forbidden", response=mock_response)

        with (
            patch("cogtrix_core.tools.brave_search._get_api_key", return_value="test-key"),
            patch("requests.get", side_effect=http_error),
        ):
            result = brave_search("test")

        assert "Error" in result
        assert "403" in result

    def test_brave_search_connection_error(self):
        from cogtrix_core.tools.brave_search import brave_search

        with (
            patch("cogtrix_core.tools.brave_search._get_api_key", return_value="test-key"),
            patch(
                "requests.get",
                side_effect=requests.exceptions.ConnectionError("connection refused"),
            ),
        ):
            result = brave_search("test")

        assert "Error" in result

    def test_brave_search_invalid_search_type_defaults_to_web(self):
        from cogtrix_core.tools.brave_search import brave_search

        mock_response = MagicMock()
        mock_response.json.return_value = {"web": {"results": []}}
        mock_response.raise_for_status = MagicMock()

        with (
            patch("cogtrix_core.tools.brave_search._get_api_key", return_value="test-key"),
            patch("requests.get", return_value=mock_response),
        ):
            result = brave_search("test", search_type="invalid")

        assert "Brave web search results" in result

    def test_brave_search_count_clamped(self):
        from cogtrix_core.tools.brave_search import brave_search

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {"title": f"Result {i}", "url": f"https://example.com/{i}", "description": "x"}
                    for i in range(25)
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()

        with (
            patch("cogtrix_core.tools.brave_search._get_api_key", return_value="test-key"),
            patch("requests.get", return_value=mock_response),
        ):
            result = brave_search("test", count=50)

        # count should be clamped to 20
        assert result.count("Result") <= 20

    def test_brave_search_faq_and_infobox(self):
        from cogtrix_core.tools.brave_search import brave_search

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "web": {
                "results": [{"title": "T1", "url": "https://example.com", "description": "D1"}]
            },
            "faq": {
                "results": [{"question": "What is Python?", "answer": "A programming language."}]
            },
            "infobox": {"results": [{"title": "Python", "long_desc": "A high-level language."}]},
        }
        mock_response.raise_for_status = MagicMock()

        with (
            patch("cogtrix_core.tools.brave_search._get_api_key", return_value="test-key"),
            patch("requests.get", return_value=mock_response),
        ):
            result = brave_search("python")

        assert "Related Questions:" in result
        assert "What is Python?" in result
        assert "A programming language." in result
        assert "Infobox: Python" in result
        assert "A high-level language." in result

    def test_brave_search_extra_snippets_truncated(self):
        from cogtrix_core.tools.brave_search import brave_search

        long_snippet = "x" * 600
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {
                        "title": "T1",
                        "url": "https://example.com",
                        "description": "D1",
                        "extra_snippets": [long_snippet],
                    }
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()

        with (
            patch("cogtrix_core.tools.brave_search._get_api_key", return_value="test-key"),
            patch("requests.get", return_value=mock_response),
        ):
            result = brave_search("test")

        assert "..." in result
        # Verify the snippet was truncated to 500 chars + "..."
        assert long_snippet[:500] in result
        # The full 600-char string should not appear
        assert long_snippet not in result


class TestBraveConfigure:
    """Unit tests for configuration helpers."""

    def test_is_configured_false_when_no_key(self):
        from cogtrix_core.tools.brave_search import is_configured

        with patch("cogtrix_core.tools.brave_search._get_api_key", return_value=None):
            assert is_configured() is False

    def test_is_configured_true_when_key_set(self):
        from cogtrix_core.tools.brave_search import is_configured

        with patch("cogtrix_core.tools.brave_search._get_api_key", return_value="test-key"):
            assert is_configured() is True

    def test_configure_brave_sets_key(self):
        from cogtrix_core.tools.brave_search import _get_api_key, configure_brave

        configure_brave({"api_key": "my-key"})
        assert _get_api_key() == "my-key"
        # Reset
        configure_brave({"api_key": ""})
        assert _get_api_key() is None or _get_api_key() == ""


class TestBraveSearchInput:
    """Unit tests for the Pydantic input schema."""

    def test_brave_search_input_defaults(self):
        from cogtrix_core.tools.brave_search import BraveSearchInput

        schema = BraveSearchInput(query="test")
        assert schema.query == "test"
        assert schema.count == 5
        assert schema.search_type == "web"
        assert schema.freshness == ""

    def test_brave_search_input_custom_values(self):
        from cogtrix_core.tools.brave_search import BraveSearchInput

        schema = BraveSearchInput(query="test", count=10, search_type="news", freshness="pd")
        assert schema.count == 10
        assert schema.search_type == "news"
        assert schema.freshness == "pd"
