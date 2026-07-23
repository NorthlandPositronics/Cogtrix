"""Tests for the SerpAPI Search tool."""

from unittest.mock import MagicMock, patch


class TestSerpAPISearch:
    """Unit tests for serpapi_search()."""

    def test_serpapi_search_not_available_returns_error(self):
        from src.tools.serpapi_search import serpapi_search

        with patch("src.tools.serpapi_search.SERPAPI_AVAILABLE", False):
            result = serpapi_search("python")

        assert "Error" in result
        assert "google-search-results is not installed" in result

    def test_serpapi_search_missing_api_key(self):
        from src.tools.serpapi_search import serpapi_search

        with (
            patch("src.tools.serpapi_search.SERPAPI_AVAILABLE", True),
            patch("src.tools.serpapi_search._get_api_key", return_value=None),
        ):
            result = serpapi_search("python")

        assert "Error" in result
        assert "SerpAPI key" in result

    def test_serpapi_search_empty_query_returns_error(self):
        from src.tools.serpapi_search import serpapi_search

        with (
            patch("src.tools.serpapi_search.SERPAPI_AVAILABLE", True),
            patch("src.tools.serpapi_search._get_api_key", return_value="test-key"),
        ):
            result = serpapi_search("   ")

        assert "Error" in result
        assert "Empty" in result

    def test_serpapi_search_returns_results(self):
        from src.tools.serpapi_search import serpapi_search

        mock_search = MagicMock()
        mock_search.get_dict.return_value = {
            "organic_results": [
                {
                    "title": "Python.org",
                    "link": "https://python.org",
                    "snippet": "Python programming language",
                    "date": "2024-01-01",
                    "rich_snippet": {"top": {"extensions": ["Official site", "Programming"]}},
                },
                {"title": "PyPI", "link": "https://pypi.org", "snippet": "Python package index"},
            ],
            "related_questions": [
                {"question": "What is Python?", "snippet": "A programming language."}
            ],
        }

        mock_google_search_class = MagicMock(return_value=mock_search)

        with (
            patch("src.tools.serpapi_search.SERPAPI_AVAILABLE", True),
            patch("src.tools.serpapi_search.GoogleSearch", mock_google_search_class),
            patch("src.tools.serpapi_search._get_api_key", return_value="test-key"),
        ):
            result = serpapi_search("python", num_results=2)

        assert "SerpAPI (google) search results for: python" in result
        assert "Python.org" in result
        assert "https://python.org" in result
        assert "Python programming language" in result
        assert "2024-01-01" in result
        assert "Official site" in result
        assert "People Also Ask:" in result
        assert "What is Python?" in result

    def test_serpapi_search_answer_box_and_knowledge_graph(self):
        from src.tools.serpapi_search import serpapi_search

        mock_search = MagicMock()
        mock_search.get_dict.return_value = {
            "answer_box": {"answer": "42", "title": "Answer"},
            "knowledge_graph": {
                "title": "Python",
                "type": "Programming Language",
                "description": "A high-level language.",
            },
            "organic_results": [],
        }

        mock_google_search_class = MagicMock(return_value=mock_search)

        with (
            patch("src.tools.serpapi_search.SERPAPI_AVAILABLE", True),
            patch("src.tools.serpapi_search.GoogleSearch", mock_google_search_class),
            patch("src.tools.serpapi_search._get_api_key", return_value="test-key"),
        ):
            result = serpapi_search("python")

        assert "**Direct Answer (Answer):** 42" in result
        assert "**Knowledge Graph: Python (Programming Language)**" in result
        assert "A high-level language." in result

    def test_serpapi_search_no_results(self):
        from src.tools.serpapi_search import serpapi_search

        mock_search = MagicMock()
        mock_search.get_dict.return_value = {"organic_results": []}

        mock_google_search_class = MagicMock(return_value=mock_search)

        with (
            patch("src.tools.serpapi_search.SERPAPI_AVAILABLE", True),
            patch("src.tools.serpapi_search.GoogleSearch", mock_google_search_class),
            patch("src.tools.serpapi_search._get_api_key", return_value="test-key"),
        ):
            result = serpapi_search("xyzzy_nothing_matches")

        assert "No results found" in result

    def test_serpapi_search_error_in_response(self):
        from src.tools.serpapi_search import serpapi_search

        mock_search = MagicMock()
        mock_search.get_dict.return_value = {"error": "Invalid API key"}

        mock_google_search_class = MagicMock(return_value=mock_search)

        with (
            patch("src.tools.serpapi_search.SERPAPI_AVAILABLE", True),
            patch("src.tools.serpapi_search.GoogleSearch", mock_google_search_class),
            patch("src.tools.serpapi_search._get_api_key", return_value="test-key"),
        ):
            result = serpapi_search("test")

        assert "Error from SerpAPI" in result
        assert "Invalid API key" in result

    def test_serpapi_search_exception(self):
        from src.tools.serpapi_search import serpapi_search

        mock_google_search_class = MagicMock(side_effect=Exception("network error"))

        with (
            patch("src.tools.serpapi_search.SERPAPI_AVAILABLE", True),
            patch("src.tools.serpapi_search.GoogleSearch", mock_google_search_class),
            patch("src.tools.serpapi_search._get_api_key", return_value="test-key"),
        ):
            result = serpapi_search("test")

        assert "Error" in result
        assert "network error" not in result  # sanitized: no raw exception content
        assert "request failed" in result.lower()  # sanitized message present

    def test_serpapi_search_num_results_clamped(self):
        from src.tools.serpapi_search import serpapi_search

        mock_search = MagicMock()
        mock_search.get_dict.return_value = {
            "organic_results": [
                {"title": f"Title {i}", "link": f"https://example.com/{i}", "snippet": "x"}
                for i in range(25)
            ]
        }

        mock_google_search_class = MagicMock(return_value=mock_search)

        with (
            patch("src.tools.serpapi_search.SERPAPI_AVAILABLE", True),
            patch("src.tools.serpapi_search.GoogleSearch", mock_google_search_class),
            patch("src.tools.serpapi_search._get_api_key", return_value="test-key"),
        ):
            result = serpapi_search("test", num_results=50)

        # Should be clamped to 20 (count lines starting with a number)
        result_lines = [
            ln for ln in result.splitlines() if ln.strip().startswith(("1.", "2.", "10.", "20."))
        ]
        assert len(result_lines) <= 20

    def test_serpapi_search_invalid_engine_defaults_to_google(self):
        from src.tools.serpapi_search import serpapi_search

        mock_search = MagicMock()
        mock_search.get_dict.return_value = {"organic_results": []}

        mock_google_search_class = MagicMock(return_value=mock_search)

        with (
            patch("src.tools.serpapi_search.SERPAPI_AVAILABLE", True),
            patch("src.tools.serpapi_search.GoogleSearch", mock_google_search_class),
            patch("src.tools.serpapi_search._get_api_key", return_value="test-key"),
        ):
            serpapi_search("test", engine="invalid")

        # Verify GoogleSearch was called with engine='google'
        call_args = mock_google_search_class.call_args[0][0]
        assert call_args["engine"] == "google"

    def test_serpapi_search_time_period_param(self):
        from src.tools.serpapi_search import serpapi_search

        mock_search = MagicMock()
        mock_search.get_dict.return_value = {"organic_results": []}

        mock_google_search_class = MagicMock(return_value=mock_search)

        with (
            patch("src.tools.serpapi_search.SERPAPI_AVAILABLE", True),
            patch("src.tools.serpapi_search.GoogleSearch", mock_google_search_class),
            patch("src.tools.serpapi_search._get_api_key", return_value="test-key"),
        ):
            serpapi_search("test", time_period="qdr:w")

        call_args = mock_google_search_class.call_args[0][0]
        assert call_args["tbs"] == "qdr:w"


class TestSerpAPIConfigure:
    """Unit tests for configuration helpers."""

    def test_is_configured_false_when_no_key(self):
        from src.tools.serpapi_search import is_configured

        with (
            patch("src.tools.serpapi_search.SERPAPI_AVAILABLE", True),
            patch("src.tools.serpapi_search._get_api_key", return_value=None),
        ):
            assert is_configured() is False

    def test_is_configured_false_when_not_available(self):
        from src.tools.serpapi_search import is_configured

        with (
            patch("src.tools.serpapi_search.SERPAPI_AVAILABLE", False),
            patch("src.tools.serpapi_search._get_api_key", return_value="test-key"),
        ):
            assert is_configured() is False

    def test_is_configured_true_when_available_and_key_set(self):
        from src.tools.serpapi_search import is_configured

        with (
            patch("src.tools.serpapi_search.SERPAPI_AVAILABLE", True),
            patch("src.tools.serpapi_search._get_api_key", return_value="test-key"),
        ):
            assert is_configured() is True

    def test_configure_serpapi_sets_key(self):
        from src.tools.serpapi_search import _get_api_key, configure_serpapi

        configure_serpapi({"api_key": "my-key"})
        assert _get_api_key() == "my-key"
        # Reset
        configure_serpapi({"api_key": ""})


class TestSerpAPISearchInput:
    """Unit tests for the Pydantic input schema."""

    def test_serpapi_search_input_defaults(self):
        from src.tools.serpapi_search import SerpAPISearchInput

        schema = SerpAPISearchInput(query="test")
        assert schema.query == "test"
        assert schema.engine == "google"
        assert schema.num_results == 10
        assert schema.search_type == ""
        assert schema.time_period == ""

    def test_serpapi_search_input_custom_values(self):
        from src.tools.serpapi_search import SerpAPISearchInput

        schema = SerpAPISearchInput(
            query="test", engine="bing", num_results=5, search_type="nws", time_period="qdr:d"
        )
        assert schema.engine == "bing"
        assert schema.num_results == 5
        assert schema.search_type == "nws"
        assert schema.time_period == "qdr:d"
