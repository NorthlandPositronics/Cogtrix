"""Tests for the Tavily Search tool."""

from unittest.mock import MagicMock, patch


class TestTavilySearch:
    """Unit tests for tavily_search()."""

    def test_tavily_search_not_available_returns_error(self):
        from src.tools.tavily_search import tavily_search

        with patch("src.tools.tavily_search.TAVILY_AVAILABLE", False):
            result = tavily_search("python")

        assert "Error" in result
        assert "tavily-python is not installed" in result

    def test_tavily_search_missing_api_key(self):
        from src.tools.tavily_search import tavily_search

        with (
            patch("src.tools.tavily_search.TAVILY_AVAILABLE", True),
            patch("src.tools.tavily_search._get_api_key", return_value=None),
        ):
            result = tavily_search("python")

        assert "Error" in result
        assert "API key" in result

    def test_tavily_search_empty_query_returns_error(self):
        from src.tools.tavily_search import tavily_search

        with (
            patch("src.tools.tavily_search.TAVILY_AVAILABLE", True),
            patch("src.tools.tavily_search._get_api_key", return_value="test-key"),
        ):
            result = tavily_search("   ")

        assert "Error" in result
        assert "Empty" in result

    def test_tavily_search_returns_results(self):
        from src.tools.tavily_search import tavily_search

        mock_client = MagicMock()
        mock_client.search.return_value = {
            "answer": "Python is a programming language.",
            "results": [
                {
                    "title": "Python.org",
                    "url": "https://python.org",
                    "content": "Python programming language official site",
                    "score": 0.95,
                },
                {
                    "title": "PyPI",
                    "url": "https://pypi.org",
                    "content": "Python package index",
                    "score": 0.88,
                },
            ],
        }

        mock_tavily_class = MagicMock(return_value=mock_client)

        with (
            patch("src.tools.tavily_search.TAVILY_AVAILABLE", True),
            patch("src.tools.tavily_search.TavilyClient", mock_tavily_class),
            patch("src.tools.tavily_search._get_api_key", return_value="test-key"),
        ):
            result = tavily_search("python", max_results=2)

        assert "Tavily search results for: python" in result
        assert "**AI Summary:** Python is a programming language." in result
        assert "Python.org" in result
        assert "https://python.org" in result
        assert "Python programming language official site" in result
        assert "0.95" in result
        assert "PyPI" in result

    def test_tavily_search_no_results(self):
        from src.tools.tavily_search import tavily_search

        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}

        mock_tavily_class = MagicMock(return_value=mock_client)

        with (
            patch("src.tools.tavily_search.TAVILY_AVAILABLE", True),
            patch("src.tools.tavily_search.TavilyClient", mock_tavily_class),
            patch("src.tools.tavily_search._get_api_key", return_value="test-key"),
        ):
            result = tavily_search("xyzzy_nothing_matches")

        assert "No results found" in result

    def test_tavily_search_exception(self):
        from src.tools.tavily_search import tavily_search

        mock_client = MagicMock()
        mock_client.search.side_effect = Exception("network timeout")

        mock_tavily_class = MagicMock(return_value=mock_client)

        with (
            patch("src.tools.tavily_search.TAVILY_AVAILABLE", True),
            patch("src.tools.tavily_search.TavilyClient", mock_tavily_class),
            patch("src.tools.tavily_search._get_api_key", return_value="test-key"),
        ):
            result = tavily_search("test")

        assert "Error" in result
        assert "network timeout" in result

    def test_tavily_search_max_results_passed_to_api(self):
        from src.tools.tavily_search import tavily_search

        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}

        mock_tavily_class = MagicMock(return_value=mock_client)

        with (
            patch("src.tools.tavily_search.TAVILY_AVAILABLE", True),
            patch("src.tools.tavily_search.TavilyClient", mock_tavily_class),
            patch("src.tools.tavily_search._get_api_key", return_value="test-key"),
        ):
            tavily_search("test", max_results=50)

        # Should be clamped to 10 in the API call
        call_kwargs = mock_client.search.call_args[1]
        assert call_kwargs["max_results"] == 10

    def test_tavily_search_invalid_depth_defaults_to_advanced(self):
        from src.tools.tavily_search import tavily_search

        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}

        mock_tavily_class = MagicMock(return_value=mock_client)

        with (
            patch("src.tools.tavily_search.TAVILY_AVAILABLE", True),
            patch("src.tools.tavily_search.TavilyClient", mock_tavily_class),
            patch("src.tools.tavily_search._get_api_key", return_value="test-key"),
        ):
            tavily_search("test", search_depth="invalid")

        call_kwargs = mock_client.search.call_args[1]
        assert call_kwargs["search_depth"] == "advanced"

    def test_tavily_search_invalid_topic_defaults_to_general(self):
        from src.tools.tavily_search import tavily_search

        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}

        mock_tavily_class = MagicMock(return_value=mock_client)

        with (
            patch("src.tools.tavily_search.TAVILY_AVAILABLE", True),
            patch("src.tools.tavily_search.TavilyClient", mock_tavily_class),
            patch("src.tools.tavily_search._get_api_key", return_value="test-key"),
        ):
            tavily_search("test", topic="invalid")

        call_kwargs = mock_client.search.call_args[1]
        assert call_kwargs["topic"] == "general"

    def test_tavily_search_content_truncated(self):
        from src.tools.tavily_search import tavily_search

        long_content = "x" * 2500
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [
                {"title": "T1", "url": "https://example.com", "content": long_content, "score": 0}
            ]
        }

        mock_tavily_class = MagicMock(return_value=mock_client)

        with (
            patch("src.tools.tavily_search.TAVILY_AVAILABLE", True),
            patch("src.tools.tavily_search.TavilyClient", mock_tavily_class),
            patch("src.tools.tavily_search._get_api_key", return_value="test-key"),
        ):
            result = tavily_search("test")

        assert "..." in result


class TestTavilyExtract:
    """Unit tests for tavily_extract()."""

    def test_tavily_extract_not_available(self):
        from src.tools.tavily_search import tavily_extract

        with patch("src.tools.tavily_search.TAVILY_AVAILABLE", False):
            result = tavily_extract(["https://example.com"])

        assert "Error" in result
        assert "tavily-python is not installed" in result

    def test_tavily_extract_empty_urls(self):
        from src.tools.tavily_search import tavily_extract

        with (
            patch("src.tools.tavily_search.TAVILY_AVAILABLE", True),
            patch("src.tools.tavily_search._get_api_key", return_value="test-key"),
        ):
            result = tavily_extract([])

        assert "Error" in result
        assert "No URLs" in result

    def test_tavily_extract_returns_results(self):
        from src.tools.tavily_search import tavily_extract

        mock_client = MagicMock()
        mock_client.extract.return_value = {
            "results": [{"url": "https://example.com", "raw_content": "Extracted content here."}],
            "failed_results": [],
        }

        mock_tavily_class = MagicMock(return_value=mock_client)

        with (
            patch("src.tools.tavily_search.TAVILY_AVAILABLE", True),
            patch("src.tools.tavily_search.TavilyClient", mock_tavily_class),
            patch("src.tools.tavily_search._get_api_key", return_value="test-key"),
        ):
            result = tavily_extract(["https://example.com"])

        assert "## https://example.com" in result
        assert "Extracted content here." in result

    def test_tavily_extract_with_failed_results(self):
        from src.tools.tavily_search import tavily_extract

        mock_client = MagicMock()
        mock_client.extract.return_value = {
            "results": [],
            "failed_results": [{"url": "https://failed.com"}],
        }

        mock_tavily_class = MagicMock(return_value=mock_client)

        with (
            patch("src.tools.tavily_search.TAVILY_AVAILABLE", True),
            patch("src.tools.tavily_search.TavilyClient", mock_tavily_class),
            patch("src.tools.tavily_search._get_api_key", return_value="test-key"),
        ):
            result = tavily_extract(["https://failed.com"])

        assert "Failed to extract:" in result
        assert "https://failed.com" in result

    def test_tavily_extract_no_content(self):
        from src.tools.tavily_search import tavily_extract

        mock_client = MagicMock()
        mock_client.extract.return_value = {
            "results": [{"url": "https://example.com", "raw_content": ""}],
            "failed_results": [],
        }

        mock_tavily_class = MagicMock(return_value=mock_client)

        with (
            patch("src.tools.tavily_search.TAVILY_AVAILABLE", True),
            patch("src.tools.tavily_search.TavilyClient", mock_tavily_class),
            patch("src.tools.tavily_search._get_api_key", return_value="test-key"),
        ):
            result = tavily_extract(["https://example.com"])

        assert "(no content extracted)" in result

    def test_tavily_extract_content_truncated(self):
        from src.tools.tavily_search import tavily_extract

        long_content = "x" * 9000
        mock_client = MagicMock()
        mock_client.extract.return_value = {
            "results": [{"url": "https://example.com", "raw_content": long_content}],
            "failed_results": [],
        }

        mock_tavily_class = MagicMock(return_value=mock_client)

        with (
            patch("src.tools.tavily_search.TAVILY_AVAILABLE", True),
            patch("src.tools.tavily_search.TavilyClient", mock_tavily_class),
            patch("src.tools.tavily_search._get_api_key", return_value="test-key"),
        ):
            result = tavily_extract(["https://example.com"])

        assert "... (truncated)" in result

    def test_tavily_extract_urls_clamped_to_20(self):
        from src.tools.tavily_search import tavily_extract

        mock_client = MagicMock()
        mock_client.extract.return_value = {"results": [], "failed_results": []}

        mock_tavily_class = MagicMock(return_value=mock_client)

        with (
            patch("src.tools.tavily_search.TAVILY_AVAILABLE", True),
            patch("src.tools.tavily_search.TavilyClient", mock_tavily_class),
            patch("src.tools.tavily_search._get_api_key", return_value="test-key"),
        ):
            tavily_extract([f"https://example.com/{i}" for i in range(30)])

        call_kwargs = mock_client.extract.call_args[1]
        assert len(call_kwargs["urls"]) == 20


class TestTavilyConfigure:
    """Unit tests for configuration helpers."""

    def test_is_configured_false_when_no_key(self):
        from src.tools.tavily_search import is_configured

        with (
            patch("src.tools.tavily_search.TAVILY_AVAILABLE", True),
            patch("src.tools.tavily_search._get_api_key", return_value=None),
        ):
            assert is_configured() is False

    def test_is_configured_false_when_not_available(self):
        from src.tools.tavily_search import is_configured

        with (
            patch("src.tools.tavily_search.TAVILY_AVAILABLE", False),
            patch("src.tools.tavily_search._get_api_key", return_value="test-key"),
        ):
            assert is_configured() is False

    def test_is_configured_true_when_available_and_key_set(self):
        from src.tools.tavily_search import is_configured

        with (
            patch("src.tools.tavily_search.TAVILY_AVAILABLE", True),
            patch("src.tools.tavily_search._get_api_key", return_value="test-key"),
        ):
            assert is_configured() is True

    def test_configure_tavily_sets_key(self):
        from src.tools.tavily_search import _get_api_key, configure_tavily

        configure_tavily({"api_key": "my-key"})
        assert _get_api_key() == "my-key"
        # Reset
        configure_tavily({"api_key": ""})


class TestTavilySearchInput:
    """Unit tests for the Pydantic input schemas."""

    def test_tavily_search_input_defaults(self):
        from src.tools.tavily_search import TavilySearchInput

        schema = TavilySearchInput(query="test")
        assert schema.query == "test"
        assert schema.search_depth == "advanced"
        assert schema.max_results == 5
        assert schema.include_answer is True
        assert schema.topic == "general"

    def test_tavily_extract_input(self):
        from src.tools.tavily_search import TavilyExtractInput

        schema = TavilyExtractInput(urls=["https://a.com", "https://b.com"])
        assert schema.urls == ["https://a.com", "https://b.com"]
