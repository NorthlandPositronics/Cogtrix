"""Tests for the Exa Search tool."""

from unittest.mock import MagicMock, patch


class TestExaSearch:
    """Unit tests for exa_search()."""

    def test_exa_search_not_available_returns_error(self):
        from cogtrix_core.tools.exa_search import exa_search

        with patch("cogtrix_core.tools.exa_search.EXA_AVAILABLE", False):
            result = exa_search("python")

        assert "Error" in result
        assert "exa-py is not installed" in result

    def test_exa_search_missing_api_key(self):
        from cogtrix_core.tools.exa_search import exa_search

        with (
            patch("cogtrix_core.tools.exa_search.EXA_AVAILABLE", True),
            patch("cogtrix_core.tools.exa_search._get_api_key", return_value=None),
        ):
            result = exa_search("python")

        assert "Error" in result
        assert "API key" in result

    def test_exa_search_empty_query_returns_error(self):
        from cogtrix_core.tools.exa_search import exa_search

        with (
            patch("cogtrix_core.tools.exa_search.EXA_AVAILABLE", True),
            patch("cogtrix_core.tools.exa_search._get_api_key", return_value="test-key"),
        ):
            result = exa_search("   ")

        assert "Error" in result
        assert "Empty" in result

    def test_exa_search_returns_results(self):
        from cogtrix_core.tools.exa_search import exa_search

        mock_result = MagicMock()
        mock_result.title = "Python.org"
        mock_result.url = "https://python.org"
        mock_result.score = 0.95
        mock_result.text = "Python programming language"

        mock_results = MagicMock()
        mock_results.results = [mock_result]

        mock_client = MagicMock()
        mock_client.search.return_value = mock_results

        mock_exa_class = MagicMock(return_value=mock_client)

        with (
            patch("cogtrix_core.tools.exa_search.EXA_AVAILABLE", True),
            patch("cogtrix_core.tools.exa_search.Exa", mock_exa_class),
            patch("cogtrix_core.tools.exa_search._get_api_key", return_value="test-key"),
        ):
            result = exa_search("python", num_results=1)

        assert "Exa search results for: python" in result
        assert "Python.org" in result
        assert "https://python.org" in result
        assert "0.9500" in result
        assert "Python programming language" in result

    def test_exa_search_no_results(self):
        from cogtrix_core.tools.exa_search import exa_search

        mock_results = MagicMock()
        mock_results.results = []

        mock_client = MagicMock()
        mock_client.search.return_value = mock_results

        mock_exa_class = MagicMock(return_value=mock_client)

        with (
            patch("cogtrix_core.tools.exa_search.EXA_AVAILABLE", True),
            patch("cogtrix_core.tools.exa_search.Exa", mock_exa_class),
            patch("cogtrix_core.tools.exa_search._get_api_key", return_value="test-key"),
        ):
            result = exa_search("xyzzy_nothing_matches")

        assert "No results found" in result

    def test_exa_search_credit_error(self):
        from cogtrix_core.tools.exa_search import exa_search

        mock_client = MagicMock()
        mock_client.search.side_effect = Exception("402 Payment Required: credits exhausted")

        mock_exa_class = MagicMock(return_value=mock_client)

        with (
            patch("cogtrix_core.tools.exa_search.EXA_AVAILABLE", True),
            patch("cogtrix_core.tools.exa_search.Exa", mock_exa_class),
            patch("cogtrix_core.tools.exa_search._get_api_key", return_value="test-key"),
        ):
            result = exa_search("test")

        assert "Error" in result
        assert "credits exhausted" in result

    def test_exa_search_generic_error(self):
        from cogtrix_core.tools.exa_search import exa_search

        mock_client = MagicMock()
        mock_client.search.side_effect = Exception("network timeout")

        mock_exa_class = MagicMock(return_value=mock_client)

        with (
            patch("cogtrix_core.tools.exa_search.EXA_AVAILABLE", True),
            patch("cogtrix_core.tools.exa_search.Exa", mock_exa_class),
            patch("cogtrix_core.tools.exa_search._get_api_key", return_value="test-key"),
        ):
            result = exa_search("test")

        assert "Error" in result
        assert "Operation failed" in result

    def test_exa_search_num_results_passed_to_api(self):
        from cogtrix_core.tools.exa_search import exa_search

        mock_results = MagicMock()
        mock_results.results = []

        mock_client = MagicMock()
        mock_client.search.return_value = mock_results

        mock_exa_class = MagicMock(return_value=mock_client)

        with (
            patch("cogtrix_core.tools.exa_search.EXA_AVAILABLE", True),
            patch("cogtrix_core.tools.exa_search.Exa", mock_exa_class),
            patch("cogtrix_core.tools.exa_search._get_api_key", return_value="test-key"),
        ):
            exa_search("test", num_results=50)

        # Should be clamped to 10 in the API kwargs
        call_kwargs = mock_client.search.call_args[1]
        assert call_kwargs["num_results"] == 10

    def test_exa_search_invalid_type_defaults_to_auto(self):
        from cogtrix_core.tools.exa_search import exa_search

        mock_results = MagicMock()
        mock_results.results = []

        mock_client = MagicMock()
        mock_client.search.return_value = mock_results

        mock_exa_class = MagicMock(return_value=mock_client)

        with (
            patch("cogtrix_core.tools.exa_search.EXA_AVAILABLE", True),
            patch("cogtrix_core.tools.exa_search.Exa", mock_exa_class),
            patch("cogtrix_core.tools.exa_search._get_api_key", return_value="test-key"),
        ):
            exa_search("test", search_type="invalid")

        # Verify search was called with type='auto'
        call_kwargs = mock_client.search.call_args[1]
        assert call_kwargs["type"] == "auto"


class TestExaFindSimilar:
    """Unit tests for exa_find_similar()."""

    def test_exa_find_similar_not_available(self):
        from cogtrix_core.tools.exa_search import exa_find_similar

        with patch("cogtrix_core.tools.exa_search.EXA_AVAILABLE", False):
            result = exa_find_similar("https://example.com")

        assert "Error" in result
        assert "exa-py is not installed" in result

    def test_exa_find_similar_empty_url(self):
        from cogtrix_core.tools.exa_search import exa_find_similar

        with (
            patch("cogtrix_core.tools.exa_search.EXA_AVAILABLE", True),
            patch("cogtrix_core.tools.exa_search._get_api_key", return_value="test-key"),
        ):
            result = exa_find_similar("   ")

        assert "Error" in result
        assert "Empty" in result

    def test_exa_find_similar_returns_results(self):
        from cogtrix_core.tools.exa_search import exa_find_similar

        mock_result = MagicMock()
        mock_result.title = "Similar Page"
        mock_result.url = "https://similar.example.com"
        mock_result.score = 0.88
        mock_result.text = "Similar content"

        mock_results = MagicMock()
        mock_results.results = [mock_result]

        mock_client = MagicMock()
        mock_client.find_similar.return_value = mock_results

        mock_exa_class = MagicMock(return_value=mock_client)

        with (
            patch("cogtrix_core.tools.exa_search.EXA_AVAILABLE", True),
            patch("cogtrix_core.tools.exa_search.Exa", mock_exa_class),
            patch("cogtrix_core.tools.exa_search._get_api_key", return_value="test-key"),
        ):
            result = exa_find_similar("https://example.com", num_results=1)

        assert "Exa search results for: pages similar to https://example.com" in result
        assert "Similar Page" in result
        assert "0.8800" in result


class TestExaGetContents:
    """Unit tests for exa_get_contents()."""

    def test_exa_get_contents_not_available(self):
        from cogtrix_core.tools.exa_search import exa_get_contents

        with patch("cogtrix_core.tools.exa_search.EXA_AVAILABLE", False):
            result = exa_get_contents(["https://example.com"])

        assert "Error" in result
        assert "exa-py is not installed" in result

    def test_exa_get_contents_empty_urls(self):
        from cogtrix_core.tools.exa_search import exa_get_contents

        with (
            patch("cogtrix_core.tools.exa_search.EXA_AVAILABLE", True),
            patch("cogtrix_core.tools.exa_search._get_api_key", return_value="test-key"),
        ):
            result = exa_get_contents([])

        assert "Error" in result
        assert "No URLs" in result

    def test_exa_get_contents_returns_results(self):
        from cogtrix_core.tools.exa_search import exa_get_contents

        mock_result = MagicMock()
        mock_result.url = "https://example.com"
        mock_result.text = "Extracted page content"

        mock_results = MagicMock()
        mock_results.results = [mock_result]

        mock_client = MagicMock()
        mock_client.get_contents.return_value = mock_results

        mock_exa_class = MagicMock(return_value=mock_client)

        with (
            patch("cogtrix_core.tools.exa_search.EXA_AVAILABLE", True),
            patch("cogtrix_core.tools.exa_search.Exa", mock_exa_class),
            patch("cogtrix_core.tools.exa_search._get_api_key", return_value="test-key"),
        ):
            result = exa_get_contents(["https://example.com"])

        assert "## https://example.com" in result
        assert "Extracted page content" in result

    def test_exa_get_contents_no_content(self):
        from cogtrix_core.tools.exa_search import exa_get_contents

        mock_result = MagicMock()
        mock_result.url = "https://example.com"
        mock_result.text = ""

        mock_results = MagicMock()
        mock_results.results = [mock_result]

        mock_client = MagicMock()
        mock_client.get_contents.return_value = mock_results

        mock_exa_class = MagicMock(return_value=mock_client)

        with (
            patch("cogtrix_core.tools.exa_search.EXA_AVAILABLE", True),
            patch("cogtrix_core.tools.exa_search.Exa", mock_exa_class),
            patch("cogtrix_core.tools.exa_search._get_api_key", return_value="test-key"),
        ):
            result = exa_get_contents(["https://example.com"])

        assert "(no content extracted)" in result

    def test_exa_get_contents_truncates_long_text(self):
        from cogtrix_core.tools.exa_search import exa_get_contents

        long_text = "x" * 9000
        mock_result = MagicMock()
        mock_result.url = "https://example.com"
        mock_result.text = long_text

        mock_results = MagicMock()
        mock_results.results = [mock_result]

        mock_client = MagicMock()
        mock_client.get_contents.return_value = mock_results

        mock_exa_class = MagicMock(return_value=mock_client)

        with (
            patch("cogtrix_core.tools.exa_search.EXA_AVAILABLE", True),
            patch("cogtrix_core.tools.exa_search.Exa", mock_exa_class),
            patch("cogtrix_core.tools.exa_search._get_api_key", return_value="test-key"),
        ):
            result = exa_get_contents(["https://example.com"])

        assert "... (truncated)" in result


class TestExaConfigure:
    """Unit tests for configuration helpers."""

    def test_is_configured_false_when_no_key(self):
        from cogtrix_core.tools.exa_search import is_configured

        with (
            patch("cogtrix_core.tools.exa_search.EXA_AVAILABLE", True),
            patch("cogtrix_core.tools.exa_search._get_api_key", return_value=None),
        ):
            assert is_configured() is False

    def test_is_configured_false_when_not_available(self):
        from cogtrix_core.tools.exa_search import is_configured

        with (
            patch("cogtrix_core.tools.exa_search.EXA_AVAILABLE", False),
            patch("cogtrix_core.tools.exa_search._get_api_key", return_value="test-key"),
        ):
            assert is_configured() is False

    def test_is_configured_true_when_available_and_key_set(self):
        from cogtrix_core.tools.exa_search import is_configured

        with (
            patch("cogtrix_core.tools.exa_search.EXA_AVAILABLE", True),
            patch("cogtrix_core.tools.exa_search._get_api_key", return_value="test-key"),
        ):
            assert is_configured() is True

    def test_configure_exa_sets_key(self):
        from cogtrix_core.tools.exa_search import _get_api_key, configure_exa

        configure_exa({"api_key": "my-key"})
        assert _get_api_key() == "my-key"
        # Reset
        configure_exa({"api_key": ""})


class TestExaSearchInput:
    """Unit tests for the Pydantic input schemas."""

    def test_exa_search_input_defaults(self):
        from cogtrix_core.tools.exa_search import ExaSearchInput

        schema = ExaSearchInput(query="test")
        assert schema.query == "test"
        assert schema.num_results == 5
        assert schema.include_text is True
        assert schema.search_type == "auto"

    def test_exa_find_similar_input_defaults(self):
        from cogtrix_core.tools.exa_search import ExaFindSimilarInput

        schema = ExaFindSimilarInput(url="https://example.com")
        assert schema.url == "https://example.com"
        assert schema.num_results == 5
        assert schema.include_text is True

    def test_exa_get_contents_input(self):
        from cogtrix_core.tools.exa_search import ExaGetContentsInput

        schema = ExaGetContentsInput(urls=["https://a.com", "https://b.com"])
        assert schema.urls == ["https://a.com", "https://b.com"]
