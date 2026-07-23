"""Tests for the SearXNG search tool."""

from unittest.mock import MagicMock, patch


def test_searxng_not_configured_returns_error():
    from src.tools.searxng_search import configure_searxng, searxng_search

    configure_searxng({})
    with patch("src.tools.searxng_search._get_url", return_value=None):
        result = searxng_search("python")
    assert "Error" in result
    assert "SEARXNG_URL" in result


def test_searxng_empty_query_returns_error():
    from src.tools.searxng_search import searxng_search

    with patch("src.tools.searxng_search._get_url", return_value="http://localhost:8888"):
        result = searxng_search("   ")
    assert "Error" in result


def test_searxng_returns_formatted_results():
    from src.tools.searxng_search import searxng_search

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {
                "title": "Python.org",
                "url": "https://python.org",
                "content": "Python programming language",
            },
            {"title": "PyPI", "url": "https://pypi.org", "content": "Python package index"},
        ]
    }
    mock_response.raise_for_status = MagicMock()

    with (
        patch("src.tools.searxng_search._get_url", return_value="http://localhost:8888"),
        patch("httpx.get", return_value=mock_response),
    ):
        result = searxng_search("python", num_results=2)

    assert "Python.org" in result
    assert "https://python.org" in result
    assert "SearXNG results for: python" in result


def test_searxng_no_results():
    from src.tools.searxng_search import searxng_search

    mock_response = MagicMock()
    mock_response.json.return_value = {"results": []}
    mock_response.raise_for_status = MagicMock()

    with (
        patch("src.tools.searxng_search._get_url", return_value="http://localhost:8888"),
        patch("httpx.get", return_value=mock_response),
    ):
        result = searxng_search("xyzzy_nothing_matches")

    assert "No results found" in result


def test_searxng_http_error():
    import httpx

    from src.tools.searxng_search import searxng_search

    mock_response = MagicMock()
    mock_response.status_code = 403
    http_error = httpx.HTTPStatusError("403", request=MagicMock(), response=mock_response)

    with (
        patch("src.tools.searxng_search._get_url", return_value="http://localhost:8888"),
        patch("httpx.get", side_effect=http_error),
    ):
        result = searxng_search("test")

    assert "Error" in result
    assert "403" in result


def test_searxng_connection_error():
    import httpx

    from src.tools.searxng_search import searxng_search

    with (
        patch("src.tools.searxng_search._get_url", return_value="http://localhost:8888"),
        patch("httpx.get", side_effect=httpx.RequestError("connection refused")),
    ):
        result = searxng_search("test")

    assert "Error" in result
    assert "connect" in result.lower()


def test_is_configured_false_when_no_url():
    from src.tools.searxng_search import is_configured

    with patch("src.tools.searxng_search._get_url", return_value=None):
        assert is_configured() is False


def test_is_configured_true_when_url_set():
    from src.tools.searxng_search import is_configured

    with patch("src.tools.searxng_search._get_url", return_value="http://localhost:8888"):
        assert is_configured() is True


def test_num_results_clamped():
    from src.tools.searxng_search import searxng_search

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"title": f"Result {i}", "url": f"http://example.com/{i}", "content": "x"}
            for i in range(20)
        ]
    }
    mock_response.raise_for_status = MagicMock()

    with (
        patch("src.tools.searxng_search._get_url", return_value="http://localhost:8888"),
        patch("httpx.get", return_value=mock_response),
    ):
        result = searxng_search("test", num_results=50)

    # Should be clamped to 10
    assert result.count("Result") <= 10


def test_configure_searxng_sets_url():
    from src.tools.searxng_search import _get_url, configure_searxng

    configure_searxng({"url": "http://my-searxng:8888"})
    assert _get_url() == "http://my-searxng:8888"
    # Reset
    configure_searxng({"url": ""})
