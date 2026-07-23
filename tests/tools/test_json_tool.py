"""Tests for cogtrix_core/tools/json_tool.py — dot-path guard and path traversal."""

import pytest

from cogtrix_core.tools.json_tool import _get_by_path, query_json


class TestGetByPathDotGuard:
    """_get_by_path must reject '.' and '..' to prevent path-traversal queries."""

    def test_single_dot_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            _get_by_path({"a": 1}, ".")

    def test_double_dot_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            _get_by_path({"a": 1}, "..")

    def test_valid_path_still_works(self) -> None:
        data = {"users": [{"name": "alice"}]}
        result = _get_by_path(data, "users[0].name")
        assert result == "alice"


class TestQueryJsonDotGuard:
    """query_json must return an error string (not raise) for dot-only paths."""

    def test_query_single_dot_returns_error(self) -> None:
        result = query_json('{"a": 1}', ".")
        assert "error" in result.lower() or "invalid" in result.lower()

    def test_query_double_dot_returns_error(self) -> None:
        result = query_json('{"a": 1}', "..")
        assert "error" in result.lower() or "invalid" in result.lower()

    def test_query_valid_path_returns_value(self) -> None:
        result = query_json('{"key": "val"}', "key")
        assert result == "val"

    def test_query_missing_key_returns_error(self) -> None:
        result = query_json('{"a": 1}', "nonexistent")
        assert "error" in result.lower()
