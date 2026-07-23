"""Tests for src/orchestration/runner.py metric wiring."""

import sys
from unittest.mock import MagicMock

from src.orchestration.runner import log_tool_calls_from_result


class TestLogToolCallsMetrics:
    """Unit tests for tool_calls_total metric increments."""

    def _ai_msg(self, tool_calls):
        """Create a fake AIMessage with given tool_calls."""
        from langchain_core.messages import AIMessage

        return AIMessage(content="", tool_calls=tool_calls)

    def _tool_msg(self, content="ok", name="search_web", tool_call_id="tc1"):
        """Create a fake ToolMessage."""
        from langchain_core.messages import ToolMessage

        return ToolMessage(content=content, name=name, tool_call_id=tool_call_id)

    def _patch_metric(self, monkeypatch):
        """Return a mock counter and inject a fake metrics module."""
        mock_counter = MagicMock()
        fake_module = MagicMock()
        fake_module.TOOL_CALLS_TOTAL = mock_counter
        monkeypatch.setitem(sys.modules, "src.api.routes.metrics", fake_module)
        return mock_counter

    def test_success_increments_metric(self, monkeypatch):
        mock_counter = self._patch_metric(monkeypatch)

        ai = self._ai_msg([{"name": "search_web", "args": {}, "id": "tc1"}])
        tool = self._tool_msg(content="results", name="search_web", tool_call_id="tc1")

        log_tool_calls_from_result({"messages": [ai, tool]}, prior_count=0)

        mock_counter.labels.assert_called_once_with(tool_name="search_web", status="success")
        mock_counter.labels.return_value.inc.assert_called_once()

    def test_error_increments_metric(self, monkeypatch):
        mock_counter = self._patch_metric(monkeypatch)

        ai = self._ai_msg([{"name": "read_file", "args": {}, "id": "tc1"}])
        tool = self._tool_msg(
            content="Error: permission denied", name="read_file", tool_call_id="tc1"
        )

        log_tool_calls_from_result({"messages": [ai, tool]}, prior_count=0)

        mock_counter.labels.assert_called_once_with(tool_name="read_file", status="error")
        mock_counter.labels.return_value.inc.assert_called_once()

    def test_multiple_tool_calls(self, monkeypatch):
        mock_counter = self._patch_metric(monkeypatch)

        ai = self._ai_msg(
            [
                {"name": "search_web", "args": {}, "id": "tc1"},
                {"name": "read_file", "args": {}, "id": "tc2"},
            ]
        )
        tool1 = self._tool_msg(content="results", name="search_web", tool_call_id="tc1")
        tool2 = self._tool_msg(content="Error: not found", name="read_file", tool_call_id="tc2")

        log_tool_calls_from_result({"messages": [ai, tool1, tool2]}, prior_count=0)

        assert mock_counter.labels.call_count == 2
        mock_counter.labels.assert_any_call(tool_name="search_web", status="success")
        mock_counter.labels.assert_any_call(tool_name="read_file", status="error")
        assert mock_counter.labels.return_value.inc.call_count == 2

    def test_no_tool_calls_no_increment(self, monkeypatch):
        mock_counter = self._patch_metric(monkeypatch)

        log_tool_calls_from_result({"messages": []}, prior_count=0)

        mock_counter.labels.assert_not_called()

    def test_prior_count_skips_history(self, monkeypatch):
        mock_counter = self._patch_metric(monkeypatch)

        history_ai = self._ai_msg([{"name": "old_tool", "args": {}, "id": "tc0"}])
        history_tool = self._tool_msg(content="old_result", name="old_tool", tool_call_id="tc0")
        ai = self._ai_msg([{"name": "search_web", "args": {}, "id": "tc1"}])
        tool = self._tool_msg(content="results", name="search_web", tool_call_id="tc1")

        log_tool_calls_from_result(
            {"messages": [history_ai, history_tool, ai, tool]}, prior_count=2
        )

        mock_counter.labels.assert_called_once_with(tool_name="search_web", status="success")
