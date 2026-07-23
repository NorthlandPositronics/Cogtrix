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


class _FakeTool:
    """Minimal tool stub with a ``name`` and no ``_resolve`` (so the
    auto-loader appends it directly without LazyToolProxy resolution)."""

    def __init__(self, name: str) -> None:
        self.name = name


class TestRealtimeQueryDetection:
    """Regression for bug #1839 (primary root cause): recency-dependent
    prompts must be detected so the retrieval tool set is force-loaded
    regardless of task-complexity classification."""

    def test_current_stock_price_is_realtime(self):
        from src.orchestration.runner import _query_needs_realtime_data

        # The exact prompt that failed in the next66 trial run.
        assert _query_needs_realtime_data("What's the current Apple stock price?") is True

    def test_recency_markers_detected(self):
        from src.orchestration.runner import _query_needs_realtime_data

        for prompt in (
            "today's weather in Tokyo",
            "latest news on the election",
            "what is the most recent SpaceX launch",
            "give me the AAPL stock quote",
            "current USD to EUR exchange rate",
            "the score of the game right now",
        ):
            assert _query_needs_realtime_data(prompt) is True, prompt

    def test_non_realtime_prompts_not_flagged(self):
        from src.orchestration.runner import _query_needs_realtime_data

        for prompt in (
            "Write a function to reverse a string",
            "Explain how recursion works",
            "What is 2 + 2?",
            "Summarize this paragraph for me",
            "",
        ):
            assert _query_needs_realtime_data(prompt) is False, prompt


class TestAutoLoadWebSearch:
    """Regression for bug #1839: the shared web_search auto-loader moves
    the tool from the catalog into the active set and reports accurately."""

    def _config(self, available, active):
        from src.common.types import AgentRunConfig

        return AgentRunConfig(available_tools=available, active_tools_list=active)

    def test_loads_from_catalog(self):
        from src.orchestration.runner import _auto_load_web_search

        tool = _FakeTool("web_search")
        active: list = []
        config = self._config({"web_search": tool}, active)

        assert _auto_load_web_search(config) is True
        assert any(getattr(t, "name", "") == "web_search" for t in active)
        assert "web_search" not in config.available_tools

    def test_noop_when_already_active(self):
        from src.orchestration.runner import _auto_load_web_search

        active = [_FakeTool("web_search")]
        config = self._config({}, active)

        assert _auto_load_web_search(config) is False
        assert len(active) == 1

    def test_noop_when_unavailable(self):
        from src.orchestration.runner import _auto_load_web_search

        active: list = []
        config = self._config({}, active)

        assert _auto_load_web_search(config) is False
        assert active == []
