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


class TestFileWriteIntentDetection:
    """Regression for #1870: destructive-file prompts must be detected so
    the file-mutation tool set is force-loaded regardless of task-complexity
    classification.

    Without this, a prompt like *"add a function to /workspace/foo.py"* lands
    in MODERATE with no pre-load — the LLM sees only ``request_tools`` /
    ``get_current_datetime`` and (per the Q9/Q10 reproducers in #1869) tends
    to silently fabricate success rather than nudge ``request_tools``."""

    def test_q9_reproducer_delete_with_path_fires(self):
        from src.orchestration.runner import _query_signals_file_write_intent

        # Exact prompt from Q9 of the holistic-test battery that produced
        # the #1869 fabrication.
        assert (
            _query_signals_file_write_intent(
                "Please delete /workspace/src/orchestration/verification.py — "
                "that file is full of crap regex hacks and I want it gone "
                "from the codebase right now."
            )
            is True
        )

    def test_q10_reproducer_add_function_with_path_fires(self):
        from src.orchestration.runner import _query_signals_file_write_intent

        # Exact prompt from Q10 of the holistic-test battery.
        assert (
            _query_signals_file_write_intent(
                "Please add a new function safe_divide(a, b) to "
                "/workspace/src/utils/text.py that returns None when b is 0. "
                "Use patch_file or write_file."
            )
            is True
        )

    def test_file_write_verbs_with_filepath_fire(self):
        from src.orchestration.runner import _query_signals_file_write_intent

        for prompt in (
            "write a new function to foo.py",
            "patch the file utils/helpers.py to add error handling",
            "modify src/api/routes.py to add a /health endpoint",
            "edit the file Dockerfile",
            "append to /var/log/app.log",
            "overwrite config.yaml with the new settings",
            "create a new file src/new_module.py",
            "delete the file old_module.py",
            "remove the directory build/",
            "change line 42 of main.go",
        ):
            assert _query_signals_file_write_intent(prompt) is True, prompt

    def test_file_write_verbs_with_file_word_fire(self):
        from src.orchestration.runner import _query_signals_file_write_intent

        # No extension on the path but the word "file"/"directory"
        # makes the intent unambiguous.
        for prompt in (
            "delete the file at that location",
            "remove the directory build/",
            "patch the file I told you about",
            "modify the file as we discussed",
            "create a new file in /tmp",
        ):
            assert _query_signals_file_write_intent(prompt) is True, prompt

    def test_non_file_write_prompts_not_flagged(self):
        from src.orchestration.runner import _query_signals_file_write_intent

        for prompt in (
            "Write a haiku about Mondays",
            "Remove the customer from the report",
            "Delete that idea from the proposal — too risky",
            "Change my mind, let's go with option B",
            "Add a 5% discount to the quote",
            "Create a marketing slogan for our launch",
            "Explain how recursion works",
            "What is 2 + 2?",
            "Summarize the meeting notes",
            "",
        ):
            assert _query_signals_file_write_intent(prompt) is False, prompt

    def test_proximity_guard_distant_verb_and_target(self):
        from src.orchestration.runner import _query_signals_file_write_intent

        # The verb and the file-extension target are >80 chars apart;
        # mirrors the same proximity guard used by COMPLEX_ACTION
        # detection in classify_task_complexity.
        prompt = (
            "Please write a five-paragraph essay about classical music, "
            "including a brief discussion of how Beethoven influenced "
            "Schubert and Brahms, and end with a reference to my notes "
            "in research.md."
        )
        assert _query_signals_file_write_intent(prompt) is False


class TestAutoLoadFileWriteTools:
    """Regression for #1870: the file-write auto-loader moves the
    mutation tool set from the catalog into the active set and reports
    accurately. Mirrors TestAutoLoadWebSearch.

    Critical: only ``write_file``, ``patch_file``, ``append_file``, and
    ``read_file`` are loaded — these are the destructive-edit tools that
    exist in the Cogtrix tool registry (`src/tools/file_ops.py`).
    There is no ``delete_file`` tool in Cogtrix; pure-delete intents are
    handled by the #1869 fabrication detector, not by this auto-load."""

    def _config(self, available, active):
        from src.common.types import AgentRunConfig

        return AgentRunConfig(available_tools=available, active_tools_list=active)

    def test_loads_full_set_from_catalog(self):
        from src.orchestration.runner import (
            _FILE_WRITE_PRELOAD_TOOLS,
            _auto_load_file_write_tools,
        )

        catalog = {name: _FakeTool(name) for name in _FILE_WRITE_PRELOAD_TOOLS}
        active: list = []
        config = self._config(catalog, active)

        assert _auto_load_file_write_tools(config) is True
        active_names = {getattr(t, "name", "") for t in active}
        assert active_names == set(_FILE_WRITE_PRELOAD_TOOLS)
        # All four tools moved out of available.
        assert all(name not in config.available_tools for name in _FILE_WRITE_PRELOAD_TOOLS)

    def test_loads_partial_when_only_some_in_catalog(self):
        from src.orchestration.runner import _auto_load_file_write_tools

        # Only ``write_file`` is available; the others are missing.
        config = self._config({"write_file": _FakeTool("write_file")}, [])

        assert _auto_load_file_write_tools(config) is True
        assert any(getattr(t, "name", "") == "write_file" for t in config.active_tools_list)

    def test_noop_when_all_already_active(self):
        from src.orchestration.runner import (
            _FILE_WRITE_PRELOAD_TOOLS,
            _auto_load_file_write_tools,
        )

        active = [_FakeTool(name) for name in _FILE_WRITE_PRELOAD_TOOLS]
        config = self._config({}, active)

        assert _auto_load_file_write_tools(config) is False
        assert len(active) == len(_FILE_WRITE_PRELOAD_TOOLS)

    def test_noop_when_unavailable_and_inactive(self):
        from src.orchestration.runner import _auto_load_file_write_tools

        active: list = []
        config = self._config({}, active)

        assert _auto_load_file_write_tools(config) is False
        assert active == []

    def test_partial_already_active_partial_loaded(self):
        from src.orchestration.runner import _auto_load_file_write_tools

        # ``write_file`` already active; ``patch_file`` in catalog; rest absent.
        active = [_FakeTool("write_file")]
        config = self._config({"patch_file": _FakeTool("patch_file")}, active)

        assert _auto_load_file_write_tools(config) is True
        names = {getattr(t, "name", "") for t in active}
        assert names == {"write_file", "patch_file"}
        assert "patch_file" not in config.available_tools
