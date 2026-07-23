"""Tests for cogtrix_core/orchestration/intent.py — BUG-042 and related."""

from __future__ import annotations

from unittest.mock import MagicMock


class TestClassifyThinkTaskPromptFormat:
    """BUG-042: verify XML-tag delimiters are used instead of triple-quotes."""

    def _make_llm(self, label: str = "general") -> MagicMock:
        mock_llm = MagicMock()
        response = MagicMock()
        response.content = label
        mock_llm.invoke.return_value = response
        return mock_llm

    def test_prompt_uses_xml_task_tags(self):
        from cogtrix_core.orchestration.intent import classify_think_task

        llm = self._make_llm("general")
        # Task matches 2+ keyword categories → forces LLM fallback path
        classify_think_task("compare and plan a project", llm)

        assert llm.invoke.called
        prompt = llm.invoke.call_args[0][0]
        assert "<task_text>" in prompt
        assert "</task_text>" in prompt

    def test_prompt_does_not_use_triple_quotes(self):
        from cogtrix_core.orchestration.intent import classify_think_task

        llm = self._make_llm("general")
        classify_think_task("some task", llm)

        prompt = llm.invoke.call_args[0][0]
        assert '"""' not in prompt

    def test_task_text_embedded_between_xml_tags(self):
        from cogtrix_core.orchestration.intent import classify_think_task

        llm = self._make_llm("general")
        task = "analyze the codebase"
        classify_think_task(task, llm)

        prompt = llm.invoke.call_args[0][0]
        assert f"<task_text>{task}</task_text>" in prompt

    def test_triple_quote_injection_does_not_escape_delimiter(self):
        """A task containing triple quotes must not break the delimiter structure."""
        from cogtrix_core.orchestration.intent import classify_think_task

        llm = self._make_llm("general")
        # Task with triple-quote sequences; matches 2+ categories → forces LLM path
        task = 'compare """plan""" strategy'
        classify_think_task(task, llm)

        prompt = llm.invoke.call_args[0][0]
        assert "<task_text>" in prompt
        assert "</task_text>" in prompt
        # The prompt must still be well-formed with exactly one open and close tag
        assert prompt.count("<task_text>") == 1
        assert prompt.count("</task_text>") == 1

    def test_double_quotes_replaced_with_single_quotes_in_sanitized(self):
        from cogtrix_core.orchestration.intent import classify_think_task

        llm = self._make_llm("general")
        task = 'say "hello"'
        classify_think_task(task, llm)

        prompt = llm.invoke.call_args[0][0]
        # Double quotes in the task are replaced with single quotes
        assert "say 'hello'" in prompt

    def test_newlines_replaced_with_spaces_in_sanitized(self):
        from cogtrix_core.orchestration.intent import classify_think_task

        llm = self._make_llm("general")
        task = "line1\nline2\r\nline3"
        classify_think_task(task, llm)

        prompt = llm.invoke.call_args[0][0]
        assert "line1 line2  line3" in prompt or "line1 line2" in prompt

    def test_null_bytes_stripped_from_sanitized(self):
        from cogtrix_core.orchestration.intent import classify_think_task

        llm = self._make_llm("general")
        task = "task\x00with\x00nulls"
        classify_think_task(task, llm)

        prompt = llm.invoke.call_args[0][0]
        assert "\x00" not in prompt

    def test_returns_category_on_valid_label(self):
        from cogtrix_core.orchestration.intent import THINK_DEFAULT_CATEGORY, classify_think_task

        first_non_default_name = None
        from cogtrix_core.orchestration.intent import THINK_CATEGORIES

        for cat in THINK_CATEGORIES:
            if cat.name != THINK_DEFAULT_CATEGORY.name:
                first_non_default_name = cat.name
                break

        if first_non_default_name is None:
            return  # Only one category exists — skip

        llm = self._make_llm(first_non_default_name)
        result = classify_think_task("some task", llm)
        assert result.name == first_non_default_name

    def test_returns_default_category_on_unknown_label(self):
        from cogtrix_core.orchestration.intent import THINK_DEFAULT_CATEGORY, classify_think_task

        llm = self._make_llm("totally_unknown_label_xyz")
        result = classify_think_task("some task", llm)
        assert result == THINK_DEFAULT_CATEGORY

    def test_returns_default_category_on_llm_exception(self):
        from cogtrix_core.orchestration.intent import THINK_DEFAULT_CATEGORY, classify_think_task

        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("connection error")
        result = classify_think_task("some task", llm)
        assert result == THINK_DEFAULT_CATEGORY


class TestClassifyThinkTaskTimeout:
    """Regression: hung LLM must not block classify_think_task indefinitely."""

    def test_returns_default_category_on_llm_timeout(self):
        """A mock LLM that sleeps longer than the timeout should return default."""
        import time
        from unittest.mock import patch

        from cogtrix_core.orchestration.intent import THINK_DEFAULT_CATEGORY, classify_think_task

        llm = MagicMock()

        def _slow_invoke(_prompt: str) -> MagicMock:
            # sleeps longer than the patched timeout but short enough
            # that shutdown(wait=True) does not stall the test suite
            time.sleep(2)
            return MagicMock(content="general")

        llm.invoke.side_effect = _slow_invoke

        # Patch timeout to a very small value so the test completes quickly.
        with patch("cogtrix_core.orchestration.intent._CLASSIFY_TIMEOUT_SECONDS", 0.1):
            # Use a task that does not match any single keyword category,
            # forcing the LLM fallback path.
            result = classify_think_task("compare and plan a project", llm)

        assert result == THINK_DEFAULT_CATEGORY
