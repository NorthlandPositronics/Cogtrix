"""Tests for pure-logic functions in src/agent/core.py.

Tests only deterministic, side-effect-free functions that do not require
a live LLM, FAISS, or any external service.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.agent.core import (
    _estimate_msg_tokens,
    _format_model_detail,
    _format_models_table,
    _trim_to_token_budget,
    _truncate_content,
    build_system_prompt,
    format_milestone_instructions,
    prepare_messages_with_context,
)

# ---------------------------------------------------------------------------
# _format_model_detail
# ---------------------------------------------------------------------------


class TestFormatModelDetail:
    def test_string_passthrough(self):
        assert _format_model_detail("gpt-4") == "gpt-4"

    def test_dict_basic(self):
        result = _format_model_detail({"provider": "openai", "model": "gpt-4"})
        assert result == "openai/gpt-4"

    def test_dict_with_temperature(self):
        result = _format_model_detail({"provider": "openai", "model": "gpt-4", "temperature": 0.7})
        assert "temp=0.7" in result

    def test_dict_with_context_window(self):
        result = _format_model_detail(
            {"provider": "ollama", "model": "qwen3", "context_window": 8192}
        )
        assert "ctx=8192" in result

    def test_dict_with_num_ctx(self):
        result = _format_model_detail({"provider": "ollama", "model": "qwen3", "num_ctx": 4096})
        assert "ctx=4096" in result

    def test_dict_missing_keys_uses_question_mark(self):
        result = _format_model_detail({})
        assert "?" in result

    def test_model_config_object(self):
        from src.config import ModelConfig

        mc = ModelConfig(provider="anthropic", model="claude-sonnet-4-5")
        result = _format_model_detail(mc)
        assert "anthropic/claude-sonnet-4-5" in result

    def test_model_config_with_temperature(self):
        from src.config import ModelConfig

        mc = ModelConfig(provider="openai", model="gpt-4", temperature=0.5)
        result = _format_model_detail(mc)
        assert "temp=0.5" in result

    def test_model_config_with_context_window(self):
        from src.config import ModelConfig

        mc = ModelConfig(provider="openai", model="gpt-4", context_window=16384)
        result = _format_model_detail(mc)
        assert "ctx=16384" in result

    def test_unknown_type_returns_str(self):
        result = _format_model_detail(42)
        assert result == "42"


# ---------------------------------------------------------------------------
# _format_models_table
# ---------------------------------------------------------------------------


class TestFormatModelsTable:
    def test_empty_dict_returns_empty_string(self):
        assert _format_models_table({}) == ""

    def test_single_model(self):
        result = _format_models_table({"default": "gpt-4"})
        assert "default" in result
        assert "gpt-4" in result

    def test_delegation_models_section(self):
        models = {"fast": "gpt-3.5", "smart": "gpt-4"}
        result = _format_models_table(models, delegation_models=["fast"])
        assert "Delegation targets" in result
        assert "fast" in result

    def test_delegation_models_others_section(self):
        models = {"fast": "gpt-3.5", "smart": "gpt-4"}
        result = _format_models_table(models, delegation_models=["fast"])
        # "smart" should be in the "Other models" section
        assert "Other models" in result
        assert "smart" in result

    def test_delegation_models_all_delegated(self):
        models = {"fast": "gpt-3.5", "smart": "gpt-4"}
        result = _format_models_table(models, delegation_models=["fast", "smart"])
        # No "Other models" section when all are delegation targets
        assert "Other models" not in result

    def test_delegation_model_not_in_registry_skipped(self):
        models = {"fast": "gpt-3.5"}
        result = _format_models_table(models, delegation_models=["nonexistent"])
        # nonexistent not in models, shouldn't appear in table
        assert "nonexistent" not in result

    def test_returns_string_with_header(self):
        result = _format_models_table({"m": "v"})
        assert "Available Models" in result


# ---------------------------------------------------------------------------
# format_milestone_instructions
# ---------------------------------------------------------------------------


class TestFormatMilestoneInstructions:
    def test_empty_list(self):
        result = format_milestone_instructions([])
        assert "Milestones" in result
        assert "report_progress" in result

    def test_single_milestone(self):
        m = MagicMock()
        m.index = 1
        m.title = "Research phase"
        result = format_milestone_instructions([m])
        assert "1. Research phase" in result

    def test_multiple_milestones(self):
        milestones = []
        for i, title in enumerate(["Plan", "Build", "Test"], start=1):
            m = MagicMock()
            m.index = i
            m.title = title
            milestones.append(m)
        result = format_milestone_instructions(milestones)
        assert "1. Plan" in result
        assert "2. Build" in result
        assert "3. Test" in result

    def test_includes_focus_rule(self):
        result = format_milestone_instructions([])
        assert "Focus rule" in result


# ---------------------------------------------------------------------------
# build_system_prompt
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    def test_default_prompt_used_when_base_none(self):
        from src.agent.core import DEFAULT_SYSTEM_PROMPT

        result = build_system_prompt()
        assert DEFAULT_SYSTEM_PROMPT in result

    def test_custom_base_prompt(self):
        result = build_system_prompt(base_prompt="My custom instructions")
        assert "My custom instructions" in result

    def test_mode_additions_appended(self):
        result = build_system_prompt(mode_additions="## Code mode context")
        assert "## Code mode context" in result

    def test_tool_instructions_appended(self):
        result = build_system_prompt(tool_instructions="Use JSON for tool calls.")
        assert "Use JSON for tool calls." in result

    def test_milestone_instructions_appended(self):
        result = build_system_prompt(milestone_instructions="## Milestones\n1. Done")
        assert "## Milestones" in result

    def test_models_table_included_when_no_active_tools(self):
        models = {"fast": "gpt-3.5"}
        result = build_system_prompt(models=models)
        assert "fast" in result

    def test_models_table_excluded_when_no_delegation_tool_active(self):
        models = {"fast": "gpt-3.5"}
        # active_tool_names without delegate_task or delegate_parallel
        result = build_system_prompt(models=models, active_tool_names={"web_search", "shell"})
        assert "fast" not in result

    def test_models_table_included_when_delegate_task_active(self):
        models = {"fast": "gpt-3.5"}
        result = build_system_prompt(
            models=models, active_tool_names={"delegate_task", "web_search"}
        )
        assert "fast" in result

    def test_parts_joined_with_double_newline(self):
        result = build_system_prompt(
            base_prompt="base", mode_additions="mode", tool_instructions="tools"
        )
        assert "\n\n" in result

    def test_empty_models_no_table(self):
        result = build_system_prompt(models={})
        assert "Available Models" not in result


# ---------------------------------------------------------------------------
# _estimate_msg_tokens
# ---------------------------------------------------------------------------


class TestEstimateMsgTokens:
    def test_message_with_content_attr(self):
        msg = MagicMock()
        msg.content = "hello world"  # 11 chars → ~2 tokens
        result = _estimate_msg_tokens(msg)
        assert result >= 1

    def test_dict_with_content(self):
        result = _estimate_msg_tokens({"content": "a" * 400})  # 400 chars → 100 tokens
        assert result == 100

    def test_empty_message_returns_overhead(self):
        msg = MagicMock()
        msg.content = ""
        result = _estimate_msg_tokens(msg)
        assert result == 10

    def test_list_content_summed(self):
        msg = MagicMock()
        msg.content = ["hello", "world"]  # 10 chars → 2 tokens
        result = _estimate_msg_tokens(msg)
        assert result >= 1

    def test_minimum_is_one_for_non_empty(self):
        msg = MagicMock()
        msg.content = "a"  # 1 char → max(0, 1) = 1
        result = _estimate_msg_tokens(msg)
        assert result >= 1


# ---------------------------------------------------------------------------
# _truncate_content
# ---------------------------------------------------------------------------


class TestTruncateContent:
    def test_short_content_unchanged(self):
        text = "hello"
        assert _truncate_content(text, max_tokens=100) == text

    def test_long_content_truncated(self):
        text = "a" * 10000
        result = _truncate_content(text, max_tokens=100)
        assert len(result) < len(text)
        assert "truncated" in result

    def test_truncated_keeps_both_ends(self):
        # Build text with distinguishable start and end
        text = "START" + "x" * 5000 + "END"
        result = _truncate_content(text, max_tokens=50)
        assert "START" in result
        assert "END" in result

    def test_exact_limit_not_truncated(self):
        max_tokens = 100
        text = "a" * (max_tokens * 4)  # exactly at limit
        assert _truncate_content(text, max_tokens) == text

    def test_non_positive_max_tokens_returns_unchanged(self):
        text = "hello world"
        assert _truncate_content(text, max_tokens=0) == text
        assert _truncate_content(text, max_tokens=-1) == text
        assert _truncate_content(text, max_tokens=-100) == text


# ---------------------------------------------------------------------------
# prepare_messages_with_context
# ---------------------------------------------------------------------------


class TestPrepareMessagesWithContext:
    def test_basic_user_input(self):
        result = prepare_messages_with_context([], "hello")
        # Last message is the user input
        last = result[-1]
        content = last.content if hasattr(last, "content") else last.get("content", "")
        assert content == "hello"

    def test_history_included(self):
        try:
            from langchain_core.messages import HumanMessage
        except ImportError:
            pytest.skip("langchain not installed")

        history = [HumanMessage(content="past message")]
        result = prepare_messages_with_context(history, "new input")
        assert len(result) >= 2

    def test_context_prefix_injected_as_message(self):
        try:
            from langchain_core.messages import HumanMessage
        except ImportError:
            pytest.skip("langchain not installed")

        result = prepare_messages_with_context([], "hello", context_prefix="Some context")
        # First message should contain the context prefix (HumanMessage for
        # strict-provider compatibility — Qwen3/vLLM reject SystemMessage
        # outside position 0).
        first = result[0]
        assert isinstance(first, HumanMessage)
        assert "Some context" in first.content

    def test_no_context_prefix_no_system_message(self):
        try:
            from langchain_core.messages import SystemMessage
        except ImportError:
            pytest.skip("langchain not installed")

        result = prepare_messages_with_context([], "hello")
        for msg in result:
            assert not isinstance(msg, SystemMessage)

    def test_token_budget_trims_history(self):
        try:
            from langchain_core.messages import HumanMessage
        except ImportError:
            pytest.skip("langchain not installed")

        # Create a very large history that exceeds a small token budget
        big_text = "x" * 8000
        history = [HumanMessage(content=big_text) for _ in range(5)]
        result = prepare_messages_with_context(history, "new input", max_context_tokens=512)
        # Result should be smaller than original history + input
        assert len(result) <= len(history) + 1

    def test_fallback_without_langchain(self):
        from unittest.mock import patch

        with patch("src.agent.core.HumanMessage", None):
            result = prepare_messages_with_context(
                [{"type": "human", "content": "history"}], "new input"
            )
        # Fallback returns list with history + new input dict
        assert len(result) >= 1
        last = result[-1]
        assert isinstance(last, dict)
        assert last["content"] == "new input"


# ---------------------------------------------------------------------------
# _trim_to_token_budget — role alternation guard
# ---------------------------------------------------------------------------


class TestTrimToTokenBudget:
    """Verify _trim_to_token_budget never produces a leading AIMessage."""

    def _human(self, text: str):
        from langchain_core.messages import HumanMessage

        return HumanMessage(content=text)

    def _ai(self, text: str):
        from langchain_core.messages import AIMessage

        return AIMessage(content=text)

    def _tool(self, text: str, tool_call_id: str = "tc1"):
        from langchain_core.messages import ToolMessage

        return ToolMessage(content=text, tool_call_id=tool_call_id)

    def test_leading_ai_message_is_removed(self):
        """If trimming exposes an AIMessage at the head, drop it."""
        msgs = [
            self._human("prefix"),
            self._human("h1"),
            self._ai("a1"),
            self._human("h2"),
            self._ai("a2"),
            self._human("tail"),
        ]
        # Budget small enough that prefix + h1 are dropped, exposing a1
        result = _trim_to_token_budget(msgs, max_context_tokens=64)
        # First message after any internal dropping must not be AIMessage
        assert not isinstance(result[0], type(self._ai("")))

    def test_leading_tool_message_after_ai_removal_is_also_dropped(self):
        """Dropping an AIMessage may orphan a following ToolMessage — remove both."""
        msgs = [
            self._human("prefix"),
            self._human("h1"),
            self._ai("a1"),
            self._tool("t1"),
            self._human("h2"),
            self._ai("a2"),
            self._human("tail"),
        ]
        result = _trim_to_token_budget(msgs, max_context_tokens=64)
        assert not isinstance(result[0], type(self._ai("")))
        assert not isinstance(result[0], type(self._tool("")))

    def test_valid_history_unchanged_when_under_budget(self):
        """When everything fits, the message order is preserved."""
        msgs = [
            self._human("h1"),
            self._ai("a1"),
            self._human("h2"),
            self._ai("a2"),
            self._human("tail"),
        ]
        result = _trim_to_token_budget(msgs, max_context_tokens=8192)
        assert len(result) == len(msgs)
        assert result[0].content == "h1"
        assert result[-1].content == "tail"

    def test_system_message_preserved_as_fixed_head(self):
        """SystemMessage stays at position 0 even when history is trimmed."""
        from langchain_core.messages import SystemMessage

        msgs = [
            SystemMessage(content="sys"),
            self._human("h1"),
            self._ai("a1"),
            self._human("h2"),
            self._ai("a2"),
            self._human("tail"),
        ]
        result = _trim_to_token_budget(msgs, max_context_tokens=64)
        assert isinstance(result[0], SystemMessage)
        # Ensure no AIMessage immediately follows SystemMessage if h1 was dropped
        if len(result) > 1:
            assert not isinstance(result[1], type(self._ai("")))
