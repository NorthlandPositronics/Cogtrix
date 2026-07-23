"""Regression tests for domain-shift detection and rolling summary reset.

Covers:
- ``_extract_recent_user_prompts()`` helper
- ``_get_domain_shift_threshold()`` env-var configuration
- ``_check_domain_shift()`` counter, reset trigger, and facts distillation

These tests validate the fix for issue #583: when conversation domain shifts
(e.g. code → research), the rolling summary must reset (after distilling facts)
to prevent stale context from polluting agent memory.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from cogtrix_core.memory.modes.conversation import ConversationMemoryManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeStore:
    def __init__(self):
        self._history: dict[str, list] = {}

    def load_history(self, session_id):
        return list(self._history.get(session_id, []))

    def save_history(self, session_id, messages):
        self._history[session_id] = list(messages)


def _make_manager(session_id="test-session", config=None):
    """Build a ConversationMemoryManager with a fake store and no real LLM."""
    manager = ConversationMemoryManager(_FakeStore(), session_id, config or {})
    # Attach a no-op LLM so reset_summary() doesn't crash when it tries to distill.
    manager._llm = MagicMock()
    return manager


# ---------------------------------------------------------------------------
# _extract_recent_user_prompts
# ---------------------------------------------------------------------------


class TestExtractRecentUserPrompts:
    """Unit tests for BaseMemoryManager._extract_recent_user_prompts()."""

    def _msgs(self, *types_and_contents):
        """Build a message list from (role, content) pairs.

        role is 'human' or 'ai'. content is a string.
        """
        from langchain_core.messages import AIMessage, HumanMessage

        result = []
        for role, content in types_and_contents:
            if role == "human":
                result.append(HumanMessage(content=content))
            else:
                result.append(AIMessage(content=content))
        return result

    def test_empty_list_returns_empty(self):
        manager = _make_manager()
        assert manager._extract_recent_user_prompts([]) == []

    def test_single_human_returns_one(self):
        msgs = self._msgs(("human", "write a function"))
        manager = _make_manager()
        assert manager._extract_recent_user_prompts(msgs) == ["write a function"]

    def test_multiple_humans_returns_last_three_newest_first(self):
        msgs = self._msgs(
            ("human", "first"),
            ("ai", "response1"),
            ("human", "second"),
            ("ai", "response2"),
            ("human", "third"),
            ("ai", "response3"),
            ("human", "fourth"),
            ("ai", "response4"),
        )
        manager = _make_manager()
        # Newest 3 human prompts, newest first in the reversed scan, then reversed back
        result = manager._extract_recent_user_prompts(msgs)
        assert result == ["second", "third", "fourth"]

    def test_limit_respects_max(self):
        msgs = self._msgs(
            ("human", "a"),
            ("ai", "b"),
            ("human", "c"),
            ("ai", "d"),
            ("human", "e"),
        )
        manager = _make_manager()
        result = manager._extract_recent_user_prompts(msgs, limit=2)
        assert result == ["c", "e"]

    def test_skips_ai_messages(self):
        msgs = self._msgs(
            ("ai", "I will help you"),
            ("human", "debug this"),
            ("ai", "Here is the fix"),
        )
        manager = _make_manager()
        result = manager._extract_recent_user_prompts(msgs)
        assert result == ["debug this"]

    def test_dict_messages_human_type(self):
        # Some code paths pass plain dicts instead of LangChain messages.
        msgs = [
            {"type": "human", "content": "analyze this dataset"},
            {"type": "ai", "content": "analysis complete"},
            {"type": "human", "content": "now plot it"},
        ]
        manager = _make_manager()
        result = manager._extract_recent_user_prompts(msgs)
        assert result == ["analyze this dataset", "now plot it"]


# ---------------------------------------------------------------------------
# _get_domain_shift_threshold
# ---------------------------------------------------------------------------


class TestDomainShiftThresholdEnvVar:
    """Tests for COGTRIX_DOMAIN_SHIFT_THRESHOLD env-var configuration."""

    def test_default_is_3(self):
        manager = _make_manager()
        assert manager._get_domain_shift_threshold() == 3

    def test_env_var_respected(self):
        with patch.dict(os.environ, {"COGTRIX_DOMAIN_SHIFT_THRESHOLD": "5"}):
            manager = _make_manager()
            assert manager._get_domain_shift_threshold() == 5

    def test_env_var_zero_ignored(self):
        with patch.dict(os.environ, {"COGTRIX_DOMAIN_SHIFT_THRESHOLD": "0"}):
            manager = _make_manager()
            assert manager._get_domain_shift_threshold() == 3

    def test_env_var_negative_ignored(self):
        with patch.dict(os.environ, {"COGTRIX_DOMAIN_SHIFT_THRESHOLD": "-1"}):
            manager = _make_manager()
            assert manager._get_domain_shift_threshold() == 3

    def test_env_var_non_numeric_ignored(self):
        with patch.dict(os.environ, {"COGTRIX_DOMAIN_SHIFT_THRESHOLD": "abc"}):
            manager = _make_manager()
            assert manager._get_domain_shift_threshold() == 3


# ---------------------------------------------------------------------------
# _check_domain_shift
# ---------------------------------------------------------------------------


class TestCheckDomainShift:
    """Regression tests for domain-shift detection and summary reset trigger."""

    def _fake_llm(self):
        """Return a MagicMock LLM that returns a valid summary on invoke."""
        response = MagicMock()
        response.content = "key fact: the user prefers dark mode"
        llm = MagicMock()
        llm.invoke.return_value = response
        return llm

    def test_initial_mode_set_on_load(self):
        manager = _make_manager()
        manager.load()
        assert manager._initial_mode == "conversation"

    def test_no_switch_resets_counter(self):
        manager = _make_manager()
        manager.load()
        manager._consecutive_domain_shifts = 2  # simulate prior shifts

        # All 3 prompts classify as "conversation" — no switch from initial mode
        manager._check_domain_shift(["hello", "how are you", "thanks"])

        assert manager._consecutive_domain_shifts == 0

    def test_switch_to_different_domain_increments_counter(self):
        manager = _make_manager()
        manager.load()

        # should_switch_mode requires at least 2 prompts; 2 reasoning prompts
        # classify as "reasoning" (different from "conversation" initial mode)
        manager._check_domain_shift(["analyze this dataset", "compare approaches"])

        assert manager._consecutive_domain_shifts == 1

    def test_counter_increments_across_multiple_calls(self):
        manager = _make_manager()
        manager.load()

        # Each call passes 2 prompts so should_switch_mode always returns a suggestion
        reasoning_pairs = [
            ["analyze this dataset", "compare approaches"],
            ["evaluate the tradeoffs", "design a solution"],
            ["research alternatives", "assess the risks"],
        ]
        for pair in reasoning_pairs:
            manager._check_domain_shift(pair)

        assert manager._consecutive_domain_shifts == 3

    def test_returning_to_initial_mode_resets_counter(self):
        manager = _make_manager()
        manager.load()
        manager._consecutive_domain_shifts = 2

        # Back to conversation-mode prompts (3 needed for switch detection)
        manager._check_domain_shift(["hello", "how are you", "thanks"])

        assert manager._consecutive_domain_shifts == 0

    def test_threshold_triggers_summary_reset(self):
        manager = _make_manager()
        manager.load()
        manager._llm = self._fake_llm()

        # Pre-populate a summary so there is something to reset
        manager._summary = "old summary about code"
        manager._summary_msg_idx = 10

        # Mock the facts store so we can verify distillation is attempted
        manager._facts_store = MagicMock()

        # Simulate 3 consecutive reasoning turns (default threshold = 3)
        # Each turn passes 2 prompts so should_switch_mode returns "reasoning"
        reasoning_pairs = [
            ["analyze this dataset", "compare approaches"],
            ["evaluate the tradeoffs", "design a solution"],
            ["research alternatives", "assess the risks"],
        ]
        for pair in reasoning_pairs:
            manager._check_domain_shift(pair)

        # Summary should be cleared
        assert manager._summary is None
        assert manager._summary_msg_idx == 0
        assert manager._summary_last_updated_at is None

        # Facts store save should have been called (distillation)
        manager._facts_store.save.assert_called_once()

    def test_reset_summary_called_on_threshold(self):
        manager = _make_manager()
        manager.load()
        manager._llm = self._fake_llm()
        manager._facts_store = MagicMock()
        manager._summary = "some old content"

        with patch.object(manager, "reset_summary", wraps=manager.reset_summary) as mock_reset:
            pairs = [
                ["analyze this", "compare those"],
                ["evaluate this", "design that"],
                ["research alternatives", "assess tradeoffs"],
            ]
            for pair in pairs:
                manager._check_domain_shift(pair)

            mock_reset.assert_called_once()

    def test_empty_prompts_does_nothing(self):
        manager = _make_manager()
        manager.load()
        manager._consecutive_domain_shifts = 2

        manager._check_domain_shift([])

        # Counter should be unchanged
        assert manager._consecutive_domain_shifts == 2

    def test_env_var_threshold_2_triggers_early_reset(self):
        with patch.dict(os.environ, {"COGTRIX_DOMAIN_SHIFT_THRESHOLD": "2"}):
            manager = _make_manager()
            manager.load()
            manager._llm = self._fake_llm()
            manager._facts_store = MagicMock()
            manager._summary = "old content"

            # Only 2 shifts needed with threshold=2
            manager._check_domain_shift(["analyze this dataset", "compare approaches"])
            assert manager._consecutive_domain_shifts == 1
            assert manager._summary is not None  # not yet reset

            manager._check_domain_shift(["evaluate the tradeoffs", "design a solution"])
            assert manager._summary is None  # reset on 2nd

    def test_single_prompt_no_switch_no_counter_increment(self):
        manager = _make_manager()
        manager.load()
        manager._llm = self._fake_llm()
        manager._facts_store = MagicMock()
        manager._summary = "old content"

        # Single prompt → should_switch_mode returns None → counter stays 0
        manager._check_domain_shift(["analyze this dataset"])

        assert manager._consecutive_domain_shifts == 0
        assert manager._summary is not None  # not reset yet

    def test_facts_distilled_before_reset(self):
        """Verify that facts are extracted from the old summary before it is cleared."""
        manager = _make_manager()
        manager.load()
        manager._llm = self._fake_llm()
        manager._facts_store = MagicMock()
        manager._summary = "the user is working on a Python project in /src"

        # distill_summary is imported inside reset_summary() from cogtrix_core.memory.distillation
        with patch("cogtrix_core.memory.distillation.distill_summary") as mock_distill:
            mock_distill.return_value = ["user working on Python project"]

            pairs = [
                ["analyze this", "compare those"],
                ["evaluate this", "design that"],
                ["research alternatives", "assess tradeoffs"],
            ]
            for pair in pairs:
                manager._check_domain_shift(pair)

            # distill_summary should have been called with the old summary text
            mock_distill.assert_called_once_with(
                manager._llm, "the user is working on a Python project in /src"
            )
            # And those facts should have been saved
            manager._facts_store.save.assert_called_once()


# ---------------------------------------------------------------------------
# Integration: update() triggers domain shift check end-to-end
# ---------------------------------------------------------------------------


class TestDomainShiftViaUpdate:
    """End-to-end test: calling manager.update() eventually triggers domain shift."""

    def _fake_llm(self):
        response = MagicMock()
        response.content = "distilled fact"
        llm = MagicMock()
        llm.invoke.return_value = response
        return llm

    def test_update_accumulates_shifts_and_resets_on_threshold(self):
        manager = _make_manager()
        manager.load()
        manager._llm = self._fake_llm()
        manager._facts_store = MagicMock()
        manager._summary = "old conversation about coding"

        # Simulate 4 reasoning-domain turns via update()
        # should_switch_mode needs >= 2 prompts to return a suggestion.
        # Turn 1: 1 prompt → no switch. Turn 2-4: 2-3 prompts → reasoning detected.
        # After 4 turns: counter reaches 3 (threshold) → summary resets.
        for _ in range(4):
            manager.update(
                "analyze this architecture and compare it to alternatives",
                "Here is my analysis...",
            )

        # After 4 reasoning turns, counter reaches 3 (threshold) and summary resets
        assert manager._summary is None

    def test_single_domain_no_reset(self):
        manager = _make_manager()
        manager.load()
        manager._llm = self._fake_llm()
        manager._facts_store = MagicMock()
        manager._summary = "old summary"

        # 2 conversation prompts — no domain shift from "conversation" mode
        manager.update("hello", "hi there")
        manager.update("how are you", "I'm fine")

        # Summary should still be intact (no shift, counter reset each time)
        assert manager._summary == "old summary"
        assert not manager._facts_store.save.called

    def test_consecutive_shifts_below_threshold_no_reset(self):
        manager = _make_manager()
        manager.load()
        manager._llm = self._fake_llm()
        manager._facts_store = MagicMock()
        manager._summary = "old summary"

        # 3 reasoning prompts via update() — below default threshold of 3
        # Turn 1: 1 prompt → no switch (counter stays 0)
        # Turn 2: 2 prompts → reasoning detected (counter = 1)
        # Turn 3: 3 prompts → reasoning detected (counter = 2)
        manager.update("analyze this", "analysis here")
        manager.update("compare alternatives", "comparison here")
        manager.update("evaluate the design", "evaluation here")

        # Counter at 2 (below threshold 3) — summary not reset yet
        assert manager._consecutive_domain_shifts == 2
        assert manager._summary is not None  # not reset yet
