"""Tests for cogtrix_core/memory/summarizer.py.

Covers:
  * The historical conversation-summary path (purpose default).
  * The new ``purpose="web_search_synthesis"`` path added in ADR-0056
    PR-D.
  * The contract violations (existing_summary with synthesis purpose,
    unknown purpose).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from cogtrix_core.memory.summarizer import generate_summary


def _fake_llm(content: str | None) -> Any:
    """Build a mock LLM whose ``.invoke(prompt)`` returns an object
    with the given ``content``. ``None`` makes the mock return an
    object with empty content (simulates an empty LLM response)."""
    llm = MagicMock()
    response = MagicMock()
    response.content = content if content is not None else ""
    llm.invoke.return_value = response
    return llm


def _make_message(role: str, content: str) -> Any:
    """Build a LangChain-shape message stub."""
    msg = MagicMock()
    msg.content = content
    type(msg).__name__ = role
    return msg


# ── Conversation path (historical behaviour) ─────────────────────────


class TestConversationPath:
    def test_returns_existing_when_empty_messages(self) -> None:
        result = generate_summary(_fake_llm("ignored"), [], existing_summary="prior")
        assert result == "prior"

    def test_first_call_no_existing_summary(self) -> None:
        llm = _fake_llm("New summary.")
        msgs = [_make_message("HumanMessage", "Hello there.")]
        result = generate_summary(llm, msgs, existing_summary=None)
        assert result == "New summary."
        # Verify the LLM saw a system + human pair.
        prompt_arg = llm.invoke.call_args[0][0]
        assert len(prompt_arg) == 2

    def test_incremental_merges_with_existing(self) -> None:
        llm = _fake_llm("Merged summary.")
        msgs = [_make_message("HumanMessage", "Another exchange.")]
        result = generate_summary(llm, msgs, existing_summary="prior")
        assert result == "Merged summary."
        # The human prompt mentions the prior summary.
        human = llm.invoke.call_args[0][0][1].content
        assert "prior" in human

    def test_empty_llm_response_returns_existing(self) -> None:
        llm = _fake_llm("")
        msgs = [_make_message("HumanMessage", "x")]
        result = generate_summary(llm, msgs, existing_summary="prior")
        assert result == "prior"

    def test_llm_raises_returns_existing(self) -> None:
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("boom")
        msgs = [_make_message("HumanMessage", "x")]
        result = generate_summary(llm, msgs, existing_summary="prior")
        assert result == "prior"


class TestOperatorInstructionExclusion:
    """#2367: operator-instruction turns must NOT be folded into the rolling
    summary — they carry operator strategy that would leak into the
    contact-visible summary. The contact-facing conversation around them is
    still summarized normally."""

    def test_operator_instruction_excluded_from_summary_input(self) -> None:
        llm = _fake_llm("Summary.")
        msgs = [
            _make_message(
                "HumanMessage",
                "[Operator instruction] Push for 90K — Marina Height 2 is the best lead.",
            ),
            _make_message("AIMessage", "Hi! Is the 2BR in Marina Height 2 still available?"),
            _make_message("HumanMessage", "Yes, asking 95K."),
        ]
        generate_summary(llm, msgs, existing_summary=None)

        # The human prompt the summarizer LLM saw = intro + formatted messages.
        prompt_text = llm.invoke.call_args[0][0][1].content
        # Operator strategy must be absent.
        assert "Operator instruction" not in prompt_text
        assert "90K" not in prompt_text
        assert "best lead" not in prompt_text
        # The contact-facing exchange is still summarized.
        assert "still available" in prompt_text
        assert "95K" in prompt_text


# ── Web search synthesis path ────────────────────────────────────────


class TestWebSearchSynthesisPath:
    def test_synthesis_returns_text(self) -> None:
        llm = _fake_llm("## Key findings\n### A\nfact [①]")
        human = _make_message("HumanMessage", "User query: x\n\n【①】 example.com\n...")
        result = generate_summary(
            llm, [human], existing_summary=None, purpose="web_search_synthesis"
        )
        assert result == "## Key findings\n### A\nfact [①]"

    def test_uses_synthesis_system_prompt(self) -> None:
        """The first message in the prompt should be a SystemMessage
        whose content is the synthesis prompt (not the conversation
        prompt)."""
        llm = _fake_llm("## Gaps\n- none")
        human = _make_message("HumanMessage", "User query: x\n\n【①】 example.com")
        generate_summary(llm, [human], purpose="web_search_synthesis")
        prompt_arg = llm.invoke.call_args[0][0]
        # System message contains the verbatim Rule 10 marker.
        sys_content = prompt_arg[0].content
        assert "CITATION-CORRECTNESS SELF-CHECK" in sys_content
        # The synthesis prompt also includes Rule 11 (language).
        assert "LANGUAGE" in sys_content

    def test_existing_summary_with_synthesis_purpose_raises(self) -> None:
        llm = _fake_llm("synthesis")
        human = _make_message("HumanMessage", "Query")
        with pytest.raises(ValueError, match="existing_summary must be None"):
            generate_summary(
                llm,
                [human],
                existing_summary="prior",
                purpose="web_search_synthesis",
            )

    def test_unknown_purpose_raises(self) -> None:
        llm = _fake_llm("x")
        with pytest.raises(ValueError, match="Unsupported purpose"):
            generate_summary(llm, [], purpose="frobnicate")

    def test_custom_timeout_respected(self) -> None:
        """Explicit timeout_seconds overrides the per-purpose default."""
        import time

        llm = MagicMock()

        def slow_invoke(_prompt: Any) -> Any:
            time.sleep(0.4)  # > 0.1s
            mock = MagicMock()
            mock.content = "Too late."
            return mock

        llm.invoke.side_effect = slow_invoke
        human = _make_message("HumanMessage", "query")
        result = generate_summary(
            llm,
            [human],
            purpose="web_search_synthesis",
            timeout_seconds=1,  # min effective val; actual sleep is 0.4s so this passes
        )
        # 1s budget > 0.4s sleep → should succeed.
        assert result == "Too late."

    def test_empty_messages_with_synthesis_returns_none(self) -> None:
        """No-messages call on the synthesis path returns
        ``existing_summary`` (None) — matches the conversation path's
        early-exit shape."""
        llm = _fake_llm("never called")
        result = generate_summary(llm, [], existing_summary=None, purpose="web_search_synthesis")
        assert result is None
        llm.invoke.assert_not_called()
