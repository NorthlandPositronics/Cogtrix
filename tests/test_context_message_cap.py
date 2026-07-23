"""Tests for pair-safe context message capping."""

from __future__ import annotations

import os

os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from src.orchestration.graph import _apply_context_message_cap  # noqa: E402


def _ai(tool_call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"id": tool_call_id, "name": "lookup", "args": {}}],
    )


def _tool(tool_call_id: str) -> ToolMessage:
    return ToolMessage(content="ok", tool_call_id=tool_call_id)


class TestContextMessageCap:
    def test_trims_oldest_messages_without_splitting_tool_pair(self) -> None:
        msgs = [
            HumanMessage(content="oldest"),
            _ai("call_1"),
            _tool("call_1"),
            HumanMessage(content="latest"),
        ]

        result = _apply_context_message_cap(msgs, 3)

        assert len(result) == 3
        assert isinstance(result[0], AIMessage)
        assert result[0].tool_calls[0]["id"] == "call_1"
        assert isinstance(result[1], ToolMessage)
        assert result[1].tool_call_id == "call_1"
        assert result[2].content == "latest"

    def test_truncation_logs_warning(self, caplog) -> None:
        msgs = [
            HumanMessage(content="oldest"),
            _ai("call_1"),
            _tool("call_1"),
            HumanMessage(content="latest"),
        ]

        with caplog.at_level("WARNING"):
            _apply_context_message_cap(msgs, 3)

        assert any("context_max_messages=3" in record.message for record in caplog.records)

    def test_zero_disables_cap(self) -> None:
        msgs = [HumanMessage(content="a"), _ai("call_2"), _tool("call_2")]

        result = _apply_context_message_cap(msgs, 0)

        assert result == msgs

    def test_trims_by_token_budget(self) -> None:
        msgs = [
            HumanMessage(content="oldest message"),
            HumanMessage(content="middle message"),
            HumanMessage(content="latest"),
        ]

        result = _apply_context_message_cap(msgs, max_messages=10, max_tokens=3)

        assert result == [msgs[-1]]

    def test_combined_caps_use_stricter_limit(self) -> None:
        msgs = [
            HumanMessage(content="oldest"),
            HumanMessage(content="middle"),
            HumanMessage(content="latest"),
        ]

        result = _apply_context_message_cap(msgs, max_messages=2, max_tokens=100)

        assert result == msgs[-2:]

    def test_oversized_latest_message_is_preserved(self) -> None:
        msgs = [
            HumanMessage(content="oldest"),
            HumanMessage(content="x" * 100),
        ]

        result = _apply_context_message_cap(msgs, max_messages=10, max_tokens=1)

        assert result == [msgs[-1]]

    def test_history_at_1_5x_max_tokens_preserves_latest_user_question_and_tool_pair(
        self,
    ) -> None:
        """Bug E #1711 regression — when prior-turn history exceeds the
        token cap by ~1.5×, the cap must still preserve the user's most
        recent question AND the latest AIMessage+ToolMessage pair
        (which holds the freshly-gathered evidence). Issue acceptance:
        "compression preserves the most-recent tool messages (or at
        least preserves the user's most recent question)" at 1.5×
        context_max_tokens.

        The cogtrix56 reproducer's failure mode (verbatim repetition of
        the prior turn's final answer) was caused by the thinking-break
        path firing on a stale arm — that root cause is closed by Bug
        H's PR #1723 (``_force_thinking_break[0] = False`` on new
        checkpoint). This test pins the orthogonal acceptance bound:
        even when the cap is forced to drop several chunks, the chunk
        carrying the current turn's fresh evidence must survive so the
        model has something distinct to synthesise from.
        """
        # _CHARS_PER_TOKEN is 4 (cogtrix default). Build history that
        # adds up to ~1.5 * max_tokens. Each "old turn" chunk is ~4000
        # chars (~1000 tokens) so 6 of them ≈ 6000 tokens, well over a
        # 4000-token cap.
        old_chunks: list = []
        for i in range(6):
            old_chunks.extend(
                [
                    HumanMessage(content=f"old user turn {i} " + ("x" * 3000)),
                    _ai(f"old_call_{i}"),
                    _tool(f"old_call_{i}"),
                ]
            )
        latest_user = HumanMessage(content="latest user question — surname etymology?")
        latest_ai = _ai("fresh_call")
        latest_tool = ToolMessage(
            content="Shiklo concentrated in Belarus, Grigoriev pan-Slavic, "
            "Ramasheuski Western Ukrainian",
            tool_call_id="fresh_call",
        )
        msgs = [*old_chunks, latest_user, latest_ai, latest_tool]

        # max_tokens = 1000 — old history alone is ~6× that. After
        # trimming, the newest chunks (latest user + AI+tool pair) must
        # survive even if older chunks must be dropped.
        result = _apply_context_message_cap(msgs, max_messages=0, max_tokens=1000)

        # The latest AI + ToolMessage pair is one chunk; the latest
        # HumanMessage is its own chunk. Both must be present at the
        # tail of the result, in order, with the tool_call_id intact.
        assert latest_tool in result, "Latest ToolMessage (fresh evidence) was dropped"
        assert latest_ai in result, "Latest AIMessage (fresh tool call) was dropped"
        assert latest_user in result, "Latest user question was dropped"

        # Order must be preserved (chronological).
        idx_user = result.index(latest_user)
        idx_ai = result.index(latest_ai)
        idx_tool = result.index(latest_tool)
        assert idx_user < idx_ai < idx_tool

        # The pair must remain adjacent — if the cap split tool_call_id
        # from its AIMessage, langgraph downstream would emit a repair
        # warning and the model would lose the tool grounding.
        assert idx_tool == idx_ai + 1

    def test_history_at_1_5x_max_tokens_preserves_pair_even_when_oversize(
        self,
    ) -> None:
        """Stronger boundary: when the latest AI+tool chunk by itself
        already exceeds max_tokens, the cap must still keep that whole
        chunk together rather than splitting the tool_call_id pair.
        Mirrors the cogtrix56 scenario where a single round's
        web_search returned ~6 KB of synthesis text — bigger than the
        per-call budget by itself."""
        old = [
            HumanMessage(content="old user " + "x" * 2000),
            _ai("old_call"),
            _tool("old_call"),
        ]
        latest_user = HumanMessage(content="latest q")
        latest_ai = AIMessage(
            content="",
            tool_calls=[{"id": "fresh", "name": "web_search", "args": {"query": "q"}}],
        )
        # Tool message content is intentionally huge (~5000 chars =
        # ~1250 tokens) — bigger than max_tokens by itself.
        latest_tool = ToolMessage(
            content="search-result " * 400,
            tool_call_id="fresh",
        )
        msgs = [*old, latest_user, latest_ai, latest_tool]

        result = _apply_context_message_cap(msgs, max_messages=0, max_tokens=500)

        # The latest tool pair must stay together AND be in the result,
        # even though the chunk alone exceeds the cap (the cap
        # always-keep-newest rule applies).
        assert latest_ai in result
        assert latest_tool in result
        idx_ai = result.index(latest_ai)
        idx_tool = result.index(latest_tool)
        assert idx_tool == idx_ai + 1, "AI+tool pair was split by the cap"
