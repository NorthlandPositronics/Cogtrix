"""Tests for pair-safe context message capping."""

from __future__ import annotations

import os

os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

from langchain_core.messages import (  # noqa: E402
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from cogtrix_core.orchestration.graph import _apply_context_message_cap  # noqa: E402


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

        # #1943 PR #1: eviction prepends a SystemMessage marker.  The
        # cap budget still applies to actual conversation content; the
        # marker is metadata.
        content = [m for m in result if not isinstance(m, SystemMessage)]
        assert len(content) == 3
        assert isinstance(content[0], AIMessage)
        assert content[0].tool_calls[0]["id"] == "call_1"
        assert isinstance(content[1], ToolMessage)
        assert content[1].tool_call_id == "call_1"
        assert content[2].content == "latest"

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

        content = [m for m in result if not isinstance(m, SystemMessage)]
        assert content == [msgs[-1]]

    def test_combined_caps_use_stricter_limit(self) -> None:
        msgs = [
            HumanMessage(content="oldest"),
            HumanMessage(content="middle"),
            HumanMessage(content="latest"),
        ]

        result = _apply_context_message_cap(msgs, max_messages=2, max_tokens=100)

        content = [m for m in result if not isinstance(m, SystemMessage)]
        assert content == msgs[-2:]

    def test_oversized_latest_message_is_preserved(self) -> None:
        msgs = [
            HumanMessage(content="oldest"),
            HumanMessage(content="x" * 100),
        ]

        result = _apply_context_message_cap(msgs, max_messages=10, max_tokens=1)

        content = [m for m in result if not isinstance(m, SystemMessage)]
        assert content == [msgs[-1]]

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


class TestEvictionMarker:
    """#1943 PR #1: when the cap evicts messages, prepend a SystemMessage
    marker so the agent has a recoverable signal that data was lost.
    Without the marker the agent sees a normal-looking message history
    and may confidently synthesise content from training-data knowledge
    instead of recognising the eviction (the failure mode in #1943).
    """

    def test_marker_prepended_when_messages_dropped(self) -> None:
        msgs = [
            HumanMessage(content="oldest"),
            _ai("c1"),
            _tool("c1"),
            HumanMessage(content="middle"),
            _ai("c2"),
            _tool("c2"),
            HumanMessage(content="newest"),
        ]
        result = _apply_context_message_cap(msgs, max_messages=2)
        # First element MUST be the marker; the newest content follows.
        assert isinstance(result[0], SystemMessage)
        assert "CONTEXT NOTICE" in result[0].content
        assert "removed" in result[0].content

    def test_marker_carries_cogtrix_kind_metadata(self) -> None:
        msgs = [HumanMessage(content="oldest"), HumanMessage(content="newest")]
        result = _apply_context_message_cap(msgs, max_messages=1)
        assert isinstance(result[0], SystemMessage)
        # Same convention as the #1923 cogtrix.kind metadata family so
        # downstream detectors can route on the kind rather than on the
        # prose.
        assert result[0].additional_kwargs.get("cogtrix.kind") == "context_evicted"

    def test_marker_quotes_actual_dropped_count(self) -> None:
        msgs = [HumanMessage(content=f"m{i}") for i in range(10)]
        result = _apply_context_message_cap(msgs, max_messages=3)
        # 10 input, kept the newest 3 → 7 dropped (but the marker takes
        # one slot, so the user-facing budget honours max_messages=3 for
        # the actual conversational content).
        marker = result[0]
        assert isinstance(marker, SystemMessage)
        # Exact dropped count must appear in the marker prose for the
        # operator-debugging view.
        assert "7" in marker.content

    def test_no_marker_when_nothing_dropped(self) -> None:
        msgs = [HumanMessage(content="m1"), HumanMessage(content="m2")]
        result = _apply_context_message_cap(msgs, max_messages=5)
        # Cap larger than input → no eviction, no marker.
        assert all(not isinstance(m, SystemMessage) for m in result)
        assert result == msgs

    def test_marker_mentions_recovery_advice(self) -> None:
        """The marker prose should tell the agent what to do — re-read
        or surface the loss — not just announce the eviction."""
        msgs = [HumanMessage(content=f"m{i}") for i in range(10)]
        result = _apply_context_message_cap(msgs, max_messages=2)
        marker_content = result[0].content
        assert "re-read" in marker_content or "request a re-read" in marker_content
        assert "do NOT claim" in marker_content or "do not claim" in marker_content.lower()


class TestEvictionMarkerWithRollingSummary:
    """#1943 PR #3: when the memory layer has a rolling summary covering
    the evicted span, the marker embeds it as a semantic anchor.  Without
    a summary the marker falls back to the PR #1 prose unchanged — never
    regresses the existing eviction-marker contract.
    """

    def test_marker_embeds_summary_when_provided(self) -> None:
        msgs = [HumanMessage(content=f"m{i}") for i in range(10)]
        summary = (
            "Earlier the user asked about deploying the new cogtrix release; "
            "we discussed CI gates and the Dockerfile non-root hardening."
        )
        result = _apply_context_message_cap(msgs, max_messages=2, evicted_summary=summary)
        marker_content = result[0].content
        assert summary in marker_content
        # Anti-fabrication guard from PR #1 must remain — the summary is
        # broad-strokes context, never licence to invent specifics.
        assert "do NOT invent specifics" in marker_content

    def test_marker_falls_back_to_pr1_prose_when_summary_is_none(self) -> None:
        msgs = [HumanMessage(content=f"m{i}") for i in range(10)]
        result_none = _apply_context_message_cap(msgs, max_messages=2, evicted_summary=None)
        result_default = _apply_context_message_cap(msgs, max_messages=2)
        # When no summary is available the marker is byte-identical to the
        # PR #1 fallback (which is exercised by the existing
        # TestEvictionMarker class above).
        assert result_none[0].content == result_default[0].content

    def test_marker_falls_back_to_pr1_prose_when_summary_is_empty(self) -> None:
        msgs = [HumanMessage(content=f"m{i}") for i in range(10)]
        result_empty = _apply_context_message_cap(msgs, max_messages=2, evicted_summary="")
        result_default = _apply_context_message_cap(msgs, max_messages=2)
        assert result_empty[0].content == result_default[0].content

    def test_marker_falls_back_when_summary_is_whitespace_only(self) -> None:
        """A summary that's just whitespace conveys nothing — fall back to
        the no-summary prose rather than embedding a blank block."""
        msgs = [HumanMessage(content=f"m{i}") for i in range(10)]
        result_ws = _apply_context_message_cap(msgs, max_messages=2, evicted_summary="   \n\t  \n")
        result_default = _apply_context_message_cap(msgs, max_messages=2)
        assert result_ws[0].content == result_default[0].content

    def test_marker_kind_metadata_unchanged_with_summary(self) -> None:
        """``cogtrix.kind`` metadata must remain ``context_evicted``
        regardless of whether a summary was embedded — downstream
        detectors (planned PR #4) route on this kind."""
        msgs = [HumanMessage(content=f"m{i}") for i in range(10)]
        result = _apply_context_message_cap(
            msgs, max_messages=2, evicted_summary="some summary text"
        )
        marker = result[0]
        assert marker.additional_kwargs.get("cogtrix.kind") == "context_evicted"

    def test_marker_with_summary_still_quotes_dropped_count(self) -> None:
        """Summary embedding must not replace the dropped-count
        statistics — operators rely on the count for debugging."""
        msgs = [HumanMessage(content=f"m{i}") for i in range(10)]
        result = _apply_context_message_cap(
            msgs, max_messages=3, evicted_summary="brief earlier context"
        )
        marker_content = result[0].content
        # 10 input, kept 3, marker takes 1 slot → 7 dropped (same
        # arithmetic as test_marker_quotes_actual_dropped_count above).
        assert "7" in marker_content
