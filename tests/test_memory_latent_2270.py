"""#2270 — latent memory fix (S1) carried from the #2131 audit.

S1 (tier_cache): the compression lookup was keyed by ``tool_call_id`` alone, so a
Tier-1 snapshot entry clobbered the Tier-2 entry for the same tool call (the tier1
population loop ran after tier2). A Tier-2 routing could then reuse the longer
Tier-1 text. Fixed by keying snapshot entries by ``(tool_call_id, tier)``.

S2 (recall trailing human-only flush) is intentionally NOT changed: an existing
test (``tests/memory/test_recall.py::test_human_only_flushed_at_end``) asserts a
trailing human-only message IS flushed — a standalone user question/statement is
meant to be recallable. The #2270 S2 finding is a misdiagnosis of that intended
behaviour; see the issue comment.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from cogtrix_core.memory.tier_cache import CompressedMessage, TierCacheSnapshot, roll_forward


class TestS1TierCacheKeyedByTier:
    def test_tier2_routing_uses_tier2_cache_not_tier1(self) -> None:
        """When the same tool_call_id is cached at BOTH tiers, a Tier-2 routing
        must reuse the Tier-2 (short) text, not the clobbering Tier-1 (long) text."""
        tcid = "callX"
        tier1_text = "TIER1-LONG-" + ("x" * 400)  # distinctly long
        tier2_text = "TIER2-SHORT"
        snapshot = TierCacheSnapshot(
            tier1_messages=[CompressedMessage(tcid, "web_search", tier1_text, "ToolMessage")],
            tier2_messages=[CompressedMessage(tcid, "web_search", tier2_text, "ToolMessage")],
        )

        # Build history that forces the target tool message into Tier 2: a big
        # filler tool result eats Tier 1, the newest messages fill Tier 0 verbatim.
        target = ToolMessage(
            content="original X content " * 50, tool_call_id=tcid, name="web_search"
        )
        filler = ToolMessage(content="filler " * 2000, tool_call_id="filler", name="read_file")
        messages = [
            HumanMessage(content="anchor"),
            target,  # older → compressed (Tier 1/2)
            filler,  # older filler to exhaust Tier 1
            AIMessage(content="recent answer that stays verbatim in Tier 0"),
            HumanMessage(content="most recent prompt (Tier 0)"),
        ]

        result = roll_forward(
            messages,
            current_snapshot=snapshot,
            summary="",
            summary_msg_idx=0,
            max_context_tokens=3_000,  # small → tight Tier-1/2 budgets force routing
            llm=None,  # no LLM → cache hits or truncation only (no new compression)
        )

        # Wherever the target's tcid landed, its content must be the correctly
        # tiered cached value — never the Tier-1 text served for a Tier-2 slot.
        for cm in result.tier2_messages:
            if cm.tool_call_id == tcid:
                assert cm.content == tier2_text, "Tier-2 slot got Tier-1 (clobbered) text"
        for cm in result.tier1_messages:
            if cm.tool_call_id == tcid:
                assert cm.content == tier1_text

    def test_external_cache_remains_tier_agnostic_fallback(self) -> None:
        """A populated external compression_cache (keyed by tcid only) is still
        honored as a fallback when there's no tiered snapshot entry."""
        tcid = "extX"
        messages = [
            HumanMessage(content="anchor"),
            ToolMessage(content="big original " * 100, tool_call_id=tcid, name="web_search"),
            AIMessage(content="recent"),
            HumanMessage(content="newest"),
        ]
        result = roll_forward(
            messages,
            current_snapshot=None,
            summary="",
            summary_msg_idx=0,
            max_context_tokens=3_000,
            llm=None,
            compression_cache={tcid: "EXTERNAL-CACHED"},
        )
        found = [
            cm.content
            for cm in (result.tier1_messages + result.tier2_messages)
            if cm.tool_call_id == tcid
        ]
        # If the tool was compressed (not kept verbatim in Tier 0), the external
        # cache value is reused rather than re-truncated.
        if found:
            assert "EXTERNAL-CACHED" in found
