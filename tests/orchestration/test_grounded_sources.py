"""Tests for the ``GroundedSources`` value object (#1964 Item C).

Pins three contracts:

1. ``GroundedSources`` bundles tool results + user prompt + system
   prompt and exposes ``iter_text() / blob`` consistently.
2. ``collect_grounded_sources(messages, turn_start_idx)`` builds the
   value object from a LangChain messages list.
3. The three grounding-aware detectors —
   ``detect_unsupported_quote``, ``detect_unsupported_attribution``,
   ``detect_unverified_entities`` — treat the **system prompt** as a
   grounding source.  That's the structural fix for the false-fire
   class that PRs #1961 and #1962 papered over with refusal-aware
   short-circuits.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from cogtrix_core.orchestration.verification import (
    GroundedSources,
    collect_grounded_sources,
    detect_unsupported_attribution,
    detect_unsupported_quote,
    detect_unverified_entities,
)

# ── Value-object semantics ─────────────────────────────────────────────


class TestGroundedSourcesValueObject:
    def test_default_empty(self) -> None:
        s = GroundedSources()
        assert s.tool_results == ()
        assert s.user_prompt == ""
        assert s.system_prompt == ""
        assert s.blob == ""
        assert s.iter_text() == []

    def test_iter_text_order_tool_user_system(self) -> None:
        """Order is tool_results → user_prompt → system_prompt (most-
        specific evidence first; persona policy last)."""
        s = GroundedSources(
            tool_results=("TOOL_A", "TOOL_B"),
            user_prompt="USER",
            system_prompt="SYSTEM",
        )
        assert s.iter_text() == ["TOOL_A", "TOOL_B", "USER", "SYSTEM"]
        assert s.blob == "TOOL_A\nTOOL_B\nUSER\nSYSTEM"

    def test_iter_text_skips_empty_user_and_system(self) -> None:
        s = GroundedSources(tool_results=("TOOL",))
        # Empty prompts are dropped — no stray blank lines in the blob.
        assert s.iter_text() == ["TOOL"]
        assert s.blob == "TOOL"

    def test_from_inputs_filters_non_string_tool_results(self) -> None:
        s = GroundedSources.from_inputs(
            tool_message_contents=["ok", 123, None, "also-ok"],  # type: ignore[list-item]
            user_prompt="user",
            system_prompt="sys",
        )
        # Non-strings dropped; types normalised.
        assert s.tool_results == ("ok", "also-ok")
        assert s.user_prompt == "user"
        assert s.system_prompt == "sys"

    def test_from_inputs_treats_none_as_empty(self) -> None:
        s = GroundedSources.from_inputs()
        assert s.tool_results == ()
        assert s.user_prompt == ""
        assert s.system_prompt == ""

    def test_frozen(self) -> None:
        s = GroundedSources(tool_results=("a",))
        try:
            s.tool_results = ("b",)  # type: ignore[misc]
        except Exception:
            return
        raise AssertionError("GroundedSources should be frozen (immutable)")


# ── collect_grounded_sources from LangChain message list ──────────────


class TestCollectGroundedSources:
    def test_extracts_system_user_and_tool_results(self) -> None:
        msgs = [
            SystemMessage(content="PERSONA POLICY"),
            HumanMessage(content="Hi, how are you?"),
            AIMessage(content=""),  # tool-call placeholder
            ToolMessage(content="tool result A", tool_call_id="t1"),
            ToolMessage(content="tool result B", tool_call_id="t2"),
            AIMessage(content="final answer"),
        ]
        # turn_start_idx points at the HumanMessage at index 1.
        sources = collect_grounded_sources(msgs, turn_start_idx=1)
        assert sources.system_prompt == "PERSONA POLICY"
        assert sources.user_prompt == "Hi, how are you?"
        assert sources.tool_results == ("tool result A", "tool result B")

    def test_system_prompt_first_match_wins(self) -> None:
        """When (somehow) multiple SystemMessages exist, the FIRST is used."""
        msgs = [
            SystemMessage(content="FIRST"),
            SystemMessage(content="SECOND"),
            HumanMessage(content="user"),
        ]
        sources = collect_grounded_sources(msgs, turn_start_idx=2)
        assert sources.system_prompt == "FIRST"

    def test_no_system_message_yields_empty_system_prompt(self) -> None:
        msgs = [HumanMessage(content="user"), AIMessage(content="reply")]
        sources = collect_grounded_sources(msgs, turn_start_idx=0)
        assert sources.system_prompt == ""
        assert sources.user_prompt == "user"

    def test_tool_results_scoped_to_current_turn(self) -> None:
        """Only ToolMessages AT OR AFTER turn_start_idx are collected.

        Prior-turn evidence is intentionally out of scope — the
        per-turn grounding contract.
        """
        msgs = [
            SystemMessage(content="SYS"),
            HumanMessage(content="turn-1 user"),
            ToolMessage(content="turn-1 tool", tool_call_id="t1"),
            AIMessage(content="turn-1 final"),
            HumanMessage(content="turn-2 user"),
            ToolMessage(content="turn-2 tool", tool_call_id="t2"),
            AIMessage(content="turn-2 final"),
        ]
        sources = collect_grounded_sources(msgs, turn_start_idx=4)
        assert sources.user_prompt == "turn-2 user"
        # Only turn-2's tool result.
        assert sources.tool_results == ("turn-2 tool",)

    def test_out_of_range_turn_start_returns_empty_user_prompt(self) -> None:
        msgs = [HumanMessage(content="user")]
        sources = collect_grounded_sources(msgs, turn_start_idx=99)
        assert sources.user_prompt == ""
        assert sources.tool_results == ()


# ── System-prompt grounding semantics (the #1964 Item C upgrade) ───────


class TestSystemPromptCountsAsGrounding:
    """The key behavioural change: a verbatim quote, attribution, or
    user-supplied entity that appears in the SYSTEM PROMPT is grounded
    — not unverified.  Pre-#1964 the detectors only inspected tool
    results + user prompt, so the agent quoting its own persona policy
    in a refusal looked like a fabricated quote."""

    _PERSONA = (
        "You are FinanceBot.  pay_invoice MUST NEVER be called unless "
        "the invoice has been approved by the controller in writing."
    )

    def test_quote_grounded_by_system_prompt(self) -> None:
        """The agent's refusal verbatim-quotes its own persona policy.
        With the system prompt in scope, the quote is grounded."""
        response = (
            'I cannot proceed with that.  Per our policy, "pay_invoice '
            "MUST NEVER be called unless the invoice has been approved "
            'by the controller in writing".'
        )
        sources = GroundedSources(
            tool_results=(),
            user_prompt="Please pay invoice INV-001",
            system_prompt=self._PERSONA,
        )
        # The refusal short-circuit fires first — verify by passing a
        # non-refusal response that still contains the quote.
        non_refusal_quote = (
            'For context, our policy text reads: "pay_invoice MUST NEVER be '
            "called unless the invoice has been approved by the controller "
            'in writing".  Hope that helps clarify the workflow.'
        )
        # Without the system prompt: quote would flag.
        legacy_sources = GroundedSources(
            tool_results=(),
            user_prompt="Please pay invoice INV-001",
            # system_prompt empty
        )
        flagged_legacy = detect_unsupported_quote(non_refusal_quote, sources=legacy_sources)
        assert flagged_legacy, (
            "Pre-condition: with no system prompt, the verbatim policy quote "
            "should flag — otherwise this test is not exercising the upgrade."
        )

        # With the system prompt: quote is grounded.
        flagged_with_sys = detect_unsupported_quote(non_refusal_quote, sources=sources)
        assert not flagged_with_sys, (
            "Quote that appears verbatim in the system prompt must be treated "
            "as grounded (#1964 Item C); detector falsely flagged it."
        )
        # Suppress unused-variable warning when refusal-shape response above
        # is kept as a documentation anchor.
        assert response  # noqa: S101 — kept for narrative

    def test_attribution_grounded_by_system_prompt(self) -> None:
        """An attribution paragraph crediting the system prompt's
        policy is grounded against persona tokens, not flagged.

        The attribution-paragraph detector checks the *fraction* of
        distinctive content tokens absent from the grounded blob; the
        threshold is 0.35.  We keep the paragraph close to the persona
        wording so the missing-fraction is well under the threshold —
        the test isn't trying to game the metric, just to confirm that
        a paragraph whose distinctive content IS in the persona
        validates against the persona blob.
        """
        # Non-refusal phrasing — the refusal short-circuit doesn't
        # apply here.  Wording is tight to the persona so the missing-
        # token fraction stays under the 0.35 threshold.
        response = (
            "According to policy, pay_invoice must never be called "
            "unless the invoice has been approved by the controller in writing."
        )
        sources_no_sys = GroundedSources(tool_results=(), user_prompt="")
        sources_with_sys = GroundedSources(
            tool_results=(),
            user_prompt="",
            system_prompt=self._PERSONA,
        )
        # Pre-condition: without persona in scope, the attribution flags.
        assert detect_unsupported_attribution(
            response, sources=sources_no_sys
        ), "Pre-condition: with no system prompt, the attribution should flag"
        # Upgrade: with persona in scope, it does not.
        assert not detect_unsupported_attribution(response, sources=sources_with_sys), (
            "Attribution credits content that appears in the system prompt — "
            "must be treated as grounded (#1964 Item C)."
        )

    def test_unverified_entity_grounded_by_system_prompt(self) -> None:
        """A user-supplied entity also mentioned in the system prompt
        is grounded, not unverified.

        Scenario: the user names the controller's account number
        ``ACME-12345`` that the persona policy ALSO names; the agent
        repeating it should not be flagged as unverified.
        """
        user_prompt = "Pay invoice for vendor ACME-12345."
        # Non-refusal response so the short-circuit doesn't bypass the
        # detector — phrase the response as an informational acknowledgement.
        response = (
            "Got it — I will queue the payment request for vendor ACME-12345 "
            "and route it through the controller workflow."
        )
        persona = (
            "FinanceBot persona.  Known approved vendors include ACME-12345 "
            "(annual contract) and BETA-99999 (one-shot)."
        )
        sources_no_sys = GroundedSources(tool_results=(), user_prompt=user_prompt)
        sources_with_sys = GroundedSources(
            tool_results=(), user_prompt=user_prompt, system_prompt=persona
        )
        # Pre-condition: without persona, the entity would flag.
        flagged_no_sys = detect_unverified_entities(response, sources=sources_no_sys)
        assert "ACME-12345" in flagged_no_sys, (
            "Pre-condition: without system prompt, the entity must flag — "
            "otherwise the test is not exercising the upgrade."
        )
        # Upgrade: with persona, it does not.
        flagged_with_sys = detect_unverified_entities(response, sources=sources_with_sys)
        assert "ACME-12345" not in flagged_with_sys, (
            "Entity mentioned in the system prompt must be treated as " "grounded (#1964 Item C)."
        )


# ── Backward compatibility: legacy kwargs path still works ─────────────


class TestLegacyKwargsBackCompat:
    """Callers that still pass ``tool_message_contents=`` and
    ``user_prompt=`` keyword arguments (no ``sources=``) must continue
    to work unchanged.  Migration is incremental — we don't force
    every callsite to flip at once."""

    def test_unverified_entities_legacy_kwargs(self) -> None:
        # User names a vendor; tool result doesn't confirm; agent repeats.
        result = detect_unverified_entities(
            response_content="I will pay ACME-12345 now.",
            user_prompt="Pay invoice for vendor ACME-12345.",
            tool_message_contents=["unrelated tool output"],
        )
        assert "ACME-12345" in result

    def test_unsupported_quote_legacy_kwargs(self) -> None:
        # A verbatim quote with no grounding from tool results.
        result = detect_unsupported_quote(
            response_content='The doc states: "the deployment runs every Friday at midnight UTC".',
            tool_message_contents=["nothing useful here"],
            user_prompt="When does deployment run?",
        )
        assert result, "Quote with no grounding should flag"

    def test_unsupported_attribution_legacy_kwargs(self) -> None:
        # Attribution with distinctive tokens that no tool result supports.
        result = detect_unsupported_attribution(
            response_content=(
                "According to the engineering charter, the canary rollout window "
                "for production wasm modules is restricted to off-peak hours "
                "and supervised by the platform reliability lead."
            ),
            tool_message_contents=["completely unrelated tool output"],
            user_prompt="",
        )
        assert result, "Attribution with no grounding should flag"
