"""Unit tests for the unverified-claim safety guard (Bug L).

Covers the detector, the tool-name collector, and the recovery node
factory. The integration with the orchestration graph (routing) is
exercised in higher-level tests.
"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.modifier import RemoveMessage

from src.orchestration.nodes.recovery import (
    build_handle_unverified_claim_node,
    build_handle_unverified_entity_node,
)
from src.orchestration.verification import (
    VERIFICATION_RULES,
    VerificationRule,
    _extract_specific_entities,
    collect_tool_message_contents,
    collect_tool_names_this_turn,
    detect_unverified_claim,
    detect_unverified_entities,
    format_unverified_entity_nudge,
)


class _DummyLogger:
    def __init__(self) -> None:
        self.warnings: list[tuple[object, ...]] = []
        self.infos: list[tuple[object, ...]] = []

    def warning(self, *args: object) -> None:
        self.warnings.append(args)

    def info(self, *args: object) -> None:
        self.infos.append(args)


class TestRegistry:
    """The registered set has been intentionally pruned of ``date_claim``
    — see the module docstring on why (the orchestration already injects
    today's date, the tool call adds nothing). The rules below are the
    ones where the agent does NOT have an injected ground-truth source.
    """

    def test_expected_rules_registered(self) -> None:
        names = {r.name for r in VERIFICATION_RULES}
        assert names == {
            "weather_claim",
            "exchange_rate_claim",
            "latest_version_claim",
            "file_content_claim",
        }

    def test_date_claim_intentionally_absent(self) -> None:
        # Pin the removal so an accidental re-introduction surfaces
        # as a failing test rather than re-adding the redundant tool
        # round-trip (see module docstring).
        names = {r.name for r in VERIFICATION_RULES}
        assert "date_claim" not in names

    def test_weather_claim_accepts_either_tool(self) -> None:
        rule = next(r for r in VERIFICATION_RULES if r.name == "weather_claim")
        assert "get_weather" in rule.required_tools
        assert "web_search" in rule.required_tools
        assert "get_weather" in rule.nudge_template
        assert "web_search" in rule.nudge_template

    def test_exchange_rate_required_tool(self) -> None:
        rule = next(r for r in VERIFICATION_RULES if r.name == "exchange_rate_claim")
        assert rule.required_tools == ("web_search",)
        assert "web_search" in rule.nudge_template

    def test_latest_version_required_tool(self) -> None:
        rule = next(r for r in VERIFICATION_RULES if r.name == "latest_version_claim")
        assert rule.required_tools == ("web_search",)

    def test_file_content_required_tool(self) -> None:
        rule = next(r for r in VERIFICATION_RULES if r.name == "file_content_claim")
        assert rule.required_tools == ("read_file",)


class TestWeatherClaimDetector:
    def test_sunny_with_temperature_cogtrix47(self) -> None:
        # cogtrix47 reproducer: the agent answered "weather tomorrow"
        # with concrete temperature + condition data and no
        # get_weather / web_search call would now nudge here.
        text = (
            "Tomorrow in London it will be sunny with temperatures around 22°C "
            "and light winds from the southwest."
        )
        rule = detect_unverified_claim(text, [])
        assert rule is not None
        assert rule.name == "weather_claim"

    def test_forecast_idiom_high_of(self) -> None:
        # Forecaster idiom — "high of N" — qualifies as weather data
        # even without a separate condition word.
        text = "High of 22°C, low of 11°C expected."
        assert detect_unverified_claim(text, []) is not None

    def test_temperature_with_following_weather_context(self) -> None:
        # Bare temperature is a weather signal when a weather-context
        # word follows within range.
        text = "It will be 22°C tomorrow."
        assert detect_unverified_claim(text, []) is not None

    def test_temperature_with_preceding_weather_context(self) -> None:
        # Bare temperature is a weather signal when a weather-context
        # word precedes within range.
        text = "Today's weather hit 22°C around lunchtime."
        assert detect_unverified_claim(text, []) is not None

    def test_satisfied_by_get_weather(self) -> None:
        text = "It will be sunny tomorrow with 22°C."
        assert detect_unverified_claim(text, ["get_weather"]) is None

    def test_satisfied_by_web_search_fallback(self) -> None:
        text = "It will be sunny tomorrow with 22°C."
        assert detect_unverified_claim(text, ["web_search"]) is None

    def test_negative_no_weather_token(self) -> None:
        # Generic prose mentioning "weather" without specifics
        # must not fire.
        text = "Weather data fluctuates seasonally."
        assert detect_unverified_claim(text, []) is None

    # ── cogtrix55 reproducer: temperatures in non-weather contexts ──

    def test_negative_cooking_temperature(self) -> None:
        # Reproducer for cogtrix55: a 500-calorie lunch recipe mentioned
        # the chicken's internal temperature, the orchestrator nudged
        # `get_weather`, and the response was dropped. A bare degree
        # reading in a recipe must NOT trip weather_claim.
        text = (
            "Add chicken and cook for 5-6 minutes until fully cooked " "(internal temp 165°F/74°C)."
        )
        assert detect_unverified_claim(text, []) is None

    def test_negative_oven_temperature(self) -> None:
        # Generic baking instruction.
        text = "Preheat the oven to 350°F."
        assert detect_unverified_claim(text, []) is None

    def test_negative_body_temperature(self) -> None:
        # Medical / clinical context.
        text = "A normal body temperature is around 37°C."
        assert detect_unverified_claim(text, []) is None

    def test_negative_engine_temperature(self) -> None:
        # Engineering / mechanical context.
        text = "The engine runs at 80°C under load."
        assert detect_unverified_claim(text, []) is None

    def test_negative_storage_temperature(self) -> None:
        # Storage instruction — pharmacy / food prep.
        text = "Store the product at 4°C until use."
        assert detect_unverified_claim(text, []) is None

    def test_negative_high_score_pattern_outside_weather(self) -> None:
        # "High of 22" inside a clearly non-weather sentence still
        # trips the forecast-idiom branch — accepted false positive,
        # documented here so the trade-off is explicit.
        text = "She scored a season high of 22 points."
        # Tripped by _FORECAST_IDIOM; this is an accepted FP, since the
        # surrounding response will rarely also lack web_search / get_weather.
        assert detect_unverified_claim(text, []) is not None


class TestExchangeRateClaimDetector:
    def test_explicit_conversion_cogtrix47(self) -> None:
        # cogtrix47 USD→GBP turn reproducer.
        text = "1 USD = 0.74735 GBP"
        rule = detect_unverified_claim(text, [])
        assert rule is not None
        assert rule.name == "exchange_rate_claim"

    def test_nzd_to_eur_form(self) -> None:
        text = "1 NZD = 0.5036 EUR (approximately)"
        assert detect_unverified_claim(text, []) is not None

    def test_exchange_rate_is_form(self) -> None:
        text = "The exchange rate is 1.23 from EUR to USD."
        assert detect_unverified_claim(text, []) is not None

    def test_slash_pair_form(self) -> None:
        text = "USD/GBP at 0.74735"
        assert detect_unverified_claim(text, []) is not None

    def test_satisfied_by_web_search(self) -> None:
        text = "1 USD = 0.7473 GBP."
        assert detect_unverified_claim(text, ["web_search"]) is None

    def test_negative_no_currency_context(self) -> None:
        text = "I had 100 items."
        assert detect_unverified_claim(text, []) is None


class TestLatestVersionClaimDetector:
    def test_canonical(self) -> None:
        text = "The latest version of Python is 3.13."
        rule = detect_unverified_claim(text, [])
        assert rule is not None
        assert rule.name == "latest_version_claim"

    def test_current_release_form(self) -> None:
        text = "The current release is v4.2.1"
        assert detect_unverified_claim(text, []) is not None

    def test_released_as_form(self) -> None:
        text = "Released as 2.1.0 last month."
        assert detect_unverified_claim(text, []) is not None

    def test_trailing_form(self) -> None:
        text = "3.13.2 is the most recent stable version."
        assert detect_unverified_claim(text, []) is not None

    def test_satisfied_by_web_search(self) -> None:
        text = "The latest version of Python is 3.13."
        assert detect_unverified_claim(text, ["web_search"]) is None

    def test_negative_vague_reference(self) -> None:
        # No specific version named — just "we use the latest version".
        # Not a verifiable claim.
        text = "We use the latest version of the framework."
        assert detect_unverified_claim(text, []) is None


class TestFileContentClaimDetector:
    def test_file_contains_form(self) -> None:
        text = "The file config.yaml contains the database URL."
        rule = detect_unverified_claim(text, [])
        assert rule is not None
        assert rule.name == "file_content_claim"

    def test_path_prefix(self) -> None:
        text = "src/orchestration/graph.py defines build_agent_graph."
        assert detect_unverified_claim(text, []) is not None

    def test_according_to_form(self) -> None:
        text = "According to pyproject.toml, the version is 0.2.10."
        assert detect_unverified_claim(text, []) is not None

    def test_satisfied_by_read_file(self) -> None:
        text = "The file config.yaml contains the database URL."
        assert detect_unverified_claim(text, ["read_file"]) is None

    def test_negative_no_file_path(self) -> None:
        text = "The file is small and well-organised."
        assert detect_unverified_claim(text, []) is None

    def test_negative_extension_in_prose(self) -> None:
        # A filename mention WITHOUT a content-claim verb shouldn't
        # fire. We require the agent to make a content assertion.
        text = "Open foo.py in your editor."
        assert detect_unverified_claim(text, []) is None


class TestDateClaimGoneFromBehaviour:
    """The agent already has today's date injected via the system prompt
    and per-message ``[YYYY-MM-DD HH:MM:SS UTC]`` prefix. The detector
    must NOT nudge for date assertions — calling get_current_datetime
    would return the same value the agent was just handed.
    """

    def test_today_is_date_passes_through(self) -> None:
        assert detect_unverified_claim("Today is May 20, 2026.", []) is None

    def test_as_of_date_passes_through(self) -> None:
        assert detect_unverified_claim("Exchange Rate (as of May 20, 2026):", []) is None

    def test_weekday_date_passes_through(self) -> None:
        assert detect_unverified_claim("Thursday, May 21, 2026 forecast", []) is None


class TestSharedNegatives:
    def test_negative_empty_string(self) -> None:
        assert detect_unverified_claim("", []) is None

    def test_negative_whitespace_only(self) -> None:
        assert detect_unverified_claim("   \n  ", []) is None

    def test_case_insensitive_weather(self) -> None:
        text = "TOMORROW IT WILL BE SUNNY AND 22°C."
        assert detect_unverified_claim(text, []) is not None


class TestCollectToolNamesThisTurn:
    def test_extracts_from_ai_tool_calls(self) -> None:
        msgs = [
            HumanMessage(content="what's the FX rate?"),
            AIMessage(
                content="",
                tool_calls=[{"name": "web_search", "args": {}, "id": "c1"}],
            ),
        ]
        assert collect_tool_names_this_turn(msgs, 0) == ["web_search"]

    def test_ignores_messages_before_turn_start(self) -> None:
        # Tool called in a PRIOR turn should not count.
        msgs = [
            HumanMessage(content="prior turn"),
            AIMessage(
                content="",
                tool_calls=[{"name": "web_search", "args": {}, "id": "c1"}],
            ),
            ToolMessage(content="ok", tool_call_id="c1", name="web_search"),
            HumanMessage(content="current turn — guard should not see prior call"),
            AIMessage(content="1 USD = 0.74 GBP", id="ai-1"),
        ]
        # Turn starts at index 3 (the second HumanMessage).
        assert collect_tool_names_this_turn(msgs, 3) == []

    def test_empty_list(self) -> None:
        assert collect_tool_names_this_turn([], 0) == []

    def test_out_of_bounds_idx(self) -> None:
        msgs = [HumanMessage(content="hi")]
        assert collect_tool_names_this_turn(msgs, 99) == []


class TestUnverifiedClaimRecoveryNode:
    def test_injects_nudge_on_first_detection(self) -> None:
        counter = [0]
        logger = _DummyLogger()
        node = build_handle_unverified_claim_node(counter, max_retries=1, logger=lambda: logger)

        state = {
            "messages": [
                HumanMessage(content="what's the USD/GBP rate?"),
                AIMessage(content="1 USD = 0.7473 GBP today.", id="ai-1"),
            ]
        }
        result = node(state)

        assert counter[0] == 1
        msgs = result["messages"]
        assert len(msgs) == 2
        assert msgs[0] == RemoveMessage(id="ai-1")
        assert isinstance(msgs[1], HumanMessage)
        # The exchange-rate nudge names web_search as the required tool.
        assert "web_search" in msgs[1].content
        assert logger.warnings

    def test_short_circuits_when_no_rule_matches(self) -> None:
        counter = [0]
        logger = _DummyLogger()
        node = build_handle_unverified_claim_node(counter, max_retries=1, logger=lambda: logger)

        # Plain greeting — no rule fires. Re-detection inside the node
        # is the second line of defence after the route_after_model
        # check.
        state = {
            "messages": [
                HumanMessage(content="hi"),
                AIMessage(content="Hello!", id="ai-2"),
            ]
        }
        result = node(state)
        # Counter still increments (the node fired), but the messages
        # update is empty so the response ships unchanged.
        assert counter[0] == 1
        assert result["messages"] == []

    def test_accepts_response_after_max_retries(self) -> None:
        counter = [1]  # already at max for max_retries=1
        logger = _DummyLogger()
        node = build_handle_unverified_claim_node(counter, max_retries=1, logger=lambda: logger)

        state = {
            "messages": [
                HumanMessage(content="what's the USD/GBP rate?"),
                AIMessage(content="1 USD = 0.7473 GBP today.", id="ai-3"),
            ]
        }
        result = node(state)
        # counter becomes 2 (>max_retries=1) → accept response as-is.
        assert counter[0] == 2
        assert result["messages"] == []
        # Logged the give-up message.
        assert any("accepting" in str(args).lower() for args in logger.infos)

    def test_handles_non_string_content_gracefully(self) -> None:
        counter = [0]
        logger = _DummyLogger()
        node = build_handle_unverified_claim_node(counter, max_retries=1, logger=lambda: logger)
        # Some providers return a list of parts. We accept the response
        # rather than risk a false-positive nudge loop.
        state = {
            "messages": [
                HumanMessage(content="hi"),
                AIMessage(content=[{"type": "text", "text": "..."}], id="ai-4"),
            ]
        }
        result = node(state)
        assert result["messages"] == []


class TestRulePatternProperties:
    """Structural properties we want to hold for every registered rule."""

    def test_every_rule_has_compiled_regex(self) -> None:
        for rule in VERIFICATION_RULES:
            assert isinstance(rule, VerificationRule)
            assert isinstance(rule.claim_re, re.Pattern)

    def test_every_rule_has_required_tools_and_nudge(self) -> None:
        for rule in VERIFICATION_RULES:
            assert rule.required_tools  # non-empty tuple
            assert all(isinstance(t, str) and t for t in rule.required_tools)
            assert rule.nudge_template
            # The nudge MUST name at least one acceptable tool so the
            # model knows what to call. Multi-tool rules can name them
            # all but the canonical/preferred one (index 0) is required.
            assert rule.required_tools[0] in rule.nudge_template

    def test_required_tool_back_compat_property(self) -> None:
        # ``rule.required_tool`` is a back-compat accessor that
        # returns the first acceptable tool. Existing callers (log
        # messages, debug renders) keep working without rewrite.
        for rule in VERIFICATION_RULES:
            assert rule.required_tool == rule.required_tools[0]


# ── cogtrix47 Issues 5+6 — unverified-entity detection ────────────────


class TestExtractSpecificEntities:
    """``_extract_specific_entities`` parses high-specificity
    identifiers out of a free-form text. Three pattern categories —
    each must be tight enough that ordinary prose ("Vienna's hardware
    stores", "the Soudal brand") doesn't surface as a candidate.
    """

    def test_sku_alphanumeric_with_hyphen(self) -> None:
        # cogtrix47 reproducer.
        ents = _extract_specific_entities("the product code is 1GH-EJ4 right")
        assert "1GH-EJ4" in ents

    def test_sku_iso_code(self) -> None:
        ents = _extract_specific_entities("ISO-9001 certified facility")
        assert "ISO-9001" in ents

    def test_sku_must_contain_digit(self) -> None:
        # No digit → not a SKU. ``ABC-XYZ`` is just shouting prose.
        ents = _extract_specific_entities("the ABC-XYZ standard")
        assert "ABC-XYZ" not in ents

    def test_sku_must_contain_hyphen(self) -> None:
        # No hyphen → not the SKU shape we care about. Plain
        # ``USA``, ``EUR``, etc. won't match.
        ents = _extract_specific_entities("USA EUR USD120 prices")
        assert "USA" not in ents
        assert "EUR" not in ents

    def test_store_qualifier_phrase(self) -> None:
        # cogtrix47 reproducer.
        ents = _extract_specific_entities("except the PowerTool shop by Praterstern in Vienna")
        assert "PowerTool" in ents

    def test_store_qualifier_with_multi_word_name(self) -> None:
        ents = _extract_specific_entities("visit the Acme Hardware store on Main St")
        # The qualifier-phrase regex captures the leading TitleCase
        # word(s) before the qualifier noun.
        assert any("Acme" in e for e in ents)

    def test_three_titlecase_words(self) -> None:
        # cogtrix47 reproducer for the SKU's verbose name.
        ents = _extract_specific_entities("the Soudal Fix All Silirub product line")
        assert "Soudal Fix All Silirub" in ents

    def test_two_titlecase_words_not_enough(self) -> None:
        # ``New York`` etc. is too common to be a high-specificity
        # match. The pattern requires 3+ consecutive TitleCase words.
        ents = _extract_specific_entities("travel to New York for the meeting")
        assert "New York" not in ents

    def test_empty_input(self) -> None:
        assert _extract_specific_entities("") == []


class TestDetectUnverifiedEntities:
    def test_cogtrix47_full_reproducer(self) -> None:
        user_prompt = (
            "I'll be in Vienna today and need to buy as many items of product "
            "(the Sealant from Soudal in the Soudal Fix All Silirub 1GH-EJ4 "
            "Sealant line that has sealant type sanitary sealant) as possible "
            "(except the PowerTool shop by Praterstern in Vienna). "
            "How many can I buy for $100 NZD?"
        )
        response = (
            "I could not retrieve current data. Product Identification: "
            "Soudal Fix All Silirub 1GH-EJ4 is a sanitary sealant product line. "
            "I was unable to find specific retailers in Vienna (excluding the "
            "PowerTool shop by Praterstern as requested)."
        )
        tool_contents = [
            "Soudal Fix All Crystal 300g priced at €9.98 at Hornbach Austria",
            "Soudal Sanitary Silicone 290ml priced at €7.49 at Co-op Superstores Ireland",
            "Soudal Trade Sanitary Silicone 290ml starting from €8.95 at MD O'Shea Ireland",
        ]
        unverified = detect_unverified_entities(response, user_prompt, tool_contents)
        # All three high-specificity identifiers should fire:
        # the SKU, the store qualifier phrase, the verbose product name.
        assert "1GH-EJ4" in unverified
        assert "PowerTool" in unverified
        assert "Soudal Fix All Silirub" in unverified

    def test_no_match_when_entity_appears_in_tool_result(self) -> None:
        # SKU IS in the tool result — verification satisfied.
        user_prompt = "Tell me about product code 1GH-EJ4."
        response = "Product 1GH-EJ4 is a sanitary sealant."
        tool_contents = ["Soudal sanitary sealant 1GH-EJ4 retail spec: tube size 300ml..."]
        unverified = detect_unverified_entities(response, user_prompt, tool_contents)
        assert "1GH-EJ4" not in unverified

    def test_no_match_when_response_drops_the_entity(self) -> None:
        # Agent correctly dropped the unverified entity from its
        # response — no nudge needed.
        user_prompt = "Find me the Soudal Fix All Silirub 1GH-EJ4 SKU."
        response = (
            "I couldn't find any product matching that SKU. The closest is "
            "Soudal Fix All Crystal at Hornbach Austria."
        )
        tool_contents = ["Soudal Fix All Crystal 300g priced at €9.98 at Hornbach Austria"]
        unverified = detect_unverified_entities(response, user_prompt, tool_contents)
        assert unverified == []

    def test_capped_at_max_returned(self) -> None:
        # Lots of unverified entities — caller should cap so the
        # nudge message stays scannable.
        user_prompt = (
            "Products: A1-X1, B2-Y2, C3-Z3, D4-W4, E5-V5 from the AlphaCo "
            "Bravo Charlie Delta product line at the Tango shop"
        )
        response = (
            "I found A1-X1, B2-Y2, C3-Z3, D4-W4, E5-V5 and the Tango shop "
            "carries the AlphaCo Bravo Charlie Delta line."
        )
        unverified = detect_unverified_entities(response, user_prompt, [])
        assert len(unverified) <= 3

    def test_empty_response(self) -> None:
        assert detect_unverified_entities("", "user", ["t"]) == []

    def test_empty_user_prompt(self) -> None:
        assert detect_unverified_entities("response", "", ["t"]) == []

    def test_case_insensitive_verification(self) -> None:
        # Agent capitalises differently than the tool result; should
        # still treat the entity as verified.
        user_prompt = "tell me about 1GH-EJ4"
        response = "the 1GH-EJ4 spec is..."
        tool_contents = ["product 1gh-ej4 datasheet"]
        unverified = detect_unverified_entities(response, user_prompt, tool_contents)
        assert unverified == []


class TestUnverifiedEntityRecoveryNode:
    def _msgs(self, user_prompt: str, tool_result: str, response_text: str):
        return [
            HumanMessage(content=user_prompt),
            AIMessage(
                content="",
                tool_calls=[{"name": "web_search", "args": {"q": "x"}, "id": "c1"}],
            ),
            ToolMessage(content=tool_result, tool_call_id="c1", name="web_search"),
            AIMessage(content=response_text, id="ai-final"),
        ]

    def test_injects_nudge_on_first_detection(self) -> None:
        counter = [0]
        log = _DummyLogger()
        node = build_handle_unverified_entity_node(counter, max_retries=1, logger=lambda: log)

        msgs = self._msgs(
            user_prompt="I want product 1GH-EJ4.",
            tool_result="Soudal Fix All Crystal at Hornbach Austria",
            response_text="Product 1GH-EJ4 is a sealant sold at €9.98.",
        )
        result = node({"messages": msgs})

        assert counter[0] == 1
        out = result["messages"]
        assert len(out) == 2
        assert isinstance(out[0], RemoveMessage)
        assert out[0].id == "ai-final"
        assert isinstance(out[1], HumanMessage)
        # Nudge mentions the unverified entity and the three options.
        assert "1GH-EJ4" in out[1].content
        assert "could not verify" in out[1].content
        assert "do not exist" in out[1].content
        assert log.warnings

    def test_short_circuits_when_no_entity_unverified(self) -> None:
        counter = [0]
        log = _DummyLogger()
        node = build_handle_unverified_entity_node(counter, max_retries=1, logger=lambda: log)
        msgs = self._msgs(
            user_prompt="hello",
            tool_result="anything",
            response_text="Hi there!",
        )
        result = node({"messages": msgs})
        assert result["messages"] == []

    def test_accepts_response_after_max_retries(self) -> None:
        counter = [1]  # already at max for max_retries=1
        log = _DummyLogger()
        node = build_handle_unverified_entity_node(counter, max_retries=1, logger=lambda: log)
        msgs = self._msgs(
            user_prompt="I want 1GH-EJ4.",
            tool_result="unrelated",
            response_text="1GH-EJ4 is a sealant.",
        )
        result = node({"messages": msgs})
        # Counter becomes 2 (>max_retries=1) — accept.
        assert counter[0] == 2
        assert result["messages"] == []
        # Logged the give-up message.
        assert any("retries exhausted" in str(args).lower() for args in log.infos)

    def test_handles_non_string_content(self) -> None:
        counter = [0]
        log = _DummyLogger()
        node = build_handle_unverified_entity_node(counter, max_retries=1, logger=lambda: log)
        msgs = [
            HumanMessage(content="I want 1GH-EJ4"),
            AIMessage(content=[{"type": "text", "text": "..."}], id="ai-x"),
        ]
        result = node({"messages": msgs})
        assert result["messages"] == []


class TestFormatUnverifiedEntityNudge:
    def test_single_entity_singular_grammar(self) -> None:
        msg = format_unverified_entity_nudge(["1GH-EJ4"])
        assert "one specific identifier" in msg
        assert "'1GH-EJ4'" in msg
        # No trailing "s" on "identifier".
        assert "identifiers" not in msg.split("\n")[0]

    def test_multiple_entities_plural_grammar(self) -> None:
        msg = format_unverified_entity_nudge(["1GH-EJ4", "PowerTool"])
        assert "two specific identifiers" in msg
        assert "'1GH-EJ4'" in msg
        assert "'PowerTool'" in msg

    def test_three_options_always_listed(self) -> None:
        msg = format_unverified_entity_nudge(["X"])
        # The recovery prompt enumerates (a), (b), (c) revision options.
        assert "(a)" in msg
        assert "(b)" in msg
        assert "(c)" in msg


class TestCollectToolMessageContents:
    def test_collects_only_tool_messages_in_current_turn(self) -> None:
        msgs = [
            HumanMessage(content="prior turn"),
            ToolMessage(content="prior result", tool_call_id="c0", name="t"),
            HumanMessage(content="current turn"),
            ToolMessage(content="current result A", tool_call_id="c1", name="t"),
            ToolMessage(content="current result B", tool_call_id="c2", name="t"),
            AIMessage(content="answer"),
        ]
        # Current turn starts at index 2.
        assert collect_tool_message_contents(msgs, 2) == [
            "current result A",
            "current result B",
        ]

    def test_empty_when_idx_out_of_range(self) -> None:
        assert collect_tool_message_contents([HumanMessage(content="hi")], 99) == []
