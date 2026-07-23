"""Tests for the Option A canary-substitution harness.

Issue #1587: LLM training corpora eventually ingest any fictional-entity
name published in an open-source test suite, causing canaries to lose their
discriminative power.  Option A eliminates the decay by generating a fresh
fictional product name at scenario-run time and substituting it into prompts
and assertions via template placeholders.
"""

from __future__ import annotations

from tests.evaluation.runner import (
    EvalScenario,
    Turn,
    _generate_canary_name,
    _substitute_canary,
)


def test_generate_canary_name_has_expected_shape() -> None:
    """Canary names must look like real tech products: PrefixSuffix-token-Category."""
    name = _generate_canary_name()
    parts = name.rsplit("-", 2)
    assert len(parts) == 3, f"Expected 3 dash-separated parts, got: {name}"
    prefix_suffix, token, category = parts
    assert prefix_suffix, "Prefix+Suffix must be non-empty"
    assert len(token) == 4, f"Token must be 4 hex chars, got: {token}"
    assert int(token, 16) >= 0, "Token must be valid hexadecimal"
    assert category, "Category must be non-empty"


def test_generate_canary_name_produces_different_names() -> None:
    """Two successive calls must produce different names (birthday-paradox safe)."""
    name1 = _generate_canary_name()
    name2 = _generate_canary_name()
    assert name1 != name2, "Successive canary names must differ"


def test_substitute_canary_no_placeholder_returns_unchanged() -> None:
    """Scenarios without canary placeholders must pass through unchanged."""
    scenario = EvalScenario(
        id="no-canary",
        domain="test",
        title="No canary",
        description="Plain text without placeholders.",
        user_prompt="Hello world",
        success_criteria=["response_contains: hello"],
    )
    result = _substitute_canary(scenario)
    assert result.user_prompt == "Hello world"
    assert result.success_criteria == ["response_contains: hello"]


def test_substitute_canary_replaces_single_placeholder() -> None:
    """{canary_name} is replaced in user_prompt and success_criteria."""
    scenario = EvalScenario(
        id="single-canary",
        domain="test",
        title="Single canary",
        description="Test scenario.",
        user_prompt="Tell me about {canary_name}.",
        success_criteria=[
            "response_not_contains: github.com/{canary_name_lower}",
        ],
    )
    result = _substitute_canary(scenario)
    assert "{canary_name}" not in result.user_prompt
    assert "{canary_name}" not in result.success_criteria[0]
    assert "{canary_name_lower}" not in result.success_criteria[0]
    # The lower variant should be hyphen-stripped and lowercased
    lower_name = result.user_prompt.split("about ")[1].rstrip(".")
    assert result.success_criteria[0].endswith(lower_name.lower().replace("-", ""))


def test_substitute_canary_replaces_multi_turn_placeholders() -> None:
    """Multi-turn scenarios with {canary_name} and {canary_name_2} get distinct names."""
    scenario = EvalScenario(
        id="multi-canary",
        domain="test",
        title="Multi canary",
        description="Test.",
        turns=[
            Turn(
                user_prompt="What is {canary_name}?",
                success_criteria=["response_not_contains: {canary_name_lower}.com"],
            ),
            Turn(
                user_prompt="And {canary_name_2}?",
                success_criteria=["response_not_contains: {canary_name_2_lower}.com"],
            ),
        ],
    )
    result = _substitute_canary(scenario)

    turn1_name = result.turns[0].user_prompt.split("is ")[1].rstrip("?")
    turn2_name = result.turns[1].user_prompt.split("And ")[1].rstrip("?")

    assert turn1_name != turn2_name, "Distinct placeholders must get distinct names"
    assert "{canary_name}" not in result.turns[0].user_prompt
    assert "{canary_name_2}" not in result.turns[1].user_prompt
    assert (
        result.turns[0].success_criteria[0].endswith(turn1_name.lower().replace("-", "") + ".com")
    )
    assert (
        result.turns[1].success_criteria[0].endswith(turn2_name.lower().replace("-", "") + ".com")
    )


def test_substitute_canary_preserves_scenario_id_and_metadata() -> None:
    """Non-text fields must be preserved exactly."""
    scenario = EvalScenario(
        id="meta-preservation",
        domain="regression",
        title="Meta test",
        description="Check metadata survives substitution.",
        user_prompt="Talk about {canary_name}.",
        system_prompt="You are a bot.",
        tools_required=["search_web"],
        tools_available=["http_get"],
        max_turns=15,
        timeout_seconds=90,
        tags=["smoke", "safety"],
        budget_usd_estimate=0.12,
        success_criteria=["response_contains: yes"],
    )
    result = _substitute_canary(scenario)
    assert result.id == "meta-preservation"
    assert result.domain == "regression"
    assert result.system_prompt == "You are a bot."
    assert result.tools_required == ["search_web"]
    assert result.tools_available == ["http_get"]
    assert result.max_turns == 15
    assert result.timeout_seconds == 90
    assert result.tags == ["smoke", "safety"]
    assert result.budget_usd_estimate == 0.12


def test_substitute_canary_legacy_fields() -> None:
    """Legacy single-turn scenarios using user_prompt + success_criteria are handled."""
    scenario = EvalScenario(
        id="legacy",
        domain="test",
        title="Legacy",
        description="Legacy shape.",
        user_prompt="Explain {canary_name}.",
        success_criteria=["response_not_contains: {canary_name_lower}.ai"],
    )
    result = _substitute_canary(scenario)
    assert "{canary_name}" not in result.user_prompt
    assert "{canary_name_lower}" not in result.success_criteria[0]
