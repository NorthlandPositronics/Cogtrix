"""Tests for the PM role-test scorecard's measurable-signal computation.

#2023 Track B (cycle-12 finding): gpt-oss-120b occasionally invents
tool names (`query_risk_register`, `query_rag`, harmony-leaked
`query_x<|channel|>commentary`).  The dispatcher rejects these with a
`KIND_TOOL_NAME_INVALID` ToolMessage, so no real tool ever runs — but
the prior scorecard still counted the invented call against
`extraneous_tool_calls`, producing a false-positive Cluster B
regression.  The refactor splits invented-and-rejected calls into a
separate `invalid_tool_names_count` field and excludes them from
`extraneous_tool_calls`.
"""

from __future__ import annotations

from tests.role_pm.scorecard import compute_measurable

_SCENARIO = {
    "tools_available": ["query_knowledge_base"],
    "tools_required": ["query_knowledge_base"],
    "tags": [],
}


class TestExtraneousVsInvalid:
    """The acceptance-gating ``extraneous_tool_calls`` metric must NOT
    count names the dispatcher rejected (``invalid_tool_names``).  Those
    calls never executed — they're a quality signal, not a runtime
    defect."""

    def test_invalid_name_not_counted_as_extraneous(self) -> None:
        # Model invented `query_risk_register`; dispatcher rejected it
        # at name resolution.  Should land in `invalid_tool_names_count`
        # and NOT in `extraneous_tool_calls`.
        m = compute_measurable(
            scenario=_SCENARIO,
            tool_calls_made=["query_knowledge_base", "query_risk_register"],
            invalid_tool_names=["query_risk_register"],
            final_response="ok",
            turn_count=1,
            latency_ms=100,
            criteria_passed=0,
            criteria_failed=0,
            criteria_total=0,
        )
        assert m.extraneous_tool_calls == 0
        assert m.invalid_tool_names_count == 1

    def test_real_tool_outside_whitelist_still_counted_as_extraneous(self) -> None:
        # Model called a REAL tool (web_search) that wasn't in the
        # scenario's whitelist.  The dispatcher executed it (no rejection).
        # That IS a runtime defect — keep counting it as extraneous.
        m = compute_measurable(
            scenario=_SCENARIO,
            tool_calls_made=["query_knowledge_base", "web_search"],
            invalid_tool_names=[],  # dispatcher did not reject
            final_response="ok",
            turn_count=1,
            latency_ms=100,
            criteria_passed=0,
            criteria_failed=0,
            criteria_total=0,
        )
        assert m.extraneous_tool_calls == 1
        assert m.invalid_tool_names_count == 0

    def test_mixed_invalid_plus_real_extraneous(self) -> None:
        # One invented (rejected) + one real-tool-outside-whitelist
        # (executed) on the same iteration.  They split cleanly.
        m = compute_measurable(
            scenario=_SCENARIO,
            tool_calls_made=["query_knowledge_base", "query_risk_register", "web_search"],
            invalid_tool_names=["query_risk_register"],
            final_response="ok",
            turn_count=1,
            latency_ms=100,
            criteria_passed=0,
            criteria_failed=0,
            criteria_total=0,
        )
        assert m.extraneous_tool_calls == 1  # web_search only
        assert m.invalid_tool_names_count == 1  # query_risk_register only

    def test_repeated_invalid_calls_aggregate(self) -> None:
        # The C12 gpt-oss-120b pattern: same invented name called
        # multiple times in one iteration.  Each call counts.
        m = compute_measurable(
            scenario=_SCENARIO,
            tool_calls_made=[
                "query_knowledge_base",
                "query_risk_register",
                "query_knowledge_base",
                "query_risk_register",
            ],
            invalid_tool_names=["query_risk_register", "query_risk_register"],
            final_response="ok",
            turn_count=1,
            latency_ms=100,
            criteria_passed=0,
            criteria_failed=0,
            criteria_total=0,
        )
        assert m.extraneous_tool_calls == 0
        assert m.invalid_tool_names_count == 2

    def test_legacy_call_without_invalid_param_preserves_old_behaviour(self) -> None:
        # When `invalid_tool_names` is omitted (None), every
        # non-whitelisted name continues to count as extraneous.  Keeps
        # any older harness invocation working unchanged.
        m = compute_measurable(
            scenario=_SCENARIO,
            tool_calls_made=["query_knowledge_base", "query_risk_register"],
            final_response="ok",
            turn_count=1,
            latency_ms=100,
            criteria_passed=0,
            criteria_failed=0,
            criteria_total=0,
        )
        assert m.extraneous_tool_calls == 1  # legacy: counts as extraneous
        assert m.invalid_tool_names_count == 0

    def test_invalid_tool_names_does_not_contribute_to_bug_count(self) -> None:
        from tests.role_pm.scorecard import ScenarioScorecard

        m = compute_measurable(
            scenario=_SCENARIO,
            tool_calls_made=["query_risk_register"],
            invalid_tool_names=["query_risk_register"],
            final_response="ok",
            turn_count=1,
            latency_ms=100,
            criteria_passed=0,
            criteria_failed=0,
            criteria_total=0,
        )
        sc = ScenarioScorecard(scenario_id="test", measurable=m)
        # `invalid_tool_names_count` is a quality signal, not a defect.
        # bug_count should be 0.
        assert sc.bug_count == 0

    def test_single_rejection_of_a_repeated_name_does_not_inflate(self) -> None:
        """#2027 (DeepSeek V4 Pro cycle-20 bug): when a name appears
        N times in tool_calls_made and ONE of those calls was rejected,
        ``invalid_tool_names_count`` must report 1 — not N.  The
        previous set-membership formula counted all N matching calls."""
        m = compute_measurable(
            scenario=_SCENARIO,
            # 11 calls to query_knowledge_base, 1 of which was rejected
            # (race during tool reactivation observed in C20).
            tool_calls_made=["query_knowledge_base"] * 11,
            invalid_tool_names=["query_knowledge_base"],
            final_response="ok",
            turn_count=1,
            latency_ms=100,
            criteria_passed=0,
            criteria_failed=0,
            criteria_total=0,
        )
        assert m.invalid_tool_names_count == 1, (
            "Single rejection of a repeated name must report count=1, "
            "not the total occurrences in tool_calls_made"
        )
        # All 11 calls also count toward correct_tool_calls (query_kb is
        # in tools_required); none toward extraneous (all in available).
        assert m.correct_tool_calls_count == 11
        assert m.extraneous_tool_calls == 0

    def test_partial_rejection_of_non_whitelist_name(self) -> None:
        """When N calls of a non-whitelist name happen and M are
        rejected (M ≤ N), only M land in ``invalid_tool_names_count``
        and the remaining (N-M) land in ``extraneous_tool_calls``.
        Counter-consumption avoids double-counting."""
        m = compute_measurable(
            scenario=_SCENARIO,
            tool_calls_made=["web_search"] * 5,
            invalid_tool_names=["web_search", "web_search"],
            final_response="ok",
            turn_count=1,
            latency_ms=100,
            criteria_passed=0,
            criteria_failed=0,
            criteria_total=0,
        )
        assert m.invalid_tool_names_count == 2
        # 5 web_search calls, 2 rejected → 3 actually ran outside whitelist
        assert m.extraneous_tool_calls == 3
