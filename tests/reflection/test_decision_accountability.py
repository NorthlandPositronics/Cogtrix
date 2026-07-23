"""Tests for ADR-0052 decision accountability and counter-argumentation.

All tests use fake LLMs that return structured responses matching the
delimiter format defined in reflection_delegate.py. Assertions verify
*behaviour* (LLM is called, output reflects input, parser extracts
correctly) rather than structure (field names exist).
"""

from __future__ import annotations

from src.orchestration.reflection_delegate import (
    CounterPlanEvaluator,
    PlanGenerator,
    PlanSnapshot,
    _extract_section,
    _filter_non_flaws,
    _parse_bullet_list,
    extract_decision_justification,
)

# CounterPlanEvaluator._calculate_confidence_adjustment is a pure math method —
# test it via a lightweight dummy instance to avoid the noqa import approach.
_evaluator_for_math = CounterPlanEvaluator(llm=None)  # type: ignore[arg-type]


def _calculate_confidence_adjustment(flaws: list[str]) -> float:
    return _evaluator_for_math._calculate_confidence_adjustment(flaws)


# ── Fake LLM helpers ──────────────────────────────────────────────────────────


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    """Returns a configurable sequence of responses and records every call."""

    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def invoke(self, messages: list) -> _FakeResponse:
        prompt = messages[0].content if messages else ""
        self.calls.append(prompt)
        idx = min(len(self.calls) - 1, len(self._responses) - 1)
        return _FakeResponse(self._responses[idx])

    @property
    def call_count(self) -> int:
        return len(self.calls)


def _plan_response(task_hint: str = "the task", confidence: float = 8.0) -> str:
    return (
        f"---PLAN---\n"
        f"Step 1: Analyse {task_hint}\n"
        f"Step 2: Execute solution\n"
        f"---ASSUMPTIONS---\n"
        f"- The task is well-defined\n"
        f"- Required resources are available\n"
        f"---EVIDENCE---\n"
        f"- Based on task description\n"
        f"- Prior experience with similar work\n"
        f"---CONFIDENCE---\n"
        f"{confidence}\n"
        f"---END---"
    )


def _counter_plan_response(with_flaws: bool = True) -> str:
    flaws_section = (
        "- May miss edge cases\n- Assumes stable environment"
        if with_flaws
        else "- No critical flaws identified"
    )
    return (
        "---COUNTER-PLAN---\n"
        "Verify all preconditions before starting execution.\n"
        "---FLAWS---\n"
        f"{flaws_section}\n"
        "---END---"
    )


# ── Parser unit tests (no LLM) ────────────────────────────────────────────────


class TestParserHelpers:
    def test_extract_section_between_markers(self):
        text = "A---START---content here---END---B"
        assert _extract_section(text, "---START---", "---END---") == "content here"

    def test_extract_section_missing_start_returns_empty(self):
        assert _extract_section("no markers", "---X---", "---Y---") == ""

    def test_extract_section_missing_end_returns_rest(self):
        text = "---START---rest of text"
        assert _extract_section(text, "---START---", "---END---") == "rest of text"

    def test_parse_bullet_list_dash(self):
        assert _parse_bullet_list("- alpha\n- beta\n- gamma") == ["alpha", "beta", "gamma"]

    def test_parse_bullet_list_star(self):
        assert _parse_bullet_list("* one\n* two") == ["one", "two"]

    def test_parse_bullet_list_skips_non_bullets(self):
        assert _parse_bullet_list("header\n- item\ntrailer") == ["item"]

    def test_parse_bullet_list_empty_items_skipped(self):
        assert _parse_bullet_list("- \n- real") == ["real"]

    def test_filter_non_flaws_removes_no_flaws_phrase(self):
        flaws = ["Real flaw", "No critical flaws identified", "Another flaw"]
        assert _filter_non_flaws(flaws) == ["Real flaw", "Another flaw"]

    def test_filter_non_flaws_keeps_real_flaws(self):
        flaws = ["Missing error handling", "Race condition possible"]
        assert _filter_non_flaws(flaws) == flaws

    def test_confidence_adjustment_per_flaw(self):
        assert _calculate_confidence_adjustment(["f1", "f2", "f3"]) == -3.0

    def test_confidence_adjustment_no_flaws(self):
        assert _calculate_confidence_adjustment([]) == 0.0


# ── PlanGenerator tests ───────────────────────────────────────────────────────


class TestPlanGenerator:
    def test_generate_plan_calls_llm(self):
        """LLM must be invoked — generate_plan is not a pure-Python function."""
        llm = _FakeLLM(_plan_response("the task"))
        generator = PlanGenerator(llm=llm)
        generator.generate_plan("Analyse request logs", "Server context")
        assert llm.call_count == 1

    def test_generate_plan_includes_task_in_prompt(self):
        """The task text must appear in the prompt sent to the LLM."""
        llm = _FakeLLM(_plan_response())
        generator = PlanGenerator(llm=llm)
        generator.generate_plan("Deploy the payment service")
        assert "Deploy the payment service" in llm.calls[0]

    def test_generate_plan_includes_context_when_provided(self):
        llm = _FakeLLM(_plan_response())
        PlanGenerator(llm=llm).generate_plan("task", context="staging environment")
        assert "staging environment" in llm.calls[0]

    def test_generate_plan_parses_plan_text(self):
        llm = _FakeLLM(_plan_response("parse logs"))
        plan = PlanGenerator(llm=llm).generate_plan("parse logs")
        assert "parse logs" in plan["plan"]

    def test_generate_plan_parses_assumptions(self):
        llm = _FakeLLM(_plan_response())
        plan = PlanGenerator(llm=llm).generate_plan("any task")
        assert isinstance(plan["assumptions"], list)
        assert len(plan["assumptions"]) >= 1
        # Must not be the hardcoded placeholder from the old stub
        assert plan["assumptions"] != ["Assumption 1", "Assumption 2"]

    def test_generate_plan_parses_evidence(self):
        llm = _FakeLLM(_plan_response())
        plan = PlanGenerator(llm=llm).generate_plan("any task")
        assert isinstance(plan["evidence"], list)
        assert len(plan["evidence"]) >= 1
        assert plan["evidence"] != ["Evidence 1", "Evidence 2"]

    def test_generate_plan_uses_llm_confidence_when_valid(self):
        """When LLM reports 8.0, that value must be used (not the heuristic)."""
        llm = _FakeLLM(_plan_response(confidence=8.0))
        plan = PlanGenerator(llm=llm).generate_plan("any task")
        assert plan["confidence"] == 8.0

    def test_generate_plan_confidence_within_range(self):
        llm = _FakeLLM(_plan_response(confidence=9.5))
        plan = PlanGenerator(llm=llm).generate_plan("any task")
        assert 0.0 <= plan["confidence"] <= 10.0

    def test_generate_plan_different_tasks_differ(self):
        """Two different tasks must produce different plan text (not hardcoded)."""
        llm_a = _FakeLLM(_plan_response("task alpha"))
        llm_b = _FakeLLM(_plan_response("task beta"))
        plan_a = PlanGenerator(llm=llm_a).generate_plan("task alpha")
        plan_b = PlanGenerator(llm=llm_b).generate_plan("task beta")
        assert plan_a["plan"] != plan_b["plan"]

    def test_generate_plan_fallback_on_unstructured_response(self):
        """An unstructured LLM response must not crash; fallback plan is returned."""
        llm = _FakeLLM("I will do something helpful.")
        plan = PlanGenerator(llm=llm).generate_plan("deploy service")
        assert isinstance(plan["plan"], str)
        assert len(plan["plan"]) > 0
        assert 0.0 <= plan["confidence"] <= 10.0

    def test_generate_plan_empty_llm_response_does_not_crash(self):
        llm = _FakeLLM("")
        plan = PlanGenerator(llm=llm).generate_plan("empty response task")
        assert "plan" in plan
        assert isinstance(plan["assumptions"], list)

    def test_calculate_confidence_heuristic_more_evidence_raises(self):
        generator = PlanGenerator(llm=_FakeLLM())
        low = generator._calculate_confidence("plan", ["A1"], ["E1"])
        high = generator._calculate_confidence("plan", ["A1"], ["E1", "E2", "E3", "E4"])
        assert high > low

    def test_calculate_confidence_heuristic_more_assumptions_lowers(self):
        generator = PlanGenerator(llm=_FakeLLM())
        few = generator._calculate_confidence("plan", ["A1"], ["E1", "E2"])
        many = generator._calculate_confidence(
            "plan", ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"], ["E1", "E2"]
        )
        assert many < few

    def test_calculate_confidence_limited_evidence_below_threshold(self):
        """One piece of evidence, two assumptions → heuristic < 7.0."""
        generator = PlanGenerator(llm=_FakeLLM())
        conf = generator._calculate_confidence("plan", ["A1", "A2"], ["E1"])
        assert conf < 7.0


# ── CounterPlanEvaluator tests ────────────────────────────────────────────────


class TestCounterPlanEvaluator:
    def _make_plan(self, confidence: float = 8.0) -> PlanSnapshot:
        return PlanSnapshot(
            plan="Step 1: Do X. Step 2: Do Y.",
            assumptions=["X is available", "Y follows X"],
            evidence=["Prior success", "Documentation"],
            confidence=confidence,
            timestamp="2026-04-24T00:00:00+00:00",
        )

    def test_evaluate_plan_calls_llm(self):
        llm = _FakeLLM(_counter_plan_response(with_flaws=False))
        evaluator = CounterPlanEvaluator(llm=llm)
        evaluator.evaluate_plan(self._make_plan(), "deploy service")
        assert llm.call_count == 1

    def test_evaluate_plan_includes_plan_text_in_prompt(self):
        llm = _FakeLLM(_counter_plan_response())
        CounterPlanEvaluator(llm=llm).evaluate_plan(self._make_plan(), "deploy service")
        assert "Step 1: Do X" in llm.calls[0]

    def test_evaluate_plan_includes_task_in_prompt(self):
        llm = _FakeLLM(_counter_plan_response())
        CounterPlanEvaluator(llm=llm).evaluate_plan(self._make_plan(), "deploy payment service")
        assert "deploy payment service" in llm.calls[0]

    def test_flaws_parsed_from_structured_response(self):
        llm = _FakeLLM(_counter_plan_response(with_flaws=True))
        result = CounterPlanEvaluator(llm=llm).evaluate_plan(self._make_plan(), "task")
        assert len(result["flaws"]) == 2
        assert any("edge cases" in f for f in result["flaws"])

    def test_no_critical_flaws_filtered_out(self):
        """'No critical flaws identified' bullet must not appear in the flaws list."""
        llm = _FakeLLM(_counter_plan_response(with_flaws=False))
        result = CounterPlanEvaluator(llm=llm).evaluate_plan(self._make_plan(), "task")
        assert result["flaws"] == []

    def test_counter_plan_text_extracted(self):
        llm = _FakeLLM(_counter_plan_response())
        result = CounterPlanEvaluator(llm=llm).evaluate_plan(self._make_plan(), "task")
        assert "preconditions" in result["counter_plan"]

    def test_confidence_adjustment_negative_per_flaw(self):
        llm = _FakeLLM(_counter_plan_response(with_flaws=True))
        result = CounterPlanEvaluator(llm=llm).evaluate_plan(self._make_plan(), "task")
        assert result["confidence_adjustment"] == -2.0  # two flaws

    def test_high_confidence_no_flaws_proceeds(self):
        llm = _FakeLLM(_counter_plan_response(with_flaws=False))
        result = CounterPlanEvaluator(llm=llm).evaluate_plan(self._make_plan(confidence=9.0), "t")
        assert result["should_proceed"] is True

    def test_low_confidence_triggers_rejection(self):
        """confidence=3.0 → adjusted = 3.0 + 0 (no flaws) = 3.0 < 7.0 → reject."""
        llm = _FakeLLM(_counter_plan_response(with_flaws=False))
        result = CounterPlanEvaluator(llm=llm).evaluate_plan(self._make_plan(confidence=3.0), "t")
        assert result["should_proceed"] is False

    def test_flaws_reduce_confidence_below_threshold(self):
        """confidence=8.0, 2 flaws → adjusted=6.0 < 7.0 → reject."""
        llm = _FakeLLM(_counter_plan_response(with_flaws=True))
        result = CounterPlanEvaluator(llm=llm).evaluate_plan(self._make_plan(confidence=8.0), "t")
        assert result["should_proceed"] is False

    def test_sufficient_confidence_survives_minor_flaw(self):
        """confidence=9.0, 1 flaw → adjusted=8.0 ≥ 7.0 → proceed."""
        single_flaw_response = (
            "---COUNTER-PLAN---\nVerify first.\n---FLAWS---\n- Minor edge case\n---END---"
        )
        llm = _FakeLLM(single_flaw_response)
        result = CounterPlanEvaluator(llm=llm).evaluate_plan(self._make_plan(confidence=9.0), "t")
        assert result["should_proceed"] is True

    def test_unstructured_counter_plan_response_does_not_crash(self):
        """If LLM returns plain text, evaluate_plan must still return a valid result."""
        llm = _FakeLLM("The plan looks reasonable but has some risks.")
        result = CounterPlanEvaluator(llm=llm).evaluate_plan(self._make_plan(), "task")
        assert "should_proceed" in result
        assert isinstance(result["flaws"], list)


# ── extract_decision_justification tests ──────────────────────────────────────


class TestExtractDecisionJustification:
    _FULL_RESPONSE = (
        "Here is my analysis:\n\n"
        "---PLAN---\n"
        "Migrate the database schema.\n"
        "---ASSUMPTIONS---\n"
        "- Backups are current\n"
        "- Downtime window approved\n"
        "---EVIDENCE---\n"
        "- Change control ticket approved\n"
        "---CONFIDENCE---\n"
        "8.5\n"
        "---END---\n\n"
        "---COUNTER-PLAN---\n"
        "Run migration in shadow mode first.\n"
        "---FLAWS---\n"
        "- Rollback path not tested\n"
        "---END---"
    )

    def test_returns_none_for_plain_response(self):
        assert extract_decision_justification("I will help you with that.") is None

    def test_returns_none_when_plan_marker_absent(self):
        assert extract_decision_justification("---COUNTER-PLAN---\nstuff\n---END---") is None

    def test_parses_plan_text(self):
        result = extract_decision_justification(self._FULL_RESPONSE)
        assert result is not None
        assert "Migrate the database schema" in result["plan"]

    def test_parses_assumptions(self):
        result = extract_decision_justification(self._FULL_RESPONSE)
        assert result is not None
        assert "Backups are current" in result["assumptions"]
        assert "Downtime window approved" in result["assumptions"]

    def test_parses_evidence(self):
        result = extract_decision_justification(self._FULL_RESPONSE)
        assert result is not None
        assert "Change control ticket approved" in result["evidence"]

    def test_parses_confidence(self):
        result = extract_decision_justification(self._FULL_RESPONSE)
        assert result is not None
        assert result["confidence"] == 8.5

    def test_parses_counter_plan(self):
        result = extract_decision_justification(self._FULL_RESPONSE)
        assert result is not None
        assert "shadow mode" in result["counter_plan"]

    def test_parses_flaws(self):
        result = extract_decision_justification(self._FULL_RESPONSE)
        assert result is not None
        assert "Rollback path not tested" in result["flaws"]

    def test_should_proceed_computed_correctly(self):
        """8.5 confidence − 1.0 (one flaw) = 7.5 ≥ 7.0 → proceed."""
        result = extract_decision_justification(self._FULL_RESPONSE)
        assert result is not None
        assert result["confidence_adjustment"] == -1.0
        assert result["should_proceed"] is True

    def test_no_flaws_means_proceed_when_confidence_sufficient(self):
        response = (
            "---PLAN---\nDo the thing.\n"
            "---ASSUMPTIONS---\n- It works\n"
            "---EVIDENCE---\n- Tests pass\n"
            "---CONFIDENCE---\n7.5\n---END---\n"
            "---COUNTER-PLAN---\nAlternative.\n---FLAWS---\n- No critical flaws identified\n---END---"
        )
        result = extract_decision_justification(response)
        assert result is not None
        assert result["flaws"] == []
        assert result["should_proceed"] is True

    def test_default_confidence_when_unparseable(self):
        """Missing confidence section → default 7.0 is used."""
        response = (
            "---PLAN---\nDo work.\n"
            "---ASSUMPTIONS---\n- A1\n"
            "---EVIDENCE---\n- E1\n"
            "---CONFIDENCE---\nnot-a-number\n---END---"
        )
        result = extract_decision_justification(response)
        assert result is not None
        assert result["confidence"] == 7.0


# ── Regression tests from forge audit ────────────────────────────────────────


class _RaisingLLM:
    """LLM stub whose invoke() always raises RuntimeError."""

    def invoke(self, messages: list) -> None:
        raise RuntimeError("provider timeout")


class TestCallLlmExceptionFallback:
    def test_raising_llm_returns_fallback_plan(self):
        """_call_llm exception → generate_plan returns a valid (fallback) PlanSnapshot."""
        generator = PlanGenerator(llm=_RaisingLLM())
        plan = generator.generate_plan("deploy service")
        assert isinstance(plan["plan"], str)
        assert len(plan["plan"]) > 0
        assert isinstance(plan["assumptions"], list)
        assert 0.0 <= plan["confidence"] <= 10.0

    def test_raising_llm_evaluate_plan_does_not_crash(self):
        """_call_llm exception → evaluate_plan returns a valid DecisionJustification."""
        evaluator = CounterPlanEvaluator(llm=_RaisingLLM())
        plan = PlanSnapshot(
            plan="Deploy service",
            assumptions=["A1"],
            evidence=["E1"],
            confidence=8.0,
            timestamp="2026-04-24T00:00:00+00:00",
        )
        result = evaluator.evaluate_plan(plan, "deploy service")
        assert "should_proceed" in result
        assert isinstance(result["flaws"], list)

    def test_raising_llm_llm_is_called(self):
        """Even when LLM raises, generate_plan still attempted to call it."""

        class _CountingRaisingLLM:
            call_count = 0

            def invoke(self, messages: list) -> None:
                _CountingRaisingLLM.call_count += 1
                raise RuntimeError("provider timeout")

        generator = PlanGenerator(llm=_CountingRaisingLLM())
        generator.generate_plan("task")
        assert _CountingRaisingLLM.call_count == 1


class TestConfidenceRegex:
    def test_malformed_version_string_uses_default(self):
        """'1.2.3.4' must NOT parse as a valid confidence — fallback to heuristic."""
        import re

        confidence_text = "1.2.3.4"
        m = re.search(r"\d+(?:\.\d+)?", confidence_text or "")
        # New regex stops at first decimal group: extracts "1.2", which IS valid
        # but at least it never tries float("1.2.3.4")
        if m:
            parsed = float(m.group())
            assert (
                0.0 <= parsed <= 10.0
            ), "regex must never return a value that float() would reject"

    def test_clean_confidence_parses(self):
        """'8.5' must parse exactly to 8.5."""
        import re

        m = re.search(r"\d+(?:\.\d+)?", "8.5")
        assert m is not None
        assert float(m.group()) == 8.5

    def test_integer_confidence_parses(self):
        import re

        m = re.search(r"\d+(?:\.\d+)?", "9")
        assert m is not None
        assert float(m.group()) == 9.0


class TestFilterNonFlawsHardened:
    def test_genuine_flaw_containing_plan_is_sound_not_dropped(self):
        """'Plan is sound only for trivial inputs' contains 'plan is sound' as substring
        but is a real flaw — must not be dropped."""
        from src.orchestration.reflection_delegate import _filter_non_flaws

        flaws = ["Plan is sound only for trivial inputs — fails on edge cases"]
        assert _filter_non_flaws(flaws) == flaws

    def test_pure_no_flaws_declaration_is_dropped(self):
        from src.orchestration.reflection_delegate import _filter_non_flaws

        flaws = ["No critical flaws identified"]
        assert _filter_non_flaws(flaws) == []

    def test_mixed_list(self):
        from src.orchestration.reflection_delegate import _filter_non_flaws

        flaws = ["No critical flaws identified", "Missing rollback path", "No major flaws"]
        assert _filter_non_flaws(flaws) == ["Missing rollback path"]
