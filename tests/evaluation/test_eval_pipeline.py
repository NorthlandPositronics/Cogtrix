"""Gate 2 end-to-end integration test — runner → judge → dashboard.

Exercises the full evaluation pipeline with mocked LLMs and real scenario
YAMLs.  No live API calls are made.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from tests.evaluation.dashboard import generate_dashboard
from tests.evaluation.judge import judge_response, judge_result
from tests.evaluation.runner import (
    ModelConfig,
    load_all_scenarios,
    run_scenario,
    save_results,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


@dataclass
class _ScriptedLLM:
    """Deterministic mock LLM that yields responses in order.

    Supports ``bind_tools`` so it can be wired into ``build_agent_graph``.
    """

    responses: list[AIMessage]
    prompts: list[str] = field(default_factory=list)
    _idx: int = 0

    def __copy__(self) -> _ScriptedLLM:
        return self

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> _ScriptedLLM:
        return self

    def invoke(self, messages: list[Any], config: Any = None) -> AIMessage:
        prompt = messages[0].content if messages else ""
        self.prompts.append(str(prompt))
        if self._idx >= len(self.responses):
            raise AssertionError(
                f"Unexpected LLM call #{self._idx + 1} "
                f"(only {len(self.responses)} responses scripted)"
            )
        msg = self.responses[self._idx]
        self._idx += 1
        return msg


def _make_mock_model_config(model_id: str = "mock-model") -> ModelConfig:
    """Return a minimal ModelConfig for mocked runs."""
    return ModelConfig(
        id=model_id,
        provider="openai",
        display_name="Mock",
        tier="test",
        smoke=True,
        env_key="MOCK_API_KEY",
        model_id="mock-model",
    )


def _responses_for_scenario(scenario: Any) -> list[AIMessage]:
    """Build mock LLM responses that call all required tools then reply."""
    tool_names = scenario.tools_required
    responses: list[AIMessage] = []
    for tname in tool_names:
        responses.append(
            AIMessage(
                content="",
                tool_calls=[{"name": tname, "args": {"query": ""}, "id": f"call_{tname}"}],
            )
        )
    # Final text must contain every "contains:" success-criterion keyword
    # so the runner's binary check marks the scenario as passed.
    keywords = []
    for criterion in scenario.success_criteria:
        if criterion.startswith("contains:"):
            keywords.append(criterion[len("contains:") :].strip())
    final_text = " ".join(keywords) if keywords else "Done."
    responses.append(AIMessage(content=final_text))
    return responses


# ── Tests ────────────────────────────────────────────────────────────────────


def test_all_scenarios_loaded() -> None:
    """All Gate 2 scenario YAMLs across every domain folder are discoverable."""
    scenarios = load_all_scenarios()
    ids = {s.id for s in scenarios}
    assert ids == {
        "procurement_po_approval_basic",
        "procurement_supplier_registration",
        "procurement_three_quote_comparison",
        "finance_invoice_approval_workflow",
        "finance_budget_variance_report",
        "regression_recovery_synthesis_no_meta_analysis",
        "regression_stuck_loop_identical_tool_calls",
        "regression_deepseek_native_tool_call_format",
        "regression_per_tool_budget_cutoff",
        "safety_refuse_unauthorized_payment",
    }


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    """Full pipeline: runner → judge → dashboard with mocked LLMs."""
    scenarios = load_all_scenarios()
    # Pick the procurement smoke scenario (fewest tools → shortest mock chain)
    scenario = next(s for s in scenarios if s.id == "procurement_po_approval_basic")

    # 1. Runner — mock the agent LLM
    runner_llm = _ScriptedLLM(_responses_for_scenario(scenario))
    model = _make_mock_model_config()

    with patch.dict("os.environ", {"MOCK_API_KEY": "fake-key"}):
        with patch("tests.evaluation.runner._build_llm", return_value=runner_llm):
            result = run_scenario(scenario, model)

    assert result.error is None
    assert result.passed is True
    assert set(result.tool_calls_made) == set(scenario.tools_required)
    assert result.scenario_id == scenario.id

    # 2. Judge — mock the judge LLM (MagicMock so invoke can be called repeatedly)
    judge_llm = MagicMock()
    judge_llm.invoke.return_value = MagicMock(
        content=json.dumps({"score": 0.9, "reason": "all criteria met"})
    )
    mock_judge_cfg = _make_mock_model_config("judge-model")

    with patch.dict("os.environ", {"MOCK_API_KEY": "fake-key"}):
        with patch("tests.evaluation.judge._build_llm", return_value=judge_llm):
            with patch(
                "tests.evaluation.judge.get_model",
                return_value=mock_judge_cfg,
            ):
                score = judge_response(scenario, result, judge_model="judge-model")
                assert score == pytest.approx(0.9)

                judged = judge_result(scenario, result, judge_model="judge-model")

    assert judged.passed is True
    assert "judge_score=0.90" in judged.notes

    # 3. Dashboard — write JSONL and generate report
    results_dir = tmp_path / "results" / "v0.8.0"
    results_dir.mkdir(parents=True)
    save_results([judged], results_dir / "mock-model.jsonl")

    output_md = tmp_path / "report.md"
    output_csv = tmp_path / "report.csv"

    generate_dashboard(results_dir, output_md, output_csv)

    md = output_md.read_text()
    assert judged.model_display_name in md
    assert "100% ✓" in md  # pass indicator

    with open(output_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["model"] == "Mock"
    assert rows[0]["overall"] == "100.0"


def test_run_scenario_forwards_active_key_to_build_llm() -> None:
    """run_scenario must pass active_key into _build_llm so OpenRouter / Cerebras
    routing actually takes effect. Without this, the model falls through to its
    native env_key (e.g. ANTHROPIC_API_KEY) and fails when only OPENROUTER_API_KEY
    is set in CI."""
    scenarios = load_all_scenarios()
    scenario = next(s for s in scenarios if s.id == "procurement_po_approval_basic")

    runner_llm = _ScriptedLLM(_responses_for_scenario(scenario))
    # Anthropic-style model with no native key in env — only the active key works.
    model = ModelConfig(
        id="claude-sonnet-4-6",
        provider="anthropic",
        display_name="Claude Sonnet 4.6",
        tier="A",
        smoke=True,
        env_key="ANTHROPIC_API_KEY",
        model_id="claude-sonnet-4-6",
        openrouter_model_id="anthropic/claude-sonnet-4-6",
    )

    captured: dict[str, Any] = {}

    def fake_build_llm(model_arg: ModelConfig, **kwargs: Any) -> Any:
        captured["active_key"] = kwargs.get("active_key")
        return runner_llm

    # Note: ANTHROPIC_API_KEY is intentionally absent from the environment so
    # any fallback to native routing would raise OSError before _build_llm is
    # called the second time.
    with patch.dict("os.environ", {}, clear=False):
        with patch("tests.evaluation.runner._build_llm", side_effect=fake_build_llm):
            run_scenario(scenario, model, active_key=("OPENROUTER_API_KEY", "or-test-key"))

    assert captured["active_key"] == ("OPENROUTER_API_KEY", "or-test-key"), (
        "run_scenario dropped active_key when calling _build_llm — "
        "models will fall back to native env keys and fail when only the "
        "priority key (e.g. OPENROUTER_API_KEY) is set."
    )


def test_check_success_criteria_searches_tool_calls() -> None:
    """Success criteria like ``contains: classify_invoice`` reference tool names
    that never appear in the natural-language reply. The check must include
    tool call names and arguments in the haystack."""
    from tests.evaluation.runner import _check_success_criteria_failed

    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "classify_invoice",
                    "args": {"amount": 12500, "vendor": "Acme Supplies"},
                    "id": "call_1",
                }
            ],
        ),
        AIMessage(content="I've classified the invoice and routed it to the VP."),
    ]
    final_text = "I've classified the invoice and routed it to the VP."

    # Tool name is in tool_calls, not in the text — must still match.
    criteria_pass = [
        "contains: classify_invoice",
        "contains: 12500",
        "contains: Acme Supplies",
        "contains: VP",
        "not_contains: error",
    ]
    assert _check_success_criteria_failed(criteria_pass, final_text, messages) is False

    # A criterion that nothing satisfies still fails.
    assert (
        _check_success_criteria_failed(["contains: nonexistent_keyword"], final_text, messages)
        is True
    )

    # not_contains must also scan tool args.
    assert (
        _check_success_criteria_failed(["not_contains: Acme Supplies"], final_text, messages)
        is True
    )


def test_build_stub_tools_default_description_describes_operation() -> None:
    """Default stub descriptions name the operation the tool performs but
    do NOT instruct the model when or in what order to call them.

    Test-integrity requirement: tool descriptions in evaluation scenarios
    must describe WHAT the tool does, not script the agent's workflow.
    A description that says "use this first" or "you must call this"
    turns a finance-reasoning evaluation into a reading-comprehension
    one — see the discussion that landed this commit.
    """
    from tests.evaluation.runner import _build_stub_tools

    tools = _build_stub_tools(["classify_invoice"])
    description = tools["classify_invoice"].description.lower()
    # Names the operation — every word from the snake_case tool name
    # appears somewhere in the description (verb / noun form irrelevant).
    for word in "classify_invoice".split("_"):
        assert (
            word in description
        ), f"Default description omits operation word {word!r}: {description!r}"
    # Must NOT script the workflow.
    forbidden = ["use this first", "use this after", "must invoke", "must call", "do not answer"]
    for phrase in forbidden:
        assert phrase not in description, f"Default description leaks workflow guidance: {phrase!r}"


def test_build_stub_tools_honours_per_tool_descriptions() -> None:
    """When a scenario provides tool_descriptions, those override the default."""
    from tests.evaluation.runner import _build_stub_tools

    tools = _build_stub_tools(
        ["x_tool"],
        descriptions={"x_tool": "Custom description for X."},
    )
    assert tools["x_tool"].description == "Custom description for X."


def test_no_smoke_scenario_uses_tool_descriptions_override() -> None:
    """Smoke scenarios must rely on the central stub_tool_registry for
    descriptions and schemas — not per-scenario tool_descriptions overrides.

    The registry is the source of truth for description, input schema, and
    return shape.  A smoke scenario carrying a tool_descriptions block is a
    regression: it bypasses the registry and risks per-model description
    drift returning.  Non-smoke scenarios may still use overrides as an
    escape hatch.
    """
    from tests.evaluation.runner import load_all_scenarios

    offenders: list[str] = []
    for scenario in load_all_scenarios():
        is_smoke = "smoke" in (scenario.tags or []) or not scenario.tags
        if is_smoke and scenario.tool_descriptions:
            offenders.append(scenario.id)
    assert not offenders, (
        f"Smoke scenarios still carry tool_descriptions overrides: {offenders}. "
        f"Move these into tests/evaluation/stub_tool_registry.py instead."
    )


def test_scenario_tool_descriptions_do_not_script_workflow() -> None:
    """No scenario's tool_descriptions may bake workflow ordering into the
    tool description.  Phrases like "use this first", "use this after X",
    or "you must call this tool" turn a reasoning evaluation into a
    reading-comprehension one — the model just needs to read instructions
    rather than plan.

    This guard fires across every scenario so future YAMLs stay honest.
    """
    from tests.evaluation.runner import load_all_scenarios

    forbidden_phrases = [
        "use this first",
        "use this last",
        "use this after",
        "use this before",
        "you must invoke",
        "you must call",
        "must invoke this tool",
        "must call this tool",
        "do not answer in prose",
    ]
    offences: list[str] = []
    for scenario in load_all_scenarios():
        for tool_name, description in (scenario.tool_descriptions or {}).items():
            lower = description.lower()
            for phrase in forbidden_phrases:
                if phrase in lower:
                    offences.append(f"{scenario.id}::{tool_name} contains forbidden {phrase!r}")
    assert not offences, "Scenario tool_descriptions leak workflow guidance:\n  " + "\n  ".join(
        offences
    )


def test_eval_scenario_supports_tool_descriptions_field() -> None:
    """tool_descriptions YAML field is loaded into EvalScenario and forwarded."""
    import tempfile
    import textwrap
    from pathlib import Path

    from tests.evaluation.runner import load_scenario

    yaml_text = textwrap.dedent("""\
        id: test_scenario
        domain: test
        title: t
        description: d
        user_prompt: p
        system_prompt: s
        tools_required: [foo]
        expected_outcome: e
        success_criteria: ["contains: foo"]
        tool_descriptions:
          foo: "Concrete description for foo."
        """)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fp:
        fp.write(yaml_text)
        fp_path = Path(fp.name)

    try:
        scenario = load_scenario(fp_path)
        assert scenario.tool_descriptions == {"foo": "Concrete description for foo."}
    finally:
        fp_path.unlink()


def test_response_only_predicates_ignore_tool_call_args() -> None:
    """``response_not_contains:`` and ``response_contains:`` restrict matching
    to the user-visible reply, ignoring tool-call args.

    Without this, a regression test asserting "no <tool_call> XML in the
    user reply" would falsely fire whenever a tool's input legitimately
    contained the same string (e.g. an instruction to write XML).
    """
    from tests.evaluation.runner import _check_success_criteria_failed

    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "write_file",
                    # Tool arg legitimately contains the string we want to
                    # forbid in the user-facing reply.
                    "args": {"content": "<tool_call>example</tool_call>"},
                    "id": "call_1",
                }
            ],
        ),
        AIMessage(content="Mattermost supports OAuth 2.0 and bot accounts."),
    ]
    final_text = "Mattermost supports OAuth 2.0 and bot accounts."

    # response_not_contains must NOT see the tool arg — only the reply.
    assert (
        _check_success_criteria_failed(["response_not_contains: <tool_call>"], final_text, messages)
        is False
    )

    # If the same string DOES appear in the reply, it must trigger.
    final_text_with_xml = "Here is the answer. <tool_call>x</tool_call>"
    assert (
        _check_success_criteria_failed(
            ["response_not_contains: <tool_call>"], final_text_with_xml, messages
        )
        is True
    )

    # response_contains must match against the reply only.
    assert (
        _check_success_criteria_failed(["response_contains: oauth"], final_text, messages) is False
    )
    # A keyword that's only in the tool args, not the reply, must fail.
    assert (
        _check_success_criteria_failed(["response_contains: example"], final_text, messages) is True
    )


def test_check_success_criteria_normalizes_thousands_separators() -> None:
    """``$7,500`` in the response satisfies ``contains: 7500``.

    Models routinely format dollar amounts with commas in natural-language
    replies while structured tool-call args carry the raw number.  The
    matcher normalises both surfaces so YAML criteria are written once,
    using the canonical bare-digit form.
    """
    from tests.evaluation.runner import _check_success_criteria_failed

    messages = [AIMessage(content="The invoice is for $7,500 and was classified as standard tier.")]
    final_text = str(messages[-1].content)
    assert _check_success_criteria_failed(["contains: 7500"], final_text, messages) is False

    # Multi-comma values normalise too: 1,234,567 → 1234567.
    messages_big = [AIMessage(content="Annual budget is $1,234,567.")]
    assert (
        _check_success_criteria_failed(
            ["contains: 1234567"], str(messages_big[-1].content), messages_big
        )
        is False
    )

    # Comma between non-three-digit groups is NOT a thousands separator
    # and must not be stripped — protects ID-like strings from being
    # accidentally rewritten.
    messages_id = [AIMessage(content="Reference 12,34 stays intact.")]
    # 1234 should NOT match because "12,34" is not a thousands group.
    assert (
        _check_success_criteria_failed(
            ["contains: 1234"], str(messages_id[-1].content), messages_id
        )
        is True
    )


def test_check_success_criteria_normalizes_decimal_zeros() -> None:
    """JSON parses ``487.50`` as Python float ``487.5``; ``str(float)`` then
    normalizes to ``"487.5"``. The criteria check must treat ``487.50`` and
    ``487.5`` as equivalent so YAML scenarios stay readable."""
    from tests.evaluation.runner import _check_success_criteria_failed

    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "create_po",
                    "args": {"total": 487.50, "unit_price": 9.75, "quantity": 50},
                    "id": "c1",
                }
            ],
        ),
        AIMessage(content="PO created."),
    ]
    # YAML keeps the human-readable cents: 487.50.  Tool call serializes to 487.5.
    criteria = [
        "contains: 487.50",
        "contains: 9.75",
        "contains: 50",
    ]
    assert _check_success_criteria_failed(criteria, "PO created.", messages) is False


def test_pipeline_with_failed_scenario(tmp_path: Path) -> None:
    """A scenario that fails runner checks still flows through judge+dashboard."""
    scenarios = load_all_scenarios()
    scenario = next(s for s in scenarios if s.id == "procurement_po_approval_basic")

    # Agent LLM calls no required tools — immediate text-only response
    runner_llm = _ScriptedLLM(
        [
            AIMessage(content="Incomplete response missing keywords."),
        ]
    )
    model = _make_mock_model_config()

    with patch.dict("os.environ", {"MOCK_API_KEY": "fake-key"}):
        with patch("tests.evaluation.runner._build_llm", return_value=runner_llm):
            result = run_scenario(scenario, model)

    assert result.passed is False
    assert result.tool_selection_rate == 0.0

    # Judge falls back to heuristic (no tools called → 0.0)
    mock_judge_cfg = _make_mock_model_config("judge-model")
    with patch("tests.evaluation.judge.get_model", return_value=mock_judge_cfg):
        score = judge_response(scenario, result, judge_model="judge-model")
    assert score == pytest.approx(0.0)

    with patch("tests.evaluation.judge.get_model", return_value=mock_judge_cfg):
        judged = judge_result(scenario, result, judge_model="judge-model")
    assert judged.passed is False

    results_dir = tmp_path / "results" / "v0.8.0"
    results_dir.mkdir(parents=True)
    save_results([judged], results_dir / "mock-model.jsonl")

    output_md = tmp_path / "report.md"
    output_csv = tmp_path / "report.csv"

    generate_dashboard(results_dir, output_md, output_csv)

    md = output_md.read_text()
    assert judged.model_display_name in md
    assert "0% ⚠" in md  # fail indicator

    with open(output_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["overall"] == "0.0"


# ── tool_not_called: predicate (B2 — refusal scenarios) ──────────────────────


def test_tool_not_called_passes_when_tool_absent_from_calls() -> None:
    """``tool_not_called: pay_invoice`` succeeds when no AIMessage.tool_calls
    entry has that name, even if the tool name appears in response text."""
    from tests.evaluation.runner import _check_success_criteria_failed

    final_text = "I cannot pay_invoice without an approval record."
    messages = [
        AIMessage(
            content="",
            tool_calls=[{"name": "classify_invoice", "args": {"a": 1}, "id": "c1"}],
        ),
        AIMessage(content=final_text),
    ]

    # Tool was not called → criterion should pass.
    assert (
        _check_success_criteria_failed(["tool_not_called: pay_invoice"], final_text, messages)
        is False
    )


def test_tool_not_called_fails_when_tool_was_called() -> None:
    """``tool_not_called`` fails (returns True) when the tool DID get invoked,
    regardless of how the natural-language reply phrases the action."""
    from tests.evaluation.runner import _check_success_criteria_failed

    messages = [
        AIMessage(
            content="",
            tool_calls=[{"name": "pay_invoice", "args": {"id": "INV-001"}, "id": "c1"}],
        ),
        AIMessage(content="Done."),
    ]
    final_text = "Done."

    assert (
        _check_success_criteria_failed(["tool_not_called: pay_invoice"], final_text, messages)
        is True
    )


def test_tool_not_called_is_case_insensitive() -> None:
    """Tool-name matching for tool_not_called: lowers both sides — scenario
    YAMLs are written in lower_snake_case but defensive symmetry helps."""
    from tests.evaluation.runner import _check_success_criteria_failed

    messages = [
        AIMessage(
            content="",
            tool_calls=[{"name": "Pay_Invoice", "args": {}, "id": "c1"}],
        ),
    ]
    assert _check_success_criteria_failed(["tool_not_called: pay_invoice"], "", messages) is True


# ── max_total_tool_calls: predicate (A1 — stuck-loop guard) ──────────────────


def test_max_total_tool_calls_passes_within_limit() -> None:
    """A trace with two tool calls passes ``max_total_tool_calls: 5``."""
    from tests.evaluation.runner import _check_success_criteria_failed

    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "get_current_datetime", "args": {}, "id": "c1"},
                {"name": "get_current_datetime", "args": {}, "id": "c2"},
            ],
        ),
        AIMessage(content="Today is 2026-05-08."),
    ]
    assert (
        _check_success_criteria_failed(
            ["max_total_tool_calls: 5"], "Today is 2026-05-08.", messages
        )
        is False
    )


def test_max_total_tool_calls_fails_when_exceeded() -> None:
    """The Hermes 33-call regression: trace exceeds the bound → fail."""
    from tests.evaluation.runner import _check_success_criteria_failed

    # 33 calls across one AIMessage (the actual incident pattern was many
    # turns of 1 call each; aggregation is identical).
    big_call_list = [{"name": "get_current_datetime", "args": {}, "id": f"c{i}"} for i in range(33)]
    messages = [
        AIMessage(content="", tool_calls=big_call_list),
        AIMessage(content="Today is 2026-05-08."),
    ]
    assert (
        _check_success_criteria_failed(
            ["max_total_tool_calls: 5"], "Today is 2026-05-08.", messages
        )
        is True
    )


def test_max_total_tool_calls_malformed_value_fails_closed() -> None:
    """A non-integer bound is treated as a failed criterion (fail-closed),
    not a parse error that crashes the runner."""
    from tests.evaluation.runner import _check_success_criteria_failed

    messages = [AIMessage(content="ok")]
    assert (
        _check_success_criteria_failed(["max_total_tool_calls: not-a-number"], "ok", messages)
        is True
    )


# ── tools_available wiring (B2) ──────────────────────────────────────────────


def test_eval_scenario_loads_tools_available_field(tmp_path: Path) -> None:
    """``tools_available`` is parsed from YAML and defaults to []."""
    import textwrap

    from tests.evaluation.runner import load_scenario

    yaml_text = textwrap.dedent("""\
        id: t_avail
        domain: test
        title: t
        description: d
        user_prompt: p
        system_prompt: s
        tools_required: [foo]
        tools_available: [bar, baz]
        expected_outcome: e
        success_criteria: ["contains: foo"]
        """)
    fp = tmp_path / "t.yaml"
    fp.write_text(yaml_text)

    scenario = load_scenario(fp)
    assert scenario.tools_required == ["foo"]
    assert scenario.tools_available == ["bar", "baz"]


def test_eval_scenario_tools_available_defaults_empty(tmp_path: Path) -> None:
    """Existing YAMLs without tools_available continue to load (default []).

    Backward compatibility check — every pre-existing scenario must still
    parse after the new field was added.
    """
    import textwrap

    from tests.evaluation.runner import load_scenario

    yaml_text = textwrap.dedent("""\
        id: t_no_avail
        domain: test
        title: t
        description: d
        user_prompt: p
        system_prompt: s
        tools_required: [foo]
        expected_outcome: e
        success_criteria: ["contains: foo"]
        """)
    fp = tmp_path / "t.yaml"
    fp.write_text(yaml_text)

    scenario = load_scenario(fp)
    assert scenario.tools_available == []


# ── Cost helpers (D2) ────────────────────────────────────────────────────────


def test_sum_token_usage_aggregates_input_and_output() -> None:
    """``_sum_token_usage`` walks AIMessages and sums usage_metadata."""
    from tests.evaluation.runner import _sum_token_usage

    msgs = [
        AIMessage(
            content="a",
            usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        ),
        AIMessage(
            content="b",
            usage_metadata={"input_tokens": 200, "output_tokens": 75, "total_tokens": 275},
        ),
    ]
    prompt, completion = _sum_token_usage(msgs)
    assert prompt == 300
    assert completion == 125


def test_sum_token_usage_missing_metadata_returns_zero() -> None:
    """When no AIMessage carries usage_metadata, totals stay at zero — the
    cost ceiling treats zero as 'unknown, do not gate'."""
    from tests.evaluation.runner import _sum_token_usage

    msgs = [AIMessage(content="a"), AIMessage(content="b")]
    assert _sum_token_usage(msgs) == (0, 0)


def test_estimate_cost_usd_uses_per_1m_rates() -> None:
    """1000 input tokens at $3/1M + 500 output at $15/1M = $0.0105."""
    from tests.evaluation.runner import _estimate_cost_usd

    model = ModelConfig(
        id="t",
        provider="anthropic",
        display_name="t",
        tier="A",
        smoke=False,
        env_key="X",
        model_id="t",
        input_price_per_1m=3.0,
        output_price_per_1m=15.0,
    )
    cost = _estimate_cost_usd(model, 1000, 500)
    assert cost == pytest.approx(0.0105)


def test_estimate_cost_usd_zero_when_prices_missing() -> None:
    """A model without prices opts out of the cost ceiling — returns 0.0."""
    from tests.evaluation.runner import _estimate_cost_usd

    model = ModelConfig(
        id="t",
        provider="openai",
        display_name="t",
        tier="C",
        smoke=False,
        env_key="X",
        model_id="t",
    )
    assert _estimate_cost_usd(model, 1000, 500) == 0.0


def test_estimate_cost_usd_zero_when_no_tokens() -> None:
    """Zero token counts produce zero cost regardless of rates."""
    from tests.evaluation.runner import _estimate_cost_usd

    model = ModelConfig(
        id="t",
        provider="anthropic",
        display_name="t",
        tier="A",
        smoke=False,
        env_key="X",
        model_id="t",
        input_price_per_1m=3.0,
        output_price_per_1m=15.0,
    )
    assert _estimate_cost_usd(model, 0, 0) == 0.0


# ── D2 cost-ceiling check in ci_gate2 ────────────────────────────────────────


def test_cost_ceiling_breached_within_budget() -> None:
    """A run within 3× budget passes the ceiling check."""
    from tests.evaluation.ci_gate2 import _cost_ceiling_breached
    from tests.evaluation.runner import EvalResult, EvalScenario

    scenario = EvalScenario(
        id="x",
        domain="d",
        title="t",
        description="",
        user_prompt="",
        system_prompt="",
        tools_required=[],
        expected_outcome="",
        success_criteria=[],
        budget_usd_estimate=0.10,
    )
    result = EvalResult(
        scenario_id="x",
        model_id="m",
        model_display_name="M",
        passed=True,
        tool_calls_made=[],
        tool_calls_required=[],
        turns_used=1,
        elapsed_seconds=1.0,
        final_response="ok",
        actual_cost_usd=0.20,  # 2× budget — under the 3× ceiling
    )
    assert _cost_ceiling_breached(scenario, result) is False


def test_cost_ceiling_breached_above_3x_budget() -> None:
    """A run that overshoots 3× budget breaches the ceiling — flips to fail."""
    from tests.evaluation.ci_gate2 import _cost_ceiling_breached
    from tests.evaluation.runner import EvalResult, EvalScenario

    scenario = EvalScenario(
        id="x",
        domain="d",
        title="t",
        description="",
        user_prompt="",
        system_prompt="",
        tools_required=[],
        expected_outcome="",
        success_criteria=[],
        budget_usd_estimate=0.10,
    )
    result = EvalResult(
        scenario_id="x",
        model_id="m",
        model_display_name="M",
        passed=True,
        tool_calls_made=[],
        tool_calls_required=[],
        turns_used=33,
        elapsed_seconds=10.0,
        final_response="ok",
        actual_cost_usd=0.50,  # 5× budget
    )
    assert _cost_ceiling_breached(scenario, result) is True


def test_cost_ceiling_skipped_when_actual_cost_zero() -> None:
    """Provider that didn't return usage metadata ⇒ cost=0 ⇒ no gate fires.

    This protects us from false-failing every scenario when a new provider
    doesn't surface token counts in usage_metadata yet.
    """
    from tests.evaluation.ci_gate2 import _cost_ceiling_breached
    from tests.evaluation.runner import EvalResult, EvalScenario

    scenario = EvalScenario(
        id="x",
        domain="d",
        title="t",
        description="",
        user_prompt="",
        system_prompt="",
        tools_required=[],
        expected_outcome="",
        success_criteria=[],
        budget_usd_estimate=0.10,
    )
    result = EvalResult(
        scenario_id="x",
        model_id="m",
        model_display_name="M",
        passed=True,
        tool_calls_made=[],
        tool_calls_required=[],
        turns_used=1,
        elapsed_seconds=1.0,
        final_response="ok",
        actual_cost_usd=0.0,
    )
    assert _cost_ceiling_breached(scenario, result) is False


def test_cost_ceiling_skipped_when_budget_zero() -> None:
    """A scenario with budget=0 explicitly opts out of the ceiling."""
    from tests.evaluation.ci_gate2 import _cost_ceiling_breached
    from tests.evaluation.runner import EvalResult, EvalScenario

    scenario = EvalScenario(
        id="x",
        domain="d",
        title="t",
        description="",
        user_prompt="",
        system_prompt="",
        tools_required=[],
        expected_outcome="",
        success_criteria=[],
        budget_usd_estimate=0.0,
    )
    result = EvalResult(
        scenario_id="x",
        model_id="m",
        model_display_name="M",
        passed=True,
        tool_calls_made=[],
        tool_calls_required=[],
        turns_used=1,
        elapsed_seconds=1.0,
        final_response="ok",
        actual_cost_usd=99.99,
    )
    assert _cost_ceiling_breached(scenario, result) is False


# ── Strict gate (issue #1268): structural completion AND judge approval ─────


def _gate_eval_inputs(
    *,
    task_completion: bool,
    error: str | None = None,
    actual_cost_usd: float = 0.0,
) -> tuple[Any, Any]:
    """Build a (scenario, result) pair tuned for the strict-gate tests."""
    from tests.evaluation.runner import EvalResult, EvalScenario

    scenario = EvalScenario(
        id="gate_test",
        domain="d",
        title="t",
        description="",
        user_prompt="",
        system_prompt="",
        tools_required=["a", "b", "c"],
        expected_outcome="",
        success_criteria=[],
        budget_usd_estimate=0.10,
    )
    result = EvalResult(
        scenario_id="gate_test",
        model_id="m",
        model_display_name="M",
        passed=task_completion,
        tool_calls_made=["a", "b", "c"] if task_completion else ["a"],
        tool_calls_required=["a", "b", "c"],
        turns_used=2,
        elapsed_seconds=1.0,
        final_response="ok",
        error=error,
        task_completion=task_completion,
        actual_cost_usd=actual_cost_usd,
    )
    return scenario, result


def test_strict_gate_passes_when_completion_and_judge_both_pass() -> None:
    """Issue #1268: full happy path — all required tools called AND judge
    score above threshold → run passes."""
    from tests.evaluation.ci_gate2 import _final_passed

    scenario, result = _gate_eval_inputs(task_completion=True)
    assert _final_passed(scenario, result, score=0.95) is True


def test_strict_gate_fails_when_partial_completion_even_with_judge_pass() -> None:
    """Issue #1268 root cause: DeepSeek-V3 called 1 of 3 required tools and
    the judge gave it a 0.50 (text sounded coherent).  Today that flips
    final_passed to True; the strict gate must flip it to False so a real
    partial-completion failure cannot squeak past on judge prose alone.
    """
    from tests.evaluation.ci_gate2 import _final_passed

    scenario, result = _gate_eval_inputs(task_completion=False)
    # Judge thinks the response is fine — but only one of three tools fired.
    assert _final_passed(scenario, result, score=0.95) is False
    # Even at exactly the judge threshold, partial completion still fails.
    assert _final_passed(scenario, result, score=0.50) is False


def test_strict_gate_fails_when_judge_below_threshold() -> None:
    """Existing behaviour preserved: a judge below 0.5 fails regardless of
    structural completion.  Confirms we did not regress the qualitative
    half of the gate."""
    from tests.evaluation.ci_gate2 import _final_passed

    scenario, result = _gate_eval_inputs(task_completion=True)
    assert _final_passed(scenario, result, score=0.49) is False


def test_strict_gate_fails_when_run_errored() -> None:
    """A non-empty error trumps both gates (auth fail, timeout, etc.)."""
    from tests.evaluation.ci_gate2 import _final_passed

    scenario, result = _gate_eval_inputs(task_completion=True, error="boom")
    assert _final_passed(scenario, result, score=1.0) is False


def test_strict_gate_fails_when_cost_ceiling_breached() -> None:
    """The D2 cost-ceiling check must still fire on top of the strict gate.
    A run that calls every tool AND wins the judge but spends 5× budget
    is still a failure."""
    from tests.evaluation.ci_gate2 import _final_passed

    scenario, result = _gate_eval_inputs(
        task_completion=True,
        actual_cost_usd=0.50,  # 5× the $0.10 budget — over 3× ceiling
    )
    assert _final_passed(scenario, result, score=1.0) is False


# ── Eval temperature pinning (issue #1268, Path C item 2) ────────────────────


def test_run_scenario_does_not_pin_agent_temperature() -> None:
    """Gate 2 evaluation must NOT pin the agent LLM temperature.

    PR #1276 pinned ``temperature=0.0`` to make Gate 2 deterministic for
    issue #1268 (DeepSeek-V3 partial completion).  That pin was reverted
    after CI runs against ``next`` showed DeepSeek-V3 routed through
    OpenRouter falling into a deterministic empty-response dead-end on
    procurement_supplier_registration: 60–80% empty-response rate at
    T=0.0, 0% at the provider default.

    The strict gate (``ci_gate2._final_passed``) catches real partial
    completion regardless of temperature, so the temperature pin gave
    no benefit while introducing a worse failure mode.  This test
    pins the revert in place: a future change that re-introduces the
    ``temperature=0.0`` pin will break it, prompting a re-evaluation.
    """
    from tests.evaluation.runner import (
        ModelConfig,
        load_all_scenarios,
        run_scenario,
    )

    scenarios = load_all_scenarios()
    scenario = next(s for s in scenarios if s.id == "procurement_po_approval_basic")

    runner_llm = _ScriptedLLM(_responses_for_scenario(scenario))
    model = ModelConfig(
        id="m",
        provider="openai",
        display_name="M",
        tier="A",
        smoke=True,
        env_key="MOCK_API_KEY",
        model_id="m",
    )

    captured: dict[str, Any] = {}

    def fake_build_llm(model_arg: ModelConfig, **kwargs: Any) -> Any:
        captured["temperature"] = kwargs.get("temperature")
        return runner_llm

    with patch.dict("os.environ", {"MOCK_API_KEY": "fake-key"}):
        with patch("tests.evaluation.runner._build_llm", side_effect=fake_build_llm):
            run_scenario(scenario, model)

    # Either kwarg absent or explicitly None (provider default).
    assert captured["temperature"] is None, (
        "run_scenario must not pin temperature — pinning to 0.0 caused "
        "DeepSeek-V3 empty-response dead-ends; provider default works."
    )


# ── New scenario YAML smoke checks ───────────────────────────────────────────


def test_new_regression_and_safety_scenarios_load() -> None:
    """The four scenarios added in this PR load and parse cleanly,
    including the new tools_available and predicate types.

    Smoke check that catches typos, malformed YAML, missing required
    fields, etc., without executing a live LLM run.
    """
    from tests.evaluation.runner import load_all_scenarios

    by_id = {s.id: s for s in load_all_scenarios()}

    # A1 — stuck-loop regression
    a1 = by_id["regression_stuck_loop_identical_tool_calls"]
    assert "max_total_tool_calls: 5" in a1.success_criteria
    assert "smoke" in a1.tags

    # A2 — DeepSeek native tool-call format regression
    a2 = by_id["regression_deepseek_native_tool_call_format"]
    assert "smoke" in a2.tags
    # Asserts at least one of the unicode special tokens is forbidden
    # in the user-visible reply.
    assert any("response_not_contains: <｜tool" in c for c in a2.success_criteria)

    # A3 — per-tool budget cutoff regression (NOT in smoke)
    a3 = by_id["regression_per_tool_budget_cutoff"]
    assert "smoke" not in a3.tags
    assert a3.tools_required == ["lookup_vendor"]

    # B2 — refusal / unauthorized payment
    b2 = by_id["safety_refuse_unauthorized_payment"]
    assert "smoke" in b2.tags
    assert b2.tools_required == []
    assert "pay_invoice" in b2.tools_available
    assert "tool_not_called: pay_invoice" in b2.success_criteria


def test_procurement_supplier_registration_tool_descriptions_domain_context() -> None:
    """Regression: weak-reasoning models (DeepSeek-V3) used to fail to call
    any tools when descriptions were too generic.  Ownership of tool
    descriptions has moved from scenario YAMLs into
    ``stub_tool_registry.STUB_TOOL_REGISTRY``; this test now asserts that
    the registry entries for the procurement supplier flow continue to
    carry the domain-concrete language needed by strict-schema models.
    """
    from tests.evaluation.stub_tool_registry import STUB_TOOL_REGISTRY

    register = STUB_TOOL_REGISTRY["register_supplier"].description.lower()
    validate = STUB_TOOL_REGISTRY["validate_supplier_data"].description.lower()

    assert "supplier" in register, "register_supplier description must name the supplier domain"
    assert any(
        w in validate for w in ("completeness", "format", "valid", "required", "errors")
    ), "validate_supplier_data description must reference validation checks"


def test_invoice_approval_tool_descriptions_domain_context() -> None:
    """Regression: weak-reasoning models (DeepSeek-V3) used to stop early
    when descriptions were too generic.  Ownership of tool descriptions
    has moved from scenario YAMLs into
    ``stub_tool_registry.STUB_TOOL_REGISTRY``; this test now asserts that
    the registry entries for the finance invoice flow continue to carry
    the domain-concrete language needed by strict-schema models.
    """
    from tests.evaluation.stub_tool_registry import STUB_TOOL_REGISTRY

    classify = STUB_TOOL_REGISTRY["classify_invoice"].description.lower()
    route = STUB_TOOL_REGISTRY["route_for_approval"].description.lower()
    notify = STUB_TOOL_REGISTRY["notify_approver"].description.lower()

    assert (
        "tier" in classify or "high-value" in classify
    ), "classify_invoice description must reference tier labels"
    assert any(
        w in route for w in ("queue", "tier", "approval")
    ), "route_for_approval description must reference approval routing"
    assert any(
        w in notify for w in ("notify", "approver", "review")
    ), "notify_approver description must reference notification semantics"
