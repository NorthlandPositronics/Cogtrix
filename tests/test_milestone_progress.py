"""Tests for the milestone progress tracking feature.

Covers:
- PromptPlan / Milestone dataclasses (cogtrix_core/prompt/optimizer.py)
- _parse_plan_response parser
- optimize_prompt short-circuit and milestone-aware paths
- report_progress callback mechanism (cogtrix_core/tools/report_progress.py)
- create_report_progress_tool factory
- ActivityIndicator context helpers (cogtrix_core/ui/spinner.py)
- format_milestone_instructions / build_system_prompt (cogtrix_core/agent/core.py)
"""

from __future__ import annotations

import threading

from cogtrix_core.agent.core import build_system_prompt, format_milestone_instructions
from cogtrix_core.prompt.optimizer import (
    Milestone,
    PromptPlan,
    _parse_plan_response,
    optimize_prompt,
)
from cogtrix_core.tools.report_progress import (
    create_report_progress_tool,
    report_progress,
    set_progress_callback,
)
from cogtrix_core.ui.spinner import ActivityIndicator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeLLM:
    """Minimal LLM stub whose invoke() returns a configurable content string."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.call_count = 0

    def invoke(self, prompt: str) -> _FakeResponse:
        self.call_count += 1
        return _FakeResponse(self._content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


# ---------------------------------------------------------------------------
# PromptPlan / Milestone dataclasses
# ---------------------------------------------------------------------------


class TestPromptPlan:
    def test_prompt_plan_str_compat(self) -> None:
        """str(PromptPlan) must equal the .text field."""
        plan = PromptPlan(text="hello")
        assert str(plan) == "hello"

    def test_prompt_plan_no_milestones_has_milestones_false(self) -> None:
        """has_milestones must be False when milestones list is empty."""
        plan = PromptPlan(text="x")
        assert plan.has_milestones is False

    def test_prompt_plan_with_milestones(self) -> None:
        """has_milestones must be True when milestones are present."""
        plan = PromptPlan(text="x", milestones=[Milestone(1, "A"), Milestone(2, "B")])
        assert plan.has_milestones is True

    def test_prompt_plan_equality_with_string(self) -> None:
        """PromptPlan does not compare equal to a plain string — only to another PromptPlan."""
        plan = PromptPlan(text="hello")
        assert plan != "hello"
        assert (plan == "hello") is False

    def test_prompt_plan_equality_with_plan(self) -> None:
        """Two PromptPlans with the same text and milestones must be equal."""
        a = PromptPlan(text="x", milestones=[Milestone(1, "A")])
        b = PromptPlan(text="x", milestones=[Milestone(1, "A")])
        assert a == b

    def test_milestone_fields(self) -> None:
        m = Milestone(index=3, title="Deploy")
        assert m.index == 3
        assert m.title == "Deploy"


# ---------------------------------------------------------------------------
# _parse_plan_response
# ---------------------------------------------------------------------------


class TestParsePlanResponse:
    def test_no_milestones_returns_text_unchanged(self) -> None:
        """Plain text with no markers returns a plan with that text."""
        plan = _parse_plan_response("just a prompt", "original")
        assert plan.text == "just a prompt"
        assert plan.milestones == []

    def test_with_milestones_parses_correctly(self) -> None:
        """Response with full structured sections is parsed into text + milestones."""
        raw = (
            "---PROMPT---\n"
            "Do the thing\n"
            "---MILESTONES---\n"
            "1. Gather requirements\n"
            "2. Implement feature\n"
            "3. Write tests\n"
            "---END---"
        )
        plan = _parse_plan_response(raw, "original")
        assert plan.text == "Do the thing"
        assert len(plan.milestones) == 3
        assert plan.milestones[0].title == "Gather requirements"
        assert plan.milestones[1].title == "Implement feature"
        assert plan.milestones[2].title == "Write tests"
        # 1-based indices
        assert plan.milestones[0].index == 1
        assert plan.milestones[2].index == 3

    def test_single_milestone_discarded(self) -> None:
        """A milestones section with only 1 entry must yield an empty list."""
        raw = "---PROMPT---\n" "Do something\n" "---MILESTONES---\n" "1. Only step\n" "---END---"
        plan = _parse_plan_response(raw, "original")
        assert plan.milestones == []
        assert plan.text == "Do something"

    def test_strips_markers_when_no_milestones_section(self) -> None:
        """---PROMPT--- and ---END--- markers must be stripped from plain text."""
        raw = "---PROMPT---\nClean text\n---END---"
        plan = _parse_plan_response(raw, "original")
        assert "---PROMPT---" not in plan.text
        assert "---END---" not in plan.text
        assert "Clean text" in plan.text

    def test_empty_prompt_part_falls_back_to_original(self) -> None:
        """When the text section is blank the original prompt is preserved."""
        raw = "---MILESTONES---\n1. A\n2. B\n---END---"
        plan = _parse_plan_response(raw, "original")
        assert plan.text == "original"

    def test_two_milestones_minimum_is_kept(self) -> None:
        """Exactly 2 milestones must be retained (the minimum acceptable count)."""
        raw = (
            "---PROMPT---\n"
            "Do something\n"
            "---MILESTONES---\n"
            "1. First\n"
            "2. Second\n"
            "---END---"
        )
        plan = _parse_plan_response(raw, "original")
        assert len(plan.milestones) == 2


# ---------------------------------------------------------------------------
# optimize_prompt
# ---------------------------------------------------------------------------


class TestOptimizePrompt:
    def test_short_prompt_returns_plan_without_llm_call(self) -> None:
        """Prompts shorter than 400 chars skip the LLM entirely."""
        llm = _FakeLLM("should not be called")
        plan = optimize_prompt("short", llm)
        assert llm.call_count == 0
        assert isinstance(plan, PromptPlan)
        assert plan.text == "short"
        assert plan.milestones == []

    def test_short_prompt_has_no_milestones(self) -> None:
        """PromptPlan returned for short input must have empty milestones."""
        llm = _FakeLLM("irrelevant")
        plan = optimize_prompt("hi", llm)
        assert plan.has_milestones is False

    def test_optimize_with_milestones_parses_structured_response(self) -> None:
        """Long prompt + plan_milestones=True populates milestones from LLM reply."""
        structured_response = (
            "---PROMPT---\n"
            "Refactored long task prompt\n"
            "---MILESTONES---\n"
            "1. Research topic\n"
            "2. Draft outline\n"
            "3. Write content\n"
            "4. Review and polish\n"
            "---END---"
        )
        llm = _FakeLLM(structured_response)
        long_prompt = "x" * 450
        plan = optimize_prompt(long_prompt, llm, plan_milestones=True)

        assert llm.call_count == 1
        assert isinstance(plan, PromptPlan)
        assert plan.has_milestones is True
        assert len(plan.milestones) == 4
        assert plan.milestones[0].title == "Research topic"
        assert plan.milestones[3].title == "Review and polish"

    def test_optimize_without_milestones_flag_returns_empty_milestones(self) -> None:
        """plan_milestones=False must always produce an empty milestones list."""
        # The LLM response includes milestone markers, but because the flag is
        # False the response is treated as plain text and milestones are not parsed.
        structured_response = (
            "---PROMPT---\n"
            "Refined prompt\n"
            "---MILESTONES---\n"
            "1. Step one\n"
            "2. Step two\n"
            "---END---"
        )
        llm = _FakeLLM(structured_response)
        long_prompt = "x" * 450
        plan = optimize_prompt(long_prompt, llm, plan_milestones=False)

        assert llm.call_count == 1
        assert plan.has_milestones is False
        assert plan.milestones == []

    def test_optimize_force_bypasses_length_gate(self) -> None:
        """force=True must call the LLM even for short prompts."""
        llm = _FakeLLM("optimized short prompt result that is long enough")
        plan = optimize_prompt("tiny", llm, force=True)
        assert llm.call_count == 1
        assert isinstance(plan, PromptPlan)

    def test_optimize_exception_returns_original_as_plan(self) -> None:
        """LLM exceptions must be caught and the original prompt returned."""

        class _BrokenLLM:
            def invoke(self, prompt: str) -> None:
                raise RuntimeError("network error")

        long_prompt = "x" * 450
        plan = optimize_prompt(long_prompt, _BrokenLLM())
        assert plan.text == long_prompt
        assert plan.milestones == []


# ---------------------------------------------------------------------------
# report_progress / set_progress_callback
# ---------------------------------------------------------------------------


class TestReportProgress:
    def setup_method(self) -> None:
        # Reset the module-level callback to None before each test.
        set_progress_callback(None)  # type: ignore[arg-type]

    def teardown_method(self) -> None:
        set_progress_callback(None)  # type: ignore[arg-type]

    def test_callback_receives_milestone_and_status(self) -> None:
        """Registered callback is called with the correct (index, status) pair."""
        received: list[tuple[int, str]] = []

        def _cb(idx: int, status: str) -> None:
            received.append((idx, status))

        set_progress_callback(_cb)
        report_progress(2, "doing work")
        assert received == [(2, "doing work")]

    def test_no_callback_does_not_raise(self) -> None:
        """report_progress with no callback registered must not raise."""
        result = report_progress(1, "step one")
        assert "1" in result

    def test_callback_default_status_is_empty_string(self) -> None:
        """Omitting status argument passes an empty string to the callback."""
        received: list[tuple[int, str]] = []

        def _cb(idx: int, status: str) -> None:
            received.append((idx, status))

        set_progress_callback(_cb)
        report_progress(3)
        assert received == [(3, "")]

    def test_return_value_contains_milestone_index(self) -> None:
        """report_progress return string must mention the milestone index."""
        result = report_progress(5, "finalizing")
        assert "5" in result

    def test_thread_safe_callback_invocation(self) -> None:
        """Concurrent calls to report_progress must each trigger the callback."""
        call_log: list[int] = []
        lock = threading.Lock()

        def _cb(idx: int, status: str) -> None:
            with lock:
                call_log.append(idx)

        set_progress_callback(_cb)

        threads = [threading.Thread(target=report_progress, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sorted(call_log) == list(range(10))


# ---------------------------------------------------------------------------
# create_report_progress_tool
# ---------------------------------------------------------------------------


class TestCreateReportProgressTool:
    def test_tool_name_is_report_progress(self) -> None:
        milestones = [Milestone(1, "Alpha"), Milestone(2, "Beta")]
        tool = create_report_progress_tool(milestones)
        assert tool.name == "report_progress"

    def test_tool_description_contains_all_milestone_titles(self) -> None:
        milestones = [Milestone(1, "Research"), Milestone(2, "Implement"), Milestone(3, "Test")]
        tool = create_report_progress_tool(milestones)
        desc = tool.description
        assert "Research" in desc
        assert "Implement" in desc
        assert "Test" in desc

    def test_tool_description_contains_milestone_indices(self) -> None:
        milestones = [Milestone(1, "First"), Milestone(2, "Second")]
        tool = create_report_progress_tool(milestones)
        desc = tool.description
        assert "1" in desc
        assert "2" in desc

    def test_tool_is_callable(self) -> None:
        """The created tool must be directly invocable."""
        milestones = [Milestone(1, "Step")]
        tool = create_report_progress_tool(milestones)
        result = tool.invoke({"milestone_index": 1, "status": "starting"})
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# ActivityIndicator context helpers
# ---------------------------------------------------------------------------


class TestSpinnerContext:
    def setup_method(self) -> None:
        self.spinner = ActivityIndicator()

    def test_set_context_stores_value(self) -> None:
        """set_context must update _context on the instance."""
        self.spinner.set_context("[1/3] Testing")
        assert self.spinner._context == "[1/3] Testing"

    def test_clear_context_resets_to_empty_string(self) -> None:
        """clear_context must set _context back to an empty string."""
        self.spinner.set_context("[2/3] Running")
        self.spinner.clear_context()
        assert self.spinner._context == ""

    def test_context_starts_empty(self) -> None:
        """A freshly created spinner must have an empty _context."""
        assert self.spinner._context == ""

    def test_set_context_overwrites_previous_value(self) -> None:
        self.spinner.set_context("first")
        self.spinner.set_context("second")
        assert self.spinner._context == "second"


# ---------------------------------------------------------------------------
# format_milestone_instructions
# ---------------------------------------------------------------------------


class TestFormatMilestoneInstructions:
    def test_output_contains_header(self) -> None:
        milestones = [Milestone(1, "A"), Milestone(2, "B"), Milestone(3, "C")]
        output = format_milestone_instructions(milestones)
        assert "## Progress Milestones" in output

    def test_output_contains_all_milestone_titles(self) -> None:
        milestones = [Milestone(1, "Alpha"), Milestone(2, "Beta"), Milestone(3, "Gamma")]
        output = format_milestone_instructions(milestones)
        assert "Alpha" in output
        assert "Beta" in output
        assert "Gamma" in output

    def test_output_contains_all_milestone_indices(self) -> None:
        milestones = [Milestone(1, "One"), Milestone(2, "Two"), Milestone(3, "Three")]
        output = format_milestone_instructions(milestones)
        assert "1." in output
        assert "2." in output
        assert "3." in output

    def test_empty_milestones_list_still_returns_string(self) -> None:
        output = format_milestone_instructions([])
        assert isinstance(output, str)


# ---------------------------------------------------------------------------
# build_system_prompt with milestone_instructions
# ---------------------------------------------------------------------------


class TestBuildSystemPromptWithMilestones:
    def test_milestone_instructions_appear_in_prompt(self) -> None:
        """Milestone text passed as milestone_instructions must appear in output."""
        milestone_text = "## Milestones\n1. Do X\n2. Do Y"
        prompt = build_system_prompt(milestone_instructions=milestone_text)
        assert "## Milestones" in prompt
        assert "Do X" in prompt
        assert "Do Y" in prompt

    def test_no_milestone_instructions_omits_section(self) -> None:
        """Without milestone_instructions the prompt must not contain that header."""
        prompt = build_system_prompt()
        assert "## Progress Milestones" not in prompt

    def test_milestone_instructions_appended_after_base(self) -> None:
        """Milestone section must appear after the base prompt content."""
        milestone_text = "## Progress Milestones\n1. Start"
        base = "Base prompt text"
        prompt = build_system_prompt(base_prompt=base, milestone_instructions=milestone_text)
        base_pos = prompt.index(base)
        milestone_pos = prompt.index("## Progress Milestones")
        assert milestone_pos > base_pos

    def test_format_and_build_integration(self) -> None:
        """format_milestone_instructions output integrates cleanly into build_system_prompt."""
        milestones = [Milestone(1, "Phase one"), Milestone(2, "Phase two")]
        instructions = format_milestone_instructions(milestones)
        prompt = build_system_prompt(milestone_instructions=instructions)
        assert "Phase one" in prompt
        assert "Phase two" in prompt
        assert "## Progress Milestones" in prompt
