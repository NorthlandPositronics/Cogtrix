"""Pipeline tests: optimizer instructions, scenario prompts, intent detection, and milestones.

Layer 1 — Optimizer instruction correctness
Layer 2 — Scenario-based pipeline tests with canned responses
Layer 3 — Intent detection regression (parametrized)
Layer 4 — Agent behavior with milestones (full pipeline)
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from cogtrix import _build_agent_graph, _cleanup_milestones, _inject_milestones
from src.agent.core import build_system_prompt, format_milestone_instructions
from src.orchestration.intent import (
    classify_think_task,
    prompt_requests_action,
    user_wants_deep_think,
    user_wants_delegation,
)
from src.orchestration.phases import agent_performed_writes
from src.prompt.optimizer import Milestone, PromptPlan, optimize_prompt
from src.tools.report_progress import create_report_progress_tool, set_progress_callback

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeLLM:
    """Minimal LLM stub whose invoke() returns a configurable content string."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.call_count = 0
        self.last_prompt: str | None = None

    def invoke(self, prompt: str) -> _FakeResponse:
        self.call_count += 1
        self.last_prompt = prompt
        return _FakeResponse(self._content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


def _make_mock_llm(responses: list[AIMessage]) -> MagicMock:
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.invoke.side_effect = responses
    return mock_llm


def _make_registry(requires_confirmation: bool = False) -> MagicMock:
    mock_registry = MagicMock()
    mock_registry.requires_confirmation.return_value = requires_confirmation
    return mock_registry


# ---------------------------------------------------------------------------
# Canned optimizer responses
# ---------------------------------------------------------------------------

DOCKER_PROMPT = (
    "Build the cogtrix docker image. The image is already available locally tagged "
    "as cogtrix:latest. Do not rebuild it. Run docker compose up with the production "
    "configuration at docker/docker-compose.prod.yml. Make sure the health check passes "
    "before proceeding. If any container fails to start, check the logs and fix the issue. "
    "Use port 8080 for the web interface. The database password is stored in .env file. "
    "After all services are running, run the integration test suite against the running "
    "containers. Report results."
)

DOCKER_OPTIMIZER_RESPONSE = (
    "---PROMPT---\n"
    "Deploy the existing cogtrix Docker stack using the production compose file.\n\n"
    "**Key constraints (do NOT violate):**\n"
    "- The image is already available locally as cogtrix:latest — do NOT rebuild it\n"
    "- Use port 8080 for the web interface\n"
    "- Database password is in .env file\n\n"
    "**Phases:**\n"
    "1. Run docker compose up with docker/docker-compose.prod.yml\n"
    "2. Wait for health checks to pass\n"
    "3. If containers fail, check logs and fix\n"
    "4. Run integration test suite against running containers\n"
    "5. Report results\n"
    "---MILESTONES---\n"
    "1. Start Docker stack\n"
    "2. Verify health checks\n"
    "3. Fix startup failures\n"
    "4. Run integration tests\n"
    "5. Report results\n"
    "---END---"
)

REFACTOR_PROMPT = (
    "Refactor the authentication module to use JWT tokens instead of sessions. "
    "The config file is at /etc/app/config.yaml and contains the session secret key "
    "which should be replaced with a JWT signing key. Update all API endpoints in "
    "src/api/ to validate JWT tokens from the Authorization header. The user model "
    "is in src/models/user.py and needs a refresh_token field added. Write migration "
    "scripts for the database schema change. Ensure all existing tests in tests/auth/ "
    "still pass and add new tests for JWT token validation, expiry, and refresh flow. "
    "The frontend at src/ui/auth.js also needs updating to store and send JWT tokens."
)

REFACTOR_OPTIMIZER_RESPONSE = (
    "---PROMPT---\n"
    "Migrate authentication from session-based to JWT tokens.\n\n"
    "**Constraints:**\n"
    "- Config at /etc/app/config.yaml — replace session secret with JWT signing key\n"
    "- User model at src/models/user.py — add refresh_token field\n"
    "- Frontend at src/ui/auth.js — store and send JWT tokens\n\n"
    "**Phases:**\n"
    "1. Update config and user model\n"
    "2. Update API endpoints in src/api/ for JWT validation\n"
    "3. Write database migration scripts\n"
    "4. Update frontend auth handling\n"
    "5. Fix existing tests and add JWT-specific tests\n"
    "---MILESTONES---\n"
    "1. Update config and models\n"
    "2. Migrate API endpoints\n"
    "3. Write DB migrations\n"
    "4. Update frontend auth\n"
    "5. Update and add tests\n"
    "---END---"
)

RESEARCH_PROMPT = (
    "Research the latest React 19 features and create a comprehensive migration guide "
    "document. The guide should cover the new use() hook, React Server Components, "
    "the new compiler, Actions and transitions, and any breaking changes from React 18. "
    "Include code examples for each feature showing the React 18 way vs the React 19 way. "
    "Search for official React team blog posts and RFC documents. Create the guide as "
    "docs/react19-migration.md with proper markdown formatting, table of contents, and "
    "version compatibility matrix."
)

RESEARCH_OPTIMIZER_RESPONSE = (
    "---PROMPT---\n"
    "Research React 19 features and produce a migration guide at docs/react19-migration.md.\n\n"
    "**Key topics to cover:**\n"
    "- use() hook\n"
    "- React Server Components\n"
    "- New compiler\n"
    "- Actions and transitions\n"
    "- Breaking changes from React 18\n\n"
    "**Requirements:**\n"
    "- Code examples: React 18 way vs React 19 way for each feature\n"
    "- Sources: official React team blog posts and RFC documents\n"
    "- Format: markdown with TOC and version compatibility matrix\n"
    "---MILESTONES---\n"
    "1. Research React 19 features\n"
    "2. Draft migration guide structure\n"
    "3. Write feature comparisons\n"
    "4. Add compatibility matrix\n"
    "---END---"
)

# ---------------------------------------------------------------------------
# Long prompts for Layer 1 tests
# ---------------------------------------------------------------------------

_LONG_NOACTION = (
    "The system should handle multiple concurrent connections efficiently while "
    "maintaining data consistency across distributed nodes in the cluster. "
    "Each node processes incoming requests independently but must synchronize "
    "state through a consensus protocol. The current implementation suffers "
    "from occasional split-brain scenarios when network partitions occur "
    "between the primary and secondary data centers. We need a thorough "
    "analysis of the failure modes and recommended mitigations."
)

_LONG_ACTION_500 = (
    "Create a comprehensive monitoring dashboard for the distributed system "
    "that tracks CPU usage, memory consumption, network latency, and disk I/O "
    "across all cluster nodes. The dashboard should include real-time charts, "
    "historical trend lines, anomaly detection alerts, and capacity planning "
    "projections. Integrate with the existing Prometheus metrics pipeline "
    "and configure Grafana panels for each service."
)

_LONG_ACTION_650 = _LONG_ACTION_500 + (
    " Additionally, set up alert rules for critical thresholds: CPU above 90%, "
    "memory above 85%, disk usage above 80%, and response latency above 500ms. "
    "Configure PagerDuty integration for on-call escalation."
)

# ---------------------------------------------------------------------------
# Layer 1 — TestOptimizerInstructions
# ---------------------------------------------------------------------------


class TestOptimizerInstructions:
    def test_fact_preservation_bullet_present(self) -> None:
        llm = _FakeLLM(
            "This is a sufficiently optimized prompt that preserves all the original intent."
        )
        optimize_prompt(_LONG_NOACTION, llm)
        assert llm.last_prompt is not None
        assert "Preserve all user-stated facts" in llm.last_prompt

    def test_no_drop_facts_instruction(self) -> None:
        llm = _FakeLLM(
            "This is a sufficiently optimized prompt that preserves all the original intent."
        )
        optimize_prompt(_LONG_NOACTION, llm)
        assert llm.last_prompt is not None
        assert "Never drop or paraphrase factual claims" in llm.last_prompt

    def test_milestone_appendix_when_true(self) -> None:
        llm = _FakeLLM(
            "This is a sufficiently optimized prompt that preserves all the original intent."
        )
        optimize_prompt(_LONG_NOACTION, llm, plan_milestones=True)
        assert llm.last_prompt is not None
        assert "---MILESTONES---" in llm.last_prompt
        assert "milestone plan" in llm.last_prompt

    def test_milestone_appendix_absent_when_false(self) -> None:
        llm = _FakeLLM(
            "This is a sufficiently optimized prompt that preserves all the original intent."
        )
        optimize_prompt(_LONG_NOACTION, llm, plan_milestones=False)
        assert llm.last_prompt is not None
        assert "---MILESTONES---" not in llm.last_prompt

    def test_nonce_delimiters_wrap_input(self) -> None:
        llm = _FakeLLM(
            "This is a sufficiently optimized prompt that preserves all the original intent."
        )
        optimize_prompt(_LONG_NOACTION, llm)
        assert llm.last_prompt is not None
        assert re.search(r"__USER_INPUT_[0-9a-f]{16}_START__", llm.last_prompt)
        assert re.search(r"__USER_INPUT_[0-9a-f]{16}_END__", llm.last_prompt)

    def test_user_input_between_delimiters(self) -> None:
        llm = _FakeLLM(
            "This is a sufficiently optimized prompt that preserves all the original intent."
        )
        optimize_prompt(_LONG_NOACTION, llm)
        assert llm.last_prompt is not None
        prompt = llm.last_prompt
        start_m = re.search(r"__USER_INPUT_[0-9a-f]{16}_START__", prompt)
        end_m = re.search(r"__USER_INPUT_[0-9a-f]{16}_END__", prompt)
        assert start_m is not None
        assert end_m is not None
        between = prompt[start_m.end() : end_m.start()]
        assert "cluster" in between

    def test_action_verb_skip_under_600(self) -> None:
        llm = _FakeLLM(
            "This is a sufficiently optimized prompt that preserves all the original intent."
        )
        plan = optimize_prompt(_LONG_ACTION_500, llm)
        assert llm.call_count == 0
        assert plan.text == _LONG_ACTION_500

    def test_action_verb_no_skip_above_600(self) -> None:
        llm = _FakeLLM(
            "This is a sufficiently optimized prompt that preserves all the original intent."
        )
        optimize_prompt(_LONG_ACTION_650, llm)
        assert llm.call_count == 1

    def test_no_action_verb_calls_llm(self) -> None:
        llm = _FakeLLM(
            "This is a sufficiently optimized prompt that preserves all the original intent."
        )
        optimize_prompt(_LONG_NOACTION, llm)
        assert llm.call_count == 1

    def test_force_bypasses_all_gates(self) -> None:
        llm = _FakeLLM(
            "This is a sufficiently optimized prompt that preserves all the original intent."
        )
        optimize_prompt("hello world", llm, force=True)
        assert llm.call_count == 1
        assert llm.last_prompt is not None
        assert "ALWAYS rewrite" in llm.last_prompt


# ---------------------------------------------------------------------------
# Layer 2 — Scenario tests
# ---------------------------------------------------------------------------


class TestDockerScenario:
    def test_action_verb_short_circuit(self) -> None:
        llm = _FakeLLM(
            "This is a sufficiently optimized prompt that preserves all the original intent."
        )
        plan = optimize_prompt(DOCKER_PROMPT, llm)
        assert llm.call_count == 0
        assert plan.text == DOCKER_PROMPT
        assert not plan.has_milestones

    def test_forced_optimization_parses_milestones(self) -> None:
        llm = _FakeLLM(DOCKER_OPTIMIZER_RESPONSE)
        plan = optimize_prompt(DOCKER_PROMPT, llm, force=True, plan_milestones=True)
        assert len(plan.milestones) == 5
        titles = [m.title for m in plan.milestones]
        assert "Start Docker stack" in titles
        assert "Verify health checks" in titles
        assert "Fix startup failures" in titles
        assert "Run integration tests" in titles
        assert "Report results" in titles

    def test_facts_preserved_in_optimized_text(self) -> None:
        llm = _FakeLLM(DOCKER_OPTIMIZER_RESPONSE)
        plan = optimize_prompt(DOCKER_PROMPT, llm, force=True, plan_milestones=True)
        assert "cogtrix:latest" in plan.text
        assert "port 8080" in plan.text
        assert "do NOT rebuild" in plan.text


class TestRefactorScenario:
    def test_triggers_llm_call(self) -> None:
        llm = _FakeLLM(REFACTOR_OPTIMIZER_RESPONSE)
        optimize_prompt(REFACTOR_PROMPT, llm, plan_milestones=True)
        assert llm.call_count == 1

    def test_milestones_parsed(self) -> None:
        llm = _FakeLLM(REFACTOR_OPTIMIZER_RESPONSE)
        plan = optimize_prompt(REFACTOR_PROMPT, llm, plan_milestones=True)
        assert len(plan.milestones) == 5
        titles = [m.title for m in plan.milestones]
        assert "Update config and models" in titles
        assert "Migrate API endpoints" in titles
        assert "Write DB migrations" in titles
        assert "Update frontend auth" in titles
        assert "Update and add tests" in titles

    def test_system_prompt_includes_milestones(self) -> None:
        llm = _FakeLLM(REFACTOR_OPTIMIZER_RESPONSE)
        plan = optimize_prompt(REFACTOR_PROMPT, llm, plan_milestones=True)
        instr = format_milestone_instructions(plan.milestones)
        sys = build_system_prompt(milestone_instructions=instr)
        assert "## Progress Milestones" in sys
        for m in plan.milestones:
            assert m.title in sys

    def test_report_progress_tool_created(self) -> None:
        llm = _FakeLLM(REFACTOR_OPTIMIZER_RESPONSE)
        plan = optimize_prompt(REFACTOR_PROMPT, llm, plan_milestones=True)
        tool = create_report_progress_tool(plan.milestones)
        assert tool.name == "report_progress"
        for m in plan.milestones:
            assert m.title in tool.description


class TestResearchScenario:
    def test_milestones_count(self) -> None:
        llm = _FakeLLM(RESEARCH_OPTIMIZER_RESPONSE)
        plan = optimize_prompt(RESEARCH_PROMPT, llm, force=True, plan_milestones=True)
        assert len(plan.milestones) == 4

    def test_facts_preserved(self) -> None:
        llm = _FakeLLM(RESEARCH_OPTIMIZER_RESPONSE)
        plan = optimize_prompt(RESEARCH_PROMPT, llm, force=True, plan_milestones=True)
        assert "use() hook" in plan.text
        assert "React Server Components" in plan.text
        assert "docs/react19-migration.md" in plan.text

    def test_milestone_indices_sequential(self) -> None:
        llm = _FakeLLM(RESEARCH_OPTIMIZER_RESPONSE)
        plan = optimize_prompt(RESEARCH_PROMPT, llm, force=True, plan_milestones=True)
        assert [m.index for m in plan.milestones] == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Layer 3 — Intent detection regression
# ---------------------------------------------------------------------------


class TestUserWantsDeepThink:
    @pytest.mark.parametrize(
        "prompt, expected",
        [
            ("think deeply about the architecture", True),
            ("Please think deeply about this problem", True),
            ("analyze thoroughly the architecture", True),
            ("think step by step about the solution", True),
            ("comprehensive analysis of the codebase", True),
            ("deep reasoning about the tradeoffs", True),
            ("deep think about the design", True),
            ("carefully analyze the error logs", True),
            ("examine the code thoroughly", True),
            ("think this through before answering", True),
            ("I need a thorough analysis of this data", True),
            ("deep analysis of the performance bottleneck", True),
            ("please think through the implications", True),
            ("analyze the results in depth", True),
            ("consider all angles of this problem", True),
            ("give me your best analysis of the situation", True),
            ("think carefully about the edge cases", True),
            # Negative
            ("think about what to have for lunch", False),
            ("just analyze it", False),
            ("give me a quick summary", False),
            ("create a deep link to the page", False),
            ("what do you think?", False),
            ("analyze this", False),
            ("run the analysis script", False),
            ("think of a name for the variable", False),
        ],
    )
    def test_deep_think_detection(self, prompt: str, expected: bool) -> None:
        assert user_wants_deep_think(prompt) is expected


class TestUserWantsDelegation:
    @pytest.mark.parametrize(
        "prompt, expected",
        [
            ("compare React vs Vue for our frontend", True),
            ("top 5 databases for time series data", True),
            ("pros and cons of microservices architecture", True),
            ("for each of the APIs, fetch the documentation", True),
            ("research Python, Rust, and Go and compare performance", True),
            ("what are the differences between REST and GraphQL", True),
            ("translate the README into French, Spanish, and German", True),
            ("evaluate React, Angular, and Svelte and pick the best", True),
            ("3 best Python web frameworks for beginners", True),
            ("each of these endpoints needs testing", True),
            # Negative
            ("use React for the frontend", False),
            ("install the database", False),
            ("the top of the file has the imports", False),
            ("configure the API endpoint", False),
            ("write a comparison section in the docs", False),
        ],
    )
    def test_delegation_detection(self, prompt: str, expected: bool) -> None:
        assert user_wants_delegation(prompt) is expected


class TestPromptRequestsAction:
    @pytest.mark.parametrize(
        "prompt, expected",
        [
            ("create a new module for authentication", True),
            ("write test files for the parser", True),
            ("fix the configuration file", True),
            ("update the README with new instructions", True),
            ("generate a report document", True),
            ("implement the login component", True),
            ("build the deployment script", True),
            ("add a new function to the utils module", True),
            ("replace the old config with the new format", True),
            ("refactor the test suite", True),
            ("modify the class to accept new parameters", True),
            ("set up the project directory structure", True),
            ("save the results to a JSON file", True),
            ("append the log entry to the file", True),
            ("patch the configuration module", True),
            # Negative
            ("explain the module architecture", False),
            ("what does this code do", False),
            ("analyze the test results", False),
            ("explain how to create files in Python", False),
            ("describe the configuration format", False),
            ("how does the module work", False),
            ("tell me about the test framework", False),
            ("show me the file contents", False),
            # Edge: explain before action verb -> False
            ("explain and then create the module", False),
        ],
    )
    def test_action_detection(self, prompt: str, expected: bool) -> None:
        assert prompt_requests_action(prompt) is expected


class TestClassifyThinkTask:
    @pytest.mark.parametrize(
        "prompt, expected_category",
        [
            ("I need a thorough code review", "code_analysis"),
            ("search for the latest AI papers", "research"),
            ("run a benchmark test of the database", "comparison"),
            ("analyze this traceback from the server", "debugging"),
            ("come up with fresh ideas for the product feature", "ideation"),
            ("explain how TCP works under the hood", "technical"),
            ("write a blog post about cloud computing", "writing"),
            ("analyze the revenue data for Q3", "business"),
        ],
    )
    def test_single_keyword_match(self, prompt: str, expected_category: str) -> None:
        llm = _FakeLLM("should not be called")
        result = classify_think_task(prompt, llm)
        assert result.name == expected_category
        assert llm.call_count == 0

    def test_orm_substring_does_not_match_information(self) -> None:
        # "ORM" must not match as a substring inside "information" — regression for
        # the bug where "find information about Synechron" was classified as database.
        llm = _FakeLLM("research")
        result = classify_think_task(
            "Please find information about Synechron company. "
            "Their website is synechron.com. The company has office in Abu Dhabi, UAE.",
            llm,
        )
        assert (
            result.name != "database"
        ), "keyword 'ORM' must not match inside 'information' — use word-boundary matching"

    def test_orm_still_matches_legitimate_orm_prompt(self) -> None:
        # ORM as a standalone token must still trigger the database category.
        llm = _FakeLLM("should not be called")
        result = classify_think_task("use ORM to query the schema", llm)
        assert result.name == "database"
        assert llm.call_count == 0

    def test_morphological_variants_match_keyword_root(self) -> None:
        # Prefix-match: keyword "refactor" must match "refactoring", "refactored".
        # Regression for the word-boundary fix that broke morphological variants.
        llm = _FakeLLM("should not be called")
        result = classify_think_task("I need help refactoring this module", llm)
        assert (
            result.name == "code_analysis"
        ), "keyword 'refactor' must match 'refactoring' via prefix match"
        assert llm.call_count == 0

    def test_no_keyword_uses_llm(self) -> None:
        llm = _FakeLLM("research")
        result = classify_think_task("what is the most efficient cache eviction policy", llm)
        assert llm.call_count == 1
        assert result.name == "research"

    def test_unrecognized_label_fallback(self) -> None:
        llm = _FakeLLM("nonexistent_category")
        result = classify_think_task("what is the most efficient cache eviction policy", llm)
        assert result.name == "general"

    def test_llm_exception_fallback(self) -> None:
        class _BrokenLLM:
            def invoke(self, prompt: str) -> None:
                raise RuntimeError("connection error")

        result = classify_think_task(
            "what is the most efficient cache eviction policy", _BrokenLLM()
        )
        assert result.name == "general"

    def test_multiple_keywords_triggers_llm(self) -> None:
        llm = _FakeLLM("comparison")
        result = classify_think_task("search for algorithm benchmarks", llm)
        assert llm.call_count == 1
        assert result.name == "comparison"


# ---------------------------------------------------------------------------
# TestAgentPerformedWrites
# ---------------------------------------------------------------------------


class TestAgentPerformedWrites:
    def test_write_file_success(self) -> None:
        msgs = [
            ToolMessage(content="File written successfully", tool_call_id="t1", name="write_file")
        ]
        assert agent_performed_writes(msgs) is True

    def test_append_file_success(self) -> None:
        msgs = [
            ToolMessage(content="Content appended to file", tool_call_id="t1", name="append_file")
        ]
        assert agent_performed_writes(msgs) is True

    def test_shell_success(self) -> None:
        msgs = [
            ToolMessage(
                content="total 42\ndrwxr-xr-x",
                tool_call_id="t1",
                name="execute_shell_command",
            )
        ]
        assert agent_performed_writes(msgs) is True

    def test_read_only_false(self) -> None:
        msgs = [ToolMessage(content="file contents here", tool_call_id="t1", name="read_file")]
        assert agent_performed_writes(msgs) is False

    def test_write_error_false(self) -> None:
        msgs = [
            ToolMessage(content="Error: permission denied", tool_call_id="t1", name="write_file")
        ]
        assert agent_performed_writes(msgs) is False

    def test_shell_denied_false(self) -> None:
        msgs = [
            ToolMessage(
                content="User denied execution",
                tool_call_id="t1",
                name="execute_shell_command",
            )
        ]
        assert agent_performed_writes(msgs) is False

    def test_shell_cancelled_false(self) -> None:
        msgs = [
            ToolMessage(
                content="User cancelled agent workflow",
                tool_call_id="t1",
                name="execute_shell_command",
            )
        ]
        assert agent_performed_writes(msgs) is False

    def test_empty_messages_false(self) -> None:
        assert agent_performed_writes([]) is False

    def test_tool_exec_error_false(self) -> None:
        msgs = [
            ToolMessage(
                content="Tool execution error: timeout",
                tool_call_id="t1",
                name="write_file",
            )
        ]
        assert agent_performed_writes(msgs) is False

    def test_mixed_with_one_success(self) -> None:
        msgs = [
            ToolMessage(content="file contents", tool_call_id="t1", name="read_file"),
            ToolMessage(content="File written successfully", tool_call_id="t2", name="write_file"),
        ]
        assert agent_performed_writes(msgs) is True


# ---------------------------------------------------------------------------
# Layer 4 — Agent behavior with milestones
# ---------------------------------------------------------------------------


class TestMilestoneInjectionCleanup:
    def _make_plan(self, n: int = 3) -> PromptPlan:
        milestones = [Milestone(index=i + 1, title=f"Step {i + 1}") for i in range(n)]
        return PromptPlan(text="optimized prompt", milestones=milestones)

    def test_inject_adds_tool_and_augments_prompt(self) -> None:
        plan = self._make_plan(3)
        tools_list: list = []
        store: list = []
        with patch("cogtrix._spinner"):
            tool, augmented = _inject_milestones(plan, tools_list, store, "base prompt")
        assert tool is not None
        assert tool.name == "report_progress"
        assert "## Progress Milestones" in augmented
        assert len(store) == 3
        assert tool in tools_list

    def test_inject_without_milestones_noop(self) -> None:
        plan = PromptPlan(text="simple prompt")
        tools_list: list = []
        store: list = []
        tool, augmented = _inject_milestones(plan, tools_list, store, "base prompt")
        assert tool is None
        assert augmented == "base prompt"
        assert len(tools_list) == 0
        assert len(store) == 0

    def test_cleanup_removes_tool_clears_store(self) -> None:
        plan = self._make_plan(3)
        tools_list: list = []
        store: list = []
        with patch("cogtrix._spinner"):
            tool, _ = _inject_milestones(plan, tools_list, store, "base prompt")
            _cleanup_milestones(tool, tools_list, store)
        assert len(tools_list) == 0
        assert len(store) == 0

    def test_cleanup_none_tool_noop(self) -> None:
        tools_list = [MagicMock(name="some_tool")]
        store = [Milestone(index=1, title="Step 1")]
        _cleanup_milestones(None, tools_list, store)
        assert len(tools_list) == 1
        assert len(store) == 1

    def test_inject_preserves_existing_tools(self) -> None:
        plan = self._make_plan(2)
        existing = MagicMock()
        existing.name = "existing_tool"
        tools_list = [existing]
        store: list = []
        with patch("cogtrix._spinner"):
            _inject_milestones(plan, tools_list, store, "base")
        assert existing in tools_list
        assert len(tools_list) == 2


class TestAgentPipelineWithMilestones:
    def setup_method(self) -> None:
        self._callback_calls: list[tuple[int, str]] = []
        set_progress_callback(self._on_progress)

    def teardown_method(self) -> None:
        set_progress_callback(None)  # type: ignore[arg-type]

    def _on_progress(self, milestone_index: int, status: str) -> None:
        self._callback_calls.append((milestone_index, status))

    def _run_pipeline(self) -> tuple[dict, PromptPlan]:
        """Shared pipeline: optimize -> inject -> build graph -> run."""
        # Step 1: Optimize
        llm = _FakeLLM(REFACTOR_OPTIMIZER_RESPONSE)
        plan = optimize_prompt(REFACTOR_PROMPT, llm, force=True, plan_milestones=True)
        assert plan.has_milestones

        # Step 2: Build system prompt with milestones
        instr = format_milestone_instructions(plan.milestones)
        sys_prompt = build_system_prompt(milestone_instructions=instr)

        # Step 3: Create report_progress tool
        rp_tool = create_report_progress_tool(plan.milestones)

        # Step 4: Mock agent LLM that reports progress then answers
        rp_call_1 = {
            "name": "report_progress",
            "args": {"milestone_index": 1, "status": "starting"},
            "id": "rp1",
        }
        rp_call_2 = {
            "name": "report_progress",
            "args": {"milestone_index": 2, "status": "implementing"},
            "id": "rp2",
        }
        agent_responses = [
            AIMessage(content="", tool_calls=[rp_call_1], id="a1"),
            AIMessage(content="", tool_calls=[rp_call_2], id="a2"),
            AIMessage(content="Migration complete. All endpoints updated.", id="a3"),
        ]
        agent_llm = _make_mock_llm(agent_responses)

        # Step 5: Build and run graph
        graph = _build_agent_graph(
            llm=agent_llm,
            system_prompt=sys_prompt,
            active_tools_list=[rp_tool],
            available_tools={},
            registry=_make_registry(),
            approvals=set(),
        )
        result = graph.invoke({"messages": [HumanMessage(content=plan.text)]})
        return result, plan

    def test_full_pipeline_optimizer_to_agent(self) -> None:
        result, plan = self._run_pipeline()
        messages = result["messages"]
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_msgs) >= 2
        progress_msgs = [m for m in tool_msgs if m.name == "report_progress"]
        assert len(progress_msgs) == 2
        for msg in progress_msgs:
            assert "Progress reported" in msg.content
        ai_msgs = [m for m in messages if isinstance(m, AIMessage) and m.content]
        assert any("Migration complete" in m.content for m in ai_msgs)

    def test_progress_callback_receives_correct_args(self) -> None:
        self._run_pipeline()
        assert len(self._callback_calls) == 2
        assert self._callback_calls[0] == (1, "starting")
        assert self._callback_calls[1] == (2, "implementing")

    def test_agent_completes_after_milestone_reports(self) -> None:
        result, _ = self._run_pipeline()
        messages = result["messages"]
        last_ai = [m for m in messages if isinstance(m, AIMessage) and m.content]
        assert len(last_ai) >= 1
        assert last_ai[-1].content.strip() != ""
