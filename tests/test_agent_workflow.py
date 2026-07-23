"""Integration tests for the agent message-handling workflow.

These tests spin up a **real** LangGraph ReAct agent backed by live LLM
endpoints and exercise the full message lifecycle:

    user input → memory context → LLM invocation → tool calls → response → memory update

Every scenario enforces:
    * **Correctness** – the response satisfies a content predicate.
    * **Timeliness**  – the round-trip completes within a wall-clock budget.
    * **Efficiency**  – the number of LLM "steps" (messages returned by the
      agent) stays within a bounded range, ensuring the agent is not looping
      or making superfluous tool calls.

Requirements
~~~~~~~~~~~~
* A valid Cogtrix configuration file that provides access to two models
  accessible as provider entries or model aliases:
    - ``gpt-oss``   (OpenAI-compatible reasoning model)
    - ``qwen3-coder`` (Ollama-hosted coding model)
* Both endpoints must be reachable from the test runner.

Run
~~~
::

    pytest tests/test_agent_workflow.py -v --timeout=300

Skip when the infrastructure is unavailable::

    pytest tests/ -v -m "not agent_workflow"
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.prebuilt import create_react_agent

from src.agent.core import (
    _estimate_msg_tokens,
    build_system_prompt,
    create_llm_from_provider_config,
    prepare_messages_with_context,
)
from src.config import Config, ProviderConfig, load_config
from src.memory.json_store import JsonFileMemoryStore
from src.memory.modes.conversation import ConversationMemoryManager
from src.registry import ToolRegistry

# Mark every test in this module as an integration test requiring live LLM endpoints.
# Excluded from CI via: pytest -m "not agent_workflow"
pytestmark = pytest.mark.agent_workflow


# ---------------------------------------------------------------------------
# Dataclass for capturing workflow metrics
# ---------------------------------------------------------------------------
@dataclass
class WorkflowMetrics:
    """Captures measurable aspects of a single agent invocation."""

    wall_seconds: float = 0.0
    total_messages: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    tool_names: list[str] = field(default_factory=list)
    final_response: str = ""

    @property
    def summary(self) -> str:
        tools = ", ".join(self.tool_names) if self.tool_names else "none"
        return (
            f"time={self.wall_seconds:.1f}s  msgs={self.total_messages}  "
            f"llm_calls={self.llm_calls}  tool_calls={self.tool_calls}  "
            f"tools=[{tools}]"
        )


def _invoke_agent(
    agent,
    messages: list[Any],
    max_context_tokens: int | None = None,
) -> tuple[str, list[BaseMessage], WorkflowMetrics]:
    """Invoke the agent and return (final_text, all_messages, metrics).

    Uses ``stream(stream_mode="values")`` — the same path the production
    ``cogtrix.py`` loop takes — so the test exercises the real code path.
    """
    start = time.monotonic()
    result: dict[str, Any] = {"messages": []}

    for chunk in agent.stream(
        {"messages": messages},
        config={"recursion_limit": 40},
        stream_mode="values",
    ):
        if isinstance(chunk, dict) and "messages" in chunk:
            result = chunk

    elapsed = time.monotonic() - start
    all_msgs: list[BaseMessage] = result.get("messages", [])

    # Derive metrics
    metrics = WorkflowMetrics(wall_seconds=round(elapsed, 2))
    metrics.total_messages = len(all_msgs)

    for msg in all_msgs:
        if isinstance(msg, AIMessage):
            metrics.llm_calls += 1
            for tc in getattr(msg, "tool_calls", []) or []:
                metrics.tool_calls += 1
                metrics.tool_names.append(tc.get("name", "?"))
        elif isinstance(msg, ToolMessage):
            pass  # counted via tool_calls on the AIMessage

    # The final AI message is the answer
    for msg in reversed(all_msgs):
        if isinstance(msg, AIMessage) and msg.content:
            metrics.final_response = (
                msg.content if isinstance(msg.content, str) else str(msg.content)
            )
            break

    return metrics.final_response, all_msgs, metrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cogtrix_config() -> Config:
    """Load the real Cogtrix configuration from the standard search path."""
    try:
        config = load_config()
    except Exception as exc:
        pytest.skip(f"Cannot load Cogtrix config: {exc}")
    return config


def _resolve_provider_config(
    config: Config,
    model_or_alias: str,
) -> ProviderConfig:
    """Turn a model name/alias into a usable ProviderConfig.

    Handles three cases:
    1. ``model_or_alias`` matches a named provider → use it directly.
    2. ``model_or_alias`` matches a model_alias → resolve the alias.
    3. Fall back to the default provider and set the model explicitly.
    """
    # Case 1: named provider
    if model_or_alias in config.providers:
        return config.providers[model_or_alias]

    # Case 2: model alias → resolve
    aliases = config.model_aliases or {}
    if model_or_alias in aliases:
        alias_val = aliases[model_or_alias]
        if isinstance(alias_val, dict):
            prov_name = alias_val.get("provider", config.provider)
            prov_cfg = config.providers.get(prov_name)
            if prov_cfg is None:
                pytest.skip(f"Alias '{model_or_alias}' references unknown provider '{prov_name}'")
            return ProviderConfig(
                name=prov_name,
                type=prov_cfg.type,
                base_url=prov_cfg.base_url,
                api_key=prov_cfg.api_key,
                model=alias_val.get("model", prov_cfg.model),
                temperature=alias_val.get("temperature", prov_cfg.temperature),
                num_ctx=alias_val.get("num_ctx", prov_cfg.num_ctx),
            )
        if isinstance(alias_val, str) and "/" in alias_val:
            prov_name, model_name = alias_val.split("/", 1)
            prov_cfg = config.providers.get(prov_name)
            if prov_cfg is None:
                pytest.skip(f"Alias '{model_or_alias}' references unknown provider '{prov_name}'")
            return ProviderConfig(
                name=prov_name,
                type=prov_cfg.type,
                base_url=prov_cfg.base_url,
                api_key=prov_cfg.api_key,
                model=model_name,
                temperature=prov_cfg.temperature,
                num_ctx=prov_cfg.num_ctx,
            )

    # Case 3: use as literal model name on the default provider
    try:
        base_prov = config.get_provider_config()
    except ValueError:
        pytest.skip(f"No default provider configured and '{model_or_alias}' is not an alias")
    return ProviderConfig(
        name=base_prov.name,
        type=base_prov.type,
        base_url=base_prov.base_url,
        api_key=base_prov.api_key,
        model=model_or_alias,
        temperature=base_prov.temperature,
        num_ctx=base_prov.num_ctx,
    )


@pytest.fixture(scope="module")
def gpt_oss_provider(cogtrix_config) -> ProviderConfig:
    return _resolve_provider_config(cogtrix_config, "gpt-oss")


@pytest.fixture(scope="module")
def qwen3_coder_provider(cogtrix_config) -> ProviderConfig:
    return _resolve_provider_config(cogtrix_config, "qwen3-coder")


@pytest.fixture(scope="module")
def safe_tools() -> list:
    """Load a curated subset of tools that don't require external API keys.

    This avoids flaky failures from missing service credentials and
    limits the agent's action space so tests are more deterministic.
    """
    registry = ToolRegistry()
    all_tools = registry.load_all_tools()
    safe_names = {
        "calculate",
        "get_current_datetime",
        "convert_timezone",
        "parse_date",
        "word_count",
        "find_replace",
        "extract_urls",
        "extract_emails",
        "text_compare",
        "split_text",
        "trim_text",
    }
    return [t for name, t in all_tools.items() if name in safe_names]


def _build_agent(provider_config: ProviderConfig, tools: list, prompt: str | None = None):
    """Build a LangGraph ReAct agent from a ProviderConfig."""
    llm = create_llm_from_provider_config(provider_config)
    system_prompt = prompt or build_system_prompt()
    return create_react_agent(model=llm, tools=tools, prompt=system_prompt)


def _make_memory(session_tag: str = "") -> ConversationMemoryManager:
    """Create a throwaway in-memory conversation manager."""
    sid = f"test-workflow-{session_tag}-{uuid.uuid4().hex[:8]}"
    store = JsonFileMemoryStore(base_dir=f"/tmp/cogtrix_test_{sid}")
    mm = ConversationMemoryManager(store, sid, {"working_memory_size": 25})
    mm.load()
    return mm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _assert_efficiency(
    metrics: WorkflowMetrics,
    max_seconds: float,
    max_llm_calls: int,
    label: str,
) -> None:
    """Shared assertion block for timing and efficiency bounds."""
    assert metrics.wall_seconds <= max_seconds, (
        f"[{label}] Took {metrics.wall_seconds}s, budget was {max_seconds}s. "
        f"Metrics: {metrics.summary}"
    )
    assert metrics.llm_calls <= max_llm_calls, (
        f"[{label}] Made {metrics.llm_calls} LLM calls, limit was {max_llm_calls}. "
        f"Metrics: {metrics.summary}"
    )


# ===================================================================
#  Test scenarios
# ===================================================================
@pytest.mark.agent_workflow
class TestSimpleResponses:
    """Scenarios where the agent should reply directly without tools."""

    def test_greeting(self, gpt_oss_provider, safe_tools):
        """A simple greeting must produce a non-empty reply in ≤1 LLM call."""
        agent = _build_agent(gpt_oss_provider, safe_tools)
        msgs = [HumanMessage(content="Hello, how are you?")]

        text, all_msgs, m = _invoke_agent(agent, msgs)
        print(f"  [greeting] {m.summary}")

        assert text, "Agent returned empty response"
        _assert_efficiency(m, max_seconds=60, max_llm_calls=2, label="greeting")

    def test_factual_knowledge(self, gpt_oss_provider, safe_tools):
        """A straightforward factual question answered from parametric knowledge."""
        agent = _build_agent(gpt_oss_provider, safe_tools)
        msgs = [HumanMessage(content="What is the chemical formula for water?")]

        text, _, m = _invoke_agent(agent, msgs)
        print(f"  [factual] {m.summary}")

        normalized = text.upper().replace("\u2082", "2")
        assert "H2O" in normalized, f"Expected 'H2O' in response: {text[:200]}"
        _assert_efficiency(m, max_seconds=60, max_llm_calls=2, label="factual")

    def test_short_explanation(self, qwen3_coder_provider, safe_tools):
        """A coding concept explanation should be answered directly."""
        agent = _build_agent(qwen3_coder_provider, safe_tools)
        msgs = [HumanMessage(content="Explain what a Python decorator is in 2-3 sentences.")]

        text, _, m = _invoke_agent(agent, msgs)
        print(f"  [explanation] {m.summary}")

        assert len(text) > 30, "Response too short for an explanation"
        assert "decorator" in text.lower() or "function" in text.lower()
        _assert_efficiency(m, max_seconds=90, max_llm_calls=2, label="explanation")


@pytest.mark.agent_workflow
class TestToolUsage:
    """Scenarios where the agent must invoke one or more tools."""

    def test_calculator(self, gpt_oss_provider, safe_tools):
        """A non-trivial arithmetic question should trigger the calculator."""
        agent = _build_agent(gpt_oss_provider, safe_tools)
        msgs = [HumanMessage(content="What is 17 * 23 + 891 / 3?")]

        text, _, m = _invoke_agent(agent, msgs)
        print(f"  [calculator] {m.summary}")

        # 17*23 = 391, 891/3 = 297 → 688
        assert "688" in text, f"Expected '688' in response: {text[:300]}"
        _assert_efficiency(m, max_seconds=60, max_llm_calls=4, label="calculator")

    def test_datetime(self, gpt_oss_provider, safe_tools):
        """Asking for the current date/time should use the datetime tool."""
        agent = _build_agent(gpt_oss_provider, safe_tools)
        msgs = [HumanMessage(content="What is the current date and time in UTC?")]

        text, _, m = _invoke_agent(agent, msgs)
        print(f"  [datetime] {m.summary}")

        assert (
            "get_current_datetime" in m.tool_names
        ), f"Expected datetime tool usage, got: {m.tool_names}"
        assert re.search(r"20\d{2}", text), "Response should contain a year"
        _assert_efficiency(m, max_seconds=60, max_llm_calls=4, label="datetime")

    def test_word_count(self, qwen3_coder_provider, safe_tools):
        """A text analysis request should invoke the word_count tool."""
        agent = _build_agent(qwen3_coder_provider, safe_tools)
        sample = "The quick brown fox jumps over the lazy dog. " * 10
        msgs = [HumanMessage(content=f"Count the words in this text:\n\n{sample}")]

        text, _, m = _invoke_agent(agent, msgs)
        print(f"  [word_count] {m.summary}")

        assert "word_count" in m.tool_names, f"Expected word_count tool, got: {m.tool_names}"
        assert (
            "90" in text or "Words" in text
        ), f"Expected word count info in response: {text[:300]}"
        _assert_efficiency(m, max_seconds=90, max_llm_calls=4, label="word_count")


@pytest.mark.agent_workflow
class TestCodeGeneration:
    """Test the coding model on code-producing tasks."""

    def test_python_function(self, qwen3_coder_provider, safe_tools):
        """Ask for a Python function and verify the output contains valid code."""
        agent = _build_agent(qwen3_coder_provider, safe_tools)
        msgs = [
            HumanMessage(
                content=(
                    "Write a Python function called `fibonacci` that takes an "
                    "integer n and returns the n-th Fibonacci number using iteration."
                )
            )
        ]

        text, _, m = _invoke_agent(agent, msgs)
        print(f"  [code_gen] {m.summary}")

        assert "def fibonacci" in text, f"Expected function definition in response: {text[:400]}"
        assert "return" in text.lower()
        _assert_efficiency(m, max_seconds=90, max_llm_calls=3, label="code_gen")

    def test_code_with_tool(self, qwen3_coder_provider, safe_tools):
        """Coding model asked to verify a computation — may combine code + calculator."""
        agent = _build_agent(qwen3_coder_provider, safe_tools)
        msgs = [
            HumanMessage(
                content="Calculate 2^20 and then write a one-liner Python "
                "expression that produces the same result."
            )
        ]

        text, _, m = _invoke_agent(agent, msgs)
        print(f"  [code+calc] {m.summary}")

        assert "1048576" in text or "2**20" in text or "2 ** 20" in text
        _assert_efficiency(m, max_seconds=90, max_llm_calls=5, label="code+calc")


@pytest.mark.agent_workflow
class TestMultiTurnConversation:
    """Multi-turn exchanges testing context retention and memory updates."""

    def test_two_turn_context_retention(self, gpt_oss_provider, safe_tools):
        """The agent must remember facts stated in the first turn."""
        agent = _build_agent(gpt_oss_provider, safe_tools)
        mm = _make_memory("ctx-retain")

        # Turn 1: introduce a fact
        ctx1 = mm.prepare_context("My favourite colour is teal.")
        msgs1 = prepare_messages_with_context(ctx1.messages, "My favourite colour is teal.")
        text1, all1, m1 = _invoke_agent(agent, msgs1)
        agent_msgs1 = [m for m in all1 if not isinstance(m, (HumanMessage, SystemMessage))]
        mm.update("My favourite colour is teal.", text1, agent_msgs1)
        mm.save()
        print(f"  [turn1] {m1.summary}")

        # Turn 2: ask about the fact
        ctx2 = mm.prepare_context("What is my favourite colour?")
        msgs2 = prepare_messages_with_context(ctx2.messages, "What is my favourite colour?")
        text2, _, m2 = _invoke_agent(agent, msgs2)
        print(f"  [turn2] {m2.summary}")

        assert (
            "teal" in text2.lower()
        ), f"Agent forgot the colour from turn 1. Response: {text2[:300]}"
        _assert_efficiency(m2, max_seconds=60, max_llm_calls=2, label="turn2-recall")

    def test_three_turn_task_continuation(self, gpt_oss_provider, safe_tools):
        """A three-turn exchange where each turn builds on the previous."""
        agent = _build_agent(gpt_oss_provider, safe_tools)
        mm = _make_memory("3turn")

        turns = [
            ("Let's work with the number 42.", None),
            ("Multiply it by 3.", "126"),
            ("Now add 8 to the result.", "134"),
        ]

        for i, (user_input, expected_substr) in enumerate(turns):
            ctx = mm.prepare_context(user_input)
            msgs = prepare_messages_with_context(ctx.messages, user_input)
            text, all_m, m = _invoke_agent(agent, msgs)
            agent_msgs = [
                msg for msg in all_m if not isinstance(msg, (HumanMessage, SystemMessage))
            ]
            mm.update(user_input, text, agent_msgs)
            mm.save()
            print(f"  [turn{i+1}] {m.summary}")

            if expected_substr:
                assert (
                    expected_substr in text
                ), f"Turn {i+1}: expected '{expected_substr}' in: {text[:300]}"

        _assert_efficiency(m, max_seconds=90, max_llm_calls=5, label="3turn-final")


@pytest.mark.agent_workflow
class TestMessageBudget:
    """Verify that message preparation respects token budgets."""

    def test_context_trimming(self, gpt_oss_provider, safe_tools):
        """When history is large, the agent must still respond (trimmed context)."""
        agent = _build_agent(gpt_oss_provider, safe_tools)
        mm = _make_memory("trim")

        # Seed 40 dummy turns to exceed the 25-message working window
        for i in range(40):
            mm.update(
                f"User message number {i} with some padding text " * 3,
                f"AI response number {i} confirming receipt " * 3,
            )
        mm.save()

        ctx = mm.prepare_context("What was the last thing I said?")
        assert (
            ctx.context_messages_count <= 25
        ), f"Working window should cap at 25, got {ctx.context_messages_count}"

        msgs = prepare_messages_with_context(
            ctx.messages,
            "What was the last thing I said?",
            context_prefix=ctx.context_prefix,
            max_context_tokens=8192,
        )

        total_tokens = sum(_estimate_msg_tokens(m) for m in msgs)
        assert total_tokens < 8192, f"Prepared messages exceed token budget: {total_tokens}"

        text, _, m = _invoke_agent(agent, msgs)
        print(f"  [trimmed] {m.summary}")

        assert text, "Agent returned empty response on trimmed context"
        _assert_efficiency(m, max_seconds=60, max_llm_calls=20, label="trimmed")

    def test_empty_history(self, gpt_oss_provider, safe_tools):
        """Agent should work fine with zero conversation history."""
        agent = _build_agent(gpt_oss_provider, safe_tools)
        mm = _make_memory("empty")

        ctx = mm.prepare_context("Say hello.")
        assert ctx.total_messages_stored == 0

        msgs = prepare_messages_with_context(ctx.messages, "Say hello.")
        text, _, m = _invoke_agent(agent, msgs)
        print(f"  [empty_hist] {m.summary}")

        assert text, "Agent returned empty response on empty history"
        _assert_efficiency(m, max_seconds=60, max_llm_calls=2, label="empty_hist")


@pytest.mark.agent_workflow
class TestEdgeCases:
    """Boundary and adversarial inputs."""

    def test_very_short_prompt(self, gpt_oss_provider, safe_tools):
        """A single-word prompt should not crash or loop."""
        agent = _build_agent(gpt_oss_provider, safe_tools)
        msgs = [HumanMessage(content="Hi")]

        text, _, m = _invoke_agent(agent, msgs)
        print(f"  [short] {m.summary}")

        assert text, "Empty response to short prompt"
        _assert_efficiency(m, max_seconds=60, max_llm_calls=2, label="short")

    def test_prompt_with_special_characters(self, gpt_oss_provider, safe_tools):
        """Prompts with unicode and special chars must not break the pipeline."""
        agent = _build_agent(gpt_oss_provider, safe_tools)
        msgs = [HumanMessage(content="What does the symbol \u03c0 represent in mathematics?")]

        text, _, m = _invoke_agent(agent, msgs)
        print(f"  [unicode] {m.summary}")

        assert "pi" in text.lower() or "\u03c0" in text or "3.14" in text
        _assert_efficiency(m, max_seconds=60, max_llm_calls=2, label="unicode")

    def test_multi_tool_prompt(self, gpt_oss_provider, safe_tools):
        """A prompt that can benefit from multiple tools in one turn."""
        agent = _build_agent(gpt_oss_provider, safe_tools)
        msgs = [
            HumanMessage(
                content=(
                    "First, tell me the current UTC time. " "Then, calculate sqrt(144) + sqrt(256)."
                )
            )
        ]

        text, _, m = _invoke_agent(agent, msgs)
        print(f"  [multi_tool] {m.summary}")

        # sqrt(144)=12, sqrt(256)=16 → 28
        assert m.tool_calls >= 1, "Expected at least one tool call"
        assert "28" in text or ("12" in text and "16" in text)
        _assert_efficiency(m, max_seconds=90, max_llm_calls=6, label="multi_tool")


@pytest.mark.agent_workflow
class TestCrossModelComparison:
    """Run the same prompt on both models to ensure both produce valid output."""

    PROMPT = "Write a Python function that checks whether a string is a palindrome."

    def test_gpt_oss_palindrome(self, gpt_oss_provider, safe_tools):
        agent = _build_agent(gpt_oss_provider, safe_tools)
        msgs = [HumanMessage(content=self.PROMPT)]

        text, _, m = _invoke_agent(agent, msgs)
        print(f"  [gpt-oss] {m.summary}")

        assert "def " in text and "palindrome" in text.lower()
        _assert_efficiency(m, max_seconds=90, max_llm_calls=3, label="gpt-oss-palindrome")

    def test_qwen3_coder_palindrome(self, qwen3_coder_provider, safe_tools):
        agent = _build_agent(qwen3_coder_provider, safe_tools)
        msgs = [HumanMessage(content=self.PROMPT)]

        text, _, m = _invoke_agent(agent, msgs)
        print(f"  [qwen3-coder] {m.summary}")

        assert "def " in text and "palindrome" in text.lower()
        _assert_efficiency(m, max_seconds=90, max_llm_calls=3, label="qwen-palindrome")


@pytest.mark.agent_workflow
class TestMemoryWorkflow:
    """Validate the full memory lifecycle: load → prepare → update → save."""

    def test_memory_roundtrip(self, gpt_oss_provider, safe_tools):
        """Messages survive a save/load cycle and remain usable."""
        agent = _build_agent(gpt_oss_provider, safe_tools)
        mm = _make_memory("roundtrip")

        # Turn 1
        ctx = mm.prepare_context("Remember that the secret code is 7734.")
        msgs = prepare_messages_with_context(ctx.messages, "Remember that the secret code is 7734.")
        text, all_m, m = _invoke_agent(agent, msgs)
        agent_msgs = [msg for msg in all_m if not isinstance(msg, (HumanMessage, SystemMessage))]
        mm.update("Remember that the secret code is 7734.", text, agent_msgs)
        mm.save()

        assert mm.get_message_count() >= 2, "Expected at least user + AI messages"

        # Simulate restart: create a new manager from the same store/session
        mm2 = ConversationMemoryManager(mm.store, mm.session_id, {"working_memory_size": 25})
        mm2.load()
        assert mm2.get_message_count() == mm.get_message_count()

        # Turn 2 on restored memory
        ctx2 = mm2.prepare_context("What is the secret code?")
        msgs2 = prepare_messages_with_context(ctx2.messages, "What is the secret code?")
        text2, _, m2 = _invoke_agent(agent, msgs2)
        print(f"  [roundtrip] {m2.summary}")

        assert "7734" in text2, f"Agent lost the secret code after reload: {text2[:300]}"

    def test_sanitize_history_filters_errors(self):
        """Ensure error messages in history are cleaned by sanitize_history."""
        from src.memory.manager import BaseMemoryManager

        history: list[Any] = [
            HumanMessage(content="Do something"),
            AIMessage(content="**Error:** Connection failed"),
            HumanMessage(content="Try again"),
            AIMessage(content="Here is your answer"),
        ]

        cleaned = BaseMemoryManager.sanitize_history(history)
        assert len(cleaned) == 2, f"Expected 2 messages after sanitize, got {len(cleaned)}"
        assert isinstance(cleaned[0], HumanMessage)
        assert isinstance(cleaned[1], AIMessage)
        assert "answer" in cleaned[1].content

    def test_context_prefix_includes_summary(self):
        """When a summary exists, prepare_context must inject it."""
        mm = _make_memory("prefix")
        mm._summary = "User previously discussed Python decorators."

        ctx = mm.prepare_context("Tell me more.")

        assert ctx.context_prefix is not None
        assert "decorators" in ctx.context_prefix.lower()
