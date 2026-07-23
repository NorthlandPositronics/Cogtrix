"""Regression tests for simple greeting responses.

These tests verify that the agent responds appropriately to simple greetings
like "Hi", "Hello", etc. and does not return empty content.

Bug context:
- Original issue: Agent returned `content=""` for simple greetings, triggering
  recovery flow instead of responding directly.
- Root cause: Gemma model produces messages with `content=""` and
  `tool_calls=[...]` for greetings, where reasoning_content contains the actual
  response.
- Fixes applied:
  1. Updated DEFAULT_SYSTEM_PROMPT in src/agent/core.py to be more explicit
     about simple greetings
  2. Updated extract_ai_content in src/orchestration/runner.py to extract
     reasoning-only content even when tool_calls is present
  3. Increased DEFAULT_RECURSION_LIMIT from 90 to 300 in src/orchestration/graph.py
  4. Added max_steps configuration support in src/orchestration/run_config.py
"""

from __future__ import annotations

import urllib.request
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Availability check — runs once at collection time
# ---------------------------------------------------------------------------

_GEMMA_BASE_URL = "http://localhost:18080"


def _gemma_is_available() -> bool:
    try:
        r = urllib.request.urlopen(f"{_GEMMA_BASE_URL}/health", timeout=3)
        return r.status == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Pytest marks / global skip
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.live_llm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def require_gemma_container() -> None:
    """Skip the entire module when the Gemma container is not reachable."""
    if not _gemma_is_available():
        pytest.skip(
            "Gemma container not reachable at localhost:8080",
            allow_module_level=True,
        )


@pytest.fixture(scope="module")
def gemma_provider():
    """Return (ProviderConfig, ModelConfig) pointing at the local Gemma container."""
    from src.config import ModelConfig, ProviderConfig

    pc = ProviderConfig(
        name="gemma-local",
        type="openai",
        base_url=f"{_GEMMA_BASE_URL}/v1",
        api_key="not-required",
    )
    mc = ModelConfig(
        provider="gemma-local",
        model="gemma-3-270m",
        temperature=0.0,
        max_tokens=128,
    )
    return pc, mc


@pytest.fixture(scope="module")
def gemma_llm(gemma_provider):
    """LangChain ChatOpenAI instance backed by the Gemma container."""
    from src.agent.core import create_llm_from_provider_config

    pc, mc = gemma_provider
    return create_llm_from_provider_config(pc, mc)


@pytest.fixture
def session_state():
    """Fresh no-confirm SessionState for each test."""
    from src.orchestration.session_state import SessionState

    return SessionState(no_confirm=True)


@pytest.fixture(scope="module")
def safe_tools_dict() -> dict[str, Any]:
    """Minimal set of deterministic tools that require no external services."""
    from src.registry import ToolRegistry

    registry = ToolRegistry()
    from src.registry import LazyToolProxy

    all_tools = registry.load_all_tools()
    keep = {"calculate", "get_current_datetime", "word_count", "find_replace"}
    result = {}
    for name, t in all_tools.items():
        if name in keep:
            if isinstance(t, LazyToolProxy):
                t = t._resolve()
            if t is not None:
                result[name] = t
    return result


@pytest.fixture(scope="module")
def tool_registry() -> Any:
    """A ToolRegistry instance for passing to run_agent."""
    from src.registry import ToolRegistry

    registry = ToolRegistry()
    registry.load_all_tools()
    return registry


@pytest.fixture
def run_config(gemma_llm, safe_tools_dict, session_state):
    """AgentRunConfig suitable for single-turn integration tests."""
    from src.orchestration.run_config import AgentRunConfig

    return AgentRunConfig(
        llm=gemma_llm,
        system_prompt=(
            "You are a helpful assistant. "
            "Answer questions concisely. "
            "If you have access to tools, use them when appropriate."
        ),
        available_tools=safe_tools_dict,
        active_tools_list=[],  # agent starts with request_tools only
        max_context_tokens=4096,
        preset_tools=set(),
        context_compression=False,  # keep tests fast
        session_state=session_state,
        parallel_tool_execution=False,  # deterministic ordering
    )


# ===========================================================================
# 1 — Direct extract_ai_content tests
# ===========================================================================


class TestExtractAiContentGreeting:
    """Tests for extract_ai_content with greeting-style model outputs."""

    def test_extract_ai_content_with_reasoning_only(self):
        """extract_ai_content extracts reasoning_content even when content is empty."""
        from src.orchestration.runner import extract_ai_content

        # Simulate a message with no content but reasoning_content
        msg = type(
            "MockMessage",
            (),
            {
                "content": "",
                "tool_calls": [],
                "additional_kwargs": {"reasoning_content": "Hi there! How can I help you today?"},
            },
        )()

        result = extract_ai_content(msg)
        assert result is not None, "extract_ai_content should return reasoning_content"
        assert "Hi there" in result, f"Expected greeting, got: {result!r}"

    def test_extract_ai_content_with_empty_content_and_tool_calls(self):
        """extract_ai_content extracts reasoning_content when tool_calls is present."""
        from src.orchestration.runner import extract_ai_content

        # Simulate a message with tool_calls but no content, reasoning has the answer
        msg = type(
            "MockMessage",
            (),
            {
                "content": "",
                "tool_calls": [
                    {"name": "some_tool", "args": {}, "id": "tc-1", "type": "tool_call"}
                ],
                "additional_kwargs": {"reasoning_content": "Hello! I'm here to help."},
            },
        )()

        result = extract_ai_content(msg)
        assert result is not None, "extract_ai_content should extract reasoning_content"
        assert "Hello" in result, f"Expected greeting, got: {result!r}"

    def test_extract_ai_content_with_thinking_field(self):
        """extract_ai_content extracts thinking field when reasoning_content is missing."""
        from src.orchestration.runner import extract_ai_content

        msg = type(
            "MockMessage",
            (),
            {
                "content": "",
                "tool_calls": [],
                "additional_kwargs": {"thinking": "Hi, how can I assist you today?"},
            },
        )()

        result = extract_ai_content(msg)
        assert result is not None, "extract_ai_content should extract thinking content"
        assert "Hi" in result, f"Expected greeting, got: {result!r}"

    def test_extract_ai_content_with_empty_content_empty_tool_calls(self):
        """extract_ai_content returns None when content is empty and no reasoning."""
        from src.orchestration.runner import extract_ai_content

        msg = type(
            "MockMessage",
            (),
            {
                "content": "",
                "tool_calls": [],
                "additional_kwargs": {},
            },
        )()

        result = extract_ai_content(msg)
        assert result is None, "extract_ai_content should return None when no content"

    def test_extract_ai_content_with_normal_content(self):
        """extract_ai_content returns normal content when present."""
        from src.orchestration.runner import extract_ai_content

        msg = type(
            "MockMessage",
            (),
            {
                "content": "Hi there! How can I help you?",
                "tool_calls": [],
                "additional_kwargs": {},
            },
        )()

        result = extract_ai_content(msg)
        assert result == "Hi there! How can I help you?"


# ===========================================================================
# 2 — Integration tests: simple greetings via run_agent
# ===========================================================================


class TestGreetingResponses:
    """Integration tests for simple greeting responses via run_agent.

    These tests verify the bug fix for the issue where Gemma produces
    messages with content="" and tool_calls=[...] for greetings, with
    the actual response in reasoning_content.
    """

    def test_simple_greeting_hi(self, run_config, tool_registry):
        """Agent responds to simple greeting "Hi" with appropriate greeting content."""
        from src.orchestration.runner import run_agent

        response = run_agent(
            user_input="Hi",
            history_messages=[],
            registry=tool_registry,
            approvals=set(),
            config=run_config,
        )

        # The response must be a non-empty string
        assert isinstance(response, str), f"Expected str, got {type(response)}"
        assert response.strip(), "run_agent returned empty response for greeting"

        # The response should contain greeting content, not an apology or step limit
        response_lower = response.lower()
        assert (
            "hi" in response_lower or "hello" in response_lower or "hey" in response_lower
        ), f"Expected greeting content in: {response!r}"

        # Should NOT be a step-limit apology or empty response recovery message
        assert (
            "steps" not in response_lower or "step" not in response_lower
        ), f"Response should not be step-limit related: {response!r}"

    def test_simple_greeting_hello(self, run_config, tool_registry):
        """Agent responds to "Hello" with appropriate greeting content."""
        from src.orchestration.runner import run_agent

        response = run_agent(
            user_input="Hello",
            history_messages=[],
            registry=tool_registry,
            approvals=set(),
            config=run_config,
        )

        assert isinstance(response, str) and response.strip()
        response_lower = response.lower()
        assert (
            "hi" in response_lower or "hello" in response_lower or "hey" in response_lower
        ), f"Expected greeting content in: {response!r}"

    def test_simple_greeting_good_morning(self, run_config, tool_registry):
        """Agent responds to "Good morning" with appropriate greeting content."""
        from src.orchestration.runner import run_agent

        response = run_agent(
            user_input="Good morning",
            history_messages=[],
            registry=tool_registry,
            approvals=set(),
            config=run_config,
        )

        assert isinstance(response, str) and response.strip()
        response_lower = response.lower()
        assert (
            "good morning" in response_lower or "hi" in response_lower or "hello" in response_lower
        ), f"Expected greeting content in: {response!r}"

    def test_simple_greeting_hey_there(self, run_config, tool_registry):
        """Agent responds to "Hey there" with appropriate greeting content."""
        from src.orchestration.runner import run_agent

        response = run_agent(
            user_input="Hey there",
            history_messages=[],
            registry=tool_registry,
            approvals=set(),
            config=run_config,
        )

        assert isinstance(response, str) and response.strip()
        response_lower = response.lower()
        assert (
            "hey" in response_lower
            or "hello" in response_lower
            or "hi" in response_lower
            or "ready" in response_lower
        )

    def test_greeting_with_empty_history(self, run_config, tool_registry):
        """Agent handles greeting with empty history correctly."""
        from src.orchestration.runner import run_agent

        # No history at all
        response = run_agent(
            user_input="Hi",
            history_messages=[],
            registry=tool_registry,
            approvals=set(),
            config=run_config,
        )

        assert isinstance(response, str) and response.strip()
        # Should be a greeting, not an error or recovery message
        assert "empty" not in response.lower()
        assert "recovery" not in response.lower()

    def test_greeting_with_existing_history(self, run_config, tool_registry):
        """Agent handles greeting after prior conversation history."""
        from langchain_core.messages import AIMessage, HumanMessage

        from src.orchestration.runner import run_agent

        history = [
            HumanMessage(content="Hello, I have a question."),
            AIMessage(content="Hello! I'm here to help. What's your question?"),
        ]

        response = run_agent(
            user_input="Hi",
            history_messages=history,
            registry=tool_registry,
            approvals=set(),
            config=run_config,
        )

        assert isinstance(response, str) and response.strip()
        # Should respond to greeting, not repeat or recover
        assert "empty" not in response.lower()
        assert "step" not in response.lower()


# ===========================================================================
# 3 — Agent behavior verification tests
# ===========================================================================


class TestAgentGreetingBehavior:
    """Tests to verify agent behavior with greetings doesn't trigger recovery flow."""

    def test_agent_does_not_trigger_recovery_for_greeting(self, run_config, tool_registry):
        """Agent completes greeting interaction without recovery/step-limit messages."""
        from src.orchestration.runner import run_agent

        # Try multiple greeting variations
        greetings = ["Hi", "Hello", "Hey", "Good morning", "Greetings"]

        for greeting in greetings:
            response = run_agent(
                user_input=greeting,
                history_messages=[],
                registry=tool_registry,
                approvals=set(),
                config=run_config,
            )

            assert isinstance(response, str), f"Response type for '{greeting}': {type(response)}"
            assert response.strip(), f"Response is empty for greeting: {greeting!r}"

            # Should not contain recovery or step-limit indicators
            response_lower = response.lower()
            assert (
                "step" not in response_lower or len(response) > 100
            ), f"Response appears to be step-limit related: {response!r}"
            assert (
                "recovery" not in response_lower
            ), f"Response indicates recovery flow: {response!r}"

    def test_greeting_response_contains_meaningful_content(self, run_config, tool_registry):
        """Greeting responses contain actual greeting content, not placeholders."""
        from src.orchestration.runner import run_agent

        response = run_agent(
            user_input="Hi",
            history_messages=[],
            registry=tool_registry,
            approvals=set(),
            config=run_config,
        )

        # Response should have substantial content (not just "I'll help" without greeting)
        words = response.split()
        assert len(words) >= 2, f"Greeting response too short: {response!r}"

        # Should contain actual greeting words
        response_lower = response.lower()
        assert any(
            word in response_lower for word in ["hi", "hello", "hey", "good morning"]
        ), f"Response doesn't contain greeting: {response!r}"

    def test_greeting_response_is_not_empty_string(self, run_config, tool_registry):
        """Regression test: Agent must never return empty string for greeting."""
        from src.orchestration.runner import run_agent

        # This is the core regression - we must never get content=""
        response = run_agent(
            user_input="Hi",
            history_messages=[],
            registry=tool_registry,
            approvals=set(),
            config=run_config,
        )

        # Core assertions that would fail if the bug returns
        assert response != "", "Agent returned empty string (BUG: content='')"
        assert response != " ", "Agent returned whitespace-only string"
        assert response is not None, "Agent returned None"


# ===========================================================================
# 4 — Multi-turn greeting tests
# ===========================================================================


class TestMultiTurnGreeting:
    """Tests for greeting responses across multiple conversation turns."""

    def test_greeting_followed_by_question(self, gemma_llm, tool_registry):
        """Agent handles greeting -> question flow correctly."""
        from langchain_core.messages import AIMessage, HumanMessage

        from src.orchestration.run_config import AgentRunConfig
        from src.orchestration.runner import run_agent
        from src.orchestration.session_state import SessionState

        cfg = AgentRunConfig(
            llm=gemma_llm,
            system_prompt="You are a helpful assistant. Be concise.",
            available_tools={},
            active_tools_list=[],
            max_context_tokens=4096,
            preset_tools=set(),
            context_compression=False,
            session_state=SessionState(no_confirm=True),
            parallel_tool_execution=False,
        )

        # Turn 1: Greeting
        greeting = run_agent(
            "Hi",
            history_messages=[],
            registry=tool_registry,
            approvals=set(),
            config=cfg,
        )
        assert isinstance(greeting, str) and greeting.strip()

        # Turn 2: Question with history
        question = run_agent(
            "What is 5 + 3?",
            history_messages=[
                HumanMessage(content="Hi"),
                AIMessage(content=greeting),
            ],
            registry=tool_registry,
            approvals=set(),
            config=cfg,
        )
        assert isinstance(question, str) and question.strip()

        # Should answer the question
        assert (
            "8" in question or "8" in question.lower() or "5 + 3" in question
        ), f"Expected answer '8' in: {question!r}"

    def test_multiple_greetings_in_sequence(self, gemma_llm, tool_registry):
        """Agent handles multiple greetings in sequence without issues."""

        from src.orchestration.run_config import AgentRunConfig
        from src.orchestration.runner import run_agent
        from src.orchestration.session_state import SessionState

        cfg = AgentRunConfig(
            llm=gemma_llm,
            system_prompt="You are a helpful assistant.",
            available_tools={},
            active_tools_list=[],
            max_context_tokens=4096,
            preset_tools=set(),
            context_compression=False,
            session_state=SessionState(no_confirm=True),
            parallel_tool_execution=False,
        )

        # Multiple greetings in sequence
        for i, greeting in enumerate(["Hi", "Hello", "Hey there"]):
            response = run_agent(
                greeting,
                history_messages=[],
                registry=tool_registry,
                approvals=set(),
                config=cfg,
            )
            assert isinstance(response, str) and response.strip(), f"Turn {i}: empty response"
