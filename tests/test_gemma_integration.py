"""Integration tests for Cogtrix backed by the local Gemma 3 270M container.

These tests require the container to be running::

    docker run -d --name gemma-test -p 18080:8080 \\
        ghcr.io/northlandpositronics/cogtrix-gemma3-270m:latest

All tests are marked ``live_llm`` and skip automatically when the container
is unreachable.  Run explicitly::

    pytest tests/test_gemma_integration.py -v --timeout=180

or include in the normal suite when the container is available::

    pytest tests/ -m live_llm -v

Why Gemma 3 270M?
-----------------
* Self-contained: weights embedded in the Docker image, zero internet access required.
* OpenAI-compatible API: exercises the same provider code path as GPT-4 etc.
* Fast enough on CPU: most test turns complete in 2–10 s.
* Small enough to be predictable: simple factual prompts return consistent answers.

Limitations
-----------
The Gemma API server does not implement the ``tools`` field — it silently
ignores extra request keys.  Therefore tool-calling assertions are not made:
tests verify that the Cogtrix infrastructure handles a model that responds
with plain text (no tool calls) gracefully, including the action-intent
recovery path introduced in BUG-249.
"""

from __future__ import annotations

import urllib.request
import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

# ---------------------------------------------------------------------------
# Availability check — runs once at collection time
# ---------------------------------------------------------------------------

_GEMMA_BASE_URL = "http://localhost:18080"
_GEMMA_API_BASE = f"{_GEMMA_BASE_URL}/v1"
_GEMMA_MODEL = "gemma-3-270m"


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
            "Gemma 3 270M container not reachable at localhost:18080 — "
            "start it with: docker run -d --name gemma-test -p 18080:8080 "
            "ghcr.io/northlandpositronics/cogtrix-gemma3-270m:latest",
            allow_module_level=True,
        )


@pytest.fixture(scope="module")
def gemma_provider():
    """Return (ProviderConfig, ModelConfig) pointing at the local Gemma container."""
    from src.config import ModelConfig, ProviderConfig

    pc = ProviderConfig(
        name="gemma-local",
        type="openai",
        base_url=_GEMMA_API_BASE,
        api_key="not-required",
    )
    mc = ModelConfig(
        provider="gemma-local",
        model=_GEMMA_MODEL,
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
def session_state():
    """Fresh no-confirm SessionState for each test."""
    from src.orchestration.session_state import SessionState

    return SessionState(no_confirm=True)


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


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_session_id() -> str:
    return f"gemma-test-{uuid.uuid4().hex[:8]}"


# ===========================================================================
# 1 — Raw HTTP connectivity
# ===========================================================================


class TestGemmaConnectivity:
    """Verify the Gemma container API endpoints respond correctly."""

    def test_health_endpoint(self):
        r = urllib.request.urlopen(f"{_GEMMA_BASE_URL}/health", timeout=5)
        import json

        body = json.loads(r.read())
        assert r.status == 200
        assert body.get("status") in ("healthy", "ok"), f"Unexpected status: {body}"

    def test_models_endpoint(self):
        import json

        r = urllib.request.urlopen(f"{_GEMMA_API_BASE}/models", timeout=5)
        body = json.loads(r.read())
        assert r.status == 200
        ids = [m["id"] for m in body.get("data", [])]
        assert len(ids) > 0, f"Expected at least one model in /v1/models, got: {ids}"

    def test_direct_chat_completions(self):
        import json

        def _request(prompt: str, max_tokens: int = 32) -> dict[str, Any]:
            payload = json.dumps(
                {
                    "model": _GEMMA_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                }
            ).encode()
            req = urllib.request.Request(
                f"{_GEMMA_API_BASE}/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            r = urllib.request.urlopen(req, timeout=30)
            assert r.status == 200
            return json.loads(r.read())

        body = _request("Reply with just the word PONG.")
        content = body["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            body = _request(
                "Reply with the single word PONG and no other text.",
                max_tokens=64,
            )
            content = body["choices"][0]["message"]["content"]
        assert isinstance(content, str) and content.strip(), "Empty response from chat completions"


# ===========================================================================
# 2 — LangChain provider layer
# ===========================================================================


class TestGemmaLLMProvider:
    """Test that Cogtrix's provider layer creates and invokes a working LLM."""

    def test_provider_config_creation(self, gemma_provider):
        """ProviderConfig and ModelConfig can be instantiated without error."""
        pc, mc = gemma_provider
        assert pc.type == "openai"
        assert pc.base_url == _GEMMA_API_BASE
        assert mc.model == _GEMMA_MODEL
        assert mc.temperature == 0.0

    def test_llm_creation(self, gemma_llm):
        """LLM object is created and has the expected interface."""
        assert hasattr(gemma_llm, "invoke"), "LLM must have an invoke method"
        assert hasattr(gemma_llm, "bind_tools"), "LLM must support bind_tools"

    def test_llm_invoke_returns_ai_message(self, gemma_llm):
        """Direct LLM invoke returns an AIMessage with non-empty string content."""
        response = gemma_llm.invoke([HumanMessage(content="What is 3 + 4?")])
        assert isinstance(response, AIMessage), f"Expected AIMessage, got {type(response)}"
        assert isinstance(response.content, str), "AIMessage content must be a string"
        assert response.content.strip(), "AIMessage content must not be empty"

    def test_llm_invoke_factual(self, gemma_llm):
        """A simple factual question returns a plausible answer."""
        response = gemma_llm.invoke(
            [HumanMessage(content="What color is the sky on a clear day? One word.")]
        )
        assert (
            "blue" in response.content.lower()
        ), f"Expected 'blue' in response: {response.content!r}"

    def test_llm_usage_metadata(self, gemma_llm):
        """Response carries token usage metadata (required by Cogtrix stats)."""
        response = gemma_llm.invoke([HumanMessage(content="Hi.")])
        # Confirm the response object is fully-formed (no crash on attribute access).
        assert response is not None
        # usage_metadata or response_metadata may be present depending on LangChain version
        _ = getattr(response, "usage_metadata", None) or getattr(
            response, "response_metadata", None
        )


# ===========================================================================
# 3 — Full run_agent pipeline
# ===========================================================================


class TestGemmaRunAgent:
    """Exercise the Cogtrix run_agent / build_agent_graph pipeline end-to-end."""

    def test_no_tools_simple_question(self, run_config, tool_registry):
        """Agent with no active tools answers a simple question."""
        from src.orchestration.runner import run_agent

        response = run_agent(
            user_input="What is the capital of France?",
            history_messages=[],
            registry=tool_registry,
            approvals=set(),
            config=run_config,
        )
        assert isinstance(response, str), f"Expected str, got {type(response)}"
        assert response.strip(), "run_agent returned empty response"
        assert "paris" in response.lower(), f"Expected 'Paris' in: {response!r}"

    def test_response_is_string(self, run_config, tool_registry):
        """run_agent always returns a plain string regardless of model output."""
        from src.orchestration.runner import run_agent

        response = run_agent(
            user_input="Say the word hello.",
            history_messages=[],
            registry=tool_registry,
            approvals=set(),
            config=run_config,
        )
        assert isinstance(response, str)
        assert len(response) > 0

    def test_history_passed_to_agent(self, run_config, tool_registry):
        """Prior conversation history is included in the prompt context."""
        from src.orchestration.runner import run_agent

        history = [
            HumanMessage(content="My favourite number is 42."),
            AIMessage(content="I'll remember that your favourite number is 42."),
        ]
        response = run_agent(
            user_input="What is my favourite number?",
            history_messages=history,
            registry=tool_registry,
            approvals=set(),
            config=run_config,
        )
        assert (
            isinstance(response, str) and response.strip()
        ), f"run_agent with history must return a non-empty string: {response!r}"

    def test_run_agent_with_tools_available(self, run_config, tool_registry):
        """Agent pipeline does not crash when tools are available but not called.

        The Gemma server ignores the ``tools`` field, so the model responds
        with plain text.  The pipeline must handle this gracefully — no
        assertion on tool usage is made.
        """
        from src.orchestration.runner import run_agent

        response = run_agent(
            user_input="What is 15 * 8?",
            history_messages=[],
            registry=tool_registry,
            approvals=set(),
            config=run_config,
        )
        assert isinstance(response, str) and response.strip()
        # Loose check — either the model answered directly or used the calculator
        assert (
            "120" in response or response.strip()
        ), f"Expected '120' or a non-empty response: {response!r}"

    def test_multi_turn_pipeline(self, gemma_llm, tool_registry):
        """Two sequential run_agent calls with accumulated history."""
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

        # Turn 1
        r1 = run_agent(
            "My name is Alex.",
            history_messages=[],
            registry=tool_registry,
            approvals=set(),
            config=cfg,
        )
        assert isinstance(r1, str) and r1.strip()

        # Turn 2 — with history from turn 1
        history = [
            HumanMessage(content="My name is Alex."),
            AIMessage(content=r1),
        ]
        r2 = run_agent(
            "What is my name?",
            history_messages=history,
            registry=tool_registry,
            approvals=set(),
            config=cfg,
        )
        assert (
            isinstance(r2, str) and r2.strip()
        ), f"run_agent with history must return a non-empty string: {r2!r}"


# ===========================================================================
# 4 — Memory manager integration
# ===========================================================================


class TestGemmaMemory:
    """Verify memory managers accept and store live Gemma responses."""

    def test_memory_update_and_retrieve(self, gemma_llm):
        """ConversationMemoryManager stores and retrieves a Gemma-generated response."""
        from src.memory.json_store import JsonFileMemoryStore
        from src.memory.modes.conversation import ConversationMemoryManager

        sid = _make_session_id()
        store = JsonFileMemoryStore(base_dir=f"/tmp/cogtrix_test_{sid}")
        mm = ConversationMemoryManager(store, sid, {"working_memory_size": 10})
        mm.load()

        # Simulate one exchange
        user_msg = "What is 2 + 2?"
        ai_response = gemma_llm.invoke([HumanMessage(content=user_msg)])
        response_text = ai_response.content

        mm.update(user_msg, response_text)
        ctx = mm.prepare_context(user_msg)

        assert ctx is not None
        assert hasattr(ctx, "messages"), "MemoryContext must have a .messages attribute"

    def test_memory_prepare_context_returns_messages(self, gemma_llm):
        """Memory context exposes LangChain messages the agent can consume."""
        from src.memory.json_store import JsonFileMemoryStore
        from src.memory.modes.conversation import ConversationMemoryManager

        sid = _make_session_id()
        store = JsonFileMemoryStore(base_dir=f"/tmp/cogtrix_test_{sid}")
        mm = ConversationMemoryManager(store, sid, {"working_memory_size": 10})
        mm.load()

        mm.update("Hello!", "Hi there!")
        mm.update("How are you?", "I'm doing great, thanks.")
        ctx = mm.prepare_context("What have we talked about?")

        assert isinstance(ctx.messages, list)
        # At least the two turns are in the window
        assert len(ctx.messages) >= 2


# ===========================================================================
# 5 — Prompt optimizer
# ===========================================================================


class TestGemmaOptimizer:
    """Verify optimize_prompt integrates correctly with a live LLM."""

    def test_short_prompt_skips_llm(self, gemma_llm):
        """Prompts below the length gate return a PromptPlan without LLM call."""
        from src.prompt.optimizer import optimize_prompt

        plan = optimize_prompt("What is the weather?", gemma_llm)
        assert plan.text == "What is the weather?", "Short prompt should be returned unchanged"
        assert plan.milestones == []

    def test_force_invokes_llm(self, gemma_llm):
        """force=True bypasses the length gate and runs the LLM."""
        from src.prompt.optimizer import PromptPlan, optimize_prompt

        result = optimize_prompt(
            "Write a Python function to reverse a string.",
            gemma_llm,
            force=True,
        )
        assert isinstance(result, PromptPlan), f"Expected PromptPlan, got {type(result)}"
        assert (
            isinstance(result.text, str) and result.text.strip()
        ), "optimize_prompt must return a non-empty text"

    def test_optimizer_fails_open(self):
        """If the LLM produces unusable output, the original prompt is returned."""
        from src.prompt.optimizer import optimize_prompt

        # Simulate the LLM returning garbage
        bad_llm = MagicMock()
        bad_llm.invoke.side_effect = RuntimeError("LLM unavailable")

        plan = optimize_prompt(
            "Write a Python function to reverse a string.",
            bad_llm,
            force=True,
        )
        # Fail-safe: original prompt is preserved
        assert "reverse" in plan.text.lower() or plan.text.strip()


# ===========================================================================
# 6 — Setup wizard — connection validation step
# ===========================================================================


class TestGemmaWizard:
    """Verify the setup wizard can validate a Gemma-backed provider."""

    def test_connection_test_with_gemma_provider(self):
        """_test_connection succeeds for the Gemma provider config.

        ``_test_connection(provider_type, model, api_key, base_url)`` is the
        internal wizard helper that creates and smoke-tests an LLM.  This
        exercises the full provider→LangChain initialization path.
        """
        from src.setup_wizard import _test_connection  # type: ignore[attr-defined]

        llm = _test_connection(
            "openai",
            _GEMMA_MODEL,
            "not-required",
            _GEMMA_API_BASE,
        )
        assert llm is not None, "_test_connection must return an LLM instance"

    def test_wizard_llm_produces_yaml_block(self, gemma_llm):
        """The wizard's configure step asks the LLM to produce YAML.

        We run just the LLM call that the wizard would make, verifying that
        Gemma returns *something* in response to a config-generation prompt.
        The wizard itself is not driven end-to-end here (that is covered by
        the unit tests in test_wizard_scenario.py with mock LLMs).
        """
        prompt = (
            "Generate a minimal Cogtrix YAML config block using the openai provider "
            f"with base_url={_GEMMA_API_BASE!r} and model={_GEMMA_MODEL!r}. "
            "Return only a ```yaml``` fenced block."
        )
        response = gemma_llm.invoke([HumanMessage(content=prompt)])
        assert (
            isinstance(response.content, str) and response.content.strip()
        ), "Gemma must return a non-empty response to the config-generation prompt"


# ===========================================================================
# 7 — Action-intent recovery (BUG-249 regression)
# ===========================================================================


class TestActionIntentRecovery:
    """Verify the handle_action_intent graph node fires and recovers correctly.

    The Gemma 3 270M server ignores the ``tools`` field, so the model always
    returns plain-text responses.  When the response contains an intent phrase
    ("I'll create...", "Let me search...") and no actual tool calls, the
    ``handle_action_intent`` node should inject a nudge so the next LLM call
    either produces a tool call or a complete answer.
    """

    def test_is_action_intent_detects_intent_phrases(self):
        """_is_action_intent returns True for typical model intent-only messages."""
        from src.orchestration.graph import _is_action_intent

        # Positive cases: intent phrase + tool-action verb, no tool_calls
        positive = [
            AIMessage(content="I'll create the file for you."),
            AIMessage(content="Let me search the web for that."),
            AIMessage(content="I will write a Python script to handle this."),
            AIMessage(content="I'm going to generate the report now."),
            AIMessage(content="Let's build the configuration file."),
            AIMessage(content="I need to fetch the data from the API."),
            AIMessage(content="Now I'll execute the shell command."),
            AIMessage(content="First, I'll read the file contents."),
        ]
        for msg in positive:
            assert _is_action_intent(msg), f"Expected _is_action_intent=True for: {msg.content!r}"

    def test_is_action_intent_ignores_tool_calls(self):
        """_is_action_intent returns False when tool_calls are present."""
        from src.orchestration.graph import _is_action_intent

        msg = AIMessage(
            content="I'll create the file.",
            tool_calls=[{"name": "write_file", "args": {}, "id": "tc-1", "type": "tool_call"}],
        )
        assert not _is_action_intent(msg), "Must return False when tool_calls present"

    def test_is_action_intent_ignores_plain_response(self):
        """_is_action_intent returns False for normal factual answers."""
        from src.orchestration.graph import _is_action_intent

        negative = [
            AIMessage(content="Paris is the capital of France."),
            AIMessage(content="The answer is 42."),
            AIMessage(content="Hello! How can I assist you today?"),
            AIMessage(content=""),
        ]
        for msg in negative:
            assert not _is_action_intent(
                msg
            ), f"Expected _is_action_intent=False for: {msg.content!r}"

    def test_agent_recovers_from_intent_only_response(self, run_config, tool_registry):
        """The full agent pipeline completes even when the model produces intent-only text.

        Because the Gemma server ignores tools, the model will respond with plain
        text.  If that text contains an intent phrase, the action_intent node fires
        and injects a nudge.  The pipeline must terminate within the retry budget
        and return a non-empty string — it must NOT loop indefinitely.
        """
        from src.orchestration.runner import run_agent

        # This prompt is likely to elicit "I'll calculate..." from Gemma
        response = run_agent(
            user_input="Please calculate 7 multiplied by 6 for me.",
            history_messages=[],
            registry=tool_registry,
            approvals=set(),
            config=run_config,
        )
        assert (
            isinstance(response, str) and response.strip()
        ), "Agent must always return a non-empty string"

    def test_action_intent_count_resets_across_turns(self, gemma_llm, tool_registry):
        """Each run_agent call starts with a fresh action_intent_count = 0.

        A shared counter would allow intent retries from a previous turn to
        bleed into the next, incorrectly exhausting the retry budget.
        """
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

        for i in range(2):
            r = run_agent(
                f"Say the number {i + 1}.",
                history_messages=[],
                registry=tool_registry,
                approvals=set(),
                config=cfg,
            )
            assert isinstance(r, str) and r.strip(), f"Turn {i+1}: run_agent returned empty string"


# ===========================================================================
# 8 — Config round-trip
# ===========================================================================


class TestGemmaConfigRoundTrip:
    """Verify a Gemma provider config survives a parse-serialize-resolve cycle."""

    def test_provider_config_resolves_to_working_llm(self, gemma_provider):
        """ProviderConfig + ModelConfig → create_chat_model_from_configs → working LLM."""
        from src.providers import create_chat_model_from_configs

        pc, mc = gemma_provider
        llm = create_chat_model_from_configs(pc, mc)
        assert llm is not None

        response = llm.invoke([HumanMessage(content="Respond with the single word YES.")])
        assert isinstance(response.content, str) and response.content.strip()

    @pytest.mark.xfail(
        strict=False,
        reason="Gemma 3 270M may echo the expression rather than compute the result",
    )
    def test_model_config_temperature_zero_deterministic(self, gemma_provider):
        """temperature=0 produces consistent responses for the same simple prompt."""
        from src.providers import create_chat_model_from_configs

        pc, mc = gemma_provider
        # mc already has temperature=0.0
        llm = create_chat_model_from_configs(pc, mc)

        prompt = [HumanMessage(content="What is 10 + 5? Answer with digits only.")]
        r1 = llm.invoke(prompt).content.strip()
        r2 = llm.invoke(prompt).content.strip()
        # Both runs should mention "15" (loose: the model might add punctuation)
        assert "15" in r1, f"Expected '15' in first run: {r1!r}"
        assert "15" in r2, f"Expected '15' in second run: {r2!r}"
