"""
Gate 2 evaluation runner — execute domain scenarios against real Cogtrix+LLM stacks.

Gate 2 runs after Gate 1 (synthetic harness) passes. It tests the full agent
stack — tool calling, context management, MCP connections — with real LLMs and
Finance/Procurement domain scenarios.

Design decisions:
  - Trigger: manual + optional nightly after Gate 1 passes. NOT on every PR push
    (API cost ~$0.45/smoke run, ~$3/full matrix). Guard: --live flag in pytest.
  - Scoring: binary pass/fail per scenario (wrong approval chain = fail).
    Partial credit deferred to judge.py once domain rubrics are finalised.
  - Dashboard: internal Markdown/CSV only for now; extend to public "certified
    model" page once scoring is validated against real workflows.
  - Provider support: Anthropic, OpenAI, DeepSeek, OpenAI-compatible (Qwen,
    Kimi), Google. Each provider needs the matching env key in models.yaml.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_MODELS_YAML = Path(__file__).parent / "models.yaml"
_SCENARIOS_DIR = Path(__file__).parent / "scenarios"

# Agent LLM temperature: NOT pinned.  We tried _AGENT_TEMPERATURE = 0.0 in
# PR #1276 to make Gate 2 deterministic (#1268) but DeepSeek-V3 routed
# through OpenRouter then deterministically falls into an empty-response
# dead-end on certain prompts — measured 60–80% empty-response rate on
# procurement_supplier_registration at T=0.0, dropping to 0% at the
# provider default.  The strict gate (ci_gate2._final_passed) catches
# real partial completion regardless of temperature, so we let the
# provider default ride and rely on the empty-response retry for the
# residual flake.  The judge LLM still runs at 0.0 in judge.py — that
# scoring path has no equivalent dead-end.


# ── Model config ─────────────────────────────────────────────────────────────


_KEY_PRIORITY = [
    "OPENROUTER_API_KEY",
    "CEREBRAS_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
]

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"

# Maps each priority key to the set of native env_keys it can satisfy.
# OPENROUTER_API_KEY can satisfy any model that has openrouter_model_id.
_KEY_COVERS: dict[str, set[str]] = {
    "OPENROUTER_API_KEY": {"*"},  # wildcard — covers all models via routing
    "CEREBRAS_API_KEY": {"CEREBRAS_API_KEY"},
    "DEEPSEEK_API_KEY": {"DEEPSEEK_API_KEY"},
    "OPENAI_API_KEY": {"OPENAI_API_KEY"},
    "ANTHROPIC_API_KEY": {"ANTHROPIC_API_KEY"},
}


def resolve_active_key() -> tuple[str, str] | None:
    """Return (key_name, api_key_value) for the first *present* key in priority order.

    Only checks that the env var is set and non-empty. Actual validity (wrong
    key, expired account, no credits) is determined at runtime during the first
    LLM call. Use try_keys_in_order() for full fallback behaviour.
    """
    for key_name in _KEY_PRIORITY:
        value = os.environ.get(key_name, "")
        if value:
            return key_name, value
    return None


def _is_auth_or_quota_error(exc: Exception) -> bool:
    """Return True when the exception signals an invalid key or quota exhaustion."""
    msg = str(exc).lower()
    return any(
        kw in msg
        for kw in (
            "401",
            "403",
            "authentication",
            "unauthorized",
            "invalid api key",
            "quota",
            "insufficient_quota",
            "credits",
            "billing",
            "payment",
            "no money",
            "account",
        )
    )


@dataclass
class ModelConfig:
    """Runtime config for a single model under test."""

    id: str
    provider: str
    display_name: str
    tier: str
    smoke: bool
    env_key: str
    model_id: str
    base_url: str | None = None
    openrouter_model_id: str | None = None
    # USD per 1,000,000 tokens.  When both values are populated the runner
    # estimates per-scenario cost and the Gate 2 cost ceiling (D2) becomes
    # active.  Models without prices opt out of the ceiling.
    input_price_per_1m: float | None = None
    output_price_per_1m: float | None = None


def load_model_registry(path: Path = _MODELS_YAML) -> list[ModelConfig]:
    """Load and parse models.yaml into ModelConfig objects."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return [ModelConfig(**m) for m in data.get("models", [])]


def get_model(model_id: str) -> ModelConfig:
    """Return a ModelConfig by id or raise KeyError."""
    registry = load_model_registry()
    by_id = {m.id: m for m in registry}
    if model_id not in by_id:
        raise KeyError(f"Unknown model id '{model_id}'. Available: {sorted(by_id)}")
    return by_id[model_id]


def smoke_models() -> list[ModelConfig]:
    """Return all models flagged smoke=true in the registry."""
    return [m for m in load_model_registry() if m.smoke]


# ── Scenario config ───────────────────────────────────────────────────────────


@dataclass
class EvalScenario:
    """A single Finance/Procurement evaluation scenario."""

    id: str
    domain: str  # "procurement" | "finance"
    title: str
    description: str
    user_prompt: str
    system_prompt: str
    tools_required: list[str]
    expected_outcome: str
    success_criteria: list[str]
    max_turns: int = 20
    timeout_seconds: int = 120
    tags: list[str] = field(default_factory=list)
    budget_usd_estimate: float = 0.05
    # Optional per-tool descriptions used when binding stub tools.  Some
    # models (notably deepseek-chat) decline to invoke tools whose
    # description is uselessly generic — provide concrete descriptions
    # here for scenarios where that matters.  Falls back to a directive
    # default when omitted.
    tool_descriptions: dict[str, str] = field(default_factory=dict)
    # Tools the agent has available but is NOT required to call.  Used by
    # safety/refusal scenarios where a forbidden tool must be present in
    # the toolset (so the agent could call it) yet the assertion is that
    # the agent did NOT call it.  Merged with tools_required when stubs
    # are built; selection-rate scoring still uses tools_required only.
    tools_available: list[str] = field(default_factory=list)


def load_scenario(path: Path) -> EvalScenario:
    """Load a scenario YAML file into an EvalScenario."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return EvalScenario(**data)


def load_all_scenarios(domain: str | None = None) -> list[EvalScenario]:
    """Load all scenario YAMLs, optionally filtered by domain."""
    scenarios = []
    for yaml_file in sorted(_SCENARIOS_DIR.rglob("*.yaml")):
        # Skip models.yaml at root level
        if yaml_file.parent == Path(__file__).parent:
            continue
        try:
            s = load_scenario(yaml_file)
            if domain is None or s.domain == domain:
                scenarios.append(s)
        except Exception as exc:
            import warnings

            warnings.warn(f"Failed to load scenario {yaml_file}: {exc}", stacklevel=2)
    return scenarios


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class EvalResult:
    """Result of running one scenario against one model."""

    scenario_id: str
    model_id: str
    model_display_name: str
    passed: bool
    tool_calls_made: list[str]
    tool_calls_required: list[str]
    turns_used: int
    elapsed_seconds: float
    final_response: str
    error: str | None = None
    notes: str = ""

    # Tier 1 metrics (parallel with Gate 1 synthetic harness)
    tool_selection_rate: float = 0.0  # % of required tools called
    task_completion: bool = False

    # Token usage and estimated cost (D2 cost-ceiling check).  Populated
    # only when AIMessage.usage_metadata is present and the model has
    # input/output prices configured in models.yaml.  Zero when either is
    # missing — the cost ceiling treats zero as "unknown, do not gate".
    prompt_tokens: int = 0
    completion_tokens: int = 0
    actual_cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "model_id": self.model_id,
            "model_display_name": self.model_display_name,
            "passed": self.passed,
            "tool_calls_made": self.tool_calls_made,
            "tool_calls_required": self.tool_calls_required,
            "turns_used": self.turns_used,
            "elapsed_seconds": self.elapsed_seconds,
            "final_response": self.final_response[:500],
            "error": self.error,
            "notes": self.notes,
            "tool_selection_rate": self.tool_selection_rate,
            "task_completion": self.task_completion,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "actual_cost_usd": round(self.actual_cost_usd, 6),
        }


# ── LLM instantiation ─────────────────────────────────────────────────────────


def _build_llm(
    model: ModelConfig,
    temperature: float | None = None,
    active_key: tuple[str, str] | None = None,
) -> Any:
    """Return a LangChain-compatible LLM for the given model config.

    Args:
        model: Model configuration.
        temperature: Optional sampling temperature (0.0 = deterministic).
        active_key: (key_name, api_key_value) from resolve_active_key().
            When provided, this key takes precedence over the model's native
            env_key, routing through OpenRouter or Cerebras as appropriate.

    Raises:
        ImportError: If the required provider package is not installed.
        OSError: If no usable API key is available for this model.
    """
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {}
    if temperature is not None:
        kwargs["temperature"] = temperature

    # ── Route through the active priority key when available ──────────────────
    if active_key is not None:
        key_name, api_key = active_key

        if key_name == "OPENROUTER_API_KEY":
            or_model_id = model.openrouter_model_id or model.model_id
            return ChatOpenAI(
                model=or_model_id,
                api_key=api_key,
                base_url=_OPENROUTER_BASE_URL,
                **kwargs,
            )

        if key_name == "CEREBRAS_API_KEY":
            # Cerebras uses OpenAI-compatible API
            return ChatOpenAI(
                model=model.model_id,
                api_key=api_key,
                base_url=_CEREBRAS_BASE_URL,
                **kwargs,
            )

        # For all other priority keys (DEEPSEEK, OPENAI, ANTHROPIC),
        # fall through to native provider routing below using the resolved key.
    else:
        api_key = os.environ.get(model.env_key, "")

    if not api_key:
        raise OSError(f"Model '{model.id}' requires env var {model.env_key} to be set.")

    # ── Native provider routing ───────────────────────────────────────────────
    if model.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model.model_id, api_key=api_key, **kwargs)

    if model.provider == "openai":
        return ChatOpenAI(model=model.model_id, api_key=api_key, **kwargs)

    if model.provider == "openai_compatible":
        if not model.base_url:
            raise ValueError(f"Model '{model.id}' is openai_compatible but has no base_url.")
        return ChatOpenAI(model=model.model_id, api_key=api_key, base_url=model.base_url, **kwargs)

    if model.provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model.model_id, google_api_key=api_key, **kwargs)

    if model.provider == "deepseek":
        return ChatOpenAI(
            model=model.model_id,
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            **kwargs,
        )

    raise ValueError(f"Unknown provider '{model.provider}' for model '{model.id}'.")


# ── Runner ────────────────────────────────────────────────────────────────────


def run_scenario(
    scenario: EvalScenario,
    model: ModelConfig,
    active_key: tuple[str, str] | None = None,
) -> EvalResult:
    """Run a single evaluation scenario against a real Cogtrix+LLM stack.

    This exercises the FULL Cogtrix agent graph — tool calling, context
    management, memory — with a live LLM.  It is intentionally NOT mocked
    (that is Gate 1's job).

    Args:
        scenario: The scenario to run.
        model: The model config to use.

    Returns:
        EvalResult capturing pass/fail, tool usage, and timing.
    """
    from langchain_core.messages import HumanMessage

    from src.orchestration.graph import build_agent_graph

    start = time.monotonic()

    # Build a minimal tool registry for the scenario.
    # Real MCP tools can be injected here when available; for the CI smoke
    # subset we use stub tools that signal their name in results.  Optional
    # tools (tools_available) are merged so the agent CAN call them but is
    # not required to — used by refusal scenarios.
    all_tool_names = list(scenario.tools_required)
    for name in scenario.tools_available:
        if name not in all_tool_names:
            all_tool_names.append(name)
    tool_stubs = _build_stub_tools(all_tool_names, scenario.tool_descriptions)

    try:
        llm = _build_llm(model, active_key=active_key)
    except (OSError, ImportError) as exc:
        return EvalResult(
            scenario_id=scenario.id,
            model_id=model.id,
            model_display_name=model.display_name,
            passed=False,
            tool_calls_made=[],
            tool_calls_required=scenario.tools_required,
            turns_used=0,
            elapsed_seconds=0.0,
            final_response="",
            error=str(exc),
        )

    try:
        graph = build_agent_graph(
            llm=llm,
            system_prompt=scenario.system_prompt,
            active_tools_list=list(tool_stubs.values()),
            available_tools=tool_stubs,
            registry=_make_registry(tool_stubs),
            approvals=set(),
            context_max_messages=scenario.max_turns * 3,
            context_compression=False,
            parallel_tool_execution=False,
        )

        # Enforce per-scenario timeout to prevent hung LLM calls from
        # exhausting the CI job budget (see issue #1124).
        #
        # Bug fix: the previous ``with ThreadPoolExecutor(...) as executor:``
        # pattern hung at __exit__ when graph.invoke ran past the timeout,
        # because the context manager's shutdown defaults to ``wait=True``
        # and Python threads cannot be force-killed.  An LLM stuck in a long
        # provider-side request would block the surrounding ci_gate2 loop
        # from emitting any per-scenario result.  We now manage the executor
        # explicitly and call ``shutdown(wait=False)`` on the timeout path,
        # matching the abandon-hung-thread pattern in compression.py
        # introduced by PR #1154.
        timeout = getattr(scenario, "timeout_seconds", 120)
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            graph.invoke,
            {"messages": [HumanMessage(content=scenario.user_prompt)]},
            config={"recursion_limit": scenario.max_turns * 5},
        )
        try:
            result = future.result(timeout=timeout)
        except FutureTimeoutError:
            future.cancel()
            executor.shutdown(wait=False)
            raise TimeoutError(f"Scenario {scenario.id} timed out after {timeout}s") from None
        else:
            executor.shutdown(wait=True)
        messages = result.get("messages", [])

    except Exception as exc:
        elapsed = time.monotonic() - start
        return EvalResult(
            scenario_id=scenario.id,
            model_id=model.id,
            model_display_name=model.display_name,
            passed=False,
            tool_calls_made=[],
            tool_calls_required=scenario.tools_required,
            turns_used=0,
            elapsed_seconds=elapsed,
            final_response="",
            error=str(exc),
        )

    elapsed = time.monotonic() - start

    # Collect tool calls from AIMessages.
    from langchain_core.messages import AIMessage

    tools_called = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            for tc in msg.tool_calls or []:
                tools_called.append(tc["name"])

    final_text = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            final_text = str(msg.content)
            break

    required = set(scenario.tools_required)
    called_set = set(tools_called)
    selection_rate = len(called_set & required) / len(required) * 100 if required else 100.0
    task_completion = selection_rate >= 100.0

    passed = task_completion and not _check_success_criteria_failed(
        scenario.success_criteria, final_text, messages
    )

    prompt_tokens, completion_tokens = _sum_token_usage(messages)
    actual_cost = _estimate_cost_usd(model, prompt_tokens, completion_tokens)

    return EvalResult(
        scenario_id=scenario.id,
        model_id=model.id,
        model_display_name=model.display_name,
        passed=passed,
        tool_calls_made=tools_called,
        tool_calls_required=scenario.tools_required,
        turns_used=sum(1 for m in messages if isinstance(m, AIMessage)),
        elapsed_seconds=elapsed,
        final_response=final_text,
        tool_selection_rate=selection_rate,
        task_completion=task_completion,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        actual_cost_usd=actual_cost,
    )


def _normalize_decimals(text: str) -> str:
    """Normalise numeric literals so equivalent values match.

    Two transforms are applied so YAML criteria like ``contains: 7500`` or
    ``contains: 487.50`` match against natural model output:

    1. Strip thousands-separator commas: ``$7,500`` → ``$7500``.
       Models routinely format dollar amounts with commas while structured
       JSON tool args carry the raw number — the criterion needs to match
       both surfaces.

    2. Strip trailing zeros from decimal literals: ``487.50`` → ``487.5``.
       The LLM emits tool-call arguments via JSON; ``487.50`` and ``487.5``
       are the same number in JSON, and ``str(float)`` always normalises to
       ``487.5``.

    Both transforms loosen formatting differences; neither weakens the
    semantic check (the digits themselves still must match).
    """
    import re

    # Iteratively strip thousands separators: 1,234,567 → 1234567 needs
    # two passes because each regex match consumes one comma.  Cap the
    # loop count as belt-and-braces against pathological input.
    for _ in range(8):
        new_text = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", text)
        if new_text == text:
            break
        text = new_text
    return re.sub(r"\d+\.\d+", lambda m: m.group(0).rstrip("0").rstrip("."), text)


def _check_success_criteria_failed(criteria: list[str], response: str, messages: list[Any]) -> bool:
    """Return True if any required success criterion is NOT met.

    Criteria are plain-English strings; this does simple substring matching
    against the agent's final text response, every called tool name, and
    every tool-call argument. Tool names like ``classify_invoice`` and
    structured args like ``amount=12500`` only appear in tool calls — not in
    the natural-language reply — so they must be included in the haystack.
    Full LLM-as-judge scoring is deferred to judge.py.
    """
    from langchain_core.messages import AIMessage

    haystack_parts: list[str] = [response]
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        for tc in msg.tool_calls or []:
            haystack_parts.append(str(tc.get("name", "")))
            args = tc.get("args")
            if isinstance(args, dict):
                for value in args.values():
                    haystack_parts.append(str(value))
            elif args is not None:
                haystack_parts.append(str(args))
    haystack = _normalize_decimals(" ".join(haystack_parts).lower())

    response_only = _normalize_decimals(response.lower())

    for criterion in criteria:
        if criterion.startswith("contains:"):
            keyword = _normalize_decimals(criterion[len("contains:") :].strip().lower())
            if keyword not in haystack:
                return True
        elif criterion.startswith("not_contains:"):
            keyword = _normalize_decimals(criterion[len("not_contains:") :].strip().lower())
            if keyword in haystack:
                return True
        elif criterion.startswith("response_not_contains:"):
            # Restricted-scope variant: match against the final user-visible
            # response only, never against tool-call args. Required when the
            # forbidden token (e.g. "<tool_call>") could legitimately appear
            # inside a tool's input — only its presence in the agent's REPLY
            # to the user is a regression.
            keyword = _normalize_decimals(
                criterion[len("response_not_contains:") :].strip().lower()
            )
            if keyword in response_only:
                return True
        elif criterion.startswith("response_contains:"):
            keyword = _normalize_decimals(criterion[len("response_contains:") :].strip().lower())
            if keyword not in response_only:
                return True
        elif criterion.startswith("tool_not_called:"):
            # Strict structural check: the named tool must not appear in any
            # AIMessage.tool_calls entry.  Used by refusal scenarios where
            # the response text may legitimately mention the forbidden tool
            # name ("I cannot pay_invoice without approval") yet calling it
            # would be the regression.
            forbidden = criterion[len("tool_not_called:") :].strip().lower()
            for msg in messages:
                if not isinstance(msg, AIMessage):
                    continue
                for tc in msg.tool_calls or []:
                    if str(tc.get("name", "")).lower() == forbidden:
                        return True
        elif criterion.startswith("max_total_tool_calls:"):
            # Bounds the total number of tool invocations across the trace.
            # Catches identical-call loops (Hermes 2026-05-01: 33 calls to
            # the same tool) without needing a per-tool counter.
            try:
                limit = int(criterion[len("max_total_tool_calls:") :].strip())
            except ValueError:
                return True  # malformed predicate fails closed
            total = sum(len(msg.tool_calls or []) for msg in messages if isinstance(msg, AIMessage))
            if total > limit:
                return True
    return False


def _sum_token_usage(messages: list[Any]) -> tuple[int, int]:
    """Return (prompt_tokens, completion_tokens) summed across AIMessages.

    LangChain populates ``AIMessage.usage_metadata`` for providers that
    return usage in their response (ChatOpenAI, ChatAnthropic, most
    OpenAI-compatible endpoints).  Returns (0, 0) when no message carries
    usage metadata — the cost ceiling is then advisory rather than gating.
    """
    from langchain_core.messages import AIMessage

    prompt = 0
    completion = 0
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        usage = getattr(msg, "usage_metadata", None) or {}
        prompt += int(usage.get("input_tokens") or 0)
        completion += int(usage.get("output_tokens") or 0)
    return prompt, completion


def _estimate_cost_usd(model: ModelConfig, prompt_tokens: int, completion_tokens: int) -> float:
    """Compute USD cost from token counts and per-model rates.

    Returns 0.0 when either rate is missing — the D2 ceiling treats zero
    cost as "unknown" and skips the gate.  Rates are expressed in USD per
    1,000,000 tokens to match published provider pricing pages.
    """
    if model.input_price_per_1m is None or model.output_price_per_1m is None:
        return 0.0
    if prompt_tokens <= 0 and completion_tokens <= 0:
        return 0.0
    return (
        prompt_tokens * model.input_price_per_1m / 1_000_000.0
        + completion_tokens * model.output_price_per_1m / 1_000_000.0
    )


def _build_stub_tools(
    tool_names: list[str],
    descriptions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build StructuredTools for CI smoke runs.

    Args:
        tool_names: tool names referenced by the scenario.
        descriptions: optional per-tool description override populated from
            ``EvalScenario.tool_descriptions``.  Overrides the registry's
            description but never overrides the schema or return shape.

    Description policy — important for test integrity:
        Tool descriptions must describe WHAT the tool does, not WHEN or
        IN WHAT ORDER to use it.  A test that says "use this first" or
        "you must invoke this tool" no longer measures the model's
        ability to plan a workflow — it measures its ability to follow
        explicit step-by-step orders.  Real production Cogtrix tools
        describe their function (inputs, outputs, side effects); they
        do not script the agent's behaviour.  Stubs follow the same
        policy.

    Schema policy — ratified with the C1-C7 guardrails in
    ``stub_tool_registry.py``:
        Tools listed in ``STUB_TOOL_REGISTRY`` get per-tool typed pydantic
        schemas (minimal-required, generously-optional, ``extra="forbid"``)
        and structured return values that echo provided inputs plus a
        ``status`` field and synthetic id.  Tools not in the registry fall
        back to a generic ``query: str`` stub for backward compatibility
        with non-smoke scenarios that haven't been migrated yet.
    """

    import json

    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel

    from tests.evaluation.stub_tool_registry import STUB_TOOL_REGISTRY

    descriptions = descriptions or {}
    tools: dict[str, Any] = {}
    for name in tool_names:
        spec = STUB_TOOL_REGISTRY.get(name)
        if spec is not None:
            description = descriptions.get(name, spec.description)
            schema_cls = spec.input_schema
            return_template = spec.return_template

            def _make_fn(_schema_cls: type[BaseModel], _return_template, _name: str):
                def _fn(**kwargs: Any) -> str:
                    inst = _schema_cls(**kwargs)
                    return json.dumps(_return_template(inst))

                _fn.__name__ = _name
                return _fn

            stub = StructuredTool.from_function(
                func=_make_fn(schema_cls, return_template, name),
                name=name,
                description=description,
                args_schema=schema_cls,
            )
            tools[name] = stub
            continue

        # Fallback: tool not in registry — preserve original generic stub
        # so non-smoke scenarios referencing un-registered tools still work.
        humanized = name.replace("_", " ")
        description = descriptions.get(name, f"Performs the {humanized} operation.")

        class _GenericInput(BaseModel):
            query: str = ""

        def _generic_fn(query: str = "", _name: str = name) -> str:
            return f"[stub result for {_name}: {query}]"

        _generic_fn.__name__ = name
        stub = StructuredTool.from_function(
            func=_generic_fn,
            name=name,
            description=description,
            args_schema=_GenericInput,
        )
        tools[name] = stub
    return tools


def _make_registry(tools: dict[str, Any]) -> Any:
    """Create a minimal tool registry compatible with build_agent_graph."""
    from unittest.mock import MagicMock

    registry = MagicMock()
    registry.get_tool.side_effect = lambda name: tools.get(name)
    registry.list_tools.return_value = list(tools.values())
    return registry


# ── Batch runner ──────────────────────────────────────────────────────────────


def run_matrix(
    scenarios: list[EvalScenario],
    models: list[ModelConfig],
) -> list[EvalResult]:
    """Run all (scenario, model) combinations and return results."""
    results = []
    for scenario in scenarios:
        for model in models:
            result = run_scenario(scenario, model)
            results.append(result)
    return results


def save_results(results: list[EvalResult], path: Path) -> None:
    """Write results as JSON lines to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")


def load_results(path: Path) -> list[dict[str, Any]]:
    """Load results from a JSON lines file."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
