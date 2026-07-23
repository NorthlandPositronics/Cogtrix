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
import random
import re
import secrets
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
    "FIREWORKS_API_KEY",
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
    "FIREWORKS_API_KEY": {"FIREWORKS_API_KEY"},
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


# Provider error patterns that identify an invalid key or exhausted quota.
#
# Matching is phrase-anchored rather than substring-loose. The pre-fix
# implementation matched bare words like ``payment`` / ``unauthorized`` /
# ``credits`` / ``account``, which collided with scenario names — e.g.
# ``Scenario safety_refuse_unauthorized_payment timed out after 90s``
# tripped on both ``unauthorized`` and ``payment``, causing a transient
# timeout to be mis-routed to the KEY_FAIL branch and the
# ``_is_transient_error`` retry path to be skipped entirely (see #1885).
#
# Each pattern below either pins an HTTP status (``401``/``402``/``403``)
# with a word boundary, or names a phrase that providers actually emit
# (``payment required``, ``insufficient credits``, ``billing issue``,
# ``account suspended``, ``you exceeded your current quota``). Adding
# a new pattern requires the same discipline — anchor it tightly enough
# that a scenario named ``test_<word>`` cannot accidentally match.
_AUTH_OR_QUOTA_PATTERNS = re.compile(
    r"""
    \b401\b
    | \b402\b                            # Payment Required
    | \b403\b
    | \binvalid\s+api\s+key\b
    | \bincorrect\s+api\s+key\b
    | \bauthentication\s+(?:failed|error|required|denied)\b
    | \bunauthorized:\s*                 # colon makes 'unauthorized' specific to error format
    | \bunauthorized\.                   # or sentence-terminating period
    | \bunauthorized\s+access\b
    | \bunauthorized\s+request\b
    | \binsufficient_quota\b
    | \binsufficient\s+(?:quota|credits|balance|funds)\b
    | \byou\s+(?:have\s+)?exceeded\s+your\s+(?:current\s+)?quota\b
    | \bquota\s+(?:has\s+been\s+|is\s+)?(?:exceeded|exhausted)\b
    | \bbilling\s+(?:issue|problem|error|disabled)\b
    | \bpayment\s+(?:required|method\s+(?:missing|invalid|expired))\b
    | \bno\s+payment\s+method\b
    | \bno\s+money\b
    | \baccount\s+(?:suspended|disabled|locked|inactive|terminated)\b
    | \bapi\s+key\s+(?:invalid|expired|revoked|missing|not\s+found)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _is_auth_or_quota_error(exc: Exception) -> bool:
    """Return True when the exception signals an invalid key or quota exhaustion.

    Phrase-anchored (#1885) — see the ``_AUTH_OR_QUOTA_PATTERNS`` comment
    block for the rationale. Loose substring matches on common words
    like ``payment`` / ``unauthorized`` / ``credits`` collide with
    scenario names; this matcher requires provider-error phrasing.
    """
    return bool(_AUTH_OR_QUOTA_PATTERNS.search(str(exc)))


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
    # When True, ``_candidate_keys_for_model`` does NOT promote this
    # model's native env_key ahead of OpenRouter — OpenRouter (or
    # whichever key sits first in _KEY_PRIORITY) is tried first
    # instead.  Used as a per-model escape hatch when the native
    # provider has a known integration bug that OpenRouter happens to
    # tolerate or work around.  Track each opt-out with a linked issue
    # in the models.yaml comment so it can be flipped back off once
    # the underlying bug is fixed.
    prefer_openrouter: bool = False
    # Declarative inverse of ``prefer_openrouter`` (#2359).  When True and the
    # model's own ``env_key`` is present in the environment, ``_build_llm``
    # drops any higher-priority routing key (OpenRouter/Cerebras) and routes
    # through the model's NATIVE provider (its ``base_url`` + ``env_key``).
    # Used for native Moonshot Kimi so it runs against api.moonshot.* with the
    # operator's Moonshot key rather than being hijacked by OpenRouter (which
    # resolve_active_key returns first).  No effect if the native key is unset.
    prefer_native: bool = False
    # Per-model retry backoff in seconds.  When set to a positive value,
    # ``_try_run_with_key`` sleeps this long *before* a transient /
    # empty-response retry attempt against this model.  Default 0 keeps
    # the historical immediate-retry behaviour for every other model.
    #
    # Issue #1994: kimi-k2-5 routed through OpenRouter periodically
    # produces empty-response flakes (turns=0, no content, no error)
    # when Moonshot's upstream capacity window closes.  An immediate
    # retry hits the same closed window and burns the retry budget
    # without ever giving the capacity a chance to recover.  A short
    # backoff (60s, well under any per-scenario timeout_seconds) lets
    # the window roll forward before the second attempt, turning what
    # used to be a guaranteed flake-fail into a likely recovery.
    #
    # Each model that opts in MUST link the tracking issue in the
    # models.yaml comment so it can be flipped back to 0 once the
    # underlying capacity flakiness clears.
    retry_backoff_seconds: int = 0
    # Per-model sampling temperature.  Default ``None`` keeps the historical
    # behaviour (the runner does not pin temperature; the underlying chat
    # model uses its own default).  Set this when a model REQUIRES a specific
    # temperature: native Moonshot Kimi K2.7 only accepts ``temperature: 1``
    # and returns an API error for any other value (#2359).  When the caller
    # passes an explicit temperature to ``_build_llm`` that still wins; this
    # is the fallback used when the caller (e.g. the role-test harness) does
    # not pin one.  Applies on every route (native / OpenRouter / Cerebras),
    # since the constraint is a property of the model, not the endpoint.
    temperature: float | None = None
    # #2484: the model's real INPUT context window (tokens).  Declared so
    # ``_build_llm`` can stamp it onto the built llm (``_cogtrix_context_window``),
    # letting the pre-flight compression guard + ``build_agent_graph``'s default-cap
    # scaler size to the true window instead of the 32768 ModelConfig default.
    # Without it, big-window models (Kimi 262k, DeepSeek 128k) are force-compressed
    # to a flat 40k on EVERY turn — the compression storm that made RAG-heavy
    # role-test scenarios minutes-per-turn slow and unrepresentative of production
    # (which resolves the cap via ``Config.resolve_context_max_tokens``).
    context_window: int | None = None


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
class Turn:
    """A single user turn in a multi-turn evaluation scenario.

    Per-turn ``success_criteria`` are evaluated against the slice of
    messages produced by that turn only, so a ``tool_called: foo``
    assertion in turn 1 does not match a tool call from turn 2.

    A scenario passes iff every turn's criteria pass AND every required
    tool was called at least once across the session.

    The LLM-as-judge scores each turn independently and aggregates
    scores by a weighted average — ``judge_weight`` controls each turn's
    contribution to the final aggregate.  Authoring guidance:

    * For **related** turn sequences (turn N builds on turn N-1's
      outcome), set a higher weight on the final turn so the score
      reflects whether the overall workflow succeeded.
    * For **unrelated** turns (one scenario testing several independent
      capabilities back-to-back), keep weights at 1.0 across the board.
    * The default 1.0 means "treat this turn equally with its
      neighbours" — safe when in doubt.
    """

    user_prompt: str
    success_criteria: list[str] = field(default_factory=list)
    judge_weight: float = 1.0


@dataclass
class EvalScenario:
    """A single Finance/Procurement evaluation scenario.

    Two YAML shapes are accepted:

    * Legacy single-turn: top-level ``user_prompt`` + ``success_criteria``.
    * Multi-turn: top-level ``turns:`` list of ``user_prompt`` + per-turn
      ``success_criteria``.  Mutually exclusive with the legacy fields.

    After ``load_scenario`` parses either shape, ``scenario.turns`` is
    always non-empty and the runner only reads ``scenario.turns``.
    """

    id: str
    domain: str  # "procurement" | "finance"
    title: str
    description: str
    # Legacy single-turn fields — either these, or `turns:`, must be
    # populated.  See class docstring.
    user_prompt: str = ""
    system_prompt: str = ""
    tools_required: list[str] = field(default_factory=list)
    expected_outcome: str = ""
    success_criteria: list[str] = field(default_factory=list)
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
    # Populated by ``load_scenario`` from either ``turns:`` (multi-turn)
    # or from the legacy ``user_prompt`` / ``success_criteria`` fields
    # (single-turn).  The runner only reads this; the legacy fields are
    # kept on the dataclass solely so ``EvalScenario(**yaml_dict)`` keeps
    # working for the existing single-turn shape.
    turns: list[Turn] = field(default_factory=list)


def load_scenario(path: Path) -> EvalScenario:
    """Load a scenario YAML file into an EvalScenario.

    Normalises both legacy and multi-turn YAML shapes so the runner only
    has to traverse ``scenario.turns``.  Rejects ambiguous YAMLs that
    mix top-level ``user_prompt`` / ``success_criteria`` with a
    ``turns:`` block.
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    raw_turns = data.pop("turns", None)
    parsed_turns: list[Turn] = []
    if raw_turns is not None:
        scenario_label = data.get("id", path.name)
        if not isinstance(raw_turns, list) or not raw_turns:
            raise ValueError(f"Scenario {scenario_label}: `turns:` must be a non-empty list.")
        for idx, raw_turn in enumerate(raw_turns):
            if not isinstance(raw_turn, dict):
                raise ValueError(f"Scenario {scenario_label} turn[{idx}]: must be a mapping.")
            if "user_prompt" not in raw_turn:
                raise ValueError(f"Scenario {scenario_label} turn[{idx}]: missing `user_prompt`.")
            raw_weight = raw_turn.get("judge_weight", 1.0)
            try:
                judge_weight = float(raw_weight)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Scenario {scenario_label} turn[{idx}]: `judge_weight` must be a number."
                ) from exc
            if judge_weight < 0:
                raise ValueError(
                    f"Scenario {scenario_label} turn[{idx}]: `judge_weight` must be non-negative."
                )
            parsed_turns.append(
                Turn(
                    user_prompt=raw_turn["user_prompt"],
                    success_criteria=list(raw_turn.get("success_criteria") or []),
                    judge_weight=judge_weight,
                )
            )

    scenario = EvalScenario(**data, turns=parsed_turns)

    # Backward-compat shim: fold legacy fields into a 1-element turns
    # list so the runner only has one code path.  Reject ambiguous YAMLs
    # that supply both shapes — silent precedence rules in this layer
    # would make scenario debugging painful.
    if not scenario.turns:
        if not scenario.user_prompt:
            raise ValueError(
                f"Scenario {scenario.id}: must provide either `turns:` or `user_prompt`."
            )
        scenario.turns = [
            Turn(
                user_prompt=scenario.user_prompt,
                success_criteria=list(scenario.success_criteria),
            )
        ]
    elif scenario.user_prompt or scenario.success_criteria:
        raise ValueError(
            f"Scenario {scenario.id}: `turns:` is mutually exclusive with "
            "top-level `user_prompt` / `success_criteria`."
        )

    return scenario


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


# ── Canary substitution (Option A) ────────────────────────────────────────────


_CANARY_PREFIXES: tuple[str, ...] = (
    "Astro",
    "Nova",
    "Orbit",
    "Pulse",
    "Vertex",
    "Prism",
    "Flux",
    "Helix",
    "Nexus",
    "Cipher",
    "Drift",
    "Echo",
    "Lumen",
    "Mirage",
    "Quark",
    "Sol",
    "Terra",
    "Vortex",
    "Zen",
    "Arc",
    "Bolt",
    "Crest",
    "Dusk",
    "Ember",
    "Frost",
    "Gale",
    "Haven",
    "Iris",
    "Jade",
    "Keystone",
)

_CANARY_SUFFIXES: tuple[str, ...] = (
    "Flow",
    "Core",
    "Grid",
    "Link",
    "Mesh",
    "Node",
    "Path",
    "Ring",
    "Spark",
    "Stream",
    "Sync",
    "Wave",
    "Base",
    "Box",
    "Bridge",
    "Cast",
    "Drive",
    "Forge",
    "Hub",
    "Kit",
    "Lab",
    "Net",
    "Port",
    "Scope",
    "Shift",
    "Stack",
    "Studio",
    "Vault",
    "View",
    "Wire",
    "Works",
    "Zone",
)

_CANARY_CATEGORIES: tuple[str, ...] = (
    "Framework",
    "Platform",
    "Toolkit",
    "Engine",
    "Runtime",
    "Library",
    "SDK",
    "IDE",
    "Suite",
    "System",
    "Service",
    "Agent",
    "Daemon",
    "Module",
    "Component",
    "Interface",
    "Gateway",
    "Router",
    "Proxy",
)


def _generate_canary_name() -> str:
    """Generate a fresh fictional product name unlikely to exist in any corpus.

    Uses a phonotactically-plausible compound name with a random hex suffix
    to ensure near-zero collision probability with real-world products or
    previously-generated canaries.  The result looks like a real tech product
    (e.g. ``NovaForge-a3b2-Toolkit``) so the agent treats it naturally in
    prompts, but the random suffix guarantees it has never appeared in training
    data.
    """
    prefix = random.choice(_CANARY_PREFIXES)
    suffix = random.choice(_CANARY_SUFFIXES)
    token = secrets.token_hex(2)  # 4 hex chars — 65k uniqueness space
    category = random.choice(_CANARY_CATEGORIES)
    return f"{prefix}{suffix}-{token}-{category}"


def _substitute_canary(scenario: EvalScenario) -> EvalScenario:
    """Replace canary placeholders with freshly-generated names.

    Supported placeholders:

    * ``{canary_name}`` / ``{canary_name_N}`` — the generated product
      name (original case).  A distinct name is generated for each
      unique placeholder suffix so multi-turn scenarios can use
      unrelated fictional products.
    * ``{canary_name_lower}`` / ``{canary_name_N_lower}`` — lowercase,
      hyphen-stripped version for URL assertions
      (e.g. ``github.com/{canary_name_lower}``).

    When no placeholder is present the scenario is returned unchanged.
    """
    import re

    scenario_text = str(scenario.__dict__)
    placeholders = set(re.findall(r"\{canary_name(?:_\d+)?(?:_lower)?\}", scenario_text))
    if not placeholders:
        return scenario

    # Generate one unique name per distinct placeholder suffix.
    # {canary_name} and {canary_name_lower} share the same root name.
    # {canary_name_2} and {canary_name_2_lower} share a second name, etc.
    canary_map: dict[str, str] = {}
    seen_roots: set[str] = set()
    for ph in placeholders:
        # Extract root: "canary_name" or "canary_name_2"
        root_match = re.match(r"\{(canary_name(?:_\d+)?)", ph)
        if root_match is None:
            continue
        root = root_match.group(1)
        if root not in seen_roots:
            seen_roots.add(root)
            canary_map[root] = _generate_canary_name()

    def _replace(text: str) -> str:
        for ph in placeholders:
            root_match = re.match(r"\{(canary_name(?:_\d+)?)", ph)
            if root_match is None:
                continue
            root = root_match.group(1)
            canary = canary_map[root]
            if ph.endswith("_lower}"):
                value = canary.lower().replace("-", "")
            else:
                value = canary
            text = text.replace(ph, value)
        return text

    new_turns = [
        Turn(
            user_prompt=_replace(turn.user_prompt),
            success_criteria=[_replace(c) for c in turn.success_criteria],
            judge_weight=turn.judge_weight,
        )
        for turn in scenario.turns
    ]

    new_scenario = EvalScenario(
        id=scenario.id,
        domain=scenario.domain,
        title=_replace(scenario.title),
        description=_replace(scenario.description),
        system_prompt=_replace(scenario.system_prompt),
        tools_required=list(scenario.tools_required),
        expected_outcome=_replace(scenario.expected_outcome),
        success_criteria=[_replace(c) for c in scenario.success_criteria],
        max_turns=scenario.max_turns,
        timeout_seconds=scenario.timeout_seconds,
        tags=list(scenario.tags),
        budget_usd_estimate=scenario.budget_usd_estimate,
        tool_descriptions={k: _replace(v) for k, v in scenario.tool_descriptions.items()},
        tools_available=list(scenario.tools_available),
        turns=new_turns,
    )

    # Preserve legacy single-turn fields when they were populated
    if scenario.user_prompt:
        new_scenario.user_prompt = _replace(scenario.user_prompt)

    return new_scenario


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class TurnResult:
    """Per-turn outputs captured by the runner for a multi-turn scenario.

    Used by ``judge.py`` to score each turn independently.  Empty for
    legacy programmatic ``EvalResult`` construction (test fixtures that
    don't go through ``run_scenario``); the judge falls back to its
    single-call code path when ``turn_results`` is empty or has length 1.
    """

    final_response: str = ""
    tool_calls_made: list[str] = field(default_factory=list)


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

    # #2212: True when a turn hit the LangGraph recursion cap and was finalized
    # via the production-equivalent step-limit recovery (re-invoke once with a
    # "answer now, no more tools" nudge) instead of crashing — mirroring
    # run_agent / recover_from_step_limit and the role_sysadmin (#2368) / role_swe
    # harnesses. REPORTED only; it never gates ``passed`` (the recovered turn is
    # still scored on its content), so a looped-but-recovered pass stays
    # distinguishable from a clean first-pass and honest failures still fail.
    recovered_from_step_limit: bool = False

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

    # Tool-call errors observed during the run.  Each entry is one short
    # "<tool_name>: <truncated error text>" line.  Bug L follow-up
    # (2026-05-20) folded these into the pass gate so a graceful "could
    # not find" answer alongside a pydantic ValidationError on http_get
    # is no longer silently scored as a pass.
    #
    # This list captures EVERY detected error for diagnostic visibility
    # (used by dashboard + the TOOL_ERRORS log line in ci_gate2).
    # Issue #1787: the pass-gate decision uses ``tool_errors_unrecovered``
    # below instead — recovered errors (same tool retried successfully
    # later in the run) no longer force ``passed=False``.
    tool_errors: list[str] = field(default_factory=list)

    # Subset of ``tool_errors`` containing only failures the model did
    # NOT recover from on a later retry of the same tool.  Computed by
    # ``_collect_tool_errors_with_recovery``.  This is the list the pass
    # gate consults; a populated ``tool_errors`` with an empty
    # ``tool_errors_unrecovered`` means the model self-corrected and the
    # scenario should be eligible to pass on the other gates (judge
    # score, task completion, budget).
    tool_errors_unrecovered: list[str] = field(default_factory=list)

    # Per-turn outputs.  Populated by ``run_scenario`` (one entry per
    # turn in ``scenario.turns``).  Empty when the result is constructed
    # programmatically without going through the runner — the judge
    # detects that and falls back to single-call mode.
    turn_results: list[TurnResult] = field(default_factory=list)

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
            "recovered_from_step_limit": self.recovered_from_step_limit,
            "tool_selection_rate": self.tool_selection_rate,
            "task_completion": self.task_completion,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "actual_cost_usd": round(self.actual_cost_usd, 6),
            "tool_errors": list(self.tool_errors),
            "tool_errors_unrecovered": list(self.tool_errors_unrecovered),
            "turn_results": [
                {
                    "final_response": tr.final_response[:500],
                    "tool_calls_made": tr.tool_calls_made,
                }
                for tr in self.turn_results
            ],
        }


# ── LLM instantiation ─────────────────────────────────────────────────────────


def _build_raw_llm(
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
    # An explicit caller temperature wins; otherwise fall back to the model's
    # own pinned temperature (e.g. Moonshot Kimi K2.7 requires temperature=1,
    # #2359).  Applied here — before the route branches below — so it reaches
    # the OpenRouter / Cerebras / native ChatOpenAI alike via **kwargs.
    _temperature = temperature if temperature is not None else model.temperature
    if _temperature is not None:
        kwargs["temperature"] = _temperature

    # ── Force the model's NATIVE provider when it asks for it (#2359) ──────────
    # A higher-priority routing key (e.g. OpenRouter, which resolve_active_key
    # returns first) would otherwise hijack a model that the operator wants run
    # natively.  ``prefer_native`` is the declarative inverse of
    # ``prefer_openrouter``: when set AND the model's own env_key is present, we
    # drop the routing key so the native-provider branch below uses the model's
    # base_url + key (native Moonshot Kimi runs against api.moonshot.* with the
    # COGTRIX_PROVIDER_KIMI_API_KEY, not via OpenRouter).
    if model.prefer_native and os.environ.get(model.env_key, ""):
        active_key = None

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
        # Use the factory so _DeepSeekChatModel (reasoning_content preservation)
        # is selected automatically for api.deepseek.com. Direct ChatOpenAI
        # instantiation bypasses the subclass and causes HTTP 400 on turn ≥ 2
        # for thinking-mode models (deepseek-reasoner / deepseek-v4-flash).
        from cogtrix_core.providers import create_chat_model

        return create_chat_model(
            "openai",
            model=model.model_id,
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            **kwargs,
        )

    raise ValueError(f"Unknown provider '{model.provider}' for model '{model.id}'.")


def _build_llm(
    model: ModelConfig,
    temperature: float | None = None,
    active_key: tuple[str, str] | None = None,
) -> Any:
    """Build the model's LLM (see :func:`_build_raw_llm`) and stamp its declared
    input window on the result (#2484).

    Production's ``create_chat_model`` stamps ``_cogtrix_context_window`` on its
    wrapper so the pre-flight compression guard sizes to the real window; the
    eval's raw ``ChatOpenAI`` routes don't, so a big-window model got capped at
    the flat 40k default and force-compressed every turn.  Stamping the declared
    ``model.context_window`` here (a harmless no-op when undeclared) makes the
    eval measure production's resolved behaviour.
    """
    llm = _build_raw_llm(model, temperature=temperature, active_key=active_key)
    win = model.context_window
    if isinstance(win, int) and not isinstance(win, bool) and win > 0:
        try:
            llm._cogtrix_context_window = win  # type: ignore[attr-defined]
        except (AttributeError, TypeError):  # pragma: no cover — frozen/exotic objects
            pass
    return llm


# ── Runner ────────────────────────────────────────────────────────────────────

#: Tight budget for the step-limit recovery re-invoke — mirrors
#: ``recover_from_step_limit``'s ``retry_config["recursion_limit"] = 4`` in
#: cogtrix_core/orchestration/phases.py (at most 1 tool call + a final answer). Same
#: constant as the role_sysadmin (#2368) / role_swe harnesses.
_STEP_LIMIT_RECOVERY_LIMIT = 4


def _invoke_with_step_limit_recovery(
    graph: Any, invoke_state: dict, config: dict
) -> tuple[dict, bool]:
    """Drive one turn to completion, recovering at the recursion cap like production.

    The Gate-2 harness used to call ``graph.invoke`` raw, so a
    ``GraphRecursionError`` propagated and the turn was scored as a crash
    (``tools=0 turns=0 error=Recursion limit…`` — #2212). But production's
    ``run_agent`` does NOT crash there: ``recover_from_step_limit`` re-invokes
    once with a tight "answer now, no more tools" nudge and finalizes a
    best-effort turn. The raw-invoke harness OVER-reported failures vs the live
    product — a weak model that gathered what it needed and then looped read as a
    hard crash on exactly the runs we most want to score.

    Mirror production (same fix as tests/role_sysadmin #2368 and tests/role_swe):
    stream so we keep the latest full state, and on ``GraphRecursionError``
    re-invoke once with the finalize nudge under ``_STEP_LIMIT_RECOVERY_LIMIT``
    instead of propagating the crash. Returns ``(final_state, recovered)`` where
    ``recovered`` is reported but never gates the pass — the recovered turn is
    still scored on its content, so honest failures still fail.
    """
    from langchain_core.messages import HumanMessage
    from langgraph.errors import GraphRecursionError

    last: dict = dict(invoke_state)

    def _drain(seed_messages: list, limit: int) -> None:
        nonlocal last
        cfg = dict(config)
        cfg["recursion_limit"] = limit
        for state in graph.stream({"messages": seed_messages}, config=cfg, stream_mode="values"):
            if isinstance(state, dict) and "messages" in state:
                last = state

    # The harness always sets recursion_limit (scenario.max_turns * 5); the
    # fallback only guards a programmatic caller that omitted it.
    base_limit = config.get("recursion_limit", 60)
    recovered = False
    try:
        _drain(list(invoke_state.get("messages", [])), base_limit)
    except GraphRecursionError:
        recovered = True
        nudge = HumanMessage(
            content=(
                "Please provide your final response now. Summarize what you have "
                "found so far. Do NOT call any more tools — just answer with the "
                "information you already have."
            )
        )
        try:
            _drain(list(last.get("messages", [])) + [nudge], _STEP_LIMIT_RECOVERY_LIMIT)
        except GraphRecursionError:
            # Recovery itself capped — keep the trail we have; still no crash.
            pass
    return last, recovered


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
    from langchain_core.messages import AIMessage, HumanMessage

    from cogtrix_core.orchestration.graph import build_agent_graph

    start = time.monotonic()

    # Option A: generate a fresh canary name at runtime and substitute
    # placeholders so the scenario is resilient to corpus poisoning.
    scenario = _substitute_canary(scenario)

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

    # Backward-compat shim for scenarios constructed programmatically
    # (i.e. not through ``load_scenario``).  Legacy callers fill
    # ``user_prompt`` + ``success_criteria`` directly and leave
    # ``turns`` empty; fold those into a single Turn so the loop below
    # has something to iterate.
    effective_turns: list[Turn] = (
        scenario.turns
        if scenario.turns
        else [
            Turn(
                user_prompt=scenario.user_prompt,
                success_criteria=list(scenario.success_criteria),
            )
        ]
    )

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

        # Enforce per-turn timeout to prevent hung LLM calls from
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
        #
        # Multi-turn loop: ``state["messages"]`` accumulates across turns
        # so the graph sees the full conversation history.  After each
        # turn we slice ``messages[msg_offset:]`` for per-turn
        # assertions, so a ``tool_called: foo`` predicate in turn 2 does
        # not match a tool call from turn 1.
        timeout = getattr(scenario, "timeout_seconds", 120)
        state: dict[str, Any] = {"messages": []}
        per_turn_failed: list[bool] = []
        per_turn_results: list[TurnResult] = []
        recovered_from_step_limit = False  # #2212: reported, never gates the pass

        for turn_idx, turn in enumerate(effective_turns):
            msg_offset = len(state["messages"])
            invoke_state = dict(state)
            invoke_state["messages"] = list(state["messages"]) + [
                HumanMessage(content=turn.user_prompt)
            ]
            executor = ThreadPoolExecutor(max_workers=1)
            # #2212: recover at the recursion cap like production instead of
            # scoring a crash — see _invoke_with_step_limit_recovery.
            future = executor.submit(
                _invoke_with_step_limit_recovery,
                graph,
                invoke_state,
                {"recursion_limit": scenario.max_turns * 5},
            )
            try:
                turn_result, turn_recovered = future.result(timeout=timeout)
            except FutureTimeoutError:
                future.cancel()
                executor.shutdown(wait=False)
                raise TimeoutError(
                    f"Scenario {scenario.id} timed out after {timeout}s "
                    f"on turn {turn_idx + 1}/{len(effective_turns)}"
                ) from None
            else:
                executor.shutdown(wait=True)

            recovered_from_step_limit = recovered_from_step_limit or turn_recovered
            state = turn_result
            turn_messages = state["messages"][msg_offset:]
            turn_final = ""
            for msg in reversed(turn_messages):
                if isinstance(msg, AIMessage) and msg.content:
                    turn_final = str(msg.content)
                    break
            turn_tools: list[str] = []
            for msg in turn_messages:
                if isinstance(msg, AIMessage):
                    for tc in msg.tool_calls or []:
                        turn_tools.append(tc["name"])
            per_turn_failed.append(
                _check_success_criteria_failed(turn.success_criteria, turn_final, turn_messages)
            )
            per_turn_results.append(
                TurnResult(final_response=turn_final, tool_calls_made=turn_tools)
            )

        messages = state["messages"]

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

    # Bug L follow-up (2026-05-20): tool errors during the run MUST fail
    # the scenario, even when the success criteria only inspect the final
    # response text.  Previously a scenario where http_get raised a
    # pydantic ValidationError and the model fell back to an honest
    # "could not find" answer was reported as a pass because the response
    # criteria still matched.  This silently masked real tool-shape bugs.
    #
    # Issue #1787 refinement: a tool error that the model recovered from
    # on a successful retry of the same tool is no longer a hard fail.
    # ``tool_errors_observed`` still captures the full diagnostic list
    # for the log; ``tool_errors_unrecovered`` is what the pass gate
    # consults.
    tool_errors_observed, tool_errors_unrecovered = _collect_tool_errors_with_recovery(messages)

    # Scenario passes iff (a) every required tool was called somewhere in
    # the session AND (b) every per-turn success_criteria block passed
    # against its own message slice AND (c) no tool error went
    # unrecovered.
    all_turns_passed = not any(per_turn_failed)
    no_unrecovered_tool_errors = not tool_errors_unrecovered
    passed = task_completion and all_turns_passed and no_unrecovered_tool_errors

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
        tool_errors=tool_errors_observed,
        tool_errors_unrecovered=tool_errors_unrecovered,
        turn_results=per_turn_results,
        recovered_from_step_limit=recovered_from_step_limit,
    )


_TOOL_ERROR_MARKERS: tuple[str, ...] = (
    # The orchestration graph wraps any tool-raised exception with this
    # prefix (see cogtrix_core/orchestration/graph.py: _invoke_one — "Error
    # executing {tool}: {exc}"). Pydantic ValidationErrors, network
    # exceptions, and uncaught provider errors all surface this way.
    "error executing ",
    # Many tools handle errors gracefully and return a plain-text message
    # starting with "Error:" instead of raising. Both http_get and
    # http_post emit "Error: <sanitised exc>" on timeout / connection /
    # validation failures. These are real tool failures from the
    # scenario's perspective even though no exception propagated.
    "error:",
)

# Substring patterns inside ToolMessage content that signal a real tool
# error even when the leading "Error:" prefix is not the first token —
# e.g. wrapped cached results "[Duplicate call — returning cached result. Do NOT
# repeat this call.]\n\nError: ...".
_TOOL_ERROR_CONTAINS: tuple[str, ...] = (
    "\nerror executing ",
    "\nerror: ",
    "validation error",
    "is no longer active",
)


def _classify_tool_messages(messages: list[Any]) -> list[tuple[str, bool, str]]:
    """Walk ``messages`` and classify each ``ToolMessage`` as success or error.

    Returns one entry per ToolMessage in source order:
    ``(tool_name, is_error, "<tool>: <snippet>" or "")``.  The snippet is
    populated only when ``is_error`` is True (matches the legacy
    ``_collect_tool_errors`` line shape so callers can reuse it verbatim).

    Detection is conservative: a ToolMessage counts as a failure only
    when its content contains an unambiguous error marker (see
    ``_TOOL_ERROR_MARKERS`` / ``_TOOL_ERROR_CONTAINS``).  Normal tool
    results that legitimately mention the word "error" (e.g. a search
    result page title) do not match because the markers anchor on the
    start of the line.
    """
    from langchain_core.messages import ToolMessage

    classified: list[tuple[str, bool, str]] = []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        content = msg.content
        if not isinstance(content, str):
            # Multi-part content shapes (rare) — flatten conservatively.
            try:
                content = " ".join(
                    str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content
                )
            except (TypeError, AttributeError):
                content = str(content)
        tool_name = getattr(msg, "name", None) or "<unknown>"
        if not content:
            classified.append((tool_name, False, ""))
            continue
        lowered = content.lower().lstrip()
        is_error = any(lowered.startswith(marker) for marker in _TOOL_ERROR_MARKERS) or any(
            marker in lowered for marker in _TOOL_ERROR_CONTAINS
        )
        if is_error:
            snippet = content.strip().splitlines()[0][:200]
            classified.append((tool_name, True, f"{tool_name}: {snippet}"))
        else:
            classified.append((tool_name, False, ""))
    return classified


def _collect_tool_errors(messages: list[Any]) -> list[str]:
    """Return one short ``<tool>: <truncated error text>`` per failed tool call.

    Bug L follow-up: tool-call errors during a scenario run must fail the
    test, otherwise the scenario can ship as a pass when the model
    happens to produce a graceful-looking answer despite a broken tool
    invocation.

    Back-compat surface: this returns *every* detected error, including
    those that the model recovered from on a subsequent retry of the
    same tool.  The recovery-aware subset used by the pass gate is
    computed by :func:`_collect_tool_errors_with_recovery`.
    """
    return [line for (_n, is_err, line) in _classify_tool_messages(messages) if is_err]


def _tool_call_args_index(messages: list[Any]) -> list[tuple[int, str, Any]]:
    """Walk ``messages`` and return one entry per tool_call across all AIMessages.

    Returns ``[(toolmsg_position, tool_name, args_repr)]``.  The
    ``toolmsg_position`` is the index of the ToolMessage this call
    *would* produce in the classified ToolMessage stream — so callers
    can correlate AIMessage-side intent with ToolMessage-side outcome
    even when the dispatcher dropped a call (cap-blocked, dedup'd,
    pre-validated).  Used by the recovery heuristic in
    :func:`_collect_tool_errors_with_recovery` to recognise an
    attempted retry with materially-different args even when the
    ToolMessage isn't there.

    ``args_repr`` is ``repr(args)`` so two args dicts compare exactly
    iff they are structurally identical — including dict-key order.
    Materially-different args are recognised by simple inequality.
    """
    from langchain_core.messages import AIMessage, ToolMessage

    entries: list[tuple[int, str, Any]] = []
    toolmsg_position = 0
    for msg in messages:
        if isinstance(msg, AIMessage):
            for tc in getattr(msg, "tool_calls", None) or []:
                if isinstance(tc, dict):
                    name = tc.get("name", "<unknown>") or "<unknown>"
                    args = tc.get("args")
                else:
                    name = getattr(tc, "name", "<unknown>") or "<unknown>"
                    args = getattr(tc, "args", None)
                entries.append((toolmsg_position, name, repr(args)))
        elif isinstance(msg, ToolMessage):
            # Each ToolMessage advances the position counter.  Note: a
            # batched parallel call produces ONE AIMessage with N
            # tool_calls and then N ToolMessages — the entries above
            # all share the same ``toolmsg_position`` of the FIRST
            # ToolMessage in the batch.  That's fine for recovery
            # purposes since we only ask "did the same tool re-appear
            # LATER" — within-batch ordering doesn't matter.
            toolmsg_position += 1
    return entries


def _collect_tool_errors_with_recovery(
    messages: list[Any],
) -> tuple[list[str], list[str]]:
    """Return ``(all_errors, unrecovered_errors)`` for one scenario message stream.

    Issue #1787: kimi-k2-5 (and other models, less frequently) regularly
    hallucinate a tool-call argument name on first invocation, get a
    pydantic ValidationError, then retry the same tool with the correct
    schema and produce a correct final response.  Gate 2 was counting
    every such recovered error as a hard fail, conflating "model never
    made a mistake" with "model arrived at the correct outcome".

    Recovery is detected by EITHER of two signals:

    1. **Same-tool success** (the #1787 baseline): tool ``T`` succeeds
       at a later index in the classified ToolMessage stream.

    2. **Retried with materially-different args** (#1993 follow-up):
       the agent emitted a *later* AIMessage tool_call for the same
       tool with non-byte-identical args, indicating an intentional
       retry attempt — regardless of whether the dispatcher produced
       a separately-classifiable ToolMessage for that retry.  This
       catches the gpt-4o ``create_po`` failure mode where the second
       call's outcome got bundled into a parallel-batch
       ToolMessage that the position-based forward-look didn't reach.

    Recovery is *not* detected by:

    - Same tool, byte-identical args (the agent retried the SAME bad
      call — that's not recovery, it's a loop).
    - Subsequent calls to different tools (could indicate the agent
      gave up on the failed tool, which doesn't recover the original
      error).

    Errors with neither signal stay in ``unrecovered_errors`` and
    remain a hard veto for the pass gate.  All errors stay in
    ``all_errors`` for the diagnostic log.
    """
    classified = _classify_tool_messages(messages)

    # Signal #1: same-tool success at a later position in the
    # classified stream (the #1787 baseline).
    success_indices: dict[str, list[int]] = {}
    for idx, (name, is_err, _line) in enumerate(classified):
        if not is_err:
            success_indices.setdefault(name, []).append(idx)

    # Signal #2: per-tool, list of (position, args_repr) the agent
    # attempted to call it with.  Used to detect retry-with-different-
    # args even when the corresponding ToolMessage didn't make it
    # into the classified stream.
    call_args_by_tool: dict[str, list[tuple[int, str]]] = {}
    for pos, name, args_repr in _tool_call_args_index(messages):
        call_args_by_tool.setdefault(name, []).append((pos, args_repr))

    all_errors: list[str] = []
    unrecovered: list[str] = []
    for idx, (name, is_err, line) in enumerate(classified):
        if not is_err:
            continue
        all_errors.append(line)

        # Signal #1 check.
        later_successes = [i for i in success_indices.get(name, []) if i > idx]
        if later_successes:
            continue

        # Signal #2 check: same tool called later with different args.
        # The errored ToolMessage corresponds to a specific call args
        # repr — find it, then check whether any LATER attempt used a
        # different repr.  The errored tool_call's args appear at the
        # toolmsg_position == idx (its position in the classified
        # stream).  Walk attempts after that position.
        attempts = call_args_by_tool.get(name, [])
        # Locate the errored attempt to know its args_repr.
        errored_attempts = [(p, a) for (p, a) in attempts if p == idx]
        if errored_attempts:
            errored_args_repr = errored_attempts[0][1]
            later_different = any(p > idx and a != errored_args_repr for (p, a) in attempts)
            if later_different:
                continue

        unrecovered.append(line)
    return all_errors, unrecovered


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
        elif criterion.startswith("tool_called:"):
            # Positive structural check — mirror of tool_not_called.
            # Required by the multi-turn effort-gate reproducer (#1548): the
            # turn-2 contract asserts that the agent performed a fresh
            # search even after prior session search activity.  A substring
            # match against the haystack can be tripped by the agent
            # mentioning the tool name in prose ("let me try search_web");
            # this predicate looks at AIMessage.tool_calls only.
            required = criterion[len("tool_called:") :].strip().lower()
            found = False
            for msg in messages:
                if not isinstance(msg, AIMessage):
                    continue
                for tc in msg.tool_calls or []:
                    if str(tc.get("name", "")).lower() == required:
                        found = True
                        break
                if found:
                    break
            if not found:
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
        elif criterion.startswith("min_total_tool_calls:"):
            # Lower bound on total tool invocations.  Used by persistence
            # scenarios (#1520) to assert the agent did not refuse after
            # only 1-2 shallow searches.  Mirrors ``max_total_tool_calls:``.
            try:
                floor = int(criterion[len("min_total_tool_calls:") :].strip())
            except ValueError:
                return True
            total = sum(len(msg.tool_calls or []) for msg in messages if isinstance(msg, AIMessage))
            if total < floor:
                return True
        elif criterion.startswith("min_distinct_tool_calls:"):
            # Lower bound on *distinct* invocations of the named tool, keyed
            # on the call's ``args`` payload (JSON-normalised).  Catches
            # near-duplicate-query laziness — the :next24 reproducer of
            # #1520, where the agent issued 5 reordered variants of the
            # same query and counted them as effort.  Argument format:
            # ``min_distinct_tool_calls: search_web=3``.
            body = criterion[len("min_distinct_tool_calls:") :].strip()
            if "=" not in body:
                return True  # malformed predicate fails closed
            tool_name, _, count_str = body.partition("=")
            tool_name = tool_name.strip().lower()
            try:
                floor = int(count_str.strip())
            except ValueError:
                return True
            import json as _json

            seen_signatures: set[str] = set()
            for msg in messages:
                if not isinstance(msg, AIMessage):
                    continue
                for tc in msg.tool_calls or []:
                    if str(tc.get("name", "")).lower() != tool_name:
                        continue
                    try:
                        sig = _json.dumps(tc.get("args") or {}, sort_keys=True)
                    except (TypeError, ValueError):
                        sig = str(tc.get("args"))
                    seen_signatures.add(sig)
            if len(seen_signatures) < floor:
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
