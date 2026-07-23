"""PM role-test harness — entry point (#1948).

One-shot harness that runs the six PM role-test scenarios against a
fully equipped Cogtrix agent (PM system prompt + Project Nimbus RAG
corpus + tool whitelist) and emits a decomposed scorecard focused on
hallucination + flawed-logic detection.

Usage::

    python -m tests.role_pm.run                           # all 6 scenarios
    python -m tests.role_pm.run --scenario 01,03,06       # selected
    python -m tests.role_pm.run --model qwen3-coder       # override default
    python -m tests.role_pm.run --output role_pm_run.json # JSON report
    python -m tests.role_pm.run --repeat 3                # N-repeat aggregation
    python -m tests.role_pm.run --judge qwen3-coder       # add LLM judge_score

Not wired into Gate 2.  Runs only when invoked manually.

This module deliberately avoids importing the heavy bits at module
load (LangChain, FAISS, Cogtrix orchestration) so ``--help`` is
fast and importing for static analysis is cheap.  All heavy imports
live inside the run functions.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Paths -------------------------------------------------------------

_HARNESS_DIR = Path(__file__).resolve().parent
# Everything for the role_pm harness lives under tests/role_pm/.  The
# scenarios used to live under tests/evaluation/scenarios/role_pm/ — but
# Gate 2's load_all_scenarios() rglobs that tree and choked on the
# harness-only ``system_prompt_file:`` extension (raised a UserWarning
# on every shard's Gate 2 cell).  Moved out for clean separation.
_CORPUS_DIR = _HARNESS_DIR / "corpus"
_SCENARIOS_DIR = _HARNESS_DIR / "scenarios"

# Post-#1951: both ``IngestConfig.vectordb_dir`` and
# ``configure_rag({"vectordb_dir": ...})`` mean the same thing — the
# directory that DIRECTLY holds ``index.faiss``.  No bridge needed.
_FAISS_INDEX_DIR = _HARNESS_DIR / "rag" / "faiss_index"


# Default model ----------------------------------------------------

# Per the design discussion (#1948): the test is deliberately run
# against a mediocre model so flaws surface faster.  The user named
# ``qwen3-coder``; that exact id is not in
# ``tests/evaluation/models.yaml`` at the time of writing — when the
# operator adds it, we pick it up automatically.  Until then we fall
# back to ``qwen3-32b`` (closest comparable size in the registry).
_DEFAULT_MODEL_CANDIDATES: tuple[str, ...] = (
    "qwen3-coder",
    "qwen3-coder-next",
    "qwen3-32b",
)


# ── Scenario loading ───────────────────────────────────────────────


def _load_scenario_yaml(path: Path) -> dict[str, Any]:
    """Load and minimally normalise one scenario file.

    Resolves ``system_prompt_file`` relative to the scenario's own
    directory and inlines it into ``system_prompt``.  Normalises
    legacy single-turn shape into a ``turns`` list so the runner
    can iterate uniformly.
    """
    import yaml

    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Scenario {path.name}: expected a YAML mapping")

    # Resolve system_prompt_file if present.
    sys_prompt_file = raw.pop("system_prompt_file", None)
    if sys_prompt_file:
        sys_prompt_path = (path.parent / sys_prompt_file).resolve()
        if not sys_prompt_path.exists():
            raise FileNotFoundError(
                f"Scenario {path.name}: system_prompt_file not found at {sys_prompt_path}"
            )
        raw["system_prompt"] = sys_prompt_path.read_text()

    # Normalise to turns shape.
    if not raw.get("turns") and raw.get("user_prompt"):
        raw["turns"] = [
            {
                "user_prompt": raw["user_prompt"],
                "success_criteria": list(raw.get("success_criteria") or []),
            }
        ]
        # Keep the originals for the scorecard view.

    return raw


def _load_scenarios(filter_ids: list[str] | None) -> list[dict[str, Any]]:
    paths = sorted(_SCENARIOS_DIR.glob("*.yaml"))
    scenarios: list[dict[str, Any]] = []
    for path in paths:
        scenario = _load_scenario_yaml(path)
        if filter_ids:
            # Allow filter by either the file's numeric prefix (01..06)
            # or the full scenario id (e.g. role_pm_01_status_update).
            number = path.stem.split("_", 1)[0]
            matches = number in filter_ids or scenario["id"] in filter_ids
            if not matches:
                continue
        scenarios.append(scenario)
    return scenarios


# ── Criteria evaluation ────────────────────────────────────────────


def _criterion_matches(
    criterion: str,
    response: str,
    tool_calls_made: list[str] | None = None,
) -> tuple[bool, str]:
    """Evaluate one success_criteria entry.

    Supports the standard Gate 2 operators:
    - ``contains: <substring>`` — passes when the substring is in the response.
    - ``not_contains: <substring>`` — passes when the substring is absent
      from the response.

    Plus the harness-specific extensions:
    - ``any_contains: opt1 | opt2 | opt3`` — passes when ANY of the
      pipe-separated alternatives appears in the response.
    - ``at_least_n_contains: <N> | opt1 | opt2 | opt3 | ...`` (#2024) —
      passes when at LEAST ``N`` of the pipe-separated options appear in
      the response.  Use this when you want a comprehensiveness signal
      but it would be wrong to insist on every specific token; e.g.
      ``at_least_n_contains: 2 | SC-1 | SC-2 | SC-3 | SC-4 | SC-5``
      requires the model to surface 2-of-5 special concerns rather than
      enumerate all five.  Avoids penalising honestly-terse models that
      refuse to mention items they have no grounded evidence for (see
      #2024 for cross-model failure data).
    - ``tool_called: <tool_name>`` — passes when *tool_name* appears in
      the agent's tool-call log (``tool_calls_made``).  Use this when
      the intent is *"did the agent invoke this tool"*, NOT *"does the
      tool name appear verbatim in the response text"*.  Many compliant
      responses do not mention the tool name even when the tool was
      called; ``contains:`` is the wrong operator for that question.

    Returns ``(passed, description)``.  ``description`` is a short
    human-readable label used in the failure list of the report.
    """
    text = criterion.strip()
    if text.lower().startswith("contains:"):
        needle = text.split(":", 1)[1].strip()
        return (needle in response, f"contains: {needle!r}")
    if text.lower().startswith("not_contains:"):
        needle = text.split(":", 1)[1].strip()
        return (needle not in response, f"not_contains: {needle!r}")
    if text.lower().startswith("any_contains:"):
        body = text.split(":", 1)[1]
        options = [opt.strip() for opt in body.split("|") if opt.strip()]
        return (
            any(opt in response for opt in options),
            f"any_contains: {options!r}",
        )
    if text.lower().startswith("at_least_n_contains:"):
        body = text.split(":", 1)[1]
        parts = [p.strip() for p in body.split("|") if p.strip()]
        if len(parts) < 2:
            # Malformed: need at least <N> + one option.  Treat as a
            # rubric and pass — the evaluator should never silently
            # fail a misconfigured criterion.
            return (True, f"(malformed at_least_n_contains): {text!r}")
        try:
            threshold = int(parts[0])
        except ValueError:
            return (True, f"(at_least_n_contains needs integer threshold): {text!r}")
        options = parts[1:]
        matched = sum(1 for opt in options if opt in response)
        return (
            matched >= threshold,
            f"at_least_n_contains: {threshold}/{len(options)} of {options!r} "
            f"(matched {matched})",
        )
    if text.lower().startswith("tool_called:"):
        tool_name = text.split(":", 1)[1].strip()
        calls = tool_calls_made or []
        return (tool_name in calls, f"tool_called: {tool_name!r}")
    # Unknown operator — treat as freeform rubric, not auto-evaluable.
    return (True, f"(rubric, not auto-evaluated): {text!r}")


def _evaluate_criteria(
    criteria: list[str],
    response: str,
    tool_calls_made: list[str] | None = None,
) -> tuple[int, int, list[str]]:
    """Return ``(passed, failed, failed_descriptions)``."""
    passed = 0
    failed = 0
    failed_desc: list[str] = []
    for c in criteria:
        ok, desc = _criterion_matches(c, response, tool_calls_made)
        if ok:
            passed += 1
        else:
            failed += 1
            failed_desc.append(desc)
    return passed, failed, failed_desc


# ── Single-scenario run ────────────────────────────────────────────


def _resolve_model(model_id: str | None) -> Any:
    """Resolve the model id to a Gate 2 ``ModelConfig``.

    Tries the explicit ``--model`` first, then the default candidate
    chain.  Raises with a clear message if nothing is found.
    """
    from tests.evaluation.runner import load_model_registry

    registry = load_model_registry()

    if model_id:
        for m in registry:
            if m.id == model_id:
                return m
        raise SystemExit(
            f"Model id {model_id!r} not found in tests/evaluation/models.yaml.\n"
            f"Available ids: {[m.id for m in registry]}"
        )

    for candidate in _DEFAULT_MODEL_CANDIDATES:
        for m in registry:
            if m.id == candidate:
                log.info("Using model %s (default chain)", candidate)
                return m

    raise SystemExit(
        "None of the default model candidates "
        f"({_DEFAULT_MODEL_CANDIDATES}) are in "
        "tests/evaluation/models.yaml.  Add one or pass --model explicitly."
    )


_TOOL_WHITELIST_PREAMBLE_TEMPLATE = (
    "\n\n## Strict tool whitelist (enforced — #1948 / #2016)\n\n"
    "This scenario enforces a hard tool whitelist.  The ONLY tool "
    "names you may emit in a structured tool call are:\n\n"
    "{whitelist_block}\n\n"
    "If you emit a tool call to any other name (``read_file``, "
    "``write_file``, ``get_weather``, etc.), the dispatcher will "
    "return a hard-refusal ToolMessage and the call will be counted "
    "as a scenario rule violation in the scorecard.  Do not retry a "
    "forbidden tool name.  If ``query_knowledge_base`` does not "
    "surface the value you need after 4–6 queries, surface that gap "
    'to the user plainly: *"I retrieved <N> chunks about <topic> but '
    'did not find <specific-thing>; recommend <action>."*  Reaching '
    "for ``read_file`` as a RAG-fallback is the cycle-2 / cycle-3 / "
    "cycle-4 Cluster B failure mode (#1987, #2006, #2016) — do not "
    "repeat it."
)


def _format_whitelist_block(tools_required: list[str], tools_available: list[str]) -> str:
    """Render the tool whitelist as a bullet block for the preamble.

    Deduplicated, sorted, prefixed with ``-`` so the model sees a
    clear inventory rather than a comma-separated string that can
    be misread as continuous prose.
    """
    names = sorted(set(tools_required) | set(tools_available))
    return "\n".join(f"- ``{n}``" for n in names) or "- (none — this scenario uses no tools)"


def _build_llm_and_graph(
    model: Any,
    system_prompt: str,
    tools_required: list[str],
    tools_available: list[str],
    *,
    enforce_tools_available: bool = False,
    corpus_attribution_detector: Any = None,
) -> Any:
    """Build a Cogtrix agent graph wired with the REAL
    ``query_knowledge_base`` tool (not the Gate 2 stub).

    The graph's compression + context-cap settings are intentionally
    left at the build_agent_graph defaults so the #1943 cascade
    (eviction marker + pre-flight guard + rolling summary + recovery
    node) is exercised end-to-end.  The eviction scenario relies on
    this — overriding the cap here would defeat the test.

    When ``enforce_tools_available`` is ``True``, a strict-whitelist
    preamble (see ``_TOOL_WHITELIST_PREAMBLE_TEMPLATE``) is appended
    to ``system_prompt``.  The active tool list is unchanged — the
    dispatcher already refuses out-of-whitelist calls via the #1919
    resolver; the preamble strengthens the model-side discipline so
    the refused calls don't keep recurring within a single turn (the
    cycle-4 Cluster B regression #2016 fingerprint).

    When ``corpus_attribution_detector`` is non-None, it is passed
    through to ``build_agent_graph`` so the #2015 corpus-aware
    attribution-mismatch recovery node runs in the recovery cascade.
    The PM harness's ``main()`` builds the detector closure around the
    harness-local ``AttributionIndex``; production code paths leave it
    as ``None`` and the orchestration layer stays corpus-agnostic.
    """
    from langchain_core.tools import StructuredTool

    from src.orchestration.graph import build_agent_graph
    from src.tools.rag import KnowledgeQueryInput, query_knowledge_base
    from tests.evaluation.runner import _build_llm, resolve_active_key

    # Build the live LLM.  Mirror the Gate 2 runner pattern (runner.py:831)
    # of passing the resolved active key so OPENROUTER_API_KEY can route
    # models that don't have a native upstream key set locally — e.g.
    # ``gpt-oss-20b-fireworks`` evaluated without ``FIREWORKS_API_KEY``.
    llm = _build_llm(model, active_key=resolve_active_key())

    # Build the real query_knowledge_base tool against the
    # role-test FAISS index (configured via configure_rag earlier
    # in main()).
    kb_tool = StructuredTool.from_function(
        func=query_knowledge_base,
        name="query_knowledge_base",
        description=(
            "Search the project knowledge base for information related "
            "to Project Nimbus.  Returns the most relevant document "
            "chunks with their source filenames.  Use this for every "
            "question about Project Nimbus before stating any "
            "project-specific fact.\n\n"
            # Per #1952 diagnostic — qwen3-embedding's discriminative
            # retrieval is weak.  Steer the agent toward k=8 so the
            # right document is more likely to appear in the returned
            # set even when the embedding doesn't rank it #1.
            "IMPORTANT: pass ``k=8`` (rather than the default 4) for "
            "every call — the embedding index needs more candidates to "
            "surface the right document reliably."
        ),
        args_schema=KnowledgeQueryInput,
    )

    # Active tool list: only the role-test whitelist.
    active_tools: list[Any] = []
    available_by_name: dict[str, Any] = {}

    whitelisted = set(tools_required) | set(tools_available)
    if "query_knowledge_base" in whitelisted:
        active_tools.append(kb_tool)
        available_by_name["query_knowledge_base"] = kb_tool

    # #2016 — when the scenario opts in, append the strict-whitelist
    # preamble so the model sees an explicit rule it can quote back to
    # itself rather than a soft suggestion buried in the cycle-2
    # preamble.  Empty whitelist short-circuits to "uses no tools"
    # which is harmless boilerplate.
    effective_system_prompt = system_prompt
    if enforce_tools_available:
        effective_system_prompt = system_prompt + _TOOL_WHITELIST_PREAMBLE_TEMPLATE.format(
            whitelist_block=_format_whitelist_block(tools_required, tools_available),
        )

    graph = build_agent_graph(
        llm=llm,
        system_prompt=effective_system_prompt,
        active_tools_list=active_tools,
        available_tools=available_by_name,
        registry=None,
        approvals=set(),
        # Deliberately keep the defaults for context_max_messages,
        # context_max_tokens, context_compression so the cascade is
        # exercised.  The eviction scenario depends on it.
        parallel_tool_execution=False,
        # #2015 — wire the curated-index corpus-aware mismatch detector
        # into the recovery cascade when the harness's main() has built
        # one.  ``None`` is the safe default (every non-PM caller of
        # build_agent_graph).
        corpus_attribution_detector=corpus_attribution_detector,
    )

    return graph


def _extract_tool_calls(messages: list[Any]) -> list[str]:
    """Flatten every AIMessage's tool_calls into a list of names."""
    names: list[str] = []
    for msg in messages:
        for call in getattr(msg, "tool_calls", None) or []:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if name:
                names.append(name)
    return names


def _extract_invalid_tool_names(messages: list[Any]) -> list[str]:
    """Return the names of tool calls that the dispatcher REJECTED at
    name resolution time (#2023 Track B).

    The dispatcher tags rejection ``ToolMessage``s with a
    ``cogtrix.kind`` of ``KIND_TOOL_NAME_INVALID`` or
    ``KIND_TOOL_RESOLUTION_FAILED``.  We pair each rejection back to
    its originating tool call via ``tool_call_id`` and surface the
    invented name.  These calls never actually executed, so a model
    that scores high on the count is producing quality noise rather
    than mis-firing tools — the PM scorecard splits them into the
    ``invalid_tool_names_count`` metric and excludes them from the
    acceptance-gating ``extraneous_tool_calls`` count.
    """
    from langchain_core.messages import ToolMessage

    from src.orchestration.tool_message_kinds import is_resolution_failure_message

    rejected_ids: set[str] = set()
    for msg in messages:
        if isinstance(msg, ToolMessage) and is_resolution_failure_message(msg):
            call_id = getattr(msg, "tool_call_id", "") or ""
            if call_id:
                rejected_ids.add(call_id)
    if not rejected_ids:
        return []

    invalid_names: list[str] = []
    for msg in messages:
        for call in getattr(msg, "tool_calls", None) or []:
            call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if name and call_id in rejected_ids:
                invalid_names.append(name)
    return invalid_names


def _extract_final_response(messages: list[Any]) -> str:
    """Return the content of the most recent AIMessage with text content."""
    from langchain_core.messages import AIMessage

    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = getattr(msg, "content", "")
            if isinstance(content, str) and content.strip():
                return content
    return ""


def _run_one_scenario(
    scenario: dict[str, Any],
    graph: Any,
) -> dict[str, Any]:
    """Run one scenario end-to-end and return the per-turn outputs."""
    from langchain_core.messages import HumanMessage

    turn_outputs: list[dict[str, Any]] = []
    accumulated_messages: list[Any] = []
    t0 = time.monotonic()

    turns = scenario.get("turns") or []
    for turn_idx, turn in enumerate(turns):
        user_prompt = turn["user_prompt"]
        accumulated_messages.append(HumanMessage(content=user_prompt))

        # Per-turn invocation.
        #
        # ``recursion_limit`` budget: ``max_turns * 8`` (was * 4 pre-#2005).
        # Each agent turn typically consumes 2-3 graph nodes (LLM → tool
        # → return); the historical × 4 multiplier was sized for
        # synthetic Gate 2 scenarios where the model converges in a few
        # turns.  PM cycle 3 (2026-06-03) showed 3 of 18 iterations
        # crashing with GraphRecursionError on the genuinely-distinct-
        # query path that the polling-loop / dedup detectors correctly
        # whitelist (issue #2005).  Doubling the multiplier gives the
        # model room to complete the distinct-args exploration without
        # weakening any other guard.  The harness crash-containment
        # wrapper (#2003) still catches anything that overruns this
        # higher budget so a runaway loop can't kill the whole run.
        result = graph.invoke(
            {"messages": list(accumulated_messages)},
            config={
                "recursion_limit": scenario.get("max_turns", 12) * 8,
            },
        )

        new_messages = result.get("messages", [])
        accumulated_messages = list(new_messages)

        turn_response = _extract_final_response(accumulated_messages)
        turn_window = accumulated_messages[len(accumulated_messages) - len(new_messages) :]
        turn_tool_calls = _extract_tool_calls(turn_window)
        turn_invalid_tool_names = _extract_invalid_tool_names(turn_window)

        criteria = turn.get("success_criteria") or []
        passed, failed, failed_desc = _evaluate_criteria(criteria, turn_response, turn_tool_calls)

        turn_outputs.append(
            {
                "turn_idx": turn_idx,
                "user_prompt": user_prompt,
                "final_response": turn_response,
                "tool_calls_made": turn_tool_calls,
                "invalid_tool_names": turn_invalid_tool_names,
                "criteria_passed": passed,
                "criteria_failed": failed,
                "criteria_failed_descriptions": failed_desc,
            }
        )

        # Reset graph state between turns IS NOT done — multi-turn
        # scenarios need the conversation history (see scenario 06).

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    return {
        "turn_outputs": turn_outputs,
        "elapsed_ms": elapsed_ms,
        "all_messages": accumulated_messages,
    }


# ── Scorecard rollup ───────────────────────────────────────────────


def _construct_eval_scenario(scenario: dict[str, Any]) -> Any:
    """Build a Gate 2 ``EvalScenario`` dataclass from the YAML dict.

    Used solely as the input shape to ``judge_response`` — the judge
    reads ``description``, ``expected_outcome``, and ``turns``
    (multi-turn) or ``user_prompt`` + ``success_criteria`` (single-turn)
    to build its prompt.  The harness-only ``system_prompt_file`` field
    is dropped before construction so the dataclass doesn't choke.
    """
    from tests.evaluation.runner import EvalScenario, Turn

    turns_dicts = scenario.get("turns") or []
    turns = [
        Turn(
            user_prompt=t.get("user_prompt", ""),
            success_criteria=list(t.get("success_criteria") or []),
            judge_weight=float(t.get("judge_weight", 1.0)),
        )
        for t in turns_dicts
    ]
    # The judge tolerates either shape; populate both for safety.
    return EvalScenario(
        id=scenario["id"],
        domain=scenario.get("domain", "role_pm"),
        title=scenario.get("title", scenario["id"]),
        description=scenario.get("description", ""),
        user_prompt=scenario.get("user_prompt", "") if not turns else "",
        system_prompt=scenario.get("system_prompt", ""),
        tools_required=list(scenario.get("tools_required") or []),
        expected_outcome=scenario.get("expected_outcome", ""),
        success_criteria=list(scenario.get("success_criteria") or []),
        max_turns=int(scenario.get("max_turns", 20)),
        timeout_seconds=int(scenario.get("timeout_seconds", 120)),
        tags=list(scenario.get("tags") or []),
        budget_usd_estimate=float(scenario.get("budget_usd_estimate", 0.05)),
        tools_available=list(scenario.get("tools_available") or []),
        turns=turns,
    )


def _construct_eval_result(
    scenario_id: str, run_output: dict[str, Any], turn_outputs: list[dict[str, Any]]
) -> Any:
    """Build a Gate 2 ``EvalResult`` dataclass from the harness's
    per-iteration run data.  Carries turn-by-turn results so the judge
    runs per-turn for multi-turn scenarios."""
    from tests.evaluation.runner import EvalResult, TurnResult

    flat_tool_calls: list[str] = []
    for t in turn_outputs:
        flat_tool_calls.extend(t.get("tool_calls_made") or [])
    final_response = turn_outputs[-1]["final_response"] if turn_outputs else ""

    turn_results = [
        TurnResult(
            final_response=t.get("final_response", ""),
            tool_calls_made=list(t.get("tool_calls_made") or []),
        )
        for t in turn_outputs
    ]

    return EvalResult(
        scenario_id=scenario_id,
        model_id="role-pm-harness",  # not consumed by the judge
        model_display_name="role-pm-harness",
        passed=False,  # the judge re-derives pass/fail; this is a placeholder
        tool_calls_made=flat_tool_calls,
        tool_calls_required=[],
        turns_used=len(turn_outputs),
        elapsed_seconds=run_output.get("elapsed_ms", 0) / 1000.0,
        final_response=final_response,
        turn_results=turn_results,
    )


def _build_scenario_scorecard(
    scenario: dict[str, Any],
    run_output: dict[str, Any],
    judge_model: str | None = None,
    attribution_index: Any = None,
) -> dict[str, Any]:
    """Build a per-iteration scorecard.

    When *judge_model* is provided, additionally invoke Gate 2's
    LLM-as-judge (``tests.evaluation.judge.judge_response``) to populate
    ``quality.judge_score``.  Costs one extra LLM call per iteration —
    off by default to keep ad-hoc runs cheap.

    When *attribution_index* is provided (the corpus name→entity
    mapping from :mod:`tests.role_pm.attribution_index`), the
    measurable scorecard's ``attribution_mismatches`` list is
    populated with any cases where the response stitches a real
    stakeholder name onto the wrong entity.  Each mismatch counts
    toward ``bug_count`` independently.
    """
    from tests.role_pm.attribution_index import detect_attribution_mismatches
    from tests.role_pm.scorecard import ScenarioScorecard, compute_measurable

    turn_outputs = run_output["turn_outputs"]
    # The decomposed scorecard is computed across all turns: the final
    # response is the LAST turn's response, but tool calls are summed.
    final_response = turn_outputs[-1]["final_response"] if turn_outputs else ""
    all_tool_calls: list[str] = []
    all_invalid_tool_names: list[str] = []
    criteria_passed = 0
    criteria_failed = 0
    criteria_total = 0
    for t in turn_outputs:
        all_tool_calls.extend(t["tool_calls_made"])
        all_invalid_tool_names.extend(t.get("invalid_tool_names") or [])
        criteria_passed += t["criteria_passed"]
        criteria_failed += t["criteria_failed"]
        criteria_total += t["criteria_passed"] + t["criteria_failed"]

    measurable = compute_measurable(
        scenario=scenario,
        tool_calls_made=all_tool_calls,
        final_response=final_response,
        turn_count=len(turn_outputs),
        latency_ms=run_output["elapsed_ms"],
        criteria_passed=criteria_passed,
        criteria_failed=criteria_failed,
        criteria_total=criteria_total,
        invalid_tool_names=all_invalid_tool_names,
    )

    # Quality scorecard.  In this first-cut harness we set the
    # LLM-judged signals from a simple rule that derives from the
    # measurable signals (hallucination ↔ failed not_contains: criteria,
    # task_done ↔ all required contains: criteria passed, etc.).
    # A future revision can plug in the Gate 2 LLM judge here for
    # finer-grained scoring; the structure is in place.
    hallucination = any(
        d.startswith("not_contains:") for d in _all_failed_descriptions(turn_outputs)
    )
    task_done = criteria_failed == 0
    on_role = measurable.extraneous_tool_calls == 0 and (
        "out_of_role" not in (scenario.get("tags") or []) or measurable.refusal_on_out_of_role
    )

    scorecard = ScenarioScorecard(
        scenario_id=scenario["id"],
        measurable=measurable,
    )
    scorecard.quality.hallucination_present = hallucination
    scorecard.quality.task_done = task_done
    scorecard.quality.on_role = on_role

    # Cycle-2 item #4 — attribution-mismatch detection.  Scan the final
    # response for ``<entity_id> ... <stakeholder>`` patterns and cross-
    # reference against the corpus.  Each mismatch becomes a distinct
    # bug_count entry that names the specific entity + claimed owner +
    # valid owner set, so the operator sees ATTRIBUTION SWAPS as their
    # own signal rather than rolled into a generic hallucination flag.
    if attribution_index is not None and final_response:
        mismatches = detect_attribution_mismatches(final_response, attribution_index)
        scorecard.measurable.attribution_mismatches = [m.describe() for m in mismatches]

    # Optional LLM judge.  When --judge MODEL_ID is passed, invoke the
    # Gate 2 judge per iteration.  judge_score becomes the aggregate
    # 0-1 score.  flawed_logic remains stubbed (False) — it's a
    # separate semantic check that would need its own judge prompt;
    # tracked in the cycle-2 findings doc as a follow-up.
    if judge_model:
        try:
            from tests.evaluation.judge import judge_response

            ev_scenario = _construct_eval_scenario(scenario)
            ev_result = _construct_eval_result(scenario["id"], run_output, turn_outputs)
            judge_score = judge_response(ev_scenario, ev_result, judge_model=judge_model)
            scorecard.quality.judge_score = judge_score
        except Exception as exc:  # noqa: BLE001 — judge must never crash the harness
            log.warning(
                "Judge call failed for %s (%s); leaving judge_score=0.0",
                scenario["id"],
                exc,
            )

    scorecard.quality.flawed_logic = False  # TODO: separate judge prompt

    return {
        "scenario_id": scenario["id"],
        "scorecard": scorecard,
        "turn_outputs": turn_outputs,
    }


def _all_failed_descriptions(turn_outputs: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for t in turn_outputs:
        out.extend(t.get("criteria_failed_descriptions", []))
    return out


# ── Reporting ──────────────────────────────────────────────────────


def _print_iteration_block(iteration: dict[str, Any], prefix: str = "") -> None:
    """Render one iteration's scorecard.  ``prefix`` is "  " when the
    iteration appears under a per-scenario aggregate header."""
    sc = iteration["scorecard"]
    m = sc.measurable
    q = sc.quality
    clean = sc.clean_pass()
    print(f"{prefix}Clean pass: {'YES' if clean else 'NO'}  |  bug_count={sc.bug_count}")
    print(
        f"{prefix}measurable:"
        f"  rag_consulted={m.rag_consulted}  "
        f"correct_tools={m.correct_tool_calls_count}  "
        f"extraneous_tools={m.extraneous_tool_calls}  "
        f"invalid_tool_names={m.invalid_tool_names_count}  "
        f"citation={m.citation_present}  "
        f"format={m.format_adherence}  "
        f"refusal_on_out_of_role={m.refusal_on_out_of_role}  "
        f"turns={m.turn_count}  "
        f"latency_ms={m.latency_total_ms}"
    )
    print(
        f"{prefix}quality:"
        f"  task_done={q.task_done}  "
        f"on_role={q.on_role}  "
        f"hallucination_present={q.hallucination_present}  "
        f"flawed_logic={q.flawed_logic}  "
        f"judge_score={q.judge_score:.2f}"
    )
    print(
        f"{prefix}criteria: {m.criteria_passed}/{m.criteria_total} passed; "
        f"{m.criteria_failed} failed"
    )
    if m.criteria_failed > 0:
        for fd in _all_failed_descriptions(iteration["turn_outputs"]):
            print(f"{prefix}  - failed: {fd}")
    if m.attribution_mismatches:
        print(f"{prefix}attribution_mismatches ({len(m.attribution_mismatches)}):")
        for mm in m.attribution_mismatches:
            print(f"{prefix}  - {mm}")


def _reproducibility(iterations: list[dict[str, Any]]) -> str:
    """Classify a scenario's outcome across iterations.

    ``always_pass`` — every iteration was a clean pass.  Solid.
    ``always_fail`` — no iteration was a clean pass.  Stable bug.
    ``flaky``       — some iterations passed, others didn't.  Run-to-run
                      noise; the underlying behaviour is unstable.
    """
    n = len(iterations)
    if n == 0:
        return "no_data"
    cleans = sum(1 for it in iterations if it["scorecard"].clean_pass())
    if cleans == n:
        return "always_pass"
    if cleans == 0:
        return "always_fail"
    return "flaky"


def _print_report(rolled: list[dict[str, Any]], repeat: int = 1) -> None:
    print()
    print("=" * 78)
    print("PM ROLE-TEST HARNESS — RESULTS")
    if repeat > 1:
        print(f"(repeat={repeat} — each scenario run N times)")
    print("=" * 78)
    print()

    total_bugs_min = 0
    total_bugs_max = 0
    clean_pass_scenarios = 0  # scenarios where ALL iterations were clean
    for entry in rolled:
        iterations = entry["iterations"]
        if repeat == 1 and len(iterations) == 1:
            # Backward-compatible single-iteration display.
            it = iterations[0]
            sc = it["scorecard"]
            print(f"Scenario: {sc.scenario_id}")
            _print_iteration_block(it, prefix="  ")
            print()
            total_bugs_min += sc.bug_count
            total_bugs_max += sc.bug_count
            if sc.clean_pass():
                clean_pass_scenarios += 1
            continue

        # Multi-iteration display.
        repro = _reproducibility(iterations)
        cleans = sum(1 for it in iterations if it["scorecard"].clean_pass())
        bug_counts = [it["scorecard"].bug_count for it in iterations]
        total_bugs_min += min(bug_counts)
        total_bugs_max += max(bug_counts)
        if repro == "always_pass":
            clean_pass_scenarios += 1

        print(f"Scenario: {entry['scenario_id']}  [{repro}]")
        print(
            f"  Clean passes: {cleans}/{len(iterations)};  "
            f"bug_count range [{min(bug_counts)}, {max(bug_counts)}]"
        )
        for i, it in enumerate(iterations):
            print(f"  --- iteration {i + 1}/{len(iterations)} ---")
            _print_iteration_block(it, prefix="    ")
        print()

    print("-" * 78)
    if repeat > 1:
        print(
            f"Summary: {clean_pass_scenarios}/{len(rolled)} scenarios "
            f"with ALL iterations clean;  "
            f"total bug_count range across scenarios "
            f"[{total_bugs_min}, {total_bugs_max}]"
        )
    else:
        print(
            f"Summary: {clean_pass_scenarios}/{len(rolled)} clean passes;  "
            f"total bug_count across scenarios: {total_bugs_max}"
        )
    print("=" * 78)
    print()


def _iteration_to_json(iteration: dict[str, Any]) -> dict[str, Any]:
    sc = iteration["scorecard"]
    return {
        "bug_count": sc.bug_count,
        "clean_pass": sc.clean_pass(),
        "measurable": {
            "rag_consulted": sc.measurable.rag_consulted,
            "correct_tool_calls_count": sc.measurable.correct_tool_calls_count,
            "extraneous_tool_calls": sc.measurable.extraneous_tool_calls,
            "invalid_tool_names_count": sc.measurable.invalid_tool_names_count,
            "citation_present": sc.measurable.citation_present,
            "format_adherence": sc.measurable.format_adherence,
            "refusal_on_out_of_role": sc.measurable.refusal_on_out_of_role,
            "turn_count": sc.measurable.turn_count,
            "latency_total_ms": sc.measurable.latency_total_ms,
            "criteria_passed": sc.measurable.criteria_passed,
            "criteria_failed": sc.measurable.criteria_failed,
            "criteria_total": sc.measurable.criteria_total,
            "attribution_mismatches": list(sc.measurable.attribution_mismatches),
        },
        "quality": {
            "task_done": sc.quality.task_done,
            "on_role": sc.quality.on_role,
            "hallucination_present": sc.quality.hallucination_present,
            "flawed_logic": sc.quality.flawed_logic,
            "judge_score": sc.quality.judge_score,
        },
        "turn_outputs": [
            {
                "turn_idx": t["turn_idx"],
                "user_prompt": t["user_prompt"][:500],
                # #2026: raised cap 2000 → 10000.  Verbose models
                # (DeepSeek V4 Pro, claude-sonnet-4-6) routinely
                # produce >2000-char responses for status / memo
                # scenarios; the 2000-char cap was hiding the
                # mismatch evidence needed for post-mortem analysis.
                # 10000 covers the longest observed responses.
                "final_response": t["final_response"][:10000],
                "tool_calls_made": t["tool_calls_made"],
                "criteria_passed": t["criteria_passed"],
                "criteria_failed": t["criteria_failed"],
                "criteria_failed_descriptions": t["criteria_failed_descriptions"],
            }
            for t in iteration["turn_outputs"]
        ],
    }


def _dump_json(rolled: list[dict[str, Any]], path: Path) -> None:
    payload = []
    for entry in rolled:
        iterations_json = [_iteration_to_json(it) for it in entry["iterations"]]
        cleans = sum(1 for ij in iterations_json if ij["clean_pass"])
        bugs = [ij["bug_count"] for ij in iterations_json]
        payload.append(
            {
                "scenario_id": entry["scenario_id"],
                "iteration_count": len(iterations_json),
                "clean_pass_count": cleans,
                "reproducibility": _reproducibility(entry["iterations"]),
                "bug_count_min": min(bugs) if bugs else 0,
                "bug_count_max": max(bugs) if bugs else 0,
                "iterations": iterations_json,
            }
        )
    path.write_text(json.dumps(payload, indent=2))


# ── Main ───────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PM role-test harness")
    parser.add_argument(
        "--scenario",
        default=None,
        help=(
            "Comma-separated scenario ids or numeric prefixes "
            "(e.g. '01,03' or 'role_pm_06_eviction_aware_honesty')"
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the default model id (see tests/evaluation/models.yaml)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write a JSON report to this path",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "Run each scenario N times and aggregate.  Cycle 1 of "
            "#1948 surfaced significant qwen3-coder run-to-run "
            "variance (e.g. R-13 mention present in run 4 but missing "
            "in run 3 of scenario 02).  Single-run signal is unstable; "
            "N=3 is recommended for cycle-2 quality runs.  Default 1 "
            "for backward compatibility."
        ),
    )
    parser.add_argument(
        "--judge",
        default=None,
        metavar="MODEL_ID",
        help=(
            "Enable Gate 2's LLM-as-judge per iteration.  The model id "
            "(e.g. 'qwen3-coder' or 'claude-sonnet-4-6') must exist in "
            "tests/evaluation/models.yaml AND have its API key in env. "
            "Populates the quality.judge_score field (0-1).  Costs one "
            "extra LLM call per scenario per iteration; off by default."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    args = parser.parse_args(argv)
    if args.repeat < 1:
        print(f"--repeat must be >= 1; got {args.repeat}", file=sys.stderr)
        return 2

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    filter_ids = [s.strip() for s in args.scenario.split(",")] if args.scenario else None
    scenarios = _load_scenarios(filter_ids)
    if not scenarios:
        print(f"No scenarios matched filter {filter_ids!r}", file=sys.stderr)
        return 2

    # Ingest corpus (idempotent).
    from tests.role_pm.corpus_ingest import ingest_corpus_idempotent

    # Pick up the user's existing embedding provider config if set;
    # otherwise default to OpenAI embeddings.  We do NOT touch the
    # user's ~/.cogtrix.yaml; the embedding config is read from
    # environment variables.
    embedding_provider = os.environ.get("ROLE_PM_EMBEDDING_PROVIDER", "openai")
    ingest_result = ingest_corpus_idempotent(
        corpus_dir=_CORPUS_DIR,
        # Post-#1951: ``vectordb_dir`` is the exact FAISS index directory.
        vectordb_dir=_FAISS_INDEX_DIR,
        embedding_provider=embedding_provider,
        embedding_model=os.environ.get("ROLE_PM_EMBEDDING_MODEL"),
        base_url=os.environ.get("ROLE_PM_EMBEDDING_BASE_URL"),
        api_key=os.environ.get("ROLE_PM_EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY"),
    )
    log.info(
        "Corpus ingest: skipped=%s, docs=%d, chunks=%d, vectordb=%s",
        ingest_result.skipped,
        ingest_result.documents_loaded,
        ingest_result.chunks_created,
        ingest_result.vectordb_dir,
    )

    # Build the corpus attribution index once.  Cheap (pure-text
    # regex scan); reused across all scenarios + iterations.  See
    # tests/role_pm/attribution_index.py for the parser + detector.
    from tests.role_pm.attribution_index import (
        build_attribution_index,
        detect_attribution_mismatches,
    )

    attribution_index = build_attribution_index(_CORPUS_DIR)
    log.info(
        "Attribution index built: %d entities, %d stakeholders",
        len(attribution_index.owners),
        len(attribution_index.known_stakeholders),
    )

    # #2015 — build the closure the orchestration layer consumes.  The
    # closure captures ``attribution_index`` so ``src/orchestration/``
    # can call it as a simple ``response_text → list[str]`` predicate
    # without ever importing tests/role_pm/attribution_index.py.  The
    # detector returns the structured ``describe()`` strings the
    # scorecard already uses, so the recovery node's nudge is the
    # SAME human-readable shape the operator sees in the JSON report.
    def _corpus_attribution_detector(response_text: str) -> list[str]:
        mismatches = detect_attribution_mismatches(response_text, attribution_index)
        return [m.describe() for m in mismatches]

    # Configure the rag tool to read from our isolated FAISS index
    # only — we do NOT want the test consulting the user's global
    # ~/.cogtrix vectordb.
    from src.tools.rag import configure_rag

    configure_rag(
        {
            "vectordb_dir": str(_FAISS_INDEX_DIR),
            "api_uploads_dir": None,
            "embedding_provider": embedding_provider,
            "embedding_model": os.environ.get("ROLE_PM_EMBEDDING_MODEL"),
            "base_url": os.environ.get("ROLE_PM_EMBEDDING_BASE_URL"),
            "api_key": os.environ.get("ROLE_PM_EMBEDDING_API_KEY")
            or os.environ.get("OPENAI_API_KEY"),
            "score_threshold": 0.0,
            # #2004 / #1952 Option A — enable the cross-encoder re-ranker
            # for the PM harness.  Issue #1952 was discovered DURING the
            # cycle-2 run; the PM harness is the cleanest representative
            # workload we have for those regime-B (numeric) and regime-C
            # (role-based) retrieval misses.  PR #1999 shipped the CE
            # re-rank stage as opt-in via this flag; cycle 3 confirmed
            # the same misses are still active with the flag at its
            # default False.  Flipping it here makes the next cycle-N
            # run an end-to-end test of PR #1999's actual effect.
            #
            # Requires the ``[rag-rerank]`` extra to be installed:
            #   ``uv sync --extra rag --extra rag-rerank``
            # When the extra is missing, the re-rank stage degrades
            # gracefully (returns the un-re-ranked pool) — retrieval is
            # never *worse* than the baseline, see src/rag/reranker.py
            # for the per-failure-mode contract.
            "use_cross_encoder_rerank": True,
        }
    )

    # Resolve model.
    model = _resolve_model(args.model)
    log.info("Running %d scenario(s) against model %s", len(scenarios), model.id)

    # Run each scenario, repeated --repeat times.
    rolled: list[dict[str, Any]] = []
    for scenario in scenarios:
        iterations: list[dict[str, Any]] = []
        for iter_idx in range(args.repeat):
            if args.repeat == 1:
                log.info("== %s ==", scenario["id"])
            else:
                log.info(
                    "== %s (iteration %d/%d) ==",
                    scenario["id"],
                    iter_idx + 1,
                    args.repeat,
                )
            # Rebuild the graph each iteration so per-run state
            # (counters, force_thinking_break flag, etc.) resets.
            graph = _build_llm_and_graph(
                model=model,
                system_prompt=scenario.get("system_prompt", ""),
                tools_required=list(scenario.get("tools_required") or []),
                tools_available=list(scenario.get("tools_available") or []),
                enforce_tools_available=bool(scenario.get("enforce_tools_available", False)),
                # #2015 — wire the closure in for every iteration so
                # the corpus-aware recovery node fires on PM responses.
                corpus_attribution_detector=_corpus_attribution_detector,
            )
            # Any uncaught exception from the agent graph (recursion
            # limit, provider 5xx, malformed tool args, etc.) MUST be
            # contained at the iteration boundary — the harness ships
            # a partial scorecard for the failed iter rather than
            # crashing the whole multi-scenario run.  The synthetic
            # empty run_output flows through ``_build_scenario_scorecard``
            # cleanly (criteria_total=0, all quality signals False),
            # producing an iteration that ``clean_pass()`` correctly
            # rejects so exit code stays 1.  Judging is skipped for
            # crashed iters (judge_model=None) to avoid paying for a
            # judge call against an empty response.
            crashed_with: str | None = None
            try:
                run_output = _run_one_scenario(scenario, graph)
            except Exception as exc:  # noqa: BLE001 — crash containment
                log.warning(
                    "Scenario %s iteration %d/%d crashed: %s: %s",
                    scenario["id"],
                    iter_idx + 1,
                    args.repeat,
                    type(exc).__name__,
                    exc,
                )
                crashed_with = f"{type(exc).__name__}: {exc}"
                run_output = {
                    "turn_outputs": [],
                    "elapsed_ms": 0,
                    "all_messages": [],
                    "_crashed_with": crashed_with,
                }
            iter_scorecard = _build_scenario_scorecard(
                scenario,
                run_output,
                judge_model=None if crashed_with else args.judge,
                attribution_index=attribution_index,
            )
            if crashed_with:
                # Surface the crash on the scorecard so the report is
                # self-explanatory without operators digging through
                # stdout logs.  An empty run_output otherwise yields
                # criteria_total=0 and bug_count=0, which ``clean_pass()``
                # would (incorrectly) treat as a pass.  Bumping
                # ``criteria_failed`` and ``hallucination_present``
                # ensures the iter is correctly recorded as a non-pass
                # and contributes to the aggregate bug count.
                sc = iter_scorecard["scorecard"]
                sc.measurable.criteria_failed += 1
                sc.measurable.criteria_total += 1
                sc.quality.hallucination_present = True
                sc.notes.append(f"harness_crash: {crashed_with}")
            iterations.append(iter_scorecard)
        rolled.append({"scenario_id": scenario["id"], "iterations": iterations})

    _print_report(rolled, repeat=args.repeat)

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        _dump_json(rolled, out_path)
        log.info("JSON report written to %s", out_path)

    # Exit code: 0 when EVERY scenario's EVERY iteration is a clean
    # pass; 1 otherwise.  Stricter than "any iteration was clean" —
    # a flaky scenario still returns 1.  This convention lets the
    # harness be wired into a future CI gate without changing it.
    if all(it["scorecard"].clean_pass() for entry in rolled for it in entry["iterations"]):
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
