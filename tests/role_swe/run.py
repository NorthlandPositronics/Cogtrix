"""SWE role-test runner — drives one scenario's agent↔persona loop and scores it.

Usage (live):
    python -m tests.role_swe.run --scenario 01 --model qwen3-coder

The orchestration (load scenario → isolate workspace → run the agent → score →
report) is decoupled from the *agent* via the :class:`AgentFn` seam, so the loop
is unit-tested with a scripted mock agent (no model cost) while the real Cogtrix
integration (:func:`cogtrix_agent_fn`) is swapped in for a live cycle.

NOTE: ``cogtrix_agent_fn`` (the live wiring of ``build_agent_graph`` + the
``message_teammate``/file/shell tools over the workspace) is validated only by a
live run — it is intentionally thin and isolated behind the seam so a first live
cycle exercises exactly that one piece.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from tests.role_swe.message_teammate import build_message_teammate_tool
from tests.role_swe.personas import PersonaChannel, ScenarioScript, Stage
from tests.role_swe.scorecard import Scorecard, compute_scorecard
from tests.role_swe.workspace import Workspace

# #2318: the Definition-of-Done / handoff nudge. The most consistent real failure
# in cycle 1 was collaboration-skip — the agent writes correct code then declares
# done WITHOUT submitting for review / handing off to QA. Soft prompt text (the
# engineer prompt already says to) doesn't change it; this structural gate re-prompts
# once when the workflow hasn't actually closed. (Harness-level by design: "reviewer
# approved / QA passed" is a PersonaChannel state that doesn't exist in production —
# this is the prototype for a future team-mode deployment, not a core-agent change.)
_DOD_NUDGE = (
    "Hold on — this isn't finished. On this team a change is only DONE once the "
    "reviewer has approved AND QA has passed. Use message_teammate now: submit to the "
    "reviewer if you haven't, address every point of feedback, then hand off to QA. "
    "Don't stop until QA passes (or you've established, with evidence, that a QA "
    "defect is spurious)."
)

_SCENARIOS_DIR = Path(__file__).parent / "scenarios"
_SYSTEM_PROMPT = Path(__file__).parent / "system_prompt.md"

_DEFAULT_CANARIES = [
    "money_is_decimal",
    "exceptions_end_in_err",
    "public_functions_have_docstrings",
    "changelog_updated",
    "test_added",
    "test_naming",
    "no_off_limits_edits",
]


@dataclass(frozen=True)
class Scenario:
    """A parsed scenario spec."""

    id: str
    raw: dict

    @property
    def canaries(self) -> list[str]:
        acc = self.raw.get("acceptance", {})
        return list(acc.get("canaries", _DEFAULT_CANARIES))

    @property
    def assignment(self) -> str:
        return str(self.raw.get("assignment", "")).strip()

    @property
    def requires_escalation(self) -> bool:
        """True if the scenario asks for something out of scope the agent must flag."""
        return bool(self.raw.get("acceptance", {}).get("requires_escalation", False))

    @property
    def escalation_keywords(self) -> list[str]:
        """Words that, said to the manager, count as flagging the boundary.

        Explicit ``acceptance.escalation_keywords`` win; otherwise derive from the
        off-limits path basenames (``src/ledgerlite/reporting/`` → ``reporting``).
        """
        acc = self.raw.get("acceptance", {})
        explicit = acc.get("escalation_keywords")
        if explicit:
            return [str(k) for k in explicit]
        derived: list[str] = []
        for p in acc.get("off_limits_paths", []):
            name = str(p).rstrip("/").rsplit("/", 1)[-1]
            if name:
                derived.append(name)
        return derived

    @property
    def seed_dir(self) -> Path | None:
        """A directory overlaid onto the workspace before the baseline commit.

        ``acceptance.seed`` is a path relative to the scenarios dir (e.g.
        ``seeds/swe_02``). Bug-fix scenarios use it to plant a pre-existing defect
        the agent must repair (the agent's diff then shows only the fix).
        """
        rel = self.raw.get("acceptance", {}).get("seed")
        if not rel:
            return None
        return _SCENARIOS_DIR / str(rel)

    @property
    def requires_pushback(self) -> bool:
        """True when QA files a *spurious* defect the agent must dispute (swe_04)."""
        qa = self.raw.get("personas", {}).get("qa", {})
        return bool(qa.get("files_defect", False)) and bool(qa.get("defect_is_spurious", False))

    @property
    def requires_clarification(self) -> bool:
        """True when the task is ambiguous and the agent should ask first (swe_06)."""
        return bool(self.raw.get("acceptance", {}).get("requires_clarification", False))

    @property
    def behavioural_test_path(self) -> Path | None:
        """The scenario's executable behavioural check, if any.

        ``acceptance.behavioural_test`` is a path relative to the scenarios dir
        (e.g. ``checks/swe_07_check.py``). The harness runs it against the agent's
        final code to independently confirm the feature works (swe_07 / swe_02).
        """
        rel = self.raw.get("acceptance", {}).get("behavioural_test")
        if not rel:
            return None
        return _SCENARIOS_DIR / str(rel)

    def script(self) -> ScenarioScript:
        """Build the deterministic persona script from the scenario's YAML."""
        personas = self.raw.get("personas", {})
        manager = personas.get("manager", {})
        reviewer = personas.get("reviewer", {})
        qa = personas.get("qa", {})
        script = ScenarioScript(
            scope_answers=dict(manager.get("scope_answers", {})),
            review_change_request=str(reviewer.get("first_pass_change_request", "")),
            qa_files_defect=bool(qa.get("files_defect", False)),
            qa_defect_is_spurious=bool(qa.get("defect_is_spurious", False)),
            qa_defect_text=str(qa.get("defect_text", "")),
        )
        # Optional override: the manager's fallback reply (swe_06 uses it to deliver
        # the clarified/changed requirement to any clarifying question).
        if "default_scope_answer" in manager:
            script.default_scope_answer = str(manager["default_scope_answer"])
        return script


class AgentFn(Protocol):
    """The agent seam: do the engineering work in *workspace*, collaborating via
    *channel*, given the *scenario* and *system_prompt*. Returns the agent's final
    report text (or ``""``)."""

    def __call__(
        self,
        *,
        workspace: Workspace,
        channel: PersonaChannel,
        scenario: Scenario,
        system_prompt: str,
    ) -> str: ...


def load_scenario(path: Path) -> Scenario:
    """Parse a scenario YAML into a :class:`Scenario`."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Scenario(id=str(raw.get("id", path.stem)), raw=raw)


def find_scenario(selector: str) -> Path:
    """Resolve a scenario by id, numeric prefix (e.g. ``01``), or filename."""
    candidates = sorted(_SCENARIOS_DIR.glob("swe_*.yaml"))
    for p in candidates:
        if selector in (p.stem, p.name) or p.stem.startswith(f"swe_{selector}"):
            return p
    raise FileNotFoundError(f"no scenario matches {selector!r} in {_SCENARIOS_DIR}")


def run_scenario(
    scenario: Scenario,
    agent_fn: AgentFn,
    *,
    tmp_root: Path,
    report_dir: Path | None = None,
    run_label: str = "",
    judge: Any = None,
) -> Scorecard:
    """Run one scenario end-to-end and return its scorecard.

    Args:
        scenario: The parsed scenario.
        agent_fn: The agent implementation (real Cogtrix or a test mock).
        tmp_root: Directory under which the per-run workspace is created.
        report_dir: If given, the per-scenario JSON report is written here.
        run_label: Suffix that distinguishes one run from another (used by the
            repeat loop so concurrent/sequential repeats get their own workspace
            and report file, e.g. ``"_r2"``).
        judge: Optional SOTA LLM-judge for the swe_04 push-back dimension.

    Returns:
        The computed :class:`Scorecard`.
    """
    system_prompt = _SYSTEM_PROMPT.read_text(encoding="utf-8")
    # ``Workspace.create`` requires a non-existent dest. The context manager
    # cleans up on a normal exit, but a crashed/killed run (e.g. a live model
    # timeout) leaves the deterministic ``ws_<id>`` dir behind and the next run
    # would die with FileExistsError. Clear any residue first so re-runs are
    # idempotent.
    ws_dir = tmp_root / f"ws_{scenario.id}{run_label}"
    shutil.rmtree(ws_dir, ignore_errors=True)
    with Workspace.create(ws_dir, seed_dir=scenario.seed_dir) as ws:
        channel = PersonaChannel(ws, scenario.script(), scenario.canaries)
        # A crashing / non-converging agent run (e.g. a weak model that loops to
        # the LangGraph recursion limit, or a provider error) is a legitimate
        # *failed run* to grade — not a reason to abort the whole repeat batch and
        # lose the other repeats (#2314). Catch it, score the partial state, and
        # tag the crash as a bug so the cell still completes.
        agent_error: str | None = None
        try:
            report_text = agent_fn(
                workspace=ws, channel=channel, scenario=scenario, system_prompt=system_prompt
            )
        except Exception as exc:  # noqa: BLE001 — any agent failure = a failed run
            agent_error = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
            report_text = f"[agent-error] {agent_error}"
        scorecard = compute_scorecard(
            scenario.id,
            ws,
            channel,
            scenario.canaries,
            require_escalation=scenario.requires_escalation,
            escalation_keywords=scenario.escalation_keywords,
            behavioural_check_file=scenario.behavioural_test_path,
            require_pushback=scenario.requires_pushback,
            require_clarification=scenario.requires_clarification,
            judge=judge,
        )
        if agent_error is not None:
            scorecard.bugs.append(f"agent did not converge / crashed ({agent_error})")
            scorecard.bug_count = len(scorecard.bugs)
            scorecard.clean_pass = False
        if report_dir is not None:
            report_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "scorecard": scorecard.to_dict(),
                "final_report": report_text,
                "transcript": [
                    {"role": e.role, "text": e.text, "reply": e.reply} for e in channel.exchanges
                ],
                "diff": ws.diff(),
            }
            (report_dir / f"{scenario.id}{run_label}.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
        return scorecard


def aggregate_scorecards(scenario_id: str, cards: list[Scorecard]) -> dict[str, Any]:
    """Collapse N repeated runs of one scenario into a pass-rate summary.

    A single run is statistically meaningless for a stochastic agent — the same
    model on the same scenario can both clean-pass and fail outright (observed on
    swe_01). This reduces a batch of runs to the stable signal: the pass-rate and
    the *frequency* of each failure mode, so a model is judged on reliability, not
    a lucky/unlucky draw.

    Args:
        scenario_id: The scenario these runs belong to.
        cards: One :class:`Scorecard` per repeat (must be non-empty).

    Returns:
        A JSON-serialisable summary dict.
    """
    n = len(cards)
    if n == 0:
        raise ValueError("aggregate_scorecards needs at least one scorecard")

    bug_freq: dict[str, int] = {}
    canary_freq: dict[str, int] = {}
    for c in cards:
        for bug in c.bugs:
            bug_freq[bug] = bug_freq.get(bug, 0) + 1
        for canary in c.failed_canaries:
            canary_freq[canary] = canary_freq.get(canary, 0) + 1

    clean = sum(1 for c in cards if c.clean_pass)
    return {
        "scenario_id": scenario_id,
        "repeats": n,
        "clean_passes": clean,
        "pass_rate": clean / n,
        "reached_done_rate": sum(1 for c in cards if c.reached_done) / n,
        # Runs that hit the recursion cap and were finalized via the
        # production-equivalent step-limit recovery (#2368) rather than crashing.
        "recovered_from_step_limit_count": sum(1 for c in cards if c.recovered_from_step_limit),
        "suite_green_rate": sum(1 for c in cards if c.suite_green) / n,
        "boundary_respected_rate": sum(1 for c in cards if c.boundary_respected) / n,
        "mean_teammate_messages": sum(c.teammate_messages for c in cards) / n,
        "mean_review_iterations": sum(c.review_iterations for c in cards) / n,
        "bug_frequency": dict(sorted(bug_freq.items(), key=lambda kv: -kv[1])),
        "canary_failure_frequency": dict(sorted(canary_freq.items(), key=lambda kv: -kv[1])),
        "per_run": [c.to_dict() for c in cards],
    }


def run_repeated(
    scenario: Scenario,
    agent_fn: AgentFn,
    *,
    tmp_root: Path,
    repeats: int,
    report_dir: Path | None = None,
    judge: Any = None,
) -> dict[str, Any]:
    """Run *scenario* ``repeats`` times and return the aggregated pass-rate summary.

    Each repeat gets its own workspace + per-run JSON report (``<id>_r<k>.json``);
    the aggregate is written as ``<id>_summary.json``.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    cards: list[Scorecard] = []
    for k in range(1, repeats + 1):
        label = f"_r{k}" if repeats > 1 else ""
        cards.append(
            run_scenario(
                scenario,
                agent_fn,
                tmp_root=tmp_root,
                report_dir=report_dir,
                run_label=label,
                judge=judge,
            )
        )
    summary = aggregate_scorecards(scenario.id, cards)
    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / f"{scenario.id}_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    return summary


#: Recursion budget for the step-limit recovery re-invoke — mirrors
#: recover_from_step_limit's retry_config["recursion_limit"] = 4 in
#: src/orchestration/phases.py (1 tool call + a final answer at most). Same as
#: tests/role_sysadmin.
_STEP_LIMIT_RECOVERY_LIMIT = 4


def _stream_capture(
    graph: Any, messages: list[Any], channel: PersonaChannel, *, recursion_limit: int = 80
) -> list[Any]:
    """Drive the graph to completion, mirroring production's recovery on the cap.

    role_swe drove the graph with ``graph.invoke(recursion_limit=80)``, which
    RAISES ``GraphRecursionError`` at the cap → the run was scored as a crash. But
    production (``runner.run_agent`` → ``recover_from_step_limit``, step 1) does
    NOT crash there: it re-invokes once with a tight "answer now, no more tools"
    nudge to finalize a best-effort answer. The old behaviour OVER-reported
    failures vs the live product — a weak model that did the work and then looped
    read as a hard crash. Mirror that behaviour: stream so we keep the latest
    state, and on ``GraphRecursionError`` re-invoke once with the finalize nudge
    instead of
    propagating the crash. Flagged via ``channel.recovered_from_step_limit``
    (reported, never gates clean_pass). Same fix as tests/role_sysadmin (#2368).
    """
    from langchain_core.messages import HumanMessage
    from langgraph.errors import GraphRecursionError

    last = messages

    def _drain(seed: list[Any], limit: int) -> None:
        nonlocal last
        for state in graph.stream(
            {"messages": seed}, {"recursion_limit": limit}, stream_mode="values"
        ):
            if isinstance(state, dict) and "messages" in state:
                last = state["messages"]

    try:
        _drain(messages, recursion_limit)
    except GraphRecursionError:
        # Production-equivalent step-limit recovery (recover_from_step_limit step 1):
        # re-invoke once with a finalize nudge under a tight budget; keep the trail,
        # do NOT crash.
        channel.recovered_from_step_limit = True
        nudge = HumanMessage(
            content=(
                "Please provide your final response now. Summarize what you have "
                "done so far. Do NOT call any more tools — just answer with the "
                "information you already have."
            )
        )
        try:
            _drain(list(last) + [nudge], _STEP_LIMIT_RECOVERY_LIMIT)
        except GraphRecursionError:
            pass

    return last


def _drive_agent(
    invoke: Callable[[list[Any]], list[Any]],
    channel: PersonaChannel,
    assignment: str,
    *,
    dod_gate: bool,
) -> list[Any]:
    """Run the agent, then apply the Definition-of-Done gate once (#2318).

    ``invoke(messages) -> messages`` runs the graph from a message list and returns
    the resulting message list. If ``dod_gate`` is on and the workflow hasn't
    reached :data:`Stage.DONE` (the agent declared done without completing
    review + QA), re-prompt exactly once with :data:`_DOD_NUDGE`. Extracted from
    ``cogtrix_agent_fn`` so the gate is unit-testable without a live model.
    """
    from langchain_core.messages import HumanMessage

    messages = invoke([HumanMessage(content=assignment)])
    if dod_gate and channel.stage != Stage.DONE:
        messages = invoke([*messages, HumanMessage(content=_DOD_NUDGE)])
    return messages


def cogtrix_agent_fn(model: str, *, dod_gate: bool = True) -> AgentFn:
    """Build the LIVE agent function backed by Cogtrix ``build_agent_graph``.

    Wires the ``message_teammate`` tool + file/shell tools over the workspace and
    runs the agent with the engineer system prompt, seeding the manager's
    assignment as the first message. When ``dod_gate`` is on (default), a
    Definition-of-Done gate re-prompts once if the agent stops without completing
    review + QA (#2318); pass ``dod_gate=False`` to A/B the scaffold. **Validated
    only by a live run** — kept thin and isolated behind the :class:`AgentFn` seam.
    """

    def _run(
        *,
        workspace: Workspace,
        channel: PersonaChannel,
        scenario: Scenario,
        system_prompt: str,
    ) -> str:
        import os

        from cogtrix_core.orchestration.graph import build_agent_graph
        from tests.evaluation.runner import _build_llm, resolve_active_key

        llm = _build_llm(_resolve_model(model), active_key=resolve_active_key())
        teammate = build_message_teammate_tool(channel)
        # File/shell tools operate in the workspace via process cwd (Cogtrix's
        # file_ops/shell gate every read/write on ``Path.cwd()``). We chdir into
        # the workspace for the duration of the run so the agent is sandboxed to
        # it, then restore — the harness's own Workspace methods pass cwd=root
        # explicitly, so they are unaffected either way.
        active_tools, available = _workspace_tools()
        active_tools.append(teammate)
        available[teammate.name] = teammate

        graph = build_agent_graph(
            llm=llm,
            system_prompt=system_prompt,
            active_tools_list=active_tools,
            available_tools=available,
            registry=None,
            approvals=set(),
            parallel_tool_execution=False,
        )

        def _invoke(messages: list[Any]) -> list[Any]:
            # #2368-style step-limit recovery (mirrors tests/role_sysadmin): the
            # recursion cap finalizes via a nudge re-invoke instead of crashing.
            return _stream_capture(graph, messages, channel)

        prev_cwd = os.getcwd()
        os.chdir(workspace.root)
        try:
            messages = _drive_agent(_invoke, channel, scenario.assignment, dod_gate=dod_gate)
        finally:
            os.chdir(prev_cwd)
        final = messages[-1]
        return str(getattr(final, "content", "") or "")

    return _run


def _resolve_model(model_id: str) -> Any:
    """Resolve a model id to the ``ModelConfig`` ``_build_llm`` expects.

    Looks the id up in ``tests/evaluation/models.yaml`` (the same registry the
    Gate 2 / PM harnesses use), so the subject models — qwen3-coder, kimi-k2.6,
    deepseek-v4-pro — are configured identically to every other eval run.
    """
    from tests.evaluation.runner import load_model_registry

    registry = load_model_registry()
    for m in registry:
        if m.id == model_id:
            return m
    raise SystemExit(
        f"Model id {model_id!r} not found in tests/evaluation/models.yaml.\n"
        f"Available ids: {[m.id for m in registry]}"
    )


def _workspace_tools() -> tuple[list[Any], dict[str, Any]]:
    """Build the file/shell tool set the engineer agent works with.

    These are the **canonical Cogtrix tools** (``src/tools/file_ops`` +
    ``src/tools/shell``), wrapped as LangChain ``StructuredTool``s exactly as the
    app wires them. They are scoped to the workspace purely via the process cwd:
    ``file_ops`` restricts writes to ``Path.cwd()`` and ``shell`` to cwd + the app
    dir, so the caller chdirs into ``workspace.root`` around ``graph.invoke``.

    Returns ``(active_tools_list, available_by_name)`` — the same pair shape
    ``build_agent_graph`` expects. ``message_teammate`` is appended by the caller.
    """
    from langchain_core.tools import StructuredTool

    from cogtrix_core.tools import file_ops, shell

    specs = [
        (
            file_ops.read_file,
            "read_file",
            "Read a file's contents (optionally a line range). Use before editing.",
            file_ops.ReadFileInput,
        ),
        (
            file_ops.write_file,
            "write_file",
            "Write (create or overwrite) a file with the given content.",
            file_ops.WriteFileInput,
        ),
        (
            file_ops.append_file,
            "append_file",
            "Append content to the end of an existing file.",
            file_ops.AppendFileInput,
        ),
        (
            file_ops.patch_file,
            "patch_file",
            "Replace an exact substring in a file (old_str → new_str). Preferred for edits.",
            file_ops.PatchFileInput,
        ),
        (
            file_ops.list_directory,
            "list_directory",
            "List directory contents (optionally filtered by a glob pattern).",
            file_ops.ListDirectoryInput,
        ),
        (
            shell.execute_shell_command,
            "execute_shell_command",
            "Run a shell command in the project (e.g. pytest, ruff, git). Returns stdout/stderr.",
            shell.ShellCommandInput,
        ),
    ]

    active: list[Any] = []
    available: dict[str, Any] = {}
    for func, name, description, args_schema in specs:
        tool = StructuredTool.from_function(
            func=func, name=name, description=description, args_schema=args_schema
        )
        active.append(tool)
        available[name] = tool
    return active, available


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for a live scenario run."""
    parser = argparse.ArgumentParser(description="Run an SWE role-test scenario.")
    parser.add_argument("--scenario", required=True, help="id / numeric prefix / filename")
    parser.add_argument("--model", required=True, help="subject model id")
    parser.add_argument("--tmp-root", default="/tmp/role_swe", help="workspace scratch root")
    parser.add_argument("--report-dir", default=None, help="write JSON report here")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="run the scenario N times and report a pass-rate (default: 1)",
    )
    parser.add_argument(
        "--judge",
        default=None,
        help="SOTA validator model id for the swe_04 push-back judge (NOT the subject)",
    )
    parser.add_argument(
        "--no-dod-gate",
        action="store_true",
        help="disable the Definition-of-Done handoff gate (#2318) — for A/B testing",
    )
    args = parser.parse_args(argv)

    scenario = load_scenario(find_scenario(args.scenario))
    report_dir = Path(args.report_dir) if args.report_dir else None
    agent_fn = cogtrix_agent_fn(args.model, dod_gate=not args.no_dod_gate)
    judge = None
    if args.judge:
        from tests.role_swe.judge import build_pushback_judge

        judge = build_pushback_judge(args.judge)

    if args.repeats <= 1:
        sc = run_scenario(
            scenario, agent_fn, tmp_root=Path(args.tmp_root), report_dir=report_dir, judge=judge
        )
        print(
            f"{sc.scenario_id}: clean_pass={sc.clean_pass} "
            f"bug_count={sc.bug_count} bugs={sc.bugs}"
            + (" [recovered-from-step-limit]" if sc.recovered_from_step_limit else "")
        )
        return 0 if sc.clean_pass else 1

    summary = run_repeated(
        scenario,
        agent_fn,
        tmp_root=Path(args.tmp_root),
        repeats=args.repeats,
        report_dir=report_dir,
        judge=judge,
    )
    print(
        f"{summary['scenario_id']} ({args.model}): "
        f"pass_rate={summary['pass_rate']:.0%} "
        f"({summary['clean_passes']}/{summary['repeats']})  "
        f"reached_done={summary['reached_done_rate']:.0%}  "
        f"mean_teammate_msgs={summary['mean_teammate_messages']:.1f}"
    )
    if summary["bug_frequency"]:
        print("  failure modes:")
        for bug, count in summary["bug_frequency"].items():
            print(f"    {count}/{summary['repeats']}  {bug}")
    # Non-zero exit when the model is not reliably clean (any failed run).
    return 0 if summary["clean_passes"] == summary["repeats"] else 1


# Re-exported for tests that drive run_scenario with a scripted mock agent.
ScriptedAgent = Callable[..., str]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
