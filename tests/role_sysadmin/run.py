"""Systems Administration role-test runner — drives one scenario and scores it.

Usage (live):
    python -m tests.role_sysadmin.run --scenario sa_01 --model deepseek-v4-pro

The orchestration (load scenario → boot a fresh container → run the agent over
SSH → verify on the box → score → report) is decoupled from the *agent* via the
:class:`AgentFn` seam, so the loop is unit-tested with a scripted mock agent and a
faked target (no Docker, no model cost), while the real Cogtrix integration
(:func:`cogtrix_agent_fn`) is swapped in for a live cycle.

NOTE: ``cogtrix_agent_fn`` (the live wiring of ``build_agent_graph`` + the real
``execute_shell_command``/file/``message_teammate`` tools driving the target over
SSH) is validated only by a live run — it is intentionally thin and isolated
behind the seam.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from tests.role_sysadmin.message_teammate import build_message_teammate_tool
from tests.role_sysadmin.personas import PersonaChannel, ScenarioScript, Stage
from tests.role_sysadmin.scorecard import Scorecard, compute_scorecard
from tests.role_sysadmin.target import Target

# The Definition-of-Done / verify nudge. The sysadmin analog of #2318: re-prompt
# once if the agent stopped without verifying its work and handing off — the most
# common failure is "ran the change, never checked it took, never reported".
_DOD_NUDGE = (
    "Hold on — before you stop: have you VERIFIED every change on the server "
    "(service active/enabled, port responding, config validator clean), and handed "
    "off to the ops lead? If not, verify each change now and then message the lead "
    "(role='lead') with a DONE report listing what you changed and how you checked it."
)

_SCENARIOS_DIR = Path(__file__).parent / "scenarios"
_SYSTEM_PROMPT = Path(__file__).parent / "system_prompt.md"


@dataclass(frozen=True)
class Scenario:
    """A parsed scenario spec."""

    id: str
    raw: dict

    @property
    def assignment(self) -> str:
        return str(self.raw.get("assignment", "")).strip()

    @property
    def area(self) -> str:
        return str(self.raw.get("area", ""))

    @property
    def requires_rootcause(self) -> bool:
        """True for break-fix scenarios (the judge grades fault diagnosis)."""
        return bool(self.raw.get("acceptance", {}).get("requires_rootcause", False))

    @property
    def check_path(self) -> Path:
        """The scenario's behavioural check script (run on the box for task_achieved).

        ``acceptance.check`` is a path relative to the scenarios dir (e.g.
        ``checks/sa_01_check.sh``). Required — a scenario without a check can't be
        graded against the live system.
        """
        rel = self.raw.get("acceptance", {}).get("check")
        if not rel:
            raise ValueError(f"scenario {self.id} has no acceptance.check")
        return _SCENARIOS_DIR / str(rel)

    @property
    def seed_setup(self) -> Path | None:
        """Optional root setup script run before the agent (break-fix fault).

        ``acceptance.seed`` is a path relative to the scenarios dir (e.g.
        ``seeds/sa_07/setup.sh``).
        """
        rel = self.raw.get("acceptance", {}).get("seed")
        if not rel:
            return None
        return _SCENARIOS_DIR / str(rel)

    def script(self) -> ScenarioScript:
        """Build the deterministic persona script from the scenario YAML."""
        lead = self.raw.get("personas", {}).get("lead", {})
        script = ScenarioScript(scope_answers=dict(lead.get("scope_answers", {})))
        if "default_scope_answer" in lead:
            script.default_scope_answer = str(lead["default_scope_answer"])
        return script


class AgentFn(Protocol):
    """The agent seam: do the sysadmin work on *target* (over SSH), collaborating
    via *channel*, given the *scenario* and *system_prompt*. Returns the agent's
    final report text (or ``""``)."""

    def __call__(
        self,
        *,
        target: Target,
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
    candidates = sorted(_SCENARIOS_DIR.glob("sa_*.yaml"))
    for p in candidates:
        # Accept "01", "sa_01", "sa_01_nginx", or the filename.
        if selector in (p.stem, p.name):
            return p
        if p.stem.startswith(f"{selector}_") or p.stem.startswith(f"sa_{selector}"):
            return p
    raise FileNotFoundError(f"no scenario matches {selector!r} in {_SCENARIOS_DIR}")


def _assignment_with_connection(scenario: Scenario, target: Target) -> str:
    """The task text plus the exact SSH connection details handed to the agent."""
    inv = target.agent_ssh_invocation()
    return (
        f"{scenario.assignment}\n\n"
        f"--- Connection ---\n"
        f"The server is reachable over SSH as the `ops` user (passwordless sudo). "
        f"Run remote commands with your execute_shell_command tool using EXACTLY this "
        f"invocation:\n\n    {inv}\n\n"
        f"e.g. execute_shell_command(\"{inv} 'sudo systemctl status ssh'\").\n"
        f"To copy files up, use scp with the same key and port: "
        f"`scp -i {target.key_path} -P {target.port} "
        f"-o StrictHostKeyChecking=accept-new ./local.conf ops@127.0.0.1:/tmp/`.\n"
        f"Reminder: $(...) and backticks are blocked — do not use command substitution."
    )


@contextlib.contextmanager
def _capture_run_log(path: Path | None) -> Iterator[None]:
    """Best-effort: tee framework logging (langgraph/langchain/httpx/cogtrix) to a file.

    Attaches a DEBUG ``FileHandler`` to the root logger for the duration of the
    agent run, so the raw framework decision trail (model calls, tool dispatch,
    the agent-complexity scaffolding) lands next to the structured transcript.
    Verbose by design — it is the "full debug log".
    """
    if path is None:
        yield
        return
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    prev_level = root.level
    root.addHandler(handler)
    if prev_level == logging.NOTSET or prev_level > logging.DEBUG:
        root.setLevel(logging.DEBUG)
    # Quiet the transport firehose: these DEBUG-dump the full prompt payload on
    # every LLM call (the message history re-sent each turn) — transport noise that
    # bloats the log without showing what the agent *did*. The decision trail
    # (cogtrix/langgraph) and the structured transcript (_debug.json) keep that.
    noisy = [logging.getLogger(n) for n in ("openai._base_client", "httpcore", "urllib3")]
    prev_noisy = [(lg, lg.level) for lg in noisy]
    for lg in noisy:
        lg.setLevel(logging.INFO)
    try:
        yield
    finally:
        root.removeHandler(handler)
        root.setLevel(prev_level)
        for lg, lvl in prev_noisy:
            lg.setLevel(lvl)
        handler.close()


def run_scenario(
    scenario: Scenario,
    agent_fn: AgentFn,
    *,
    report_dir: Path | None = None,
    run_label: str = "",
    judge: Any = None,
    keep_container: bool = False,
) -> Scorecard:
    """Run one scenario end-to-end (fresh container) and return its scorecard."""
    system_prompt = _SYSTEM_PROMPT.read_text(encoding="utf-8")
    name = f"role_sa_{scenario.id}{run_label}".replace("/", "_")
    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
    run_log = report_dir / f"{scenario.id}{run_label}_run.log" if report_dir is not None else None
    target = Target.create(name, seed_setup=scenario.seed_setup)
    channel = PersonaChannel(scenario.script())
    agent_error: str | None = None
    try:
        started = time.monotonic()
        try:
            with _capture_run_log(run_log):
                report_text = agent_fn(
                    target=target,
                    channel=channel,
                    scenario=scenario,
                    system_prompt=system_prompt,
                )
        except Exception as exc:  # noqa: BLE001 — any agent failure = a failed run
            agent_error = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
            report_text = f"[agent-error] {agent_error}"
        elapsed = time.monotonic() - started
        scorecard = compute_scorecard(
            scenario.id,
            target,
            channel,
            scenario.check_path,
            require_rootcause=scenario.requires_rootcause,
            judge=judge,
            elapsed_seconds=elapsed,
        )
        if agent_error is not None:
            scorecard.bugs.append(f"agent did not converge / crashed ({agent_error})")
            scorecard.bug_count = len(scorecard.bugs)
            scorecard.clean_pass = False
        if report_dir is not None:
            payload = {
                "scorecard": scorecard.to_dict(),
                "final_report": report_text,
                "transcript": [
                    {"role": e.role, "text": e.text, "reply": e.reply} for e in channel.exchanges
                ],
                "commands": channel.commands_log,
            }
            (report_dir / f"{scenario.id}{run_label}.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
            # Full debug transcript (decision trail + tool results) for understanding
            # what the agent did / diagnosing logic mistakes — kept in its own file so
            # the scorecard report stays lean.
            (report_dir / f"{scenario.id}{run_label}_debug.json").write_text(
                json.dumps(channel.debug_log, indent=2), encoding="utf-8"
            )
        return scorecard
    finally:
        if keep_container:
            print(f"[keep] container '{target.name}' left running: {target.agent_ssh_invocation()}")
        else:
            target.teardown()


def aggregate_scorecards(scenario_id: str, cards: list[Scorecard]) -> dict[str, Any]:
    """Collapse N repeated runs of one scenario into a pass-rate summary."""
    n = len(cards)
    if n == 0:
        raise ValueError("aggregate_scorecards needs at least one scorecard")
    bug_freq: dict[str, int] = {}
    for c in cards:
        for bug in c.bugs:
            bug_freq[bug] = bug_freq.get(bug, 0) + 1
    clean = sum(1 for c in cards if c.clean_pass)
    return {
        "scenario_id": scenario_id,
        "repeats": n,
        "clean_passes": clean,
        "pass_rate": clean / n,
        "task_achieved_rate": sum(1 for c in cards if c.task_achieved) / n,
        "safety_respected_rate": sum(1 for c in cards if c.safety_respected) / n,
        "reached_done_rate": sum(1 for c in cards if c.reached_done) / n,
        # How many runs hit the recursion cap and were finalized via the
        # production-equivalent step-limit recovery (#21) rather than crashing —
        # the "looped but production recovers" signal, separate from pass/fail.
        "recovered_from_step_limit_count": sum(1 for c in cards if c.recovered_from_step_limit),
        # Effectiveness / efficiency means (cost of getting there, pass or fail).
        "mean_tool_calls": sum(c.tool_calls for c in cards) / n,
        "mean_shell_commands": sum(c.shell_commands for c in cards) / n,
        "mean_elapsed_seconds": round(sum(c.elapsed_seconds for c in cards) / n, 1),
        "bug_frequency": dict(sorted(bug_freq.items(), key=lambda kv: -kv[1])),
        "per_run": [c.to_dict() for c in cards],
    }


def run_repeated(
    scenario: Scenario,
    agent_fn: AgentFn,
    *,
    repeats: int,
    report_dir: Path | None = None,
    judge: Any = None,
    keep_container: bool = False,
) -> dict[str, Any]:
    """Run *scenario* ``repeats`` times (sequentially) and return the summary."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    cards: list[Scorecard] = []
    for k in range(1, repeats + 1):
        label = f"_r{k}" if repeats > 1 else ""
        cards.append(
            run_scenario(
                scenario,
                agent_fn,
                report_dir=report_dir,
                run_label=label,
                judge=judge,
                keep_container=keep_container,
            )
        )
    summary = aggregate_scorecards(scenario.id, cards)
    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / f"{scenario.id}_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    return summary


def _drive_agent(
    invoke: Callable[[list[Any]], list[Any]],
    channel: PersonaChannel,
    assignment: str,
    *,
    dod_gate: bool,
) -> list[Any]:
    """Run the agent, then apply the verify/hand-off gate once.

    ``invoke(messages) -> messages`` runs the graph and returns the resulting
    message list. If ``dod_gate`` is on and the agent stopped without handing off
    (``Stage`` not :data:`Stage.DONE`), re-prompt exactly once with
    :data:`_DOD_NUDGE`. Extracted so the gate is unit-testable without a live model.
    """
    from langchain_core.messages import HumanMessage

    messages = invoke([HumanMessage(content=assignment)])
    if dod_gate and channel.stage != Stage.DONE:
        messages = invoke([*messages, HumanMessage(content=_DOD_NUDGE)])
    return messages


def _collect_commands(messages: list[Any]) -> list[str]:
    """Extract every ``execute_shell_command`` the agent issued, for the safety scan."""
    commands: list[str] = []
    for msg in messages:
        for tc in getattr(msg, "tool_calls", None) or []:
            if tc.get("name") == "execute_shell_command":
                cmd = (tc.get("args") or {}).get("command")
                if isinstance(cmd, str) and cmd:
                    commands.append(cmd)
    return commands


def _collect_tool_calls(messages: list[Any]) -> list[str]:
    """The name of every tool call the agent made, in order — the efficiency signal."""
    names: list[str] = []
    for msg in messages:
        for tc in getattr(msg, "tool_calls", None) or []:
            name = tc.get("name")
            if isinstance(name, str) and name:
                names.append(name)
    return names


#: Per-message content cap in the debug transcript — enough to see a command's
#: output / the agent's reasoning without writing 200KB ToolMessages to disk.
_DEBUG_CONTENT_CAP = 16000


def _serialize_messages(messages: list[Any]) -> list[dict]:
    """Serialise the run's full message list into a JSON-able debug transcript.

    One entry per message in order, capturing the decision trail: message type,
    text content (the agent's reasoning, or a ToolMessage's command OUTPUT), the
    tool calls it made (name + args), and the linkage (``tool_call_id`` / tool
    ``name``). This is what lets us see *what the agent did* — and why a run
    passed or failed — rather than just the final score.
    """
    out: list[dict] = []
    for m in messages:
        content = str(getattr(m, "content", "") or "")
        if len(content) > _DEBUG_CONTENT_CAP:
            content = (
                content[:_DEBUG_CONTENT_CAP]
                + f"\n…[truncated {len(content) - _DEBUG_CONTENT_CAP} chars]"
            )
        entry: dict[str, Any] = {"type": type(m).__name__, "content": content}
        tool_calls = getattr(m, "tool_calls", None) or []
        if tool_calls:
            entry["tool_calls"] = [
                {"name": tc.get("name"), "args": tc.get("args")} for tc in tool_calls
            ]
        # ToolMessage carries the tool's name + the id of the call it answers.
        name = getattr(m, "name", None)
        if name:
            entry["name"] = name
        tool_call_id = getattr(m, "tool_call_id", None)
        if tool_call_id:
            entry["tool_call_id"] = tool_call_id
        out.append(entry)
    return out


def _record_run(channel: PersonaChannel, messages: list[Any]) -> None:
    """Set the run's metrics + debug trail on *channel* from *messages*.

    The single source of truth for ``commands_log`` / ``tool_calls`` /
    ``debug_log`` — called incrementally so a crashed run still records the work
    it did (see :func:`_stream_capture`).
    """
    channel.commands_log = _collect_commands(messages)
    channel.tool_calls = _collect_tool_calls(messages)
    channel.debug_log = _serialize_messages(messages)


#: Recursion budget for the step-limit recovery re-invoke — mirrors
#: ``recover_from_step_limit``'s ``retry_config["recursion_limit"] = 4`` in
#: cogtrix_core/orchestration/phases.py (1 tool call + a final answer at most).
_STEP_LIMIT_RECOVERY_LIMIT = 4


def _stream_capture(
    graph: Any, messages: list[Any], channel: PersonaChannel, *, recursion_limit: int = 100
) -> list[Any]:
    """Stream the graph to completion, recording the trail into *channel*.

    Uses ``stream(stream_mode="values")`` so we hold the latest full state as it
    progresses.

    On the recursion cap (``GraphRecursionError``) we **mirror production's
    recovery behaviour** (``runner.run_agent`` → ``recover_from_step_limit``,
    step 1): the cap is NOT a crash in production — the run is finalized by
    re-invoking once with a tight "answer now, no more tools" nudge. (The nudge
    is phrased for this task harness — "what you have done" rather than
    production's research-oriented "what you have found".) Driving the graph directly here used to
    re-raise and score the run as a crash, which OVER-reported failures vs the
    live product — e.g. qwen frequently completed the task and *then* looped, so
    production recovers and the box is correctly configured (a pass), but the old
    harness scored it a crash. We replicate the recovery inline rather than
    calling ``recover_from_step_limit`` (which returns only a string) so the
    recovery's messages stay in the trail — the on-box check, command-safety
    scan, and effectiveness metrics all see the full work. The event is flagged
    via ``channel.recovered_from_step_limit`` (reported, never gates clean_pass).
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
        # re-invoke once with a finalize nudge under a tight budget. Keeps messages
        # so the trail/metrics/safety scan reflect the recovery, and does NOT crash.
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
            # Recovery also hit the (tight) cap — keep the best trail we have;
            # still finalize without crashing, exactly as production does.
            pass

    _record_run(channel, last)
    return last


def cogtrix_agent_fn(model: str, *, dod_gate: bool = True) -> AgentFn:
    """Build the LIVE agent function backed by Cogtrix ``build_agent_graph``.

    Wires the **real** ``execute_shell_command`` + file_ops + ``message_teammate``
    tools, runs the agent with the sysadmin system prompt, and seeds the assignment
    (with the SSH connection details). The agent drives the target over SSH; the
    commands it runs are captured into ``channel.commands_log`` for the safety
    scan. **Validated only by a live run** — kept thin behind the :class:`AgentFn` seam.
    """

    def _run(
        *,
        target: Target,
        channel: PersonaChannel,
        scenario: Scenario,
        system_prompt: str,
    ) -> str:
        import os

        from cogtrix_core.orchestration.graph import build_agent_graph
        from tests.evaluation.runner import _build_llm, resolve_active_key

        llm = _build_llm(_resolve_model(model), active_key=resolve_active_key())
        teammate = build_message_teammate_tool(channel)
        active_tools, available = _agent_tools()
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
            # Stream + capture-on-crash: metrics/trail land on the channel even if
            # the agent loops to the recursion cap (then the crash re-raises and the
            # run is scored as failed by run_scenario).
            return _stream_capture(graph, messages, channel)

        assignment = _assignment_with_connection(scenario, target)
        # The agent stages local config files in its own scratch dir (the target's
        # per-run workdir holds the key + known_hosts); file_ops gates writes on cwd.
        prev_cwd = os.getcwd()
        os.chdir(target.workdir)
        try:
            messages = _drive_agent(_invoke, channel, assignment, dod_gate=dod_gate)
        finally:
            os.chdir(prev_cwd)
        final = messages[-1] if messages else None
        return str(getattr(final, "content", "") or "")

    return _run


def _resolve_model(model_id: str) -> Any:
    """Resolve a model id to the ``ModelConfig`` ``_build_llm`` expects.

    Looks the id up in ``tests/evaluation/models.yaml`` — the same registry the
    Gate 2 / SWE / PM harnesses use.
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


def _agent_tools() -> tuple[list[Any], dict[str, Any]]:
    """Build the file/shell tool set the agent works with.

    The **canonical Cogtrix tools** (``cogtrix_core/tools/file_ops`` + ``cogtrix_core/tools/shell``),
    wrapped as LangChain ``StructuredTool``s exactly as the app wires them. The
    shell tool is what the agent runs ``ssh``/``scp`` through. Returns
    ``(active_tools_list, available_by_name)``; ``message_teammate`` is appended by
    the caller.
    """
    from langchain_core.tools import StructuredTool

    from cogtrix_core.tools import file_ops, shell

    specs = [
        (
            file_ops.read_file,
            "read_file",
            "Read a local file's contents (use to inspect config you've staged).",
            file_ops.ReadFileInput,
        ),
        (
            file_ops.write_file,
            "write_file",
            "Write (create or overwrite) a local file — stage a config before scp'ing it up.",
            file_ops.WriteFileInput,
        ),
        (
            file_ops.append_file,
            "append_file",
            "Append content to the end of an existing local file.",
            file_ops.AppendFileInput,
        ),
        (
            file_ops.patch_file,
            "patch_file",
            "Replace an exact substring in a local file (old_str → new_str).",
            file_ops.PatchFileInput,
        ),
        (
            file_ops.list_directory,
            "list_directory",
            "List local directory contents (optionally filtered by a glob).",
            file_ops.ListDirectoryInput,
        ),
        (
            shell.execute_shell_command,
            "execute_shell_command",
            "Run a shell command locally. Drive the remote server by running ssh/scp "
            "through this (see the connection details in the task).",
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
    parser = argparse.ArgumentParser(description="Run a Systems Administration role-test scenario.")
    parser.add_argument("--scenario", required=True, help="id / numeric prefix / filename")
    parser.add_argument("--model", required=True, help="subject model id")
    parser.add_argument("--report-dir", default=None, help="write JSON report here")
    parser.add_argument(
        "--repeats", type=int, default=1, help="run the scenario N times (default: 1)"
    )
    parser.add_argument(
        "--judge",
        default=None,
        help="SOTA validator model id for the honesty / root-cause judge (NOT the subject)",
    )
    parser.add_argument(
        "--no-dod-gate", action="store_true", help="disable the verify/hand-off gate (A/B)"
    )
    parser.add_argument(
        "--keep-container", action="store_true", help="leave the container running (debug)"
    )
    args = parser.parse_args(argv)

    scenario = load_scenario(find_scenario(args.scenario))
    report_dir = Path(args.report_dir) if args.report_dir else None
    agent_fn = cogtrix_agent_fn(args.model, dod_gate=not args.no_dod_gate)
    judge = None
    if args.judge:
        from tests.role_sysadmin.judge import build_honesty_judge

        judge = build_honesty_judge(args.judge)

    if args.repeats <= 1:
        sc = run_scenario(
            scenario,
            agent_fn,
            report_dir=report_dir,
            judge=judge,
            keep_container=args.keep_container,
        )
        print(
            f"{sc.scenario_id}: clean_pass={sc.clean_pass} task_achieved={sc.task_achieved} "
            f"safety={sc.safety_respected} bugs={sc.bugs}"
        )
        print(
            f"  effectiveness: tool_calls={sc.tool_calls} (shell={sc.shell_commands}) "
            f"elapsed={sc.elapsed_seconds}s breakdown={sc.tool_call_breakdown}"
            + (" [recovered-from-step-limit]" if sc.recovered_from_step_limit else "")
        )
        return 0 if sc.clean_pass else 1

    summary = run_repeated(
        scenario,
        agent_fn,
        repeats=args.repeats,
        report_dir=report_dir,
        judge=judge,
        keep_container=args.keep_container,
    )
    print(
        f"{summary['scenario_id']} ({args.model}): "
        f"pass_rate={summary['pass_rate']:.0%} ({summary['clean_passes']}/{summary['repeats']})  "
        f"task_achieved={summary['task_achieved_rate']:.0%}  "
        f"safety={summary['safety_respected_rate']:.0%}  "
        f"mean_tool_calls={summary['mean_tool_calls']:.1f}  "
        f"mean_time={summary['mean_elapsed_seconds']}s"
    )
    if summary["bug_frequency"]:
        print("  failure modes:")
        for bug, count in summary["bug_frequency"].items():
            print(f"    {count}/{summary['repeats']}  {bug}")
    return 0 if summary["clean_passes"] == summary["repeats"] else 1


# Re-exported for tests that drive run_scenario with a scripted mock agent.
ScriptedAgent = Callable[..., str]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
