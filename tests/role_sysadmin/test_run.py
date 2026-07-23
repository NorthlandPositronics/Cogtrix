"""Runner-logic unit tests: the verify gate, command capture, scenario loading.

These cover the orchestration seam without Docker or an LLM. The live wiring
(``cogtrix_agent_fn``) and the real container (``Target``) are exercised by a live
run and by ``test_target.py`` (docker-marked) respectively.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from tests.role_sysadmin.personas import PersonaChannel, ScenarioScript, Stage
from tests.role_sysadmin.run import (
    _collect_commands,
    _collect_tool_calls,
    _drive_agent,
    _serialize_messages,
    _stream_capture,
    find_scenario,
    load_scenario,
)


class _FakeGraph:
    """A graph whose .stream() yields value-states, optionally crashing.

    Models the two-call shape ``_stream_capture`` exercises: the first
    ``stream()`` is the main run; a second call is the step-limit recovery
    re-invoke. ``recovery_states`` / ``recovery_crash`` drive that second call.
    """

    def __init__(
        self,
        states: list[dict],
        *,
        crash: bool = False,
        recovery_states: list[dict] | None = None,
        recovery_crash: bool = False,
    ) -> None:
        self._states = states
        self._crash = crash
        self._recovery_states = recovery_states or []
        self._recovery_crash = recovery_crash
        self.calls = 0

    def stream(self, _inp: dict, _config: dict, *, stream_mode: str | None = None) -> Any:
        self.calls += 1
        if self.calls == 1:
            yield from self._states
            if self._crash:
                raise GraphRecursionError("Recursion limit of 100 reached")
        else:
            yield from self._recovery_states
            if self._recovery_crash:
                raise GraphRecursionError("Recursion limit of 4 reached")


def _shell_turn() -> list[Any]:
    return [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "execute_shell_command", "args": {"command": "ssh h 'ls'"}, "id": "1"}
            ],
        ),
        ToolMessage(content="ok", tool_call_id="1", name="execute_shell_command"),
    ]


def test_stream_capture_records_trail_on_success() -> None:
    ch = PersonaChannel(ScenarioScript())
    msgs = _shell_turn()
    out = _stream_capture(_FakeGraph([{"messages": msgs}]), [], ch)
    assert out == msgs
    assert ch.tool_calls == ["execute_shell_command"]
    assert ch.commands_log == ["ssh h 'ls'"]
    assert len(ch.debug_log) == 2


def test_stream_capture_recovers_on_step_limit_instead_of_crashing() -> None:
    # #21: hitting the recursion cap mirrors production (recover_from_step_limit)
    # — re-invoke once with a finalize nudge instead of re-raising as a crash.
    ch = PersonaChannel(ScenarioScript())
    msgs = _shell_turn()
    final = msgs + [AIMessage(content="Done: nginx installed and verified.")]
    g = _FakeGraph([{"messages": msgs}], crash=True, recovery_states=[{"messages": final}])

    out = _stream_capture(g, [], ch)  # must NOT raise

    assert ch.recovered_from_step_limit is True
    assert g.calls == 2  # original run + exactly one recovery re-invoke
    assert out == final  # the recovery's finalized messages are retained
    # The trail (commands/tool_calls/debug) reflects the full work, incl. recovery.
    assert ch.commands_log == ["ssh h 'ls'"]
    assert ch.tool_calls == ["execute_shell_command"]


def test_stream_capture_recovery_also_capped_still_no_crash() -> None:
    # If the recovery re-invoke ALSO hits its (tight) cap, we still finalize
    # without crashing — keeping the best trail we have, exactly as production.
    ch = PersonaChannel(ScenarioScript())
    msgs = _shell_turn()
    g = _FakeGraph(
        [{"messages": msgs}],
        crash=True,
        recovery_states=[{"messages": msgs}],
        recovery_crash=True,
    )

    out = _stream_capture(g, [], ch)  # must NOT raise

    assert ch.recovered_from_step_limit is True
    assert out == msgs
    assert ch.commands_log == ["ssh h 'ls'"]


def test_stream_capture_success_does_not_flag_recovery() -> None:
    # A normal run never sets the recovery flag and never re-invokes.
    ch = PersonaChannel(ScenarioScript())
    msgs = _shell_turn()
    g = _FakeGraph([{"messages": msgs}])
    _stream_capture(g, [], ch)
    assert ch.recovered_from_step_limit is False
    assert g.calls == 1


def test_drive_agent_nudges_once_when_not_done() -> None:
    ch = PersonaChannel(ScenarioScript())
    calls: list[int] = []

    def invoke(messages: list[Any]) -> list[Any]:
        calls.append(len(messages))
        return [*messages, AIMessage(content="ok")]

    _drive_agent(invoke, ch, "do the task", dod_gate=True)
    assert len(calls) == 2  # initial + one DoD nudge (never reached DONE)


def test_drive_agent_no_nudge_when_done() -> None:
    ch = PersonaChannel(ScenarioScript())
    calls: list[int] = []

    def invoke(messages: list[Any]) -> list[Any]:
        calls.append(len(messages))
        ch.message("lead", "all done — verified nginx is active")  # flips stage to DONE
        return [*messages, AIMessage(content="done")]

    _drive_agent(invoke, ch, "do the task", dod_gate=True)
    assert len(calls) == 1
    assert ch.stage == Stage.DONE


def test_drive_agent_gate_disabled() -> None:
    ch = PersonaChannel(ScenarioScript())
    calls: list[int] = []

    def invoke(messages: list[Any]) -> list[Any]:
        calls.append(len(messages))
        return [*messages, AIMessage(content="stopped early")]

    _drive_agent(invoke, ch, "do the task", dod_gate=False)
    assert len(calls) == 1  # no nudge even though not DONE


def test_collect_commands_extracts_only_shell() -> None:
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "execute_shell_command", "args": {"command": "ssh h 'ls'"}, "id": "1"},
                {"name": "read_file", "args": {"path": "x"}, "id": "2"},
                {"name": "execute_shell_command", "args": {"command": "scp f h:/tmp"}, "id": "3"},
            ],
        ),
        AIMessage(content="no tools here"),
    ]
    assert _collect_commands(messages) == ["ssh h 'ls'", "scp f h:/tmp"]
    # All tool calls (the efficiency signal) include non-shell tools too.
    assert _collect_tool_calls(messages) == [
        "execute_shell_command",
        "read_file",
        "execute_shell_command",
    ]


def test_serialize_messages_captures_decision_trail() -> None:
    messages = [
        HumanMessage(content="set up nginx"),
        AIMessage(
            content="I'll install nginx.",
            tool_calls=[
                {
                    "name": "execute_shell_command",
                    "args": {"command": "ssh h 'apt install nginx'"},
                    "id": "c1",
                }
            ],
        ),
        ToolMessage(content="nginx installed OK", tool_call_id="c1", name="execute_shell_command"),
    ]
    out = _serialize_messages(messages)
    assert [e["type"] for e in out] == ["HumanMessage", "AIMessage", "ToolMessage"]
    # AI message records the tool call it made...
    assert out[1]["tool_calls"][0]["name"] == "execute_shell_command"
    # ...and the ToolMessage records the RESULT the agent saw, linked by id.
    assert out[2]["content"] == "nginx installed OK"
    assert out[2]["tool_call_id"] == "c1"
    assert out[2]["name"] == "execute_shell_command"


def test_serialize_messages_truncates_huge_content() -> None:
    big = "x" * 50_000
    out = _serialize_messages([ToolMessage(content=big, tool_call_id="c1", name="t")])
    assert len(out[0]["content"]) < 50_000
    assert "truncated" in out[0]["content"]


def test_scenarios_load_and_resolve() -> None:
    nginx = load_scenario(find_scenario("sa_01"))
    assert nginx.id == "role_sa_01_nginx"
    assert "nginx" in nginx.assignment.lower()
    assert nginx.check_path.name == "sa_01_check.sh"
    assert nginx.check_path.exists()
    assert nginx.requires_rootcause is False
    assert nginx.seed_setup is None

    ssh = load_scenario(find_scenario("sa_03"))
    assert ssh.area == "security"
    assert ssh.check_path.exists()
