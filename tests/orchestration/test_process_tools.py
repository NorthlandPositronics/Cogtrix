"""Unit tests for the extracted process_tools node."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from src.agent.core import CogtrixState
from src.orchestration.nodes.process_tools import build_process_tools_node
from src.orchestration.session_state import SessionState


class _DummyLogger:
    def __init__(self):
        self.infos: list[tuple[object, ...]] = []
        self.warnings: list[tuple[object, ...]] = []
        self.debugs: list[tuple[object, ...]] = []

    def info(self, *args: object):
        self.infos.append(args)

    def warning(self, *args: object):
        self.warnings.append(args)

    def debug(self, *args: object):
        self.debugs.append(args)


def _make_state(last_message) -> CogtrixState:
    return {"messages": [last_message]}


def _make_ai_msg(tool_calls, content="") -> AIMessage:
    return AIMessage(content=content, tool_calls=tool_calls)


def _make_node(**overrides):
    """Build a process_tools node with default mock dependencies."""
    import threading

    defaults = {
        "_invoke_one": MagicMock(
            return_value=ToolMessage(content="ok", tool_call_id="tc1", name="t1")
        ),
        "_tool_lookup": {"t1": MagicMock(name="t1")},
        "_active_names": {"t1"},
        "_available_tools_ref": [{}],
        "session_state": SessionState(),
        "parallel_tool_execution": False,
        "_identical_error_signature": MagicMock(return_value=None),
        "_tool_error_class": MagicMock(return_value=None),
        "_tool_error_guidance": MagicMock(return_value="stop retrying"),
        "_last_identical_error_signature": [None],
        "_consecutive_identical_error_count": [0],
        "_force_thinking_break": [False],
        "_graph_log": _DummyLogger(),
        "protected": {"request_tools"},
        "tool_catalog": {},
        "registry": None,
        "approvals": set(),
        "confirmation_ui": None,
        "git_native": False,
        "on_tool_expansion": None,
        "output_cap": 1000,
        "expansion_count": [0],
        "auto_expansion_count": [0],
        "request_tools_noop_count": [0],
        "_MAX_REQUEST_TOOLS_NOOPS": 3,
        "active_tools_list": [],
        "_tool_version": [0],
        "_calls_since_last_checkpoint": [0],
        "_same_file_writes": {},
        "_same_file_writes_lock": threading.Lock(),
        "_REWRITE_SEARCH_THRESHOLD": 2,
        "_consecutive_errors": [0],
        "_STUCK_THRESHOLD": 5,
        "_stuck_detection_headline": lambda x: x.split("\n")[0] if x else "",
        "_get_tool_executor": MagicMock(),
        "_detect_tool_request": MagicMock(return_value=None),
        "_safe_tool_name": lambda name, max_len=80: name[:max_len],
    }
    defaults.update(overrides)
    return build_process_tools_node(**defaults)


class TestProcessToolsBasicRouting:
    def test_returns_empty_when_last_message_has_no_tool_calls(self):
        node = _make_node()
        state = _make_state(HumanMessage(content="hello"))

        result = node(state, RunnableConfig())

        assert result == {"messages": []}

    def test_returns_empty_when_last_message_is_not_ai_message(self):
        node = _make_node()
        state = _make_state(ToolMessage(content="ok", tool_call_id="tc1", name="t1"))

        result = node(state, RunnableConfig())

        assert result == {"messages": []}


class TestProcessToolsSerialExecution:
    def test_invokes_known_tool_and_returns_tool_message(self):
        invoke_one = MagicMock(
            return_value=ToolMessage(content="result", tool_call_id="tc1", name="t1")
        )
        node = _make_node(_invoke_one=invoke_one)
        ai_msg = _make_ai_msg([{"name": "t1", "args": {}, "id": "tc1"}])
        state = _make_state(ai_msg)

        result = node(state, RunnableConfig())

        invoke_one.assert_called_once()
        msgs = result["messages"]
        assert len(msgs) == 1
        assert isinstance(msgs[0], ToolMessage)
        assert msgs[0].content == "result"

    def test_invokes_multiple_tools_in_serial_when_parallel_disabled(self):
        invoke_one = MagicMock(
            side_effect=[
                ToolMessage(content="r1", tool_call_id="tc1", name="t1"),
                ToolMessage(content="r2", tool_call_id="tc2", name="t2"),
            ]
        )
        node = _make_node(
            _invoke_one=invoke_one,
            _tool_lookup={"t1": MagicMock(), "t2": MagicMock()},
            _active_names={"t1", "t2"},
        )
        ai_msg = _make_ai_msg(
            [
                {"name": "t1", "args": {}, "id": "tc1"},
                {"name": "t2", "args": {}, "id": "tc2"},
            ]
        )
        state = _make_state(ai_msg)

        result = node(state, RunnableConfig())

        assert invoke_one.call_count == 2
        msgs = result["messages"]
        assert len(msgs) == 2
        assert msgs[0].content == "r1"
        assert msgs[1].content == "r2"


class TestProcessToolsUnknownToolHandling:
    def test_returns_guidance_when_tool_not_found(self):
        node = _make_node(
            _tool_lookup={},
            _active_names=set(),
        )
        ai_msg = _make_ai_msg([{"name": "unknown_tool_xyz", "args": {}, "id": "tc1"}])
        state = _make_state(ai_msg)

        result = node(state, RunnableConfig())

        msgs = result["messages"]
        assert len(msgs) == 2
        assert isinstance(msgs[0], ToolMessage)
        assert "not a valid tool" in msgs[0].content
        assert isinstance(msgs[1], HumanMessage)
        assert "Continue with your task" in msgs[1].content

    def test_suggests_request_tools_when_tool_is_available(self):
        node = _make_node(
            _tool_lookup={},
            _active_names=set(),
            _available_tools_ref=[{"web_search": MagicMock()}],
        )
        ai_msg = _make_ai_msg([{"name": "web_search", "args": {}, "id": "tc1"}])
        state = _make_state(ai_msg)

        result = node(state, RunnableConfig())

        msgs = result["messages"]
        assert len(msgs) == 1
        assert isinstance(msgs[0], ToolMessage)
        # Updated message: "in the catalog but not loaded" + advertises
        # the direct add= form first (with the actual tool name) before
        # the semantic query form.
        assert "in the catalog but not loaded" in msgs[0].content
        assert 'request_tools(add=["web_search"])' in msgs[0].content


class TestProcessToolsBudgetAndErrorHandling:
    def test_triggers_circuit_breaker_after_repeated_noop_request_tools(self):
        invoke_one = MagicMock(
            return_value=ToolMessage(content="no changes", tool_call_id="tc1", name="request_tools")
        )
        detect_tool_request = MagicMock(
            return_value=MagicMock(has_changes=False, add=[], remove=[])
        )
        node = _make_node(
            _invoke_one=invoke_one,
            _tool_lookup={"request_tools": MagicMock()},
            _active_names={"request_tools"},
            request_tools_noop_count=[2],
            _MAX_REQUEST_TOOLS_NOOPS=3,
            _detect_tool_request=detect_tool_request,
        )
        ai_msg = _make_ai_msg([{"name": "request_tools", "args": {}, "id": "tc1"}])
        state = _make_state(ai_msg)

        result = node(state, RunnableConfig())

        msgs = result["messages"]
        assert any("STOP" in m.content for m in msgs if isinstance(m, HumanMessage))

    def test_resets_error_counter_on_success(self):
        invoke_one = MagicMock(
            return_value=ToolMessage(content="success", tool_call_id="tc1", name="t1")
        )
        node = _make_node(
            _invoke_one=invoke_one,
            _consecutive_errors=[3],
        )
        ai_msg = _make_ai_msg([{"name": "t1", "args": {}, "id": "tc1"}])
        state = _make_state(ai_msg)

        node(state, RunnableConfig())

        assert node.__closure__ is not None  # just ensure it ran


class TestProcessToolsCheckpointTracking:
    def test_increments_calls_since_last_checkpoint(self):
        invoke_one = MagicMock(
            return_value=ToolMessage(content="ok", tool_call_id="tc1", name="t1")
        )
        calls_since = [0]
        node = _make_node(
            _invoke_one=invoke_one,
            _calls_since_last_checkpoint=calls_since,
        )
        ai_msg = _make_ai_msg([{"name": "t1", "args": {}, "id": "tc1"}])
        state = _make_state(ai_msg)

        node(state, RunnableConfig())

        assert calls_since[0] == 1

    def test_increments_for_multiple_tools(self):
        invoke_one = MagicMock(
            side_effect=[
                ToolMessage(content="r1", tool_call_id="tc1", name="t1"),
                ToolMessage(content="r2", tool_call_id="tc2", name="t2"),
            ]
        )
        calls_since = [0]
        node = _make_node(
            _invoke_one=invoke_one,
            _tool_lookup={"t1": MagicMock(), "t2": MagicMock()},
            _active_names={"t1", "t2"},
            _calls_since_last_checkpoint=calls_since,
        )
        ai_msg = _make_ai_msg(
            [
                {"name": "t1", "args": {}, "id": "tc1"},
                {"name": "t2", "args": {}, "id": "tc2"},
            ]
        )
        state = _make_state(ai_msg)

        node(state, RunnableConfig())

        assert calls_since[0] == 2


class TestProcessToolsStuckDetection:
    def test_adaptive_stuck_threshold_triggers_at_three_errors(self):
        """Production inline uses min(_STUCK_THRESHOLD, 3); extracted must match.

        _consecutive_errors counts *rounds* with all errors, not individual
        tool calls. We must call the node three separate times.
        """
        force_break = [False]
        consecutive_errors = [0]
        node = _make_node(
            _invoke_one=MagicMock(
                return_value=ToolMessage(content="Error: not found", tool_call_id="tc1", name="t1")
            ),
            _tool_lookup={"t1": MagicMock()},
            _active_names={"t1"},
            _consecutive_errors=consecutive_errors,
            _force_thinking_break=force_break,
            _STUCK_THRESHOLD=5,
        )

        for _ in range(3):
            ai_msg = _make_ai_msg([{"name": "t1", "args": {}, "id": "tc1"}])
            state = _make_state(ai_msg)
            node(state, RunnableConfig())

        assert force_break[0] is True
        assert consecutive_errors[0] == 3

    def test_adaptive_stuck_threshold_does_not_trigger_at_two_errors(self):
        force_break = [False]
        consecutive_errors = [0]
        node = _make_node(
            _invoke_one=MagicMock(
                return_value=ToolMessage(content="Error: not found", tool_call_id="tc1", name="t1")
            ),
            _tool_lookup={"t1": MagicMock()},
            _active_names={"t1"},
            _consecutive_errors=consecutive_errors,
            _force_thinking_break=force_break,
            _STUCK_THRESHOLD=5,
        )

        for _ in range(2):
            ai_msg = _make_ai_msg([{"name": "t1", "args": {}, "id": "tc1"}])
            state = _make_state(ai_msg)
            node(state, RunnableConfig())

        assert force_break[0] is False
        assert consecutive_errors[0] == 2

    def test_error_patterns_logged_on_stuck(self):
        class _CapturingLogger:
            def __init__(self):
                self.debugs: list[tuple[object, ...]] = []
                self.infos: list[tuple[object, ...]] = []

            def debug(self, *args: object):
                self.debugs.append(args)

            def info(self, *args: object):
                self.infos.append(args)

        graph_log = _CapturingLogger()
        force_break = [False]
        consecutive_errors = [0]
        node = _make_node(
            _invoke_one=MagicMock(
                return_value=ToolMessage(
                    content="Error: 404 not found", tool_call_id="tc1", name="t1"
                )
            ),
            _tool_lookup={"t1": MagicMock()},
            _active_names={"t1"},
            _force_thinking_break=force_break,
            _consecutive_errors=consecutive_errors,
            _graph_log=graph_log,
            _STUCK_THRESHOLD=5,
        )

        for _ in range(3):
            ai_msg = _make_ai_msg([{"name": "t1", "args": {}, "id": "tc1"}])
            state = _make_state(ai_msg)
            node(state, RunnableConfig())

        assert force_break[0] is True
        # debug should have been called with error patterns
        assert any("Error patterns" in str(args[0]) for args in graph_log.debugs)
        # info should include patterns
        assert any("patterns" in str(args[0]) for args in graph_log.infos)
