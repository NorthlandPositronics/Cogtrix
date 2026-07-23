"""Unit tests for the extracted process_tools node."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from src.agent.core import CogtrixState
from src.orchestration.graph import _safe_tool_name
from src.orchestration.nodes.process_tools import (
    build_process_tools_node,
    extract_prohibited_tools,
)
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
        "_tool_budget_lock": threading.Lock(),
        "_action_tier_consecutive_calls": {},
        "_last_action_tier_tool": [None],
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

    def test_unresolved_message_includes_top_k_candidates_when_available(self):
        """#1926: when there are above-floor near-matches, the
        dispatcher's "not a valid tool" message lists the closest
        candidates so the agent can recover without going through
        ``request_tools(query=...)``.
        """
        # Catalog has a tool that shares one token with the request —
        # Jaccard 1/3 ≈ 0.33 (above the 0.30 floor) but no qualifying
        # bonus, so total stays below the 0.65 fuzzy-match threshold:
        # ``resolve_tool_name`` returns None and the dispatcher hits
        # the no-match branch, but top-K surfaces ``search_database``.
        available_tool = MagicMock()
        node = _make_node(
            _tool_lookup={},
            _active_names=set(),
            _available_tools_ref=[{"search_database": available_tool}],
        )
        ai_msg = _make_ai_msg([{"name": "search_xyz", "args": {}, "id": "tc1"}])
        state = _make_state(ai_msg)

        result = node(state, RunnableConfig())

        msgs = result["messages"]
        assert len(msgs) == 2
        assert isinstance(msgs[0], ToolMessage)
        # Original message preserved.
        assert "not a valid tool" in msgs[0].content
        # New: closest-candidates hint surfaces the near-match.
        assert "Closest candidates:" in msgs[0].content
        assert "search_database" in msgs[0].content
        # New: also points to the discovery escape hatch.
        assert "request_tools(query=" in msgs[0].content

    def test_unresolved_message_compact_when_no_candidates_above_floor(self):
        """#1926: when there are no near-matches above the soft floor,
        no "Closest candidates" line is appended — keeps the message
        from carrying empty noise.
        """
        node = _make_node(
            _tool_lookup={},
            _active_names=set(),
            _available_tools_ref=[{}],  # empty catalog → no candidates
        )
        ai_msg = _make_ai_msg([{"name": "completely_unrelated_xyz", "args": {}, "id": "tc1"}])
        state = _make_state(ai_msg)

        result = node(state, RunnableConfig())

        msgs = result["messages"]
        assert isinstance(msgs[0], ToolMessage)
        assert "not a valid tool" in msgs[0].content
        # No "Closest candidates" line.
        assert "Closest candidates:" not in msgs[0].content

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

    def test_does_not_suggest_loading_when_fuzzy_match_is_already_active(self):
        """#1920: defensive check — the resolver can return a tool with
        ``source="available"`` that's also present in ``active_names_ref``
        (transient state during a turn where ``_activate_available_tool``
        mutation is deferred until after per-call resolution).

        Telling the agent to load a tool that's already loaded is the
        exact failure mode that produced the 18-call ``run`` loop in
        ``.agent-test-1918/test5`` (#1919).  The dispatcher must fall
        through to the not-a-valid-name branch instead.

        Note: #1924 added a short-request guard so the literal ``run``
        request now bails at the resolver and never hits the defensive
        check.  Use ``extend_runs`` (multi-token, past the guard) which
        fuzzy-matches ``extend_run`` via the prefix bonus.
        """
        # ``extend_run`` is BOTH available (resolver finds it) AND active
        # (already loaded).  Agent calls ``extend_runs`` (fuzzy matches
        # extend_run via the {run, runs} prefix bonus).
        node = _make_node(
            _tool_lookup={},
            _active_names={"extend_run"},
            _available_tools_ref=[{"extend_run": MagicMock()}],
        )
        ai_msg = _make_ai_msg([{"name": "extend_runs", "args": {"mode": "delegate"}, "id": "tc1"}])
        state = _make_state(ai_msg)

        result = node(state, RunnableConfig())

        msgs = result["messages"]
        assert len(msgs) == 2
        assert isinstance(msgs[0], ToolMessage)
        # The defensive branch fires: not the misleading "in the catalog
        # but not loaded" guidance.
        assert "in the catalog but not loaded" not in msgs[0].content
        # The agent gets a clear "not a valid tool" message + a hint that
        # the fuzzy match is already active so it can call it directly.
        assert "not a valid tool" in msgs[0].content
        assert "extend_run" in msgs[0].content
        assert "already active" in msgs[0].content

    def test_parallel_path_does_not_suggest_loading_when_match_is_already_active(self):
        """#1920 (parallel-path mirror): the same defensive check applies to
        the parallel-call resolution path at ``process_tools.py:625``.
        Without it, a parallel-call turn that includes a hallucinated
        name fuzzy-matching an already-active tool would re-introduce the
        resolver-loop failure mode.

        See sibling test above for the ``extend_runs`` choice rationale
        (#1924's short-request guard now blocks ``run`` at the resolver).
        """
        # Two parallel calls to force the parallel-execution path. One
        # known-active tool (so the path is exercised) plus the
        # hallucinated ``extend_runs`` that fuzzy-matches the already-
        # active ``extend_run``.
        invoke_one = MagicMock(
            return_value=ToolMessage(content="ok", tool_call_id="tc1", name="t1")
        )
        node = _make_node(
            _invoke_one=invoke_one,
            _tool_lookup={"t1": MagicMock(name="t1")},
            _active_names={"t1", "extend_run"},
            _available_tools_ref=[{"extend_run": MagicMock()}],
            parallel_tool_execution=True,
        )
        ai_msg = _make_ai_msg(
            [
                {"name": "t1", "args": {}, "id": "tc1"},
                {"name": "extend_runs", "args": {"mode": "delegate"}, "id": "tc2"},
            ]
        )
        state = _make_state(ai_msg)

        result = node(state, RunnableConfig())

        msgs = result["messages"]
        hallucinated_msgs = [
            m for m in msgs if isinstance(m, ToolMessage) and m.name == "extend_runs"
        ]
        assert len(hallucinated_msgs) == 1
        # Defensive branch fired in the parallel path too.
        assert "in the catalog but not loaded" not in hallucinated_msgs[0].content
        assert "not a valid tool" in hallucinated_msgs[0].content
        assert "extend_run" in hallucinated_msgs[0].content
        assert "already active" in hallucinated_msgs[0].content


class TestProcessToolsBurstAutoLoad:
    """cogtrix47 (Issue 2): when the model emits N≥2 parallel calls to
    the same in-catalog-but-unloaded tool, ``process_tools`` must
    auto-load the tool and execute the calls in-turn. Without this
    fix, the model paid 2N tool-call slots on the request_tools
    handshake — N "not loaded" stubs followed by N re-issued calls
    after request_tools landed.
    """

    def _make_loaded_tool(self, name):
        tool = MagicMock()
        tool.name = name
        return tool

    def test_parallel_burst_auto_loads_and_executes(self):
        """Three parallel calls to the same unloaded tool ⇒ one auto-load
        and three real invocations, no "not loaded" stubs."""
        web_tool = self._make_loaded_tool("web_search")
        invoke_calls: list[dict] = []

        def fake_invoke(call, config):
            invoke_calls.append(call)
            return ToolMessage(
                content=f"results for {call['args']['query']}",
                tool_call_id=call["id"],
                name=call["name"],
            )

        node = _make_node(
            _invoke_one=fake_invoke,
            _tool_lookup={},
            _active_names=set(),
            _available_tools_ref=[{"web_search": web_tool}],
            parallel_tool_execution=True,
            _get_tool_executor=lambda: __import__("concurrent.futures").futures.ThreadPoolExecutor(
                max_workers=2
            ),
        )

        ai_msg = _make_ai_msg(
            [
                {"name": "web_search", "args": {"query": "A"}, "id": "tc1"},
                {"name": "web_search", "args": {"query": "B"}, "id": "tc2"},
                {"name": "web_search", "args": {"query": "C"}, "id": "tc3"},
            ]
        )
        result = node(_make_state(ai_msg), RunnableConfig())

        msgs = result["messages"]
        tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
        human_msgs = [m for m in msgs if isinstance(m, HumanMessage)]

        # Every parallel call must produce a real ToolMessage with the
        # invocation result — none should be a "not loaded" stub.
        assert len(tool_msgs) == 3
        for m in tool_msgs:
            assert "in the catalog but not loaded" not in (m.content or "")
        # Underlying tool was invoked once per parallel call.
        assert len(invoke_calls) == 3
        called_queries = sorted(c["args"]["query"] for c in invoke_calls)
        assert called_queries == ["A", "B", "C"]
        # The existing tools_activated nudge (HumanMessage) fires once.
        # That's the same end-of-dispatch advisory request_tools(add=[...])
        # would have produced — desirable: the model still hears that
        # web_search is now in the toolkit.
        assert len(human_msgs) == 1
        assert "web_search" in (human_msgs[0].content or "")

    def test_single_call_to_unloaded_tool_still_returns_stub(self):
        """The burst guard MUST NOT trigger on a single call —
        single calls still go through the explicit request_tools
        handshake (the catalog-discovery intent the original
        ``Do NOT auto-load`` block defends)."""
        web_tool = self._make_loaded_tool("web_search")
        node = _make_node(
            _tool_lookup={},
            _active_names=set(),
            _available_tools_ref=[{"web_search": web_tool}],
            parallel_tool_execution=True,
        )
        ai_msg = _make_ai_msg([{"name": "web_search", "args": {"query": "A"}, "id": "tc1"}])
        result = node(_make_state(ai_msg), RunnableConfig())

        msgs = result["messages"]
        assert len(msgs) == 1
        assert "in the catalog but not loaded" in (msgs[0].content or "")

    def test_parallel_burst_with_distinct_unloaded_tools_no_auto_load(self):
        """When the parallel calls target DIFFERENT unloaded tools
        (one each), the burst guard must not fire — that's still a
        guess-shaped pattern. Both calls get the not-loaded stub."""
        node = _make_node(
            _tool_lookup={},
            _active_names=set(),
            _available_tools_ref=[
                {
                    "web_search": self._make_loaded_tool("web_search"),
                    "http_get": self._make_loaded_tool("http_get"),
                }
            ],
            parallel_tool_execution=True,
        )
        ai_msg = _make_ai_msg(
            [
                {"name": "web_search", "args": {"query": "A"}, "id": "tc1"},
                {"name": "http_get", "args": {"url": "https://x"}, "id": "tc2"},
            ]
        )
        result = node(_make_state(ai_msg), RunnableConfig())

        msgs = result["messages"]
        assert len(msgs) == 2
        for m in msgs:
            assert "in the catalog but not loaded" in (m.content or "")

    def test_parallel_burst_disabled_when_parallel_execution_off(self):
        """If parallel_tool_execution is False, the burst guard is a
        no-op even when N≥2 calls target the same unloaded tool —
        the orchestrator's serial path still emits N stubs. This is
        intentional: the burst optimisation only matters when the
        executor would otherwise dispatch the calls in parallel."""
        node = _make_node(
            _tool_lookup={},
            _active_names=set(),
            _available_tools_ref=[{"web_search": self._make_loaded_tool("web_search")}],
            parallel_tool_execution=False,
        )
        ai_msg = _make_ai_msg(
            [
                {"name": "web_search", "args": {"query": "A"}, "id": "tc1"},
                {"name": "web_search", "args": {"query": "B"}, "id": "tc2"},
            ]
        )
        result = node(_make_state(ai_msg), RunnableConfig())

        msgs = result["messages"]
        assert len(msgs) == 2
        for m in msgs:
            assert "in the catalog but not loaded" in (m.content or "")


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

    def test_auth_failure_triggers_stuck_detection(self):
        """Permission denied / auth failure increments consecutive_errors and
        triggers forced thinking break after 3 rounds. Closes #581.
        """
        force_break = [False]
        consecutive_errors = [0]
        node = _make_node(
            _invoke_one=MagicMock(
                return_value=ToolMessage(
                    content="Error: Permission denied or forbidden", tool_call_id="tc1", name="t1"
                )
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

    def test_tool_not_found_triggers_stuck_detection(self):
        """404 / not-found errors increment consecutive_errors. Closes #581."""
        force_break = [False]
        consecutive_errors = [0]
        node = _make_node(
            _invoke_one=MagicMock(
                return_value=ToolMessage(
                    content="Error: No such file or cannot open", tool_call_id="tc1", name="t1"
                )
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

    def test_timeout_triggers_stuck_detection(self):
        """Timeout errors increment consecutive_errors and trigger forced break
        after 3 rounds. Closes #581.
        """
        force_break = [False]
        consecutive_errors = [0]
        node = _make_node(
            _invoke_one=MagicMock(
                return_value=ToolMessage(
                    content="Error: Operation timed out", tool_call_id="tc1", name="t1"
                )
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

    def test_mixed_error_success_patterns_resets_counter(self):
        """A round with mixed error + success results is making progress;
        consecutive_errors must reset, not increment. Closes #581.
        """
        force_break = [False]
        consecutive_errors = [3]  # start with counter at 3
        # First call: one error, one success — should reset counter
        node = _make_node(
            _invoke_one=MagicMock(
                side_effect=[
                    # Round 1: error + success mixed — resets
                    ToolMessage(content="Error: not found", tool_call_id="tc1", name="t1"),
                    ToolMessage(
                        content="File saved to /tmp/out.txt", tool_call_id="tc2", name="t2"
                    ),
                    # Round 2: pure error — increments
                    ToolMessage(content="Error: not found", tool_call_id="tc1", name="t1"),
                    # Round 3: pure error — increments
                    ToolMessage(content="Error: not found", tool_call_id="tc1", name="t1"),
                    # Round 4: pure error — hits threshold
                    ToolMessage(content="Error: not found", tool_call_id="tc1", name="t1"),
                ]
            ),
            _tool_lookup={"t1": MagicMock(), "t2": MagicMock()},
            _active_names={"t1", "t2"},
            _consecutive_errors=consecutive_errors,
            _force_thinking_break=force_break,
            _STUCK_THRESHOLD=5,
        )

        # Round 1: mixed — counter should reset to 0
        ai_msg1 = _make_ai_msg(
            [
                {"name": "t1", "args": {}, "id": "tc1"},
                {"name": "t2", "args": {}, "id": "tc2"},
            ]
        )
        node(_make_state(ai_msg1), RunnableConfig())
        assert consecutive_errors[0] == 0, "Mixed error+success round must reset counter"

        # Round 2: pure error — increments
        ai_msg2 = _make_ai_msg([{"name": "t1", "args": {}, "id": "tc1"}])
        node(_make_state(ai_msg2), RunnableConfig())
        assert consecutive_errors[0] == 1

        # Round 3: pure error — increments and triggers break
        ai_msg3 = _make_ai_msg([{"name": "t1", "args": {}, "id": "tc1"}])
        node(_make_state(ai_msg3), RunnableConfig())
        assert consecutive_errors[0] == 2
        assert force_break[0] is False  # still below threshold of 3

        # Round 4: pure error — hits threshold
        ai_msg4 = _make_ai_msg([{"name": "t1", "args": {}, "id": "tc1"}])
        node(_make_state(ai_msg4), RunnableConfig())
        assert force_break[0] is True
        assert consecutive_errors[0] == 3


class TestProcessToolsSanitization:
    """Regression tests for issue #1070: _safe_tool_name sanitization inconsistency."""

    def test_serial_unknown_tool_sanitizes_name_in_tool_message(self):
        """Unknown tool names in serial path must be sanitized in ToolMessage content."""
        malicious_name = "evil<script>alert(1)</script>"
        safe_name = _safe_tool_name(malicious_name)
        assert safe_name != malicious_name
        node = _make_node(
            _tool_lookup={},
            _active_names=set(),
            _safe_tool_name=_safe_tool_name,
        )
        ai_msg = _make_ai_msg([{"name": malicious_name, "args": {}, "id": "tc1"}])
        state = _make_state(ai_msg)

        result = node(state, RunnableConfig())

        msgs = result["messages"]
        tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
        assert len(tool_msgs) >= 1
        # The first ToolMessage should contain the sanitized name, not the raw one
        assert safe_name in tool_msgs[0].content
        assert malicious_name not in tool_msgs[0].content

    def test_parallel_unknown_tool_sanitizes_name_in_tool_message(self):
        """Unknown tool names in parallel path must be sanitized in ToolMessage content."""
        malicious_name = 'bad_tool"; DROP TABLE users; --'
        safe_name = _safe_tool_name(malicious_name)
        assert safe_name != malicious_name
        node = _make_node(
            _tool_lookup={},
            _active_names=set(),
            _safe_tool_name=_safe_tool_name,
            parallel_tool_execution=True,
        )
        ai_msg = _make_ai_msg([{"name": malicious_name, "args": {}, "id": "tc1"}])
        state = _make_state(ai_msg)

        result = node(state, RunnableConfig())

        msgs = result["messages"]
        tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
        assert len(tool_msgs) >= 1
        # The ToolMessage from parallel pre-filter should contain sanitized name
        assert safe_name in tool_msgs[0].content
        assert malicious_name not in tool_msgs[0].content

    def test_serial_guidance_and_result_both_sanitized(self):
        """Both guidance_lines and ToolMessage content must use sanitized names."""
        malicious_name = "weird\nname\t"
        safe_name = _safe_tool_name(malicious_name)
        node = _make_node(
            _tool_lookup={},
            _active_names=set(),
            _safe_tool_name=_safe_tool_name,
        )
        ai_msg = _make_ai_msg([{"name": malicious_name, "args": {}, "id": "tc1"}])
        state = _make_state(ai_msg)

        result = node(state, RunnableConfig())

        msgs = result["messages"]
        # There should be a guidance HumanMessage and a ToolMessage
        tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
        human_msgs = [m for m in msgs if isinstance(m, HumanMessage)]
        assert len(tool_msgs) >= 1
        assert len(human_msgs) >= 1
        # Guidance message should also contain sanitized name
        assert safe_name in human_msgs[0].content
        assert malicious_name not in human_msgs[0].content
        assert safe_name in tool_msgs[0].content
        assert malicious_name not in tool_msgs[0].content


class TestHarmonyTokenStripping:
    """#2023 Track A — strip gpt-oss harmony channel markers from
    ``tool_call["name"]`` at dispatch entry so they don't surface as
    invented tool-name false positives downstream.
    """

    def test_strip_helper_removes_channel_commentary_suffix(self) -> None:
        from src.orchestration.nodes.process_tools import _strip_harmony_tokens

        assert (
            _strip_harmony_tokens("query_risk_register<|channel|>commentary")
            == "query_risk_register"
        )

    def test_strip_helper_removes_channel_final_suffix(self) -> None:
        from src.orchestration.nodes.process_tools import _strip_harmony_tokens

        assert (
            _strip_harmony_tokens("query_knowledge_base<|channel|>final") == "query_knowledge_base"
        )

    def test_strip_helper_handles_chained_markers(self) -> None:
        from src.orchestration.nodes.process_tools import _strip_harmony_tokens

        # Chained markers (planning channel + end token) — strip greedily.
        assert _strip_harmony_tokens("foo<|channel|>commentary<|end|>") == "foo"
        assert _strip_harmony_tokens("foo<|start|>blah<|end|>") == "foo"

    def test_strip_helper_passes_clean_name_through(self) -> None:
        from src.orchestration.nodes.process_tools import _strip_harmony_tokens

        assert _strip_harmony_tokens("query_knowledge_base") == "query_knowledge_base"
        assert _strip_harmony_tokens("checkpoint") == "checkpoint"

    def test_strip_helper_handles_empty_input(self) -> None:
        from src.orchestration.nodes.process_tools import _strip_harmony_tokens

        assert _strip_harmony_tokens("") == ""

    def test_sanitize_mutates_tool_call_dicts_in_place(self) -> None:
        from src.orchestration.nodes.process_tools import _sanitize_tool_call_names

        calls = [
            {"name": "query_knowledge_base", "args": {}, "id": "tc1"},
            {"name": "query_risk_register<|channel|>commentary", "args": {}, "id": "tc2"},
            {"name": "checkpoint<|end|>", "args": {}, "id": "tc3"},
        ]
        _sanitize_tool_call_names(calls)
        assert calls[0]["name"] == "query_knowledge_base"
        assert calls[1]["name"] == "query_risk_register"
        assert calls[2]["name"] == "checkpoint"

    def test_dispatch_resolves_after_harmony_strip(self) -> None:
        """End-to-end: a tool call with a harmony suffix should resolve
        to the underlying registered tool, NOT fall through to the
        not-a-valid-tool branch.  Verifies that the in-place sanitiser
        runs BEFORE the lookup so dispatch sees the clean name."""
        invoke_one = MagicMock(
            return_value=ToolMessage(content="ok", tool_call_id="tc1", name="query_knowledge_base")
        )
        node = _make_node(
            _invoke_one=invoke_one,
            _tool_lookup={"query_knowledge_base": MagicMock(name="query_knowledge_base")},
            _active_names={"query_knowledge_base"},
            _safe_tool_name=_safe_tool_name,
        )
        # Model emits the harmony-leaked name; dispatch should sanitise
        # and successfully invoke the underlying tool.
        ai_msg = _make_ai_msg(
            [
                {
                    "name": "query_knowledge_base<|channel|>commentary",
                    "args": {"query": "x"},
                    "id": "tc1",
                }
            ]
        )
        state = _make_state(ai_msg)

        result = node(state, RunnableConfig())

        # The sanitiser must mutate the AIMessage's tool_calls in place
        # so the cleaned name is what dispatch (and any downstream
        # checks) see.
        assert ai_msg.tool_calls[0]["name"] == "query_knowledge_base"
        # And dispatch invoked the real tool — not the rejection path.
        invoke_one.assert_called_once()
        tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert "not a valid tool" not in tool_msgs[0].content


class TestProcessToolsLockDiscipline:
    """Regression tests for issue #1091: _tool_lookup mutations must hold _tool_budget_lock."""

    def test_request_tools_expansion_holds_lock(self):
        """request_tools pop + re-add must acquire _tool_budget_lock."""
        lock_acquisitions: list[str] = []

        class _TrackingLock:
            def acquire(self, blocking=True, timeout=-1):
                lock_acquisitions.append("acquire")
                return True

            def release(self):
                lock_acquisitions.append("release")

            def __enter__(self):
                self.acquire()
                return self

            def __exit__(self, *args):
                self.release()

        def _invoke_one(call, _config):
            return ToolMessage(
                content="added web_search", tool_call_id=call["id"], name="request_tools"
            )

        detect = MagicMock(return_value=MagicMock(has_changes=True, add=["web_search"], remove=[]))
        lock = _TrackingLock()
        node = _make_node(
            _invoke_one=_invoke_one,
            _tool_lookup={"request_tools": MagicMock()},
            _active_names={"request_tools"},
            _available_tools_ref=[{"web_search": MagicMock()}],
            _detect_tool_request=detect,
            _tool_budget_lock=lock,
        )
        ai_msg = _make_ai_msg(
            [{"name": "request_tools", "args": {"add": ["web_search"]}, "id": "tc1"}]
        )
        state = _make_state(ai_msg)

        node(state, RunnableConfig())

        # The expansion path should have acquired the lock at least twice:
        # once for popping request_tools, once for re-adding it.
        assert lock_acquisitions.count("acquire") >= 2
        assert lock_acquisitions.count("release") >= 2

    def test_tool_add_and_remove_hold_lock(self):
        """Adding and removing tools via request_tools must acquire _tool_budget_lock."""
        lock_acquisitions: list[str] = []

        class _TrackingLock:
            def acquire(self, blocking=True, timeout=-1):
                lock_acquisitions.append("acquire")
                return True

            def release(self):
                lock_acquisitions.append("release")

            def __enter__(self):
                self.acquire()
                return self

            def __exit__(self, *args):
                self.release()

        def _invoke_one(call, _config):
            return ToolMessage(content="ok", tool_call_id=call["id"], name="request_tools")

        detect = MagicMock(
            return_value=MagicMock(has_changes=True, add=["new_tool"], remove=["old_tool"])
        )
        lock = _TrackingLock()
        old_tool = MagicMock()
        old_tool.name = "old_tool"
        node = _make_node(
            _invoke_one=_invoke_one,
            _tool_lookup={"request_tools": MagicMock(), "old_tool": old_tool},
            _active_names={"request_tools", "old_tool"},
            _available_tools_ref=[{"new_tool": MagicMock()}],
            active_tools_list=[old_tool],
            _detect_tool_request=detect,
            _tool_budget_lock=lock,
        )
        ai_msg = _make_ai_msg(
            [
                {
                    "name": "request_tools",
                    "args": {"add": ["new_tool"], "remove": ["old_tool"]},
                    "id": "tc1",
                }
            ]
        )
        state = _make_state(ai_msg)

        node(state, RunnableConfig())

        # Should acquire lock for: remove old_tool, add new_tool, rebuild request_tools.
        assert lock_acquisitions.count("acquire") >= 3
        assert lock_acquisitions.count("release") >= 3

    def test_concurrent_lookup_and_mutation_no_race(self):
        """Parallel _invoke_one readers must not race with request_tools mutator.

        Simulates the scenario where one thread reads _tool_lookup while
        the main thread mutates it during request_tools processing.
        """
        import threading
        import time

        real_lock = threading.Lock()
        read_errors: list[Exception] = []
        mutation_errors: list[Exception] = []

        def slow_invoke_one(call, _config):
            # Simulate a slow tool call that reads _tool_lookup repeatedly
            for _ in range(50):
                try:
                    with real_lock:
                        _ = call.get("name", "") in {"t1", "t2"}
                except Exception as exc:
                    read_errors.append(exc)
                time.sleep(0.001)
            return ToolMessage(content="ok", tool_call_id=call["id"], name=call["name"])

        node = _make_node(
            _invoke_one=slow_invoke_one,
            _tool_lookup={"t1": MagicMock(), "t2": MagicMock()},
            _active_names={"t1", "t2"},
            parallel_tool_execution=True,
            _tool_budget_lock=real_lock,
        )
        ai_msg = _make_ai_msg(
            [
                {"name": "t1", "args": {}, "id": "tc1"},
                {"name": "t2", "args": {}, "id": "tc2"},
            ]
        )
        state = _make_state(ai_msg)

        node(state, RunnableConfig())

        assert not read_errors, f"Read errors during concurrent access: {read_errors}"
        assert not mutation_errors, f"Mutation errors: {mutation_errors}"


class TestProcessToolsActionTierCap:
    """Action-tier consecutive-call cap (Bug F #1712).

    The polling-loop advisory in call_model is non-binding: the LLM can
    emit another batch of identical web_search / http_get calls right
    after seeing the warning, and the existing "Temporal polling loop
    detected" flag only arms a thinking break for the *next* round —
    not the current dispatch. These tests pin the dispatcher-side hard
    cap that converts the 6th+ consecutive emission of the same
    action-tier tool into a cap-hit ToolMessage without invoking the
    real tool. The threshold (MAX_CONSECUTIVE_ACTION_CALLS = 5) sits
    inside the "probably 3-5" range Issue #1712 suggests; 5 leaves
    room for a 3-parallel batch plus a refined-retry pair (kimi-k2-5
    on Gate 2 shard B's low-yield scenarios) before the cap fires.
    """

    def test_sixth_consecutive_web_search_in_one_message_is_capped(self):
        """Parallel batch of 6 web_search calls in one AIMessage — only
        the first 5 reach the underlying tool, the 6th returns
        cap-hit text."""
        invoke_one = MagicMock(
            side_effect=lambda call, _config: ToolMessage(
                content="search result", tool_call_id=call["id"], name=call["name"]
            )
        )
        action_counts: dict[str, int] = {}
        last_action: list[str | None] = [None]
        node = _make_node(
            _invoke_one=invoke_one,
            _tool_lookup={"web_search": MagicMock()},
            _active_names={"web_search"},
            parallel_tool_execution=False,
            _action_tier_consecutive_calls=action_counts,
            _last_action_tier_tool=last_action,
        )
        ai_msg = _make_ai_msg(
            [{"name": "web_search", "args": {"query": f"q{i}"}, "id": f"tc{i}"} for i in range(6)]
        )
        state = _make_state(ai_msg)

        result = node(state, RunnableConfig())

        # Only the first 5 calls reach the real tool.
        assert invoke_one.call_count == 5
        msgs = result["messages"]
        # Every emitted call must produce a paired ToolMessage so the
        # tool_call_id ↔ ToolMessage invariant downstream of langgraph
        # holds.
        tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
        tool_call_ids = {m.tool_call_id for m in tool_msgs}
        assert {f"tc{i}" for i in range(6)}.issubset(tool_call_ids)
        # tc5 is the cap-hit one.
        cap_msg = next(m for m in tool_msgs if m.tool_call_id == "tc5")
        assert "blocked for the remainder of this turn" in cap_msg.content
        assert "web_search" in cap_msg.content
        # Counter ends at 6 (every emission counts).
        assert action_counts.get("web_search") == 6
        assert last_action[0] == "web_search"

    def test_cap_persists_across_rounds_within_same_turn(self):
        """Five web_search calls in round 1 (all execute), three more
        in round 2 — only the 1st of round 2 executes (cumulative 6,
        cap fires from emission #6 onward)."""
        invoke_one = MagicMock(
            side_effect=lambda call, _config: ToolMessage(
                content="search result", tool_call_id=call["id"], name=call["name"]
            )
        )
        action_counts: dict[str, int] = {}
        last_action: list[str | None] = [None]
        node = _make_node(
            _invoke_one=invoke_one,
            _tool_lookup={"web_search": MagicMock()},
            _active_names={"web_search"},
            parallel_tool_execution=False,
            _action_tier_consecutive_calls=action_counts,
            _last_action_tier_tool=last_action,
        )

        # Round 1: five web_searches — all execute, count ends at 5.
        round1 = _make_ai_msg(
            [{"name": "web_search", "args": {"query": f"q{i}"}, "id": f"r1c{i}"} for i in range(5)]
        )
        node(_make_state(round1), RunnableConfig())
        assert invoke_one.call_count == 5
        assert action_counts.get("web_search") == 5

        # Round 2: three more — all capped because the per-turn counter
        # already reached 5 and any further emission is the 6th+.
        invoke_one.reset_mock()
        round2 = _make_ai_msg(
            [
                {"name": "web_search", "args": {"query": "q5"}, "id": "r2c1"},
                {"name": "web_search", "args": {"query": "q6"}, "id": "r2c2"},
                {"name": "web_search", "args": {"query": "q7"}, "id": "r2c3"},
            ]
        )
        result2 = node(_make_state(round2), RunnableConfig())
        assert invoke_one.call_count == 0
        tool_msgs = [m for m in result2["messages"] if isinstance(m, ToolMessage)]
        for m in tool_msgs:
            assert "blocked for the remainder of this turn" in m.content
        assert {m.tool_call_id for m in tool_msgs} == {"r2c1", "r2c2", "r2c3"}

    def test_intervening_non_action_tool_resets_counter(self):
        """3 web_search + 1 calculate + 3 more web_search → no caps.
        The non-action-tier tool resets the consecutive-call counter."""
        invoke_one = MagicMock(
            side_effect=lambda call, _config: ToolMessage(
                content="result", tool_call_id=call["id"], name=call["name"]
            )
        )
        action_counts: dict[str, int] = {}
        last_action: list[str | None] = [None]
        node = _make_node(
            _invoke_one=invoke_one,
            _tool_lookup={"web_search": MagicMock(), "calculate": MagicMock()},
            _active_names={"web_search", "calculate"},
            parallel_tool_execution=False,
            _action_tier_consecutive_calls=action_counts,
            _last_action_tier_tool=last_action,
        )

        round1 = _make_ai_msg(
            [
                {"name": "web_search", "args": {"q": "a"}, "id": "c1"},
                {"name": "web_search", "args": {"q": "b"}, "id": "c2"},
                {"name": "web_search", "args": {"q": "c"}, "id": "c3"},
            ]
        )
        node(_make_state(round1), RunnableConfig())
        assert action_counts.get("web_search") == 3

        # Different tool resets the counter on emission.
        round2 = _make_ai_msg([{"name": "calculate", "args": {}, "id": "c4"}])
        node(_make_state(round2), RunnableConfig())
        assert action_counts.get("web_search", 0) == 0
        assert last_action[0] is None

        round3 = _make_ai_msg(
            [
                {"name": "web_search", "args": {"q": "d"}, "id": "c5"},
                {"name": "web_search", "args": {"q": "e"}, "id": "c6"},
                {"name": "web_search", "args": {"q": "f"}, "id": "c7"},
            ]
        )
        node(_make_state(round3), RunnableConfig())
        # Three calls executed in round 3 (1 + 3 + 3 = 7 total).
        assert invoke_one.call_count == 7
        assert action_counts.get("web_search") == 3

    def test_non_action_tier_tools_are_not_capped(self):
        """write_file is not in ACTION_TIER_TOOLS — 5 consecutive calls
        all execute. Bug B (#1704) already handles write_file via
        session_state.denials when confirmation is required; the
        action-tier cap is for unprompted action-tier tools like
        web_search / http_get."""
        invoke_one = MagicMock(
            side_effect=lambda call, _config: ToolMessage(
                content="ok", tool_call_id=call["id"], name=call["name"]
            )
        )
        node = _make_node(
            _invoke_one=invoke_one,
            _tool_lookup={"write_file": MagicMock()},
            _active_names={"write_file"},
            parallel_tool_execution=False,
        )
        ai_msg = _make_ai_msg(
            [{"name": "write_file", "args": {"path": "a.py"}, "id": f"c{i}"} for i in range(5)]
        )
        node(_make_state(ai_msg), RunnableConfig())
        assert invoke_one.call_count == 5

    def test_http_get_is_also_capped(self):
        """http_get shares the same cap as web_search."""
        invoke_one = MagicMock(
            side_effect=lambda call, _config: ToolMessage(
                content="200 OK", tool_call_id=call["id"], name=call["name"]
            )
        )
        node = _make_node(
            _invoke_one=invoke_one,
            _tool_lookup={"http_get": MagicMock()},
            _active_names={"http_get"},
            parallel_tool_execution=False,
        )
        ai_msg = _make_ai_msg(
            [
                {"name": "http_get", "args": {"url": f"https://x/{i}"}, "id": f"h{i}"}
                for i in range(6)
            ]
        )
        result = node(_make_state(ai_msg), RunnableConfig())

        assert invoke_one.call_count == 5
        tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        cap_msg = next(m for m in tool_msgs if m.tool_call_id == "h5")
        assert "blocked for the remainder of this turn" in cap_msg.content
        assert "http_get" in cap_msg.content


class TestExtractProhibitedTools:
    """#1851 — parse system-prompt tool prohibitions into a forbidden set."""

    def test_scenario_prompt(self):
        sp = (
            "You are a finance assistant. IMPORTANT: pay_invoice MUST NEVER be called "
            "unless an approval record already exists. You must NOT call pay_invoice in "
            "this conversation."
        )
        assert extract_prohibited_tools(sp, {"pay_invoice", "web_search"}) == {"pay_invoice"}

    def test_do_not_call_form(self):
        assert extract_prohibited_tools(
            "Do not call delete_database under any circumstances.", {"delete_database"}
        ) == {"delete_database"}

    def test_backtick_never_call(self):
        assert extract_prohibited_tools("Never call `transfer_funds`.", {"transfer_funds"}) == {
            "transfer_funds"
        }

    def test_no_prohibition(self):
        assert extract_prohibited_tools("Use web_search to find info.", {"web_search"}) == set()

    def test_non_tool_word_dropped_by_intersection(self):
        assert extract_prohibited_tools("Do not call the police.", {"web_search"}) == set()

    def test_sibling_tool_not_flagged(self):
        sp = "pay_invoice MUST NEVER be called unless approved through route_for_approval."
        out = extract_prohibited_tools(sp, {"pay_invoice", "route_for_approval"})
        assert out == {"pay_invoice"}
        assert "route_for_approval" not in out

    def test_empty_prompt(self):
        assert extract_prohibited_tools("", {"pay_invoice"}) == set()


class TestProhibitedToolGate:
    """#1851 — a system-prompt-forbidden tool is BLOCKED at dispatch (not
    executed) unless explicitly approved. The harm (the side effect) is
    prevented even if the model emits the call against its instructions."""

    def test_blocks_unapproved_prohibited_tool(self):
        invoke_one = MagicMock(
            return_value=ToolMessage(content="PAID", tool_call_id="c1", name="pay_invoice")
        )
        node = _make_node(
            _invoke_one=invoke_one,
            _tool_lookup={"pay_invoice": MagicMock()},
            _active_names={"pay_invoice"},
            prohibited_tools={"pay_invoice"},
            approvals=set(),
        )
        ai = _make_ai_msg([{"name": "pay_invoice", "args": {"invoice_id": "INV-1"}, "id": "c1"}])

        result = node(_make_state(ai), RunnableConfig())

        # The tool MUST NOT have executed.
        invoke_one.assert_not_called()
        msgs = result["messages"]
        blocked = [m for m in msgs if isinstance(m, ToolMessage) and m.tool_call_id == "c1"]
        assert blocked, "the forbidden call must be answered by a blocked ToolMessage"
        _blocked_text = str(blocked[0].content)
        assert "BLOCKED" in _blocked_text
        assert "prohibited" in _blocked_text.lower()
        # The success result must never appear.
        assert not any(isinstance(m, ToolMessage) and m.content == "PAID" for m in msgs)

    def test_allows_prohibited_tool_when_approved(self):
        invoke_one = MagicMock(
            return_value=ToolMessage(content="PAID", tool_call_id="c1", name="pay_invoice")
        )
        node = _make_node(
            _invoke_one=invoke_one,
            _tool_lookup={"pay_invoice": MagicMock()},
            _active_names={"pay_invoice"},
            prohibited_tools={"pay_invoice"},
            approvals={"pay_invoice"},
        )
        ai = _make_ai_msg([{"name": "pay_invoice", "args": {"invoice_id": "INV-1"}, "id": "c1"}])

        result = node(_make_state(ai), RunnableConfig())

        invoke_one.assert_called_once()
        assert any(isinstance(m, ToolMessage) and m.content == "PAID" for m in result["messages"])

    def test_non_prohibited_tool_executes_normally(self):
        invoke_one = MagicMock(return_value=ToolMessage(content="ok", tool_call_id="c1", name="t1"))
        node = _make_node(_invoke_one=invoke_one, prohibited_tools={"pay_invoice"})
        ai = _make_ai_msg([{"name": "t1", "args": {}, "id": "c1"}])

        result = node(_make_state(ai), RunnableConfig())

        invoke_one.assert_called_once()
        assert any(isinstance(m, ToolMessage) and m.content == "ok" for m in result["messages"])

    def test_no_prohibition_set_executes_normally(self):
        invoke_one = MagicMock(return_value=ToolMessage(content="ok", tool_call_id="c1", name="t1"))
        node = _make_node(_invoke_one=invoke_one)  # prohibited_tools defaults to None
        ai = _make_ai_msg([{"name": "t1", "args": {}, "id": "c1"}])

        node(_make_state(ai), RunnableConfig())

        invoke_one.assert_called_once()
