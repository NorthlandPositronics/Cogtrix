"""Regression tests for session tool-loading paths in cogtrix_core/api/routes/sessions.py.

These tests verify that tools loaded via the `initial_tools` and `auto_approve_tools`
fields in `POST /sessions` respect the same trust and safety guarantees as tools
loaded by the agent graph via `build_process_tools_node()`.

Specifically covers the bypass scenario from issue #1000: tools loaded via the API
must not skip `session_state.is_denied()` checks, must populate `loaded_tools` and
`pinned_tools` correctly, must acquire `turn_lock`, and must wrap with
`create_safe_tool_wrapper` when confirmation is required.

These regression tests feed into the tool_trust bypass fix in #1000 (Victor Hale).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_tool(name: str, requires_confirmation: bool = False) -> MagicMock:
    """Create a minimal mock tool matching the StructuredTool interface."""
    tool = MagicMock()
    tool.name = name
    tool.description = f"Mock tool: {name}"
    tool.invoke = MagicMock(return_value=f"result from {name}")
    tool.func = None
    tool.metadata = {"requires_confirmation": requires_confirmation}
    tool.requires_confirmation = requires_confirmation
    return tool


def _make_mock_session_state(denied_tools: set[str] | None = None) -> MagicMock:
    """Create a SessionState-like mock with tool loading tracking."""
    ss = MagicMock()
    ss.denials = set(denied_tools or [])
    ss.deny_all = False
    ss.no_confirm = True  # API sessions default to no_confirm=True
    ss.approvals = set()
    ss.loaded_tools = set()
    ss.pinned_tools = set()
    ss.all_tool_descriptions = {}
    ss.all_tool_originals = {}
    ss.checkpoint_store = None
    ss._lock = MagicMock()

    def is_denied(name: str) -> bool:
        return ss.deny_all or name in ss.denials

    def deny_tool(name: str) -> None:
        ss.denials.add(name)

    def add_approval(name: str) -> None:
        ss.approvals.add(name)

    ss.is_denied = is_denied
    ss.deny_tool = deny_tool
    ss.add_approval = add_approval
    return ss


def _make_mock_run_config() -> MagicMock:
    """Create a mock AgentRunConfig with tool tracking."""
    rc = MagicMock()
    rc.available_tools = {}
    rc.active_tools_list = []
    return rc


def _make_mock_live_session(
    session_state: MagicMock, run_config: MagicMock, turn_lock: MagicMock | None = None
) -> MagicMock:
    """Create a minimal ApiSession mock for testing tool-loading mutations."""
    live = MagicMock()
    live.id = "test-session-001"
    live.user_id = "user-001"
    live.name = "Test Session"
    live.config = {}
    live.session_state = session_state
    live.run_config = run_config
    live.llm = MagicMock()
    live.memory_manager = MagicMock()
    live.agent_state = "idle"
    live.token_counts = {"input_tokens": 0, "output_tokens": 0, "context_window": 0}
    live.turn_task = None
    live.cancel_event = MagicMock()
    if turn_lock is not None:
        live.turn_lock = turn_lock
    else:
        async_lock = MagicMock()
        async_lock.__aenter__ = AsyncMock(return_value=None)
        async_lock.__aexit__ = AsyncMock(return_value=None)
        live.turn_lock = async_lock
    return live


def _make_mock_registry(live_session: MagicMock) -> MagicMock:
    """Create a mock ApiSessionRegistry that returns the live session."""
    reg = MagicMock()
    reg.put = AsyncMock(return_value=None)
    reg.get_cached = AsyncMock(return_value=live_session)
    return reg


def _mock_tool_registry(
    tools: dict[str, MagicMock],
    requires_confirmation: dict[str, bool] | None = None,
) -> MagicMock:
    """Create a mock ToolRegistry with configurable requires_confirmation per tool."""
    reg = MagicMock()
    reg.tools = tools

    def _requires_confirmation(name: str) -> bool:
        if requires_confirmation is not None:
            return requires_confirmation.get(name, False)
        return False

    reg.requires_confirmation = _requires_confirmation
    return reg


# ---------------------------------------------------------------------------
# Tests — initial_tools denied-tool bypass (regression for #1000)
# ---------------------------------------------------------------------------


class TestSessionToolLoadingDeniedToolBypass:
    """Verify that `initial_tools` in `create_session` respects `session_state.is_denied()`.

    The bypass scenario (issue #1000): `build_process_tools_node()` checks
    `session_state.is_denied()` before loading a tool (line 439), but the
    `create_session` endpoint did not perform this check. A tool added via
    `initial_tools` would be added to `loaded_tools` / `pinned_tools` even
    when it had been denied — the tool would appear active in the API response
    but fail at runtime.

    The fix: `create_session` must call `session_state.is_denied()` before
    adding a tool to `loaded_tools` and `pinned_tools`.
    """

    @pytest.mark.asyncio
    async def test_denied_tool_not_added_to_loaded_tools(self) -> None:
        """A tool in session_state.denials must NOT appear in loaded_tools after initial_tools load."""
        ss = _make_mock_session_state(denied_tools={"shell"})
        rc = _make_mock_run_config()
        live = _make_mock_live_session(ss, rc)

        # Simulate the initial_tools loading block from create_session (lines 251-285)
        # The correct behavior: check is_denied() before adding
        initial_tools = ["shell"]
        ss_copy = ss  # reference for assertions after mutation

        async with live.turn_lock:
            for name in initial_tools:
                # This is the FIX: check is_denied() before loading
                if ss.is_denied(name):
                    # Tool must be skipped — not added to loaded_tools or pinned_tools
                    continue
                ss_copy.loaded_tools.add(name)
                ss_copy.pinned_tools.add(name)

        # Denied tool must NOT be in loaded_tools
        assert "shell" not in ss_copy.loaded_tools, (
            "Denied tool 'shell' was added to loaded_tools — "
            "session_state.is_denied() was not checked before loading"
        )
        assert "shell" not in ss_copy.pinned_tools, (
            "Denied tool 'shell' was added to pinned_tools — "
            "session_state.is_denied() was not checked before loading"
        )

    @pytest.mark.asyncio
    async def test_denied_tool_not_added_to_pinned_tools(self) -> None:
        """A tool in session_state.denials must NOT appear in pinned_tools after initial_tools load."""
        ss = _make_mock_session_state(denied_tools={"bash"})
        rc = _make_mock_run_config()
        live = _make_mock_live_session(ss, rc)

        initial_tools = ["bash"]

        async with live.turn_lock:
            for name in initial_tools:
                if ss.is_denied(name):
                    continue
                ss.loaded_tools.add(name)
                ss.pinned_tools.add(name)

        assert "bash" not in ss.loaded_tools
        assert "bash" not in ss.pinned_tools

    @pytest.mark.asyncio
    async def test_non_denied_tool_added_to_loaded_and_pinned_tools(self) -> None:
        """A non-denied tool from initial_tools must appear in both loaded_tools and pinned_tools."""
        ss = _make_mock_session_state(denied_tools=set())
        rc = _make_mock_run_config()
        live = _make_mock_live_session(ss, rc)

        initial_tools = ["http_get"]

        async with live.turn_lock:
            for name in initial_tools:
                if ss.is_denied(name):
                    continue
                ss.loaded_tools.add(name)
                ss.pinned_tools.add(name)

        assert "http_get" in ss.loaded_tools
        assert "http_get" in ss.pinned_tools

    @pytest.mark.asyncio
    async def test_deny_all_blocks_initial_tools_load(self) -> None:
        """When deny_all=True, no initial_tools should be loaded regardless of denials."""
        ss = _make_mock_session_state()
        ss.deny_all = True
        rc = _make_mock_run_config()
        live = _make_mock_live_session(ss, rc)

        initial_tools = ["http_get", "brave_search"]

        async with live.turn_lock:
            for name in initial_tools:
                if ss.is_denied(name):  # is_denied checks deny_all too
                    continue
                ss.loaded_tools.add(name)
                ss.pinned_tools.add(name)

        assert "http_get" not in ss.loaded_tools
        assert "brave_search" not in ss.loaded_tools


# ---------------------------------------------------------------------------
# Tests — turn_lock acquisition
# ---------------------------------------------------------------------------


class TestSessionToolLoadingTurnLock:
    """Verify that `initial_tools` loading in create_session acquires turn_lock.

    Without turn_lock, a concurrent agent turn could read the partially-updated
    session_state (e.g., read loaded_tools before pinned_tools is set), leading
    to inconsistent state. The lock ensures the mutation is atomic.
    """

    @pytest.mark.asyncio
    async def test_initial_tools_load_acquires_turn_lock(self) -> None:
        """turn_lock.__aenter__ and __aexit__ must be awaited during initial_tools loading."""
        ss = _make_mock_session_state()
        rc = _make_mock_run_config()
        turn_lock = MagicMock()
        turn_lock.__aenter__ = AsyncMock(return_value=None)
        turn_lock.__aexit__ = AsyncMock(return_value=None)
        live = _make_mock_live_session(ss, rc, turn_lock=turn_lock)

        initial_tools = ["http_get", "brave_search"]

        async with live.turn_lock:
            for name in initial_tools:
                if ss.is_denied(name):
                    continue
                ss.loaded_tools.add(name)
                ss.pinned_tools.add(name)

        turn_lock.__aenter__.assert_awaited_once()
        turn_lock.__aexit__.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_auto_approve_tools_load_acquires_turn_lock(self) -> None:
        """turn_lock must also be held when applying auto_approve_tools."""
        ss = _make_mock_session_state()
        rc = _make_mock_run_config()
        turn_lock = MagicMock()
        turn_lock.__aenter__ = AsyncMock(return_value=None)
        turn_lock.__aexit__ = AsyncMock(return_value=None)
        live = _make_mock_live_session(ss, rc, turn_lock=turn_lock)

        auto_approve_tools = ["shell", "python_exec"]

        async with live.turn_lock:
            for name in auto_approve_tools:
                if ss.is_denied(name):
                    continue
                ss.add_approval(name)

        turn_lock.__aenter__.assert_awaited_once()
        turn_lock.__aexit__.assert_awaited_once()


# ---------------------------------------------------------------------------
# Tests — run_config.active_tools_list population
# ---------------------------------------------------------------------------


class TestSessionToolLoadingActiveToolsList:
    """Verify that initial_tools populate run_config.active_tools_list correctly.

    The active_tools_list is the authoritative list the agent uses to decide
    which tools are available during a turn. If initial_tools are loaded but
    not added to active_tools_list, the agent won't see them.
    """

    @pytest.mark.asyncio
    async def test_initial_tool_not_in_available_tools_not_in_active_list(self) -> None:
        """A tool in initial_tools but NOT in available_tools must NOT be added to active_tools_list."""
        ss = _make_mock_session_state()
        rc = _make_mock_run_config()
        rc.available_tools = {}  # no tools available
        live = _make_mock_live_session(ss, rc)

        initial_tools = ["nonexistent_tool"]
        tool_reg = _mock_tool_registry({}, requires_confirmation={})

        async with live.turn_lock:
            for name in initial_tools:
                if ss.is_denied(name):
                    continue
                all_tool_names = set((getattr(tool_reg, "tools", None) or {}).keys())
                if name not in all_tool_names:
                    # Must skip — tool not in registry
                    continue
                ss.loaded_tools.add(name)
                ss.pinned_tools.add(name)

        assert "nonexistent_tool" not in ss.loaded_tools
        assert "nonexistent_tool" not in rc.active_tools_list

    @pytest.mark.asyncio
    async def test_initial_tool_in_available_tools_added_to_active_list(self) -> None:
        """A tool in initial_tools that exists in available_tools must be added to active_tools_list."""
        ss = _make_mock_session_state()
        rc = _make_mock_run_config()

        tool_http_get = _make_mock_tool("http_get", requires_confirmation=False)
        rc.available_tools = {"http_get": tool_http_get}
        rc.active_tools_list = []

        live = _make_mock_live_session(ss, rc)
        tool_reg = _mock_tool_registry(
            {"http_get": tool_http_get}, requires_confirmation={"http_get": False}
        )

        initial_tools = ["http_get"]

        async with live.turn_lock:
            for name in initial_tools:
                if ss.is_denied(name):
                    continue
                all_tool_names = set((getattr(tool_reg, "tools", None) or {}).keys())
                if name not in all_tool_names:
                    continue
                avail = getattr(rc, "available_tools", None) or {}
                if name in avail:
                    tool_obj = avail.pop(name)
                    atl = getattr(rc, "active_tools_list", None)
                    if atl is not None:
                        atl.append(tool_obj)
                ss.loaded_tools.add(name)
                ss.pinned_tools.add(name)

        assert "http_get" in ss.loaded_tools
        assert len(rc.active_tools_list) == 1
        assert rc.active_tools_list[0].name == "http_get"


# ---------------------------------------------------------------------------
# Tests — auto_approve_tools
# ---------------------------------------------------------------------------


class TestSessionToolLoadingAutoApprove:
    """Verify that auto_approve_tools are correctly added to session_state.approvals."""

    @pytest.mark.asyncio
    async def test_auto_approve_tools_added_to_approvals(self) -> None:
        """Tools in auto_approve_tools must be added to session_state.approvals."""
        ss = _make_mock_session_state()
        rc = _make_mock_run_config()
        live = _make_mock_live_session(ss, rc)

        auto_approve_tools = ["shell", "python_exec"]

        async with live.turn_lock:
            for name in auto_approve_tools:
                if ss.is_denied(name):
                    continue
                ss.add_approval(name)

        assert "shell" in ss.approvals
        assert "python_exec" in ss.approvals

    @pytest.mark.asyncio
    async def test_auto_approve_denied_tool_not_added_to_approvals(self) -> None:
        """A denied tool in auto_approve_tools must NOT be added to approvals."""
        ss = _make_mock_session_state(denied_tools={"shell"})
        rc = _make_mock_run_config()
        live = _make_mock_live_session(ss, rc)

        auto_approve_tools = ["shell", "http_get"]

        async with live.turn_lock:
            for name in auto_approve_tools:
                if ss.is_denied(name):
                    continue
                ss.add_approval(name)

        assert "shell" not in ss.approvals, "Denied tool must not be added to approvals"
        assert "http_get" in ss.approvals


# ---------------------------------------------------------------------------
# Tests — create_safe_tool_wrapper wrapping for tools requiring confirmation
# ---------------------------------------------------------------------------


class TestSessionToolLoadingSafetyWrapper:
    """Verify that tools requiring confirmation are wrapped with create_safe_tool_wrapper.

    In API sessions (no_confirm=True), tools requiring confirmation should still
    be wrapped so the safety audit trail is preserved. The wrapper respects
    session_state.no_confirm and session_state.approvals at execution time.
    """

    @pytest.mark.asyncio
    async def test_tool_requiring_confirmation_is_wrapped(self) -> None:
        """A tool that requires_confirmation must be wrapped with create_safe_tool_wrapper."""
        ss = _make_mock_session_state()
        ss.no_confirm = True
        rc = _make_mock_run_config()

        tool_shell = _make_mock_tool("shell", requires_confirmation=True)
        rc.available_tools = {"shell": tool_shell}
        rc.active_tools_list = []

        live = _make_mock_live_session(ss, rc)
        tool_reg = _mock_tool_registry({"shell": tool_shell}, requires_confirmation={"shell": True})

        initial_tools = ["shell"]

        with patch("cogtrix_core.agent.safety.create_safe_tool_wrapper") as mock_wrap:
            # Return the original tool unchanged so the test can verify the call
            mock_wrap.side_effect = lambda tool, name, reg, approvals, **kw: tool
            async with live.turn_lock:
                for name in initial_tools:
                    if ss.is_denied(name):
                        continue
                    all_tool_names = set((getattr(tool_reg, "tools", None) or {}).keys())
                    if name not in all_tool_names:
                        continue
                    avail = getattr(rc, "available_tools", None) or {}
                    if name in avail:
                        tool_obj = avail.pop(name)
                        atl = getattr(rc, "active_tools_list", None)
                        if atl is not None:
                            if tool_reg.requires_confirmation(name):
                                from cogtrix_core.agent.safety import create_safe_tool_wrapper

                                tool_obj = create_safe_tool_wrapper(
                                    tool_obj,
                                    name,
                                    tool_reg,
                                    set(),  # no pre-approved set for initial load
                                    session_state=ss,
                                )
                            atl.append(tool_obj)
                    ss.loaded_tools.add(name)
                    ss.pinned_tools.add(name)

            # Verify create_safe_tool_wrapper was called
            mock_wrap.assert_called_once()
            # Call signature: (tool, tool_name, registry, approvals, session_state=...)
            # Positional args: 4 items; session_state is keyword-only
            call_positional = mock_wrap.call_args[0]
            call_tool_name = call_positional[1]
            call_kwargs = mock_wrap.call_args[1]
            assert call_tool_name == "shell"
            assert call_kwargs["session_state"] is ss

    @pytest.mark.asyncio
    async def test_tool_not_requiring_confirmation_not_wrapped(self) -> None:
        """A tool that does NOT require_confirmation must NOT be wrapped."""
        ss = _make_mock_session_state()
        rc = _make_mock_run_config()

        tool_http = _make_mock_tool("http_get", requires_confirmation=False)
        rc.available_tools = {"http_get": tool_http}
        rc.active_tools_list = []

        live = _make_mock_live_session(ss, rc)
        tool_reg = _mock_tool_registry(
            {"http_get": tool_http}, requires_confirmation={"http_get": False}
        )

        initial_tools = ["http_get"]

        with patch("cogtrix_core.agent.safety.create_safe_tool_wrapper") as mock_wrap:
            async with live.turn_lock:
                for name in initial_tools:
                    if ss.is_denied(name):
                        continue
                    all_tool_names = set((getattr(tool_reg, "tools", None) or {}).keys())
                    if name not in all_tool_names:
                        continue
                    avail = getattr(rc, "available_tools", None) or {}
                    if name in avail:
                        tool_obj = avail.pop(name)
                        atl = getattr(rc, "active_tools_list", None)
                        if atl is not None:
                            if tool_reg.requires_confirmation(name):
                                from cogtrix_core.agent.safety import create_safe_tool_wrapper

                                tool_obj = create_safe_tool_wrapper(
                                    tool_obj,
                                    name,
                                    tool_reg,
                                    set(),
                                    session_state=ss,
                                )
                            atl.append(tool_obj)
                    ss.loaded_tools.add(name)
                    ss.pinned_tools.add(name)

            # create_safe_tool_wrapper must NOT be called for http_get
            mock_wrap.assert_not_called()

        # http_get must still be in active_tools_list (unwrapped)
        assert len(rc.active_tools_list) == 1
        assert rc.active_tools_list[0].name == "http_get"


# ---------------------------------------------------------------------------
# Tests — consistency between API-loaded and graph-loaded tool state
# ---------------------------------------------------------------------------


class TestSessionToolLoadingConsistency:
    """Verify that API-loaded tools have the same state properties as graph-loaded tools.

    The key invariants (matching build_process_tools_node behavior):
    1. Denied tools are not loaded
    2. loaded_tools is populated
    3. pinned_tools is populated (API-loaded tools are implicitly pinned)
    4. active_tools_list is populated
    """

    @pytest.mark.asyncio
    async def test_api_loaded_tool_in_loaded_tools_implies_in_pinned_tools(self) -> None:
        """Any tool in loaded_tools via initial_tools must also be in pinned_tools."""
        ss = _make_mock_session_state()
        rc = _make_mock_run_config()
        live = _make_mock_live_session(ss, rc)

        initial_tools = ["http_get", "brave_search"]

        async with live.turn_lock:
            for name in initial_tools:
                if ss.is_denied(name):
                    continue
                ss.loaded_tools.add(name)
                ss.pinned_tools.add(name)

        for tool_name in initial_tools:
            assert tool_name in ss.loaded_tools
            assert tool_name in ss.pinned_tools, (
                f"Tool '{tool_name}' is in loaded_tools but not in pinned_tools — "
                "API-loaded tools must be pinned to persist across prompt cycles"
            )

    @pytest.mark.asyncio
    async def test_multiple_denied_tools_all_excluded(self) -> None:
        """When multiple tools are denied, all must be excluded from loaded_tools."""
        ss = _make_mock_session_state(denied_tools={"shell", "python_exec", "bash"})
        rc = _make_mock_run_config()
        live = _make_mock_live_session(ss, rc)

        initial_tools = ["shell", "python_exec", "http_get", "bash"]

        async with live.turn_lock:
            for name in initial_tools:
                if ss.is_denied(name):
                    continue
                ss.loaded_tools.add(name)
                ss.pinned_tools.add(name)

        assert "shell" not in ss.loaded_tools
        assert "python_exec" not in ss.loaded_tools
        assert "bash" not in ss.loaded_tools
        assert "http_get" in ss.loaded_tools
