"""Tests for python_exec: session isolation and sandbox security."""

import pytest

from cogtrix_core.tools.python_exec import (
    _MAX_SESSIONS,
    DANGEROUS_ATTRS,
    PythonExecInput,
    _check_ast_imports,
    _check_code_safety,
    _execute_code_internal,
    _get_session_state,
    _safe_getattr,
    _safe_setattr,
    _session_states,
    clear_context,
    execute_python,
    get_context,
    get_history,
    reset_default_session,
    set_session,
)


class TestSessionIsolation:
    """Verify that different session_ids produce fully isolated state."""

    def setup_method(self) -> None:
        """Clear all session state before each test."""
        _session_states.clear()
        reset_default_session()

    def test_variable_overwrite_does_not_cross_sessions(self) -> None:
        """Overwriting a variable in one session must not affect another."""
        execute_python("val = 1", session_id="session_a")
        execute_python("val = 2", session_id="session_b")
        execute_python("val = 10", session_id="session_a")

        assert get_context("session_a").get("val") == 10
        assert get_context("session_b").get("val") == 2

    def test_history_is_isolated_between_sessions(self) -> None:
        """Execution history must be scoped per session_id."""
        execute_python("1 + 1", session_id="hist_a")
        execute_python("2 + 2", session_id="hist_b")
        execute_python("3 + 3", session_id="hist_b")

        hist_a = get_history("hist_a")
        hist_b = get_history("hist_b")

        assert len(hist_a) == 1
        assert len(hist_b) == 2

    def test_default_session_does_not_use_global(self) -> None:
        """When session_id is omitted, _get_session_state uses an auto-generated UUID, not global."""
        set_session("some_other_session")

        state = _get_session_state()
        assert "some_other_session" not in _session_states or state is not _session_states.get(
            "some_other_session"
        )
        assert _session_states.get("default") is not state

    def test_clear_context_only_affects_target_session(self) -> None:
        """clear_context for one session_id must leave other sessions intact."""
        execute_python("z = 7", session_id="keep")
        execute_python("z = 7", session_id="wipe")

        clear_context("wipe")

        assert get_context("keep").get("z") == 7
        assert get_context("wipe") == {}

    def test_execute_python_uses_session_id_kwarg(self) -> None:
        """execute_python must route state through the provided session_id."""
        execute_python("n = 55", session_id="explicit")
        result = execute_python("%vars", session_id="explicit")

        assert "n" in result

    def test_execute_python_different_sessions_do_not_share_context(self) -> None:
        """Two concurrent-style calls with different session_ids stay isolated."""
        execute_python("shared_name = 'alice'", session_id="user_alice")
        execute_python("shared_name = 'bob'", session_id="user_bob")

        output_alice = execute_python("shared_name", session_id="user_alice")
        output_bob = execute_python("shared_name", session_id="user_bob")

        assert "alice" in output_alice
        assert "bob" in output_bob


class TestGetSessionStateDefault:
    """Unit tests for _get_session_state fallback behavior."""

    def setup_method(self) -> None:
        _session_states.clear()
        reset_default_session()

    def test_none_maps_to_auto_generated_key(self) -> None:
        state = _get_session_state(None)
        assert "default" not in _session_states
        assert state in _session_states.values()

    def test_explicit_default_string_maps_to_auto_generated_key(self) -> None:
        state = _get_session_state("default")
        assert "default" not in _session_states
        assert state in _session_states.values()

    def test_named_session_creates_distinct_state(self) -> None:
        default_state = _get_session_state(None)
        named_state = _get_session_state("chat_99")
        assert default_state is not named_state

    def test_global_set_session_does_not_affect_default_fallback(self) -> None:
        """Changing _current_session via set_session must not redirect _get_session_state(None)."""
        set_session("injected")
        state = _get_session_state(None)
        assert "default" not in _session_states
        assert state in _session_states.values()


class TestPythonExecInputSchema:
    """Verify that PythonExecInput exposes the session_id field."""

    def test_schema_has_session_id_field(self) -> None:
        schema = PythonExecInput.model_fields
        assert "session_id" in schema

    def test_session_id_defaults_to_default(self) -> None:
        inp = PythonExecInput(code="1+1")
        assert inp.session_id == "default"

    def test_session_id_can_be_set(self) -> None:
        inp = PythonExecInput(code="1+1", session_id="my_chat")
        assert inp.session_id == "my_chat"


class TestSafeGetattr:
    def test_allows_safe_attribute(self):
        result = _safe_getattr([1, 2, 3], "__len__")
        assert callable(result)

    def test_blocks_dangerous_attr(self):
        with pytest.raises(AttributeError, match="blocked in sandbox"):
            _safe_getattr([], "__subclasses__")

    def test_blocks_globals(self):
        def f():
            pass

        with pytest.raises(AttributeError, match="blocked in sandbox"):
            _safe_getattr(f, "__globals__")

    def test_blocks_mro(self):
        with pytest.raises(AttributeError, match="blocked in sandbox"):
            _safe_getattr(list, "__mro__")

    def test_blocks_class(self):
        with pytest.raises(AttributeError, match="blocked in sandbox"):
            _safe_getattr([], "__class__")

    def test_default_value_returned_for_missing(self):
        result = _safe_getattr([], "nonexistent_attr_xyz", "default")
        assert result == "default"

    def test_raises_attribute_error_for_missing_without_default(self):
        with pytest.raises(AttributeError):
            _safe_getattr([], "nonexistent_attr_xyz")

    def test_non_string_name_raises_type_error(self):
        with pytest.raises(TypeError):
            _safe_getattr([], 42)  # type: ignore[arg-type]

    def test_all_dangerous_attrs_blocked(self):
        obj = []
        for attr in DANGEROUS_ATTRS:
            with pytest.raises(AttributeError, match="blocked in sandbox"):
                _safe_getattr(obj, attr)


class TestSafeSetattr:
    def test_allows_safe_attribute(self):
        class Obj:
            pass

        o = Obj()
        _safe_setattr(o, "x", 42)
        assert o.x == 42  # type: ignore[attr-defined]

    def test_blocks_dangerous_attr(self):
        class Obj:
            pass

        o = Obj()
        with pytest.raises(AttributeError, match="blocked in sandbox"):
            _safe_setattr(o, "__class__", int)

    def test_non_string_name_raises_type_error(self):
        class Obj:
            pass

        o = Obj()
        with pytest.raises(TypeError):
            _safe_setattr(o, 42, "value")  # type: ignore[arg-type]


class TestSandboxEscapeViaRuntimeAttr:
    """Tests that verify the runtime getattr bypass is closed."""

    def test_chr_constructed_subclasses_blocked(self):
        code = (
            "sc_name = ''.join([chr(95), chr(95), 's', 'u', 'b', 'c', 'l', 'a', 's', 's', 'e', 's', chr(95), chr(95)])\n"
            "x = getattr(type([]), sc_name)\n"
        )
        result = _execute_code_internal(code, {})
        assert not result["success"]
        assert "blocked in sandbox" in result.get("error", "")

    def test_join_constructed_globals_blocked(self):
        code = "attr = '__' + 'gl' + 'obals__'\ndef f(): pass\nx = getattr(f, attr)\n"
        result = _execute_code_internal(code, {})
        assert not result["success"]
        assert "blocked in sandbox" in result.get("error", "")

    def test_literal_subclasses_still_blocked_by_pattern(self):
        is_safe, msg = _check_code_safety("[].__subclasses__()")
        assert not is_safe

    def test_setattr_escape_blocked(self):
        code = "attr = '__' + 'class' + '__'\nsetattr([], attr, int)\n"
        result = _execute_code_internal(code, {})
        assert not result["success"]
        assert "blocked in sandbox" in result.get("error", "")


class TestLegitimateGetattr:
    """Tests that safe getattr usage still works correctly."""

    def test_getattr_on_list_len(self):
        code = "x = getattr([1, 2, 3], '__len__')()\n"
        result = _execute_code_internal(code, {})
        assert result["success"]

    def test_getattr_with_default(self):
        code = "x = getattr([], 'no_such_attr', 'fallback')\n"
        result = _execute_code_internal(code, {})
        assert result["success"]

    def test_getattr_normal_attr(self):
        code = "import math\nx = getattr(math, 'pi')\n"
        result = _execute_code_internal(code, {})
        assert result["success"]

    def test_setattr_normal_usage(self):
        code = "import math\nsetattr(math, 'custom_val', 42)\nresult = math.custom_val\n"
        result = _execute_code_internal(code, {})
        assert result["success"]


class TestLoopLimiterRejectsOnFailure:
    """BUG-1240: AST transformation failure must reject code, not execute without limits."""

    def test_add_loop_limits_failure_rejects_code(self, monkeypatch):
        """If _add_loop_limits raises, the code must be rejected."""

        def _boom(_code):
            raise RuntimeError("AST explosion")

        monkeypatch.setattr("cogtrix_core.tools.python_exec._add_loop_limits", _boom)

        result = _execute_code_internal("x = 1 + 1", {})

        assert result["success"] is False
        assert "could not be safely bounded" in result["error"]

    def test_add_loop_limits_failure_logs_warning(self, monkeypatch, caplog):
        """If _add_loop_limits raises, a warning must be logged."""

        def _boom(_code):
            raise RuntimeError("AST explosion")

        monkeypatch.setattr("cogtrix_core.tools.python_exec._add_loop_limits", _boom)

        with caplog.at_level("WARNING", logger="cogtrix.python_exec"):
            _execute_code_internal("x = 1 + 1", {})

        assert "Loop limiter AST transformation failed" in caplog.text
        assert "AST explosion" in caplog.text


class TestCodeSafetyChecks:
    def test_blocks_eval(self):
        is_safe, _ = _check_code_safety("eval('1+1')")
        assert not is_safe

    def test_blocks_exec(self):
        is_safe, _ = _check_code_safety("exec('x=1')")
        assert not is_safe

    def test_blocks_import_os(self):
        is_safe, _ = _check_code_safety("import os")
        assert not is_safe

    def test_allows_math_import(self):
        is_safe, _ = _check_code_safety("import math")
        assert is_safe

    def test_blocks_dunder_class_literal(self):
        is_safe, _ = _check_code_safety("x = [].__class__")
        assert not is_safe

    def test_blocks_getattr_with_dangerous_literal(self):
        is_safe, _ = _check_code_safety("getattr([], '__subclasses__')")
        assert not is_safe


class TestAstImportChecks:
    """BUG-025: AST-based import detection avoids substring false positives."""

    def test_import_os_is_blocked(self):
        is_safe, msg = _check_code_safety("import os")
        assert not is_safe
        assert "os" in msg

    def test_from_subprocess_import_run_is_blocked(self):
        is_safe, msg = _check_code_safety("from subprocess import run")
        assert not is_safe
        assert "subprocess" in msg

    def test_pathlib_as_string_literal_is_allowed(self):
        is_safe, _ = _check_code_safety('x = "pathlib"')
        assert is_safe

    def test_shutil_as_variable_value_is_allowed(self):
        is_safe, _ = _check_code_safety('x = "shutil"')
        assert is_safe

    def test_pathlib_in_variable_name_is_allowed(self):
        is_safe, _ = _check_code_safety("my_pathlib_wrapper = 42")
        assert is_safe

    def test_eval_still_blocked_as_call_pattern(self):
        is_safe, _ = _check_code_safety("eval('1+1')")
        assert not is_safe

    def test_exec_still_blocked_as_call_pattern(self):
        is_safe, _ = _check_code_safety("exec('x=1')")
        assert not is_safe

    def test_check_ast_imports_blocks_os(self):
        import ast as _ast

        tree = _ast.parse("import os")
        is_safe, msg = _check_ast_imports(tree)
        assert not is_safe
        assert "os" in msg

    def test_check_ast_imports_allows_math(self):
        import ast as _ast

        tree = _ast.parse("import math")
        is_safe, _ = _check_ast_imports(tree)
        assert is_safe

    def test_check_ast_imports_blocks_from_sys_import_path(self):
        import ast as _ast

        tree = _ast.parse("from sys import path")
        is_safe, msg = _check_ast_imports(tree)
        assert not is_safe
        assert "sys" in msg


class TestSessionLruEviction:
    """BUG-026: _session_states uses OrderedDict with LRU eviction."""

    def setup_method(self) -> None:
        _session_states.clear()
        reset_default_session()

    def test_eviction_occurs_when_exceeding_max_sessions(self, monkeypatch) -> None:
        import cogtrix_core.tools.python_exec as mod

        monkeypatch.setattr(mod, "_MAX_SESSIONS", 3)

        mod._get_session_state("s1")
        mod._get_session_state("s2")
        mod._get_session_state("s3")
        assert len(_session_states) == 3
        assert "s1" in _session_states

        mod._get_session_state("s4")
        assert len(_session_states) == 3
        assert "s1" not in _session_states
        assert "s4" in _session_states

    def test_accessing_existing_session_moves_it_to_end(self, monkeypatch) -> None:
        import cogtrix_core.tools.python_exec as mod

        monkeypatch.setattr(mod, "_MAX_SESSIONS", 3)

        mod._get_session_state("s1")
        mod._get_session_state("s2")
        mod._get_session_state("s3")

        # Access s1 to promote it to most-recently-used
        mod._get_session_state("s1")

        # Adding s4 should evict s2 (now oldest), not s1
        mod._get_session_state("s4")
        assert len(_session_states) == 3
        assert "s1" in _session_states
        assert "s2" not in _session_states
        assert "s4" in _session_states

    def test_max_sessions_constant_exists(self) -> None:
        assert _MAX_SESSIONS == 1000


class TestDefaultSessionContextIsolation:
    """BUG-1073: Default session must be isolated across concurrent contexts."""

    def setup_method(self) -> None:
        _session_states.clear()
        reset_default_session()

    def test_same_thread_reuses_auto_session(self) -> None:
        """Single-threaded callers get persistence via the cached auto session."""
        state_a = _get_session_state(None)
        state_b = _get_session_state("default")
        state_c = _get_session_state(None)
        assert state_a is state_b is state_c

    def test_different_threads_get_distinct_auto_sessions(self) -> None:
        """Concurrent threads must not share the auto-generated default session."""
        import concurrent.futures

        def _get_sid() -> str:
            reset_default_session()
            state = _get_session_state(None)
            # Return the key that maps to this state
            for k, v in _session_states.items():
                if v is state:
                    return k
            raise RuntimeError("state not found")

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            future_a = pool.submit(_get_sid)
            future_b = pool.submit(_get_sid)
            sid_a = future_a.result()
            sid_b = future_b.result()

        assert sid_a != sid_b

    def test_explicit_named_session_not_affected_by_auto_sessions(self) -> None:
        """Named sessions remain independent of auto-generated default sessions."""
        auto_state = _get_session_state(None)
        named_state = _get_session_state("named_session")
        assert auto_state is not named_state

    def test_cross_session_data_leakage_blocked(self) -> None:
        """Variables set in one thread's default session must not leak to another."""
        import concurrent.futures

        def _thread_a() -> str:
            reset_default_session()
            return execute_python("leak_test_var = 42")

        def _thread_b() -> str:
            reset_default_session()
            return execute_python("result = leak_test_var * 2")

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            future_a = pool.submit(_thread_a)
            future_b = pool.submit(_thread_b)
            result_a = future_a.result(timeout=5)
            result_b = future_b.result(timeout=5)

        # thread_a should succeed
        assert "Error" not in result_a
        # thread_b should fail with NameError because leak_test_var is isolated
        assert "Name Error" in result_b

    def test_reset_default_session_creates_fresh_uuid(self) -> None:
        """After reset_default_session(), the next lookup must use a new UUID."""
        state_first = _get_session_state(None)
        reset_default_session()
        state_second = _get_session_state(None)
        assert state_first is not state_second


class TestToolConfigDescription:
    """Regression: TOOL_CONFIG description must stay in sync with module state."""

    def test_configure_datascience_modules_updates_description(self, monkeypatch) -> None:
        """configure_datascience_modules(True) must refresh TOOL_CONFIG['description']."""
        import cogtrix_core.tools.python_exec as mod

        # Enable datascience modules — only actually-importable ones are listed
        mod.configure_datascience_modules(True)

        updated = mod.TOOL_CONFIG["description"]
        # numpy is available in the test environment; pandas/scipy may not be
        assert "numpy" in updated

        # Restore state
        monkeypatch.setattr(mod, "_AVAILABLE_OPTIONAL", {})
        mod.configure_datascience_modules(False)


# ── Issue #926: time.sleep DoS cap ────────────────────────────────────────────


class TestTimeSleepDoSCap:
    """Issue #926: ``time.sleep`` in the python_exec sandbox could
    occupy a worker slot for the full execution timeout (up to 60s)
    without doing useful work.  In concurrent sessions a few
    sleep-heavy calls drained the multiprocessing pool.

    The fix shims ``time`` with a SimpleNamespace whose ``sleep``
    caps the requested duration at :data:`_TIME_SLEEP_CAP_SECONDS`
    (5s).  Other read-only time functions (``time``, ``monotonic``,
    ``ctime``, ...) are passed through unchanged.

    These tests pin: cap behaviour, identity of preloaded vs imported
    shim, and that legitimate short-sleep usage still works.
    """

    def test_sleep_is_capped_at_five_seconds(self) -> None:
        """Asking for a 999-second sleep must complete in ≤ 5 s + slack."""
        import time

        from cogtrix_core.tools.python_exec import execute_python

        start = time.monotonic()
        # ``execute_python`` returns once the sandboxed call returns;
        # if the cap works the call returns in ~5s, not 60s.
        result = execute_python("import time; time.sleep(999); 'reached'", timeout=20)
        elapsed = time.monotonic() - start

        assert "reached" in result, f"sleep(999) did not return cleanly; result={result[:200]!r}"
        # Cap is 5s; allow generous slack for subprocess spawn + IPC.
        assert elapsed < 15.0, (
            f"time.sleep(999) took {elapsed:.1f}s — the #926 cap is missing "
            "or set too high.  Worker-pool DoS surface is open."
        )

    def test_short_sleep_still_works_uncapped(self) -> None:
        """A 0.1-second sleep must NOT be modified by the cap — only
        sleeps above the cap are clamped.  This protects legitimate
        rate-limit testing and small animation loops.
        """
        import time

        from cogtrix_core.tools.python_exec import execute_python

        start = time.monotonic()
        result = execute_python("import time; time.sleep(0.1); 'short_sleep_ok'")
        elapsed = time.monotonic() - start

        assert "short_sleep_ok" in result
        # Should take ≥ 0.1s (the sleep ran) but well under the 5s cap.
        assert elapsed >= 0.1
        assert elapsed < 5.0

    def test_time_time_still_returns_real_timestamp(self) -> None:
        """``time.time()`` must still return the real Unix timestamp.
        Capping ``sleep`` should not regress other time functions."""
        from cogtrix_core.tools.python_exec import execute_python

        result = execute_python("import time; t = time.time(); 'got=' + str(int(t))")
        assert "got=" in result
        # The integer captured should look like a recent Unix epoch
        # (post-2025-01-01 = 1735689600).  Pinning a recent year keeps
        # this test from drifting in the far future without rewriting.
        import re as _re

        m = _re.search(r"got=(\d+)", result)
        assert m, f"timestamp not captured in result: {result[:200]!r}"
        assert int(m.group(1)) > 1735689600, (
            "time.time() returned a stale or fake value — the shim should "
            "delegate to the real clock"
        )

    def test_negative_sleep_rejected(self) -> None:
        """Negative sleeps are rejected by the shim, matching the
        contract of the real ``time.sleep``."""
        from cogtrix_core.tools.python_exec import execute_python

        result = execute_python("import time; time.sleep(-1)")
        # The error surfaces as a Value Error from the sandboxed run.
        assert "Error" in result
        assert "non-negative" in result.lower() or "negative" in result.lower()

    def test_sleep_via_preloaded_module_is_capped(self) -> None:
        """Even without an explicit ``import time``, the preloaded
        ``time`` global must be the capped shim — otherwise old code
        relying on the implicit module would bypass the cap.
        """
        import time

        from cogtrix_core.tools.python_exec import execute_python

        start = time.monotonic()
        # No ``import time`` — uses the preloaded global.
        result = execute_python("time.sleep(999); 'preloaded_ok'", timeout=20)
        elapsed = time.monotonic() - start

        assert (
            "preloaded_ok" in result
        ), f"preloaded time.sleep did not return; result={result[:200]!r}"
        assert elapsed < 15.0, (
            f"preloaded time.sleep(999) took {elapsed:.1f}s — the cap "
            "applies to import but not to the preloaded module reference"
        )

    def test_safe_time_module_helper_omits_dangerous_attrs(self) -> None:
        """The shim must NOT expose attributes like ``__loader__`` or
        ``__spec__`` that could be used to reach back to the real
        module via reflection.
        """
        from cogtrix_core.tools.python_exec import _make_safe_time_module

        shim = _make_safe_time_module()
        # SimpleNamespace lacks module dunders by construction; double-
        # check the obvious escape vectors are absent.
        for attr in ("__loader__", "__spec__", "__file__", "__builtins__"):
            assert not hasattr(
                shim, attr
            ), f"safe-time shim exposes {attr} — possible sandbox escape"
        # Capped sleep must be present and callable.
        assert callable(shim.sleep)

    def test_sleep_cap_constant_is_exported(self) -> None:
        """The cap value is a module-level constant so future tuning
        is a one-line edit and operators can audit it.
        """
        from cogtrix_core.tools.python_exec import _TIME_SLEEP_CAP_SECONDS

        assert isinstance(_TIME_SLEEP_CAP_SECONDS, int | float)
        assert _TIME_SLEEP_CAP_SECONDS > 0
        assert _TIME_SLEEP_CAP_SECONDS <= 30, (
            "Cap > 30s defeats the purpose: the worker pool's per-call "
            "timeout is 60s, so a 30s+ cap still drains slots significantly."
        )
