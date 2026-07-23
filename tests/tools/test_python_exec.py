"""Tests for python_exec: session isolation and sandbox security."""

import pytest

from src.tools.python_exec import (
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
    set_session,
)


class TestSessionIsolation:
    """Verify that different session_ids produce fully isolated state."""

    def setup_method(self) -> None:
        """Clear all session state before each test."""
        _session_states.clear()

    def test_variables_are_isolated_between_sessions(self) -> None:
        """Variables set in session A must not appear in session B."""
        execute_python("x = 42", session_id="session_a")
        execute_python("y = 99", session_id="session_b")

        ctx_a = get_context("session_a")
        ctx_b = get_context("session_b")

        assert ctx_a.get("x") == 42
        assert "x" not in ctx_b
        assert ctx_b.get("y") == 99
        assert "y" not in ctx_a

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
        """When session_id is omitted, _get_session_state falls back to 'default', not global."""
        set_session("some_other_session")

        state = _get_session_state()
        assert "some_other_session" not in _session_states or state is not _session_states.get(
            "some_other_session"
        )
        assert _session_states.get("default") is state

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

    def test_none_maps_to_default_key(self) -> None:
        state = _get_session_state(None)
        assert "default" in _session_states
        assert _session_states["default"] is state

    def test_explicit_default_string_maps_to_default_key(self) -> None:
        state = _get_session_state("default")
        assert _session_states["default"] is state

    def test_named_session_creates_distinct_state(self) -> None:
        default_state = _get_session_state(None)
        named_state = _get_session_state("chat_99")
        assert default_state is not named_state

    def test_global_set_session_does_not_affect_default_fallback(self) -> None:
        """Changing _current_session via set_session must not redirect _get_session_state(None)."""
        set_session("injected")
        state = _get_session_state(None)
        assert "default" in _session_states
        assert _session_states["default"] is state


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
        code = "attr = '__' + 'gl' + 'obals__'\n" "def f(): pass\n" "x = getattr(f, attr)\n"
        result = _execute_code_internal(code, {})
        assert not result["success"]
        assert "blocked in sandbox" in result.get("error", "")

    def test_literal_subclasses_still_blocked_by_pattern(self):
        is_safe, msg = _check_code_safety("[].__subclasses__()")
        assert not is_safe

    def test_setattr_escape_blocked(self):
        code = "attr = '__' + 'class' + '__'\n" "setattr([], attr, int)\n"
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

    def test_eviction_occurs_when_exceeding_max_sessions(self, monkeypatch) -> None:
        import src.tools.python_exec as mod

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
        import src.tools.python_exec as mod

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
