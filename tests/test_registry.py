"""Tests for cogtrix_core/registry.py — BUG-047 fallback schema matching."""

from __future__ import annotations

import types

from pydantic import BaseModel

from cogtrix_core.registry import _func_to_schema_name

# ── _func_to_schema_name ──────────────────────────────────────────────────────


class TestFuncToSchemaName:
    def test_simple_name(self):
        assert _func_to_schema_name("search") == "SearchInput"

    def test_snake_case_two_words(self):
        assert _func_to_schema_name("foo_bar") == "FooBarInput"

    def test_snake_case_three_words(self):
        assert _func_to_schema_name("read_the_file") == "ReadTheFileInput"

    def test_single_word(self):
        assert _func_to_schema_name("query") == "QueryInput"

    def test_all_lowercase(self):
        assert _func_to_schema_name("my_tool_func") == "MyToolFuncInput"

    def test_already_camel_single(self):
        # Single word with mixed case — each "word" from split("_") is capitalized
        assert _func_to_schema_name("mytool") == "MytoolInput"

    def test_empty_string(self):
        assert _func_to_schema_name("") == "Input"

    def test_trailing_underscore(self):
        # Trailing underscore produces an empty word, which capitalizes to ""
        result = _func_to_schema_name("foo_")
        assert result.endswith("Input")
        assert result.startswith("Foo")


# ── Fallback discovery by name convention ─────────────────────────────────────


def _make_module_with_schemas(
    func_name: str,
    schema_name: str,
    *,
    has_tool_config: bool = False,
) -> types.ModuleType:
    """Build a minimal fake module with one function and one *Input schema."""
    mod = types.ModuleType(f"fake_{func_name}")

    class _Schema(BaseModel):
        query: str

    _Schema.__name__ = schema_name
    _Schema.__qualname__ = schema_name
    setattr(mod, schema_name, _Schema)

    def _fn(**kwargs):  # type: ignore[return]
        """Do something useful."""

    _fn.__name__ = func_name
    _fn.__qualname__ = func_name
    setattr(mod, func_name, _fn)

    if has_tool_config:
        mod.TOOL_CONFIG = {  # type: ignore[attr-defined]
            "name": func_name,
            "description": "configured",
            "input_schema": _Schema,
            "requires_confirmation": False,
        }

    return mod


class TestFallbackDiscovery:
    """BUG-047: fallback pairs function with schema by name convention."""

    def _extract(self, module: types.ModuleType) -> list[tuple]:
        from cogtrix_core.registry import ToolRegistry

        registry = ToolRegistry.__new__(ToolRegistry)
        registry.tools = {}
        registry.tool_metadata = {}
        return registry.extract_tool_functions(module)

    def test_named_match_found(self):
        """foo_bar should be paired with FooBarInput."""
        mod = _make_module_with_schemas("foo_bar", "FooBarInput")
        results = self._extract(mod)
        assert len(results) == 1
        func, config = results[0]
        assert func.__name__ == "foo_bar"
        assert config["input_schema"].__name__ == "FooBarInput"

    def test_no_match_schema_name_mismatch_single_schema_skips_function(self):
        """When the sole schema name does not match the function, the function is skipped."""
        mod = _make_module_with_schemas("search_web", "SomeOtherInput")
        results = self._extract(mod)
        assert len(results) == 0

    def test_no_match_multiple_schemas_skips_function(self):
        """With multiple schemas and no name match, the function is skipped."""
        mod = _make_module_with_schemas("search_web", "SomeOtherInput")

        class _AnotherSchema(BaseModel):
            x: int

        _AnotherSchema.__name__ = "YetAnotherInput"
        mod.YetAnotherInput = _AnotherSchema

        results = self._extract(mod)
        assert all(config["name"] != "search_web" for _, config in results)

    def test_tool_config_takes_precedence_over_fallback(self):
        """Modules with TOOL_CONFIG should not use fallback at all."""
        mod = _make_module_with_schemas("foo_bar", "FooBarInput", has_tool_config=True)
        results = self._extract(mod)
        assert len(results) == 1
        _, config = results[0]
        assert config["description"] == "configured"

    def test_private_functions_skipped_in_fallback(self):
        """Functions starting with _ must not appear in fallback results."""
        mod = _make_module_with_schemas("foo_bar", "FooBarInput")

        def _private(**kwargs):  # type: ignore[return]
            """Private helper."""

        _private.__name__ = "_private"
        mod._private = _private

        results = self._extract(mod)
        names = [config["name"] for _, config in results]
        assert "_private" not in names

    def test_functions_without_docstring_skipped_in_fallback(self):
        """Functions without docstrings are skipped in fallback discovery."""
        mod = _make_module_with_schemas("foo_bar", "FooBarInput")

        def no_doc(**kwargs):  # type: ignore[return]
            pass  # no docstring

        no_doc.__name__ = "no_doc"
        mod.no_doc = no_doc

        results = self._extract(mod)
        names = [config["name"] for _, config in results]
        assert "no_doc" not in names

    def test_description_taken_from_first_docstring_paragraph(self):
        """The description field uses only the first paragraph of the docstring."""
        mod = _make_module_with_schemas("foo_bar", "FooBarInput")

        def foo_bar(**kwargs):  # type: ignore[return]
            """First paragraph.

            Second paragraph with more detail.
            """

        foo_bar.__name__ = "foo_bar"
        mod.foo_bar = foo_bar

        results = self._extract(mod)
        assert len(results) == 1
        _, config = results[0]
        assert config["description"] == "First paragraph."


# ── Discovery hygiene (cogtrix52 startup-noise) ──────────────────────────────


class TestDiscoverToolModulesHygiene:
    """cogtrix52.log surfaced ~20 startup WARNINGs of the form
    ``Module src.tools._foo: no tools resolved from TOOL_CONFIG/TOOL_CONFIGS``.

    Two separate sources of that noise:

    1. ``_``-prefixed PRIVATE helpers (``_ddg``, ``_http_safety``,
       ``_native_safety``, …) — Python convention says these aren't
       tools; the discovery should skip them entirely so no log
       fires at any level.
    2. Public infra modules (``checkpoint``, ``configure``,
       ``error_sanitizer``, …) that legitimately have no
       TOOL_CONFIG/CONFIGS — those should log at DEBUG, not WARNING,
       because the absence is intentional.
    """

    def _make_registry_with_dir(self, tools_dir, files: dict[str, str]):
        """Materialise ``files`` (name → contents) under ``tools_dir`` and
        return a ToolRegistry pointed at it."""
        from cogtrix_core.registry import ToolRegistry

        for name, content in files.items():
            (tools_dir / name).write_text(content)
        return ToolRegistry(tools_directory=str(tools_dir))

    def test_private_modules_skipped_in_discovery(self, tmp_path):
        """``_helper.py`` must not appear in the discovered-module list —
        no scan, no log, no warning."""
        reg = self._make_registry_with_dir(
            tmp_path,
            {
                "real_tool.py": "TOOL_CONFIG = {}\n",
                "_helper.py": "# private helper\n",
                "_another_private.py": "x = 1\n",
                "__init__.py": "",
            },
        )
        discovered = reg.scan_tools()
        assert "real_tool" in discovered
        assert "_helper" not in discovered
        assert "_another_private" not in discovered

    def test_dunder_init_still_skipped(self, tmp_path):
        """Belt-and-braces: ``__init__.py`` keeps being skipped even
        after the leading-underscore filter is added."""
        reg = self._make_registry_with_dir(
            tmp_path,
            {
                "real_tool.py": "TOOL_CONFIG = {}\n",
                "__init__.py": "TOOL_CONFIG = {'name': 'should not load'}\n",
            },
        )
        assert reg.scan_tools() == ["real_tool"]

    def test_no_tools_resolved_logs_at_debug_not_warning(self, caplog):
        """A module with no TOOL_CONFIG/CONFIGS and no fallback-discoverable
        function ends up with ``results == []``. The log line must fire at
        DEBUG, not WARNING — the absence is intentional for infra modules
        like ``checkpoint`` / ``configure`` that legitimately don't expose
        agent tools."""
        import logging

        from cogtrix_core.registry import ToolRegistry

        # Module with nothing tool-shaped — no TOOL_CONFIG, no *Input class,
        # no function pairings.
        mod = types.ModuleType("fake_infra_module")

        def _internal_helper():  # type: ignore[return]
            """Internal helper, not an agent tool."""

        mod._internal_helper = _internal_helper  # type: ignore[attr-defined]

        registry = ToolRegistry.__new__(ToolRegistry)
        registry.tools = {}
        registry.tool_metadata = {}

        with caplog.at_level(logging.DEBUG, logger="cogtrix"):
            results = registry.extract_tool_functions(mod)

        assert results == []
        # Must appear at DEBUG level.
        debug_matches = [
            r
            for r in caplog.records
            if r.levelno == logging.DEBUG and "no tools resolved" in r.getMessage()
        ]
        warning_matches = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "no tools resolved" in r.getMessage()
        ]
        assert len(debug_matches) == 1, (
            "expected exactly one DEBUG 'no tools resolved' line, got "
            f"{len(debug_matches)} debug + {len(warning_matches)} warning"
        )
        assert warning_matches == [], (
            "no-tools-resolved must NOT fire at WARNING — the absence is "
            "intentional for infra modules and just produces startup noise"
        )
