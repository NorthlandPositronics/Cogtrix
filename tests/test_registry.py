"""Tests for src/registry.py — BUG-047 fallback schema matching."""

from __future__ import annotations

import types

from pydantic import BaseModel

from src.registry import _func_to_schema_name

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
        from src.registry import ToolRegistry

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
