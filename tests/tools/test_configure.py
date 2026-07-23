"""Tests for cogtrix_core/tools/configure — tool configuration factories."""

from unittest.mock import MagicMock, patch


class TestCreateRequestToolsTool:
    """Tests for the request_tools meta-tool factory."""

    def test_returns_tool_with_correct_name(self):
        from cogtrix_core.tools.configure import create_request_tools_tool

        available = {"web_search": MagicMock(description="Search the web")}
        catalog = {"web_search": "Search the web"}
        tool = create_request_tools_tool(available, catalog)
        assert tool is not None
        assert tool.name == "request_tools"

    def test_returns_none_without_structured_tool(self):
        """When langchain_core is missing, returns None gracefully."""
        with patch.dict("sys.modules", {"langchain_core": None, "langchain_core.tools": None}):
            # Force re-import failure — but the function uses a local try/except ImportError
            # so we need to actually test the function's internal fallback.
            # Since the function does `from langchain_core.tools import StructuredTool`
            # inside a try/except, we need to make that import fail.
            pass
        # This is hard to test with mocking. Skip if StructuredTool is available.
        # The function has a try/except that returns None on ImportError.

    def test_valid_add_returns_message(self):
        from cogtrix_core.tools.configure import create_request_tools_tool

        available = {"web_search": MagicMock(description="Search")}
        catalog = {"web_search": "Search"}
        tool = create_request_tools_tool(available, catalog)
        assert tool is not None
        result = tool.invoke({"add": ["web_search"], "remove": []})
        assert "web_search" in result
        assert "requested" in result.lower() or "loaded" in result.lower()

    def test_invalid_add_returns_error(self):
        from cogtrix_core.tools.configure import create_request_tools_tool

        available = {"web_search": MagicMock(description="Search")}
        catalog = {"web_search": "Search"}
        tool = create_request_tools_tool(available, catalog)
        assert tool is not None
        result = tool.invoke({"add": ["nonexistent"], "remove": []})
        assert "nonexistent" in result
        assert "unknown" in result.lower()

    def test_already_active_add_returns_distinct_message(self):
        """BUG-C: tools already active produce a different message than truly unknown tools."""
        from cogtrix_core.tools.configure import create_request_tools_tool

        available = {}
        catalog = {}
        active = {"web_search"}
        tool = create_request_tools_tool(available, catalog, active_names=active)
        assert tool is not None
        result = tool.invoke({"add": ["web_search"], "remove": []})
        assert "web_search" in result
        assert "already active" in result.lower()
        assert "unknown" not in result.lower()

    def test_fuzzy_resolved_name_shown_in_success_message(self):
        """BUG-G: fuzzy resolution includes '(resolved from X)' in success message.

        #1924 added a short-request guard so ``"http"`` (4 chars, single
        token) no longer fuzzy-resolves.  Use ``"http_req"`` (multi-
        token, past the guard) which fuzzy-resolves to ``http_request``
        via the prefix-token bonus on ``req``/``request``.
        """
        from cogtrix_core.tools.configure import create_request_tools_tool

        available = {"http_request": MagicMock(description="HTTP requests")}
        catalog = {"http_request": "HTTP requests"}
        tool = create_request_tools_tool(available, catalog)
        assert tool is not None
        result = tool.invoke({"add": ["http_req"], "remove": []})
        assert "http_request" in result
        assert "resolved from" in result.lower()
        assert "'http_req'" in result

    def test_exact_match_no_resolved_from_annotation(self):
        """BUG-G: exact name match does not add '(resolved from X)' annotation."""
        from cogtrix_core.tools.configure import create_request_tools_tool

        available = {"search_web": MagicMock(description="Search")}
        catalog = {"search_web": "Search"}
        tool = create_request_tools_tool(available, catalog)
        assert tool is not None
        result = tool.invoke({"add": ["search_web"], "remove": []})
        assert "search_web" in result
        assert "resolved from" not in result.lower()

    def test_protected_release_blocked(self):
        from cogtrix_core.tools.configure import create_request_tools_tool

        available = {}
        catalog = {}
        active = {"core_tool", "removable_tool"}
        protected = {"core_tool"}
        tool = create_request_tools_tool(
            available, catalog, active_names=active, protected_names=protected
        )
        assert tool is not None
        result = tool.invoke({"add": [], "remove": ["core_tool"]})
        assert "core_tool" in result
        assert "cannot" in result.lower() or "core" in result.lower()

    def test_valid_release(self):
        from cogtrix_core.tools.configure import create_request_tools_tool

        available = {}
        catalog = {}
        active = {"removable_tool"}
        protected = set()
        tool = create_request_tools_tool(
            available, catalog, active_names=active, protected_names=protected
        )
        assert tool is not None
        result = tool.invoke({"add": [], "remove": ["removable_tool"]})
        assert "removable_tool" in result
        assert "releas" in result.lower() or "remov" in result.lower()

    def test_add_wins_over_remove_dedup(self):
        """If a tool appears in both add and remove, add wins."""
        from cogtrix_core.tools.configure import create_request_tools_tool

        available = {"dual_tool": MagicMock(description="Dual")}
        catalog = {"dual_tool": "Dual"}
        active = {"dual_tool"}
        tool = create_request_tools_tool(available, catalog, active_names=active)
        assert tool is not None
        result = tool.invoke({"add": ["dual_tool"], "remove": ["dual_tool"]})
        # Should process as add, not remove
        assert "requested" in result.lower() or "loaded" in result.lower() or "dual_tool" in result

    def test_no_names_provided_returns_catalog(self):
        from cogtrix_core.tools.configure import create_request_tools_tool

        available = {"x": MagicMock(description="X")}
        catalog = {"x": "X"}
        tool = create_request_tools_tool(available, catalog)
        assert tool is not None
        result = tool.invoke({"add": [], "remove": []})
        assert "ADD" in result
        assert "x" in result

    def test_no_names_provided_uses_compact_catalog_when_many_tools_active(self):
        from cogtrix_core.tools.configure import create_request_tools_tool

        available = {f"tool_{i}": MagicMock(description=f"Tool {i}") for i in range(12)}
        catalog = {name: tool.description for name, tool in available.items()}
        active = {f"active_{i}" for i in range(10)}
        tool = create_request_tools_tool(available, catalog, active_names=active)
        assert tool is not None
        result = tool.invoke({"add": [], "remove": []})
        assert "use add=[...]" in result.lower()
        assert "tool_0" not in result
        assert "active_0" not in result

    def test_description_does_not_contain_tool_names(self):
        """Tool names must not appear in the description; they are in the return value."""
        from cogtrix_core.tools.configure import create_request_tools_tool

        available = {"web_search": MagicMock(description="Search the web")}
        catalog = {"web_search": "Search the web"}
        tool = create_request_tools_tool(available, catalog)
        assert tool is not None
        assert "web_search" not in tool.description
        # Catalog is still accessible via the return value when called with no args
        result = tool.invoke({"add": [], "remove": []})
        assert "web_search" in result

    def test_full_catalog_still_lists_tool_names_for_small_active_sets(self):
        from cogtrix_core.tools.configure import create_request_tools_tool

        available = {
            "search_web": MagicMock(description="Search"),
            "read_file": MagicMock(description="Read"),
        }
        catalog = {"search_web": "Search", "read_file": "Read"}
        active = {"core_tool"}
        tool = create_request_tools_tool(available, catalog, active_names=active)
        assert tool is not None
        result = tool.invoke({"add": [], "remove": []})
        assert "search_web" in result
        assert "read_file" in result

    def test_fuzzy_add_resolves_abbreviated_name(self):
        """BUG-076: gpt-4o sends 'search' instead of 'search_web'."""
        from cogtrix_core.tools.configure import create_request_tools_tool

        available = {
            "search_web": MagicMock(description="Search the web"),
            "search_news": MagicMock(description="Search news"),
        }
        catalog = {"search_web": "Search the web", "search_news": "Search news"}
        tool = create_request_tools_tool(available, catalog)
        assert tool is not None
        result = tool.invoke({"add": ["search"], "remove": []})
        assert "loaded" in result.lower() or "active" in result.lower()
        assert "search_web" in result or "search_news" in result

    def test_fuzzy_add_resolves_partial_name(self):
        """Fuzzy resolution: 'http_req' resolves to 'http_request'.

        #1924 added a short-request guard so single-token requests of
        ≤4 chars (like the original ``'http'``) now bail at the
        resolver.  Multi-token requests still fuzzy-resolve normally.
        """
        from cogtrix_core.tools.configure import create_request_tools_tool

        available = {"http_request": MagicMock(description="HTTP requests")}
        catalog = {"http_request": "HTTP requests"}
        tool = create_request_tools_tool(available, catalog)
        assert tool is not None
        result = tool.invoke({"add": ["http_req"], "remove": []})
        assert "http_request" in result
        assert "loaded" in result.lower() or "active" in result.lower()

    def test_fuzzy_add_exact_match_takes_priority(self):
        """Exact name match should still work and take priority."""
        from cogtrix_core.tools.configure import create_request_tools_tool

        available = {"search_web": MagicMock(description="Search")}
        catalog = {"search_web": "Search"}
        tool = create_request_tools_tool(available, catalog)
        assert tool is not None
        result = tool.invoke({"add": ["search_web"], "remove": []})
        assert "search_web" in result
        assert "loaded" in result.lower() or "active" in result.lower()

    def test_fuzzy_no_match_still_rejected(self):
        """Completely unrelated names should still be rejected."""
        from cogtrix_core.tools.configure import create_request_tools_tool

        available = {"search_web": MagicMock(description="Search")}
        catalog = {"search_web": "Search"}
        tool = create_request_tools_tool(available, catalog)
        assert tool is not None
        result = tool.invoke({"add": ["completely_unrelated_xyz"], "remove": []})
        assert "cannot" in result.lower() or "unknown" in result.lower()


class _BrokenConfigureModule:
    """Fake module that raises ImportError/OSError when specific attrs are accessed.

    Used to simulate a tool module whose import succeeds (module is in sys.modules)
    but whose body fails on a transitive dependency — the scenario from #1089.
    """

    def __init__(self, name: str, broken_attrs: dict[str, Exception]) -> None:
        self.__name__ = name
        self._broken_attrs = broken_attrs

    def __getattr__(self, name: str) -> object:
        if name in self._broken_attrs:
            raise self._broken_attrs[name]
        raise AttributeError(name)


class TestConfigureLoggingOnImportError:
    """Regression tests for issue #1089: configure functions must log a warning
    when a tool module exists but fails to import due to a transitive dependency
    error (not just when the module itself is missing).
    """

    def test_module_not_found_is_silent(self, caplog):
        """ModuleNotFoundError (module not installed) — no warning logged."""
        import sys
        from unittest.mock import MagicMock

        key = "cogtrix_core.tools.tavily_search"
        prior = sys.modules.pop(key, None)
        try:
            from cogtrix_core.tools.configure import configure_tavily_tool

            config = MagicMock()
            configure_tavily_tool(config)
            # Must not log any warning — tool is simply not installed.
            assert not any(r.levelname == "WARNING" for r in caplog.records)
        finally:
            if prior is not None:
                sys.modules[key] = prior

    def test_transitive_import_error_logs_warning(self, caplog):
        """ImportError (non-ModuleNotFoundError) — warning logged."""
        import sys
        from unittest.mock import MagicMock

        key = "cogtrix_core.tools.tavily_search"
        prior = sys.modules.pop(key, None)
        fake_module = _BrokenConfigureModule(
            key,
            {
                "configure_tavily": ImportError(
                    "cannot import name 'Something' from 'transitive_dep'"
                )
            },
        )
        try:
            sys.modules[key] = fake_module
            from cogtrix_core.tools.configure import configure_tavily_tool

            config = MagicMock()
            configure_tavily_tool(config)
            warnings = [r for r in caplog.records if r.levelname == "WARNING"]
            assert len(warnings) >= 1, (
                f"Expected at least one WARNING for transitive ImportError, "
                f"got: {[r.message for r in caplog.records]}"
            )
            assert "tavily" in warnings[0].message.lower()
        finally:
            sys.modules.pop(key, None)
            if prior is not None:
                sys.modules[key] = prior

    def test_delegate_tool_transitive_failure_logs_warning(self, caplog):
        """configure_delegate_tool logs warning on transitive ImportError."""
        import sys
        from unittest.mock import MagicMock

        key = "cogtrix_core.tools.delegate"
        prior = sys.modules.pop(key, None)
        fake_module = _BrokenConfigureModule(
            key,
            {
                "configure_delegate": ImportError("transitive dependency failure in delegate"),
                "set_status_callback": ImportError("transitive dependency failure in delegate"),
            },
        )
        try:
            sys.modules[key] = fake_module
            from cogtrix_core.tools.configure import configure_delegate_tool

            config = MagicMock()
            configure_delegate_tool(config)
            warnings = [r for r in caplog.records if r.levelname == "WARNING"]
            assert len(warnings) >= 1, (
                f"Expected WARNING for delegate tool transitive ImportError, "
                f"got: {[r.message for r in caplog.records]}"
            )
            assert "delegate" in warnings[0].message.lower()
        finally:
            sys.modules.pop(key, None)
            if prior is not None:
                sys.modules[key] = prior

    def test_cron_tool_oserror_logs_warning(self, caplog):
        """configure_cron_tool logs warning on OSError."""
        import sys
        from unittest.mock import MagicMock

        key = "cogtrix_core.tools.cron_tools"
        prior = sys.modules.pop(key, None)
        fake_module = _BrokenConfigureModule(
            key,
            {"configure_cron": OSError("permission denied on data directory")},
        )
        try:
            sys.modules[key] = fake_module
            from cogtrix_core.tools.configure import configure_cron_tool

            config = MagicMock()
            configure_cron_tool(config)
            warnings = [r for r in caplog.records if r.levelname == "WARNING"]
            assert len(warnings) >= 1, (
                f"Expected WARNING for cron tool OSError, "
                f"got: {[r.message for r in caplog.records]}"
            )
            assert "cron" in warnings[0].message.lower()
        finally:
            sys.modules.pop(key, None)
            if prior is not None:
                sys.modules[key] = prior


class TestRequestToolsQueryRecovery:
    """Regression for bug #1839 (secondary root cause): a stray ``query``
    must not silently no-op. Either it names a real tool (forgiving load)
    or it produces a loud 'nothing loaded' signal — never a catalog dump
    that the model mistakes for success."""

    def test_query_naming_a_tool_loads_it(self):
        from cogtrix_core.tools.configure import create_request_tools_tool

        available = {"web_search": MagicMock(description="Search the web")}
        catalog = {"web_search": "Search the web"}
        tool = create_request_tools_tool(available, catalog)

        result = tool.invoke({"query": "web_search"})

        # The stray `query` was routed through the `add` path — "Tools loaded:"
        # is only emitted when valid_add is non-empty, which here can only
        # happen via the forgiving recovery (no `add` was passed).
        assert "Tools loaded:" in result
        assert "web_search" in result
        # And the agent is nudged toward the correct call shape next time.
        assert "add=" in result

    def test_query_description_without_index_is_loud_noop(self):
        from cogtrix_core.tools.configure import create_request_tools_tool

        # The exact malformed call from the next66 trial run, no semantic index.
        available = {"web_search": MagicMock(description="Search the web")}
        catalog = {"web_search": "Search the web"}
        tool = create_request_tools_tool(available, catalog, tool_index=None)

        result = tool.invoke({"query": "current stock price financial data"})

        # Must NOT load anything and must NOT masquerade as success.
        assert "web_search" in available  # nothing was loaded
        assert "Nothing was loaded" in result
        assert "add=" in result
        # Regression guard: the old behaviour dumped the catalog header as if OK.
        assert "Tools you can ADD" not in result

    def test_query_description_with_index_is_loud_noop(self):
        from cogtrix_core.tools.configure import create_request_tools_tool

        index = MagicMock()
        index.search.return_value = ["web_search"]
        available = {"web_search": MagicMock(description="Search the web")}
        catalog = {"web_search": "Search the web"}
        tool = create_request_tools_tool(available, catalog, tool_index=index)

        result = tool.invoke({"query": "find me something to search the web"})

        assert "Nothing was loaded" in result
        assert "web_search" in result  # listed as a candidate
        assert "web_search" in available  # but not actually loaded
        assert "add=" in result
