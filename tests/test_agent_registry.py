"""Tests for src.agent.registry — AgentConfig, AgentRegistry, and module singleton."""

from __future__ import annotations

from src.agent.registry import (
    AgentConfig,
    AgentRegistry,
    clear,
    filter_tools_for_agent,
    get,
    list_agents,
    register,
)

# ---------------------------------------------------------------------------
# AgentConfig defaults
# ---------------------------------------------------------------------------


def test_agent_config_defaults() -> None:
    a = AgentConfig(name="tester")
    assert a.description == ""
    assert a.system_prompt == ""
    assert a.tools_include == []
    assert a.tools_exclude == []
    assert a.model_alias == ""
    assert a.memory_mode == ""
    assert a.max_steps == 20
    assert a.temperature == -1.0


def test_agent_config_custom_fields() -> None:
    a = AgentConfig(
        name="coder",
        description="A coding agent",
        system_prompt="You write code.",
        tools_include=["write_file", "shell"],
        tools_exclude=["web_search"],
        model_alias="fast",
        memory_mode="code",
        max_steps=10,
        temperature=0.2,
    )
    assert a.name == "coder"
    assert a.tools_include == ["write_file", "shell"]
    assert a.max_steps == 10
    assert a.temperature == 0.2


# ---------------------------------------------------------------------------
# AgentRegistry CRUD
# ---------------------------------------------------------------------------


class TestAgentRegistry:
    def setup_method(self) -> None:
        self.reg = AgentRegistry()

    def test_register_and_get(self) -> None:
        a = AgentConfig(name="alpha")
        self.reg.register(a)
        assert self.reg.get("alpha") is a

    def test_get_missing_returns_none(self) -> None:
        assert self.reg.get("nonexistent") is None

    def test_list_sorted_by_name(self) -> None:
        self.reg.register(AgentConfig(name="zebra"))
        self.reg.register(AgentConfig(name="alpha"))
        self.reg.register(AgentConfig(name="middle"))
        names = [a.name for a in self.reg.list()]
        assert names == ["alpha", "middle", "zebra"]

    def test_list_empty(self) -> None:
        assert self.reg.list() == []

    def test_register_overwrites(self) -> None:
        self.reg.register(AgentConfig(name="dup", description="first"))
        self.reg.register(AgentConfig(name="dup", description="second"))
        assert self.reg.get("dup").description == "second"  # type: ignore[union-attr]

    def test_clear(self) -> None:
        self.reg.register(AgentConfig(name="x"))
        self.reg.clear()
        assert self.reg.list() == []


# ---------------------------------------------------------------------------
# load_from_config
# ---------------------------------------------------------------------------


class TestLoadFromConfig:
    def setup_method(self) -> None:
        self.reg = AgentRegistry()

    def _cfg(self, agents_dict: object) -> object:
        """Minimal config-like object."""

        class _Cfg:
            agents = agents_dict

        return _Cfg()

    def test_load_basic(self) -> None:
        cfg = self._cfg(
            {
                "researcher": {
                    "description": "Researches topics",
                    "system_prompt": "You research.",
                    "max_steps": 5,
                }
            }
        )
        self.reg.load_from_config(cfg)
        a = self.reg.get("researcher")
        assert a is not None
        assert a.description == "Researches topics"
        assert a.max_steps == 5

    def test_load_ignores_unknown_fields(self) -> None:
        cfg = self._cfg({"x": {"description": "ok", "unknown_field": "ignored"}})
        self.reg.load_from_config(cfg)
        a = self.reg.get("x")
        assert a is not None
        assert a.description == "ok"

    def test_load_skips_non_dict_value(self) -> None:
        cfg = self._cfg({"bad": "not-a-dict"})
        self.reg.load_from_config(cfg)
        assert self.reg.get("bad") is None

    def test_load_skips_non_dict_agents(self) -> None:
        cfg = self._cfg("not-a-dict")
        self.reg.load_from_config(cfg)
        assert self.reg.list() == []

    def test_load_no_agents_attr(self) -> None:
        class _Empty:
            pass

        self.reg.load_from_config(_Empty())
        assert self.reg.list() == []

    def test_load_agents_none(self) -> None:
        cfg = self._cfg(None)
        self.reg.load_from_config(cfg)
        assert self.reg.list() == []

    def test_load_multiple_agents(self) -> None:
        cfg = self._cfg(
            {
                "a1": {"description": "first"},
                "a2": {"description": "second"},
            }
        )
        self.reg.load_from_config(cfg)
        assert len(self.reg.list()) == 2

    def test_load_tools_include_exclude(self) -> None:
        cfg = self._cfg(
            {
                "tooled": {
                    "tools_include": ["web_search", "shell"],
                    "tools_exclude": ["write_file"],
                }
            }
        )
        self.reg.load_from_config(cfg)
        a = self.reg.get("tooled")
        assert a is not None
        assert a.tools_include == ["web_search", "shell"]
        assert a.tools_exclude == ["write_file"]


# ---------------------------------------------------------------------------
# merge_from_agents_md
# ---------------------------------------------------------------------------


class _FakeAgentDef:
    """Minimal AgentDefinition-like object for testing merge_from_agents_md."""

    def __init__(self, name: str, description: str = "", system_prompt: str = "") -> None:
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.tools_include: list[str] = []
        self.tools_exclude: list[str] = []
        self.model_alias = ""
        self.memory_mode = ""


class TestMergeFromAgentsMd:
    def setup_method(self) -> None:
        self.reg = AgentRegistry()

    def test_merge_adds_new_agent(self) -> None:
        defn = _FakeAgentDef("researcher", description="From AGENTS.md")
        self.reg.merge_from_agents_md({"researcher": defn})
        a = self.reg.get("researcher")
        assert a is not None
        assert a.description == "From AGENTS.md"

    def test_config_takes_precedence_over_agents_md(self) -> None:
        self.reg.register(AgentConfig(name="researcher", description="From config"))
        defn = _FakeAgentDef("researcher", description="From AGENTS.md")
        self.reg.merge_from_agents_md({"researcher": defn})
        assert self.reg.get("researcher").description == "From config"  # type: ignore[union-attr]

    def test_merge_empty_dict(self) -> None:
        self.reg.merge_from_agents_md({})
        assert self.reg.list() == []

    def test_merge_uses_defn_name_over_key(self) -> None:
        defn = _FakeAgentDef("Real Name")
        self.reg.merge_from_agents_md({"some_key": defn})
        assert self.reg.get("Real Name") is not None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


class TestModuleSingleton:
    def setup_method(self) -> None:
        clear()

    def teardown_method(self) -> None:
        clear()

    def test_register_and_get(self) -> None:
        a = AgentConfig(name="mod_agent")
        register(a)
        assert get("mod_agent") is a

    def test_get_missing(self) -> None:
        assert get("does_not_exist") is None

    def test_list_agents(self) -> None:
        register(AgentConfig(name="b_agent"))
        register(AgentConfig(name="a_agent"))
        names = [a.name for a in list_agents()]
        assert names == ["a_agent", "b_agent"]

    def test_clear(self) -> None:
        register(AgentConfig(name="temp"))
        clear()
        assert list_agents() == []

    def test_load_from_config_via_singleton(self) -> None:
        from src.agent.registry import load_from_config

        class _Cfg:
            agents = {"singleton_agent": {"description": "loaded from config"}}

        load_from_config(_Cfg())
        a = get("singleton_agent")
        assert a is not None
        assert a.description == "loaded from config"

    def test_merge_from_agents_md_via_singleton(self) -> None:
        from src.agent.registry import merge_from_agents_md

        defn = _FakeAgentDef("md_agent", description="from md")
        merge_from_agents_md({"md_agent": defn})
        a = get("md_agent")
        assert a is not None
        assert a.description == "from md"


# ---------------------------------------------------------------------------
# filter_tools_for_agent
# ---------------------------------------------------------------------------


def _make_tool_dict(names: list[str]) -> dict[str, object]:
    """Create a fake tool dict keyed by name."""
    return {n: object() for n in names}


class TestFilterToolsForAgent:
    def setup_method(self) -> None:
        clear()

    def teardown_method(self) -> None:
        clear()

    def test_none_agent_name_returns_all(self) -> None:
        tools = _make_tool_dict(["a", "b", "c"])
        f_dict, f_list = filter_tools_for_agent(None, tools)
        assert f_dict == tools
        assert len(f_list) == 3

    def test_unknown_agent_returns_all(self) -> None:
        tools = _make_tool_dict(["a", "b", "c"])
        f_dict, f_list = filter_tools_for_agent("unknown", tools)
        assert f_dict == tools
        assert len(f_list) == 3

    def test_empty_include_allows_all_with_excludes(self) -> None:
        register(AgentConfig(name="agent1", tools_exclude=["b"]))
        tools = _make_tool_dict(["a", "b", "c"])
        f_dict, f_list = filter_tools_for_agent("agent1", tools)
        assert set(f_dict.keys()) == {"a", "c"}
        assert len(f_list) == 2

    def test_include_restricts_and_exclude_removes(self) -> None:
        register(
            AgentConfig(
                name="agent2",
                tools_include=["a", "b", "c"],
                tools_exclude=["b"],
            )
        )
        tools = _make_tool_dict(["a", "b", "c", "d", "e"])
        f_dict, f_list = filter_tools_for_agent("agent2", tools)
        assert set(f_dict.keys()) == {"a", "c"}
        assert len(f_list) == 2

    def test_include_only_keeps_listed(self) -> None:
        register(AgentConfig(name="agent3", tools_include=["a", "c"]))
        tools = _make_tool_dict(["a", "b", "c", "d"])
        f_dict, f_list = filter_tools_for_agent("agent3", tools)
        assert set(f_dict.keys()) == {"a", "c"}
        assert len(f_list) == 2

    def test_no_filters_returns_all(self) -> None:
        register(AgentConfig(name="agent4"))
        tools = _make_tool_dict(["a", "b", "c"])
        f_dict, f_list = filter_tools_for_agent("agent4", tools)
        assert f_dict == tools
        assert len(f_list) == 3

    def test_exclude_only_removes_specified(self) -> None:
        register(AgentConfig(name="agent5", tools_exclude=["a", "c"]))
        tools = _make_tool_dict(["a", "b", "c", "d"])
        f_dict, f_list = filter_tools_for_agent("agent5", tools)
        assert set(f_dict.keys()) == {"b", "d"}
        assert len(f_list) == 2

    def test_include_nonexistent_tool_handled_gracefully(self) -> None:
        register(AgentConfig(name="agent6", tools_include=["a", "zzz"]))
        tools = _make_tool_dict(["a", "b"])
        f_dict, f_list = filter_tools_for_agent("agent6", tools)
        assert set(f_dict.keys()) == {"a"}
        assert len(f_list) == 1

    def test_empty_tools_dict(self) -> None:
        register(AgentConfig(name="agent7", tools_include=["a"]))
        f_dict, f_list = filter_tools_for_agent("agent7", {})
        assert f_dict == {}
        assert f_list == []

    def test_filter_preserves_object_identity(self) -> None:
        register(AgentConfig(name="agent8", tools_include=["a"]))
        tools = _make_tool_dict(["a", "b"])
        f_dict, f_list = filter_tools_for_agent("agent8", tools)
        assert f_dict["a"] is tools["a"]
        assert f_list[0] is tools["a"]
