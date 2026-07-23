"""Tests for src/agent/agents_md.py"""

from __future__ import annotations

import textwrap
from pathlib import Path

from src.agent.agents_md import (
    AgentDefinition,
    _agent_key,
    get_agent,
    list_agents,
    load_agents_md,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "AGENTS.md"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


# ── _agent_key ────────────────────────────────────────────────────────────────


def test_agent_key_lowercases():
    assert _agent_key("My Agent") == "my_agent"


def test_agent_key_multiple_spaces():
    assert _agent_key("  Code  Reviewer  ") == "code_reviewer"


def test_agent_key_already_normalised():
    assert _agent_key("researcher") == "researcher"


# ── load_agents_md — single agent ─────────────────────────────────────────────


def test_single_agent_minimal(tmp_path):
    p = _write(
        tmp_path,
        """
        ## My Agent

        This is the description.

        ```yaml
        system_prompt: "You are helpful."
        model_alias: fast
        memory_mode: conversation
        ```
        """,
    )
    agents = load_agents_md(p)
    assert len(agents) == 1
    a = agents["my_agent"]
    assert isinstance(a, AgentDefinition)
    assert a.name == "My Agent"
    assert "description" in a.description
    assert a.system_prompt == "You are helpful."
    assert a.model_alias == "fast"
    assert a.memory_mode == "conversation"
    assert a.tools_include == []
    assert a.tools_exclude == []


def test_single_agent_tools(tmp_path):
    p = _write(
        tmp_path,
        """
        ## Searcher

        ```yaml
        tools_include:
          - web_search
          - exa_search
        tools_exclude:
          - shell
        ```
        """,
    )
    agents = load_agents_md(p)
    a = agents["searcher"]
    assert a.tools_include == ["web_search", "exa_search"]
    assert a.tools_exclude == ["shell"]


# ── load_agents_md — multiple agents ──────────────────────────────────────────


def test_multiple_agents(tmp_path):
    p = _write(
        tmp_path,
        """
        ## Researcher

        Does research.

        ```yaml
        model_alias: smart
        ```

        ## Coder

        Writes code.

        ```yaml
        memory_mode: code
        ```
        """,
    )
    agents = load_agents_md(p)
    assert set(agents.keys()) == {"researcher", "coder"}
    assert agents["researcher"].model_alias == "smart"
    assert agents["coder"].memory_mode == "code"


def test_multiple_agents_descriptions(tmp_path):
    p = _write(
        tmp_path,
        """
        ## Alpha

        Alpha description here.

        ## Beta

        Beta description here.
        """,
    )
    agents = load_agents_md(p)
    assert "Alpha description here." in agents["alpha"].description
    assert "Beta description here." in agents["beta"].description


# ── prompt_file ───────────────────────────────────────────────────────────────


def test_prompt_file_loaded(tmp_path):
    prompt_path = tmp_path / "prompts" / "system.txt"
    prompt_path.parent.mkdir()
    prompt_path.write_text("You are a coding expert.", encoding="utf-8")

    p = _write(
        tmp_path,
        """
        ## Coder

        ```yaml
        prompt_file: prompts/system.txt
        ```
        """,
    )
    agents = load_agents_md(p)
    a = agents["coder"]
    assert a.system_prompt == "You are a coding expert."
    assert a.prompt_file == "prompts/system.txt"


def test_prompt_file_missing_warns(tmp_path, caplog):
    import logging

    p = _write(
        tmp_path,
        """
        ## Ghost

        ```yaml
        prompt_file: nonexistent/prompt.txt
        ```
        """,
    )
    with caplog.at_level(logging.WARNING, logger="cogtrix.agents_md"):
        agents = load_agents_md(p)
    assert "ghost" in agents  # agent still present
    assert agents["ghost"].system_prompt == ""  # but no system_prompt loaded
    assert any("prompt_file" in r.message or "Cannot read" in r.message for r in caplog.records)


def test_prompt_file_path_traversal_is_blocked(tmp_path, caplog):
    """prompt_file values with '..' path traversal must not read outside the AGENTS.md dir."""
    import logging

    # Write a sensitive file outside the agents_md directory
    sensitive_file = tmp_path.parent / "secret.txt"
    sensitive_file.write_text("TOP SECRET", encoding="utf-8")

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    p = agents_dir / "AGENTS.md"
    p.write_text(
        "## Attacker\n\n```yaml\nprompt_file: ../../secret.txt\n```\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger="cogtrix.agents_md"):
        agents = load_agents_md(p)
    # Agent is still present but no prompt was loaded from outside the directory
    assert "attacker" in agents
    assert agents["attacker"].system_prompt == ""
    assert "TOP SECRET" not in agents["attacker"].system_prompt
    # Warning was logged about the traversal attempt
    assert any("outside" in r.message for r in caplog.records)


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_missing_file_returns_empty(tmp_path):
    agents = load_agents_md(tmp_path / "AGENTS.md")
    assert agents == {}


def test_empty_file_returns_empty(tmp_path):
    p = tmp_path / "AGENTS.md"
    p.write_text("", encoding="utf-8")
    agents = load_agents_md(p)
    assert agents == {}


def test_no_headings_returns_empty(tmp_path):
    p = _write(
        tmp_path,
        """
        # Top-level heading only

        Some text without level-2 headings.
        """,
    )
    agents = load_agents_md(p)
    assert agents == {}


def test_malformed_yaml_block_returns_partial(tmp_path, caplog):
    import logging

    p = _write(
        tmp_path,
        """
        ## Good Agent

        ```yaml
        model_alias: fast
        ```

        ## Bad Agent

        ```yaml
        {this is: [invalid yaml
        ```
        """,
    )
    with caplog.at_level(logging.WARNING, logger="cogtrix.agents_md"):
        agents = load_agents_md(p)

    # Good agent must be present; bad agent present but with empty config
    assert "good_agent" in agents
    assert agents["good_agent"].model_alias == "fast"
    assert "bad_agent" in agents  # still created, config just empty
    assert agents["bad_agent"].model_alias == ""


def test_agent_with_no_yaml_block(tmp_path):
    p = _write(
        tmp_path,
        """
        ## Plain Agent

        Just a description, no yaml block at all.
        """,
    )
    agents = load_agents_md(p)
    a = agents["plain_agent"]
    assert "Just a description" in a.description
    assert a.system_prompt == ""
    assert a.model_alias == ""


def test_tools_include_single_string(tmp_path):
    """tools_include / tools_exclude accept a bare scalar (coerced to list)."""
    p = _write(
        tmp_path,
        """
        ## Solo

        ```yaml
        tools_include: web_search
        ```
        """,
    )
    agents = load_agents_md(p)
    assert agents["solo"].tools_include == ["web_search"]


def test_tools_null_becomes_empty_list(tmp_path):
    p = _write(
        tmp_path,
        """
        ## Null Tools

        ```yaml
        tools_include: null
        tools_exclude: null
        ```
        """,
    )
    agents = load_agents_md(p)
    a = agents["null_tools"]
    assert a.tools_include == []
    assert a.tools_exclude == []


# ── Registry functions ────────────────────────────────────────────────────────


def test_get_agent_returns_definition(tmp_path, monkeypatch):
    from src.agent import agents_md as _mod

    p = _write(
        tmp_path,
        """
        ## Finder

        ```yaml
        model_alias: fast
        ```
        """,
    )
    loaded = load_agents_md(p)
    monkeypatch.setattr(_mod, "_agents_registry", loaded)

    a = get_agent("finder")
    assert a is not None
    assert a.name == "Finder"
    assert a.model_alias == "fast"


def test_get_agent_unknown_returns_none(monkeypatch):
    from src.agent import agents_md as _mod

    monkeypatch.setattr(_mod, "_agents_registry", {})
    assert get_agent("nonexistent") is None


def test_get_agent_normalises_name(tmp_path, monkeypatch):
    from src.agent import agents_md as _mod

    p = _write(tmp_path, "## My Cool Agent\n\n```yaml\nmodel_alias: smart\n```\n")
    monkeypatch.setattr(_mod, "_agents_registry", load_agents_md(p))

    # Name with different casing and spaces must resolve
    assert get_agent("My Cool Agent") is not None
    assert get_agent("my cool agent") is not None
    assert get_agent("my_cool_agent") is not None


def test_list_agents_returns_all(tmp_path, monkeypatch):
    from src.agent import agents_md as _mod

    p = _write(
        tmp_path,
        """
        ## Alpha
        ## Beta
        ## Gamma
        """,
    )
    monkeypatch.setattr(_mod, "_agents_registry", load_agents_md(p))
    agents = list_agents()
    assert len(agents) == 3
    names = {a.name for a in agents}
    assert names == {"Alpha", "Beta", "Gamma"}


def test_list_agents_empty(monkeypatch):
    from src.agent import agents_md as _mod

    monkeypatch.setattr(_mod, "_agents_registry", {})
    assert list_agents() == []


# ── load_default_agents ───────────────────────────────────────────────────────


def test_load_default_agents_no_file(monkeypatch, tmp_path):
    from src.agent import agents_md as _mod

    monkeypatch.setattr(_mod, "get_agents_md_path", lambda: None)
    result = _mod.load_default_agents()
    assert result == {}
    assert _mod._agents_registry == {}


def test_load_default_agents_from_file(monkeypatch, tmp_path):
    from src.agent import agents_md as _mod

    p = _write(tmp_path, "## Helper\n\n```yaml\nmodel_alias: fast\n```\n")
    monkeypatch.setattr(_mod, "get_agents_md_path", lambda: p)
    result = _mod.load_default_agents()
    assert "helper" in result
    assert _mod._agents_registry is result


# ── get_agents_md_path ────────────────────────────────────────────────────────


def test_get_agents_md_path_cwd_preferred(tmp_path, monkeypatch):
    from src.agent import agents_md as _mod

    cwd_agents = tmp_path / "AGENTS.md"
    cwd_agents.write_text("", encoding="utf-8")
    home_agents = tmp_path / "home" / "AGENTS.md"
    home_agents.parent.mkdir()
    home_agents.write_text("", encoding="utf-8")

    monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))

    result = _mod.get_agents_md_path()
    assert result == cwd_agents


def test_get_agents_md_path_falls_back_to_home(tmp_path, monkeypatch):
    from src.agent import agents_md as _mod

    home_agents = tmp_path / "AGENTS.md"
    home_agents.write_text("", encoding="utf-8")

    monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: tmp_path / "empty_cwd"))
    (tmp_path / "empty_cwd").mkdir(exist_ok=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    result = _mod.get_agents_md_path()
    assert result == home_agents


def test_get_agents_md_path_none_when_missing(tmp_path, monkeypatch):
    from src.agent import agents_md as _mod

    empty = tmp_path / "no_agents_here"
    empty.mkdir()
    monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: empty))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: empty))

    assert _mod.get_agents_md_path() is None
