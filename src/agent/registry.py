"""Agent registry — named agent configurations loaded from config file and AGENTS.md.

Combines two sources:
  1. ``agents:`` section in .cogtrix.yaml / .cogtrix.json (via Config.agents)
  2. AGENTS.md agent definitions loaded by ``src.agent.agents_md``

Config-file entries take precedence over AGENTS.md entries of the same name.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("cogtrix.agent.registry")


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class AgentConfig:
    """Configuration for a named agent."""

    name: str
    description: str = ""
    system_prompt: str = ""
    tools_include: list[str] = field(default_factory=list)  # empty = all tools allowed
    tools_exclude: list[str] = field(default_factory=list)
    model_alias: str = ""  # empty = use session default
    memory_mode: str = ""  # empty = use session default
    max_steps: int = 20
    temperature: float = -1.0  # -1.0 = use model default


# ── Registry class ────────────────────────────────────────────────────────────


class AgentRegistry:
    """In-process registry of named AgentConfig objects."""

    def __init__(self) -> None:
        self._agents: dict[str, AgentConfig] = {}

    def register(self, agent: AgentConfig) -> None:
        """Add or overwrite an agent by name."""
        self._agents[agent.name] = agent

    def get(self, name: str) -> AgentConfig | None:
        """Return the AgentConfig for *name* or None if not found."""
        return self._agents.get(name)

    def list(self) -> list[AgentConfig]:
        """Return all registered agents sorted by name."""
        return sorted(self._agents.values(), key=lambda a: a.name)

    def load_from_config(self, config: Any) -> None:
        """Populate the registry from ``config.agents`` dict.

        Each key in ``config.agents`` is the agent name; the value is a free-form
        dict of AgentConfig fields.  Unknown fields are silently ignored.
        """
        agents_dict = getattr(config, "agents", {}) or {}
        if not isinstance(agents_dict, dict):
            log.warning("Config 'agents' key is not a mapping — skipping")
            return

        _known = {f.name for f in AgentConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]

        for name, raw in agents_dict.items():
            if not isinstance(raw, dict):
                log.warning("Agent config for %r is not a mapping — skipping", name)
                continue
            kwargs: dict[str, Any] = {"name": str(name)}
            for k, v in raw.items():
                if k == "name":
                    continue  # name comes from the key
                if k not in _known:
                    log.debug("Agent %r: unknown field %r — ignoring", name, k)
                    continue
                kwargs[k] = v
            try:
                self._agents[str(name)] = AgentConfig(**kwargs)
            except (TypeError, ValueError) as exc:
                log.warning("Could not load agent config for %r: %s", name, exc)

    def merge_from_agents_md(self, agents: dict[str, Any]) -> None:
        """Merge AgentDefinition objects from agents_md into the registry.

        Config-file entries take precedence: if a name already exists in the
        registry it is not overwritten.
        """
        for key, defn in agents.items():
            name = getattr(defn, "name", key)
            if name in self._agents:
                log.debug(
                    "Agent %r already registered from config — skipping AGENTS.md entry", name
                )
                continue
            try:
                self._agents[name] = AgentConfig(
                    name=name,
                    description=getattr(defn, "description", ""),
                    system_prompt=getattr(defn, "system_prompt", ""),
                    tools_include=list(getattr(defn, "tools_include", []) or []),
                    tools_exclude=list(getattr(defn, "tools_exclude", []) or []),
                    model_alias=getattr(defn, "model_alias", ""),
                    memory_mode=getattr(defn, "memory_mode", ""),
                )
            except (TypeError, ValueError) as exc:
                log.warning("Could not merge AGENTS.md agent %r: %s", name, exc)

    def clear(self) -> None:
        """Remove all registered agents."""
        self._agents.clear()


# ── Module-level singleton ────────────────────────────────────────────────────

_registry = AgentRegistry()


def register(agent: AgentConfig) -> None:
    """Register an agent in the module-level registry."""
    _registry.register(agent)


def get(name: str) -> AgentConfig | None:
    """Return the AgentConfig for *name* from the module-level registry."""
    return _registry.get(name)


def list_agents() -> list[AgentConfig]:
    """Return all registered agents from the module-level registry, sorted by name."""
    return _registry.list()


def load_from_config(config: Any) -> None:
    """Populate the module-level registry from the config file's ``agents:`` section."""
    _registry.load_from_config(config)


def merge_from_agents_md(agents: dict[str, Any]) -> None:
    """Merge AGENTS.md definitions into the module-level registry.

    Config-file entries take precedence over AGENTS.md entries.
    """
    _registry.merge_from_agents_md(agents)


def clear() -> None:
    """Clear the module-level registry."""
    _registry.clear()
