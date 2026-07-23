"""AGENTS.md support — load named agent definitions from a Markdown file.

Format::

    ## Agent Name

    Optional description text (everything between headings and yaml block).

    ```yaml
    system_prompt: |
      You are a helpful assistant.
    tools_include:
      - web_search
    tools_exclude: []
    model_alias: fast
    memory_mode: conversation
    ```

Fields in the yaml block
  system_prompt   inline prompt text (mutually exclusive with prompt_file)
  prompt_file     path relative to AGENTS.md directory; loaded from disk
  tools_include   list of tool names to enable (empty = all)
  tools_exclude   list of tool names to block
  model_alias     model alias from config (e.g. "fast", "smart") or ""
  memory_mode     "conversation", "code", "reasoning" or ""
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("cogtrix.agents_md")

try:
    import yaml as _yaml

    _HAS_YAML = True
except ImportError:  # pragma: no cover
    _HAS_YAML = False
    _yaml = None  # type: ignore[assignment]

# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class AgentDefinition:
    name: str
    description: str = ""
    system_prompt: str = ""
    prompt_file: str = ""
    tools_include: list[str] = field(default_factory=list)
    tools_exclude: list[str] = field(default_factory=list)
    model_alias: str = ""
    memory_mode: str = ""


# ── Parser ────────────────────────────────────────────────────────────────────

_HEADING_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)
_YAML_FENCE_RE = re.compile(r"```ya?ml\s*\n(.*?)```", re.DOTALL)
_KNOWN_FIELDS = {
    "system_prompt",
    "prompt_file",
    "tools_include",
    "tools_exclude",
    "model_alias",
    "memory_mode",
}


def _agent_key(name: str) -> str:
    """Normalise agent name to a registry key: lowercase, spaces → underscores."""
    return re.sub(r"\s+", "_", name.strip().lower())


def _parse_yaml_block(raw: str, agents_dir: Path) -> dict[str, Any]:
    """Parse a yaml block and return a dict of known fields."""
    if not _HAS_YAML:
        log.warning("PyYAML not installed — skipping yaml agent config block")
        return {}
    assert _yaml is not None
    try:
        data = _yaml.safe_load(raw)
    except Exception as exc:
        log.warning("Malformed AGENTS.md yaml block: %s", exc)
        return {}
    if not isinstance(data, dict):
        log.warning("AGENTS.md yaml block is not a mapping — ignoring")
        return {}

    out: dict[str, Any] = {}
    for key in _KNOWN_FIELDS:
        if key not in data:
            continue
        val = data[key]
        if key in ("tools_include", "tools_exclude"):
            if isinstance(val, list):
                out[key] = [str(v) for v in val]
            elif val is None:
                out[key] = []
            else:
                out[key] = [str(val)]
        elif key == "prompt_file":
            if val:
                pf = agents_dir / str(val)
                try:
                    out["system_prompt"] = pf.read_text(encoding="utf-8")
                    out["prompt_file"] = str(val)
                except OSError as exc:
                    log.warning("Cannot read prompt_file %s: %s", pf, exc)
        else:
            out[key] = str(val) if val is not None else ""
    return out


def load_agents_md(path: str | Path) -> dict[str, AgentDefinition]:
    """Parse *path* (AGENTS.md) and return a dict of AgentDefinition keyed by name.

    Returns an empty dict if the file does not exist or cannot be parsed.
    Each ``## Heading`` starts a new agent block.  Unknown agents or malformed
    yaml blocks produce a warning and are skipped.
    """
    path = Path(path)
    if not path.exists():
        return {}

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("Cannot read AGENTS.md at %s: %s", path, exc)
        return {}

    agents_dir = path.parent
    headings = list(_HEADING_RE.finditer(text))
    if not headings:
        return {}

    result: dict[str, AgentDefinition] = {}

    for idx, match in enumerate(headings):
        name = match.group(1).strip()
        key = _agent_key(name)
        section_start = match.end()
        section_end = headings[idx + 1].start() if idx + 1 < len(headings) else len(text)
        section = text[section_start:section_end]

        # Extract description: text between the heading and the first yaml block
        yaml_match = _YAML_FENCE_RE.search(section)
        if yaml_match:
            desc_raw = section[: yaml_match.start()]
        else:
            desc_raw = section

        description = desc_raw.strip()

        # Parse yaml config
        cfg: dict[str, Any] = {}
        if yaml_match:
            cfg = _parse_yaml_block(yaml_match.group(1), agents_dir)

        agent = AgentDefinition(
            name=name,
            description=description,
            system_prompt=cfg.get("system_prompt", ""),
            prompt_file=cfg.get("prompt_file", ""),
            tools_include=cfg.get("tools_include", []),
            tools_exclude=cfg.get("tools_exclude", []),
            model_alias=cfg.get("model_alias", ""),
            memory_mode=cfg.get("memory_mode", ""),
        )
        result[key] = agent

    return result


# ── Registry ──────────────────────────────────────────────────────────────────

_agents_registry: dict[str, AgentDefinition] = {}


def get_agents_md_path() -> Path | None:
    """Return the first AGENTS.md found in cwd or home directory, or None."""
    for candidate in (Path.cwd() / "AGENTS.md", Path.home() / "AGENTS.md"):
        if candidate.exists():
            return candidate
    return None


def load_default_agents() -> dict[str, AgentDefinition]:
    """Load agents from the default AGENTS.md path and update the registry.

    Returns the loaded dict (may be empty if no file is found).
    """
    global _agents_registry
    p = get_agents_md_path()
    if p is None:
        _agents_registry = {}
        return {}
    agents = load_agents_md(p)
    _agents_registry = agents
    if agents:
        log.debug("Loaded %d agent(s) from %s", len(agents), p)
    return agents


def get_agent(name: str) -> AgentDefinition | None:
    """Return the AgentDefinition for *name* (normalised key) or None."""
    return _agents_registry.get(_agent_key(name))


def list_agents() -> list[AgentDefinition]:
    """Return all loaded AgentDefinition objects."""
    return list(_agents_registry.values())
