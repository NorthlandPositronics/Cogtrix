"""Integrity guard for the API dangerous-tool deny set (#2118).

``_API_DENIED_DANGEROUS_TOOLS`` mixes two kinds of entries with different roles:

* canonical, agent-callable tool names (``execute_shell_command`` /
  ``execute_python``) — the load-bearing entries that ``is_denied`` matches;
* module/alias name strings (``shell`` / ``bash`` / ``python_exec``) — which do
  NOT match ``is_denied`` (a no-op there) but DO guard the tools-route
  ``load``/``enable`` block.

These tests make the split explicit and **fail loud on drift**:
* a canonical entry must resolve to a really-registered tool (a rename that
  silently re-opened the RCE hole would break this);
* an alias must NOT be a canonical tool name — if a tool literally named
  ``shell`` is ever registered, a previously-inert no-op would silently become
  load-bearing, and this test forces a re-evaluation.
"""

from __future__ import annotations

from pathlib import Path

from cogtrix_core.api.session_bridge import (
    _API_DENIED_DANGEROUS_TOOL_ALIASES,
    _API_DENIED_DANGEROUS_TOOLS,
    _API_DENIED_EXEC_TOOLS,
)


def _all_canonical_tool_names() -> set[str]:
    """Canonical tool names declared across ``cogtrix_core/tools/*.py`` (AST scan, no import)."""
    import cogtrix_core.registry as registry

    tools_dir = Path(registry.__file__).parent / "tools"
    names: set[str] = set()
    for file_path in sorted(tools_dir.glob("*.py")):
        if file_path.name == "__init__.py" or file_path.name.startswith("_"):
            continue
        for entry in registry._scan_tool_metadata_from_file(file_path):
            name = entry.get("name")
            if name:
                names.add(name)
    return names


def test_combined_set_is_exactly_canonical_plus_aliases() -> None:
    assert (
        _API_DENIED_DANGEROUS_TOOLS == _API_DENIED_EXEC_TOOLS + _API_DENIED_DANGEROUS_TOOL_ALIASES
    )


def test_exec_and_alias_tuples_are_disjoint() -> None:
    assert not (set(_API_DENIED_EXEC_TOOLS) & set(_API_DENIED_DANGEROUS_TOOL_ALIASES))


def test_canonical_exec_names_resolve_to_registered_tools() -> None:
    """Drift guard: a rename of the exec tools must not silently re-open the
    deny (the #2050 RCE hole). Every canonical entry must be a real tool name."""
    canonical = _all_canonical_tool_names()
    # Sanity: the scan found a non-trivial registry.
    assert "execute_shell_command" in canonical and "execute_python" in canonical
    for name in _API_DENIED_EXEC_TOOLS:
        assert name in canonical, (
            f"deny-set canonical entry {name!r} no longer matches a registered tool — "
            "is_denied() would be a silent no-op (RCE-reopen risk, #2050/#2118)"
        )


def test_aliases_are_not_canonical_tool_names() -> None:
    """If an alias ever becomes a real canonical tool, the previously-inert
    string entry silently turns load-bearing — force a re-evaluation here."""
    canonical = _all_canonical_tool_names()
    for alias in _API_DENIED_DANGEROUS_TOOL_ALIASES:
        assert alias not in canonical, (
            f"alias {alias!r} is now a registered canonical tool — move it to "
            "_API_DENIED_EXEC_TOOLS so is_denied() actually enforces it (#2118)"
        )
