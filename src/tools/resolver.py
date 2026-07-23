"""Canonical fuzzy tool-name resolver.

Single source of truth for all tool-name resolution across Cogtrix.
Used by both the orchestration graph (process_tools node) and the
request_tools meta-tool factory.
"""

from __future__ import annotations

from typing import Any

FUZZY_MATCH_THRESHOLD = (
    0.65  # raised from 0.40; blocks read_file↔write_file (0.63), search_web↔search_news (0.63)
)


def _is_word_contained(short: str, long: str) -> bool:
    """True when *short* appears in *long* on underscore word boundaries."""
    if short == long:
        return True
    return long.startswith(short + "_") or long.endswith("_" + short) or f"_{short}_" in long


def resolve_tool_name(
    requested: str,
    available_tools: dict[str, Any],
    active_tool_names: set[str] | None = None,
) -> tuple[str | None, str]:
    """Map *requested* (the name the LLM used) to an actual registered tool.

    Resolution order:
      1. Exact match in the on-demand pool  → ``("name", "available")``
      2. Exact match among active tools     → ``("name", "active")``
      3. Fuzzy match (token overlap /
         substring containment)             → ``("name", "available"|"active")``
      4. No match                           → ``(None, "none")``

    Args:
        requested: The raw tool name produced by the LLM.
        available_tools: ``{name: tool}`` dict of on-demand (not yet loaded) tools.
        active_tool_names: Names of tools currently loaded in the agent.
            Pass ``None`` or an empty set when only searching the available pool.

    Returns:
        ``(resolved_name, source)`` where *source* is one of ``"available"``,
        ``"active"``, or ``"none"``.
    """
    _active: set[str] = active_tool_names or set()

    if requested in available_tools:
        return requested, "available"
    if requested in _active:
        return requested, "active"

    req_norm = requested.lower().replace("-", "_")
    req_tokens = set(req_norm.split("_"))

    best: tuple[str | None, float, str] = (None, 0.0, "none")

    pools: list[tuple[dict[str, Any] | set[str], str]] = [
        (available_tools, "available"),
        (_active, "active"),
    ]
    for pool, source in pools:
        for tool_name in pool:
            tn_norm = tool_name.lower().replace("-", "_")
            tn_tokens = set(tn_norm.split("_"))

            intersection = req_tokens & tn_tokens
            union = req_tokens | tn_tokens
            score = len(intersection) / len(union) if union else 0.0

            if _is_word_contained(req_norm, tn_norm) or _is_word_contained(tn_norm, req_norm):
                score += 0.40
            if req_tokens.issubset(tn_tokens) or tn_tokens.issubset(req_tokens):
                score += 0.20

            _prefix_hit = any(
                (len(a) >= 3 and len(b) >= 3 and a != b and (b.startswith(a) or a.startswith(b)))
                for a in req_tokens
                for b in tn_tokens
            )
            if _prefix_hit:
                score += 0.35

            if score > best[1]:
                best = (tool_name, score, source)

    if best[1] >= FUZZY_MATCH_THRESHOLD:
        return best[0], best[2]
    return None, "none"
