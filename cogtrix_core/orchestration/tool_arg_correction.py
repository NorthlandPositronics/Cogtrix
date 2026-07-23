"""Tool-argument correction utilities for the orchestration graph.

Extracted from ``cogtrix_core/orchestration/graph.py`` as the third step of the
graph.py 5-module split proposed by the /forge audit
(architect finding A1.3, 2026-05-23). All helpers operate on a single
tool object + its incoming args dict and have no graph-build /
langgraph-runtime dependency.

Public surface (preserves leading-underscore names from graph.py for
back-compat with existing imports in nodes/process_tools.py and the
test suite):

* :func:`_correct_tool_args` — best-effort correction of misnamed /
  mistyped tool arguments. Combines Pydantic alias resolution, a
  static well-known-remap table, fuzzy name matching with a
  blocklist, and type coercion for list/dict ↔ string mismatches.
* :func:`_safe_tool_name` — sanitiser used when echoing
  model-supplied tool names back to the model.
* ``_tool_arg_schema_cache`` — module-level cache of per-tool schema
  introspection. Bounded by ``_TOOL_ARG_SCHEMA_CACHE_MAX_SIZE`` with
  FIFO eviction. Guarded by ``_tool_arg_cache_lock``.
* ``_FUZZY_ARG_BLOCKLIST`` — short common names ("data", "path", etc.)
  that the fuzzy matcher refuses to remap because they collide too
  often with unrelated tool fields.
"""

from __future__ import annotations

import json as _json
import re
import threading
import types
import typing
from difflib import SequenceMatcher
from typing import Any

from cogtrix_core.logging_config import get_logger

# Cache for _correct_tool_args schema introspection results.
# Keyed by logical schema identity (tool name + sorted field names) so MCP
# reconnects that recreate equivalent Pydantic models reuse the same cache entry.
_ToolArgSchemaCacheKey = tuple[str, tuple[str, ...]]
_TOOL_ARG_SCHEMA_CACHE_MAX_SIZE = 512
_tool_arg_schema_cache: dict[
    _ToolArgSchemaCacheKey, tuple[dict[str, Any], dict[str, str], dict[str, str]]
] = {}
_tool_arg_cache_lock = threading.Lock()

_FUZZY_ARG_BLOCKLIST: frozenset[str] = frozenset(
    {
        "data",
        "name",
        "port",
        "code",
        "type",
        "text",
        "path",
        "file",
        "mode",
        "size",
        "body",
        "host",
        "user",
        "role",
        "args",
        "keys",
    }
)

# #1862: common-suffix mapping for quantifier-style arg names. Weak models
# routinely emit a wrong suffix on a shared prefix — ``max_points`` for
# ``max_results``, ``num_items`` for ``num_results``, ``max_records`` for
# ``max_results``. SequenceMatcher misses these because the suffix differs
# in nearly every character. We map them only when (a) the prefix matches
# exactly and ends at an underscore boundary, AND (b) BOTH the model's
# suffix and the schema's expected suffix are in the equivalent-
# quantifier set below. That keeps the rule from ever mapping
# semantically distinct fields like ``min_results`` → ``max_results``
# (different prefixes) or ``max_depth`` → ``max_results`` (suffix not in
# the set).
_EQUIVALENT_QUANTIFIER_SUFFIXES: frozenset[str] = frozenset(
    {
        "results",
        "points",
        "items",
        "count",
        "limit",
        "entries",
        "records",
        "rows",
        "matches",
        "hits",
        "values",
        "elements",
        "objects",
    }
)
_MIN_QUANTIFIER_PREFIX_LEN = 3  # avoid mapping single-letter prefixes like ``n_``


# #1862: antonym-prefix collision guard. Pairs like ``min_results`` /
# ``max_results`` score 0.82 on SequenceMatcher (above the 0.75 remap
# threshold) and would be silently flipped to their semantic opposite —
# a real data hazard. When two names share every underscore-bounded
# component except ONE position, and at that position the parts are both
# in the antonym set, refuse to remap regardless of the character ratio.
_ANTONYM_PREFIXES: frozenset[str] = frozenset(
    {
        "min",
        "max",
        "lower",
        "upper",
        "first",
        "last",
        "start",
        "end",
        "src",
        "dst",
        "source",
        "dest",
        "before",
        "after",
        "top",
        "bottom",
    }
)


def _is_antonym_collision(unk_lower: str, exp_lower: str) -> bool:
    """Return True when ``unk_lower`` and ``exp_lower`` differ in EXACTLY one
    underscore-bounded component and at that position the two parts are
    both well-known antonyms (``min``↔``max``, ``start``↔``end``, …)."""
    unk_parts = unk_lower.split("_")
    exp_parts = exp_lower.split("_")
    if len(unk_parts) != len(exp_parts):
        return False
    diff_positions = [
        i for i, (u, e) in enumerate(zip(unk_parts, exp_parts, strict=True)) if u != e
    ]
    if len(diff_positions) != 1:
        return False
    i = diff_positions[0]
    return unk_parts[i] in _ANTONYM_PREFIXES and exp_parts[i] in _ANTONYM_PREFIXES


def _is_equivalent_quantifier_pair(unk_lower: str, exp_lower: str) -> bool:
    """Return True when ``unk_lower`` and ``exp_lower`` differ only in a
    quantifier-noun suffix on the same underscore-bounded prefix.

    Examples that match (safe to remap):
        ``max_points`` ↔ ``max_results``
        ``num_items``  ↔ ``num_results``
        ``max_records`` ↔ ``max_results``

    Examples that DO NOT match (deliberately):
        ``min_results`` ↔ ``max_results`` (different prefixes)
        ``max_depth``   ↔ ``max_results`` (suffix not in the set)
        ``n_points``    ↔ ``max_results`` (prefix below length floor)
    """
    if "_" not in unk_lower or "_" not in exp_lower:
        return False
    unk_prefix, _, unk_suffix = unk_lower.rpartition("_")
    exp_prefix, _, exp_suffix = exp_lower.rpartition("_")
    if not unk_prefix or unk_prefix != exp_prefix:
        return False
    if len(unk_prefix) < _MIN_QUANTIFIER_PREFIX_LEN:
        return False
    if unk_suffix == exp_suffix:
        return False
    return (
        unk_suffix in _EQUIVALENT_QUANTIFIER_SUFFIXES
        and exp_suffix in _EQUIVALENT_QUANTIFIER_SUFFIXES
    )


def _correct_tool_args(tool: Any, args: dict) -> dict:
    """Best-effort correction of misnamed tool arguments.

    Weaker LLMs sometimes send wrong parameter names (e.g. ``cmd`` instead of
    ``command``).  This function compares provided keys against the tool's
    Pydantic ``args_schema`` and applies two heuristics:

    1. **Fuzzy name match** — uses substring containment and SequenceMatcher
       to remap unknown arg names to the closest expected field.
    2. **Type coercion** — if the schema expects ``str`` and the value is a
       ``list`` or ``dict``, serialise it to a JSON string.

    Returns the (possibly corrected) args dict.  On any error, returns the
    original args unchanged.
    """
    schema = getattr(tool, "args_schema", None)
    if schema is None:
        return args

    try:
        expected: dict[str, Any] = {}
        if hasattr(schema, "model_fields"):
            expected = schema.model_fields  # Pydantic v2
        elif hasattr(schema, "__fields__"):
            expected = schema.__fields__  # Pydantic v1
        if not expected:
            return args
    except (AttributeError, TypeError) as exc:
        tool_name = str(getattr(tool, "name", "") or "<unknown_tool>")
        get_logger().warning(
            "_correct_tool_args: schema introspection failed for tool %r: %s — returning args unchanged",
            tool_name,
            exc,
        )
        return args

    expected = dict(expected)
    tool_name = str(getattr(tool, "name", "") or "<unknown_tool>")
    cache_key: _ToolArgSchemaCacheKey = (tool_name, tuple(sorted(expected.keys())))

    with _tool_arg_cache_lock:
        _cached = _tool_arg_schema_cache.get(cache_key)
        if _cached is not None:
            expected, alias_map, _well_known_remaps_cached = _cached
        else:
            # --- Alias resolution -------------------------------------------------
            # Pydantic aliases (Field(alias=...)) are not visible as field names.
            # Map known aliases to their canonical field name so LLMs that send the
            # alias (e.g. "cmd" instead of "command") get corrected before fuzzy match.
            alias_map: dict[str, str] = {}
            for fname, finfo in expected.items():
                _alias = getattr(finfo, "alias", None)
                if _alias and _alias != fname:
                    alias_map[_alias] = fname
                # Also check validation_alias (Pydantic v2)
                _valias = getattr(finfo, "validation_alias", None)
                if isinstance(_valias, str) and _valias != fname:
                    alias_map[_valias] = fname

            _well_known_remaps_cached: dict[str, str] = {}
            # Evict oldest entries if cache exceeds max size (FIFO eviction)
            if len(_tool_arg_schema_cache) >= _TOOL_ARG_SCHEMA_CACHE_MAX_SIZE:
                # Pop the first (oldest) item - Python 3.7+ dicts maintain insertion order
                _tool_arg_schema_cache.pop(next(iter(_tool_arg_schema_cache)))
            _tool_arg_schema_cache[cache_key] = (expected, alias_map, _well_known_remaps_cached)

    expected_names = set(expected.keys())
    provided_names = set(args.keys())

    corrected = dict(args)

    for alias_key, canonical in alias_map.items():
        if alias_key in corrected and canonical not in corrected:
            corrected[canonical] = corrected.pop(alias_key)
            log = get_logger()
            log.info("Tool arg alias resolved: '%s' → '%s'", alias_key, canonical)

    # --- Well-known parameter variations ────────────────────────────
    # LLMs frequently use common synonyms that fall below the fuzzy
    # threshold (0.75).  Explicit remaps for the most common cases.
    _WELL_KNOWN_REMAPS: dict[str, list[str]] = {
        "filename": ["path"],
        "file_path": ["path"],
        "filepath": ["path"],
        "file_name": ["path"],
        "file_content": ["content"],
        "text": ["content", "prompt"],
        "body": ["content"],
        "cmd": ["command"],
        "query_string": ["query"],
        "search_query": ["query"],
        "dir": ["path"],
        "directory": ["path"],
        # Additional common LLM variants (ratio 0.75–0.84 — below old threshold)
        "infile": ["input_file"],
        "input_file": ["infile"],
        "workdir": ["working_dir"],
        "working_dir": ["workdir"],
        "verbose": ["verbosity"],
        "verbosity": ["verbose"],
        "filenamestr": ["file_name"],
        # cron_add: LLMs commonly use "pattern" for cron expressions (#520)
        "pattern": ["schedule"],
        "expression": ["schedule"],
        # GitHub PR tools: LLMs use pr_number / number for pull_number
        "pr_number": ["pull_number"],
        "pull_request_number": ["pull_number"],
        # list_pull_requests: LLMs use status for state
        "status": ["state"],
        # Tools that expect "prompt" but LLM sends content/message
        "content": ["prompt"],
        "message": ["prompt"],
        # checkpoint(finding=...): LLMs (observed: kimi-k2-5 on
        # regression_multi_turn_effort_gate_no_carryover, PR #1723
        # Gate 2 shard D failure) frequently emit synonyms when
        # recording progress observations. SequenceMatcher ratio for
        # 'reason' ↔ 'finding' is ~0.15, far below the 0.75 fuzzy
        # threshold, so without explicit remaps these names hard-fail
        # at pydantic validation and the scenario is marked failed
        # (runner.py:910 — any tool error fails the run, by design).
        "reason": ["finding"],
        "note": ["finding"],
        "observation": ["finding"],
        "discovery": ["finding"],
        "result": ["finding"],
        "conclusion": ["finding"],
        "summary": ["finding"],
        "outcome": ["finding"],
    }
    for provided_key in list(corrected.keys()):
        if provided_key in expected_names:
            continue  # already matches a field — skip
        for canonical in _WELL_KNOWN_REMAPS.get(provided_key, []):
            if canonical in expected_names and canonical not in corrected:
                corrected[canonical] = corrected.pop(provided_key)
                log = get_logger()
                log.info("Tool arg well-known remap: '%s' → '%s'", provided_key, canonical)
                break

    provided_names = set(corrected.keys())

    # --- Name remapping ---------------------------------------------------
    unknown = provided_names - expected_names
    missing = expected_names - provided_names

    if unknown and missing:
        _REMAP_THRESHOLD = 0.75
        for unk in unknown:
            unk_lower = unk.lower()
            best: str | None = None
            best_ratio = 0.0
            tied = False
            for exp in missing:
                exp_lower = exp.lower()
                # #1862: refuse antonym-prefix pairs (min↔max, start↔end,
                # …) regardless of character ratio. They score high on
                # SequenceMatcher but a silent remap would flip semantics.
                if _is_antonym_collision(unk_lower, exp_lower):
                    continue
                # Substring containment — only trust when the shorter
                # string is long enough to be meaningful.
                shorter_len = min(len(unk_lower), len(exp_lower))
                longer_len = max(len(unk_lower), len(exp_lower))
                if (
                    shorter_len >= 5
                    and shorter_len / longer_len >= 0.5
                    and unk_lower not in _FUZZY_ARG_BLOCKLIST
                    and (unk_lower in exp_lower or exp_lower in unk_lower)
                ):
                    ratio = 1.0
                elif _is_equivalent_quantifier_pair(unk_lower, exp_lower):
                    # #1862: shared exact prefix at underscore boundary +
                    # both suffixes are quantifier nouns → safe to remap.
                    ratio = 1.0
                else:
                    ratio = SequenceMatcher(None, unk_lower, exp_lower).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best = exp
                    tied = False
                elif abs(ratio - best_ratio) < 1e-9 and ratio >= _REMAP_THRESHOLD:
                    tied = True
            if best is not None and best_ratio >= _REMAP_THRESHOLD and not tied:
                corrected[best] = corrected.pop(unk)
                missing.discard(best)
                log = get_logger()
                log.info("Tool arg corrected: '%s' → '%s' (score=%.2f)", unk, best, best_ratio)

    # --- Type coercion: schema expects list but got JSON-encoded string → decode.
    import json as _json_mod

    for key, value in list(corrected.items()):
        if key not in expected:
            continue
        if not isinstance(value, str):
            continue
        field_info = expected[key]
        annotation = getattr(field_info, "annotation", None) or getattr(
            field_info, "outer_type_", None
        )
        origin = typing.get_origin(annotation)
        if origin is typing.Union or isinstance(annotation, types.UnionType):
            type_args = [a for a in typing.get_args(annotation) if a is not type(None)]
            if len(type_args) == 1:
                annotation = type_args[0]
        if annotation is list or (typing.get_origin(annotation) is list):
            stripped = value.strip()
            if stripped.startswith("["):
                try:
                    parsed = _json_mod.loads(stripped)
                    if isinstance(parsed, list):
                        corrected[key] = parsed
                        log = get_logger()
                        log.debug("Tool arg '%s' coerced from JSON string to list", key)
                except (ValueError, KeyError):
                    pass

    # --- Type coercion: schema expects str but got list/dict → JSON-encode.
    for key, value in list(corrected.items()):
        if key not in expected:
            continue
        if not isinstance(value, (list, dict)):
            continue
        field_info = expected[key]
        annotation = getattr(field_info, "annotation", None) or getattr(
            field_info, "outer_type_", None
        )
        # Unwrap Optional[str] / str | None → str
        origin = typing.get_origin(annotation)
        if origin is typing.Union or isinstance(annotation, types.UnionType):
            type_args = [a for a in typing.get_args(annotation) if a is not type(None)]
            if len(type_args) == 1:
                annotation = type_args[0]
        if annotation is str:
            if isinstance(value, list) and all(isinstance(v, str) for v in value):
                corrected[key] = " ".join(value)
            else:
                corrected[key] = _json.dumps(value)

    return corrected


def _safe_tool_name(name: str, max_len: int = 80) -> str:
    """Strip everything except word chars, hyphens, and dots; truncate.

    Prevents model-supplied names from carrying injection payload into
    guidance messages that are fed back to the model.
    """
    sanitized = re.sub(r"[^\w\-\.]", "", name)
    return sanitized[:max_len] if sanitized else "<unknown>"


# url-fetch tools that take a concrete ``url`` (not a search query).
_URL_FETCH_TOOLS: frozenset[str] = frozenset({"http_get", "http_post"})

# Argument names a model reaches for when it mistakes a url-fetch tool for a
# web-search tool. Kept tight (search-intent words) to stay high-precision.
_QUERY_LIKE_KEYS: tuple[str, ...] = (
    "query",
    "q",
    "search",
    "search_query",
    "query_string",
    "search_terms",
    "keywords",
)

# A value that already looks like a URL/host is a field-naming mistake, not
# search misuse — don't redirect those to web_search.
_URL_LIKE_RE = re.compile(
    r"^\s*(?:https?://|ftp://|www\.|[\w-]+\.[a-z]{2,}(?:[/:]|$))", re.IGNORECASE
)


def detect_url_tool_misuse(tool_name: str, args: Any) -> str | None:
    """Return a redirect message when a url-fetch tool is called as a search (#2293).

    Weaker / open-weight models (qwen3 observed in production v0.4.1 logs, ~1-in-4
    ``http_get`` calls) treat ``http_get``/``http_post`` like ``web_search`` —
    passing a natural-language ``query`` with no ``url``. The call then hard-fails
    Pydantic with a cryptic ``url Field required``; weak models retry the same bad
    shape and loop. This detector spots the misuse so the dispatcher can return an
    actionable redirect to ``web_search`` instead.

    Fires ONLY when:
      * the tool is a url-fetch tool, AND
      * there is no usable ``url`` (missing / empty / not URL-shaped), AND
      * a query-like key holds a non-empty string that is NOT itself URL-shaped
        (a URL-shaped value is a field-naming slip, left to ``_correct_tool_args``).

    Returns the guidance string, or ``None`` when it is not this misuse.
    """
    if tool_name not in _URL_FETCH_TOOLS or not isinstance(args, dict):
        return None
    url_val = args.get("url")
    if isinstance(url_val, str) and _URL_LIKE_RE.match(url_val):
        return None  # a real url is present — normal call
    for key in _QUERY_LIKE_KEYS:
        val = args.get(key)
        if isinstance(val, str) and val.strip() and not _URL_LIKE_RE.match(val):
            snippet = val.strip()
            if len(snippet) > 80:
                snippet = snippet[:77] + "..."
            return (
                f"'{_safe_tool_name(tool_name)}' fetches a single web page and needs a "
                f"concrete 'url' (e.g. https://example.com), not a search query. It did "
                f'NOT run. To find information about "{snippet}", call '
                f"web_search(query=...) first, then '{_safe_tool_name(tool_name)}' the "
                f"specific result URL you want to read."
            )
    return None


__all__ = [
    "_FUZZY_ARG_BLOCKLIST",
    "_TOOL_ARG_SCHEMA_CACHE_MAX_SIZE",
    "_ToolArgSchemaCacheKey",
    "_correct_tool_args",
    "_safe_tool_name",
    "_tool_arg_cache_lock",
    "_tool_arg_schema_cache",
    "detect_url_tool_misuse",
]
