"""Agent runner — orchestrates a single prompt-to-response cycle.

Houses the main ``run_agent`` entry point and the helper functions it
delegates to: response extraction, error formatting, tool-call logging,
and phantom-call detection.
"""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future
from typing import Any

from src.agent.core import prepare_messages_with_context
from src.agent.safety import AgentExecutionError, UserCancelledRun
from src.logging_config import get_logger, is_trace, is_verbose, log_tool_call
from src.orchestration.compression import (
    COMPRESSION_MIN_AGE_CYCLES,
    COMPRESSION_MIN_CHARS,
    _get_compression_pool,
    apply_message_compression,
)
from src.orchestration.graph import DEFAULT_RECURSION_LIMIT, build_agent_graph
from src.orchestration.intent import (
    OwnershipMode,
    OwnershipResult,
    TaskComplexity,
    classify_task_complexity,
    classify_task_ownership,
)
from src.orchestration.run_config import AgentRunConfig
from src.tools.extend_run import ExtendRunState

# Persistent caches that survive across graph rebuilds.
# _persistent_bound_cache stores LLM bind_tools() results (expensive to recreate).
# _persistent_compression_cache stores compressed tool message summaries.
# _cached_llm_id tracks which LLM generation the bound cache was built for;
# when the LLM changes, advance_llm_generation() is called and the cache is cleared.
_MAX_COMPRESSION_CACHE_SIZE = 256
_MAX_BOUND_CACHE_SIZE = 16
_persistent_bound_cache: OrderedDict = OrderedDict()
_persistent_compression_cache: OrderedDict = OrderedDict()
_cached_llm_id: tuple[int, int] | None = None
_cache_lock = threading.Lock()
_llm_generation: int = 0
_pending_background_compression_jobs: list[
    tuple[Future[OrderedDict[str, str]], OrderedDict[str, str]]
] = []

# Compiled graph cache — avoids the ~25ms StateGraph.compile() cost per turn.
# Keyed by (llm_id, llm_generation, active_fingerprint, available_fingerprint,
# system_prompt_hash).  When the key matches, _reset_for_new_run() is called
# to refresh per-run mutable state before the graph is reused.
# Only used in CLI mode (not API mode, which manages per-session state
# independently via AgentRunConfig.bound_cache / compression_cache).
_MAX_GRAPH_CACHE_SIZE = 4
_persistent_graph_cache: OrderedDict = OrderedDict()  # key → compiled graph

# Simple-task preload set. Intentionally excludes "search_web" since
# PR-G (ADR-0056) retired the legacy in-process DDG tool from the
# agent catalogue — `web_search` (subprocess-isolated) supersedes it.
# Listing "search_web" here is a footgun: if anything ever re-registers
# the legacy name, this preload would auto-load the in-process variant
# and re-introduce the Bug D / cogtrix46 heap-corruption crash (curl_cffi
# loaded into a process that also has httpx). The modern `web_search`
# tool intentionally lives outside this preload — it is loaded lazily
# via `request_tools` for complex tasks (see the COMPLEX_ACTION /
# COMPLEX_RESEARCH branch below) so its subprocess overhead is paid
# only when actually needed.
_SIMPLE_PRELOAD_TOOLS: tuple[str, ...] = (
    "calculate",
    "read_file",
    "get_current_datetime",
)

# File-mutation tool set (bug #1870). Pre-loaded when the prompt signals
# destructive-file intent (see `_query_signals_file_write_intent`).
# `read_file` is included because most realistic edits via `patch_file`
# require reading the current content first to know the ``old_string``
# to replace. There is intentionally no `delete_file` tool in Cogtrix —
# pure-delete intents cannot be satisfied by this preload; they must be
# caught by the #1869 fabricated-success detector instead.
_FILE_WRITE_PRELOAD_TOOLS: tuple[str, ...] = (
    "write_file",
    "patch_file",
    "append_file",
    "read_file",
)

# Real-time / recency markers (bug #1839). The task-complexity classifier
# scores *linguistic* complexity, so a short factual question like
# "What's the current Apple stock price?" lands in SIMPLE and never gets
# `web_search` auto-loaded — leaving the agent to deflect with "I don't
# have real-time data". These markers add an orthogonal *information-
# recency* signal: when a prompt is asking for fresh external data, the
# retrieval tool set is force-loaded regardless of complexity score.
# Kept high-precision (strong temporal markers + clear real-time domains)
# so we don't pay web_search's subprocess cost on ordinary prompts.
_REALTIME_QUERY_MARKERS = re.compile(
    r"\b("
    r"current(?:ly)?|latest|most[ -]recent|today'?s?|tonight|"
    r"right[ -]now|as[ -]of[ -](?:now|today|this)|"
    r"this[ -](?:week|month|year|morning|afternoon|evening)|"
    r"up[ -]?to[ -]?date|real[ -]?time|"
    r"stock|share[ -]price|\bquote\b|exchange[ -]rate|"
    r"weather|forecast|\bnews\b|headlines|\bscores?\b"
    r")\b",
    re.IGNORECASE,
)

# File-mutation intent markers (bug #1870). The task-complexity classifier
# only catches build/install/deploy + research patterns, so a destructive
# prompt like "delete /workspace/foo.py" or "add a function to bar.py"
# lands in SIMPLE/MODERATE with no pre-load. The agent then sees only
# ``get_current_datetime`` + ``request_tools`` (the ``code`` memory-mode
# preset) and — per Q9/Q10 of the #1869 holistic-test battery — tends to
# silently fabricate success instead of nudging ``request_tools``.
#
# These markers add an orthogonal *destructive-intent* signal: when a
# prompt asks the agent to mutate a file (write/patch/append/modify/
# edit/overwrite/create/delete/remove/change/add) and the target is
# clearly file-shaped (literal "file"/"directory"/"folder" token, an
# apparent path with a recognised extension, or a multi-segment slash
# path), the mutation tool set is force-loaded regardless of complexity.
#
# Verbs and targets must appear within 80 characters of each other,
# mirroring the proximity guard used by COMPLEX_ACTION detection in
# :func:`classify_task_complexity` — this suppresses false positives on
# distant verb/path mentions like long prose with a closing reference
# to a research note path.
_FILE_WRITE_INTENT_VERBS = re.compile(
    r"\b(?:"
    r"writ(?:e|es|ing)|"
    r"patch(?:es|ing)?|"
    r"append(?:s|ing)?|"
    r"modif(?:y|ies|ying)|"
    r"edit(?:s|ing)?|"
    r"overwrit(?:e|es|ing)|"
    r"creat(?:e|es|ing)|"
    r"delet(?:e|es|ing)|"
    r"remov(?:e|es|ing)|"
    r"chang(?:e|es|ing)|"
    r"add(?:s|ing)?"
    r")\b",
    re.IGNORECASE,
)

_FILE_WRITE_INTENT_TARGETS = re.compile(
    r"(?:"
    # Explicit "file" / "files" / "directory" / "folder" / "dir" tokens
    r"\b(?:file|files|folder|folders|directory|directories|dir)\b" r"|"
    # Apparent file path with a recognised source/config extension.
    # Extension list intentionally bounded — we don't want short English
    # words to slip through. Each alternative requires the dot before it
    # (``\.``) so plain words can't match.
    r"\b[\w./_-]+\.(?:py|pyi|pyx|js|ts|tsx|jsx|java|c|h|cc|cpp|cxx|hpp|cs|"
    r"go|rs|rb|swift|kt|kts|m|mm|php|pl|sh|bash|zsh|fish|"
    r"json|jsonc|yaml|yml|toml|ini|cfg|conf|md|markdown|rst|txt|"
    r"html|htm|css|scss|sass|less|xml|csv|tsv|tab|log|sql|"
    r"gradle|cmake|make|mk|env|properties|service|lock|"
    r"pb|proto|graphql|gql|tf|tfvars|hcl|nix|dockerfile|"
    r"jinja|j2|tmpl|template)\b"
    r"|"
    # Multi-segment absolute or rooted path. At least two leading slash
    # segments so we don't false-positive on token pairs like "to/from".
    r"(?:/[\w.-]+){2,}/?" r")",
    re.IGNORECASE,
)


def _query_needs_realtime_data(prompt: str) -> bool:
    """Heuristic: does the prompt ask for information past the model's
    training cutoff (current prices, today's weather, latest news, …)?

    Orthogonal to task complexity — a recency-dependent prompt can be
    linguistically trivial yet still require web retrieval. See bug #1839.
    """
    if not prompt:
        return False
    return _REALTIME_QUERY_MARKERS.search(prompt) is not None


def _query_signals_file_write_intent(prompt: str) -> bool:
    """Heuristic: does the prompt ask the agent to write, patch, modify,
    delete, or otherwise mutate a file?

    Orthogonal to task complexity — see bug #1870. Without this signal,
    a short imperative like ``"delete /workspace/foo.py"`` lands in
    SIMPLE / MODERATE and the agent's active tool set stays at the
    ``code`` memory-mode preset (``{get_current_datetime, request_tools}``),
    leaving the mutation tools idle in the catalog and creating a wide
    fabrication surface (Q9/Q10 reproducers of #1869).

    Tight precision: both a mutation verb and a file-shaped target (the
    literal word ``file`` / ``directory`` / ``folder`` / ``dir``, an
    apparent path with a recognised extension, or a multi-segment
    slash path) must appear within 80 characters of each other —
    mirroring the proximity guard used by ``COMPLEX_ACTION`` detection
    in :func:`classify_task_complexity`.
    """
    if not prompt:
        return False
    verb_match = _FILE_WRITE_INTENT_VERBS.search(prompt)
    if not verb_match:
        return False
    target_match = _FILE_WRITE_INTENT_TARGETS.search(prompt)
    if not target_match:
        return False
    return abs(verb_match.start() - target_match.start()) < 80


def _auto_load_web_search(config: AgentRunConfig) -> bool:
    """Move ``web_search`` from the on-demand catalog into the active set.

    Shared by the COMPLEX-task path and the real-time-query path so the
    agent has web research from the first round without a ``request_tools``
    round-trip. The modern ``web_search`` tool is subprocess-isolated for
    the DDG provider (Bug D); the legacy in-process ``search_web`` name was
    retired by PR-G and must never be auto-loaded — see the comment on
    ``_SIMPLE_PRELOAD_TOOLS`` above.

    Returns ``True`` only when the tool was actually moved into the active
    set; ``False`` when it was already active, unavailable, or failed to
    resolve (so callers can log accurately).
    """
    avail = config.available_tools
    active = config.active_tools_list
    if not avail or active is None or "web_search" not in avail:
        return False
    if any(getattr(t, "name", "") == "web_search" for t in active):
        return False
    search_tool = avail.pop("web_search")
    # Resolve LazyToolProxy before adding to active tools — bind_tools()
    # requires real StructuredTool objects.
    if hasattr(search_tool, "_resolve"):
        try:
            search_tool = search_tool._resolve()
        except Exception as exc:
            get_logger().warning("Failed to resolve web_search tool: %s", exc)
            avail["web_search"] = search_tool
            return False
    if search_tool is None:
        return False
    active.append(search_tool)
    return True


def advance_llm_generation() -> None:
    """Increment the LLM generation counter when the LLM is switched."""
    global _llm_generation
    with _cache_lock:
        _llm_generation += 1
        _pending_background_compression_jobs.clear()


def invalidate_llm_caches() -> None:
    """Clear all module-level LLM-related caches — call on provider/model switch.

    This is a no-op for API sessions that supply per-session caches via
    ``AgentRunConfig.bound_cache`` / ``AgentRunConfig.compression_cache``;
    those sessions manage their own cache lifecycle independently.
    """
    global _llm_generation
    with _cache_lock:
        _llm_generation += 1
        _persistent_bound_cache.clear()
        _persistent_compression_cache.clear()
        _persistent_graph_cache.clear()
        _pending_background_compression_jobs.clear()


def _clear_pending_background_compression_jobs() -> None:
    """Drop any queued background compression jobs after an LLM reset."""
    _pending_background_compression_jobs.clear()


def _merge_compression_cache(
    target_cache: OrderedDict[str, str], source_cache: dict[str, str]
) -> None:
    """Merge compressed cache entries into an LRU cache with size bounds."""
    for key, value in source_cache.items():
        target_cache[key] = value
        target_cache.move_to_end(key)
    while len(target_cache) > _MAX_COMPRESSION_CACHE_SIZE:
        target_cache.popitem(last=False)


def _drain_background_compression_jobs(
    target_cache: OrderedDict[str, str] | None = None,
) -> None:
    """Merge any finished warm-up jobs into their target caches without waiting.

    When *target_cache* is supplied, only jobs queued for that exact cache
    object are processed.  This prevents one API session's ``finally`` block
    from draining jobs belonging to another concurrent session.
    """
    log = get_logger()
    finished: list[tuple[Future[OrderedDict[str, str]], OrderedDict[str, str]]] = []
    with _cache_lock:
        if not _pending_background_compression_jobs:
            return
        still_pending: list[tuple[Future[OrderedDict[str, str]], OrderedDict[str, str]]] = []
        for future, job_target_cache in _pending_background_compression_jobs:
            if target_cache is not None and job_target_cache is not target_cache:
                still_pending.append((future, job_target_cache))
                continue
            if future.done():
                finished.append((future, job_target_cache))
            else:
                still_pending.append((future, job_target_cache))
        _pending_background_compression_jobs[:] = still_pending

    for future, job_target_cache in finished:
        try:
            snapshot = future.result()
        except Exception as exc:  # pragma: no cover - defensive background logging
            log.debug("Background compression warm-up failed: %s", exc, exc_info=True)
            continue
        if snapshot:
            with _cache_lock:
                _merge_compression_cache(job_target_cache, snapshot)


def _queue_background_compression(
    messages: list,
    target_cache: OrderedDict[str, str],
    *,
    call_count: int,
    llm: Any,
    max_context_tokens: int | None,
    min_age_cycles: int,
    min_chars: int,
    emergency_threshold: float,
    human_msg_max_chars: int,
    actual_input_tokens: int = 0,
    min_age_override: int | None = None,
) -> None:
    """Warm the compression cache in the background without delaying the turn."""
    if llm is None or max_context_tokens is None or max_context_tokens < 16_384:
        return

    messages_snapshot = list(messages)
    cache_snapshot = OrderedDict(target_cache)

    def _warm() -> OrderedDict[str, str]:
        apply_message_compression(
            messages_snapshot,
            call_count=call_count,
            compression_cache=cache_snapshot,
            llm=llm,
            max_context_tokens=max_context_tokens,
            min_age_cycles=min_age_cycles,
            min_chars=min_chars,
            emergency_threshold=emergency_threshold,
            human_msg_max_chars=human_msg_max_chars,
            actual_input_tokens=actual_input_tokens,
            min_age_override=min_age_override,
        )
        return cache_snapshot

    future = _get_compression_pool().submit(_warm)
    with _cache_lock:
        _pending_background_compression_jobs.append((future, target_cache))


def _auto_load_simple_tools(config: AgentRunConfig) -> None:
    """Preload a small default tool set for simple tasks.

    The agent still starts lean for complex work, but short/simple prompts
    can skip the request_tools bootstrap round-trip when the common tools
    are already in the active tool list.
    """
    available_tools = config.available_tools
    active_tools_list = config.active_tools_list
    if not available_tools or active_tools_list is None:
        return

    active_names = {getattr(tool, "name", "") for tool in active_tools_list}
    loaded: list[str] = []
    for tool_name in _SIMPLE_PRELOAD_TOOLS:
        if tool_name in active_names or tool_name not in available_tools:
            continue

        tool = available_tools.pop(tool_name)
        if hasattr(tool, "_resolve"):
            try:
                tool = tool._resolve()
            except Exception as exc:
                get_logger().warning("Failed to resolve preload tool %r: %s", tool_name, exc)
                available_tools[tool_name] = tool
                continue

        if tool is None:
            available_tools[tool_name] = tool
            continue

        active_tools_list.append(tool)
        active_names.add(tool_name)
        loaded.append(tool_name)

    if loaded:
        get_logger().info("Auto-loaded common tools for simple task: %s", ", ".join(loaded))


def _auto_load_file_write_tools(config: AgentRunConfig) -> bool:
    """Move the file-mutation tool set from the catalog into the active set.

    Mirrors :func:`_auto_load_web_search` for the destructive-file-intent
    signal added for bug #1870. When the user's prompt signals an intent
    to write, patch, append, modify, edit, overwrite, create, delete,
    remove, change, or add to a file (with a file/dir target within 80
    chars of the verb), pre-load the mutation tools so the agent does
    not silently fabricate success when only ``get_current_datetime`` /
    ``request_tools`` are active (the default ``code`` memory-mode preset
    — see ``src/tools/configure.py``).

    ``read_file`` is included alongside the mutators because most
    realistic edits (``patch_file``) require reading the current content
    first to know the ``old_string`` to replace.

    Returns:
        True only when at least one tool was actually moved from the
        catalog into the active set; False otherwise (so callers can
        log accurately).

    Note:
        Cogtrix has no ``delete_file`` tool — only ``write_file``,
        ``patch_file``, ``append_file``. Pure-delete intents (e.g.
        ``rm foo.py``) cannot be satisfied by this preload; they must
        be caught by the #1869 fabricated-success detector and answered
        with an honest "I do not have a delete tool" instead.
    """
    available_tools = config.available_tools
    active_tools_list = config.active_tools_list
    if not available_tools or active_tools_list is None:
        return False

    active_names = {getattr(tool, "name", "") for tool in active_tools_list}
    loaded: list[str] = []
    for tool_name in _FILE_WRITE_PRELOAD_TOOLS:
        if tool_name in active_names or tool_name not in available_tools:
            continue

        tool = available_tools.pop(tool_name)
        if hasattr(tool, "_resolve"):
            try:
                tool = tool._resolve()
            except Exception as exc:
                get_logger().warning(
                    "Failed to resolve file-write preload tool %r: %s", tool_name, exc
                )
                available_tools[tool_name] = tool
                continue

        if tool is None:
            available_tools[tool_name] = tool
            continue

        active_tools_list.append(tool)
        active_names.add(tool_name)
        loaded.append(tool_name)

    if loaded:
        get_logger().info(
            "Auto-loaded file-write tools for destructive-intent prompt: %s",
            ", ".join(loaded),
        )
        return True
    return False


class ToolCallLogger:
    """Callback handler that logs tool calls.

    Tracks invocations by ``call_id`` (LangChain's unique tool-call ID)
    so that concurrent runs of the *same* tool each get an accurate
    duration measurement.
    """

    _STALE_TIMEOUT = 600  # 10 minutes
    _EVICT_INTERVAL = 60.0  # minimum seconds between eviction passes

    def __init__(self) -> None:
        self._tool_start_times: dict[str, float] = {}
        self._lock = threading.Lock()
        self._last_evict: float = 0.0

    def _evict_stale(self) -> None:
        """Remove entries older than ``_STALE_TIMEOUT`` to prevent leaks.

        Must be called with ``self._lock`` already held.
        Rate-limited to at most once per ``_EVICT_INTERVAL`` seconds.
        """
        now = time.monotonic()
        if now - self._last_evict < self._EVICT_INTERVAL:
            return
        self._last_evict = now
        cutoff = now - self._STALE_TIMEOUT
        stale_keys = [k for k, ts in self._tool_start_times.items() if ts < cutoff]
        for k in stale_keys:
            self._tool_start_times.pop(k, None)

    def on_tool_start(self, tool_name: str, tool_input: dict, call_id: str = "") -> None:
        """Log when a tool starts execution."""
        key = call_id or tool_name
        with self._lock:
            self._evict_stale()
            self._tool_start_times[key] = time.monotonic()
        log_tool_call(tool_name, inputs=tool_input)

    def on_tool_end(self, tool_name: str, output: str, call_id: str = "") -> None:
        """Log when a tool finishes execution."""
        key = call_id or tool_name
        duration = None
        with self._lock:
            if key in self._tool_start_times:
                duration = time.monotonic() - self._tool_start_times.pop(key)
        log_tool_call(tool_name, output=output, duration=duration)
        # TODO(#269): wire render_tool_panel here once a console/args ref is threaded
        # through ToolCallLogger (or via a registered _display_callback similar to
        # deep_think.py's _progress_callback pattern). Inputs are: tool_name, args dict
        # (needs to be captured in on_tool_start), output, elapsed=duration.

    def on_tool_error(self, tool_name: str, error: str, call_id: str = "") -> None:
        """Log when a tool encounters an error."""
        key = call_id or tool_name
        with self._lock:
            self._tool_start_times.pop(key, None)
        log_tool_call(tool_name, error=error)


_tool_logger = ToolCallLogger()


def is_valid_response(output: str) -> bool:
    """Check if an agent response is valid and should be saved to history.

    Filters out empty responses and error messages that would poison
    the conversation context if stored in history.

    Returns:
        True if the response is a valid, meaningful AI output.
    """
    if not output or not output.strip():
        return False

    from src.common.message_validation import is_bad_ai_content

    return not is_bad_ai_content(output)


_SDK_MARKERS = (
    "Request body:",
    "Response body:",
    "Error body:",
    "Error code:",
    "Error message:",
    "request_id:",
    "headers:",
)
_SANITIZE_MAX_LEN = 500


def _sanitize_sdk_error(text: str) -> str:
    """Truncate and strip SDK-specific internals from a raw exception string."""
    for marker in _SDK_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].rstrip()
    return text[:_SANITIZE_MAX_LEN]


def _extract_api_message(error_str: str) -> str | None:
    """Extract the 'message' field from an OpenAI-compatible API error body.

    Handles the SDK format: ``Error code: 429 - {'error': {'message': "...", ...}}``
    """
    import re

    m = re.search(r"""['"]message['"]\s*:\s*['"](.+?)['"](?:\s*,|\s*\})""", error_str, re.DOTALL)
    if m:
        msg = m.group(1).strip()
        return msg[:_SANITIZE_MAX_LEN] if msg else None
    return None


def _is_user_config_error(e: Exception) -> bool:
    """True for provider 4xx errors caused by user configuration or credentials.

    Bad model IDs, invalid/missing API keys, and malformed requests are
    actionable by the operator and are *not* Cogtrix faults — they should be
    surfaced cleanly (see :func:`format_agent_error`) and logged concisely,
    without an ERROR-level stack trace that reads like an internal crash (#2124).
    """
    error_type = type(e).__name__
    error_str = str(e).lower()
    if any(t in error_type for t in ("BadRequestError", "NotFoundError", "AuthenticationError")):
        return True
    return any(
        needle in error_str
        for needle in (
            "not a valid model",
            "model_not_found",
            "invalid_api_key",
            "invalid_request_error",
        )
    )


def format_agent_error(e: Exception) -> str:
    """Format agent execution errors into user-friendly messages.

    Categorizes common error types and provides helpful guidance
    without exposing full stack traces.

    Uses Markdown formatting for proper display in rich panels.
    """
    error_str = str(e)
    error_type = type(e).__name__

    # Provider 400 "not a valid model ID" — a config error, not a Cogtrix fault.
    # Caught before the generic BadRequest branch so the operator gets an
    # actionable message naming the rejected model and how to fix it (#2124).
    if "not a valid model" in error_str.lower():
        actual = _extract_api_message(error_str) or _sanitize_sdk_error(error_str)
        return (
            f"**Invalid model ID:** {actual}\n\n"
            "The provider rejected the configured model name. Please check:\n"
            "- `models.<alias>.model` is the provider's **exact** model slug "
            "(e.g. `qwen/qwen3-...` on OpenRouter, not a bare `qwen3`)\n"
            "- The model is available to your account / API key\n"
            "- The provider `base_url` points at the intended endpoint"
        )

    if "NotFoundError" in error_type or "model_not_found" in error_str:
        if "does not exist" in error_str:
            parts = error_str.split("`")
            model = parts[1] if len(parts) >= 3 else "unknown"
            return (
                f"**Model not found:** `{model}`\n\n"
                "Please check:\n"
                "- The model name is correct\n"
                "- You have access to this model\n"
                "- Your API key has the required permissions"
            )
        return f"**Model not found:** {_sanitize_sdk_error(error_str)}"

    if "AuthenticationError" in error_type or "invalid_api_key" in error_str:
        return (
            "**Authentication failed:** "
            "Authentication failed. Please check your API key "
            "(OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, XAI_API_KEY, "
            "or DEEPSEEK_API_KEY depending on your provider)."
        )

    if "RateLimitError" in error_type or "rate_limit" in error_str.lower():
        actual_msg = _extract_api_message(error_str)
        if actual_msg:
            return f"**Rate limit / capacity error:**\n\n{actual_msg}"
        return (
            "**Rate limit exceeded.**\n\n"
            "Please wait a moment and try again, or:\n"
            "- Reduce request frequency\n"
            "- Upgrade your API plan"
        )

    if "APIConnectionError" in error_type or "Connection" in error_type:
        return (
            "**Connection error:** Unable to reach the API.\n\n"
            "Please check:\n"
            "- Your internet connection\n"
            "- The API endpoint URL is correct\n"
            "- Any firewall or proxy settings"
        )

    if "Timeout" in error_type or "timeout" in error_str.lower():
        return (
            "**Request timed out.**\n\n"
            "The model took too long to respond. Please:\n"
            "- Try again with a shorter prompt\n"
            "- Check if the service is experiencing high load"
        )

    if "BadRequestError" in error_type or "invalid_request" in error_str:
        if "max_tokens" in error_str and ("got -" in error_str or "at least 1" in error_str):
            return (
                "**Prompt too long for this model's context window.**\n\n"
                "The conversation history plus system prompt exceeds the model's "
                "maximum context size, leaving no room for a response.\n\n"
                "Try one of these:\n"
                "- `/clear` — clear conversation history and start fresh\n"
                "- Switch to a model with a larger context window (`/model <name>`)\n"
                "- Use a shorter prompt\n"
                "- Increase `context_window` in the model config (Ollama only)"
            )
        return f"**Invalid request:** {_sanitize_sdk_error(error_str)}"

    if "InternalServerError" in error_type or "500" in error_str:
        return (
            "**API server error (500).**\n\n"
            "The service is experiencing issues. Please try again later."
        )

    if "ServiceUnavailableError" in error_type or "503" in error_str:
        return (
            "**Service temporarily unavailable (503).**\n\n"
            "The API is overloaded or under maintenance. Please try again later."
        )

    if "ollama" in error_str.lower():
        if "connection refused" in error_str.lower():
            return (
                "**Cannot connect to Ollama.**\n\n"
                "Please check:\n"
                "- Ollama is running (`ollama serve`)\n"
                "- The base URL is correct (default: `http://localhost:11434`)"
            )
        if "model" in error_str.lower() and "not found" in error_str.lower():
            return (
                "**Ollama model not found.**\n\n"
                "Please check:\n"
                "- The model is downloaded (`ollama pull <model>`)\n"
                "- The model name is spelled correctly"
            )

    return f"An error occurred: {_sanitize_sdk_error(error_str)}"


def extract_ai_content(msg: Any) -> str | None:
    """Extract text content from a single message object.

    Handles string content, list content (multimodal), dict messages,
    and reasoning/thinking content produced by models like Qwen3 and QwQ
    (which may return thinking tokens in ``additional_kwargs`` rather
    than in the regular ``content`` field).

    Returns None if the message has no meaningful text.
    """
    content = getattr(msg, "content", None)

    if content is None and isinstance(msg, dict):
        content = msg.get("content", None)

    if isinstance(content, str) and content.strip():
        return content

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str) and part.strip():
                text_parts.append(part)
            elif isinstance(part, dict) and part.get("text", "").strip():
                text_parts.append(part["text"])
        if text_parts:
            return "\n".join(text_parts)

    # Thinking/reasoning content (Qwen3, QwQ, DeepSeek-R1, etc.)
    # Return reasoning even when tool_calls are present: a model greeting the user
    # ("Hello! I'm here to help.") while also calling request_tools is a real pattern,
    # and suppressing the greeting produces a silent/empty user-facing response.
    additional = getattr(msg, "additional_kwargs", None)
    if additional and isinstance(additional, dict):
        reasoning = additional.get("reasoning_content") or additional.get("thinking")
        if reasoning and isinstance(reasoning, str) and reasoning.strip():
            return reasoning

    return None


def has_phantom_tool_call(result: dict) -> bool:
    """Detect a "phantom tool call".

    The server reported finish_reason=tool_calls but the message has no
    actual tool_calls or content.

    This happens when vLLM (or another inference server) fails to parse the
    model's JSON for tool call arguments (JSONDecodeError) but still returns
    a 200 OK with finish_reason='tool_calls'.  LangChain creates an AIMessage
    with content='' and tool_calls=[] — a dead end for the agent.

    Returns True if the *last* AIMessage exhibits this pattern.
    """
    from langchain_core.messages import AIMessage

    messages = result.get("messages", [])
    if not messages:
        return False

    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue

        content = getattr(msg, "content", "")
        if isinstance(content, str) and content.strip():
            return False

        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            return False

        meta = getattr(msg, "response_metadata", None)
        if meta and isinstance(meta, dict):
            fr = meta.get("finish_reason", "")
            if fr == "tool_calls":
                return True

        # Without explicit finish_reason metadata we can't be sure this is a
        # phantom tool call.  A genuinely empty response should go through the
        # normal recovery path, not the phantom-retry path.
        return False

    return False


def extract_response(result: Any, log: Any = None, prior_count: int = 0) -> str | None:
    """Extract a meaningful AI response from the agent result.

    Walks backward through messages produced THIS TURN (after prior_count)
    to find the last AIMessage with non-empty content, skipping ToolMessages
    and empty AIMessages.  Never looks at history messages before prior_count
    to prevent returning a stale answer from a previous turn.

    Args:
        result: Agent execution result (dict with 'messages' key)
        log: Logger instance
        prior_count: Number of messages that were in the history BEFORE this
            turn.  Only messages at index >= prior_count are candidates.

    Returns:
        Response string, or None if no valid content found
    """
    if not isinstance(result, dict) or "messages" not in result:
        text = str(result)
        if text and text.strip():
            return text
        return None

    messages = result["messages"]
    if not messages:
        return None

    from langchain_core.messages import AIMessage, ToolMessage

    # Only search messages produced this turn (after prior_count)
    turn_messages = messages[prior_count:] if prior_count > 0 else messages

    # Imported here so the tests don't pay the import cost when extract_response
    # isn't on the hot path.
    from src.orchestration.phases import strip_foreign_tool_call_xml

    for msg in reversed(turn_messages):
        if isinstance(msg, ToolMessage):
            continue

        if isinstance(msg, AIMessage):
            text = extract_ai_content(msg)
            if text:
                cleaned = strip_foreign_tool_call_xml(text)
                # Only return cleaned text if there's still meaningful content
                # after stripping; an all-XML response should fall through.
                if isinstance(cleaned, str) and cleaned.strip():
                    return cleaned
            continue

        if isinstance(msg, dict) and msg.get("type") in ("ai", "aimessage"):
            text = extract_ai_content(msg)
            if text:
                cleaned = strip_foreign_tool_call_xml(text)
                if isinstance(cleaned, str) and cleaned.strip():
                    return cleaned

    if log:
        if log.isEnabledFor(10):  # logging.DEBUG
            log.debug(
                "No AI content in %d messages. Types: %s",
                len(messages),
                [type(m).__name__ for m in messages[-5:]],
            )

    return None


def build_tool_results_response(result: Any) -> str | None:
    """Build a response from tool execution results when the model failed to summarize.

    This is a last-resort fallback: if the model called tools and received results
    but then returned empty content, we present the tool results directly to the user.

    Returns:
        A formatted string with tool results, or None if no tools ran.
    """
    if not isinstance(result, dict) or "messages" not in result:
        return None

    from langchain_core.messages import ToolMessage

    tool_results: list[tuple[str, str]] = []

    for msg in result["messages"]:
        if isinstance(msg, ToolMessage):
            name = getattr(msg, "name", None) or "tool"
            content = getattr(msg, "content", "")
            if content and isinstance(content, str) and len(content) > 10:
                if not content.startswith("Error"):
                    tool_results.append((name, content))

    if not tool_results:
        return None

    parts = [
        "The model executed tools but did not summarize the results. Here is what was gathered:\n"
    ]
    for name, content in tool_results:
        parts.append(f"\n**{name}:**\n{content}\n")

    return "".join(parts)


def log_tool_calls_from_result(result: dict, prior_count: int = 0) -> None:
    """Extract and log tool calls from agent result messages.

    Parses only the new messages (since ``prior_count``) to find:
    - AIMessage with tool_calls (tool invocation requests)
    - ToolMessage (tool execution results)
    """
    try:
        from langchain_core.messages import AIMessage as AI
        from langchain_core.messages import ToolMessage as Tool
    except ImportError:
        return

    if not isinstance(result, dict) or "messages" not in result:
        return

    messages = result["messages"][prior_count:]
    log = get_logger()
    log.debug("Processing %d messages for tool calls", len(messages))

    pending_tool_calls: dict = {}

    for msg in messages:
        if isinstance(msg, AI):
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    tool_name = tc.get("name", "unknown")
                    tool_args = tc.get("args", {})
                    tool_id = tc.get("id", "")
                    pending_tool_calls[tool_id] = tool_name
                    _tool_logger.on_tool_start(tool_name, tool_args, call_id=tool_id)

        elif isinstance(msg, Tool):
            tool_call_id = getattr(msg, "tool_call_id", None) or ""
            tool_name = getattr(msg, "name", None)
            content = getattr(msg, "content", "")

            if not tool_name and tool_call_id in pending_tool_calls:
                tool_name = pending_tool_calls.pop(tool_call_id)

            if tool_name:
                from src.api.routes.metrics import TOOL_CALLS_TOTAL

                if isinstance(content, str) and content.startswith("Error"):
                    _tool_logger.on_tool_error(tool_name, content, call_id=tool_call_id)
                    if TOOL_CALLS_TOTAL is not None:
                        TOOL_CALLS_TOTAL.labels(tool_name=tool_name, status="error").inc()
                else:
                    output_str = str(content) if content else ""
                    _tool_logger.on_tool_end(tool_name, output_str, call_id=tool_call_id)
                    if TOOL_CALLS_TOTAL is not None:
                        TOOL_CALLS_TOTAL.labels(tool_name=tool_name, status="success").inc()


_EXTEND_CONTINUE_LIMIT = 300  # step budget for continuation after extend_run(mode="continue")
_EXTEND_DELEGATE_LIMIT = 50  # small budget for synthesis after delegation results arrive


def _handle_extend_run(
    extend_state: ExtendRunState,
    graph: Any,
    result: dict,
    input_messages: list,
    invoke_config: dict,
    config: Any,
    callbacks: list | None,
    log: Any,
) -> str:
    """Handle mid-run extension requests from the extend_run tool.

    Two modes:
    - **continue**: re-invoke the graph with a higher step limit so the agent
      can keep working sequentially.
    - **delegate**: run the requested subtasks via ``delegate_parallel``, then
      re-invoke the graph with the combined results so the agent synthesizes.
    """
    from src.orchestration.phases import force_delegation

    mode = extend_state.mode
    log.info(
        "Mid-run extension requested: mode=%s, subtasks=%d, reason=%s",
        mode,
        len(extend_state.subtasks),
        extend_state.reason,
    )

    messages_so_far = list(result.get("messages", input_messages))

    if mode == "delegate" and extend_state.subtasks:
        # ── Delegation mode: run subtasks in parallel, then synthesize ──
        try:
            # Build context from what the agent has done so far
            agent_context = extract_response(result, log) or ""
            tool_context = build_tool_results_response(result) or ""

            # Use force_delegation with subtask hints baked into the prompt
            subtask_list = "\n".join(
                f"  {i + 1}. {task}" for i, task in enumerate(extend_state.subtasks)
            )
            delegation_prompt = (
                f"The agent has been working on a complex task and identified these "
                f"independent subtasks that should run in parallel:\n\n{subtask_list}\n\n"
                f"Context from the agent's work so far:\n{agent_context[:2000]}"
            )
            delegation_result = force_delegation(
                delegation_prompt,
                agent_context,
                tool_context,
                config,
                log,
                llm=config.llm,
            )
            if delegation_result and delegation_result.strip():
                log.info("Delegation produced %d chars of results", len(delegation_result))
                # Feed delegation results back to the agent for synthesis
                try:
                    from langchain_core.messages import HumanMessage as HM

                    messages_so_far.append(
                        HM(
                            content=(
                                f"Sub-agent delegation results:\n\n{delegation_result}\n\n"
                                "Please synthesize these results into a complete, coherent "
                                "response. Combine the findings and provide your final answer."
                            )
                        )
                    )
                except ImportError:
                    pass  # langchain_core not installed — skip delegation context injection

                # Re-invoke with a small budget for synthesis
                synth_config = dict(invoke_config)
                synth_config["recursion_limit"] = _EXTEND_DELEGATE_LIMIT
                try:
                    synth_result: dict = {"messages": messages_so_far}
                    for chunk in graph.stream(
                        {"messages": messages_so_far},
                        config=synth_config,
                        stream_mode="values",
                    ):
                        if isinstance(chunk, dict) and "messages" in chunk:
                            synth_result = chunk
                    synth_response = extract_response(synth_result, log)
                    if synth_response:
                        return synth_response
                except RecursionError:
                    log.warning(
                        "Recursion limit hit during delegation synthesis — "
                        "returning raw delegation results (may be incomplete)"
                    )
                # Fallback: return the raw delegation results (may be incomplete
                # if synthesis hit RecursionError above)
                return delegation_result
        except Exception as e:
            log.warning("Delegation in extend_run failed: %s", e, exc_info=True)
            # Fall through to continuation mode

    # ── Continue mode: re-invoke with a higher step limit ──────────
    log.info("Continuing with extended step limit of %d", _EXTEND_CONTINUE_LIMIT)
    try:
        from langchain_core.messages import HumanMessage as HM

        messages_so_far.append(
            HM(content="Your step budget has been extended. Continue working on the task.")
        )
    except ImportError:
        pass  # langchain_core not installed — skip extension message

    continue_config = dict(invoke_config)
    continue_config["recursion_limit"] = _EXTEND_CONTINUE_LIMIT
    if callbacks:
        continue_config["callbacks"] = callbacks

    try:
        continue_result: dict = {"messages": messages_so_far}
        for chunk in graph.stream(
            {"messages": messages_so_far},
            config=continue_config,
            stream_mode="values",
        ):
            if isinstance(chunk, dict) and "messages" in chunk:
                continue_result = chunk

        response = extract_response(continue_result, log)
        if response:
            return response
    except RecursionError:
        log.warning("Extended run also hit recursion limit")

    # Final fallback
    from src.orchestration.phases import recover_from_step_limit

    return recover_from_step_limit(graph, result, input_messages, invoke_config, log)


def _build_ownership_constraint(mode: OwnershipMode) -> str:
    """Return a system prompt suffix that constrains the agent to INFORM or ADVISE mode."""
    if mode == OwnershipMode.INFORM:
        return (
            "\n\n---\nTASK MODE: INFORMATIONAL\n"
            "The user has requested information, not execution. Research and explain; "
            "do not take actions that change system state. Describe what steps would "
            "be needed rather than executing them."
        )
    if mode == OwnershipMode.ADVISE:
        return (
            "\n\n---\nTASK MODE: ADVISORY\n"
            "The user is seeking guidance. Present options with tradeoffs. "
            "Do not execute — let the user decide."
        )
    return ""


def run_agent(
    user_input: str,
    history_messages: list,
    registry: Any,
    approvals: set,
    context_prefix: str | None = None,
    recursion_limit: int | None = None,
    callbacks: list | None = None,
    result_messages: list | None = None,
    *,
    config: AgentRunConfig,
    task_complexity: TaskComplexity | None = None,
) -> str:
    """Run agent using a custom LangGraph StateGraph.

    This replaces run_agent_with_safety with a graph-based approach where
    tool expansion, phantom recovery, and tool validation are handled as
    graph nodes with proper routing.

    Args:
        user_input: Current user input
        history_messages: Conversation history
        registry: Tool registry
        approvals: Set of approved tools
        context_prefix: Mode-specific context to inject
        recursion_limit: Maximum graph node visits (default: 150, ~75 tool calls)
        callbacks: Optional callback handlers for LLM observability
        result_messages: Optional output list for caller inspection
        config: Session-constant parameters bundle (required)
        task_complexity: Optional precomputed complexity; when omitted the
            function classifies the task automatically.

    Returns:
        Agent response as string
    """
    from src.orchestration.phases import is_step_limit_apology, recover_from_step_limit
    from src.tools.web_search import set_synthesis_llm

    # Scope the stage-5 synthesiser LLM to this run. The API path
    # reaches run_agent through ``asyncio.to_thread``, which gives us
    # a copied ContextVar context per call → multi-tenant isolation.
    # The CLI/assistant call run_agent sequentially per session on
    # the same thread → the overwrite-on-entry is correct because
    # runs don't overlap. Single-tenant is the degenerate case.
    set_synthesis_llm(config.llm, getattr(config, "compression_llm", None))

    _base_system_prompt = config.system_prompt
    _run_system_prompt = _base_system_prompt

    _compression_min_age = config.compression_min_age
    _compression_min_chars = config.compression_min_chars

    if recursion_limit is None:
        _complexity = task_complexity or classify_task_complexity(user_input)
        if _complexity == TaskComplexity.COMPLEX_ACTION:
            recursion_limit = 300  # ~150 tool-call cycles for builds/installs
            get_logger().info(
                "Complex action task detected — raising step limit to %d", recursion_limit
            )
        elif _complexity == TaskComplexity.COMPLEX_RESEARCH:
            recursion_limit = 200  # research tasks may need extra rounds too
            get_logger().info(
                "Complex research task detected — raising step limit to %d", recursion_limit
            )
        else:
            recursion_limit = DEFAULT_RECURSION_LIMIT

        if _complexity == TaskComplexity.SIMPLE:
            _auto_load_simple_tools(config)
            # A SIMPLE-classified prompt can still need fresh external data
            # (e.g. "What's the current Apple stock price?"). Force-load the
            # retrieval set on a recency signal so the agent doesn't deflect
            # for lack of a search tool (bug #1839).
            if _query_needs_realtime_data(user_input) and _auto_load_web_search(config):
                get_logger().info("Auto-loaded web_search for real-time query")

        # Auto-load `web_search` for complex tasks so the agent has web
        # research available from the first round without needing to call
        # `request_tools`. Addresses the RBA (Research-Before-Action) gap
        # where agents skip loading search when they're confident in
        # training data.
        elif _complexity in (TaskComplexity.COMPLEX_ACTION, TaskComplexity.COMPLEX_RESEARCH):
            if _auto_load_web_search(config):
                get_logger().info("Auto-loaded web_search for complex task")

        # File-mutation intent — orthogonal to complexity (a short imperative
        # like "delete /workspace/foo.py" is SIMPLE linguistically but still
        # requires the mutation toolset to be live). Without this preload,
        # the default `code` memory-mode preset leaves the agent with only
        # `get_current_datetime` + `request_tools` and the model tends to
        # silently fabricate success on destructive operations rather than
        # invoke `request_tools` honestly (Q9/Q10 of #1869's holistic-test
        # battery). See bug #1870.
        if _query_signals_file_write_intent(user_input):
            _auto_load_file_write_tools(config)

    # ── Task ownership classification ──────────────────────────────────────
    if getattr(config, "task_ownership_classifier_enabled", True):
        _toc_llm_fallback = getattr(config, "task_ownership_classifier_llm_fallback", False)
        _toc_ambiguous_action = getattr(config, "task_ownership_ambiguous_action", "ask")

        _ownership = classify_task_ownership(
            user_input,
            llm=config.llm if _toc_llm_fallback else None,
            llm_fallback_enabled=_toc_llm_fallback,
            llm_timeout_seconds=min(getattr(config, "llm_timeout", 180), 10),
        )
        get_logger().info(
            "Task ownership: mode=%s confidence=%.2f signal=%s",
            _ownership.mode.name,
            _ownership.confidence,
            _ownership.raw_signal,
        )

        if _ownership.mode == OwnershipMode.AMBIGUOUS:
            if _toc_ambiguous_action == "ask":
                # Let the graph run normally so history, token counts, and
                # WS events are handled correctly.  Inject a prompt constraint
                # that guides the agent to ask one clarifying question instead
                # of executing.  Returning a bare string here would corrupt
                # conversation history (C1: AMBIGUOUS short-circuit bug).
                _action = _ownership.inferred_action or user_input[:40]
                _ambiguous_constraint = (
                    "\n\n---\nTASK MODE: CLARIFICATION NEEDED\n"
                    f"The request '{_action}' is ambiguous — it could mean "
                    "performing an action OR explaining how to do it. "
                    "Ask the user ONE specific question to determine their "
                    "intent before doing anything. Do not execute, install, "
                    "delete, or modify anything until the intent is confirmed."
                )
                if _run_system_prompt:
                    _run_system_prompt = _run_system_prompt + _ambiguous_constraint
                else:
                    _run_system_prompt = _ambiguous_constraint
            elif _toc_ambiguous_action == "inform":
                _ownership = OwnershipResult(
                    mode=OwnershipMode.INFORM,
                    confidence=_ownership.confidence,
                    is_reversible=_ownership.is_reversible,
                    raw_signal="ambiguous_forced_inform",
                    inferred_action=_ownership.inferred_action,
                )
            # else "execute": fall through without constraint injection

        if _ownership.mode in (OwnershipMode.INFORM, OwnershipMode.ADVISE):
            _constraint = _build_ownership_constraint(_ownership.mode)
            if _run_system_prompt:
                _run_system_prompt = _run_system_prompt + _constraint
            else:
                _run_system_prompt = _constraint

        config.system_prompt = _run_system_prompt

    if _compression_min_age is None:
        _compression_min_age = COMPRESSION_MIN_AGE_CYCLES
    if _compression_min_chars is None:
        _compression_min_chars = COMPRESSION_MIN_CHARS

    log = get_logger()
    _t = [time.monotonic()]

    def _mark(label: str) -> None:
        now = time.monotonic()
        log.debug("⏱ %s: %.0fms", label, (now - _t[0]) * 1000)
        _t[0] = now

    try:
        input_messages = prepare_messages_with_context(
            history_messages=history_messages,
            user_input=user_input,
            context_prefix=context_prefix,
            max_context_tokens=config.max_context_tokens,
        )
        _mark("prepare_messages")

        if is_verbose():
            log.debug("Sending %d messages to agent", len(input_messages))
        if is_trace():
            for i, msg in enumerate(input_messages):
                msg_type = type(msg).__name__
                content = ""
                if hasattr(msg, "content"):
                    content = msg.content
                elif isinstance(msg, dict) and "content" in msg:
                    content = msg["content"]
                log.debug("  [%d] %s: %s", i, msg_type, content)

        # extend_run tool is wired into the graph below so the agent can call it
        # explicitly to request more steps or delegate subtasks mid-run.
        _extend_state = ExtendRunState()

        invoke_config: dict[str, Any] = {"recursion_limit": recursion_limit}
        if callbacks:
            invoke_config["callbacks"] = callbacks

        # Decide whether to use per-session caches (API mode) or module globals (CLI mode).
        # Per-session caches are set by session_bridge._build_run_config(); when present
        # they are fully isolated so concurrent sessions with different LLMs cannot
        # poison each other's bind_tools / compression results.
        use_per_session_caches = (
            config.bound_cache is not None and config.compression_cache is not None
        )

        if use_per_session_caches:
            # API mode: snapshot local copies from per-session caches; merge back after.
            # ``use_per_session_caches`` is True exactly when both caches are
            # non-None (computed three lines up), and nothing between mutates
            # them — so the inner check that previously raised RuntimeError
            # here was unreachable (#1090).  The ``assert`` keeps the type
            # narrowing pyright needs for the indexing below without lying
            # about being a runtime defence.
            assert config.bound_cache is not None and config.compression_cache is not None
            local_bound_cache = OrderedDict(config.bound_cache)
            local_compression_cache = OrderedDict(config.compression_cache)
            compression_cache_target = config.compression_cache
            current_llm_id = (id(config.llm), 0)
        else:
            global _persistent_bound_cache, _persistent_compression_cache, _cached_llm_id

            # Drain finished background compression warm-up jobs BEFORE the
            # snapshot so their results are included in local_compression_cache.
            # Must run outside _cache_lock — _drain acquires the same lock
            # internally and threading.Lock is non-reentrant.
            _drain_background_compression_jobs()

            llm_changed: bool
            with _cache_lock:
                current_llm_id = (id(config.llm), _llm_generation)
                llm_changed = _cached_llm_id is not None and _cached_llm_id != current_llm_id
                if llm_changed:
                    _persistent_bound_cache.clear()
                    _persistent_compression_cache.clear()
                    _clear_pending_background_compression_jobs()
                _cached_llm_id = current_llm_id
                local_bound_cache = OrderedDict(_persistent_bound_cache)
                local_compression_cache = OrderedDict(_persistent_compression_cache)
                compression_cache_target = _persistent_compression_cache

        _mark("cache_setup")

        # ── Compiled-graph cache (CLI mode only) ─────────────────────────────
        # Build a fingerprint from factors that determine whether an existing
        # compiled graph can be safely reused.  API-mode sessions each have
        # their own bound/compression caches so they bypass this cache.
        graph: Any
        if not use_per_session_caches:
            _active_fp = tuple(getattr(t, "name", "") for t in (config.active_tools_list or []))
            _avail_fp = tuple(sorted((config.available_tools or {}).keys()))
            _sp_hash = hash(config.system_prompt or "")
            _graph_key = (current_llm_id, _active_fp, _avail_fp, _sp_hash)
            with _cache_lock:
                _cached_graph = _persistent_graph_cache.get(_graph_key)
                if _cached_graph is not None:
                    _persistent_graph_cache.move_to_end(_graph_key)
            if _cached_graph is not None and hasattr(_cached_graph, "_reset_for_new_run"):
                _cached_graph._reset_for_new_run(  # type: ignore[attr-defined]
                    config.available_tools or {},
                    local_bound_cache,
                    local_compression_cache,
                    extend_run_state=_extend_state,
                )
                graph = _cached_graph
                log.debug("Graph cache hit — reusing compiled graph")
            else:
                graph = build_agent_graph(
                    config=config,
                    registry=registry,
                    approvals=approvals,
                    compression_min_age=_compression_min_age,
                    compression_min_chars=_compression_min_chars,
                    bound_cache=local_bound_cache,
                    compression_cache_in=local_compression_cache,
                    tool_context_limit_pct=getattr(config, "tool_context_limit_pct", 0.80),
                    extend_run_state=_extend_state,
                )
                with _cache_lock:
                    _persistent_graph_cache[_graph_key] = graph
                    _persistent_graph_cache.move_to_end(_graph_key)
                    while len(_persistent_graph_cache) > _MAX_GRAPH_CACHE_SIZE:
                        _persistent_graph_cache.popitem(last=False)
        else:
            graph = build_agent_graph(
                config=config,
                registry=registry,
                approvals=approvals,
                compression_min_age=_compression_min_age,
                compression_min_chars=_compression_min_chars,
                bound_cache=local_bound_cache,
                compression_cache_in=local_compression_cache,
                tool_context_limit_pct=getattr(config, "tool_context_limit_pct", 0.80),
                extend_run_state=_extend_state,
            )
        _mark("build_graph")

        if config.context_compression:
            _queue_background_compression(
                input_messages,
                compression_cache_target,
                call_count=0,
                llm=config.compression_llm or config.llm,
                max_context_tokens=config.max_context_tokens,
                min_age_cycles=_compression_min_age,
                min_chars=_compression_min_chars,
                emergency_threshold=getattr(
                    config, "context_compression_emergency_threshold", 0.85
                ),
                human_msg_max_chars=getattr(
                    config, "context_compression_human_msg_max_chars", 20_000
                ),
            )
        _mark("compression")

        hit_recursion_limit = False
        prior_msg_count = len(input_messages)
        result: dict = {"messages": input_messages}
        try:
            try:
                for chunk in graph.stream(
                    {"messages": input_messages},
                    config=invoke_config,
                    stream_mode="values",
                ):
                    if isinstance(chunk, dict) and "messages" in chunk:
                        result = chunk
            except RecursionError:
                hit_recursion_limit = True
                log.warning("Agent hit the recursion limit")
            _mark("graph.stream")

            log_tool_calls_from_result(result, prior_count=prior_msg_count)

            # Post-turn compression: warm the next turn's cache without holding
            # response delivery. The current turn already has its answer; the
            # compressed copy is only needed for future turns.
            _tcc_active = getattr(config, "tier_cache_enabled", False)
            if config.context_compression and result.get("messages") and not _tcc_active:
                _queue_background_compression(
                    result["messages"],
                    compression_cache_target,
                    call_count=999,  # ensures all messages are "old enough"
                    llm=config.compression_llm or config.llm,
                    max_context_tokens=config.max_context_tokens,
                    min_age_cycles=1,
                    min_chars=_compression_min_chars,
                    emergency_threshold=getattr(
                        config, "context_compression_emergency_threshold", 0.85
                    ),
                    human_msg_max_chars=getattr(
                        config, "context_compression_human_msg_max_chars", 20_000
                    ),
                )

            if result_messages is not None:
                result_messages.extend(result.get("messages", []))

            if hit_recursion_limit:
                # ── Mid-run extension: continue or delegate ──────────────
                if _extend_state.requested:
                    return _handle_extend_run(
                        _extend_state,
                        graph,
                        result,
                        input_messages,
                        invoke_config,
                        config,
                        callbacks,
                        log,
                    )
                return recover_from_step_limit(graph, result, input_messages, invoke_config, log)

            response = extract_response(result, log, prior_count=prior_msg_count)
            if response and not is_step_limit_apology(response):
                return response

            if response and is_step_limit_apology(response):
                log.warning(
                    "Agent returned a step-limit apology instead of a real answer, "
                    "attempting recovery"
                )
            else:
                log.warning("Agent returned empty content, attempting recovery")

            return recover_from_step_limit(graph, result, input_messages, invoke_config, log)
        finally:
            _drain_background_compression_jobs(compression_cache_target)
            if use_per_session_caches:
                # Merge local snapshots back into the per-session caches.
                # Same invariant as at the entry branch: ``use_per_session_caches``
                # is True only when both caches were non-None, and ``config`` is
                # treated as session-constant per the AgentRunConfig docstring.
                # ``assert`` for pyright narrowing; the previous ``raise
                # RuntimeError`` here was unreachable (#1090).
                assert config.bound_cache is not None and config.compression_cache is not None
                with config.cache_lock:
                    for key, value in local_bound_cache.items():
                        config.bound_cache[key] = value
                        config.bound_cache.move_to_end(key)
                    while len(config.bound_cache) > _MAX_BOUND_CACHE_SIZE:
                        config.bound_cache.popitem(last=False)
                    _merge_compression_cache(config.compression_cache, local_compression_cache)
            else:
                with _cache_lock:
                    if _cached_llm_id == current_llm_id:
                        for key, value in local_bound_cache.items():
                            _persistent_bound_cache[key] = value
                            _persistent_bound_cache.move_to_end(key)
                        while len(_persistent_bound_cache) > _MAX_BOUND_CACHE_SIZE:
                            _persistent_bound_cache.popitem(last=False)
                        _merge_compression_cache(
                            _persistent_compression_cache, local_compression_cache
                        )

    except UserCancelledRun:
        raise
    except Exception as e:
        if _is_user_config_error(e):
            # Provider rejected the request for a user-actionable reason (bad
            # model id, invalid key, malformed request). Log concisely without a
            # stack trace — it's a config error, not a Cogtrix fault (#2124).
            log.warning(
                "Agent run rejected by the model provider (user-actionable): %s",
                _sanitize_sdk_error(str(e)),
            )
        else:
            log.error("Agent execution failed: %s", e, exc_info=True)
        # #2124: signal failure to callers via a typed exception instead of
        # returning an error string that masquerades as a normal answer. Callers
        # (CLI / API turn runner / assistant) decide how to surface it; the API
        # path turns this into a proper error frame rather than a 200 reply.
        raise AgentExecutionError(format_agent_error(e)) from e
    finally:
        # Always restore the session-constant system prompt (#2172). run_agent
        # mutates config.system_prompt above to apply per-run task-ownership /
        # clarification constraints; an exception raised before the inner
        # graph.stream try (e.g. in prepare_messages_with_context or
        # build_agent_graph) would otherwise skip the restore and leave the
        # caller's AgentRunConfig dirty, stacking TASK MODE constraint blocks on
        # any cross-turn reuse.
        config.system_prompt = _base_system_prompt
