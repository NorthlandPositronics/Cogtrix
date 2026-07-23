"""Agent runner — orchestrates a single prompt-to-response cycle.

Houses the main ``run_agent`` entry point and the helper functions it
delegates to: response extraction, error formatting, tool-call logging,
and phantom-call detection.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any

from src.agent.core import prepare_messages_with_context
from src.agent.safety import UserCancelledRun
from src.logging_config import get_logger, is_trace, is_verbose, log_tool_call
from src.orchestration.compression import (
    COMPRESSION_MIN_AGE_CYCLES,
    COMPRESSION_MIN_CHARS,
    apply_message_compression,
)
from src.orchestration.graph import DEFAULT_RECURSION_LIMIT, build_agent_graph
from src.orchestration.run_config import AgentRunConfig

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

# Compiled graph cache — avoids the ~25ms StateGraph.compile() cost per turn.
# Keyed by (llm_id, llm_generation, active_fingerprint, available_fingerprint,
# system_prompt_hash).  When the key matches, _reset_for_new_run() is called
# to refresh per-run mutable state before the graph is reused.
# Only used in CLI mode (not API mode, which manages per-session state
# independently via AgentRunConfig.bound_cache / compression_cache).
_MAX_GRAPH_CACHE_SIZE = 4
_persistent_graph_cache: OrderedDict = OrderedDict()  # key → compiled graph


def advance_llm_generation() -> None:
    """Increment the LLM generation counter when the LLM is switched."""
    global _llm_generation
    with _cache_lock:
        _llm_generation += 1


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

    from src.memory.manager import _is_bad_ai_content

    return not _is_bad_ai_content(output)


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


def format_agent_error(e: Exception) -> str:
    """Format agent execution errors into user-friendly messages.

    Categorizes common error types and provides helpful guidance
    without exposing full stack traces.

    Uses Markdown formatting for proper display in rich panels.
    """
    error_str = str(e)
    error_type = type(e).__name__

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
            "(OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, or XAI_API_KEY "
            "depending on your provider)."
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
    # Only use this fallback for *final* messages — NOT for tool-calling messages.
    # A tool-calling AIMessage has content="" plus tool_calls=[...]; its reasoning
    # is just internal deliberation about which tool to invoke.
    tool_calls = getattr(msg, "tool_calls", None)
    has_tool_calls = bool(tool_calls)

    if not has_tool_calls:
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

    for msg in reversed(turn_messages):
        if isinstance(msg, ToolMessage):
            continue

        if isinstance(msg, AIMessage):
            text = extract_ai_content(msg)
            if text:
                return text
            continue

        if isinstance(msg, dict) and msg.get("type") in ("ai", "aimessage"):
            text = extract_ai_content(msg)
            if text:
                return text

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
        "The model executed tools but did not summarize the results. "
        "Here is what was gathered:\n"
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
                if isinstance(content, str) and content.startswith("Error"):
                    _tool_logger.on_tool_error(tool_name, content, call_id=tool_call_id)
                else:
                    output_str = str(content) if content else ""
                    _tool_logger.on_tool_end(tool_name, output_str, call_id=tool_call_id)


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
    config: AgentRunConfig | None = None,
    llm: Any = None,
    system_prompt: str | None = None,
    available_tools: dict | None = None,
    active_tools_list: list | None = None,
    max_context_tokens: int | None = None,
    preset_tools: set[str] | None = None,
    context_compression: bool = True,
    compression_min_age: int | None = None,
    compression_min_chars: int | None = None,
    compression_llm: Any = None,
    tool_call_guard: Any | None = None,
    session_state: Any = None,
    confirmation_ui: Any | None = None,
    on_tool_expansion: Any | None = None,
    parallel_tool_execution: bool = True,
    git_native: bool = False,
    tool_context_limit_pct: float = 0.80,
    tier_cache_enabled: bool = True,
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
        config: Session-constant parameters bundle (preferred over individual kwargs)
        llm: Pre-created LLM instance (fallback when config is None)
        system_prompt: System prompt (fallback when config is None)
        available_tools: {name: tool} of tools available on request (fallback when config is None)
        active_tools_list: List of tool objects currently active (fallback when config is None)
        max_context_tokens: Context budget (fallback when config is None)
        preset_tools: Tool names that cannot be released (fallback when config is None)

    Returns:
        Agent response as string
    """
    from src.orchestration.phases import is_step_limit_apology, recover_from_step_limit

    if config is None:
        config = AgentRunConfig(
            llm=llm,
            system_prompt=system_prompt,
            available_tools=available_tools,
            active_tools_list=active_tools_list,
            max_context_tokens=max_context_tokens,
            preset_tools=preset_tools,
            context_compression=context_compression,
            compression_min_age=compression_min_age,
            compression_min_chars=compression_min_chars,
            compression_llm=compression_llm,
            tool_call_guard=tool_call_guard,
            session_state=session_state,
            confirmation_ui=confirmation_ui,
            on_tool_expansion=on_tool_expansion,
            parallel_tool_execution=parallel_tool_execution,
            git_native=git_native,
            tool_context_limit_pct=tool_context_limit_pct,
            tier_cache_enabled=tier_cache_enabled,
        )

    _compression_min_age = config.compression_min_age
    _compression_min_chars = config.compression_min_chars

    if recursion_limit is None:
        recursion_limit = DEFAULT_RECURSION_LIMIT
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
            # Asserts narrow the type for Pyright — the boolean guard already guarantees
            # these are non-None when use_per_session_caches is True.
            assert config.bound_cache is not None  # noqa: S101
            assert config.compression_cache is not None  # noqa: S101
            local_bound_cache = OrderedDict(config.bound_cache)
            local_compression_cache = OrderedDict(config.compression_cache)
            current_llm_id = (id(config.llm), 0)
        else:
            global _persistent_bound_cache, _persistent_compression_cache, _cached_llm_id

            llm_changed: bool
            with _cache_lock:
                current_llm_id = (id(config.llm), _llm_generation)
                llm_changed = _cached_llm_id is not None and _cached_llm_id != current_llm_id
                if llm_changed:
                    _persistent_bound_cache.clear()
                    _persistent_compression_cache.clear()
                _cached_llm_id = current_llm_id
                local_bound_cache = OrderedDict(_persistent_bound_cache)
                local_compression_cache = OrderedDict(_persistent_compression_cache)

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
            )
        _mark("build_graph")

        if config.context_compression:
            input_messages = apply_message_compression(
                input_messages,
                call_count=0,
                compression_cache=local_compression_cache,
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

            # Post-turn compression: compress the final message list so the
            # memory manager stores a compact context and the next turn starts
            # within budget.  Without this, the context bar shows the raw
            # uncompressed size and the next turn inherits inflated history.
            # When TCC is active the background roll-forward handles compression
            # incrementally, so this O(N) bulk pass is redundant.
            _tcc_active = getattr(config, "tier_cache_enabled", False)
            if config.context_compression and result.get("messages") and not _tcc_active:
                _post_llm = config.compression_llm or config.llm
                if _post_llm is not None:
                    result["messages"] = apply_message_compression(
                        result["messages"],
                        call_count=999,  # high count ensures all messages are "old enough"
                        compression_cache=local_compression_cache,
                        llm=_post_llm,
                        max_context_tokens=config.max_context_tokens,
                        min_age_cycles=1,
                        min_chars=_compression_min_chars,
                    )

            if result_messages is not None:
                result_messages.extend(result.get("messages", []))

            if hit_recursion_limit:
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
            if use_per_session_caches:
                # Merge local snapshots back into the per-session caches.
                # Asserts narrow OrderedDict | None for Pyright — the boolean guard
                # already guarantees non-None when use_per_session_caches is True.
                assert config.bound_cache is not None  # noqa: S101
                assert config.compression_cache is not None  # noqa: S101
                for key, value in local_bound_cache.items():
                    config.bound_cache[key] = value
                    config.bound_cache.move_to_end(key)
                while len(config.bound_cache) > _MAX_BOUND_CACHE_SIZE:
                    config.bound_cache.popitem(last=False)
                for key, value in local_compression_cache.items():
                    config.compression_cache[key] = value
                    config.compression_cache.move_to_end(key)
                while len(config.compression_cache) > _MAX_COMPRESSION_CACHE_SIZE:
                    config.compression_cache.popitem(last=False)
            else:
                with _cache_lock:
                    if _cached_llm_id == current_llm_id:
                        for key, value in local_bound_cache.items():
                            _persistent_bound_cache[key] = value
                            _persistent_bound_cache.move_to_end(key)
                        while len(_persistent_bound_cache) > _MAX_BOUND_CACHE_SIZE:
                            _persistent_bound_cache.popitem(last=False)
                        for key, value in local_compression_cache.items():
                            _persistent_compression_cache[key] = value
                            _persistent_compression_cache.move_to_end(key)
                        while len(_persistent_compression_cache) > _MAX_COMPRESSION_CACHE_SIZE:
                            _persistent_compression_cache.popitem(last=False)

    except UserCancelledRun:
        raise
    except Exception as e:
        log.error("Agent execution failed: %s", e, exc_info=True)
        return format_agent_error(e)
