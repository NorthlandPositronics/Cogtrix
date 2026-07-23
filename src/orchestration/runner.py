"""Agent runner — orchestrates a single prompt-to-response cycle.

Houses the main ``run_agent`` entry point and the helper functions it
delegates to: response extraction, error formatting, tool-call logging,
and phantom-call detection.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.agent.safety import UserCancelledRun
from src.logging_config import get_logger, log_tool_call
from src.orchestration.run_config import AgentRunConfig


class ToolCallLogger:
    """Callback handler that logs tool calls.

    Tracks invocations by ``call_id`` (LangChain's unique tool-call ID)
    so that concurrent runs of the *same* tool each get an accurate
    duration measurement.
    """

    _STALE_TIMEOUT = 600  # 10 minutes

    def __init__(self) -> None:
        self._tool_start_times: dict[str, float] = {}

    def _evict_stale(self) -> None:
        """Remove entries older than ``_STALE_TIMEOUT`` to prevent leaks."""
        cutoff = time.time() - self._STALE_TIMEOUT
        stale_keys = [k for k, ts in self._tool_start_times.items() if ts < cutoff]
        for k in stale_keys:
            self._tool_start_times.pop(k, None)

    def on_tool_start(self, tool_name: str, tool_input: dict, call_id: str = "") -> None:
        """Log when a tool starts execution."""
        self._evict_stale()
        key = call_id or tool_name
        self._tool_start_times[key] = time.time()
        log_tool_call(tool_name, inputs=tool_input)

    def on_tool_end(self, tool_name: str, output: str, call_id: str = "") -> None:
        """Log when a tool finishes execution."""
        key = call_id or tool_name
        duration = None
        if key in self._tool_start_times:
            duration = time.time() - self._tool_start_times.pop(key)

        log_tool_call(tool_name, output=output, duration=duration)

    def on_tool_error(self, tool_name: str, error: str, call_id: str = "") -> None:
        """Log when a tool encounters an error."""
        key = call_id or tool_name
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
                "- Increase `num_ctx` in the provider config (Ollama only)"
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
    messages = result.get("messages", [])
    if not messages:
        return False

    for msg in reversed(messages):
        if type(msg).__name__ != "AIMessage":
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


def extract_response(result: Any, log: Any = None) -> str | None:
    """Extract a meaningful AI response from the agent result.

    Walks backward through messages to find the last AIMessage with
    non-empty content, skipping ToolMessages and empty AIMessages.

    Args:
        result: Agent execution result (dict with 'messages' key)
        log: Logger instance

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

    for msg in reversed(messages):
        msg_type = type(msg).__name__

        if msg_type == "ToolMessage":
            continue

        if msg_type == "AIMessage":
            text = extract_ai_content(msg)
            if text:
                return text
            continue

        if isinstance(msg, dict) and msg.get("type") in ("ai", "aimessage"):
            text = extract_ai_content(msg)
            if text:
                return text

    if log:
        log.debug(
            f"No AI content in {len(messages)} messages. "
            f"Types: {[type(m).__name__ for m in messages[-5:]]}"
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

    tool_results: list[tuple[str, str]] = []

    for msg in result["messages"]:
        if type(msg).__name__ == "ToolMessage":
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


def log_tool_calls_from_result(result: dict) -> None:
    """Extract and log tool calls from agent result messages.

    Parses the message sequence to find:
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

    messages = result["messages"]
    log = get_logger()
    log.debug(f"Processing {len(messages)} messages for tool calls")

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
    from src.agent.core import prepare_messages_with_context
    from src.orchestration.compression import COMPRESSION_MIN_AGE_CYCLES, COMPRESSION_MIN_CHARS
    from src.orchestration.graph import DEFAULT_RECURSION_LIMIT, build_agent_graph
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

    try:
        input_messages = prepare_messages_with_context(
            history_messages=history_messages,
            user_input=user_input,
            context_prefix=context_prefix,
            max_context_tokens=config.max_context_tokens,
        )

        log.debug("Sending %d messages to agent", len(input_messages))
        if log.isEnabledFor(logging.DEBUG):
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

        graph = build_agent_graph(
            config=config,
            registry=registry,
            approvals=approvals,
            compression_min_age=_compression_min_age,
            compression_min_chars=_compression_min_chars,
        )

        hit_recursion_limit = False
        result: dict = {"messages": input_messages}
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

        log_tool_calls_from_result(result)

        if result_messages is not None:
            result_messages.extend(result.get("messages", []))

        if hit_recursion_limit:
            return recover_from_step_limit(graph, result, input_messages, invoke_config, log)

        response = extract_response(result, log)
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

    except UserCancelledRun:
        raise
    except Exception as e:
        return format_agent_error(e)
