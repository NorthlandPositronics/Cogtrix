"""
Logging configuration for Cogtrix Agent.

Provides centralized logging setup with support for:
- File logging with configurable path
- Debug mode with verbose output
- Request ID tracking for conversation context
- Custom formatters for different log levels
"""

import logging
import re as _re
import sys
import uuid
from contextvars import ContextVar
from pathlib import Path

# Context variable for request/conversation tracking
_request_id: ContextVar[str] = ContextVar("request_id", default="")

# Module-level logger
logger = logging.getLogger("cogtrix")

# Default log file name
DEFAULT_LOG_FILE = "cogtrix.log"

# Verbose logging flag - when True, log full content without truncation
_verbose_logging: bool = False

# Verbosity level: 0=normal, 1=debug, 2=verbose, 3=trace
_verbosity: int = 0


def get_verbosity() -> int:
    """Return the current verbosity level (0–3)."""
    return _verbosity


def set_verbosity(level: int) -> None:
    """Set the verbosity level (0–3), clamped to valid range."""
    global _verbosity, _verbose_logging
    _verbosity = max(0, min(3, int(level)))
    _verbose_logging = _verbosity >= 2


def is_verbose() -> bool:
    """Return True when verbosity is 2 (verbose) or higher."""
    return _verbosity >= 2


def is_trace() -> bool:
    """Return True when verbosity is 3 (trace)."""
    return _verbosity >= 3


# Patterns that look like secrets (compiled once at import time)
_SENSITIVE_RE = _re.compile(
    r"""
    (?:api[_-]?key|secret|token|password|authorization)
    \s*[:=]\s*
    ["']?([A-Za-z0-9_\-/.]{8,})["']?
    """,
    _re.IGNORECASE | _re.VERBOSE,
)
_BEARER_RE = _re.compile(r"Bearer\s+[A-Za-z0-9_\-/.]{8,}", _re.IGNORECASE)
_SK_RE = _re.compile(r"sk-[A-Za-z0-9]{8,}")
_KEY_LIKE_RE = _re.compile(r"(?:xai-|exa-|tvly-|BSA)[A-Za-z0-9_\-]{6,}")


def _scrub_secrets(text: str) -> str:
    """Replace likely API keys / tokens with a redacted placeholder."""
    text = _SENSITIVE_RE.sub("***REDACTED***", text)
    text = _BEARER_RE.sub("Bearer ***", text)
    text = _SK_RE.sub("sk-***", text)
    text = _KEY_LIKE_RE.sub("***REDACTED***", text)
    return text


def _truncate(text: str, max_len: int) -> str:
    """Truncate text if verbose logging is disabled, scrub secrets."""
    text = _scrub_secrets(text)
    if _verbose_logging or len(text) <= max_len:
        return text
    return text[:max_len] + "..."


class CogtrixFormatter(logging.Formatter):
    """Custom formatter with support for request IDs and varied formats."""

    def __init__(self, debug: bool = False):
        self.debug = debug
        if debug:
            # Detailed format with milliseconds and request ID
            fmt = "%(asctime)s.%(msecs)03d [%(levelname)s] [%(request_id)s] %(message)s"
            datefmt = "%Y-%m-%d %H:%M:%S"
        else:
            # Standard format
            fmt = "%(asctime)s [%(levelname)s] %(message)s"
            datefmt = "%Y-%m-%d %H:%M:%S"
        super().__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        # Add request_id to record if not present
        if not hasattr(record, "request_id"):
            record.request_id = _request_id.get() or "-"
        return super().format(record)


class CogtrixLoggerAdapter(logging.LoggerAdapter):  # type: ignore[type-arg]
    """Logger adapter that automatically includes request ID."""

    def process(self, msg: str, kwargs: dict) -> tuple:  # type: ignore[override]
        # Add request_id to extra
        extra = kwargs.get("extra", {})
        extra["request_id"] = _request_id.get() or "-"
        kwargs["extra"] = extra
        return msg, kwargs


class _MaxLevelFilter(logging.Filter):
    """Passes only records strictly below a given level (used to split stdout/stderr)."""

    def __init__(self, max_level: int) -> None:
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < self.max_level


def setup_logging(
    log_file: str | None = None,
    debug: bool = False,
    console_output: bool = False,
    verbose: bool = True,
    stream_output: bool = False,
    verbosity: int = 0,
) -> logging.Logger:
    """
    Configure logging for Cogtrix application.

    Args:
        log_file: Path to log file. If None and stream_output is False, logging
                  is disabled. If empty string, uses DEFAULT_LOG_FILE.
        debug: Enable debug level logging with verbose output.
        console_output: Also output logs to console (stderr).
        verbose: Log full message content without truncation (default: True).
        stream_output: Route logs to stdout/stderr instead of a file.
                       DEBUG/INFO → stdout; WARNING/ERROR/CRITICAL → stderr.
                       Ignored when log_file is explicitly provided.
        verbosity: Verbosity level 0–3.  0=INFO, 1+=DEBUG.  Levels 2+ enable
                   full tool I/O logging; level 3 enables LangGraph/memory trace
                   logging.  When provided, overrides the ``debug`` bool for
                   level selection and sets the module-level ``_verbosity`` flag
                   so ``is_verbose()`` / ``is_trace()`` reflect the active level.

    Returns:
        Configured logger instance.
    """
    global _verbose_logging
    # verbosity takes precedence over the legacy debug/verbose booleans when non-zero
    if verbosity:
        set_verbosity(verbosity)
        debug = verbosity >= 1
        verbose = verbosity >= 2
    else:
        _verbose_logging = verbose

    # Clear any existing handlers
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # --log-file always takes priority; stream_output only applies when no file is given
    if log_file is None and stream_output:
        fmt = CogtrixFormatter(debug=debug)
        level = logging.DEBUG if debug else logging.INFO

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(level)
        stdout_handler.addFilter(_MaxLevelFilter(logging.WARNING))
        stdout_handler.setFormatter(fmt)
        logger.addHandler(stdout_handler)

        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)
        stderr_handler.setFormatter(fmt)
        logger.addHandler(stderr_handler)

        logger.info("Logging initialized: stdout/stderr (debug=%s)", debug)
        return logger

    # If no log file specified, return logger without handlers (no-op logging)
    if log_file is None:
        logger.addHandler(logging.NullHandler())
        return logger

    # Use default log file if empty string
    if log_file == "":
        log_file = DEFAULT_LOG_FILE

    # Create log directory if needed
    log_path = Path(log_file)
    try:
        if log_path.parent and not log_path.parent.exists():
            log_path.parent.mkdir(parents=True, exist_ok=True)

        # File handler
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
        file_handler.setFormatter(CogtrixFormatter(debug=debug))
        logger.addHandler(file_handler)
    except OSError as exc:
        print(f"Warning: could not set up log file {log_file!r}: {exc}", file=sys.stderr)

    # Optional console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
        console_handler.setFormatter(CogtrixFormatter(debug=debug))
        logger.addHandler(console_handler)

    # Log initial message
    logger.info("Logging initialized: %s (debug=%s)", log_file, debug)

    return logger


def get_logger() -> CogtrixLoggerAdapter:
    """Get a logger adapter with request ID support."""
    return CogtrixLoggerAdapter(logger, {})


def new_request_id() -> str:
    """Generate and set a new request ID for the current context."""
    request_id = str(uuid.uuid4())[:8]
    _request_id.set(request_id)
    return request_id


def get_request_id() -> str:
    """Get the current request ID."""
    return _request_id.get() or "-"


def clear_request_id() -> None:
    """Clear the current request ID."""
    _request_id.set("")


# Convenience functions for structured logging
def log_user_message(message: str) -> None:
    """Log a user message."""
    log = get_logger()
    log.info("User: %s", _truncate(message, 200))


def log_agent_response(response: str, token_count: int | None = None) -> None:
    """Log an agent response."""
    log = get_logger()
    token_info = f" ({token_count} tokens)" if token_count else ""
    log.info("Agent response%s", token_info)
    log.debug("Agent: %s", _truncate(response, 500))


def log_tool_call(
    tool_name: str,
    inputs: dict | None = None,
    output: str | None = None,
    duration: float | None = None,
    error: str | None = None,
) -> None:
    """Log a tool call with optional details."""
    log = get_logger()

    if error:
        log.error("Tool failed: %s - %s", tool_name, error)
        if inputs:
            log.debug("Tool input: %s", inputs)
        return

    duration_str = f" ({duration:.2f}s)" if duration else ""
    log.info("Tool: %s%s", tool_name, duration_str)

    if inputs:
        log.debug("Tool input: %s", inputs)
    if output:
        log.debug("Tool output: %s", _truncate(output, 300))


def log_llm_call(
    provider: str,
    model: str,
    reasoning: str | None = None,
    token_count: int | None = None,
) -> None:
    """Log an LLM invocation."""
    log = get_logger()
    token_info = f", {token_count} tokens" if token_count else ""
    log.info("LLM: %s/%s%s", provider, model, token_info)

    if reasoning:
        log.debug("LLM reasoning: %s", _truncate(reasoning, 400))


def log_memory_context(
    mode: str,
    message_count: int,
    token_estimate: int | None = None,
) -> None:
    """Log memory context preparation."""
    log = get_logger()
    token_info = f", ~{token_estimate} tokens" if token_estimate else ""
    log.debug("Context: mode=%s, %d messages%s", mode, message_count, token_info)


def log_error(
    error: Exception,
    context: str | None = None,
    include_trace: bool = False,
) -> None:
    """Log an error with optional context and traceback."""
    log = get_logger()
    error_type = type(error).__name__
    error_msg = str(error)

    if context:
        log.error("%s: %s - %s", context, error_type, error_msg)
    else:
        log.error("%s: %s", error_type, error_msg)

    if include_trace:
        import traceback

        log.debug("Traceback:\n%s", traceback.format_exc())


def log_session_info(
    session_id: str,
    message_count: int,
    memory_mode: str,
    provider: str,
    model: str,
) -> None:
    """Log session information at startup."""
    log = get_logger()
    log.info("Session started: %s", session_id)
    log.info("Provider: %s/%s", provider, model)
    log.info("Memory mode: %s", memory_mode)
    log.debug("Existing messages: %d", message_count)


def log_delegation(
    target_model: str,
    task: str,
    response_format: str,
    success: bool,
    duration: float | None = None,
    error: str | None = None,
) -> None:
    """Log a delegation task."""
    log = get_logger()
    duration_str = f" ({duration:.2f}s)" if duration else ""

    if success:
        log.info("Delegation to %s: success%s", target_model, duration_str)
    else:
        log.error("Delegation to %s: failed - %s", target_model, error)

    log.debug("Delegation task: %s", _truncate(task, 200))
    log.debug("Response format: %s", response_format)


# LangChain callback handler for LLM observability
try:
    import json
    import time as _time_module
    from typing import Any

    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult

    class LLMObservabilityHandler(BaseCallbackHandler):
        """
        Callback handler that logs all LLM interactions for debugging.

        Captures:
        - LLM start/end events with prompts and responses
        - Token streaming (real-time output)
        - Tool calls and their results
        - Thinking/reasoning content from models that support it
        """

        def __init__(self, verbose: bool = True):
            """
            Initialize the handler.

            Args:
                verbose: If True, log full content. If False, truncate.
            """
            self.verbose = verbose
            self._current_tokens: list[str] = []
            self._token_count = 0
            self._start_time: float = 0.0

        def _log(self, level: str, message: str) -> None:
            """Log a message at the specified level."""
            message = _scrub_secrets(message)
            log = get_logger()
            if level == "debug":
                log.debug(message)
            elif level == "info":
                log.info(message)
            elif level == "warning":
                log.warning(message)
            elif level == "error":
                log.error(message)

        def _format_content(self, content: str, max_len: int = 500) -> str:
            """Format content with optional truncation."""
            content = _scrub_secrets(content)
            if self.verbose or len(content) <= max_len:
                return content
            return content[:max_len] + "..."

        def on_llm_start(
            self,
            serialized: dict[str, Any],
            prompts: list[str],
            **kwargs: Any,
        ) -> None:
            """Called when LLM starts processing."""
            model = serialized.get("kwargs", {}).get("model", "unknown")
            self._log("debug", f"LLM_START: model={model}, prompts={len(prompts)}")
            self._current_tokens = []
            self._token_count = 0

        def on_chat_model_start(
            self,
            serialized: dict[str, Any],
            messages: list[list[BaseMessage]],
            **kwargs: Any,
        ) -> None:
            """Called when chat model starts processing."""
            model = serialized.get("kwargs", {}).get("model", "unknown")
            total_msgs = sum(len(batch) for batch in messages)

            # Estimate context size in characters and tokens
            total_chars = 0
            for batch in messages:
                for msg in batch:
                    if hasattr(msg, "content"):
                        content = msg.content
                        if isinstance(content, str):
                            total_chars += len(content)
                        elif isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and "text" in item:
                                    total_chars += len(item["text"])
                                elif isinstance(item, str):
                                    total_chars += len(item)
            estimated_tokens = total_chars // 4  # Rough estimate

            self._log(
                "info",
                f"LLM_CHAT_START: model={model}, messages={total_msgs}, "
                f"~{total_chars:,} chars, ~{estimated_tokens:,} tokens",
            )
            self._log("debug", "LLM_PROMPT_EVAL: waiting for model to process context...")
            self._start_time = _time_module.time()
            self._current_tokens = []
            self._token_count = 0

        def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
            """Called for each new token streamed from the LLM."""
            # Log first token with prompt eval time
            if self._token_count == 0 and self._start_time > 0:
                elapsed = _time_module.time() - self._start_time
                self._log(
                    "info",
                    f"LLM_FIRST_TOKEN: prompt eval completed in {elapsed:.2f}s",
                )

            self._current_tokens.append(token)
            self._token_count += 1

            # Log each token in verbose mode for debugging
            if self.verbose and token.strip():
                self._log("debug", f"LLM_TOKEN: {repr(token)}")

            # Check for chunk metadata (may contain thinking content)
            chunk = kwargs.get("chunk")
            if chunk:
                # Some models provide additional_kwargs with thinking content
                if hasattr(chunk, "additional_kwargs") and chunk.additional_kwargs:
                    ak = chunk.additional_kwargs
                    for key in ["thinking", "reasoning", "thought"]:
                        if key in ak and ak[key]:
                            self._log("debug", f"LLM_CHUNK_THINKING: {ak[key]}")

            # Log periodically to show progress (every 50 tokens)
            if self._token_count % 50 == 0:
                partial = "".join(self._current_tokens[-100:])  # Last 100 tokens
                self._log(
                    "debug",
                    f"LLM_STREAMING: {self._token_count} tokens... {repr(partial[-200:])}",
                )

        def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
            """Called when LLM finishes processing."""
            # Log total time
            if self._start_time > 0:
                total_time = _time_module.time() - self._start_time
                self._log(
                    "info",
                    f"LLM_COMPLETE: {self._token_count} tokens in {total_time:.2f}s",
                )
            # Log final streamed content if we collected tokens
            if self._current_tokens:
                full_response = "".join(self._current_tokens)
                self._log("debug", f"LLM_STREAM_COMPLETE: {self._token_count} tokens")
                self._log(
                    "debug",
                    f"LLM_FULL_OUTPUT: {self._format_content(full_response, 2000)}",
                )
                self._current_tokens = []

            # Also log from the response object
            if response.generations:
                for gen_list in response.generations:
                    for gen in gen_list:
                        text = gen.text if hasattr(gen, "text") else str(gen)
                        if text:
                            self._log(
                                "debug",
                                f"LLM_GENERATION: {self._format_content(text, 2000)}",
                            )

                        # Log generation info (includes tool calls, thinking, etc.)
                        if hasattr(gen, "generation_info") and gen.generation_info:
                            info = gen.generation_info
                            self._log(
                                "debug",
                                f"LLM_GEN_INFO: {json.dumps(info, default=str)}",
                            )

                        # Log message content for chat models
                        if hasattr(gen, "message"):
                            msg = gen.message
                            # Log content (may be string or list for multimodal)
                            if hasattr(msg, "content") and msg.content:
                                content_str = str(msg.content)
                                # Check for thinking tags in content
                                if "<think>" in content_str or "</think>" in content_str:
                                    formatted = self._format_content(content_str, 5000)
                                    self._log("debug", f"LLM_THINKING_RAW: {formatted}")
                                else:
                                    formatted = self._format_content(content_str, 2000)
                                    self._log("debug", f"LLM_MESSAGE: {formatted}")

                            # Log response_metadata (Ollama uses this)
                            if hasattr(msg, "response_metadata") and msg.response_metadata:
                                rm = msg.response_metadata
                                self._log(
                                    "debug",
                                    f"LLM_RESPONSE_META: {json.dumps(rm, default=str)}",
                                )

                            # Log tool calls
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    tool_name = tc.get("name", "unknown")
                                    tool_args = tc.get("args", {})
                                    args_str = _scrub_secrets(str(tool_args))
                                    self._log(
                                        "info",
                                        f"LLM_TOOL_CALL: {tool_name} args={args_str}",
                                    )

                            # Log additional kwargs (may contain thinking, reasoning)
                            if hasattr(msg, "additional_kwargs") and msg.additional_kwargs:
                                ak = msg.additional_kwargs
                                # Check for thinking/reasoning content
                                for key in [
                                    "thinking",
                                    "reasoning",
                                    "thought",
                                    "thoughts",
                                ]:
                                    if key in ak:
                                        content = self._format_content(str(ak[key]), 5000)
                                        self._log(
                                            "debug",
                                            f"LLM_THINKING ({key}): {content}",
                                        )

                                # Log any other additional kwargs
                                other_keys = [
                                    k
                                    for k in ak.keys()
                                    if k
                                    not in [
                                        "thinking",
                                        "reasoning",
                                        "thought",
                                        "thoughts",
                                    ]
                                ]
                                if other_keys:
                                    other_data = {k: ak[k] for k in other_keys}
                                    self._log(
                                        "debug",
                                        f"LLM_ADDITIONAL: {json.dumps(other_data, default=str)}",
                                    )

            # Log token usage if available
            if response.llm_output:
                usage = response.llm_output.get("token_usage") or response.llm_output.get("usage")
                if usage:
                    self._log("info", f"LLM_TOKENS: {usage}")

        def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
            """Called when LLM encounters an error."""
            self._log("error", f"LLM_ERROR: {type(error).__name__}: {error}")

        def on_tool_start(
            self,
            serialized: dict[str, Any],
            input_str: str,
            **kwargs: Any,
        ) -> None:
            """Called when a tool starts executing."""
            tool_name = serialized.get("name", "unknown")
            self._log("info", f"TOOL_START: {tool_name}")
            self._log(
                "debug", f"TOOL_INPUT: {self._format_content(_scrub_secrets(input_str), 1000)}"
            )

        def on_tool_end(self, output: Any, **kwargs: Any) -> None:
            """Called when a tool finishes executing."""
            if not isinstance(output, str):
                output = getattr(output, "content", None) or str(output)
            self._log("debug", f"TOOL_OUTPUT: {self._format_content(output, 1000)}")

        def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
            """Called when a tool encounters an error."""
            self._log("error", f"TOOL_ERROR: {type(error).__name__}: {error}")

        def on_agent_action(self, action: Any, **kwargs: Any) -> None:
            """Called when agent takes an action."""
            tool = getattr(action, "tool", "unknown")
            tool_input = getattr(action, "tool_input", {})
            self._log("info", f"AGENT_ACTION: {tool}")
            self._log("debug", f"AGENT_ACTION_INPUT: {tool_input}")

        def on_agent_finish(self, finish: Any, **kwargs: Any) -> None:
            """Called when agent finishes."""
            output = getattr(finish, "return_values", {})
            self._log("info", f"AGENT_FINISH: {self._format_content(str(output), 500)}")

    CALLBACK_HANDLER_AVAILABLE = True

except ImportError:
    LLMObservabilityHandler = None  # type: ignore[misc, assignment]
    CALLBACK_HANDLER_AVAILABLE = False


def create_observability_handler(verbose: bool = True) -> Any | None:
    """
    Create an LLM observability callback handler.

    Args:
        verbose: If True, log full content without truncation.

    Returns:
        LLMObservabilityHandler instance or None if not available.
    """
    if CALLBACK_HANDLER_AVAILABLE and LLMObservabilityHandler is not None:
        return LLMObservabilityHandler(verbose=verbose)
    return None
