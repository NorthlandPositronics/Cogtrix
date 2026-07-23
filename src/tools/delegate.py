"""
Delegate Tool: Delegate tasks to other LLM models.

Enables the primary agent to offload subtasks to other LLM models,
supporting parallel execution, structured JSON responses, and timeouts.
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from src.logging_config import get_logger, log_delegation

log = get_logger()

# LangChain imports with graceful fallback
try:
    from langchain_core.messages import HumanMessage, SystemMessage

    LANGCHAIN_MESSAGES_AVAILABLE = True
except ImportError:
    HumanMessage = None  # type: ignore[misc, assignment]
    SystemMessage = None  # type: ignore[misc, assignment]
    LANGCHAIN_MESSAGES_AVAILABLE = False


# Module-level configuration (set by cogtrix.py at startup)
_delegate_config: dict[str, Any] = {
    "enabled": True,
    "default_timeout": 60,
    "default_provider": "ollama",
    "default_model": None,
    "allowed_providers": ["openai", "ollama", "anthropic", "google"],
    "models": {},
    "providers": {},  # Named provider configurations
    "max_consecutive_failures": 5,
    "circuit_breaker_cooldown": 300,  # seconds before retry
}

# Tools available to delegate agents.  Set by the host application
# via ``set_delegate_tools()``.  When non-empty, delegates can run
# as full ReAct agents with tool access instead of plain LLM calls.
# Stored per-thread so concurrent assistant-mode sessions don't overwrite each other.
_delegate_tools_tls: threading.local = threading.local()


def get_delegate_tools() -> list[Any]:
    """Return the delegate tool list for the current thread."""
    return getattr(_delegate_tools_tls, "tools", [])


# Names excluded from the delegate tool set to prevent recursion
# and keep delegate execution fast.
_DELEGATE_EXCLUDED_TOOLS = frozenset(
    {
        "delegate_task",
        "delegate_parallel",
        "deep_think",
        "request_tools",
    }
)


def set_delegate_tools(
    active_tools: list[Any],
    available_tools: dict[str, Any] | None = None,
) -> None:
    """Register tools that delegate agents can use.

    Delegates receive **all** tools — both the currently active set and
    any on-demand tools that the main agent hasn't activated yet.  This
    means delegates can read files, run shell commands, search the web,
    etc. from the very first invocation without needing to request tools.

    Delegation tools and ``deep_think`` are automatically excluded to
    prevent recursion.

    Parameters
    ----------
    active_tools:
        The main agent's currently active tool list.
    available_tools:
        On-demand tools not yet active in the main agent.  Merged into
        the delegate toolset so delegates have full capabilities.
    """
    seen: set[str] = set()
    merged: list[Any] = []

    for t in active_tools:
        name = getattr(t, "name", "")
        if name not in _DELEGATE_EXCLUDED_TOOLS and name not in seen:
            merged.append(t)
            seen.add(name)

    if available_tools:
        for name, t in available_tools.items():
            if name not in _DELEGATE_EXCLUDED_TOOLS and name not in seen:
                merged.append(t)
                seen.add(name)

    _delegate_tools_tls.tools = merged


# Optional callback for real-time delegation status messages.
# Signature: (message: str) -> None
# Set by the host application (e.g. cogtrix.py) to display delegation
# activity in the UI while the spinner is running.
_status_callback: Any = None
_status_callback_lock = threading.Lock()


def set_status_callback(callback) -> None:
    """Register a callback that receives delegation status messages."""
    global _status_callback
    with _status_callback_lock:
        _status_callback = callback


def _emit_status(message: str) -> None:
    """Emit a status message if a callback is registered."""
    with _status_callback_lock:
        cb = _status_callback
    if cb is not None:
        try:
            cb(message)
        except Exception as exc:
            log.debug("Status callback error: %s", exc)


@dataclass
class ModelCircuitBreaker:
    """Track model availability status for circuit breaker pattern."""

    consecutive_failures: int = 0
    is_unavailable: bool = False
    last_failure_time: float = 0.0
    last_error: str = ""
    last_used: float = 0.0

    def record_failure(self, error: str, max_failures: int = 5) -> bool:
        """Record a failure and potentially mark model as unavailable.

        Returns True if the circuit breaker just tripped (model now unavailable).
        """
        self.consecutive_failures += 1
        self.last_failure_time = time.time()
        self.last_error = error
        if self.consecutive_failures >= max_failures:
            self.is_unavailable = True
            return True
        return False

    def record_success(self) -> None:
        """Record a success and reset failure count."""
        self.consecutive_failures = 0
        self.is_unavailable = False
        self.last_error = ""

    def _check_availability_locked(self, cooldown: float = 300.0) -> tuple[bool, str | None]:
        """Check if model is available.

        Caller MUST hold ``_circuit_breaker_lock`` before calling this method.

        Returns:
            Tuple of (is_available, reason_if_unavailable)
        """
        if not self.is_unavailable:
            return True, None

        elapsed = time.time() - self.last_failure_time
        if elapsed >= cooldown:
            self.is_unavailable = False
            self.consecutive_failures = 0
            return True, None

        remaining = int(cooldown - elapsed)
        return False, (
            f"Model marked unavailable after {self.consecutive_failures} "
            f"consecutive failures. Last error: {self.last_error}. "
            f"Will retry in {remaining}s."
        )

    def check_availability(self, cooldown: float = 300.0) -> tuple[bool, str | None]:
        """Check if model is available. Thread-safe; acquires ``_circuit_breaker_lock``.

        Returns:
            Tuple of (is_available, reason_if_unavailable)
        """
        with _circuit_breaker_lock:
            return self._check_availability_locked(cooldown)


# Circuit breaker registry: tracks failures per "provider/model" key
_circuit_breakers: dict[str, ModelCircuitBreaker] = {}
_circuit_breaker_lock = threading.RLock()

_MAX_CIRCUIT_BREAKERS = 200

_CIRCUIT_BREAKER_IDLE_SECONDS = 3600.0  # 1 hour


# Must only be called while holding _circuit_breaker_lock.
def _evict_stale_breakers() -> None:
    """Remove stale entries from the circuit breaker registry.

    Two-pass eviction:
    1. Remove entries with zero consecutive failures that haven't been used
       within the idle window — they carry no state worth keeping.
    2. If the registry still exceeds the cap, remove the oldest entries by
       ``last_used`` timestamp until it fits within the limit.
    """
    now = time.time()
    idle_cutoff = now - _CIRCUIT_BREAKER_IDLE_SECONDS

    stale_keys = [
        k
        for k, b in _circuit_breakers.items()
        if b.consecutive_failures == 0 and b.last_used < idle_cutoff
    ]
    for k in stale_keys:
        del _circuit_breakers[k]

    if len(_circuit_breakers) >= _MAX_CIRCUIT_BREAKERS:
        sorted_keys = sorted(_circuit_breakers, key=lambda k: _circuit_breakers[k].last_used)
        excess = len(_circuit_breakers) - _MAX_CIRCUIT_BREAKERS + 1
        for k in sorted_keys[:excess]:
            del _circuit_breakers[k]


def _get_circuit_breaker(provider: str, model: str) -> ModelCircuitBreaker:
    """Get or create circuit breaker for a provider/model combination."""
    key = f"{provider}/{model}"
    with _circuit_breaker_lock:
        if key not in _circuit_breakers:
            if len(_circuit_breakers) >= _MAX_CIRCUIT_BREAKERS:
                _evict_stale_breakers()
            _circuit_breakers[key] = ModelCircuitBreaker(last_used=time.time())
        else:
            _circuit_breakers[key].last_used = time.time()
        return _circuit_breakers[key]


def get_model_status() -> dict[str, Any]:
    """
    Get status of all tracked models.

    Returns:
        Dictionary with model availability status.
    """
    cooldown = _delegate_config.get("circuit_breaker_cooldown", 300)
    with _circuit_breaker_lock:
        snapshot = list(_circuit_breakers.items())

    status = {}
    for key, breaker in snapshot:
        with _circuit_breaker_lock:
            available, reason = breaker._check_availability_locked(cooldown)
            consecutive_failures = breaker.consecutive_failures
            last_error = breaker.last_error if not available else None
        status[key] = {
            "available": available,
            "consecutive_failures": consecutive_failures,
            "last_error": last_error,
            "reason": reason,
        }
    return status


def reset_model_status(provider: str | None = None, model: str | None = None):
    """
    Reset circuit breaker for a specific model or all models.

    Args:
        provider: Provider name (if None with model=None, resets all)
        model: Model name
    """
    with _circuit_breaker_lock:
        if provider is None and model is None:
            _circuit_breakers.clear()
        elif provider and model:
            key = f"{provider}/{model}"
            if key in _circuit_breakers:
                _circuit_breakers[key] = ModelCircuitBreaker()


def configure_delegate(config: dict[str, Any]) -> None:
    """
    Configure the delegate tool with runtime settings.

    Called by cogtrix.py at startup to pass configuration.
    """
    global _delegate_config
    # Atomic reference swap — safe for concurrent readers without a lock
    _delegate_config = {**_delegate_config, **config}


@dataclass
class DelegateResult:
    """Result from a delegated task."""

    success: bool
    response: str
    format_valid: bool
    parsed_json: dict[str, Any] | None
    model_used: str
    provider: str
    duration_seconds: float
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "response": self.response,
            "format_valid": self.format_valid,
            "parsed_json": self.parsed_json,
            "model_used": self.model_used,
            "provider": self.provider,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
        }


class DelegateInput(BaseModel):
    """Input schema for single task delegation."""

    task: str = Field(description="Clear description of what the delegated model should do")
    context: str = Field(
        default="",
        description=(
            "Relevant context/data for the task (code, text, documents, etc.). "
            "Required when use_tools=False. When use_tools=True the delegate "
            "can gather data itself, but providing context still helps."
        ),
    )
    use_tools: bool = Field(
        default=True,
        description=(
            "When True (default), the delegate runs as a full agent with "
            "tool access (file I/O, shell, search, etc.) and can gather "
            "data independently. When False, the delegate is LLM-only and "
            "can only reason about text provided in 'context'."
        ),
    )
    response_format: str = Field(
        default="text",
        description="Expected response format: 'text', 'json', 'code', 'markdown'",
    )
    json_schema: str | None = Field(
        default=None,
        description="If response_format='json', describe the expected JSON structure",
    )
    provider: str | None = Field(
        default=None,
        description="LLM provider: 'openai', 'ollama', 'anthropic', 'google', or alias from config",
    )
    model: str | None = Field(
        default=None,
        description="Model name (e.g., 'gpt-4.1-mini', 'qwen3:8b', 'claude-sonnet-4-5') or alias from config",
    )
    timeout: int = Field(
        default=60,
        description="Maximum seconds to wait for response (10-300)",
    )
    temperature: float = Field(
        default=0.7,
        description="Model temperature (0.0-2.0). Lower = more deterministic",
    )


class DelegateParallelInput(BaseModel):
    """Input schema for parallel task delegation."""

    tasks: list[dict[str, Any]] = Field(
        description=(
            "List of tasks to run in parallel. Each task is a dict with: "
            "'task' (required), 'context', 'use_tools' (default True), "
            "'response_format', 'json_schema', 'provider', 'model', "
            "'temperature'"
        )
    )
    timeout: int = Field(
        default=120,
        description="Maximum seconds to wait for ALL tasks to complete (30-600)",
    )


def resolve_model_alias(provider: str | None, model: str | None) -> tuple:
    """
    Resolve model aliases from configuration.

    Supports two alias formats:
    1. String: "provider/model" or just "model"
    2. Object: {"provider": "...", "model": "...", "timeout": 300, "temperature": 0.5, "context_window": 32768}

    Returns:
        Tuple of (resolved_provider, resolved_model, alias_config)
        where alias_config is a dict with optional 'timeout', 'temperature', 'context_window' overrides
    """
    aliases = _delegate_config.get("models", {})
    alias_config: dict[str, Any] = {}

    def _extract_alias_config(alias_value: dict) -> None:
        """Extract config values from alias dict into alias_config."""
        if "timeout" in alias_value:
            alias_config["timeout"] = alias_value["timeout"]
        if "temperature" in alias_value:
            alias_config["temperature"] = alias_value["temperature"]
        if "context_window" in alias_value:
            alias_config["context_window"] = alias_value["context_window"]
        elif "context_length" in alias_value:
            alias_config["context_window"] = alias_value["context_length"]
        elif "num_ctx" in alias_value:
            alias_config["context_window"] = alias_value["num_ctx"]

    # Check if model is an alias
    if model and model in aliases:
        alias_value = aliases[model]

        # Object format: {"provider": "...", "model": "...", "timeout": 300, "context_window": 32768}
        if isinstance(alias_value, dict):
            resolved_provider = alias_value.get("provider", provider)
            resolved_model = alias_value.get("model", model)
            _extract_alias_config(alias_value)
            return resolved_provider, resolved_model, alias_config

        # String format: "provider/model"
        if "/" in alias_value:
            parts = alias_value.split("/", 1)
            return parts[0], parts[1], alias_config
        else:
            # Just model name, use provided or default provider
            return provider, alias_value, alias_config

    # Check if provider is an alias (e.g., "fast" -> "ollama")
    if provider and provider in aliases:
        alias_value = aliases[provider]

        # Object format
        if isinstance(alias_value, dict):
            resolved_provider = alias_value.get("provider", provider)
            resolved_model = model or alias_value.get("model")
            _extract_alias_config(alias_value)
            return resolved_provider, resolved_model, alias_config

        # String format
        if "/" in alias_value:
            parts = alias_value.split("/", 1)
            return parts[0], model or parts[1], alias_config
        else:
            return alias_value, model, alias_config

    return provider, model, alias_config


def create_delegate_llm(
    provider: str,
    model: str | None = None,
    temperature: float = 0.5,
    num_ctx: int | None = None,
) -> Any:
    """
    Create an LLM instance for the specified provider.

    Delegates to the centralized ``src.providers`` registry.
    Supports named providers from config and legacy provider names.

    Args:
        provider: Provider name (e.g., 'openai', 'ollama', 'anthropic', 'my-server')
        model: Model name (overrides provider config default)
        temperature: Sampling temperature
        num_ctx: Context window size in tokens (Ollama only)

    Returns:
        LLM instance

    Raises:
        ValueError: If provider is not supported or not available
    """
    from src.providers import create_chat_model

    allowed = _delegate_config.get("allowed_providers", ["openai", "ollama", "anthropic", "google"])
    if provider not in allowed:
        raise ValueError(f"Provider '{provider}' not allowed. Allowed: {allowed}")

    providers = _delegate_config.get("providers", {})

    # Check for named provider in config
    if provider in providers:
        prov_cfg = providers[provider]
        prov_type = prov_cfg.get("type", provider)
        # model must come from caller (resolved via alias); providers carry only
        # connection info in the new format.  The legacy "model" key in prov_cfg
        # is kept as a fallback for old config dict shapes.
        final_model = model or prov_cfg.get("model")

        return create_chat_model(
            prov_type,
            model=final_model,
            api_key=prov_cfg.get("api_key"),
            base_url=prov_cfg.get("base_url"),
            temperature=temperature,
            num_ctx=num_ctx,
        )

    # Provider not in named config — raise a clear error
    raise ValueError(
        f"Provider '{provider}' is not configured. "
        f"Add it to the 'providers' section of your config file."
    )


def _build_prompt(
    task: str,
    context: str,
    response_format: str,
    json_schema: str | None,
) -> list[Any]:
    """
    Build the prompt messages for the delegated task.

    Returns:
        List of messages (SystemMessage, HumanMessage)
    """
    # Build system message based on response format
    system_parts = ["You are a helpful assistant completing a delegated task."]

    if response_format == "json":
        system_parts.append("You MUST respond with valid JSON only. No explanations or markdown.")
        if json_schema:
            system_parts.append(f"Expected JSON structure: {json_schema}")
    elif response_format == "code":
        system_parts.append("Respond with code only. No explanations unless in comments.")
    elif response_format == "markdown":
        system_parts.append("Format your response using Markdown.")

    system_content = "\n".join(system_parts)

    # Build user message
    user_parts = [f"**Task:** {task}"]
    if context:
        user_parts.append(f"\n**Context:**\n{context}")

    user_content = "\n".join(user_parts)

    if LANGCHAIN_MESSAGES_AVAILABLE:
        return [
            SystemMessage(content=system_content),
            HumanMessage(content=user_content),
        ]
    else:
        # Fallback to dict format
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]


def _validate_json_response(response: str) -> tuple:
    """
    Validate and parse JSON response.

    Returns:
        Tuple of (is_valid, parsed_json_or_none)
    """
    # Try to extract JSON from response (might be wrapped in markdown)
    text = response.strip()

    # Remove markdown code blocks if present — search instead of startswith
    # so prose written before the fence is tolerated
    fence_idx = text.find("```json")
    if fence_idx != -1:
        text = text[fence_idx + 7 :]
        # Find the matching closing fence (first ``` after the opening)
        close_idx = text.find("```")
        if close_idx != -1:
            text = text[:close_idx]
    elif text.startswith("```"):
        text = text[3:]
        close_idx = text.find("```")
        if close_idx != -1:
            text = text[:close_idx]
    text = text.strip()

    try:
        parsed = json.loads(text)
        return True, parsed
    except json.JSONDecodeError:
        return False, None


def resolve_delegate_defaults(
    provider: str | None,
    model: str | None,
) -> tuple[str, str]:
    """Apply default provider/model when values are still ``None``.

    Call **after** alias resolution so defaults only fill in gaps.
    Resolves through ``default_model_alias`` → models dict when present;
    falls back to legacy ``default_provider``/``default_model`` keys.

    Returns:
        ``(provider, model)`` with no ``None`` values.
    """
    if provider and model:
        return provider, model

    alias = _delegate_config.get("default_model_alias")
    models = _delegate_config.get("models", {})

    if alias and alias in models:
        entry = models[alias]
        resolved_provider: str = provider or entry.get("provider", "ollama")
        resolved_model: str = model or entry.get("model", alias)
    else:
        # Alias not in models — try legacy keys, then treat alias as literal model name
        resolved_provider = provider or str(_delegate_config.get("default_provider", "ollama"))
        resolved_model = model or str(_delegate_config.get("default_model") or alias or "default")

    return resolved_provider, resolved_model


def _check_allowed_model(model: str | None) -> str | None:
    """Validate that the requested model/alias is in ``allowed_models``.

    If ``allowed_models`` is not configured (``None``), every model is
    permitted (backward-compatible behaviour).

    Returns:
        ``None`` if the model is allowed, or an error message string.
    """
    allowed: list[str] | None = _delegate_config.get("allowed_models")
    if allowed is None:
        return None  # no restriction

    if not model:
        return None  # no model specified; defaults will apply

    if model in allowed:
        return None  # explicitly allowed

    # Also accept aliases that resolve to an allowed alias
    # (e.g. "code" and "coder" might both be defined)
    return (
        f"Model '{model}' is not in the allowed delegation list. "
        f"Allowed models: {', '.join(allowed)}"
    )


_DELEGATE_AGENT_RECURSION_LIMIT = 30
_DELEGATE_AGENT_SYSTEM_PROMPT = (
    "You are a delegate agent completing a specific task. "
    "Use your tools to gather information and complete the task. "
    "Be thorough but efficient — you have a limited number of steps. "
    "When done, provide your final answer directly.\n\n"
    "IMPORTANT:\n"
    "- Do NOT delegate to other models (you have no delegation tools).\n"
    "- Do NOT say 'I need more steps'. Deliver what you have.\n"
    "- Focus on the specific task assigned to you.\n"
)


def _extract_content(result: Any) -> str:
    """Extract text content from an LLM response object.

    Multi-modal responses (tool_use blocks, reasoning blocks) are reduced
    to their text parts.  Non-text blocks are logged at debug level.
    """
    if hasattr(result, "content"):
        content = result.content
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
                elif isinstance(part, dict):
                    block_type = part.get("type", "unknown")
                    log.debug(
                        "_extract_content: dropping non-text content block of type '%s'",
                        block_type,
                    )
            return "\n".join(text_parts) if text_parts else str(content)
        return str(content) if content is not None else ""
    return str(result)


def run_delegate_agent(
    llm: Any,
    task: str,
    context: str,
    tools_override: list[Any] | None = None,
) -> str:
    """Run a delegate as a full ReAct agent with tool access.

    Returns the final text response from the agent.  Falls back to
    plain LLM invocation if the agent framework is unavailable or
    the model does not support tool calling.
    """
    log = get_logger()

    use_tools = tools_override if tools_override is not None else get_delegate_tools()

    try:
        from langgraph.prebuilt import create_react_agent
    except ImportError:
        log.debug("LangGraph unavailable — falling back to plain LLM for delegate")
        return ""  # empty signals caller to fall back

    prompt = _DELEGATE_AGENT_SYSTEM_PROMPT
    if context:
        prompt += f"\n## Provided Context\n\n{context}\n"

    try:
        agent = create_react_agent(
            model=llm,
            tools=use_tools,
            prompt=prompt,
        )
    except Exception as exc:
        log.warning("Delegate agent creation/execution failed: %s", exc, exc_info=True)
        return f"Error: delegate agent failed ({type(exc).__name__}: {exc})"

    try:
        if LANGCHAIN_MESSAGES_AVAILABLE:
            input_msg = HumanMessage(content=task)
        else:
            input_msg = {"role": "user", "content": task}

        result = agent.invoke(
            {"messages": [input_msg]},
            {"recursion_limit": _DELEGATE_AGENT_RECURSION_LIMIT},
        )

        messages = result.get("messages", [])
        for msg in reversed(messages):
            msg_content = _extract_content(msg)
            if not msg_content.strip():
                continue
            msg_type = type(msg).__name__.lower()
            if msg_type in ("aimessage", "ai"):
                if not getattr(msg, "tool_calls", None):
                    return msg_content
        return _extract_content(messages[-1]) if messages else ""

    except Exception as exc:
        if "recursion" in type(exc).__name__.lower() or "recursion" in str(exc).lower():
            log.warning("Delegate exceeded step limit: %s", exc, exc_info=True)
            return f"Error: delegate exceeded step limit ({exc})"
        log.warning("Delegate agent creation/execution failed: %s", exc, exc_info=True)
        return f"Error: delegate agent failed ({type(exc).__name__}: {exc})"


def _execute_single_task(
    task: str,
    context: str = "",
    response_format: str = "text",
    json_schema: str | None = None,
    provider: str = "ollama",
    model: str = "default",
    temperature: float = 0.5,
    num_ctx: int | None = None,
    use_tools: bool = True,
) -> DelegateResult:
    """
    Execute a single delegated task.

    When *use_tools* is ``True`` and delegate tools are configured,
    the task runs as a full ReAct agent with tool access.  Otherwise
    it falls back to a plain LLM call.

    **Callers must resolve aliases and defaults before calling this
    function.**  Use ``resolve_model_alias`` followed by
    ``resolve_delegate_defaults`` to prepare the arguments.

    Includes circuit breaker logic to prevent repeated calls to failing
    models.

    Returns:
        DelegateResult with response and metadata
    """
    log = get_logger()
    start_time = time.time()

    target_model = f"{provider}/{model}"
    log.debug("Delegation starting: %s", target_model)
    log.debug("Task: %s%s", task[:100], "..." if len(task) > 100 else "")

    # Check circuit breaker status
    circuit_breaker = _get_circuit_breaker(provider, model)
    cooldown = _delegate_config.get("circuit_breaker_cooldown", 300)
    with _circuit_breaker_lock:
        is_available, unavailable_reason = circuit_breaker._check_availability_locked(cooldown)

    if not is_available:
        log.info("Delegation blocked: %s - circuit breaker open", target_model)
        return DelegateResult(
            success=False,
            response="",
            format_valid=False,
            parsed_json=None,
            model_used=model,
            provider=provider,
            duration_seconds=0,
            error=unavailable_reason,
        )

    try:
        # Create LLM
        llm = create_delegate_llm(provider, model, temperature, num_ctx)

        # ── Agent mode: run as full ReAct agent with tools ──────
        response_text = ""
        if use_tools and get_delegate_tools():
            log.debug(
                "Running delegate agent with %d tools: %s",
                len(get_delegate_tools()),
                target_model,
            )
            response_text = run_delegate_agent(llm, task, context)

        # ── Fallback: plain LLM call ────────────────────────────
        if not response_text:
            if not use_tools:
                log.debug("Invoking delegate LLM (no tools): %s", target_model)
            else:
                log.debug("Invoking delegate LLM (agent fallback): %s", target_model)
            messages = _build_prompt(task, context, response_format, json_schema)
            result = llm.invoke(messages)
            response_text = _extract_content(result)

        # Validate JSON if requested
        format_valid = True
        parsed_json = None
        if response_format == "json":
            format_valid, parsed_json = _validate_json_response(response_text)

        duration = time.time() - start_time
        model_name = model or (llm.model if hasattr(llm, "model") else "unknown")

        # Record success - reset circuit breaker
        with _circuit_breaker_lock:
            circuit_breaker.record_success()

        # Log successful delegation
        log_delegation(
            target_model=target_model,
            task=task,
            response_format=response_format,
            success=True,
            duration=duration,
        )

        return DelegateResult(
            success=True,
            response=response_text,
            format_valid=format_valid,
            parsed_json=parsed_json,
            model_used=str(model_name),
            provider=provider,
            duration_seconds=round(duration, 2),
            error=None,
        )

    except Exception as e:
        duration = time.time() - start_time
        error_msg = str(e)

        # Record failure in circuit breaker
        max_failures = _delegate_config.get("max_consecutive_failures", 5)
        with _circuit_breaker_lock:
            tripped = circuit_breaker.record_failure(error_msg, max_failures)
            failure_count = circuit_breaker.consecutive_failures

        # Add circuit breaker status to error if model is now unavailable
        if tripped:
            error_msg = (
                f"{error_msg} | Model marked unavailable after "
                f"{failure_count} consecutive failures."
            )

        # Log failed delegation
        log_delegation(
            target_model=target_model,
            task=task,
            response_format=response_format,
            success=False,
            duration=duration,
            error=error_msg,
        )

        return DelegateResult(
            success=False,
            response="",
            format_valid=False,
            parsed_json=None,
            model_used=model or "unknown",
            provider=provider or "unknown",
            duration_seconds=round(duration, 2),
            error=error_msg,
        )


def delegate_task(
    task: str,
    context: str = "",
    use_tools: bool = True,
    response_format: str = "text",
    json_schema: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    timeout: int = 60,
    temperature: float = 0.5,
) -> str:
    """
    Delegate a task to another LLM model.

    When *use_tools* is True (default), the delegate runs as a full
    agent with tool access (file I/O, shell, search, etc.).  When
    False, the delegate is LLM-only and requires non-empty *context*.

    Args:
        task: Clear description of what the delegated model should do
        context: Relevant context/data for the task
        use_tools: Whether the delegate can use tools (default True)
        response_format: Expected format ('text', 'json', 'code', 'markdown')
        json_schema: If JSON, the expected structure description
        provider: LLM provider ('openai', 'ollama') or alias
        model: Model name or alias from config
        timeout: Maximum seconds to wait (10-300), can be overridden by alias config
        temperature: Model temperature (0.0-2.0), can be overridden by alias config

    Returns:
        Formatted string with the delegation result
    """
    if not _delegate_config.get("enabled", True):
        return "**Delegation disabled.** Enable in configuration."

    # Reject LLM-only delegation with empty context
    if not use_tools and not context.strip():
        return (
            "**Delegation rejected:** context is empty and use_tools=False. "
            "Either provide context data for the delegate to analyse, or "
            "set use_tools=True so the delegate can gather data itself."
        )

    # Validate against allowed_models before any resolution
    error = _check_allowed_model(model)
    if error:
        return f"**Delegation blocked:** {error}"

    # Resolve aliases first to get any timeout/temperature/context_window overrides
    resolved_provider, resolved_model, alias_config = resolve_model_alias(provider, model)

    # Apply alias overrides (alias config takes precedence over parameters)
    if "timeout" in alias_config:
        timeout = alias_config["timeout"]
    if "temperature" in alias_config:
        temperature = alias_config["temperature"]
    num_ctx = alias_config.get("context_window") or alias_config.get("num_ctx")

    # Fill in defaults for anything still None after alias resolution
    resolved_provider, resolved_model = resolve_delegate_defaults(resolved_provider, resolved_model)

    # Clamp timeout and temperature to valid ranges
    timeout = max(10, min(600, timeout))  # Allow up to 600s for reasoning models
    temperature = max(0.0, min(2.0, temperature))

    # Show delegation activity to the user
    alias_hint = f" (alias: {model})" if model and model != resolved_model else ""
    mode_hint = " (with tools)" if use_tools and get_delegate_tools() else ""
    _emit_status(f"→ Delegating to {resolved_provider}/{resolved_model}" f"{alias_hint}{mode_hint}")

    # Execute with timeout — use explicit executor management so that
    # a timeout does not block in executor.__exit__(wait=True).
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(
            _execute_single_task,
            task=task,
            context=context,
            response_format=response_format,
            json_schema=json_schema,
            provider=resolved_provider,
            model=resolved_model,
            temperature=temperature,
            num_ctx=num_ctx,
            use_tools=use_tools,
        )

        try:
            result = future.result(timeout=timeout)
        except FuturesTimeoutError:
            future.cancel()
            _emit_status(f"✗ Delegation timed out after {timeout}s")
            return f"**Delegation timed out** after {timeout}s.\n\n" f"Task: {task[:100]}..."
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    # Format output
    if result.success:
        _emit_status(
            f"✓ {result.provider}/{result.model_used} responded in {result.duration_seconds}s"
        )
        output_parts = [
            f"**Delegated to:** `{result.provider}/{result.model_used}`",
            f"**Duration:** {result.duration_seconds}s",
        ]

        if response_format == "json":
            if result.format_valid:
                output_parts.append("**JSON Valid:** ✓")
            else:
                output_parts.append("**JSON Valid:** ✗ (response may not be valid JSON)")

        output_parts.append(f"\n**Response:**\n{result.response}")
        return "\n".join(output_parts)
    else:
        return f"**Delegation failed:** {result.error}"


def delegate_parallel(
    tasks: list[dict[str, Any]],
    timeout: int = 120,
) -> str:
    """
    Delegate multiple tasks in parallel to LLM models.

    Run several independent subtasks concurrently for efficiency.
    Each task can use a different provider/model.

    Args:
        tasks: List of task definitions, each with:
            - task (required): Task description
            - context: Context for the task
            - use_tools: Run as agent with tools (default True)
            - response_format: 'text', 'json', 'code', 'markdown'
            - json_schema: Expected JSON structure
            - provider: 'openai', 'ollama', or alias
            - model: Model name or alias
            - temperature: 0.0-2.0
        timeout: Max seconds for ALL tasks (30-600)

    Returns:
        Formatted string with all results
    """
    if not _delegate_config.get("enabled", True):
        return "**Delegation disabled.** Enable in configuration."

    if not tasks:
        return "**No tasks provided.**"

    # Validate context for LLM-only tasks and allowed_models up front
    for i, task_def in enumerate(tasks):
        error = _check_allowed_model(task_def.get("model"))
        if error:
            return f"**Delegation blocked (task {i + 1}):** {error}"
        task_use_tools = task_def.get("use_tools", True)
        task_context = task_def.get("context", "")
        if not task_use_tools and not task_context.strip():
            return (
                f"**Delegation rejected (task {i + 1}):** context is empty "
                f"and use_tools=False. Provide context or set use_tools=True."
            )

    # Clamp timeout
    timeout = max(30, min(600, timeout))

    # Build summary of delegation targets for the status message
    target_summaries = []
    for task_def in tasks:
        prov_peek, mdl_peek, _ = resolve_model_alias(
            task_def.get("provider"), task_def.get("model")
        )
        prov_peek, mdl_peek = resolve_delegate_defaults(prov_peek, mdl_peek)
        label = task_def.get("model") or mdl_peek
        target_summaries.append(f"{prov_peek}/{label}")
    _emit_status(f"→ Delegating {len(tasks)} tasks in parallel: {', '.join(target_summaries)}")

    start_time = time.time()
    results: list[tuple] = []  # (index, DelegateResult)

    def execute_indexed(index: int, task_def: dict[str, Any]) -> tuple:
        """Execute task and return with index for ordering."""
        prov, mdl, alias_cfg = resolve_model_alias(
            task_def.get("provider"),
            task_def.get("model"),
        )
        temp = task_def.get("temperature", 0.5)
        if "temperature" in alias_cfg and alias_cfg["temperature"] is not None:
            temp = alias_cfg["temperature"]
        if temp is None:
            temp = 0.5
        temp = max(0.0, min(2.0, temp))
        num_ctx = alias_cfg.get("context_window") or alias_cfg.get("num_ctx")

        prov, mdl = resolve_delegate_defaults(prov, mdl)

        result = _execute_single_task(
            task=task_def.get("task", ""),
            context=task_def.get("context", ""),
            response_format=task_def.get("response_format", "text"),
            json_schema=task_def.get("json_schema"),
            provider=prov,
            model=mdl,
            temperature=temp,
            num_ctx=num_ctx,
            use_tools=task_def.get("use_tools", True),
        )
        return (index, result)

    # Execute all tasks in parallel — use explicit executor management so that
    # a timeout does not block in executor.__exit__(wait=True).
    executor = ThreadPoolExecutor(max_workers=min(len(tasks), 10))
    try:
        futures = [
            executor.submit(execute_indexed, i, task_def) for i, task_def in enumerate(tasks)
        ]

        # Collect results with timeout.
        # Use enumerate to track the original task index so that
        # timeout entries are associated with the correct task.
        for i, future in enumerate(futures):
            try:
                remaining = timeout - (time.time() - start_time)
                if remaining <= 0:
                    future.cancel()
                    results.append(
                        (
                            i,
                            DelegateResult(
                                success=False,
                                response="",
                                format_valid=False,
                                parsed_json=None,
                                model_used="unknown",
                                provider="unknown",
                                duration_seconds=0,
                                error="Timeout exceeded",
                            ),
                        )
                    )
                else:
                    results.append(future.result(timeout=remaining))
            except FuturesTimeoutError:
                future.cancel()
                results.append(
                    (
                        i,
                        DelegateResult(
                            success=False,
                            response="",
                            format_valid=False,
                            parsed_json=None,
                            model_used="unknown",
                            provider="unknown",
                            duration_seconds=0,
                            error="Task timed out",
                        ),
                    )
                )
            except Exception as exc:
                results.append(
                    (
                        i,
                        DelegateResult(
                            success=False,
                            response="",
                            format_valid=False,
                            parsed_json=None,
                            model_used="unknown",
                            provider="unknown",
                            duration_seconds=0,
                            error=f"Task failed: {exc}",
                        ),
                    )
                )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    # Sort by original index
    results.sort(key=lambda x: x[0])

    total_duration = time.time() - start_time
    success_count = sum(1 for _, r in results if r.success)

    _emit_status(
        f"✓ Parallel delegation complete: {success_count}/{len(tasks)} "
        f"succeeded in {round(total_duration, 1)}s"
    )

    # Format output
    output_parts = [
        f"**Parallel Delegation:** {len(tasks)} tasks",
        f"**Completed:** {success_count}/{len(tasks)} successful",
        f"**Total Duration:** {round(total_duration, 2)}s",
        "",
    ]

    for i, (_idx, result) in enumerate(results):
        task_desc = tasks[i].get("task", "")[:50]
        output_parts.append(f"---\n### Task {i + 1}: {task_desc}...")

        if result.success:
            output_parts.append(
                f"**Model:** `{result.provider}/{result.model_used}` "
                f"({result.duration_seconds}s)"
            )
            if tasks[i].get("response_format") == "json":
                status = "✓" if result.format_valid else "✗"
                output_parts.append(f"**JSON Valid:** {status}")
            output_parts.append(f"\n{result.response}\n")
        else:
            if result.error and "timed out" in result.error.lower():
                output_parts.append(f"**⏱ Timed out** (>{timeout}s)\n")
            elif result.error and "timeout exceeded" in result.error.lower():
                output_parts.append("**⏱ Timed out** (total timeout exceeded)\n")
            else:
                output_parts.append(f"**✗ Error:** {result.error}\n")

    # Recovery guidance for partial or total failures
    failed_count = len(tasks) - success_count
    if failed_count > 0:
        timeout_failures = sum(
            1
            for _, r in results
            if not r.success
            and r.error
            and ("timed out" in r.error.lower() or "timeout exceeded" in r.error.lower())
        )
        other_failures = failed_count - timeout_failures

        guidance = ["---", "### Recovery Guidance"]
        if success_count == 0:
            guidance.append(f"**All {len(tasks)} tasks failed.** Suggested next steps:")
        else:
            guidance.append(f"**{failed_count} task(s) did not complete.** Suggested next steps:")

        if timeout_failures > 0:
            guidance.append(
                f"- **Timeout ({timeout_failures} task(s)):** Re-run with a higher "
                f"`timeout` (current: {timeout}s, max: 600s), or use `delegate_task` "
                f"to run each failed task individually with `timeout=300`."
            )
        if other_failures > 0:
            guidance.append(
                "- **Error failures:** Check that the specified provider/model is "
                "available. Try `delegate_task` individually to isolate the problem."
            )
        if success_count > 0 and failed_count > 0:
            guidance.append(
                "- **Partial results above are usable.** Incorporate the successful "
                "results and retry only the failed tasks."
            )

        output_parts.extend(guidance)

    return "\n".join(output_parts)


# Tool configuration for registry (use TOOL_CONFIGS for multiple tools)
TOOL_CONFIGS = [
    {
        "name": "delegate_task",
        "description": (
            "Delegate a task to another LLM model. By default the "
            "delegate runs as a full agent with tool access (file I/O, "
            "shell, search, etc.). Set use_tools=False for pure text "
            "analysis (faster, but requires non-empty context).\n"
            "\n"
            "USE THIS TOOL WHEN:\n"
            "- A subtask can run independently (code review, research, "
            "testing, data gathering)\n"
            "- You need a second opinion or specialised model\n"
            "- The user asks you to delegate or use a specific model\n"
            "- Multiple independent subtasks can run in parallel\n"
            "\n"
            "RULES:\n"
            "- NEVER use use_tools=False with empty context\n"
            "- Provide context when possible to avoid redundant work\n"
            "- Delegates cannot see your conversation history\n"
            "\n"
            "Use for: specialized work (code, research), second opinions, verification, "
            "text analysis (use_tools=False with context parameter)."
        ),
        "input_schema": DelegateInput,
        "function": delegate_task,
        "requires_confirmation": False,
    },
    {
        "name": "delegate_parallel",
        "description": (
            "Run multiple independent tasks in parallel across LLM "
            "models. Each task can target a different model and each "
            "delegate runs as a full agent with tool access by default.\n"
            "\n"
            "USE THIS TOOL WHEN:\n"
            "- You have multiple independent subtasks (e.g., review "
            "3 modules, research 5 topics, run tests in parallel)\n"
            "- You need to compare outputs from different models\n"
            "- Batch processing where tasks don't depend on each other\n"
            "\n"
            "Set use_tools=False per task for pure text analysis "
            "(requires non-empty context). Delegates cannot see "
            "your conversation history — pass relevant data in context.\n"
            "\n"
            "Use for: independent subtasks (fan out), large-scale processing, "
            "batch analysis, summarization of multiple sources."
        ),
        "input_schema": DelegateParallelInput,
        "function": delegate_parallel,
        "requires_confirmation": False,
    },
]

__all__ = [
    "delegate_task",
    "delegate_parallel",
    "configure_delegate",
    "set_delegate_tools",
    "set_status_callback",
    "get_model_status",
    "reset_model_status",
    "DelegateInput",
    "DelegateParallelInput",
    "DelegateResult",
    "ModelCircuitBreaker",
    # Public helpers (renamed from private)
    "create_delegate_llm",
    "get_delegate_tools",
    "resolve_delegate_defaults",
    "resolve_model_alias",
    "run_delegate_agent",
    # Other helpers
    "_build_prompt",
    "_check_allowed_model",
    "_circuit_breaker_lock",
    "_get_circuit_breaker",
    "_MAX_CIRCUIT_BREAKERS",
    "_validate_json_response",
    "_extract_content",
    "TOOL_CONFIGS",
]

# Backward-compatible aliases for the renamed functions
_create_llm = create_delegate_llm
_get_delegate_tools = get_delegate_tools
_resolve_defaults = resolve_delegate_defaults
_resolve_model_alias = resolve_model_alias
_run_delegate_agent = run_delegate_agent


def __getattr__(name: str) -> Any:
    if name == "_delegate_tools":
        return get_delegate_tools()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
