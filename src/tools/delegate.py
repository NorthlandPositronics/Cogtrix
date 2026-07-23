"""
Delegate Tool: Delegate tasks to other LLM models.

Enables the primary agent to offload subtasks to other LLM models,
supporting parallel execution, structured JSON responses, and timeouts.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from src.logging_config import get_logger, log_delegation

# LangChain imports with graceful fallback
try:
    from langchain_openai import ChatOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    ChatOpenAI = None  # type: ignore[misc, assignment]
    OPENAI_AVAILABLE = False

try:
    from langchain_ollama import ChatOllama

    OLLAMA_AVAILABLE = True
except ImportError:
    ChatOllama = None  # type: ignore[misc, assignment]
    OLLAMA_AVAILABLE = False

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
    "max_depth": 3,
    "default_timeout": 60,
    "default_provider": "ollama",
    "default_model": None,
    "allowed_providers": ["openai", "ollama"],
    "model_aliases": {},
    "providers": {},  # Named provider configurations
    # Legacy fallbacks
    "openai_api_key": None,
    "ollama_base_url": "http://localhost:11434",
    "max_consecutive_failures": 5,
    "circuit_breaker_cooldown": 300,  # seconds before retry
}


@dataclass
class ModelCircuitBreaker:
    """Track model availability status for circuit breaker pattern."""

    consecutive_failures: int = 0
    is_unavailable: bool = False
    last_failure_time: float = 0.0
    last_error: str = ""

    def record_failure(self, error: str, max_failures: int = 5) -> None:
        """Record a failure and potentially mark model as unavailable."""
        self.consecutive_failures += 1
        self.last_failure_time = time.time()
        self.last_error = error
        if self.consecutive_failures >= max_failures:
            self.is_unavailable = True

    def record_success(self) -> None:
        """Record a success and reset failure count."""
        self.consecutive_failures = 0
        self.is_unavailable = False
        self.last_error = ""

    def check_availability(self, cooldown: float = 300.0) -> tuple:
        """
        Check if model is available.

        Returns:
            Tuple of (is_available, reason_if_unavailable)
        """
        if not self.is_unavailable:
            return True, None

        # Check if cooldown period has passed
        elapsed = time.time() - self.last_failure_time
        if elapsed >= cooldown:
            # Reset and allow retry
            self.is_unavailable = False
            self.consecutive_failures = 0
            return True, None

        remaining = int(cooldown - elapsed)
        return False, (
            f"Model marked unavailable after {self.consecutive_failures} "
            f"consecutive failures. Last error: {self.last_error}. "
            f"Will retry in {remaining}s."
        )


# Circuit breaker registry: tracks failures per "provider/model" key
_circuit_breakers: dict[str, ModelCircuitBreaker] = {}


def _get_circuit_breaker(provider: str, model: str) -> ModelCircuitBreaker:
    """Get or create circuit breaker for a provider/model combination."""
    key = f"{provider}/{model}"
    if key not in _circuit_breakers:
        _circuit_breakers[key] = ModelCircuitBreaker()
    return _circuit_breakers[key]


def get_model_status() -> dict[str, Any]:
    """
    Get status of all tracked models.

    Returns:
        Dictionary with model availability status.
    """
    status = {}
    cooldown = _delegate_config.get("circuit_breaker_cooldown", 300)

    for key, breaker in _circuit_breakers.items():
        available, reason = breaker.check_availability(cooldown)
        status[key] = {
            "available": available,
            "consecutive_failures": breaker.consecutive_failures,
            "last_error": breaker.last_error if not available else None,
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
    _delegate_config.update(config)


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
        description="Relevant context/data for the task (code, text, documents, etc.)",
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
        description="LLM provider: 'openai', 'ollama', or alias from config",
    )
    model: str | None = Field(
        default=None,
        description="Model name (e.g., 'gpt-4', 'llama3:8b') or alias from config",
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
            "'task' (required), 'context', 'response_format', 'json_schema', "
            "'provider', 'model', 'temperature'"
        )
    )
    timeout: int = Field(
        default=120,
        description="Maximum seconds to wait for ALL tasks to complete (30-600)",
    )


def _resolve_model_alias(provider: str | None, model: str | None) -> tuple:
    """
    Resolve model aliases from configuration.

    Supports two alias formats:
    1. String: "provider/model" or just "model"
    2. Object: {"provider": "...", "model": "...", "timeout": 300, "temperature": 0.5, "num_ctx": 32768}

    Returns:
        Tuple of (resolved_provider, resolved_model, alias_config)
        where alias_config is a dict with optional 'timeout', 'temperature', 'num_ctx' overrides
    """
    aliases = _delegate_config.get("model_aliases", {})
    alias_config: dict[str, Any] = {}

    def _extract_alias_config(alias_value: dict) -> None:
        """Extract config values from alias dict into alias_config."""
        if "timeout" in alias_value:
            alias_config["timeout"] = alias_value["timeout"]
        if "temperature" in alias_value:
            alias_config["temperature"] = alias_value["temperature"]
        if "num_ctx" in alias_value:
            alias_config["num_ctx"] = alias_value["num_ctx"]

    # Check if model is an alias
    if model and model in aliases:
        alias_value = aliases[model]

        # Object format: {"provider": "...", "model": "...", "timeout": 300, "num_ctx": 32768}
        if isinstance(alias_value, dict):
            resolved_provider = alias_value.get("provider", provider)
            resolved_model = alias_value.get("model")
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


def _create_llm(
    provider: str,
    model: str | None = None,
    temperature: float = 0.7,
    num_ctx: int | None = None,
) -> Any:
    """
    Create an LLM instance for the specified provider.

    Supports both named providers from config and legacy providers.

    Args:
        provider: Provider name (e.g., 'openai', 'ollama', 'my-server')
        model: Model name (overrides provider config default)
        temperature: Sampling temperature
        num_ctx: Context window size (Ollama only)

    Returns:
        LLM instance

    Raises:
        ValueError: If provider is not supported or not available
    """
    allowed = _delegate_config.get("allowed_providers", ["openai", "ollama"])
    if provider not in allowed:
        raise ValueError(f"Provider '{provider}' not allowed. Allowed: {allowed}")

    providers = _delegate_config.get("providers", {})

    # Check for named provider in config
    if provider in providers:
        prov_cfg = providers[provider]
        prov_type = prov_cfg.get("type", provider)
        base_url = prov_cfg.get("base_url")
        api_key = prov_cfg.get("api_key")
        default_model = prov_cfg.get("model")
        final_model = model or default_model
        # Use alias num_ctx if provided, else fall back to provider config
        final_num_ctx = num_ctx if num_ctx is not None else prov_cfg.get("num_ctx")

        if prov_type == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI not available. Install: pip install langchain-openai")
            kwargs: dict[str, Any] = {
                "model": final_model or "gpt-4o-mini",
                "temperature": temperature,
                "max_retries": 3,
            }
            if api_key:
                kwargs["api_key"] = api_key
            if base_url:
                kwargs["base_url"] = base_url
            return ChatOpenAI(**kwargs)

        elif prov_type == "ollama":
            if not OLLAMA_AVAILABLE:
                raise ImportError("Ollama not available. Install: pip install langchain-ollama")
            ollama_kwargs: dict[str, Any] = {
                "model": final_model or "llama3:8b",
                "base_url": base_url or "http://localhost:11434",
                "temperature": temperature,
            }
            if final_num_ctx is not None:
                ollama_kwargs["num_ctx"] = final_num_ctx
            return ChatOllama(**ollama_kwargs)

        else:
            raise ValueError(
                f"Unknown provider type '{prov_type}' for '{provider}'. "
                "Use 'openai' or 'ollama'."
            )

    # Legacy fallback for built-in provider names
    if provider == "openai":
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI not available. Install: pip install langchain-openai")
        api_key = _delegate_config.get("openai_api_key")
        return ChatOpenAI(
            model=model or "gpt-4o-mini",
            temperature=temperature,
            api_key=api_key,  # type: ignore[arg-type]
            max_retries=3,
        )

    elif provider == "ollama":
        if not OLLAMA_AVAILABLE:
            raise ImportError("Ollama not available. Install: pip install langchain-ollama")
        base_url = _delegate_config.get("ollama_base_url", "http://localhost:11434")
        return ChatOllama(
            model=model or "llama3:8b",
            base_url=base_url,
            temperature=temperature,
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}")


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

    # Remove markdown code blocks if present
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        parsed = json.loads(text)
        return True, parsed
    except json.JSONDecodeError:
        return False, None


def _execute_single_task(
    task: str,
    context: str = "",
    response_format: str = "text",
    json_schema: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.7,
    num_ctx: int | None = None,
) -> DelegateResult:
    """
    Execute a single delegated task.

    Includes circuit breaker logic to prevent repeated calls to failing models.

    Returns:
        DelegateResult with response and metadata
    """
    log = get_logger()
    start_time = time.time()

    # Resolve aliases and defaults
    provider, model, alias_config = _resolve_model_alias(provider, model)
    provider = provider or _delegate_config.get("default_provider", "ollama")
    model = model or _delegate_config.get("default_model") or "default"

    # Apply alias config overrides (alias takes precedence over parameter)
    if "temperature" in alias_config:
        temperature = alias_config["temperature"]
    if "num_ctx" in alias_config:
        num_ctx = alias_config["num_ctx"]

    target_model = f"{provider}/{model}"
    log.debug(f"Delegation starting: {target_model}")
    log.debug(f"Task: {task[:100]}{'...' if len(task) > 100 else ''}")

    # Check circuit breaker status
    circuit_breaker = _get_circuit_breaker(provider, model)
    cooldown = _delegate_config.get("circuit_breaker_cooldown", 300)
    is_available, unavailable_reason = circuit_breaker.check_availability(cooldown)

    if not is_available:
        log.info(f"Delegation blocked: {target_model} - circuit breaker open")
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
        llm = _create_llm(provider, model, temperature, num_ctx)

        # Build prompt
        messages = _build_prompt(task, context, response_format, json_schema)

        # Invoke LLM
        log.debug(f"Invoking delegate LLM: {target_model}")
        result = llm.invoke(messages)

        # Extract response content
        if hasattr(result, "content"):
            content = result.content
            if isinstance(content, list):
                # Multimodal message — extract text parts
                text_parts = []
                for part in content:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])
                response_text = "\n".join(text_parts) if text_parts else str(content)
            else:
                response_text = str(content) if content is not None else ""
        else:
            response_text = str(result)

        # Validate JSON if requested
        format_valid = True
        parsed_json = None
        if response_format == "json":
            format_valid, parsed_json = _validate_json_response(response_text)

        duration = time.time() - start_time
        model_name = model or (llm.model if hasattr(llm, "model") else "unknown")

        # Record success - reset circuit breaker
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
        circuit_breaker.record_failure(error_msg, max_failures)

        # Add circuit breaker status to error if model is now unavailable
        if circuit_breaker.is_unavailable:
            error_msg = (
                f"{error_msg} | Model marked unavailable after "
                f"{circuit_breaker.consecutive_failures} consecutive failures."
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
    response_format: str = "text",
    json_schema: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    timeout: int = 60,
    temperature: float = 0.7,
) -> str:
    """
    Delegate a task to another LLM model.

    This tool allows the agent to offload subtasks to other models,
    enabling specialization, cost optimization, and parallel processing.

    Args:
        task: Clear description of what the delegated model should do
        context: Relevant context/data for the task
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

    # Resolve aliases first to get any timeout/temperature overrides
    resolved_provider, resolved_model, alias_config = _resolve_model_alias(provider, model)

    # Apply alias overrides (alias config takes precedence)
    if "timeout" in alias_config:
        timeout = alias_config["timeout"]
    if "temperature" in alias_config:
        temperature = alias_config["temperature"]
    num_ctx = alias_config.get("num_ctx")

    # Clamp timeout
    timeout = max(10, min(600, timeout))  # Allow up to 600s for reasoning models

    # Execute with timeout
    with ThreadPoolExecutor(max_workers=1) as executor:
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
        )

        try:
            result = future.result(timeout=timeout)
        except FuturesTimeoutError:
            return f"**Delegation timed out** after {timeout}s.\n\n" f"Task: {task[:100]}..."

    # Format output
    if result.success:
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

    # Clamp timeout
    timeout = max(30, min(600, timeout))

    start_time = time.time()
    results: list[tuple] = []  # (index, DelegateResult)

    def execute_indexed(index: int, task_def: dict[str, Any]) -> tuple:
        """Execute task and return with index for ordering."""
        result = _execute_single_task(
            task=task_def.get("task", ""),
            context=task_def.get("context", ""),
            response_format=task_def.get("response_format", "text"),
            json_schema=task_def.get("json_schema"),
            provider=task_def.get("provider"),
            model=task_def.get("model"),
            temperature=task_def.get("temperature", 0.7),
        )
        return (index, result)

    # Execute all tasks in parallel
    with ThreadPoolExecutor(max_workers=min(len(tasks), 10)) as executor:
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

    # Sort by original index
    results.sort(key=lambda x: x[0])

    total_duration = time.time() - start_time
    success_count = sum(1 for _, r in results if r.success)

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
            output_parts.append(f"**Failed:** {result.error}\n")

    return "\n".join(output_parts)


# Tool configuration for registry (use TOOL_CONFIGS for multiple tools)
TOOL_CONFIGS = [
    {
        "name": "delegate_task",
        "description": (
            "Delegate a task to another LLM model. Offload subtasks to "
            "specialized or cost-effective models while you orchestrate "
            "the overall workflow.\n"
            "\n"
            "USE THIS TOOL WHEN:\n"
            "- A subtask would benefit from a specialized model (e.g., "
            "code review → code model, translation → multilingual model)\n"
            "- You need a second opinion or independent verification of "
            "your own analysis\n"
            "- The task involves processing large text (summarization, "
            "extraction, rewriting) that can be offloaded\n"
            "- The user mentions model aliases ('use the fast model', "
            "'ask the code model') or asks you to delegate\n"
            "- You are orchestrating a multi-step workflow and individual "
            "steps can be handled by other models\n"
            "- A cheaper/faster model can handle routine work while you "
            "focus on synthesis and reasoning\n"
            "\n"
            "DO NOT delegate when you can answer directly with equal "
            "quality, or when the task requires your conversation context."
        ),
        "input_schema": DelegateInput,
        "function": delegate_task,
        "requires_confirmation": False,
    },
    {
        "name": "delegate_parallel",
        "description": (
            "Run multiple independent tasks in parallel across LLM models. "
            "Each task can target a different model.\n"
            "\n"
            "USE THIS TOOL WHEN:\n"
            "- You have multiple independent subtasks that don't depend "
            "on each other (e.g., summarize 5 documents, review 3 files)\n"
            "- You need to compare outputs from different models on the "
            "same task\n"
            "- Batch processing: extract data from several sources, "
            "translate multiple texts, analyze multiple items\n"
            "- Speed matters and tasks can run concurrently\n"
            "\n"
            "DO NOT use for sequential tasks where each step depends on "
            "the previous result — use delegate_task in sequence instead."
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
    "get_model_status",
    "reset_model_status",
    "DelegateInput",
    "DelegateParallelInput",
    "DelegateResult",
    "ModelCircuitBreaker",
    "TOOL_CONFIGS",
]
