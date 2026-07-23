"""
Agent core setup using LangGraph ReAct Agent.
Decoupled from CLI; callable from any interface.
Supports multiple LLM providers: OpenAI, Ollama, and OpenAI-compatible APIs.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Any, TypedDict

from src.logging_config import get_logger

if TYPE_CHECKING:
    from src.config import ProviderConfig

# LangGraph agent creation (modern API)
try:
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import create_react_agent

    LANGGRAPH_AVAILABLE = True
except ImportError:
    add_messages = None  # type: ignore[misc, assignment]
    create_react_agent = None  # type: ignore[misc, assignment]
    LANGGRAPH_AVAILABLE = False

# LangChain message types (separate import to avoid cascade failure)
try:
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        SystemMessage,
    )
except ImportError:
    BaseMessage = None  # type: ignore[misc, assignment]
    HumanMessage = None  # type: ignore[misc, assignment]
    AIMessage = None  # type: ignore[misc, assignment]
    SystemMessage = None  # type: ignore[misc, assignment]


class CogtrixState(TypedDict):
    """State schema for the Cogtrix agent graph."""

    messages: Annotated[
        Sequence[BaseMessage], add_messages  # pyright: ignore[reportInvalidTypeForm]
    ]


DEFAULT_SYSTEM_PROMPT = """You are a capable AI assistant that COMPLETES TASKS end-to-end.

## Core Principles
- Fully execute every task step-by-step until complete. Never stop halfway
  or ask "what would you like me to do?" when the task is clear.
- Use tools proactively to gather information — never ask the user for data
  you can obtain yourself.
- After gathering data, synthesize, analyze, organize, and deliver one
  complete, polished response (never just raw lists or partial output).
- Stay focused on the current task; do not offer tangential help until done.
- For complex requests: break it down, execute each part with tools, combine
  results into one coherent deliverable.

## Accuracy and Grounding
- Base answers **strictly on data returned by tools**. Do NOT fill gaps with
  assumptions or prior knowledge — state what the tools found and explicitly
  note when information was not available.
- If sources do not contain the requested details, say so clearly (e.g.,
  "This information was not found in the sources I checked") rather than
  guessing.
- Clearly distinguish confirmed facts from inferences; use hedging language
  ("likely", "appears to be") for inferences.
- Cite URLs or source names from tool results when presenting factual claims.

## Forbidden Behaviors
- Never say "I'm ready to help!" or "What would you like me to do?" — do
  the work.
- Never stop after using tools — synthesize results into your answer.
- NEVER invent numbers, dates, parameter counts, version numbers, URLs, or
  any specifics not found in tool results.
- **NEVER** say "I need more steps" or "I ran out of steps" — deliver your
  best answer with whatever you have gathered. Partial real information is
  always better than an apology.

## Tools

You start with **one meta-tool**: `request_tools`.  It lists every tool
available in the catalog.  Before you can use any tool you must request it
first:

1. Read the catalog inside `request_tools` to see what is available.
2. Call `request_tools(add=["tool_a", "tool_b"])` to load what you need.
3. The requested tools become available on your **next** turn — do NOT
   attempt to call them in the same turn you requested them.
4. When you no longer need a tool, release it with
   `request_tools(remove=["tool_a"])` to keep your toolkit lean.

Request only the tools relevant to the current task.  Don't load tools
speculatively.

## Context Budget

You have a **limited context window**.  Every tool output consumes part of it.

**Be strategic:**
- Prefer `list_directory` first, then read only the files you need.
- Don't read entire large files — use `start_line` and `max_lines` to page
  (e.g. 200 lines at a time).
- If output shows "[truncated]", read only the needed section instead of
  re-reading everything.
- Delegate independent subtasks to free up your own context.

## Deep Reasoning

When the user asks for deep or thorough analysis, invoke the `deep_think`
tool.  It explores multiple solution paths in parallel using Tree-of-Thought
reasoning.  Use it for architecture decisions, strategy, complex debugging,
or multi-angle analysis.  You may combine it with prior tool calls.

CRITICAL: `deep_think` runs in ISOLATION — it cannot see conversation history
or previous tool results.  You MUST copy the FULL text of all relevant data
into the `context` parameter.  Do NOT pass references like "see above" —
pass the actual text, or the tool will hallucinate.

Do not use `deep_think` for simple factual or straightforward tasks.

## Task Delegation

Use `delegate_task` and `delegate_parallel` for:
- Independent subtasks (fan out with `delegate_parallel`)
- Specialized work (code, research, etc.)
- Second opinions or verification
- Large-scale processing, summarization, batch analysis
- Pure text analysis (`use_tools=False` — must provide `context`)

Rules:
- Delegates have the same tools you do (except delegation and deep_think).
- Delegates cannot see conversation history — include relevant findings
  in `context`.
- Never leave `context` empty when `use_tools=False`.
- In parallel calls, assign different model aliases to spread load.
- After receiving results, synthesize into one polished response — don't
  just list raw outputs.

Do not delegate simple questions or tasks requiring conversation memory.
"""

DEFAULT_TOOL_INSTRUCTIONS = (
    "When a tool is needed, output ONLY a valid tool call in the exact "
    "OpenAI format.\n"
    "NEVER add explanations, thoughts, markdown, or extra text outside "
    "the JSON.\n"
    "The arguments MUST be valid JSON \u2014 escape quotes, no trailing "
    "commas, correct types.\n\n"
    "Example of correct output when calling a tool:\n"
    '{"name": "get_weather", "arguments": '
    '"{\\"location\\": \\"Dubai\\", \\"unit\\": \\"celsius\\"}"}\n\n'
    "For final answers (no tool needed), just respond normally.\n\n"
    "Repeat: Output tool calls as pure JSON only \u2014 nothing else."
)


def build_agent_executor(
    tools: list,
    llm: Any = None,
    system_prompt: str | None = None,
    # Legacy parameters (deprecated - use llm parameter instead)
    provider: str = "openai",
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> Any:
    """
    Create a LangGraph ReAct agent with provided tools.

    Args:
        tools: List of LangChain tools
        llm: Pre-created LLM instance (preferred)
        system_prompt: Custom system prompt (default: generic assistant)
        provider: (deprecated) LLM provider - use llm parameter instead
        model: (deprecated) Model name - use llm parameter instead
        base_url: (deprecated) Server URL - use llm parameter instead
        api_key: (deprecated) API key - use llm parameter instead

    Returns:
        Compiled agent executor ready for use

    Raises:
        ImportError: If required packages are not installed
        ValueError: If provider is not supported
    """
    log = get_logger()

    if not LANGGRAPH_AVAILABLE or create_react_agent is None:
        raise ImportError("LangGraph not installed. Run: pip install langgraph")

    # Use pre-created LLM if provided, otherwise create one (legacy path)
    if llm is None:
        from src.providers import create_chat_model, get_default_model

        resolved_model = model or get_default_model(provider)
        log.debug("Creating LLM: provider=%s, model=%s", provider, resolved_model)

        llm = create_chat_model(
            provider,
            model=resolved_model,
            api_key=api_key,
            base_url=base_url,
        )
        log.debug("LLM created: %s", type(llm).__name__)

    # Use custom system prompt or default
    final_prompt = system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT

    # Create agent using LangGraph's create_react_agent
    # This creates a compiled graph that handles tool calling
    log.debug(f"Creating ReAct agent with {len(tools)} tools")
    agent_executor = create_react_agent(
        model=llm,
        tools=tools,
        prompt=final_prompt,
    )

    log.info(f"Agent initialized: {type(llm).__name__}")
    return agent_executor


def _format_alias_detail(value: Any) -> str:
    """Return a human-readable description of one alias value."""
    if isinstance(value, dict):
        provider = value.get("provider", "?")
        model = value.get("model", "?")
        extras: list[str] = []
        if "temperature" in value:
            extras.append(f"temp={value['temperature']}")
        if "timeout" in value:
            extras.append(f"timeout={value['timeout']}s")
        if "num_ctx" in value:
            extras.append(f"ctx={value['num_ctx']}")
        detail = f"{provider}/{model}"
        if extras:
            detail += f" ({', '.join(extras)})"
        return detail
    if isinstance(value, str):
        return value
    return str(value)


def _format_alias_table(
    model_aliases: dict[str, Any],
    delegation_models: list[str] | None = None,
) -> str:
    """Format model aliases into a readable table for the system prompt.

    If *delegation_models* is provided, the table is split into two
    groups: aliases that are allowed for delegation (highlighted) and
    the remaining aliases which are only available via ``/model``.

    Returns an empty string when there are no aliases to show.
    """
    if not model_aliases:
        return ""

    lines: list[str] = ["## Available Model Aliases", ""]

    if delegation_models:
        # Show delegation targets prominently
        lines.append("### Delegation targets (use with `delegate_task` / `delegate_parallel`):")
        lines.append("")
        for alias in delegation_models:
            if alias in model_aliases:
                detail = _format_alias_detail(model_aliases[alias])
                lines.append(f"- **{alias}** → `{detail}`")
        lines.append("")

        # Show remaining aliases
        others = {k: v for k, v in model_aliases.items() if k not in delegation_models}
        if others:
            lines.append("### Other aliases (available via `/model` command):")
            lines.append("")
            for alias, value in others.items():
                detail = _format_alias_detail(value)
                lines.append(f"- **{alias}** → `{detail}`")
            lines.append("")
    else:
        lines.append(
            "Use these names as the `model` parameter in `delegate_task` / `delegate_parallel`:"
        )
        lines.append("")
        for alias, value in model_aliases.items():
            detail = _format_alias_detail(value)
            lines.append(f"- **{alias}** → `{detail}`")
        lines.append("")

    return "\n".join(lines)


def build_system_prompt(
    base_prompt: str | None = None,
    mode_additions: str | None = None,
    model_aliases: dict[str, Any] | None = None,
    delegation_models: list[str] | None = None,
    tool_instructions: str | None = None,
) -> str:
    """
    Build a complete system prompt with mode-specific additions.

    Args:
        base_prompt: Base system prompt (uses default if None)
        mode_additions: Additional instructions from memory mode
        model_aliases: User-defined model aliases to expose to the agent
        delegation_models: Subset of alias names allowed for delegation
            (``None`` means all aliases are allowed)
        tool_instructions: Optional tool-call formatting instructions to
            append.  Defaults to ``None`` (no instructions), since
            LangGraph ``bind_tools()`` handles tool-call formatting at the
            API level for all supported providers.  Set to a non-empty
            string to inject custom instructions for providers that need
            explicit guidance.

    Returns:
        Combined system prompt
    """
    base = base_prompt if base_prompt else DEFAULT_SYSTEM_PROMPT

    parts = [base]

    # Inject alias table so the agent knows what models are available
    alias_section = _format_alias_table(
        model_aliases or {},
        delegation_models=delegation_models,
    )
    if alias_section:
        parts.append(alias_section)

    if mode_additions:
        parts.append(mode_additions)

    # Inject tool-call formatting instructions only when explicitly provided.
    # bind_tools() handles formatting at the API level, so injecting raw-JSON
    # examples into the system prompt can conflict with the structured
    # tool_calls response format and cause parsing failures (especially on
    # vLLM's openai_tool_parser).
    if tool_instructions:
        parts.append(tool_instructions)

    return "\n\n".join(parts)


_DEFAULT_CONTEXT_WINDOW = 32_768
_MIN_RESPONSE_TOKENS = 1024
_RESPONSE_RESERVE_RATIO = 0.25
_MAX_SINGLE_MESSAGE_TOKENS = 6_000


def _estimate_msg_tokens(msg: Any) -> int:
    """Estimate token count for a single message (~4 chars per token)."""
    if hasattr(msg, "content") and msg.content:
        content = msg.content
    elif isinstance(msg, dict) and msg.get("content"):
        content = msg["content"]
    else:
        return 10  # overhead for empty/structural messages
    if isinstance(content, list):
        total = sum(len(str(p)) for p in content)
    else:
        total = len(str(content))
    return max(total // 4, 1)


def _truncate_content(content: str, max_tokens: int) -> str:
    """Truncate a string to roughly *max_tokens* (at ~4 chars/token)."""
    max_chars = max_tokens * 4
    if len(content) <= max_chars:
        return content
    half = max_chars // 2
    return (
        content[:half] + f"\n\n[... truncated: {len(content) - max_chars} chars removed "
        f"to fit context window ...]\n\n" + content[-half:]
    )


def _trim_to_token_budget(
    messages: list[Any],
    max_context_tokens: int,
) -> list[Any]:
    """Drop the oldest history messages so the total fits the budget.

    **Never removed:** the first message (if it's a SystemMessage) and the
    last message (current user input).  Everything in between (history) is
    trimmed oldest-first.  Oversized individual messages are truncated as a
    last resort.
    """
    _log = get_logger()

    response_reserve = max(
        _MIN_RESPONSE_TOKENS,
        int(max_context_tokens * _RESPONSE_RESERVE_RATIO),
    )
    budget = max_context_tokens - response_reserve

    if budget <= 0:
        _log.warning(
            "Context window (%d tokens) too small for response reserve (%d)",
            max_context_tokens,
            response_reserve,
        )
        budget = max_context_tokens // 2

    # Separate fixed parts from trimmable history
    fixed_head: list[Any] = []
    fixed_tail: list[Any] = []
    history: list[Any] = list(messages)

    if history and SystemMessage is not None and isinstance(history[0], SystemMessage):
        fixed_head.append(history.pop(0))

    if history:
        fixed_tail.append(history.pop(-1))

    # Truncate oversized fixed parts (system prompt, user input) to ensure
    # they don't individually consume the entire budget.
    max_fixed_single = budget // 2
    for bucket in (fixed_head, fixed_tail):
        for idx, msg in enumerate(bucket):
            est = _estimate_msg_tokens(msg)
            if est > max_fixed_single and max_fixed_single > 0:
                content = ""
                if hasattr(msg, "content") and isinstance(msg.content, str):
                    content = msg.content
                elif isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    content = msg["content"]
                if content:
                    trimmed = _truncate_content(content, max_fixed_single)
                    if isinstance(msg, dict):
                        bucket[idx] = {**msg, "content": trimmed}
                    elif hasattr(msg, "copy"):
                        bucket[idx] = msg.copy(update={"content": trimmed})

    fixed_cost = sum(_estimate_msg_tokens(m) for m in fixed_head + fixed_tail)

    history_budget = budget - fixed_cost
    if history_budget <= 0:
        _log.warning(
            "System prompt + user input alone (~%d tokens) nearly fills "
            "the context window (%d tokens) — sending without history",
            fixed_cost,
            max_context_tokens,
        )
        return fixed_head + fixed_tail

    # Truncate individual oversized messages in history.
    # Create copies to avoid mutating the caller's original messages.
    for idx, msg in enumerate(history):
        est = _estimate_msg_tokens(msg)
        if est > _MAX_SINGLE_MESSAGE_TOKENS:
            content = ""
            if hasattr(msg, "content") and isinstance(msg.content, str):
                content = msg.content
            elif isinstance(msg, dict) and isinstance(msg.get("content"), str):
                content = msg["content"]
            if content:
                trimmed = _truncate_content(content, _MAX_SINGLE_MESSAGE_TOKENS)
                if isinstance(msg, dict):
                    history[idx] = {**msg, "content": trimmed}
                elif hasattr(msg, "copy"):
                    clone = msg.copy(update={"content": trimmed})
                    history[idx] = clone
                else:
                    try:
                        from copy import copy as _shallow_copy

                        clone = _shallow_copy(msg)
                        clone.content = trimmed
                        history[idx] = clone
                    except Exception:
                        pass

    # Drop oldest history messages until we fit
    total_history = sum(_estimate_msg_tokens(m) for m in history)
    dropped = 0
    while history and total_history > history_budget:
        removed = history.pop(0)
        total_history -= _estimate_msg_tokens(removed)
        dropped += 1

    # Remove orphaned ToolMessages at the head of the trimmed history.
    # A ToolMessage is orphaned if no preceding AIMessage carries its tool_call_id.
    try:
        from langchain_core.messages import ToolMessage as _ToolMessage
    except ImportError:
        _ToolMessage = None  # type: ignore[assignment, misc]

    if _ToolMessage is not None and AIMessage is not None:
        while history and isinstance(history[0], _ToolMessage):
            removed = history.pop(0)
            total_history -= _estimate_msg_tokens(removed)
            dropped += 1

    if dropped:
        _log.info(
            "Context trimmed: dropped %d oldest messages to fit "
            "token budget (%d / %d used, %d reserved for response)",
            dropped,
            fixed_cost + total_history,
            max_context_tokens,
            response_reserve,
        )

    return fixed_head + history + fixed_tail


def prepare_messages_with_context(
    history_messages: list[Any],
    user_input: str,
    context_prefix: str | None = None,
    max_context_tokens: int | None = None,
) -> list[Any]:
    """
    Prepare messages for the agent with optional context prefix.

    The context prefix (containing mode-specific state like current task,
    tracked files, reasoning state, etc.) is injected as a SystemMessage
    before the conversation history.

    When *max_context_tokens* is provided, the assembled message list is
    trimmed to fit the token budget.  Oldest history messages are dropped
    first; oversized individual messages are truncated as a last resort.

    Args:
        history_messages: Conversation history from memory manager
        user_input: Current user input
        context_prefix: Mode-specific context to inject
        max_context_tokens: Token budget for the full prompt (from
            ``ProviderConfig.num_ctx`` or a sensible default).  When set,
            the function trims history to leave room for the LLM response.

    Returns:
        List of messages ready for agent invocation
    """
    if HumanMessage is None:
        # LangChain not available, return basic structure
        fallback_msgs = list(history_messages)
        fallback_msgs.append({"type": "human", "content": user_input})
        return fallback_msgs

    result: list[Any] = []

    # Inject context prefix as a SystemMessage if present
    if context_prefix and SystemMessage is not None:
        ctx_content = f"Current context:\n{context_prefix}"
        result.append(SystemMessage(content=ctx_content))

    # Add conversation history
    result.extend(history_messages)

    # Add current user input
    result.append(HumanMessage(content=user_input))

    # Trim to budget if a limit is set
    if max_context_tokens and max_context_tokens > 0:
        result = _trim_to_token_budget(result, max_context_tokens)

    return result


def create_llm_from_provider_config(provider_config: "ProviderConfig") -> Any:
    """
    Create an LLM instance from a ProviderConfig.

    Delegates to the centralized ``src.providers`` registry which
    supports openai, ollama, anthropic, google, and OpenAI-compatible
    providers (xAI, vLLM, Groq, etc.).

    Args:
        provider_config: Provider configuration object

    Returns:
        LLM instance ready for use

    Raises:
        ImportError: If required packages are not installed
        ValueError: If provider type is not supported
    """
    log = get_logger()

    from src.providers import create_chat_model_from_config

    log.debug(
        "Creating LLM from config: name=%s, type=%s, model=%s",
        provider_config.name,
        provider_config.type,
        provider_config.get_model(),
    )

    llm = create_chat_model_from_config(provider_config)

    log.debug("LLM created: %s", type(llm).__name__)
    return llm


def ensure_base_messages(history: list[Any]) -> list[Any]:
    """
    Convert stored history (dicts or BaseMessages) into BaseMessages.

    Preserves tool_calls on AIMessages and correctly handles ToolMessages
    so that the full agent chain survives a dict→BaseMessage round-trip.

    Args:
        history: List of message dicts or BaseMessage objects

    Returns:
        List of BaseMessage objects (or original if LangChain unavailable)
    """
    if BaseMessage is None or HumanMessage is None:
        return history

    try:
        from langchain_core.messages import ToolMessage
    except ImportError:
        ToolMessage = None  # type: ignore[assignment, misc]

    converted = []
    for msg in history:
        if isinstance(msg, BaseMessage):
            converted.append(msg)
            continue
        if isinstance(msg, dict) and "content" in msg:
            role = msg.get("type", "human")
            additional: dict[str, Any] = {}
            ts = msg.get("timestamp")
            if ts:
                additional["_ts"] = ts

            if role == "tool" and ToolMessage is not None:
                converted.append(
                    ToolMessage(
                        content=msg["content"],
                        name=msg.get("name", ""),
                        tool_call_id=msg.get("tool_call_id", ""),
                    )
                )
            elif role == "ai" and AIMessage is not None:
                kwargs: dict[str, Any] = {
                    "content": msg["content"],
                }
                if additional:
                    kwargs["additional_kwargs"] = additional
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    kwargs["tool_calls"] = tool_calls
                converted.append(AIMessage(**kwargs))
            elif role == "system" and SystemMessage is not None:
                converted.append(SystemMessage(content=msg["content"]))
            else:
                kwargs_h: dict[str, Any] = {"content": msg["content"]}
                if additional:
                    kwargs_h["additional_kwargs"] = additional
                converted.append(HumanMessage(**kwargs_h))
    return converted
