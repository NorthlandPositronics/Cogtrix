"""
Agent core setup using LangGraph ReAct Agent.
Decoupled from CLI; callable from any interface.
Supports multiple LLM providers: OpenAI, Ollama, and OpenAI-compatible APIs.
"""

from typing import TYPE_CHECKING, Any

from src.logging_config import get_logger

if TYPE_CHECKING:
    from src.config import ProviderConfig

# LangGraph agent creation (modern API)
try:
    from langgraph.prebuilt import create_react_agent

    LANGGRAPH_AVAILABLE = True
except ImportError:
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

# OpenAI provider
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None  # type: ignore[misc, assignment]

# Ollama provider
try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None  # type: ignore[misc, assignment]


DEFAULT_SYSTEM_PROMPT = """You are a capable AI assistant that COMPLETES TASKS end-to-end.

## Core Principles

1. **COMPLETE TASKS FULLY** - When given a task, work through it step-by-step until finished. Never stop halfway to ask "what would you like me to do?" when the task is clear.

2. **USE TOOLS PROACTIVELY** - If you need information, use tools to get it. Don't ask the user for information you can gather yourself.

3. **SYNTHESIZE AND DELIVER** - After using tools to gather information, synthesize your findings into a complete, useful response. Don't just list what you found - analyze, organize, and present it meaningfully.

4. **STAY FOCUSED** - Keep working on the current task. Don't offer tangential help or ask what the user wants to do next until you've completed what they asked for.

5. **HANDLE MULTI-STEP TASKS** - For complex requests:
   - Break down the task mentally
   - Execute each step using appropriate tools
   - Combine results into a coherent deliverable
   - Present the final result

## Accuracy and Grounding

When answering questions that require factual information:
- Base your answer **strictly on data returned by tools** (search results, web
  pages, file contents, etc.).  Do NOT fill gaps with assumptions or prior
  knowledge — state what the tools found and explicitly note when information
  was not available.
- If search results or web pages do not contain the requested details, say so
  clearly (e.g., "This information was not found in the sources I checked")
  rather than guessing.
- Clearly distinguish between what the tools confirmed vs. what you are
  inferring.  Use hedging language ("likely", "appears to be") for inferences.
- Cite URLs or source names from tool results when presenting factual claims.

## What NOT To Do

- DON'T say "I'm ready to help!" when you should be doing the work
- DON'T ask "What would you like me to do?" when the task is already specified
- DON'T provide tangential information instead of completing the task
- DON'T stop after using tools - synthesize the results into your answer
- DON'T give up on complex tasks - break them into manageable steps
- DON'T make up facts, URLs, parameter counts, version numbers, or other
  specific details that were not in the tool results
- **NEVER** say "I need more steps", "I ran out of steps", or any variation. If you have been working through many tool calls, STOP calling tools and deliver your answer with whatever you have gathered so far. A partial answer with real information is always better than an apology with no content.

## Example Behavior

If asked to "analyze this codebase":
1. List directories and files using tools
2. Read key files to understand structure
3. Identify patterns, architecture, dependencies
4. Synthesize findings into a structured analysis
5. Present complete analysis to user

NOT: Read one file and say "Let me know what you'd like to explore!"

## Deep Reasoning

IMPORTANT: When the user includes phrases like "think deep", "think deeply",
"deep think", "analyze thoroughly", "think step by step", "consider all angles",
or "explore multiple approaches", you MUST invoke the `deep_think` tool.
These phrases are explicit requests for the Tree-of-Thought reasoning engine,
not general instructions to be more careful.

The `deep_think` tool explores several solution paths in parallel, evaluates
each with structured reflection, and synthesizes the best elements into an
improved solution through iterative cycles.  Use it for architecture decisions,
strategy planning, complex debugging, or multi-angle analysis.

You may combine `deep_think` with other tools: for example, first gather
information via `search_web` or `http_get`, then call `deep_think` to reason
deeply about the collected data.  Do NOT skip the `deep_think` call when the
user explicitly asks for deep or thorough thinking.

CRITICAL: `deep_think` runs in ISOLATION — it cannot see your conversation
history or previous tool results.  When calling it, you MUST copy the FULL
text of all relevant data (search results, web page content, etc.) into the
`context` parameter.  Do NOT pass references like "see search result 1" or
"the data above" — pass the actual text, or the tool will hallucinate.

Do NOT use `deep_think` for simple factual questions or straightforward tasks.

## Task Delegation

You can delegate subtasks to other LLM models using `delegate_task` or `delegate_parallel`. Use delegation when a subtask would benefit from a specialized model (code review, translation, summarization), when you need a second opinion, when processing multiple independent items in parallel, or when a cheaper/faster model can handle routine work while you focus on orchestration and synthesis. Model aliases (like "fast", "code", "smart") are configured by the user — use them when the task matches their purpose. Do NOT delegate when you can answer directly with equal quality or when the task needs your conversation context.
"""


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
        # Resolve model name with provider-specific defaults
        if provider == "ollama":
            resolved_model = model or "qwen3:32b"
        else:
            resolved_model = model or "gpt-4o-mini"

        log.debug(f"Creating LLM: provider={provider}, model={resolved_model}")

        # Create LLM based on provider
        if provider == "ollama":
            llm = _create_ollama_llm(resolved_model, base_url)
            log.debug(f"Ollama LLM created: base_url={base_url or 'localhost:11434'}")
        elif provider == "openai":
            llm = _create_openai_llm(resolved_model, api_key, base_url)
            log.debug("OpenAI LLM created")
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'ollama'.")

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


def build_system_prompt(
    base_prompt: str | None = None,
    mode_additions: str | None = None,
) -> str:
    """
    Build a complete system prompt with mode-specific additions.

    Args:
        base_prompt: Base system prompt (uses default if None)
        mode_additions: Additional instructions from memory mode

    Returns:
        Combined system prompt
    """
    base = base_prompt if base_prompt else DEFAULT_SYSTEM_PROMPT

    if mode_additions:
        return f"{base}\n\n{mode_additions}"

    return base


def prepare_messages_with_context(
    history_messages: list[Any],
    user_input: str,
    context_prefix: str | None = None,
) -> list[Any]:
    """
    Prepare messages for the agent with optional context prefix.

    The context prefix (containing mode-specific state like current task,
    tracked files, reasoning state, etc.) is injected as a SystemMessage
    before the conversation history.

    Args:
        history_messages: Conversation history from memory manager
        user_input: Current user input
        context_prefix: Mode-specific context to inject

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

    return result


def _create_openai_llm(
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0,
):
    """
    Create OpenAI or OpenAI-compatible chat model.

    Args:
        model: Model name
        api_key: API key (None = use OPENAI_API_KEY env var)
        base_url: Custom API endpoint (None = use OpenAI default)
        temperature: Sampling temperature
    """
    if ChatOpenAI is None:
        raise ImportError("langchain-openai not installed. Run: pip install langchain-openai")

    kwargs: dict = {
        "model": model or "gpt-4o-mini",
        "temperature": temperature,
        # Retry on transient errors (malformed JSON from vLLM tool
        # parser, 5xx, etc.).  LangChain's ChatOpenAI passes this
        # through to the underlying HTTP client.
        "max_retries": 3,
    }

    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url

    return ChatOpenAI(**kwargs)


def _create_ollama_llm(
    model: str | None = None,
    base_url: str | None = None,
    temperature: float = 0,
    num_ctx: int | None = None,
):
    """Create Ollama chat model."""
    if ChatOllama is None:
        raise ImportError("langchain-ollama not installed. Run: pip install langchain-ollama")

    kwargs: dict[str, Any] = {
        "model": model or "qwen3:32b",
        "base_url": base_url or "http://localhost:11434",
        "temperature": temperature,
    }

    # Set context window size if specified
    if num_ctx is not None:
        kwargs["num_ctx"] = num_ctx

    return ChatOllama(**kwargs)  # type: ignore[arg-type]


def create_llm_from_provider_config(provider_config: "ProviderConfig") -> Any:
    """
    Create an LLM instance from a ProviderConfig.

    Args:
        provider_config: Provider configuration object

    Returns:
        LLM instance ready for use

    Raises:
        ImportError: If required packages are not installed
        ValueError: If provider type is not supported
    """
    log = get_logger()

    provider_type = provider_config.type
    model = provider_config.get_model()
    base_url = provider_config.get_base_url()
    temperature = provider_config.temperature or 0

    log.debug(
        f"Creating LLM from config: name={provider_config.name}, "
        f"type={provider_type}, model={model}"
    )

    if provider_type == "openai":
        llm = _create_openai_llm(
            model=model,
            api_key=provider_config.api_key,
            base_url=base_url,
            temperature=temperature,
        )
        url_info = f" (base_url={base_url})" if base_url else ""
        log.debug(f"OpenAI LLM created{url_info}")

    elif provider_type == "ollama":
        llm = _create_ollama_llm(
            model=model,
            base_url=base_url,
            temperature=temperature,
            num_ctx=provider_config.num_ctx,
        )
        ctx_info = f", num_ctx={provider_config.num_ctx}" if provider_config.num_ctx else ""
        log.debug(f"Ollama LLM created: base_url={base_url}{ctx_info}")

    else:
        raise ValueError(
            f"Unsupported provider type: '{provider_type}'. " "Use 'openai' or 'ollama'."
        )

    return llm


def ensure_base_messages(history: list[Any]) -> list[Any]:
    """
    Convert stored history (dicts or BaseMessages) into BaseMessages.

    Args:
        history: List of message dicts or BaseMessage objects

    Returns:
        List of BaseMessage objects (or original if LangChain unavailable)
    """
    if BaseMessage is None or HumanMessage is None:
        return history

    converted = []
    for msg in history:
        if isinstance(msg, BaseMessage):
            converted.append(msg)
            continue
        if isinstance(msg, dict) and "content" in msg:
            role = msg.get("type", "human")
            if role == "ai" and AIMessage is not None:
                converted.append(AIMessage(content=msg["content"]))
            elif role == "system" and SystemMessage is not None:
                converted.append(SystemMessage(content=msg["content"]))
            else:
                converted.append(HumanMessage(content=msg["content"]))
    return converted
