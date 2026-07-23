"""
Agent core setup using LangGraph ReAct Agent.
Decoupled from CLI; callable from any interface.
Supports multiple LLM providers: OpenAI, Ollama, and OpenAI-compatible APIs.
"""

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Annotated, Any, Protocol, TypedDict, runtime_checkable

from src.common.types import AgentRunConfig
from src.logging_config import get_logger

if TYPE_CHECKING:
    from src.config import ModelConfig, ProviderConfig

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
- Execute every task step-by-step until complete. Never stop halfway.
- Use tools when needed — if a task requires external data or actions, use the
  appropriate tools. For conversational exchanges, respond directly.
- Synthesize tool results into one complete, polished response.
- For complex requests: break down, execute each part, combine results.

## When NOT to use tools
For conversational messages, greetings, or questions answerable from training data
(e.g. "hi", "how are you", "what is 2+2", "explain X"), respond directly without
calling any tools. Reserve tools for tasks that genuinely require:
- External data (web search, file read, API call)
- Computation or actions beyond text generation
- Information that may have changed since training

## Accuracy
- Base answers strictly on tool results. Do not fill gaps with assumptions.
- Note when information was not available; distinguish facts from inferences.
- Cite URLs/sources from tool results for factual claims.

## Forbidden
- Never say "I'm ready to help!" — do the work.
- Never stop after using tools — synthesize results.
- Never invent numbers, dates, URLs, or specifics not from tool results.
- Never say "I need more steps" — deliver your best answer with what you have.
- **Never preface a response with "You're absolutely right" / "You're right" / "I apologize" / "You're raising an important point" / similar validation phrases unless you are about to substantively revise your prior answer.** If new evidence has changed your conclusion, state plainly what changed and why. If your conclusion is unchanged after considering the user's input, say so explicitly — e.g., "I've reviewed what you said and my conclusion is unchanged: [conclusion]. What would change it is [evidence]." Validating the user and then repeating the same answer is dishonest; it gives the illusion of update without an actual update and amplifies the trust loss when the user notices.

## Tools
You start with one meta-tool: `request_tools`. Load tools by name: `request_tools(add=["web_search"])`. Release with `request_tools(remove=["web_search"])`. If unsure which tool to use, call `request_tools()` with no arguments to see the catalog. Request only tools relevant to the current task.

For research queries the canonical tool is **`web_search`** — it runs a multi-provider fan-out and returns a structured Markdown picture (sources + extracts + coverage). If `web_search` returns a `## Synthesis unavailable` section, read the source extracts that follow and synthesise from them in your own response before replying to the user.

**When `web_search` yields low or no content** (the Coverage block shows `Fetched: 0/N successful` or `Synthesis: skipped` with no extracts), do NOT recommend external resources like "check XE.com / Amazon / [some site]" — that is URL fabrication. Instead either (a) issue ONE refined `web_search` with a more specific query (different terms, region keyword, named retailer), or (b) if you've already retried once, state plainly what was found and what wasn't (e.g., "I found Soudal product pages but none with Vienna pricing"). Never invent URLs or vendor names not present in the tool output.

**When a `web_search` source is annotated `(snippet-only)`** (the URL is real but the fan-out fetcher couldn't extract page content — common on financial sites, paywalled news, anti-bot pages), try `http_get` on that surfaced URL before refusing. The `snippet-only` annotation is a signal, not a verdict — `http_get` uses a different fetch path and frequently succeeds where the search fan-out did not. Only refuse after `http_get` has also failed.

**When the user explicitly provides a URL in their message** — e.g., *"check this page: https://example.com/x"* or *"look at https://foo.org/bar"* — your FIRST action should be `http_get` on that URL, not `web_search` with a `site:` query. `web_search` is for discovery (you don't know where the information lives); `http_get` is for retrieval (the user has told you exactly where). Searching the web for a URL that was already given to you is wasted budget and frequently fails when the index doesn't surface the exact URL in top results. Only fall back to `web_search` if `http_get` fails or the page returns no meaningful content.

**When the user names a specific service or API by name** — e.g., *"use the Wayback Machine"*, *"check the GitHub releases"*, *"look at the RSS feed for X"* — and the service has a known URL/API surface, `http_get` against the canonical URL is the right tool, NOT `web_search` *about* the service. Searching the web for "Wayback Machine snapshot of X 2022" doesn't actually query the Wayback Machine — it returns articles ABOUT the Wayback Machine. Examples of services that should be reached via `http_get` rather than `web_search`: Wayback Machine snapshots, the GitHub releases API for a known repository, a user-supplied RSS or Atom feed URL. Construct the URL from what you already know about the service's standard surface (e.g., the Wayback Machine takes a date and a target URL; the GitHub releases endpoint takes an owner and repo) — do not copy literal URLs from these instructions into your tool calls or response.

If `http_get` against the canonical surface returns no result or 404, report that honestly ("the Wayback Machine has no snapshot of X for date Y") — do NOT downgrade to `web_search` *about* the service and conclude from those results, because they don't reflect what the service actually contains. Do NOT fabricate or guess URLs for services the user did not name and you have not actually fetched — only mention a URL in your response if it came from an actual tool result.

**When the user asks about citation counts, "most-cited" / "top-cited" papers, or wants academic papers ranked by influence** — e.g., *"5 most-cited diffusion-model papers from arXiv this year"*, *"the top-cited papers on graph neural networks since 2024"* — the canonical service is **Semantic Scholar**. Its paper-search API returns each result with a `citationCount` field, supports topic queries and a `year` filter, and includes arXiv / DOI / other external identifiers in the response so you can link back to the source. Reach for it via `http_get` against `api.semanticscholar.org`. arXiv itself does NOT publish citation counts — `http_get` against an `arxiv.org/abs/...` URL returns the abstract, not citation data; using only `web_search` for "most cited arxiv X papers 2026" returns subjective blog rankings, not citation-sorted results. Sort the Semantic Scholar response by `citationCount` descending and take the top N. Construct the URL from what you know about the API (query parameter, year filter, fields parameter naming `title`, `authors`, `citationCount`, `abstract`, `externalIds`) — do not copy literal URL templates from these instructions into your tool calls. If the API returns rate-limited (HTTP 429), wait briefly and retry once; if it still fails, report what you could not obtain honestly rather than substituting a non-citation-ranked list.

If asked for "top papers" or "best papers" without an explicit citation criterion, ask the user once whether they want "most cited" (Semantic Scholar route) or "most recent" (arXiv listings route) before burning a search round on the wrong axis.

### Batching
Batch independent tool calls in a single response for parallel execution. Keep dependent operations sequential. Keep `request_tools` calls alone.

### Efficiency
- **Write comprehensive scripts, not one-liners.** When a task has multiple steps (download, extract, configure, test), combine them into ONE script that handles the full workflow — including error handling. Don't do each step as a separate tool call.
- **Never re-check what you already know.** Your checkpoints contain confirmed facts. Don't re-verify confirmed results.
- **Read error messages before retrying.** When a command fails, the error output tells you exactly why. Fix the specific issue — don't make a random variation hoping it works.
- **Batch related operations.** If you need 5 files downloaded, write one script that downloads all 5 — not 5 separate tool calls.
- **Checkpoint exact commands, not descriptions.** When a command works, checkpoint the FULL command line so you can copy-paste it verbatim. "EXACT COMMAND: LD_LIBRARY_PATH=~/lib ~/bin/as" is useful. "as works with LD_LIBRARY_PATH" is not — you'll waste time reconstructing the invocation.

### Debugging
When something doesn't work as expected:
1. **Read the error.** The error message tells you exactly what's wrong. Read it CAREFULLY before doing anything.
2. **Isolate.** If tests fail, run the SINGLE failing test alone and read its full output. Don't re-run the entire suite.
3. **Search.** Look up the specific error or a working example — web search, man pages, Stack Overflow.
4. **Fix ONE thing.** Make a targeted change addressing the specific error. Don't rewrite the whole file.
5. **Commit to ONE strategy.** Pick either "search for a working reference and adapt it" OR "fix incrementally from error messages." Don't alternate between both.
6. **Never rewrite the same file more than twice** without searching for a working reference first.

### Work Cycle for Complex Tasks
For multi-step tasks, follow this cycle every 5-8 tool calls:
1. **PLAN** (Chain of Thought): Before doing anything, think through what specific information you need. Write out: "I need to find [X]. To get [X], I should search for [Y]." This prevents wasted actions.
2. **RESEARCH**: Search using web search and local system resources. After getting results, EVALUATE them:
   - Do the results contain a specific, actionable answer (an exact URL, a concrete command, a package name)?
   - If YES → proceed to step 3.
   - If NO → refine your search query and search AGAIN. Do NOT guess or fill in details from memory. Your training data has stale URLs and incomplete procedures. Keep searching until you have actionable specifics.
3. **ACT**: Execute the approach from your research. Make sure you have the right tools loaded. Write comprehensive scripts, not one-liners.
4. **EVALUATE**: Did it work? Checkpoint the outcome — both the result AND the exact working command.
5. **PIVOT or PROCEED**: If it failed, read the error, checkpoint what you learned, and go back to step 1. Think through WHY it failed and what DIFFERENT category of approach to try. If it worked, checkpoint and move to the next phase.

Critical rules:
- Search before you act — even if you think you know the answer.
- After a search, ask: "Do I have a SPECIFIC URL/command/package, or just general info?" If just general → search again more specifically.
- Never guess URLs or package names from memory — always search for the current, working ones.
- When something works, checkpoint the EXACT command and move on.
- When something fails, checkpoint the failure reason so you don't repeat it.

## Research
Search for information using ALL available sources. Web search is the broadest. Also use local resources: `man` pages, `--help` output, documentation directories, system package databases. For complex problem-solving: research until you have ACTIONABLE specifics, THEN act. Use `http_get` to read specific pages when search snippets aren't enough. If your first search returns vague results, refine the query and search again — don't fall back to guessing.

## Time and Date Handling
Today's date is injected into this prompt and prefixed onto every user message as `[YYYY-MM-DD HH:MM:SS UTC]`. Trust those values — you do not need to call `get_current_datetime` just to know what day it is.

Use `get_current_datetime` ONLY when you need sub-day precision (a precise timestamp for a log entry, a duration calculation, a timezone conversion). For "what day is it" / "what year is it" / "what weekday corresponds to today" the injected date is authoritative.

Never invent a date that contradicts the injected value. Never assume the injected date is wrong without first calling `get_current_datetime`.

## User Constraints
Trust user-stated facts. Don't verify unless they demonstrably fail.

## Context Budget
Limited context window. Use `list_directory` first, then read only needed files. Page large files (200 lines). Delegate independent subtasks.

## Clarification Policy
When an action is irreversible AND the target scope is ambiguous, ask one specific
question before acting ("Should I delete only *.log files or everything in /tmp/build/?").
Ask exactly ONE question, then stop. State an assumption and proceed when risk is low
or one option is a clear default. Never resolve conflicting instructions silently — surface them.

**When the user's message provides NO actionable task** — e.g. *"Do it now please."*, *"Help me"*, *"Proceed"*, *"Continue"*, or any message that refers to work but does not say what work — do NOT speculate by listing goals / tasks / inbox / tool catalog. Respond directly with one specific clarifying question asking what task they want performed. Do not call any tools before asking; speculative exploration just wastes budget and produces noise. The clarifying question itself IS the complete response for this turn.
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
    log.debug("Creating ReAct agent with %d tools", len(tools))
    agent_executor = create_react_agent(
        model=llm,
        tools=tools,
        prompt=final_prompt,
    )

    log.info("Agent initialized: %s", type(llm).__name__)
    return agent_executor


def _format_model_detail(value: Any) -> str:
    """Return a human-readable description of one model entry."""
    from src.config import ModelConfig

    if isinstance(value, ModelConfig):
        extras: list[str] = []
        if value.temperature is not None:
            extras.append(f"temp={value.temperature}")
        if value.context_window is not None:
            extras.append(f"ctx={value.context_window}")
        detail = f"{value.provider}/{value.model}"
        if extras:
            detail += f" ({', '.join(extras)})"
        return detail
    if isinstance(value, dict):
        provider = value.get("provider", "?")
        model = value.get("model", "?")
        extras = []
        if "temperature" in value:
            extras.append(f"temp={value['temperature']}")
        if "context_window" in value:
            extras.append(f"ctx={value['context_window']}")
        elif "num_ctx" in value:
            extras.append(f"ctx={value['num_ctx']}")
        detail = f"{provider}/{model}"
        if extras:
            detail += f" ({', '.join(extras)})"
        return detail
    if isinstance(value, str):
        return value
    return str(value)


def _format_models_table(
    models: dict[str, Any],
    delegation_models: list[str] | None = None,
) -> str:
    """Format models registry into a readable table for the system prompt.

    If *delegation_models* is provided, the table is split into two
    groups: models allowed for delegation (highlighted) and the remaining
    models which are only available via ``/model``.

    Returns an empty string when there are no models to show.
    """
    if not models:
        return ""

    lines: list[str] = ["## Available Models", ""]

    if delegation_models:
        lines.append("### Delegation targets (use with `delegate_task` / `delegate_parallel`):")
        lines.append("")
        for name in delegation_models:
            if name in models:
                detail = _format_model_detail(models[name])
                lines.append(f"- **{name}** → `{detail}`")
        lines.append("")

        others = {k: v for k, v in models.items() if k not in delegation_models}
        if others:
            lines.append("### Other models (available via `/model` command):")
            lines.append("")
            for name, value in others.items():
                detail = _format_model_detail(value)
                lines.append(f"- **{name}** → `{detail}`")
            lines.append("")
    else:
        lines.append(
            "Use these names as the `model` parameter in `delegate_task` / `delegate_parallel`:"
        )
        lines.append("")
        for name, value in models.items():
            detail = _format_model_detail(value)
            lines.append(f"- **{name}** → `{detail}`")
        lines.append("")

    return "\n".join(lines)


def format_milestone_instructions(milestones: list) -> str:
    """Generate a system prompt section describing the task milestones."""
    lines = [
        "## Progress Milestones",
        "",
        "This task has been decomposed into milestones. Call `report_progress` "
        "when you START each milestone.",
        "",
    ]
    for m in milestones:
        lines.append(f"{m.index}. {m.title}")
    lines.append("")
    lines.append(
        "Call `report_progress(milestone_index=N)` as you begin each milestone. "
        "This is mandatory — the user relies on these updates to track progress."
    )
    lines.append("")
    lines.append(
        "**Focus rule:** work on ONE milestone at a time. Do not revisit a "
        "completed milestone or repeat tool calls from a previous milestone. "
        "If a tool call fails, try a DIFFERENT approach — do not retry the "
        "same call more than once."
    )
    return "\n".join(lines)


_DELEGATE_TOOL_NAMES: frozenset[str] = frozenset({"delegate_task", "delegate_parallel"})
_MERGE_TOOL_NAMES: frozenset[str] = frozenset({"merge_pull_request", "github_merge_pull_request"})
_MERGE_CI_GUARD_PROMPT = """### GitHub Merge Safety

- Before every `merge_pull_request` call, fetch the current CI status first with `get_pull_request_status`.
- Only proceed if the CI Summary check is `completed/success`.
- If any check is `in_progress`, `queued`, `failure`, or missing, do not attempt the merge.
- After `Repository rule violations`, do not retry the merge in the same session unless CI state has changed.
"""


def build_system_prompt(
    base_prompt: str | None = None,
    mode_additions: str | None = None,
    models: dict[str, Any] | None = None,
    delegation_models: list[str] | None = None,
    tool_instructions: str | None = None,
    milestone_instructions: str | None = None,
    active_tool_names: set[str] | None = None,
    decision_accountability_prompt: str | None = None,
    pre_action_confirmation_prompt: str | None = None,
) -> str:
    """
    Build a complete system prompt with mode-specific additions.

    Args:
        base_prompt: Base system prompt (uses default if None)
        mode_additions: Additional instructions from memory mode
        models: Named models registry to expose to the agent
        delegation_models: Subset of model names allowed for delegation
            (``None`` means all models are allowed)
        tool_instructions: Optional tool-call formatting instructions to
            append.  Defaults to ``None`` (no instructions), since
            LangGraph ``bind_tools()`` handles tool-call formatting at the
            API level for all supported providers.  Set to a non-empty
            string to inject custom instructions for providers that need
            explicit guidance.
        milestone_instructions: Optional milestone progress section
            generated by ``format_milestone_instructions()``.  When
            non-None, appended after ``tool_instructions``.
        active_tool_names: Set of tool names currently active or available
            to the agent (active + on-demand pool).  When provided, the
            models table is only included if a delegation tool
            (``delegate_task`` or ``delegate_parallel``) is present.
            When ``None``, the table is included whenever *models* is
            non-empty (backward-compatible default).
        decision_accountability_prompt: Optional decision accountability block
            (``ACCOUNTABILITY_PROMPT`` from ``reflection_delegate.py``).
            Injected last when ``config.decision_accountability["enabled"]``
            is True (M2 integration point — ADR-0052).

    Returns:
        Combined system prompt
    """
    base = base_prompt if base_prompt else DEFAULT_SYSTEM_PROMPT

    # Inject current date so models with pre-current-year training cutoffs
    # trust real-time search results over training-data priors (issue #886).
    # "## Time and Date Handling — CRITICAL" in the base prompt already tells
    # the agent to use get_current_datetime, but the agent needs to *know*
    # today's date in absolute terms before reasoning can begin.
    date_prefix = f"Today's date is {datetime.now(timezone.utc).strftime('%B %d, %Y (UTC)')}.\n\n"  # noqa: UP017
    base = date_prefix + base

    parts = [base]

    delegation_accessible = active_tool_names is None or bool(
        active_tool_names & _DELEGATE_TOOL_NAMES
    )
    models_section = (
        _format_models_table(
            models or {},
            delegation_models=delegation_models,
        )
        if delegation_accessible
        else ""
    )
    if models_section:
        parts.append(models_section)

    if mode_additions:
        parts.append(mode_additions)

    # Inject tool-call formatting instructions only when explicitly provided.
    # bind_tools() handles formatting at the API level, so injecting raw-JSON
    # examples into the system prompt can conflict with the structured
    # tool_calls response format and cause parsing failures (especially on
    # vLLM's openai_tool_parser).
    if tool_instructions:
        parts.append(tool_instructions)

    if milestone_instructions:
        parts.append(milestone_instructions)

    merge_guard_enabled = bool(active_tool_names and active_tool_names & _MERGE_TOOL_NAMES)
    if merge_guard_enabled:
        parts.append(_MERGE_CI_GUARD_PROMPT)

    if decision_accountability_prompt:
        parts.append(decision_accountability_prompt)

    if pre_action_confirmation_prompt:
        parts.append(pre_action_confirmation_prompt)

    return "\n\n".join(parts)


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
    if max_tokens <= 0:
        return content
    max_chars = max_tokens * 4
    if len(content) <= max_chars:
        return content
    half = max_chars // 2
    return (
        content[:half] + f"\n\n[... truncated: {len(content) - max_chars} chars removed "
        f"to fit context window ...]\n\n" + content[-half:]
    )


def _copy_with_content(msg: Any, content: str) -> Any:
    """Create a message copy with updated content, Pydantic v1/v2 safe."""
    if hasattr(msg, "model_copy"):
        return msg.model_copy(update={"content": content})
    if hasattr(msg, "copy"):
        return msg.copy(update={"content": content})
    from copy import copy as _shallow_copy

    clone = _shallow_copy(msg)
    clone.content = content
    return clone


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
                    else:
                        bucket[idx] = _copy_with_content(msg, trimmed)

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

    # Truncate individual oversized messages and collect token costs in a single pass.
    token_costs: list[int] = []
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
                else:
                    history[idx] = _copy_with_content(msg, trimmed)
                est = _estimate_msg_tokens(history[idx])
        token_costs.append(est)
    total_history = sum(token_costs)
    drop_count = 0
    while drop_count < len(history) and total_history > history_budget:
        total_history -= token_costs[drop_count]
        drop_count += 1
    history = history[drop_count:]
    dropped = drop_count

    # Remove orphaned messages at the head of the trimmed history.
    # A ToolMessage is orphaned if no preceding AIMessage carries its tool_call_id.
    # An AIMessage at the head is orphaned because there is no preceding user
    # message, violating role-alternation constraints (llama.cpp Jinja, etc.).
    try:
        from langchain_core.messages import ToolMessage as _ToolMessage
    except ImportError:
        _ToolMessage = None  # type: ignore[assignment, misc]

    removed = 0
    while removed < len(history):
        msg = history[removed]
        if _ToolMessage is not None and isinstance(msg, _ToolMessage):
            total_history -= _estimate_msg_tokens(msg)
            removed += 1
            continue
        if AIMessage is not None and isinstance(msg, AIMessage):
            total_history -= _estimate_msg_tokens(msg)
            removed += 1
            continue
        break
    if removed:
        history = history[removed:]
        dropped += removed

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
            ``ModelConfig.context_window`` or a sensible default).  When set,
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

    # Inject context prefix as a HumanMessage.  Using SystemMessage here
    # would place a second system-role message after position 0, which
    # strict OpenAI-compatible providers (vLLM, Qwen3) reject with a
    # validation error (BUG-238).
    if context_prefix:
        ctx_content = f"Current context:\n{context_prefix}"
        result.append(HumanMessage(content=ctx_content))

    # Add conversation history
    result.extend(history_messages)

    # Add current user input
    result.append(HumanMessage(content=user_input))

    # Trim to budget if a limit is set
    if max_context_tokens and max_context_tokens > 0:
        result = _trim_to_token_budget(result, max_context_tokens)

    return result


def create_llm_from_provider_config(
    provider_config: "ProviderConfig",
    model_config: "ModelConfig | None" = None,
) -> Any:
    """
    Create an LLM instance from a ProviderConfig and optional ModelConfig.

    Delegates to the centralized ``src.providers`` registry which
    supports openai, ollama, anthropic, google, and OpenAI-compatible
    providers (xAI, vLLM, Groq, etc.).

    Args:
        provider_config: Provider connection configuration.
        model_config: Model settings (model name, temperature, etc.).
            When omitted, a default ModelConfig is synthesized from the
            provider's default model.

    Returns:
        LLM instance ready for use

    Raises:
        ImportError: If required packages are not installed
        ValueError: If provider type is not supported
    """
    from src.config import ModelConfig as _ModelConfig
    from src.providers import create_chat_model_from_configs, get_default_model

    log = get_logger()

    if model_config is None:
        model_config = _ModelConfig(
            provider=provider_config.name,
            model=get_default_model(provider_config.type),
        )

    log.debug(
        "Creating LLM from config: name=%s, type=%s, model=%s",
        provider_config.name,
        provider_config.type,
        model_config.model,
    )

    llm = create_chat_model_from_configs(provider_config, model_config)

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


@runtime_checkable
class AgentRunner(Protocol):
    """Protocol that decouples assistant handler from the concrete run_agent implementation.

    Callers pass session-constant parameters via ``config: AgentRunConfig``
    and per-call parameters as positional/keyword arguments.
    """

    def __call__(
        self,
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
        task_complexity: Any | None = None,
    ) -> str: ...
