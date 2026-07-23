"""Orchestration phases — research, deep-think, execution, recovery.

Houses the pipeline stages that run after the main agent loop:
research delegation, forced deep thinking, execution phases, and
step-limit recovery.
"""

from __future__ import annotations

import re
from copy import copy
from typing import Any

from src.agent.safety import UserCancelledRun
from src.logging_config import get_logger
from src.orchestration.run_config import AgentRunConfig

# Phrases that indicate the LLM gave up due to perceived step exhaustion
# rather than providing a real answer.  Checked case-insensitively.
STEP_LIMIT_PHRASES = (
    "need more steps",
    "need additional steps",
    "requires more steps",
    "ran out of steps",
    "not enough steps",
    "step limit",
    "iteration limit",
    "too many steps",
    "maximum number of steps",
    "exceeded the limit",
    "unable to complete within",
    "couldn't finish in the allotted",
)

# Minimum context length (chars) to consider a deep_think call
# "well-grounded".  Below this, the agent likely passed references
# ("search result 1,2,3") rather than actual data.
MIN_GOOD_CONTEXT_LEN = 500

# ── preserve_tables_for_markdown regex patterns ───────────────────────────
_TABLE_SEP_RE = re.compile(r"[━─═]{3,}")
_COL_GAP_RE = re.compile(r"\S {3,}\S")
_PIPE_TABLE_RE = re.compile(r"^\s*\|.*\|\s*$")

# ── Research-delegate pipeline ────────────────────────────────────────────

WEB_TOOL_NAMES = frozenset(
    {
        "exa_search",
        "exa_find_similar",
        "exa_get_contents",
        "search_web",
        "http_get",
        "tavily_search",
        "brave_search",
        "serpapi_search",
        "google_search",
    }
)

RESEARCH_CAP_RATIO = 0.85

# ── Execution-phase: research → analyse → ACT pipeline ───────────────────

# Tools that constitute "the agent took action" (not just reading).
ACTION_TOOL_NAMES = frozenset(
    {
        "write_file",
        "append_file",
        "execute_shell_command",
    }
)

WRITE_FAILURE_PREFIXES = (
    "Error",
    "User denied execution",
    "Tool execution error",
    "User cancelled agent workflow",
)


def was_delegation_called(messages: list) -> bool:
    """Check whether delegate_task or delegate_parallel was invoked."""
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                if isinstance(tc, dict) and tc.get("name") in (
                    "delegate_task",
                    "delegate_parallel",
                ):
                    return True
    return False


def force_delegation(
    user_input: str,
    agent_response: str,
    tool_outputs: str,
    config: Any,
    log: Any,
) -> str:
    """
    Programmatically invoke delegation when the agent failed to use it
    despite the user's query clearly requiring multi-model work.

    Uses an LLM call to decompose the task, then runs delegate_parallel.
    """
    from src.tools.delegate import _delegate_config, delegate_parallel

    log.info("Forcing delegation — agent failed to use delegate tools")

    aliases = _delegate_config.get("models", {})
    allowed = _delegate_config.get("allowed_models")

    if allowed:
        available_aliases = [a for a in allowed if a in aliases]
    else:
        available_aliases = list(aliases.keys())

    if not available_aliases:
        log.warning("No model aliases available for forced delegation")
        return agent_response

    alias_list = ", ".join(available_aliases)
    decompose_prompt = (
        "You are a task decomposer. Break the following user request into "
        "2-5 independent subtasks that can be executed in parallel by "
        "different LLM models.\n\n"
        f"Available model aliases: {alias_list}\n\n"
        "For each subtask, output a JSON object on a single line with keys:\n"
        '  "task": "the subtask description",\n'
        '  "model": "alias_name"\n\n'
        "Assign different aliases to spread the workload. If a subtask "
        "involves code, prefer a code-focused alias. If research, prefer "
        "a reasoning alias. Output ONLY the JSON objects, one per line.\n\n"
        f"User request: {user_input}"
    )

    import json as _json

    try:
        from langchain_core.messages import HumanMessage as _HM

        # Use the primary LLM to decompose
        llm = build_llm_for_decomposition(config)
        if llm is None:
            log.warning("Cannot build LLM for task decomposition")
            return agent_response

        response = llm.invoke([_HM(content=decompose_prompt)])
        raw_content = getattr(response, "content", str(response))
        if isinstance(raw_content, list):
            raw_content = " ".join(
                str(c.get("text", c) if isinstance(c, dict) else c) for c in raw_content
            )
        content = str(raw_content).strip()

        # Parse subtasks from the response
        tasks: list[dict] = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            # Try to parse each line as JSON (strip trailing commas
            # that appear when LLMs format items in an array)
            if line.startswith("{"):
                clean = line.rstrip(",")
                try:
                    task_def = _json.loads(clean)
                    if "task" in task_def:
                        # Validate model alias
                        model = task_def.get("model", "")
                        if model not in aliases:
                            task_def["model"] = available_aliases[
                                len(tasks) % len(available_aliases)
                            ]
                        tasks.append(task_def)
                except _json.JSONDecodeError:
                    continue

        # Fallback: extract JSON array if the LLM returned one
        if not tasks and "[" in content:
            try:
                arr = _json.loads(content[content.index("[") : content.rindex("]") + 1])
                if isinstance(arr, list):
                    for i, item in enumerate(arr):
                        if isinstance(item, dict) and "task" in item:
                            model = item.get("model", "")
                            if model not in aliases:
                                item["model"] = available_aliases[i % len(available_aliases)]
                            tasks.append(item)
            except (_json.JSONDecodeError, ValueError):
                pass

        if not tasks:
            log.warning("Task decomposition produced no subtasks")
            return agent_response

        # Add context from what the agent already gathered
        combined_context = ""
        if tool_outputs.strip():
            combined_context += tool_outputs + "\n\n"
        if agent_response.strip():
            combined_context += "Previous analysis:\n" + agent_response

        if combined_context.strip():
            for t in tasks:
                existing = t.get("context", "")
                t["context"] = (combined_context + "\n\n" + existing).strip()

        # Execute delegation
        result = delegate_parallel(tasks=tasks, timeout=300)
        if result and result.strip():
            return result

    except ImportError:
        log.warning("Required imports for forced delegation not available")
    except Exception as exc:
        log.error(f"Forced delegation failed: {exc}")

    return agent_response


def build_llm_for_decomposition(config: Any) -> Any:
    """Build a lightweight LLM instance for task decomposition.

    Reuses the primary model configuration since decomposition is a
    quick classification-style call.
    """
    try:
        from src.providers import create_chat_model

        provider_cfg = config.get_provider_config()
        return create_chat_model(
            provider_cfg.type,
            model=provider_cfg.get_model(),
            api_key=provider_cfg.api_key,
            base_url=provider_cfg.get_base_url(),
            temperature=0.3,
            num_ctx=provider_cfg.num_ctx if provider_cfg.type == "ollama" else None,
            max_tokens=provider_cfg.max_tokens,
        )
    except Exception:
        pass
    return None


def extract_turn_messages(all_messages: list, boundary: object | None = None) -> list:
    """Extract the agent's response chain from the current turn.

    The current turn begins right after the *last* ``HumanMessage`` in
    *all_messages* (which is the user's input).  Everything after it —
    ``AIMessage`` (with or without ``tool_calls``), ``ToolMessage``, and
    the final ``AIMessage`` — is the agent's work product that should be
    persisted so the agent can continue iterating ("Ralph Loop").

    When *boundary* is provided, the scan looks for that **specific object**
    (identity comparison via ``is``) instead of the last ``HumanMessage``.
    This prevents mis-anchoring when the message list contains multiple
    ``HumanMessage`` instances (e.g. after an execution phase).
    """
    try:
        from langchain_core.messages import HumanMessage as _HM
    except ImportError:
        _HM = None  # type: ignore[assignment, misc]

    for i in range(len(all_messages) - 1, -1, -1):
        msg = all_messages[i]
        if boundary is not None:
            if msg is boundary:
                return all_messages[i + 1 :]
        elif _HM is not None:
            if isinstance(msg, _HM):
                return all_messages[i + 1 :]
        else:
            if type(msg).__name__ == "HumanMessage":
                return all_messages[i + 1 :]
    return []


def was_deep_think_called(messages: list) -> bool:
    """Check whether the `deep_think` tool was invoked in the agent messages."""
    try:
        from langchain_core.messages import ToolMessage
    except ImportError:
        ToolMessage = None  # type: ignore[assignment, misc]

    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                if isinstance(tc, dict) and tc.get("name") == "deep_think":
                    return True
        if ToolMessage is not None and isinstance(msg, ToolMessage):
            if getattr(msg, "name", None) == "deep_think":
                return True
    return False


def deep_think_had_good_context(messages: list) -> bool:
    """
    Return True if deep_think was called with substantial context data.

    Inspects the AIMessage tool_calls to find the `context` argument
    that was passed to deep_think.  If it's shorter than
    MIN_GOOD_CONTEXT_LEN characters, the agent likely passed
    references instead of actual data.
    """
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            continue
        for tc in tool_calls:
            if not isinstance(tc, dict) or tc.get("name") != "deep_think":
                continue
            args = tc.get("args", {})
            context = args.get("context", "")
            if isinstance(context, str) and len(context) >= MIN_GOOD_CONTEXT_LEN:
                return True
    return False


def preserve_tables_for_markdown(text: str) -> str:
    """Pre-process text so that table-like sections survive Rich ``Markdown()``.

    Rich's Markdown renderer collapses consecutive spaces in normal
    paragraphs, which destroys manually-aligned tables that LLMs often
    produce.  This function detects such sections and wraps them in
    fenced code blocks (````` ```) so they render monospaced.

    **Markdown pipe tables** (lines like ``| col | col |``) are left
    untouched — Rich renders them natively as proper tables.

    Detection heuristics (a line is "table-like" if it matches any):
    * Contains box-drawing separators (━ ─ ═ repeated 3+)
    * Contains 3+ consecutive spaces between non-space characters
      (typical column padding) — **unless** the line is a pipe table row
    """
    lines = text.split("\n")
    result: list[str] = []
    table_buf: list[str] = []
    in_fence = False  # track if we're already inside a code fence

    def _flush_table() -> None:
        if table_buf:
            result.append("```")
            result.extend(table_buf)
            result.append("```")
            table_buf.clear()

    for line in lines:
        stripped = line.strip()

        # Track existing code fences — don't double-wrap
        if stripped.startswith("```"):
            _flush_table()
            in_fence = not in_fence
            result.append(line)
            continue

        if in_fence:
            result.append(line)
            continue

        # Markdown pipe tables — Rich handles these natively; skip
        if _PIPE_TABLE_RE.match(line):
            _flush_table()
            result.append(line)
            continue

        is_table_line = bool(_TABLE_SEP_RE.search(line) or _COL_GAP_RE.search(line))

        if is_table_line:
            table_buf.append(line)
        else:
            _flush_table()
            result.append(line)

    _flush_table()
    return "\n".join(result)


def collect_tool_outputs(messages: list) -> str:
    """Concatenate all non-error ToolMessage outputs into a single string."""
    try:
        from langchain_core.messages import ToolMessage
    except ImportError:
        ToolMessage = None  # type: ignore[assignment, misc]

    parts: list[str] = []
    for msg in messages:
        if ToolMessage is None or not isinstance(msg, ToolMessage):
            continue
        name = getattr(msg, "name", "tool")
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content.strip() and not content.startswith("Error"):
            parts.append(f"=== {name} ===\n{content}")
    return "\n\n".join(parts)


def extract_fetched_urls(messages: list) -> list[str]:
    """Extract URLs the agent visited via web/content tools.

    Scans AIMessage tool_calls for ``exa_get_contents`` (``urls`` arg),
    ``http_get`` (``url`` arg), and ``exa_search``/``exa_find_similar``/
    ``search_web`` (extracts URLs from the corresponding ToolMessage results).
    """
    try:
        from langchain_core.messages import AIMessage, ToolMessage
    except ImportError:
        AIMessage = ToolMessage = None  # type: ignore[assignment, misc]

    urls: list[str] = []

    for msg in messages:
        if AIMessage is not None and isinstance(msg, AIMessage):
            for call in getattr(msg, "tool_calls", []):
                name = call.get("name", "")
                args = call.get("args", {})
                if name == "exa_get_contents":
                    urls.extend(args.get("urls", []))
                elif name == "http_get":
                    url = args.get("url", "")
                    if url:
                        urls.append(url)

        # Also harvest URLs from exa_search/search_web result text (lines starting with "   URL: ")
        if ToolMessage is not None and isinstance(msg, ToolMessage):
            name = getattr(msg, "name", "")
            if name in ("exa_search", "exa_find_similar", "search_web"):
                content = getattr(msg, "content", "")
                if isinstance(content, str):
                    for line in content.splitlines():
                        stripped = line.strip()
                        if stripped.startswith("URL: "):
                            urls.append(stripped[5:].strip())
            elif name in WEB_TOOL_NAMES:
                content = getattr(msg, "content", "")
                if isinstance(content, str):
                    found = re.findall(r'https?://[^\s"\'<>\[\]]+', content)
                    found = [re.sub(r"[),.:;!?]+$", "", u) for u in found]
                    urls.extend(u for u in found if len(u) > 15)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)

    # Filter SSRF-unsafe URLs
    try:
        from src.tools.http_request import _validate_url

        unique = [u for u in unique if _validate_url(u)[0]]
    except ImportError:
        pass  # http_request tool not available; keep URLs as-is

    return unique


def agent_used_web_tools(messages: list) -> bool:
    """Return True if any web/content retrieval tool was called."""
    try:
        from langchain_core.messages import ToolMessage
    except ImportError:
        ToolMessage = None  # type: ignore[assignment, misc]

    for msg in messages:
        if ToolMessage is not None and isinstance(msg, ToolMessage):
            if getattr(msg, "name", "") in WEB_TOOL_NAMES:
                return True
    return False


def run_research_delegate(
    urls: list[str],
    task: str,
    max_context_tokens: int | None = None,
    timeout: int = 300,
    cap_ratio: float = RESEARCH_CAP_RATIO,
) -> str:
    """Delegate web research to a sub-agent with a large context budget.

    The delegate fetches pages with a high context cap (default 85% vs
    the normal 10%) and returns structured specifications — not summaries.
    """
    log = get_logger()

    if not urls:
        log.debug("run_research_delegate called with no URLs — skipping")
        return ""

    from src.tools.delegate import (
        create_delegate_llm,
        get_delegate_tools,
        resolve_delegate_defaults,
        resolve_model_alias,
        run_delegate_agent,
    )

    current_delegate_tools = get_delegate_tools()
    if not current_delegate_tools:
        log.warning("Research delegate: no delegate tools configured — skipping")
        return ""

    # Build a focused research prompt
    url_list = "\n".join(f"  - {u}" for u in urls[:10])
    research_prompt = (
        f"## Research task\n\n{task}\n\n"
        "## URLs to fetch and extract\n\n"
        f"{url_list}\n\n"
        "## Instructions\n\n"
        "1. Use `exa_get_contents` or `http_get` to fetch each URL above.\n"
        "2. Extract ONLY actionable specifications from each page:\n"
        "   - Exact file formats, field names, YAML/JSON schemas\n"
        "   - Exact tool names, model aliases, CLI commands\n"
        "   - Complete code/config examples — copy them VERBATIM\n"
        "   - File paths and directory structures\n"
        "3. DO NOT summarize, paraphrase, or omit details.\n"
        "   Copy exact syntax, field names, and examples.\n"
        "4. DO NOT add your own interpretation or recommendations.\n"
        "5. Organize output by topic with clear headings.\n"
        "6. If a page is very long, focus on sections containing "
        "configuration syntax, examples, and reference material.\n"
    )

    # Build per-invocation copies of web tools with a high output cap.
    # Shallow-copy each web tool so the originals are never mutated —
    # concurrent assistant-mode calls each get their own wrapper.
    import functools

    high_cap = int(max_context_tokens * cap_ratio * 4) if max_context_tokens else 100_000

    delegate_tools: list[Any] = []
    for tool_obj in current_delegate_tools:
        tname = getattr(tool_obj, "name", "")
        if tname in WEB_TOOL_NAMES:
            tool_copy = copy(tool_obj)
            true_original = (
                getattr(tool_obj, "_uncapped_func", None)
                or getattr(tool_obj, "func", None)
                or getattr(tool_obj, "_run", None)
            )
            if true_original is not None:

                @functools.wraps(true_original)
                def _high_cap_wrapper(
                    *args: Any,
                    _orig: Any = true_original,
                    _cap: int = high_cap,
                    **kwargs: Any,
                ) -> Any:
                    result = _orig(*args, **kwargs)
                    if isinstance(result, str) and len(result) > _cap:
                        half = _cap // 2
                        return (
                            result[:half]
                            + "\n\n[... truncated for research budget ...]\n\n"
                            + result[-half:]
                        )
                    return result

                if hasattr(tool_copy, "func"):
                    tool_copy.func = _high_cap_wrapper
                else:
                    tool_copy._run = _high_cap_wrapper
            delegate_tools.append(tool_copy)
        else:
            delegate_tools.append(tool_obj)

    try:
        prov, mdl, alias_cfg = resolve_model_alias(None, None)
        prov, mdl = resolve_delegate_defaults(prov, mdl)
        timeout = max(60, min(600, alias_cfg.get("timeout", timeout)))

        log.info(
            "Running research delegate (%s/%s) for %d URLs, cap=%d chars, timeout=%ds",
            prov,
            mdl,
            len(urls),
            high_cap,
            timeout,
        )

        llm = create_delegate_llm(prov, mdl, temperature=0.3, num_ctx=alias_cfg.get("num_ctx"))
        response_text = run_delegate_agent(llm, research_prompt, "", tools_override=delegate_tools)

        if response_text.strip():
            log.info("Research delegate returned %d chars", len(response_text))
            return response_text
        else:
            log.warning("Research delegate returned empty response")
            return ""

    except Exception as exc:  # noqa: BLE001
        log.warning("Research delegate error: %s", exc)
        return ""


def force_deep_think(
    user_input: str,
    agent_response: str,
    tool_outputs: str,
    log: Any,
    *,
    research_context: str | None = None,
) -> str:
    """
    Programmatically invoke the deep_think tool when the agent failed
    to call it despite the user's explicit request.

    Passes all gathered data (tool outputs + agent's initial response)
    as context so deep_think can reason about real facts.

    When *research_context* is provided (from the research delegate),
    it is preferred over the raw *tool_outputs* because it contains
    structured, high-fidelity extractions rather than lossy summaries.
    """
    from src.orchestration.intent import DEEP_THINK_TRIGGERS
    from src.tools.deep_think import deep_think

    log.info("Programmatically invoking deep_think (agent skipped it)")

    # Build context — prefer structured research over raw tool dumps
    context_parts: list[str] = []
    if research_context and research_context.strip():
        context_parts.append(
            "## Structured research data (extracted from web pages)\n\n" + research_context
        )
        log.info(
            "Using research delegate output (%d chars) as primary deep_think context",
            len(research_context),
        )
    elif tool_outputs.strip():
        context_parts.append(
            "## Gathered data (from web searches and other tools)\n\n" + tool_outputs
        )
    if agent_response.strip():
        context_parts.append("## Agent's initial analysis\n\n" + agent_response)
    full_context = "\n\n---\n\n".join(context_parts)

    # Strip trigger phrases from the task so deep_think focuses on the
    # actual question, not on "think deep" as literal text.
    task = DEEP_THINK_TRIGGERS.sub("", user_input).strip().rstrip(".")
    if not task:
        task = user_input

    try:
        result = deep_think(
            task=task,
            context=full_context,
            max_iterations=3,
            num_branches=3,
            beam_width=2,
        )
        if result and result.strip():
            return result
        log.warning("deep_think returned empty result, using agent response")
        return agent_response
    except Exception as e:  # noqa: BLE001 — must not crash
        log.warning(f"Programmatic deep_think failed: {e}")
        return agent_response


def agent_performed_writes(messages: list) -> bool:
    """Return True if any write-oriented tool was called in *messages*."""
    try:
        from langchain_core.messages import ToolMessage
    except ImportError:
        ToolMessage = None  # type: ignore[assignment, misc]

    for msg in messages:
        if ToolMessage is not None and isinstance(msg, ToolMessage):
            name = getattr(msg, "name", "")
            if name in ACTION_TOOL_NAMES:
                content = getattr(msg, "content", "")
                if isinstance(content, list):
                    content = " ".join(
                        str(c.get("text", c) if isinstance(c, dict) else c) for c in content
                    )
                if isinstance(content, str) and not content.startswith(WRITE_FAILURE_PREFIXES):
                    return True
    return False


def run_execution_phase(
    analysis: str,
    original_prompt: str,
    context_messages: list,
    registry: Any,
    approvals: set,
    context_prefix: str | None = None,
    callbacks: list | None = None,
    *,
    config: AgentRunConfig | None = None,
    llm: Any = None,
    system_prompt: str | None = None,
    available_tools: dict | None = None,
    active_tools_list: list | None = None,
    max_context_tokens: int | None = None,
    preset_tools: set[str] | None = None,
    session_state: Any = None,
    on_tool_expansion: Any = None,
    parallel_tool_execution: bool = True,
) -> tuple[str, list]:
    """Feed the analysis back to the agent with an explicit 'execute now' prompt.

    Returns ``(output_text, agent_messages)`` from the execution pass.
    If the execution pass fails or produces nothing, returns ``("", [])``.

    When *config* is provided it is forwarded directly to ``run_agent``;
    individual keyword arguments are used only when *config* is ``None``.
    """
    from src.orchestration.runner import run_agent

    log = get_logger()
    log.info("Running execution phase — feeding analysis back to agent for action")

    # Truncate analysis if very long to leave room for tool work.
    max_analysis = 12_000
    if len(analysis) > max_analysis:
        analysis = analysis[:max_analysis] + "\n\n[... analysis truncated for brevity ...]"

    exec_prompt = (
        "You have just completed a thorough analysis. "
        "Now EXECUTE the plan — create every file, make every change.\n\n"
        "RULES:\n"
        "• Call write_file (or append_file) for EACH file that needs to be "
        "created or modified. Do NOT just describe them — actually create them.\n"
        "• Work through the plan systematically: create files one at a time.\n"
        "• After creating all files, briefly confirm what was done.\n\n"
        f"## Original request\n{original_prompt}\n\n"
        f"## Analysis / plan\n{analysis}"
    )

    exec_msgs: list = []
    try:
        if config is not None:
            exec_config = copy(config)
            if exec_config.available_tools is not None:
                exec_config.available_tools = dict(exec_config.available_tools)
            result = run_agent(
                exec_prompt,
                context_messages,
                registry,
                approvals,
                context_prefix=context_prefix,
                callbacks=callbacks,
                result_messages=exec_msgs,
                config=exec_config,
            )
        else:
            result = run_agent(
                exec_prompt,
                context_messages,
                registry,
                approvals,
                context_prefix=context_prefix,
                callbacks=callbacks,
                result_messages=exec_msgs,
                llm=llm,
                system_prompt=system_prompt,
                available_tools=dict(available_tools) if available_tools else available_tools,
                active_tools_list=active_tools_list,
                max_context_tokens=max_context_tokens,
                preset_tools=preset_tools,
                session_state=session_state,
                on_tool_expansion=on_tool_expansion,
                parallel_tool_execution=parallel_tool_execution,
            )
        if result and result.strip():
            wrote = agent_performed_writes(exec_msgs)
            log.info(
                "Execution phase complete — files written: %s",
                "yes" if wrote else "no",
            )
            return result, exec_msgs
    except UserCancelledRun:
        raise
    except Exception as exc:  # noqa: BLE001
        log.warning("Execution phase failed: %s", exc)

    return "", []


def is_step_limit_apology(text: str) -> bool:
    """
    Detect when the LLM returns a vague "need more steps" apology
    instead of an actual answer.

    When the model senses it has been going for many rounds it sometimes
    produces a short, unhelpful message like "Sorry, I need more steps
    to process this request" and stops.  This is technically non-empty
    content, so the normal empty-response guard misses it.  We treat it
    the same way: trigger recovery to extract partial results or retry.

    Only flags short messages (< 300 chars) to avoid false positives on
    legitimate long answers that happen to mention "steps".
    """
    if not text or len(text) > 300:
        return False
    lower = text.lower()
    return any(phrase in lower for phrase in STEP_LIMIT_PHRASES)


def extract_partial_results(messages: list) -> str | None:
    """
    Extract useful information from partial agent messages.

    When recursion limit is hit, try to gather what the agent learned
    from tool calls before failing.
    """
    if not messages:
        return None

    try:
        from langchain_core.messages import AIMessage, ToolMessage
    except ImportError:
        AIMessage = ToolMessage = None  # type: ignore[assignment, misc]

    tool_results = []
    last_ai_content = None

    for msg in messages:
        if ToolMessage is not None and isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", "tool")
            content = getattr(msg, "content", "")
            if content and not content.startswith("Error"):
                tool_results.append(f"**{tool_name}:** {content}")

        elif AIMessage is not None and isinstance(msg, AIMessage):
            content = getattr(msg, "content", "")
            if content and len(content) > 50:
                last_ai_content = content

    if not tool_results and not last_ai_content:
        return None

    # NOTE: The output deliberately avoids _ERROR_PREFIXES so that
    # _is_valid_response() returns True and the turn is saved to
    # history.  This is essential for the "Ralph Loop" pattern where
    # the agent iterates on a complex task across multiple turns —
    # discarding partial progress would force it to restart from
    # scratch every time.
    parts = ["The task could not be completed in one pass. " "Here is the progress so far:\n\n"]

    if tool_results:
        parts.append("*Information gathered:*\n")
        # Keep up to 10 results.  Early results (from initial searches)
        # tend to be most relevant, so we take from the front.
        for result in tool_results[:10]:
            parts.append(f"- {result}\n")

    if last_ai_content:
        parts.append(f"\n*Last response attempt:*\n{last_ai_content}")

    return "".join(parts)


def recover_from_step_limit(
    agent_executor: Any,
    result: dict,
    input_messages: list,
    invoke_config: dict,
    log: Any,
) -> str:
    """
    Attempt to recover a useful answer after the agent exhausts its steps.

    Recovery cascade:
      1. Re-invoke with a nudge (via ``stream()`` so intermediate tool
         results are preserved even if the retry also hits its limit).
      2. Build a response from tool results gathered in the retry *or*
         the original run.
      3. Use the structured partial-results extractor.
      4. Return an explicit, actionable error message.
    """
    from src.orchestration.runner import build_tool_results_response, extract_response

    # Collect all messages across both runs for fallback extraction
    all_messages: list = list(result.get("messages", []))

    # ── Step 1: Retry with a nudge ────────────────────────────────
    retry_result: dict = {"messages": []}
    try:
        log.info("Recovery: re-invoking agent with nudge prompt")
        retry_messages = list(result.get("messages", input_messages))

        try:
            from langchain_core.messages import HumanMessage as HM

            retry_messages.append(
                HM(
                    content=(
                        "Please provide your final response now. "
                        "Summarize what you have found so far. "
                        "Do NOT call any more tools — just answer "
                        "with the information you already have."
                    )
                )
            )
        except ImportError:
            retry_messages.append(
                {
                    "type": "human",
                    "content": (
                        "Please provide your final response now. "
                        "Summarize what you have found so far."
                    ),
                }
            )

        retry_config = dict(invoke_config)
        # Tight limit: allow at most 1 tool call + final answer.
        # The nudge says "Do NOT call tools" but some models ignore
        # that; a low limit prevents wasting the recovery budget.
        retry_config["recursion_limit"] = 4

        # Use stream() (like the main flow) so we keep intermediate
        # messages even if GraphRecursionError fires.
        try:
            for chunk in agent_executor.stream(
                {"messages": retry_messages},
                config=retry_config,
                stream_mode="values",
            ):
                if isinstance(chunk, dict) and "messages" in chunk:
                    retry_result = chunk
        except RecursionError:
            log.warning("Recovery retry also hit recursion limit")

        # Merge retry messages into the combined pool for fallback
        all_messages.extend(retry_result.get("messages", []))

        retry_response = extract_response(retry_result, log)
        if retry_response and not is_step_limit_apology(retry_response):
            log.info("Recovery succeeded: got response on retry")
            return retry_response

    except UserCancelledRun:
        raise
    except Exception as retry_err:  # noqa: BLE001 — recovery must not crash
        log.warning(f"Recovery retry failed: {retry_err}")

    # ── Step 2: Build response from tool results ──────────────────
    # Check retry messages first (more recent), then original run
    combined_result: dict = {"messages": all_messages}
    tool_response = build_tool_results_response(combined_result)
    if tool_response:
        log.info("Recovery: returning tool results to user")
        return tool_response

    # ── Step 3: Structured partial-results extractor ──────────────
    partial = extract_partial_results(all_messages)
    if partial:
        log.info("Recovery: returning partial results to user")
        return partial

    # ── Step 4: Nothing worked — tell the user but keep the turn
    #    in history so the agent can retry on the next invocation
    #    (Ralph Loop).  The message deliberately avoids error
    #    prefixes in _ERROR_PREFIXES to pass _is_valid_response(). ──
    log.error("All recovery attempts failed — no usable content")
    return (
        "I was unable to complete this task in the allotted steps. "
        "The query may require many tool calls.\n\n"
        "You can say **continue** and I will pick up where I left off, "
        "or you can rephrase / break the question into smaller parts."
    )
