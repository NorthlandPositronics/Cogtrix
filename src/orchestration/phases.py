"""Orchestration phases — research, deep-think, execution, recovery.

Houses the pipeline stages that run after the main agent loop:
research delegation, forced deep thinking, execution phases, and
step-limit recovery.
"""

from __future__ import annotations

import re
import secrets
import unicodedata
import urllib.parse
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
        # ADR-0056 PR-G renamed search_web → web_search. Both kept here
        # so any future re-introduction of the legacy tool still scopes
        # as a web tool.
        "search_web",
        "web_search",
        "http_get",
        "tavily_search",
        "brave_search",
        "serpapi_search",
        "google_search",
    }
)

RESEARCH_CAP_RATIO = 0.85

# ── Research-query heuristic ──────────────────────────────────────────────

_RESEARCH_INTENT_PHRASES = frozenset(
    {
        "research",
        "find information",
        "find out",
        "search for",
        "look up",
        "look into",
        "what is",
        "what are",
        "who is",
        "who are",
        "how does",
        "how do",
        "how is",
        "tell me about",
        "explain",
        "give me information",
        "give me details",
        "learn about",
        "find the best",
        "find examples",
        "show me examples",
        "summarize",
        "summarise",
        "overview of",
        "list of",
    }
)

_ACTION_INTENT_PHRASES = frozenset(
    {
        "write",
        "create",
        "generate",
        "edit",
        "fix",
        "run",
        "execute",
        "build",
        "implement",
        "deploy",
        "install",
        "configure",
        "set up",
        "refactor",
        "debug",
        "test",
        "commit",
        "push",
        "delete",
        "remove",
    }
)


def _looks_like_research_query(text: str) -> bool:
    """Lightweight heuristic: does this query look like a web research request?

    No LLM call — fast enough to run synchronously before the main agent.
    Returns True when the text contains research-intent phrases and does NOT
    contain action-intent phrases (write/create/edit/run/etc.).

    Multi-word research phrases use substring matching (they are specific
    enough).  Single-word action phrases use word-boundary matching to avoid
    false positives from substrings (e.g. "test" inside "latest").
    """
    lower = text.lower()

    # Check for negation preceding matched phrases (e.g. "don't research", "no need to find")
    _neg_re = re.compile(
        r"\b(?:not|no|never|don'?t|doesn'?t|can'?t|cannot|without|stop|avoid)\b",
        re.IGNORECASE,
    )

    def _phrase_is_negated(phrase: str) -> bool:
        pos = lower.find(phrase)
        if pos < 0:
            return False
        window = lower[max(0, pos - 25) : pos]
        return bool(_neg_re.search(window))

    has_research_intent = any(
        phrase in lower and not _phrase_is_negated(phrase) for phrase in _RESEARCH_INTENT_PHRASES
    )
    has_action_intent = any(
        (
            re.search(r"\b" + re.escape(phrase) + r"\b", lower) is not None
            if " " not in phrase
            else phrase in lower
        )
        for phrase in _ACTION_INTENT_PHRASES
    )
    return has_research_intent and not has_action_intent


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
    llm: Any = None,
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
    _nonce = secrets.token_hex(8)
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
        f"User request: [{_nonce}]{user_input}[/{_nonce}]"
    )

    import json as _json

    try:
        from langchain_core.messages import HumanMessage as _HM

        # Use the primary LLM to decompose
        llm = llm if llm is not None else build_llm_for_decomposition(config)
        if llm is None:
            log.warning("Cannot build LLM for task decomposition")
            return agent_response

        # Wrap the decomposition LLM call in a temporary executor so we can
        # enforce a timeout.  Python threads cannot be cancelled;
        # shutdown(wait=False) lets the hung thread die in the background
        # without blocking the caller.
        import concurrent.futures as _cf

        _pool = _cf.ThreadPoolExecutor(max_workers=1)
        try:
            _fut = _pool.submit(llm.invoke, [_HM(content=decompose_prompt)])
            try:
                response = _fut.result(timeout=60)
            except _cf.TimeoutError:
                _fut.cancel()
                _pool.shutdown(wait=False)
                log.warning(
                    "Task decomposition LLM call timed out after 60s — "
                    "returning original agent response"
                )
                return agent_response
        finally:
            _pool.shutdown(wait=False)
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

        # Add context from what the agent already gathered.
        # Wrap untrusted content in nonce delimiters to prevent prompt injection
        # via adversarial web content or echoed LLM output (BUG-148 / ARCH-403).
        _MAX_DELEGATION_CONTEXT = 12_000
        combined_context = ""
        if tool_outputs.strip():
            safe_tool_outputs = f"<tool_data_{_nonce}>{tool_outputs}</tool_data_{_nonce}>"
            combined_context += safe_tool_outputs + "\n\n"
        if agent_response.strip():
            safe_response = f"<prior_response_{_nonce}>{agent_response}</prior_response_{_nonce}>"
            combined_context += "Previous analysis:\n" + safe_response

        if len(combined_context) > _MAX_DELEGATION_CONTEXT:
            combined_context = combined_context[:_MAX_DELEGATION_CONTEXT] + "\n[... truncated]"

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
    except UserCancelledRun:
        raise
    except Exception as exc:
        log.error("Forced delegation failed: %s", exc, exc_info=True)

    return agent_response


def build_llm_for_decomposition(config: Any) -> Any:
    """Build a lightweight LLM instance for task decomposition.

    Reuses the primary model configuration since decomposition is a
    quick classification-style call.
    """
    log = get_logger()
    try:
        from src.providers import create_chat_model

        pc, mc = config.resolve_llm_config()
        return create_chat_model(
            pc.type,
            model=mc.model,
            api_key=pc.api_key,
            base_url=pc.get_base_url(),
            temperature=0.3,
            num_ctx=mc.context_window if pc.type == "ollama" else None,
            max_tokens=mc.max_tokens,
        )
    except Exception as exc:
        log.debug("Failed to build decomposition LLM: %s", exc)
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


def _normalize_url(raw: str) -> str | None:
    """Normalize a raw URL string before SSRF validation.

    Applies percent-decoding, strips control characters, and Unicode NFC
    normalization, then re-encodes to a canonical form.  Returns ``None``
    if the URL cannot be safely normalized (malformed, empty host, etc.).

    This prevents bypass attempts such as:
      * Percent-encoded private IPs: ``http://192.%31.1.1/``
      * Control-character injection: ``http://example.com\\x00@internal/``
      * Unicode lookalike tricks in the host component
    """
    try:
        # Strip surrounding whitespace and ASCII control chars (< 0x20 or DEL)
        stripped = raw.strip()
        stripped = "".join(ch for ch in stripped if ord(ch) >= 0x20 and ord(ch) != 0x7F)
        if not stripped:
            return None

        # Decode percent-encoding so that encoded private IPs are exposed
        parsed = urllib.parse.urlparse(stripped)
        netloc_decoded = urllib.parse.unquote(parsed.netloc)
        path_decoded = urllib.parse.unquote(parsed.path)

        # Apply NFC Unicode normalization to collapse lookalike characters
        netloc_norm = unicodedata.normalize("NFC", netloc_decoded)
        path_norm = unicodedata.normalize("NFC", path_decoded)

        # Re-encode with safe characters preserved so the URL remains valid
        netloc_reenc = urllib.parse.quote(netloc_norm, safe=".:@[]!$&'()*+,;=-")
        path_reenc = urllib.parse.quote(path_norm, safe="/:@!$&'()*+,;=-")

        normalized = urllib.parse.urlunparse(
            (
                parsed.scheme.lower(),
                netloc_reenc,
                path_reenc,
                urllib.parse.quote(urllib.parse.unquote(parsed.params), safe=";="),
                urllib.parse.quote(urllib.parse.unquote(parsed.query), safe="=&+"),
                urllib.parse.quote(urllib.parse.unquote(parsed.fragment), safe=""),
            )
        )
        # Reject URLs that have no scheme or no host after normalization
        reparsed = urllib.parse.urlparse(normalized)
        if reparsed.scheme not in ("http", "https") or not reparsed.netloc:
            return None
        return normalized
    except Exception:
        return None


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

        # Also harvest URLs from exa_search / search_web / web_search
        # result text (lines starting with "   URL: ").
        if ToolMessage is not None and isinstance(msg, ToolMessage):
            name = getattr(msg, "name", "")
            if name in ("exa_search", "exa_find_similar", "search_web", "web_search"):
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

    # Normalize + deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for u in urls:
        normalized = _normalize_url(u)
        if normalized is None:
            continue
        if normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)

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

    _URL_BATCH = 10
    _dropped_count = max(0, len(urls) - _URL_BATCH)
    if _dropped_count:
        log.warning(
            "run_research_delegate: %d URLs provided, processing first %d; %d dropped",
            len(urls),
            _URL_BATCH,
            _dropped_count,
        )

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
    url_list = "\n".join(f"  - {u}" for u in urls[:_URL_BATCH])
    _truncation_note = (
        f"\n\n**Note:** {_dropped_count} URL(s) were omitted due to the "
        f"batch limit ({_URL_BATCH}). Research covers only the first {_URL_BATCH} URLs."
        if _dropped_count
        else ""
    )
    research_prompt = (
        f"## Research task\n\n{task}{_truncation_note}\n\n"
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
        "\n"
        "## Source Diversity Requirements\n\n"
        "7. Identify the domain/origin of each source (e.g., github.com, wikipedia.org, arxiv.org).\n"
        "8. Count unique origins, not just source count.\n"
        "9. If 3+ sources share the same origin, treat them as a single piece of evidence.\n"
        "10. Explicitly ask: 'What would disprove this claim?'.\n"
        "11. Report source diversity metrics in your analysis.\n"
        "12. If diversity is low (< 0.5), indicate uncertainty in your findings.\n"
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

        llm = create_delegate_llm(
            prov,
            mdl,
            temperature=0.3,
            num_ctx=alias_cfg.get("context_window") or alias_cfg.get("num_ctx"),
        )
        response_text = run_delegate_agent(llm, research_prompt, "", tools_override=delegate_tools)

        # Source diversity tracking (M6.1)
        from src.orchestration.research_delegate import SourceTracker

        source_tracker = SourceTracker()
        for idx, url in enumerate(urls[:_URL_BATCH]):
            source_tracker.add_source(
                source_id=f"source_{idx}",
                url=url,
                content="",  # Domain-based origin only; per-URL content not available here
            )
        diversity_score = source_tracker.diversity_score()
        dominant_ratio = source_tracker.dominant_origin_ratio()
        log.info(
            "Research diversity: score=%.2f, dominant_ratio=%.2f", diversity_score, dominant_ratio
        )

        if response_text.strip():
            log.info("Research delegate returned %d chars", len(response_text))
            return response_text
        else:
            log.warning("Research delegate returned empty response")
            return ""

    except UserCancelledRun:
        raise
    except Exception as exc:  # noqa: BLE001
        log.warning("Research delegate error: %s", exc, exc_info=True)
        return ""


def force_deep_think(
    user_input: str,
    agent_response: str,
    tool_outputs: str,
    log: Any,
    *,
    research_context: str | None = None,
    llm: Any = None,
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

    import secrets as _secrets

    _dt_nonce = _secrets.token_hex(8)
    safe_task = f"<user_input_{_dt_nonce}>{task}</user_input_{_dt_nonce}>"

    try:
        result = deep_think(
            task=safe_task,
            context=full_context,
            max_iterations=3,
            num_branches=3,
            beam_width=2,
            llm=llm,
        )
        if result and result.strip():
            return result
        log.warning("deep_think returned empty result, using agent response")
        return agent_response
    except UserCancelledRun:
        raise
    except Exception as e:  # noqa: BLE001 — must not crash
        log.warning("Programmatic deep_think failed: %s", e, exc_info=True)
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
    config: AgentRunConfig,
) -> tuple[str, list]:
    """Feed the analysis back to the agent with an explicit 'execute now' prompt.

    Returns ``(output_text, agent_messages)`` from the execution pass.
    If the execution pass fails or produces nothing, returns ``("", [])``.
    """
    from src.orchestration.runner import run_agent

    log = get_logger()
    log.info("Running execution phase — feeding analysis back to agent for action")

    # Truncate analysis if very long to leave room for tool work.
    max_analysis = 12_000
    if len(analysis) > max_analysis:
        analysis = analysis[:max_analysis] + "\n\n[... analysis truncated for brevity ...]"

    _nonce = secrets.token_hex(8)
    exec_prompt = (
        "You have just completed a thorough analysis. "
        "Now EXECUTE the plan — create every file, make every change.\n\n"
        "RULES:\n"
        "• Call write_file (or append_file) for EACH file that needs to be "
        "created or modified. Do NOT just describe them — actually create them.\n"
        "• Work through the plan systematically: create files one at a time.\n"
        "• After creating all files, briefly confirm what was done.\n\n"
        f"## Original request\n[{_nonce}]{original_prompt}[/{_nonce}]\n\n"
        f"## Analysis / plan\n{analysis}"
    )

    exec_msgs: list = []
    try:
        exec_config = copy(config)
        if exec_config.available_tools is not None:
            exec_config.available_tools = dict(exec_config.available_tools)
        if exec_config.active_tools_list is not None:
            exec_config.active_tools_list = list(exec_config.active_tools_list)
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
    # Check for negation: "I don't need more steps" should NOT trigger recovery
    _NEGATION_RE = re.compile(
        r"\b(?:not|no|never|don'?t|doesn'?t|can'?t|cannot|without)\b"
        r".{0,20}(?:need|require|want)",
        re.IGNORECASE,
    )
    for phrase in STEP_LIMIT_PHRASES:
        if phrase in lower:
            # Find phrase position and check for negation in the surrounding context
            pos = lower.find(phrase)
            window = lower[max(0, pos - 30) : pos + len(phrase) + 10]
            if _NEGATION_RE.search(window):
                continue  # negated — not a step-limit apology
            return True
    return False


# Foreign tool-call formats that some models (notably qwen3-coder) emit
# inside AIMessage content rather than via structured tool_calls.
# These slip past the LangChain tool-call extractor and render as raw
# text in the user-facing UI unless we strip them.
_FOREIGN_TOOL_CALL_PATTERNS = (
    # Qwen3 XML format: <tool_call><function=name(args)></function></tool_call>
    re.compile(r"<tool_call>\s*<function=[^>]*?>\s*</?function>\s*</tool_call>", re.DOTALL),
    # Qwen3 with newlines and arguments
    re.compile(
        r"<tool_call>\s*<function=[^>]*?>.*?</function>\s*</tool_call>", re.DOTALL | re.IGNORECASE
    ),
    # Bare <function=...></function> without enclosing <tool_call>
    re.compile(r"<function=[^>]*?>\s*</function>", re.DOTALL),
)


# DeepSeek's native chat template emits tool calls using special tokens
# embedded in the assistant content, rather than via the structured
# ``tool_calls`` field that langchain-openai parses.  Seen specifically
# when DeepSeek-V3 is routed through OpenRouter — the OpenAI-compatible
# wrapper does not normalise the response into structured tool_calls.
#
# Format:
#   <｜tool▁calls▁begin｜>
#     <｜tool▁call▁begin｜>function<｜tool▁sep｜>{tool_name}
#     ```json
#     {arg_json}
#     ```
#     <｜tool▁call▁end｜>
#     ... more tool calls ...
#   <｜tool▁calls▁end｜>
#
# The unicode characters used:
#   ｜ — FULLWIDTH VERTICAL LINE (U+FF5C)
#   ▁ — LOWER ONE EIGHTH BLOCK (U+2581)
_DEEPSEEK_TOOL_CALL_RE = re.compile(
    r"<｜tool▁call▁begin｜>"  # <｜tool▁call▁begin｜>
    r"\s*(\w+)"  # tool kind (usually "function")
    r"<｜tool▁sep｜>"  # <｜tool▁sep｜>
    r"\s*([\w.-]+)"  # tool name
    r"\s*```(?:[\w-]*)\s*"  # ```json (or other lang) opening fence
    r"([\s\S]*?)"  # JSON args (non-greedy)
    r"```\s*"  # closing fence
    r"<｜tool▁call▁end｜>",  # <｜tool▁call▁end｜>
    re.UNICODE,
)
_DEEPSEEK_TOOL_CALLS_WRAPPER_RE = re.compile(
    r"<｜tool▁calls▁(?:begin|end)｜>",
    re.UNICODE,
)


def extract_deepseek_native_tool_calls(content: Any) -> tuple[list[dict], Any]:
    """Parse DeepSeek native special-token tool calls from message content.

    DeepSeek-V3 (notably when routed through OpenRouter) emits tool calls
    as special tokens in the assistant content stream rather than via the
    structured ``tool_calls`` field.  langchain-openai does not parse this
    format, so the calls are silently dropped — the agent thinks the model
    answered in prose when it actually intended to invoke tools.

    Returns
    -------
    tuple[list[dict], str]
        ``(extracted_tool_calls, content_with_tokens_removed)`` —
        each tool_call has the standard LangChain shape:
        ``{"name": str, "args": dict, "id": str, "type": "tool_call"}``.
        Returns ``([], content)`` unchanged for non-string input or when
        no DeepSeek tokens are present.
    """
    if not isinstance(content, str) or "｜tool▁call" not in content:
        return [], content

    extracted: list[dict] = []
    for idx, match in enumerate(_DEEPSEEK_TOOL_CALL_RE.finditer(content)):
        kind = match.group(1)
        if kind != "function":
            continue
        name = match.group(2)
        args_text = match.group(3).strip()
        try:
            import json

            args = json.loads(args_text) if args_text else {}
        except (json.JSONDecodeError, ValueError):
            # Malformed args — skip rather than raise; the agent will see
            # zero tool calls and respond accordingly, which is at least
            # consistent with structured-tool-call failure modes.
            continue
        if not isinstance(args, dict):
            continue
        extracted.append(
            {
                "name": name,
                "args": args,
                "id": f"deepseek_{idx}_{abs(hash(args_text)) & 0xFFFFFFFF:08x}",
                "type": "tool_call",
            }
        )

    cleaned = _DEEPSEEK_TOOL_CALL_RE.sub("", content)
    cleaned = _DEEPSEEK_TOOL_CALLS_WRAPPER_RE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return extracted, cleaned


def normalize_native_tool_calls(message: Any) -> Any:
    """Pull native tool-call tokens out of message content into structured form.

    Composes ``extract_deepseek_native_tool_calls`` (extract+strip) with
    ``strip_foreign_tool_call_xml`` (strip-only, for qwen3 XML).  When the
    message has no recognisable native tokens, returns it unchanged.

    Otherwise returns a new ``AIMessage`` with:
      • ``content``: tokens removed
      • ``tool_calls``: existing tool_calls + any extracted ones
      • other fields preserved (id, response_metadata, additional_kwargs)
    """
    try:
        from langchain_core.messages import AIMessage
    except ImportError:
        return message
    if not isinstance(message, AIMessage):
        return message
    content = getattr(message, "content", "")
    if not isinstance(content, str) or not content:
        return message

    extracted_calls, after_deepseek = extract_deepseek_native_tool_calls(content)
    after_xml = strip_foreign_tool_call_xml(after_deepseek)
    final_content = after_xml if isinstance(after_xml, str) else after_deepseek

    if not extracted_calls and final_content == content:
        return message

    existing_calls = list(getattr(message, "tool_calls", None) or [])
    return AIMessage(
        content=final_content,
        tool_calls=existing_calls + extracted_calls,
        id=getattr(message, "id", None),
        response_metadata=getattr(message, "response_metadata", None) or {},
        additional_kwargs=getattr(message, "additional_kwargs", None) or {},
    )


def strip_foreign_tool_call_xml(content: Any) -> Any:
    """Strip Qwen3-style XML tool-call markup from message content.

    Some models (qwen3-coder seen in the wild) emit tool calls in their
    content stream using ``<tool_call><function=name(args)></function></tool_call>``
    XML rather than the structured ``tool_calls`` field. Cogtrix uses
    LangChain's structured tool-call API exclusively, so this XML never
    executes — but it does render as raw text in the user UI unless
    stripped here.

    Returns the input unchanged for non-string content.
    """
    if not isinstance(content, str):
        return content
    cleaned = content
    for pattern in _FOREIGN_TOOL_CALL_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    # Tidy up multiple blank lines left behind by the strip
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _extract_checkpoint_findings(messages: list) -> list[str]:
    """Collect ``finding`` arguments from successful checkpoint tool calls.

    Checkpoints are the highest-signal artifacts the agent produces — when
    the user prompt finishes via a give-up path, the checkpoint findings
    are usually the actual answer to the question.  Returned in the order
    they were recorded.
    """
    try:
        from langchain_core.messages import AIMessage
    except ImportError:
        return []

    findings: list[str] = []
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        for tc in getattr(msg, "tool_calls", None) or []:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if name != "checkpoint":
                continue
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
            if not isinstance(args, dict):
                continue
            finding = args.get("finding")
            if isinstance(finding, str) and finding.strip():
                findings.append(finding.strip())
    return findings


def synthesize_answer_from_state(messages: list) -> str | None:
    """Build a clean user-facing answer from accumulated agent state.

    Used when the graph exits via a give-up branch (action-intent or
    phantom-tool-call retries exhausted) and we don't want the user to see
    the model's last stuck-thinking output.

    Priority cascade:
    1. Checkpoint findings — the agent explicitly summarised what it learned.
    2. Tool-result snippets via ``build_tool_results_response``.
    3. The last AI content with foreign tool-call XML stripped, if it's
       substantive (>80 chars after stripping).

    Returns ``None`` when nothing usable can be synthesized.
    """
    findings = _extract_checkpoint_findings(messages)
    if findings:
        if len(findings) == 1:
            return findings[0]
        # Latest checkpoint usually subsumes earlier ones; prefer it but
        # mention the count so power users know there were multiple.
        return findings[-1]

    try:
        from src.orchestration.runner import build_tool_results_response

        tool_response = build_tool_results_response({"messages": messages})
        if tool_response:
            return tool_response
    except Exception:  # noqa: BLE001 — synthesis must not crash
        pass

    try:
        from langchain_core.messages import AIMessage
    except ImportError:
        return None

    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        content = getattr(msg, "content", "")
        if not isinstance(content, str) or not content.strip():
            continue
        cleaned = strip_foreign_tool_call_xml(content)
        # 30 chars is enough to recognise a real answer ("Yes — Mattermost
        # supports OAuth"), short enough to filter out "OK" / "Done" stubs
        # and the empty leftovers from XML-only responses.
        if isinstance(cleaned, str) and len(cleaned) > 30:
            return cleaned
    return None


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

    # NOTE: The output deliberately avoids error prefixes so that
    # _is_valid_response() returns True and the turn is saved to
    # history.  This is essential for the "Ralph Loop" pattern where
    # the agent iterates on a complex task across multiple turns —
    # discarding partial progress would force it to restart from
    # scratch every time.
    parts = ["The task could not be completed in one pass. Here is the progress so far:\n\n"]

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
        log.warning("Recovery retry failed: %s", retry_err)

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
    #    error prefixes to pass _is_valid_response(). ──
    log.error("All recovery attempts failed — no usable content")
    return (
        "I was unable to complete this task in the allotted steps. "
        "The query may require many tool calls.\n\n"
        "You can say **continue** and I will pick up where I left off, "
        "or you can rephrase / break the question into smaller parts."
    )
