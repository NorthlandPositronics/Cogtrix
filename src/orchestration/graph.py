"""LangGraph agent graph for Cogtrix.

Builds a custom StateGraph with three nodes:
- call_model: binds active tools to LLM and invokes it
- process_tools: executes tool calls, handles fuzzy matching and expansion
- handle_phantom: recovers from phantom tool calls
"""

import atexit
import concurrent.futures
import json as _json
import re
import threading
import time
import types
import typing
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from difflib import SequenceMatcher
from typing import Any

from opentelemetry.trace import Status, StatusCode

from src.agent.core import CogtrixState
from src.agent.safety import UserCancelledRun
from src.api.telemetry import start_span
from src.logging_config import get_logger
from src.orchestration.compression import (
    _CHARS_PER_TOKEN,
    _EMERGENCY_THRESHOLD_RATIO,
    _MID_TURN_COMPRESSION_THRESHOLD,
    COMPRESSION_MIN_AGE_CYCLES,
    COMPRESSION_MIN_CHARS,
    _content_len,
    apply_message_compression,
    truncate_tool_output,
)
from src.orchestration.nodes.process_tools import build_process_tools_node
from src.orchestration.nodes.recovery import (
    build_handle_action_intent_node,
    build_handle_phantom_node,
)
from src.orchestration.run_config import AgentRunConfig
from src.orchestration.session_state import SessionState
from src.registry import LazyToolProxy as _LazyToolProxy
from src.tools.configure import (
    TOOL_OUTPUT_CAP_MIN_CHARS,
    build_tool_catalog,
    compute_tool_output_cap,
)

DEFAULT_RECURSION_LIMIT = 90
EMPTY_RESPONSE_MSG = "**Error:** The model returned an empty response. Please try again."
_PARALLEL_TOOL_WORKERS = 8
_HISTORY_TOOL_MESSAGE_CAP_CHARS = 30_000

_TOOL_EXECUTOR: concurrent.futures.ThreadPoolExecutor | None = None
_TOOL_EXECUTOR_LOCK = threading.Lock()

_LLM_EXECUTOR: concurrent.futures.ThreadPoolExecutor | None = None
_LLM_EXECUTOR_LOCK = threading.Lock()
_LLM_EXECUTOR_WORKERS = 4


def _get_tool_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Return the module-level parallel tool executor, creating it on first use."""
    global _TOOL_EXECUTOR
    if _TOOL_EXECUTOR is None:
        with _TOOL_EXECUTOR_LOCK:
            if _TOOL_EXECUTOR is None:
                _TOOL_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                    max_workers=_PARALLEL_TOOL_WORKERS,
                    thread_name_prefix="tool",
                )
                atexit.register(_TOOL_EXECUTOR.shutdown, wait=False, cancel_futures=True)
    return _TOOL_EXECUTOR


def _get_llm_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Return the module-level LLM executor, creating it on first use.

    Use a shared bounded pool instead of creating a fresh
    ThreadPoolExecutor per LLM call to avoid thread leakage when
    calls time out and the underlying OS thread stays blocked in I/O.
    """
    global _LLM_EXECUTOR
    if _LLM_EXECUTOR is None:
        with _LLM_EXECUTOR_LOCK:
            if _LLM_EXECUTOR is None:
                _LLM_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                    max_workers=_LLM_EXECUTOR_WORKERS,
                    thread_name_prefix="llm",
                )
                atexit.register(_LLM_EXECUTOR.shutdown, wait=False, cancel_futures=True)
    return _LLM_EXECUTOR


def _extract_llm_labels(llm: Any) -> tuple[str, str]:
    """Extract provider and model labels from a LangChain LLM instance.

    Falls back to ``"unknown"`` when the LLM object does not expose the
    expected attributes.
    """
    if llm is None:
        return "unknown", "unknown"

    # Model name — try common attribute names across LangChain providers.
    model = getattr(llm, "model_name", None) or getattr(llm, "model", None) or ""
    if not model:
        _ident = getattr(llm, "_identifying_params", None) or {}
        model = _ident.get("model_name") or _ident.get("model") or "unknown"

    # Provider — normalize from _llm_type or class name.
    _llm_type = getattr(llm, "_llm_type", None)
    if _llm_type:
        provider = _llm_type.lower().replace("chat-", "").replace("-chat", "")
    else:
        cls_name = type(llm).__name__.lower()
        if "openai" in cls_name:
            provider = "openai"
        elif "anthropic" in cls_name:
            provider = "anthropic"
        elif "google" in cls_name:
            provider = "google"
        elif "ollama" in cls_name:
            provider = "ollama"
        elif "deepseek" in cls_name:
            provider = "deepseek"
        elif "xai" in cls_name:
            provider = "xai"
        else:
            provider = "unknown"

    return provider, model


_INVALID_TOOL_RE = re.compile(r"^Error:\s*(\S+)\s+is not a valid tool")

_CONTEXT_OVERFLOW_PATTERNS = (
    "context_length_exceeded",
    "context window",
    "too long",
    "maximum context",
    "reduce the length",
    "input is too long",
    "prompt is too long",
)


def _is_context_overflow_error(exc: Exception) -> bool:
    """Return True if *exc* is a provider context-length rejection.

    Checks structured provider error fields first (OpenAI ``error.code``,
    HTTP status codes, Anthropic ``type``), then falls back to string
    matching against ``_CONTEXT_OVERFLOW_PATTERNS``.
    """
    # Structured checks — faster and language-independent
    # OpenAI / OpenAI-compatible: BadRequestError with code field
    code = getattr(exc, "code", None) or getattr(getattr(exc, "error", None), "code", None)
    if code and "context_length" in str(code).lower():
        return True
    # HTTP status 400 paired with overflow-related type/param
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status == 400:
        etype = getattr(exc, "type", None) or getattr(getattr(exc, "error", None), "type", None)
        if etype and any(p in str(etype).lower() for p in ("context", "length", "token")):
            return True
    # String fallback — covers providers that embed the message in the exc string
    msg = str(exc).lower()
    return any(p in msg for p in _CONTEXT_OVERFLOW_PATTERNS)


def _stable_tool_call_value(value: Any, seen: set[int] | None = None) -> Any:
    """Return a deterministic JSON-safe representation for tool-call args."""
    if seen is None:
        seen = set()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"__type__": "builtins.bytes", "__value__": value.hex()}
    if isinstance(value, bytearray):
        return {"__type__": "builtins.bytearray", "__value__": bytes(value).hex()}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
            "__state__": _stable_tool_call_value(asdict(value), seen),
        }

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
        except Exception:
            pass
        else:
            return {
                "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
                "__state__": _stable_tool_call_value(dumped, seen),
            }

    obj_id = id(value)
    if obj_id in seen:
        return {
            "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
            "__cycle__": True,
        }

    if isinstance(value, dict):
        seen.add(obj_id)
        try:
            return {
                str(key): _stable_tool_call_value(value[key], seen)
                for key in sorted(value, key=lambda key: str(key))
            }
        finally:
            seen.discard(obj_id)

    if isinstance(value, list):
        seen.add(obj_id)
        try:
            return [_stable_tool_call_value(item, seen) for item in value]
        finally:
            seen.discard(obj_id)

    if isinstance(value, tuple):
        seen.add(obj_id)
        try:
            return {
                "__type__": "builtins.tuple",
                "__items__": [_stable_tool_call_value(item, seen) for item in value],
            }
        finally:
            seen.discard(obj_id)

    if isinstance(value, (set, frozenset)):
        seen.add(obj_id)
        try:
            items = [_stable_tool_call_value(item, seen) for item in value]
            items.sort(key=lambda item: _json.dumps(item, sort_keys=True, separators=(",", ":")))
            return {
                "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
                "__items__": items,
            }
        finally:
            seen.discard(obj_id)

    if hasattr(value, "__dict__"):
        seen.add(obj_id)
        try:
            return {
                "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
                "__state__": {
                    key: _stable_tool_call_value(attr, seen)
                    for key, attr in sorted(vars(value).items())
                    if not key.startswith("__")
                },
            }
        finally:
            seen.discard(obj_id)

    slots = getattr(value, "__slots__", None)
    if slots:
        seen.add(obj_id)
        try:
            state: dict[str, Any] = {}
            slot_names = (slots,) if isinstance(slots, str) else tuple(slots)
            for slot in slot_names:
                if hasattr(value, slot):
                    state[slot] = _stable_tool_call_value(getattr(value, slot), seen)
            return {
                "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
                "__state__": state,
            }
        finally:
            seen.discard(obj_id)

    return {
        "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
        "__value__": str(value),
    }


def _apply_context_budget_guard(
    response: Any,
    *,
    max_context_tokens: int | None,
    tool_context_limit_pct: float,
) -> Any:
    """Return a warning AIMessage when tool calls exceed the turn budget."""
    if not max_context_tokens or not getattr(response, "tool_calls", None):
        return response

    um = getattr(response, "usage_metadata", None)
    if not um or not isinstance(um, dict):
        return response

    turn_input = um.get("input_tokens", 0)
    if not isinstance(turn_input, int) or turn_input <= max_context_tokens * tool_context_limit_pct:
        return response

    from langchain_core.messages import AIMessage

    pct_used = int(turn_input * 100 / max_context_tokens)
    warning = (
        f"[Context budget reached — {pct_used}% of {max_context_tokens:,} "
        f"tokens used this turn (limit: {int(tool_context_limit_pct * 100)}%). "
        "Tool execution halted. Summarising based on available information.]"
    )
    return AIMessage(
        content=warning,
        id=getattr(response, "id", None),
        response_metadata={"budget_guard": True},
    )


_TOPIC_SWITCH_MESSAGE_WINDOW = 8
_TOPIC_SWITCH_MAX_WORDS = 15
_TOPIC_SWITCH_MIN_SIMILARITY = 0.40
_TOPIC_SWITCH_NUDGE = (
    "The user has changed topic. Answer the new question directly without reference "
    "to the prior task."
)
_TOPIC_SWITCH_STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "at",
    "be",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "our",
    "please",
    "that",
    "the",
    "their",
    "this",
    "to",
    "what",
    "what's",
    "whats",
    "with",
    "you",
    "your",
}


def _topic_switch_tokens(text: str) -> list[str]:
    """Return normalized content tokens used by the topic-switch heuristic."""
    normalized = text.lower().replace("'s", "")
    return [
        token
        for token in re.findall(r"[a-z0-9]+", normalized)
        if len(token) > 2 and token not in _TOPIC_SWITCH_STOPWORDS
    ]


def _should_reset_summary_for_topic_switch(messages: list[Any]) -> bool:
    """Return True when the latest user message appears to switch topics."""
    if not messages:
        return False

    last_human_idx = -1
    last_human_text = ""
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if getattr(msg, "type", None) == "human":
            last_human_idx = idx
            content = getattr(msg, "content", "")
            last_human_text = content if isinstance(content, str) else ""
            break

    if last_human_idx <= 0 or not last_human_text:
        return False

    if len(last_human_text.split()) >= _TOPIC_SWITCH_MAX_WORDS:
        return False

    current_tokens = _topic_switch_tokens(last_human_text)
    if not current_tokens:
        return False

    reference_messages = messages[
        max(0, last_human_idx - _TOPIC_SWITCH_MESSAGE_WINDOW) : last_human_idx
    ]
    reference_tokens: list[str] = []
    for msg in reference_messages:
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content:
            reference_tokens.extend(_topic_switch_tokens(content))

    if not reference_tokens:
        return False

    current_counts = Counter(current_tokens)
    reference_counts = Counter(reference_tokens)
    overlap = sum(min(count, reference_counts[token]) for token, count in current_counts.items())
    similarity = overlap / max(len(current_tokens), len(reference_tokens))

    has_question = "?" in last_human_text
    if has_question:
        return overlap <= 1 and similarity < _TOPIC_SWITCH_MIN_SIMILARITY

    # Imperative commands ("check slack", "look at github") are valid topic
    # switches even without a question mark.  Use a stricter threshold —
    # zero overlap and at least 2 meaningful tokens — to reduce false positives
    # on short continuations like "okay proceed".
    return overlap == 0 and len(current_tokens) >= 2


def _infer_llm_provider_name(llm: Any) -> str:
    """Infer a stable provider label for telemetry."""
    for attr in ("provider", "provider_name", "_provider_name"):
        value = getattr(llm, attr, None)
        if isinstance(value, str) and value:
            return value

    module = getattr(llm.__class__, "__module__", "").lower()
    if "openai" in module:
        return "openai"
    if "anthropic" in module:
        return "anthropic"
    if "ollama" in module:
        return "ollama"
    if "google" in module or "genai" in module:
        return "google"
    return llm.__class__.__name__.lower()


def _infer_llm_model_name(llm: Any) -> str:
    """Infer the model identifier for telemetry."""
    for attr in ("model", "model_name", "model_id", "model_name_or_path"):
        value = getattr(llm, attr, None)
        if isinstance(value, str) and value:
            return value
    kwargs = getattr(llm, "_default_params", None)
    if isinstance(kwargs, dict):
        for key in ("model", "model_name", "model_id"):
            value = kwargs.get(key)
            if isinstance(value, str) and value:
                return value
    return llm.__class__.__name__


# ── Action-intent detection ───────────────────────────────────────────────────
# Catches "I'll create X" / "Let me write Y" responses that contain no tool
# calls — the model expressed intent but didn't act on it.

# Phrases that introduce a planned future action
_INTENT_LEAD_RE = re.compile(
    r"\b(?:"
    # First-person modal / future forms
    r"I(?:"
    r"'ll\b"  # I'll
    r"|\s+will\b"  # I will
    r"|\s+am\s+going\s+to\b"  # I am going to
    r"|'m\s+going\s+to\b"  # I'm going to
    r"|'m\s+about\s+to\b"  # I'm about to
    r"|\s+need\s+to\b"  # I need to
    r"|\s+have\s+to\b"  # I have to
    r"|\s+should\b"  # I should
    r"|\s+must\b"  # I must
    r"|\s+will\s+now\b"  # I will now
    r"|'ll\s+now\b"  # I'll now
    r"|'m\s+now\b"  # I'm now [verb-ing]
    r"|'ll\s+(?:proceed|go\s+ahead|start|begin|continue)\b"  # I'll proceed/...
    r")"
    # Imperative / collaborative
    r"|Let(?:'s|\s+(?:me|us))\b"  # Let me / Let's / Let us
    # Temporal modifiers + intent
    r"|Now\s+(?:I(?:'ll\b|\s+will\b)|let\s+me\b)"  # Now I'll / Now let me
    r"|(?:First|Next|Then|Finally|Additionally|Also)(?:\s*,)?\s+"
    r"(?:I(?:'ll\b|\s+will\b)|let\s+me\b)"  # First/Next/Then ... I'll
    # Implicit future
    r"|(?:Going|About)\s+to\b"  # Going to / About to
    r"|Time\s+to\b"  # Time to
    r"|I\s+can\s+now\b"  # I can now
    r")",
    re.IGNORECASE,
)

# Verb stems that indicate a tool-requiring operation.
# Long stems (5+ chars) use \w{0,8} for inflection coverage.
# Short/ambiguous stems use explicit endings to prevent overmatch.
_TOOL_VERB_RE = re.compile(
    r"\b(?:"
    # File / content operations — long stems
    r"creat\w{0,8}|generat\w{0,8}|overwrite\w{0,8}|append\w{0,8}"
    r"|delet\w{0,8}|remov\w{0,8}|replac\w{0,8}|insert\w{0,8}|rename\w{0,8}"
    r"|modif\w{0,8}|patch\w{0,8}|refactor\w{0,8}"
    # Read / fetch
    r"|fetch\w{0,8}|retriev\w{0,8}|download\w{0,8}" r"|search\w{0,8}|crawl\w{0,8}|scrape\w{0,8}"
    # Build / execution
    r"|build\w{0,8}|compil\w{0,8}|execut\w{0,8}|launch\w{0,8}"
    r"|install\w{0,8}|deploy\w{0,8}|configur\w{0,8}|initializ\w{0,8}"
    r"|bootstrap\w{0,8}|provision\w{0,8}|scaffold\w{0,8}"
    # Code / dev
    r"|implement\w{0,8}|develop\w{0,8}|debug\w{0,8}|resolv\w{0,8}"
    # Network / API
    r"|upload\w{0,8}|commit\w{0,8}|invok\w{0,8}|submit\w{0,8}|request\w{0,8}"
    r"|publish\w{0,8}|broadcast\w{0,8}"
    # Data processing
    r"|transform\w{0,8}|extract\w{0,8}|analyz\w{0,8}|analys\w{0,8}"
    r"|comput\w{0,8}|calculat\w{0,8}|process\w{0,8}|pars\w{0,8}"
    r"|export\w{0,8}|migrat\w{0,8}|convert\w{0,8}|classif\w{0,8}"
    # Infra
    r"|register\w{0,8}|connect\w{0,8}|verif\w{0,8}|inspect\w{0,8}|examin\w{0,8}" r"|clone\w{0,8}"
    # Short stems — explicit inflections only
    r"|writ(?:e|es|ing|ten)\b"
    r"|read(?:s|ing)?\b"
    r"|open(?:s|ing|ed)?\b"
    r"|load(?:s|ing|ed)?\b"
    r"|run(?:s|ning)?\b"
    r"|start(?:s|ing|ed)?\b"
    r"|send(?:s|ing)?\b|sent\b"
    r"|post(?:s|ing|ed)?\b"
    r"|call(?:s|ing|ed)?\b"
    r"|pull(?:s|ing|ed)?\b"
    r"|push(?:es|ing|ed)?\b"
    r"|fix(?:es|ing|ed)?\b"
    r"|test(?:s|ing|ed)?\b"
    r"|edit(?:s|ing|ed)?\b"
    r"|update(?:s|d|ing)?\b"
    r"|sav(?:e|es|ing|ed)\b"
    r"|stor(?:e|es|ing|ed)\b"
    r"|add(?:s|ing|ed)?\b"
    r"|cod(?:e|es|ing|ed)\b"
    r"|defin(?:e|es|ing|ed)\b"
    r"|list(?:s|ing|ed)?\b"
    r"|output(?:s|ting|ted)?\b"
    r"|check(?:s|ing|ed)?\b"
    # Multi-word constructions
    r"|set\s+up\b" r"|spin\s+up\b" r"|look\s+up\b" r"|wire\s+up\b" r"|stand\s+up\b" r")\b",
    re.IGNORECASE,
)


# Phrases that LOOK like intent leads but are actually conversational.
# "let me know" is the most common false positive.
_INTENT_FALSE_POSITIVE_RE = re.compile(
    r"""
    \b(?:
        let\s+me\s+know\b                   # "let me know if..."
      | let\s+me\s+(?:explain|clarify|describe|summarize|outline|walk\s+you\s+through)\b
      | I\s+(?:want\s+to|need\s+to|would\s+like\s+to)\s+(?:mention|note|clarify|explain|point\s+out)\b
      | I\s+should\s+(?:mention|note|clarify|point\s+out)\b
      | please\s+(?:note|be\s+aware|keep\s+in\s+mind)\b
      | (?:^|(?:please|just|do)\s+)note\s+that\b    # "note that" / "please note that"
      | (?:^|(?:please|just)\s+)keep\s+in\s+mind\b  # sentence-initial or after polite prefix
      | (?:it(?:'s|\s+is)\s+)?worth\s+(?:noting|mentioning)\b
      | it(?:'s|\s+is)\s+(?:worth|important\s+to)\s+(?:note|mention)\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _is_action_intent(message: Any) -> bool:
    """Return True when the model describes a planned action but emits no tool calls.

    Checks for an intent-lead phrase (``I'll``, ``Let me``, ``Going to``, etc.)
    paired with a tool-action verb (``create``, ``run``, ``fetch``, etc.)
    **in the same sentence**.  The sentence-locality requirement prevents false
    positives where the lead phrase appears in a closing remark (e.g. "Feel
    free to let me know") and the verb appears elsewhere (e.g. "reading" in a
    weather table).
    """
    if getattr(message, "tool_calls", None):
        return False
    content = getattr(message, "content", "")
    if not isinstance(content, str):
        return False
    text = content.strip()
    if not text:
        return False

    # Split into sentences (on . ! ? or newlines) and check each independently.
    # This prevents a verb in one sentence from pairing with an intent lead
    # in a completely different sentence.
    sentences = re.split(r"[.!?\n]", text)
    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue
        has_intent = bool(_INTENT_LEAD_RE.search(s))
        has_verb = bool(_TOOL_VERB_RE.search(s))
        if not (has_intent and has_verb):
            continue
        # A real intent+verb pair exists in this sentence.  Only suppress it
        # when the FP phrase is the SOLE intent-shaped construct — i.e. it
        # appears but there is no additional "I'll / Let me / Going to" lead-in
        # that could pair with the verb.  This prevents "please note that I'll
        # run the build" from being silently dropped.  The original "let me
        # know" case still suppresses correctly because "let me know" fires the
        # FP regex AND does not typically combine with an action verb in the
        # same sentence.
        fp_match = _INTENT_FALSE_POSITIVE_RE.search(s)
        if fp_match:
            # Re-check: is the intent lead-in ONLY the FP phrase itself, or is
            # there a genuine action lead-in beyond the FP match?
            text_after_fp = s[fp_match.end() :]
            text_before_fp = s[: fp_match.start()]
            if not (
                _INTENT_LEAD_RE.search(text_after_fp) or _INTENT_LEAD_RE.search(text_before_fp)
            ):
                continue  # FP phrase is the only lead-in — suppress
        return True
    return False


# Phrases that signal a multi-step task is incomplete — the model used
# sequential language ("first", "to start") but stopped before finishing.
# Only fires when paired with an action-intent detection (intent lead +
# tool verb in the same sentence), which guards against false positives
# from purely conversational uses of these words.
_INCOMPLETENESS_SIGNAL_RE = re.compile(
    r"\b(?:"
    r"first\b"  # "create the PO first" → there's a second step
    r"|to\s+start\b"  # "to start, let me..."
    r"|to\s+begin\b"  # "to begin with..."
    r"|initially\b"  # "initially create the PO"
    r"|step\s*1\b"  # "step 1: create the PO"
    r")",
    re.IGNORECASE,
)


def _has_incompleteness_signal(text: str) -> bool:
    """Return True when the text signals an incomplete multi-step operation.

    Detects language that implies the model planned more steps but
    stopped early — e.g. "first", "to start", "step 1".  Only relevant
    when ``_is_action_intent`` has already identified an intent-lead
    + tool-verb pair in the same sentence; this function narrows the
    nudge from generic to specific.
    """
    if not text or not isinstance(text, str):
        return False
    # Restrict to the sentence containing the incompleteness signal
    # so a "first" in a generic intro doesn't contaminate a later sentence.
    sentences = re.split(r"[.!?\n]", text)
    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue
        if _INCOMPLETENESS_SIGNAL_RE.search(s) and _INTENT_LEAD_RE.search(s):
            return True
    return False


# Past-tense action verbs at the start of numbered or bulleted list items.
# Used to detect hallucinated-completion summaries where the model claims
# "1. Notified the VP..." without actually calling notify_approver.
_PAST_TENSE_LIST_VERB_RE = re.compile(
    r"^\s*(?:\d+[.)]|[-*•])\s+([A-Z][a-z]{2,}(?:ied|ed))\b",
    re.MULTILINE,
)


def _is_hallucinated_completion(
    message: Any,
    messages: typing.Sequence[Any],
    available_tool_names: list[str],
) -> bool:
    """Return True when the final response claims past-tense completion
    of an action whose corresponding tool was never actually called.

    Pattern observed on gpt-oss-20b for finance_invoice_approval_workflow:

        1. Classified as "medium" tier (amount $12,500).
        2. Routed to the VP-level approval queue.
        3. Notified the VP for review.   ← notify_approver never called

    The standard ``_is_action_intent`` detector matches only future-tense
    intent leads ("I'll notify", "Let me..."); past-tense passive claims
    slip past it.  A weaker/cheaper model that confabulates a completion
    summary without executing every required step would otherwise reach
    the user, fail scenario assertions, and burn the run budget with no
    recovery cycle.

    Heuristic:
    1. Find past-tense verbs at the start of numbered/bulleted list items
       in the final text response.
    2. For each verb, derive a stem by stripping ``ed`` / ``ied``.
    3. If the stem appears in an *available* tool name, AND that tool was
       not called in the conversation, treat as hallucinated completion.

    Stem-matching is intentionally permissive ("notif" matches
    ``notify_approver``); the false-positive ceiling is bounded by the
    "in the available tool list" check — a verb that doesn't correspond
    to any tool the agent could have called never triggers.
    """
    if getattr(message, "tool_calls", None):
        return False
    content = getattr(message, "content", "")
    if not isinstance(content, str) or not content.strip():
        return False

    verbs = _PAST_TENSE_LIST_VERB_RE.findall(content)
    if not verbs:
        return False

    called_names: set[str] = set()
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            name = tc.get("name") if isinstance(tc, dict) else None
            if name:
                called_names.add(name.lower())

    available_lower = [t.lower() for t in available_tool_names]

    for verb in verbs:
        stem = re.sub(r"(?:ied|ed)$", "", verb.lower())
        if len(stem) < 4:
            # Too short to disambiguate ("Read"-ed → "read" is fine but
            # "Set"-? doesn't end in ed/ied so won't match anyway).
            continue
        for tool_name in available_lower:
            if stem in tool_name and tool_name not in called_names:
                return True
    return False


_PHANTOM_TOOL_MARKUP_RE = re.compile(
    r"<\s*(?:function_calls|invoke|Call|antml:function_calls)\b",
    re.IGNORECASE,
)

# JSON array of {"tool": "...", "arguments": {...}} objects emitted as literal text
# instead of structured tool_calls (e.g. model hallucinates tool calls as JSON).
_PHANTOM_JSON_TOOL_RE = re.compile(
    r'"tool"\s*:\s*"[^"\n]{1,120}"',
    re.IGNORECASE,
)

_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:[a-zA-Z0-9_+-]+)?\s*\n(?P<body>.*)\n```\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _unwrap_code_fence(content: str) -> str:
    """Return the body of a fenced code block, or the original content."""
    match = _CODE_FENCE_RE.match(content)
    if match:
        return match.group("body")
    return content


def _looks_like_phantom_tool_markup(message: Any) -> bool:
    """Return True when plain text resembles raw tool-call markup.

    Catches two families:
    - XML-style:  ``<function_calls>``, ``<invoke>``, ``<Call>``, etc.
    - JSON-style: ``[{"tool": "...", "arguments": {...}}]`` arrays in text

    Both should be treated as phantom tool-call responses so the graph can
    recover instead of accepting hallucinated markup as final prose.
    """
    if getattr(message, "tool_calls", None):
        return False
    content = getattr(message, "content", "")
    if not isinstance(content, str):
        return False
    candidate = _unwrap_code_fence(content)
    if _PHANTOM_TOOL_MARKUP_RE.search(candidate):
        return True
    # JSON phantom: require the message to look like raw JSON at the start so
    # prose that merely mentions {"tool": ...} or "arguments" does not trip it.
    stripped = candidate.lstrip()
    if not stripped or stripped[0] not in "[{":
        return False
    return bool(_PHANTOM_JSON_TOOL_RE.search(candidate) and '"arguments"' in candidate)


_MARKDOWN_TABLE_ROW_RE = re.compile(r"^\s*\|.+\|.+\|", re.MULTILINE)
_NUMBERED_SECTION_RE = re.compile(r"^#{1,4}\s+\d+\.", re.MULTILINE)
_SUCCESS_CLAIM_RE = re.compile(
    r"(?:✅|:white_check_mark:|\bsuccess(?:ful|fully)?\b|\b(?:created|completed|finished)\s+successfully\b)",
    re.IGNORECASE,
)
_NEGATED_SUCCESS_RE = re.compile(
    r"\b(?:not|no|failed|unable|cannot|can't|couldn't|didn't)\b.{0,24}"
    r"(?:success(?:ful|fully)?|created|completed|finished)\b",
    re.IGNORECASE,
)
_TOOL_ERROR_INDICATORS = (
    "error:",
    "failed",
    "http error",
    "timed out",
    "permission denied",
    "access denied",
    "not found",
    "tool not loaded",
    "path outside allowed",
    "cannot",
)


def _looks_like_markdown_phantom_report(message: Any) -> bool:
    """Return True when the response is a fabricated structured markdown report.

    The "markdown phantom" variant generates a plausible-looking tool-output
    report (numbered sections, tables) from training-data memory without calling
    any tools.  Unlike XML/JSON phantoms it contains no markup syntax — just
    normal prose and tables — so ``_looks_like_phantom_tool_markup`` misses it.

    Detection signal: markdown table rows AND numbered section headers together
    in a response with no tool calls.  Legitimate responses either have tool_calls
    before producing structured output, or produce plain prose without tables.
    """
    if getattr(message, "tool_calls", None):
        return False
    content = getattr(message, "content", "")
    if not isinstance(content, str) or len(content) < 80:
        return False
    return bool(_MARKDOWN_TABLE_ROW_RE.search(content) and _NUMBERED_SECTION_RE.search(content))


def _looks_like_fabricated_success_after_tool_errors(
    messages: typing.Sequence[Any],
    last_message: Any,
) -> bool:
    """Return True when final success text contradicts immediately prior tool errors.

    Guard scope is intentionally narrow:
    - only inspects the contiguous ToolMessage block immediately before ``last_message``
    - only fires when *all* of those tool results look like errors
    - only fires when ``last_message`` contains an explicit success claim
    """
    if getattr(last_message, "tool_calls", None):
        return False
    content = getattr(last_message, "content", "")
    if not isinstance(content, str) or not content.strip():
        return False
    if not _SUCCESS_CLAIM_RE.search(content):
        return False
    if _NEGATED_SUCCESS_RE.search(content):
        return False

    i = len(messages) - 2
    recent_tool_messages: list[Any] = []
    while i >= 0 and hasattr(messages[i], "tool_call_id"):
        recent_tool_messages.append(messages[i])
        i -= 1
    if not recent_tool_messages:
        return False

    for tool_msg in recent_tool_messages:
        tool_content = getattr(tool_msg, "content", "")
        if not isinstance(tool_content, str):
            return False
        headline = _stuck_detection_headline(tool_content).lower()
        headline = headline.lstrip()
        if not any(headline.startswith(ind) for ind in _TOOL_ERROR_INDICATORS):
            return False

    return True


def _stuck_detection_headline(content: str) -> str:
    """Return the first non-empty line used for stuck-detection heuristics."""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


@dataclass
class ToolManagementRequest:
    """Result of scanning agent messages for ``request_tools`` calls."""

    add: list[str]
    remove: list[str]

    @property
    def has_changes(self) -> bool:
        return bool(self.add or self.remove)


def _detect_tool_request(messages: list, start_idx: int = 0) -> ToolManagementRequest | None:
    """
    Scan agent messages for a ``request_tools`` invocation.

    Supports both the new schema (``add`` / ``remove``) and the legacy
    schema (``names`` treated as additions).

    Args:
        messages: Full message list from the agent result.
        start_idx: Index to start scanning from (skip history messages).

    Returns a ``ToolManagementRequest`` or *None* if no request was made.
    """
    all_add: list[str] = []
    all_remove: list[str] = []

    for i in range(start_idx, len(messages)):
        msg = messages[i]
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            continue
        for tc in tool_calls:
            if isinstance(tc, dict) and tc.get("name") == "request_tools":
                args = tc.get("args", {})

                # New schema: add / remove
                add_names = args.get("add", [])
                remove_names = args.get("remove", [])

                # Legacy fallback: bare ``names`` list → treat as add
                if not add_names and not remove_names:
                    add_names = args.get("names", [])

                # Normalize bare strings to single-element lists so
                # {"add": "web_search"} works the same as {"add": ["web_search"]}
                # (BUG-204).
                if isinstance(add_names, str):
                    add_names = [add_names]
                if isinstance(remove_names, str):
                    remove_names = [remove_names]

                if isinstance(add_names, list):
                    all_add.extend(str(n) for n in add_names)
                if isinstance(remove_names, list):
                    all_remove.extend(str(n) for n in remove_names)

    if not all_add and not all_remove:
        return None
    # Deduplicate and strip empty strings so LLM-generated garbage (e.g.
    # duplicate names, empty strings from JSON coercion) produces a clear
    # no-op or guidance line rather than a silent failure.
    seen_add: set[str] = set()
    deduped_add: list[str] = []
    for n in all_add:
        if n and n not in seen_add:
            seen_add.add(n)
            deduped_add.append(n)
    seen_rem: set[str] = set()
    deduped_rem: list[str] = []
    for n in all_remove:
        if n and n not in seen_rem:
            seen_rem.add(n)
            deduped_rem.append(n)
    return ToolManagementRequest(add=deduped_add, remove=deduped_rem)


def _detect_invalid_tool_calls(
    messages: list,
    start_idx: int = 0,
) -> list[str]:
    """
    Scan *messages* from *start_idx* for **any** "is not a valid tool"
    ToolMessage error, regardless of whether the tool is in the on-demand
    pool.

    Returns a de-duplicated, ordered list of tool names the LLM tried.
    """
    from langchain_core.messages import ToolMessage

    found: list[str] = []
    seen: set[str] = set()
    for i in range(start_idx, len(messages)):
        msg = messages[i]
        if not isinstance(msg, ToolMessage):
            continue
        content = getattr(msg, "content", "")
        if not isinstance(content, str):
            continue
        m = _INVALID_TOOL_RE.match(content)
        if m:
            tool_name = m.group(1)
            if tool_name not in seen:
                found.append(tool_name)
                seen.add(tool_name)
    return found


def _strip_failed_tool_messages(messages: list, tool_names: set[str]) -> list:
    """
    Return a copy of *messages* with ToolMessage errors (and their matching
    AIMessage tool_calls) removed for tools in *tool_names*.

    This cleans up the conversation history after auto-activation so the
    resumed agent doesn't see the failed "is not a valid tool" attempts.
    """
    from langchain_core.messages import AIMessage, ToolMessage

    tool_call_ids_to_remove: set[str] = set()
    cleaned: list = []

    for msg in messages:
        if isinstance(msg, ToolMessage):
            name = getattr(msg, "name", "")
            content = getattr(msg, "content", "")
            if name in tool_names and isinstance(content, str) and "is not a valid tool" in content:
                tcid = getattr(msg, "tool_call_id", "")
                if tcid:
                    tool_call_ids_to_remove.add(tcid)
                continue
        cleaned.append(msg)

    if not tool_call_ids_to_remove:
        return cleaned

    final: list = []
    for msg in cleaned:
        if isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                remaining = [tc for tc in tool_calls if tc.get("id") not in tool_call_ids_to_remove]
                if len(remaining) != len(tool_calls):
                    extra = dict(getattr(msg, "additional_kwargs", {}))
                    extra.pop("tool_calls", None)
                    new_msg = AIMessage(
                        content=getattr(msg, "content", ""),
                        tool_calls=remaining,
                        additional_kwargs=extra,
                    )
                    if not remaining and not (
                        isinstance(new_msg.content, str) and new_msg.content.strip()
                    ):
                        continue
                    final.append(new_msg)
                    continue
        final.append(msg)
    return final


def _repair_tool_message_pairs(messages: list) -> list:
    """Remove ToolMessages whose tool_call_id has no valid preceding AIMessage.

    OpenAI (and compatible providers) reject requests where a ToolMessage is not
    preceded by an AIMessage that contains a tool_call with a matching id.  This
    situation arises when:
    - An MCP/tool call raises an exception (e.g. ClosedResourceError) and the
      ToolMessage error is stored in state, but the triggering AIMessage was empty
      or had a malformed / truncated tool_calls list.
    - Message compression strips tool_calls from an AIMessage while retaining the
      paired ToolMessages.

    The repair pass collects every tool_call id that appears in an AIMessage
    (checking .tool_calls, additional_kwargs["tool_calls"], and Anthropic/Bedrock
    content blocks), then drops any ToolMessage whose tool_call_id is absent from
    that set or appears before the declaring AIMessage.  Truly empty AIMessages
    (no content, no tool_calls of any kind) that no longer serve as a pair anchor
    are also dropped.
    """
    from langchain_core.messages import AIMessage, ToolMessage

    def _collect_tool_call_ids(msg: AIMessage) -> set[str]:
        """Return all tool_call ids declared by an AIMessage across all encoding styles."""
        ids: set[str] = set()
        # Standard LangChain attribute (OpenAI, Anthropic modern, etc.)
        for tc in getattr(msg, "tool_calls", None) or []:
            tcid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            if tcid:
                ids.add(tcid)
        # OpenAI additional_kwargs encoding (some providers / older LangChain)
        for tc in (getattr(msg, "additional_kwargs", None) or {}).get("tool_calls") or []:
            tcid = tc.get("id") if isinstance(tc, dict) else None
            if tcid:
                ids.add(tcid)
        # Anthropic/Bedrock content-block encoding: content=[{type:tool_use, id:...}]
        content = getattr(msg, "content", None)
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tcid = block.get("id")
                    if tcid:
                        ids.add(tcid)
        return ids

    def _msg_has_content(msg: AIMessage) -> bool:
        """True when the message carries text, tool-calls, or content blocks."""
        if _collect_tool_call_ids(msg):
            return True
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            return bool(content.strip())
        if isinstance(content, list):
            return bool(content)  # any content blocks (including tool_use) count
        return bool(content)

    # Pass 1 — collect declared tool_call ids and their first declaring position.
    declared_ids: set[str] = set()
    declared_positions: dict[str, int] = {}
    for idx, msg in enumerate(messages):
        if not isinstance(msg, AIMessage):
            continue
        tool_call_ids = _collect_tool_call_ids(msg)
        declared_ids |= tool_call_ids
        for tcid in tool_call_ids:
            declared_positions.setdefault(tcid, idx)

    # Pass 2 — identify orphaned and misordered ToolMessage tool_call_ids.
    orphaned_ids: set[str] = set()
    misordered_ids: set[str] = set()
    for msg_idx, msg in enumerate(messages):
        if isinstance(msg, ToolMessage):
            tcid = getattr(msg, "tool_call_id", None)
            if tcid and tcid not in declared_ids:
                orphaned_ids.add(tcid)
                continue
            if tcid and declared_positions.get(tcid) is not None:
                if msg_idx < declared_positions[tcid]:
                    misordered_ids.add(tcid)

    if not orphaned_ids and not misordered_ids:
        return messages

    import logging as _logging

    _logging.getLogger("cogtrix.orchestration.graph").warning(
        "Repairing %d orphaned and %d misordered ToolMessage(s) (orphans: %s; misordered: %s) — "
        "likely caused by ClosedResourceError, malformed tool_calls, or compressed history",
        len(orphaned_ids),
        len(misordered_ids),
        ", ".join(sorted(orphaned_ids)) if orphaned_ids else "none",
        ", ".join(sorted(misordered_ids)) if misordered_ids else "none",
    )

    repaired: list = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tcid = getattr(msg, "tool_call_id", None)
            if tcid in orphaned_ids or tcid in misordered_ids:
                continue  # drop orphaned or misordered ToolMessage
        elif isinstance(msg, AIMessage):
            # Drop truly empty AIMessages: no text, no tool_calls, no content blocks
            if not _msg_has_content(msg):
                continue
        repaired.append(msg)
    return repaired


def _apply_context_message_cap(
    messages: list,
    max_messages: int | None,
    max_tokens: int | None = None,
) -> list:
    """Trim oldest message pairs when history exceeds the configured cap(s).

    Consecutive AIMessage + ToolMessage runs are treated as a single logical
    chunk so tool-call pairs are never split.  Oldest chunks are dropped until
    both the message-count and token budgets fit.  The newest chunk is always
    preserved even if it exceeds the configured budget on its own.
    """
    if (not max_messages or max_messages <= 0) and (not max_tokens or max_tokens <= 0):
        return messages

    from langchain_core.messages import AIMessage, ToolMessage

    def _tool_ids(msg: Any) -> set[str]:
        ids: set[str] = set()
        for tc in getattr(msg, "tool_calls", None) or []:
            tcid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            if tcid:
                ids.add(tcid)
        return ids

    def _msg_tokens(msg: Any) -> int:
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            return max(1, len(content) // _CHARS_PER_TOKEN)
        if isinstance(content, list):
            chars = 0
            for item in content:
                if isinstance(item, str):
                    chars += len(item)
                elif isinstance(item, dict):
                    chars += len(item.get("text", ""))
            return max(1, chars // _CHARS_PER_TOKEN)
        return 1

    chunks: list[list[Any]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if isinstance(msg, AIMessage):
            tc_ids = _tool_ids(msg)
            if tc_ids:
                chunk: list[Any] = [msg]
                j = i + 1
                while j < len(messages):
                    nxt = messages[j]
                    if not isinstance(nxt, ToolMessage):
                        break
                    if getattr(nxt, "tool_call_id", None) not in tc_ids:
                        break
                    chunk.append(nxt)
                    j += 1
                chunks.append(chunk)
                i = j
                continue
        chunks.append([msg])
        i += 1

    kept: list[list[Any]] = []
    kept_count = 0
    kept_tokens = 0
    for chunk in reversed(chunks):
        chunk_count = len(chunk)
        chunk_tokens = sum(_msg_tokens(msg) for msg in chunk)
        if not kept:
            kept.append(chunk)
            kept_count += chunk_count
            kept_tokens += chunk_tokens
            continue
        if max_messages and max_messages > 0 and kept_count + chunk_count > max_messages:
            break
        if max_tokens and max_tokens > 0 and kept_tokens + chunk_tokens > max_tokens:
            break
        kept.append(chunk)
        kept_count += chunk_count
        kept_tokens += chunk_tokens

    if not kept:
        return messages

    kept.reverse()
    truncated = [m for chunk in kept for m in chunk]
    dropped = len(messages) - len(truncated)
    if dropped > 0:
        import logging as _log_mod

        _log_mod.getLogger("cogtrix.orchestration.graph").warning(
            "context_max_messages=%s context_max_tokens=%s: dropped %d oldest message(s)",
            max_messages if max_messages is not None else 0,
            max_tokens if max_tokens is not None else 0,
            dropped,
        )
    return truncated


# Cache for _correct_tool_args schema introspection results.
# Keyed by logical schema identity (tool name + sorted field names) so MCP
# reconnects that recreate equivalent Pydantic models reuse the same cache entry.
_ToolArgSchemaCacheKey = tuple[str, tuple[str, ...]]
_TOOL_ARG_SCHEMA_CACHE_MAX_SIZE = 512
_tool_arg_schema_cache: dict[
    _ToolArgSchemaCacheKey, tuple[dict[str, Any], dict[str, str], dict[str, str]]
] = {}
_tool_arg_cache_lock = threading.Lock()

_FUZZY_ARG_BLOCKLIST: frozenset[str] = frozenset(
    {
        "data",
        "name",
        "port",
        "code",
        "type",
        "text",
        "path",
        "file",
        "mode",
        "size",
        "body",
        "host",
        "user",
        "role",
        "args",
        "keys",
    }
)


def _correct_tool_args(tool: Any, args: dict) -> dict:
    """Best-effort correction of misnamed tool arguments.

    Weaker LLMs sometimes send wrong parameter names (e.g. ``cmd`` instead of
    ``command``).  This function compares provided keys against the tool's
    Pydantic ``args_schema`` and applies two heuristics:

    1. **Fuzzy name match** — uses substring containment and SequenceMatcher
       to remap unknown arg names to the closest expected field.
    2. **Type coercion** — if the schema expects ``str`` and the value is a
       ``list`` or ``dict``, serialise it to a JSON string.

    Returns the (possibly corrected) args dict.  On any error, returns the
    original args unchanged.
    """
    schema = getattr(tool, "args_schema", None)
    if schema is None:
        return args

    try:
        expected: dict[str, Any] = {}
        if hasattr(schema, "model_fields"):
            expected = schema.model_fields  # Pydantic v2
        elif hasattr(schema, "__fields__"):
            expected = schema.__fields__  # Pydantic v1
        if not expected:
            return args
    except (AttributeError, TypeError) as exc:
        tool_name = str(getattr(tool, "name", "") or "<unknown_tool>")
        get_logger().warning(
            "_correct_tool_args: schema introspection failed for tool %r: %s — returning args unchanged",
            tool_name,
            exc,
        )
        return args

    expected = dict(expected)
    tool_name = str(getattr(tool, "name", "") or "<unknown_tool>")
    cache_key: _ToolArgSchemaCacheKey = (tool_name, tuple(sorted(expected.keys())))

    with _tool_arg_cache_lock:
        _cached = _tool_arg_schema_cache.get(cache_key)
        if _cached is not None:
            expected, alias_map, _well_known_remaps_cached = _cached
        else:
            # --- Alias resolution -------------------------------------------------
            # Pydantic aliases (Field(alias=...)) are not visible as field names.
            # Map known aliases to their canonical field name so LLMs that send the
            # alias (e.g. "cmd" instead of "command") get corrected before fuzzy match.
            alias_map: dict[str, str] = {}
            for fname, finfo in expected.items():
                _alias = getattr(finfo, "alias", None)
                if _alias and _alias != fname:
                    alias_map[_alias] = fname
                # Also check validation_alias (Pydantic v2)
                _valias = getattr(finfo, "validation_alias", None)
                if isinstance(_valias, str) and _valias != fname:
                    alias_map[_valias] = fname

            _well_known_remaps_cached: dict[str, str] = {}
            # Evict oldest entries if cache exceeds max size (FIFO eviction)
            if len(_tool_arg_schema_cache) >= _TOOL_ARG_SCHEMA_CACHE_MAX_SIZE:
                # Pop the first (oldest) item - Python 3.7+ dicts maintain insertion order
                _tool_arg_schema_cache.pop(next(iter(_tool_arg_schema_cache)))
            _tool_arg_schema_cache[cache_key] = (expected, alias_map, _well_known_remaps_cached)

    expected_names = set(expected.keys())
    provided_names = set(args.keys())

    corrected = dict(args)

    for alias_key, canonical in alias_map.items():
        if alias_key in corrected and canonical not in corrected:
            corrected[canonical] = corrected.pop(alias_key)
            log = get_logger()
            log.info("Tool arg alias resolved: '%s' → '%s'", alias_key, canonical)

    # --- Well-known parameter variations ────────────────────────────
    # LLMs frequently use common synonyms that fall below the fuzzy
    # threshold (0.75).  Explicit remaps for the most common cases.
    _WELL_KNOWN_REMAPS: dict[str, list[str]] = {
        "filename": ["path"],
        "file_path": ["path"],
        "filepath": ["path"],
        "file_name": ["path"],
        "file_content": ["content"],
        "text": ["content", "prompt"],
        "body": ["content"],
        "cmd": ["command"],
        "query_string": ["query"],
        "search_query": ["query"],
        "dir": ["path"],
        "directory": ["path"],
        # Additional common LLM variants (ratio 0.75–0.84 — below old threshold)
        "infile": ["input_file"],
        "input_file": ["infile"],
        "workdir": ["working_dir"],
        "working_dir": ["workdir"],
        "verbose": ["verbosity"],
        "verbosity": ["verbose"],
        "filenamestr": ["file_name"],
        # cron_add: LLMs commonly use "pattern" for cron expressions (#520)
        "pattern": ["schedule"],
        "expression": ["schedule"],
        # GitHub PR tools: LLMs use pr_number / number for pull_number
        "pr_number": ["pull_number"],
        "pull_request_number": ["pull_number"],
        # list_pull_requests: LLMs use status for state
        "status": ["state"],
        # Tools that expect "prompt" but LLM sends content/message
        "content": ["prompt"],
        "message": ["prompt"],
    }
    for provided_key in list(corrected.keys()):
        if provided_key in expected_names:
            continue  # already matches a field — skip
        for canonical in _WELL_KNOWN_REMAPS.get(provided_key, []):
            if canonical in expected_names and canonical not in corrected:
                corrected[canonical] = corrected.pop(provided_key)
                log = get_logger()
                log.info("Tool arg well-known remap: '%s' → '%s'", provided_key, canonical)
                break

    provided_names = set(corrected.keys())

    # --- Name remapping ---------------------------------------------------
    unknown = provided_names - expected_names
    missing = expected_names - provided_names

    if unknown and missing:
        _REMAP_THRESHOLD = 0.75
        for unk in unknown:
            unk_lower = unk.lower()
            best: str | None = None
            best_ratio = 0.0
            tied = False
            for exp in missing:
                exp_lower = exp.lower()
                # Substring containment — only trust when the shorter
                # string is long enough to be meaningful.
                shorter_len = min(len(unk_lower), len(exp_lower))
                longer_len = max(len(unk_lower), len(exp_lower))
                if (
                    shorter_len >= 5
                    and shorter_len / longer_len >= 0.5
                    and unk_lower not in _FUZZY_ARG_BLOCKLIST
                    and (unk_lower in exp_lower or exp_lower in unk_lower)
                ):
                    ratio = 1.0
                else:
                    ratio = SequenceMatcher(None, unk_lower, exp_lower).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best = exp
                    tied = False
                elif abs(ratio - best_ratio) < 1e-9 and ratio >= _REMAP_THRESHOLD:
                    tied = True
            if best is not None and best_ratio >= _REMAP_THRESHOLD and not tied:
                corrected[best] = corrected.pop(unk)
                missing.discard(best)
                log = get_logger()
                log.info("Tool arg corrected: '%s' → '%s' (score=%.2f)", unk, best, best_ratio)

    # --- Type coercion: schema expects list but got JSON-encoded string → decode.
    import json as _json_mod

    for key, value in list(corrected.items()):
        if key not in expected:
            continue
        if not isinstance(value, str):
            continue
        field_info = expected[key]
        annotation = getattr(field_info, "annotation", None) or getattr(
            field_info, "outer_type_", None
        )
        origin = typing.get_origin(annotation)
        if origin is typing.Union or isinstance(annotation, types.UnionType):
            type_args = [a for a in typing.get_args(annotation) if a is not type(None)]
            if len(type_args) == 1:
                annotation = type_args[0]
        if annotation is list or (typing.get_origin(annotation) is list):
            stripped = value.strip()
            if stripped.startswith("["):
                try:
                    parsed = _json_mod.loads(stripped)
                    if isinstance(parsed, list):
                        corrected[key] = parsed
                        log = get_logger()
                        log.debug("Tool arg '%s' coerced from JSON string to list", key)
                except (ValueError, KeyError):
                    pass

    # --- Type coercion: schema expects str but got list/dict → JSON-encode.
    for key, value in list(corrected.items()):
        if key not in expected:
            continue
        if not isinstance(value, (list, dict)):
            continue
        field_info = expected[key]
        annotation = getattr(field_info, "annotation", None) or getattr(
            field_info, "outer_type_", None
        )
        # Unwrap Optional[str] / str | None → str
        origin = typing.get_origin(annotation)
        if origin is typing.Union or isinstance(annotation, types.UnionType):
            type_args = [a for a in typing.get_args(annotation) if a is not type(None)]
            if len(type_args) == 1:
                annotation = type_args[0]
        if annotation is str:
            if isinstance(value, list) and all(isinstance(v, str) for v in value):
                corrected[key] = " ".join(value)
            else:
                corrected[key] = _json.dumps(value)

    return corrected


def _safe_tool_name(name: str, max_len: int = 80) -> str:
    """Strip everything except word chars, hyphens, and dots; truncate.

    Prevents model-supplied names from carrying injection payload into
    guidance messages that are fed back to the model.
    """
    sanitized = re.sub(r"[^\w\-\.]", "", name)
    return sanitized[:max_len] if sanitized else "<unknown>"


@dataclass
class PerRunState:
    """All mutable per-run state for a compiled agent graph.

    Scalar fields are list-wrapped so that closures mutated inside
    graph nodes see the current value without rebinding the closure.
    A fresh instance is created by ``_reset_for_new_run()`` between
    agent turns, which guarantees every counter and collection is
    zeroed — a new field added here is automatically reset.
    """

    # Retry / loop counters
    phantom_count: list[int] = field(default_factory=lambda: [0])
    fabrication_count: list[int] = field(default_factory=lambda: [0])
    action_intent_count: list[int] = field(default_factory=lambda: [0])
    incompleteness_nudge_given: list[int] = field(default_factory=lambda: [0])
    expansion_count: list[int] = field(default_factory=lambda: [0])
    auto_expansion_count: list[int] = field(default_factory=lambda: [0])
    call_count: list[int] = field(default_factory=lambda: [0])
    last_input_tokens: list[int] = field(default_factory=lambda: [0])
    request_tools_noop_count: list[int] = field(default_factory=lambda: [0])

    # Tool tracking
    tool_version: list[int] = field(default_factory=lambda: [0])
    last_tool_version: list[int] = field(default_factory=lambda: [-1])
    tool_call_history: OrderedDict[str, str] = field(default_factory=OrderedDict)
    tool_call_counts: dict[str, int] = field(default_factory=dict)

    # Reflection / health-check pacing
    last_reflection_at: list[int] = field(default_factory=lambda: [0])
    last_tool_health_check_at: list[int] = field(default_factory=lambda: [0])

    # Stuck-detection state
    stuck_threshold_calibrated: list[bool] = field(default_factory=lambda: [False])
    stuck_no_checkpoint_threshold: list[int] = field(default_factory=lambda: [15])
    consecutive_errors: list[int] = field(default_factory=lambda: [0])
    force_thinking_break: list[bool] = field(default_factory=lambda: [False])
    consecutive_identical_error_count: list[int] = field(default_factory=lambda: [0])
    last_identical_error_signature: list[tuple[str, str] | None] = field(
        default_factory=lambda: [None]
    )

    # Checkpoint pacing
    last_checkpoint_count: list[int] = field(default_factory=lambda: [0])
    rounds_since_checkpoint: list[int] = field(default_factory=lambda: [0])
    calls_since_last_checkpoint: list[int] = field(default_factory=lambda: [0])

    # File-write tracking
    same_file_writes: dict[str, int] = field(default_factory=dict)

    # Cache / lookup state
    bound_cache: OrderedDict = field(default_factory=OrderedDict)
    compression_cache: dict[str, str] = field(default_factory=dict)
    tool_lookup: dict[str, Any] = field(default_factory=dict)
    active_names: set[str] = field(default_factory=set)
    tool_catalog: dict[str, str] = field(default_factory=dict)
    available_tools_ref: list[dict] = field(default_factory=list)


def build_agent_graph(
    llm: Any = None,
    system_prompt: str = "",
    active_tools_list: list | None = None,
    available_tools: dict | None = None,
    registry: Any = None,
    approvals: set | None = None,
    max_context_tokens: int | None = None,
    preset_tools: set[str] | None = None,
    context_compression: bool = True,
    compression_min_age: int = COMPRESSION_MIN_AGE_CYCLES,
    compression_min_chars: int = COMPRESSION_MIN_CHARS,
    compression_llm: Any = None,
    context_max_messages: int = 200,
    context_max_tokens: int = 40_000,
    tool_call_guard: Any | None = None,
    session_state: SessionState | None = None,
    confirmation_ui: Any | None = None,
    on_tool_expansion: Any | None = None,
    parallel_tool_execution: bool = True,
    git_native: bool = False,
    tool_context_limit_pct: float = 0.80,
    extend_run_state: Any = None,
    *,
    config: AgentRunConfig | None = None,
    bound_cache: OrderedDict | None = None,
    compression_cache_in: dict[str, str] | None = None,
    checkpoint_store: Any | None = None,
) -> Any:
    """Build a custom LangGraph StateGraph for the Cogtrix agent.

    The graph has three nodes:
    - call_model: binds active tools to LLM and invokes it
    - process_tools: executes tool calls, handles fuzzy matching and expansion
    - handle_phantom: recovers from phantom tool calls (malformed JSON)

    Tool management uses closured mutable references: active_tools_list and
    available_tools are modified in-place, so callers see the changes after
    graph execution.

    When *config* is provided, its fields take precedence over the individual
    keyword arguments (backward-compat layer).
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain_core.messages.modifier import RemoveMessage
    from langgraph.graph import END, StateGraph

    _model_timeout = 180  # default LLM request timeout (seconds)

    if config is not None:
        if config.llm is not None:
            llm = config.llm
        if config.system_prompt is not None:
            system_prompt = config.system_prompt
        if config.active_tools_list is not None:
            active_tools_list = config.active_tools_list
        if config.available_tools is not None:
            available_tools = config.available_tools
        if config.max_context_tokens is not None:
            max_context_tokens = config.max_context_tokens
        if config.preset_tools is not None:
            preset_tools = config.preset_tools
        if config.tool_call_guard is not None:
            tool_call_guard = config.tool_call_guard
        if config.session_state is not None:
            session_state = config.session_state
        memory_manager = getattr(config, "memory_manager", None)
        if config.confirmation_ui is not None:
            confirmation_ui = config.confirmation_ui
        if config.on_tool_expansion is not None:
            on_tool_expansion = config.on_tool_expansion
        parallel_tool_execution = config.parallel_tool_execution
        git_native = config.git_native
        context_compression = config.context_compression
        _context_max_messages = getattr(config, "context_max_messages", context_max_messages) or 0
        _context_max_tokens = getattr(config, "context_max_tokens", context_max_tokens) or 0
        _model_timeout = getattr(config, "llm_timeout", 180)
        if config.compression_llm is not None:
            compression_llm = config.compression_llm
        if config.compression_min_age is not None:
            compression_min_age = config.compression_min_age
        if config.compression_min_chars is not None:
            compression_min_chars = config.compression_min_chars
        if hasattr(config, "tool_context_limit_pct"):
            tool_context_limit_pct = config.tool_context_limit_pct
        if hasattr(config, "checkpoint_store"):
            checkpoint_store = config.checkpoint_store
        tools_ready = getattr(config, "tools_ready", None)
        _tier_cache_enabled = getattr(config, "tier_cache_enabled", True)
        _da_enabled = getattr(config, "decision_accountability_enabled", False)
        _da_report_uncertainty = getattr(config, "decision_accountability_report_uncertainty", True)
        _da_min_confidence = getattr(config, "decision_accountability_min_confidence", 7.0)
    else:
        _context_max_messages = context_max_messages or 0
        _context_max_tokens = context_max_tokens or 0
        tools_ready = None
        memory_manager = None
        _tier_cache_enabled = False
        _da_enabled = False
        _da_report_uncertainty = True
        _da_min_confidence = 7.0

    if active_tools_list is None:
        active_tools_list = []
    if available_tools is None:
        available_tools = {}
    if approvals is None:
        approvals = set()

    if session_state is None:
        session_state = SessionState()

    # Mutable container so _reset_for_new_run can swap the state each run
    # without rebuilding the compiled graph.
    extend_run_state_ref: list[Any] = [extend_run_state]
    if extend_run_state is not None:
        try:
            from langchain_core.tools import StructuredTool as _ST

            from src.tools.extend_run import ExtendRunInput

            def _extend_run_fn(
                mode: str = "continue",
                subtasks: list[str] | None = None,
                reason: str = "",
            ) -> str:
                _state = extend_run_state_ref[0]
                if _state is None:
                    return "Error: extend_run is not available in this run context."
                if mode == "delegate" and not subtasks:
                    return (
                        "Error: mode='delegate' requires a non-empty 'subtasks' list. "
                        "Provide 2-5 independent subtask descriptions."
                    )
                _state.request_extension(mode=mode, subtasks=subtasks or [], reason=reason)
                if mode == "delegate":
                    count = len(subtasks or [])
                    return (
                        f"Extension registered: {count} subtask(s) queued for parallel "
                        "delegation. Continue sequential work; delegation runs after this run."
                    )
                return (
                    "Extension registered: the step budget will be increased when the "
                    "current limit is reached. Continue working on the task."
                )

            _extend_tool = _ST.from_function(
                func=_extend_run_fn,
                name="extend_run",
                description=(
                    "Request more execution steps or delegate work to parallel sub-agents. "
                    "Call when the task needs significantly more turns than available.\n\n"
                    "Modes:\n"
                    "- 'continue': Request more sequential steps.\n"
                    "- 'delegate': Split into parallel sub-agents (requires 'subtasks' list).\n\n"
                    "Call EARLY — don't wait until almost out of steps."
                ),
                args_schema=ExtendRunInput,
            )
            if not any(getattr(tool, "name", "") == "extend_run" for tool in active_tools_list):
                active_tools_list.append(_extend_tool)
            available_tools["extend_run"] = _extend_tool
        except ImportError:
            pass
    _MAX_PHANTOM_RETRIES = 3
    _MAX_FABRICATION_RETRIES = 3
    _MAX_ACTION_INTENT_RETRIES = 3
    # After the 3 standard action-intent nudges are exhausted, the model
    # gets exactly one more chance if the response contains incompleteness
    # language ("first", "to start", "step 1") — a stronger nudge that
    # demands completion rather than a generic "call the appropriate tool".
    _MAX_INCOMPLETENESS_NUDGES = 1
    _MAX_TOOL_EXPANSIONS = 3
    _MAX_REQUEST_TOOLS_NOOPS = 3
    _MAX_TOOL_CALL_HISTORY = 256
    _history_lock = threading.Lock()
    # Pending events for in-flight tool calls (BUG-1293): maps call_key -> threading.Event.
    # Used to block duplicate parallel threads until the first thread stores the result.
    _pending_events: dict[str, threading.Event] = {}
    # Per-tool call counter: tracks how many times each tool is called this turn.
    # After _TOOL_BUDGET_SOFT calls, a synthesis hint is appended to the output.
    # After _TOOL_BUDGET_HARD calls, the tool returns a stop message.
    _tool_budget_lock = (
        threading.Lock()
    )  # Protects _per_run_state[0].tool_call_counts and active_tools_list
    _TOOL_BUDGET_SOFT = 5  # nudge: "please synthesize"
    _TOOL_BUDGET_HARD = 8  # stop: "budget exhausted"
    _TOOL_BUDGET_SOFT_EXEMPT = {
        "request_tools",
        "report_progress",
        "queue_reply",
        "list_scheduled_messages",
        "edit_scheduled_message",
        "cancel_scheduled_message",
        "defer_processing",
        "suppress_reply",
        # Action tools that naturally require many sequential calls for
        # complex tasks (building software, multi-file edits, etc.).
        # The budget is designed to prevent runaway *search* loops, not
        # to throttle legitimate action sequences.
        "execute_shell_command",
        "write_file",
        "append_file",
        "patch_file",
        # Progress tracking — must always be callable.
        "checkpoint",
    }
    _TOOL_BUDGET_HARD_EXEMPT = _TOOL_BUDGET_SOFT_EXEMPT | {
        # Search tools should not hard-stop at the fixed cutoff because
        # legitimate research often requires many progressive searches.
        "search_web",
        "search_news",
        "google_search",
        "brave_search",
        "exa_search",
        "tavily_search",
        "serpapi_search",
        "searxng_search",
        "search_email",
        "calendar_search_events",
    }
    _DUPLICATE_EXEMPT = {
        "request_tools",
        "report_progress",
        "queue_reply",
        "list_scheduled_messages",
        "edit_scheduled_message",
        "cancel_scheduled_message",
        # These control tools must always return a fresh result; caching a
        # prior error (e.g. from a ToolCallGuard block) would cause a retry
        # to receive the stale "duplicate" error instead of being evaluated
        # on its own merits (BUG-237).
        "suppress_reply",
        "defer_processing",
    }
    protected = (preset_tools or set()) | {"request_tools"}
    _bound_cache_lock = threading.Lock()
    _REFLECTION_INTERVAL = 10  # inject reflection every N call_model cycles
    _TOOL_HEALTH_CHECK_INTERVAL = (
        getattr(config, "tool_health_check_interval", 20) if config is not None else 20
    )
    _TOOL_QUALITY_GATE_ENABLED = (
        getattr(config, "tool_quality_gate_enabled", True) if config is not None else True
    )
    _TOPIC_SWITCH_DETECTION_ENABLED = (
        getattr(config, "topic_switch_detection_enabled", True) if config is not None else True
    )

    # ── Stuck detection ───────────────────────────────────────────────
    # Tracks consecutive tool calls that produce errors.  When the count
    # reaches _STUCK_THRESHOLD, the next call_model invokes the LLM
    # WITHOUT tools (forced thinking break) so the model must produce
    # a text-only Chain-of-Thought response before it can resume tool use.
    _STUCK_THRESHOLD = 5  # consecutive error results before forcing a break
    _CHECKPOINT_NUDGE_INTERVAL = 8  # nudge after N tool calls without checkpoint
    _same_file_writes_lock = threading.Lock()
    _REWRITE_SEARCH_THRESHOLD = 2  # search reminder after N writes to same file

    _cached_fingerprint: list[tuple[str, ...]] = [()]
    output_cap = (
        compute_tool_output_cap(max_context_tokens)
        if max_context_tokens
        else TOOL_OUTPUT_CAP_MIN_CHARS
    )
    _sys_msg = SystemMessage(content=system_prompt) if system_prompt else None

    # ── Per-run mutable state (structurally reset) ────────────────────
    # All counters, collections, and lookup tables that must be zeroed
    # between agent turns live in a single dataclass.  A fresh instance
    # is created by _reset_for_new_run(), so a newly-added field is
    # automatically reset — no risk of forgetting a manual reset line.
    _tool_lookup_init: dict[str, Any] = {getattr(t, "name", ""): t for t in active_tools_list}
    _tool_lookup_init.pop("", None)
    _active_names_init: set[str] = set(_tool_lookup_init.keys())
    _per_run_state: list[PerRunState] = [
        PerRunState(
            tool_lookup=_tool_lookup_init,
            active_names=_active_names_init,
            tool_catalog=build_tool_catalog(available_tools),
            available_tools_ref=[available_tools],
            bound_cache=(bound_cache if bound_cache is not None else OrderedDict()),
            compression_cache=(compression_cache_in if compression_cache_in is not None else {}),
        )
    ]

    # ── Checkpoint store ──────────────────────────────────────────────
    from src.tools.checkpoint import CheckpointStore, create_checkpoint_tool

    if checkpoint_store is not None:
        _checkpoint_store: CheckpointStore = checkpoint_store
    else:
        _checkpoint_store = CheckpointStore()
    _checkpoint_store_lock = threading.Lock()

    _checkpoint_tool = create_checkpoint_tool(_checkpoint_store)
    if _checkpoint_tool is not None and active_tools_list is not None:
        _existing_names = {getattr(t, "name", "") for t in active_tools_list}
        if "checkpoint" not in _existing_names:
            active_tools_list.append(_checkpoint_tool)
            _per_run_state[0].active_names.add("checkpoint")
            _per_run_state[0].tool_lookup["checkpoint"] = _checkpoint_tool

    _graph_log = get_logger()

    def _warm_bound_cache() -> None:
        """Seed the bind_tools cache for the initial active tool set."""
        if llm is None or not active_tools_list:
            return
        if tools_ready is not None and not tools_ready.is_set():
            _graph_log.debug("Skipping bind_tools warm-up until MCP tools finish reconnecting")
            return
        tool_list = list(active_tools_list)
        _seen_names_rev: set[str] = set()
        deduped_rev: list[Any] = []
        for _t in reversed(tool_list):
            _tname = getattr(_t, "name", "")
            if _tname not in _seen_names_rev:
                _seen_names_rev.add(_tname)
                deduped_rev.append(_t)
        tool_list = list(reversed(deduped_rev))
        normalized_tools: list[Any] = []
        for tool_obj in tool_list:
            if isinstance(tool_obj, _LazyToolProxy):
                try:
                    tool_obj = tool_obj._resolve()
                except Exception as exc:
                    _graph_log.warning(
                        "bind_tools warm-up failed to resolve lazy tool %r: %s",
                        getattr(tool_obj, "name", ""),
                        exc,
                    )
                    continue
                if tool_obj is None:
                    continue
            normalized_tools.append(tool_obj)
        if not normalized_tools:
            return
        fingerprint = tuple(getattr(t, "name", "") for t in normalized_tools)
        if fingerprint in _per_run_state[0].bound_cache:
            return
        try:
            if len(_per_run_state[0].bound_cache) >= 8:
                _per_run_state[0].bound_cache.popitem(last=False)
            _per_run_state[0].bound_cache[fingerprint] = llm.bind_tools(normalized_tools)
            _cached_fingerprint[0] = fingerprint
            _graph_log.debug("⏱ bind_tools warm-up: %d tool(s)", len(normalized_tools))
        except Exception as exc:
            _graph_log.warning("Initial bind_tools warm-up failed: %s", exc)

    _warm_bound_cache()

    def _maybe_compress(msgs: list) -> list:
        """Pre-invoke compression check (mid-turn guard).

        Uses actual token counts from the previous model call when available,
        falling back to char-based estimates.  Fires at
        _MID_TURN_COMPRESSION_THRESHOLD (0.60) — lower than the turn-start
        token-based threshold (0.72) — so context can never grow to 100%
        during a long tool loop before compression triggers.

        When TCC is active, this guard is a safety net only — the background
        roll-forward handles compression incrementally.  The threshold is raised
        to 0.80 to avoid redundant mid-turn LLM calls.

        At 85%+ char pressure (emergency), min_age_override=0 forces all
        eligible ToolMessages to be compressed regardless of age.
        """
        _comp_llm = compression_llm or llm
        if not context_compression or _comp_llm is None:
            return msgs
        if max_context_tokens is None or max_context_tokens < 16_384:
            return msgs
        total_chars = sum(_content_len(m) for m in msgs)
        context_chars = max_context_tokens * _CHARS_PER_TOKEN
        if context_chars <= 0:
            return msgs
        ratio = total_chars / context_chars
        # Also check token-based ratio when real data is available — the
        # char estimate underestimates web/JSON content density.
        token_ratio = 0.0
        last_tokens = _per_run_state[0].last_input_tokens[0]
        if last_tokens > 0 and max_context_tokens > 0:
            token_ratio = last_tokens / max_context_tokens
        effective_ratio = max(ratio, token_ratio)
        # When TCC is active, raise the threshold to 0.80 — the mid-turn guard
        # is a safety net only; roll-forward handles most compression.
        _mid_turn_threshold = 0.80 if _tier_cache_enabled else _MID_TURN_COMPRESSION_THRESHOLD
        if effective_ratio < _mid_turn_threshold:
            return msgs
        # Emergency: min_age_override=0 compresses regardless of message age.
        # Non-emergency: min_age_override=compression_min_age bypasses the
        # internal token/char threshold check while keeping the age guard.
        min_age_ovr = 0 if effective_ratio >= _EMERGENCY_THRESHOLD_RATIO else compression_min_age
        return apply_message_compression(
            msgs,
            call_count=_per_run_state[0].call_count[0],
            compression_cache=_per_run_state[0].compression_cache,
            llm=_comp_llm,
            max_context_tokens=max_context_tokens,
            min_age_cycles=compression_min_age,
            min_chars=compression_min_chars,
            min_age_override=min_age_ovr,
            actual_input_tokens=last_tokens,
        )

    # ── LLM call with timeout ─────────────────────────────────────
    # Prevents indefinite hangs when the LLM backend disconnects.
    _LLM_RETRY_TIMEOUT = 300  # seconds — retry timeout after first attempt fails
    _LLM_MAX_RETRIES = 3  # total attempts (1 initial + 2 retries)
    _LLM_RETRY_BASE_DELAY = 2.0  # seconds — doubles on each retry (2, 4)

    def _is_retryable_error(exc: Exception) -> bool:
        """Return True for transient errors worth retrying (rate limits, 5xx)."""
        msg = str(exc).lower()
        return any(
            p in msg
            for p in (
                "rate limit",
                "rate_limit",
                "too many requests",
                "429",
                "503",
                "502",
                "500",
                "server error",
                "overloaded",
                "capacity",
                "temporarily",
            )
        )

    def _invoke_with_timeout(_model: Any, _messages: list, _cfg: Any, _timeout: int) -> Any:
        import concurrent.futures as _cf

        _executor = _get_llm_executor()
        last_exc: Exception | None = None
        for _attempt in range(_LLM_MAX_RETRIES):
            _fut = _executor.submit(_model.invoke, _messages, _cfg)
            try:
                _timeout_for_attempt = _LLM_RETRY_TIMEOUT if _attempt > 0 else _timeout
                return _fut.result(timeout=_timeout_for_attempt)
            except _cf.TimeoutError:
                # Cancel the future so the shared executor can reclaim the
                # slot.  If the underlying LLM I/O is stuck the OS thread may
                # continue running, but it is bounded by the pool's max_workers.
                _fut.cancel()
                last_exc = RuntimeError(
                    f"LLM backend not responding (timed out after {_timeout_for_attempt}s)"
                )
                _graph_log.warning(
                    "LLM call timed out after %ds (attempt %d/%d)",
                    _timeout_for_attempt,
                    _attempt + 1,
                    _LLM_MAX_RETRIES,
                )
            except Exception as _exc:
                if _is_retryable_error(_exc) and _attempt < _LLM_MAX_RETRIES - 1:
                    last_exc = _exc
                    _delay = _LLM_RETRY_BASE_DELAY * (2**_attempt)
                    _graph_log.warning(
                        "LLM call failed with retryable error (attempt %d/%d, "
                        "retrying in %.0fs): %s",
                        _attempt + 1,
                        _LLM_MAX_RETRIES,
                        _delay,
                        _exc,
                    )
                    time.sleep(_delay)
                    continue
                raise
            if _attempt < _LLM_MAX_RETRIES - 1:
                _delay = _LLM_RETRY_BASE_DELAY * (2**_attempt)
                time.sleep(_delay)
        raise last_exc or RuntimeError("LLM invocation failed after all retries")

    # ── Tool output quality gate helpers ──────────────────────────────
    _SUBSTANCELESS_PREFIXES = ("error:", "no results", "0 results")

    def _is_substanceless(content: Any) -> bool:
        """Return True if a tool result lacks actionable substance."""
        if content is None:
            return True
        if not isinstance(content, str):
            return False
        stripped = content.strip()
        if not stripped:
            return True
        # An empty JSON array/object is valid "nothing found" data, not no-data.
        # list_pull_requests returning [] means "no open PRs" — the quality gate
        # must not fire when prior turns already returned valid results.
        if stripped in ("[]", "{}", "[ ]", "{ }"):
            return False
        if len(stripped) < 20:
            return True
        lower = stripped.lower()
        if lower.startswith(_SUBSTANCELESS_PREFIXES):
            return True
        return False

    def _all_tool_results_substanceless(messages: list[Any]) -> bool:
        """Return True when the most recent contiguous ToolMessage block is non-empty
        and every message in it is substanceless.
        """
        tool_msgs: list[Any] = []
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                tool_msgs.append(msg)
            else:
                break
        if not tool_msgs:
            return False
        return all(_is_substanceless(getattr(m, "content", None)) for m in tool_msgs)

    from src.orchestration.nodes.call_model import CallModelContext, build_call_model_node

    call_model = build_call_model_node(
        CallModelContext(
            llm=llm,
            tools_ready=tools_ready,
            active_tools_list=active_tools_list,
            active_names=_per_run_state[0].active_names,
            bound_cache=_per_run_state[0].bound_cache,
            bound_cache_lock=_bound_cache_lock,
            cached_fingerprint=_cached_fingerprint,
            compression_cache=_per_run_state[0].compression_cache,
            tool_version=_per_run_state[0].tool_version,
            last_tool_version=_per_run_state[0].last_tool_version,
            call_count=_per_run_state[0].call_count,
            last_input_tokens=_per_run_state[0].last_input_tokens,
            max_context_tokens=max_context_tokens,
            context_max_messages=_context_max_messages,
            context_max_tokens=_context_max_tokens,
            model_max_tokens=getattr(llm, "max_tokens", None),
            compression_llm=compression_llm,
            memory_manager=memory_manager,
            checkpoint_store=_checkpoint_store,
            calls_since_last_checkpoint=_per_run_state[0].calls_since_last_checkpoint,
            last_checkpoint_count=_per_run_state[0].last_checkpoint_count,
            rounds_since_checkpoint=_per_run_state[0].rounds_since_checkpoint,
            force_thinking_break=_per_run_state[0].force_thinking_break,
            consecutive_errors=_per_run_state[0].consecutive_errors,
            last_identical_error_signature=_per_run_state[0].last_identical_error_signature,
            consecutive_identical_error_count=_per_run_state[0].consecutive_identical_error_count,
            last_reflection_at=_per_run_state[0].last_reflection_at,
            tool_health_check_interval=_TOOL_HEALTH_CHECK_INTERVAL,
            last_tool_health_check_at=_per_run_state[0].last_tool_health_check_at,
            tool_quality_gate_enabled=_TOOL_QUALITY_GATE_ENABLED,
            topic_switch_detection_enabled=_TOPIC_SWITCH_DETECTION_ENABLED,
            stuck_threshold=_STUCK_THRESHOLD,
            stuck_no_checkpoint_threshold=_per_run_state[0].stuck_no_checkpoint_threshold,
            stuck_threshold_calibrated=_per_run_state[0].stuck_threshold_calibrated,
            checkpoint_nudge_interval=_CHECKPOINT_NUDGE_INTERVAL,
            reflection_interval=_REFLECTION_INTERVAL,
            max_request_tools_noops=_MAX_REQUEST_TOOLS_NOOPS,
            sys_msg=_sys_msg,
            model_timeout=_model_timeout,
            tool_context_limit_pct=tool_context_limit_pct,
            da_enabled=_da_enabled,
            da_report_uncertainty=_da_report_uncertainty,
            da_min_confidence=_da_min_confidence,
            apply_context_message_cap=_apply_context_message_cap,
            maybe_compress=_maybe_compress,
            invoke_with_timeout=_invoke_with_timeout,
            all_tool_results_substanceless=_all_tool_results_substanceless,
        )
    )

    handle_phantom = build_handle_phantom_node(
        phantom_count=_per_run_state[0].phantom_count,
        max_retries=_MAX_PHANTOM_RETRIES,
    )
    handle_action_intent = build_handle_action_intent_node(
        action_intent_count=_per_run_state[0].action_intent_count,
        max_retries=_MAX_ACTION_INTENT_RETRIES,
        incompleteness_check=_has_incompleteness_signal,
    )

    def handle_fabrication(state: CogtrixState) -> dict:
        _per_run_state[0].fabrication_count[0] += 1
        last = state["messages"][-1]
        log = get_logger()
        log.warning(
            "Fabricated success-after-error detected, attempt %d/%d. Injecting correction.",
            _per_run_state[0].fabrication_count[0],
            _MAX_FABRICATION_RETRIES,
        )
        if _per_run_state[0].fabrication_count[0] > _MAX_FABRICATION_RETRIES:
            return {
                "messages": [
                    RemoveMessage(id=last.id),
                    AIMessage(
                        content=(
                            "I reported success incorrectly after tool errors and could not "
                            "recover safely. Please retry your request."
                        )
                    ),
                ]
            }
        return {
            "messages": [
                RemoveMessage(id=last.id),
                HumanMessage(
                    content=(
                        "Some of the tools you called returned errors, but your response claims "
                        "success. Report honestly what the tools returned. Do not fabricate "
                        "success messages."
                    )
                ),
            ]
        }

    def handle_incompleteness(state: CogtrixState) -> dict:
        """Inject a strongly-worded final nudge when the model signalled
        incomplete multi-step work but exhausted standard action-intent retries.
        """
        log = get_logger()
        log.warning(
            "Incompleteness signal detected after action-intent retries exhausted. "
            "Injecting critical completion nudge."
        )
        return {
            "messages": [
                HumanMessage(
                    content=(
                        "CRITICAL: The task is incomplete. "
                        "You used language like 'first' or 'to start', "
                        "which implies there are more steps to complete. "
                        "Do not explain what comes next — call the remaining "
                        "tool(s) NOW."
                    )
                )
            ]
        }

    def _tool_call_key(call: dict) -> str | None:
        """Compute the deduplication key for a tool call, or None if not serializable.

        Normalizes to the canonical tool name via ``_per_run_state[0].tool_lookup`` so that an
        alias and its resolved canonical name share the same cache key.  When the
        alias is not in ``_per_run_state[0].tool_lookup`` (e.g. during the auto-expansion serial
        path) the raw call name is used instead (BUG-234).
        """
        tool_name = call["name"]
        if tool_name in _DUPLICATE_EXEMPT:
            return None
        # Prefer the canonical name stored on the live tool object.
        tool_obj = _per_run_state[0].tool_lookup.get(tool_name)
        if tool_obj is not None:
            tool_name = getattr(tool_obj, "name", tool_name) or tool_name
        args_json = _json.dumps(_stable_tool_call_value(call.get("args", {})), sort_keys=True)
        return tool_name + ":" + args_json

    def _identical_error_signature(call: dict) -> str | None:
        """Return a stable signature for repeated identical-error detection.

        Uses the tool name plus the first meaningful argument so that retry
        loops on the same action are grouped together without requiring the
        full argument payload to match byte-for-byte.
        """
        tool_name = call.get("name", "")
        if not tool_name:
            return None
        args = call.get("args", {})
        if not isinstance(args, dict):
            return None
        primary_keys = (
            "pull_number",
            "path",
            "url",
            "query",
            "command",
            "name",
            "text",
            "repo",
            "email",
            "username",
            "branch",
        )
        primary_key = next(
            (
                key
                for key in primary_keys
                if key in args and args.get(key) not in (None, "", [], {})
            ),
            None,
        )
        if primary_key is None:
            if not args:
                return None
            primary_key = next(iter(sorted(args.keys())))
        try:
            primary_value = _json.dumps(args.get(primary_key), sort_keys=True, default=str)
        except (TypeError, ValueError):
            primary_value = str(args.get(primary_key))
        return f"{tool_name}:{primary_key}={primary_value}"

    def _tool_error_class(content: str) -> str | None:
        """Normalize an error ToolMessage into a coarse error class."""
        normalized = content.strip()
        if normalized.lower().startswith("[duplicate call"):
            normalized = normalized.split("\n\n", 1)[-1]
        content_lower = _stuck_detection_headline(normalized).lower()
        if "repository rule violations" in content_lower:
            return "repository_rule_violations"
        if "permission denied" in content_lower or "forbidden" in content_lower:
            return "permission_denied"
        if "timed out" in content_lower or "timeout" in content_lower:
            return "timeout"
        if (
            "not found" in content_lower
            or "404" in content_lower
            or "no such file" in content_lower
            or "cannot open" in content_lower
        ):
            return "not_found"
        if (
            content_lower.startswith("error")
            or "error executing" in content_lower
            or "traceback" in content_lower
            or "failed" in content_lower
        ):
            return "generic_error"
        return None

    def _tool_error_guidance(error_class: str, tool_name: str) -> str:
        """Return short guidance tailored to the repeated error class."""
        if error_class == "repository_rule_violations":
            return (
                f"'{_safe_tool_name(tool_name)}' hit repository rule violations. "
                "Stop retrying and verify CI/branch protections or ask a maintainer to merge."
            )
        if error_class == "permission_denied":
            return (
                f"'{_safe_tool_name(tool_name)}' was denied. Stop retrying and verify "
                "authentication or permissions before trying again."
            )
        if error_class == "timeout":
            return (
                f"'{_safe_tool_name(tool_name)}' timed out repeatedly. Stop retrying and "
                "switch to a different approach or inspect the service health."
            )
        if error_class == "not_found":
            return (
                f"'{_safe_tool_name(tool_name)}' cannot find the target repeatedly. "
                "Verify the identifier or path before trying again."
            )
        return (
            f"'{_safe_tool_name(tool_name)}' has returned the same error repeatedly. "
            "Stop retrying and inspect the last failure before continuing."
        )

    def _check_duplicate(call: dict, key: str | None = None) -> ToolMessage | None:
        """Return a cached ToolMessage if this exact call was seen before."""
        tool_name = call["name"]
        if key is None:
            key = _tool_call_key(call)
        if key is None:
            return None
        with _history_lock:
            cached = _per_run_state[0].tool_call_history.get(key)
            if cached is not None:
                _per_run_state[0].tool_call_history.move_to_end(key)
        if cached is None:
            return None
        log = get_logger()
        log.warning("Duplicate tool call detected: %s (returning cached result)", tool_name)
        return ToolMessage(
            content=(
                "[Duplicate call — returning cached result. Do NOT repeat this call.]\n\n" + cached
            ),
            tool_call_id=call["id"],
            name=tool_name,
        )

    def _store_call_result(call: dict, result_text: str, key: str | None = None) -> None:
        """Store a tool call result for duplicate detection."""
        if key is None:
            key = _tool_call_key(call)
        if key is None:
            return
        with _history_lock:
            _per_run_state[0].tool_call_history[key] = result_text[:500]
            _per_run_state[0].tool_call_history.move_to_end(key)
            if len(_per_run_state[0].tool_call_history) > _MAX_TOOL_CALL_HISTORY:
                _per_run_state[0].tool_call_history.popitem(last=False)

    def _cap_history_tool_content(content: str) -> str:
        """Cap tool output before it is stored in message history."""
        if len(content) <= _HISTORY_TOOL_MESSAGE_CAP_CHARS:
            return content
        return truncate_tool_output(content, _HISTORY_TOOL_MESSAGE_CAP_CHARS)

    def _invoke_one(call: dict, run_config: Any) -> Any:
        """Execute a single tool call already in tool_lookup. Returns ToolMessage."""
        call_key = _tool_call_key(call)
        dup = _check_duplicate(call, key=call_key)
        if dup is not None:
            return dup

        # ── TOCTOU guard (BUG-1293) ───────────────────────────────────────
        # Atomically check-and-reserve the cache slot so that parallel
        # duplicate tool calls invoke the tool only once.  Threads that
        # arrive while another thread is executing block on an Event until
        # the result is stored, then return the cached result.
        if call_key is not None:
            with _history_lock:
                cached = _per_run_state[0].tool_call_history.get(call_key)
                if cached is not None:
                    _per_run_state[0].tool_call_history.move_to_end(call_key)
                    return ToolMessage(
                        content=(
                            "[Duplicate call — returning cached result. Do NOT repeat this call.]\n\n"
                            + cached
                        ),
                        tool_call_id=call["id"],
                        name=call["name"],
                    )
                if call_key in _pending_events:
                    _wait_event = _pending_events[call_key]
                else:
                    _wait_event = None
                    _pending_events[call_key] = threading.Event()
            if _wait_event is not None:
                _wait_event.wait(timeout=30.0)
                with _history_lock:
                    cached = _per_run_state[0].tool_call_history.get(call_key)
                    if cached is not None:
                        _per_run_state[0].tool_call_history.move_to_end(call_key)
                        return ToolMessage(
                            content=(
                                "[Duplicate call — returning cached result. Do NOT repeat this call.]\n\n"
                                + cached
                            ),
                            tool_call_id=call["id"],
                            name=call["name"],
                        )
                # Should not reach here, but fall through to execute if it does
                _log = get_logger()
                _log.warning(
                    "TOCTOU wait timed out for %s — falling through to execute",
                    call_key,
                )
        # ───────────────────────────────────────────────────────────────────

        tool_name = call["name"]

        # ── Per-tool call budget ──────────────────────────────────────────
        # Prevents runaway search loops where the model calls the same tool
        # 10+ times with diminishing returns.  Exempt tools (request_tools,
        # report_progress, etc.) are not counted.
        if tool_name not in _TOOL_BUDGET_HARD_EXEMPT:
            # Critical section: protect compound read-increment-write on
            # _per_run_state[0].tool_call_counts and concurrent removal from active_tools_list
            with _tool_budget_lock:
                count = _per_run_state[0].tool_call_counts.get(tool_name, 0) + 1
                _per_run_state[0].tool_call_counts[tool_name] = count
                if count > _TOOL_BUDGET_HARD:
                    # Remove from active set AND add to denials so the model
                    # can't re-load it via request_tools(add=[...]).
                    # Also remove from active_tools_list so bind_tools stops
                    # advertising the disabled tool to the LLM, and so
                    # _reset_for_new_run doesn't silently re-enable it by
                    # rebuilding _per_run_state[0].tool_lookup from the stale list (root cause
                    # of the "Tool names must be unique" 400 on re-add).
                    _per_run_state[0].tool_lookup.pop(tool_name, None)
                    _per_run_state[0].active_names.discard(tool_name)
                    session_state.deny_tool(tool_name)
                    _disabled_obj = next(
                        (t for t in active_tools_list if getattr(t, "name", "") == tool_name),
                        None,
                    )
                    if _disabled_obj is not None:
                        with _bound_cache_lock:
                            try:
                                active_tools_list.remove(_disabled_obj)
                            except ValueError:
                                pass  # already removed by a concurrent invocation
                    _per_run_state[0].tool_version[0] += 1  # force bind_tools refresh
                    return ToolMessage(
                        content=(
                            f"Tool '{tool_name}' has been disabled after {_TOOL_BUDGET_HARD} calls "
                            f"and is no longer available. Please synthesize your findings into a "
                            f"final response now using the data you already have."
                        ),
                        tool_call_id=call["id"],
                        name=tool_name,
                    )

        tool_input = {**call, "type": "tool_call"}

        if tool_call_guard is not None:
            _guard_result = tool_call_guard(tool_name, call.get("args", {}))
            if hasattr(_guard_result, "is_safe") and not _guard_result.is_safe:
                log = get_logger()
                log.warning(
                    "Tool call blocked [%s]: %s — %s",
                    getattr(_guard_result, "guard_name", ""),
                    tool_name,
                    getattr(_guard_result, "reason", ""),
                )
                return ToolMessage(
                    content=(
                        f"Tool call blocked by security policy: "
                        f"{getattr(_guard_result, 'reason', 'blocked')}"
                    ),
                    tool_call_id=call["id"],
                    name=tool_name,
                )
        try:
            with _tool_budget_lock:
                tool = _per_run_state[0].tool_lookup.get(tool_name)
            if tool is None:
                return ToolMessage(
                    content=f"Tool '{tool_name}' is no longer active.",
                    tool_call_id=call["id"],
                    name=tool_name,
                )
            _corrected = _correct_tool_args(tool, call.get("args", {}))
            _corrected_input = {**tool_input, "args": _corrected}
            _tool_t0 = time.monotonic()
            with start_span(
                "src.orchestration.graph",
                "tool.call",
                attributes={"tool.name": tool_name},
            ) as _tool_span:
                try:
                    result = tool.invoke(_corrected_input, run_config)
                except Exception as exc:
                    _tool_span.record_exception(exc)
                    _tool_span.set_attribute("tool.status", "error")
                    _tool_span.set_attribute(
                        "tool.duration_ms", int((time.monotonic() - _tool_t0) * 1000)
                    )
                    _tool_span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise

                # Soft budget nudge: after N calls to the same tool, hint to synthesize.
                with _tool_budget_lock:
                    _cnt = _per_run_state[0].tool_call_counts.get(tool_name, 0)
                _nudge = ""
                if _cnt >= _TOOL_BUDGET_SOFT and tool_name not in _TOOL_BUDGET_SOFT_EXEMPT:
                    _nudge = (
                        f"\n\n[Note: You have called {tool_name} {_cnt} times this turn. "
                        "You likely have enough data — please synthesize your findings "
                        "into a complete response now rather than searching further.]"
                    )

                if isinstance(result, ToolMessage):
                    content = result.content if isinstance(result.content, str) else ""
                    if _nudge:
                        content += _nudge
                    content = _cap_history_tool_content(content)
                    if call_key is not None:
                        # Inlined from _store_call_result() so the history write
                        # and Event signalling happen atomically under _history_lock.
                        # Splitting them re-introduces the TOCTOU race (BUG-1293).
                        with _history_lock:
                            _per_run_state[0].tool_call_history[call_key] = content[:500]
                            _per_run_state[0].tool_call_history.move_to_end(call_key)
                            if len(_per_run_state[0].tool_call_history) > _MAX_TOOL_CALL_HISTORY:
                                _per_run_state[0].tool_call_history.popitem(last=False)
                            _event = _pending_events.pop(call_key, None)
                        if _event is not None:
                            _event.set()
                    result.content = content
                    _tool_span.set_attribute("tool.status", "success")
                    _tool_span.set_attribute(
                        "tool.duration_ms", int((time.monotonic() - _tool_t0) * 1000)
                    )
                    _tool_span.set_status(Status(StatusCode.OK))
                    return result
                text = str(result) if result is not None else ""
                text = _cap_history_tool_content(text)
                if call_key is not None:
                    # Inlined from _store_call_result() so the history write
                    # and Event signalling happen atomically under _history_lock.
                    # Splitting them re-introduces the TOCTOU race (BUG-1293).
                    with _history_lock:
                        _per_run_state[0].tool_call_history[call_key] = text[:500]
                        _per_run_state[0].tool_call_history.move_to_end(call_key)
                        if len(_per_run_state[0].tool_call_history) > _MAX_TOOL_CALL_HISTORY:
                            _per_run_state[0].tool_call_history.popitem(last=False)
                        _event = _pending_events.pop(call_key, None)
                    if _event is not None:
                        _event.set()
                _tool_span.set_attribute("tool.status", "success")
                _tool_span.set_attribute(
                    "tool.duration_ms", int((time.monotonic() - _tool_t0) * 1000)
                )
                _tool_span.set_status(Status(StatusCode.OK))
                return ToolMessage(
                    content=text,
                    tool_call_id=call["id"],
                    name=tool_name,
                )
        except UserCancelledRun:
            if call_key is not None:
                with _history_lock:
                    _event = _pending_events.pop(call_key, None)
                if _event is not None:
                    _event.set()
            raise
        except Exception as exc:
            if call_key is not None:
                with _history_lock:
                    _event = _pending_events.pop(call_key, None)
                if _event is not None:
                    _event.set()
            log = get_logger()
            log.warning("Tool %s raised: %s", tool_name, exc, exc_info=True)
            return ToolMessage(
                content=_cap_history_tool_content(f"Error executing {tool_name}: {exc}"),
                tool_call_id=call["id"],
                name=tool_name,
            )

    process_tools = build_process_tools_node(
        _invoke_one=_invoke_one,
        _tool_lookup=_per_run_state[0].tool_lookup,
        _active_names=_per_run_state[0].active_names,
        _available_tools_ref=_per_run_state[0].available_tools_ref,
        session_state=session_state,
        parallel_tool_execution=parallel_tool_execution,
        _identical_error_signature=_identical_error_signature,
        _tool_error_class=_tool_error_class,
        _tool_error_guidance=_tool_error_guidance,
        _last_identical_error_signature=_per_run_state[0].last_identical_error_signature,
        _consecutive_identical_error_count=_per_run_state[0].consecutive_identical_error_count,
        _force_thinking_break=_per_run_state[0].force_thinking_break,
        _graph_log=_graph_log,
        protected=protected,
        tool_catalog=_per_run_state[0].tool_catalog,
        registry=registry,
        approvals=approvals,
        confirmation_ui=confirmation_ui,
        git_native=git_native,
        on_tool_expansion=on_tool_expansion,
        output_cap=output_cap,
        expansion_count=_per_run_state[0].expansion_count,
        auto_expansion_count=_per_run_state[0].auto_expansion_count,
        request_tools_noop_count=_per_run_state[0].request_tools_noop_count,
        _MAX_REQUEST_TOOLS_NOOPS=_MAX_REQUEST_TOOLS_NOOPS,
        active_tools_list=active_tools_list,
        _tool_version=_per_run_state[0].tool_version,
        _calls_since_last_checkpoint=_per_run_state[0].calls_since_last_checkpoint,
        _same_file_writes=_per_run_state[0].same_file_writes,
        _same_file_writes_lock=_same_file_writes_lock,
        _REWRITE_SEARCH_THRESHOLD=_REWRITE_SEARCH_THRESHOLD,
        _consecutive_errors=_per_run_state[0].consecutive_errors,
        _STUCK_THRESHOLD=_STUCK_THRESHOLD,
        _stuck_detection_headline=_stuck_detection_headline,
        _get_tool_executor=lambda: _get_tool_executor(),
        _detect_tool_request=_detect_tool_request,
        _safe_tool_name=_safe_tool_name,
        tool_trust=config.tool_trust if config is not None else None,
    )

    def route_after_model(state: CogtrixState) -> str:
        msgs = state["messages"]
        if not msgs:
            return END

        last = msgs[-1]
        if isinstance(last, AIMessage):
            content = getattr(last, "content", "")
            has_content = isinstance(content, str) and bool(content.strip())
            tool_calls = getattr(last, "tool_calls", None)
            meta = getattr(last, "response_metadata", None)

            if not has_content and not tool_calls:
                if meta and isinstance(meta, dict):
                    if meta.get("finish_reason") == "tool_calls":
                        return "handle_phantom"
                return END

            if meta and isinstance(meta, dict) and meta.get("budget_guard"):
                return END

            if tool_calls:
                return "process_tools"

            if _looks_like_phantom_tool_markup(last):
                return "handle_phantom"

            if _looks_like_markdown_phantom_report(last):
                return "handle_phantom"

            if _looks_like_fabricated_success_after_tool_errors(msgs, last):
                return "handle_fabrication"

            # Has content but no tool calls — check for intention-without-action.
            # Suppress the nudge when the agent is responding to an access-denied
            # tool failure: the model is offering alternatives, not planning to act.
            # Nudging it again causes a counterproductive retry of the same blocked path.
            if _is_action_intent(last):
                msgs = state.get("messages", [])
                recent_tool_errors = [
                    getattr(m, "content", "") or "" for m in msgs[-6:] if hasattr(m, "tool_call_id")
                ]
                if any(
                    "Access denied" in err or "path outside allowed" in err
                    for err in recent_tool_errors
                ):
                    pass  # skip nudge — agent handled the error gracefully
                else:
                    return "handle_action_intent"

            # Hallucinated completion: model wrote a past-tense summary
            # ("Notified the VP...") claiming it called a tool that it never
            # actually invoked. Route through the same retry/synthesis path
            # so the model gets a chance to execute the missing step.
            _available_names = [getattr(t, "name", "") for t in (active_tools_list or [])]
            if _is_hallucinated_completion(last, msgs, _available_names):
                return "handle_action_intent"

        return END

    def route_after_phantom(state: CogtrixState) -> str:
        if _per_run_state[0].phantom_count[0] > _MAX_PHANTOM_RETRIES:
            return END
        return "call_model"

    def route_after_action_intent(state: CogtrixState) -> str:  # noqa: ARG001
        if _per_run_state[0].action_intent_count[0] > _MAX_ACTION_INTENT_RETRIES:
            # Standard retries exhausted.  Before ending, check whether
            # the model used incompleteness language ("first", "to start")
            # — a strong signal that it planned more steps but stopped.
            # Give exactly one more chance with a targeted nudge.
            if _per_run_state[0].incompleteness_nudge_given[0] < _MAX_INCOMPLETENESS_NUDGES:
                msgs = state.get("messages") or []
                last = msgs[-1] if msgs else None
                content = getattr(last, "content", "") if last is not None else ""
                if isinstance(content, str) and _has_incompleteness_signal(content):
                    _per_run_state[0].incompleteness_nudge_given[0] += 1
                    return "handle_incompleteness"
            return END
        return "call_model"

    def route_after_fabrication(state: CogtrixState) -> str:  # noqa: ARG001
        if _per_run_state[0].fabrication_count[0] > _MAX_FABRICATION_RETRIES:
            return END
        return "call_model"

    def _reset_for_new_run(
        new_available_tools: dict,
        new_bound_cache: "OrderedDict",
        new_compression_cache: dict,
        extend_run_state: Any = None,
    ) -> None:
        """Reset all per-run mutable state so the compiled graph can be reused.

        Called by ``run_agent()`` when the graph fingerprint matches the
        cached graph.  A fresh ``PerRunState`` instance is built and its
        values are copied in-place into the existing instance so that
        closures holding direct references to mutable fields still see
        the reset values.  Any new field added to ``PerRunState`` is
        automatically handled — no manual reset line required.
        """
        _fresh_tool_lookup = {
            getattr(t, "name", ""): t for t in active_tools_list if getattr(t, "name", "")
        }
        fresh = PerRunState(
            tool_lookup=_fresh_tool_lookup,
            active_names=set(_fresh_tool_lookup.keys()),
            tool_catalog=build_tool_catalog(new_available_tools),
            available_tools_ref=[new_available_tools],
            bound_cache=(new_bound_cache if new_bound_cache is not None else OrderedDict()),
            compression_cache=(new_compression_cache if new_compression_cache is not None else {}),
            tool_version=[_per_run_state[0].tool_version[0] + 1],
            last_tool_version=[-1],
        )

        # Copy fresh values into the existing PerRunState instance in-place.
        # This preserves object identity so closures that captured direct
        # references to mutable fields (e.g. call_count, bound_cache) still
        # see the reset values.
        for _f in fields(PerRunState):
            _current = getattr(_per_run_state[0], _f.name)
            _new = getattr(fresh, _f.name)
            if isinstance(_current, list):
                _current[:] = _new
            elif isinstance(_current, (dict, OrderedDict, set)):
                _current.clear()
                if _new:
                    _current.update(_new)
            else:
                setattr(_per_run_state[0], _f.name, _new)

        with _history_lock:
            _pending_events.clear()

        with _checkpoint_store_lock:
            _checkpoint_store.clear()

        if extend_run_state is not None:
            extend_run_state_ref[0] = extend_run_state

    graph: Any = StateGraph(CogtrixState)
    graph.add_node("call_model", call_model)
    graph.add_node("handle_phantom", handle_phantom)
    graph.add_node("handle_fabrication", handle_fabrication)
    graph.add_node("handle_action_intent", handle_action_intent)
    graph.add_node("handle_incompleteness", handle_incompleteness)
    graph.add_node("process_tools", process_tools)
    graph.set_entry_point("call_model")
    graph.add_conditional_edges(
        "call_model",
        route_after_model,
        {
            "process_tools": "process_tools",
            "handle_phantom": "handle_phantom",
            "handle_fabrication": "handle_fabrication",
            "handle_action_intent": "handle_action_intent",
            "handle_incompleteness": "handle_incompleteness",
            END: END,
        },
    )
    graph.add_edge("process_tools", "call_model")
    graph.add_conditional_edges(
        "handle_phantom",
        route_after_phantom,
        {"call_model": "call_model", END: END},
    )
    graph.add_conditional_edges(
        "handle_action_intent",
        route_after_action_intent,
        {"call_model": "call_model", END: END},
    )
    graph.add_conditional_edges(
        "handle_fabrication",
        route_after_fabrication,
        {"call_model": "call_model", END: END},
    )
    # handle_incompleteness always routes back to call_model — exactly
    # one chance to finish the task after a stronger nudge.
    graph.add_edge("handle_incompleteness", "call_model")
    compiled = graph.compile()
    compiled._reset_for_new_run = _reset_for_new_run  # type: ignore[attr-defined]
    compiled._per_run_state = _per_run_state  # type: ignore[attr-defined]
    return compiled
