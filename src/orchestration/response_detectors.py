"""Response-content detectors for the orchestration router.

Extracted from ``src/orchestration/graph.py`` as the second step of the
graph.py 5-module split proposed by the /forge audit
(architect finding A1.2, 2026-05-23). All six detectors operate on a
single AIMessage's content (plus, in two cases, the surrounding
``messages`` list) and return ``bool``. They have no graph-build /
langgraph-runtime dependency.

Detectors (in order of original definition):

* :func:`_is_action_intent` — sentence-local intent-lead + tool-verb
  match. Used to catch "I'll create the file" with no tool call.
* :func:`_has_incompleteness_signal` — narrows intent detection to
  multi-step sentences ("first", "to start", "step 1").
* :func:`_is_hallucinated_completion` — past-tense list-item verbs
  whose stem matches an *available* but never-called tool name.
* :func:`_looks_like_phantom_tool_markup` — XML or JSON-array tool-call
  markup emitted as final text.
* :func:`_looks_like_markdown_phantom_report` — fabricated structured
  markdown report (table + numbered section + claim-of-action signal).
* :func:`_looks_like_fabricated_success_after_tool_errors` — explicit
  success claim immediately after a contiguous block of tool errors.

Also re-exported: the regex constants each detector consumes, the
``_unwrap_code_fence`` helper, and ``_stuck_detection_headline``
(used both inside this module and by other graph.py code paths —
graph.py re-imports it).

Leading underscores on names are preserved from the original graph.py
module for back-compat with existing call sites in the orchestration
nodes and the test suite.
"""

from __future__ import annotations

import re
import typing
from typing import Any

# ── _is_action_intent regex set ───────────────────────────────────────

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


# ── _has_incompleteness_signal ────────────────────────────────────────

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


# ── _is_hallucinated_completion ───────────────────────────────────────

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


# ── _looks_like_phantom_tool_markup ───────────────────────────────────

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


# ── _looks_like_markdown_phantom_report + success/error helpers ───────

_MARKDOWN_TABLE_ROW_RE = re.compile(r"^\s*\|.+\|.+\|", re.MULTILINE)
_NUMBERED_SECTION_RE = re.compile(r"^#{1,4}\s+\d+\.", re.MULTILINE)

# Fake-tool-output content signals. The original
# ``_looks_like_markdown_phantom_report`` fired on *any* response with a
# markdown table + numbered headings, which caught every well-structured
# educational answer (cogtrix45.log turn 3 — Bug K). The fix is to also
# require a "claim-of-action" content signal so the detector matches
# fabricated tool reports without tripping on legitimate prose that
# happens to be well structured.
#
# Each alternative below corresponds to a distinct fabrication pattern:
#
# 1. Bullet-point past-tense claim with a retrieval verb:
#    ``- Retrieved last 8 messages``
# 2. First-person past-tense action with a typical object phrase:
#    ``I retrieved the 8 most recent messages``
# 3. Report-style section headers:
#    ``Sources:``, ``References:``, ``Tool Results:``, ``Search Results:``
# 4. "According to" tool/search/results claim:
#    ``According to my search, …``
# 5. Results/search/data acting as the subject of a reporting verb:
#    ``The results show X``, ``Search returned Y``
#
# What this deliberately does NOT match (the false-positive corpus):
# - Tutorials describing a *technique* in imperative or infinitive form
#   ("Before each turn, retrieve top-3 chunks") — verb not at bullet
#   start and not in first-person past tense.
# - Reflective prose ("Here are the core strategies I use") — "I use"
#   is not in the past-tense verb list.
# - Apologetic prefixes ("I'll address it directly") — same reason.
_FAKE_TOOL_OUTPUT_SIGNAL_RE = re.compile(
    r"(?im)"
    # 1. Bullet-point past-tense claim of action.
    r"^\s*[-*]\s+(?:Retrieved|Fetched|Found|Got|Pulled|Searched|Queried|Returned|Listed)\s+\S"
    r"|"
    # 2. First-person past-tense + typical object phrase ("I retrieved
    #    the/a/N/all/recent …"). The trailing alternation prevents
    #    matching educational phrasing like "I find that …".
    r"\bI\s+(?:retrieved|fetched|queried|searched|found|looked\s+up|checked|ran|executed|pulled)"
    r"\s+(?:the|a|an|\d+|all|some|recent|last|top)\b"
    r"|"
    # 3. Report-style section headers (line on its own).
    r"^\s*(?:Sources?|References?|Search Results?|Tool Results?|Output)\s*:\s*$"
    r"|"
    # 4. "According to my/the search/results/findings".
    r"\baccording\s+to\s+(?:my|the)\s+(?:search|query|results?|findings?)\b"
    r"|"
    # 5. Reporting-verb structure with results/search/data as subject.
    r"\b(?:results?|search|query|data)\s+(?:show(?:s|ed)?|indicate(?:s|d)?|return(?:ed|s)?)\b"
)

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

    Detection signal — all three required:

    1. A markdown table row, and
    2. A numbered section header, and
    3. A claim-of-action content signal (see
       ``_FAKE_TOOL_OUTPUT_SIGNAL_RE``).

    The third condition was added after Bug K (cogtrix45.log turn 3): the
    original two-signal heuristic misclassified well-structured *educational*
    answers as fabricated tool reports, causing the agent to topic-drift back
    to stale conversation history. Fabricated reports almost always contain
    a past-tense claim of retrieval ("Retrieved last 8 messages") or a
    report-style header ("Sources:") — the new signal captures that intent
    without tripping on tutorials that compare techniques in tables.
    """
    if getattr(message, "tool_calls", None):
        return False
    content = getattr(message, "content", "")
    if not isinstance(content, str) or len(content) < 80:
        return False
    return bool(
        _MARKDOWN_TABLE_ROW_RE.search(content)
        and _NUMBERED_SECTION_RE.search(content)
        and _FAKE_TOOL_OUTPUT_SIGNAL_RE.search(content)
    )


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


__all__ = [
    "_CODE_FENCE_RE",
    "_FAKE_TOOL_OUTPUT_SIGNAL_RE",
    "_INCOMPLETENESS_SIGNAL_RE",
    "_INTENT_FALSE_POSITIVE_RE",
    "_INTENT_LEAD_RE",
    "_MARKDOWN_TABLE_ROW_RE",
    "_NEGATED_SUCCESS_RE",
    "_NUMBERED_SECTION_RE",
    "_PAST_TENSE_LIST_VERB_RE",
    "_PHANTOM_JSON_TOOL_RE",
    "_PHANTOM_TOOL_MARKUP_RE",
    "_SUCCESS_CLAIM_RE",
    "_TOOL_ERROR_INDICATORS",
    "_TOOL_VERB_RE",
    "_has_incompleteness_signal",
    "_is_action_intent",
    "_is_hallucinated_completion",
    "_looks_like_fabricated_success_after_tool_errors",
    "_looks_like_markdown_phantom_report",
    "_looks_like_phantom_tool_markup",
    "_stuck_detection_headline",
    "_unwrap_code_fence",
]
