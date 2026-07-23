"""Unverified-claim safety guard.

The agent sometimes answers with a categorical claim about external
state — *"The latest release is X"*, *"The file says Y"*, *"The
USD/GBP rate is Z"* — without calling the verification tool that
would actually fetch ground truth. This module routes such
responses back to a recovery node that nudges the model to verify
before answering.

Why ``date_claim`` is NOT here
============================================================
A prior iteration of this module included a ``date_claim`` rule
that fired when the response named a calendar date without a
preceding ``get_current_datetime`` call. We removed it: the
orchestration already injects today's date into the system prompt
(``src/agent/core.py:build_system_prompt``) AND prefixes every
``HumanMessage`` with ``[YYYY-MM-DD HH:MM:SS UTC]``
(``src/memory/manager.BaseMemoryManager._inject_timestamps``).
A response that names today's date is repeating ground truth that
the agent was just handed — calling ``get_current_datetime`` would
return the same value. The tool call is pure friction.

Rules in this module target categories where the agent does NOT
have an injected ground-truth source:

* **weather_claim** — current/forecast weather, requires
  ``get_weather`` (preferred) or ``web_search``.
* **exchange_rate_claim** — FX conversion rate, requires
  ``web_search``.
* **latest_version_claim** — "latest version of X", requires
  ``web_search``.
* **file_content_claim** — assertions about a specific file's
  contents, require ``read_file`` (or a sibling reader).

Design properties
============================================================

* **Rule registry.** Each ``VerificationRule`` declares a regex
  for the claim shape, the required tool name (or alternatives),
  and the recovery nudge text. Adding a new rule is a single
  registry entry — no orchestration changes needed.
* **Conservative detection.** Each rule's regex requires both a
  *claim phrase* and an *evidence token* (a real number, version
  string, filename, etc.) so legitimate prose paraphrase doesn't
  trip.
* **Bounded retries.** One revision attempt; after that the
  agent's answer ships as-is — we'd rather surface a possibly-
  stale answer than loop forever.

Out of scope here (deliberate)
============================================================

* No content rewriting. We never modify the model's text directly
  — only re-prompt and let the model produce a new answer.
* No semantic verification. We don't check whether the *value* the
  agent claimed matches the *value* the tool would return; we only
  check whether the tool was called. The tool's job is to provide
  ground truth; the agent's job is to use it. Mismatch between the
  agent's claim and the tool's response is a separate bug class
  (fabricated-success-after-tool-error, which has its own
  detector).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# ── Shared helpers ────────────────────────────────────────────────────

# Currency / FX context cues that distinguish a real exchange-rate
# claim from prose that happens to mention numbers. Money-symbol
# tokens (``$``, ``€``, ``£``, ``¥``, ``₹``) and 3-letter ISO codes
# cover the common shapes search results emit.
_CURRENCY_SYMBOL = r"[$€£¥₹]"
_CURRENCY_CODE = (
    r"USD|EUR|GBP|NZD|AUD|CAD|JPY|CNY|INR|CHF|SGD|HKD|MXN|BRL|"
    r"ZAR|SEK|NOK|DKK|PLN|CZK|TRY|RUB|KRW"
)

# Semantic-version token: 1.2.3 / 1.2 / v1.2.3 / v3.0.0-rc1 — anchored
# enough that ordinary numeric prose ("about 1.2 million") can't pass.
_VERSION_TOKEN = r"\bv?\d+\.\d+(?:\.\d+)?(?:-(?:rc|beta|alpha|dev|pre)\.?\d*)?\b"


# ── weather_claim ─────────────────────────────────────────────────────
# Fires on shapes like "It is sunny in London", "Forecast: 22°C
# tomorrow", "Light rain expected". Requires a real meteorological
# token (temperature reading, weather condition word) — generic prose
# mentioning "weather" without specifics doesn't trip.

_WEATHER_CONDITION = (
    r"(?:sunny|cloudy|overcast|rainy?|raining|drizzl(?:e|ing)|"
    r"snow(?:y|ing)?|fog(?:gy)?|mist(?:y)?|haz(?:y|e)|"
    r"thunderstorms?|stormy?|wind(?:y)?|gust(?:y|s)|hail|sleet|"
    r"clear(?:\s+skies)?|partly\s+cloud(?:y|less)|"
    r"humid(?:ity)?|dry|chilly|cold|warm|hot|mild|freezing)"
)
# Forecaster idiom — "high of 22", "low of 11". Domain-specific enough
# that it stays a standalone weather signal without a sibling context
# word. ("Low of 22" can appear outside weather — golf scores, etc. —
# but in those domains the surrounding prose virtually always names
# the domain, and the false-positive cost is bounded.)
_FORECAST_IDIOM = r"(?:high|low)\s+of\s+\d{1,3}(?:\s*°?\s*[CF]?)?\b"

# Raw degree reading — "22°C", "72°F", "-5°C". Bare temperatures must
# NOT be a standalone weather signal: the same shape appears in
# cooking recipes ("internal temp 165°F"), oven temps ("350°F"),
# medical ("fever of 38°C"), engineering ("engine at 80°C"). We
# only treat them as weather when a weather-context word is nearby
# (see _WEATHER_CONTEXT below). This avoids the cogtrix55 regression
# where a recipe answer tripped weather_claim and the orchestrator
# dropped the response.
_BARE_TEMPERATURE = r"(?:[-+]?\d{1,3}\s*°\s*[CF]\b|\d{1,3}\s*°\b)"

# Weather-specific context words. Deliberately NOT including the
# singular "temperature" (which is mostly used in non-weather senses:
# body temperature, internal temperature, engine temperature). The
# plural "temperatures" overwhelmingly refers to weather/climate, so
# it IS included.
_WEATHER_CONTEXT = (
    r"(?:weather|forecast|outside|outdoor(?:s)?|"
    r"today|tomorrow|tonight|this\s+(?:morning|afternoon|evening)|"
    r"currently|right\s+now|temperatures|"
    r"sunrise|sunset|humidity|precipitation|"
    r"wind\s+speed|chance\s+of\s+(?:rain|snow))"
)

_WEATHER_CLAIM_RE = re.compile(
    r"(?im)"
    # Direct condition assertions
    rf"\b(?:it|today|tomorrow|forecast|weather)\b.{{0,50}}?{_WEATHER_CONDITION}"
    r"|"
    rf"{_WEATHER_CONDITION}.{{0,50}}?\b(?:today|tomorrow|currently|now)\b"
    r"|"
    # Forecaster idiom ("high of 22", "low of 11") — standalone
    rf"{_FORECAST_IDIOM}"
    r"|"
    # Bare temperature reading with weather context preceding
    rf"\b{_WEATHER_CONTEXT}\b.{{0,80}}?{_BARE_TEMPERATURE}"
    r"|"
    # Bare temperature reading with weather context following
    rf"{_BARE_TEMPERATURE}.{{0,30}}?\b{_WEATHER_CONTEXT}\b"
)


# ── exchange_rate_claim ───────────────────────────────────────────────
# Fires on shapes like "1 USD = 0.74 GBP", "100 NZD ≈ €50",
# "the exchange rate is 1.23". Requires both a currency context AND
# a numeric value, so prose like "exchange rates fluctuate daily"
# doesn't match.

_EXCHANGE_RATE_CLAIM_RE = re.compile(
    r"(?im)"
    # "1 USD = 0.74 GBP" / "100 NZD = €50" — explicit conversion form
    rf"\b\d+(?:[.,]\d+)?\s*(?:{_CURRENCY_CODE}|{_CURRENCY_SYMBOL})\s*"
    rf"(?:=|≈|to|→|->|/)\s*"
    rf"(?:{_CURRENCY_SYMBOL}\s*)?\d+(?:[.,]\d+)?\s*"
    rf"(?:{_CURRENCY_CODE})?\b"
    r"|"
    # "exchange rate is 0.74"
    rf"\b(?:exchange|conversion|fx|forex|spot)\s+rate\s+"
    rf"(?:is|of|=|≈)\s*[~]?\s*\d+(?:[.,]\d+)?"
    r"|"
    # "USD/GBP at 0.74" / "GBP-USD rate 1.23"
    rf"\b(?:{_CURRENCY_CODE})[/\-](?:{_CURRENCY_CODE})\s+"
    rf"(?:rate\s+|at\s+)?\d+(?:[.,]\d+)?"
)


# ── latest_version_claim ──────────────────────────────────────────────
# Fires on shapes like "The latest version of Python is 3.13",
# "Released version 2.1.0", "current release is v4.2". The version
# token is required so prose like "we use the latest version" (no
# specific version named) doesn't trip — that's a vague reference,
# not a verifiable claim.

_LATEST_VERSION_CLAIM_RE = re.compile(
    rf"(?im)"
    rf"\b(?:the\s+)?(?:latest|current|newest|stable|most\s+recent)\s+"
    rf"(?:released?\s+)?(?:version|release|build)\s+"
    rf"(?:of\s+\S+\s+)?(?:is|=|:)?\s*{_VERSION_TOKEN}"
    r"|"
    rf"\b(?:released?|published?|shipped)\s+(?:as\s+)?{_VERSION_TOKEN}"
    r"|"
    rf"\b{_VERSION_TOKEN}\s+(?:is|was)\s+(?:the\s+)?"
    rf"(?:latest|current|newest|stable|most\s+recent)\b"
)


# ── file_content_claim ────────────────────────────────────────────────
# Fires on shapes like "The file config.yaml contains X",
# "src/foo.py defines bar", "the README says Y". Requires a file-path
# shape (extension OR path separator) so prose like "the file is
# small" doesn't match.
#
# Required tool: read_file (or a sibling reader — see required_tools).

_FILE_PATH_TOKEN = (
    # path/to/file.ext (any common ext) OR just a filename with a
    # known extension. Backticks tolerated.
    r"`?(?:[\w/.\-]+/)?[\w\-]+\.(?:py|js|ts|tsx|jsx|json|ya?ml|"
    r"toml|md|txt|cfg|ini|sh|bash|zsh|fish|rs|go|rb|php|java|c|cpp|h|hpp|"
    r"html?|css|scss|less|sql|xml|csv|tsv|log|env|lock|gitignore|"
    r"dockerfile|makefile|cmake)`?"
)
_FILE_CONTENT_CLAIM_RE = re.compile(
    r"(?im)"
    rf"\b{_FILE_PATH_TOKEN}\s+(?:contains?|defines?|declares?|"
    rf"says?|reads?|sets?|specifies|exports?|imports?|includes?|has)\b"
    r"|"
    rf"\b(?:the\s+|inside\s+|in\s+)?{_FILE_PATH_TOKEN}\s*[,:]"
    rf"\s+(?:the|it)\s+(?:contains?|defines?|says?|reads?|sets?|"
    rf"declares?|specifies|exports?|has)\b"
    r"|"
    rf"\baccording\s+to\s+{_FILE_PATH_TOKEN}\b"
)


@dataclass(frozen=True)
class VerificationRule:
    """One rule in the verification registry.

    Adding a new claim category (time, exchange rate, version, file
    content, etc.) means adding a ``VerificationRule`` to
    ``VERIFICATION_RULES`` — nothing else needs to change.

    Attributes
    ----------
    name:
        Short identifier surfaced in log messages and recovery hints.
    claim_re:
        Compiled regex matching the *claim shape* in the response
        text. Must require an actual evidence token (number, version,
        filename, etc.), not just a verb phrase, to avoid false
        positives on legitimate prose.
    required_tools:
        Tuple of tool names that satisfy the verification — ANY ONE
        of these called this turn marks the claim as verified. A
        single-tool rule passes ``("only_acceptable_tool",)``.
        Multi-tool rules let the agent satisfy the guard via
        sibling readers (e.g. ``read_file`` or ``view_file``).
    nudge_template:
        The recovery prompt injected when the rule fires. Should
        explain *what* claim was caught and *which* tool to call.
        The phrase MUST end with explicit instruction so the model
        doesn't repeat the unverified answer.
    """

    name: str
    claim_re: re.Pattern[str]
    required_tools: tuple[str, ...]
    nudge_template: str

    @property
    def required_tool(self) -> str:
        """Back-compat accessor for callers that read a single tool name.

        Returns the first acceptable tool. Existing code paths that
        log or render ``rule.required_tool`` keep working without
        change; new code should prefer ``rule.required_tools``.
        """
        return self.required_tools[0]


# ── GroundedSources value object (#1964 Item C) ───────────────────────
#
# The grounding-aware detectors below — ``detect_unverified_entities``,
# ``detect_unsupported_quote``, ``detect_unsupported_attribution`` —
# answer the same shape of question: *is this span in the response
# supported by anything the agent could legitimately cite this turn?*
#
# Pre-#1964 each detector took its grounding inputs as separate kwargs
# (``tool_message_contents``, ``user_prompt``) and the **system prompt
# was not threaded in at all**.  That left a structural gap: when the
# agent quotes its own persona's policy statement in a refusal — *"per
# our policy, pay_invoice MUST NEVER be called unless …"* — the quote
# IS grounded (the system prompt is the agent's source of truth) but
# the detector only saw the tool blob and false-fired.  PRs #1961 and
# #1962 papered over that with refusal-aware short-circuits, but the
# real fix is to give the detectors access to the system prompt as a
# first-class grounding source.
#
# ``GroundedSources`` bundles the three grounding inputs into one value
# object so:
#
#   * adding a new grounding source (e.g. RAG retrieval results) only
#     touches this dataclass + ``iter_text()``, not every detector;
#   * detectors uniformly access ``.iter_text()`` instead of re-stitching
#     three separate iterables;
#   * the contract is explicit at the call site — readers see exactly
#     which sources the detector considers grounding.
#
# The detectors accept ``sources: GroundedSources | None`` (new path) and
# keep the original kwargs for back-compat.  When ``sources`` is given,
# the old kwargs are ignored.
@dataclass(frozen=True, slots=True)
class GroundedSources:
    """Bundle of text the agent may legitimately cite as a grounding source.

    Attributes
    ----------
    tool_results:
        Iterable of ToolMessage content strings from the current turn.
        The ground-truth corpus the agent fetched on this turn.
    user_prompt:
        The user's most recent message text.  Quoting the user's own
        words back is legitimate, so it counts as grounding.
    system_prompt:
        The agent's persona / system prompt.  Refusals and policy
        explanations frequently quote the system prompt's policy
        statement — that IS grounded by construction, since the system
        prompt is the agent's declared source of truth.  Empty string
        when not available.
    """

    tool_results: tuple[str, ...] = ()
    user_prompt: str = ""
    system_prompt: str = ""

    @classmethod
    def from_inputs(
        cls,
        *,
        tool_message_contents: Iterable[str] | None = None,
        user_prompt: str = "",
        system_prompt: str = "",
    ) -> GroundedSources:
        """Build a ``GroundedSources`` from raw inputs, normalising types.

        Non-string entries in ``tool_message_contents`` are filtered out
        (the existing detectors already did this defensively).
        """
        contents = tuple(c for c in (tool_message_contents or ()) if isinstance(c, str))
        return cls(
            tool_results=contents,
            user_prompt=user_prompt or "",
            system_prompt=system_prompt or "",
        )

    def iter_text(self) -> list[str]:
        """Yield the grounding texts in priority order.

        Order: tool results first (most specific, this-turn evidence),
        then user prompt (user's own words), then system prompt (the
        agent's persona / policy statements).  Callers that need a
        single blob ``"\\n".join(sources.iter_text())``.
        """
        out: list[str] = list(self.tool_results)
        if self.user_prompt:
            out.append(self.user_prompt)
        if self.system_prompt:
            out.append(self.system_prompt)
        return out

    @property
    def blob(self) -> str:
        """Concatenated grounding text — one newline-separated string."""
        return "\n".join(self.iter_text())


def collect_grounded_sources(
    messages: list[Any],
    turn_start_idx: int,
) -> GroundedSources:
    """Build a ``GroundedSources`` from the conversation state.

    Walks the messages list to gather:

    * every ToolMessage content emitted since ``turn_start_idx``
      (the current turn's evidence);
    * the user prompt that opened the current turn (``messages[turn_start_idx]``);
    * the first SystemMessage in the conversation, if any (the agent's
      persona / system prompt).

    Pure walk — does not import LangChain message classes; relies on
    duck-typed attribute checks so the helper stays usable from tests
    that pass plain stand-in objects.
    """
    # User prompt — the HumanMessage at the turn boundary.
    user_prompt = ""
    if 0 <= turn_start_idx < len(messages):
        up_content = getattr(messages[turn_start_idx], "content", "") or ""
        if isinstance(up_content, str):
            user_prompt = up_content

    # System prompt — first message whose type is "system".  We don't
    # import ``SystemMessage`` to keep this helper independent of the
    # LangChain message class hierarchy; ``type`` is the public protocol.
    system_prompt = ""
    for msg in messages:
        msg_type = getattr(msg, "type", None)
        if msg_type == "system":
            sp_content = getattr(msg, "content", "") or ""
            if isinstance(sp_content, str):
                system_prompt = sp_content
            break

    # Tool results — every ToolMessage in the current turn.
    tool_contents: list[str] = []
    if 0 <= turn_start_idx < len(messages):
        for msg in messages[turn_start_idx:]:
            if not hasattr(msg, "tool_call_id"):
                continue
            content = getattr(msg, "content", "") or ""
            if isinstance(content, str):
                tool_contents.append(content)

    return GroundedSources(
        tool_results=tuple(tool_contents),
        user_prompt=user_prompt,
        system_prompt=system_prompt,
    )


# ── GroundedDetector protocol + registry (#1964 Item B) ───────────────
#
# Pre-#1964 every grounding-aware detector had a slightly different
# parameter list — ``detect_unverified_entities(response, user_prompt,
# tool_contents)`` vs ``detect_unsupported_quote(response, tool_contents,
# user_prompt)`` vs ``detect_unsupported_attribution(response,
# tool_contents, user_prompt)`` — and the **system prompt wasn't
# threaded in at all**.  Adding a new detector meant inventing a new
# kwarg shape, picking up some grounding sources but not others, and
# wiring an inline conditional in ``route_after_model``.  The drift was
# the structural cause of the #1960 false-fire class.
#
# This section formalises the contract:
#
# * ``GroundedDetector`` (Protocol) — the call shape every grounding-
#   aware detector must satisfy.  ``response_content`` positionally,
#   ``sources: GroundedSources`` keyword-only, ``**kwargs`` reserved
#   for detector-specific tuning knobs.  Returns ``list[str]`` of
#   offending spans (empty list = no fire).
#
# * ``GroundedDetectorSpec`` — metadata bundle: detector callable +
#   canonical name + handler-node identifier + which
#   ``GroundedSources`` fields the detector actually consumes (so the
#   "what grounding does this detector see?" question is answered
#   declaratively at registration time, not by reading the
#   implementation).
#
# * ``GROUNDED_DETECTORS`` — registry tuple.  Future detectors register
#   here; the orchestration router does NOT yet iterate over the
#   registry (that's a follow-up — it would touch every inline
#   conditional in ``route_after_model`` and warrants its own
#   architectural review).  Today the registry serves three purposes:
#
#     1. A discovery surface — readers see all grounding-aware
#        detectors in one place with their grounding-source declarations.
#     2. A test anchor — ``test_grounded_detector_protocol.py`` asserts
#        every entry conforms to the protocol shape.
#     3. A migration target — the runtime conditional in
#        ``route_after_model`` can be rewritten in a future PR to
#        iterate over ``GROUNDED_DETECTORS`` once the cascade-budget
#        observability shows the inline form's behaviour matches the
#        registry-driven form's behaviour exactly.


@runtime_checkable
class GroundedDetector(Protocol):
    """Call-shape contract for grounding-aware detectors.

    Every detector that checks the agent's response against
    ``GroundedSources`` MUST satisfy this protocol.  Concrete return
    shape is detector-specific (``list[str]`` of offending entities,
    quoted spans, paragraphs) but the shape is uniformly a list — an
    empty list means *no fire*.

    The ``sources`` parameter is **keyword-only** so callers cannot
    accidentally swap positional grounding-source arguments — the bug
    class that drove this protocol formalisation.

    ``**kwargs`` is reserved for detector-specific tuning knobs
    (``max_returned``, ``min_quote_words``, etc.) that callers should
    not normally need to override.
    """

    def __call__(
        self,
        response_content: str,
        *,
        sources: GroundedSources,
        **kwargs: Any,
    ) -> list[str]: ...


@dataclass(frozen=True, slots=True)
class GroundedDetectorSpec:
    """Metadata for one entry in the ``GROUNDED_DETECTORS`` registry.

    Attributes
    ----------
    name:
        Canonical detector name surfaced in logs and the
        ``recovery_firings_history`` observability stream.  Matches
        the ``handle_<name>`` recovery-node identifier with the
        ``handle_`` prefix stripped.
    detect:
        The detector callable.  Must satisfy the
        ``GroundedDetector`` protocol.
    handler_node:
        The router branch this detector activates (``"handle_<name>"``).
    consumes_tool_results:
        ``True`` when the detector reads ``sources.tool_results``.
    consumes_user_prompt:
        ``True`` when the detector reads ``sources.user_prompt``.
        ``detect_unverified_entities`` reads it to extract candidates,
        not to *verify* against — flagged via the field below.
    consumes_system_prompt:
        ``True`` when the detector counts the system prompt as a
        grounding source for verification.  Pre-#1964 this was
        ``False`` for all three detectors (the structural gap).
    extracts_candidates_from_user_prompt:
        ``True`` for detectors that EXTRACT candidate spans from the
        user prompt (currently only ``detect_unverified_entities``).
        Such detectors must NOT include the user prompt in the
        verification blob — every candidate would self-match.
    issue_refs:
        Tracking issue numbers — handy when readers want the trail
        of why this detector exists.
    """

    name: str
    detect: Callable[..., list[str]]
    handler_node: str
    consumes_tool_results: bool = True
    consumes_user_prompt: bool = True
    consumes_system_prompt: bool = True
    extracts_candidates_from_user_prompt: bool = False
    issue_refs: tuple[int, ...] = field(default_factory=tuple)


VERIFICATION_RULES: tuple[VerificationRule, ...] = (
    VerificationRule(
        name="weather_claim",
        claim_re=_WEATHER_CLAIM_RE,
        # ``get_weather`` is the preferred verifier (structured,
        # current). ``web_search`` is the broad fallback for places
        # the dedicated tool can't reach (e.g. micro-regions).
        required_tools=("get_weather", "web_search"),
        nudge_template=(
            "Your response made a weather assertion (current conditions, "
            "forecast, or temperature) without calling `get_weather` or "
            "`web_search`. Weather data is time-sensitive — your training "
            "snapshot is not a reliable source. Call `get_weather` (or "
            "`web_search` as a fallback) now, then revise your answer "
            "using only the data the tool returns. If neither tool yields "
            "useful data, say so honestly rather than guessing."
        ),
    ),
    VerificationRule(
        name="exchange_rate_claim",
        claim_re=_EXCHANGE_RATE_CLAIM_RE,
        required_tools=("web_search",),
        nudge_template=(
            "Your response quoted a specific exchange / FX rate "
            "(e.g. ``1 USD = 0.74 GBP``) without calling `web_search` "
            "to fetch a current quote. FX rates change continuously — "
            "your training data is months or years out of date. Call "
            "`web_search` now to look up the rate, then revise your "
            "answer with the value the search returns. Cite the source "
            "and the time the rate was retrieved."
        ),
    ),
    VerificationRule(
        name="latest_version_claim",
        claim_re=_LATEST_VERSION_CLAIM_RE,
        required_tools=("web_search",),
        nudge_template=(
            "Your response asserted a specific latest / current version "
            "number without calling `web_search` to verify it. Releases "
            "ship constantly — the version you remember from training is "
            "almost certainly behind by now. Call `web_search` for the "
            "official release announcement (project changelog or release "
            "page), then revise the answer using the verified value."
        ),
    ),
    VerificationRule(
        name="file_content_claim",
        claim_re=_FILE_CONTENT_CLAIM_RE,
        # ``read_file`` is canonical. ``patch_file`` and similar readers
        # don't satisfy the guard — they don't return content. A future
        # ``view_file`` (read-only mirror) would be added here.
        required_tools=("read_file",),
        nudge_template=(
            "Your response claimed something about the contents of a "
            "specific file without calling `read_file` (or a sibling "
            "reader) to inspect it. You cannot describe what a file says "
            "from training data — files in this workspace are arbitrary "
            "and unknown to your training. Call `read_file` now on the "
            "exact path you cited, then revise the answer to quote or "
            "summarise what the tool actually returned."
        ),
    ),
)


def detect_unverified_claim(
    response_content: str,
    tool_names_called_this_turn: Iterable[str],
) -> VerificationRule | None:
    """Return the first ``VerificationRule`` that matched the response
    without its required tool having been called this turn, or ``None``.

    Parameters
    ----------
    response_content:
        The final AIMessage content text (not tool-call args, not
        prior turns — only the current response).
    tool_names_called_this_turn:
        Iterable of tool names invoked during the current turn. The
        guard checks this against each rule's ``required_tool``.

    Returns
    -------
    The first matching rule, or ``None`` if no unverified claim is
    present. We return the rule itself (not just the name) so the
    caller can read ``nudge_template`` for the recovery message.

    A rule is considered SATISFIED if ANY of its ``required_tools``
    was called this turn. Multi-tool rules let the agent satisfy
    the guard via sibling readers — e.g. ``weather_claim`` accepts
    either ``get_weather`` or ``web_search``.
    """
    if not response_content or not response_content.strip():
        return None

    called = set(tool_names_called_this_turn)
    for rule in VERIFICATION_RULES:
        if any(t in called for t in rule.required_tools):
            continue
        if rule.claim_re.search(response_content):
            return rule
    return None


def collect_tool_names_this_turn(messages: list[Any], turn_start_idx: int) -> list[str]:
    """Walk forward from ``turn_start_idx`` and collect the names of
    every tool that was called this turn.

    Parameters
    ----------
    messages:
        The full conversation message list (langchain-style).
    turn_start_idx:
        Index of the first message belonging to the current turn —
        typically the index of the user's HumanMessage that opened
        this turn. Messages strictly before this index belong to
        prior turns and are ignored.

    Returns
    -------
    Tool names in invocation order. A tool called twice is listed
    twice; deduplication is the caller's choice via ``set()``.
    """
    if turn_start_idx >= len(messages):
        return []

    names: list[str] = []
    for msg in messages[turn_start_idx:]:
        # Two shapes of tool-call evidence:
        # 1. AIMessage with tool_calls=[{"name": ..., "args": ...}, ...]
        # 2. ToolMessage with .name set (the response side; pair to a call)
        tcs = getattr(msg, "tool_calls", None)
        if tcs:
            for tc in tcs:
                if isinstance(tc, dict):
                    n = tc.get("name")
                    if isinstance(n, str) and n:
                        names.append(n)
        else:
            n = getattr(msg, "name", None)
            if isinstance(n, str) and n and hasattr(msg, "tool_call_id"):
                # ToolMessage path — the response side of a tool call.
                # We could double-count, but the call side is the
                # canonical signal so we prefer that. Only pick this
                # up if no tool_calls were captured upstream
                # (defensive — shouldn't happen in well-formed runs).
                if n not in names:
                    names.append(n)
    return names


# ── Unverified-entity detection (cogtrix47 Issues 5 + 6) ─────────────
#
# When the user supplies a high-specificity identifier in their prompt
# (a SKU, a specific retailer name, a multi-word product/service name)
# and the agent repeats it in the final response WITHOUT any tool
# result confirming the entity exists, the response is operating on an
# unverified premise. The cogtrix47 run did this with two such
# identifiers:
#
#   * ``Soudal Fix All Silirub 1GH-EJ4`` — SKU that appears in zero
#     search results; only ``Fix All Crystal`` / ``Sanitary Silicone``
#     came back.
#   * ``PowerTool shop by Praterstern`` — store name never confirmed
#     by any tool result.
#
# Three pattern categories anchor the detector. Each must be specific
# enough that ordinary prose (city names, single-word brands like
# ``Soudal`` or ``Vienna``) cannot trip:
#
#   1. SKU-shape — alphanumeric token with at least one digit AND one
#      hyphen, length 4-30 (catches ``1GH-EJ4``, ``ISO-9001``).
#   2. Qualifier phrase — ``<TitleCase> {shop|store|retailer|...}``
#      (catches ``PowerTool shop``).
#   3. Three or more consecutive TitleCase words (catches
#      ``Fix All Silirub``).

_SKU_RE = re.compile(
    # Lookahead: token must contain ≥1 digit AND ≥1 hyphen so we
    # don't match plain ALLCAPS words like "USA" or "EUR".
    r"\b(?=[A-Z0-9\-]*\d)(?=[A-Z0-9\-]*-)[A-Z0-9\-]{4,30}\b"
)

_STORE_QUALIFIER_RE = re.compile(
    r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)"
    r"\s+(?:shop|store|retailer|outlet|warehouse|depot|"
    r"branch|location|office|showroom|boutique|kiosk|stall)\b"
)

_MULTI_TITLE_RE = re.compile(
    # 3+ consecutive TitleCase words. Each word starts with capital,
    # contains 2+ letters total. Trailing-word boundary anchored.
    r"\b(?:[A-Z][a-z]{1,}\s+){2,}[A-Z][a-z]{1,}\b"
)


def _extract_specific_entities(text: str) -> list[str]:
    """Return the set of high-specificity identifiers in *text*.

    Three pattern categories — SKU, store-qualifier phrase, 3+ TitleCase
    words. The categories are intentionally narrow so ordinary
    descriptive prose ("Vienna's hardware stores") doesn't surface as
    a candidate. Duplicates collapse; ordering preserves first
    appearance for downstream readability.
    """
    seen: dict[str, None] = {}
    for ent in _SKU_RE.findall(text):
        seen.setdefault(ent, None)
    for ent in _STORE_QUALIFIER_RE.findall(text):
        seen.setdefault(ent, None)
    for ent in _MULTI_TITLE_RE.findall(text):
        seen.setdefault(ent, None)
    return list(seen.keys())


def detect_unverified_entities(
    response_content: str,
    user_prompt: str = "",
    tool_message_contents: Iterable[str] | None = None,
    *,
    sources: GroundedSources | None = None,
    max_returned: int = 3,
) -> list[str]:
    """Return user-supplied specific entities the agent repeated in its
    response without any grounding source confirming their existence.

    Parameters
    ----------
    response_content:
        The final AIMessage content text.
    user_prompt:
        The user's most recent ``HumanMessage`` content — source of
        the entity candidates.  Ignored when ``sources`` is supplied
        (the value object carries it).
    tool_message_contents:
        Iterable of ToolMessage content strings from the current turn.
        Ignored when ``sources`` is supplied.
    sources:
        Bundled grounding inputs (#1964 Item C).  When supplied this
        is the preferred grounding path — the detector reads
        ``sources.user_prompt`` for the candidate identifiers and the
        full ``sources.blob`` for verification, so entities mentioned
        in the system prompt also count as grounded.
    max_returned:
        Cap on the returned list to avoid swamping the recovery nudge
        when the user introduced many specific identifiers.

    Returns
    -------
    A list of entity strings in first-appearance order. Empty list
    means no unverified entities — the response is internally
    grounded for whatever identifiers it cites.
    """
    if not response_content or not response_content.strip():
        return []

    # Build the grounded sources view once.  ``sources`` is the new
    # path; legacy kwargs are honored when ``sources`` is not provided.
    if sources is None:
        sources = GroundedSources.from_inputs(
            tool_message_contents=tool_message_contents,
            user_prompt=user_prompt,
        )

    if not sources.user_prompt or not sources.user_prompt.strip():
        return []

    # #1960: a refusal naturally echoes the user-supplied identifier
    # while declining to act on it (``I cannot pay invoice INV-...``).
    # That isn't an unverified entity — the agent is referencing what
    # the user named, then refusing.  Short-circuit; same precedent
    # as #1851 for action_intent.
    from src.orchestration.response_detectors import text_is_refusal

    if text_is_refusal(response_content):
        return []

    candidates = _extract_specific_entities(sources.user_prompt)
    if not candidates:
        return []

    response_lower = response_content.lower()
    # Verify against tool results AND the system prompt — but NOT the
    # user prompt.  Candidates were extracted FROM the user prompt; if
    # we included it in the verification blob every candidate would
    # always self-match and the detector would never fire.  Pre-#1964
    # the verification blob was tool-results-only; adding the system
    # prompt is the #1964-C semantic upgrade (entities introduced by
    # the agent's persona policy statement are grounded, not unverified).
    verification_parts = list(sources.tool_results)
    if sources.system_prompt:
        verification_parts.append(sources.system_prompt)
    verification_lower = "\n".join(verification_parts).lower()

    unverified: list[str] = []
    for ent in candidates:
        ent_lower = ent.lower()
        if ent_lower not in response_lower:
            # Agent didn't repeat this identifier — nothing to verify.
            continue
        if ent_lower in verification_lower:
            # A grounding source confirmed (or at least mentioned) it —
            # the agent has grounds.
            continue
        unverified.append(ent)
        if len(unverified) >= max_returned:
            break
    return unverified


def collect_tool_message_contents(messages: list[Any], turn_start_idx: int) -> list[str]:
    """Walk forward from ``turn_start_idx`` and collect the content of
    every ToolMessage in the current turn.

    Pairs with ``detect_unverified_entities`` — the entity detector
    needs to scan tool results for verifying mentions.
    """
    if turn_start_idx >= len(messages):
        return []
    out: list[str] = []
    for msg in messages[turn_start_idx:]:
        if not hasattr(msg, "tool_call_id"):
            continue
        content = getattr(msg, "content", "") or ""
        if isinstance(content, str):
            out.append(content)
    return out


_UNVERIFIED_ENTITY_NUDGE = (
    "Your response repeats {n_word} specific identifier{plural} from "
    "the user's prompt that NO tool result this turn has confirmed: "
    "{entities}. The agent is treating user-supplied names as ground "
    "truth without grounding them in evidence — exactly the failure "
    "mode that produces confident-but-fabricated answers about "
    "products / shops / SKUs that do not exist.\n\n"
    "Revise your response to either:\n"
    "  (a) cite the tool result that confirms each identifier "
    "(by URL or extract); or\n"
    "  (b) state plainly that you could not verify the identifier and "
    'suggest the user clarify it (e.g. "I could not find any product '
    "matching the SKU 'X' — could you confirm the exact code?\"); or\n"
    "  (c) drop the unverified identifier and answer with the closest "
    "verified alternative the search did return.\n\n"
    "Do NOT fabricate a confirmation. Do NOT silently substitute a "
    "different identifier and continue as if the user's name were "
    "correct."
)


def format_unverified_entity_nudge(entities: list[str]) -> str:
    """Render the recovery nudge for one or more unverified entities."""
    n = len(entities)
    plural = "s" if n != 1 else ""
    n_word = {1: "one", 2: "two", 3: "three"}.get(n, str(n))
    quoted = ", ".join(f"'{e}'" for e in entities)
    return _UNVERIFIED_ENTITY_NUDGE.format(n_word=n_word, plural=plural, entities=quoted)


# ── Output-fidelity guard (#1841) ─────────────────────────────────────
#
# The rule registry and the entity detector both check *research effort*
# (was a tool called? is a user-named identifier present in a tool
# result?). Neither checks *research fidelity*: that a verbatim quote or
# explicitly-attributed statement the model puts in its answer actually
# appears in a tool result this turn. The next67 trial surfaced the gap —
# the model fabricated a blockquote ("`kimi-k2.6` was officially
# discontinued …") that inverted the fetched source (which said the
# deprecated `kimi-k2` *series* was discontinued and to USE `kimi-k2.6`).
#
# This detector targets QUOTED spans (double-quoted text and `>`
# blockquotes) only — explicit verbatim claims of "the source says
# exactly this." Free paraphrase is deliberately out of scope
# (paraphrase-fidelity is a separate, false-positive-prone problem).

# Markdown emphasis / inline-code marks to drop before comparison so the
# model's reformatting (``**bold**``, `` `code` ``) doesn't cause a
# spurious mismatch against the raw tool text.
_FIDELITY_MD_STRIP_RE = re.compile(r"[`*_]+")
_FIDELITY_WS_RE = re.compile(r"\s+")
_QUOTE_GLYPHS = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})

# Quoted-span extractors.
_BLOCKQUOTE_RE = re.compile(r"^\s*>\s?(.*)$", re.MULTILINE)
_DOUBLE_QUOTE_RE = re.compile(r'"([^"\n]+)"')
_ELLIPSIS_SPLIT_RE = re.compile(r"\.{3,}|…")

# A quoted span shorter than this (in significant words) is not treated
# as a substantive claim — quoting "OK" or a single product token must
# not trip the guard.
_MIN_QUOTE_WORDS = 6


def _normalize_for_fidelity(text: str) -> str:
    """Casefold + strip markdown marks + collapse whitespace so a faithful
    quote matches the raw tool text despite cosmetic reformatting."""
    t = text.translate(_QUOTE_GLYPHS).lower()
    t = _FIDELITY_MD_STRIP_RE.sub("", t)
    return _FIDELITY_WS_RE.sub(" ", t).strip()


def _strip_quote_wrapping(span: str) -> str:
    """Remove surrounding whitespace and a single layer of wrapping quote
    characters from an extracted span."""
    s = span.strip()
    if len(s) >= 2 and s[0] in "\"'“‘" and s[-1] in "\"'”’":
        s = s[1:-1].strip()
    return s


def _extract_quoted_spans(text: str) -> list[str]:
    """Return the substantive quoted spans in *text* — `>` blockquote
    lines and double-quoted substrings, de-wrapped of quote characters."""
    spans: list[str] = []
    for m in _BLOCKQUOTE_RE.finditer(text):
        spans.append(_strip_quote_wrapping(m.group(1)))
    for m in _DOUBLE_QUOTE_RE.finditer(text):
        spans.append(_strip_quote_wrapping(m.group(1)))
    return [s for s in spans if s]


def _quote_is_grounded(norm_span: str, grounded_blob: str) -> bool:
    """A normalized quote is grounded if it (or, for elided quotes, each
    of its non-trivial segments) is a substring of the grounded blob."""
    if not norm_span:
        return True
    if norm_span in grounded_blob:
        return True
    segments = [seg.strip() for seg in _ELLIPSIS_SPLIT_RE.split(norm_span)]
    segments = [seg for seg in segments if len(seg.split()) >= 3]
    if len(segments) >= 2:
        return all(seg in grounded_blob for seg in segments)
    return False


def detect_unsupported_quote(
    response_content: str,
    tool_message_contents: Iterable[str] | None = None,
    user_prompt: str = "",
    *,
    sources: GroundedSources | None = None,
    min_quote_words: int = _MIN_QUOTE_WORDS,
    max_returned: int = 3,
) -> list[str]:
    """Return substantive quoted spans in the response that appear in no
    grounding source (and aren't quotes of the user's own prompt or
    the system prompt's policy statements).

    Targets fabricated verbatim quotes / fabricated source citations —
    the output-fidelity gap (#1841). Quoted spans only; paraphrase is out
    of scope by design.

    Parameters
    ----------
    response_content: the final AIMessage content text.
    tool_message_contents: ToolMessage content strings from this turn —
        the ground-truth corpus a quote must be traceable to.  Ignored
        when ``sources`` is supplied.
    user_prompt: the user's most recent message; quoting the user's own
        words back is legitimate, so it counts as grounding.  Ignored
        when ``sources`` is supplied.
    sources:
        Bundled grounding inputs (#1964 Item C).  When supplied this is
        the preferred grounding path — the detector additionally treats
        the system prompt as a grounding source, so refusals that quote
        the persona's own policy statement no longer false-fire.
    """
    if not response_content or not response_content.strip():
        return []

    if sources is None:
        sources = GroundedSources.from_inputs(
            tool_message_contents=tool_message_contents,
            user_prompt=user_prompt,
        )

    # #1960: refusals frequently quote the system prompt's own policy
    # statement (``pay_invoice MUST NEVER be called unless ...``) as
    # part of explaining the decline.  Now that ``GroundedSources``
    # carries ``system_prompt``, the policy quote IS in the grounded
    # blob and would match cleanly — but the short-circuit stays for
    # two reasons: (a) refusals are recognised cheaply by regex; the
    # regex sweep over every quoted span is more expensive, and
    # (b) when the caller still uses the legacy kwargs path the
    # system prompt isn't threaded in and the short-circuit remains
    # the only line of defence.  See #1964 Item A audit + #1962.
    from src.orchestration.response_detectors import text_is_refusal

    if text_is_refusal(response_content):
        return []

    spans = _extract_quoted_spans(response_content)
    if not spans:
        return []

    grounded_blob = _normalize_for_fidelity(sources.blob)

    unsupported: list[str] = []
    seen: set[str] = set()
    for span in spans:
        norm = _normalize_for_fidelity(span)
        if len(norm.split()) < min_quote_words:
            continue  # too short to be a substantive claim
        if norm in seen:
            continue
        seen.add(norm)
        if not grounded_blob or not _quote_is_grounded(norm, grounded_blob):
            unsupported.append(span)
            if len(unsupported) >= max_returned:
                break
    return unsupported


_UNSUPPORTED_QUOTE_NUDGE = (
    "Your response presents {n_word} quoted/attributed statement{plural} "
    "that appear in NO tool result from this turn: {quotes}. Presenting a "
    "verbatim quote or a source attribution that you cannot trace to "
    "fetched content is fabrication — even when the surrounding topic was "
    "researched. This is the exact failure mode where a model invents an "
    'authoritative-sounding quote (e.g. "the official platform states …") '
    "that the source never said, or inverts what it did say.\n\n"
    "Revise your response to either:\n"
    "  (a) quote VERBATIM from an actual tool result (copy the exact text, "
    "and only attribute it to the source that actually contains it); or\n"
    "  (b) drop the quotation marks / attribution and state the point as "
    "your own summary, clearly grounded in what the tools returned; or\n"
    "  (c) if the tools did not establish the point, say so plainly rather "
    "than manufacturing a citation.\n\n"
    "Do NOT fabricate a quote. Do NOT attribute a statement to a source "
    "that does not contain it. Do NOT change the subject of a real quote "
    "(e.g. attributing a statement about one model/version to a different "
    "one)."
)


def format_unsupported_quote_nudge(quotes: list[str]) -> str:
    """Render the recovery nudge for one or more unsupported quotes."""
    n = len(quotes)
    plural = "s" if n != 1 else ""
    n_word = {1: "one", 2: "two", 3: "three"}.get(n, str(n))

    def _clip(q: str) -> str:
        q = q.strip()
        return q if len(q) <= 120 else q[:117] + "..."

    quoted = "; ".join(f'"{_clip(q)}"' for q in quotes)
    return _UNSUPPORTED_QUOTE_NUDGE.format(n_word=n_word, plural=plural, quotes=quoted)


# ── Version-scope-collapse guard (#1843) ──────────────────────────────
#
# The fidelity guard (#1841) catches a *fabricated quote* — text the model
# attributes to a source that the source never said. The version-scope
# guard catches a subtler, adjacent failure: the model takes a status that
# the evidence genuinely scopes to one identifier and *re-scopes* it onto a
# more specific (child / newer) identifier. In the next67 trial the fetched
# docs said the deprecated ``kimi-k2`` *series* was discontinued and to USE
# ``kimi-k2.6``; the model reported that ``kimi-k2.6`` itself was
# discontinued, and under challenge invented that ``kimi-k2.5`` was *also*
# discontinued. Both are children of the ``kimi-k2`` parent the source
# actually scoped the status to. Textually ``kimi-k2`` ⊂ ``kimi-k2.6`` /
# ``kimi-k2.5``; semantically they are opposite — the classic
# series→version substring confusion.
#
# Why effort/fidelity guards miss it: ``detect_unverified_claim`` only
# checks the verifier was called (it was); ``detect_unsupported_quote``
# only fires on *quoted* spans (a status stated as prose or in a table cell
# is not a quote); and every presence-based check is satisfied because
# ``kimi-k2`` *is* present in the tool blob and is a substring of the
# child ID. Scope requires modelling *which* identifier a status attaches
# to — not merely whether a token appears.
#
# Detection is deliberately conservative to satisfy the "no false positives
# on legitimate version-specific claims" requirement:
#
#   * **Sentence-scoped, nearest-ID attribution.** A status attaches to the
#     nearest model-ID *within its own sentence / table row*. This handles
#     the contrastive form ("``kimi-k2.6`` is current, unlike the
#     discontinued ``kimi-k2`` series") — there the status's nearest ID is
#     the parent, so the child is never marked claimed.
#   * **Negation aware.** "is not discontinued" / "isn't deprecated" do not
#     count as a status assertion.
#   * **Canonical IDs only.** We match canonical hyphen/dotted identifiers
#     (``kimi-k2.6``, ``gpt-4``) — the form tool outputs and the model's
#     own citations use. Display variants ("Kimi K2.6") are out of scope.
#   * **Flag only true scope-collapse.** A child claim is flagged only when
#     (a) the source does NOT support that status for the child's exact ID,
#     AND (b) the source scopes that status to a prefix-*parent* of the
#     child. A genuinely version-specific claim the source supports, or a
#     pure fabrication with no parent in the evidence, is left to the other
#     guards.
#
# Lifecycle-status only (discontinued / deprecated / retired / sunset /
# EOL): these are the high-harm misattributions (telling a user a current
# model is dead). Release/launch status is intentionally excluded — version
# release dates legitimately differ and the false-positive cost is higher.

_NEG_STATUS_RE = re.compile(
    r"\b(?:discontinued|deprecated|sunset(?:ted|ting)?|retired|"
    r"decommissioned|end[\s-]of[\s-]life|eol|"
    r"no\s+longer\s+(?:available|supported|maintained|offered))\b",
    re.IGNORECASE,
)

# Negation tokens that, when they immediately precede a status word, mean
# the model is NOT asserting that status ("kimi-k2.6 is not discontinued").
_PRE_NEGATION_RE = re.compile(
    r"\b(?:not|never|no|isn[’']t|wasn[’']t|aren[’']t|weren[’']t|n[’']t)\b",
    re.IGNORECASE,
)

# Canonical model/version identifier: an alpha stem of ≥2 letters, at least
# one digit somewhere, and at least one ``-``/``.`` separator group. Matches
# ``kimi-k2``, ``kimi-k2.6``, ``gpt-4``, ``claude-opus-4-7``. The greedy
# separator group means ``kimi-k2.6`` is captured whole (not just
# ``kimi-k2``), so the child and parent are distinct tokens. The leading
# ``[a-z]{2,}`` and the digit lookahead exclude bare versions (``1.2.3``),
# abbreviations (``e.g``), Slack/object IDs (``u0b115kt39d``), and
# dotted-but-digitless names (``node.js``).
_MODEL_ID_RE = re.compile(r"\b(?=[a-z0-9.\-]*\d)[a-z]{2,}[a-z0-9]*(?:[-.][a-z0-9]+)+\b")

# Split into sentences / table rows WITHOUT breaking version tokens: a
# sentence ends on .!? followed by whitespace (so the ``.`` inside
# ``kimi-k2.6`` — followed by a digit — never splits), or on a newline.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|[\n\r]+")

# A status attached to an ID more than this many characters away (within a
# single long sentence) is too distant to be a reliable attribution.
_SCOPE_WINDOW = 100


@dataclass(frozen=True)
class VersionScopeMismatch:
    """One version-scope-collapse finding.

    ``claimed_id`` is the specific identifier the response attaches the
    status to; ``scoped_to_id`` is the prefix-parent the *evidence*
    actually scopes that status to; ``status`` is the matched lifecycle
    word (``discontinued`` etc.).
    """

    claimed_id: str
    status: str
    scoped_to_id: str


def _nearest_id(
    ids: list[tuple[str, int, int]], s_start: int, s_end: int, *, window: int = _SCOPE_WINDOW
) -> str | None:
    """Return the model-ID token nearest to a status span, or ``None`` if
    the closest ID is farther than *window* characters."""
    best: str | None = None
    best_gap: int | None = None
    for tok, i_start, i_end in ids:
        if i_end <= s_start:
            gap = s_start - i_end
        elif i_start >= s_end:
            gap = i_start - s_end
        else:
            gap = 0
        if gap <= window and (best_gap is None or gap < best_gap):
            best, best_gap = tok, gap
    return best


def _nonnegated_status_spans(lowered_segment: str) -> list[tuple[int, int, str]]:
    """Return ``(start, end, word)`` for each non-negated lifecycle-status
    mention in a (lowercased) sentence/row."""
    out: list[tuple[int, int, str]] = []
    for sm in _NEG_STATUS_RE.finditer(lowered_segment):
        pre = lowered_segment[max(0, sm.start() - 24) : sm.start()]
        if _PRE_NEGATION_RE.search(pre):
            continue
        out.append((sm.start(), sm.end(), sm.group(0)))
    return out


def _response_status_claims(text: str) -> list[tuple[str, str]]:
    """Return ``(model_id, status_word)`` pairs the response asserts —
    status attached to its nearest model-ID within each sentence/row."""
    claims: list[tuple[str, str]] = []
    for seg in _SENTENCE_SPLIT_RE.split(text):
        lowered = seg.lower()
        ids = [(m.group(0), m.start(), m.end()) for m in _MODEL_ID_RE.finditer(lowered)]
        if not ids:
            continue
        for s_start, s_end, word in _nonnegated_status_spans(lowered):
            nearest = _nearest_id(ids, s_start, s_end)
            if nearest is not None:
                claims.append((nearest, word))
    return claims


def _source_status_scope(tool_blob: str) -> tuple[set[str], set[str]]:
    """Return ``(supported_ids, primary_ids)`` from the evidence.

    ``supported_ids`` is every ID that shares a sentence/row with a
    non-negated lifecycle status — the generous set used to decide whether
    a claimed ID's status is genuinely backed (avoids false positives when
    the source really does scope the status to that exact ID).
    ``primary_ids`` is the nearest ID to each status — the precise set used
    to find the parent a status is actually scoped to.
    """
    supported: set[str] = set()
    primary: set[str] = set()
    for seg in _SENTENCE_SPLIT_RE.split(tool_blob):
        lowered = seg.lower()
        ids = [(m.group(0), m.start(), m.end()) for m in _MODEL_ID_RE.finditer(lowered)]
        if not ids:
            continue
        spans = _nonnegated_status_spans(lowered)
        if not spans:
            continue
        supported.update(item[0] for item in ids)
        for span in spans:
            nearest = _nearest_id(ids, span[0], span[1])
            if nearest is not None:
                primary.add(nearest)
    return supported, primary


def _best_parent(child_id: str, parent_candidates: set[str]) -> str | None:
    """Return the longest ID in *parent_candidates* that is a prefix-parent
    of *child_id* (proper prefix ending on a ``.``/``-`` version boundary),
    or ``None``. ``kimi-k2`` is a parent of ``kimi-k2.6``; ``kimi-k2`` is
    NOT a parent of ``kimi-k20`` (boundary char ``0`` is not a separator)."""
    parents = [
        y
        for y in parent_candidates
        if y != child_id and child_id.startswith(y) and child_id[len(y)] in ".-"
    ]
    return max(parents, key=len) if parents else None


def detect_version_scope_mismatch(
    response_content: str,
    tool_message_contents: Iterable[str],
    *,
    max_returned: int = 3,
) -> list[VersionScopeMismatch]:
    """Return version-scope-collapse findings (#1843).

    Flags each case where the response attaches a lifecycle status to a
    specific model-ID while the evidence scopes that status only to a
    prefix-*parent* of that ID (and does not back it for the exact ID).
    Returns ``[]`` when nothing is found — including when the response makes
    no status claim, when no tool content is available to check against, or
    when every claim is either source-supported or a pure fabrication with
    no parent in the evidence (the latter is the other guards' concern).
    """
    if not response_content or not response_content.strip():
        return []
    claims = _response_status_claims(response_content)
    if not claims:
        return []
    blob = "\n".join(c for c in tool_message_contents if isinstance(c, str))
    if not blob.strip():
        return []
    supported, primary = _source_status_scope(blob)
    if not primary:
        return []

    results: list[VersionScopeMismatch] = []
    seen: set[tuple[str, str]] = set()
    for claimed_id, status in claims:
        if claimed_id in supported:
            continue  # evidence backs this exact ID — a true claim
        parent = _best_parent(claimed_id, primary)
        if parent is None:
            continue  # no parent scope in evidence — not a scope collapse
        key = (claimed_id, parent)
        if key in seen:
            continue
        seen.add(key)
        results.append(
            VersionScopeMismatch(claimed_id=claimed_id, status=status, scoped_to_id=parent)
        )
        if len(results) >= max_returned:
            break
    return results


_VERSION_SCOPE_NUDGE = (
    "Your response attaches a lifecycle status to {n_word} specific "
    "model/version identifier{plural} that the fetched sources scope to a "
    "DIFFERENT (broader / parent) identifier: {pairs}. This is version-scope "
    "collapse — a statement about a deprecated *series* or a different "
    "version is being reattributed to a more specific version. Textually "
    "`{parent}` is a substring of `{child}`, but they are not the same "
    "thing, and the sources may actually list `{child}` as current or "
    "available.\n\n"
    "Re-read the tool results and revise your answer to:\n"
    '  (a) attach the "{status}" status to the EXACT identifier the '
    "source scopes it to (`{parent}`), not to `{child}`; and\n"
    "  (b) state the actual status of `{child}` using ONLY what the sources "
    "say about that exact identifier — do not assume it inherits the "
    "parent's status.\n\n"
    "Do NOT carry a parent/series status down onto a specific version. Do "
    "NOT invent a status for a version the sources do not mention."
)


def format_version_scope_nudge(mismatches: list[VersionScopeMismatch]) -> str:
    """Render the recovery nudge for one or more version-scope mismatches."""
    n = len(mismatches)
    plural = "s" if n != 1 else ""
    n_word = {1: "one", 2: "two", 3: "three"}.get(n, str(n))
    pairs = "; ".join(
        f'`{m.claimed_id}` (sources scope "{m.status}" to `{m.scoped_to_id}`)' for m in mismatches
    )
    example = mismatches[0]
    return _VERSION_SCOPE_NUDGE.format(
        n_word=n_word,
        plural=plural,
        pairs=pairs,
        child=example.claimed_id,
        parent=example.scoped_to_id,
        status=example.status,
    )


# ── Attributed-claim guard (#1860) ─────────────────────────────────────
#
# #1841's quote detector only fires on verbatim quoted/blockquoted spans.
# The next69 trial surfaced the residual gap: the model fabricated a
# factual claim in PROSE — *"Voices of The Void has both a native Linux
# build … as confirmed by community guides"* — with no source actually
# confirming a native Linux build. Quoted-only guards cannot catch
# attributed prose claims.
#
# This detector targets ATTRIBUTED paragraphs only — blocks where the
# model credits a source/authority ("as confirmed by", "according to",
# "officially documented", "the docs say", "sources confirm", …). For
# each such paragraph it extracts distinctive multi-word content phrases
# and checks whether they appear in the grounded blob (tool output + user
# prompt). When several distinctive phrases are ungrounded the
# attributed content is treated as fabricated. Free paraphrase is
# deliberately out of scope — only attributed paragraphs are inspected,
# keeping the false-positive surface narrow.

_ATTRIBUTION_MARKERS_RE = re.compile(
    r"(?im)\b(?:"
    r"as\s+(?:confirmed|stated|documented|noted|reported|explained|"
    r"described|shown|mentioned|established|verified|attested)\s+"
    r"(?:by|in|on|at|via)\b"
    r"|according\s+to\b"
    r"|as\s+per\b"
    r"|per\s+(?:the|its|their)\s+\w+"
    r"|(?:the|its|their)\s+(?:docs|documentation|guides?|community|website|"
    r"site|page|manual|reference)\s+(?:say|says|states?|notes?|confirms?|"
    r"indicates?|reports?|mentions?|explains?|claims?|describes?)\b"
    r"|(?:sources?|reports?|documents?)\s+(?:confirm|confirms|say|says|"
    r"indicate|indicates|note|notes|state|states|report|reports|"
    r"mention|mentions|describe|describes)\b"
    r"|it\s+is\s+(?:documented|stated|confirmed|noted|reported|established|"
    r"described)\s+(?:that|in|by|on)\b"
    r"|officially\s+(?:confirmed|stated|documented|noted|reported|"
    r"established|verified|released|announced|deprecated|discontinued|"
    r"retired|recommended)\b"
    # ── #1867 Group A: self-introspection attributions ─────────────────
    # The model claims it re-read the artifact and the artifact agrees
    # with its (often flipped) new position. Manufactured authority of
    # the "I looked, it agrees with me now" shape. Q3 reproducer:
    # ``Re-reading the file confirms _is_sycophantic_prefix has no check
    # for tool_calls`` — directly contradicted by the actual file.
    r"|(?:the|its)\s+(?:file|code|source(?:\s+(?:code|file))?|"
    r"implementation)\s+(?:says|states?|confirms?|shows?|reveals?|"
    r"indicates?|demonstrates?)\b"
    r"|re-reading\s+(?:the|its)\s+(?:file|code|source|implementation)\s+"
    r"(?:confirms?|shows?|reveals?|indicates?)\b"
    r"|looking\s+at\s+(?:the|its)\s+(?:file|code|source|implementation)\s+"
    r"(?:more\s+carefully\s+)?(?:confirms?|shows?|reveals?|indicates?)\b"
    r"|(?:inspecting|examining|reviewing)\s+(?:the|its)\s+"
    r"(?:file|code|source|implementation)\s+"
    r"(?:confirms?|shows?|reveals?|indicates?)\b"
    # ── #1867 Group B: generic third-party corroborating attributions ──
    # The model credits a third-party document (article, blog, README,
    # release notes, …) for content the document does not actually
    # contain.
    r"|(?:the|its|their)\s+(?:article|paper|blog(?:\s+post)?|post|study|"
    r"report|wiki|readme|changelog|release\s+notes?)\s+"
    r"(?:say|says|states?|confirms?|shows?|notes?|indicates?|describes?|"
    r"mentions?|explains?)\b"
    r"|as\s+the\s+(?:article|paper|blog(?:\s+post)?|post|study|report|"
    r"wiki|readme|changelog|release\s+notes?)\s+"
    r"(?:say|says|states?|confirms?|shows?|notes?|indicates?)\b"
    r")"
)

# Stop-word set for distinctive-n-gram filtering. Extremely common tokens
# don't make a phrase distinctive enough to be a fabrication signal.
_ATTRIBUTION_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "and",
        "or",
        "but",
        "if",
        "in",
        "on",
        "at",
        "to",
        "by",
        "for",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "can",
        "could",
        "may",
        "might",
        "should",
        "shall",
        "must",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "they",
        "them",
        "their",
        "there",
        "here",
        "when",
        "where",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "how",
        "why",
        "with",
        "without",
        "from",
        "as",
        "also",
        "more",
        "most",
        "some",
        "any",
        "all",
        "not",
        "no",
        "yes",
        "one",
        "two",
        "both",
        "via",
        "use",
        "used",
        "using",
        "you",
        "your",
        "we",
        "our",
        "us",
        "i",
        "me",
        "my",
        "he",
        "she",
        "him",
        "her",
        "his",
        "hers",
    }
)

# Grounding rule for attributed paragraphs: a paragraph is flagged when a
# meaningful fraction of its DISTINCTIVE CONTENT TOKENS (≥ 4 chars, not in
# the stop-word set) is absent from the grounded blob's distinctive-token
# set. This is paraphrase-tolerant — a legitimate attribution to grounded
# content shares most of its content tokens with the source even when the
# wording is rephrased — while still catching fabrication, where the
# specific content (entities + properties) is genuinely missing from
# every source.
_ATTRIBUTION_MISSING_FRACTION = 0.35
_ATTRIBUTION_MIN_DISTINCTIVE = 4


def _attribution_paragraphs(text: str) -> list[str]:
    """Split *text* into paragraphs and return those carrying an
    attribution marker. Paragraphs are blank-line- or heading-separated
    blocks."""
    if not text or not isinstance(text, str):
        return []
    blocks = re.split(r"\n\s*\n+|\n(?=#{1,6}\s)", text)
    return [b for b in blocks if _ATTRIBUTION_MARKERS_RE.search(b)]


def _distinctive_token_set(text: str) -> set[str]:
    """Return distinctive content tokens of *text* — lowercased, non-stop,
    length ≥ 4 — with the attribution markers themselves stripped first so
    they cannot trivially match the grounded blob."""
    norm = _normalize_for_fidelity(text)
    if not norm:
        return set()
    norm = _ATTRIBUTION_MARKERS_RE.sub(" ", norm)
    tokens = re.findall(r"[a-z][a-z0-9\-]{3,}", norm)
    return {t for t in tokens if t not in _ATTRIBUTION_STOPWORDS}


def detect_unsupported_attribution(
    response_content: str,
    tool_message_contents: Iterable[str] | None = None,
    user_prompt: str = "",
    *,
    sources: GroundedSources | None = None,
    missing_fraction_threshold: float = _ATTRIBUTION_MISSING_FRACTION,
    min_distinctive_tokens: int = _ATTRIBUTION_MIN_DISTINCTIVE,
    max_returned: int = 3,
) -> list[str]:
    """Return attributed paragraphs whose distinctive content is not grounded.

    Scope: only paragraphs containing an attribution marker
    ("as confirmed by …", "according to …", "officially …", …) are
    inspected — free paraphrase is deliberately out of scope (#1841 already
    handles quoted spans). A paragraph is flagged when the FRACTION of its
    distinctive content tokens absent from the grounded blob's
    distinctive-token set is at least ``missing_fraction_threshold``.
    Paragraphs with fewer than ``min_distinctive_tokens`` are skipped — too
    little signal to flag with confidence. Returns at most ``max_returned``
    snippets in first-appearance order.

    When ``sources`` is supplied (#1964 Item C), the system prompt also
    contributes to the grounded token set — attributions to the agent's
    own policy statements are grounded by construction.
    """
    if not response_content or not response_content.strip():
        return []

    if sources is None:
        sources = GroundedSources.from_inputs(
            tool_message_contents=tool_message_contents,
            user_prompt=user_prompt,
        )

    # #1960: a refusal commonly cites the system prompt's policy
    # statement (``According to our payment policies, ...``).  Now that
    # ``GroundedSources`` carries ``system_prompt``, those attributions
    # are grounded against the persona's own tokens and would no longer
    # false-fire — but the short-circuit stays for callers still on the
    # legacy kwargs path (no system prompt threaded in) and for the
    # latency win.  See #1962 + #1964 Item A audit.
    from src.orchestration.response_detectors import text_is_refusal

    if text_is_refusal(response_content):
        return []

    candidate_paragraphs = _attribution_paragraphs(response_content)
    if not candidate_paragraphs:
        return []

    grounded_tokens = _distinctive_token_set(sources.blob)

    flagged: list[str] = []
    for p in candidate_paragraphs:
        para_tokens = _distinctive_token_set(p)
        if len(para_tokens) < min_distinctive_tokens:
            continue  # too little signal to assess
        if not grounded_tokens:
            # No grounding at all — every attributed paragraph with enough
            # distinctive content is unsupported by construction.
            flagged.append(p.strip()[:240])
        else:
            missing = para_tokens - grounded_tokens
            if len(missing) / len(para_tokens) >= missing_fraction_threshold:
                flagged.append(p.strip()[:240])
        if len(flagged) >= max_returned:
            break
    return flagged


_UNSUPPORTED_ATTRIBUTION_NUDGE = (
    "Your response credits a source/authority for {n_word} statement{plural} "
    "whose specific content is NOT supported by any tool result this turn: "
    '{snippets}. Phrases like "as confirmed by …", "according to …", '
    '"officially …" manufacture authority for a claim — when the claim\'s '
    "specific content (entities, properties, facts) is absent from your "
    "fetched sources, you are fabricating the source itself, not just "
    "paraphrasing.\n\n"
    "Revise your response to either:\n"
    "  (a) re-check the tool results and only attribute claims whose "
    "specific content (the entity + property pair, the version, the date) "
    "appears in the fetched extracts — quote or paraphrase faithfully; or\n"
    "  (b) drop the attribution phrase and state the claim as your own "
    "summary, clearly grounded in what the tools returned; or\n"
    "  (c) if the tools did not establish the claim, say so plainly — do "
    "not invent a citing source.\n\n"
    'Do NOT credit "community guides", "the docs", "sources", or any '
    "named authority for content that is not in the fetched results."
)


def format_unsupported_attribution_nudge(snippets: list[str]) -> str:
    """Render the recovery nudge for one or more unsupported attributions."""
    n = len(snippets)
    plural = "s" if n != 1 else ""
    n_word = {1: "one", 2: "two", 3: "three"}.get(n, str(n))

    def _clip(s: str) -> str:
        s = " ".join(s.split())
        return s if len(s) <= 160 else s[:157] + "..."

    quoted = "; ".join(f'"{_clip(s)}"' for s in snippets)
    return _UNSUPPORTED_ATTRIBUTION_NUDGE.format(n_word=n_word, plural=plural, snippets=quoted)


# ── #1988: entity-owner mismatch detector ──────────────────────────────
#
# Sister to ``detect_unsupported_attribution``, specialised for the
# structured-identifier failure mode catalogued in #1987 (PM role-test
# cycle-2 post-mortem).  ``detect_unsupported_attribution`` catches a
# paragraph whose distinctive content tokens are absent from the
# grounded blob.  But an entity-owner swap can pass that check — the
# agent's response contains many corpus-overlap tokens (the entity
# title, status, mitigations) AND a stakeholder name that IS in the
# corpus (just paired with the wrong entity).  The unsupported-
# attribution detector sees ample overlap and lets it through.
#
# This detector targets the specific shape *<entity-id> ... <stakeholder
# name>* co-mentions.  For each pair found in the response, it verifies
# that the SAME pair appears within a same-window co-occurrence in
# at least one tool result.  Map-free: works against any structured
# corpus that uses canonical-form identifiers (R-NN, DEC-YYYY-MM-DD-NN,
# CHG-NIMB-NN, NIMB-WBS-NN, etc.) and proper-noun owner names.

# Canonical entity-ID patterns found in the Project Nimbus corpus + a
# generic fall-back for any ``[A-Z]+-\d+`` identifier.  Keep this
# specific enough that ordinary version strings (``v1.2.3``) and
# free numbers don't trip.
_ENTITY_ID_RE = re.compile(
    r"\b("
    r"R-\d{1,3}"
    r"|DEC-\d{4}-\d{2}-\d{2}-\d{1,3}"
    r"|CHG-[A-Z]+-\d{1,3}"
    r"|NIMB-WBS-\d{1,3}"
    r"|[A-Z]{2,8}-\d{1,4}"
    r")\b"
)

# Proper-noun (capitalised multi-word) name pattern.  Matches "Tomislav
# Hessford", "Beatriz Cazadora-Olesen", "Hyeon-Jin Park", "Aldous
# Pemberton-Riggs".  Avoids matching ordinary sentence-start words by
# requiring at least two capitalized tokens.  Hyphens permitted in
# tokens to handle compound names; apostrophes too.  3-letter minimum
# per token to filter sentence-start "The Risk" / "We Will" noise.
_STAKEHOLDER_NAME_RE = re.compile(r"\b([A-Z][a-zA-Z'\-]{2,}(?:\s+[A-Z][a-zA-Z'\-]{2,}){1,3})\b")

# Window size (chars) around an entity-id mention in the response within
# which we collect candidate stakeholder names.  240 chars covers a
# table-row + immediate next sentence; tight enough that we don't
# pair an entity with a name two paragraphs away.
_OWNER_ATTR_RESPONSE_WINDOW = 240

# Window size (chars) within which we require <entity-id> + <name> to
# co-appear in a tool result for the pair to be "grounded".  Wider than
# the response window because corpus chunks often split header/body
# across lines or use tables; 400 chars allows reasonable spacing.
_OWNER_ATTR_GROUNDING_WINDOW = 400


# #2006 cycle-10 bias-audit: tokens that, when present in a candidate
# multi-word TitleCase phrase, suppress that phrase from being treated
# as a stakeholder name.  ``_STAKEHOLDER_NAME_RE`` happily matches
# section-header phrases like "Risk Register", "Project Charter",
# "Decision Log", "Status Report" — these are document-structure
# nouns, not people.  Filtering on the presence of any of these
# tokens prevents the entity-owner detector from reporting
# "<entity> co-mentioned with 'Risk Register'" false positives.
#
# These are TUNED FOR PROJECT-MANAGEMENT / BUSINESS-DOCUMENT corpora.
# An operator running Cogtrix against a meaningfully different corpus
# (legal, medical, scientific) may need a different blocklist — for
# example "Risk Manager Smith" or "Project Lead Jones" would be
# wrongly filtered by this set.  Pass ``stakeholder_name_blocklist=``
# to :func:`detect_entity_owner_mismatch` to override.
_DEFAULT_STAKEHOLDER_NAME_BLOCKLIST: frozenset[str] = frozenset(
    {
        # Generic English business-document structural nouns
        "summary",
        "status",
        "report",
        "update",
        # PM-corpus section heads (also common in many business corpora)
        "project",
        "risk",
        "register",
        "decision",
        "change",
    }
)


def _co_occurs_in_text(text: str, entity_id: str, name: str, window: int) -> bool:
    """True when ``entity_id`` and ``name`` appear in ``text`` within
    ``window`` characters of each other (in either order).

    Used to verify a (entity_id, owner) pair is grounded in a tool
    result.  ``window`` is character-distance, not token-distance —
    appropriate for the structured-table layout the corpus uses.
    """
    if not text or not entity_id or not name:
        return False
    # Find all entity-id positions
    entity_positions = [m.start() for m in re.finditer(re.escape(entity_id), text)]
    if not entity_positions:
        return False
    name_positions = [m.start() for m in re.finditer(re.escape(name), text)]
    if not name_positions:
        return False
    for ep in entity_positions:
        for np_ in name_positions:
            if abs(ep - np_) <= window:
                return True
    return False


def detect_entity_owner_mismatch(
    response_content: str,
    tool_message_contents: Iterable[str] | None = None,
    user_prompt: str = "",
    *,
    sources: GroundedSources | None = None,
    max_returned: int = 3,
    stakeholder_name_blocklist: frozenset[str] | None = None,
) -> list[str]:
    """Return ``<entity-id> + <stakeholder-name>`` pairs the response
    mentions co-located that do NOT co-appear in any tool result or in
    the system-prompt grounding this turn.

    Targets the entity-owner swap failure mode from #1987 (PM role-test
    cycle-2): the agent emits a response with corpus entity IDs and
    corpus stakeholder names, but stitches the wrong name onto the
    wrong entity (e.g. *"R-13 ... Hyeon-Jin Park (Migration Squad)"*
    when the corpus says R-13's owner is Tomislav Hessford).

    Heuristic:

    1. Find all entity-IDs in ``response_content`` (matching
       ``_ENTITY_ID_RE``).
    2. For each, collect candidate stakeholder names appearing within
       ``_OWNER_ATTR_RESPONSE_WINDOW`` chars.
    3. For each ``(entity_id, name)`` pair, verify both tokens
       co-occur within ``_OWNER_ATTR_GROUNDING_WINDOW`` chars in at
       least one tool result OR in the system prompt.  If not, the
       pair is unsupported.

    Refusal short-circuit: refusals naturally name stakeholders the
    user mentioned without claiming an ownership relation; we skip.
    """
    if not response_content or not response_content.strip():
        return []

    effective_blocklist = (
        stakeholder_name_blocklist
        if stakeholder_name_blocklist is not None
        else _DEFAULT_STAKEHOLDER_NAME_BLOCKLIST
    )

    if sources is None:
        sources = GroundedSources.from_inputs(
            tool_message_contents=tool_message_contents,
            user_prompt=user_prompt,
        )

    from src.orchestration.response_detectors import text_is_refusal

    if text_is_refusal(response_content):
        return []

    entity_matches = list(_ENTITY_ID_RE.finditer(response_content))
    if not entity_matches:
        return []

    # Build a grounded-text union for verification.  Tool results +
    # system prompt; user prompt EXCLUDED (the user's prompt may name
    # the entity-id without specifying an owner, and including it would
    # let "user named R-13" satisfy the pair without corpus grounding).
    grounded_texts: list[str] = list(sources.tool_results)
    if sources.system_prompt:
        grounded_texts.append(sources.system_prompt)
    if not grounded_texts:
        return []  # No grounding to verify against — caller's fault, fail-open.

    flagged: list[str] = []
    seen_pairs: set[tuple[str, str]] = set()

    for em in entity_matches:
        entity_id = em.group(1)
        # Collect names in the windowed neighbourhood.
        start = max(0, em.start() - _OWNER_ATTR_RESPONSE_WINDOW)
        end = em.end() + _OWNER_ATTR_RESPONSE_WINDOW
        window_text = response_content[start:end]
        for nm in _STAKEHOLDER_NAME_RE.finditer(window_text):
            name = nm.group(1).strip()
            # Filter section-header phrases (e.g. "Project Nimbus",
            # "Risk Register") — multi-word TitleCase strings that
            # match the stakeholder-name regex but aren't people.
            # Default blocklist is tuned for business-document
            # corpora; pass ``stakeholder_name_blocklist=`` to
            # override.  See ``_DEFAULT_STAKEHOLDER_NAME_BLOCKLIST``
            # for the bias-audit notes.
            if any(tok.lower() in effective_blocklist for tok in name.split()):
                continue
            pair_key = (entity_id, name)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            grounded = any(
                _co_occurs_in_text(t, entity_id, name, _OWNER_ATTR_GROUNDING_WINDOW)
                for t in grounded_texts
            )
            if not grounded:
                flagged.append(
                    f"{entity_id} co-mentioned with {name!r} in response but "
                    f"the pair does not co-appear in any tool result this turn"
                )
                if len(flagged) >= max_returned:
                    return flagged

    return flagged


# Runtime LLM-facing nudge.  Intentionally abstract — no
# corpus-specific entity-ids, stakeholder names, or file paths in
# the production string (#2006 cycle-10 bias-leakage rule).
_ENTITY_OWNER_MISMATCH_NUDGE = (
    "Your response co-mentions {n_word} entity-owner pair{plural} that "
    "do NOT co-appear in any tool result this turn: {snippets}. This "
    "is the entity-owner attribution failure mode catalogued in #1987 — "
    "the agent stitches a plausible-sounding stakeholder onto a wrong "
    "entity ID because the stakeholder's apparent role matches the "
    "entity's topic.  Stakeholder roles are CORPUS facts about people, "
    "NOT inferences from entity topics.\n\n"
    "Revise your response to do ONE of:\n"
    "  (a) re-query the knowledge base specifically for each cited "
    "entity's Owner / Decided-By / Approver field (a focused query "
    "naming the entity-id and the literal token ``owner``) and use "
    "the verbatim owner name + role qualifier from the returned "
    "chunk; or\n"
    "  (b) drop the owner attribution from any entity where the "
    "corpus owner field is not present in your fetched chunks — "
    "say plainly that the entity is documented in your retrieved "
    "sources but the owner field is not in the retrieved excerpts; "
    "or\n"
    "  (c) if you have no corpus evidence for any of the cited "
    "entities, defer the question rather than fabricating.\n\n"
    "Do NOT attribute an entity to a stakeholder solely because the "
    "stakeholder's listed role topically matches the entity's "
    "subject; that is exactly the failure pattern this detector "
    "catches."
)


def format_entity_owner_mismatch_nudge(mismatches: list[str]) -> str:
    """Render the recovery nudge for one or more entity-owner mismatches."""
    n = len(mismatches)
    plural = "s" if n != 1 else ""
    n_word = {1: "one", 2: "two", 3: "three"}.get(n, str(n))

    def _clip(s: str) -> str:
        s = " ".join(s.split())
        return s if len(s) <= 160 else s[:157] + "..."

    quoted = "; ".join(f'"{_clip(s)}"' for s in mismatches)
    return _ENTITY_OWNER_MISMATCH_NUDGE.format(n_word=n_word, plural=plural, snippets=quoted)


# ── #1989: topic-substitution detector ─────────────────────────────────
#
# Cluster C from the #1987 PM role-test cycle-2 post-mortem: the user
# asks about a subject (e.g. *"CompactSync codebase tech debt"*) that
# does NOT appear in the corpus.  Expected: agent defers with the
# standard out-of-scope template.  Actual: agent silently retitles its
# response to a related in-corpus subject (*"Project Nimbus Technical
# Debt Risks"*) and answers THAT.
#
# Existing detectors target fabrication and unsupported attribution.
# They don't catch *silent question reframing* where the response body
# is technically grounded (Project Nimbus IS in the corpus) but
# addresses a different topic from what was asked.  The LLM-as-judge
# also misses this — it scores response-quality without comparing
# response-topic to question-topic.
#
# Heuristic detector:
#
# 1. Extract distinctive subjects from the user prompt — CamelCase
#    compound words ("CompactSync"), TitleCase multi-word phrases
#    ("Project Nimbus"), all-caps acronyms ("AWS").
# 2. For each, check whether it appears in the agent's response, in
#    any tool result this turn, or in the system prompt.
# 3. If absent from ALL three AND the response is substantive (above
#    a length threshold), flag silent topic substitution.

# Multi-token TitleCase phrase: "Project Nimbus", "New York", "AcmeCloud Migration".
# Two or more capitalised tokens with allowed letters + apostrophes + hyphens.
#
# Inter-token whitespace is restricted to HORIZONTAL whitespace only
# (``[ \t]+``) — the original ``\s+`` matched newlines, causing the
# detector to read across line boundaries.  Concretely, a user prompt
# containing
#     "Primary product category: Electronics
#      Please register this supplier and validate the information."
# yielded ``"Electronics\nPlease"`` as a "missing subject", which then
# fired ``detect_topic_substitution`` (#1989) on every kimi-k2-5
# response that didn't mention the literal ``Electronics\nPlease``
# substring.  The resulting recovery cascade burned the scenario
# timeout and produced ``tools=0 turns=0`` PARTIAL_COMPLETION on
# ``procurement_supplier_registration`` × kimi-k2-5 in PR #1999 CI.
# A TitleCase topic-subject phrase that spans a paragraph break is
# never the user's actual subject — it's almost always a regex false
# positive.
_DISTINCTIVE_TITLE_PHRASE_RE = re.compile(
    r"\b([A-Z][a-zA-Z'\-]{2,}(?:[ \t]+[A-Z][a-zA-Z'\-]{2,}){1,4})\b"
)

# CamelCase / PascalCase single-token compound: "CompactSync", "AcmeDB", "AcmeCloud".
# At least 5 chars (else "Bob", "Joe" trip), contains lowercase
# and at least one internal uppercase — i.e. NOT a sentence-start
# normal word.
_DISTINCTIVE_CAMEL_RE = re.compile(r"\b([A-Z][a-z]+[A-Z][a-zA-Z]+)\b")

# All-caps acronym: "AWS", "GCP", "HTTPS", "PMBOK".  Length 3-8.
# Excluded as too generic / common at sentence start: "I", "A", "AN".
_DISTINCTIVE_ACRONYM_RE = re.compile(r"\b([A-Z]{3,8})\b")

# Acronyms that are too generic to be meaningful topic subjects.
_GENERIC_ACRONYMS: frozenset[str] = frozenset(
    {
        "USD",
        "EUR",
        "GBP",
        "USA",
        "EU",
        "UK",
        "PDF",
        "HTML",
        "CSV",
        "XML",
        "API",
        "URL",
        "URI",
        "ID",
        "IDS",
        "IDS.",
        "OK",
        "YES",
        "NO",
        "FAQ",
        "FAQS",
        "TBD",
        "PM",
        "TODO",
        "TODOS",
        "GMT",
        "UTC",
        "EST",
        "PST",
        "Q1",
        "Q2",
        "Q3",
        "Q4",
        "AM",
    }
)

# Title-phrase structural tokens we filter out so that TitleCase
# section/heading phrases ("Project Charter", "Decision Log",
# "Status Report") don't get treated as topic-subject candidates.
#
# #2006 cycle-10 bias-audit: this set was originally tuned during
# PM-test work.  The tokens themselves are generic English
# business-document structural nouns and apply cleanly to most
# corporate corpora — but an operator running Cogtrix against a
# meaningfully different corpus (legal, medical, scientific) may
# need a different set.  Pass ``title_phrase_stopwords=`` to
# :func:`detect_topic_substitution` to override.
_DEFAULT_TITLE_PHRASE_STOPWORDS: frozenset[str] = frozenset(
    {
        "project",
        "risk",
        "register",
        "report",
        "decision",
        "decisions",
        "change",
        "changes",
        "schedule",
        "summary",
        "status",
        "update",
        "executive",
        "current",
        "key",
        "next",
        "previous",
        "last",
        "first",
    }
)
# Minimum char length for distinctive subjects (filters two- and three-
# letter capitalised tokens that are usually sentence-start noise).
_DISTINCTIVE_MIN_CHARS = 5

# Threshold above which a response is "substantive enough" to count as
# a topic-substitution if the user's subject is absent.  ~400 chars is
# a reasonable lower bound for "answers the question at length" — short
# responses are more often clarifications or refusals.
_TOPIC_SUBSTITUTION_MIN_RESPONSE_CHARS = 400


def _extract_distinctive_subjects(
    text: str,
    *,
    title_phrase_stopwords: frozenset[str] | None = None,
) -> list[str]:
    """Extract distinctive topic-subject candidates from *text*.

    Combines three pattern categories — CamelCase compound, TitleCase
    multi-word phrase, all-caps acronym — and filters out generic /
    structural noise so the candidates reflect what a user is most
    plausibly asking ABOUT.

    ``title_phrase_stopwords`` lets the caller override the structural
    stopword set applied to TitleCase phrases.  When ``None``, uses
    :data:`_DEFAULT_TITLE_PHRASE_STOPWORDS` (tuned for business-document
    corpora — see that constant's docstring for the bias-audit notes).

    Preserves first-appearance order; deduplicates by exact string.
    """
    if not text or not text.strip():
        return []
    effective_stopwords = (
        title_phrase_stopwords
        if title_phrase_stopwords is not None
        else _DEFAULT_TITLE_PHRASE_STOPWORDS
    )
    seen: dict[str, None] = {}

    # CamelCase compounds (e.g. "CompactSync", "AcmeCloud", "AcmeDB")
    for m in _DISTINCTIVE_CAMEL_RE.finditer(text):
        candidate = m.group(1)
        if len(candidate) < _DISTINCTIVE_MIN_CHARS:
            continue
        seen.setdefault(candidate, None)

    # TitleCase multi-word phrases (e.g. "Project Nimbus", "New York")
    for m in _DISTINCTIVE_TITLE_PHRASE_RE.finditer(text):
        candidate = m.group(1)
        if len(candidate) < _DISTINCTIVE_MIN_CHARS:
            continue
        # Filter phrases whose every token is a structural stopword
        # (e.g. "Status Update" — keep "Project Nimbus" because
        # "Nimbus" is not in the stopword set).
        tokens = [t.lower() for t in candidate.split()]
        non_stopword_tokens = [t for t in tokens if t not in effective_stopwords]
        if not non_stopword_tokens:
            continue
        seen.setdefault(candidate, None)

    # All-caps acronyms (e.g. "AWS", "PMBOK")
    for m in _DISTINCTIVE_ACRONYM_RE.finditer(text):
        candidate = m.group(1)
        if candidate in _GENERIC_ACRONYMS:
            continue
        if len(candidate) < 3:
            continue
        seen.setdefault(candidate, None)

    return list(seen.keys())


def detect_topic_substitution(
    response_content: str,
    tool_message_contents: Iterable[str] | None = None,
    user_prompt: str = "",
    *,
    sources: GroundedSources | None = None,
    min_response_chars: int = _TOPIC_SUBSTITUTION_MIN_RESPONSE_CHARS,
    max_returned: int = 3,
    title_phrase_stopwords: frozenset[str] | None = None,
) -> list[str]:
    """Return distinctive subjects from the user prompt that the
    response, tool results, and system prompt all FAIL to cover —
    evidence of silent topic substitution.

    Targets the failure mode from #1989: user asks about subject X
    (e.g. *"CompactSync codebase tech debt"*); agent silently retitles
    to in-corpus subject Y (e.g. *"Project Nimbus Technical Debt
    Risks"*) and produces a substantive answer about Y without
    acknowledging the swap.

    Conservative — only fires when:

    * the user prompt has at least one distinctive subject (CamelCase
      compound, TitleCase multi-word phrase, or non-generic all-caps
      acronym);
    * the response is substantive (above ``min_response_chars``);
    * the response is NOT a refusal (matches the #1962 convention);
    * the distinctive subject is absent from ALL of: the response
      itself, every tool result this turn, and the system prompt.

    Returns the missing subjects in user-prompt order, capped at
    ``max_returned``.
    """
    if not response_content or not response_content.strip():
        return []

    if sources is None:
        sources = GroundedSources.from_inputs(
            tool_message_contents=tool_message_contents,
            user_prompt=user_prompt,
        )

    if not sources.user_prompt or not sources.user_prompt.strip():
        return []

    # Substantive-response gate.  Short responses are typically
    # refusals, clarifications, or "I don't have data on X" replies —
    # those are the CORRECT behaviour the detector should not punish.
    if len(response_content.strip()) < min_response_chars:
        return []

    from src.orchestration.response_detectors import text_is_refusal

    if text_is_refusal(response_content):
        return []

    subjects = _extract_distinctive_subjects(
        sources.user_prompt, title_phrase_stopwords=title_phrase_stopwords
    )
    if not subjects:
        return []

    # #1992 follow-up: when EVERY tool result this turn is empty (the
    # corpus / search stub returned nothing), the agent's response not
    # naming the user's subject is NOT topic substitution — it's the
    # honest *"I searched but found nothing"* shape that
    # ``regression_persist_before_refusing`` and similar scenarios
    # explicitly require.  Substitution requires the agent to PIVOT
    # to a different topic with content; if no content was retrieved
    # at all, there's no pivot.  Short-circuit to avoid false-firing
    # on empty-corpus-result scenarios.
    has_nonempty_tool_results = any(isinstance(s, str) and s.strip() for s in sources.tool_results)
    if not has_nonempty_tool_results:
        return []

    # Case-insensitive containment check on the combined corpus +
    # response + system prompt.  Use lowercased blobs for cheap
    # substring matching.
    response_lower = response_content.lower()
    tool_blob_lower = "\n".join(s for s in sources.tool_results if isinstance(s, str)).lower()
    sys_lower = (sources.system_prompt or "").lower()

    missing: list[str] = []
    for subject in subjects:
        sl = subject.lower()
        if sl in response_lower or sl in tool_blob_lower or sl in sys_lower:
            continue
        missing.append(subject)
        if len(missing) >= max_returned:
            break
    return missing


_TOPIC_SUBSTITUTION_NUDGE = (
    "Your response substantively addresses a topic that does NOT match "
    "what the user asked about.  The user named {n_word} distinctive "
    "subject{plural}: {subjects}.  None of those subjects appears in "
    "your response, in any tool result this turn, or in the system "
    "prompt — yet your response is substantive.  That is silent "
    "topic substitution (#1987 Cluster C): the agent reframes the "
    "user's question to fit what it can answer instead of acknowledging "
    "the out-of-scope.\n\n"
    "Revise your response to do ONE of:\n"
    "  (a) re-query the corpus specifically for the user's subject "
    "(e.g. ``query_knowledge_base({first_subject!r})``) and use the "
    "results if anything relevant comes back; or\n"
    "  (b) defer plainly: *'I don't have information about "
    "{first_subject} in the corpus.  This may be outside my scope — "
    "recommend escalating to the appropriate role.'*\n\n"
    "Do NOT silently rename your response heading to fit a related "
    "in-corpus topic.  Do NOT proceed to 'the closest in-scope topic' "
    "without explicit user confirmation."
)


def format_topic_substitution_nudge(missing_subjects: list[str]) -> str:
    """Render the recovery nudge for one or more missing user subjects."""
    if not missing_subjects:
        # Should not happen — defensive.
        return _TOPIC_SUBSTITUTION_NUDGE.format(
            n_word="zero", plural="s", subjects="(none)", first_subject="(none)"
        )
    n = len(missing_subjects)
    plural = "s" if n != 1 else ""
    n_word = {1: "one", 2: "two", 3: "three"}.get(n, str(n))
    listed = ", ".join(f"``{s}``" for s in missing_subjects)
    first = missing_subjects[0]
    return _TOPIC_SUBSTITUTION_NUDGE.format(
        n_word=n_word, plural=plural, subjects=listed, first_subject=first
    )


# ── #1868: non-canonical GitHub-fork-recommendation detector ───────────
#
# Q5 of the 2026-05-28 holistic-test battery against
# ``cogtrix:release-next`` @ ``2bb52c7``: asked for *"three currently-
# active open-source projects on GitHub that implement WebAssembly tools
# for security analysis"*, the agent returned one canonical entry plus
# two **personal / inactive forks** (DharitriOne/wasmer,
# wasm-wasi-rs/runtimes__wasmtime) presented with the canonical
# projects' descriptions and recommendation framing. A user clicking the
# link in good faith would land on the fork, not on the canonical home.
#
# Failure pattern: ``canonical-name + canonical-blurb → non-canonical
# URL``. The agent uses web_search to find ANY ``github.com/<owner>/<repo>``
# URL whose name matches the project keyword and treats it as authoritative.
#
# First-pass detector (this PR — see ticket #1868 for proposed
# strengthenings):
#   1. Extract all ``github.com/<owner>/<repo>`` URLs from the response.
#   2. For each URL, classify ``(owner, repo)`` against an explicit
#      canonical allowlist plus three structural heuristics
#      (``owner == repo``, ``repo`` substring of ``owner``, recognised
#      org-name suffixes).
#   3. If non-canonical, require the surrounding 200-char context to
#      contain explicit RECOMMENDATION language (``actively maintained``,
#      ``stable release``, ``production-ready``, ``universal runtime``,
#      ``recommended``, …) — incidental URLs (issue trackers, code
#      examples) do not trip the detector.
#   4. Suppress entirely when the user's prompt explicitly asked for
#      forks (``"show me hardened forks of …"``).
#
# Out of scope for this PR (could come later if the heuristic proves
# insufficient): live GitHub-API check of ``stars`` / ``forks`` /
# ``is_fork``.  That requires token/rate-limit/mock-fixture work that
# is not worth coupling to the regex-only pass.

_GITHUB_REPO_URL_RE = re.compile(
    r"https?://(?:www\.)?github\.com/([\w][\w.-]*)/([\w][\w.-]*)",
    re.IGNORECASE,
)

# Recommendation-context markers. Triggered when ANY of these appears
# within ~200 chars of a flagged GitHub URL — implies the response is
# presenting the URL as an authoritative project home rather than as
# an incidental link (issue tracker, code example, etc.).
_RECOMMENDATION_LANGUAGE_RE = re.compile(
    r"\b(?:"
    r"actively\s+(?:maintained|developed|supported|updated)"
    r"|currently\s+(?:active|maintained|supported|updated)"
    r"|stable\s+(?:release|version|build|stream)"
    r"|production[- ]ready|battle[- ]tested|widely\s+(?:used|adopted)"
    r"|industry[- ]standard|de\s+facto|flagship|canonical"
    r"|the\s+(?:de[- ]facto|standard|reference|official|canonical)\s+"
    r"(?:implementation|library|tool|runtime|engine|client|server)"
    r"|universal\s+(?:runtime|engine|tool|client|library|framework)"
    r"|official(?:ly)?\s+(?:supported|backed|maintained|released)"
    r"|recommended|recommend(?:ing)?\s+(?:this|the)"
    r"|recent\s+(?:commits?|releases?)|last\s+commit"
    r"|mature\s+(?:library|tool|project|runtime|engine|framework)"
    r"|popular\s+(?:choice|library|tool|project)"
    r"|I\s+recommend|here\s+(?:is|are)\s+(?:three\s+|some\s+|several\s+|a\s+|an\s+)?"
    r"(?:active|maintained|currently-?active|currently\s+active|popular|recommended)"
    r")",
    re.IGNORECASE,
)

# User-prompt suppression: the user explicitly asked for forks; any
# non-canonical recommendation is on-task, not a fabrication. Conservative
# match — requires the literal token "fork" near a request verb.
_USER_ASKED_FOR_FORK_RE = re.compile(
    r"\b(?:fork|forks|forked)\b",
    re.IGNORECASE,
)

# Org-name suffixes that imply a canonical project organisation
# (lowercased, with or without the dash). Adding here only ever
# *suppresses* firing so additions are safe.
_CANONICAL_ORG_SUFFIXES: tuple[str, ...] = (
    "-io",
    "-team",
    "-org",
    "-orgs",
    "-project",
    "-projects",
    "-labs",
    "-lab",
    "-foundation",
    "-fdn",
    "-community",
    "-developers",
    "-devs",
    "-group",
    "-network",
)

# Known canonical GitHub owners — major orgs plus a small set of
# individual maintainers whose repos are canonical (``torvalds/linux``,
# ``ggerganov/llama.cpp``, etc.). Lowercased on lookup. Additions are
# always safe (only ever suppress firing); deletions need care.
_KNOWN_CANONICAL_OWNERS: frozenset[str] = frozenset(
    {
        # Major orgs (broad)
        "google",
        "googleapis",
        "googlecloudplatform",
        "googlechrome",
        "microsoft",
        "azure",
        "vscode",
        "apple",
        "amazon",
        "amzn",
        "aws",
        "aws-samples",
        "facebook",
        "facebookresearch",
        "meta",
        "meta-llama",
        "openai",
        "anthropics",
        # Cloud-native / infra
        "kubernetes",
        "kubernetes-sigs",
        "cncf",
        "containerd",
        "etcd-io",
        "envoyproxy",
        "helm",
        "istio",
        "linkerd",
        "prometheus",
        "grafana",
        "fluent",
        "hashicorp",
        # Language ecosystems
        "python",
        "pypa",
        "psf",
        "rust-lang",
        "golang",
        "nodejs",
        "dotnet",
        "ruby",
        "rails",
        "php",
        "swift-lang",
        "swiftlang",
        "dart-lang",
        "kotlin",
        "jetbrains",
        "scala",
        "clojure",
        "erlang",
        "elixir-lang",
        "haskell",
        # Major individual / canonical projects
        "torvalds",
        "ggerganov",
        "openssl",
        "git",
        "vim",
        "neovim",
        "tmux",
        "alacritty",
        # WebAssembly ecosystem (Q5 reproducer)
        "bytecodealliance",
        "wasmerio",
        "webassembly",
        "wasmcloud",
        "wasm-tool",
        "wasmedge",
        "extism",
        # JS ecosystem
        "vercel",
        "vuejs",
        "angular",
        "expressjs",
        "webpack",
        "babel",
        "rollup",
        "vitejs",
        "nestjs",
        "fastify",
        "yarnpkg",
        "npm",
        "denoland",
        "oven-sh",
        "remix-run",
        "sveltejs",
        # Python ecosystem
        "pallets",
        "django",
        "fastapi",
        "tiangolo",
        "pydantic",
        "pytest-dev",
        "scrapy",
        "celery",
        "sqlalchemy",
        # ML / AI
        "pytorch",
        "tensorflow",
        "keras-team",
        "huggingface",
        "scikit-learn",
        "scipy",
        "numpy",
        "pandas-dev",
        "ipython",
        "jupyter",
        "jupyterhub",
        # Other
        "apache",
        "eclipse",
        "mozilla",
        "linuxfoundation",
        "redhat",
        "ibm",
        "intel",
        "nvidia",
        "openzeppelin",
        "ethereum",
        "spotify",
        "slack",
        "github",
        "shopify",
        "stripe",
        "elastic",
        "redis",
        "mongodb",
        "postgresql",
        "moby",
        "docker",
        "ansible",
        "wireguard",
        "asciinema",
        "spf13",
    }
)


def _looks_canonical_owner(owner: str, repo: str) -> bool:
    """Heuristic: is the ``(owner, repo)`` pair likely a canonical home?

    Returns True for known-canonical orgs/maintainers and for several
    structural patterns common to canonical-project URL shapes. Returns
    False when nothing matches — caller treats False as "potentially
    non-canonical fork".

    Rules (in order):
      1. ``owner`` (lower) is in :data:`_KNOWN_CANONICAL_OWNERS`.
      2. ``owner == repo`` (``kubernetes/kubernetes``,
         ``prometheus/prometheus``).
      3. ``repo`` is a substring of ``owner`` (``wasmerio/wasmer``).
      4. ``owner`` ends with a recognised org-name suffix from
         :data:`_CANONICAL_ORG_SUFFIXES`.
    """
    o = owner.lower()
    r = repo.lower()
    if o in _KNOWN_CANONICAL_OWNERS:
        return True
    if o == r:
        return True
    if r and r in o:
        return True
    return any(o.endswith(suffix) for suffix in _CANONICAL_ORG_SUFFIXES)


def detect_noncanonical_fork_recommendation(
    response: str,
    user_prompt: str = "",
) -> list[str]:
    """Return URLs in *response* that recommend a non-canonical GitHub fork.

    See the module-level comment block above :data:`_GITHUB_REPO_URL_RE`
    for the full detection flow and scope rationale (#1868).

    Args:
        response: The model's final AIMessage content.
        user_prompt: The user's most recent HumanMessage content. When
            it contains the literal token ``fork``, the detector returns
            ``[]`` — the user explicitly asked for forks, so a fork URL
            is on-task, not a fabrication.

    Returns:
        Distinct flagged GitHub URLs in order of first appearance.
    """
    if not response or not response.strip():
        return []
    if user_prompt and _USER_ASKED_FOR_FORK_RE.search(user_prompt):
        return []

    flagged: list[str] = []
    seen: set[str] = set()
    for match in _GITHUB_REPO_URL_RE.finditer(response):
        owner = match.group(1)
        repo = match.group(2).rstrip(".,;:?!\"')]")
        url = match.group(0).rstrip(".,;:?!\"')]")
        if not repo:
            continue
        if _looks_canonical_owner(owner, repo):
            continue
        # Only flag URLs whose surrounding context frames them as an
        # authoritative recommendation; suppress incidental links.
        start = max(0, match.start() - 200)
        end = min(len(response), match.end() + 200)
        if not _RECOMMENDATION_LANGUAGE_RE.search(response[start:end]):
            continue
        if url in seen:
            continue
        seen.add(url)
        flagged.append(url)
    return flagged


_NONCANONICAL_FORK_NUDGE = (
    "Your response recommends {n_word} GitHub URL{plural} that may "
    "point to a non-canonical fork rather than the canonical home of "
    "the project: {urls}.\n\n"
    "Each cited URL has an owner that is not in the known set of "
    "canonical organisations / maintainers, and the surrounding text "
    "describes the URL with authoritative recommendation language "
    "(``actively maintained``, ``stable release``, ``universal "
    "runtime``, etc.). When a recommendation URL is non-canonical, "
    "the user will land on a fork instead of the project they think "
    "they are clicking through to.\n\n"
    "Re-emit your recommendation using ONE of these honest paths — do "
    "NOT keep the suspicious URL as-is:\n"
    "  (a) Replace the URL with the canonical home of the project "
    "(verify it via a fresh ``web_search`` or by checking that the "
    "owner is the project's official organisation).\n"
    "  (b) State plainly that you cannot identify a canonical active "
    "project matching the criteria, rather than recommending an "
    "uncertain URL.\n"
    "  (c) If you intended to recommend a fork specifically (because "
    "the user asked for forks), say so clearly — name the upstream "
    "project and note that the URL points to a fork."
)


def format_noncanonical_fork_nudge(urls: list[str]) -> str:
    """Render the recovery nudge for one or more flagged fork URLs."""
    n = len(urls)
    plural = "s" if n != 1 else ""
    n_word = {1: "one", 2: "two", 3: "three"}.get(n, str(n))
    joined = ", ".join(urls)
    return _NONCANONICAL_FORK_NUDGE.format(n_word=n_word, plural=plural, urls=joined)


# ── Synthesis-after-eviction guard (#1943 PR #4) ──────────────────────
#
# PR #1 (#1944) added a ``[CONTEXT NOTICE]`` SystemMessage marker that
# the cap prepends when it evicts older messages.  PR #3 (#1946) wired
# the memory-layer rolling summary INTO that marker so the agent has a
# semantic anchor.  This detector closes the cascade by catching the
# residual failure mode: the model emits a substantive final answer
# AFTER the marker, ran NO new tools this turn, and the response carries
# none of the compliant-acknowledgement phrases — meaning it's drawing
# its specifics either from the (visible) marker summary or from
# training-data fabrication, and there's no way to tell which without
# semantic validation.  We treat it as a high-suspicion synthesis and
# route to a recovery node that nudges the model to either (a) ground
# every specific claim in what's visible, (b) honestly surface the
# loss, or (c) call the appropriate tool to re-gather evidence.
#
# Detection signals (all must be true to fire):
# 1. A ``SystemMessage`` with ``additional_kwargs["cogtrix.kind"] ==
#    "context_evicted"`` is in the visible context.
# 2. The latest message is an ``AIMessage`` with no ``tool_calls`` (it's
#    a final answer, not a tool dispatch).
# 3. The response content is substantive (length >= the threshold below).
# 4. The current turn — messages from the most-recent ``HumanMessage``
#    onwards — contains zero ``AIMessage.tool_calls`` (the model did not
#    gather fresh evidence this turn).
# 5. The response does NOT contain a compliant-acknowledgement phrase
#    (it didn't honestly say "context was lost, please re-share").

_SYNTHESIS_MIN_RESPONSE_CHARS: int = 200

# Phrases the model uses when it correctly acknowledged the eviction
# rather than fabricating around it.  All matched case-insensitively
# against the response text; ANY hit short-circuits the detector.
_EVICTION_COMPLIANT_PHRASES: tuple[str, ...] = (
    "context was lost",
    "context was removed",
    "context has been removed",
    "context has been evicted",
    "prior context was removed",
    "prior conversation was removed",
    "earlier messages were removed",
    "older messages were removed",
    "older messages have been removed",
    "do not have access to the prior",
    "do not have access to that prior",
    "don't have access to the prior",
    "i no longer have access",
    "could you re-share",
    "could you re-send",
    "please re-share",
    "please share again",
    "please re-send",
    "context notice",
    "please provide the earlier",
    "please provide the prior",
    "can you provide the earlier",
    "can you provide the prior",
    "i cannot recall",
    "i can't recall",
    "i'm unable to recall",
    "no longer have the original",
)

_EVICTION_MARKER_KIND: str = "context_evicted"


def _has_eviction_marker(messages: Iterable[Any]) -> bool:
    """Return True when any ``SystemMessage`` in *messages* carries
    ``additional_kwargs["cogtrix.kind"] == "context_evicted"`` — the
    marker that PR #1's ``_apply_context_message_cap`` prepends on
    eviction.
    """
    for msg in messages:
        # Match on the kind metadata, not on prose substring, so prose
        # rewording in future PRs does not silently disable the detector.
        kwargs = getattr(msg, "additional_kwargs", None)
        if isinstance(kwargs, dict) and kwargs.get("cogtrix.kind") == _EVICTION_MARKER_KIND:
            return True
    return False


def _current_turn_made_tool_calls(messages: list, turn_start: int) -> bool:
    """Return True when any ``AIMessage`` from *turn_start* (the last
    ``HumanMessage`` in *messages*) onwards declares tool calls.  The
    final ``AIMessage`` itself is excluded — that's the response under
    inspection, not a fresh-evidence signal.
    """
    if turn_start >= len(messages):
        return False
    # Last message is the response under inspection; scan everything
    # after the turn-start HumanMessage but BEFORE the final AIMessage.
    for msg in messages[turn_start : len(messages) - 1]:
        if getattr(msg, "tool_calls", None):
            return True
    return False


def detect_synthesis_after_eviction(
    response_content: str,
    messages: list,
    turn_start: int,
    *,
    min_response_chars: int = _SYNTHESIS_MIN_RESPONSE_CHARS,
) -> bool:
    """Return True when the response is suspected synthesis-after-eviction.

    All five detection signals must hold simultaneously.  The detector is
    deliberately conservative — short replies, refusals, and compliant
    acknowledgements all short-circuit.  Bounded by the recovery node's
    retry counter so a stubborn model gets exactly one revision attempt
    before the response ships with a warning logged.

    Parameters
    ----------
    response_content
        Final ``AIMessage`` content text.
    messages
        Full message list as seen by the orchestrator at routing time
        (must include the response as the last message).
    turn_start
        Index of the most-recent ``HumanMessage`` — see
        :func:`_find_current_turn_start` in ``nodes/recovery.py``.
    min_response_chars
        Minimum response length to be considered "substantive".  Short
        conversational replies and refusals don't trip.
    """
    # Signal 3 — substantive content.  Short replies (acks, single
    # questions, brief refusals) are the normal compliant behaviour we
    # do NOT want to disturb.
    if not response_content or len(response_content) < min_response_chars:
        return False

    # Signal 1 — eviction marker is in the visible context.
    if not _has_eviction_marker(messages):
        return False

    # Signal 2 — response is a final answer (no tool_calls).  The
    # orchestrator only routes here after observing the final AIMessage,
    # so a tool-calling response would never reach this branch in
    # practice — but the defensive check keeps the detector callable
    # directly from tests with hand-rolled message lists.
    if not messages:
        return False
    last = messages[-1]
    if getattr(last, "tool_calls", None):
        return False

    # Signal 4 — no fresh evidence gathered this turn.
    if _current_turn_made_tool_calls(messages, turn_start):
        return False

    # Signal 5 — no compliant-acknowledgement phrase in the response.
    lowered = response_content.lower()
    for phrase in _EVICTION_COMPLIANT_PHRASES:
        if phrase in lowered:
            return False

    return True


_SYNTHESIS_AFTER_EVICTION_NUDGE = (
    "Your previous response made substantive claims that appear to "
    "draw on conversation content that was evicted earlier this "
    "session — the [CONTEXT NOTICE] SystemMessage above explicitly "
    "stated that older messages were removed, and you ran NO tools "
    "this turn, so the claims could not have come from fresh evidence.\n\n"
    "Revise your response to do ONE of the following:\n"
    "  (a) ground every specific claim in what is still visible "
    "(quote or cite the surviving message that supports it, and "
    "explicitly say which one); or\n"
    "  (b) honestly tell the user that the prior context was lost and "
    'ask them to re-share the relevant detail (use the phrase "the '
    'earlier context was removed" or similar so the user understands '
    "what happened); or\n"
    "  (c) call the appropriate tool to re-gather the evidence from "
    "scratch this turn.\n\n"
    "Do NOT restate the previous answer with the same fabricated "
    "specifics.  Do NOT invent names, quotes, numbers, file paths, or "
    "any other detail that is not visible in the surviving conversation "
    "above.  If the rolling-summary block in the [CONTEXT NOTICE] is "
    "your only source, it is broad-strokes context only — you may "
    "summarise it but you must not present its broad strokes as "
    "verbatim specifics."
)


def format_synthesis_after_eviction_nudge() -> str:
    """Render the recovery nudge for the synthesis-after-eviction guard."""
    return _SYNTHESIS_AFTER_EVICTION_NUDGE


# ── GroundedDetector registry (#1964 Item B) ──────────────────────────
#
# Declarative roster of every detector that consumes ``GroundedSources``.
# Each entry pairs the detector callable with its name, handler-node
# identifier, and a precise declaration of which ``GroundedSources``
# fields the detector actually reads.  See the protocol section earlier
# in this module for the contract.
#
# Adding a new grounding-aware detector:
#   1. Implement the detector with the protocol signature
#      ``def detect_<x>(response_content: str, *, sources: GroundedSources, **kwargs) -> list[str]``.
#   2. Register it here with the correct grounding-source declarations.
#   3. Wire the recovery node + retry counter in
#      ``src/orchestration/nodes/recovery.py`` and the router branch
#      in ``src/orchestration/graph.py:route_after_model``.
#
# ``test_grounded_detector_protocol.py`` asserts every entry below
# satisfies the protocol shape and that its handler_node convention
# matches ``handle_<name>``.


GROUNDED_DETECTORS: tuple[GroundedDetectorSpec, ...] = (
    GroundedDetectorSpec(
        name="unverified_entity",
        detect=detect_unverified_entities,
        handler_node="handle_unverified_entity",
        # Entity candidates are EXTRACTED from the user prompt — so the
        # user prompt itself is NOT used as verification grounding
        # (every candidate would self-match).  Tool results + system
        # prompt verify the candidates.
        consumes_tool_results=True,
        consumes_user_prompt=False,
        consumes_system_prompt=True,
        extracts_candidates_from_user_prompt=True,
        issue_refs=(1714, 1726, 1960, 1964),
    ),
    GroundedDetectorSpec(
        name="unsupported_quote",
        detect=detect_unsupported_quote,
        handler_node="handle_unsupported_quote",
        consumes_tool_results=True,
        consumes_user_prompt=True,
        consumes_system_prompt=True,
        issue_refs=(1841, 1960, 1964),
    ),
    GroundedDetectorSpec(
        name="unsupported_attribution",
        detect=detect_unsupported_attribution,
        handler_node="handle_unsupported_attribution",
        consumes_tool_results=True,
        consumes_user_prompt=True,
        consumes_system_prompt=True,
        issue_refs=(1860, 1960, 1964),
    ),
    GroundedDetectorSpec(
        name="entity_owner_mismatch",
        detect=detect_entity_owner_mismatch,
        handler_node="handle_entity_owner_mismatch",
        # Entity-ID + owner pairs are EXTRACTED from the response, then
        # verified against tool results + system prompt.  User prompt
        # excluded from verification: a user can name an entity-ID
        # without specifying its owner, and including the user prompt
        # would let the user's bare mention satisfy a (entity, name)
        # pair that the response co-located.
        consumes_tool_results=True,
        consumes_user_prompt=False,
        consumes_system_prompt=True,
        issue_refs=(1948, 1987, 1988),
    ),
    GroundedDetectorSpec(
        name="topic_substitution",
        detect=detect_topic_substitution,
        handler_node="handle_topic_substitution",
        # Subjects are EXTRACTED from the user prompt, then verified
        # against the response itself + tool results + system prompt.
        # ``consumes_user_prompt=False`` because the user prompt is the
        # SOURCE of candidates, not part of the grounding blob — same
        # convention as ``unverified_entity``.
        consumes_tool_results=True,
        consumes_user_prompt=False,
        consumes_system_prompt=True,
        extracts_candidates_from_user_prompt=True,
        issue_refs=(1948, 1987, 1989),
    ),
)


__all__ = [
    "GROUNDED_DETECTORS",
    "GroundedDetector",
    "GroundedDetectorSpec",
    "GroundedSources",
    "VERIFICATION_RULES",
    "VerificationRule",
    "VersionScopeMismatch",
    "collect_grounded_sources",
    "collect_tool_message_contents",
    "collect_tool_names_this_turn",
    "detect_entity_owner_mismatch",
    "detect_noncanonical_fork_recommendation",
    "detect_topic_substitution",
    "detect_synthesis_after_eviction",
    "detect_unsupported_attribution",
    "detect_unsupported_quote",
    "detect_unverified_claim",
    "detect_unverified_entities",
    "detect_version_scope_mismatch",
    "format_entity_owner_mismatch_nudge",
    "format_noncanonical_fork_nudge",
    "format_topic_substitution_nudge",
    "format_synthesis_after_eviction_nudge",
    "format_unsupported_attribution_nudge",
    "format_unsupported_quote_nudge",
    "format_unverified_entity_nudge",
    "format_version_scope_nudge",
]
