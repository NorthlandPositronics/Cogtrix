"""Structured ``cogtrix.kind`` markers for dispatcher-synthesised ToolMessages.

Issue #1921 / #1919 (Finding 6): the dispatcher in
``src/orchestration/nodes/process_tools.py`` constructs synthetic
``ToolMessage`` instances whenever it needs to respond to a tool call
that can't be executed (tool not loaded, tool name unresolved, tool
disabled by user, ...). Downstream detectors — most importantly
``_looks_like_fabricated_success_after_tool_errors`` in
``response_detectors.py`` — need to classify these as errors so the
agent can't follow them with a fabricated success claim.

Before this module, the detector substring-matched the message content
against a hand-curated allowlist of phrases ("tool not loaded", ...).
That allowlist drifted out of sync with the dispatcher's actual
phrasings — when the dispatcher's "is in the catalog but not loaded"
message was added (clearer than the previous "tool not loaded"), the
detector silently stopped recognising it. The 18-call ``run`` loop in
``.agent-test-1918/test5`` ended with the agent fabricating a "tests
passed" final response, and the detector missed it.

The fix: the dispatcher tags each synthetic ToolMessage with a
``cogtrix.kind`` value in ``additional_kwargs``. Detectors consult the
kind first; substring matching is retained as a fallback for
ToolMessages produced by real tools (those can't carry the flag
without per-tool changes).

Adding a new kind here is the canonical extension point — any new
dispatcher-synthesised error shape should land in this set so the
detector chain catches it automatically.
"""

from __future__ import annotations

# Synthetic-ToolMessage discriminator placed in ``additional_kwargs``.
COGTRIX_KIND_KEY = "cogtrix.kind"

#: Tool name matched a catalog entry that's not currently loaded.
#: Dispatcher tells the agent to issue ``request_tools(add=[...])``.
KIND_TOOL_NOT_LOADED = "tool_not_loaded"

#: Tool name was denied by the user in the current session.
KIND_TOOL_DISABLED = "tool_disabled"

#: Tool name was unresolvable, but a fuzzy match points to an already-active
#: tool. Dispatcher returns a "Did you mean 'X'? It is already active." hint.
KIND_TOOL_NAME_INVALID = "tool_name_invalid"

#: Tool name has no match in either pool. Dispatcher returns
#: "'X' is not a valid tool and could not be resolved."
KIND_TOOL_RESOLUTION_FAILED = "tool_resolution_failed"

#: A url-fetch tool (http_get/http_post) was called with a search query and no
#: URL — the model confused it with web_search (#2293). Dispatcher short-circuits
#: with a redirect message instead of letting the call hit a Pydantic
#: ``url Field required`` error. The tool did NOT execute.
KIND_TOOL_MISUSE_REDIRECT = "tool_misuse_redirect"

#: Set of kinds that signal a dispatcher-synthesised resolution failure —
#: the agent's tool call did NOT execute and any "I did it" claim that
#: follows is fabricated.  Consumers (see
#: ``_looks_like_fabricated_success_after_tool_errors``) treat any kind
#: in this set as a tool-error event.
TOOL_RESOLUTION_FAILURE_KINDS: frozenset[str] = frozenset(
    {
        KIND_TOOL_NOT_LOADED,
        KIND_TOOL_DISABLED,
        KIND_TOOL_NAME_INVALID,
        KIND_TOOL_RESOLUTION_FAILED,
        KIND_TOOL_MISUSE_REDIRECT,
    }
)


def is_resolution_failure_message(message: object) -> bool:
    """Return True iff *message* is a ToolMessage carrying a kind in
    :data:`TOOL_RESOLUTION_FAILURE_KINDS`.

    Accepts any duck-typed object exposing ``additional_kwargs``; returns
    False for anything else.  The helper centralises the kind lookup so
    callers don't reach into ``additional_kwargs`` directly.
    """
    extras = getattr(message, "additional_kwargs", None)
    if not isinstance(extras, dict):
        return False
    return extras.get(COGTRIX_KIND_KEY) in TOOL_RESOLUTION_FAILURE_KINDS


__all__ = [
    "COGTRIX_KIND_KEY",
    "KIND_TOOL_NOT_LOADED",
    "KIND_TOOL_DISABLED",
    "KIND_TOOL_NAME_INVALID",
    "KIND_TOOL_RESOLUTION_FAILED",
    "KIND_TOOL_MISUSE_REDIRECT",
    "TOOL_RESOLUTION_FAILURE_KINDS",
    "is_resolution_failure_message",
]
