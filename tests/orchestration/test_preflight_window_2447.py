"""#2447 — pre-flight compression guard must size on the INPUT context window.

The pre-flight token-size guard in ``call_model`` (added with #1943) used the
model's *output* completion cap (``model_max_tokens``) as a proxy for the
*input* context window, and let it floor the window via ``min()``. A model with
a small ``max_tokens`` but a large ``context_window`` (e.g. deepseek-v4-pro:
ctx 131072, max_tokens 8000) got its input floored to 8000 → threshold ~6400 →
force-compressed on *every* turn, dropping recent context.

The fix threads the model's real ``context_window`` (stamped onto the llm at
provider-creation time) as the authoritative input-window signal; the output
cap is only a last-resort fallback when ``context_window`` is unknown. Operator
compression caps (``max_context_tokens`` / ``context_max_tokens``) still tighten
the window — only the *output* cap must never silently floor it.
"""

from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from cogtrix_core.config import ModelConfig, ProviderConfig
from cogtrix_core.providers import create_chat_model_from_configs
from tests.orchestration.test_call_model import _make_node, _make_state

# _CHARS_PER_TOKEN == 2, so a message body of N chars ⇒ ~N/2 estimated tokens.
_BIG_BODY = "x" * 40_000  # ⇒ ~20_000 est tokens
_HUGE_BODY = "x" * 400_000  # ⇒ ~200_000 est tokens


def _record_compression(monkeypatch) -> list[dict]:
    """Patch the module-level ``apply_message_compression`` and record calls."""
    calls: list[dict] = []

    def _fake(msgs, **kwargs):
        calls.append(kwargs)
        return msgs

    monkeypatch.setattr(
        "cogtrix_core.orchestration.nodes.call_model.apply_message_compression", _fake
    )
    return calls


def _run(monkeypatch, *, body: str = _BIG_BODY, **ctx_overrides):
    calls = _record_compression(monkeypatch)
    node = _make_node(
        invoke_with_timeout=lambda _llm, _msgs, _cfg, _to: AIMessage(content="ok"),
        # Passthrough so the separate context-message-cap path (triggered when
        # context_max_tokens > 0) doesn't interfere with the pre-flight guard
        # assertion; it accepts the ``evicted_summary`` kwarg the node passes.
        apply_context_message_cap=lambda msgs, *_a, **_k: msgs,
        tool_context_limit_pct=0.5,
        **ctx_overrides,
    )
    node(_make_state([HumanMessage(content=body)]), {})
    return calls


class TestPreFlightWindowGuard:
    def test_large_context_window_not_floored_by_small_output_cap(self, monkeypatch):
        # THE BUG: ctx 131072, output cap 8000. ~20k est tokens is well under
        # 131072 × 0.5 = 65536 → must NOT force pre-flight compression.
        calls = _run(
            monkeypatch,
            model_context_window=131_072,
            model_max_tokens=8_000,
            context_max_tokens=0,
            max_context_tokens=None,
        )
        assert calls == [], "output cap floored the input window — #2447 regression"

    def test_missing_context_window_falls_back_to_output_cap(self, monkeypatch):
        # No declared context_window → the guard must still protect: fall back to
        # the output cap (8000 × 0.5 = 4000 < ~20k est) → compression forced.
        calls = _run(
            monkeypatch,
            model_context_window=None,
            model_max_tokens=8_000,
            context_max_tokens=0,
            max_context_tokens=None,
        )
        assert len(calls) == 1, "guard must not be disabled when context_window is unknown"

    def test_operator_compression_cap_still_tightens_window(self, monkeypatch):
        # A tighter operator cap (context_max_tokens=10000) must still lower the
        # effective window even with a huge context_window: min(131072, 10000) ×
        # 0.5 = 5000 < ~20k est → compression forced.
        calls = _run(
            monkeypatch,
            model_context_window=131_072,
            model_max_tokens=8_000,
            context_max_tokens=10_000,
            max_context_tokens=None,
        )
        assert len(calls) == 1, "operator compression cap must still tighten the guard"

    def test_oversized_input_still_compresses_within_big_window(self, monkeypatch):
        # ~200k est tokens exceeds even 131072 × 0.5 = 65536 → the guard still
        # fires. The fix widens the window; it does not neuter the guard.
        calls = _run(
            monkeypatch,
            body=_HUGE_BODY,
            model_context_window=131_072,
            model_max_tokens=8_000,
            context_max_tokens=0,
            max_context_tokens=None,
        )
        assert len(calls) == 1, "guard must still trip when input truly exceeds the window"


class TestProviderStampsContextWindow:
    """The real input window is stamped onto the llm at provider-creation time
    so ``call_model`` (via graph.py) reads it back with
    ``getattr(llm, "_cogtrix_context_window", None)``."""

    def _configs(self, context_window):
        return (
            ProviderConfig(name="p", type="openai", api_key="sk-test"),
            ModelConfig(provider="p", model="m", context_window=context_window, max_tokens=8_000),
        )

    def test_stamps_configured_context_window(self, monkeypatch):
        monkeypatch.setattr(
            "cogtrix_core.providers.create_chat_model", lambda *_a, **_k: SimpleNamespace()
        )
        pc, mc = self._configs(131_072)
        llm = create_chat_model_from_configs(pc, mc)
        assert getattr(llm, "_cogtrix_context_window", None) == 131_072

    def test_stamps_default_when_context_window_unset(self, monkeypatch):
        monkeypatch.setattr(
            "cogtrix_core.providers.create_chat_model", lambda *_a, **_k: SimpleNamespace()
        )
        pc, mc = self._configs(None)
        llm = create_chat_model_from_configs(pc, mc)
        assert getattr(llm, "_cogtrix_context_window", None) == ModelConfig.DEFAULT_CONTEXT_WINDOW

    def test_unstampable_model_does_not_crash(self, monkeypatch):
        class _Frozen:
            __slots__ = ()  # cannot set arbitrary attributes

        monkeypatch.setattr("cogtrix_core.providers.create_chat_model", lambda *_a, **_k: _Frozen())
        pc, mc = self._configs(131_072)
        # Must not raise — the stamp is best-effort.
        llm = create_chat_model_from_configs(pc, mc)
        assert getattr(llm, "_cogtrix_context_window", None) is None
