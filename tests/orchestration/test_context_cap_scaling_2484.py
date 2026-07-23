"""Regression tests for #2484 — compression-storm root cause.

The low-level ``build_agent_graph(llm=...)`` path (eval harnesses + direct
embedders) does NOT run ``Config.resolve_context_max_tokens()`` (#2360), so a
big-window model was capped at the flat 40k default and force-compressed on
*every* turn — minutes-per-turn latency + timeouts on RAG tasks.

``_scale_default_cap_to_window`` fixes this at the point of use: when the cap is
left at its default, it scales to ``max(40_000, window // 2)`` using the input
window the provider factory stamps on the llm (``_cogtrix_context_window``). It
only ever *raises* the cap, so nothing that previously fit can start being
truncated.
"""

from __future__ import annotations

from types import SimpleNamespace

from cogtrix_core.orchestration.graph import (
    _DEFAULT_CONTEXT_MAX_TOKENS,
    _scale_default_cap_to_window,
)


def _llm(window: object) -> SimpleNamespace:
    return SimpleNamespace(_cogtrix_context_window=window)


class TestScaleDefaultCapToWindow:
    def test_big_window_default_cap_scales_to_half_window(self) -> None:
        # Kimi 262k → cap becomes window//2 (131072), not the flat 40k that
        # triggered compression every turn.
        assert _scale_default_cap_to_window(_DEFAULT_CONTEXT_MAX_TOKENS, _llm(262_144)) == 131_072

    def test_small_window_keeps_the_40k_floor(self) -> None:
        # window//2 (16384) < 40k → stays 40k. No regression for small models.
        assert _scale_default_cap_to_window(_DEFAULT_CONTEXT_MAX_TOKENS, _llm(32_768)) == 40_000

    def test_explicit_operator_cap_is_never_touched(self) -> None:
        # A non-default value = an explicit caller/operator cap; it wins, even on
        # a big-window model, and is never lowered.
        assert _scale_default_cap_to_window(20_000, _llm(262_144)) == 20_000
        assert _scale_default_cap_to_window(500_000, _llm(262_144)) == 500_000

    def test_unknown_window_keeps_default(self) -> None:
        # Mock llm / no stamped window → unchanged, no crash.
        assert (
            _scale_default_cap_to_window(_DEFAULT_CONTEXT_MAX_TOKENS, SimpleNamespace()) == 40_000
        )
        assert _scale_default_cap_to_window(_DEFAULT_CONTEXT_MAX_TOKENS, _llm(None)) == 40_000

    def test_bool_and_nonpositive_window_ignored(self) -> None:
        # isinstance-bool guard + >0 guard (MagicMock/exotic values are common in
        # tests; must never scale off a truthy-but-invalid window).
        assert _scale_default_cap_to_window(_DEFAULT_CONTEXT_MAX_TOKENS, _llm(True)) == 40_000
        assert _scale_default_cap_to_window(_DEFAULT_CONTEXT_MAX_TOKENS, _llm(0)) == 40_000
        assert _scale_default_cap_to_window(_DEFAULT_CONTEXT_MAX_TOKENS, _llm(-5)) == 40_000

    def test_never_lowers_a_big_default_below_the_floor(self) -> None:
        # Sanity: the result is always >= the default (only ever raises).
        for win in (8_000, 32_768, 131_072, 262_144, 1_000_000):
            assert (
                _scale_default_cap_to_window(_DEFAULT_CONTEXT_MAX_TOKENS, _llm(win))
                >= _DEFAULT_CONTEXT_MAX_TOKENS
            )
