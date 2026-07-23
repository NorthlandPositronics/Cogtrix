"""Regression pin for #2481 — qwen3-coder must route to the LOCAL Spark cluster.

qwen3-coder is served from a local vLLM/Spark endpoint and is NOT hosted on
OpenRouter. Without ``prefer_native``, an ambient ``OPENROUTER_API_KEY`` makes
``resolve_active_key()`` return OpenRouter first and ``_build_llm`` routes the
model there → auth/routing errors that invalidate the whole qwen column (the
2026-07-07 role-test finding). The ``prefer_native`` *mechanism* is covered in
``test_deepseek_reasoning.py``; this test pins the qwen registry entry so the
fix cannot silently regress.
"""

from __future__ import annotations

from tests.evaluation.runner import get_model


def test_qwen3_coder_prefers_native_local_route() -> None:
    m = get_model("qwen3-coder")
    assert m.prefer_native is True, (
        "qwen3-coder must set prefer_native: true (#2481) — otherwise an ambient "
        "OPENROUTER_API_KEY routes it to OpenRouter, which 404s the model."
    )
    # It is a local-Spark model: native env_key + a LAN base_url, no OpenRouter id.
    assert m.env_key == "VLLM_LOCAL_API_KEY"
    assert m.base_url and m.base_url.startswith("http://")
    assert m.openrouter_model_id is None
