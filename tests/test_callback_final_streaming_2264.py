"""#2264 — opt-in live streaming of the post-tool final answer.

#2251 suppressed post-tool final-answer tokens (delivered once at turn end) to
avoid a double-render when verification-recovery regenerates — the side effect
was dead air on tool turns. With ``stream_final_answer=True`` the handler streams
final tokens live and emits a single ``discard_pending`` control frame when a
regeneration (a NEW LLM run_id) supersedes a partially-streamed answer, so the
client can drop the stale partial. Default (flag off) keeps the #2251 behaviour.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastapi")

from cogtrix_core.api.callbacks import WebSocketCallbackHandler  # noqa: E402


def _handler(
    stream_final: bool,
) -> tuple[WebSocketCallbackHandler, asyncio.Queue, asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    q: asyncio.Queue = asyncio.Queue()
    handler = WebSocketCallbackHandler(ws_queue=q, loop=loop, stream_final_answer=stream_final)
    return handler, q, loop


def _drain(q: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> list[dict]:
    loop.run_until_complete(asyncio.sleep(0))  # flush call_soon_threadsafe
    items: list[dict] = []
    while not q.empty():
        items.append(q.get_nowait())
    return items


class TestFinalAnswerLiveStreaming:
    def test_final_tokens_stream_live_when_opted_in(self) -> None:
        handler, q, loop = _handler(stream_final=True)
        handler.tool_call_count = 1  # a tool ran; none in-flight → is_final
        handler.on_llm_new_token("final chunk", run_id="r1")
        frames = _drain(q, loop)
        loop.close()
        assert frames == [{"type": "token", "payload": {"text": "final chunk", "final": True}}]
        # Must NOT flag buffered — turn runner would otherwise re-emit at turn end.
        assert handler.final_answer_buffered is False

    def test_single_generation_emits_no_discard(self) -> None:
        handler, q, loop = _handler(stream_final=True)
        handler.tool_call_count = 1
        for tok in ("Answer ", "one"):
            handler.on_llm_new_token(tok, run_id="r1")
        frames = _drain(q, loop)
        loop.close()
        assert all(f["type"] == "token" for f in frames), "no discard on a single generation"
        assert [f["payload"]["text"] for f in frames] == ["Answer ", "one"]
        assert all(f["payload"]["final"] is True for f in frames)

    def test_regeneration_emits_one_discard_then_streams_survivor(self) -> None:
        handler, q, loop = _handler(stream_final=True)
        handler.tool_call_count = 2
        # Generation 1 (discarded by verification recovery) — run_id r1.
        for tok in ("A", "B"):
            handler.on_llm_new_token(tok, run_id="r1")
        # Generation 2 (survivor) — NEW run_id r2 → one discard_pending first.
        for tok in ("C", "D"):
            handler.on_llm_new_token(tok, run_id="r2")
        frames = _drain(q, loop)
        loop.close()

        types = [f["type"] for f in frames]
        assert types.count("discard_pending") == 1, "exactly one discard on regeneration"
        discard_idx = types.index("discard_pending")
        texts_before = [f["payload"]["text"] for f in frames[:discard_idx]]
        texts_after = [f["payload"]["text"] for f in frames[discard_idx + 1 :]]
        assert texts_before == ["A", "B"], "discard must follow the superseded partial"
        assert texts_after == ["C", "D"], "survivor streams after the discard"
        assert handler.final_answer_buffered is False

    def test_default_off_preserves_suppression(self) -> None:
        # Regression guard: without the opt-in, #2251 behaviour is unchanged.
        handler, q, loop = _handler(stream_final=False)
        handler.tool_call_count = 1
        handler.on_llm_new_token("final chunk", run_id="r1")
        frames = _drain(q, loop)
        loop.close()
        assert frames == [], "default must still suppress final tokens (no live stream)"
        assert handler.final_answer_buffered is True

    def test_non_final_tokens_still_stream_with_flag_on(self) -> None:
        # Inter-tool reasoning (tool in flight) streams live as non-final regardless.
        handler, q, loop = _handler(stream_final=True)
        handler.on_tool_start({"name": "search"}, "{}", run_id="t1")
        _drain(q, loop)
        handler.on_llm_new_token("thinking", run_id="r1")
        frames = _drain(q, loop)
        loop.close()
        assert frames == [{"type": "token", "payload": {"text": "thinking", "final": False}}]
