"""#2251 — post-tool final-answer tokens are buffered (suppressed) from the live
stream so a verification-recovery regeneration can't double-render.

The verification-recovery loop can discard a fully-generated final answer
(``RemoveMessage``) and regenerate; streaming each generation live rendered TWO
answers on the WS client while only the surviving one was persisted. The WS
callback now suppresses ``is_final`` tokens and flags ``final_answer_buffered`` so
the turn runner emits the single surviving answer once at turn end. Non-final
tokens (no-tool answers, inter-tool reasoning, in-flight-tool tokens) still stream.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastapi")

from cogtrix_core.api.callbacks import WebSocketCallbackHandler  # noqa: E402


def _handler_and_loop() -> (
    tuple[WebSocketCallbackHandler, asyncio.Queue, asyncio.AbstractEventLoop]
):
    loop = asyncio.new_event_loop()
    q: asyncio.Queue = asyncio.Queue()
    handler = WebSocketCallbackHandler(ws_queue=q, loop=loop)
    return handler, q, loop


def _drain(q: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> list[dict]:
    loop.run_until_complete(asyncio.sleep(0))  # flush call_soon_threadsafe
    items: list[dict] = []
    while not q.empty():
        items.append(q.get_nowait())
    return items


class TestFinalAnswerSuppression:
    def test_final_token_suppressed_and_flagged(self) -> None:
        handler, q, loop = _handler_and_loop()
        handler.tool_call_count = 1  # a tool ran; no tool in-flight → is_final
        handler.on_llm_new_token("final answer chunk")
        frames = _drain(q, loop)
        loop.close()
        assert frames == [], "post-tool final-answer tokens must NOT stream live"
        assert handler.final_answer_buffered is True

    def test_no_tool_token_streams_live(self) -> None:
        handler, q, loop = _handler_and_loop()
        # No tools called this turn → tokens are never 'final' → stream live.
        handler.on_llm_new_token("hello ")
        handler.on_llm_new_token("world")
        frames = _drain(q, loop)
        loop.close()
        assert [f["payload"]["text"] for f in frames] == ["hello ", "world"]
        assert all(f["type"] == "token" for f in frames)
        assert handler.final_answer_buffered is False

    def test_in_flight_tool_token_streams_live(self) -> None:
        handler, q, loop = _handler_and_loop()
        handler.on_tool_start({"name": "search"}, "{}", run_id="r1")  # tool in flight
        _drain(q, loop)  # discard the tool_start frame
        handler.on_llm_new_token("reasoning while tool runs")
        frames = _drain(q, loop)
        loop.close()
        # tool_call_count>0 but a tool is in-flight → not final → streams.
        assert [f["payload"]["text"] for f in frames] == ["reasoning while tool runs"]
        assert handler.final_answer_buffered is False

    def test_recovery_double_generation_emits_no_live_final_tokens(self) -> None:
        """Two post-tool final generations (a recovery regeneration) → BOTH are
        suppressed live, so the client never receives a duplicated answer; the
        single surviving answer is delivered by the turn runner at turn end."""
        handler, q, loop = _handler_and_loop()
        handler.tool_call_count = 2
        # Generation 1 (later discarded by verification recovery)
        for tok in ("Answer ", "one"):
            handler.on_llm_new_token(tok)
        # Generation 2 (the survivor)
        for tok in ("Answer ", "two"):
            handler.on_llm_new_token(tok)
        frames = _drain(q, loop)
        loop.close()
        token_frames = [f for f in frames if f["type"] == "token"]
        assert token_frames == [], "no final-answer tokens may stream live (no double-render)"
        assert handler.final_answer_buffered is True
