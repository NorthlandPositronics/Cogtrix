"""WebSocket callback handler for LangChain agent streaming.

Bridges synchronous LangChain callbacks to an async WebSocket queue using
``asyncio.run_coroutine_threadsafe``.  The handler runs on the agent thread
(inside ``asyncio.to_thread``) and safely enqueues typed messages onto the
event loop's queue so the WebSocket drain task can forward them to the client.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Any

log = logging.getLogger("cogtrix.api.callbacks")

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:  # pragma: no cover
    BaseCallbackHandler = object  # type: ignore[misc, assignment]


class WebSocketCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that forwards events to a WebSocket queue.

    Instantiated per agent turn.  The owning coroutine captures the running
    event loop at connect time and passes it here so ``run_coroutine_threadsafe``
    can safely cross the thread boundary.

    Also accumulates token usage so the ``done`` payload and session
    token_counts are populated correctly.
    """

    def __init__(
        self,
        ws_queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
        *,
        stream_final_answer: bool = False,
    ) -> None:
        super().__init__()
        self._queue = ws_queue
        self._loop = loop
        self._tool_starts: dict[str, float] = {}  # str(run_id) -> start_time
        self._tool_starts_lock = threading.Lock()  # guards _tool_starts against concurrent access
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.tool_call_count: int = 0
        # #2251: True once any post-tool *final-answer* token has been buffered
        # (suppressed from the live stream). The turn runner consults this to emit
        # the surviving answer once at turn end — see on_llm_new_token.
        self.final_answer_buffered: bool = False
        # #2264: opt-in live streaming of the post-tool final answer. When True,
        # final-answer tokens stream live (final: True) and a ``discard_pending``
        # control frame is emitted if a verification-recovery regeneration starts a
        # NEW final-answer LLM run (detected by run_id change), so the client can
        # drop the superseded partial. Default False preserves the #2251 suppress +
        # single-frame behaviour for clients that don't handle ``discard_pending``.
        self._stream_final_answer: bool = stream_final_answer
        # run_id of the final-answer LLM run currently streaming live (#2264); set on
        # the first is_final token of a run. A different run_id arriving with is_final
        # after this is set = a regeneration → emit one discard_pending, then adopt it.
        self._final_run_id: str | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enqueue(self, msg_type: str, payload: dict[str, Any]) -> None:
        """Thread-safe enqueue via call_soon_threadsafe.

        Uses put_nowait so that a full queue (no active WS drain task) drops
        the message rather than blocking or growing the queue unboundedly
        (BUG-FORGE-004).
        """
        item = {"type": msg_type, "payload": payload}
        try:
            self._loop.call_soon_threadsafe(self._try_put_nowait, item)
        except RuntimeError:
            pass  # event loop closed

    def _try_put_nowait(self, item: dict) -> None:
        """Synchronous put_nowait called from the event loop thread."""
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            pass

    # ------------------------------------------------------------------
    # LLM events
    # ------------------------------------------------------------------

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        """Log the start of an LLM call."""
        if log.isEnabledFor(logging.DEBUG):
            model_name = (serialized or {}).get("name") or (serialized or {}).get("id", ["?"])[-1]
            # Rough token estimate: ~4 chars per token
            prompt_chars = sum(len(p) for p in (prompts or []))
            log.debug("LLM call started: model=%s prompt_chars=%d", model_name, prompt_chars)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Forward a streaming token to the WebSocket.

        #2251: post-tool *final-answer* tokens (``is_final``) are NOT streamed
        live. The verification-recovery loop can discard a fully-generated final
        answer (``RemoveMessage``) and regenerate, so streaming each generation
        live would render TWO answers on the client while only the surviving one
        is persisted. Instead, these tokens are buffered (suppressed); the turn
        runner emits the single surviving answer once at turn end (the text from
        ``extract_response``). Non-final tokens — including no-tool answers and
        inter-tool reasoning — still stream live as before.
        """
        # BUG-218: final is True only when tool calls have been seen AND none
        # are currently in-flight — avoids marking intermediate reasoning tokens
        # between tool calls as final.
        with self._tool_starts_lock:
            is_final = self.tool_call_count > 0 and len(self._tool_starts) == 0
        if is_final:
            if not self._stream_final_answer:
                # #2251 default: suppress from the live stream; the surviving final
                # answer is emitted once at turn end. Flag it for the turn runner.
                self.final_answer_buffered = True
                return
            # #2264 opt-in: stream the final answer live. A verification-recovery
            # regeneration is a NEW LLM run, so a run_id change after we've begun a
            # final-answer stream means the earlier partial was discarded — tell the
            # client to drop it before the survivor streams.
            run_id = kwargs.get("run_id")
            run_id_str = str(run_id) if run_id is not None else None
            if self._final_run_id is not None and run_id_str != self._final_run_id:
                self._enqueue("discard_pending", {})
            self._final_run_id = run_id_str
            self._enqueue("token", {"text": token, "final": True})
            return
        self._enqueue("token", {"text": token, "final": False})

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Accumulate token usage from each LLM call.

        Handles three token-count locations used by different providers:
        1. ``llm_output.token_usage``  — OpenAI (prompt_tokens / completion_tokens)
        2. ``llm_output.usage``        — some OpenAI-compat providers (same keys)
        3. ``generation.message.usage_metadata`` — LangChain standard field;
           may be a dict (most providers) or an object with attributes (older
           LangChain versions) — both are handled via duck-typing (BUG-FORGE-003).
        """
        llm_output = getattr(response, "llm_output", None)
        if llm_output:
            usage = llm_output.get("token_usage") or llm_output.get("usage")
            if usage:
                self.input_tokens += usage.get("prompt_tokens", 0)
                self.output_tokens += usage.get("completion_tokens", 0)
                log.debug(
                    "LLM call ended: completion_tokens=%d",
                    usage.get("completion_tokens", 0),
                )
                return
        gens = getattr(response, "generations", None)
        if gens:
            for gen_list in gens:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    if msg:
                        um = getattr(msg, "usage_metadata", None)
                        if um:
                            if isinstance(um, dict):
                                self.input_tokens += um.get("input_tokens", 0)
                                out = um.get("output_tokens", 0)
                                self.output_tokens += out
                            else:
                                self.input_tokens += getattr(um, "input_tokens", 0)
                                out = getattr(um, "output_tokens", 0)
                                self.output_tokens += out
                            log.debug("LLM call ended: output_tokens=%d", out)

    def on_llm_error(self, error: BaseException | str, **kwargs: Any) -> None:
        """Forward an LLM-level error to the WebSocket."""
        log.warning("LLM error: %s", error)
        self._enqueue("error", {"code": "AGENT_ERROR", "message": str(error)})

    # ------------------------------------------------------------------
    # Tool events
    # ------------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str | dict,
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """Notify the WebSocket that a tool invocation has started."""
        key = str(run_id) if run_id is not None else ""
        with self._tool_starts_lock:
            self._tool_starts[key] = time.monotonic()
            self.tool_call_count += 1
        tool_name: str = serialized.get("name", "unknown") if serialized else "unknown"
        if isinstance(input_str, dict):
            tool_input: dict = input_str
        else:
            try:
                parsed = json.loads(input_str)
                tool_input = parsed if isinstance(parsed, dict) else {}
            except Exception:
                tool_input = {}
        input_preview = str(tool_input)[:120]
        log.debug("Tool start: %s input=%.120s", tool_name, input_preview)
        self._enqueue(
            "tool_start",
            {
                "tool_name": tool_name,
                "tool_call_id": key,
                "input": tool_input,
            },
        )

    def on_tool_end(self, output: Any, *, run_id: Any = None, **kwargs: Any) -> None:
        """Notify the WebSocket that a tool invocation completed successfully."""
        key = str(run_id) if run_id is not None else ""
        with self._tool_starts_lock:
            start = self._tool_starts.pop(key, time.monotonic())
        duration_ms = int((time.monotonic() - start) * 1000)
        tool_name: str = kwargs.get("name", "unknown")
        log.debug(
            "Tool end: %s duration_ms=%d output=%.120s", tool_name, duration_ms, str(output)[:120]
        )
        self._enqueue(
            "tool_end",
            {
                "tool_name": tool_name,
                "tool_call_id": key,
                "duration_ms": duration_ms,
                "error": None,
            },
        )

    def on_tool_error(
        self, error: BaseException | str, *, run_id: Any = None, **kwargs: Any
    ) -> None:
        """Notify the WebSocket that a tool invocation failed."""
        key = str(run_id) if run_id is not None else ""
        tool_name_err: str = kwargs.get("name", "unknown")
        log.warning("Tool error: %s error=%s", tool_name_err, error)
        with self._tool_starts_lock:
            start = self._tool_starts.pop(key, time.monotonic())
        duration_ms = int((time.monotonic() - start) * 1000)
        tool_name: str = kwargs.get("name", "unknown")
        self._enqueue(
            "tool_end",
            {
                "tool_name": tool_name,
                "tool_call_id": key,
                "duration_ms": duration_ms,
                "error": str(error),
            },
        )
