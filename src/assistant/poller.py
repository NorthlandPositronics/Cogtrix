"""
Channel polling loop for Cogtrix assistant mode.

Spawns one polling thread per channel and one background session-eviction thread.
New messages are submitted to a shared ThreadPoolExecutor so different chats are
processed concurrently while same-chat ordering is enforced by session.lock.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from src.assistant.channel import Channel, IncomingMessage

log = logging.getLogger("cogtrix")

_DEFAULT_POLL_INTERVALS: dict[str, float] = {
    "whatsapp": 5.0,
    "telegram": 1.0,
}


class MessageBuffer:
    """Per-chat message buffer with debounce dispatch.

    Collects messages for the same chat and dispatches them as a batch
    after a configurable quiet window, preventing rapid-fire messages
    from triggering separate agent runs.
    """

    def __init__(
        self,
        handler: Any,
        executor: Any,
        debounce_seconds: float = 3.0,
    ) -> None:
        self._handler = handler
        self._executor = executor
        self._debounce = debounce_seconds
        self._buffers: dict[str, list[tuple[IncomingMessage, Channel]]] = {}
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def add(self, msg: IncomingMessage, channel: Channel) -> None:
        """Buffer a message; reset the debounce timer for its chat."""
        key = msg.session_key
        with self._lock:
            if key not in self._buffers:
                self._buffers[key] = []
            self._buffers[key].append((msg, channel))
            if key in self._timers:
                self._timers[key].cancel()
            timer = threading.Timer(self._debounce, self._flush, args=(key,))
            timer.daemon = True
            timer.start()
            self._timers[key] = timer

    def _flush(self, key: str) -> None:
        """Dispatch all buffered messages for *key* as a single batch."""
        with self._lock:
            batch = self._buffers.pop(key, [])
            self._timers.pop(key, None)
        if not batch:
            return
        messages = [m for m, _ in batch]
        channel = batch[0][1]
        self._executor.submit(self._handler.handle_batch, messages, channel)

    def flush_all(self) -> None:
        """Force-flush all pending buffers (called on shutdown)."""
        with self._lock:
            keys = list(self._buffers.keys())
            for key in keys:
                timer = self._timers.pop(key, None)
                if timer:
                    timer.cancel()
        for key in keys:
            self._flush(key)
            with self._lock:
                timer = self._timers.pop(key, None)
                if timer:
                    timer.cancel()


class ChannelPoller:
    """Manages per-channel polling threads and a session-eviction thread.

    Args:
        channels: List of Channel instances to poll.
        handler: MessageHandler called for each new message.
        executor: ThreadPoolExecutor for concurrent message processing.
        config: Assistant-mode config dict (services.assistant section).
        session_mgr: ChatSessionManager whose evict_idle() is called periodically.
    """

    def __init__(
        self,
        channels: list[Channel],
        handler: Any,
        executor: Any,
        config: dict[str, Any],
        session_mgr: Any,
        debounce_seconds: float = 3.0,
    ) -> None:
        self._channels = channels
        self._handler = handler
        self._executor = executor
        self._config = config
        self._session_mgr = session_mgr
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._buffer = MessageBuffer(handler, executor, debounce_seconds)

    def start(self) -> None:
        """Start per-channel polling threads and the eviction thread."""
        for ch in self._channels:
            t = threading.Thread(
                target=self._poll_loop,
                args=(ch,),
                daemon=True,
                name=f"poller-{ch.name}",
            )
            t.start()
            self._threads.append(t)

        evictor = threading.Thread(
            target=self._eviction_loop,
            daemon=True,
            name="session-evictor",
        )
        evictor.start()
        self._threads.append(evictor)

    def stop(self) -> None:
        """Signal all threads to stop and wait for them to finish."""
        self._stop_event.set()
        self._buffer.flush_all()
        for t in self._threads:
            t.join(timeout=10)

    def _poll_loop(self, channel: Channel) -> None:
        base_interval = self._get_poll_interval(channel.name)
        ch_cfgs: dict[str, Any] = self._config.get("channels", {})
        ch_cfg: dict[str, Any] = ch_cfgs.get(channel.name, {})
        min_interval = float(ch_cfg.get("poll_interval_min", base_interval))
        max_interval = float(ch_cfg.get("poll_interval_max", 60.0))
        backoff_factor = max(1.0, float(ch_cfg.get("poll_backoff_factor", 1.5)))
        recovery_factor = max(1.0, float(ch_cfg.get("poll_recovery_factor", 2.0)))

        current_interval = min_interval

        while not self._stop_event.is_set():
            had_activity = False
            try:
                messages = channel.poll()
                had_activity = bool(messages)
                for msg in messages:
                    self._buffer.add(msg, channel)
            except Exception as exc:
                log.error("Poll error on %s: %s", channel.name, exc)
                self._stop_event.wait(timeout=current_interval)
                continue

            if had_activity:
                current_interval = max(min_interval, current_interval / recovery_factor)
            else:
                current_interval = min(max_interval, current_interval * backoff_factor)

            self._stop_event.wait(timeout=current_interval)

    def _eviction_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._session_mgr.evict_idle()
            except Exception as exc:
                log.error("Session eviction failed: %s", exc)
            self._stop_event.wait(timeout=60)

    def _get_poll_interval(self, channel_name: str) -> float:
        ch_cfgs: dict[str, Any] = self._config.get("channels", {})
        ch_cfg: dict[str, Any] = ch_cfgs.get(channel_name, {})
        default = _DEFAULT_POLL_INTERVALS.get(channel_name, 5.0)
        return float(ch_cfg.get("poll_interval", default))
