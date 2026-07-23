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

from src.assistant.channel import Channel

log = logging.getLogger("cogtrix")

_DEFAULT_POLL_INTERVALS: dict[str, float] = {
    "whatsapp": 5.0,
    "telegram": 1.0,
}


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
    ) -> None:
        self._channels = channels
        self._handler = handler
        self._executor = executor
        self._config = config
        self._session_mgr = session_mgr
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

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
        for t in self._threads:
            t.join(timeout=10)

    def _poll_loop(self, channel: Channel) -> None:
        interval = self._get_poll_interval(channel.name)
        while not self._stop_event.is_set():
            try:
                messages = channel.poll()
                for msg in messages:
                    self._executor.submit(self._handler.handle, msg, channel)
            except Exception as exc:
                log.error("Poll error on %s: %s", channel.name, exc)
            self._stop_event.wait(timeout=interval)

    def _eviction_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._session_mgr.evict_idle()
            except Exception:
                pass
            self._stop_event.wait(timeout=60)

    def _get_poll_interval(self, channel_name: str) -> float:
        ch_cfgs: dict[str, Any] = self._config.get("channels", {})
        ch_cfg: dict[str, Any] = ch_cfgs.get(channel_name, {})
        default = _DEFAULT_POLL_INTERVALS.get(channel_name, 5.0)
        return float(ch_cfg.get("poll_interval", default))
