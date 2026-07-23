"""API-side tool confirmation UI.

Implements the ``ConfirmationUI`` Protocol from ``src/agent/safety.py`` for the
WebSocket session layer.  Instead of rendering a Rich panel in the terminal,
``render_prompt`` enqueues a ``tool_confirm_request`` message on the session's
WebSocket queue and ``read_choice`` blocks (on the agent thread) until the
WebSocket handler resolves the pending confirmation or the 5-minute timeout
expires.

Action mapping (WebSocket → CLI):
    allow      → "y"
    deny       → "n"
    allow_all  → "a"
    disable    → "d"
    forbid_all → "f"
    cancel     → "c"
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from typing import Any

log = logging.getLogger("cogtrix.api.confirmation")

# Map WebSocket action strings to CLI single-character codes.
_ACTION_MAP: dict[str, str] = {
    "allow": "y",
    "deny": "n",
    "allow_all": "a",
    "disable": "d",
    "forbid_all": "f",
    "cancel": "c",
}

# Default to deny on timeout (5 minutes).
_TIMEOUT_SECONDS = 300


class ApiConfirmationUI:
    """Tool confirmation UI that communicates via the session WebSocket queue.

    Implements the ``ConfirmationUI`` Protocol — safe to pass as
    ``confirmation_ui`` in ``AgentRunConfig`` for API sessions.
    """

    def __init__(self, ws_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
        self._queue = ws_queue
        self._loop = loop
        self._lock = threading.Lock()
        self._pending_event: threading.Event | None = None
        self._pending_action: str = "n"
        self._confirmation_id: str = ""
        self._cancel_requested: bool = False

    # ------------------------------------------------------------------
    # ConfirmationUI Protocol implementation
    # ------------------------------------------------------------------

    def render_prompt(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        last_keys: frozenset[str],
        preview_limit: int,
    ) -> None:
        """Enqueue a ``tool_confirm_request`` message to the WebSocket queue."""
        with self._lock:
            if self._pending_event is not None and not self._pending_event.is_set():
                # A previous read_choice() caller is still blocking on the old event.
                # Unblock it immediately with a denial so it never waits the full timeout.
                self._pending_action = "n"
                self._pending_event.set()
                log.warning(
                    "ApiConfirmationUI.render_prompt: displaced a pending confirmation for %s",
                    self._confirmation_id,
                )
            self._confirmation_id = str(uuid.uuid4())
            self._pending_event = threading.Event()
            self._pending_action = "n"
            self._cancel_requested = False
            confirmation_id = self._confirmation_id

        try:
            asyncio.run_coroutine_threadsafe(
                self._queue.put(
                    {
                        "type": "tool_confirm_request",
                        "payload": {
                            "confirmation_id": confirmation_id,
                            "tool": tool_name,
                            "parameters": tool_input,
                            "message": f"Tool '{tool_name}' requires confirmation",
                        },
                    }
                ),
                self._loop,
            )
        except Exception as exc:  # pragma: no cover
            log.warning("ApiConfirmationUI.render_prompt enqueue failed: %s", exc)

    def read_choice(self) -> str:
        """Block the agent thread until the WebSocket handler resolves the confirmation.

        Returns the CLI-equivalent action character ("y", "n", "a", "d", "f", "c").
        Defaults to "n" (deny) on timeout.

        Polls in short intervals so that task cancellation (via asyncio.to_thread
        unwinding) can be detected promptly via the cancel_requested flag rather
        than waiting the full 5-minute timeout.
        """
        event: threading.Event | None
        with self._lock:
            event = self._pending_event

        if event is not None:
            deadline = time.monotonic() + _TIMEOUT_SECONDS
            _POLL_INTERVAL = 0.5  # seconds per poll cycle
            while time.monotonic() < deadline:
                if event.wait(timeout=_POLL_INTERVAL):
                    break
                with self._lock:
                    cancelled = self._cancel_requested
                if cancelled:
                    break

        with self._lock:
            return self._pending_action

    def resolve(self, confirmation_id: str, action: str) -> bool:
        """Called by the WebSocket message handler when the client responds.

        Args:
            confirmation_id: Must match the ID sent in the ``tool_confirm_request``.
            action: WebSocket action string (e.g. "allow", "deny").

        Returns:
            True if the confirmation was resolved; False if the ID did not match.
        """
        with self._lock:
            if confirmation_id != self._confirmation_id or self._pending_event is None:
                return False
            self._pending_action = _ACTION_MAP.get(action, "n")
            self._pending_event.set()
            return True

    def cancel(self) -> None:
        """Signal the polling loop in read_choice to exit immediately.

        Called by the WebSocket cancel handler so a pending confirmation
        unblocks within one poll interval (~0.5 s) rather than after the
        full 5-minute timeout.
        """
        with self._lock:
            self._cancel_requested = True
            if self._pending_event is not None:
                self._pending_event.set()

    # ------------------------------------------------------------------
    # No-op spinner / message stubs (API has no terminal)
    # ------------------------------------------------------------------

    def show_message(self, message: str, style: str) -> None:  # noqa: ARG002
        pass

    def pause_spinner(self) -> None:
        pass

    def resume_spinner(self) -> None:
        pass
