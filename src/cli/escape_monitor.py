"""Background Escape-key monitor that translates Escape to SIGINT.

Runs a daemon thread that reads raw keystrokes from stdin while the
spinner is active.  A standalone Escape byte (not part of an escape
sequence like arrow keys) sends SIGINT to the main process, which
the existing KeyboardInterrupt handler in cogtrix.py catches.

No-op on non-tty stdin, missing termios (Windows), or when imported
by a non-interactive process (assistant mode, piped input).
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Any

# Guard: only available on Unix with a real terminal fd
_AVAILABLE = False

try:
    import select
    import signal
    import termios
    import tty

    if hasattr(sys.stdin, "fileno"):
        try:
            _fd_check = sys.stdin.fileno()
            if os.isatty(_fd_check):
                _AVAILABLE = True
        except (ValueError, OSError):
            pass
except ImportError:
    pass


class EscapeMonitor:
    """Monitors stdin for standalone Escape keypresses in a background thread.

    Lifecycle mirrors ``ActivityIndicator``: start / stop / pause / resume.
    """

    _ESCAPE_DELAY = 0.05  # 50 ms — disambiguate Escape from escape sequences
    _POLL_INTERVAL = 0.1  # 100 ms — select() timeout per iteration
    _WARMUP_SECS = 0.3  # 300 ms — ignore input after cbreak entry
    _MAX_CONSECUTIVE_ERRORS = 5  # BUG-105: break out after this many select() errors

    def __init__(self) -> None:
        self._running = False
        self._paused = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._terminal_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._saved_attrs: list[Any] | None = None
        self._fd: int | None = None

    @property
    def available(self) -> bool:
        return _AVAILABLE

    # -- public lifecycle ---------------------------------------------------

    def start(self) -> None:
        if not _AVAILABLE:
            return
        with self._lock:
            if self._running:
                return
            self._running = True
            self._paused = False
            self._stop_event.clear()
        if self._fd is None:
            try:
                self._fd = sys.stdin.fileno()
            except (ValueError, OSError):
                self._running = False
                return
        # cbreak mode is entered inside _monitor_loop, not here — the
        # thread needs a warmup window to drain stale terminal bytes
        # before arming Escape detection.
        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="escape-monitor"
        )
        self._thread.start()

    def stop(self) -> None:
        # BUG-101: always join and clear _thread regardless of _running state.
        if not _AVAILABLE:
            return
        with self._lock:
            already_stopped = not self._running
            self._running = False
        if not already_stopped:
            self._stop_event.set()
            self._restore_terminal()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def pause(self) -> None:
        """Restore normal terminal mode for ``input()`` / readline."""
        if not _AVAILABLE:
            return
        with self._lock:
            if not self._running or self._paused:
                return
            self._paused = True
        self._restore_terminal()

    def resume(self) -> None:
        """Re-enter cbreak mode after ``input()`` completes."""
        if not _AVAILABLE:
            return
        with self._lock:
            if not self._running or not self._paused:
                return
            self._paused = False
        self._enter_cbreak()

    # -- terminal mode management -------------------------------------------

    def _enter_cbreak(self) -> None:
        with self._terminal_lock:
            fd = self._fd
            if fd is None:
                return
            try:
                if self._saved_attrs is None:
                    self._saved_attrs = termios.tcgetattr(fd)
            except (termios.error, OSError, ValueError):
                return
            try:
                tty.setcbreak(fd)
            except (termios.error, OSError, ValueError):
                if self._saved_attrs is not None:
                    try:
                        termios.tcsetattr(fd, termios.TCSADRAIN, self._saved_attrs)
                    except (termios.error, OSError, ValueError):
                        pass

    def _restore_terminal(self) -> None:
        with self._terminal_lock:
            try:
                if self._saved_attrs is not None and self._fd is not None:
                    termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved_attrs)
                    self._saved_attrs = None
            except (termios.error, OSError, ValueError):
                pass

    # -- monitor loop -------------------------------------------------------

    def _monitor_loop(self) -> None:
        fd = self._fd
        if fd is None:
            return

        # ── Warmup ────────────────────────────────────────────────────
        # After readline returns from input(), the terminal emulator may
        # still be sending responses (bracketed-paste disable, cursor-
        # position replies, mode acknowledgements).  These contain \x1b
        # bytes that look like standalone Escape if we enter cbreak too
        # soon.  Strategy:
        #   1. Wait in cooked mode so stale bytes are captured by the
        #      line discipline (not delivered to us).
        #   2. Flush the kernel input buffer.
        #   3. Enter cbreak mode.
        #   4. Drain anything that slipped through during the switch.
        if self._stop_event.wait(timeout=self._WARMUP_SECS):
            return

        # BUG-100: if pause() was called during the warmup window, skip
        # flush/cbreak/drain and wait until unpaused or stopped.
        with self._lock:
            paused_after_warmup = self._paused

        if paused_after_warmup:
            while True:
                if self._stop_event.is_set():
                    return
                with self._lock:
                    if not self._paused:
                        break
                self._stop_event.wait(timeout=self._POLL_INTERVAL)

        try:
            termios.tcflush(fd, termios.TCIFLUSH)
        except (termios.error, OSError, ValueError):
            pass

        self._enter_cbreak()

        # Short drain after cbreak entry — catch bytes that arrived
        # between the flush and the mode switch.
        for _ in range(3):  # 3 × 30 ms = 90 ms
            # BUG-104: also exit drain if paused during this window.
            if self._stop_event.is_set() or self._paused:
                return
            try:
                ready, _, _ = select.select([fd], [], [], 0.03)
                if ready:
                    os.read(fd, 256)  # discard
            except (InterruptedError, OSError, ValueError):
                pass

        # ── Main detection loop ───────────────────────────────────────
        consecutive_errors = 0  # BUG-105: track select() failures
        while not self._stop_event.is_set():
            with self._lock:
                if not self._running:
                    break
                paused = self._paused

            if paused:
                self._stop_event.wait(timeout=self._POLL_INTERVAL)
                continue

            # BUG-103: belt-and-suspenders lockless check before select()
            if self._paused:
                continue

            try:
                ready, _, _ = select.select([fd], [], [], self._POLL_INTERVAL)
                consecutive_errors = 0  # reset on success
            except (InterruptedError, OSError, ValueError) as exc:
                # BUG-105: only OSError/ValueError count toward the error budget
                if isinstance(exc, (OSError, ValueError)):
                    consecutive_errors += 1
                    if consecutive_errors >= self._MAX_CONSECUTIVE_ERRORS:
                        break
                continue

            if not ready:
                continue

            try:
                byte = os.read(fd, 1)
            except (InterruptedError, OSError):
                continue

            if not byte:
                continue

            if byte == b"\x1b":
                # Could be standalone Escape or start of an escape sequence
                # (e.g. \x1b[A for Up arrow).  Wait briefly for more bytes.
                try:
                    follow, _, _ = select.select([fd], [], [], self._ESCAPE_DELAY)
                except (InterruptedError, OSError, ValueError):
                    continue

                if follow:
                    # Escape sequence — drain remaining bytes
                    for _ in range(64):  # cap at 64 iterations to avoid CPU spin on large pastes
                        if self._stop_event.is_set():
                            break
                        try:
                            more, _, _ = select.select([fd], [], [], 0.01)
                            if not more:
                                break
                            os.read(fd, 16)
                        except (InterruptedError, OSError):
                            break
                else:
                    # Standalone Escape — restore terminal then fire SIGINT
                    self._restore_terminal()
                    with self._lock:
                        self._running = False
                    self._stop_event.set()
                    try:
                        os.kill(os.getpid(), signal.SIGINT)
                    except OSError:
                        pass
            # Other bytes during LLM processing are silently discarded.
