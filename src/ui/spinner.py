"""Animated activity spinner for the Cogtrix CLI."""

from __future__ import annotations

import sys
import threading
from typing import Any

_SPINNER_MESSAGES = [
    "Processing request ",
    "Warming up the circuits ",
    "Feeding data to the neurons ",
    "Parsing the input ",
    "Aligning the tokens ",
    "Traversing the attention layers ",
    "Computing hidden states ",
    "Evaluating possibilities ",
    "Thinking very hard ",
    "Exploring the latent space ",
    "Generating insights ",
    "Refining the answer ",
    "Cross-checking the results ",
    "Almost there... ",
    "Polishing the response ",
    "Patience, greatness takes time ",
    "Solar flares disrupting the attention mechanism ☀️ ",
    "Static from nylon underwear causing token embedding drift ⚡ ",
    "Fat electrons clogging the transformer layers 🍔 ",
    "Secretary plugged a hairdryer into the GPU power supply 💇 ",
    "Cosmic rays flipping bits in the weights 🪐 ",
    "Bogon emissions from the dataset poisoning the loss 🌫️ ",
    "Little hamster in the inference wheel needs coffee 🐹 ",
    "Gradient descent interrupted by tectonic stress 🌍 ",
    "Overfitting due to luser prompt injection 😈 ",
    "Hallucinations caused by floating point overflow in the decoder 🤯 ",
    "Waiting for the phone company to fix the context window 📞 ",
    "Positron router malfunction in the embedding space ⚛️ ",
    "We're upgrading /dev/attention for more heads 🔧 ",
    "Evil dogs hypnotized the training cluster 🐶 ",
    "Runt packets lost in the attention bottleneck 📦 ",
    "Mouse chewed through the fiber to the datacenter 🐭 ",
    "Temporal routing anomaly in the recurrent layers ⏳ ",
    "Daemons loose in the parameter server 👹 ",
    "UPS failed—blame the janitor's vacuum cleaner 🔌 ",
    "Nesting roaches shorted the tensor cores 🪳 ",
    "Quantum dynamics affecting the optimizer steps ⚛️ ",
    "The model is calculating pi on the hidden states 🧮 ",
    "High pressure system failure in the VRAM 🌪️ ",
    "Boss' kid fine-tuned the model on cat memes 😹 ",
    "Electromagnetic pulses from prompt engineering 📡 ",
    "Bit bucket overflow in the generation buffer 🗑️ ",
    "Zombie processes haunting the inference queue 🧟 ",
    "The Borg tried to assimilate the weights—resistance is futile 🛸 ",
    "Fluorescent lights generating negative gradients 💡 ",
    "Your prompt caused a divide-by-zero in the softmax ÷0 ",
    "We're wrapping the datacenter in aluminum foil 🛡️ ",
    "Lunar radiation interfering with backpropagation 🌕 ",
    "The kernel panicked: too many tokens in /dev/null 😱 ",
    "Small animal kamikaze attack on the cooling fans 🐦 ",
    "Vendor no longer supports this attention pattern 🚫 ",
    "Sticky bits on the learned representations 🧲 ",
    "Runaway cat on the server room floor 🐱 ",
    "Post-it note sludge leaked into the optimizer 📝 ",
    "The curls in the ethernet cable lost electricity 🌀 ",
    "Pygmy packets broadcast by a rogue tokenizer 🍼 ",
    "Fanout dropping voltage—try cutting traces on the GPU 🔪 ",
    "Due to budget cuts, we're training on CPU only 💸 ",
    "Lightning strike on the cloud provider ⚡ ",
    "The UPS is on strike—send coffee ☕ ",
    "Neutrino overload on the parameter server 🌌 ",
    "Melting hard drives from excessive inference 🔥 ",
    "Your flux capacitor needs realignment 🔋 ",
    "Interference between the keyboard and the chair 👨\u200d💻 ",
    "We ran out of compute credits—waiting for recharge 💳 ",
    "The token fell out of the ring—call when you find it 💍 ",
    "High altitude condensation contaminated the subnet mask ☁️ ",
    "Electrons on a bender in the attention heads 🍻 ",
    "Telecommunications downgrading the context length 📉 ",
    "Hard drive sleeping—let it wake up naturally 😴 ",
    "The CPU has shifted and become decentralized 🌐 ",
    "We ran out of dial tone for the API endpoint 📞 ",
    "Microelectronic Riemannian curved-space fault in the latent space 🌀 ",
    "Fractal radiation jamming the generation backbone 🌐 ",
    "IRQ problems with the Uninterruptible Prompt Supply ⚠️ ",
    "CPU-angle has exceeded velocity parameters 🚀 ",
    "Slow/Narrow attention interface problem ⏱️ ",
]

_SPINNER_CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"  # Smooth braille spinner

# Gradient palette for the spinner character — cycles through these
# ANSI 256-color codes to create a smooth color transition.
# Cyan → Blue → Magenta → Red → Yellow → Green → Cyan
_SPINNER_GRADIENT = [
    51,
    50,
    49,
    48,
    47,
    46,  # cyan → green
    82,
    118,
    154,
    190,
    226,  # green → yellow
    220,
    214,
    208,
    202,
    196,  # yellow → red
    197,
    198,
    199,
    200,
    201,  # red → magenta
    165,
    129,
    93,
    57,
    21,  # magenta → blue
    27,
    33,
    39,
    45,
    51,  # blue → cyan
]


class ActivityIndicator:
    """Animated spinner shown while the LLM is processing.

    Uses Rich ``console.print`` when available, falls back to raw ANSI.
    Exposes ``pause()`` / ``resume()`` so tool-confirmation prompts can
    temporarily clear the spinner line without stopping the thread.
    """

    # Change message roughly every 7 seconds (at ~0.1 s/frame)
    _MSG_INTERVAL = 70

    # Index of the first "fun" message (after "Patience, greatness takes time")
    _FUN_START = 16

    def __init__(self) -> None:
        self._msg_index = 0
        self._message = _SPINNER_MESSAGES[0]
        self._running = False
        self._pause_count: int = 0
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._escape_monitor: Any = None
        self._context: str = ""

    def set_escape_monitor(self, monitor: Any) -> None:
        """Attach an ``EscapeMonitor`` whose lifecycle mirrors the spinner."""
        self._escape_monitor = monitor

    def set_context(self, context: str) -> None:
        """Set prefix shown before the fun message. Thread-safe."""
        with self._lock:
            self._context = context

    def clear_context(self) -> None:
        with self._lock:
            self._context = ""

    def _next_message(self) -> str:
        """Return the next spinner message.

        The first 16 messages play in order (up to and including
        "Patience, greatness takes time").  After that, a random
        fun phrase is picked each time.
        """
        import random

        self._msg_index += 1
        if self._msg_index < self._FUN_START:
            return _SPINNER_MESSAGES[self._msg_index]
        return random.choice(_SPINNER_MESSAGES[self._FUN_START :])  # noqa: E203  # nosec B311

    # -- public API ---------------------------------------------------------

    @staticmethod
    def _tty_output_enabled() -> bool:
        import os

        if os.environ.get("NO_COLOR") is not None:
            return False
        return sys.stdout.isatty() or bool(os.environ.get("FORCE_COLOR"))

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._pause_count = 0
            self._msg_index = 0
            self._message = _SPINNER_MESSAGES[0]
            self._context = ""
            self._thread = threading.Thread(target=self._animate, daemon=True)
            self._thread.start()
        if self._escape_monitor is not None:
            self._escape_monitor.start()

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
        if self._escape_monitor is not None:
            self._escape_monitor.stop()
        if self._thread:
            self._thread.join(timeout=2)
        self._clear_line()

    def pause(self) -> None:
        """Temporarily hide the spinner (e.g. for user prompts)."""
        should_clear = False
        should_pause_monitor = False
        with self._lock:
            self._pause_count += 1
            if self._running and self._pause_count == 1:
                should_clear = True
                should_pause_monitor = True
        if should_clear:
            self._clear_line()
        if should_pause_monitor and self._escape_monitor is not None:
            self._escape_monitor.pause()

    def resume(self) -> None:
        """Re-show the spinner after a pause."""
        should_resume_monitor = False
        with self._lock:
            prev = self._pause_count
            self._pause_count = max(0, self._pause_count - 1)
            if prev == 1 and self._pause_count == 0 and self._running:
                should_resume_monitor = True
        if should_resume_monitor and self._escape_monitor is not None:
            self._escape_monitor.resume()

    # -- internals ----------------------------------------------------------

    def _animate(self) -> None:
        import time

        if not self._tty_output_enabled():
            return
        idx = 0
        frame_count = 0
        grad_len = len(_SPINNER_GRADIENT)
        while self._running:
            # Hold the lock only long enough to read state and build the
            # frame string — stdout I/O happens outside to prevent a
            # deadlock when the pipe buffer is full and flush() blocks
            # while the main thread waits on the same lock via pause().
            frame: str | None = None
            with self._lock:
                if self._pause_count == 0:
                    if frame_count % self._MSG_INTERVAL == 0 and frame_count > 0:
                        self._message = self._next_message()
                    char = _SPINNER_CHARS[idx % len(_SPINNER_CHARS)]
                    color = _SPINNER_GRADIENT[idx % grad_len]
                    ctx = self._context
                    # Always use raw stdout — Rich console.print doesn't
                    # handle carriage-return rewriting correctly.
                    # \033[2K = erase entire line, \r = return to column 0
                    # \033[38;5;Nm = 256-color foreground
                    if ctx:
                        frame = (
                            f"\033[2K\r\033[1;38;5;{color}m{char}\033[0m"
                            f" \033[1m{ctx}\033[22m: \033[2m{self._message}\033[0m"
                        )
                    else:
                        frame = (
                            f"\033[2K\r\033[1;38;5;{color}m{char}\033[0m"
                            f" \033[2m{self._message}\033[0m"
                        )
                    idx += 1
                    frame_count += 1
            if frame is not None:
                sys.stdout.write(frame)
                sys.stdout.flush()
            time.sleep(0.1)

    @staticmethod
    def _clear_line() -> None:
        if not ActivityIndicator._tty_output_enabled():
            return
        # \033[2K erases the entire line regardless of width; \r returns to column 0
        sys.stdout.write("\033[2K\r")
        sys.stdout.flush()


# Global spinner instance — shared so tool-confirmation can pause it.
_spinner = ActivityIndicator()
