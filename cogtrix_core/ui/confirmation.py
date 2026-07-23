"""Tool-confirmation UI helpers for the Cogtrix CLI."""

from __future__ import annotations

from typing import Any

from cogtrix_core.ui.spinner import _spinner

try:
    from langchain_core.callbacks import BaseCallbackHandler as _BaseCallback
except ImportError:
    _BaseCallback = object  # type: ignore[misc, assignment]

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
except ImportError:
    Console = None  # type: ignore[misc, assignment]
    Panel = None  # type: ignore[misc, assignment]
    Text = None  # type: ignore[misc, assignment]


# Module-level state (set by configure())
_console: _Console | None = None
_text: Any = None
_panel: Any = None


def configure(console: Any) -> None:
    """Configure the module with runtime dependencies.

    Called once at startup to inject the console instance.
    This ensures consistent module-level state and follows the pattern
    used by other tools modules (commands, slack_tools, rag, delegate, etc.).
    """
    global _console, _text, _panel
    _console = console
    if Text is not None:
        _text = Text
    if Panel is not None:
        _panel = Panel


def get_console() -> _Console | None:
    """Return the configured console instance."""
    return _console


def get_text() -> Any:
    """Return the configured Text class."""
    return _text


def get_panel() -> Any:
    """Return the configured Panel class."""
    return _panel


if Console is not None:

    class _Console(Console):  # type: ignore[misc, valid-type]
        """Console variant that defaults crop=False so panel borders are never clipped."""

        def print(self, *objects: Any, crop: bool = False, **kwargs: Any) -> None:
            super().print(*objects, crop=crop, **kwargs)

    _text = Text
    _panel = Panel
else:
    _Console = None  # type: ignore[assignment, misc]
    _text = None
    _panel = None

# Backward compatibility aliases
console: _Console | None = _console
Text: Any = _text
Panel: Any = _panel


class _TokenAccumulator(_BaseCallback):  # type: ignore[misc]
    """Accumulates token usage across LLM calls within a single agent run."""

    def __init__(self) -> None:
        super().__init__()
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.last_input_tokens: int = 0

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        if not hasattr(self, "input_tokens"):
            return
        llm_output = getattr(response, "llm_output", None)
        if llm_output:
            usage = llm_output.get("token_usage") or llm_output.get("usage")
            if usage:
                prompt = usage.get("prompt_tokens", 0)
                self.input_tokens += prompt
                self.output_tokens += usage.get("completion_tokens", 0)
                if prompt > 0:
                    self.last_input_tokens = prompt
                return
        gens = getattr(response, "generations", None)
        if gens:
            for gen_list in gens:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    if msg:
                        um = getattr(msg, "usage_metadata", None)
                        if um:
                            # UsageMetadata is a dict subclass — use dict access
                            inp = (
                                um.get("input_tokens", 0)
                                if isinstance(um, dict)
                                else getattr(um, "input_tokens", 0)
                            )
                            out = (
                                um.get("output_tokens", 0)
                                if isinstance(um, dict)
                                else getattr(um, "output_tokens", 0)
                            )
                            self.input_tokens += inp
                            self.output_tokens += out
                            if inp > 0:
                                self.last_input_tokens = inp
                            return
                        # Ollama returns prompt_eval_count in response_metadata
                        rm = getattr(msg, "response_metadata", None)
                        if rm and isinstance(rm, dict):
                            inp = rm.get("prompt_eval_count", 0)
                            out = rm.get("eval_count", 0)
                            if inp or out:
                                self.input_tokens += inp
                                self.output_tokens += out
                                if inp > 0:
                                    self.last_input_tokens = inp


class _RichConfirmationUI:
    """ConfirmationUI implementation using Rich panels and stdin."""

    # Keys hidden from the confirmation panel: LangChain tool_call envelope
    # metadata and default-valued parameters that add noise.
    _HIDDEN_KEYS = frozenset({"timeout", "type", "name", "id"})

    def render_prompt(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        last_keys: frozenset[str],
        preview_limit: int,
    ) -> None:
        def _preview(val: object) -> str:
            s = str(val)
            if len(s) <= preview_limit:
                return s
            return s[:preview_limit] + f"… ({len(s)} chars total)"

        def _is_hidden(key: str, value: object) -> bool:
            if key in self._HIDDEN_KEYS:
                return True
            if value is None or str(value).strip() in ("", "None"):
                return True
            return False

        # Unwrap LangChain tool_call envelope if present
        if (
            isinstance(tool_input, dict)
            and "args" in tool_input
            and isinstance(tool_input["args"], dict)
        ):
            tool_input = tool_input["args"]

        if _console is not None and _text is not None and _panel is not None:
            visible: list[tuple[str, object]] = []
            if isinstance(tool_input, dict) and tool_input:
                sorted_keys = sorted(
                    tool_input.keys(),
                    key=lambda k: (k in last_keys, len(str(tool_input[k]))),
                )
                for key in sorted_keys:
                    value = tool_input[key]
                    if not _is_hidden(key, value):
                        visible.append((key, value))

            if len(visible) == 1:
                _, value = visible[0]
                val_str = _preview(value).replace("[", chr(92) + "[")
                params_text = f"  {val_str}"
            elif visible:
                lines = []
                for key, value in visible:
                    val_str = _preview(value).replace("[", chr(92) + "[")
                    lines.append(f"  [dim cyan]{key:<18s}[/dim cyan] {val_str}")
                params_text = "\n".join(lines)
            else:
                params_text = "  [dim](no parameters)[/dim]"

            panel_title = _text()
            panel_title.append(tool_name, style="bold cyan")
            action_msg = "[bold white]Agent wants to execute:[/bold white]"
            hint_msg = (
                "[bright_green underline]Y[/bright_green underline][white]es[/white]  "
                "[bright_red underline]N[/bright_red underline][white]o[/white]  "
                "[bright_yellow underline]A[/bright_yellow underline][white]llow all[/white]  "
                "[bright_red underline]D[/bright_red underline][white]isable[/white]  "
                "[bright_red underline]F[/bright_red underline][white]orbid all[/white]  "
                "[bright_red underline]C[/bright_red underline][white]ancel[/white]"
            )
            full_body = action_msg + "\n\n" + params_text + "\n\n" + hint_msg
            _console.print()
            _console.print(
                _panel(
                    _text.from_markup(full_body),
                    title=panel_title,
                    border_style="cyan",
                    padding=(1, 2),
                )
            )
        else:
            print(f"\n--- {tool_name} ---")
            if isinstance(tool_input, dict) and tool_input:
                sorted_keys_p = sorted(
                    tool_input.keys(),
                    key=lambda k: (k in last_keys, len(str(tool_input[k]))),
                )
                visible_p = [
                    (k, tool_input[k]) for k in sorted_keys_p if not _is_hidden(k, tool_input[k])
                ]
                if len(visible_p) == 1:
                    _, value = visible_p[0]
                    print(f"  {_preview(value)}")
                elif visible_p:
                    for key, value in visible_p:
                        print(f"  {key:<18s} {_preview(value)}")
                else:
                    print("  (no parameters)")
            elif tool_input:
                print(f"  {_preview(tool_input)}")
            else:
                print("  (no parameters)")
            print("  Yes  No  Allow all  Disable  Forbid all  Cancel")

    def read_choice(self) -> str:
        return input("> ")

    def show_message(self, message: str, style: str) -> None:
        if _console is not None:
            _console.print(f"[{style}]{message}[/{style}]")
        else:
            print(message)

    def show_diff_preview(self, path: str, diff_lines: list[str]) -> None:
        if not diff_lines:
            return
        if _console is not None and _panel is not None:
            from rich.syntax import Syntax

            diff_text = "\n".join(diff_lines)
            syntax = Syntax(
                diff_text,
                "diff",
                theme="monokai",
                line_numbers=False,
                word_wrap=False,
            )
            _console.print(
                _panel(
                    syntax,
                    title=f"[bold]Proposed changes to[/bold] [cyan]{path}[/cyan]",
                    border_style="cyan",
                    padding=(0, 1),
                )
            )
        else:
            print(f"\nProposed changes to {path}:")
            for line in diff_lines:
                print(line)
            print()

    def pause_spinner(self) -> None:
        _spinner.pause()

    def resume_spinner(self) -> None:
        _spinner.resume()


__all__ = ["_TokenAccumulator", "_RichConfirmationUI"]
