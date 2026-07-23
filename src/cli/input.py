import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

try:
    import readline
except ImportError:
    readline = None  # type: ignore[assignment]

try:
    from rich.console import Console

    _console: Any = Console()
except ImportError:
    _console = None

HISTORY_MAX = 1000


def _history_dir() -> Path:
    return Path("data") / "history"


def _history_file() -> Path:
    return _history_dir() / ".input_history"


_history_disabled = False
_prefill: str = ""


def prefill_next_input(text: str) -> None:
    """Pre-fill the next ``input()`` call with *text* so the user can edit it.

    Uses ``readline.set_startup_hook`` to inject the text into the line
    buffer when the next prompt is displayed.  The hook fires once and
    clears itself so subsequent prompts are unaffected.
    """
    global _prefill
    if readline is None:
        return
    _rl = readline
    _prefill = text

    def _hook() -> None:
        global _prefill
        _rl.insert_text(_prefill)
        _prefill = ""
        _rl.set_startup_hook(None)

    _rl.set_startup_hook(_hook)


def load_input_history() -> None:
    """Load readline history from disk (if available)."""
    global _history_disabled
    if readline is None:
        return
    try:
        if _history_file().exists():
            readline.read_history_file(str(_history_file()))
        readline.set_history_length(HISTORY_MAX)
    except OSError as exc:
        _history_disabled = True
        msg = f"Could not load input history ({exc}). History will not be persisted."
        if _console is not None:
            _console.print(f"[dim yellow]{msg}[/dim yellow]")
        else:
            print(msg)


_AT_PATH_RE = re.compile(r"@([\w./\-]*)$")


_slash_commands: list[str] = []


def set_slash_commands(commands: list[str]) -> None:
    """Provide the list of slash command names (with leading /) for tab completion."""
    global _slash_commands
    _slash_commands = sorted(commands)


def _completer(text: str, state: int) -> str | None:
    """Tab-complete slash commands and @file references."""
    if readline is None:
        return None
    try:
        buf = readline.get_line_buffer()

        # Slash command completion: /com<Tab> → /compact
        if buf.startswith("/"):
            partial = buf.split()[0] if buf.strip() else buf
            matches = [c for c in _slash_commands if c.startswith(partial)]
            if state < len(matches):
                # Preserve any text after the command (e.g. "/compact aggressive")
                rest = buf[len(partial) :]
                return matches[state] + rest
            return None

        # @file path completion
        at_match = _AT_PATH_RE.search(buf)
        if not at_match:
            return None
        partial = at_match.group(1)
        base_dir = Path.cwd()
        if "/" in partial:
            parent = base_dir / Path(partial).parent
            prefix = Path(partial).name
        else:
            parent = base_dir
            prefix = partial
        try:
            matches = [
                str(p.relative_to(base_dir)) + ("/" if p.is_dir() else "")
                for p in sorted(parent.iterdir())
                if p.name.startswith(prefix)
            ]
        except OSError:
            return None
        if state < len(matches):
            return "@" + matches[state]
    except Exception:
        pass
    return None


def setup_readline_completion() -> None:
    """Register the slash-command and @-path completer with readline."""
    if readline is None:
        return
    readline.set_completer(_completer)
    readline.set_completer_delims(" \t\n")
    readline.parse_and_bind("tab: complete")


def save_input_history() -> None:
    """Persist readline history to disk."""
    global _history_disabled
    if readline is None or _history_disabled:
        return
    try:
        _history_dir().mkdir(parents=True, exist_ok=True)
        readline.write_history_file(str(_history_file()))
    except OSError as exc:
        _history_disabled = True
        msg = f"Could not save input history ({exc}). History will not be persisted."
        if _console is not None:
            _console.print(f"[dim yellow]{msg}[/dim yellow]")
        else:
            print(msg)


def read_multiline(first_line: str = "") -> str:
    """
    Read multi-line input until a closing ``\"\"\"`` delimiter.

    Used for pasting text that contains newline / carriage-return
    characters (log snippets, code blocks, data tables, web-page
    excerpts, etc.) which would otherwise be split across multiple
    prompts by :func:`input`.

    Termination:
        * A line whose stripped content is exactly ``\"\"\"``
        * ``Ctrl+D`` (EOF) — finishes input
        * ``Ctrl+C`` — cancels and returns empty string

    Args:
        first_line: Optional first line of content (when the opening
            ``\"\"\"`` was followed by text on the same line).

    Returns:
        Collected text joined by newlines, stripped.
        Empty string if the user cancelled with Ctrl+C.
    """
    lines: list[str] = []
    if first_line:
        lines.append(first_line)

    if _console:
        _console.print(
            "[dim]  Multi-line mode \u2014 paste text, then type "
            '[yellow bold]"""[/yellow bold] on a new line to send  (Ctrl+C to cancel)[/dim]'
        )
    else:
        print('  Multi-line mode \u2014 paste text, then type """ on a new line to send')

    while True:
        try:
            line = input("... ")
            if line.strip() == '"""':
                break
            lines.append(line)
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\n  (cancelled)")
            return ""

    return "\n".join(lines).strip()


def run_inline_shell(command: str) -> None:
    """Execute a shell command inline and print the output."""
    command = command.replace("\r", "")
    if not command.strip():
        if _console is not None:
            _console.print("[dim]Usage: !<command>  (e.g. !ls -la)[/dim]")
        else:
            print("Usage: !<command>  (e.g. !ls -la)")
        return

    _shell_meta = {"|", ">", "<", "&", ";", "`", "$", "(", ")", "*", "?", "{", "}"}
    needs_shell = any(ch in command for ch in _shell_meta)

    try:
        if needs_shell:
            proc = subprocess.Popen(  # nosec B602
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True,  # nosec B602
            )
        else:
            proc = subprocess.Popen(  # nosec B603
                shlex.split(command),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

        try:
            stdout, stderr = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise

        class _Result:
            def __init__(self, out: str, err: str, rc: int) -> None:
                self.stdout = out
                self.stderr = err
                self.returncode = rc

        result = _Result(stdout, stderr, proc.returncode)

        output = result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr

        _CAP = 512_000
        if len(output) > _CAP:
            half = _CAP // 2
            output = (
                output[:half]
                + f"\n\n[... {len(output) - _CAP:,} chars truncated ...]\n\n"
                + output[-half:]
            )

        if _console is not None:
            _console.rule("[dim]Shell[/dim]", style="dim green")
            if output.strip():
                _console.print(output.rstrip(), style="dim white", highlight=False)
            if result.returncode != 0:
                _console.print(f"[dim red]exit code: {result.returncode}[/dim red]")
            _console.rule(style="dim green")
        else:
            print("--- Shell ---")
            if output.strip():
                print(output.rstrip())
            if result.returncode != 0:
                print(f"exit code: {result.returncode}")
            print("-------------")

    except subprocess.TimeoutExpired:
        msg = "Command timed out after 30 seconds"
    except ValueError as e:
        msg = f"Invalid command syntax: {e}"
    except FileNotFoundError:
        cmd_name = command.split()[0] if command.split() else command
        msg = f"Command not found: {cmd_name}"
    except Exception as e:
        msg = f"Error: {e}"

    else:
        return

    # Error path — display the message
    if _console is not None:
        _console.rule("[dim]Shell[/dim]", style="dim green")
        _console.print(f"[red]{msg}[/red]")
        _console.rule(style="dim green")
    else:
        print(msg)
