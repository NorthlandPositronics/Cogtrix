"""Tests for the prompt_toolkit input session module."""


def test_update_toolbar_stats():
    import src.ui.input_session as m
    from src.ui.input_session import update_toolbar_stats

    update_toolbar_stats("session 4,812 tok   ████░░░░░░  2%")
    assert m._toolbar_stats != ""


def test_get_prompt_contains_stats_when_set():
    import re

    from src.ui.input_session import _get_prompt, update_toolbar_stats

    update_toolbar_stats("session 1,000 tok")
    result = _get_prompt()
    text = result.value if hasattr(result, "value") else str(result)
    plain = re.sub(r"\033\[[0-9;]*m", "", text)
    assert "session 1,000 tok" in plain


def test_get_prompt_contains_prompt_glyph():
    from src.ui.input_session import _get_prompt, update_toolbar_stats

    update_toolbar_stats("")
    result = _get_prompt()
    text = str(result)
    assert "\u276f" in text  # ❯


def test_prompt_glyph_is_last_line():
    """❯ must appear after the separator and stats (last line of prompt)."""
    from src.ui.input_session import _get_prompt, update_toolbar_stats

    update_toolbar_stats("session 999 tok   ████░░░░░░░  5%")
    result = _get_prompt()
    import re

    raw = result.value if hasattr(result, "value") else str(result)
    plain = re.sub(r"\033\[[0-9;]*m", "", raw)
    lines = plain.split("\n")
    assert "\u276f" in lines[-1], f"❯ not in last line: {lines}"
    assert lines[0].startswith("─"), f"First line should be separator, got: {lines[0]}"


def test_create_session_returns_session():
    from prompt_toolkit import PromptSession

    from src.ui.input_session import create_session

    s = create_session()
    assert isinstance(s, PromptSession)


def test_create_output_disables_cpr(monkeypatch):
    import src.ui.input_session as m

    class FakeOutput:
        def __init__(self) -> None:
            self.enable_cpr = True

    monkeypatch.setattr(m, "Vt100_Output", FakeOutput)
    monkeypatch.setattr(m, "create_output", lambda: FakeOutput())

    output = m._create_output()

    assert output.enable_cpr is False


def test_create_session_uses_output_helper(monkeypatch):
    from prompt_toolkit.output.base import DummyOutput

    import src.ui.input_session as m

    sentinel = DummyOutput()
    calls: list[object] = []

    monkeypatch.setattr(m, "_create_output", lambda: calls.append(sentinel) or sentinel)

    session = m.create_session()

    assert calls == [sentinel]
    assert session.output is sentinel


def test_create_session_with_history():
    from prompt_toolkit.history import InMemoryHistory

    from src.ui.input_session import create_session

    history = InMemoryHistory()
    s = create_session(history=history)
    assert s is not None


def test_get_prompt_contains_separator():
    from src.ui.input_session import _get_prompt, update_toolbar_stats

    update_toolbar_stats("")
    result = _get_prompt()
    text = str(result)
    assert "─" in text


def test_update_toolbar_module_state():
    import src.ui.input_session as m

    m.update_toolbar_stats("test value 42")
    assert m._toolbar_stats == "test value 42"
    m.update_toolbar_stats("")  # reset


def test_get_prompt_stats_right_aligned():
    """Stats line should be right-padded so it ends near terminal width."""
    import re
    import shutil

    from src.ui.input_session import _get_prompt, update_toolbar_stats

    # Use plain text (no ANSI) to test alignment logic cleanly
    update_toolbar_stats("session 4,812 tok   \u2588\u2588\u2588\u2588\u2591\u2591\u2591\u2591  2%")
    result = _get_prompt()
    text = result.value if hasattr(result, "value") else str(result)
    plain = re.sub(r"\033\[[0-9;]*m", "", text)
    lines = plain.split("\n")
    stats_line = lines[1]
    width = shutil.get_terminal_size((80, 24)).columns
    assert (
        len(stats_line) >= width - 5
    ), f"Stats line not right-aligned: len={len(stats_line)}, width={width}"


def test_get_prompt_glyph_is_teal():
    from src.ui.input_session import _get_prompt, update_toolbar_stats

    update_toolbar_stats("")
    result = _get_prompt()
    text = result.value if hasattr(result, "value") else str(result)
    assert "\033[38;5;37m" in text
    assert "\u276f" in text


def test_get_prompt_separator_is_colored():
    from src.ui.input_session import _get_prompt, update_toolbar_stats

    update_toolbar_stats("")
    result = _get_prompt()
    text = result.value if hasattr(result, "value") else str(result)
    # Teal ANSI code for separator (same as ❯)
    assert "\033[38;5;37m" in text


def test_file_history_used_in_prompt_session(tmp_path, monkeypatch):
    """PromptSession is created with FileHistory, not InMemoryHistory."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    import importlib

    import src.ui.input_session as m

    importlib.reload(m)
    from prompt_toolkit.history import FileHistory

    s = m.create_session()
    assert isinstance(s.history, FileHistory)


def test_file_history_path_is_persistent(tmp_path, monkeypatch):
    """History path resolves to a stable location under XDG data dir or home."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    import importlib

    import src.ui.input_session as m

    importlib.reload(m)
    assert "cogtrix" in m._HISTORY_PATH
    assert "history" in m._HISTORY_PATH


def test_fallback_to_in_memory_on_error(monkeypatch, tmp_path):
    """FileHistory failure falls back to InMemoryHistory without exception."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    import importlib

    import src.ui.input_session as m

    importlib.reload(m)
    from prompt_toolkit.history import FileHistory, InMemoryHistory

    _ = FileHistory.__init__

    def bad_init(self, filename):
        raise PermissionError("read-only")

    monkeypatch.setattr(FileHistory, "__init__", bad_init)
    s = m.create_session()
    assert isinstance(s.history, InMemoryHistory)


# ---------------------------------------------------------------------------
# Regression: SlashCompleter — tab completion for slash commands
# ---------------------------------------------------------------------------


class TestSlashCompleter:
    """Tab completion must work in prompt_toolkit (not just readline).

    Regression: readline completer was set up but prompt_toolkit bypasses
    readline entirely, so Tab did nothing — only right-arrow (auto-suggest
    from history) worked.
    """

    def test_slash_completer_returns_matches(self):
        from prompt_toolkit.document import Document

        from src.ui.input_session import SlashCompleter, set_slash_commands

        set_slash_commands(["/compact", "/clear", "/help", "/quit"])
        c = SlashCompleter()
        doc = Document("/com")
        completions = list(c.get_completions(doc, None))
        texts = [comp.text for comp in completions]
        assert "/compact" in texts

    def test_slash_completer_no_match_for_non_slash(self):
        from prompt_toolkit.document import Document

        from src.ui.input_session import SlashCompleter, set_slash_commands

        set_slash_commands(["/compact", "/help"])
        c = SlashCompleter()
        doc = Document("hello")
        completions = list(c.get_completions(doc, None))
        assert completions == []

    def test_slash_completer_multiple_matches(self):
        from prompt_toolkit.document import Document

        from src.ui.input_session import SlashCompleter, set_slash_commands

        set_slash_commands(["/compact", "/clear", "/c"])
        c = SlashCompleter()
        doc = Document("/c")
        completions = list(c.get_completions(doc, None))
        texts = [comp.text for comp in completions]
        assert "/compact" in texts
        assert "/clear" in texts
        assert "/c" in texts

    def test_create_session_uses_slash_completer_by_default(self):
        import src.ui.input_session as m

        s = m.create_session()
        assert isinstance(s.completer, m.SlashCompleter)
