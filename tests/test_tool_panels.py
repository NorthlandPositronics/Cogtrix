"""Tests for tool output panel rendering."""

from io import StringIO

from rich.console import Console


def _console() -> tuple[Console, StringIO]:
    buf = StringIO()
    return Console(file=buf, highlight=False, markup=False, width=100), buf


def test_render_tool_panel_contains_tool_name():
    from cogtrix_core.ui.tool_panels import render_tool_panel

    console, buf = _console()
    render_tool_panel(console, "search_web", {"query": "hello"}, "result text", 1.2)
    out = buf.getvalue()
    assert "search_web" in out


def test_render_tool_panel_contains_result():
    from cogtrix_core.ui.tool_panels import render_tool_panel

    console, buf = _console()
    render_tool_panel(console, "read_file", {"path": "/tmp/x"}, "file content here", 0.1)
    out = buf.getvalue()
    assert "file content here" in out


def test_render_tool_panel_shows_elapsed():
    from cogtrix_core.ui.tool_panels import render_tool_panel

    console, buf = _console()
    render_tool_panel(console, "run_shell", {}, "output", 3.7)
    out = buf.getvalue()
    assert "3.7s" in out


def test_render_tool_panel_collapses_long_output():
    from cogtrix_core.ui.tool_panels import render_tool_panel

    console, buf = _console()
    long_result = "x" * 5000
    render_tool_panel(
        console, "http_get", {"url": "http://x.com"}, long_result, 2.0, collapse_threshold=2000
    )
    out = buf.getvalue()
    assert "truncated" in out or "chars" in out


def test_render_tool_panel_short_output_not_collapsed():
    from cogtrix_core.ui.tool_panels import render_tool_panel

    console, buf = _console()
    render_tool_panel(console, "calculator", {}, "42", 0.0, collapse_threshold=2000)
    out = buf.getvalue()
    assert "42" in out
    assert "truncated" not in out


def test_render_tool_panel_no_args():
    from cogtrix_core.ui.tool_panels import render_tool_panel

    console, buf = _console()
    render_tool_panel(console, "get_time", {}, "12:00", 0.0)
    out = buf.getvalue()
    assert "get_time" in out


def test_render_diff_panel_contains_path():
    from cogtrix_core.ui.tool_panels import render_diff_panel

    console, buf = _console()
    diff = "+++ new\n--- old\n+added line\n-removed line"
    render_diff_panel(console, "write_file", "cogtrix_core/foo.py", diff, 0.1)
    out = buf.getvalue()
    assert "cogtrix_core/foo.py" in out


def test_render_diff_panel_contains_tool_name():
    from cogtrix_core.ui.tool_panels import render_diff_panel

    console, buf = _console()
    render_diff_panel(console, "patch_file", "x.py", "+line", 0.2)
    out = buf.getvalue()
    assert "patch_file" in out


def test_tool_panels_importable_from_src_ui():
    from cogtrix_core.ui import render_diff_panel, render_tool_panel

    assert callable(render_tool_panel)
    assert callable(render_diff_panel)
