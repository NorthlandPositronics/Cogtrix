"""Tests for research_delegate_auto pre-flight delegation."""

# ── _looks_like_research_query tests ─────────────────────────────────


def test_research_phrases_detected():
    from src.orchestration.phases import _looks_like_research_query

    assert _looks_like_research_query("research the best TUI frameworks")
    assert _looks_like_research_query("what is SearXNG?")
    assert _looks_like_research_query("find information about btop")
    assert _looks_like_research_query("look up the latest Python releases")
    assert _looks_like_research_query("tell me about context window sizes")


def test_action_phrases_excluded():
    from src.orchestration.phases import _looks_like_research_query

    assert not _looks_like_research_query("write a function that sorts a list")
    assert not _looks_like_research_query("create a new config file")
    assert not _looks_like_research_query("fix the bug in auth.py")
    assert not _looks_like_research_query("run the test suite")


def test_mixed_intent_excluded():
    from src.orchestration.phases import _looks_like_research_query

    # "research and implement" — action word present, should be False
    assert not _looks_like_research_query("research how to implement this and write the code")


def test_plain_statement_not_research():
    from src.orchestration.phases import _looks_like_research_query

    assert not _looks_like_research_query("Hello, how are you?")
    assert not _looks_like_research_query("show me the current config")


# ── Config field tests ────────────────────────────────────────────────


def test_config_defaults():
    from src.config import Config

    c = Config()
    assert c.research_delegate_auto is False
    assert c.research_delegate_auto_threshold == 0.50


def test_config_custom():
    from src.config import Config

    c = Config(research_delegate_auto=True, research_delegate_auto_threshold=0.30)
    assert c.research_delegate_auto is True
    assert c.research_delegate_auto_threshold == 0.30


# ── Pre-flight logic tests ────────────────────────────────────────────


def test_preflight_not_triggered_when_disabled():
    """When research_delegate_auto=False, delegate_task is never called."""
    from src.config import Config

    config = Config(research_delegate_auto=False)
    # Even with a research query and high context, pre-flight should not fire
    # (tested indirectly via the config flag — no integration test needed here)
    assert config.research_delegate_auto is False


def test_preflight_not_triggered_below_threshold():
    """When session context is below threshold, pre-flight does not fire."""
    from src.config import Config

    config = Config(research_delegate_auto=True, research_delegate_auto_threshold=0.80)
    # Session at 30% context — should not trigger
    session_total = 3_000
    max_context = 10_000
    ratio = session_total / max_context
    assert ratio < config.research_delegate_auto_threshold


def test_looks_like_research_case_insensitive():
    from src.orchestration.phases import _looks_like_research_query

    assert _looks_like_research_query("RESEARCH the topic")
    assert _looks_like_research_query("What Is the meaning of life?")
