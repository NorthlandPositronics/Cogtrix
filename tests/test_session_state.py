"""Tests for SessionState dataclass — reset methods and initial state."""

from src.orchestration.session_state import SessionState


def test_initial_state_empty() -> None:
    ss = SessionState()
    assert ss.denials == set()
    assert ss.deny_all is False
    assert ss.no_confirm is False
    assert ss.approvals == set()
    assert ss.loaded_tools == set()
    assert ss.pinned_tools == set()
    assert ss.all_tool_descriptions == {}
    assert ss.all_tool_originals == {}


def test_reset_for_new_session_clears_session_scoped() -> None:
    ss = SessionState(
        denials={"shell"},
        deny_all=True,
        no_confirm=True,
        approvals={"write_file"},
        loaded_tools={"web_search"},
        pinned_tools={"web_search"},
        all_tool_descriptions={"shell": "run commands"},
        all_tool_originals={"shell": object()},
    )
    ss.reset_for_new_session()

    assert ss.denials == set()
    assert ss.deny_all is False
    assert ss.loaded_tools == set()
    assert ss.pinned_tools == set()
    assert ss.approvals == set()

    assert ss.no_confirm is True
    assert ss.all_tool_descriptions == {"shell": "run commands"}
    assert "shell" in ss.all_tool_originals


def test_reset_for_new_prompt_only_clears_deny_all() -> None:
    ss = SessionState(
        denials={"shell"},
        deny_all=True,
        no_confirm=True,
        approvals={"write_file"},
        loaded_tools={"web_search", "read_file"},
        pinned_tools={"web_search"},
    )
    ss.reset_for_new_prompt()

    assert ss.deny_all is False

    assert ss.denials == {"shell"}
    assert ss.no_confirm is True
    assert ss.approvals == {"write_file"}
    # Agent-loaded (non-pinned) tools are cleared; pinned tools remain
    assert ss.loaded_tools == {"web_search"}
    assert ss.pinned_tools == {"web_search"}


def test_reset_for_new_session_preserves_catalogs() -> None:
    sentinel = object()
    ss = SessionState(
        all_tool_descriptions={"foo": "bar"},
        all_tool_originals={"foo": sentinel},
    )
    ss.reset_for_new_session()

    assert ss.all_tool_descriptions == {"foo": "bar"}
    assert ss.all_tool_originals["foo"] is sentinel


def test_session_state_fields_are_independent_instances() -> None:
    ss1 = SessionState()
    ss2 = SessionState()
    ss1.denials.add("tool_a")
    assert "tool_a" not in ss2.denials
