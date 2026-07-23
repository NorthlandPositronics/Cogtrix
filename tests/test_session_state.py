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


# ── Helper method tests ───────────────────────────────────────────────────────


def test_is_denied_false_when_empty() -> None:
    ss = SessionState()
    assert ss.is_denied("write_file") is False


def test_is_denied_true_when_in_denials() -> None:
    ss = SessionState()
    ss.deny_tool("write_file")
    assert ss.is_denied("write_file") is True


def test_is_denied_true_when_deny_all() -> None:
    ss = SessionState()
    ss.set_deny_all()
    assert ss.is_denied("any_tool") is True
    assert ss.is_denied("another_tool") is True


def test_deny_tool_adds_to_denials() -> None:
    ss = SessionState()
    ss.deny_tool("shell")
    assert "shell" in ss.denials
    assert ss.is_denied("shell") is True
    assert ss.is_denied("other") is False  # not a blanket denial


def test_allow_tool_removes_from_denials() -> None:
    ss = SessionState()
    ss.deny_tool("shell")
    ss.allow_tool("shell")
    assert "shell" not in ss.denials
    assert ss.is_denied("shell") is False


def test_allow_tool_is_idempotent_for_absent_tool() -> None:
    ss = SessionState()
    ss.allow_tool("shell")  # never denied — must not raise
    assert ss.is_denied("shell") is False


def test_set_deny_all_sets_flag() -> None:
    ss = SessionState()
    assert ss.deny_all is False
    ss.set_deny_all()
    assert ss.deny_all is True


def test_get_denials_snapshot_is_frozenset() -> None:
    ss = SessionState()
    ss.deny_tool("a")
    ss.deny_tool("b")
    snap = ss.get_denials_snapshot()
    assert isinstance(snap, frozenset)
    assert snap == frozenset({"a", "b"})


def test_get_denials_snapshot_is_immutable_copy() -> None:
    ss = SessionState()
    ss.deny_tool("a")
    snap = ss.get_denials_snapshot()
    ss.deny_tool("b")  # modify after snapshot
    assert "b" not in snap  # snapshot is not affected


def test_each_sessionstate_has_independent_lock() -> None:
    ss1 = SessionState()
    ss2 = SessionState()
    assert ss1._lock is not ss2._lock


def test_concurrent_deny_and_is_denied() -> None:
    """Basic race-free test: parallel deny_tool and is_denied calls must not raise."""
    import threading

    ss = SessionState()
    errors: list[Exception] = []

    def _writer() -> None:
        for i in range(200):
            ss.deny_tool(f"tool_{i % 10}")

    def _reader() -> None:
        for i in range(200):
            try:
                ss.is_denied(f"tool_{i % 10}")
            except Exception as e:
                errors.append(e)

    threads = [threading.Thread(target=_writer), threading.Thread(target=_reader)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"Concurrent access raised: {errors}"
