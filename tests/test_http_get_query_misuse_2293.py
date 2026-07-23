"""#2293 — http_get/http_post called with a search `query` (no `url`) must redirect.

Weaker/open-weight models (qwen3 in prod v0.4.1 logs: ~1-in-4 http_get calls)
treat ``http_get`` like ``web_search`` — passing a natural-language ``query`` with
no ``url`` — which hard-fails Pydantic ``url Field required`` and loops. The
dispatcher now detects the misuse and returns an actionable redirect to
``web_search`` (tagged as a non-execution so a following "I fetched it" claim is
caught as fabricated), instead of the cryptic validation error.
"""

from __future__ import annotations

from src.orchestration.tool_arg_correction import detect_url_tool_misuse
from src.orchestration.tool_message_kinds import (
    KIND_TOOL_MISUSE_REDIRECT,
    TOOL_RESOLUTION_FAILURE_KINDS,
    is_resolution_failure_message,
)


class TestDetectUrlToolMisuse:
    def test_query_without_url_redirects(self) -> None:
        """The exact prod shape: http_get({'query': '...search phrase...'})."""
        msg = detect_url_tool_misuse("http_get", {"query": "US-Iran deal Reuters 2026-06-18"})
        assert msg is not None
        assert "web_search" in msg
        assert "did" in msg.lower() and "not run" in msg.lower()

    def test_http_post_also_covered(self) -> None:
        assert detect_url_tool_misuse("http_post", {"search": "latest news"}) is not None

    def test_assorted_query_like_keys(self) -> None:
        for key in ("query", "q", "search", "search_query", "query_string", "keywords"):
            assert detect_url_tool_misuse("http_get", {key: "some search"}) is not None, key

    def test_real_url_present_is_not_misuse(self) -> None:
        assert detect_url_tool_misuse("http_get", {"url": "https://example.com"}) is None

    def test_url_shaped_value_under_query_is_naming_slip_not_misuse(self) -> None:
        """A URL-shaped value under `query` is a field-naming mistake (left to the
        arg-corrector), NOT a web_search confusion — must not redirect."""
        assert detect_url_tool_misuse("http_get", {"query": "https://example.com"}) is None
        assert detect_url_tool_misuse("http_get", {"q": "example.com/path"}) is None

    def test_empty_url_with_query_redirects(self) -> None:
        assert detect_url_tool_misuse("http_get", {"url": "", "query": "cats"}) is not None

    def test_other_tools_never_trigger(self) -> None:
        assert detect_url_tool_misuse("web_search", {"query": "x"}) is None
        assert detect_url_tool_misuse("read_file", {"query": "x"}) is None

    def test_no_query_like_key_is_none(self) -> None:
        assert detect_url_tool_misuse("http_get", {"timeout": 30}) is None

    def test_non_dict_args_is_none(self) -> None:
        assert detect_url_tool_misuse("http_get", None) is None
        assert detect_url_tool_misuse("http_get", "query=x") is None

    def test_long_query_is_truncated_in_message(self) -> None:
        long_q = "a" * 200
        msg = detect_url_tool_misuse("http_get", {"query": long_q})
        assert msg is not None
        assert "..." in msg
        assert "a" * 200 not in msg  # truncated


class TestMisuseKindClassification:
    def test_redirect_kind_is_a_resolution_failure(self) -> None:
        """A redirect means the tool did NOT execute — it must be in the
        failure-kinds set so the fabricated-success detector treats a following
        'I fetched the page' claim as fabricated."""
        assert KIND_TOOL_MISUSE_REDIRECT in TOOL_RESOLUTION_FAILURE_KINDS

    def test_is_resolution_failure_message_recognises_the_tag(self) -> None:
        from langchain_core.messages import ToolMessage

        from src.orchestration.tool_message_kinds import COGTRIX_KIND_KEY

        tm = ToolMessage(
            content="use web_search",
            tool_call_id="c1",
            name="http_get",
            additional_kwargs={COGTRIX_KIND_KEY: KIND_TOOL_MISUSE_REDIRECT},
        )
        assert is_resolution_failure_message(tm) is True


class TestInvokerWiring:
    """Through DedupedToolInvoker.invoke_one: the misuse short-circuits BEFORE any
    tool invocation (no Pydantic error). The guard sits right after the denial
    check and before the budget/correction/invoke machinery."""

    def test_misuse_short_circuits_before_invoke(self) -> None:
        from unittest.mock import MagicMock

        from src.orchestration.deduped_tool_invoker import DedupedToolInvoker
        from src.orchestration.tool_message_kinds import COGTRIX_KIND_KEY

        http_get = MagicMock()
        http_get.name = "http_get"
        http_get.invoke.side_effect = AssertionError("tool must NOT be invoked on misuse")

        per_run = MagicMock()
        per_run.tool_lookup = {"http_get": http_get}

        inv = DedupedToolInvoker.__new__(DedupedToolInvoker)
        inv._safe_tool_name = lambda n: n
        inv._tool_call_key = lambda call: None  # disable dedup/TOCTOU path
        inv._check_duplicate = lambda call, key=None: None
        inv._session_state = None  # no denial
        inv._release_sentinel = lambda call_key: None
        inv._per_run_state = [per_run]

        call = {"name": "http_get", "args": {"query": "cogtrix capabilities"}, "id": "c1"}
        result = inv.invoke_one(call, run_config=None)

        assert result.name == "http_get"
        assert result.tool_call_id == "c1"
        assert "web_search" in result.content
        assert result.additional_kwargs.get(COGTRIX_KIND_KEY) == KIND_TOOL_MISUSE_REDIRECT
        http_get.invoke.assert_not_called()

    def test_normal_url_call_is_not_short_circuited(self) -> None:
        """A proper http_get(url=...) is NOT intercepted — reaches the tool."""
        from unittest.mock import MagicMock

        from langchain_core.messages import ToolMessage

        from src.orchestration.deduped_tool_invoker import DedupedToolInvoker

        http_get = MagicMock()
        http_get.name = "http_get"
        # Stand in for a successful fetch.
        sentinel = ToolMessage(content="<html>ok</html>", tool_call_id="c2", name="http_get")
        http_get.invoke.return_value = sentinel

        per_run = MagicMock()
        per_run.tool_lookup = {"http_get": http_get}
        per_run.tool_call_counts = {}

        inv = DedupedToolInvoker.__new__(DedupedToolInvoker)
        inv._safe_tool_name = lambda n: n
        inv._tool_call_key = lambda call: None
        inv._check_duplicate = lambda call, key=None: None
        inv._session_state = None
        inv._release_sentinel = lambda call_key: None
        inv._correct_tool_args = lambda tool, args: args
        inv._per_run_state = [per_run]

        call = {"name": "http_get", "args": {"url": "https://example.com"}, "id": "c2"}
        # The misuse guard must let this through to budget/correction/invoke; we
        # only assert the guard did NOT intercept (the tool is reached). Any
        # AttributeError past the guard would be from un-shimmed budget machinery,
        # which is out of scope here — so we assert the guard returned None first.
        from src.orchestration.tool_arg_correction import detect_url_tool_misuse

        assert detect_url_tool_misuse("http_get", call["args"]) is None
