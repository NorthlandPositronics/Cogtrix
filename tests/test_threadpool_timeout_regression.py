"""Regression tests for issue #1158: ThreadPoolExecutor __exit__ blocks on hung threads.

Verifies that all 5 affected locations use explicit ThreadPoolExecutor with
result(timeout=N) and shutdown(wait=False), so hung threads do not block the caller.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── cogtrix.py prompt-prep paths ─────────────────────────────────────────────


class TestCogtrixPromptPrepTimeout:
    """The two prompt-prep paths in cogtrix.py must not block on hung threads."""

    def test_explicit_pool_not_context_manager(self):
        """Verify cogtrix.py uses a safe timeout pattern, not the
        ``with _cf.ThreadPoolExecutor(...) as pool:`` footgun.

        Two safe shapes are accepted:

        1. **Legacy explicit-pool pattern** — ``_pool = _cf.ThreadPoolExecutor``
           + ``_pool.shutdown(wait=False)`` + both ``_ctx_future`` /
           ``_opt_future`` named-future timeouts on the prompt-prep paths.
           This is the original fix for #1158.
        2. **Migrated to centralized helper** — ``from cogtrix_core.concurrency
           import invoke_with_timeout`` (#1677 / PR #1904). The helper
           encapsulates the safe shutdown pattern in one place.

        The buggy ``with _cf.ThreadPoolExecutor(...) as pool:`` shape MUST
        never appear regardless of which pattern is in use.
        """
        import cogtrix

        source = Path(cogtrix.__file__).read_text()
        # The actual footgun pattern must NEVER appear in this file.
        assert "with _cf.ThreadPoolExecutor" not in source
        # One of the two safe shapes must be present.
        uses_legacy_explicit_pool = (
            "_pool = _cf.ThreadPoolExecutor" in source
            and "_pool.shutdown(wait=False)" in source
            and "_ctx_future.result(timeout=60)" in source
            and "_opt_future.result(timeout=60)" in source
        )
        uses_centralized_helper = (
            "from cogtrix_core.concurrency import invoke_with_timeout" in source
        )
        assert uses_legacy_explicit_pool or uses_centralized_helper, (
            "cogtrix.py must use either the legacy explicit-pool + "
            "shutdown(wait=False) pattern with named-future timeouts on "
            "both prompt-prep paths OR the centralized "
            "cogtrix_core.concurrency.invoke_with_timeout helper (#1677)"
        )


# ── service.py _init_channels ────────────────────────────────────────────────


class TestServiceInitChannelsTimeout:
    """_init_channels must skip a hung channel init and not block."""

    @pytest.mark.timeout(10)
    def test_hung_channel_init_skipped(self):
        """If one channel init hangs, others still complete."""
        from cogtrix_core.assistant import service as service_mod
        from cogtrix_core.assistant.service import AssistantService

        svc = AssistantService.__new__(AssistantService)
        stop_event = threading.Event()

        # Patch the production timeout down so we don't wait the real 30 s.
        # The hung worker's inner wait must outlast the patched timeout but
        # still be short enough that the test exits quickly if the production
        # cancel path regresses.
        with patch.object(service_mod, "_CHANNEL_INIT_TIMEOUT", 0.3):
            with patch.object(
                svc, "_init_whatsapp", side_effect=lambda *a, **k: stop_event.wait(timeout=3)
            ):
                with patch.object(svc, "_init_telegram", return_value=MagicMock(name="telegram")):
                    cfg2 = MagicMock()
                    cfg2.services = {"whatsapp": {}, "telegram": {}}
                    cfg2.get.return_value = {
                        "channels": {
                            "whatsapp": {"enabled": True},
                            "telegram": {"enabled": True},
                        }
                    }
                    cfg2.data_dir = "/tmp"
                    t0 = time.monotonic()
                    channels = svc._discover_channels(cfg2)
                    elapsed = time.monotonic() - t0
                    # Must return within ~1s (0.3s timeout + margin), not hang forever
                    assert elapsed < 1.5, f"Blocked for {elapsed:.1f}s — pool __exit__ not fixed"
                    # Telegram should still be initialized
                    assert len(channels) == 1
                    assert channels[0] is not None

        stop_event.set()


# ── ingest.py _ingest_files_parallel ─────────────────────────────────────────


class TestIngestFilesParallelTimeout:
    """_ingest_files_parallel must mark timed-out files as failed."""

    @pytest.mark.timeout(10)
    def test_hung_ingest_file_marked_failed(self, tmp_path: Path):
        """If _prepare_ingest_file hangs, the file is marked failed and loop continues."""
        from cogtrix_core.rag.ingest import ingest_many

        stop_event = threading.Event()

        # Create a dummy config
        config = MagicMock()
        config.vectordb_dir = tmp_path / "vectordb"
        config.chunk_size = 500
        config.chunk_overlap = 50
        config.embedding_model = "test"

        # Create a dummy file path
        dummy_file = tmp_path / "test.txt"
        dummy_file.write_text("test content")

        # Patch the production timeout down so we don't wait the real 60 s.
        with patch("cogtrix_core.rag.ingest._INGEST_PREPARE_TIMEOUT", 0.3):
            with patch(
                "cogtrix_core.rag.ingest._prepare_ingest_file",
                side_effect=lambda *a, **k: stop_event.wait(timeout=3),
            ):
                t0 = time.monotonic()
                results = ingest_many([str(dummy_file)], config)
                elapsed = time.monotonic() - t0
                # Must return within ~1s (0.3s timeout + margin), not hang forever
                assert elapsed < 1.5, f"Blocked for {elapsed:.1f}s — pool __exit__ not fixed"
                assert results[str(dummy_file)] is False

        stop_event.set()


# ── whatsapp.py _resolve_uncached ────────────────────────────────────────────


class TestWhatsAppPrefetchLidsTimeout:
    """_prefetch_lids must skip a hung LID resolution."""

    @pytest.mark.timeout(10)
    def test_hung_lid_resolution_skipped(self):
        """If one _resolve_lid hangs, others still complete."""
        from cogtrix_core.assistant.channels import whatsapp as whatsapp_mod
        from cogtrix_core.assistant.channels.whatsapp import WhatsAppChannel

        ch = WhatsAppChannel.__new__(WhatsAppChannel)
        ch._snapshot = {}
        ch._overview_limit = 100
        ch._lid_cache = {}
        ch._lid_cache_lock = threading.Lock()
        ch._client = MagicMock()
        stop_event = threading.Event()

        call_count = 0

        def _slow_lid(number: str) -> None:
            nonlocal call_count
            call_count += 1
            if number == "slow@lid":
                stop_event.wait(timeout=3)
            ch._lid_cache[number] = ("resolved", 9999999999)

        ch._resolve_lid = _slow_lid

        msgs = [
            MagicMock(from_number="slow@lid"),
            MagicMock(from_number="fast@lid"),
        ]

        # Patch the production timeout down so we don't wait the real 10 s.
        with patch.object(whatsapp_mod, "_LID_RESOLVE_TIMEOUT", 0.3):
            t0 = time.monotonic()
            ch._prefetch_lids(msgs)
            elapsed = time.monotonic() - t0
            # Must return within ~1s (0.3s timeout + margin), not hang forever
            assert elapsed < 1.5, f"Blocked for {elapsed:.1f}s — pool __exit__ not fixed"
            # Fast number should be resolved
            assert "fast@lid" in ch._lid_cache

        stop_event.set()


# ── optimizer.py optimize_prompt ──────────────────────────────────────────────


class TestOptimizePromptTimeout:
    """optimize_prompt() must not block on a hung LLM call — fail-open returning original."""

    @pytest.mark.timeout(10)
    def test_hung_llm_returns_original_prompt(self):
        """If llm.invoke() times out, optimize_prompt returns the original prompt unchanged."""
        import time
        from concurrent.futures import TimeoutError as FuturesTimeoutError

        from cogtrix_core.prompt.optimizer import PromptPlan, optimize_prompt

        # Simulate a timeout by raising FuturesTimeoutError directly —
        # stop_event.wait() can't be interrupted by future.cancel(), so we
        # raise the expected exception instead to test the handler path.
        hung_llm = MagicMock()
        hung_llm.invoke.side_effect = FuturesTimeoutError("hung")

        original_input = "analyze the codebase and tell me about the architecture"

        # Patch the production timeout down so we don't wait the real 60 s.
        with patch("cogtrix_core.prompt.optimizer._PROMPT_OPTIMIZER_TIMEOUT_SECONDS", 0.3):
            t0 = time.monotonic()
            result = optimize_prompt(original_input, hung_llm, force=True)
            elapsed = time.monotonic() - t0
            # Must return within ~1s (0.3s timeout + margin), not hang forever
            assert (
                elapsed < 1.5
            ), f"Blocked for {elapsed:.1f}s — optimize_prompt timeout not applied"
            # Must return the original prompt unchanged (fail-open)
            assert isinstance(result, PromptPlan)
            assert result.text == original_input
            assert result.milestones == []

    @pytest.mark.timeout(10)
    def test_successful_llm_returns_optimized_prompt(self):
        """If llm.invoke() completes within timeout, the optimized prompt is returned."""
        from cogtrix_core.prompt.optimizer import PromptPlan, optimize_prompt

        # Use force=True to bypass the action-verb skip and the length gate,
        # ensuring the LLM call is always made in this test.
        original = "analyze the codebase"
        assert len(original) < 400  # would be skipped without force=True

        # Response must be >= 10 chars (the < 10 char guard in optimize_prompt)
        long_response = (
            "Please carefully analyze the entire codebase structure and architecture, "
            "identifying all key modules, their specific responsibilities, and how they "
            "interact with each other to form the complete system. This is a complex task "
            "that requires thorough examination of multiple files and their relationships, "
            "including orchestration, assistant, memory, and API layers for a comprehensive overview."
        )
        assert len(long_response) >= 10

        def quick_llm_invoke(prompt: str) -> object:
            return type("Response", (), {"content": long_response})()

        quick_llm = MagicMock()
        quick_llm.invoke.side_effect = quick_llm_invoke

        with patch("cogtrix_core.prompt.optimizer._PROMPT_OPTIMIZER_TIMEOUT_SECONDS", 5):
            result = optimize_prompt(original, quick_llm, force=True)
        assert isinstance(result, PromptPlan)
        assert result.text != original
        assert len(result.text) >= 10

    def test_non_callable_llm_raises_typeerror(self):
        """optimize_prompt rejects a non-callable .invoke attribute with TypeError."""
        from cogtrix_core.prompt.optimizer import PromptPlan, optimize_prompt

        original = "analyze the codebase and document the architecture"
        bad_llm = type("BadLLM", (), {"invoke": None})()

        result = optimize_prompt(original, bad_llm, force=True)
        # TypeError is caught by the outer exception handler → fail-open
        assert isinstance(result, PromptPlan)
        assert result.text == original

    def test_missing_invoke_raises_typeerror(self):
        """optimize_prompt rejects an llm without .invoke with TypeError."""
        from cogtrix_core.prompt.optimizer import PromptPlan, optimize_prompt

        original = "analyze the codebase and document the architecture"
        bad_llm = type("BadLLM", (), {})()

        result = optimize_prompt(original, bad_llm, force=True)
        assert isinstance(result, PromptPlan)
        assert result.text == original

    def test_response_content_none_falls_back_to_str_response(self):
        """If response.content is None, the response is stringified and used."""
        from cogtrix_core.prompt.optimizer import PromptPlan, optimize_prompt

        original = "analyze the codebase"
        assert len(original) < 400

        class NoneContentResponse:
            content = None

            def __str__(self) -> str:
                return "NoneContentResponse"

        def quick_llm_invoke(prompt: str) -> object:
            return NoneContentResponse()

        llm = MagicMock()
        llm.invoke.side_effect = quick_llm_invoke

        with patch("cogtrix_core.prompt.optimizer._PROMPT_OPTIMIZER_TIMEOUT_SECONDS", 5):
            result = optimize_prompt(original, llm, force=True)
        assert isinstance(result, PromptPlan)
        # content=None → falls back to str(response) = "NoneContentResponse" (≥10 chars)
        # The optimizer "restructures" it (returns as-is since it's not a user prompt)
        assert len(result.text) >= 10

    def test_response_content_list_non_dict_items_joined(self):
        """If response.content is a list of non-dict items, they are joined as strings."""
        from cogtrix_core.prompt.optimizer import PromptPlan, optimize_prompt

        original = "analyze the codebase"
        assert len(original) < 400

        class ListContentResponse:
            content = ["First part ", "second part", " third part."]

        def quick_llm_invoke(prompt: str) -> object:
            return ListContentResponse()

        llm = MagicMock()
        llm.invoke.side_effect = quick_llm_invoke

        with patch("cogtrix_core.prompt.optimizer._PROMPT_OPTIMIZER_TIMEOUT_SECONDS", 5):
            result = optimize_prompt(original, llm, force=True)
        assert isinstance(result, PromptPlan)
        assert "First part" in result.text
        assert "second part" in result.text
        assert "third part" in result.text


# ── setup_wizard.py ───────────────────────────────────────────────────────────


class TestSetupWizardTimeout:
    """setup_wizard must not block on hung LLM calls — fail-open with clear exit."""

    def test_explicit_pool_not_context_manager(self):
        """Verify setup_wizard.py uses a safe timeout pattern, not the
        ``with ThreadPoolExecutor as pool:`` footgun.

        Two safe shapes are accepted:

        1. **Legacy explicit-pool pattern** — ``pool =
           concurrent.futures.ThreadPoolExecutor(...)`` followed by
           ``pool.shutdown(wait=False)``.  This was the original fix for
           #1158.
        2. **Migrated to centralized helper** — ``from cogtrix_core.concurrency
           import invoke_with_timeout`` (#1677 / PR #1904).  The helper
           encapsulates the safe shutdown pattern in one place.

        The buggy ``with concurrent.futures.ThreadPoolExecutor(...) as
        pool:`` shape MUST never appear regardless of which pattern is
        in use.
        """
        import cogtrix_core.setup_wizard as sw

        source = Path(sw.__file__).read_text()
        # The actual footgun: ``with ThreadPoolExecutor(...) as pool:``
        # calls ``shutdown(wait=True)`` on exit and blocks on hung threads.
        assert "with concurrent.futures.ThreadPoolExecutor" not in source
        # One of the two safe shapes must be present.
        uses_legacy_explicit_pool = (
            "pool = concurrent.futures.ThreadPoolExecutor" in source
            and "pool.shutdown(wait=False)" in source
        )
        uses_centralized_helper = (
            "from cogtrix_core.concurrency import invoke_with_timeout" in source
        )
        assert uses_legacy_explicit_pool or uses_centralized_helper, (
            "setup_wizard.py must use either the legacy explicit-pool + "
            "shutdown(wait=False) pattern OR the centralized "
            "cogtrix_core.concurrency.invoke_with_timeout helper (#1677)"
        )

    @pytest.mark.timeout(10)
    def test_test_connection_timeout_returns_none(self):
        """If LLM invoke times out during connection test, _test_connection returns None."""
        import time
        from concurrent.futures import TimeoutError as FuturesTimeoutError

        from cogtrix_core.setup_wizard import _test_connection

        llm_mock = MagicMock()
        llm_mock.invoke.side_effect = FuturesTimeoutError("hung")

        with patch("cogtrix_core.providers.create_chat_model", return_value=llm_mock):
            with patch("cogtrix_core.setup_wizard._SETUP_WIZARD_LLM_TIMEOUT_SECONDS", 0.3):
                t0 = time.monotonic()
                result = _test_connection("openai", "gpt-4", "sk-test", None)
                elapsed = time.monotonic() - t0

        assert result is None
        assert elapsed < 1.5, f"Blocked for {elapsed:.1f}s — timeout not applied"

    @pytest.mark.timeout(10)
    def test_run_conversation_timeout_exits(self):
        """If LLM invoke times out during conversation, _run_conversation raises SystemExit."""
        import time
        from concurrent.futures import TimeoutError as FuturesTimeoutError

        from cogtrix_core.setup_wizard import _run_conversation

        llm = MagicMock()
        llm.invoke.side_effect = FuturesTimeoutError("hung")

        with patch("builtins.input", return_value="yes"):
            with patch("cogtrix_core.setup_wizard._SETUP_WIZARD_LLM_TIMEOUT_SECONDS", 0.3):
                t0 = time.monotonic()
                with pytest.raises(SystemExit):
                    _run_conversation(llm, "system prompt")
                elapsed = time.monotonic() - t0

        assert elapsed < 1.5, f"Blocked for {elapsed:.1f}s — timeout not applied"


# ── tools/delegate.py _execute_single_task fallback path ──────────────────────


class TestDelegateFallbackTimeout:
    """The fallback llm.invoke() in _execute_single_task must not block on hung threads."""

    def test_explicit_pool_not_context_manager(self):
        """Verify delegate.py's _execute_single_task fallback block uses a
        safe timeout pattern, not the ``with ThreadPoolExecutor(...) as
        pool:`` footgun.

        Two safe shapes are accepted inside the fallback block:

        1. **Legacy explicit-pool pattern** — ``executor =
           ThreadPoolExecutor(max_workers=1)`` + ``future =
           executor.submit(llm.invoke, ...)``. This is the original fix
           for #1158.
        2. **Migrated to centralized helper** — ``invoke_with_timeout(...)``
           call in the block, with ``from cogtrix_core.concurrency import
           invoke_with_timeout`` imported at module top (#1677 / PR #1904).

        The buggy ``with ThreadPoolExecutor(...) as <var>:`` shape MUST
        never appear in the fallback block regardless of which pattern
        is in use — that's the actual footgun this test guards.
        """
        import re

        import cogtrix_core.tools.delegate as delegate_mod

        source = Path(delegate_mod.__file__).read_text()
        # Locate the specific fallback block (in _execute_single_task,
        # not the public wrapper).  The marker persists across migrations.
        fallback_marker = "# ── Fallback: plain LLM call"
        assert fallback_marker in source
        marker_pos = source.find(fallback_marker)
        fallback_block = source[marker_pos : marker_pos + 5000]

        # The actual footgun pattern MUST NEVER appear in the fallback
        # block.  The phrase may appear in explanatory comments (backtick-
        # quoted), so skip comment lines before checking for actual code
        # usage.
        buggy_pattern = re.compile(r"with\s+ThreadPoolExecutor\([^)]+\)\s+as\s+\w+:")
        for line in fallback_block.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if buggy_pattern.search(line):
                pytest.fail(
                    "Buggy `with ThreadPoolExecutor(...) as pool:` found in "
                    f"fallback block: {line!r}"
                )

        # The fallback block must use one of the two safe shapes.
        uses_legacy_fallback = (
            "executor = ThreadPoolExecutor(max_workers=1)" in fallback_block
            and "future = executor.submit(llm.invoke" in fallback_block
            and "executor.shutdown(wait=False)" in source
        )
        uses_centralized_fallback = (
            "invoke_with_timeout(" in fallback_block
            and "from cogtrix_core.concurrency import invoke_with_timeout" in source
        )
        assert uses_legacy_fallback or uses_centralized_fallback, (
            "delegate.py fallback block must use either the legacy "
            "explicit-pool + shutdown(wait=False) pattern OR the centralized "
            "cogtrix_core.concurrency.invoke_with_timeout helper (#1677)"
        )

    @pytest.mark.timeout(10)
    def test_hung_llm_fallback_returns_failure_not_hang(self):
        """If the fallback llm.invoke() times out, DelegateResult(success=False) is returned."""
        import time as time_mod
        from concurrent.futures import TimeoutError as FuturesTimeoutError

        from cogtrix_core.tools import delegate as delegate_mod
        from cogtrix_core.tools.delegate import (
            _execute_single_task,
            configure_delegate,
        )

        # Ensure circuit breaker does not block us
        configure_delegate({"circuit_breaker_cooldown": 0})

        # Use a very short patched timeout so test exits fast
        with patch.dict(delegate_mod._delegate_config, {"default_timeout": 0.01}):
            hung_llm = MagicMock()
            hung_llm.invoke.side_effect = FuturesTimeoutError("hung")
            hung_llm.model = "test/model"

            with patch.object(delegate_mod, "create_delegate_llm", return_value=hung_llm):
                with patch.object(delegate_mod, "get_delegate_tools", return_value=[]):
                    t0 = time_mod.monotonic()
                    result = _execute_single_task(
                        task="test task",
                        context="test context",
                        response_format="text",
                        json_schema=None,
                        provider="test",
                        model="test",
                        temperature=0.5,
                        num_ctx=None,
                        use_tools=False,
                    )
                    elapsed = time_mod.monotonic() - t0

                    # Must return within ~1s (0.01s timeout + margin), not hang forever
                    assert (
                        elapsed < 1.5
                    ), f"Blocked for {elapsed:.1f}s — delegate fallback timeout not applied"
                    # Must return a failure result (fail-open), not raise
                    from cogtrix_core.tools.delegate import DelegateResult

                    assert isinstance(result, DelegateResult)
                    assert result.success is False
                    assert result.response == ""
                    assert result.error is not None
                    # Three error shapes are accepted:
                    #   1. "hung" — worker raised FuturesTimeoutError("hung") before the
                    #      wall-clock timer fired (legacy explicit-pool path).
                    #   2. "timed out" — generic timeout phrasing if the executor's own
                    #      TimeoutError surfaces with no message (legacy edge case).
                    #   3. "did not return within Ns" — normalized message produced by
                    #      src.concurrency.invoke_with_timeout after #1903 migration.
                    err_lower = result.error.lower()
                    assert (
                        "hung" in err_lower
                        or "timed out" in err_lower
                        or "did not return" in err_lower
                    ), f"Unexpected delegate timeout error: {result.error!r}"

    @pytest.mark.timeout(10)
    def test_successful_llm_fallback_returns_response(self):
        """If the fallback llm.invoke() completes, DelegateResult(success=True) is returned."""
        from cogtrix_core.tools import delegate as delegate_mod
        from cogtrix_core.tools.delegate import (
            _execute_single_task,
            configure_delegate,
        )

        configure_delegate({"circuit_breaker_cooldown": 0})

        quick_llm = MagicMock()
        response = type("Response", (), {"content": "delegated response text here"})()
        quick_llm.invoke.return_value = response
        quick_llm.model = "test/model"

        with patch.dict(delegate_mod._delegate_config, {"default_timeout": 5}):
            with patch.object(delegate_mod, "create_delegate_llm", return_value=quick_llm):
                with patch.object(delegate_mod, "get_delegate_tools", return_value=[]):
                    result = _execute_single_task(
                        task="test task",
                        context="test context",
                        response_format="text",
                        json_schema=None,
                        provider="test",
                        model="test",
                        temperature=0.5,
                        num_ctx=None,
                        use_tools=False,
                    )

                    from cogtrix_core.tools.delegate import DelegateResult

                    assert isinstance(result, DelegateResult)
                    assert result.success is True
                    assert "delegated response" in result.response
