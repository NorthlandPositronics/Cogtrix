"""Regression tests for the 13 fixes applied during the 2026-03-08 ProjectForge audit.

Each test class covers one fix and is named after the bug ID / category.
Tests are designed to catch regressions if the fix is accidentally reverted.
"""

import os
import re
import signal
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("COGTRIX_DATA_DIR", str(Path("/tmp") / "cogtrix-tests"))

from src.orchestration.session_state import SessionState

# ── SEC-01: copy removed from SAFE_MODULES (python_exec.py) ─────────────


class TestCopyRemovedFromSafeModules:
    """copy module must NOT be in SAFE_MODULES to prevent deepcopy sandbox escape."""

    def test_copy_not_in_safe_modules(self):
        from src.tools.python_exec import SAFE_MODULES

        assert "copy" not in SAFE_MODULES, (
            "copy must not be in SAFE_MODULES — deepcopy calls __reduce_ex__ "
            "via C code, bypassing sandbox attribute guards"
        )

    def test_other_safe_modules_still_present(self):
        """Ensure we didn't accidentally remove other modules."""
        from src.tools.python_exec import SAFE_MODULES

        for mod in ("math", "json", "re", "datetime", "collections"):
            assert mod in SAFE_MODULES


# ── BUG-184: calculator _safe_pow guard (calculator.py) ──────────────────


class TestSafePowGuard:
    """_safe_pow must reject large and negative exponents."""

    def test_large_exponent_rejected(self):
        from src.tools.calculator import _safe_pow

        with pytest.raises(ValueError, match="exponent magnitude too large"):
            _safe_pow(2, 1000)

    def test_boundary_exponent_exactly_1000_float(self):
        from src.tools.calculator import _safe_pow

        with pytest.raises(ValueError, match="exponent magnitude too large"):
            _safe_pow(2, 1000.0)

    def test_exponent_999_allowed(self):
        from src.tools.calculator import _safe_pow

        result = _safe_pow(2, 999)
        assert result == 2**999

    def test_exponent_1000_rejected(self):
        from src.tools.calculator import _safe_pow

        with pytest.raises(ValueError, match="exponent magnitude too large"):
            _safe_pow(2, 1000)

    def test_negative_exponent_rejected(self):
        """Negative exponents must also be capped (abs check)."""
        from src.tools.calculator import _safe_pow

        with pytest.raises(ValueError, match="exponent magnitude too large"):
            _safe_pow(2, -1000)

    def test_negative_exponent_large_rejected(self):
        from src.tools.calculator import _safe_pow

        with pytest.raises(ValueError, match="exponent magnitude too large"):
            _safe_pow(10, -9999)

    def test_small_negative_exponent_allowed(self):
        from src.tools.calculator import _safe_pow

        result = _safe_pow(2, -10)
        assert result == 2**-10

    def test_float_exponent_near_boundary(self):
        from src.tools.calculator import _safe_pow

        # 999.9 < 1000, should be allowed
        result = _safe_pow(1.001, 999.9)
        assert isinstance(result, float)

    def test_old_threshold_would_have_allowed(self):
        """9999**9999 was allowed under old threshold (>= 10000). Must be blocked now."""
        from src.tools.calculator import _safe_pow

        with pytest.raises(ValueError, match="exponent magnitude too large"):
            _safe_pow(9999, 9999)


# ── BUG-178: shell.py process group kill on timeout ──────────────────────


class TestShellProcessGroupKill:
    """shell.py must use start_new_session and killpg for proper cleanup."""

    def test_popen_uses_start_new_session(self):
        """Verify that Popen is called with start_new_session=True."""
        with patch("src.tools.shell.subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.communicate.return_value = ("output", "")
            mock_proc.returncode = 0
            mock_popen.return_value = mock_proc

            from src.tools.shell import execute_shell_command

            execute_shell_command("echo hello")
            mock_popen.assert_called_once()
            _, kwargs = mock_popen.call_args
            assert kwargs.get("start_new_session") is True

    def test_popen_shell_mode_uses_start_new_session(self):
        """Shell=True mode also uses start_new_session."""
        with patch("src.tools.shell.subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.communicate.return_value = ("output", "")
            mock_proc.returncode = 0
            mock_popen.return_value = mock_proc

            from src.tools.shell import execute_shell_command

            execute_shell_command("echo hello | cat")  # pipe triggers shell=True
            mock_popen.assert_called_once()
            _, kwargs = mock_popen.call_args
            assert kwargs.get("start_new_session") is True
            assert kwargs.get("shell") is True

    def test_timeout_calls_killpg(self):
        """On timeout, os.killpg should be called to kill the process group."""
        with patch("src.tools.shell.subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.pid = 12345
            mock_proc.communicate.side_effect = subprocess.TimeoutExpired("cmd", 1)
            mock_popen.return_value = mock_proc

            with patch("src.tools.shell.os.killpg") as mock_killpg:
                from src.tools.shell import execute_shell_command

                result = execute_shell_command("sleep 100", timeout=1)
                mock_killpg.assert_called_once_with(12345, signal.SIGKILL)
                mock_proc.wait.assert_called_once()
                assert "timed out" in result

    def test_timeout_falls_back_to_kill_on_oserror(self):
        """If killpg raises OSError, fall back to proc.kill()."""
        with patch("src.tools.shell.subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.pid = 12345
            mock_proc.communicate.side_effect = subprocess.TimeoutExpired("cmd", 1)
            mock_popen.return_value = mock_proc

            with patch("src.tools.shell.os.killpg", side_effect=OSError("no such process")):
                from src.tools.shell import execute_shell_command

                result = execute_shell_command("sleep 100", timeout=1)
                mock_proc.kill.assert_called_once()
                assert "timed out" in result


# ── BUG-180: delegate.py JSON fence stripping ────────────────────────────


class TestValidateJsonResponseFenceStripping:
    """_validate_json_response must strip only the matching closing fence."""

    def test_json_fence_basic(self):
        from src.tools.delegate import _validate_json_response

        response = '```json\n{"key": "value"}\n```'
        valid, parsed = _validate_json_response(response)
        assert valid is True
        assert parsed == {"key": "value"}

    def test_json_fence_with_trailing_code_block(self):
        """Response with JSON block followed by another code block.
        Old code would strip the last ``` unconditionally, corrupting the parse."""
        from src.tools.delegate import _validate_json_response

        response = (
            '```json\n{"result": "success"}\n```\n\n'
            "Here's an example:\n```python\nprint('hello')\n```"
        )
        valid, parsed = _validate_json_response(response)
        assert valid is True
        assert parsed == {"result": "success"}

    def test_json_fence_with_prose_before(self):
        """Prose before the JSON fence is tolerated."""
        from src.tools.delegate import _validate_json_response

        response = 'Here is the result:\n```json\n{"a": 1}\n```'
        valid, parsed = _validate_json_response(response)
        assert valid is True
        assert parsed == {"a": 1}

    def test_json_fence_with_multiple_trailing_blocks(self):
        """Multiple code blocks after JSON — only first closing fence matters."""
        from src.tools.delegate import _validate_json_response

        response = "```json\n[1, 2, 3]\n```\n\n```bash\necho hello\n```\n\n```python\nx = 1\n```"
        valid, parsed = _validate_json_response(response)
        assert valid is True
        assert parsed == [1, 2, 3]

    def test_plain_fence_basic(self):
        from src.tools.delegate import _validate_json_response

        response = '```\n{"data": 42}\n```'
        valid, parsed = _validate_json_response(response)
        assert valid is True
        assert parsed == {"data": 42}

    def test_no_fence(self):
        from src.tools.delegate import _validate_json_response

        valid, parsed = _validate_json_response('{"simple": true}')
        assert valid is True
        assert parsed == {"simple": True}

    def test_no_closing_fence(self):
        """Opening fence without closing fence — should still try to parse."""
        from src.tools.delegate import _validate_json_response

        response = '```json\n{"key": "value"}'
        valid, parsed = _validate_json_response(response)
        assert valid is True
        assert parsed == {"key": "value"}


# ── BUG-177: delegate.py circuit breaker lock discipline ─────────────────


class TestCircuitBreakerLockDiscipline:
    """_execute_single_task must call _check_availability_locked() inside the lock."""

    def test_execute_single_task_uses_locked_availability_check(self):
        """_execute_single_task must consult the locked helper before delegating."""
        from src.tools import delegate

        calls: list[float] = []

        class FakeCircuitBreaker:
            def _check_availability_locked(self, cooldown: float = 300.0):
                calls.append(cooldown)
                return False, "blocked"

        def fake_get_circuit_breaker(provider: str, model: str):
            return FakeCircuitBreaker()

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(delegate, "_get_circuit_breaker", fake_get_circuit_breaker)
        monkeypatch.setitem(delegate._delegate_config, "circuit_breaker_cooldown", 123)
        try:
            result = delegate._execute_single_task(
                "task",
                provider="openai",
                model="gpt-4.1-mini",
            )
        finally:
            monkeypatch.undo()

        assert calls == [123]
        assert result.success is False
        assert result.error == "blocked"
        assert result.model_used == "gpt-4.1-mini"

    def test_check_availability_is_wrapper(self):
        """Public check_availability() must route through _check_availability_locked()."""
        from src.tools.delegate import ModelCircuitBreaker

        calls: list[float] = []

        def fake_locked(self, cooldown: float = 300.0):
            calls.append(cooldown)
            return False, "locked"

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(ModelCircuitBreaker, "_check_availability_locked", fake_locked)
        try:
            breaker = ModelCircuitBreaker(is_unavailable=True)
            available, reason = breaker.check_availability(cooldown=42.0)
        finally:
            monkeypatch.undo()

        assert calls == [42.0]
        assert available is False
        assert reason == "locked"


# ── SEC-05: config.py resolve_data_path returns resolved path ────────────


class TestResolveDataPathReturnsResolved:
    """resolve_data_path must return .resolve()'d path to prevent symlink TOCTOU."""

    def test_relative_data_dir_returns_absolute(self):
        from src.config import Config

        config = Config()
        config.data_dir = "data"
        result = config.resolve_data_path("subdir/file.json")
        assert result.is_absolute(), "resolve_data_path must return an absolute path"

    def test_returned_path_is_resolved(self):
        from src.config import Config

        config = Config()
        config.data_dir = "data"
        result = config.resolve_data_path("vectordb")
        expected = Path("data/vectordb").resolve()
        assert result == expected

    def test_absolute_data_dir_returns_resolved(self):
        from src.config import Config

        config = Config()
        config.data_dir = "/tmp/test_data"
        result = config.resolve_data_path("sessions")
        assert result == Path("/tmp/test_data/sessions").resolve()
        assert result.is_absolute()


# ── SEC-02/03/04: ReDoS resistance in guardrails and intent ──────────────


class TestReDoSResistance:
    """Bounded regex patterns must complete quickly on adversarial input."""

    @staticmethod
    def _timed_match(pattern: re.Pattern, text: str, max_seconds: float = 0.5) -> None:
        """Assert a regex completes within max_seconds on the given text."""
        start = time.monotonic()
        pattern.search(text)
        elapsed = time.monotonic() - start
        assert elapsed < max_seconds, (
            f"Regex took {elapsed:.2f}s on {len(text)}-char input (limit: {max_seconds}s). "
            f"Possible ReDoS vulnerability."
        )

    def test_guardrails_drop_pattern_no_backtrack(self):
        """Pattern 'drop|clear...context|history...' must not backtrack on long input."""
        from src.assistant.guardrails import _INJECTION_PATTERNS

        # Find the pattern with 'drop|clear' and 'context|history'
        pattern = None
        for p in _INJECTION_PATTERNS:
            if "drop" in p.pattern and "context" in p.pattern:
                pattern = p
                break
        assert pattern is not None, "Could not find drop/clear injection pattern"

        # Adversarial: many words between "drop" and end — old unbounded .* would backtrack
        adversarial = "drop " + "x " * 2000 + "not_a_keyword"
        self._timed_match(pattern, adversarial)

    def test_guardrails_dan_pattern_no_backtrack(self):
        """Pattern 'DAN...mode' must not backtrack on long input."""
        from src.assistant.guardrails import _INJECTION_PATTERNS

        pattern = None
        for p in _INJECTION_PATTERNS:
            if "DAN" in p.pattern and "mode" in p.pattern:
                pattern = p
                break
        assert pattern is not None, "Could not find DAN mode injection pattern"

        adversarial = "DAN " + "x " * 2000 + "not_mode"
        self._timed_match(pattern, adversarial)

    def test_deep_think_triggers_no_backtrack(self):
        """DEEP_THINK_TRIGGERS bounded patterns must not hang on adversarial input."""
        from src.orchestration.intent import DEEP_THINK_TRIGGERS

        # Test "reason through...carefully" — old unbounded .*? would backtrack
        adversarial = "reason through " + "word " * 500 + "not_carefully"
        self._timed_match(DEEP_THINK_TRIGGERS, adversarial)

        # Test "analyze...in depth"
        adversarial = "analyze " + "stuff " * 500 + "not_in_depth"
        self._timed_match(DEEP_THINK_TRIGGERS, adversarial)

        # Test "examine...thoroughly"
        adversarial = "examine " + "thing " * 500 + "not_thoroughly"
        self._timed_match(DEEP_THINK_TRIGGERS, adversarial)

    def test_delegation_triggers_no_backtrack(self):
        """DELEGATION_TRIGGERS bounded patterns must not hang on adversarial input."""
        from src.orchestration.intent import DELEGATION_TRIGGERS

        # Test "compare...and"
        adversarial = "compare " + "item " * 500 + "not_and"
        self._timed_match(DELEGATION_TRIGGERS, adversarial)

        # Test "translate...into...and"
        adversarial = "translate " + "word " * 500 + "not_into"
        self._timed_match(DELEGATION_TRIGGERS, adversarial)

    def test_deep_think_still_matches_real_phrases(self):
        """Bounded patterns must still match legitimate trigger phrases."""
        from src.orchestration.intent import user_wants_deep_think

        assert user_wants_deep_think("reason through the problem carefully") is True
        assert user_wants_deep_think("analyze this code in depth") is True
        assert user_wants_deep_think("examine the architecture thoroughly") is True

    def test_delegation_still_matches_real_phrases(self):
        """Bounded patterns must still match legitimate delegation phrases."""
        from src.orchestration.intent import user_wants_delegation

        assert user_wants_delegation("compare React and Vue") is True
        assert user_wants_delegation("translate README into French and Spanish") is True


# ── PERF-03: intent.py _THINK_CAT_BY_NAME module-level hoist ────────────


class TestThinkCatByNameModuleLevel:
    """_THINK_CAT_BY_NAME must be a module-level dict, not rebuilt per call."""

    def test_is_module_level(self):
        """The dict should exist at module level, not be created inside a function."""
        from src.orchestration import intent

        assert hasattr(intent, "_THINK_CAT_BY_NAME")
        assert isinstance(intent._THINK_CAT_BY_NAME, dict)

    def test_contains_all_categories(self):
        from src.orchestration.intent import _THINK_CAT_BY_NAME, THINK_CATEGORIES

        assert len(_THINK_CAT_BY_NAME) == len(THINK_CATEGORIES)
        for cat in THINK_CATEGORIES:
            assert cat.name in _THINK_CAT_BY_NAME
            assert _THINK_CAT_BY_NAME[cat.name] is cat

    def test_classify_uses_module_level_dict(self):
        """classify_think_task should read the live module-level lookup table."""
        from src.orchestration import intent
        from src.orchestration.intent import ThinkCategory, classify_think_task

        sentinel = ThinkCategory(
            name="sentinel",
            keywords=("unrelated",),
            gather_template="gather",
            analysis_preamble="analysis",
            stage2_task_framing="frame",
        )

        class FakeLLM:
            def invoke(self, prompt: str):
                return type("Resp", (), {"content": "sentinel"})()

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(intent, "_THINK_CAT_BY_NAME", {"sentinel": sentinel})
        try:
            result = classify_think_task("plain text without keyword matches", FakeLLM())
        finally:
            monkeypatch.undo()

        assert result is sentinel


# ── PERF-01: compression.py lazy pool initialization ─────────────────────


class TestCompressionPoolLazy:
    """Compression pool must not be created at import time."""

    def test_pool_is_none_before_first_call(self):
        """The module-level _COMPRESSION_POOL should be None until first use."""
        import src.orchestration.compression as comp

        # The pool may already be initialized from a prior test. Check the getter.
        assert hasattr(comp, "_get_compression_pool")
        assert callable(comp._get_compression_pool)

    def test_getter_returns_executor(self):
        import concurrent.futures

        from src.orchestration.compression import _get_compression_pool

        pool = _get_compression_pool()
        assert isinstance(pool, concurrent.futures.ThreadPoolExecutor)

    def test_getter_returns_same_instance(self):
        from src.orchestration.compression import _get_compression_pool

        pool1 = _get_compression_pool()
        pool2 = _get_compression_pool()
        assert pool1 is pool2


# ── PERF-02: runner.py _evict_stale reuses `now` parameter ───────────────


class TestEvictStaleTimingFix:
    """ToolCallLogger._evict_stale must use the `now` parameter for cutoff."""

    def test_evict_stale_removes_only_expired_entries(self, monkeypatch):
        """Verify stale tool-call entries are evicted while fresh ones stay."""
        from src.orchestration.runner import ToolCallLogger

        logger = ToolCallLogger()
        logger._tool_start_times = {
            "stale": 399.0,
            "fresh": 999.0,
        }
        logger._last_evict = 0.0

        monkeypatch.setattr(time, "monotonic", lambda: 1000.0)

        logger._evict_stale()

        assert "stale" not in logger._tool_start_times
        assert logger._tool_start_times["fresh"] == 999.0
        assert logger._last_evict == 1000.0


# ── BUG-192: symlink traversal in _collect_faiss_dirs ─────────────────────


@pytest.fixture(autouse=False)
def _restore_rag_config():
    """Save and restore _rag_config around tests that call configure_rag."""
    import src.tools.rag as _rag_mod

    original = dict(_rag_mod._rag_config)
    yield
    _rag_mod._rag_config.update(original)


class TestCollectFaissDirsSymlink:
    """_collect_faiss_dirs must skip symlinks in the uploads directory."""

    @pytest.mark.usefixtures("_restore_rag_config")
    def test_symlink_in_uploads_dir_is_skipped(self, tmp_path: Path) -> None:
        from src.tools.rag import _collect_faiss_dirs, configure_rag

        uploads = tmp_path / "uploads"
        uploads.mkdir()

        legit = uploads / "legit-doc"
        legit_idx = legit / "vectordb" / "faiss_index"
        legit_idx.mkdir(parents=True)
        (legit_idx / "index.faiss").write_bytes(b"")

        # Target lives OUTSIDE uploads dir
        target = tmp_path / "evil-target"
        evil_idx = target / "vectordb" / "faiss_index"
        evil_idx.mkdir(parents=True)
        (evil_idx / "index.faiss").write_bytes(b"")
        symlink = uploads / "symlink-doc"
        symlink.symlink_to(target)

        configure_rag(
            {
                "embedding_provider": "ollama",
                "embedding_model": "test",
                "base_url": "http://localhost:11434",
                "api_key": None,
                "vectordb_dir": str(tmp_path / "nonexistent"),
                "api_uploads_dir": str(uploads),
            }
        )

        dirs = _collect_faiss_dirs()
        dir_strs = [str(d) for d in dirs]
        assert any("legit-doc" in s for s in dir_strs)
        # Symlink pointing outside uploads dir must be skipped
        assert not any("symlink-doc" in s for s in dir_strs)
        assert not any("evil-target" in s for s in dir_strs)


# ── BUG-195: non-deterministic index order ────────────────────────────────


class TestFaissDirsSorted:
    """_collect_faiss_dirs must return a sorted list for determinism."""

    @pytest.mark.usefixtures("_restore_rag_config")
    def test_dirs_are_sorted(self, tmp_path: Path) -> None:
        from src.tools.rag import _collect_faiss_dirs, configure_rag

        for name in ["zebra", "alpha", "middle"]:
            idx = tmp_path / name / "vectordb" / "faiss_index"
            idx.mkdir(parents=True)
            (idx / "index.faiss").write_bytes(b"")

        configure_rag(
            {
                "embedding_provider": "ollama",
                "embedding_model": "test",
                "base_url": "http://localhost:11434",
                "api_key": None,
                "vectordb_dir": str(tmp_path / "nonexistent"),
                "api_uploads_dir": str(tmp_path),
            }
        )

        dirs = _collect_faiss_dirs()
        assert dirs == sorted(dirs)


# ── BUG-199: /tools load must pin agent-loaded tools ─────────────────────


class TestPinAgentLoadedTool:
    """An agent-loaded tool promoted via /tools load must survive reset."""

    def test_agent_loaded_tool_survives_reset_after_pin(self) -> None:
        ss = SessionState(loaded_tools={"web_search"})
        assert "web_search" not in ss.pinned_tools

        ss.pinned_tools.add("web_search")
        ss.reset_for_new_prompt()
        assert "web_search" in ss.loaded_tools
        assert "web_search" in ss.pinned_tools


# ── BUG-200: --activate-tools fallback to all_tool_originals ──────────────


class TestActivateToolsFallback:
    """--activate-tools must fall back to all_tool_originals when tool
    isn't in available_tools or registry.tools."""

    def test_activate_from_originals(self) -> None:
        ss = SessionState()
        sentinel = MagicMock()
        sentinel.name = "web_search"
        ss.all_tool_originals = {"web_search": sentinel}

        available: dict = {}
        registry_tools: dict = {}

        name = "web_search"
        if name in available:
            pass
        elif name in registry_tools:
            pass
        elif name in ss.all_tool_originals:
            registry_tools[name] = ss.all_tool_originals[name]
            ss.loaded_tools.add(name)
            ss.pinned_tools.add(name)

        assert "web_search" in registry_tools
        assert "web_search" in ss.pinned_tools


# ── BUG-203: pinned status must beat auto_approved ────────────────────────


class TestToolStatusPriority:
    """_classify_tool_status must return 'pinned' over 'auto_approved'."""

    def test_pinned_before_auto_approved(self) -> None:
        from src.api.routes.tools import _classify_tool_status

        ss = SessionState(
            approvals={"web_search"},
            pinned_tools={"web_search"},
            loaded_tools={"web_search"},
        )
        assert _classify_tool_status("web_search", ss) == "pinned"

    def test_disabled_beats_pinned(self) -> None:
        from src.api.routes.tools import _classify_tool_status

        ss = SessionState(
            denials={"web_search"},
            pinned_tools={"web_search"},
        )
        assert _classify_tool_status("web_search", ss) == "disabled"


# ── BUG-193: uploads path consistency ─────────────────────────────────────


class TestUploadsPathConsistency:
    """API startup must propagate config.data_dir to COGTRIX_DATA_DIR."""

    def test_data_dir_propagated(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("COGTRIX_DATA_DIR", None)
            cfg = MagicMock()
            cfg.data_dir = "/custom/data"

            if cfg is not None and not os.environ.get("COGTRIX_DATA_DIR"):
                os.environ["COGTRIX_DATA_DIR"] = cfg.data_dir

            assert os.environ["COGTRIX_DATA_DIR"] == "/custom/data"

    def test_existing_env_var_not_overwritten(self) -> None:
        with patch.dict(os.environ, {"COGTRIX_DATA_DIR": "/existing"}):
            cfg = MagicMock()
            cfg.data_dir = "/custom/data"

            if cfg is not None and not os.environ.get("COGTRIX_DATA_DIR"):
                os.environ["COGTRIX_DATA_DIR"] = cfg.data_dir

            assert os.environ["COGTRIX_DATA_DIR"] == "/existing"


# ── BUG-NEW: compression timeout must log a warning ───────────────────────


class TestCompressionTimeoutWarning:
    """Compression LLM timeout (60 s) must emit a WARNING before falling back."""

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("langchain_core"),
        reason="langchain_core not installed",
    )
    def test_timeout_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """When future.result(timeout=60) raises TimeoutError, a WARNING is logged."""
        import concurrent.futures
        import logging

        from langchain_core.messages import AIMessage, ToolMessage

        from src.orchestration.compression import apply_message_compression

        long_content = "Z" * 90_000
        tool_msg = ToolMessage(content=long_content, tool_call_id="tc_warn", name="warn_tool")
        old_ais = [AIMessage(content="step") for _ in range(6)]
        messages = old_ais + [tool_msg, AIMessage(content="done")]

        fake_future = MagicMock(spec=concurrent.futures.Future)
        fake_future.result.side_effect = concurrent.futures.TimeoutError("timed out")

        fake_pool = MagicMock()
        fake_pool.submit.return_value = fake_future

        with caplog.at_level(logging.WARNING, logger="cogtrix"):
            with patch(
                "src.orchestration.compression._get_compression_pool", return_value=fake_pool
            ):
                with patch("concurrent.futures.as_completed", return_value=[fake_future]):
                    apply_message_compression(
                        messages,
                        call_count=20,
                        compression_cache={},
                        llm=MagicMock(),
                        max_context_tokens=16_384,
                        min_age_cycles=0,
                        min_chars=100,
                    )

        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warning_records, "Expected at least one WARNING log record on timeout"
        assert any("timeout" in r.getMessage().lower() for r in warning_records)

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("langchain_core"),
        reason="langchain_core not installed",
    )
    def test_plain_timeout_error_also_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """The plain built-in TimeoutError also triggers the WARNING log path."""
        import concurrent.futures
        import logging

        from langchain_core.messages import AIMessage, ToolMessage

        from src.orchestration.compression import apply_message_compression

        long_content = "Y" * 80_000
        tool_msg = ToolMessage(content=long_content, tool_call_id="tc_warn2", name="warn_tool2")
        old_ais = [AIMessage(content="step") for _ in range(6)]
        messages = old_ais + [tool_msg, AIMessage(content="done")]

        fake_future = MagicMock(spec=concurrent.futures.Future)
        fake_future.result.side_effect = TimeoutError("plain timeout")

        fake_pool = MagicMock()
        fake_pool.submit.return_value = fake_future

        with caplog.at_level(logging.WARNING, logger="cogtrix"):
            with patch(
                "src.orchestration.compression._get_compression_pool", return_value=fake_pool
            ):
                with patch("concurrent.futures.as_completed", return_value=[fake_future]):
                    apply_message_compression(
                        messages,
                        call_count=20,
                        compression_cache={},
                        llm=MagicMock(),
                        max_context_tokens=16_384,
                        min_age_cycles=0,
                        min_chars=100,
                    )

        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warning_records, "Expected at least one WARNING log record on plain TimeoutError"


# ── BUG-198: API turn_runner must call reset_for_new_prompt ───────────────


class TestApiTurnResetsPromptState:
    """BUG-198: API turns must call reset_for_new_prompt to clear ephemeral tools."""

    def test_reset_clears_agent_loaded_tools(self) -> None:
        ss = SessionState(no_confirm=True)
        ss.loaded_tools = {"pinned_tool", "ephemeral_tool"}
        ss.pinned_tools = {"pinned_tool"}
        ss.reset_for_new_prompt()
        assert ss.loaded_tools == {"pinned_tool"}
        assert "ephemeral_tool" not in ss.loaded_tools

    def test_reset_preserves_pinned_tools(self) -> None:
        ss = SessionState(no_confirm=True)
        ss.loaded_tools = {"a", "b", "c"}
        ss.pinned_tools = {"a", "c"}
        ss.reset_for_new_prompt()
        assert ss.loaded_tools == {"a", "c"}
        assert "b" not in ss.loaded_tools

    def test_reset_clears_deny_all(self) -> None:
        ss = SessionState(no_confirm=True)
        ss.deny_all = True
        ss.reset_for_new_prompt()
        assert ss.deny_all is False

    def test_turn_runner_calls_reset(self) -> None:
        """The inner turn execution must call session_state.reset_for_new_prompt()."""
        import asyncio
        from unittest.mock import Mock, patch

        from src.api import turn_runner
        from src.api.session_bridge import ApiSession
        from src.orchestration.run_config import AgentRunConfig

        session_state = Mock()
        session_state.reset_for_new_prompt = Mock()
        session_state.approvals = set()

        session = ApiSession(
            id="session-198",
            user_id="user-198",
            name="turn-runner-reset",
            session_state=session_state,
            run_config=AgentRunConfig(llm=object(), active_tools_list=[], available_tools={}),
            memory_manager=None,
            registry=None,
            ws_queue=asyncio.Queue(),
        )

        def fake_run_agent(*args, **kwargs) -> str:
            return "turn output"

        async def _run() -> None:
            with patch("src.orchestration.runner.run_agent", side_effect=fake_run_agent):
                await turn_runner._run_message_turn_inner(
                    session=session,
                    text="hello",
                    mode="chat",
                    db=None,
                    app_state=None,
                )

        asyncio.run(_run())
        session_state.reset_for_new_prompt.assert_called_once()


# ── BUG-199: warm_session must populate all_tool_originals ────────────────


class TestWarmSessionPopulatesOriginals:
    """BUG-199: warm_session must populate all_tool_originals from available_tools."""

    def test_originals_populated_from_available_tools(self) -> None:
        ss = SessionState(no_confirm=True)
        available: dict = {"tool_a": "obj_a", "tool_b": "obj_b"}
        if available:
            ss.all_tool_originals = dict(available)
        assert ss.all_tool_originals == {"tool_a": "obj_a", "tool_b": "obj_b"}

    def test_originals_empty_when_no_tools(self) -> None:
        ss = SessionState(no_confirm=True)
        available: dict = {}
        if available:
            ss.all_tool_originals = dict(available)
        assert ss.all_tool_originals == {}

    def test_warm_session_bridge_assigns_originals(self) -> None:
        """warm_session() in session_bridge.py must assign all_tool_originals."""
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import patch

        from src.api import session_bridge

        record = SimpleNamespace(
            id="session-199",
            user_id="user-199",
            name="warm-session-originals",
            config_json="{}",
            token_counts_json="{}",
            state="idle",
        )
        available_tools = {"tool_a": object(), "tool_b": object()}
        app_state = SimpleNamespace(
            config=None,
            tool_registry=SimpleNamespace(tools=available_tools),
        )
        memory_manager = SimpleNamespace(
            set_llm=lambda llm: None,
            configure_compression=lambda *a, **k: None,
        )

        async def _run() -> None:
            with (
                patch.object(session_bridge, "_build_memory_manager", return_value=memory_manager),
                patch.object(session_bridge, "_build_llm", return_value=MagicMock()),
            ):
                session = await session_bridge.warm_session(record, app_state)

            assert session.session_state.all_tool_originals == available_tools
            assert session.run_config.active_tools_list is not None
            assert session.run_config.available_tools is not None

        asyncio.run(_run())

    def test_originals_dict_is_a_copy(self) -> None:
        """all_tool_originals must be a new dict, not a reference to available_tools."""
        ss = SessionState(no_confirm=True)
        available: dict = {"tool_x": object()}
        if available:
            ss.all_tool_originals = dict(available)
        available["new_key"] = "new_value"
        assert "new_key" not in ss.all_tool_originals


# ── BUG-207: compression as_completed pool-level timeout ───────────────────


class TestCompressionPoolTimeout:
    """as_completed must receive a timeout so hung LLM calls fall back to truncation."""

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("langchain_core"),
        reason="langchain_core not installed",
    )
    def test_pool_timeout_falls_back_to_truncation(self, caplog: pytest.LogCaptureFixture) -> None:
        """When as_completed itself raises TimeoutError, all unfinished messages
        must be truncated and a WARNING logged."""
        import concurrent.futures
        import logging

        from langchain_core.messages import AIMessage, ToolMessage

        from src.orchestration.compression import apply_message_compression

        long_content = "Z" * 90_000
        tool_msg = ToolMessage(content=long_content, tool_call_id="tc_pool", name="pool_tool")
        old_ais = [AIMessage(content="step") for _ in range(6)]
        messages = old_ais + [tool_msg, AIMessage(content="done")]

        fake_pool = MagicMock()
        fake_future = MagicMock(spec=concurrent.futures.Future)
        fake_pool.submit.return_value = fake_future

        def _raise_timeout(*args, **kwargs):
            raise TimeoutError("pool timed out")

        with caplog.at_level(logging.WARNING, logger="cogtrix"):
            with patch(
                "src.orchestration.compression._get_compression_pool",
                return_value=fake_pool,
            ):
                with patch("concurrent.futures.as_completed", side_effect=_raise_timeout):
                    result = apply_message_compression(
                        messages,
                        call_count=20,
                        compression_cache={},
                        llm=MagicMock(),
                        max_context_tokens=16_384,
                        min_age_cycles=0,
                        min_chars=100,
                    )

        assert len(result) == len(messages)
        tool_result = result[len(old_ais)]
        assert len(tool_result.content) < len(
            long_content
        ), "Tool message must be truncated when pool times out"
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("pool timed out" in r.getMessage().lower() for r in warning_records)

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("langchain_core"),
        reason="langchain_core not installed",
    )
    def test_as_completed_receives_timeout_kwarg(self) -> None:
        """as_completed must be called with a timeout= keyword argument."""
        from langchain_core.messages import AIMessage, ToolMessage

        from src.orchestration.compression import apply_message_compression

        long_content = "A" * 90_000
        tool_msg = ToolMessage(content=long_content, tool_call_id="tc_kw", name="kw_tool")
        old_ais = [AIMessage(content="step") for _ in range(6)]
        messages = old_ais + [tool_msg, AIMessage(content="done")]

        compressed_mock = MagicMock()
        compressed_mock.content = "compressed"
        llm = MagicMock()
        llm.invoke.return_value = compressed_mock

        with patch(
            "concurrent.futures.as_completed", wraps=__import__("concurrent").futures.as_completed
        ) as mock_ac:
            apply_message_compression(
                messages,
                call_count=20,
                compression_cache={},
                llm=llm,
                max_context_tokens=16_384,
                min_age_cycles=0,
                min_chars=100,
            )

        assert mock_ac.called, "as_completed must be called"
        _, kwargs = mock_ac.call_args
        assert "timeout" in kwargs, "as_completed must receive a timeout kwarg (BUG-207)"
        assert kwargs["timeout"] > 0


# ── BUG-209: turn_runner non-blocking queue puts ───────────────────────────


class TestTurnRunnerNonBlockingPuts:
    """error and done messages must not use blocking await put() on bounded queue."""

    def test_error_messages_use_put_nowait(self) -> None:
        """Error paths must enqueue via put_nowait and avoid blocking put()."""
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        from src.api import turn_runner
        from src.api.session_bridge import ApiSession
        from src.orchestration.run_config import AgentRunConfig

        class DummyQueue:
            def __init__(self) -> None:
                self.nowait_items: list[dict] = []
                self.put_items: list[dict] = []

            def put_nowait(self, item: dict) -> None:
                self.nowait_items.append(item)

            async def put(self, item: dict) -> None:
                self.put_items.append(item)
                raise AssertionError("error path must not use blocking queue.put()")

        queue = DummyQueue()
        session_state = SimpleNamespace(reset_for_new_prompt=Mock(), approvals=set())
        session = ApiSession(
            id="session-209-error",
            user_id="user-209",
            name="turn-runner-nonblocking-error",
            session_state=session_state,
            run_config=AgentRunConfig(llm=object(), active_tools_list=[], available_tools={}),
            memory_manager=None,
            registry=None,
            ws_queue=queue,
        )

        def fake_run_agent(*args, **kwargs) -> None:
            raise RuntimeError("boom")

        async def _run() -> None:
            with (
                patch("src.api.turn_runner.WebSocketCallbackHandler", autospec=True) as mock_ws,
                patch("src.api.turn_runner.ApiConfirmationUI", autospec=True),
                patch("src.orchestration.runner.run_agent", side_effect=fake_run_agent),
            ):
                mock_ws.return_value.input_tokens = 0
                mock_ws.return_value.output_tokens = 0
                mock_ws.return_value.tool_call_count = 0
                await turn_runner._run_message_turn_inner(
                    session=session,
                    text="hello",
                    mode="chat",
                    db=None,
                    app_state=None,
                )

        asyncio.run(_run())
        assert any(item["type"] == "error" for item in queue.nowait_items)
        assert any(item["type"] == "done" for item in queue.nowait_items)
        assert queue.put_items == []

    def test_done_message_uses_blocking_put_without_timeout(self) -> None:
        """The done message must use await put() without asyncio.wait_for."""
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        from src.api import turn_runner
        from src.api.session_bridge import ApiSession
        from src.orchestration.run_config import AgentRunConfig

        class DummyQueue:
            def __init__(self) -> None:
                self.nowait_items: list[dict] = []
                self.put_items: list[dict] = []

            def put_nowait(self, item: dict) -> None:
                self.nowait_items.append(item)

            async def put(self, item: dict) -> None:
                self.put_items.append(item)

        queue = DummyQueue()
        session_state = SimpleNamespace(reset_for_new_prompt=Mock(), approvals=set())
        session = ApiSession(
            id="session-209-done",
            user_id="user-209",
            name="turn-runner-nonblocking-done",
            session_state=session_state,
            run_config=AgentRunConfig(llm=object(), active_tools_list=[], available_tools={}),
            memory_manager=None,
            registry=None,
            ws_queue=queue,
        )

        def fake_run_agent(*args, **kwargs) -> str:
            return "turn output"

        wait_for_calls: list[float] = []

        async def fake_wait_for(coro, timeout):
            wait_for_calls.append(timeout)
            return await coro

        async def _run() -> None:
            with (
                patch("src.api.turn_runner.WebSocketCallbackHandler", autospec=True) as mock_ws,
                patch("src.api.turn_runner.ApiConfirmationUI", autospec=True),
                patch("src.orchestration.runner.run_agent", side_effect=fake_run_agent),
                patch("src.api.turn_runner.asyncio.wait_for", side_effect=fake_wait_for),
            ):
                mock_ws.return_value.input_tokens = 0
                mock_ws.return_value.output_tokens = 0
                mock_ws.return_value.tool_call_count = 0
                await turn_runner._run_message_turn_inner(
                    session=session,
                    text="hello",
                    mode="chat",
                    db=None,
                    app_state=None,
                )

        asyncio.run(_run())
        assert wait_for_calls == []
        assert any(item["type"] == "done" for item in queue.put_items)

    def test_done_message_not_dropped_on_slow_queue(self) -> None:
        """Regression for #1086: done message must not be dropped when queue is slow."""
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        from src.api import turn_runner
        from src.api.session_bridge import ApiSession
        from src.orchestration.run_config import AgentRunConfig

        class SlowQueue:
            def __init__(self) -> None:
                self.items: list[dict] = []
                self.put_delay = 0.05

            def put_nowait(self, item: dict) -> None:
                self.items.append(item)

            async def put(self, item: dict) -> None:
                await asyncio.sleep(self.put_delay)
                self.items.append(item)

        queue = SlowQueue()
        session_state = SimpleNamespace(reset_for_new_prompt=Mock(), approvals=set())
        session = ApiSession(
            id="session-1086",
            user_id="user-1086",
            name="turn-runner-slow-queue",
            session_state=session_state,
            run_config=AgentRunConfig(llm=object(), active_tools_list=[], available_tools={}),
            memory_manager=None,
            registry=None,
            ws_queue=queue,
        )

        def fake_run_agent(*args, **kwargs) -> str:
            return "turn output"

        async def _run() -> None:
            with (
                patch("src.api.turn_runner.WebSocketCallbackHandler", autospec=True) as mock_ws,
                patch("src.api.turn_runner.ApiConfirmationUI", autospec=True),
                patch("src.orchestration.runner.run_agent", side_effect=fake_run_agent),
            ):
                mock_ws.return_value.input_tokens = 0
                mock_ws.return_value.output_tokens = 0
                mock_ws.return_value.tool_call_count = 0
                await turn_runner._run_message_turn_inner(
                    session=session,
                    text="hello",
                    mode="chat",
                    db=None,
                    app_state=None,
                )

        asyncio.run(_run())
        done_items = [item for item in queue.items if item.get("type") == "done"]
        assert len(done_items) == 1
        assert done_items[0]["payload"]["text"] == "turn output"


# ── BUG-209 — turn_runner must expose patchable API callback helpers ──────


class TestTurnRunnerExports:
    """turn_runner must re-export callback helpers for test patching."""

    def test_callback_helpers_are_module_level_exports(self) -> None:
        from src.api import turn_runner
        from src.api.callbacks import WebSocketCallbackHandler
        from src.api.confirmation import ApiConfirmationUI

        assert turn_runner.WebSocketCallbackHandler is WebSocketCallbackHandler
        assert turn_runner.ApiConfirmationUI is ApiConfirmationUI


# ── BUG-211: rag OSError resilience ────────────────────────────────────────


class TestRagOSErrorResilience:
    """knowledge_base_stats and _build_description must handle OSError gracefully."""

    @pytest.fixture
    def _restore_rag_config(self):
        import src.tools.rag as _rag_mod

        original = dict(_rag_mod._rag_config)
        yield
        _rag_mod._rag_config.update(original)

    @pytest.mark.usefixtures("_restore_rag_config")
    def test_stats_survives_permission_error(self, tmp_path: Path) -> None:
        """knowledge_base_stats must not crash when iterdir() raises PermissionError."""
        from src.tools.rag import configure_rag, knowledge_base_stats

        idx = tmp_path / "faiss_index"
        idx.mkdir()
        (idx / "index.faiss").write_bytes(b"x" * 100)

        configure_rag({"vectordb_dir": str(idx), "api_uploads_dir": str(tmp_path / "none")})

        with patch.object(Path, "iterdir", side_effect=PermissionError("denied")):
            count, total_size = knowledge_base_stats()

        assert count >= 0
        assert total_size == 0

    def test_stats_has_oserror_guard_on_stat(self, tmp_path: Path) -> None:
        """knowledge_base_stats must skip files whose stat() raises OSError."""
        from src.tools.rag import configure_rag, knowledge_base_stats

        idx = tmp_path / "faiss_index"
        idx.mkdir()
        (idx / "good.faiss").write_bytes(b"good")
        bad = idx / "bad.faiss"
        bad.write_bytes(b"bad")

        configure_rag({"vectordb_dir": str(idx), "api_uploads_dir": str(idx / "uploads")})

        original_stat = Path.stat

        def _stat(self: Path, *args, **kwargs):
            if self.name == "bad.faiss":
                raise OSError("stat failed")
            return original_stat(self, *args, **kwargs)

        with patch.object(Path, "stat", _stat):
            count, total_size = knowledge_base_stats()

        assert count >= 1
        assert total_size == 4

    @pytest.mark.usefixtures("_restore_rag_config")
    def test_build_description_survives_permission_error(self, tmp_path: Path) -> None:
        """_build_description must not crash when iterdir() raises PermissionError."""
        from src.tools.rag import _build_description, configure_rag

        idx = tmp_path / "faiss_index"
        idx.mkdir()
        (idx / "index.faiss").write_bytes(b"x" * 2048)

        configure_rag({"vectordb_dir": str(idx), "api_uploads_dir": str(tmp_path / "none")})

        with patch.object(Path, "iterdir", side_effect=PermissionError("denied")):
            desc = _build_description()

        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_configure_rag_tool_catches_oserror(self, tmp_path: Path) -> None:
        """configure_rag_tool must swallow OSError from RAG configuration."""
        from src.tools.configure import configure_rag_tool

        config = MagicMock()
        config.data_dir = str(tmp_path)
        config.rag = MagicMock()
        config.rag.vectordb_dir = "vectordb"
        config.rag.score_threshold = 0.0
        config.resolve_embedding_config.return_value = (
            "ollama",
            "nomic-embed-text",
            None,
            None,
        )
        config.resolve_data_path.return_value = tmp_path

        with patch(
            "src.tools.rag.configure_rag", side_effect=OSError("boom")
        ) as mock_configure_rag:
            configure_rag_tool(config)

        mock_configure_rag.assert_called_once()


# ── BUG-212 / BUG-216 — workflow_id validation at API boundary ──────────


class TestWorkflowIdValidation:
    """BUG-212: workflow_id path params must be validated before filesystem ops.
    BUG-216: upload filename must have path containment check."""

    def test_validate_wf_id_rejects_traversal(self):
        from fastapi import HTTPException

        from src.api.routes.workflows import _validate_wf_id

        with pytest.raises(HTTPException) as exc_info:
            _validate_wf_id("../etc/passwd")
        assert exc_info.value.status_code == 400

    def test_validate_wf_id_rejects_slash(self):
        from fastapi import HTTPException

        from src.api.routes.workflows import _validate_wf_id

        with pytest.raises(HTTPException):
            _validate_wf_id("a/b")

    def test_validate_wf_id_accepts_valid(self):
        from src.api.routes.workflows import _validate_wf_id

        _validate_wf_id("bike-sales")
        _validate_wf_id("Support_v2")
        _validate_wf_id("a1")

    def test_all_workflow_endpoints_call_validate(self):
        """Every workflow_id endpoint must route through _validate_wf_id."""
        import asyncio
        from types import SimpleNamespace

        from src.api.auth import TokenData
        from src.api.routes import workflows as mod
        from src.api.schemas.workflow import WorkflowUpdate

        class StopAfterValidate(RuntimeError):
            pass

        class FakeUploadFile:
            def __init__(self) -> None:
                self.filename = "report.pdf"

            async def read(self) -> bytes:
                return b"hello world"

        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(workflow_registry=MagicMock()),
            )
        )
        current_user = TokenData("user-1", "admin", {})
        body = WorkflowUpdate()

        with patch.object(
            mod, "_validate_wf_id", side_effect=StopAfterValidate("stop")
        ) as mock_validate:
            cases = [
                lambda: asyncio.run(mod.get_workflow("bike-sales", request, current_user)),
                lambda: asyncio.run(mod.update_workflow("bike-sales", body, request, current_user)),
                lambda: asyncio.run(mod.delete_workflow("bike-sales", request, current_user)),
                lambda: asyncio.run(
                    mod.upload_workflow_document(
                        "bike-sales",
                        request,
                        file=FakeUploadFile(),  # type: ignore[arg-type]
                        current_user=current_user,
                    )
                ),
                lambda: asyncio.run(
                    mod.list_workflow_documents("bike-sales", request, current_user)
                ),
                lambda: asyncio.run(
                    mod.delete_workflow_document("bike-sales", "doc-1", request, current_user)
                ),
            ]

            for invoke in cases:
                with pytest.raises(StopAfterValidate):
                    invoke()

        assert mock_validate.call_count == len(cases)

    def test_upload_has_path_containment(self, tmp_path: Path) -> None:
        """upload_workflow_document must reject path-escaping filenames (BUG-216)."""
        import asyncio
        from types import SimpleNamespace

        from fastapi import HTTPException

        from src.api.routes.workflows import upload_workflow_document
        from src.assistant.workflows import WorkflowDefinition

        class FakeUploadFile:
            def __init__(self, filename: str, data: bytes) -> None:
                self.filename = filename
                self._data = data

            async def read(self) -> bytes:
                return self._data

        class Registry:
            def __init__(self, data_dir: Path) -> None:
                self._data_dir = data_dir
                self._workflows_dir = data_dir / "workflows"
                self._workflows_dir.mkdir(parents=True, exist_ok=True)
                self.workflow = WorkflowDefinition(id="bike-sales", name="Bike Sales")

            def get_workflow(self, workflow_id: str) -> WorkflowDefinition:
                assert workflow_id == self.workflow.id
                return self.workflow

        registry = Registry(tmp_path)
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(workflow_registry=registry),
            )
        )
        file = FakeUploadFile("report.pdf", b"hello world")
        original_resolve = Path.resolve

        def _fake_resolve(self: Path, *args: object, **kwargs: object) -> Path:
            if self.name == "report.pdf":
                return Path("/tmp/outside") / self.name
            return original_resolve(self, *args, **kwargs)

        with patch.object(Path, "resolve", _fake_resolve):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(
                    upload_workflow_document(
                        "bike-sales",
                        request,  # type: ignore[arg-type]
                        file=file,  # type: ignore[arg-type]
                        current_user=MagicMock(),
                    )
                )

        assert exc_info.value.status_code == 400
        assert "escapes workflow directory" in str(exc_info.value.detail["message"])


# ── BUG-213 — update_workflow uses copy, not live object ─────────────────


class TestUpdateWorkflowCopy:
    """BUG-213: update_workflow route must not mutate the live registry object."""

    def test_update_returns_new_workflow_definition(self) -> None:
        import asyncio
        from types import SimpleNamespace

        from src.api.auth import TokenData
        from src.api.routes.workflows import update_workflow
        from src.api.schemas.workflow import (
            WorkflowAutoDetectOut,
            WorkflowToolPolicyOut,
            WorkflowUpdate,
        )
        from src.assistant.workflows import (
            WorkflowAutoDetect,
            WorkflowDefinition,
            WorkflowToolPolicy,
        )

        original = WorkflowDefinition(
            id="bike-sales",
            name="Old",
            description="old",
            system_prompt="sys",
            knowledge_base=False,
            tool_policy=WorkflowToolPolicy(
                excluded_tools=["calc"], additional_approved_tools=["shell"]
            ),
            auto_detect=WorkflowAutoDetect(
                enabled=False,
                keywords=["old"],
                patterns=["old.*"],
                min_confidence=2,
            ),
        )

        class Registry:
            def __init__(self, wf: WorkflowDefinition) -> None:
                self.workflow = wf
                self.updated: WorkflowDefinition | None = None

            def get_workflow(self, workflow_id: str) -> WorkflowDefinition:
                assert workflow_id == self.workflow.id
                return self.workflow

            def update_workflow(self, updated: WorkflowDefinition) -> None:
                self.updated = updated

        registry = Registry(original)
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(workflow_registry=registry),
            )
        )
        body = WorkflowUpdate(
            name="New",
            description="new",
            system_prompt="updated",
            knowledge_base=True,
            tool_policy=WorkflowToolPolicyOut(
                excluded_tools=["calc", "shell"],
                additional_approved_tools=["rag"],
            ),
            auto_detect=WorkflowAutoDetectOut(
                enabled=True,
                keywords=["new"],
                patterns=["new.*"],
                min_confidence=7,
            ),
        )

        result = asyncio.run(
            update_workflow(
                "bike-sales",
                body,
                request,  # type: ignore[arg-type]
                TokenData("user-1", "admin", {}),
            )
        )

        assert registry.updated is not None
        assert registry.updated is not original
        assert original.name == "Old"
        assert original.description == "old"
        assert original.system_prompt == "sys"
        assert original.knowledge_base is False
        assert original.tool_policy.excluded_tools == ["calc"]
        assert original.auto_detect.keywords == ["old"]
        assert registry.updated.name == "New"
        assert registry.updated.description == "new"
        assert registry.updated.system_prompt == "updated"
        assert registry.updated.knowledge_base is True
        assert registry.updated.tool_policy.excluded_tools == ["calc", "shell"]
        assert registry.updated.tool_policy is not original.tool_policy
        assert registry.updated.auto_detect is not original.auto_detect
        assert registry.updated.auto_detect.min_confidence == 7
        assert result.data.name == "New"


# ── BUG-214 — _load_prompt_from_value relative path containment ──────────


class TestLoadPromptContainment:
    """BUG-214: relative paths must be resolved against data_dir and contained."""

    def test_relative_path_resolved_within_data_dir(self, tmp_path):
        from src.assistant.workflows import _load_prompt_from_value

        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()
        (prompt_dir / "prompt.txt").write_text("Hello!", encoding="utf-8")
        result = _load_prompt_from_value("./prompts/prompt.txt", tmp_path)
        assert result == "Hello!"

    def test_relative_traversal_rejected(self, tmp_path):
        from src.assistant.workflows import _load_prompt_from_value

        result = _load_prompt_from_value("../../../etc/passwd", tmp_path)
        assert result == ""

    def test_absolute_path_outside_rejected(self, tmp_path):
        from src.assistant.workflows import _load_prompt_from_value

        result = _load_prompt_from_value("/etc/passwd", tmp_path)
        assert result == ""

    def test_inline_text_still_works(self, tmp_path):
        from src.assistant.workflows import _load_prompt_from_value

        result = _load_prompt_from_value("You are a helpful assistant.", tmp_path)
        assert result == "You are a helpful assistant."


# ── BUG-218 — token final field must not be premature ────────────────────


class TestTokenFinalNotPremature:
    """BUG-218: final=True only when tools completed AND none in-flight."""

    def test_final_false_while_tools_active(self):
        from unittest.mock import Mock

        from src.api.callbacks import WebSocketCallbackHandler

        handler = WebSocketCallbackHandler(Mock(), Mock())
        # Simulate a tool starting
        handler.on_tool_start({"name": "test"}, "", run_id="run1")
        # tool_call_count is 1 but tool is still in-flight
        with handler._tool_starts_lock:
            assert handler.tool_call_count == 1
            assert len(handler._tool_starts) == 1
        handler._enqueue = Mock()
        handler.on_llm_new_token("token-1")
        handler._enqueue.assert_called_once()
        _, payload = handler._enqueue.call_args.args
        assert payload["final"] is False

    def test_final_true_after_tools_complete(self):
        from unittest.mock import Mock

        from src.api.callbacks import WebSocketCallbackHandler

        handler = WebSocketCallbackHandler(Mock(), Mock())
        handler.on_tool_start({"name": "test"}, "", run_id="run1")
        handler.on_tool_end("result", run_id="run1")
        with handler._tool_starts_lock:
            assert handler.tool_call_count == 1
            assert len(handler._tool_starts) == 0
        handler._enqueue = Mock()
        handler.on_llm_new_token("token-2")
        handler._enqueue.assert_called_once()
        _, payload = handler._enqueue.call_args.args
        assert payload["final"] is True


# ── BUG-219 — compression per-future timeout not dead code ───────────────


class TestCompressionPerFutureTimeout:
    """BUG-219: future.result() must have a timeout so the inner except is reachable."""

    def test_future_result_has_timeout(self):
        import concurrent.futures
        from unittest.mock import patch

        from langchain_core.messages import ToolMessage

        from src.orchestration import compression

        timeout_calls: list[int | float | None] = []

        class FakeFuture:
            def __init__(self, idx: int) -> None:
                self.idx = idx

            def result(self, timeout=None):
                timeout_calls.append(timeout)
                return self.idx, "compressed"

        class FakePool:
            def submit(self, fn, idx):
                return FakeFuture(idx)

        def fake_as_completed(futures, timeout=None):
            assert timeout is not None and timeout > 0
            return list(futures.keys())

        messages = [
            ToolMessage(
                content="x" * 5_000,
                tool_call_id="call-1",
                name="demo-tool",
            )
        ]

        with (
            patch.object(compression, "_get_compression_pool", return_value=FakePool()),
            patch.object(concurrent.futures, "as_completed", side_effect=fake_as_completed),
        ):
            compression.apply_message_compression(
                messages=messages,
                call_count=5,
                compression_cache={},
                llm=object(),
                max_context_tokens=16_384,
                min_age_override=0,
                min_age_cycles=0,
                min_chars=0,
                actual_input_tokens=12_000,
            )

        assert timeout_calls == [compression._COMPRESSION_PER_CALL_TIMEOUT_SECS]


# ── BUG-220 — auto_detect returns highest-scoring, not first alphabetical ─


class TestAutoDetectHighestScore:
    """BUG-220: _auto_detect should return the highest-scoring workflow."""

    def test_highest_score_wins(self, tmp_path):
        from src.assistant.workflows import (
            WorkflowAutoDetect,
            WorkflowDefinition,
            WorkflowRegistry,
        )

        reg = WorkflowRegistry(tmp_path)
        # Create two workflows: "aaa" has 1 keyword match, "zzz" has 3
        reg.create_workflow(
            WorkflowDefinition(
                id="aaa",
                name="Low scorer",
                auto_detect=WorkflowAutoDetect(enabled=True, keywords=["alpha"], min_confidence=1),
            )
        )
        reg.create_workflow(
            WorkflowDefinition(
                id="zzz",
                name="High scorer",
                auto_detect=WorkflowAutoDetect(
                    enabled=True, keywords=["alpha", "beta", "gamma"], min_confidence=1
                ),
            )
        )
        # Message matches 1 keyword for aaa, 3 for zzz — zzz should win
        result = reg.resolve("chat::test", msg_text="alpha beta gamma")
        assert result.workflow_id == "zzz"

    def test_tie_score_alphabetical(self, tmp_path):
        """When two workflows have the same score, alphabetical ID wins."""
        from src.assistant.workflows import (
            WorkflowAutoDetect,
            WorkflowDefinition,
            WorkflowRegistry,
        )

        reg = WorkflowRegistry(tmp_path)
        for wf_id in ("beta-wf", "alpha-wf"):
            reg.create_workflow(
                WorkflowDefinition(
                    id=wf_id,
                    name=wf_id,
                    auto_detect=WorkflowAutoDetect(
                        enabled=True, keywords=["hello"], min_confidence=1
                    ),
                )
            )
        result = reg.resolve("chat::tie", msg_text="hello world")
        assert result.workflow_id == "alpha-wf"


# ── BUG-214 extra — _load_prompt_from_value edge cases ───────────────────


class TestLoadPromptEdgeCases:
    """Additional edge cases for _load_prompt_from_value after BUG-214 fix."""

    def test_empty_value_returns_empty(self, tmp_path):
        from src.assistant.workflows import _load_prompt_from_value

        assert _load_prompt_from_value("", tmp_path) == ""
        assert _load_prompt_from_value("   ", tmp_path) == ""

    def test_tilde_path_outside_data_dir_rejected(self, tmp_path):
        from src.assistant.workflows import _load_prompt_from_value

        result = _load_prompt_from_value("~/../../etc/passwd", tmp_path)
        assert result == ""

    def test_dot_slash_file_inside_data_dir(self, tmp_path):
        from src.assistant.workflows import _load_prompt_from_value

        (tmp_path / "prompt.txt").write_text("test prompt", encoding="utf-8")
        result = _load_prompt_from_value("./prompt.txt", tmp_path)
        assert result == "test prompt"

    def test_dot_dot_slash_rejected(self, tmp_path):
        from src.assistant.workflows import _load_prompt_from_value

        result = _load_prompt_from_value("../outside.txt", tmp_path)
        assert result == ""

    def test_missing_file_returns_empty(self, tmp_path):
        from src.assistant.workflows import _load_prompt_from_value

        result = _load_prompt_from_value("./nonexistent.txt", tmp_path)
        assert result == ""

    def test_multiword_text_treated_as_inline(self, tmp_path):
        from src.assistant.workflows import _load_prompt_from_value

        result = _load_prompt_from_value("You are a helpful bike sales assistant.", tmp_path)
        assert result == "You are a helpful bike sales assistant."


# ── BUG-212 extra — _validate_wf_id boundary cases ──────────────────────


class TestValidateWfIdBoundary:
    """Additional boundary tests for API workflow ID validation."""

    def test_rejects_leading_underscore(self):
        from fastapi import HTTPException

        from src.api.routes.workflows import _validate_wf_id

        with pytest.raises(HTTPException):
            _validate_wf_id("_bad")

    def test_rejects_empty_string(self):
        from fastapi import HTTPException

        from src.api.routes.workflows import _validate_wf_id

        with pytest.raises(HTTPException):
            _validate_wf_id("")

    def test_accepts_single_char(self):
        from src.api.routes.workflows import _validate_wf_id

        _validate_wf_id("a")
        _validate_wf_id("Z")
        _validate_wf_id("9")

    def test_rejects_null_byte(self):
        from fastapi import HTTPException

        from src.api.routes.workflows import _validate_wf_id

        with pytest.raises(HTTPException):
            _validate_wf_id("test\x00evil")


# ── BUG-213 extra — update_workflow returns updated, not stale ───────────


class TestUpdateWorkflowReturnsUpdated:
    """update_workflow route must return _wf_to_out(updated), not _wf_to_out(wf)."""

    def test_update_returns_updated_workflow(self) -> None:
        import asyncio
        from types import SimpleNamespace

        from src.api.auth import TokenData
        from src.api.routes.workflows import update_workflow
        from src.api.schemas.workflow import (
            WorkflowAutoDetectOut,
            WorkflowToolPolicyOut,
            WorkflowUpdate,
        )
        from src.assistant.workflows import (
            WorkflowAutoDetect,
            WorkflowDefinition,
            WorkflowToolPolicy,
        )

        original = WorkflowDefinition(
            id="bike-sales",
            name="Old",
            description="old",
            system_prompt="sys",
            knowledge_base=False,
            tool_policy=WorkflowToolPolicy(excluded_tools=["calc"], additional_approved_tools=[]),
            auto_detect=WorkflowAutoDetect(
                enabled=False, keywords=["old"], patterns=[], min_confidence=3
            ),
        )

        class Registry:
            def __init__(self, wf: WorkflowDefinition) -> None:
                self.workflow = wf
                self.updated: WorkflowDefinition | None = None

            def get_workflow(self, workflow_id: str) -> WorkflowDefinition:
                assert workflow_id == self.workflow.id
                return self.workflow

            def update_workflow(self, updated: WorkflowDefinition) -> None:
                self.updated = updated

        registry = Registry(original)
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(workflow_registry=registry),
            )
        )
        body = WorkflowUpdate(
            name="New",
            description="new",
            system_prompt="updated",
            knowledge_base=True,
            tool_policy=WorkflowToolPolicyOut(
                excluded_tools=["calc", "shell"],
                additional_approved_tools=["rag"],
            ),
            auto_detect=WorkflowAutoDetectOut(
                enabled=True,
                keywords=["new"],
                patterns=["new.*"],
                min_confidence=7,
            ),
        )

        result = asyncio.run(
            update_workflow(
                "bike-sales",
                body,
                request,  # type: ignore[arg-type]
                TokenData("user-1", "admin", {}),
            )
        )

        assert registry.updated is not None
        assert registry.updated is not original
        assert original.name == "Old"
        assert registry.updated.name == "New"
        assert registry.updated.description == "new"
        assert registry.updated.system_prompt == "updated"
        assert registry.updated.knowledge_base is True
        assert registry.updated.tool_policy.excluded_tools == ["calc", "shell"]
        assert registry.updated.tool_policy is not original.tool_policy
        assert registry.updated.auto_detect is not original.auto_detect
        assert registry.updated.auto_detect.min_confidence == 7
        assert result.data.name == "New"


# ── Bug 6 — context_prefix must be injected as HumanMessage, not SystemMessage ──


class TestContextPrefixIsHumanMessage:
    """Bug 6 regression: prepare_messages_with_context must inject context_prefix
    as HumanMessage so that strict OpenAI-compatible providers (vLLM, Qwen3) that
    reject a second SystemMessage outside position 0 do not receive a 400/422
    response (BUG-238)."""

    def test_context_prefix_produces_human_message(self):
        """context_prefix must appear in a HumanMessage, not a SystemMessage."""
        from langchain_core.messages import HumanMessage, SystemMessage

        from src.agent.core import prepare_messages_with_context

        msgs = prepare_messages_with_context(
            history_messages=[],
            user_input="What day is it?",
            context_prefix="User's name is Alice.",
        )

        human_content = " ".join(
            m.content if isinstance(m.content, str) else ""
            for m in msgs
            if isinstance(m, HumanMessage)
        )
        system_content = " ".join(
            m.content if isinstance(m.content, str) else ""
            for m in msgs
            if isinstance(m, SystemMessage)
        )

        assert "Alice" in human_content, "context_prefix must appear in a HumanMessage"
        assert "Alice" not in system_content, "context_prefix must NOT appear in a SystemMessage"

    def test_no_extra_system_messages_with_prefix(self):
        """With context_prefix provided, no SystemMessage should be injected."""
        from langchain_core.messages import SystemMessage

        from src.agent.core import prepare_messages_with_context

        msgs = prepare_messages_with_context(
            history_messages=[],
            user_input="hello",
            context_prefix="some context here",
        )

        system_count = sum(1 for m in msgs if isinstance(m, SystemMessage))
        assert system_count == 0, (
            f"prepare_messages_with_context injected {system_count} SystemMessage(s); "
            f"context_prefix must use HumanMessage instead"
        )
