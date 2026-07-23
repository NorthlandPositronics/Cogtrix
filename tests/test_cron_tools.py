"""Tests for cogtrix_core/tools/cron_tools.py."""

from __future__ import annotations

import json
import pathlib
import threading
import time

import pytest


@pytest.fixture()
def data_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path / "cron"


@pytest.fixture(autouse=True)
def reset_cron_module():
    """Reset module-level singletons between tests."""
    import cogtrix_core.tools.cron_tools as _mod

    orig_scheduler = _mod._scheduler
    orig_factory = _mod._llm_factory
    orig_job_runner = _mod._job_runner
    orig_data_dir = _mod._data_dir
    orig_timeout = _mod._cron_llm_timeout_seconds

    yield

    if _mod._scheduler is not None and _mod._scheduler is not orig_scheduler:
        _mod._scheduler.stop()
    _mod._scheduler = orig_scheduler
    _mod._llm_factory = orig_factory
    _mod._job_runner = orig_job_runner
    _mod._data_dir = orig_data_dir
    _mod._cron_llm_timeout_seconds = orig_timeout


# ── CronJob ───────────────────────────────────────────────────────────────────


class TestCronJob:
    def test_to_dict_roundtrip(self):
        from cogtrix_core.tools.cron_tools import CronJob

        job = CronJob(
            id="abc12345",
            name="test",
            schedule="*/5 * * * *",
            prompt="hello",
            created_at=1000.0,
            context="inherit",
            last_run=2000.0,
            next_run=3000.0,
            run_count=7,
        )
        d = job.to_dict()
        job2 = CronJob.from_dict(d)
        assert job2.id == "abc12345"
        assert job2.schedule == "*/5 * * * *"
        assert job2.run_count == 7
        assert job2.last_run == 2000.0
        assert job2.context == "inherit"

    def test_from_dict_defaults(self):
        from cogtrix_core.tools.cron_tools import CronJob

        job = CronJob.from_dict({"id": "x1", "schedule": "0 * * * *", "prompt": "hi"})
        assert job.run_count == 0
        assert job.last_run is None
        assert job.next_run is None
        assert job.context == "fresh"

    def test_next_run_human_none(self):
        from cogtrix_core.tools.cron_tools import CronJob

        job = CronJob(id="z", name="z", schedule="*", prompt="p", created_at=0.0)
        assert job.next_run_human() == "unknown"

    def test_next_run_human_timestamp(self):
        from cogtrix_core.tools.cron_tools import CronJob

        job = CronJob(id="z", name="z", schedule="*", prompt="p", created_at=0.0, next_run=0.0)
        label = job.next_run_human()
        assert "1970" in label
        assert "UTC" in label


# ── CronScheduler ─────────────────────────────────────────────────────────────


class TestCronScheduler:
    def test_add_valid_schedule(self, data_dir):
        from cogtrix_core.tools.cron_tools import CronScheduler

        s = CronScheduler(data_dir)
        job, _ = s.add("*/5 * * * *", "ping", name="my-job", context="inherit")
        assert job.id
        assert job.schedule == "*/5 * * * *"
        assert job.next_run is not None
        assert job.next_run > time.time() - 1
        assert job.context == "inherit"

    def test_add_invalid_schedule_raises(self, data_dir):
        from cogtrix_core.tools.cron_tools import CronScheduler

        s = CronScheduler(data_dir)
        with pytest.raises(ValueError, match="Invalid cron expression"):
            s.add("not-a-cron", "ping")

    def test_add_persists_to_json(self, data_dir):
        from cogtrix_core.tools.cron_tools import CronScheduler

        s = CronScheduler(data_dir)
        s.add("0 9 * * *", "morning check")
        path = data_dir / "jobs.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data["jobs"]) == 1

    def test_remove_existing_job(self, data_dir):
        from cogtrix_core.tools.cron_tools import CronScheduler

        s = CronScheduler(data_dir)
        job, _ = s.add("0 * * * *", "hourly")
        s.remove(job.id)
        assert not s.list_jobs()

    def test_remove_unknown_job_raises(self, data_dir):
        from cogtrix_core.tools.cron_tools import CronScheduler

        s = CronScheduler(data_dir)
        with pytest.raises(KeyError, match="no-such-id"):
            s.remove("no-such-id")

    def test_list_includes_next_run_human(self, data_dir):
        from cogtrix_core.tools.cron_tools import CronScheduler

        s = CronScheduler(data_dir)
        s.add("0 9 * * *", "check", name="daily")
        jobs = s.list_jobs()
        assert len(jobs) == 1
        assert "next_run_human" in jobs[0]
        assert "UTC" in jobs[0]["next_run_human"]

    def test_persistence_survives_reload(self, data_dir):
        from cogtrix_core.tools.cron_tools import CronScheduler

        s1 = CronScheduler(data_dir)
        job, _ = s1.add("*/10 * * * *", "reload-test")
        s2 = CronScheduler(data_dir)
        loaded = s2.list_jobs()
        assert any(j["id"] == job.id for j in loaded)

    def test_fire_calls_llm_factory(self, data_dir):
        from cogtrix_core.tools.cron_tools import CronScheduler

        called = []

        class FakeLLM:
            def invoke(self, msgs):
                called.append(msgs[0].content)

                class R:
                    content = "ok"

                return R()

        import cogtrix_core.tools.cron_tools as _mod

        _mod._llm_factory = lambda: FakeLLM()

        s = CronScheduler(data_dir)
        job, _ = s.add("*/5 * * * *", "test-prompt")
        # Manually trigger fire
        s._fire(job, time.time())
        assert called == ["test-prompt"]

    def test_fire_inherit_uses_runner(self, data_dir):
        from cogtrix_core.tools.cron_tools import CronScheduler

        called: list[str] = []

        import cogtrix_core.tools.cron_tools as _mod

        _mod._llm_factory = None
        _mod._job_runner = lambda job: called.append(job.prompt) or "inherited-ok"

        s = CronScheduler(data_dir)
        job, _ = s.add("*/5 * * * *", "inherit-prompt", context="inherit")
        s._fire(job, time.time())
        assert called == ["inherit-prompt"]

    def test_fire_no_factory_logs_warning(self, data_dir, caplog):
        import logging

        import cogtrix_core.tools.cron_tools as _mod
        from cogtrix_core.tools.cron_tools import CronScheduler

        _mod._llm_factory = None
        s = CronScheduler(data_dir)
        job, _ = s.add("*/5 * * * *", "orphaned")
        with caplog.at_level(logging.WARNING, logger="cogtrix.tools.cron"):
            s._fire(job, time.time())
        assert any("no LLM factory" in r.message for r in caplog.records)

    def test_fire_no_humanmessage_logs_warning(self, data_dir, caplog):
        """When langchain_core is unavailable, _invoke_llm should log a warning
        rather than raising AssertionError swallowed by the except clause."""
        import logging

        import cogtrix_core.tools.cron_tools as _mod
        from cogtrix_core.tools.cron_tools import CronScheduler

        _mod._llm_factory = lambda: object()  # factory is set
        original = _mod._HumanMessage
        try:
            _mod._HumanMessage = None  # simulate missing langchain_core
            s = CronScheduler(data_dir)
            job, _ = s.add("*/5 * * * *", "lc-missing")
            with caplog.at_level(logging.WARNING, logger="cogtrix.tools.cron"):
                s._invoke_llm(job)
        finally:
            _mod._HumanMessage = original
        assert any("langchain_core" in r.message for r in caplog.records)

    def test_fire_updates_run_count(self, data_dir):
        import cogtrix_core.tools.cron_tools as _mod
        from cogtrix_core.tools.cron_tools import CronScheduler

        class FakeLLM:
            def invoke(self, *_):
                class R:
                    content = "done"

                return R()

        _mod._llm_factory = lambda: FakeLLM()
        s = CronScheduler(data_dir)
        job, _ = s.add("*/5 * * * *", "count-me")
        assert job.run_count == 0
        s._fire(job, time.time())
        assert job.run_count == 1

    def test_fire_logs_without_printing_to_stdout(self, data_dir, caplog, capsys):
        import logging

        import cogtrix_core.tools.cron_tools as _mod
        from cogtrix_core.tools.cron_tools import CronScheduler

        class FakeLLM:
            def invoke(self, *_):
                class R:
                    content = "cron-result"

                return R()

        _mod._llm_factory = lambda: FakeLLM()
        s = CronScheduler(data_dir)
        job, _ = s.add("*/5 * * * *", "quiet-cron")

        with caplog.at_level(logging.INFO, logger="cogtrix.tools.cron"):
            s._fire(job, time.time())

        captured = capsys.readouterr()
        assert captured.out == ""
        assert "quiet-cron" in caplog.text
        assert "cron-result" in caplog.text

    def test_fire_updates_next_run(self, data_dir):
        import cogtrix_core.tools.cron_tools as _mod
        from cogtrix_core.tools.cron_tools import CronScheduler

        _mod._llm_factory = None  # no LLM needed for this check
        s = CronScheduler(data_dir)
        job, _ = s.add("*/5 * * * *", "next-check")
        first_next = job.next_run
        # Fire at first_next (the scheduled time) so get_next returns the subsequent interval
        s._fire(job, first_next)  # type: ignore[arg-type]
        assert job.next_run is not None
        assert job.next_run > first_next  # type: ignore[operator]

    def test_tick_fires_due_jobs(self, data_dir):
        import cogtrix_core.tools.cron_tools as _mod
        from cogtrix_core.tools.cron_tools import CronScheduler

        fired: list[str] = []

        class FakeLLM:
            def invoke(self, msgs):
                fired.append(msgs[0].content)

                class R:
                    content = "ok"

                return R()

        _mod._llm_factory = lambda: FakeLLM()
        s = CronScheduler(data_dir)
        job, _ = s.add("*/5 * * * *", "tick-test")
        # Force next_run into the past
        with s._lock:
            job.next_run = time.time() - 1
        s._tick()
        assert fired == ["tick-test"]

    def test_tick_skips_future_jobs(self, data_dir):
        import cogtrix_core.tools.cron_tools as _mod
        from cogtrix_core.tools.cron_tools import CronScheduler

        fired: list[str] = []

        class FakeLLM:
            def invoke(self, msgs):
                fired.append(msgs[0].content)

                class R:
                    content = "ok"

                return R()

        _mod._llm_factory = lambda: FakeLLM()
        s = CronScheduler(data_dir)
        s.add("*/5 * * * *", "future-job")  # next_run is in the future
        s._tick()
        assert not fired

    def test_start_stop(self, data_dir):
        from cogtrix_core.tools.cron_tools import CronScheduler

        s = CronScheduler(data_dir)
        s.start()
        assert s._thread is not None
        assert s._thread.is_alive()
        s.stop()

    def test_start_is_idempotent(self, data_dir):
        from cogtrix_core.tools.cron_tools import CronScheduler

        s = CronScheduler(data_dir)
        s.start()
        t1 = s._thread
        s.start()  # should not create a second thread
        assert s._thread is t1
        s.stop()

    def test_malformed_json_is_skipped_on_load(self, data_dir):
        from cogtrix_core.tools.cron_tools import CronScheduler

        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "jobs.json").write_text(
            json.dumps(
                {"jobs": [{"bad": "entry"}, {"id": "ok1", "schedule": "0 * * * *", "prompt": "p"}]}
            ),
            encoding="utf-8",
        )
        s = CronScheduler(data_dir)
        # The malformed entry is skipped; the valid one is loaded
        assert len(s.list_jobs()) == 1

    def test_missing_json_file_does_not_crash(self, data_dir):
        from cogtrix_core.tools.cron_tools import CronScheduler

        # data_dir doesn't exist yet — should not raise
        s = CronScheduler(data_dir)
        assert s.list_jobs() == []

    def test_save_uses_atomic_write(self, data_dir):
        """_save() must use atomic_write_json so a crash during write leaves the
        previous file intact rather than producing a truncated/empty file.

        We verify indirectly: after add(), the jobs.json file must exist and contain
        valid JSON — if the atomic helper were bypassed and the process killed mid-write
        the file could be empty or truncated; this at minimum confirms the happy path
        round-trips through a real write call.
        """
        from cogtrix_core.tools.cron_tools import CronScheduler

        s = CronScheduler(data_dir)
        s.add("0 9 * * *", "atomic-test")
        path = data_dir / "jobs.json"
        assert path.exists(), "jobs.json must exist after add()"
        raw = path.read_text(encoding="utf-8")
        # Must be complete JSON — truncated atomic writes leave the old file intact
        parsed = json.loads(raw)
        assert len(parsed["jobs"]) == 1
        # No leftover .tmp file from the atomic write
        tmp_files = list(data_dir.glob("*.tmp"))
        assert not tmp_files, f"Leftover temp files found: {tmp_files}"

    def test_tick_continues_after_llm_timeout(self, data_dir, caplog, monkeypatch):
        """A hung LLM call must time out so the scheduler can fire other jobs."""
        import logging

        import cogtrix_core.tools.cron_tools as _mod
        from cogtrix_core.tools.cron_tools import CronScheduler

        release = threading.Event()
        outputs: list[str] = []
        factory_calls: list[str] = []

        class SlowLLM:
            def invoke(self, msgs):
                release.wait(0.2)

                class R:
                    content = "slow"

                return R()

        class FastLLM:
            def invoke(self, msgs):
                class R:
                    content = "fast"

                return R()

        def factory():
            factory_calls.append("call")
            return SlowLLM() if len(factory_calls) == 1 else FastLLM()

        monkeypatch.setattr(_mod, "_llm_factory", factory)
        monkeypatch.setattr(_mod, "_cron_llm_timeout_seconds", 0.01)
        monkeypatch.setattr(
            _mod,
            "_print_cron_output",
            lambda job_name, prompt, response: outputs.append(response),
        )

        s = CronScheduler(data_dir)
        slow_job, _ = s.add("*/5 * * * *", "slow-job", name="slow")
        fast_job, _ = s.add("*/5 * * * *", "fast-job", name="fast")
        with s._lock:
            slow_job.next_run = time.time() - 1
            fast_job.next_run = time.time() - 1

        with caplog.at_level(logging.WARNING, logger="cogtrix.tools.cron"):
            s._tick()

        release.set()

        assert any("timed out" in record.message for record in caplog.records)
        timeout_label = f"{_mod._cron_llm_timeout_seconds:g}"
        assert outputs == [
            f"[LLM timeout after {timeout_label}s — will retry]",
            "fast",
        ]

    def test_tick_continues_after_llm_exception(self, data_dir, caplog, monkeypatch):
        """An LLM that raises RuntimeError must be caught so the scheduler can fire other jobs."""
        import logging

        import cogtrix_core.tools.cron_tools as _mod
        from cogtrix_core.tools.cron_tools import CronScheduler

        outputs: list[str] = []

        class BrokenLLM:
            def invoke(self, msgs):
                raise RuntimeError("provider down")

        class OKLLM:
            def invoke(self, msgs):

                class R:
                    content = "ok"

                return R()

        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return BrokenLLM() if call_count == 1 else OKLLM()

        monkeypatch.setattr(_mod, "_llm_factory", factory)
        monkeypatch.setattr(
            _mod,
            "_print_cron_output",
            lambda job_name, prompt, response: outputs.append(response),
        )

        s = CronScheduler(data_dir)
        broken_job, _ = s.add("*/5 * * * *", "broken-job", name="broken")
        ok_job, _ = s.add("*/5 * * * *", "ok-job", name="ok")
        with s._lock:
            broken_job.next_run = time.time() - 1
            ok_job.next_run = time.time() - 1

        with caplog.at_level(logging.WARNING, logger="cogtrix.tools.cron"):
            s._tick()

        assert any("LLM call failed" in record.message for record in caplog.records)
        assert outputs == [
            "[LLM error: provider down]",
            "ok",
        ]


# ── Tool functions ─────────────────────────────────────────────────────────────


class TestCronToolFunctions:
    @pytest.fixture(autouse=True)
    def _cron_tool_data_dir(self, tmp_path: pathlib.Path):
        """Provide an isolated data directory for each tool-function test.

        Uses a class-level autouse fixture so it runs *after* the module-level
        ``reset_cron_module`` fixture saves state and *before* that fixture
        restores it, preventing stale state from leaking between test classes.
        """
        import cogtrix_core.tools.cron_tools as _mod

        self._tmp = tmp_path / "cron"
        self._tmp.mkdir()
        _mod._data_dir = self._tmp
        yield
        # Module-level reset_cron_module fixture handles state restoration.

    def test_cron_add_returns_id_and_next_run(self):
        from cogtrix_core.tools.cron_tools import cron_add

        result = cron_add("*/5 * * * *", "test prompt", name="my-job", context="inherit")
        assert "ID:" in result
        assert "my-job" in result
        assert "inherit" in result
        assert "Next run:" in result

    def test_cron_add_default_name_is_schedule(self):
        from cogtrix_core.tools.cron_tools import cron_add

        result = cron_add("0 9 * * *", "morning")
        assert "0 9 * * *" in result  # used as name

    def test_cron_add_invalid_schedule_returns_error(self):
        from cogtrix_core.tools.cron_tools import cron_add

        result = cron_add("bad-schedule", "test")
        assert result.startswith("Error:")

    def test_cron_list_empty(self):
        from cogtrix_core.tools.cron_tools import cron_list

        result = cron_list()
        assert "No cron jobs" in result

    def test_cron_list_shows_added_job(self):
        from cogtrix_core.tools.cron_tools import cron_add, cron_list

        cron_add("0 * * * *", "hourly prompt", name="hourly", context="inherit")
        result = cron_list()
        assert "hourly" in result
        assert "0 * * * *" in result
        assert "hourly prompt" in result
        assert "inherit" in result

    def test_cron_remove_existing(self):
        from cogtrix_core.tools.cron_tools import cron_add, cron_list, cron_remove

        result = cron_add("*/30 * * * *", "half-hourly")
        job_id = result.split("ID:")[1].split()[0].strip()
        remove_result = cron_remove(job_id)
        assert job_id in remove_result
        assert "No cron jobs" in cron_list()

    def test_cron_remove_unknown_returns_error(self):
        from cogtrix_core.tools.cron_tools import cron_remove

        result = cron_remove("nonexistent")
        assert "No cron job found" in result

    def test_cron_list_truncates_long_prompt(self):
        from cogtrix_core.tools.cron_tools import cron_add, cron_list

        long_prompt = "x" * 200
        cron_add("0 * * * *", long_prompt)
        result = cron_list()
        assert "…" in result

    def test_six_field_cron_expression(self):
        """6-field (seconds) cron should be accepted by croniter."""
        from cogtrix_core.tools.cron_tools import cron_add

        result = cron_add("0 */5 * * * *", "every 5 min by seconds")
        assert "Error:" not in result

    def test_cron_add_multiple_jobs(self):
        from cogtrix_core.tools.cron_tools import cron_add, cron_list

        cron_add("0 9 * * *", "morning", name="morning")
        cron_add("0 18 * * *", "evening", name="evening", context="inherit")
        result = cron_list()
        assert "morning" in result
        assert "evening" in result


# ── configure_cron ────────────────────────────────────────────────────────────


class TestConfigureCron:
    def test_configure_cron_sets_factory(self, tmp_path):
        import cogtrix_core.tools.cron_tools as _mod

        _mod._scheduler = None
        _mod._llm_factory = None

        factory_called = []

        def my_factory():
            factory_called.append(True)

        from cogtrix_core.tools.cron_tools import configure_cron

        configure_cron(data_dir=str(tmp_path / "cron"), llm_factory=my_factory)
        assert _mod._llm_factory is my_factory
        assert _mod._scheduler is not None
        _mod._scheduler.stop()

    def test_configure_cron_seeds_initial_jobs(self, tmp_path):
        import cogtrix_core.tools.cron_tools as _mod

        _mod._scheduler = None
        _mod._llm_factory = None
        _mod._job_runner = None

        from cogtrix_core.tools.cron_tools import configure_cron

        configure_cron(
            data_dir=str(tmp_path / "cron"),
            initial_jobs=[
                {
                    "name": "nightly",
                    "schedule": "0 2 * * *",
                    "prompt": "check status",
                    "context": "inherit",
                }
            ],
        )
        assert _mod._scheduler is not None
        jobs = _mod._scheduler.list_jobs()
        assert len(jobs) == 1
        assert jobs[0]["context"] == "inherit"
        _mod._scheduler.stop()

    def test_configure_cron_twice_reuses_scheduler(self, tmp_path):
        import cogtrix_core.tools.cron_tools as _mod

        _mod._scheduler = None
        from cogtrix_core.tools.cron_tools import configure_cron

        configure_cron(data_dir=str(tmp_path / "cron"))
        first = _mod._scheduler
        configure_cron(llm_factory=lambda: None)  # second call
        assert _mod._scheduler is first  # same instance
        assert _mod._scheduler is not None
        _mod._scheduler.stop()


# ── Tenant isolation (#424) ────────────────────────────────────────────────────


class TestCronTenantIsolation:
    """Regression tests for #424 — cron operations must be scoped to owner_id."""

    def _sched(self, tmp_path: pathlib.Path):
        from cogtrix_core.tools.cron_tools import CronScheduler

        s = CronScheduler(tmp_path)
        s.start()
        return s

    def test_list_only_returns_own_jobs(self, tmp_path: pathlib.Path) -> None:
        s = self._sched(tmp_path)
        s.add("* * * * *", "ping", "job-a", owner_id="session-A")
        s.add("* * * * *", "ping", "job-b", owner_id="session-B")
        s.stop()

        a_jobs = s.list_jobs(owner_id="session-A")
        b_jobs = s.list_jobs(owner_id="session-B")
        assert len(a_jobs) == 1 and a_jobs[0]["name"] == "job-a"
        assert len(b_jobs) == 1 and b_jobs[0]["name"] == "job-b"

    def test_remove_cross_tenant_raises_permission_error(self, tmp_path: pathlib.Path) -> None:
        s = self._sched(tmp_path)
        job, _ = s.add("* * * * *", "ping", owner_id="session-A")
        s.stop()

        with pytest.raises(PermissionError):
            s.remove(job.id, owner_id="session-B")

    def test_remove_own_job_succeeds(self, tmp_path: pathlib.Path) -> None:
        s = self._sched(tmp_path)
        job, _ = s.add("* * * * *", "ping", owner_id="session-A")
        s.stop()
        s.remove(job.id, owner_id="session-A")
        assert job.id not in {j["id"] for j in s.list_jobs(owner_id="session-A")}

    def test_session_id_context_var_scopes_tool_functions(self, tmp_path: pathlib.Path) -> None:
        import cogtrix_core.tools.cron_tools as _mod
        from cogtrix_core.tools.cron_tools import cron_add, cron_list, set_cron_session_id

        orig = _mod._scheduler
        _mod._scheduler = self._sched(tmp_path)
        try:
            set_cron_session_id("sess-X")
            cron_add("* * * * *", "hello", "job-x")
            set_cron_session_id("sess-Y")
            cron_add("* * * * *", "world", "job-y")

            set_cron_session_id("sess-X")
            listing = cron_list()
            assert "job-x" in listing
            assert "job-y" not in listing
        finally:
            _mod._scheduler.stop()
            _mod._scheduler = orig
            set_cron_session_id("")

    def test_owner_id_persisted_and_restored(self, tmp_path: pathlib.Path) -> None:
        from cogtrix_core.tools.cron_tools import CronScheduler

        s1 = CronScheduler(tmp_path)
        s1.start()
        job, _ = s1.add("* * * * *", "ping", owner_id="tenant-Z")
        s1.stop()

        s2 = CronScheduler(tmp_path)
        jobs = s2.list_jobs(owner_id="tenant-Z")
        assert len(jobs) == 1 and jobs[0]["id"] == job.id
        assert jobs[0]["owner_id"] == "tenant-Z"
