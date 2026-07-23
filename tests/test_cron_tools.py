"""Tests for src/tools/cron_tools.py."""

from __future__ import annotations

import json
import pathlib
import tempfile
import time

import pytest


@pytest.fixture()
def data_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path / "cron"


@pytest.fixture(autouse=True)
def reset_cron_module():
    """Reset module-level singletons between tests."""
    import src.tools.cron_tools as _mod

    orig_scheduler = _mod._scheduler
    orig_factory = _mod._llm_factory
    orig_data_dir = _mod._data_dir

    yield

    if _mod._scheduler is not None and _mod._scheduler is not orig_scheduler:
        _mod._scheduler.stop()
    _mod._scheduler = orig_scheduler
    _mod._llm_factory = orig_factory
    _mod._data_dir = orig_data_dir


# ── CronJob ───────────────────────────────────────────────────────────────────


class TestCronJob:
    def test_to_dict_roundtrip(self):
        from src.tools.cron_tools import CronJob

        job = CronJob(
            id="abc12345",
            name="test",
            schedule="*/5 * * * *",
            prompt="hello",
            created_at=1000.0,
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

    def test_from_dict_defaults(self):
        from src.tools.cron_tools import CronJob

        job = CronJob.from_dict({"id": "x1", "schedule": "0 * * * *", "prompt": "hi"})
        assert job.run_count == 0
        assert job.last_run is None
        assert job.next_run is None

    def test_next_run_human_none(self):
        from src.tools.cron_tools import CronJob

        job = CronJob(id="z", name="z", schedule="*", prompt="p", created_at=0.0)
        assert job.next_run_human() == "unknown"

    def test_next_run_human_timestamp(self):
        from src.tools.cron_tools import CronJob

        job = CronJob(id="z", name="z", schedule="*", prompt="p", created_at=0.0, next_run=0.0)
        label = job.next_run_human()
        assert "1970" in label
        assert "UTC" in label


# ── CronScheduler ─────────────────────────────────────────────────────────────


class TestCronScheduler:
    def test_add_valid_schedule(self, data_dir):
        from src.tools.cron_tools import CronScheduler

        s = CronScheduler(data_dir)
        job = s.add("*/5 * * * *", "ping", name="my-job")
        assert job.id
        assert job.schedule == "*/5 * * * *"
        assert job.next_run is not None
        assert job.next_run > time.time() - 1

    def test_add_invalid_schedule_raises(self, data_dir):
        from src.tools.cron_tools import CronScheduler

        s = CronScheduler(data_dir)
        with pytest.raises(ValueError, match="Invalid cron expression"):
            s.add("not-a-cron", "ping")

    def test_add_persists_to_json(self, data_dir):
        from src.tools.cron_tools import CronScheduler

        s = CronScheduler(data_dir)
        s.add("0 9 * * *", "morning check")
        path = data_dir / "jobs.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data["jobs"]) == 1

    def test_remove_existing_job(self, data_dir):
        from src.tools.cron_tools import CronScheduler

        s = CronScheduler(data_dir)
        job = s.add("0 * * * *", "hourly")
        s.remove(job.id)
        assert not s.list_jobs()

    def test_remove_unknown_job_raises(self, data_dir):
        from src.tools.cron_tools import CronScheduler

        s = CronScheduler(data_dir)
        with pytest.raises(KeyError, match="no-such-id"):
            s.remove("no-such-id")

    def test_list_includes_next_run_human(self, data_dir):
        from src.tools.cron_tools import CronScheduler

        s = CronScheduler(data_dir)
        s.add("0 9 * * *", "check", name="daily")
        jobs = s.list_jobs()
        assert len(jobs) == 1
        assert "next_run_human" in jobs[0]
        assert "UTC" in jobs[0]["next_run_human"]

    def test_persistence_survives_reload(self, data_dir):
        from src.tools.cron_tools import CronScheduler

        s1 = CronScheduler(data_dir)
        job = s1.add("*/10 * * * *", "reload-test")
        s2 = CronScheduler(data_dir)
        loaded = s2.list_jobs()
        assert any(j["id"] == job.id for j in loaded)

    def test_fire_calls_llm_factory(self, data_dir):
        from src.tools.cron_tools import CronScheduler

        called = []

        class FakeLLM:
            def invoke(self, msgs):
                called.append(msgs[0].content)

                class R:
                    content = "ok"

                return R()

        import src.tools.cron_tools as _mod

        _mod._llm_factory = lambda: FakeLLM()

        s = CronScheduler(data_dir)
        job = s.add("*/5 * * * *", "test-prompt")
        # Manually trigger fire
        s._fire(job, time.time())
        assert called == ["test-prompt"]

    def test_fire_no_factory_logs_warning(self, data_dir, caplog):
        import logging

        import src.tools.cron_tools as _mod
        from src.tools.cron_tools import CronScheduler

        _mod._llm_factory = None
        s = CronScheduler(data_dir)
        job = s.add("*/5 * * * *", "orphaned")
        with caplog.at_level(logging.WARNING, logger="cogtrix.tools.cron"):
            s._fire(job, time.time())
        assert any("no LLM factory" in r.message for r in caplog.records)

    def test_fire_updates_run_count(self, data_dir):
        import src.tools.cron_tools as _mod
        from src.tools.cron_tools import CronScheduler

        class FakeLLM:
            def invoke(self, *_):
                class R:
                    content = "done"

                return R()

        _mod._llm_factory = lambda: FakeLLM()
        s = CronScheduler(data_dir)
        job = s.add("*/5 * * * *", "count-me")
        assert job.run_count == 0
        s._fire(job, time.time())
        assert job.run_count == 1

    def test_fire_updates_next_run(self, data_dir):
        import src.tools.cron_tools as _mod
        from src.tools.cron_tools import CronScheduler

        _mod._llm_factory = None  # no LLM needed for this check
        s = CronScheduler(data_dir)
        job = s.add("*/5 * * * *", "next-check")
        first_next = job.next_run
        # Fire at first_next (the scheduled time) so get_next returns the subsequent interval
        s._fire(job, first_next)  # type: ignore[arg-type]
        assert job.next_run is not None
        assert job.next_run > first_next  # type: ignore[operator]

    def test_tick_fires_due_jobs(self, data_dir):
        import src.tools.cron_tools as _mod
        from src.tools.cron_tools import CronScheduler

        fired: list[str] = []

        class FakeLLM:
            def invoke(self, msgs):
                fired.append(msgs[0].content)

                class R:
                    content = "ok"

                return R()

        _mod._llm_factory = lambda: FakeLLM()
        s = CronScheduler(data_dir)
        job = s.add("*/5 * * * *", "tick-test")
        # Force next_run into the past
        with s._lock:
            job.next_run = time.time() - 1
        s._tick()
        assert fired == ["tick-test"]

    def test_tick_skips_future_jobs(self, data_dir):
        import src.tools.cron_tools as _mod
        from src.tools.cron_tools import CronScheduler

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
        from src.tools.cron_tools import CronScheduler

        s = CronScheduler(data_dir)
        s.start()
        assert s._thread is not None
        assert s._thread.is_alive()
        s.stop()

    def test_start_is_idempotent(self, data_dir):
        from src.tools.cron_tools import CronScheduler

        s = CronScheduler(data_dir)
        s.start()
        t1 = s._thread
        s.start()  # should not create a second thread
        assert s._thread is t1
        s.stop()

    def test_malformed_json_is_skipped_on_load(self, data_dir):
        from src.tools.cron_tools import CronScheduler

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
        from src.tools.cron_tools import CronScheduler

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
        from src.tools.cron_tools import CronScheduler

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


# ── Tool functions ─────────────────────────────────────────────────────────────


class TestCronToolFunctions:
    def setup_method(self):
        import src.tools.cron_tools as _mod

        _mod._scheduler = None
        _mod._llm_factory = None
        self._tmp = tempfile.mkdtemp()
        _mod._data_dir = pathlib.Path(self._tmp)

    def test_cron_add_returns_id_and_next_run(self):
        from src.tools.cron_tools import cron_add

        result = cron_add("*/5 * * * *", "test prompt", name="my-job")
        assert "ID:" in result
        assert "my-job" in result
        assert "Next run:" in result

    def test_cron_add_default_name_is_schedule(self):
        from src.tools.cron_tools import cron_add

        result = cron_add("0 9 * * *", "morning")
        assert "0 9 * * *" in result  # used as name

    def test_cron_add_invalid_schedule_returns_error(self):
        from src.tools.cron_tools import cron_add

        result = cron_add("bad-schedule", "test")
        assert result.startswith("Error:")

    def test_cron_list_empty(self):
        from src.tools.cron_tools import cron_list

        result = cron_list()
        assert "No cron jobs" in result

    def test_cron_list_shows_added_job(self):
        from src.tools.cron_tools import cron_add, cron_list

        cron_add("0 * * * *", "hourly prompt", name="hourly")
        result = cron_list()
        assert "hourly" in result
        assert "0 * * * *" in result
        assert "hourly prompt" in result

    def test_cron_remove_existing(self):
        from src.tools.cron_tools import cron_add, cron_list, cron_remove

        result = cron_add("*/30 * * * *", "half-hourly")
        job_id = result.split("ID:")[1].split()[0].strip()
        remove_result = cron_remove(job_id)
        assert job_id in remove_result
        assert "No cron jobs" in cron_list()

    def test_cron_remove_unknown_returns_error(self):
        from src.tools.cron_tools import cron_remove

        result = cron_remove("nonexistent")
        assert result.startswith("Error:")

    def test_cron_list_truncates_long_prompt(self):
        from src.tools.cron_tools import cron_add, cron_list

        long_prompt = "x" * 200
        cron_add("0 * * * *", long_prompt)
        result = cron_list()
        assert "…" in result

    def test_six_field_cron_expression(self):
        """6-field (seconds) cron should be accepted by croniter."""
        from src.tools.cron_tools import cron_add

        result = cron_add("0 */5 * * * *", "every 5 min by seconds")
        assert "Error:" not in result

    def test_cron_add_multiple_jobs(self):
        from src.tools.cron_tools import cron_add, cron_list

        cron_add("0 9 * * *", "morning", name="morning")
        cron_add("0 18 * * *", "evening", name="evening")
        result = cron_list()
        assert "morning" in result
        assert "evening" in result


# ── configure_cron ────────────────────────────────────────────────────────────


class TestConfigureCron:
    def test_configure_cron_sets_factory(self, tmp_path):
        import src.tools.cron_tools as _mod

        _mod._scheduler = None
        _mod._llm_factory = None

        factory_called = []

        def my_factory():
            factory_called.append(True)

        from src.tools.cron_tools import configure_cron

        configure_cron(data_dir=str(tmp_path / "cron"), llm_factory=my_factory)
        assert _mod._llm_factory is my_factory
        assert _mod._scheduler is not None
        _mod._scheduler.stop()

    def test_configure_cron_twice_reuses_scheduler(self, tmp_path):
        import src.tools.cron_tools as _mod

        _mod._scheduler = None
        from src.tools.cron_tools import configure_cron

        configure_cron(data_dir=str(tmp_path / "cron"))
        first = _mod._scheduler
        configure_cron(llm_factory=lambda: None)  # second call
        assert _mod._scheduler is first  # same instance
        assert _mod._scheduler is not None
        _mod._scheduler.stop()
