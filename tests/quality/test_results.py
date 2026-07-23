"""Tests for persistent quality-harness result storage."""

from __future__ import annotations

import os

from tests.quality import results
from tests.quality.metrics import MetricsReport


def _make_report(**overrides: object) -> MetricsReport:
    base = MetricsReport(
        scenario_id="tool_call_single",
        tool_selection_rate=100.0,
        parameter_name_f1=1.0,
        parameter_value_match_rate=100.0,
        phantom_call_count=0,
        task_completion_rate=100.0,
        orphaned_pair_count=0,
        tool_readiness_violations=0,
        error_recovery_turns=0,
        turns_to_completion=2,
        prompt_tokens_per_task=0,
        post_cutoff_phantom_count=0,
        identical_error_retry_count=0,
        tier1_pass=True,
        tier2_warnings=[],
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_store_run_round_trips_latest_report(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "RESULTS_ROOT", tmp_path / "results")

    report = _make_report()
    path = results.store_run(report, "0.2.6")

    assert path.exists()
    assert path.parent == tmp_path / "results" / "0.2.6"

    loaded = results.load_previous("0.2.6")
    assert loaded == report


def test_load_previous_prefers_most_recent_file(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "RESULTS_ROOT", tmp_path / "results")

    older = results.store_run(_make_report(turns_to_completion=2), "0.2.6")
    newer = results.store_run(_make_report(turns_to_completion=3), "0.2.6")

    older.touch()
    newer.touch()
    older_mtime = 1_000_000_000
    newer_mtime = older_mtime + 10
    os.utime(older, (older_mtime, older_mtime))
    os.utime(newer, (newer_mtime, newer_mtime))

    loaded = results.load_previous("0.2.6")
    assert loaded is not None
    assert loaded.turns_to_completion == 3


def test_compare_reports_detects_regressions():
    previous = _make_report(
        tool_selection_rate=100.0,
        phantom_call_count=0,
        turns_to_completion=2,
        prompt_tokens_per_task=100,
    )
    current = _make_report(
        tool_selection_rate=95.0,
        phantom_call_count=1,
        turns_to_completion=4,
        prompt_tokens_per_task=150,
    )

    regressions = results.compare(current, previous)

    assert "metric_1 dropped from 100.0 to 95.0" in regressions
    assert "metric_4 dropped from 0 to 1" in regressions
    assert "metric_9 dropped from 2 to 4" in regressions
    assert "metric_10 dropped from 100 to 150" in regressions
