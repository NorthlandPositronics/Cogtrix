"""Tests for tests/quality/dashboard.py."""

from __future__ import annotations

from pathlib import Path

from tests.quality import dashboard, results
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
        prompt_tokens_per_task=100,
        post_cutoff_phantom_count=0,
        identical_error_retry_count=0,
        tier1_pass=True,
        tier2_warnings=[],
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_generate_report_empty_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(results, "RESULTS_ROOT", tmp_path / "results")

    report = dashboard.generate_report()

    assert "_No stored quality results found._" in report


def test_generate_report_tracks_last_five_versions_and_trends(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(results, "RESULTS_ROOT", tmp_path / "results")

    versions = [
        "0.2.9",
        "0.2.10",
        "0.2.11",
        "0.2.12",
        "0.2.13",
        "0.2.14",
    ]
    for index, version in enumerate(versions):
        results.store_run(
            _make_report(
                tool_selection_rate=90.0 + index,
                phantom_call_count=5 - index,
                turns_to_completion=2 + index,
                prompt_tokens_per_task=100 + index * 10,
            ),
            version,
        )

    report = dashboard.generate_report(limit=5)

    assert "0.2.9" not in report
    for version in versions[1:]:
        assert f"`{version}`" in report

    assert "- Versions included: 5" in report
    assert "- Latest version: `0.2.14`" in report
    assert "`tool_selection_rate`" in report
    assert "↗ improving" in report
    assert "`turns_to_completion`" in report
    assert "↘ worsening" in report


def test_generate_report_uses_version_order_not_lexicographic(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(results, "RESULTS_ROOT", tmp_path / "results")

    for version, value in [("0.2.9", 91.0), ("0.2.10", 92.0)]:
        results.store_run(_make_report(tool_selection_rate=value), version)

    report = dashboard.generate_report(limit=2)

    metric_line = next(line for line in report.splitlines() if "tool_selection_rate" in line)
    first_idx = metric_line.index("`0.2.9`")
    second_idx = metric_line.index("`0.2.10`")
    assert first_idx < second_idx
