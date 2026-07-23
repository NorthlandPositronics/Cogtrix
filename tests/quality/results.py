"""Persistent JSON storage for quality harness run results."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from tests.quality.metrics import MetricsReport

RESULTS_ROOT = Path(__file__).parent / "results"
BASELINES_ROOT = Path(__file__).parent / "baselines"

_METRIC_RULES: tuple[tuple[str, str, bool], ...] = (
    ("metric_1", "tool_selection_rate", True),
    ("metric_2", "parameter_name_f1", True),
    ("metric_3", "parameter_value_match_rate", True),
    ("metric_4", "phantom_call_count", False),
    ("metric_5", "task_completion_rate", True),
    ("metric_6", "orphaned_pair_count", False),
    ("metric_7", "tool_readiness_violations", False),
    ("metric_8", "error_recovery_turns", False),
    ("metric_9", "turns_to_completion", False),
    ("metric_10", "prompt_tokens_per_task", False),
    ("metric_11", "post_cutoff_phantom_count", False),
    ("metric_12", "identical_error_retry_count", False),
)


def _version_dir(cogtrix_version: str) -> Path:
    return RESULTS_ROOT / cogtrix_version


def _latest_result_file(cogtrix_version: str) -> Path | None:
    version_dir = _version_dir(cogtrix_version)
    if not version_dir.exists():
        return None

    candidates = [path for path in version_dir.glob("*.json") if path.is_file()]
    if not candidates:
        return None

    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def store_run(report: MetricsReport, cogtrix_version: str) -> Path:
    """Store run results as JSON in tests/quality/results/{version}/{timestamp}.json."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    target_dir = _version_dir(cogtrix_version)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{timestamp}.json"

    payload = {
        "cogtrix_version": cogtrix_version,
        "timestamp": timestamp,
        "report": asdict(report),
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return target


def load_previous(cogtrix_version: str) -> MetricsReport | None:
    """Load the most recent stored run for a given version."""
    latest = _latest_result_file(cogtrix_version)
    if latest is None:
        return None

    payload = json.loads(latest.read_text())
    report_data = payload.get("report", payload)
    return MetricsReport(**report_data)


def load_baseline(path: str = "tests/quality/baselines/baseline.json") -> MetricsReport | None:
    """Load the stored minimum thresholds from disk.

    Returns None if the baseline file doesn't exist.
    """
    baseline_path = Path(path)
    if not baseline_path.exists():
        return None

    payload = json.loads(baseline_path.read_text())
    return MetricsReport(**payload)


def check_ratchet(current: MetricsReport, baseline: MetricsReport) -> list[str]:
    """Return list of regressions where any Tier 1 metric dropped below baseline.

    Higher-is-better: tool_selection_rate, parameter_name_f1, parameter_value_match_rate,
                      task_completion_rate
    Lower-is-better: phantom_call_count, orphaned_pair_count, tool_readiness_violations,
                     error_recovery_turns
    """
    regressions: list[str] = []

    # Higher-is-better metrics
    for attr in (
        "tool_selection_rate",
        "parameter_name_f1",
        "parameter_value_match_rate",
        "task_completion_rate",
    ):
        current_value = getattr(current, attr)
        baseline_value = getattr(baseline, attr)
        if current_value < baseline_value:
            regressions.append(f"{attr} dropped from {baseline_value} to {current_value}")

    # Lower-is-better metrics
    for attr in (
        "phantom_call_count",
        "orphaned_pair_count",
        "tool_readiness_violations",
        "error_recovery_turns",
    ):
        current_value = getattr(current, attr)
        baseline_value = getattr(baseline, attr)
        if current_value > baseline_value:
            regressions.append(f"{attr} increased from {baseline_value} to {current_value}")

    return regressions


def update_baseline(
    report: MetricsReport, path: str = "tests/quality/baselines/baseline.json"
) -> None:
    """Write a new baseline.json with the current metric values."""
    baseline_path = Path(path)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)

    payload = asdict(report)
    baseline_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def compare(current: MetricsReport, previous: MetricsReport) -> list[str]:
    """Return regression strings for metrics that got worse."""
    regressions: list[str] = []

    for metric_label, attr, higher_is_better in _METRIC_RULES:
        current_value = getattr(current, attr)
        previous_value = getattr(previous, attr)

        if current_value == previous_value:
            continue

        is_worse = (
            current_value < previous_value if higher_is_better else current_value > previous_value
        )
        if is_worse:
            regressions.append(f"{metric_label} dropped from {previous_value} to {current_value}")

    return regressions
