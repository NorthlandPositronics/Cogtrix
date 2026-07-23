"""Markdown dashboard for stored quality-harness run results.

The dashboard summarizes the latest stored result per version from
``tests/quality/results/`` and shows a short trend window for each metric.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from tests.quality import results
from tests.quality.metrics import MetricsReport

_VERSION_TOKEN = re.compile(r"(\d+|\D+)")


@dataclass(frozen=True)
class _MetricSpec:
    key: str
    display_name: str
    higher_is_better: bool


_METRICS_LIST: list[_MetricSpec] = []
for _metric_label, attr, higher_is_better in results._METRIC_RULES:  # type: ignore[attr-defined]
    _METRICS_LIST.append(
        _MetricSpec(
            key=attr,
            display_name=f"`{attr}`",
            higher_is_better=higher_is_better,
        )
    )
_METRICS: tuple[_MetricSpec, ...] = tuple(_METRICS_LIST)


def generate_report(results_root: Path | None = None, limit: int = 5) -> str:
    """Return a Markdown dashboard from stored quality result JSON files."""

    root = Path(results_root) if results_root is not None else results.RESULTS_ROOT
    if limit <= 0:
        raise ValueError("limit must be positive")

    latest_runs = _load_latest_runs(root)
    if not latest_runs:
        return "# Cogtrix Quality Trend Dashboard\n\n_No stored quality results found._\n"

    selected = latest_runs[-limit:]
    versions = [version for version, _ in selected]
    reports = [report for _, report in selected]

    md_lines: list[str] = [
        "# Cogtrix Quality Trend Dashboard",
        "",
        "## Summary",
        "",
        f"- Results root: `{root}`",
        f"- Versions included: {len(selected)}",
        f"- Latest version: `{versions[-1]}`",
        "",
        "## Metric Trends",
        "",
        "| Metric | Last values | Trend |",
        "| --- | --- | --- |",
    ]

    for spec in _METRICS:
        values = [getattr(report, spec.key) for report in reports]
        window = _format_window(versions, values)
        trend = _trend(values, spec.higher_is_better)
        md_lines.append(f"| {spec.display_name} | {window} | {trend} |")

    md_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Trend direction is based on the oldest and newest values in the displayed window.",
            "- Higher-is-better metrics are marked improving when they rise; lower-is-better metrics are marked improving when they fall.",
        ]
    )

    return "\n".join(md_lines) + "\n"


def _load_latest_runs(root: Path) -> list[tuple[str, MetricsReport]]:
    if not root.exists():
        return []

    versions = sorted(
        (entry.name for entry in root.iterdir() if entry.is_dir()),
        key=_version_sort_key,
    )

    runs: list[tuple[str, MetricsReport]] = []
    for version in versions:
        report = results.load_previous(version)
        if report is not None:
            runs.append((version, report))
    return runs


def _version_sort_key(version: str) -> tuple[object, ...]:
    parts: list[object] = []
    for token in _VERSION_TOKEN.findall(version):
        if token.isdigit():
            parts.append(int(token))
        else:
            parts.append(token)
    return tuple(parts)


def _format_window(versions: Iterable[str], values: Iterable[float | int]) -> str:
    items = [f"`{version}`: {value:g}" for version, value in zip(versions, values, strict=True)]
    return " → ".join(items)


def _trend(values: list[float | int], higher_is_better: bool) -> str:
    if len(values) < 2:
        return "→ stable"

    first = values[0]
    last = values[-1]
    if first == last:
        return "→ stable"

    improved = (last > first) if higher_is_better else (last < first)
    return "↗ improving" if improved else "↘ worsening"


if __name__ == "__main__":  # pragma: no cover - convenience entrypoint
    print(generate_report())
