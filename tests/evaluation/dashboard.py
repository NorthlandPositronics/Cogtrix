"""Gate 2 evaluation dashboard — generates Markdown + CSV reports from result JSONL files.

Usage::

    from pathlib import Path
    from tests.evaluation.dashboard import generate_dashboard

    generate_dashboard(
        results_dir=Path("data/eval_results/v0.8.0"),
        output_md=Path("data/eval_results/v0.8.0/report.md"),
        output_csv=Path("data/eval_results/v0.8.0/report.csv"),
        compare_to="v0.7.9",  # optional — previous version tag
    )

The dashboard reads all ``*.jsonl`` files in *results_dir*, computes pass
rates per model × domain, and writes a readable Markdown table plus a
machine-readable CSV.  When *compare_to* is provided, it loads the prior
run from ``data/eval_results/{compare_to}`` and flags regressions
(>10 % drop in pass rate for any scenario).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tests.evaluation.runner import load_all_scenarios, load_results

# ── Types ────────────────────────────────────────────────────────────────────


class _ModelResults:
    """Internal accumulator for a single model's results."""

    def __init__(self, model_id: str, display_name: str) -> None:
        self.model_id = model_id
        self.display_name = display_name
        # domain -> list of (scenario_id, passed)
        self.by_domain: dict[str, list[tuple[str, bool]]] = defaultdict(list)

    @property
    def overall_pass_rate(self) -> float:
        all_results = [passed for results in self.by_domain.values() for _, passed in results]
        if not all_results:
            return 0.0
        return sum(all_results) / len(all_results) * 100

    def domain_pass_rate(self, domain: str) -> float:
        results = self.by_domain.get(domain, [])
        if not results:
            return 0.0
        return sum(passed for _, passed in results) / len(results) * 100


# ── Public API ───────────────────────────────────────────────────────────────


def generate_dashboard(
    results_dir: Path,
    output_md: Path,
    output_csv: Path,
    compare_to: str | None = None,
) -> None:
    """Generate Markdown + CSV dashboard from Gate 2 result JSONL files.

    Args:
        results_dir: Directory containing ``*.jsonl`` result files (one per
            model under test).
        output_md: Path to write the Markdown report.
        output_csv: Path to write the CSV report.
        compare_to: Optional previous version tag.  When provided, the
            dashboard loads ``data/eval_results/{compare_to}/*.jsonl`` and
            flags any scenario that dropped >10 % in pass rate.
    """
    results_dir = Path(results_dir)
    output_md = Path(output_md)
    output_csv = Path(output_csv)

    # Load scenario definitions so we know each scenario's domain
    scenarios = {s.id: s for s in load_all_scenarios()}
    domains = sorted({s.domain for s in scenarios.values()})

    # Load current results
    current = _load_run(results_dir, scenarios)

    # Load previous results for regression detection
    previous: dict[str, _ModelResults] = {}
    if compare_to:
        prev_dir = results_dir.parent / compare_to
        if prev_dir.exists():
            previous = _load_run(prev_dir, scenarios)

    # Build report
    version = results_dir.name
    date_str = datetime.now(UTC).strftime("%Y-%m-%d")
    md_lines: list[str] = [
        f"# Cogtrix {version} — Gate 2 Evaluation Report ({date_str})",
        "",
        "## Summary",
        "",
        _build_md_table(current, domains),
        "",
    ]

    # Regression section
    regressions = _detect_regressions(current, previous, scenarios)
    if regressions:
        md_lines.extend(
            [
                "## Regressions vs previous run",
                "",
            ]
        )
        for model_id, scenario_id, prev_rate, curr_rate in regressions:
            md_lines.append(
                f"- `{model_id}` / `{scenario_id}`: "
                f"{prev_rate:.0f}% → {curr_rate:.0f}% ⚠ REGRESSION"
            )
        md_lines.append("")
    elif compare_to:
        md_lines.extend(
            [
                "## Regressions vs previous run",
                "",
                "No regressions detected.",
                "",
            ]
        )

    # Write Markdown
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(md_lines), encoding="utf-8")

    # Write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(current, domains, output_csv)


def _load_run(
    results_dir: Path,
    scenarios: dict[str, Any],
) -> dict[str, _ModelResults]:
    """Load all JSONL result files from a directory into _ModelResults."""
    model_results: dict[str, _ModelResults] = {}

    for jsonl_file in sorted(results_dir.glob("*.jsonl")):
        for row in load_results(jsonl_file):
            model_id = row["model_id"]
            display_name = row.get("model_display_name", model_id)
            scenario_id = row["scenario_id"]
            passed = bool(row.get("passed", False))

            if model_id not in model_results:
                model_results[model_id] = _ModelResults(model_id, display_name)

            # Determine domain from scenario definition
            scenario = scenarios.get(scenario_id)
            domain = scenario.domain if scenario else "unknown"
            model_results[model_id].by_domain[domain].append((scenario_id, passed))

    return model_results


def _build_md_table(
    model_results: dict[str, _ModelResults],
    domains: list[str],
) -> str:
    """Return a Markdown table of pass rates."""
    if not model_results:
        return "_No results found._"

    # Header
    domain_headers = " | ".join(f"{d.upper()}" for d in domains)
    header = f"| Model | {domain_headers} | OVERALL |"
    separator = "|" + "|".join(" --- " for _ in range(len(domains) + 2)) + "|"

    lines = [header, separator]

    for model in sorted(model_results.values(), key=lambda m: m.overall_pass_rate, reverse=True):
        domain_cells = " | ".join(
            f"{model.domain_pass_rate(d):.0f}% {_grade_emoji(model.domain_pass_rate(d))}"
            for d in domains
        )
        overall = model.overall_pass_rate
        lines.append(
            f"| {model.display_name} | {domain_cells} | {overall:.0f}% {_grade_emoji(overall)} |"
        )

    return "\n".join(lines)


def _grade_emoji(rate: float) -> str:
    """Return a visual grade indicator based on pass rate."""
    if rate >= 90:
        return "✓"
    if rate >= 75:
        return "~"
    return "⚠"


def _detect_regressions(
    current: dict[str, _ModelResults],
    previous: dict[str, _ModelResults],
    scenarios: dict[str, Any],
) -> list[tuple[str, str, float, float]]:
    """Return list of (model_id, scenario_id, prev_rate, curr_rate) regressions.

    A regression is defined as a >10 percentage-point drop in pass rate
    for a specific scenario.
    """
    regressions: list[tuple[str, str, float, float]] = []
    if not previous:
        return regressions

    for model_id, curr_model in current.items():
        prev_model = previous.get(model_id)
        if not prev_model:
            continue

        # Build per-scenario pass rates for current and previous
        curr_by_scenario: dict[str, bool] = {}
        prev_by_scenario: dict[str, bool] = {}

        for _domain, results in curr_model.by_domain.items():
            for scenario_id, passed in results:
                curr_by_scenario[scenario_id] = passed

        for _domain, results in prev_model.by_domain.items():
            for scenario_id, passed in results:
                prev_by_scenario[scenario_id] = passed

        # Detect regressions (>10% drop in binary pass/fail is simply
        # passing → failing)
        for scenario_id in set(curr_by_scenario) | set(prev_by_scenario):
            prev_pass = prev_by_scenario.get(scenario_id, True)
            curr_pass = curr_by_scenario.get(scenario_id, True)
            if prev_pass and not curr_pass:
                regressions.append((model_id, scenario_id, 100.0, 0.0))

    return regressions


def _write_csv(
    model_results: dict[str, _ModelResults],
    domains: list[str],
    path: Path,
) -> None:
    """Write a CSV report of pass rates."""
    fieldnames = ["model", "overall"] + domains
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model in sorted(
            model_results.values(), key=lambda m: m.overall_pass_rate, reverse=True
        ):
            row = {
                "model": model.display_name,
                "overall": f"{model.overall_pass_rate:.1f}",
            }
            for d in domains:
                row[d] = f"{model.domain_pass_rate(d):.1f}"
            writer.writerow(row)
