"""#2442 (b) — cross-cell convergence-regression gate for Gate-2.

The per-cell recovery check (#2442 part a) is advisory-only because a single CI
cell is one ``(shard × model)`` with just 2-4 scenarios — far too small a sample
for a rate threshold. This module is the *strict* gate: each cell writes a
per-cell recovery report (``write_cell_report``), and a post-matrix rollup job
aggregates every report across the WHOLE matrix and fails on a real convergence
regression.

Why this is strict AND not flaky:
- A convergence regression from a code change makes a scenario loop **regardless
  of model**, so it shows up as that scenario recovering on **multiple models**.
  We fail when a scenario recovered on ``>= _MIN_MODELS_FOR_SCENARIO_GATE`` models
  — this catches the single-scenario regression the per-cell check structurally
  missed.
- An isolated weak-model nondeterministic recovery touches **one** model for that
  scenario → never trips → no reintroduced #2212 flake.
- An **overall-matrix rate** backstop catches a widespread regression (many
  scenarios recovering) even if no single scenario crosses the per-scenario floor.

Per the gates-stay-strict rule, both thresholds tune DOWN (stricter) only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: A scenario recovering on this many DISTINCT models is a convergence regression
#: (a code-path loop is model-independent), not weak-model nondeterminism.
_MIN_MODELS_FOR_SCENARIO_GATE = int(os.environ.get("GATE2_MIN_MODELS_FOR_SCENARIO_GATE", "2"))


def _parse_matrix_rate_env() -> float:
    """Overall-matrix recovery-rate backstop, parsed defensively (mirrors #2442a)."""
    raw = os.environ.get("GATE2_MAX_MATRIX_RECOVERY_RATE", "0.25")
    try:
        val = float(raw)
    except (TypeError, ValueError):
        print(f"[gate2-rollup] WARN invalid GATE2_MAX_MATRIX_RECOVERY_RATE={raw!r}; using 0.25")
        return 0.25
    return min(1.0, max(0.0, val))


_MAX_MATRIX_RECOVERY_RATE = _parse_matrix_rate_env()

_REPORT_PREFIX = "gate2-recovery-"
_REPORT_SUFFIX = ".json"


def write_cell_report(
    out_dir: str | os.PathLike[str],
    *,
    shard: str,
    model: str,
    scenarios: list[dict[str, Any]],
) -> Path:
    """Write one cell's recovery report; ``scenarios`` is a list of
    ``{"id", "recovered": bool, "passed": bool}``. Returns the path written."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{_REPORT_PREFIX}{shard}-{model}{_REPORT_SUFFIX}"
    path.write_text(json.dumps({"shard": shard, "model": model, "scenarios": scenarios}))
    return path


@dataclass
class RollupResult:
    total_runs: int
    recovered_runs: int
    matrix_rate: float
    # scenario_id -> sorted list of models that recovered on it
    scenarios_over_model_floor: dict[str, list[str]]
    matrix_rate_tripped: bool

    @property
    def tripped(self) -> bool:
        return bool(self.scenarios_over_model_floor) or self.matrix_rate_tripped


def aggregate(reports: list[dict[str, Any]]) -> RollupResult:
    """Aggregate per-cell reports into the cross-cell gate decision (pure)."""
    total = 0
    recovered = 0
    recovered_models_by_scenario: dict[str, set[str]] = defaultdict(set)
    for rep in reports:
        model = rep.get("model", "?")
        for scn in rep.get("scenarios", []):
            total += 1
            if scn.get("recovered"):
                recovered += 1
                recovered_models_by_scenario[str(scn.get("id"))].add(model)

    over_floor = {
        sid: sorted(models)
        for sid, models in recovered_models_by_scenario.items()
        if len(models) >= _MIN_MODELS_FOR_SCENARIO_GATE
    }
    rate = (recovered / total) if total else 0.0
    return RollupResult(
        total_runs=total,
        recovered_runs=recovered,
        matrix_rate=rate,
        scenarios_over_model_floor=over_floor,
        matrix_rate_tripped=(total > 0 and rate > _MAX_MATRIX_RECOVERY_RATE),
    )


def load_reports(report_dir: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Load every ``gate2-recovery-*.json`` under ``report_dir`` (recursively —
    CI downloads each artifact into its own subdir)."""
    root = Path(report_dir)
    reports: list[dict[str, Any]] = []
    for path in sorted(root.rglob(f"{_REPORT_PREFIX}*{_REPORT_SUFFIX}")):
        try:
            reports.append(json.loads(path.read_text()))
        except (
            OSError,
            ValueError,
        ) as exc:  # noqa: BLE001 — a bad artifact must not crash the gate
            print(f"[gate2-rollup] WARN skipping unreadable report {path}: {exc}")
    return reports


def run_rollup(report_dir: str | os.PathLike[str], emit: Callable[[str], None] = print) -> int:
    """Load reports, apply the strict cross-cell gate, emit a summary. Returns
    0 (pass) or 1 (convergence regression)."""
    reports = load_reports(report_dir)
    if not reports:
        # No reports at all (e.g. every cell key-exhausted/skipped) — nothing to
        # gate on. Do NOT fail: absence of data is not a regression signal.
        emit("[gate2-rollup] SKIP — no recovery reports found (no cells ran)")
        return 0
    res = aggregate(reports)
    emit(
        f"[gate2-rollup] MATRIX_SUMMARY cells={len(reports)} total_runs={res.total_runs} "
        f"recovered={res.recovered_runs} matrix_rate={res.matrix_rate:.0%} "
        f"(scenario_model_floor={_MIN_MODELS_FOR_SCENARIO_GATE}, "
        f"matrix_rate_threshold={_MAX_MATRIX_RECOVERY_RATE:.0%})"
    )
    if res.scenarios_over_model_floor:
        for sid, models in sorted(res.scenarios_over_model_floor.items()):
            emit(
                f"[gate2-rollup] CONVERGENCE_REGRESSION scenario={sid} recovered on "
                f"{len(models)} models {models} (>= {_MIN_MODELS_FOR_SCENARIO_GATE}). A "
                "scenario looping across multiple models is a code-path convergence "
                "regression, not weak-model nondeterminism."
            )
    if res.matrix_rate_tripped:
        emit(
            f"[gate2-rollup] CONVERGENCE_REGRESSION matrix recovery_rate "
            f"{res.matrix_rate:.0%} > {_MAX_MATRIX_RECOVERY_RATE:.0%} threshold — "
            "widespread looping-then-recovering across the matrix."
        )
    if res.tripped:
        emit(
            "[gate2-rollup] FAIL — convergence regression detected. Investigate the "
            "change under test; do NOT raise the thresholds to get green (#2442)."
        )
        return 1
    emit("[gate2-rollup] PASS — no cross-cell convergence regression.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gate2_recovery_rollup")
    parser.add_argument(
        "report_dir",
        help="Directory containing the downloaded per-cell gate2-recovery-*.json artifacts.",
    )
    args = parser.parse_args(argv)
    return run_rollup(args.report_dir)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
