"""#2442 (b) — cross-cell convergence-regression rollup.

Pins the strict cross-cell gate: a scenario recovering on >= N models is a
code-path convergence regression (fail), while an isolated weak-model recovery on
one model is not (pass). Plus an overall-matrix rate backstop, defensive env
parsing, and the write/load report round-trip.
"""

from __future__ import annotations

import tests.evaluation.gate2_recovery_rollup as rollup
from tests.evaluation.gate2_recovery_rollup import (
    _parse_matrix_rate_env,
    aggregate,
    load_reports,
    run_rollup,
    write_cell_report,
)


def _rep(model: str, scenarios: list[tuple[str, bool]]) -> dict:
    return {
        "shard": "X",
        "model": model,
        "scenarios": [{"id": sid, "recovered": rec, "passed": True} for sid, rec in scenarios],
    }


class TestAggregate:
    def test_single_scenario_recovering_on_two_models_is_a_regression(self) -> None:
        # The exact case per-cell gating MISSED: scenario "x" loops on 2 models,
        # overall rate is low. The per-scenario/per-model floor catches it.
        reports = [
            _rep("deepseek", [("x", True), ("y", False), ("z", False)]),
            _rep("kimi", [("x", True), ("y", False), ("z", False)]),
            _rep("gpt4o", [("x", False), ("y", False), ("z", False)]),
        ]
        res = aggregate(reports)
        assert res.tripped
        assert res.scenarios_over_model_floor == {"x": ["deepseek", "kimi"]}
        # overall rate is low (2/9) — this trips on the per-scenario floor, not rate.
        assert not res.matrix_rate_tripped

    def test_lone_model_recovery_is_not_a_regression(self) -> None:
        reports = [
            _rep("deepseek", [("x", True), ("y", False)]),  # only deepseek loops on x
            _rep("kimi", [("x", False), ("y", False)]),
            _rep("gpt4o", [("x", False), ("y", False)]),
        ]
        res = aggregate(reports)
        assert not res.tripped
        assert res.scenarios_over_model_floor == {}

    def test_widespread_regression_trips_matrix_rate_backstop(self, monkeypatch) -> None:
        # No single scenario crosses 2 models, but recoveries are widespread → the
        # overall-rate backstop catches it. Force distinct scenarios per model so
        # the per-scenario floor stays clear.
        reports = [
            _rep("m1", [("a", True), ("b", True)]),
            _rep("m2", [("c", True), ("d", True)]),
        ]
        # 4/4 recovered, all distinct scenarios → 0 over the model-floor, rate 100%.
        res = aggregate(reports)
        assert res.scenarios_over_model_floor == {}
        assert res.matrix_rate_tripped  # 100% > 25% default
        assert res.tripped

    def test_clean_matrix_passes(self) -> None:
        reports = [_rep("m1", [("a", False), ("b", False)]), _rep("m2", [("a", False)])]
        res = aggregate(reports)
        assert not res.tripped
        assert res.total_runs == 3 and res.recovered_runs == 0

    def test_thresholds_tune_down_stricter(self, monkeypatch) -> None:
        reports = [_rep("m1", [("x", True)]), _rep("m2", [("x", False)])]
        # default floor 2 → x on 1 model → not tripped
        assert not aggregate(reports).scenarios_over_model_floor
        # tighten to 1 model → now a lone recovery trips (stricter)
        monkeypatch.setattr(rollup, "_MIN_MODELS_FOR_SCENARIO_GATE", 1)
        assert aggregate(reports).scenarios_over_model_floor == {"x": ["m1"]}


class TestReportRoundTrip:
    def test_write_then_load(self, tmp_path) -> None:
        write_cell_report(
            tmp_path,
            shard="A",
            model="deepseek-v4-flash",
            scenarios=[{"id": "s1", "recovered": True, "passed": True}],
        )
        loaded = load_reports(tmp_path)
        assert len(loaded) == 1
        assert loaded[0]["model"] == "deepseek-v4-flash"
        assert loaded[0]["scenarios"][0]["recovered"] is True

    def test_load_recurses_into_artifact_subdirs(self, tmp_path) -> None:
        # CI downloads each artifact into its own subdir — loader must recurse.
        (tmp_path / "cellA").mkdir()
        (tmp_path / "cellB").mkdir()
        write_cell_report(tmp_path / "cellA", shard="A", model="m1", scenarios=[])
        write_cell_report(tmp_path / "cellB", shard="B", model="m2", scenarios=[])
        assert len(load_reports(tmp_path)) == 2

    def test_load_skips_unreadable_report_without_crashing(self, tmp_path) -> None:
        (tmp_path / "gate2-recovery-A-m1.json").write_text("{not json")
        write_cell_report(tmp_path, shard="B", model="m2", scenarios=[])
        loaded = load_reports(tmp_path)  # must not raise
        assert len(loaded) == 1 and loaded[0]["model"] == "m2"


class TestRunRollup:
    def test_no_reports_skips_not_fails(self, tmp_path) -> None:
        logs: list[str] = []
        assert run_rollup(tmp_path, emit=logs.append) == 0
        assert any("SKIP" in ln for ln in logs)

    def test_regression_fails_with_diagnostic(self, tmp_path) -> None:
        write_cell_report(
            tmp_path,
            shard="A",
            model="deepseek",
            scenarios=[{"id": "x", "recovered": True, "passed": True}],
        )
        write_cell_report(
            tmp_path,
            shard="A",
            model="kimi",
            scenarios=[{"id": "x", "recovered": True, "passed": True}],
        )
        logs: list[str] = []
        assert run_rollup(tmp_path, emit=logs.append) == 1
        assert any("CONVERGENCE_REGRESSION" in ln and "scenario=x" in ln for ln in logs)
        assert any("FAIL" in ln for ln in logs)

    def test_clean_matrix_passes(self, tmp_path) -> None:
        write_cell_report(
            tmp_path,
            shard="A",
            model="m1",
            scenarios=[{"id": "x", "recovered": False, "passed": True}],
        )
        logs: list[str] = []
        assert run_rollup(tmp_path, emit=logs.append) == 0
        assert any("PASS" in ln for ln in logs)
        assert any("MATRIX_SUMMARY" in ln for ln in logs)


class TestEnvParsing:
    def test_default(self, monkeypatch) -> None:
        monkeypatch.delenv("GATE2_MAX_MATRIX_RECOVERY_RATE", raising=False)
        assert _parse_matrix_rate_env() == 0.25

    def test_malformed_falls_back(self, monkeypatch) -> None:
        monkeypatch.setenv("GATE2_MAX_MATRIX_RECOVERY_RATE", "25%")
        assert _parse_matrix_rate_env() == 0.25

    def test_clamped(self, monkeypatch) -> None:
        monkeypatch.setenv("GATE2_MAX_MATRIX_RECOVERY_RATE", "9")
        assert _parse_matrix_rate_env() == 1.0
