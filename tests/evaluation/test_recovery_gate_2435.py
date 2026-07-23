"""#2435 follow-up — Gate-2 convergence-regression gate (strict, not loose).

#2435 made the Gate-2 harness recover at the recursion cap like production instead
of scoring a crash. A recovered run still faces the full strict ``_final_passed``
bar, so a single recovery on a nondeterministic weak model is a legitimate pass —
but a SYSTEMIC spike (many runs looping-then-recovering) is a convergence
regression that must FAIL the gate, not pass silently. These tests pin the strict
threshold: a lone recovery never gates (no reintroduced #2212 flake), a spike does.
"""

from __future__ import annotations

import tests.evaluation.ci_gate2 as ci_gate2
from tests.evaluation.ci_gate2 import (
    _parse_recovery_rate_env,
    _recovery_gate_tripped,
    run_gate2_smoke,
)
from tests.evaluation.runner import EvalResult, EvalScenario, ModelConfig


class TestRecoveryGateThreshold:
    """Pure-function coverage of the strict-but-not-flaky threshold."""

    def test_no_recoveries_never_trips(self) -> None:
        assert _recovery_gate_tripped(0, 5) is False

    def test_lone_recovery_is_tolerated(self) -> None:
        # Weak-model nondeterminism — a single recovery must NOT red-X (the #2212
        # flake we deliberately stopped failing on).
        assert _recovery_gate_tripped(1, 4) is False
        assert _recovery_gate_tripped(1, 1) is False  # even at 100%, below the floor

    def test_exactly_half_does_not_trip_default(self) -> None:
        # Default threshold 0.5 is strict-greater-than, so 2/4 == 50% does not trip.
        assert _recovery_gate_tripped(2, 4) is False

    def test_systemic_spike_trips(self) -> None:
        assert _recovery_gate_tripped(3, 4) is True  # 75% > 50%, >= 2
        assert _recovery_gate_tripped(2, 3) is True  # 67% > 50%, >= 2
        assert _recovery_gate_tripped(4, 4) is True  # 100%

    def test_zero_total_guarded(self) -> None:
        assert _recovery_gate_tripped(0, 0) is False
        assert _recovery_gate_tripped(3, 0) is False

    def test_threshold_can_only_be_tightened_in_practice(self, monkeypatch) -> None:
        # Tunable via the module constant (env at import). Raising it is looser (a
        # spike that used to fail now passes) — documented as disallowed, but the
        # mechanism works; lowering it is stricter.
        monkeypatch.setattr(ci_gate2, "_MAX_RECOVERY_RATE", 0.9)
        assert _recovery_gate_tripped(3, 4) is False  # 75% < 90% now
        monkeypatch.setattr(ci_gate2, "_MAX_RECOVERY_RATE", 0.2)
        assert _recovery_gate_tripped(2, 4) is True  # 50% > 20% now
        # The >=2 floor still protects against a lone-recovery red-X at any rate.
        assert _recovery_gate_tripped(1, 1) is False


def _scn(sid: str) -> EvalScenario:
    return EvalScenario(
        id=sid,
        domain="test",
        title=sid,
        description="",
        user_prompt="go",
        system_prompt="",
        tools_required=["t"],
        expected_outcome="",
        success_criteria=[],
    )


def _model() -> ModelConfig:
    return ModelConfig(
        id="mock",
        provider="anthropic",
        display_name="Mock",
        tier="smoke",
        smoke=True,
        env_key="ANTHROPIC_API_KEY",
        model_id="mock-model",
    )


def _passing_result(sid: str, *, recovered: bool) -> EvalResult:
    return EvalResult(
        scenario_id=sid,
        model_id="mock",
        model_display_name="Mock",
        passed=True,
        tool_calls_made=["t"],
        tool_calls_required=["t"],
        turns_used=2,
        elapsed_seconds=0.1,
        final_response="done",
        error=None,
        task_completion=True,
        tool_selection_rate=100.0,
        recovered_from_step_limit=recovered,
    )


def _wire(monkeypatch, recovered_ids: set[str]) -> None:
    from tests.evaluation.runner import _KEY_PRIORITY

    for key in _KEY_PRIORITY:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-value")
    monkeypatch.setattr(
        "tests.evaluation.ci_gate2.run_scenario",
        lambda scenario, model, active_key=None: _passing_result(
            scenario.id, recovered=scenario.id in recovered_ids
        ),
    )
    monkeypatch.setattr(
        "tests.evaluation.ci_gate2.score_result",
        lambda scenario, result, judge_model="claude-sonnet-4-6": 0.9,  # individually passing
    )


class TestRecoveryGateInGate2Smoke:
    """#2442: per-cell recovery threshold is ADVISORY-only (never fails the cell);
    the SHARD_SUMMARY observability always emits. The strict gate is the cross-cell
    rollup (follow-up), not this per-cell check over 2-4 scenarios."""

    def test_systemic_recovery_spike_is_advisory_not_a_failure(self, monkeypatch) -> None:
        ids = {"a", "b", "c"}
        _wire(monkeypatch, recovered_ids=ids)  # all 3 recovered (100%)
        logs: list[str] = []

        exit_code = run_gate2_smoke(
            [_scn(s) for s in ("a", "b", "c")], [_model()], emit=logs.append
        )

        # Individual runs passed (recovered runs still cleared _final_passed)…
        score_lines = [ln for ln in logs if "score=" in ln]
        assert len(score_lines) == 3 and all("passed=True" in ln for ln in score_lines)
        # …and the per-cell spike does NOT fail the cell (#2442 — it's advisory).
        assert exit_code == 0
        assert not any("RECOVERY_GATE_FAIL" in ln for ln in logs)
        assert any("RECOVERY_ADVISORY" in ln for ln in logs)
        summary = [ln for ln in logs if "SHARD_SUMMARY" in ln]
        assert summary and "recovery_rate=100%" in summary[0]

    def test_lone_recovery_emits_no_advisory(self, monkeypatch) -> None:
        _wire(monkeypatch, recovered_ids={"b"})  # 1 of 3 recovered
        logs: list[str] = []

        exit_code = run_gate2_smoke(
            [_scn(s) for s in ("a", "b", "c")], [_model()], emit=logs.append
        )

        assert exit_code == 0
        assert not any("RECOVERY_ADVISORY" in ln for ln in logs)  # below the floor
        assert not any("RECOVERY_GATE_FAIL" in ln for ln in logs)
        summary = [ln for ln in logs if "SHARD_SUMMARY" in ln]
        assert summary and "recovery_rate=33%" in summary[0]
        assert "recovered_from_step_limit=1" in summary[0]

    def test_shard_summary_always_emitted_for_observability(self, monkeypatch) -> None:
        _wire(monkeypatch, recovered_ids=set())  # zero recoveries
        logs: list[str] = []

        run_gate2_smoke([_scn("a"), _scn("b")], [_model()], emit=logs.append)

        summary = [ln for ln in logs if "SHARD_SUMMARY" in ln]
        assert summary, "SHARD_SUMMARY must always be emitted (recovery observability)"
        assert "recovered_from_step_limit=0" in summary[0]
        assert "recovery_rate=0%" in summary[0]

    def test_writes_per_cell_recovery_report_for_the_rollup(self, monkeypatch, tmp_path) -> None:
        # #2442(b): when a report dir is given, the cell drops a loadable per-cell
        # report carrying per-scenario recovery flags for the cross-cell rollup.
        from tests.evaluation.gate2_recovery_rollup import load_reports

        _wire(monkeypatch, recovered_ids={"b"})  # scenario b recovered
        run_gate2_smoke(
            [_scn(s) for s in ("a", "b")],
            [_model()],
            emit=lambda _l: None,
            recovery_report_dir=str(tmp_path),
            shard_label="A",
        )
        reports = load_reports(tmp_path)
        assert len(reports) == 1
        rep = reports[0]
        assert rep["shard"] == "A" and rep["model"] == "mock"
        by_id = {s["id"]: s for s in rep["scenarios"]}
        assert by_id["b"]["recovered"] is True and by_id["a"]["recovered"] is False


class TestRecoveryRateEnvParsing:
    """#2442: GATE2_MAX_RECOVERY_RATE must parse defensively — a malformed value
    must not crash every Gate-2 cell at import, and a >1 value must not silently
    disable the check forever."""

    def test_default_when_unset(self, monkeypatch) -> None:
        monkeypatch.delenv("GATE2_MAX_RECOVERY_RATE", raising=False)
        assert _parse_recovery_rate_env() == 0.5

    def test_valid_value_used(self, monkeypatch) -> None:
        monkeypatch.setenv("GATE2_MAX_RECOVERY_RATE", "0.3")
        assert _parse_recovery_rate_env() == 0.3

    def test_malformed_falls_back_to_default_not_crash(self, monkeypatch) -> None:
        for bad in ("", "50%", "abc", "0.5 x"):
            monkeypatch.setenv("GATE2_MAX_RECOVERY_RATE", bad)
            assert _parse_recovery_rate_env() == 0.5  # no ValueError at import

    def test_out_of_range_is_clamped(self, monkeypatch) -> None:
        monkeypatch.setenv("GATE2_MAX_RECOVERY_RATE", "50")  # fat-finger 50 (meant 50%)
        assert _parse_recovery_rate_env() == 1.0  # clamped — never silently disabled
        monkeypatch.setenv("GATE2_MAX_RECOVERY_RATE", "-0.2")
        assert _parse_recovery_rate_env() == 0.0
