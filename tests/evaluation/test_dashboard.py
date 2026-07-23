"""Tests for tests/evaluation/dashboard.py."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from tests.evaluation.dashboard import (
    _build_md_table,
    _detect_regressions,
    _grade_emoji,
    _ModelResults,
    _write_csv,
    generate_dashboard,
)

# ── _ModelResults ────────────────────────────────────────────────────────────


class TestModelResults:
    def test_overall_pass_rate_all_pass(self) -> None:
        m = _ModelResults("gpt-4o", "gpt-4o")
        m.by_domain["procurement"].append(("po_basic", True))
        m.by_domain["finance"].append(("invoice_check", True))
        assert m.overall_pass_rate == 100.0

    def test_overall_pass_rate_mixed(self) -> None:
        m = _ModelResults("claude", "claude-sonnet")
        m.by_domain["procurement"].append(("po_basic", True))
        m.by_domain["finance"].append(("invoice_check", False))
        assert m.overall_pass_rate == 50.0

    def test_overall_pass_rate_empty(self) -> None:
        m = _ModelResults("deepseek", "deepseek-v3")
        assert m.overall_pass_rate == 0.0

    def test_domain_pass_rate(self) -> None:
        m = _ModelResults("gpt-4o", "gpt-4o")
        m.by_domain["procurement"].append(("po_basic", True))
        m.by_domain["procurement"].append(("supplier_reg", True))
        m.by_domain["finance"].append(("invoice_check", False))
        assert m.domain_pass_rate("procurement") == 100.0
        assert m.domain_pass_rate("finance") == 0.0
        assert m.domain_pass_rate("unknown") == 0.0


# ── _grade_emoji ─────────────────────────────────────────────────────────────


class TestGradeEmoji:
    def test_excellent(self) -> None:
        assert _grade_emoji(95.0) == "✓"
        assert _grade_emoji(90.0) == "✓"

    def test_moderate(self) -> None:
        assert _grade_emoji(89.0) == "~"
        assert _grade_emoji(75.0) == "~"

    def test_poor(self) -> None:
        assert _grade_emoji(74.0) == "⚠"
        assert _grade_emoji(0.0) == "⚠"


# ── _build_md_table ──────────────────────────────────────────────────────────


class TestBuildMdTable:
    def test_empty_results(self) -> None:
        assert _build_md_table({}, ["procurement", "finance"]) == "_No results found._"

    def test_single_model(self) -> None:
        m = _ModelResults("gpt-4o", "gpt-4o")
        m.by_domain["procurement"].append(("po_basic", True))
        m.by_domain["finance"].append(("invoice_check", False))
        md = _build_md_table({"gpt-4o": m}, ["finance", "procurement"])
        assert "gpt-4o" in md
        assert "100% ✓" in md
        assert "0% ⚠" in md
        assert "50% ⚠" in md  # overall (50% < 75%)

    def test_sorted_by_overall(self) -> None:
        m1 = _ModelResults("claude", "claude-sonnet")
        m1.by_domain["procurement"].append(("po_basic", True))
        m1.by_domain["finance"].append(("invoice_check", True))

        m2 = _ModelResults("gpt", "gpt-4o")
        m2.by_domain["procurement"].append(("po_basic", True))
        m2.by_domain["finance"].append(("invoice_check", False))

        md = _build_md_table({"gpt": m2, "claude": m1}, ["finance", "procurement"])
        lines = md.splitlines()
        # claude has 100% overall, gpt has 50% — claude should appear first
        claude_idx = next(i for i, line in enumerate(lines) if "claude-sonnet" in line)
        gpt_idx = next(i for i, line in enumerate(lines) if "gpt-4o" in line)
        assert claude_idx < gpt_idx


# ── _detect_regressions ──────────────────────────────────────────────────────


class TestDetectRegressions:
    def test_no_previous(self) -> None:
        curr = {"gpt": _ModelResults("gpt", "gpt")}
        assert _detect_regressions(curr, {}, {}) == []

    def test_pass_to_fail_is_regression(self) -> None:
        curr = {"gpt": _ModelResults("gpt", "gpt")}
        curr["gpt"].by_domain["procurement"].append(("po_basic", False))

        prev = {"gpt": _ModelResults("gpt", "gpt")}
        prev["gpt"].by_domain["procurement"].append(("po_basic", True))

        regs = _detect_regressions(curr, prev, {})
        assert len(regs) == 1
        assert regs[0] == ("gpt", "po_basic", 100.0, 0.0)

    def test_fail_to_pass_is_not_regression(self) -> None:
        curr = {"gpt": _ModelResults("gpt", "gpt")}
        curr["gpt"].by_domain["procurement"].append(("po_basic", True))

        prev = {"gpt": _ModelResults("gpt", "gpt")}
        prev["gpt"].by_domain["procurement"].append(("po_basic", False))

        assert _detect_regressions(curr, prev, {}) == []

    def test_still_passing_is_not_regression(self) -> None:
        curr = {"gpt": _ModelResults("gpt", "gpt")}
        curr["gpt"].by_domain["procurement"].append(("po_basic", True))

        prev = {"gpt": _ModelResults("gpt", "gpt")}
        prev["gpt"].by_domain["procurement"].append(("po_basic", True))

        assert _detect_regressions(curr, prev, {}) == []

    def test_model_missing_in_previous(self) -> None:
        curr = {"gpt": _ModelResults("gpt", "gpt")}
        curr["gpt"].by_domain["procurement"].append(("po_basic", False))

        prev = {}  # no previous run for this model

        assert _detect_regressions(curr, prev, {}) == []


# ── _write_csv ───────────────────────────────────────────────────────────────


class TestWriteCsv:
    def test_writes_expected_rows(self, tmp_path: Path) -> None:
        m = _ModelResults("gpt-4o", "gpt-4o")
        m.by_domain["procurement"].append(("po_basic", True))
        m.by_domain["finance"].append(("invoice_check", False))

        path = tmp_path / "report.csv"
        _write_csv({"gpt-4o": m}, ["finance", "procurement"], path)

        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 1
        assert rows[0]["model"] == "gpt-4o"
        assert rows[0]["overall"] == "50.0"
        assert rows[0]["finance"] == "0.0"
        assert rows[0]["procurement"] == "100.0"

    def test_multiple_models_sorted(self, tmp_path: Path) -> None:
        m1 = _ModelResults("claude", "claude-sonnet")
        m1.by_domain["procurement"].append(("po_basic", True))

        m2 = _ModelResults("gpt", "gpt-4o")
        m2.by_domain["procurement"].append(("po_basic", False))

        path = tmp_path / "report.csv"
        _write_csv({"claude": m1, "gpt": m2}, ["procurement"], path)

        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 2
        # claude first (100% > 0%)
        assert rows[0]["model"] == "claude-sonnet"
        assert rows[1]["model"] == "gpt-4o"


# ── generate_dashboard (integration) ─────────────────────────────────────────


class TestGenerateDashboard:
    def _make_scenario_mock(self, sid: str, domain: str) -> MagicMock:
        m = MagicMock()
        m.id = sid
        m.domain = domain
        return m

    def test_generates_md_and_csv(self, tmp_path: Path) -> None:
        results_dir = tmp_path / "results" / "v0.8.0"
        results_dir.mkdir(parents=True)

        # Write a single JSONL result file
        (results_dir / "gpt-4o.jsonl").write_text(
            json.dumps(
                {
                    "model_id": "gpt-4o",
                    "model_display_name": "gpt-4o",
                    "scenario_id": "po_basic",
                    "passed": True,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "model_id": "gpt-4o",
                    "model_display_name": "gpt-4o",
                    "scenario_id": "invoice_check",
                    "passed": False,
                }
            )
            + "\n"
        )

        output_md = tmp_path / "report.md"
        output_csv = tmp_path / "report.csv"

        with patch(
            "tests.evaluation.dashboard.load_all_scenarios",
            return_value=[
                self._make_scenario_mock("po_basic", "procurement"),
                self._make_scenario_mock("invoice_check", "finance"),
            ],
        ):
            generate_dashboard(results_dir, output_md, output_csv)

        md = output_md.read_text()
        assert "# Cogtrix v0.8.0" in md
        assert "gpt-4o" in md
        assert "100% ✓" in md
        assert "0% ⚠" in md
        assert "No regressions detected." not in md  # no compare_to

        with open(output_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["model"] == "gpt-4o"
        assert rows[0]["procurement"] == "100.0"
        assert rows[0]["finance"] == "0.0"

    def test_regression_detection(self, tmp_path: Path) -> None:
        base = tmp_path / "results"
        curr_dir = base / "v0.8.0"
        prev_dir = base / "v0.7.9"
        curr_dir.mkdir(parents=True)
        prev_dir.mkdir(parents=True)

        # Current: po_basic FAILS
        (curr_dir / "gpt-4o.jsonl").write_text(
            json.dumps(
                {
                    "model_id": "gpt-4o",
                    "model_display_name": "gpt-4o",
                    "scenario_id": "po_basic",
                    "passed": False,
                }
            )
            + "\n"
        )
        # Previous: po_basic PASSES
        (prev_dir / "gpt-4o.jsonl").write_text(
            json.dumps(
                {
                    "model_id": "gpt-4o",
                    "model_display_name": "gpt-4o",
                    "scenario_id": "po_basic",
                    "passed": True,
                }
            )
            + "\n"
        )

        output_md = tmp_path / "report.md"
        output_csv = tmp_path / "report.csv"

        with patch(
            "tests.evaluation.dashboard.load_all_scenarios",
            return_value=[self._make_scenario_mock("po_basic", "procurement")],
        ):
            generate_dashboard(curr_dir, output_md, output_csv, compare_to="v0.7.9")

        md = output_md.read_text()
        assert "## Regressions vs previous run" in md
        assert "gpt-4o" in md
        assert "po_basic" in md
        assert "REGRESSION" in md

    def test_empty_results_dir(self, tmp_path: Path) -> None:
        results_dir = tmp_path / "empty"
        results_dir.mkdir()
        output_md = tmp_path / "report.md"
        output_csv = tmp_path / "report.csv"

        with patch(
            "tests.evaluation.dashboard.load_all_scenarios",
            return_value=[],
        ):
            generate_dashboard(results_dir, output_md, output_csv)

        md = output_md.read_text()
        assert "_No results found._" in md

        with open(output_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 0

    def test_compare_to_missing_dir(self, tmp_path: Path) -> None:
        results_dir = tmp_path / "results" / "v0.8.0"
        results_dir.mkdir(parents=True)
        (results_dir / "gpt-4o.jsonl").write_text(
            json.dumps(
                {
                    "model_id": "gpt-4o",
                    "model_display_name": "gpt-4o",
                    "scenario_id": "po_basic",
                    "passed": True,
                }
            )
            + "\n"
        )

        output_md = tmp_path / "report.md"
        output_csv = tmp_path / "report.csv"

        with patch(
            "tests.evaluation.dashboard.load_all_scenarios",
            return_value=[self._make_scenario_mock("po_basic", "procurement")],
        ):
            # compare_to points to non-existent directory
            generate_dashboard(results_dir, output_md, output_csv, compare_to="v0.7.9")

        md = output_md.read_text()
        # Should still generate report; comparison section says no regressions
        assert "gpt-4o" in md
        assert "## Regressions vs previous run" in md
        assert "No regressions detected." in md
