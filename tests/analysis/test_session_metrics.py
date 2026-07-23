"""Tests for src.analysis.session_metrics."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from cogtrix_core.analysis.session_metrics import compute_session_metrics, write_session_metrics

SAMPLE_LOG = """\
2026-05-01 10:00:00.000 [INFO] [a1b2c3d4] TOOL_START: read_text_file
2026-05-01 10:00:01.000 [DEBUG] [a1b2c3d4] ⏱ call_model model.invoke: 1000ms
2026-05-01 10:00:02.000 [INFO] [a1b2c3d4] TOOL_START: search_web
2026-05-01 10:00:03.000 [DEBUG] [a1b2c3d4] ⏱ call_model model.invoke: 1000ms
2026-05-01 10:00:04.000 [INFO] [a1b2c3d4] TOOL_START: write_file
2026-05-01 10:00:05.000 [DEBUG] [a1b2c3d4] ⏱ call_model model.invoke: 1000ms
2026-05-01 10:00:06.000 [INFO] [a1b2c3d4] TOOL_START: checkpoint
2026-05-01 10:00:07.000 [INFO] [a1b2c3d4] TOOL_START: read_text_file
2026-05-01 10:00:08.000 [INFO] [a1b2c3d4] TOOL_START: search_web
2026-05-01 10:00:09.000 [INFO] [a1b2c3d4] TOOL_START: edit_file
2026-05-01 10:00:10.000 [INFO] [a1b2c3d4] Checkpoint nudge fired (calls_since=8, round=6)
2026-05-01 10:00:11.000 [INFO] [a1b2c3d4] Stuck detected — forcing thinking break
2026-05-01 10:00:12.000 [DEBUG] [a1b2c3d4] ⏱ call_model thinking_break: 2000ms
2026-05-01 10:00:13.000 [INFO] [a1b2c3d4] TOOL_START: create_directory
2026-05-01 10:00:14.000 [INFO] [a1b2c3d4] No new checkpoints in 20 rounds — forcing thinking break
"""


def _make_log_file(content: str) -> Path:
    path = Path(tempfile.gettempdir()) / "test_session.log"
    path.write_text(content, encoding="utf-8")
    return path


class TestComputeSessionMetrics:
    def test_basic_metrics(self):
        log_path = _make_log_file(SAMPLE_LOG)
        result = compute_session_metrics(str(log_path))

        assert result["session_id"] == "a1b2c3d4"
        assert result["total_tool_calls"] == 8
        # call_model rounds counted from "model.invoke" lines
        assert result["total_call_model_rounds"] == 3

        m = result["metrics"]

        # CD = 1 checkpoint / 8 tools = 12.5%
        assert m["checkpoint_density"]["value"] == pytest.approx(12.5, rel=0.01)

        # RBA = search_web before first non-discovery call
        # First call is read_text_file (discovery), second is search_web
        # First non-discovery is write_file at index 2
        # search_web at index 1 <= write_file at index 2 → 100%
        assert m["research_before_action"]["value"] == 100.0

        # TTFPA = first productive tool (search_web at index 1) → 2 rounds
        assert m["time_to_first_productive_action"]["value"] == 2

        # TCE proxy = non-discovery / total = 6 / 8 = 75.0%
        assert m["tool_call_efficiency"]["value"] == pytest.approx(75.0, rel=0.01)

        # WSHR = 2 searches, both followed by non-discovery within 5 calls
        assert m["web_search_hit_rate"]["value"] == 100.0

        # CRS proxy — read_text_file called 2× (<=2 threshold) → no violations
        assert m["context_retention_score"]["value"] == 100.0

    def test_no_search_web_returns_zero_rba(self):
        log = SAMPLE_LOG.replace("search_web", "read_text_file")
        log_path = _make_log_file(log)
        result = compute_session_metrics(str(log_path))
        assert result["metrics"]["research_before_action"]["value"] == 0.0

    def test_empty_log(self):
        log_path = _make_log_file("just some garbage\nno parseable lines\n")
        result = compute_session_metrics(str(log_path))
        assert "error" in result

    def test_manual_metrics_return_none_with_note(self):
        log_path = _make_log_file(SAMPLE_LOG)
        result = compute_session_metrics(str(log_path))
        m = result["metrics"]

        assert m["task_completion_rate"]["value"] is None
        assert "manual" in m["task_completion_rate"]["note"].lower()

        assert m["stuck_detection_accuracy"]["value"] is None
        assert "thinking breaks" in m["stuck_detection_accuracy"]["note"].lower()

        assert m["pivot_quality"]["value"] is None
        assert "manual" in m["pivot_quality"]["note"].lower()

        assert m["debug_loop_efficiency"]["value"] is None
        assert "diff" in m["debug_loop_efficiency"]["note"].lower()

    def test_composite_score_present(self):
        log_path = _make_log_file(SAMPLE_LOG)
        result = compute_session_metrics(str(log_path))
        assert "composite_score" in result
        assert isinstance(result["composite_score"], float)
        assert 0 <= result["composite_score"] <= 100

    def test_wshr_partial_hits(self):
        """Regression for #1011: WSHR must evaluate each search_web call, not always the first."""
        log = """\
2026-05-01 10:00:00.000 [INFO] [a1b2c3d4] TOOL_START: search_web
2026-05-01 10:00:01.000 [INFO] [a1b2c3d4] TOOL_START: read_text_file
2026-05-01 10:00:02.000 [INFO] [a1b2c3d4] TOOL_START: list_directory
2026-05-01 10:00:03.000 [INFO] [a1b2c3d4] TOOL_START: get_file_info
2026-05-01 10:00:04.000 [INFO] [a1b2c3d4] TOOL_START: read_text_file
2026-05-01 10:00:05.000 [INFO] [a1b2c3d4] TOOL_START: list_directory
2026-05-01 10:00:06.000 [INFO] [a1b2c3d4] TOOL_START: search_web
2026-05-01 10:00:07.000 [INFO] [a1b2c3d4] TOOL_START: write_file
"""
        log_path = _make_log_file(log)
        result = compute_session_metrics(str(log_path))
        # First search_web followed only by discovery tools → no hit
        # Second search_web followed by write_file → hit
        assert result["metrics"]["web_search_hit_rate"]["value"] == 50.0

    def test_multiple_sessions_picks_largest(self):
        log = SAMPLE_LOG + "2026-05-01 10:01:00.000 [INFO] [e5f6a7b8] TOOL_START: read_text_file\n"
        log_path = _make_log_file(log)
        result = compute_session_metrics(str(log_path))
        # a1b2c3d4 has 9 tools, e5f6a7b8 has 1
        assert result["session_id"] == "a1b2c3d4"


class TestWriteSessionMetrics:
    def test_writes_json_file(self, tmp_path: Path):
        log_path = _make_log_file(SAMPLE_LOG)
        out_dir = tmp_path / "metrics"
        out_file = write_session_metrics(str(log_path), str(out_dir))

        assert out_file.exists()
        assert out_file.name == "a1b2c3d4.json"

        data = json.loads(out_file.read_text(encoding="utf-8"))
        assert data["session_id"] == "a1b2c3d4"
        assert "metrics" in data
