"""Regression tests for issue #1158: ThreadPoolExecutor __exit__ blocks on hung threads.

Verifies that all 5 affected locations use explicit ThreadPoolExecutor with
result(timeout=N) and shutdown(wait=False), so hung threads do not block the caller.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── cogtrix.py prompt-prep paths ─────────────────────────────────────────────


class TestCogtrixPromptPrepTimeout:
    """The two prompt-prep paths in cogtrix.py must not block on hung threads."""

    def test_explicit_pool_not_context_manager(self):
        """Verify cogtrix.py uses explicit pool = ThreadPoolExecutor (not `with`)."""
        import cogtrix

        source = Path(cogtrix.__file__).read_text()
        # The old buggy pattern:
        assert "with _cf.ThreadPoolExecutor" not in source
        # The new fixed pattern:
        assert "_pool = _cf.ThreadPoolExecutor" in source
        assert "_pool.shutdown(wait=False)" in source
        assert "_ctx_future.result(timeout=60)" in source
        assert "_opt_future.result(timeout=60)" in source


# ── service.py _init_channels ────────────────────────────────────────────────


class TestServiceInitChannelsTimeout:
    """_init_channels must skip a hung channel init and not block."""

    @pytest.mark.timeout(60)
    def test_hung_channel_init_skipped(self):
        """If one channel init hangs, others still complete."""
        from src.assistant.service import AssistantService

        svc = AssistantService.__new__(AssistantService)
        stop_event = threading.Event()

        with patch.object(
            svc, "_init_whatsapp", side_effect=lambda *a, **k: stop_event.wait(timeout=60)
        ):
            with patch.object(svc, "_init_telegram", return_value=MagicMock(name="telegram")):
                cfg2 = MagicMock()
                cfg2.services = {"whatsapp": {}, "telegram": {}}
                cfg2.get.return_value = {
                    "channels": {
                        "whatsapp": {"enabled": True},
                        "telegram": {"enabled": True},
                    }
                }
                cfg2.data_dir = "/tmp"
                t0 = time.monotonic()
                channels = svc._discover_channels(cfg2)
                elapsed = time.monotonic() - t0
                # Must return within ~35s (30s timeout + margin), not hang forever
                assert elapsed < 35, f"Blocked for {elapsed:.1f}s — pool __exit__ not fixed"
                # Telegram should still be initialized
                assert len(channels) == 1
                assert channels[0] is not None

        stop_event.set()


# ── ingest.py _ingest_files_parallel ─────────────────────────────────────────


class TestIngestFilesParallelTimeout:
    """_ingest_files_parallel must mark timed-out files as failed."""

    @pytest.mark.timeout(120)
    def test_hung_ingest_file_marked_failed(self, tmp_path: Path):
        """If _prepare_ingest_file hangs, the file is marked failed and loop continues."""
        from src.rag.ingest import ingest_many

        stop_event = threading.Event()

        # Create a dummy config
        config = MagicMock()
        config.vectordb_dir = tmp_path / "vectordb"
        config.chunk_size = 500
        config.chunk_overlap = 50
        config.embedding_model = "test"

        # Create a dummy file path
        dummy_file = tmp_path / "test.txt"
        dummy_file.write_text("test content")

        with patch(
            "src.rag.ingest._prepare_ingest_file",
            side_effect=lambda *a, **k: stop_event.wait(timeout=120),
        ):
            t0 = time.monotonic()
            results = ingest_many([str(dummy_file)], config)
            elapsed = time.monotonic() - t0
            # Must return within ~65s (60s timeout + margin), not hang forever
            assert elapsed < 65, f"Blocked for {elapsed:.1f}s — pool __exit__ not fixed"
            assert results[str(dummy_file)] is False

        stop_event.set()


# ── whatsapp.py _resolve_uncached ────────────────────────────────────────────


class TestWhatsAppPrefetchLidsTimeout:
    """_prefetch_lids must skip a hung LID resolution."""

    def test_hung_lid_resolution_skipped(self):
        """If one _resolve_lid hangs, others still complete."""
        from src.assistant.channels.whatsapp import WhatsAppChannel

        ch = WhatsAppChannel.__new__(WhatsAppChannel)
        ch._snapshot = {}
        ch._overview_limit = 100
        ch._lid_cache = {}
        ch._lid_cache_lock = threading.Lock()
        ch._client = MagicMock()
        stop_event = threading.Event()

        call_count = 0

        def _slow_lid(number: str) -> None:
            nonlocal call_count
            call_count += 1
            if number == "slow@lid":
                stop_event.wait(timeout=60)
            ch._lid_cache[number] = ("resolved", 9999999999)

        ch._resolve_lid = _slow_lid

        msgs = [
            MagicMock(from_number="slow@lid"),
            MagicMock(from_number="fast@lid"),
        ]

        t0 = time.monotonic()
        ch._prefetch_lids(msgs)
        elapsed = time.monotonic() - t0
        # Must return within ~15s (10s timeout + margin), not hang forever
        assert elapsed < 15, f"Blocked for {elapsed:.1f}s — pool __exit__ not fixed"
        # Fast number should be resolved
        assert "fast@lid" in ch._lid_cache

        stop_event.set()
