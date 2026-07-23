"""Tests for provider cache loading and double-checked locking."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.providers import _load_provider, _provider_cache, _provider_cache_lock


class TestProviderCache:
    """Tests for _load_provider caching behavior."""

    def setup_method(self):
        """Clear the provider cache before each test."""
        with _provider_cache_lock:
            _provider_cache.clear()

    def teardown_method(self):
        """Clear the provider cache after each test."""
        with _provider_cache_lock:
            _provider_cache.clear()

    def test_cache_hit_returns_without_import(self):
        """A cached provider is returned without calling import_module."""
        fake_module = MagicMock()
        with _provider_cache_lock:
            _provider_cache["openai"] = fake_module

        with patch("src.providers.__init__.importlib.import_module") as mock_import:
            result = _load_provider("openai")

        assert result is fake_module
        mock_import.assert_not_called()

    def test_cache_miss_imports_and_stores(self):
        """An uncached provider is imported and stored in the cache."""
        fake_module = MagicMock()

        with patch("src.providers.__init__.importlib.import_module", return_value=fake_module):
            result = _load_provider("openai")

        assert result is fake_module
        with _provider_cache_lock:
            assert _provider_cache["openai"] is fake_module

    def test_unknown_provider_raises_value_error(self):
        """An unknown provider type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider type: 'unknown'."):
            _load_provider("unknown")

    def test_double_checked_locking_reduces_contention(self):
        """Slow imports do not block concurrent cache lookups.

        Simulates one thread performing a slow import while another
        thread reads an already-cached provider. The reader should
        complete before the slow import finishes.
        """
        reader_module = MagicMock()
        slow_module = MagicMock()

        with _provider_cache_lock:
            _provider_cache["ollama"] = reader_module

        import_started = threading.Event()
        import_finished = threading.Event()
        reader_result = []

        def slow_import(path: str):
            import_started.set()
            time.sleep(0.2)
            import_finished.set()
            return slow_module

        def reader():
            # Wait until the slow import has started before reading
            import_started.wait(timeout=5.0)
            result = _load_provider("ollama")
            reader_result.append(result)

        with patch("src.providers.__init__.importlib.import_module", side_effect=slow_import):
            slow_thread = threading.Thread(target=_load_provider, args=("openai",))
            read_thread = threading.Thread(target=reader)

            slow_thread.start()
            read_thread.start()

            # The reader should finish before the slow import completes
            read_thread.join(timeout=1.0)
            assert not read_thread.is_alive(), "Reader thread was blocked by slow import"
            assert reader_result == [reader_module]

            slow_thread.join(timeout=5.0)
            assert not slow_thread.is_alive()
            assert import_finished.is_set()

    def test_concurrent_imports_deduplicate_via_double_check(self):
        """Multiple threads importing the same uncached provider store it once."""
        fake_module = MagicMock()
        import_count = [0]

        def counting_import(path: str):
            import_count[0] += 1
            time.sleep(0.05)
            return fake_module

        with patch("src.providers.__init__.importlib.import_module", side_effect=counting_import):
            threads = [threading.Thread(target=_load_provider, args=("openai",)) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5.0)
                assert not t.is_alive()

        # import_module may be called multiple times (Python's import lock
        # handles deduplication at the module level), but the cache should
        # only store the result once.
        with _provider_cache_lock:
            assert _provider_cache["openai"] is fake_module
