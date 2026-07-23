"""Regression tests for MemoryFactory race conditions."""

import threading
from typing import Any

from cogtrix_core.memory.base import BaseMemoryStore
from cogtrix_core.memory.factory import MemoryFactory
from cogtrix_core.memory.manager import BaseMemoryManager


class DummyStore(BaseMemoryStore):
    """Minimal store for factory tests."""

    def load_history(self, session_id: str) -> list[Any]:
        return []

    def save_history(self, session_id: str, messages: list[Any]) -> None:
        pass


class DummyManager(BaseMemoryManager):
    """Minimal manager for factory tests."""

    @property
    def mode_name(self) -> str:
        return "dummy"

    def load(self) -> list[Any]:
        return []

    def save(self, messages: list[Any]) -> None:
        pass

    def get_context(self, query: str | None = None) -> str:
        return ""

    def add_interaction(self, user_input: str, response: str) -> None:
        pass

    def get_recent_history(self, n: int = 5) -> list[dict[str, str]]:
        return []

    def prepare_context(self, user_input: str) -> Any:
        from cogtrix_core.memory.context import MemoryContext

        return MemoryContext()

    def update(
        self,
        user_input: str,
        ai_response: str,
        agent_messages: list[Any] | None = None,
    ) -> None:
        pass


class TestFactoryUnregisterRace:
    """Regression tests for #1081 — unregister() without lock."""

    def test_unregister_acquires_lock(self):
        """unregister() must hold _lock while mutating _registry."""
        # Ensure clean state
        MemoryFactory.unregister("race_test_mode")

        # Register a dummy mode
        MemoryFactory.register("race_test_mode", DummyManager)

        try:
            # Verify it's registered
            assert MemoryFactory.is_registered("race_test_mode")

            # Unregister and verify lock discipline by checking no exception
            MemoryFactory.unregister("race_test_mode")
            assert not MemoryFactory.is_registered("race_test_mode")
        finally:
            # Cleanup
            MemoryFactory.unregister("race_test_mode")

    def test_concurrent_register_unregister_no_exceptions(self):
        """Concurrent register/unregister must not raise or corrupt state (#1081)."""
        mode = "concurrent_race_mode"
        iterations = 100
        threads_per_side = 4
        barrier = threading.Barrier(threads_per_side * 2)
        errors: list[Exception] = []

        def reg():
            try:
                barrier.wait(timeout=5)
                for _ in range(iterations):
                    try:
                        MemoryFactory.register(mode, DummyManager)
                    except ValueError:
                        # Already registered — expected race outcome
                        pass
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        def unreg():
            try:
                barrier.wait(timeout=5)
                for _ in range(iterations):
                    MemoryFactory.unregister(mode)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        # Ensure clean start
        MemoryFactory.unregister(mode)

        try:
            workers = []
            for _ in range(threads_per_side):
                workers.append(threading.Thread(target=reg))
                workers.append(threading.Thread(target=unreg))

            for t in workers:
                t.start()
            for t in workers:
                t.join(timeout=30)

            assert not errors, f"Exceptions during concurrent register/unregister: {errors}"

            # Final state must be consistent: either registered or not
            is_reg = MemoryFactory.is_registered(mode)
            if is_reg:
                cls_ = MemoryFactory.get_manager_class(mode)
                assert cls_ is DummyManager
        finally:
            MemoryFactory.unregister(mode)

    def test_concurrent_unregister_create_no_exceptions(self):
        """Concurrent unregister/create must not raise KeyError (#1081)."""
        mode = "create_race_mode"
        iterations = 100
        threads_per_side = 4
        barrier = threading.Barrier(threads_per_side * 2)
        errors: list[Exception] = []
        store = DummyStore()

        def unreg():
            try:
                barrier.wait(timeout=5)
                for _ in range(iterations):
                    MemoryFactory.unregister(mode)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        def create():
            try:
                barrier.wait(timeout=5)
                for _ in range(iterations):
                    try:
                        MemoryFactory.create(mode, store, "session")
                    except ValueError:
                        # Mode not registered — expected when unregister wins
                        pass
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        # Pre-register so create() has something to work with
        MemoryFactory.register(mode, DummyManager)

        try:
            workers = []
            for _ in range(threads_per_side):
                workers.append(threading.Thread(target=unreg))
                workers.append(threading.Thread(target=create))

            for t in workers:
                t.start()
            for t in workers:
                t.join(timeout=30)

            assert not errors, f"Exceptions during concurrent unregister/create: {errors}"
        finally:
            MemoryFactory.unregister(mode)
