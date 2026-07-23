"""
Factory for creating memory managers by mode name.
Supports dynamic registration of new memory modes.
"""

import sys
import threading
from typing import Any

from src.memory.base import BaseMemoryStore
from src.memory.manager import BaseMemoryManager


class MemoryFactory:
    """
    Factory for creating memory managers.

    Memory modes register themselves with this factory, allowing
    dynamic creation by mode name (e.g., from CLI arguments).

    Usage:
        # Registration (in mode module)
        MemoryFactory.register("conversation", ConversationManager)

        # Creation (in main application)
        manager = MemoryFactory.create("conversation", store, session_id)
    """

    _registry: dict[str, type[BaseMemoryManager]] = {}
    _lock: threading.RLock = threading.RLock()

    @classmethod
    def register(cls, mode: str, manager_class: type[BaseMemoryManager]) -> None:
        """
        Register a memory manager class for a mode.

        Args:
            mode: Mode name (e.g., 'conversation', 'code', 'reasoning')
            manager_class: Class that implements BaseMemoryManager

        Raises:
            ValueError: If mode is empty or already registered
            TypeError: If manager_class doesn't inherit from BaseMemoryManager
        """
        if not isinstance(mode, str) or not mode:
            raise ValueError("Mode must be a non-empty string")

        if not isinstance(manager_class, type):
            raise TypeError("manager_class must be a class")

        if not issubclass(manager_class, BaseMemoryManager):
            raise TypeError(f"{manager_class.__name__} must inherit from " "BaseMemoryManager")

        with cls._lock:
            if mode in cls._registry:
                existing = cls._registry[mode].__name__
                raise ValueError(f"Mode '{mode}' is already registered to {existing}")

            cls._registry[mode] = manager_class

    @classmethod
    def unregister(cls, mode: str) -> None:
        """
        Unregister a memory mode.

        Primarily for testing purposes.

        Args:
            mode: Mode name to unregister
        """
        cls._registry.pop(mode, None)

    @classmethod
    def create(
        cls,
        mode: str,
        store: BaseMemoryStore,
        session_id: str,
        config: dict[str, Any] | None = None,
    ) -> BaseMemoryManager:
        """
        Create a memory manager for the specified mode.

        Args:
            mode: Mode name (must be registered)
            store: Storage backend for persistence
            session_id: Unique session identifier
            config: Optional mode-specific configuration

        Returns:
            Instance of the appropriate memory manager

        Raises:
            ValueError: If mode is not registered
        """
        with cls._lock:
            if mode not in cls._registry:
                available = cls.available_modes()
                raise ValueError(f"Unknown memory mode: '{mode}'. Available modes: {available}")

            manager_class = cls._registry[mode]
        return manager_class(store, session_id, config)

    @classmethod
    def available_modes(cls) -> list[str]:
        """
        Return list of registered mode names.

        Returns:
            List of available mode names (sorted)
        """
        return sorted(cls._registry.keys())

    @classmethod
    def is_registered(cls, mode: str) -> bool:
        """
        Check if a mode is registered.

        Args:
            mode: Mode name to check

        Returns:
            True if mode is registered
        """
        return mode in cls._registry

    @classmethod
    def get_manager_class(cls, mode: str) -> type[BaseMemoryManager] | None:
        """
        Get the manager class for a mode without instantiating.

        Args:
            mode: Mode name

        Returns:
            Manager class or None if not registered
        """
        return cls._registry.get(mode)

    @classmethod
    def clear_registry(cls) -> None:
        """
        Clear all registrations.

        For testing purposes only. Raises ``RuntimeError`` outside of a
        pytest session to prevent accidental production use.
        """
        if "pytest" not in sys.modules:
            raise RuntimeError("clear_registry() may only be called during testing")
        with cls._lock:
            cls._registry.clear()
