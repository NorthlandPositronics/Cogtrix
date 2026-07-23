"""
Abstract base class for memory persistence backends.
"""

from abc import ABC, abstractmethod


class BaseMemoryStore(ABC):
    """Interface for loading and saving conversation history."""

    @abstractmethod
    def load_history(self, session_id: str):
        """Return a list of chat messages for the given session_id."""
        raise NotImplementedError

    @abstractmethod
    def save_history(self, session_id: str, messages):
        """Persist the given messages for the session_id."""
        raise NotImplementedError
