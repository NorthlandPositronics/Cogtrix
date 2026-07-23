"""
JSON file-based memory store.
Saves conversation history to data/history/{session_id}.json.
"""

import json
import logging
from pathlib import Path
from typing import Any

from src.memory.base import BaseMemoryStore

log = logging.getLogger("cogtrix")

# Optional LangChain message classes
try:  # pragma: no cover
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
except ImportError:  # pragma: no cover
    BaseMessage = None  # type: ignore
    HumanMessage = None  # type: ignore
    AIMessage = None  # type: ignore


def _message_to_dict(msg: Any) -> dict:
    """Serialize a message to a dict representation."""
    if BaseMessage is not None and isinstance(msg, BaseMessage):
        role = "human" if isinstance(msg, HumanMessage) else "ai"
        return {"type": role, "content": msg.content}
    # Fallback for simple dict-like messages
    if isinstance(msg, dict) and "content" in msg:
        return {"type": msg.get("type", "human"), "content": msg["content"]}
    return {"type": "human", "content": str(msg)}


def _dict_to_message(data: dict) -> Any:
    """Deserialize dict back to a message object if LangChain is available."""
    msg_type = data.get("type", "human")
    content = data.get("content", "")
    if HumanMessage is not None and AIMessage is not None:
        if msg_type == "ai":
            return AIMessage(content=content)
        return HumanMessage(content=content)
    # Fallback: return the dict itself
    return {"type": msg_type, "content": content}


class JsonFileMemoryStore(BaseMemoryStore):
    """Persist conversation history as JSON files."""

    def __init__(self, base_dir: str = "data/history"):
        self.base_path = Path(base_dir)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        safe_id = session_id.replace("/", "_")
        return self.base_path / f"{safe_id}.json"

    def load_history(self, session_id: str):
        path = self._session_path(session_id)
        if not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return [_dict_to_message(item) for item in data]
        except json.JSONDecodeError as e:
            log.error(f"Corrupt session file {path}: {e}")
            return []
        except Exception as e:
            log.error(f"Error loading session {session_id}: {e}")
            return []

    def save_history(self, session_id: str, messages: list[Any]):
        path = self._session_path(session_id)
        serializable = [_message_to_dict(m) for m in messages]
        with path.open("w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
