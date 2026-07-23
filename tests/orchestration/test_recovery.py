"""Unit tests for extracted orchestration recovery nodes."""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.modifier import RemoveMessage

from src.orchestration.nodes.recovery import (
    build_handle_action_intent_node,
    build_handle_phantom_node,
)


class _DummyLogger:
    def __init__(self) -> None:
        self.warnings: list[tuple[object, ...]] = []
        self.infos: list[tuple[object, ...]] = []

    def warning(self, *args: object) -> None:
        self.warnings.append(args)

    def info(self, *args: object) -> None:
        self.infos.append(args)


def _state(message_id: str = "msg-1") -> dict:
    return {"messages": [AIMessage(content="stub", id=message_id)]}


class TestHandlePhantomNode:
    def test_injects_parse_retry_hint_before_exhaustion(self) -> None:
        phantom_count = [0]
        logger = _DummyLogger()
        node = build_handle_phantom_node(phantom_count, max_retries=3, logger=lambda: logger)

        result = node(_state("p1"))

        assert phantom_count[0] == 1
        assert len(result["messages"]) == 2
        assert result["messages"][0] == RemoveMessage(id="p1")
        assert isinstance(result["messages"][1], HumanMessage)
        assert "could not be parsed" in result["messages"][1].content
        assert logger.warnings

    def test_falls_back_after_retry_budget_is_exceeded(self) -> None:
        phantom_count = [3]
        logger = _DummyLogger()
        node = build_handle_phantom_node(phantom_count, max_retries=3, logger=lambda: logger)

        result = node(_state("p2"))

        assert phantom_count[0] == 4
        assert len(result["messages"]) == 2
        assert result["messages"][0] == RemoveMessage(id="p2")
        # The give-up branch now synthesizes from accumulated state instead
        # of returning the hard-coded "persistent formatting issues" message.
        # With no accumulated checkpoints / tool results, the fallback is the
        # polite "rephrase the question" prompt.
        assert isinstance(result["messages"][1], AIMessage)
        assert "rephrase" in result["messages"][1].content.lower()


class TestHandleActionIntentNode:
    def test_injects_nudge_before_exhaustion(self) -> None:
        action_intent_count = [0]
        logger = _DummyLogger()
        node = build_handle_action_intent_node(
            action_intent_count,
            max_retries=3,
            logger=lambda: logger,
        )

        result = node(_state("a1"))

        assert action_intent_count[0] == 1
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], HumanMessage)
        assert "did not call any tools" in result["messages"][0].content
        assert logger.warnings

    def test_gives_up_after_retry_budget_is_exceeded(self) -> None:
        action_intent_count = [3]
        logger = _DummyLogger()
        node = build_handle_action_intent_node(
            action_intent_count,
            max_retries=3,
            logger=lambda: logger,
        )

        result = node(_state("a2"))

        assert action_intent_count[0] == 4
        # Previously returned [] which let the model's stuck-thinking output
        # stand as the user-facing response.  After the May 2026 user
        # disaster fix, the give-up branch synthesizes a clean answer from
        # accumulated state; with no checkpoints the polite fallback fires.
        assert len(result["messages"]) == 2
        assert result["messages"][0] == RemoveMessage(id="a2")
        assert isinstance(result["messages"][1], AIMessage)
        assert "rephrase" in result["messages"][1].content.lower()
