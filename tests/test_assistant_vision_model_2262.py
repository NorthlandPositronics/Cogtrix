"""Tests for the assistant vision-model delegation path (#2262).

Covers _describe_images and the vision dispatch logic in _run_agent.
No FAISS, no network, no real LLM — all LLM calls are mocked.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

from cogtrix_core.assistant.channel import IncomingMessage
from cogtrix_core.assistant.handler import MessageHandler
from cogtrix_core.memory.context import MemoryContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session() -> MagicMock:
    session = MagicMock()
    session.session_key = "telegram::99"
    session.lock = MagicMock()
    session.lock.__enter__ = MagicMock(return_value=None)
    session.lock.__exit__ = MagicMock(return_value=False)
    session.guardrail_violations = 0
    session.last_sent_message_id = None
    session.memory_manager.prepare_context.return_value = MemoryContext(
        messages=[],
        context_prefix=None,
    )
    return session


def _make_handler(
    *,
    vision_llm: Any = None,
    conversation_supports_vision: bool | None = None,
    agent_runner: Any = None,
) -> tuple[MessageHandler, MagicMock, MagicMock]:
    """Return (handler, session_mgr, captured_agent_runner)."""
    session = _make_session()
    session_mgr = MagicMock()
    session_mgr.get_or_create.return_value = session

    if agent_runner is None:
        agent_runner = MagicMock(return_value="ok")

    handler = MessageHandler(
        session_mgr=session_mgr,
        config={},
        llm=MagicMock(),
        system_prompt="You are helpful.",
        registry=MagicMock(),
        approvals={"*"},
        available_tools={},
        active_tools=[],
        agent_runner=agent_runner,
        vision_llm=vision_llm,
        conversation_supports_vision=conversation_supports_vision,
    )
    return handler, session_mgr, agent_runner


def _make_vision_llm(description: str = "A cat on a mat.") -> MagicMock:
    """Return a mock LLM whose .invoke() returns a response with the given content."""
    vision_llm = MagicMock()
    response = MagicMock()
    response.content = description
    vision_llm.invoke.return_value = response
    return vision_llm


def _make_msg(text: str = "What is this?", images: list[str] | None = None) -> IncomingMessage:
    return IncomingMessage(
        channel="telegram",
        chat_id="99",
        message_id="m1",
        sender_id="u1",
        sender_name="Bob",
        text=text,
        timestamp=time.time(),
        images=images or [],
    )


# ---------------------------------------------------------------------------
# _describe_images unit tests
# ---------------------------------------------------------------------------


class TestDescribeImages:
    """Direct tests of MessageHandler._describe_images."""

    def test_returns_description_on_success(self) -> None:
        vision_llm = _make_vision_llm("A red fire truck parked outside.")
        handler, _, _ = _make_handler(vision_llm=vision_llm)

        result = handler._describe_images(["data:image/png;base64,abc"], "What is this?")

        assert result == "A red fire truck parked outside."

    def test_passes_images_as_image_url_blocks(self) -> None:
        vision_llm = _make_vision_llm("A dog.")
        handler, _, _ = _make_handler(vision_llm=vision_llm)
        uri = "data:image/jpeg;base64,xyz"

        handler._describe_images([uri], "Describe it.")

        call_args = vision_llm.invoke.call_args
        messages = call_args[0][0]
        assert len(messages) == 1
        content = messages[0].content
        image_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "image_url"]
        assert len(image_blocks) == 1
        assert image_blocks[0]["image_url"]["url"] == uri

    def test_returns_none_when_response_content_is_none(self) -> None:
        vision_llm = MagicMock()
        response = MagicMock()
        response.content = None
        vision_llm.invoke.return_value = response
        handler, _, _ = _make_handler(vision_llm=vision_llm)

        result = handler._describe_images(["data:image/png;base64,a"], "")

        assert result is None

    def test_returns_none_when_response_content_is_blank(self) -> None:
        vision_llm = _make_vision_llm("   ")
        handler, _, _ = _make_handler(vision_llm=vision_llm)

        result = handler._describe_images(["data:image/png;base64,a"], "")

        assert result is None

    def test_returns_none_and_warns_on_exception(self, caplog: Any) -> None:
        vision_llm = MagicMock()
        vision_llm.invoke.side_effect = RuntimeError("network error")
        handler, _, _ = _make_handler(vision_llm=vision_llm)

        import logging

        with caplog.at_level(logging.WARNING, logger="cogtrix"):
            result = handler._describe_images(["data:image/png;base64,a"], "")

        assert result is None
        assert any("Vision model description failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# _run_agent vision dispatch scenarios
# ---------------------------------------------------------------------------


class TestRunAgentVisionDispatch:
    """Integration-level tests of the vision dispatch logic inside _run_agent."""

    def _call_run_agent(
        self, handler: MessageHandler, agent_runner: MagicMock, images: list[str], text: str = "Hi"
    ) -> None:
        """Invoke _run_agent directly with the given images and user text."""
        handler._run_agent(
            user_input=text,
            history_messages=[],
            context_prefix=None,
            effective_prompt="You are helpful.",
            active_tools=[],
            session=_make_session(),
            user_images=images,
        )

    # ------------------------------------------------------------------
    # Scenario 1: vision_llm present → description folded in, images stripped
    # ------------------------------------------------------------------

    def test_vision_llm_present_images_stripped_and_description_injected(self) -> None:
        vision_llm = _make_vision_llm("A screenshot of an error message.")
        handler, _, agent_runner = _make_handler(vision_llm=vision_llm)

        self._call_run_agent(handler, agent_runner, ["data:image/png;base64,abc"], "What is this?")

        call_kwargs = agent_runner.call_args[1]
        assert call_kwargs["user_images"] is None
        assert "[Image description: A screenshot of an error message.]" in call_kwargs["user_input"]
        assert "What is this?" in call_kwargs["user_input"]

    def test_vision_llm_received_image_content(self) -> None:
        vision_llm = _make_vision_llm("A cat.")
        handler, _, agent_runner = _make_handler(vision_llm=vision_llm)

        self._call_run_agent(handler, agent_runner, ["data:image/png;base64,xyz"], "Describe.")

        assert vision_llm.invoke.called
        call_args = vision_llm.invoke.call_args[0][0]
        content = call_args[0].content
        image_types = [b["type"] for b in content if isinstance(b, dict)]
        assert "image_url" in image_types

    # ------------------------------------------------------------------
    # Scenario 2: vision_llm raises → graceful degradation, turn proceeds
    # ------------------------------------------------------------------

    def test_vision_llm_raises_images_dropped_annotation_present_agent_still_called(
        self,
    ) -> None:
        vision_llm = MagicMock()
        vision_llm.invoke.side_effect = ConnectionError("timeout")
        handler, _, agent_runner = _make_handler(vision_llm=vision_llm)

        self._call_run_agent(handler, agent_runner, ["data:image/png;base64,abc"], "Hello")

        # Agent runner must still be called
        assert agent_runner.called
        call_kwargs = agent_runner.call_args[1]
        assert call_kwargs["user_images"] is None
        assert "could not be analyzed" in call_kwargs["user_input"]

    # ------------------------------------------------------------------
    # Scenario 3: no vision_llm + conversation_supports_vision=False
    # ------------------------------------------------------------------

    def test_no_vision_llm_text_only_model_images_dropped_annotation_present(self) -> None:
        handler, _, agent_runner = _make_handler(
            vision_llm=None,
            conversation_supports_vision=False,
        )

        self._call_run_agent(handler, agent_runner, ["data:image/png;base64,abc"], "See attached")

        call_kwargs = agent_runner.call_args[1]
        assert call_kwargs["user_images"] is None
        assert "cannot analyze images" in call_kwargs["user_input"]

    # ------------------------------------------------------------------
    # Scenario 4: no vision_llm + conversation_supports_vision=None
    # ------------------------------------------------------------------

    def test_no_vision_llm_unknown_capability_images_passed_through(self) -> None:
        handler, _, agent_runner = _make_handler(
            vision_llm=None,
            conversation_supports_vision=None,
        )
        images = ["data:image/png;base64,abc"]

        self._call_run_agent(handler, agent_runner, images, "Look")

        call_kwargs = agent_runner.call_args[1]
        assert call_kwargs["user_images"] == images

    # ------------------------------------------------------------------
    # Scenario 5: no images → no vision dispatch at all
    # ------------------------------------------------------------------

    def test_no_images_vision_llm_not_called(self) -> None:
        vision_llm = _make_vision_llm("should not be called")
        handler, _, agent_runner = _make_handler(vision_llm=vision_llm)

        self._call_run_agent(handler, agent_runner, [], "Plain text message")

        assert not vision_llm.invoke.called
        call_kwargs = agent_runner.call_args[1]
        assert call_kwargs["user_images"] is None or call_kwargs["user_images"] == []

    # ------------------------------------------------------------------
    # Scenario 6: empty user_input + vision_llm → description becomes full input
    # ------------------------------------------------------------------

    def test_empty_user_input_description_becomes_full_input(self) -> None:
        vision_llm = _make_vision_llm("A receipt from a supermarket.")
        handler, _, agent_runner = _make_handler(vision_llm=vision_llm)

        self._call_run_agent(handler, agent_runner, ["data:image/png;base64,abc"], "")

        call_kwargs = agent_runner.call_args[1]
        assert "A receipt from a supermarket." in call_kwargs["user_input"]
        # Must not start with a bare newline
        assert not call_kwargs["user_input"].startswith("\n")
