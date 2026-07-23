"""Regression tests for DeepSeek reasoning_content round-trip (issue #1391).

DeepSeek thinking-mode models (deepseek-reasoner / deepseek-v4-flash) require
the assistant's `reasoning_content` field to be echoed back verbatim in the
next API call.  LangChain's ChatOpenAI serialisation drops it silently, causing
HTTP 400 on turn ≥ 2.

These tests verify that:
1. _DeepSeekChatModel is selected for api.deepseek.com via the provider factory.
2. reasoning_content is captured from the API response and stored in additional_kwargs.
3. reasoning_content is re-injected into outgoing request payloads on the next turn.
4. The evaluation runner's _build_llm() routes through the provider factory.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.memory.json_store import _dict_to_message, _message_to_dict
from src.providers import create_chat_model
from src.providers.openai import _DeepSeekChatModel, _is_deepseek_base_url


class TestDeepSeekReasoningChatModelSelection:
    """Verify _DeepSeekChatModel is selected for DeepSeek API endpoints."""

    def test_deepseek_base_url_selects_deepseek_chat_model(self):
        """create_chat_model returns _DeepSeekChatModel for api.deepseek.com."""
        llm = create_chat_model(
            "openai",
            model="deepseek-reasoner",
            api_key="sk-test",
            base_url="https://api.deepseek.com/v1",
        )
        # create_chat_model wraps in RetryableChatModel — check the inner model
        inner = getattr(llm, "_model", llm)
        assert isinstance(
            inner, _DeepSeekChatModel
        ), f"Expected _DeepSeekChatModel for api.deepseek.com, got {type(inner).__name__}"

    def test_non_deepseek_base_url_selects_plain_chat_openai(self):
        """OpenAI-compatible endpoints do NOT get the DeepSeek subclass."""
        llm = create_chat_model(
            "openai",
            model="gpt-4o",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
        )
        # create_chat_model wraps in RetryableChatModel — check the inner model
        inner = getattr(llm, "_model", llm)
        assert (
            type(inner).__name__ == "ChatOpenAI"
        ), f"Expected plain ChatOpenAI for api.openai.com, got {type(inner).__name__}"
        assert not isinstance(
            inner, _DeepSeekChatModel
        ), "api.openai.com should NOT use _DeepSeekChatModel"

    def test_is_deepseek_base_url_true_for_api_deepseek(self):
        assert _is_deepseek_base_url("https://api.deepseek.com/v1") is True
        assert _is_deepseek_base_url("https://api.deepseek.com") is True

    def test_is_deepseek_base_url_false_for_lookalikes(self):
        assert _is_deepseek_base_url("https://api.deepseek.com.evil.com/v1") is False
        assert _is_deepseek_base_url("https://deepseek.evil.com/v1") is False


class TestDeepSeekReasoningContentCapture:
    """Verify reasoning_content is captured from API responses."""

    def test_create_chat_result_captures_reasoning_content_from_response_dict(self):
        """Part A: _create_chat_result extracts reasoning_content from raw response."""
        llm = _DeepSeekChatModel(
            model="deepseek-reasoner",
            api_key="sk-test",
            base_url="https://api.deepseek.com/v1",
        )

        mock_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Final answer here.",
                        "reasoning_content": "Let me think step by step...",
                    }
                }
            ]
        }

        result = llm._create_chat_result(mock_response)

        assert len(result.generations) == 1
        gen = result.generations[0]
        assert hasattr(gen, "message")
        rc = gen.message.additional_kwargs.get("reasoning_content")
        assert (
            rc == "Let me think step by step..."
        ), f"Expected reasoning_content in additional_kwargs, got {rc!r}"

    def test_create_chat_result_captures_reasoning_content_from_streaming_chunk(self):
        """Part C: _convert_chunk_to_generation_chunk extracts reasoning_content from delta."""
        llm = _DeepSeekChatModel(
            model="deepseek-reasoner",
            api_key="sk-test",
            base_url="https://api.deepseek.com/v1",
        )

        mock_chunk = {
            "choices": [
                {
                    "delta": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "Thinking process...",
                    }
                }
            ]
        }

        result = llm._convert_chunk_to_generation_chunk(mock_chunk, MagicMock(), None)

        assert result is not None
        assert hasattr(result, "message")
        rc = result.message.additional_kwargs.get("reasoning_content")
        assert (
            rc == "Thinking process..."
        ), f"Expected reasoning_content in streaming additional_kwargs, got {rc!r}"


class TestDeepSeekReasoningContentReinject:
    """Verify reasoning_content is re-injected into outgoing request payloads."""

    def test_get_request_payload_injects_reasoning_content_for_assistant_messages(self):
        """Part B: _get_request_payload re-injects reasoning_content into assistant dicts."""
        llm = _DeepSeekChatModel(
            model="deepseek-reasoner",
            api_key="sk-test",
            base_url="https://api.deepseek.com/v1",
        )

        from langchain_core.messages import AIMessage, HumanMessage

        # First turn: assistant message with reasoning_content stored in additional_kwargs
        assistant_with_rc = AIMessage(
            content="Let me search for that.",
            additional_kwargs={"reasoning_content": "I need to search the web for current info."},
        )
        human_msg = HumanMessage(content="What's the latest on X?")

        input_messages = [human_msg, assistant_with_rc]

        payload = llm._get_request_payload(input_messages)

        # Find the assistant message dict in the payload
        msgs = payload.get("messages", [])
        assistant_dicts = [m for m in msgs if isinstance(m, dict) and m.get("role") == "assistant"]
        assert len(assistant_dicts) >= 1

        # The assistant dict should have reasoning_content at the top level
        # (where DeepSeek expects it, not buried in additional_kwargs)
        assistant_dict = assistant_dicts[0]
        rc_injected = assistant_dict.get("reasoning_content")
        assert (
            rc_injected is not None
        ), f"Expected reasoning_content in assistant message dict payload, got keys {list(assistant_dict.keys())}"
        assert rc_injected == "I need to search the web for current info."

    def test_get_request_payload_fills_empty_string_when_reasoning_content_missing(self):
        """When an assistant message lacks reasoning_content, fill with '' to satisfy DeepSeek."""
        llm = _DeepSeekChatModel(
            model="deepseek-reasoner",
            api_key="sk-test",
            base_url="https://api.deepseek.com/v1",
        )

        from langchain_core.messages import AIMessage, HumanMessage

        # Assistant message WITHOUT reasoning_content (plain AIMessage from legacy path)
        plain_assistant = AIMessage(content="Here's the answer.")
        human_msg = HumanMessage(content="What is 2+2?")

        input_messages = [human_msg, plain_assistant]

        payload = llm._get_request_payload(input_messages)

        msgs = payload.get("messages", [])
        assistant_dicts = [m for m in msgs if isinstance(m, dict) and m.get("role") == "assistant"]
        assert len(assistant_dicts) >= 1

        # Even without original reasoning_content, the field should be present (empty string)
        assistant_dict = assistant_dicts[0]
        rc_present = "reasoning_content" in assistant_dict
        assert (
            rc_present
        ), f"Expected reasoning_content key (even if empty) in plain assistant dict, got keys {list(assistant_dict.keys())}"


class TestDeepSeekJsonStoreRoundTrip:
    """Verify reasoning_content survives JSON store save/load cycle."""

    def test_message_to_dict_preserves_reasoning_content(self):
        """json_store._message_to_dict serializes reasoning_content."""
        from langchain_core.messages import AIMessage

        msg = AIMessage(
            content="Here's the result.",
            additional_kwargs={"reasoning_content": "Analysis process..."},
        )

        d = _message_to_dict(msg)

        assert (
            "reasoning_content" in d
        ), f"Expected reasoning_content in serialized dict, got keys {list(d.keys())}"
        assert d["reasoning_content"] == "Analysis process..."

    def test_message_to_dict_truncates_long_reasoning_content(self):
        """Long reasoning_content (>8192 chars) is truncated to avoid huge JSON files."""
        from langchain_core.messages import AIMessage

        long_rc = "x" * 10000
        msg = AIMessage(
            content="Short answer.",
            additional_kwargs={"reasoning_content": long_rc},
        )

        d = _message_to_dict(msg)

        rc_in_dict = d.get("reasoning_content", "")
        assert len(rc_in_dict) <= 8192 + len(
            " … [truncated]"
        ), f"Expected truncation at 8192 chars, got {len(rc_in_dict)}"
        assert "… [truncated]" in rc_in_dict

    def test_dict_to_message_restores_reasoning_content(self):
        """json_store._dict_to_message restores reasoning_content to additional_kwargs."""
        from langchain_core.messages import AIMessage

        data = {
            "type": "ai",
            "content": "Answer here.",
            "reasoning_content": "Restored reasoning chain.",
        }

        msg = _dict_to_message(data)

        assert isinstance(msg, AIMessage)
        rc = msg.additional_kwargs.get("reasoning_content")
        assert (
            rc == "Restored reasoning chain."
        ), f"Expected reasoning_content restored from dict, got {rc!r}"


class TestDeepSeekBaseUrlRedaction:
    """Regression tests for #1106 — _is_deepseek_base_url must redact credentials in logs."""

    def test_redact_url_strips_userinfo(self):
        """_redact_url removes username:password from URL netloc."""
        from src.providers import _redact_url

        redacted = _redact_url("https://user:pass@api.deepseek.com/v1")
        assert "user:pass@" not in redacted
        assert redacted == "https://api.deepseek.com/v1"

    def test_redact_url_strips_sensitive_query_params(self):
        """_redact_url replaces sensitive query params with [redacted]."""
        from src.providers import _redact_url

        redacted = _redact_url("https://api.deepseek.com/v1?api_key=sk-secret123")
        assert "sk-secret123" not in redacted
        assert "api_key=[redacted]" in redacted

    def test_is_deepseek_warning_redacts_no_hostname_url_with_userinfo(self, caplog):
        """No-hostname warning path must not leak raw credentials (#1106).

        Note: the ``except Exception`` parse-failure branch is effectively
        unreachable for valid string input (``urlparse`` is very tolerant),
        so this test covers the no-hostname branch — which is the one that
        fires in practice when a misconfigured URL has userinfo but no host
        segment.
        """
        import logging

        from src.providers.openai import _is_deepseek_base_url

        with caplog.at_level(logging.WARNING, logger="cogtrix.providers.openai"):
            result = _is_deepseek_base_url("https://user:pass@/bad-url")

        assert result is False
        assert "user:pass@" not in caplog.text
        assert "***@" not in caplog.text  # _redact_url strips userinfo entirely
        # Positive assertion: redacted URL does appear in the log
        assert "<unparseable URL>" in caplog.text

    def test_is_deepseek_warning_redacts_no_hostname_url(self, caplog):
        """No-hostname warning must not leak raw credentials (#1106)."""
        import logging

        from src.providers.openai import _is_deepseek_base_url

        with caplog.at_level(logging.WARNING, logger="cogtrix.providers.openai"):
            result = _is_deepseek_base_url("https://user:pass@/")

        assert result is False
        assert "user:pass@" not in caplog.text
        # Positive assertion: redacted URL does appear in the log
        assert "<unparseable URL>" in caplog.text

    def test_is_deepseek_warning_redacts_on_parse_exception(self, caplog, monkeypatch):
        """Parse-failure warning path must not leak raw credentials (#1106)."""
        import logging

        from src.providers import openai as _o

        def _boom(_url):
            raise ValueError("boom")

        monkeypatch.setattr(_o, "urlparse", _boom)
        with caplog.at_level(logging.WARNING, logger="cogtrix.providers.openai"):
            result = _o._is_deepseek_base_url("https://user:pass@api.deepseek.com/v1")

        assert result is False
        assert "user:pass" not in caplog.text
        # Verify the redacted URL DOES appear (not just that the raw one is absent)
        assert "api.deepseek.com" in caplog.text
        # Verify exception class name is logged for debuggability
        assert "ValueError" in caplog.text


class TestRunnerDeepSeekRouting:
    """Verify the evaluation runner uses the provider factory for deepseek models."""

    def test_runner_build_llm_routes_deepseek_through_factory(self):
        """The runner's _build_llm() must use create_chat_model for deepseek."""
        from tests.evaluation.runner import _build_llm

        # Create a minimal model config for deepseek
        model = MagicMock()
        model.id = "deepseek-v4-flash"
        model.provider = "deepseek"
        model.env_key = "DEEPSEEK_API_KEY"
        model.model_id = "deepseek-reasoner"
        model.openrouter_model_id = None
        model.temperature = None

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test-key"}):
            llm = _build_llm(model, active_key=None)

        # The returned LLM is a RetryableChatModel wrapping _DeepSeekChatModel
        inner = getattr(llm, "_model", llm)
        assert isinstance(inner, _DeepSeekChatModel), (
            f"Expected _DeepSeekChatModel via factory, got {type(inner).__name__}. "
            "The runner must use src.providers.create_chat_model for deepseek, "
            "not a plain ChatOpenAI instantiation."
        )
