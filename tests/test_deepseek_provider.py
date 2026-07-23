"""Regression tests for the DeepSeek provider wrapper.

ROOT CAUSE: deepseek-v4-pro (and any DeepSeek reasoning model) returns a
`reasoning_content` field in assistant messages.  DeepSeek's API requires
this field to be echoed back verbatim in subsequent calls.  The failure has
two components:

  1. LangChain's _convert_dict_to_message() never extracts `reasoning_content`
     from the API response dict, so AIMessage.additional_kwargs never has it.
  2. Without it in additional_kwargs, the re-injection step in
     _get_request_payload() has nothing to inject.

FIX:
  Part A (_create_chat_result): extract reasoning_content from the raw API
  response before LangChain discards it; store it in additional_kwargs.
  Part B (_get_request_payload): walk message pairs in lockstep and inject
  reasoning_content into outgoing assistant message dicts.
"""

from __future__ import annotations


class TestDeepSeekChatModelReasoning:
    """_DeepSeekChatModel correctly preserves reasoning_content."""

    def _make_model(self) -> object:
        from src.providers.openai import _DeepSeekChatModel

        return _DeepSeekChatModel(
            model="deepseek-v4-pro",
            api_key="sk-test",
            base_url="https://api.deepseek.com/v1",
        )

    def test_reasoning_content_injected_into_assistant_dict(self) -> None:
        from unittest.mock import patch

        from langchain_core.messages import AIMessage, HumanMessage

        from src.providers.openai import _DeepSeekChatModel

        model = self._make_model()
        messages = [
            HumanMessage(content="hello"),
            AIMessage(
                content="Hi there",
                additional_kwargs={"reasoning_content": "Let me think..."},
            ),
            HumanMessage(content="what's the weather?"),
        ]

        base_payload = {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "Hi there"},
                {"role": "user", "content": "what's the weather?"},
            ],
            "model": "deepseek-v4-pro",
        }

        with patch.object(
            _DeepSeekChatModel.__bases__[0],
            "_get_request_payload",
            return_value=base_payload,
        ):
            payload = model._get_request_payload(messages)

        assistant_dict = payload["messages"][1]
        assert assistant_dict["role"] == "assistant"
        assert assistant_dict.get("reasoning_content") == "Let me think...", (
            "reasoning_content must be injected into the assistant message dict "
            "so DeepSeek's API does not return 400"
        )

    def test_no_reasoning_content_gets_empty_placeholder(self) -> None:
        """Assistant messages with no reasoning_content get an empty placeholder.

        DeepSeek requires reasoning_content on every assistant message in
        reasoning-model conversations.  Using "" satisfies the API for
        messages where the original reasoning chain is unavailable (e.g. old
        history files, /think synthetic messages).
        """
        from unittest.mock import patch

        from langchain_core.messages import AIMessage, HumanMessage

        from src.providers.openai import _DeepSeekChatModel

        model = self._make_model()
        messages = [
            HumanMessage(content="hello"),
            AIMessage(content="Hi there"),
        ]
        base_payload = {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            "model": "deepseek-v4-pro",
        }

        with patch.object(
            _DeepSeekChatModel.__bases__[0],
            "_get_request_payload",
            return_value=base_payload,
        ):
            payload = model._get_request_payload(messages)

        # Always gets the field — empty string when no rc is available
        assert payload["messages"][1].get("reasoning_content") == ""

    def test_reasoning_content_not_added_to_user_messages(self) -> None:
        from unittest.mock import patch

        from langchain_core.messages import AIMessage, HumanMessage

        from src.providers.openai import _DeepSeekChatModel

        model = self._make_model()
        # Even if a HumanMessage somehow carried reasoning_content, it should
        # not be injected (only assistant messages get it)
        messages = [
            HumanMessage(content="hello"),
            AIMessage(content="Hi"),
        ]
        base_payload = {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "Hi"},
            ],
        }

        with patch.object(
            _DeepSeekChatModel.__bases__[0],
            "_get_request_payload",
            return_value=base_payload,
        ):
            payload = model._get_request_payload(messages)

        assert "reasoning_content" not in payload["messages"][0]

    def test_multiple_reasoning_turns_all_injected(self) -> None:
        """Two AI turns with reasoning_content both get the field injected."""
        from unittest.mock import patch

        from langchain_core.messages import AIMessage, HumanMessage

        from src.providers.openai import _DeepSeekChatModel

        model = self._make_model()
        messages = [
            HumanMessage(content="q1"),
            AIMessage(content="a1", additional_kwargs={"reasoning_content": "rc1"}),
            HumanMessage(content="q2"),
            AIMessage(content="a2", additional_kwargs={"reasoning_content": "rc2"}),
            HumanMessage(content="q3"),
        ]
        base_payload = {
            "messages": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
                {"role": "user", "content": "q3"},
            ],
        }

        with patch.object(
            _DeepSeekChatModel.__bases__[0],
            "_get_request_payload",
            return_value=base_payload,
        ):
            payload = model._get_request_payload(messages)

        assert payload["messages"][1].get("reasoning_content") == "rc1"
        assert payload["messages"][3].get("reasoning_content") == "rc2"


class TestDeepSeekProviderSelection:
    """create_chat_model uses _DeepSeekChatModel for the DeepSeek endpoint."""

    def test_deepseek_url_returns_deepseek_model(self) -> None:
        from src.providers.openai import _DeepSeekChatModel, create_chat_model

        model = create_chat_model(
            model="deepseek-chat",
            api_key="sk-test",
            base_url="https://api.deepseek.com/v1",
        )
        assert isinstance(
            model, _DeepSeekChatModel
        ), "DeepSeek base_url must produce a _DeepSeekChatModel, not plain ChatOpenAI"

    def test_openai_url_returns_standard_model(self) -> None:
        from langchain_openai import ChatOpenAI

        from src.providers.openai import _DeepSeekChatModel, create_chat_model

        model = create_chat_model(model="gpt-4.1-mini", api_key="sk-test")
        assert not isinstance(model, _DeepSeekChatModel)
        assert isinstance(model, ChatOpenAI)

    def test_xai_url_returns_standard_model(self) -> None:
        from src.providers.openai import _DeepSeekChatModel, create_chat_model

        model = create_chat_model(
            model="grok-4.1-fast",
            api_key="xai-test",
            base_url="https://api.x.ai/v1",
        )
        assert not isinstance(model, _DeepSeekChatModel)

    def test_deepseek_model_is_subclass_of_chatopenai(self) -> None:
        from langchain_openai import ChatOpenAI

        from src.providers.openai import _DeepSeekChatModel

        assert issubclass(_DeepSeekChatModel, ChatOpenAI)

    # ── Regression: unsafe substring match (issue #669) ──────────────

    def test_lookalike_domain_not_selected(self) -> None:
        """Lookalike domain (api.deepseek.com.evil.com) must NOT activate DeepSeek."""
        from src.providers.openai import _DeepSeekChatModel, create_chat_model

        model = create_chat_model(
            model="gpt-4.1-mini",
            api_key="sk-test",
            base_url="https://api.deepseek.com.evil.com/v1",
        )
        assert not isinstance(model, _DeepSeekChatModel), (
            "Lookalike domain api.deepseek.com.evil.com must NOT activate "
            "_DeepSeekChatModel — hostname check must use urlparse, not substring"
        )

    def test_path_contains_deepseek_not_selected(self) -> None:
        """Path segment containing 'api.deepseek.com' must NOT activate DeepSeek."""
        from src.providers.openai import _DeepSeekChatModel, create_chat_model

        model = create_chat_model(
            model="gpt-4.1-mini",
            api_key="sk-test",
            base_url="http://evil.com/proxy/api.deepseek.com/v1",
        )
        assert not isinstance(model, _DeepSeekChatModel), (
            "Path segment /proxy/api.deepseek.com/ must NOT activate "
            "_DeepSeekChatModel — hostname check must use urlparse, not substring"
        )

    def test_base_url_none_standard_model(self) -> None:
        """base_url=None must return standard ChatOpenAI, not _DeepSeekChatModel."""
        from src.providers.openai import _DeepSeekChatModel, create_chat_model

        model = create_chat_model(model="gpt-4.1-mini", api_key="sk-test")
        assert not isinstance(
            model, _DeepSeekChatModel
        ), "base_url=None must NOT produce _DeepSeekChatModel"

    def test_is_deepseek_base_url_helper(self) -> None:
        """_is_deepseek_base_url returns True only for known DeepSeek hostnames."""
        from src.providers.openai import _is_deepseek_base_url

        # Legitimate DeepSeek URLs
        assert _is_deepseek_base_url("https://api.deepseek.com/v1") is True
        assert _is_deepseek_base_url("http://api.deepseek.com/") is True
        assert _is_deepseek_base_url("https://api.deepseek.com") is True

        # Lookalike / non-DeepSeek URLs
        assert _is_deepseek_base_url("https://api.deepseek.com.evil.com/v1") is False
        assert _is_deepseek_base_url("http://evil.com/proxy/api.deepseek.com/") is False
        assert _is_deepseek_base_url("https://api.openai.com/v1") is False
        assert _is_deepseek_base_url("https://api.x.ai/v1") is False

        # Edge cases
        assert _is_deepseek_base_url(None) is False
        assert _is_deepseek_base_url("") is False
        assert _is_deepseek_base_url("not-even-a-url") is False


class TestDeepSeekChatModelCapture:
    """_DeepSeekChatModel._create_chat_result captures reasoning_content from API responses."""

    _FAKE_USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    _BASE = {
        "model": "deepseek-v4-pro",
        "id": "test-id",
        "object": "chat.completion",
        "created": 1700000000,
    }

    def _make_model(self) -> object:
        from src.providers.openai import _DeepSeekChatModel

        return _DeepSeekChatModel(
            model="deepseek-v4-pro",
            api_key="sk-test",
            base_url="https://api.deepseek.com/v1",
        )

    def _fake_response(self, *, reasoning_content: str | None = None) -> dict:
        msg: dict = {"role": "assistant", "content": "Hi there"}
        if reasoning_content:
            msg["reasoning_content"] = reasoning_content
        return {
            **self._BASE,
            "usage": self._FAKE_USAGE,
            "choices": [{"message": msg, "finish_reason": "stop", "index": 0}],
        }

    def test_reasoning_content_stored_in_additional_kwargs(self) -> None:
        model = self._make_model()
        result = model._create_chat_result(self._fake_response(reasoning_content="Let me think..."))
        ai_msg = result.generations[0].message
        assert ai_msg.additional_kwargs.get("reasoning_content") == "Let me think...", (
            "reasoning_content must be stored in AIMessage.additional_kwargs so "
            "_get_request_payload can re-inject it on the next API call"
        )

    def test_no_reasoning_content_no_additional_kwargs_pollution(self) -> None:
        model = self._make_model()
        result = model._create_chat_result(self._fake_response())
        assert "reasoning_content" not in result.generations[0].message.additional_kwargs

    def test_multiple_choices_each_get_correct_reasoning_content(self) -> None:
        model = self._make_model()
        response = {
            **self._BASE,
            "usage": self._FAKE_USAGE,
            "choices": [
                {
                    "message": {"role": "assistant", "content": "A", "reasoning_content": "rc_A"},
                    "finish_reason": "stop",
                    "index": 0,
                },
                {
                    "message": {"role": "assistant", "content": "B", "reasoning_content": "rc_B"},
                    "finish_reason": "stop",
                    "index": 1,
                },
            ],
        }
        result = model._create_chat_result(response)
        assert result.generations[0].message.additional_kwargs.get("reasoning_content") == "rc_A"
        assert result.generations[1].message.additional_kwargs.get("reasoning_content") == "rc_B"

    def test_full_round_trip_capture_then_reinject(self) -> None:
        """End-to-end: capture from response → stored → re-injected on next call."""
        from unittest.mock import patch

        from langchain_core.messages import HumanMessage

        from src.providers.openai import _DeepSeekChatModel

        model = self._make_model()

        # Step 1 — simulate receiving a tool_calls response with reasoning_content
        fake_response = {
            **self._BASE,
            "usage": self._FAKE_USAGE,
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Let me search",
                        "reasoning_content": "I need to think about this",
                    },
                    "finish_reason": "tool_calls",
                    "index": 0,
                }
            ],
        }
        result = model._create_chat_result(fake_response)
        ai_msg = result.generations[0].message
        assert (
            "reasoning_content" in ai_msg.additional_kwargs
        ), "Step 1 (capture) failed: reasoning_content not stored in additional_kwargs"

        # Step 2 — build next turn with that AIMessage in history
        messages = [
            HumanMessage(content="What's the weather?"),
            ai_msg,
            HumanMessage(content="<tool result: sunny, 28°C>"),
        ]
        base_payload = {
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": "Let me search"},
                {"role": "user", "content": "<tool result: sunny, 28°C>"},
            ],
        }
        with patch.object(
            _DeepSeekChatModel.__bases__[0], "_get_request_payload", return_value=base_payload
        ):
            payload = model._get_request_payload(messages)

        assert (
            payload["messages"][1].get("reasoning_content") == "I need to think about this"
        ), "Step 2 (reinject) failed: reasoning_content not present in outgoing assistant dict"


# ── C2: Streaming path captures reasoning_content ────────────────────────────


class TestDeepSeekStreamingCapture:
    """_convert_chunk_to_generation_chunk captures reasoning_content from streaming deltas."""

    def _make_model(self) -> object:
        from src.providers.openai import _DeepSeekChatModel

        return _DeepSeekChatModel(
            model="deepseek-v4-pro",
            api_key="sk-test",
            base_url="https://api.deepseek.com/v1",
        )

    def test_streaming_chunk_reasoning_content_captured(self) -> None:
        """reasoning_content is captured from choices[0].delta — NOT from top-level chunk."""
        from langchain_core.messages import AIMessageChunk

        model = self._make_model()
        # Real OpenAI streaming format: delta is nested under choices[0]
        chunk = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "Hi",
                        "reasoning_content": "Let me think",
                    },
                    "finish_reason": None,
                }
            ],
            "model": "deepseek-reasoner",
        }
        result = model._convert_chunk_to_generation_chunk(chunk, AIMessageChunk, {})
        if result is not None:
            assert (
                result.message.additional_kwargs.get("reasoning_content") == "Let me think"
            ), "reasoning_content from streaming delta must be stored in AIMessageChunk.additional_kwargs"

    def test_streaming_chunk_no_reasoning_no_pollution(self) -> None:
        from langchain_core.messages import AIMessageChunk

        model = self._make_model()
        chunk = {
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hi"},
                    "finish_reason": None,
                }
            ],
            "model": "deepseek-reasoner",
        }
        result = model._convert_chunk_to_generation_chunk(chunk, AIMessageChunk, {})
        if result is not None:
            assert "reasoning_content" not in result.message.additional_kwargs

    def test_streaming_result_none_handled_gracefully(self) -> None:
        """If parent returns None, override must also return None without error."""
        from unittest.mock import patch

        from langchain_core.messages import AIMessageChunk

        from src.providers.openai import _DeepSeekChatModel

        model = self._make_model()
        with patch.object(
            _DeepSeekChatModel.__bases__[0],
            "_convert_chunk_to_generation_chunk",
            return_value=None,
        ):
            result = model._convert_chunk_to_generation_chunk({}, AIMessageChunk, {})
        assert result is None


# ── H3: _set_provider_key security warning ───────────────────────────────────


class TestSetProviderKeySecurityWarning:
    def test_warns_when_base_url_differs_from_preset(self, caplog) -> None:
        import logging

        from src.config import Config, ProviderConfig, _set_provider_key

        cfg = Config()
        cfg.providers["deepseek"] = ProviderConfig(
            name="deepseek", type="openai", base_url="https://attacker.example.com/v1"
        )
        with caplog.at_level(logging.WARNING):
            _set_provider_key(cfg, "deepseek", "sk-real-key")
        assert (
            "SECURITY" in caplog.text
        ), f"Expected SECURITY warning in caplog, got: {caplog.text!r}"

    def test_no_warning_when_base_url_matches_preset(self, caplog) -> None:
        import logging

        from src.config import Config, ProviderConfig, _set_provider_key

        cfg = Config()
        cfg.providers["deepseek"] = ProviderConfig(
            name="deepseek", type="openai", base_url="https://api.deepseek.com/v1"
        )
        with caplog.at_level(logging.WARNING):
            _set_provider_key(cfg, "deepseek", "sk-real-key")
        assert "SECURITY" not in caplog.text

    def test_no_warning_for_new_provider_creation(self, caplog) -> None:
        import logging

        from src.config import Config, _set_provider_key

        cfg = Config()  # no existing deepseek provider
        with caplog.at_level(logging.WARNING):
            _set_provider_key(cfg, "deepseek", "sk-real-key")
        assert "SECURITY" not in caplog.text


# ── M4: Responses API payload "input" key ────────────────────────────────────


class TestResponsesApiPayloadInjection:
    def test_responses_api_input_key_also_injected(self) -> None:
        """reasoning_content is injected into payload['input'] for Responses API path."""
        from unittest.mock import patch

        from langchain_core.messages import AIMessage, HumanMessage

        from src.providers.openai import _DeepSeekChatModel

        model = _DeepSeekChatModel(
            model="deepseek-v4-pro", api_key="sk-test", base_url="https://api.deepseek.com/v1"
        )
        messages = [
            HumanMessage(content="hello"),
            AIMessage(content="Hi", additional_kwargs={"reasoning_content": "rc1"}),
        ]
        base_payload = {
            "input": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "Hi"},
            ]
        }
        with patch.object(
            _DeepSeekChatModel.__bases__[0], "_get_request_payload", return_value=base_payload
        ):
            payload = model._get_request_payload(messages)

        assert (
            payload["input"][1].get("reasoning_content") == "rc1"
        ), "reasoning_content must be injected into payload['input'] for Responses API"

    def test_messages_path_still_works_when_both_keys_absent(self) -> None:
        from unittest.mock import patch

        from langchain_core.messages import AIMessage, HumanMessage

        from src.providers.openai import _DeepSeekChatModel

        model = _DeepSeekChatModel(
            model="deepseek-v4-pro", api_key="sk-test", base_url="https://api.deepseek.com/v1"
        )
        messages = [HumanMessage(content="hello"), AIMessage(content="Hi")]
        # Payload with neither key
        base_payload = {"model": "deepseek-v4-pro"}
        with patch.object(
            _DeepSeekChatModel.__bases__[0], "_get_request_payload", return_value=base_payload
        ):
            payload = model._get_request_payload(messages)
        # Should not crash and should return the payload unchanged
        assert payload == base_payload

    def test_empty_rc_placeholder_when_some_messages_missing_rc(self) -> None:
        """When some AIMessages have rc and others don't, missing ones get empty placeholder.

        This covers history loaded from old JSON files that pre-date rc serialization.
        Without this, DeepSeek returns 400 because the conversation is 'in thinking mode'
        but some assistant messages are missing reasoning_content.
        """
        from unittest.mock import patch

        from langchain_core.messages import AIMessage, HumanMessage

        from src.providers.openai import _DeepSeekChatModel

        model = _DeepSeekChatModel(
            model="deepseek-reasoner", api_key="sk-test", base_url="https://api.deepseek.com/v1"
        )
        messages = [
            HumanMessage(content="q1"),
            AIMessage(content="a1"),  # old message — no reasoning_content
            HumanMessage(content="q2"),
            AIMessage(content="a2", additional_kwargs={"reasoning_content": "rc2"}),
            HumanMessage(content="q3"),
        ]
        base_payload = {
            "messages": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
                {"role": "user", "content": "q3"},
            ]
        }
        with patch.object(
            _DeepSeekChatModel.__bases__[0], "_get_request_payload", return_value=base_payload
        ):
            payload = model._get_request_payload(messages)

        # old message gets empty placeholder so DeepSeek doesn't reject the request
        assert payload["messages"][1].get("reasoning_content") == ""
        # new message gets its actual reasoning content
        assert payload["messages"][3].get("reasoning_content") == "rc2"

    def test_all_assistant_messages_get_placeholder_even_without_any_rc(self) -> None:
        """Even when NO messages carry reasoning_content, all assistant dicts get ''.

        This covers sessions loaded from old JSON history (pre-rc-serialization) and
        /think synthetic messages — both have no rc in additional_kwargs, yet the
        DeepSeek reasoning model requires the field on every assistant message.
        """
        from unittest.mock import patch

        from langchain_core.messages import AIMessage, HumanMessage

        from src.providers.openai import _DeepSeekChatModel

        model = _DeepSeekChatModel(
            model="deepseek-chat", api_key="sk-test", base_url="https://api.deepseek.com/v1"
        )
        messages = [HumanMessage(content="q"), AIMessage(content="a")]
        base_payload = {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ]
        }
        with patch.object(
            _DeepSeekChatModel.__bases__[0], "_get_request_payload", return_value=base_payload
        ):
            payload = model._get_request_payload(messages)

        assert payload["messages"][1].get("reasoning_content") == ""
        assert "reasoning_content" not in payload["messages"][0]
