"""Tests for src/orchestration/phases.py."""

from unittest.mock import MagicMock, patch

import pytest

from src.orchestration.phases import build_llm_for_decomposition


class TestBuildLlmForDecomposition:
    def _make_config(self, provider_type: str = "openai") -> MagicMock:
        provider_cfg = MagicMock()
        provider_cfg.type = provider_type
        provider_cfg.api_key = None
        provider_cfg.base_url = None
        provider_cfg.num_ctx = None
        provider_cfg.max_tokens = None
        provider_cfg.get_model.return_value = "gpt-4.1-mini"
        provider_cfg.get_base_url.return_value = None

        config = MagicMock()
        config.get_provider_config.return_value = provider_cfg
        return config

    def test_calls_create_chat_model(self):
        config = self._make_config("openai")
        fake_llm = MagicMock()

        with patch("src.providers.create_chat_model", return_value=fake_llm):
            result = build_llm_for_decomposition(config)

        assert result is fake_llm

    def test_passes_temperature_03(self):
        config = self._make_config("ollama")
        fake_llm = MagicMock()

        with patch("src.providers.create_chat_model", return_value=fake_llm) as mock_ccm:
            build_llm_for_decomposition(config)
            _, kwargs = mock_ccm.call_args
            assert kwargs.get("temperature") == pytest.approx(0.3)

    def test_returns_none_on_exception(self):
        config = MagicMock()
        config.get_provider_config.side_effect = RuntimeError("boom")

        result = build_llm_for_decomposition(config)
        assert result is None

    def test_returns_none_on_import_error(self):
        config = self._make_config("anthropic")

        with patch("src.providers.create_chat_model", side_effect=ImportError("no pkg")):
            result = build_llm_for_decomposition(config)

        assert result is None

    @pytest.mark.parametrize("provider_type", ["openai", "ollama", "anthropic", "google"])
    def test_supports_all_provider_types(self, provider_type):
        config = self._make_config(provider_type)
        fake_llm = MagicMock()

        with patch("src.providers.create_chat_model", return_value=fake_llm) as mock_ccm:
            result = build_llm_for_decomposition(config)

        assert result is fake_llm
        call_args = mock_ccm.call_args
        assert call_args[0][0] == provider_type


class TestExtractTurnMessages:
    """Tests for extract_turn_messages."""

    def test_returns_empty_for_no_messages(self):
        from src.orchestration.phases import extract_turn_messages

        assert extract_turn_messages([]) == []

    def test_returns_messages_after_last_human(self):
        from src.orchestration.phases import extract_turn_messages

        try:
            from langchain_core.messages import AIMessage, HumanMessage
        except ImportError:
            pytest.skip("langchain_core required")
        h1 = HumanMessage(content="first")
        a1 = AIMessage(content="resp1")
        h2 = HumanMessage(content="second")
        a2 = AIMessage(content="resp2")
        msgs = [h1, a1, h2, a2]
        result = extract_turn_messages(msgs)
        assert result == [a2]

    def test_boundary_anchors_to_specific_message(self):
        from src.orchestration.phases import extract_turn_messages

        try:
            from langchain_core.messages import AIMessage, HumanMessage
        except ImportError:
            pytest.skip("langchain_core required")
        h1 = HumanMessage(content="prompt")
        a1 = AIMessage(content="analysis")
        h2 = HumanMessage(content="exec prompt")
        a2 = AIMessage(content="exec result")
        msgs = [h1, a1, h2, a2]
        result = extract_turn_messages(msgs, boundary=h1)
        assert result == [a1, h2, a2]

    def test_boundary_identity_not_equality(self):
        """Two HumanMessages with identical content are distinguished by identity."""
        from src.orchestration.phases import extract_turn_messages

        try:
            from langchain_core.messages import AIMessage, HumanMessage
        except ImportError:
            pytest.skip("langchain_core required")
        h1 = HumanMessage(content="same text")
        a1 = AIMessage(content="resp1")
        h2 = HumanMessage(content="same text")  # same content, different object
        a2 = AIMessage(content="resp2")
        msgs = [h1, a1, h2, a2]
        assert extract_turn_messages(msgs) == [a2]
        assert extract_turn_messages(msgs, boundary=h1) == [a1, h2, a2]
        assert extract_turn_messages(msgs, boundary=h2) == [a2]

    def test_no_human_message_returns_empty(self):
        from src.orchestration.phases import extract_turn_messages

        try:
            from langchain_core.messages import AIMessage
        except ImportError:
            pytest.skip("langchain_core required")
        msgs = [AIMessage(content="orphan")]
        assert extract_turn_messages(msgs) == []

    def test_boundary_not_found_returns_empty(self):
        from src.orchestration.phases import extract_turn_messages

        try:
            from langchain_core.messages import AIMessage, HumanMessage
        except ImportError:
            pytest.skip("langchain_core required")
        h1 = HumanMessage(content="in list")
        a1 = AIMessage(content="resp")
        h_missing = HumanMessage(content="not in list")
        msgs = [h1, a1]
        assert extract_turn_messages(msgs, boundary=h_missing) == []
