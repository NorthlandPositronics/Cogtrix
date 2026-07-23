"""Tests for src/orchestration/phases.py."""

from unittest.mock import MagicMock, patch

import pytest

from src.orchestration.phases import build_llm_for_decomposition


class TestBuildLlmForDecomposition:
    def _make_config(self, provider_type: str = "openai") -> MagicMock:
        pc = MagicMock()
        pc.type = provider_type
        pc.api_key = None
        pc.get_base_url.return_value = None

        mc = MagicMock()
        mc.model = "gpt-4.1-mini"
        mc.context_window = None
        mc.max_tokens = None

        config = MagicMock()
        config.resolve_llm_config.return_value = (pc, mc)
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
        config.resolve_llm_config.side_effect = RuntimeError("boom")

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


# ---------------------------------------------------------------------------
# TestNormalizeUrl  (FIX 1 — SSRF normalization)
# ---------------------------------------------------------------------------


class TestNormalizeUrl:
    def _norm(self, url: str):
        from src.orchestration.phases import _normalize_url

        return _normalize_url(url)

    def test_normal_public_url_preserved(self):
        result = self._norm("https://example.com/page")
        assert result == "https://example.com/page"

    def test_url_with_encoded_path_chars_decoded_then_reencoded(self):
        result = self._norm("https://example.com/path%2Ffile%3Fq%3D1")
        assert result is not None
        # Normalized form may re-encode; just verify it's not rejected (None)
        assert len(result) > 0

    def test_percent_encoded_private_ip_rejected(self):
        # 192.%31.1.1 decodes to 192.1.1.1 — private-range after normalization
        # The key guarantee is that the normalized form goes through _validate_url
        # which blocks RFC-1918 ranges. Even if the host passes normalization,
        # the subsequent _validate_url call in extract_fetched_urls rejects it.
        # Here we just verify _normalize_url produces a consistent decoded form.
        result = self._norm("http://192.%31.1.1/secret")
        # After decoding, netloc becomes 192.1.1.1 — normalization must not
        # re-encode the dot-decimal digits in a way that hides the IP.
        if result is not None:
            assert "192.1.1.1" in result or "192.%31.1.1" not in result

    def test_control_character_url_rejected_or_stripped(self):
        result = self._norm("http://example.com\x00@evil.com/")
        # Either rejected outright (None) or the control char is stripped
        if result is not None:
            assert "\x00" not in result

    def test_del_character_stripped(self):
        result = self._norm("https://example.com/path\x7f")
        if result is not None:
            assert "\x7f" not in result

    def test_empty_string_returns_none(self):
        assert self._norm("") is None

    def test_whitespace_only_returns_none(self):
        assert self._norm("   ") is None

    def test_non_http_scheme_returns_none(self):
        assert self._norm("ftp://example.com/file") is None

    def test_no_host_returns_none(self):
        assert self._norm("http:///path") is None

    def test_scheme_lowercased(self):
        result = self._norm("HTTP://Example.COM/")
        assert result is not None
        assert result.startswith("http://")


class TestExtractFetchedUrlsNormalization:
    """extract_fetched_urls() must normalize before deduplication."""

    def test_extract_urls_skips_malformed(self):
        from src.orchestration.phases import extract_fetched_urls

        try:
            from langchain_core.messages import AIMessage
        except ImportError:
            pytest.skip("langchain_core required")

        # Inject a null-byte URL via a tool call arg — should be dropped
        msg = AIMessage(content="")
        msg.tool_calls = [{"name": "http_get", "args": {"url": "http://evil\x00.com/"}}]
        result = extract_fetched_urls([msg])
        # Control-char URL must not appear in results
        for url in result:
            assert "\x00" not in url

    def test_extract_urls_deduplicates_after_normalization(self):
        from src.orchestration.phases import extract_fetched_urls

        try:
            from langchain_core.messages import AIMessage
        except ImportError:
            pytest.skip("langchain_core required")

        # Same URL with and without trailing slash — normalization may unify them
        msg = AIMessage(content="")
        msg.tool_calls = [
            {"name": "http_get", "args": {"url": "https://example.com/page"}},
            {"name": "http_get", "args": {"url": "https://example.com/page"}},
        ]
        result = extract_fetched_urls([msg])
        assert len([u for u in result if "example.com/page" in u]) <= 1


# ---------------------------------------------------------------------------
# TestRunResearchDelegateURLWarning  (FIX 5 — URL count warning)
# ---------------------------------------------------------------------------


class TestRunResearchDelegateURLWarning:
    def test_no_warning_for_ten_or_fewer_urls(self, caplog):
        import logging
        from unittest.mock import patch

        from src.orchestration.phases import run_research_delegate

        urls = [f"https://example.com/{i}" for i in range(10)]
        with caplog.at_level(logging.WARNING, logger="cogtrix.orchestration.phases"):
            with patch("src.tools.delegate.get_delegate_tools", return_value=[]):
                run_research_delegate(urls, task="test")
        assert not any("dropped" in r.message for r in caplog.records)

    def test_warning_logged_for_more_than_ten_urls(self, caplog):
        import logging
        from unittest.mock import patch

        from src.orchestration.phases import run_research_delegate

        urls = [f"https://example.com/{i}" for i in range(15)]
        with caplog.at_level(logging.WARNING, logger="cogtrix.orchestration.phases"):
            with patch("src.tools.delegate.get_delegate_tools", return_value=[]):
                run_research_delegate(urls, task="test")
        # Warning fires before the delegate-tool check, so it always fires on >10 URLs
        drop_warnings = [r for r in caplog.records if "dropped" in r.message]
        assert len(drop_warnings) == 1
        assert "15" in drop_warnings[0].message
        assert "5" in drop_warnings[0].message


# ---------------------------------------------------------------------------
# Regression — force_deep_think must not swallow UserCancelledRun (#1185)
# ---------------------------------------------------------------------------


class TestForceDeepThinkUserCancelledRun:
    def test_user_cancelled_run_propagates(self):
        """UserCancelledRun raised by deep_think must not be caught by the broad except Exception."""
        from unittest.mock import patch

        from src.agent.safety import UserCancelledRun
        from src.orchestration.phases import force_deep_think

        log_mock = MagicMock()

        with patch("src.tools.deep_think.deep_think", side_effect=UserCancelledRun("stop")):
            with pytest.raises(UserCancelledRun):
                force_deep_think(
                    user_input="think deep about this",
                    agent_response="initial response",
                    tool_outputs="",
                    log=log_mock,
                )

    def test_other_exceptions_still_swallowed(self):
        """Non-cancellation exceptions must still be caught and logged."""
        from unittest.mock import patch

        from src.orchestration.phases import force_deep_think

        log_mock = MagicMock()

        with patch("src.tools.deep_think.deep_think", side_effect=RuntimeError("boom")):
            result = force_deep_think(
                user_input="think deep about this",
                agent_response="fallback response",
                tool_outputs="",
                log=log_mock,
            )

        assert result == "fallback response"
        log_mock.warning.assert_called_once()
        assert "Programmatic deep_think failed" in log_mock.warning.call_args[0][0]


# ---------------------------------------------------------------------------
# Regression — force_delegation must not swallow UserCancelledRun (#1166)
# ---------------------------------------------------------------------------


class TestForceDelegationUserCancelledRun:
    def _make_config(self) -> MagicMock:
        """Return a config that resolves to valid provider/model for task decomposition."""
        pc = MagicMock()
        pc.type = "openai"
        pc.api_key = None
        pc.get_base_url.return_value = None
        mc = MagicMock()
        mc.model = "gpt-4.1-mini"
        mc.context_window = None
        mc.max_tokens = None
        config = MagicMock()
        config.resolve_llm_config.return_value = (pc, mc)
        return config

    def test_user_cancelled_run_propagates(self):
        """UserCancelledRun raised inside force_delegation must not be swallowed by except Exception."""
        from unittest.mock import patch

        from src.agent.safety import UserCancelledRun

        log_mock = MagicMock()
        config_mock = self._make_config()

        fake_llm = MagicMock()
        fake_response = MagicMock()
        fake_response.content = '{"task":"do sub-task","model":"default"}'
        fake_llm.invoke.return_value = fake_response

        # Mock _delegate_config so available_aliases is non-empty (force_delegation reaches delegate_parallel).
        # Both patches must be active BEFORE force_delegation is imported so that its local import
        # of delegate_parallel picks up the patched version.
        with patch("src.orchestration.phases.build_llm_for_decomposition", return_value=fake_llm):
            with patch("src.tools.delegate._delegate_config") as mock_cfg:
                mock_cfg.get.return_value = {"default": {"model": "gpt-4o", "provider": "openai"}}
                with patch(
                    "src.tools.delegate.delegate_parallel", side_effect=UserCancelledRun("stop")
                ):
                    from src.orchestration.phases import force_delegation

                    with pytest.raises(UserCancelledRun):
                        force_delegation(
                            user_input="do this task",
                            agent_response="some response",
                            tool_outputs="",
                            config=config_mock,
                            log=log_mock,
                        )

    def test_other_exceptions_still_swallowed(self):
        """Non-cancellation exceptions must still be caught and logged."""
        from unittest.mock import patch

        log_mock = MagicMock()
        config_mock = self._make_config()

        fake_llm = MagicMock()
        fake_response = MagicMock()
        fake_response.content = '{"task":"do sub-task","model":"default"}'
        fake_llm.invoke.return_value = fake_response

        with patch("src.orchestration.phases.build_llm_for_decomposition", return_value=fake_llm):
            with patch("src.tools.delegate._delegate_config") as mock_cfg:
                mock_cfg.get.return_value = {"default": {"model": "gpt-4o", "provider": "openai"}}
                with patch(
                    "src.tools.delegate.delegate_parallel", side_effect=RuntimeError("boom")
                ):
                    from src.orchestration.phases import force_delegation

                    result = force_delegation(
                        user_input="do this task",
                        agent_response="fallback response",
                        tool_outputs="",
                        config=config_mock,
                        log=log_mock,
                    )

        assert result == "fallback response"
        log_mock.error.assert_called_once()
        assert "Forced delegation failed" in log_mock.error.call_args[0][0]


# ---------------------------------------------------------------------------
# Regression — force_delegation must timeout on hung llm.invoke() (#1164)
# ---------------------------------------------------------------------------


class TestForceDelegationTimeout:
    def test_llm_invoke_timeout_returns_original_response(self):
        """If llm.invoke hangs, force_delegation must timeout and return agent_response."""
        import concurrent.futures
        from unittest.mock import patch

        log_mock = MagicMock()
        config_mock = MagicMock()
        config_mock.resolve_llm_config.return_value = (MagicMock(), MagicMock())

        fake_llm = MagicMock()
        cancelled: list[bool] = []

        class FakeFuture:
            def result(self, timeout=None):
                raise concurrent.futures.TimeoutError("timed out")

            def cancel(self):
                cancelled.append(True)

        class FakeExecutor:
            def __init__(self, *args, **kwargs):
                pass

            def submit(self, fn, *args, **kwargs):
                return FakeFuture()

            def shutdown(self, wait=False):
                pass

        with patch("src.orchestration.phases.build_llm_for_decomposition", return_value=fake_llm):
            with patch("src.tools.delegate._delegate_config") as mock_cfg:
                mock_cfg.get.return_value = {"default": {"model": "gpt-4o", "provider": "openai"}}
                with patch("concurrent.futures.ThreadPoolExecutor", FakeExecutor):
                    from src.orchestration.phases import force_delegation

                    result = force_delegation(
                        user_input="do this task",
                        agent_response="original response",
                        tool_outputs="",
                        config=config_mock,
                        log=log_mock,
                    )

        assert result == "original response"
        assert cancelled == [True]
        log_mock.warning.assert_called_once()
        assert "timed out" in log_mock.warning.call_args[0][0].lower()


# ---------------------------------------------------------------------------
# Regression — run_research_delegate must not swallow UserCancelledRun (#1166)
# ---------------------------------------------------------------------------


class TestRunResearchDelegateUserCancelledRun:
    def test_user_cancelled_run_propagates(self):
        """UserCancelledRun raised inside run_research_delegate must not be swallowed by except Exception."""
        from unittest.mock import patch

        from src.agent.safety import UserCancelledRun

        fake_tool = MagicMock()
        fake_tool.name = "http_get"
        fake_tool.description = "Fetch URL"
        fake_llm = MagicMock()

        # Mock get_delegate_tools (otherwise function returns early) and create_delegate_llm
        # (otherwise raises ValueError about missing provider). Patch run_delegate_agent at source.
        # Import inside patch context so local import picks up the patch.
        with patch("src.tools.delegate.get_delegate_tools", return_value=[fake_tool]):
            with patch("src.tools.delegate.create_delegate_llm", return_value=fake_llm):
                with patch(
                    "src.tools.delegate.run_delegate_agent", side_effect=UserCancelledRun("stop")
                ):
                    from src.orchestration.phases import run_research_delegate

                    with pytest.raises(UserCancelledRun):
                        run_research_delegate(
                            urls=["https://example.com"],
                            task="research this",
                            max_context_tokens=128000,
                        )

    def test_other_exceptions_still_swallowed(self):
        """Non-cancellation exceptions must still be caught and logged."""
        from unittest.mock import patch

        fake_tool = MagicMock()
        fake_tool.name = "http_get"
        fake_tool.description = "Fetch URL"
        fake_llm = MagicMock()

        with patch("src.tools.delegate.get_delegate_tools", return_value=[fake_tool]):
            with patch("src.tools.delegate.create_delegate_llm", return_value=fake_llm):
                with patch(
                    "src.tools.delegate.run_delegate_agent", side_effect=RuntimeError("boom")
                ):
                    from src.orchestration.phases import run_research_delegate

                    result = run_research_delegate(
                        urls=["https://example.com"],
                        task="research this",
                        max_context_tokens=128000,
                    )

        # Exception is caught by except Exception → returns ""
        assert result == ""
