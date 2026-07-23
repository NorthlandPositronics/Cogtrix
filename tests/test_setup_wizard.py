"""Tests for src/setup_wizard.py."""

from __future__ import annotations

import os
import socket
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.setup_wizard import (
    _detect_environment,
    _extract_config_info,
    _extract_connection_error,
    _extract_yaml,
    _has_yaml_block,
    _inject_bootstrap,
    _is_safe_url,
    _list_ollama_models,
    _load_docs,
    _load_existing_config,
    _mask_secrets,
    _print_detections,
    _run_conversation,
    _sanitize_yaml_for_prompt,
    _strip_nulls,
    _test_connection,
)


class TestDetectEnvironment:
    def test_detects_openai_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            env = _detect_environment()
        assert env["openai_key"] == "sk-test123"

    def test_no_openai_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.setup_wizard.urllib.request.urlopen", side_effect=Exception):
                env = _detect_environment()
        assert "openai_key" not in env

    def test_detects_ollama_running(self):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.setup_wizard.urllib.request.urlopen", return_value=mock_resp):
                env = _detect_environment()
        assert env.get("ollama_running") is True

    def test_ollama_not_running(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.setup_wizard.urllib.request.urlopen", side_effect=Exception("refused")):
                env = _detect_environment()
        assert "ollama_running" not in env


class TestLoadDocs:
    def test_loads_embedded_docs(self):
        docs = _load_docs()
        assert "Configuration" in docs
        assert len(docs) > 100

    def test_url_fallback_on_failure(self):
        with patch("src.setup_wizard.urllib.request.urlopen", side_effect=Exception("fail")):
            docs = _load_docs(url="https://example.com/bad")
        assert "Configuration" in docs

    def test_url_success(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"# Custom Docs\nContent here"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("src.setup_wizard.urllib.request.urlopen", return_value=mock_resp):
            docs = _load_docs(url="https://example.com/docs")
        assert docs == "# Custom Docs\nContent here"


class TestLoadExistingConfig:
    def test_no_config_found(self):
        with patch("src.config.find_config_file", return_value=None):
            content, path = _load_existing_config()
        assert content == ""
        assert path is None

    def test_config_found(self, tmp_path):
        cfg_file = tmp_path / ".cogtrix.yaml"
        cfg_file.write_text("provider: ollama\nmodel: qwen3:8b\n")
        with patch("src.config.find_config_file", return_value=cfg_file):
            content, path = _load_existing_config()
        assert "provider: ollama" in content
        assert path == cfg_file


class TestHasYamlBlock:
    def test_complete_block(self):
        assert _has_yaml_block("Here is the config:\n```yaml\nprovider: ollama\n```\nDone.")

    def test_no_block(self):
        assert not _has_yaml_block("No yaml here")

    def test_unclosed_block(self):
        assert not _has_yaml_block("```yaml\nprovider: ollama\n")

    def test_only_marker(self):
        assert not _has_yaml_block("```yaml")


class TestExtractYaml:
    def test_extracts_yaml_block(self):
        text = "Some text\n```yaml\nprovider: ollama\nmodel: qwen3:8b\n```\nMore text"
        result = _extract_yaml(text)
        assert "provider: ollama" in result
        assert "model: qwen3:8b" in result

    def test_takes_last_block(self):
        text = "```yaml\nfirst: block\n```\nLater:\n```yaml\nsecond: block\n```"
        result = _extract_yaml(text)
        assert "second: block" in result
        assert "first" not in result

    def test_no_block_raises(self):
        with pytest.raises(ValueError, match="No.*yaml.*block"):
            _extract_yaml("no yaml here")

    def test_strips_whitespace(self):
        text = "```yaml\n  provider: ollama  \n```"
        result = _extract_yaml(text)
        assert result == "provider: ollama"

    def test_trailing_code_fences_not_included(self):
        # BUG: when the LLM appends a "Next steps" section with shell code fences
        # AFTER the YAML block, the old greedy fallback spanned all the way to the
        # last ``` in the response, injecting shell commands into the YAML string
        # which then failed yaml.safe_load with "character '`' cannot start token".
        text = (
            "Here is your config:\n"
            "\n"
            "```yaml\n"
            "provider: ollama\n"
            "model: qwen3:8b\n"
            "```\n"
            "\n"
            "Next steps:\n"
            "1. Create dirs:\n"
            "\n"
            "```\n"
            "mkdir -p docs vectordb\n"
            "```\n"
            "\n"
            "2. Start:\n"
            "\n"
            "```\n"
            "python cogtrix.py\n"
            "```\n"
        )
        result = _extract_yaml(text)
        assert "provider: ollama" in result
        assert "model: qwen3:8b" in result
        assert "mkdir" not in result
        assert "python cogtrix.py" not in result
        assert "```" not in result


class TestInjectBootstrap:
    def test_injects_api_key(self):
        data = {}
        bootstrap = {
            "provider": "openai",
            "model": "gpt-4.1-mini",
            "api_key": "sk-real-key",
            "base_url": None,
            "type": "openai",
        }
        _inject_bootstrap(data, bootstrap)
        assert data["providers"]["openai"]["api_key"] == "sk-real-key"
        assert "model" not in data["providers"]["openai"]
        assert data["models"]["default_model"]["model"] == "gpt-4.1-mini"

    def test_injects_ollama_base_url(self):
        data = {}
        bootstrap = {
            "provider": "ollama",
            "model": "qwen3:8b",
            "api_key": None,
            "base_url": "http://localhost:11434",
            "type": "ollama",
        }
        _inject_bootstrap(data, bootstrap)
        assert "provider" not in data
        assert data["providers"]["ollama"]["base_url"] == "http://localhost:11434"
        assert data["models"]["default"] == "default_model"

    def test_preserves_existing_providers(self):
        data = {"providers": {"other_provider": {"type": "openai", "model": "gpt-4.1"}}}
        bootstrap = {
            "provider": "ollama",
            "model": "qwen3:8b",
            "api_key": None,
            "base_url": None,
            "type": "ollama",
        }
        _inject_bootstrap(data, bootstrap)
        assert "other_provider" in data["providers"]
        assert "ollama" in data["providers"]

    def test_no_api_key_when_none(self):
        data = {}
        bootstrap = {
            "provider": "ollama",
            "model": "qwen3:8b",
            "api_key": None,
            "base_url": None,
            "type": "ollama",
        }
        _inject_bootstrap(data, bootstrap)
        assert "api_key" not in data["providers"]["ollama"]

    def test_creates_models_registry_entry(self):
        data = {}
        bootstrap = {
            "provider": "openai",
            "model": "gpt-4.1-mini",
            "api_key": "sk-key",
            "base_url": None,
            "type": "openai",
        }
        _inject_bootstrap(data, bootstrap)
        assert "models" in data
        assert data["models"]["default"] == "default_model"
        assert data["models"]["default_model"]["provider"] == "openai"
        assert data["models"]["default_model"]["model"] == "gpt-4.1-mini"
        assert "model" not in data

    def test_does_not_overwrite_existing_default_model(self):
        data = {
            "models": {
                "default_model": {"provider": "ollama", "model": "existing-model"},
            }
        }
        bootstrap = {
            "provider": "openai",
            "model": "gpt-4.1-mini",
            "api_key": "sk-key",
            "base_url": None,
            "type": "openai",
        }
        _inject_bootstrap(data, bootstrap)
        assert data["models"]["default_model"]["provider"] == "ollama"
        assert data["models"]["default_model"]["model"] == "existing-model"
        assert data["models"]["default"] == "default_model"


class TestMaskSecrets:
    def test_masks_api_key(self):
        text = 'api_key: "sk-secret-value-1234"'
        result = _mask_secrets(text)
        assert "sk-secret-value-1234" not in result
        assert 'api_key: "sk-***1234"' in result

    def test_masks_api_key_short(self):
        text = 'api_key: "abc"'
        result = _mask_secrets(text)
        assert "abc" not in result
        assert 'api_key: "***"' in result

    def test_masks_token(self):
        text = "token: bot123456:ABC"
        result = _mask_secrets(text)
        assert "bot123456" not in result
        assert "token: bot***:ABC" in result

    def test_preserves_non_secret(self):
        text = "provider: ollama\nmodel: qwen3:8b"
        result = _mask_secrets(text)
        assert "provider: ollama" in result
        assert "model: qwen3:8b" in result


class TestValidateYamlIntegration:
    def test_valid_yaml_round_trip(self):
        yaml_content = "provider: ollama\n"
        data = yaml.safe_load(yaml_content)
        assert isinstance(data, dict)
        assert data["provider"] == "ollama"


class TestExtractConfigInfo:
    def test_basic_provider_and_model(self):
        yaml_content = "provider: openai\nmodel: gpt-4.1-mini\n"
        info = _extract_config_info(yaml_content)
        assert info["provider"] == "openai"
        assert info["model"] == "gpt-4.1-mini"

    def test_inference_section_type_and_base_url(self):
        yaml_content = (
            "provider: myollama\n"
            "inference:\n"
            "  myollama:\n"
            "    type: ollama\n"
            "    model: qwen3:8b\n"
            "    base_url: http://localhost:11434\n"
        )
        info = _extract_config_info(yaml_content)
        assert info["provider"] == "myollama"
        assert info["type"] == "ollama"
        assert info["base_url"] == "http://localhost:11434"

    def test_model_falls_back_to_inference_when_top_level_absent(self):
        yaml_content = (
            "provider: ollama\ninference:\n  ollama:\n    type: ollama\n    model: llama3.2\n"
        )
        info = _extract_config_info(yaml_content)
        assert info["model"] == "llama3.2"

    def test_top_level_model_takes_precedence_over_inference_model(self):
        yaml_content = (
            "provider: ollama\n"
            "model: top-level-model\n"
            "inference:\n"
            "  ollama:\n"
            "    type: ollama\n"
            "    model: inference-model\n"
        )
        info = _extract_config_info(yaml_content)
        assert info["model"] == "top-level-model"

    def test_missing_provider_returns_empty(self):
        yaml_content = "model: gpt-4.1-mini\n"
        info = _extract_config_info(yaml_content)
        assert "provider" not in info
        assert "type" not in info

    def test_invalid_yaml_returns_empty(self):
        info = _extract_config_info("not: valid: yaml: ::::")
        assert info == {}

    def test_non_dict_yaml_returns_empty(self):
        info = _extract_config_info("- item1\n- item2\n")
        assert info == {}

    def test_empty_string_returns_empty(self):
        info = _extract_config_info("")
        assert info == {}

    def test_extracts_api_key_from_inference(self):
        yaml_content = (
            "provider: openai\n"
            "inference:\n"
            "  openai:\n"
            "    type: openai\n"
            "    model: gpt-4.1-mini\n"
            "    api_key: sk-existing-key\n"
        )
        info = _extract_config_info(yaml_content)
        assert info["api_key"] == "sk-existing-key"

    def test_no_api_key_when_absent(self):
        yaml_content = (
            "provider: ollama\ninference:\n  ollama:\n    type: ollama\n    model: qwen3:8b\n"
        )
        info = _extract_config_info(yaml_content)
        assert "api_key" not in info
        assert "base_url" not in info

    def test_providers_key_used_as_fallback_for_inference(self):
        yaml_content = (
            "provider: groq\n"
            "providers:\n"
            "  groq:\n"
            "    type: openai\n"
            "    base_url: https://api.groq.com/openai/v1\n"
        )
        info = _extract_config_info(yaml_content)
        assert info["type"] == "openai"
        assert info["base_url"] == "https://api.groq.com/openai/v1"


class TestExtractConfigInfoNewFormat:
    """Tests for _extract_config_info() with the new models.default format."""

    def test_new_format_models_default_resolves_alias(self):
        yaml_content = (
            "providers:\n"
            "  openai:\n"
            "    type: openai\n"
            "    api_key: sk-test\n"
            "models:\n"
            "  default: fast\n"
            "  fast:\n"
            "    provider: openai\n"
            "    model: gpt-4o-mini\n"
        )
        info = _extract_config_info(yaml_content)
        assert info["provider"] == "openai"
        assert info["model"] == "gpt-4o-mini"
        assert info["type"] == "openai"
        assert info["api_key"] == "sk-test"

    def test_new_format_with_base_url(self):
        yaml_content = (
            "providers:\n"
            "  spark:\n"
            "    type: openai\n"
            "    base_url: http://spark:8080/v1\n"
            "    api_key: sk-spark\n"
            "models:\n"
            "  default: oss\n"
            "  oss:\n"
            "    provider: spark\n"
            "    model: gpt-oss\n"
        )
        info = _extract_config_info(yaml_content)
        assert info["provider"] == "spark"
        assert info["model"] == "gpt-oss"
        assert info["base_url"] == "http://spark:8080/v1"

    def test_new_format_default_alias_missing_from_models(self):
        yaml_content = (
            "providers:\n"
            "  openai:\n"
            "    type: openai\n"
            "models:\n"
            "  default: nonexistent\n"
            "  fast:\n"
            "    provider: openai\n"
            "    model: gpt-4o\n"
        )
        info = _extract_config_info(yaml_content)
        # Falls through to legacy — no top-level provider/model either
        assert info.get("model") is None or "provider" not in info

    def test_new_format_takes_precedence_over_legacy(self):
        yaml_content = (
            "provider: ollama\n"
            "model: legacy-model\n"
            "providers:\n"
            "  openai:\n"
            "    type: openai\n"
            "    api_key: sk-new\n"
            "models:\n"
            "  default: smart\n"
            "  smart:\n"
            "    provider: openai\n"
            "    model: gpt-4o\n"
        )
        info = _extract_config_info(yaml_content)
        assert info["provider"] == "openai"
        assert info["model"] == "gpt-4o"

    def test_new_format_provider_not_in_providers_section(self):
        yaml_content = (
            "models:\n"
            "  default: fast\n"
            "  fast:\n"
            "    provider: missing_provider\n"
            "    model: some-model\n"
        )
        info = _extract_config_info(yaml_content)
        assert info["provider"] == "missing_provider"
        assert info["model"] == "some-model"
        assert "type" not in info

    def test_new_format_ollama_provider(self):
        yaml_content = (
            "providers:\n"
            "  local:\n"
            "    type: ollama\n"
            "    base_url: http://localhost:11434\n"
            "models:\n"
            "  default: local_model\n"
            "  local_model:\n"
            "    provider: local\n"
            "    model: qwen3:8b\n"
        )
        info = _extract_config_info(yaml_content)
        assert info["type"] == "ollama"
        assert info["base_url"] == "http://localhost:11434"
        assert info["model"] == "qwen3:8b"


class TestListOllamaModels:
    def _make_mock_urlopen(self, payload: bytes):
        mock_resp = MagicMock()
        mock_resp.read.return_value = payload
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        return MagicMock(return_value=mock_resp)

    def test_returns_model_names(self):
        payload = b'{"models": [{"name": "qwen3:8b", "size": 20000000000}, {"name": "llama3.2", "size": 500000000}]}'
        mock_urlopen = self._make_mock_urlopen(payload)
        with patch("src.setup_wizard.urllib.request.urlopen", mock_urlopen):
            names = _list_ollama_models("http://localhost:11434")
        assert names == ["qwen3:8b", "llama3.2"]

    def test_prints_model_names_and_sizes(self, capsys):
        payload = b'{"models": [{"name": "phi4", "size": 2500000000}]}'
        mock_urlopen = self._make_mock_urlopen(payload)
        with patch("src.setup_wizard.urllib.request.urlopen", mock_urlopen):
            _list_ollama_models("http://localhost:11434")
        captured = capsys.readouterr()
        assert "phi4" in captured.out
        assert "2.5GB" in captured.out

    def test_size_displayed_as_mb_when_under_1gb(self, capsys):
        payload = b'{"models": [{"name": "tiny", "size": 750000000}]}'
        mock_urlopen = self._make_mock_urlopen(payload)
        with patch("src.setup_wizard.urllib.request.urlopen", mock_urlopen):
            _list_ollama_models("http://localhost:11434")
        captured = capsys.readouterr()
        assert "MB" in captured.out

    def test_empty_models_list_returns_empty(self):
        payload = b'{"models": []}'
        mock_urlopen = self._make_mock_urlopen(payload)
        with patch("src.setup_wizard.urllib.request.urlopen", mock_urlopen):
            names = _list_ollama_models("http://localhost:11434")
        assert names == []

    def test_connection_failure_returns_empty(self):
        with patch("src.setup_wizard.urllib.request.urlopen", side_effect=Exception("refused")):
            names = _list_ollama_models("http://localhost:11434")
        assert names == []


class TestExtractConnectionError:
    """Unit tests for _extract_connection_error — error message prettifier.

    The openai SDK builds exc.message as "Error code: {status} - {body_repr}",
    so the human-readable text must be extracted from exc.body, not exc.message.
    """

    def test_body_dict_with_nested_error_message(self) -> None:
        # Mirrors actual openai.BadRequestError raised by the SDK
        exc = Exception("Error code: 400 - {'error': {'message': 'No connected db.'}}")
        exc.message = "Error code: 400 - {'error': {'message': 'No connected db.'}}"  # type: ignore[attr-defined]
        exc.body = {"error": {"message": "No connected db.", "type": "no_db_connection"}}  # type: ignore[attr-defined]
        assert _extract_connection_error(exc) == "No connected db."

    def test_body_dict_with_top_level_message(self) -> None:
        exc = Exception("Error code: 503 - {'message': 'Service unavailable'}")
        exc.body = {"message": "Service unavailable"}  # type: ignore[attr-defined]
        assert _extract_connection_error(exc) == "Service unavailable"

    def test_body_dict_error_missing_message_falls_through_to_message(self) -> None:
        # Body has no usable message; exc.message is clean (no "Error code:" prefix)
        exc = Exception("Connection error.")
        exc.message = "Connection error."  # type: ignore[attr-defined]
        exc.body = {"error": {"code": "500"}}  # type: ignore[attr-defined]
        assert _extract_connection_error(exc) == "Connection error."

    def test_clean_message_attr_used_when_no_body(self) -> None:
        # Mirrors openai.APIConnectionError (no HTTP status, clean message)
        exc = Exception("Connection error.")
        exc.message = "Connection error."  # type: ignore[attr-defined]
        assert _extract_connection_error(exc) == "Connection error."

    def test_error_code_prefix_in_message_skipped(self) -> None:
        # When body is absent, exc.message starts with "Error code:" — skip it
        exc = Exception("Error code: 400 - some ugly body")
        exc.message = "Error code: 400 - some ugly body"  # type: ignore[attr-defined]
        # No body → falls through to str(exc) which is same as message
        assert _extract_connection_error(exc) == "Error code: 400 - some ugly body"

    def test_no_special_attributes_returns_str(self) -> None:
        exc = ValueError("plain error")
        assert _extract_connection_error(exc) == "plain error"

    def test_body_not_dict_falls_through(self) -> None:
        exc = Exception("Connection error.")
        exc.message = "Connection error."  # type: ignore[attr-defined]
        exc.body = "raw string body"  # type: ignore[attr-defined]
        assert _extract_connection_error(exc) == "Connection error."


class TestTestConnection:
    def _make_llm_mock(self, response_text: str = "ok"):
        response = MagicMock()
        response.content = response_text
        llm = MagicMock()
        llm.invoke.return_value = response
        return llm

    def test_successful_connection_returns_llm(self):
        llm_mock = self._make_llm_mock("ok")
        with patch("src.providers.create_chat_model", return_value=llm_mock):
            result = _test_connection("ollama", "qwen3:8b", None, "http://localhost:11434")
        assert result is llm_mock

    def test_provider_setup_failure_returns_none(self, capsys):
        with patch(
            "src.providers.create_chat_model",
            side_effect=ValueError("bad config"),
        ):
            result = _test_connection("ollama", "bad-model", None, None)
        assert result is None
        captured = capsys.readouterr()
        assert "Provider setup failed" in captured.out

    def test_invoke_failure_returns_none(self, capsys):
        llm_mock = MagicMock()
        llm_mock.invoke.side_effect = RuntimeError("timeout")
        with patch("src.providers.create_chat_model", return_value=llm_mock):
            result = _test_connection("openai", "gpt-4.1-mini", "sk-test", None)
        assert result is None
        captured = capsys.readouterr()
        assert "Connection failed" in captured.out

    def test_empty_response_returns_none(self, capsys):
        llm_mock = self._make_llm_mock("   ")
        with patch("src.providers.create_chat_model", return_value=llm_mock):
            result = _test_connection("openai", "gpt-4.1-mini", "sk-test", None)
        assert result is None
        captured = capsys.readouterr()
        assert "Connection failed" in captured.out

    def test_response_without_content_attr_uses_str(self):
        class _NoContent:
            def __str__(self) -> str:
                return "ok"

        llm_mock = MagicMock()
        llm_mock.invoke.return_value = _NoContent()
        with patch("src.providers.create_chat_model", return_value=llm_mock):
            result = _test_connection("ollama", "llama3.2", None, "http://localhost:11434")
        assert result is llm_mock

    def test_create_chat_model_called_with_max_retries_zero(self) -> None:
        """Connection test must use max_retries=0 to fail fast on unreachable hosts."""
        llm_mock = self._make_llm_mock("ok")
        with patch("src.providers.create_chat_model", return_value=llm_mock) as mock_factory:
            _test_connection("openai", "gpt-4o", "sk-test", None)
        _kwargs = mock_factory.call_args.kwargs
        assert _kwargs.get("max_retries") == 0, (
            "_test_connection must pass max_retries=0 to avoid internal retry delays "
            "when the host is unreachable"
        )

    def test_api_error_with_body_shown_cleanly(self, capsys) -> None:
        """A structured API error must show the inner message, not the full dict repr.

        The openai SDK sets exc.message = 'Error code: 400 - {body_repr}' (ugly).
        The clean message lives in exc.body['error']['message'].
        """
        llm_mock = MagicMock()
        # Mirrors the real openai.BadRequestError raised by the SDK
        exc = RuntimeError(
            "Error code: 400 - {'error': {'message': 'No connected db.', 'type': 'no_db_connection', 'param': None, 'code': '400'}}"
        )
        exc.message = "Error code: 400 - {'error': {'message': 'No connected db.', 'type': 'no_db_connection', 'param': None, 'code': '400'}}"  # type: ignore[attr-defined]
        exc.body = {
            "error": {
                "message": "No connected db.",
                "type": "no_db_connection",
                "param": None,
                "code": "400",
            }
        }  # type: ignore[attr-defined]
        llm_mock.invoke.side_effect = exc
        with patch("src.providers.create_chat_model", return_value=llm_mock):
            _test_connection("openai", "gpt-oss", "sk-test", "http://192.168.70.254:8080/v1")
        captured = capsys.readouterr()
        assert "No connected db." in captured.out
        assert "{'error':" not in captured.out, "Raw dict repr must not appear in error output"

    def test_timeout_returns_none(self, capsys):
        """A hung LLM call must return None rather than blocking forever."""
        import time
        from concurrent.futures import TimeoutError as FuturesTimeoutError

        llm_mock = MagicMock()
        llm_mock.invoke.side_effect = FuturesTimeoutError("hung")

        with patch("src.providers.create_chat_model", return_value=llm_mock):
            with patch("src.setup_wizard._SETUP_WIZARD_LLM_TIMEOUT_SECONDS", 0.3):
                t0 = time.monotonic()
                result = _test_connection("openai", "gpt-4.1-mini", "sk-test", None)
                elapsed = time.monotonic() - t0

        assert result is None
        assert elapsed < 1.5, f"Blocked for {elapsed:.1f}s — timeout not applied"
        captured = capsys.readouterr()
        assert "timed out" in captured.out.lower()


class TestPrintDetections:
    def test_prints_openai_key_detection(self, capsys):
        _print_detections({"openai_key": "sk-abc"})
        captured = capsys.readouterr()
        assert "OPENAI_API_KEY" in captured.out

    def test_prints_ollama_detection_with_url(self, capsys):
        _print_detections({"ollama_running": True, "ollama_url": "http://localhost:11434"})
        captured = capsys.readouterr()
        assert "Ollama" in captured.out
        assert "http://localhost:11434" in captured.out

    def test_prints_ollama_default_url_when_missing(self, capsys):
        _print_detections({"ollama_running": True})
        captured = capsys.readouterr()
        assert "127.0.0.1:11434" in captured.out

    def test_both_detections(self, capsys):
        _print_detections(
            {"openai_key": "sk-xyz", "ollama_running": True, "ollama_url": "http://10.0.0.1:11434"}
        )
        captured = capsys.readouterr()
        assert "OPENAI_API_KEY" in captured.out
        assert "Ollama" in captured.out

    def test_empty_env_prints_nothing(self, capsys):
        _print_detections({})
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_no_exception_on_arbitrary_env(self):
        _print_detections({"some_unknown_key": "value"})


class TestWizardSystemPromptTemplate:
    """BUG-074 — _WIZARD_SYSTEM_PROMPT uses string.Template to handle curly braces."""

    def _full_substitute(self, **overrides):
        from src.setup_wizard import _WIZARD_SYSTEM_PROMPT

        defaults = dict(
            existing_config="cfg content",
            bootstrap_provider="openai",
            bootstrap_type="openai",
            bootstrap_base_url="https://api.openai.com/v1",
            bootstrap_model="gpt-4",
            bootstrap_has_key="yes",
            production_context="Same as bootstrap: use openai / gpt-4 as models.default.",
        )
        defaults.update(overrides)
        return _WIZARD_SYSTEM_PROMPT.substitute(**defaults)

    def test_curly_braces_in_existing_config_do_not_crash(self):
        # Docs are now injected per-turn, not in the system prompt — verify
        # curly braces in existing_config still don't crash Template.substitute.
        result = self._full_substitute(existing_config="example {curly} and {{double}} and }}}")
        assert "{curly}" in result
        assert "{{double}}" in result
        assert "}}}" in result

    def test_substitution_inserts_all_fields(self):
        result = self._full_substitute()
        # Docs are now injected per-turn; system prompt contains config and bootstrap info.
        assert "cfg content" in result
        assert "openai" in result
        assert "gpt-4" in result

    def test_bootstrap_context_included(self):
        """New bootstrap fields must appear verbatim in the rendered prompt."""
        result = self._full_substitute(
            bootstrap_provider="spark",
            bootstrap_type="openai",
            bootstrap_base_url="http://192.168.70.254:8080",
            bootstrap_model="qwen35",
            bootstrap_has_key="yes",
        )
        assert "spark" in result
        assert "http://192.168.70.254:8080" in result
        assert "qwen35" in result
        assert "yes" in result

    def test_bootstrap_no_key_renders_no(self):
        result = self._full_substitute(bootstrap_has_key="no")
        assert "no" in result

    def test_prompt_instructs_not_to_ask_about_bootstrap(self):
        """System prompt must explicitly forbid re-asking bootstrap info."""
        result = self._full_substitute()
        assert "Do NOT ask" in result or "do NOT ask" in result

    def test_prompt_forbids_key_in_comments(self):
        """System prompt must forbid leaking secrets in comments."""
        result = self._full_substitute()
        assert "not in comments" in result or "not in\ncomments" in result

    def test_production_context_rendered_in_prompt(self):
        """$production_context must appear verbatim in the rendered prompt."""
        result = self._full_substitute(
            production_context="Use prod-provider / big-model as models.default."
        )
        assert "prod-provider" in result
        assert "big-model" in result


class TestFormatProductionContext:
    """Unit tests for _format_production_context."""

    def _info(
        self,
        provider="spark",
        model="qwen35",
        ptype="openai",
        base_url="http://192.168.70.254:8080",
        api_key="sk-test",
    ):
        return {
            "provider": provider,
            "model": model,
            "type": ptype,
            "base_url": base_url,
            "api_key": api_key,
        }

    def test_same_object_returns_same_as_bootstrap(self):
        from src.setup_wizard import _format_production_context

        bootstrap = self._info()
        result = _format_production_context(bootstrap, bootstrap)
        assert "Same as bootstrap" in result

    def test_same_values_returns_same_as_bootstrap(self):
        from src.setup_wizard import _format_production_context

        bootstrap = self._info()
        production = self._info()  # equal dict but different object
        result = _format_production_context(bootstrap, production)
        assert "Same as bootstrap" in result

    def test_different_model_returns_production_context(self):
        from src.setup_wizard import _format_production_context

        bootstrap = self._info(model="small-model")
        production = self._info(model="big-model")
        result = _format_production_context(bootstrap, production)
        assert "big-model" in result
        assert "separate production model" in result

    def test_different_provider_returns_production_context(self):
        from src.setup_wizard import _format_production_context

        bootstrap = self._info(provider="local")
        production = self._info(
            provider="cloud", model="gpt-4o", base_url="https://api.openai.com/v1"
        )
        result = _format_production_context(bootstrap, production)
        assert "cloud" in result
        assert "gpt-4o" in result
        assert "models.default" in result

    def test_production_context_includes_all_fields(self):
        from src.setup_wizard import _format_production_context

        bootstrap = self._info(model="tiny")
        production = self._info(
            provider="big-provider",
            model="gpt-4o",
            ptype="openai",
            base_url="https://api.openai.com/v1",
            api_key="sk-real",
        )
        result = _format_production_context(bootstrap, production)
        assert "big-provider" in result
        assert "gpt-4o" in result
        assert "openai" in result
        assert "yes" in result  # api_key configured: yes

    def test_no_key_shows_no(self):
        from src.setup_wizard import _format_production_context

        bootstrap = self._info(model="tiny")
        production = self._info(model="big", api_key="")
        result = _format_production_context(bootstrap, production)
        assert "no" in result

    def test_no_secret_in_context(self):
        """Real API keys must never appear in the production context string."""
        from src.setup_wizard import _format_production_context

        bootstrap = self._info(model="tiny")
        production = self._info(model="big", api_key="sk-supersecret")
        result = _format_production_context(bootstrap, production)
        assert "sk-supersecret" not in result


class TestMaybeConfigureProductionModel:
    """Unit tests for _maybe_configure_production_model."""

    def _bootstrap(self):
        return {
            "provider": "spark",
            "model": "qwen35",
            "type": "openai",
            "base_url": "http://192.168.70.254:8080",
            "api_key": "sk-test",
        }

    def test_no_returns_bootstrap_unchanged(self):
        from src.setup_wizard import _maybe_configure_production_model

        bootstrap = self._bootstrap()
        with patch("builtins.input", return_value="no"):
            result = _maybe_configure_production_model(bootstrap, {})
        assert result is bootstrap

    def test_default_is_no(self):
        """Pressing Enter (empty input) must default to 'no'."""
        from src.setup_wizard import _maybe_configure_production_model

        bootstrap = self._bootstrap()
        with patch("builtins.input", return_value=""):
            result = _maybe_configure_production_model(bootstrap, {})
        assert result is bootstrap

    def test_yes_runs_bootstrap_llm_and_returns_production(self):
        from src.setup_wizard import _maybe_configure_production_model

        bootstrap = self._bootstrap()
        prod_info = {
            "provider": "openai",
            "model": "gpt-4o",
            "type": "openai",
            "base_url": None,
            "api_key": "sk-prod",
        }
        prod_llm = MagicMock()
        with (
            patch("builtins.input", return_value="yes"),
            patch("src.setup_wizard._bootstrap_llm", return_value=(prod_llm, prod_info)) as mock_bl,
        ):
            result = _maybe_configure_production_model(bootstrap, {"openai_key": "sk-prod"})
        mock_bl.assert_called_once()
        assert result is prod_info
        assert result["model"] == "gpt-4o"

    def test_output_shows_current_model(self, capsys):
        from src.setup_wizard import _maybe_configure_production_model

        bootstrap = self._bootstrap()
        with patch("builtins.input", return_value="no"):
            _maybe_configure_production_model(bootstrap, {})
        out = capsys.readouterr().out
        assert "qwen35" in out
        assert "spark" in out


class TestInjectBootstrapTwoProvider:
    """Tests for _inject_bootstrap with a separate production provider."""

    def _bootstrap(self):
        return {
            "provider": "wizard-llm",
            "model": "small-model",
            "type": "openai",
            "base_url": "http://local:8080",
            "api_key": "sk-wiz",
        }

    def _production(self):
        return {
            "provider": "cloud",
            "model": "gpt-4o",
            "type": "openai",
            "base_url": None,
            "api_key": "sk-cloud",
        }

    def test_single_provider_bootstrap_is_default(self):
        from src.setup_wizard import _inject_bootstrap

        data: dict = {}
        _inject_bootstrap(data, self._bootstrap())
        assert data["models"]["default_model"]["provider"] == "wizard-llm"
        assert data["models"]["default_model"]["model"] == "small-model"

    def test_production_provider_becomes_default(self):
        from src.setup_wizard import _inject_bootstrap

        data: dict = {}
        _inject_bootstrap(data, self._bootstrap(), production_info=self._production())
        assert data["models"]["default_model"]["provider"] == "cloud"
        assert data["models"]["default_model"]["model"] == "gpt-4o"

    def test_both_providers_in_providers_section(self):
        from src.setup_wizard import _inject_bootstrap

        data: dict = {}
        _inject_bootstrap(data, self._bootstrap(), production_info=self._production())
        assert "wizard-llm" in data["providers"]
        assert "cloud" in data["providers"]

    def test_bootstrap_key_injected(self):
        from src.setup_wizard import _inject_bootstrap

        data: dict = {}
        _inject_bootstrap(data, self._bootstrap(), production_info=self._production())
        assert data["providers"]["wizard-llm"]["api_key"] == "sk-wiz"

    def test_production_key_injected(self):
        from src.setup_wizard import _inject_bootstrap

        data: dict = {}
        _inject_bootstrap(data, self._bootstrap(), production_info=self._production())
        assert data["providers"]["cloud"]["api_key"] == "sk-cloud"

    def test_same_provider_not_duplicated(self):
        """When production == bootstrap, only one provider entry is created."""
        from src.setup_wizard import _inject_bootstrap

        bootstrap = self._bootstrap()
        data: dict = {}
        _inject_bootstrap(data, bootstrap, production_info=bootstrap)
        # Only one provider entry
        assert list(data["providers"].keys()) == ["wizard-llm"]
        assert data["models"]["default_model"]["provider"] == "wizard-llm"

    def test_no_production_key_when_none(self):
        from src.setup_wizard import _inject_bootstrap

        prod = {**self._production(), "api_key": None}
        data: dict = {}
        _inject_bootstrap(data, self._bootstrap(), production_info=prod)
        assert "api_key" not in data["providers"]["cloud"]

    def test_is_template_instance(self):
        from string import Template

        from src.setup_wizard import _WIZARD_SYSTEM_PROMPT

        assert isinstance(_WIZARD_SYSTEM_PROMPT, Template)


class TestIsUrlSafe:
    """BUG-077 — _is_safe_url blocks SSRF targets."""

    def test_loopback_is_blocked(self):
        with patch(
            "socket.getaddrinfo", return_value=[(None, None, None, None, ("127.0.0.1", 80))]
        ):
            assert _is_safe_url("http://localhost/path") is False

    def test_link_local_is_blocked(self):
        with patch(
            "socket.getaddrinfo",
            return_value=[(None, None, None, None, ("169.254.169.254", 80))],
        ):
            assert _is_safe_url("http://169.254.169.254/") is False

    def test_private_rfc1918_is_blocked(self):
        with patch(
            "socket.getaddrinfo",
            return_value=[(None, None, None, None, ("192.168.1.1", 80))],
        ):
            assert _is_safe_url("http://192.168.1.1/") is False

    def test_public_ip_is_allowed(self):
        with patch(
            "socket.getaddrinfo",
            return_value=[(None, None, None, None, ("8.8.8.8", 443))],
        ):
            assert _is_safe_url("https://docs.example.com/") is True

    def test_dns_failure_is_blocked(self):
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("no host")):
            assert _is_safe_url("http://nonexistent.invalid/") is False

    def test_load_docs_blocks_loopback_url(self):
        with patch(
            "socket.getaddrinfo",
            return_value=[(None, None, None, None, ("127.0.0.1", 9999))],
        ):
            docs = _load_docs("http://127.0.0.1:9999/docs")
        # Must fall back to embedded docs (not raise)
        assert len(docs) > 10

    def test_load_docs_blocks_link_local(self):
        with patch(
            "socket.getaddrinfo",
            return_value=[(None, None, None, None, ("169.254.169.254", 80))],
        ):
            docs = _load_docs("http://169.254.169.254/latest/meta-data/")
        assert len(docs) > 10


class TestRunConversation:
    """Unit tests for _run_conversation — the Step 2 LLM dialogue loop."""

    # ── helpers ──────────────────────────────────────────────────────

    def _make_llm(self, responses: list[str]) -> MagicMock:
        """Return a mock LLM that yields *responses* in sequence."""
        llm = MagicMock()
        side_effects = []
        for text in responses:
            resp = MagicMock()
            resp.content = text
            side_effects.append(resp)
        llm.invoke.side_effect = side_effects
        return llm

    def _yaml_response(self, extra: str = "") -> str:
        return f"Here is your config:\n```yaml\nprovider: ollama\n{extra}```\nDone."

    # ── regression: vLLM / LiteLLM 400 "No user query" (bug report 2026-03-24) ──

    def test_first_invoke_includes_human_message(self):
        """The very first llm.invoke call must contain a HumanMessage.

        Strict OpenAI-compatible backends (vLLM, LiteLLM, Qwen) reject a
        messages list that contains only a SystemMessage with:
          'No user query found in messages' (HTTP 400).
        Seed HumanMessage("Start.") must be present before the first invoke.
        """
        from langchain_core.messages import (  # type: ignore[import-untyped]
            HumanMessage,
            SystemMessage,
        )

        llm = self._make_llm([self._yaml_response()])
        with patch("builtins.input", return_value="yes"):
            _run_conversation(llm, "You are the wizard.")

        first_call_messages = llm.invoke.call_args_list[0][0][0]
        types = [type(m) for m in first_call_messages]
        assert SystemMessage in types, "SystemMessage must be present in first invoke"
        assert HumanMessage in types, (
            "HumanMessage must be present in the first invoke — "
            "vLLM/LiteLLM reject system-only message lists with HTTP 400"
        )

    def test_first_human_message_is_start(self):
        """The seed HumanMessage content must be 'Start.' (the documented trigger)."""
        from langchain_core.messages import HumanMessage  # type: ignore[import-untyped]

        llm = self._make_llm([self._yaml_response()])
        with patch("builtins.input", return_value="yes"):
            _run_conversation(llm, "You are the wizard.")

        first_call_messages = llm.invoke.call_args_list[0][0][0]
        human_msgs = [m for m in first_call_messages if isinstance(m, HumanMessage)]
        assert human_msgs, "At least one HumanMessage must be in first invoke"
        assert (
            human_msgs[0].content == "Start."
        ), f"Seed HumanMessage content must be 'Start.', got {human_msgs[0].content!r}"

    def test_system_message_is_first(self):
        """SystemMessage must be the first element (position 0) in the messages list."""
        from langchain_core.messages import SystemMessage  # type: ignore[import-untyped]

        llm = self._make_llm([self._yaml_response()])
        with patch("builtins.input", return_value="yes"):
            _run_conversation(llm, "wizard prompt text")

        first_call_messages = llm.invoke.call_args_list[0][0][0]
        assert isinstance(
            first_call_messages[0], SystemMessage
        ), "SystemMessage must be at position 0 in the messages list"
        assert "wizard prompt text" in first_call_messages[0].content

    # ── happy-path: YAML block accepted on first response ────────────

    def test_returns_response_when_yaml_accepted(self):
        yaml_resp = self._yaml_response()
        llm = self._make_llm([yaml_resp])
        with patch("builtins.input", return_value="yes"):
            result = _run_conversation(llm, "system prompt")
        assert result == yaml_resp

    def test_invokes_llm_exactly_once_when_yaml_accepted_immediately(self):
        llm = self._make_llm([self._yaml_response()])
        with patch("builtins.input", return_value="yes"):
            _run_conversation(llm, "system prompt")
        assert llm.invoke.call_count == 1

    # ── edit flow: user rejects config, then accepts revised version ─

    def test_second_invoke_after_edit_request(self):
        """User rejects config → types edit instruction → LLM invoked again."""
        llm = self._make_llm(
            [
                self._yaml_response(),  # first response has YAML
                self._yaml_response("model: qwen3:8b\n"),  # revised response
            ]
        )
        inputs = iter(["no, continue editing", "add model setting", "yes"])
        with patch("builtins.input", side_effect=inputs):
            result = _run_conversation(llm, "system prompt")
        assert llm.invoke.call_count == 2
        assert "model: qwen3:8b" in result

    # ── no-YAML flow: LLM asks a question, user answers, then YAML ──

    def test_question_answer_before_yaml(self):
        llm = self._make_llm(
            [
                "What do you want to use Cogtrix for?",  # no YAML yet
                self._yaml_response(),
            ]
        )
        inputs = iter(["I want a WhatsApp bot", "yes"])
        with patch("builtins.input", side_effect=inputs):
            result = _run_conversation(llm, "system prompt")
        assert llm.invoke.call_count == 2
        assert "provider: ollama" in result

    # ── quit / cancel ────────────────────────────────────────────────

    def test_quit_raises_system_exit(self):
        llm = self._make_llm(["What do you need?"])
        with patch("builtins.input", return_value="quit"):
            with pytest.raises(SystemExit):
                _run_conversation(llm, "system prompt")

    def test_exit_keyword_raises_system_exit(self):
        llm = self._make_llm(["What do you need?"])
        with patch("builtins.input", return_value="exit"):
            with pytest.raises(SystemExit):
                _run_conversation(llm, "system prompt")

    def test_cancel_after_rejecting_yaml_raises_system_exit(self):
        """Typing 'quit' at the edit-instruction prompt cancels the wizard.

        _ask_choice only accepts 'yes' / 'no, continue editing'; 'quit' is
        handled in the free-text input() call that follows when the user
        chooses to continue editing.
        """
        llm = self._make_llm([self._yaml_response()])
        inputs = iter(["no, continue editing", "quit"])
        with patch("builtins.input", side_effect=inputs):
            with pytest.raises(SystemExit):
                _run_conversation(llm, "system prompt")

    # ── response without .content attribute ─────────────────────────

    def test_response_without_content_attr_uses_str(self):
        class _NoContent:
            def __str__(self) -> str:
                return self._yaml_response()

            def _yaml_response(self) -> str:
                return "```yaml\nprovider: ollama\n```"

        llm = MagicMock()
        llm.invoke.return_value = _NoContent()
        with patch("builtins.input", return_value="yes"):
            result = _run_conversation(llm, "system prompt")
        assert "provider: ollama" in result

    # ── empty user input is skipped (no extra LLM call) ─────────────

    def test_empty_input_does_not_invoke_llm(self):
        llm = self._make_llm(
            [
                "What do you need?",
                self._yaml_response(),
            ]
        )
        # First empty input is skipped; second real input triggers second invoke
        inputs = iter(["", "I need a bot", "yes"])
        with patch("builtins.input", side_effect=inputs):
            _run_conversation(llm, "system prompt")
        assert llm.invoke.call_count == 2

    # ── timeout handling ─────────────────────────────────────────────

    def test_initial_invoke_timeout_exits(self):
        """If the first LLM call hangs, the wizard exits cleanly."""
        import time
        from concurrent.futures import TimeoutError as FuturesTimeoutError

        llm = MagicMock()
        llm.invoke.side_effect = FuturesTimeoutError("hung")

        with patch("src.setup_wizard._SETUP_WIZARD_LLM_TIMEOUT_SECONDS", 0.3):
            t0 = time.monotonic()
            with pytest.raises(SystemExit):
                _run_conversation(llm, "system prompt")
            elapsed = time.monotonic() - t0

        assert elapsed < 1.5, f"Blocked for {elapsed:.1f}s — timeout not applied"

    def test_edit_invoke_timeout_exits(self):
        """If the LLM hangs during the edit flow, the wizard exits cleanly."""
        import time
        from concurrent.futures import TimeoutError as FuturesTimeoutError

        responses = [self._yaml_response()]

        def _side_effect(*args, **kwargs):
            if llm.invoke.call_count >= 2:
                raise FuturesTimeoutError("hung")
            return type("Response", (), {"content": responses[0]})()

        llm = MagicMock()
        llm.invoke.side_effect = _side_effect

        inputs = iter(["no, continue editing", "add model setting"])
        with patch("builtins.input", side_effect=inputs):
            with patch("src.setup_wizard._SETUP_WIZARD_LLM_TIMEOUT_SECONDS", 0.3):
                t0 = time.monotonic()
                with pytest.raises(SystemExit):
                    _run_conversation(llm, "system prompt")
                elapsed = time.monotonic() - t0

        assert elapsed < 1.5, f"Blocked for {elapsed:.1f}s — timeout not applied"

    def test_question_invoke_timeout_exits(self):
        """If the LLM hangs during the question-answer flow, the wizard exits cleanly."""
        import time
        from concurrent.futures import TimeoutError as FuturesTimeoutError

        def _side_effect(*args, **kwargs):
            if llm.invoke.call_count >= 2:
                raise FuturesTimeoutError("hung")
            return type("Response", (), {"content": "What do you need?"})()

        llm = MagicMock()
        llm.invoke.side_effect = _side_effect

        inputs = iter(["I need a bot"])
        with patch("builtins.input", side_effect=inputs):
            with patch("src.setup_wizard._SETUP_WIZARD_LLM_TIMEOUT_SECONDS", 0.3):
                t0 = time.monotonic()
                with pytest.raises(SystemExit):
                    _run_conversation(llm, "system prompt")
                elapsed = time.monotonic() - t0

        assert elapsed < 1.5, f"Blocked for {elapsed:.1f}s — timeout not applied"


class TestStripNulls:
    """_strip_nulls removes None values and empty dicts from config structures."""

    def test_top_level_none_removed(self):
        assert _strip_nulls({"services": None, "session": "default"}) == {"session": "default"}

    def test_nested_none_removed(self):
        data = {"memory": {"mode": "conversation", "extra": None}}
        assert _strip_nulls(data) == {"memory": {"mode": "conversation"}}

    def test_all_none_values_in_dict_removes_parent(self):
        data = {"services": {"whatsapp": None, "telegram": None}, "session": "x"}
        assert _strip_nulls(data) == {"session": "x"}

    def test_non_dict_passthrough(self):
        assert _strip_nulls("hello") == "hello"
        assert _strip_nulls(42) == 42
        assert _strip_nulls([1, 2]) == [1, 2]

    def test_false_and_zero_are_kept(self):
        data = {"delegate": {"enabled": False}, "temperature": 0.0}
        assert _strip_nulls(data) == {"delegate": {"enabled": False}, "temperature": 0.0}

    def test_empty_dict_literal_is_removed(self):
        data = {"services": {}, "session": "default"}
        assert _strip_nulls(data) == {"session": "default"}


class TestSanitizeYamlForPrompt:
    """_sanitize_yaml_for_prompt() must redact secret fields before sending to LLM."""

    def test_api_key_redacted(self):
        yaml_str = "api_key: sk-super-secret-key-value\nmodel: gpt-4\n"
        result = _sanitize_yaml_for_prompt(yaml_str)
        assert "sk-super-secret-key-value" not in result
        assert "***" in result

    def test_token_and_password_redacted(self):
        yaml_str = "token: tok123\npassword: hunter2\nuser: admin\n"
        result = _sanitize_yaml_for_prompt(yaml_str)
        assert "tok123" not in result
        assert "hunter2" not in result
        assert "admin" in result

    def test_non_secret_fields_preserved(self):
        yaml_str = "model: gpt-4.1-mini\nprovider: openai\nbase_url: http://localhost\n"
        result = _sanitize_yaml_for_prompt(yaml_str)
        assert "gpt-4.1-mini" in result
        assert "openai" in result
        assert "http://localhost" in result

    def test_nested_secret_redacted(self):
        yaml_str = "services:\n  email:\n    password: smtp-pass\n    host: mail.example.com\n"
        result = _sanitize_yaml_for_prompt(yaml_str)
        assert "smtp-pass" not in result
        assert "mail.example.com" in result  # codeql[py/incomplete-url-substring-sanitization]

    def test_invalid_yaml_returns_placeholder(self):
        result = _sanitize_yaml_for_prompt("{bad: yaml: content")
        assert "redacted" in result

    def test_empty_string_returns_unchanged(self):
        assert _sanitize_yaml_for_prompt("") == ""

    def test_whitespace_only_returns_unchanged(self):
        assert _sanitize_yaml_for_prompt("   ") == "   "
