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

    def test_nested_backticks_greedy_fallback(self):
        # YAML content contains triple backticks; the non-greedy regex terminates
        # early at the nested ```, so the greedy fallback must recover the full block.
        text = (
            "```yaml\n"
            "provider: ollama\n"
            "example: |\n"
            "  ```\n"
            "  some nested code\n"
            "  ```\n"
            "model: qwen3:8b\n"
            "```"
        )
        result = _extract_yaml(text)
        assert "provider: ollama" in result
        assert "model: qwen3:8b" in result


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
        text = 'api_key: "sk-secret-value"'
        result = _mask_secrets(text)
        assert "sk-secret-value" not in result
        assert "***" in result

    def test_masks_token(self):
        text = "token: bot123456:ABC"
        result = _mask_secrets(text)
        assert "bot123456" not in result

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
            "provider: ollama\n"
            "inference:\n"
            "  ollama:\n"
            "    type: ollama\n"
            "    model: llama3.2\n"
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
            "provider: ollama\n"
            "inference:\n"
            "  ollama:\n"
            "    type: ollama\n"
            "    model: qwen3:8b\n"
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
        exc.body = {"error": {"message": "No connected db.", "type": "no_db_connection", "param": None, "code": "400"}}  # type: ignore[attr-defined]
        llm_mock.invoke.side_effect = exc
        with patch("src.providers.create_chat_model", return_value=llm_mock):
            _test_connection("openai", "gpt-oss", "sk-test", "http://192.168.70.254:8080/v1")
        captured = capsys.readouterr()
        assert "No connected db." in captured.out
        assert "{'error':" not in captured.out, "Raw dict repr must not appear in error output"


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

    def test_curly_braces_in_docs_do_not_crash(self):
        from src.setup_wizard import _WIZARD_SYSTEM_PROMPT

        result = _WIZARD_SYSTEM_PROMPT.substitute(
            docs="example {curly} and {{double}} and }}}",
            existing_config="x: 1",
            bootstrap_provider="ollama",
            bootstrap_model="qwen3:8b",
        )
        assert "{curly}" in result
        assert "{{double}}" in result
        assert "}}}" in result

    def test_substitution_inserts_all_fields(self):
        from src.setup_wizard import _WIZARD_SYSTEM_PROMPT

        result = _WIZARD_SYSTEM_PROMPT.substitute(
            docs="doc content",
            existing_config="cfg content",
            bootstrap_provider="openai",
            bootstrap_model="gpt-4",
        )
        assert "doc content" in result
        assert "cfg content" in result
        assert "openai" in result
        assert "gpt-4" in result

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
