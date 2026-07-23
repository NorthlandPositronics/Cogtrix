"""Tests for AssistantService system prompt resolution."""

from src.assistant.service import _ASSISTANT_SYSTEM_PROMPT, AssistantService


class TestBuildSystemPrompt:
    """Test the system prompt priority chain."""

    def test_cli_prompt_highest_priority(self):
        asst_cfg = {"system_prompt": "config prompt"}
        result = AssistantService._build_system_prompt(asst_cfg, "fallback", "cli prompt")
        assert result == "cli prompt"

    def test_inline_config_over_file_and_default(self, tmp_path):
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("file prompt")
        asst_cfg = {
            "system_prompt": "inline prompt",
            "system_prompt_file": str(prompt_file),
        }
        result = AssistantService._build_system_prompt(asst_cfg, "fallback")
        assert result == "inline prompt"

    def test_file_config_over_default(self, tmp_path):
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("file prompt content")
        asst_cfg = {"system_prompt_file": str(prompt_file)}
        result = AssistantService._build_system_prompt(asst_cfg, "fallback")
        assert result == "file prompt content"

    def test_file_config_missing_file_falls_to_default(self):
        asst_cfg = {"system_prompt_file": "/nonexistent/path/prompt.txt"}
        result = AssistantService._build_system_prompt(asst_cfg, "fallback")
        assert result == _ASSISTANT_SYSTEM_PROMPT

    def test_file_config_empty_file_falls_to_default(self, tmp_path):
        prompt_file = tmp_path / "empty.txt"
        prompt_file.write_text("   ")
        asst_cfg = {"system_prompt_file": str(prompt_file)}
        result = AssistantService._build_system_prompt(asst_cfg, "fallback")
        assert result == _ASSISTANT_SYSTEM_PROMPT

    def test_default_prompt_when_nothing_configured(self):
        result = AssistantService._build_system_prompt({}, "fallback")
        assert result == _ASSISTANT_SYSTEM_PROMPT

    def test_cli_prompt_overrides_everything(self, tmp_path):
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("file prompt")
        asst_cfg = {
            "system_prompt": "inline prompt",
            "system_prompt_file": str(prompt_file),
        }
        result = AssistantService._build_system_prompt(asst_cfg, "fallback", "cli wins")
        assert result == "cli wins"
