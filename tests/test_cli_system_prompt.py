"""Tests for CLI system-prompt override loading."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cogtrix import _load_cli_system_prompt


class TestLoadCliSystemPrompt:
    def test_inline_prompt_wins(self):
        args = SimpleNamespace(system_prompt="inline prompt", system_prompt_file="ignored.txt")
        assert _load_cli_system_prompt(args) == "inline prompt"

    def test_reads_prompt_file(self, tmp_path):
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("file prompt")
        args = SimpleNamespace(system_prompt=None, system_prompt_file=str(prompt_file))
        assert _load_cli_system_prompt(args) == "file prompt"

    def test_missing_prompt_file_raises(self, tmp_path):
        args = SimpleNamespace(system_prompt=None, system_prompt_file=str(tmp_path / "missing.txt"))
        with pytest.raises(FileNotFoundError):
            _load_cli_system_prompt(args)

    def test_empty_prompt_file_raises(self, tmp_path):
        prompt_file = tmp_path / "empty.txt"
        prompt_file.write_text("   ")
        args = SimpleNamespace(system_prompt=None, system_prompt_file=str(prompt_file))
        with pytest.raises(ValueError):
            _load_cli_system_prompt(args)
