"""Tests for src/tools/generate_tests.py"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config() -> MagicMock:
    """Return a minimal Config mock."""
    cfg = MagicMock()
    pc = MagicMock(name="provider_config")
    mc = MagicMock(name="model_config")
    cfg.resolve_llm_config.return_value = (pc, mc)
    return cfg


def _llm_response(text: str) -> MagicMock:
    """Return a mock LLM response object."""
    resp = MagicMock()
    resp.content = text
    return resp


def _patch_llm(response_text: str):
    """Context manager: patch create_chat_model_from_configs to return a mock LLM."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = _llm_response(response_text)
    return patch(
        "src.tools.generate_tests.create_chat_model_from_configs",
        return_value=mock_llm,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_module():
    """Reset module-level _config to None before each test."""
    import src.tools.generate_tests as mod

    original = mod._config
    mod._config = None
    yield
    mod._config = original


@pytest.fixture()
def configured(tmp_path: Path):
    """Wire a mock Config into the module."""
    import src.tools.generate_tests as mod

    cfg = _make_config()
    mod.TOOL_SETUP(cfg)
    return cfg


@pytest.fixture()
def source_file(tmp_path: Path) -> Path:
    """Write a minimal Python source file under cwd/src-like path."""
    src = tmp_path / "src" / "mymodule.py"
    src.parent.mkdir(parents=True)
    src.write_text(
        "def add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a - b\n",
        encoding="utf-8",
    )
    return src


# ---------------------------------------------------------------------------
# 1. generate_tests reads source file and calls LLM with its content
# ---------------------------------------------------------------------------


def test_llm_receives_source_content(configured, source_file, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "tests" / "test_mymodule.py"
    out.parent.mkdir(parents=True, exist_ok=True)

    captured_prompts: list[str] = []

    mock_llm = MagicMock()

    def capture_invoke(msgs):
        captured_prompts.append(msgs[0].content)
        return _llm_response("def test_add(): assert add(1,2)==3")

    mock_llm.invoke.side_effect = capture_invoke

    with patch("src.tools.generate_tests.create_chat_model_from_configs", return_value=mock_llm):
        from src.tools.generate_tests import generate_tests

        generate_tests(str(source_file), str(out))

    assert len(captured_prompts) == 1
    assert "def add(a, b):" in captured_prompts[0]
    assert "def subtract(a, b):" in captured_prompts[0]


# ---------------------------------------------------------------------------
# 2. generate_tests strips markdown fences from LLM response
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected_start",
    [
        ("```python\nimport pytest\n```", "import pytest"),
        ("```\nimport pytest\n```", "import pytest"),
        ("import pytest\n", "import pytest"),  # no fences
    ],
)
def test_strips_markdown_fences(
    configured, source_file, tmp_path, monkeypatch, raw, expected_start
):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "tests" / "test_out.py"
    out.parent.mkdir(parents=True, exist_ok=True)

    with _patch_llm(raw):
        from src.tools.generate_tests import generate_tests

        generate_tests(str(source_file), str(out))

    written = out.read_text(encoding="utf-8")
    assert written.startswith(expected_start)


# ---------------------------------------------------------------------------
# 3. output_file is auto-derived correctly from source_file path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src_name,expected_out",
    [
        ("src/tools/email_tools.py", "tests/test_email_tools.py"),
        ("src/foo/bar.py", "tests/test_bar.py"),
        ("mymodule.py", "tests/test_mymodule.py"),
    ],
)
def test_output_file_derivation(src_name, expected_out):
    from src.tools.generate_tests import _derive_output_file

    assert _derive_output_file(src_name) == expected_out


# ---------------------------------------------------------------------------
# 4. generate_tests writes output to the derived path
# ---------------------------------------------------------------------------


def test_writes_to_derived_path(configured, source_file, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)

    with _patch_llm("def test_placeholder(): pass"):
        from src.tools.generate_tests import generate_tests

        result = generate_tests(str(source_file))

    # derived path: tests/test_mymodule.py
    expected = tmp_path / "tests" / "test_mymodule.py"
    assert expected.exists()
    assert "tests/test_mymodule.py" in result


# ---------------------------------------------------------------------------
# 5. generate_tests returns error if output_file already exists
# ---------------------------------------------------------------------------


def test_error_if_output_exists(configured, source_file, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "tests" / "test_existing.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# existing", encoding="utf-8")

    from src.tools.generate_tests import generate_tests

    result = generate_tests(str(source_file), str(out))

    assert result.startswith("Error:")
    assert "already exists" in result


# ---------------------------------------------------------------------------
# 6. generate_tests rejects path traversal in source_file
# ---------------------------------------------------------------------------


def test_rejects_traversal_in_source_file(configured, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    from src.tools.generate_tests import generate_tests

    result = generate_tests("../etc/passwd")

    assert result.startswith("Error:")
    assert "traversal" in result.lower() or "path" in result.lower()


# ---------------------------------------------------------------------------
# 7. generate_tests rejects path traversal in output_file
# ---------------------------------------------------------------------------


def test_rejects_traversal_in_output_file(configured, source_file, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    from src.tools.generate_tests import generate_tests

    result = generate_tests(str(source_file), "../evil.py")

    assert result.startswith("Error:")
    assert "traversal" in result.lower() or "path" in result.lower()


# ---------------------------------------------------------------------------
# 8. focus parameter is included in the prompt
# ---------------------------------------------------------------------------


def test_focus_included_in_prompt(configured, source_file, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)

    captured: list[str] = []
    mock_llm = MagicMock()

    def capture(msgs):
        captured.append(msgs[0].content)
        return _llm_response("def test_x(): pass")

    mock_llm.invoke.side_effect = capture

    with patch("src.tools.generate_tests.create_chat_model_from_configs", return_value=mock_llm):
        from src.tools.generate_tests import generate_tests

        generate_tests(str(source_file), focus="error paths")

    assert captured, "LLM was not called"
    assert "error paths" in captured[0]


# ---------------------------------------------------------------------------
# 9. LLM error returns descriptive error string
# ---------------------------------------------------------------------------


def test_llm_error_returns_descriptive_string(configured, source_file, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = RuntimeError("connection refused")

    with patch("src.tools.generate_tests.create_chat_model_from_configs", return_value=mock_llm):
        from src.tools.generate_tests import generate_tests

        result = generate_tests(str(source_file))

    assert result.startswith("Error:")
    assert "LLM" in result
    # Raw error details sanitized — exception class and raw message not exposed to LLM
    assert "connection refused" not in result
    assert "Operation failed" in result  # sanitized fallback message


# ---------------------------------------------------------------------------
# 10. TOOL_CONFIGS has 1 entry with requires_confirmation=True
# ---------------------------------------------------------------------------


def test_tool_configs_structure():
    from src.tools.generate_tests import TOOL_CONFIGS

    assert len(TOOL_CONFIGS) == 1
    entry = TOOL_CONFIGS[0]
    assert entry["name"] == "generate_tests"
    assert entry["requires_confirmation"] is True
    assert "function" in entry
    assert "input_schema" in entry


# ---------------------------------------------------------------------------
# 11. TOOL_SETUP sets _config
# ---------------------------------------------------------------------------


def test_tool_setup_sets_config(tmp_path):
    import src.tools.generate_tests as mod

    assert mod._config is None  # reset_module fixture ensures this

    cfg = _make_config()
    mod.TOOL_SETUP(cfg)
    assert mod._config is cfg
    assert mod.is_configured()


# ---------------------------------------------------------------------------
# 12. Result message includes line count
# ---------------------------------------------------------------------------


def test_result_message_includes_line_count(configured, source_file, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)
    code = "def test_a(): pass\ndef test_b(): pass\n"

    with _patch_llm(code):
        from src.tools.generate_tests import generate_tests

        result = generate_tests(str(source_file))

    # _extract_code strips trailing whitespace, leaving 2 lines
    assert "lines" in result
    assert "2" in result


# ---------------------------------------------------------------------------
# 13. Returns error when not configured
# ---------------------------------------------------------------------------


def test_returns_error_when_not_configured(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # _config is None due to reset_module fixture

    from src.tools.generate_tests import generate_tests

    result = generate_tests("some_file.py")
    assert result.startswith("Error:")
    assert "not configured" in result


# ---------------------------------------------------------------------------
# 14. Unsupported style returns error
# ---------------------------------------------------------------------------


def test_unsupported_style_returns_error(configured, source_file, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    from src.tools.generate_tests import generate_tests

    result = generate_tests(str(source_file), style="unittest")
    assert result.startswith("Error:")
    assert "unsupported style" in result


# ---------------------------------------------------------------------------
# 15. LLM timeout returns error within bounded time
# ---------------------------------------------------------------------------


def test_llm_invoke_timeout_returns_error(configured, source_file, tmp_path, monkeypatch):
    import threading
    import time

    import src.tools.generate_tests as mod

    monkeypatch.chdir(tmp_path)
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)

    mock_llm = MagicMock()
    stop_event = threading.Event()

    def _never_return(prompt):
        stop_event.wait(timeout=60)

    mock_llm.invoke.side_effect = _never_return

    with (
        patch("src.tools.generate_tests.create_chat_model_from_configs", return_value=mock_llm),
        patch.object(mod, "_GENERATE_TESTS_LLM_TIMEOUT_SECONDS", 0.1),
    ):
        from src.tools.generate_tests import generate_tests

        start = time.monotonic()
        result = generate_tests(str(source_file))
        elapsed = time.monotonic() - start

    stop_event.set()

    assert elapsed < 5, f"Expected return within 5s, took {elapsed}s"
    assert result.startswith("Error:")
    assert "timed out" in result.lower()


# ---------------------------------------------------------------------------
# 16. UserCancelledRun is re-raised, not swallowed
# ---------------------------------------------------------------------------


def test_user_cancelled_run_is_raised(configured, source_file, tmp_path, monkeypatch):
    from src.agent.safety import UserCancelledRun

    monkeypatch.chdir(tmp_path)
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = UserCancelledRun("user cancelled")

    with patch("src.tools.generate_tests.create_chat_model_from_configs", return_value=mock_llm):
        from src.tools.generate_tests import generate_tests

        with pytest.raises(UserCancelledRun):
            generate_tests(str(source_file))


# ---------------------------------------------------------------------------
# 17. Secrets are redacted before sending to LLM
# ---------------------------------------------------------------------------


def test_secrets_redacted_in_llm_prompt(configured, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)

    src = tmp_path / "src" / "api_client.py"
    src.parent.mkdir(parents=True)
    original = (
        "API_KEY = 'sk-testsecret1234567890'\n"
        "PASSWORD = 'super_secret_password_123'\n"
        "HEADERS = {'Authorization': 'Bearer abcdef1234567890'}\n"
        "\n"
        "def fetch_data():\n"
        "    pass\n"
    )
    src.write_text(original, encoding="utf-8")

    captured: list[str] = []
    mock_llm = MagicMock()

    def capture(msgs):
        captured.append(msgs[0].content)
        return _llm_response("def test_fetch(): pass")

    mock_llm.invoke.side_effect = capture

    with patch("src.tools.generate_tests.create_chat_model_from_configs", return_value=mock_llm):
        from src.tools.generate_tests import generate_tests

        generate_tests(str(src))

    assert captured, "LLM was not called"
    prompt = captured[0]

    # Secrets must be redacted
    assert "sk-testsecret1234567890" not in prompt
    assert "super_secret_password_123" not in prompt
    assert "abcdef1234567890" not in prompt
    # Redaction placeholders must be present
    assert "***REDACTED***" in prompt or "sk-***" in prompt
    # Original file must NOT be modified
    assert src.read_text(encoding="utf-8") == original
