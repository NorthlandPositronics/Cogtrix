"""End-to-end scenario tests for the setup wizard.

These tests drive ``run_setup_wizard()`` through scripted ``input()`` queues
that mirror real interactive sessions.  Each test is a named scenario.

Infrastructure
--------------
``_scenario()`` is a ``@contextmanager`` that sets up all patches via
``contextlib.ExitStack``.  Each test passes a list of answers (in prompt order)
and any scenario-specific overrides:

  answers       — fed to ``builtins.input`` via ``side_effect=iter(answers)``
  llm           — mock LLM returned on every ``_test_connection`` call
  test_conn     — pre-built ``_test_connection`` mock (for retry tests with
                  ``side_effect=[None, llm]``); overrides *llm*
  env           — ``_detect_environment`` return value (default ``{}``)
  models        — ``_list_ollama_models`` return value (default ``[]``)
  existing      — ``_load_existing_config`` return value (default ``("", None)``)
  extra         — additional ``patch()`` objects (e.g. ``_read_masked_input``)

How answers map to prompts
--------------------------
``_ask_choice`` and ``_ask_input`` both call ``builtins.input``.
``_ask_input(secret=True)`` calls ``_read_masked_input()`` instead — patched via
*extra* when needed.
Empty string ``""`` accepts the displayed default for any prompt.
"""

from __future__ import annotations

import textwrap
from contextlib import ExitStack, contextmanager
from unittest.mock import MagicMock, patch

import pytest
import yaml

from cogtrix_core.setup_wizard import run_setup_wizard

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm(*responses: str) -> MagicMock:
    """Return a mock LLM that yields *responses* in order on ``.invoke()``."""
    llm = MagicMock()
    llm.invoke.side_effect = [MagicMock(content=r) for r in responses]
    return llm


def _yaml_response(
    provider: str = "ollama",
    model: str = "qwen3.5:9b",
    base_url: str = "http://192.168.70.200:11434",
    *,
    with_next_steps: bool = False,
) -> str:
    """Return a plausible LLM message containing a ```yaml``` block.

    When *with_next_steps* is True, appends a "Next steps" section with shell
    code fences after the YAML block — this exercises the _extract_yaml
    trailing-fence fix.
    """
    providers_line = f"  {provider}:\n" f"    type: {provider}\n" + (
        f'    base_url: "{base_url}"\n' if base_url else ""
    )
    config = (
        "providers:\n" + providers_line + "models:\n"
        "  default: main\n"
        "  main:\n"
        f"    provider: {provider}\n"
        f"    model: {model}\n"
    )
    block = f"Here is your configuration:\n\n```yaml\n{config}```\n"
    if not with_next_steps:
        return block
    return (
        block + "\nNext steps:\n\n"
        "1. Save as `~/.cogtrix.yaml`\n\n"
        "2. Create required directories:\n\n"
        "```\nmkdir -p docs vectordb\n```\n\n"
        "3. Start Cogtrix:\n\n"
        "```\npython cogtrix.py\n```\n"
    )


@contextmanager
def _scenario(
    answers,
    *,
    llm=None,
    test_conn=None,
    env=None,
    models=None,
    existing=("", None),
    extra=(),
):
    """Context manager: activate all patches for one wizard scenario.

    Uses ExitStack so the variable-length *extra* list can be added without
    tuple-unpacking hacks inside a ``with`` statement.
    """
    if test_conn is None:
        test_conn = MagicMock(return_value=llm)

    patches = [
        patch("builtins.input", side_effect=iter(answers)),
        patch("cogtrix_core.setup_wizard._test_connection", new=test_conn),
        patch("cogtrix_core.setup_wizard._detect_environment", return_value=env or {}),
        patch("cogtrix_core.setup_wizard._load_existing_config", return_value=existing),
        patch("cogtrix_core.setup_wizard._list_ollama_models", return_value=models or []),
        patch("cogtrix_core.setup_wizard._load_docs", return_value="docs content"),
        patch("cogtrix_core.config._apply_config_file"),
        *extra,
    ]
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield


# ---------------------------------------------------------------------------
# Scenario tests
# ---------------------------------------------------------------------------


class TestWizardScenario:
    """Full end-to-end wizard flow — one test per scenario."""

    # ------------------------------------------------------------------
    # 1. Ollama LAN server — happy path
    # ------------------------------------------------------------------

    def test_ollama_lan_server_creates_valid_config(self, tmp_path):
        """Ollama on a LAN IP (192.168.x.x) goes through all three steps
        and writes a valid, parseable config file.

        Prompt sequence:
          provider type   → "ollama"
          Ollama URL      → "http://192.168.70.200:11434"
          model           → "qwen3.5:9b"
          production?     → "no"
          accept YAML?    → "yes"
          write file?     → "yes"
        """
        out = tmp_path / ".cogtrix.yaml"
        llm = _make_llm(_yaml_response("ollama", "qwen3.5:9b", "http://192.168.70.200:11434"))

        answers = [
            "ollama",
            "http://192.168.70.200:11434",
            "qwen3.5:9b",
            "no",  # no separate production model
            "yes",  # accept YAML
            "yes",  # write file
        ]

        with _scenario(answers, llm=llm, models=["qwen3.5:9b", "qwen3:8b"]):
            run_setup_wizard(output_path=out)

        assert out.exists(), "config file must be written"
        cfg = yaml.safe_load(out.read_text())
        assert cfg["providers"]["ollama"]["base_url"] == "http://192.168.70.200:11434"
        # _inject_bootstrap always sets default alias to "default_model"
        assert cfg["models"]["default"] == "default_model"
        assert cfg["models"]["default_model"]["model"] == "qwen3.5:9b"

    # ------------------------------------------------------------------
    # 2. OpenAI — API key auto-detected from environment
    # ------------------------------------------------------------------

    def test_openai_with_env_key_no_key_prompt(self, tmp_path):
        """OPENAI_API_KEY in env → wizard skips the key prompt entirely and
        uses the env key directly.

        Prompt sequence:
          provider type   → "" (accept default "openai" suggested by env)
          base URL        → "" (accept default)
          model           → "" (accept default)
          production?     → "no"
          accept YAML?    → "yes"
          write file?     → "yes"
        """
        out = tmp_path / ".cogtrix.yaml"
        llm = _make_llm(_yaml_response("openai", "gpt-4.1-mini", ""))

        answers = [
            "",  # provider type — default "openai" because env has openai_key
            "",  # base URL
            "",  # model
            "no",  # no production model
            "yes",  # accept YAML
            "yes",  # write
        ]

        with _scenario(answers, llm=llm, env={"openai_key": "sk-env-key"}):
            run_setup_wizard(output_path=out)

        assert out.exists()
        cfg = yaml.safe_load(out.read_text())
        # _inject_bootstrap writes the env key into the providers section
        assert cfg["providers"]["openai"]["api_key"] == "sk-env-key"

    # ------------------------------------------------------------------
    # 3. Anthropic — key typed manually (secret / masked input)
    # ------------------------------------------------------------------

    def test_anthropic_with_manually_entered_key(self, tmp_path):
        """User types an Anthropic key via the masked-input prompt.

        _ask_input(secret=True) calls _read_masked_input(), which is patched
        separately from builtins.input — so the key does NOT appear in the
        answers list.

        Prompt sequence (via builtins.input):
          provider type   → "anthropic"
          model           → "" (accept default claude-sonnet-4-5)
          production?     → "no"
          accept YAML?    → "yes"
          write file?     → "yes"
        Secret prompt (via _read_masked_input):
          API key         → "sk-ant-test"
        """
        out = tmp_path / ".cogtrix.yaml"
        llm = _make_llm(_yaml_response("anthropic", "claude-sonnet-4-5", ""))

        answers = [
            "anthropic",
            # API key comes from _read_masked_input (see extra= below)
            "",  # model
            "no",  # no production model
            "yes",  # accept YAML
            "yes",  # write
        ]

        with _scenario(
            answers,
            llm=llm,
            extra=[
                patch("cogtrix_core.setup_wizard._read_masked_input", return_value="sk-ant-test")
            ],
        ):
            run_setup_wizard(output_path=out)

        assert out.exists()
        cfg = yaml.safe_load(out.read_text())
        assert cfg["providers"]["anthropic"]["api_key"] == "sk-ant-test"

    # ------------------------------------------------------------------
    # 4. Retry flow — first connection fails, user corrects URL and retries
    # ------------------------------------------------------------------

    def test_connection_fails_then_succeeds_on_retry(self, tmp_path):
        """_test_connection returns None first (simulated timeout); user retries
        with the correct URL; the second attempt succeeds.

        Prompt sequence (1st attempt):
          provider type   → "ollama"
          Ollama URL      → "http://bad-host:11434"
          model           → ""
          retry?          → "yes"
        Prompt sequence (2nd attempt):
          provider type   → "" (default "ollama" carried from last)
          Ollama URL      → "http://192.168.70.200:11434"
          model           → ""
        Then:
          production?     → "no"
          accept YAML?    → "yes"
          write file?     → "yes"
        """
        out = tmp_path / ".cogtrix.yaml"
        llm = _make_llm(_yaml_response())
        test_conn = MagicMock(side_effect=[None, llm])  # fail, then succeed

        answers = [
            "ollama",  # 1st: provider type
            "http://bad-host:11434",  # 1st: bad URL
            "",  # 1st: model (default)
            "yes",  # retry
            "",  # 2nd: provider type (default "ollama")
            "http://192.168.70.200:11434",  # 2nd: corrected URL
            "",  # 2nd: model (default)
            "no",  # no production model
            "yes",  # accept YAML
            "yes",  # write
        ]

        with _scenario(
            answers,
            test_conn=test_conn,
            models=["qwen3.5:9b"],
        ):
            run_setup_wizard(output_path=out)

        assert out.exists()
        assert test_conn.call_count == 2
        # Second call must use the corrected URL (4th positional arg)
        _, _, _, base_url = test_conn.call_args_list[1][0]
        assert base_url == "http://192.168.70.200:11434"

    # ------------------------------------------------------------------
    # 5. Retry declined — wizard exits without writing anything
    # ------------------------------------------------------------------

    def test_retry_declined_exits_without_writing(self, tmp_path):
        """_test_connection always fails; user says 'no' to retry → SystemExit,
        no config file created.
        """
        out = tmp_path / ".cogtrix.yaml"

        answers = [
            "ollama",
            "http://bad-host:11434",
            "",
            "no",  # decline retry → SystemExit(1)
        ]

        with _scenario(answers, test_conn=MagicMock(return_value=None)):
            with pytest.raises(SystemExit):
                run_setup_wizard(output_path=out)

        assert not out.exists()

    # ------------------------------------------------------------------
    # 6. Multi-turn conversation — LLM asks a question first
    # ------------------------------------------------------------------

    def test_llm_asks_questions_before_producing_yaml(self, tmp_path):
        """LLM sends a question (no YAML block) on the first call; user answers;
        LLM then produces the YAML on the second call.

        Prompt sequence:
          provider type         → "ollama"
          Ollama URL            → "" (default)
          model                 → ""
          production?           → "no"
          (LLM call 1 — question, no YAML block)
          answer to LLM         → "interactive assistant"
          (LLM call 2 — YAML block)
          accept YAML?          → "yes"
          write file?           → "yes"
        """
        out = tmp_path / ".cogtrix.yaml"
        llm = _make_llm(
            "What do you want to use Cogtrix for?",  # call 1: no ```yaml``` block
            _yaml_response(),  # call 2: has ```yaml``` block
        )

        answers = [
            "ollama",
            "",
            "",
            "no",
            "interactive assistant",  # user's answer to LLM question
            "yes",
            "yes",
        ]

        with _scenario(answers, llm=llm):
            run_setup_wizard(output_path=out)

        assert out.exists()
        assert llm.invoke.call_count == 2

    # ------------------------------------------------------------------
    # 7. Reject → feedback → accept revised config
    # ------------------------------------------------------------------

    def test_user_rejects_config_then_accepts_revised(self, tmp_path):
        """User rejects the first YAML, provides feedback, then accepts the
        revised version.  LLM must be invoked exactly twice.

        Prompt sequence:
          ... (bootstrap) ...
          production?           → "no"
          (LLM call 1: YAML v1 with qwen3:8b)
          accept YAML?          → "no, continue editing"
          feedback to LLM       → "use qwen3.5:9b instead"
          (LLM call 2: YAML v2 with qwen3.5:9b)
          accept YAML?          → "yes"
          write file?           → "yes"
        """
        out = tmp_path / ".cogtrix.yaml"
        llm = _make_llm(
            _yaml_response("ollama", "qwen3:8b"),  # v1 — user rejects
            _yaml_response("ollama", "qwen3.5:9b"),  # v2 — user accepts
        )

        answers = [
            "ollama",
            "",
            "",
            "no",
            "no, continue editing",  # reject first YAML
            "use qwen3.5:9b instead",  # feedback
            "yes",  # accept revised YAML
            "yes",  # write
        ]

        with _scenario(answers, llm=llm):
            run_setup_wizard(output_path=out)

        assert out.exists()
        assert llm.invoke.call_count == 2
        cfg = yaml.safe_load(out.read_text())
        # "main" alias comes from the LLM's revised YAML — must have qwen3.5:9b
        assert cfg["models"]["main"]["model"] == "qwen3.5:9b"
        # "default_model" comes from bootstrap_info (user typed "" → qwen3:8b default)
        assert cfg["models"]["default_model"]["model"] == "qwen3:8b"

    # ------------------------------------------------------------------
    # 8. Production model different from bootstrap (two-provider config)
    # ------------------------------------------------------------------

    def test_separate_production_model_written_to_config(self, tmp_path):
        """User selects a separate production model in Step 1; the wizard calls
        _test_connection twice (bootstrap ollama, then production openai).
        Both providers appear in the final config.

        Prompt sequence:
          (bootstrap)
          provider type   → "ollama" / URL → "" / model → ""
          production?     → "yes"
          (production — OpenAI key from env)
          provider type   → "openai" / base URL → "" / model → ""
          accept YAML?    → "yes"
          write file?     → "yes"
        """
        out = tmp_path / ".cogtrix.yaml"
        bootstrap_llm = _make_llm(_yaml_response("openai", "gpt-4.1-mini", ""))
        prod_llm = MagicMock()
        test_conn = MagicMock(side_effect=[bootstrap_llm, prod_llm])

        answers = [
            # Bootstrap (ollama)
            "ollama",
            "",
            "",
            # Production model prompt
            "yes",  # configure separate production model
            "openai",  # production provider type (key from env)
            "",  # base URL (default)
            "",  # model (default)
            # Step 2 / 3
            "yes",  # accept YAML
            "yes",  # write
        ]

        with _scenario(
            answers,
            test_conn=test_conn,
            env={"openai_key": "sk-prod-key"},
            models=["qwen3.5:9b"],
        ):
            run_setup_wizard(output_path=out)

        assert out.exists()
        assert test_conn.call_count == 2
        cfg = yaml.safe_load(out.read_text())
        providers = cfg.get("providers", {})
        assert "ollama" in providers
        assert "openai" in providers
        assert providers["openai"]["api_key"] == "sk-prod-key"

    # ------------------------------------------------------------------
    # 9. Existing config found — user edits it
    # ------------------------------------------------------------------

    def test_edit_existing_config_mode(self, tmp_path):
        """Wizard detects an existing config and asks 'edit existing / create new'.
        User picks 'edit existing'; existing URL and model become the defaults.

        Prompt sequence:
          mode            → "edit existing"
          provider type   → "" (default "ollama" from existing)
          Ollama URL      → "" (accept existing URL)
          model           → "" (accept existing model)
          production?     → "no"
          accept YAML?    → "yes"
          write file?     → "yes"
        """
        existing_yaml = textwrap.dedent("""
            providers:
              ollama:
                type: ollama
                base_url: "http://127.0.0.1:11434"
            models:
              default: main
              main:
                provider: ollama
                model: qwen3:8b
        """).strip()
        existing_path = tmp_path / "existing.yaml"
        existing_path.write_text(existing_yaml)

        out = tmp_path / ".cogtrix.yaml"
        llm = _make_llm(_yaml_response("ollama", "qwen3:8b", "http://127.0.0.1:11434"))

        answers = [
            "edit existing",  # mode choice (existing config was found)
            "",  # provider type — default "ollama" from existing
            "",  # Ollama URL — default from existing
            "",  # model — default from existing
            "no",  # no production model
            "yes",  # accept YAML
            "yes",  # write
        ]

        with _scenario(
            answers,
            llm=llm,
            existing=(existing_yaml, existing_path),
            models=["qwen3:8b"],
        ):
            run_setup_wizard(output_path=out)

        assert out.exists()
        cfg = yaml.safe_load(out.read_text())
        assert cfg["providers"]["ollama"]["base_url"] == "http://127.0.0.1:11434"

    # ------------------------------------------------------------------
    # 10. User declines final write — no file created
    # ------------------------------------------------------------------

    def test_write_declined_no_file_created(self, tmp_path):
        """User accepts the YAML in step 2 but says 'no' at the write prompt.
        The wizard calls SystemExit(0) and must not create the output file.

        Prompt sequence:
          ... (bootstrap) ...
          production?     → "no"
          accept YAML?    → "yes"
          write file?     → "no"   → SystemExit(0)
        """
        out = tmp_path / ".cogtrix.yaml"
        llm = _make_llm(_yaml_response())

        answers = [
            "ollama",
            "",
            "",
            "no",
            "yes",  # accept YAML in step 2
            "no",  # decline write → SystemExit(0)
        ]

        with _scenario(answers, llm=llm):
            with pytest.raises(SystemExit) as exc_info:
                run_setup_wizard(output_path=out)

        assert exc_info.value.code == 0
        assert not out.exists()

    # ------------------------------------------------------------------
    # 11. Regression: Next-steps code fences must not bleed into YAML
    # ------------------------------------------------------------------

    def test_next_steps_code_fences_do_not_corrupt_config(self, tmp_path):
        """When the LLM appends a Next-steps section with shell code fences
        (```mkdir```, ```python```) after the YAML block, the written config
        must still be valid YAML — no backtick characters anywhere.

        This is a direct replay of the bug the user reported:
        'Invalid YAML: found character \\x60 that cannot start any token'.
        The old _extract_yaml greedy fallback would span from ```yaml to the
        last ``` in the response, injecting shell commands into the YAML string.
        """
        # Exact structure produced by qwen3.5:9b in the user's session
        response = (
            "Based on your answers, here is your complete Cogtrix configuration:\n\n"
            "```yaml\n"
            "providers:\n"
            "  ollama:\n"
            "    type: ollama\n"
            '    base_url: "http://192.168.70.200:11434"\n'
            "models:\n"
            "  default: main\n"
            "  main:\n"
            "    provider: ollama\n"
            "    model: qwen3.5:9b\n"
            "```\n\n"
            "Next steps:\n\n"
            "1. Save this file as `~/.cogtrix.yaml`\n\n"
            "2. Create the required directories:\n\n"
            "```\n"
            "mkdir -p docs vectordb\n"
            "```\n\n"
            "3. Start Cogtrix:\n\n"
            "```\n"
            "python cogtrix.py\n"
            "```\n"
        )
        out = tmp_path / ".cogtrix.yaml"
        llm = _make_llm(response)

        answers = [
            "ollama",
            "http://192.168.70.200:11434",
            "qwen3.5:9b",
            "no",
            "yes",
            "yes",
        ]

        with _scenario(answers, llm=llm, models=["qwen3.5:9b"]):
            run_setup_wizard(output_path=out)

        assert out.exists()
        text = out.read_text()
        assert "`" not in text, "backtick characters must not appear in the written config"
        cfg = yaml.safe_load(text)
        assert cfg["providers"]["ollama"]["base_url"] == "http://192.168.70.200:11434"
        assert cfg["models"]["main"]["model"] == "qwen3.5:9b"
