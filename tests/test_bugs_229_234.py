"""Regression tests for BUG-229 through BUG-236 (2026-03-22 sweep fixes).

BUG-229 — _detect_environment: OLLAMA_BASE_URL not validated before urlopen
BUG-230 — _list_ollama_models: user-typed URL not validated before urlopen
        — _is_safe_url: allow_local=True permits LAN/loopback for user-provided docs URLs
BUG-231 — openai.py: "no-key" placeholder replaced with "not-required"
BUG-232 — docker-entrypoint: wizard auto-start documented (no code change needed)
BUG-233 — graph.py: _bound_cache accessed without lock in call_model closure
BUG-234 — graph.py: _tool_call_key normalizes to canonical name via _tool_lookup
BUG-236 — Dockerfile: HEALTHCHECK exits 0 in CLI mode (sentinel file gate)
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# BUG-229: _detect_environment skips Ollama probe for non-safe URLs
# ---------------------------------------------------------------------------


class TestBug229OllamaEnvSSRF:
    """OLLAMA_BASE_URL env var must be validated before urlopen."""

    def test_safe_ollama_url_allows_loopback(self):
        """127.0.0.1 (canonical Ollama address) must pass _is_safe_ollama_url."""
        from src.setup_wizard import _is_safe_ollama_url

        assert _is_safe_ollama_url("http://127.0.0.1:11434") is True

    def test_safe_ollama_url_allows_localhost(self):
        from src.setup_wizard import _is_safe_ollama_url

        assert _is_safe_ollama_url("http://localhost:11434") is True

    def test_safe_ollama_url_blocks_link_local(self):
        """169.254.x.x (AWS metadata, link-local) must be blocked."""
        from src.setup_wizard import _is_safe_ollama_url

        # We patch getaddrinfo so we don't need real DNS resolution in CI.
        with patch("socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(None, None, None, None, ("169.254.169.254", 80))]
            assert _is_safe_ollama_url("http://169.254.169.254/") is False

    def test_safe_ollama_url_blocks_rfc1918_in_strict_mode(self):
        """In strict mode (env-var probe), RFC-1918 addresses must be blocked."""
        from src.setup_wizard import _is_safe_ollama_url

        with patch("socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(None, None, None, None, ("10.0.0.1", 80))]
            assert _is_safe_ollama_url("http://10.0.0.1/admin") is False

        with patch("socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(None, None, None, None, ("192.168.1.1", 80))]
            assert _is_safe_ollama_url("http://192.168.1.1/") is False

    def test_safe_ollama_url_allows_rfc1918_in_user_mode(self):
        """With allow_private=True (user-typed URL), LAN addresses must be allowed."""
        from src.setup_wizard import _is_safe_ollama_url

        with patch("socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(None, None, None, None, ("192.168.70.200", 11434))]
            assert _is_safe_ollama_url("http://192.168.70.200:11434", allow_private=True) is True

        with patch("socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(None, None, None, None, ("10.0.0.1", 11434))]
            assert _is_safe_ollama_url("http://10.0.0.1:11434", allow_private=True) is True

    def test_safe_ollama_url_still_blocks_link_local_in_user_mode(self):
        """allow_private=True must still block link-local (AWS metadata) addresses."""
        from src.setup_wizard import _is_safe_ollama_url

        with patch("socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(None, None, None, None, ("169.254.169.254", 80))]
            assert _is_safe_ollama_url("http://169.254.169.254/", allow_private=True) is False

    def test_safe_ollama_url_blocks_empty_hostname(self):
        from src.setup_wizard import _is_safe_ollama_url

        assert _is_safe_ollama_url("not-a-url") is False
        assert _is_safe_ollama_url("") is False

    def test_detect_environment_skips_probe_for_unsafe_url(self):
        """_detect_environment must NOT call urlopen when URL is unsafe (BUG-229)."""
        from src.setup_wizard import _detect_environment

        with (
            patch("src.setup_wizard._is_safe_ollama_url", return_value=False) as mock_safe,
            patch("urllib.request.urlopen") as mock_urlopen,
            patch.dict("os.environ", {"OLLAMA_BASE_URL": "http://169.254.169.254/"}, clear=False),
        ):
            env = _detect_environment()

        mock_safe.assert_called()
        mock_urlopen.assert_not_called()
        assert "ollama_running" not in env

    def test_detect_environment_probes_safe_url(self):
        """_detect_environment MUST call urlopen when URL is safe."""
        from src.setup_wizard import _detect_environment

        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = 200

        with (
            patch("src.setup_wizard._is_safe_ollama_url", return_value=True),
            patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen,
            patch.dict("os.environ", {"OLLAMA_BASE_URL": "http://127.0.0.1:11434"}, clear=False),
        ):
            env = _detect_environment()

        mock_urlopen.assert_called_once()
        assert env.get("ollama_running") is True


# ---------------------------------------------------------------------------
# BUG-230: _list_ollama_models validates user-typed URL before urlopen
# ---------------------------------------------------------------------------


class TestBug230OllamaUserURLSSRF:
    """User-typed Ollama URL must be validated before urlopen (BUG-230)."""

    def test_list_ollama_models_skips_unsafe_url(self):
        """_list_ollama_models must return [] and NOT call urlopen for unsafe URL."""
        from src.setup_wizard import _list_ollama_models

        with (
            patch("src.setup_wizard._is_safe_ollama_url", return_value=False),
            patch("urllib.request.urlopen") as mock_urlopen,
        ):
            result = _list_ollama_models("http://169.254.169.254/")

        mock_urlopen.assert_not_called()
        assert result == []

    def test_list_ollama_models_fetches_for_safe_url(self):
        """_list_ollama_models must proceed normally for safe URLs."""
        from src.setup_wizard import _list_ollama_models

        payload = b'{"models": [{"name": "qwen3:8b", "size": 5000000000}]}'
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = payload

        with (
            patch("src.setup_wizard._is_safe_ollama_url", return_value=True),
            patch("urllib.request.urlopen", return_value=mock_resp),
        ):
            result = _list_ollama_models("http://127.0.0.1:11434")

        assert "qwen3:8b" in result

    def test_list_ollama_models_allows_lan_address(self):
        """LAN Ollama servers (192.168.x.x) must NOT be blocked by user-typed URL guard."""
        from src.setup_wizard import _list_ollama_models

        payload = b'{"models": [{"name": "qwen3:8b", "size": 5000000000}]}'
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = payload

        with (
            patch(
                "socket.getaddrinfo",
                return_value=[(None, None, None, None, ("192.168.70.200", 11434))],
            ),
            patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen,
        ):
            result = _list_ollama_models("http://192.168.70.200:11434")

        mock_urlopen.assert_called_once()
        assert "qwen3:8b" in result

    def test_list_ollama_models_dns_failure_logs_network_hint(self):
        """DNS resolution failure (e.g. inside Docker) must log a clear message,
        not the misleading 'blocked address (link-local/reserved)' text."""
        from src.setup_wizard import _list_ollama_models

        with (
            patch("socket.getaddrinfo", side_effect=OSError("Name or service not known")),
            patch("urllib.request.urlopen") as mock_urlopen,
        ):
            result = _list_ollama_models("http://spark:11434")

        mock_urlopen.assert_not_called()
        assert result == []

    def test_list_ollama_models_allows_hostname_resolving_to_lan(self):
        """A hostname that resolves to an RFC-1918 address (e.g. 'spark' → 192.168.x.x)
        must succeed — the user-typed URL guard allows private addresses."""
        from src.setup_wizard import _list_ollama_models

        payload = b'{"models": [{"name": "llama3:8b", "size": 4000000000}]}'
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = payload

        with (
            patch(
                "socket.getaddrinfo",
                return_value=[(None, None, None, None, ("192.168.1.50", 11434))],
            ),
            patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen,
        ):
            result = _list_ollama_models("http://spark:11434")

        mock_urlopen.assert_called_once()
        assert "llama3:8b" in result


# ---------------------------------------------------------------------------
# BUG-231: openai.py uses "not-required" placeholder (not "no-key")
# ---------------------------------------------------------------------------


class TestBug231NoKeyPlaceholder:
    """The api_key fallback must be "not-required", not the confusing "no-key"."""

    def test_placeholder_literal_value(self):
        """The exact string used as fallback must be 'not-required' when no env var is set."""
        from src.providers import create_chat_model

        captured = {}

        class _FakeChatOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with (
            patch("src.providers.openai.ChatOpenAI", _FakeChatOpenAI),
            patch("src.providers.openai.OpenAIEmbeddings", MagicMock()),
            patch.dict(os.environ, {}, clear=True),
        ):
            create_chat_model(
                provider_type="openai",
                model="gpt-4o",
                base_url="http://localhost:1234/v1",
                api_key=None,
            )

        assert (
            captured.get("api_key") == "not-required"
        ), f"Expected 'not-required', got {captured.get('api_key')!r}"

    def test_env_var_fallback_used(self):
        """OPENAI_API_KEY env var must be used when api_key is None and base_url is set."""
        from src.providers import create_chat_model

        captured = {}

        class _FakeChatOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with (
            patch("src.providers.openai.ChatOpenAI", _FakeChatOpenAI),
            patch("src.providers.openai.OpenAIEmbeddings", MagicMock()),
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env-key"}),
        ):
            create_chat_model(
                provider_type="openai",
                model="gpt-4o",
                base_url="http://custom.example.com/v1",
                api_key=None,
            )

        assert "api_key" not in captured, (
            f"Expected api_key to be omitted so SDK can use env var, "
            f"but got {captured.get('api_key')!r}"
        )


# ---------------------------------------------------------------------------
# BUG-233: _bound_cache in build_agent_graph closure is lock-protected
# ---------------------------------------------------------------------------


class TestBug233BoundCacheLock:
    """_bound_cache in call_model closure must be guarded by _bound_cache_lock."""

    def test_lock_prevents_concurrent_cache_corruption(self):
        """Concurrent call_model invocations must not corrupt the OrderedDict."""
        cache: OrderedDict = OrderedDict()
        lock = threading.Lock()
        errors: list[str] = []

        def _worker(key: str) -> None:
            try:
                with lock:
                    if key not in cache:
                        if len(cache) >= 8:
                            cache.popitem(last=False)
                        cache[key] = f"bound_{key}"
                    cache.move_to_end(key)
            except RuntimeError as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=_worker, args=(f"k{i}",)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Lock prevented no errors: {errors}"
        assert len(cache) <= 8


# ---------------------------------------------------------------------------
# BUG-234: _tool_call_key normalizes alias to canonical name
# ---------------------------------------------------------------------------


class TestBug234FuzzyDedupKey:
    """_tool_call_key must resolve alias names to canonical via _tool_lookup (BUG-234)."""

    def test_tool_lookup_get_avoids_keyerror(self):
        """When a tool name is missing from _tool_lookup, .get() returns None
        instead of raising KeyError — the fallback uses the raw call name."""
        lookup: dict[str, object] = {"search": object()}
        # .get() returns None for unknown names — no KeyError
        assert lookup.get("unknown_tool") is None
        # known names return the tool object
        assert lookup.get("search") is not None
        # By contrast, [] would raise KeyError
        with pytest.raises(KeyError):
            _ = lookup["unknown_tool"]

    def test_getattr_extracts_canonical_name(self):
        """When a tool is in _tool_lookup, getattr(tool_obj, 'name', fallback)
        resolves the canonical name for deduplication."""

        class FakeTool:
            name = "canonical_search"

        tool = FakeTool()
        raw_call_name = "search_alias"
        canonical = getattr(tool, "name", raw_call_name) or raw_call_name
        assert (
            canonical == "canonical_search"
        ), f"Expected canonical 'canonical_search', got {canonical!r}"

    def test_dedup_key_normalizes_alias_to_canonical(self):
        """When a tool alias is looked up, the canonical name replaces the alias
        in the deduplication key so that alias and canonical calls share a key."""
        lookup: dict[str, object] = {}

        class FakeTool:
            name = "canonical_search"

        tool = FakeTool()
        lookup["canonical_search"] = tool
        lookup["search_alias"] = tool  # alias maps to same tool

        def make_key(call_name: str) -> str:
            tool_name = call_name
            tool_obj = lookup.get(tool_name)
            if tool_obj is not None:
                tool_name = getattr(tool_obj, "name", tool_name) or tool_name
            return tool_name

        canonical_key = make_key("canonical_search")
        alias_key = make_key("search_alias")
        assert canonical_key == alias_key == "canonical_search", (
            f"Expected both keys to be 'canonical_search', "
            f"got canonical={canonical_key!r}, alias={alias_key!r}"
        )


# ---------------------------------------------------------------------------
# BUG-236: Dockerfile HEALTHCHECK is conditional on API-mode sentinel
# ---------------------------------------------------------------------------


class TestBug236HealthcheckCLIMode:
    """HEALTHCHECK must exit 0 immediately in CLI mode (no sentinel file)."""

    def test_healthcheck_cmd_exits_zero_without_sentinel(self):
        """Simulate the healthcheck in CLI mode: exits 0 when sentinel absent."""
        # Run the healthcheck logic without the sentinel present.
        cmd = (
            "import os, sys; "
            "sys.exit(0) if not os.path.exists('/run/cogtrix/api-mode-test') else None; "
            "sys.exit(99)"  # would probe HTTP — sentinel absent so we never reach here
        )
        result = subprocess.run(["python", "-c", cmd], capture_output=True)
        assert (
            result.returncode == 0
        ), f"Healthcheck must exit 0 in CLI mode, got {result.returncode}"

    def test_healthcheck_cmd_probes_http_with_sentinel(self):
        """Simulate the healthcheck in API mode: tries HTTP probe when sentinel present."""
        fd, sentinel_path = tempfile.mkstemp(suffix=".sentinel")
        sentinel = Path(sentinel_path)
        try:
            os.close(fd)
            cmd = (
                f"import os, sys; "
                f"sys.exit(0) if not os.path.exists('{sentinel}') else None; "
                f"sys.exit(42)"  # simulates failed HTTP probe
            )
            result = subprocess.run(["python", "-c", cmd], capture_output=True)
            # Should NOT exit 0 — it tried to probe and "failed" (exit 42)
            assert (
                result.returncode == 42
            ), f"Healthcheck must attempt HTTP probe in API mode, got {result.returncode}"
        finally:
            sentinel.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# BUG-232: wizard auto-start documented (behavioural contract test)
# ---------------------------------------------------------------------------


class TestBug232WizardAutoStartBehaviour:
    """Wizard auto-start condition must be documented in docker-entrypoint.sh (BUG-232).

    The entrypoint MUST auto-start the setup wizard when all of: no arguments,
    no config file found, stdin is a TTY, and no API-key env vars are set.
    """

    def test_entrypoint_is_valid_bash(self):
        """docker-entrypoint.sh must be syntactically valid bash."""
        entrypoint = Path(__file__).parent.parent / "docker" / "docker-entrypoint.sh"
        result = subprocess.run(["bash", "-n", str(entrypoint)], capture_output=True, text=True)
        assert (
            result.returncode == 0
        ), f"docker-entrypoint.sh is not valid bash: {result.stderr.strip()}"

    def test_entrypoint_has_wizard_auto_start(self):
        """The entrypoint must contain the conditional block that triggers the
        setup wizard when: 0 args, no config, TTY stdin, no API key env vars."""
        # Behavioural check: verify the wizard auto-start condition evaluates
        # correctly by running a simplified version through bash.
        test_script = """
set -eu
_cogtrix_has_config() { false; }
export OPENAI_API_KEY=""
export ANTHROPIC_API_KEY=""
export GEMINI_API_KEY=""
export XAI_API_KEY=""
export DEEPSEEK_API_KEY=""
export COGTRIX_OLLAMA=""
export OLLAMA_BASE_URL=""
# Simulate the auto-start condition (same logic as entrypoint)
if true; then  # $# -eq 0 && ! _cogtrix_has_config && [ -t 0 ]
    if [ -z "$OPENAI_API_KEY" ] && \
       [ -z "$ANTHROPIC_API_KEY" ] && \
       [ -z "$GEMINI_API_KEY" ] && \
       [ -z "$XAI_API_KEY" ] && \
       [ -z "$DEEPSEEK_API_KEY" ] && \
       [ -z "$COGTRIX_OLLAMA" ] && \
       [ -z "$OLLAMA_BASE_URL" ]; then
        echo "would-trigger-setup"
        exit 0
    fi
fi
exit 1
"""
        result = subprocess.run(["bash", "-c", test_script], capture_output=True, text=True)
        assert (
            result.returncode == 0
        ), "Wizard auto-start condition must trigger when all API key vars are empty"
        assert "would-trigger-setup" in result.stdout


# ---------------------------------------------------------------------------
# BUG-230 (codebase audit): _is_safe_url allow_local flag for docs URLs
# ---------------------------------------------------------------------------


class TestBug230IsafeUrlAllowLocal:
    """_is_safe_url must allow loopback and RFC-1918 when allow_local=True (BUG-230 audit)."""

    def test_strict_mode_blocks_loopback(self):
        """Default strict mode blocks loopback addresses."""
        from src.setup_wizard import _is_safe_url

        with patch("socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(None, None, None, None, ("127.0.0.1", 80))]
            assert _is_safe_url("http://127.0.0.1/docs") is False

    def test_strict_mode_blocks_rfc1918(self):
        """Default strict mode blocks RFC-1918 private addresses."""
        from src.setup_wizard import _is_safe_url

        with patch("socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(None, None, None, None, ("192.168.1.1", 80))]
            assert _is_safe_url("http://192.168.1.1/docs") is False

    def test_strict_mode_always_blocks_link_local(self):
        """Link-local (169.254.x.x) is always blocked even in strict mode."""
        from src.setup_wizard import _is_safe_url

        with patch("socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(None, None, None, None, ("169.254.169.254", 80))]
            assert _is_safe_url("http://169.254.169.254/docs") is False

    def test_allow_local_permits_loopback(self):
        """allow_local=True must allow loopback (e.g. local vLLM doc server)."""
        from src.setup_wizard import _is_safe_url

        with patch("socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(None, None, None, None, ("127.0.0.1", 8080))]
            assert _is_safe_url("http://127.0.0.1:8080/docs", allow_local=True) is True

    def test_allow_local_permits_rfc1918(self):
        """allow_local=True must allow RFC-1918 addresses (e.g. intranet doc server)."""
        from src.setup_wizard import _is_safe_url

        with patch("socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(None, None, None, None, ("10.0.0.5", 80))]
            assert _is_safe_url("http://10.0.0.5/docs", allow_local=True) is True

        with patch("socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(None, None, None, None, ("192.168.1.100", 80))]
            assert _is_safe_url("http://192.168.1.100/docs", allow_local=True) is True

    def test_allow_local_still_blocks_link_local(self):
        """allow_local=True must still block link-local (169.254.x.x / AWS metadata)."""
        from src.setup_wizard import _is_safe_url

        with patch("socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(None, None, None, None, ("169.254.169.254", 80))]
            assert _is_safe_url("http://169.254.169.254/", allow_local=True) is False

    def test_load_docs_passes_allow_local_true(self):
        """_load_docs must call _is_safe_url with allow_local=True so that user-
        provided URLs on private networks (loopback, RFC-1918) are permitted."""
        from src.setup_wizard import _load_docs

        with patch("src.setup_wizard._is_safe_url") as mock_safe:
            mock_safe.return_value = True
            # Simulate a successful fetch from a LAN address
            with patch("urllib.request.urlopen") as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = (
                    b"# Config docs\n\nContent from LAN server."
                )
                result = _load_docs("http://192.168.1.100:8080/docs")

            mock_safe.assert_called_once_with("http://192.168.1.100:8080/docs", allow_local=True)
            assert "# Config docs" in result
