"""Unit tests for the curl_cffi/OpenSSL coexistence guard (Bug D).

Background: ``curl_cffi`` (libcurl + BoringSSL) loaded into the same
Python process as ``httpx`` / ``urllib3`` (OpenSSL via Python stdlib
ssl) produces a glibc heap abort on the first TLS call. The
defence is to never import ``curl_cffi`` in the parent — the DDG
scraper runs in a subprocess. This module pins the *detection*
invariant so a regression that re-introduces the in-process
``curl_cffi`` import surfaces as a Python-level signal rather than
a silent ``"double free or corruption (!prev)"`` abort.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import sys
import types

import pytest

from cogtrix_core.tools import _native_safety
from cogtrix_core.tools._native_safety import (
    detect_curl_cffi_openssl_coexistence,
    warn_if_unsafe,
)


@pytest.fixture(autouse=True)
def _reset_one_shot_flag() -> None:
    """Clear the warn_if_unsafe one-shot flag between cases."""
    _native_safety._reset_warning_emitted_for_tests()


def _strip_native_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every ``curl_cffi*`` / ``httpx*`` / ``urllib3*`` entry from
    ``sys.modules`` for the duration of the test.

    A prior test in the same xdist worker may have triggered an import
    that left ``curl_cffi.curl`` / ``curl_cffi.const`` / submodules of
    ``httpx`` in ``sys.modules`` (e.g. ``tests/tools/test_ddg.py``
    exercises the real ``fetch_ddg_html`` path). The detection registry
    matches submodule names exactly, so a single leftover key causes
    false positives in the "no-coexistence" baseline tests.

    Sweeping by prefix is robust to whatever the real package layout
    happens to be in the runner's Python build — no hard-coded list to
    drift.
    """
    for key in list(sys.modules):
        if (
            key == "curl_cffi"
            or key.startswith("curl_cffi.")
            or key == "httpx"
            or key.startswith("httpx.")
            or key == "urllib3"
            or key.startswith("urllib3.")
        ):
            monkeypatch.delitem(sys.modules, key, raising=False)


@pytest.fixture
def clean_native_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip all ``curl_cffi*`` / ``httpx*`` / ``urllib3*`` entries
    from ``sys.modules`` for the duration of the test.

    Establishes the deterministic baseline. Compose with the
    ``inject_curl_cffi`` / ``inject_httpx`` helpers below to layer
    specific modules ON TOP of the clean state — those helpers
    deliberately do NOT strip, so multiple injections compose
    correctly when a test needs both backends present.
    """
    _strip_native_modules(monkeypatch)


def _inject(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setitem(sys.modules, name, types.ModuleType(name))


@contextlib.contextmanager
def _isolated_reimport(module_name: str):
    """Re-import *module_name* from scratch for a side-effect check, then fully
    restore the ORIGINAL module object.

    A plain ``del sys.modules[...] + importlib.import_module(...)`` (even via
    ``monkeypatch.delitem``) leaves the *parent package attribute* (e.g.
    ``src.tools.web_search``) pointing at the throwaway re-imported module —
    ``monkeypatch`` only restores the ``sys.modules`` dict entry, not the
    attribute. Any other test that bound ``from <module> import name`` at import
    time then reads a module whose globals ``monkeypatch.setattr`` can no longer
    reach. That decoupling broke ``test_web_search``'s outer-deadline test under
    ``--dist=loadfile`` once both files shared an xdist worker. Restore both the
    ``sys.modules`` entry and the package attribute so the re-import is fully
    isolated regardless of file ordering.
    """
    parent_name, _, attr = module_name.rpartition(".")
    parent = sys.modules.get(parent_name)
    original = sys.modules.get(module_name)
    sys.modules.pop(module_name, None)
    try:
        importlib.import_module(module_name)
        yield
    finally:
        if original is not None:
            sys.modules[module_name] = original
            if parent is not None:
                setattr(parent, attr, original)
        else:
            sys.modules.pop(module_name, None)
            if parent is not None and hasattr(parent, attr):
                delattr(parent, attr)


class TestDetect:
    def test_no_curl_cffi_no_openssl(self, clean_native_modules) -> None:
        # Clean state: neither side loaded. Should report no coexistence.
        coexists, curl_hits, openssl_hits = detect_curl_cffi_openssl_coexistence()
        assert coexists is False
        assert curl_hits == []
        assert openssl_hits == []

    def test_only_openssl_present(
        self,
        clean_native_modules,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _inject(monkeypatch, "httpx")

        coexists, curl_hits, openssl_hits = detect_curl_cffi_openssl_coexistence()
        # httpx alone is the normal agent state — must NOT trigger.
        assert coexists is False
        assert curl_hits == []
        assert "httpx" in openssl_hits

    def test_only_curl_cffi_present(
        self,
        clean_native_modules,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _inject(monkeypatch, "curl_cffi")

        coexists, curl_hits, openssl_hits = detect_curl_cffi_openssl_coexistence()
        # curl_cffi alone (e.g. the subprocess worker) is safe.
        assert coexists is False
        assert "curl_cffi" in curl_hits
        assert openssl_hits == []

    def test_both_present_triggers(
        self,
        clean_native_modules,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _inject(monkeypatch, "curl_cffi")
        _inject(monkeypatch, "httpx")

        coexists, curl_hits, openssl_hits = detect_curl_cffi_openssl_coexistence()
        assert coexists is True
        assert "curl_cffi" in curl_hits
        assert "httpx" in openssl_hits

    def test_detects_curl_cffi_submodule_even_without_parent(
        self,
        clean_native_modules,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Inject only the submodule (no top-level curl_cffi) plus httpx
        # so the coexistence check fires off the submodule alone.
        _inject(monkeypatch, "curl_cffi.requests")
        _inject(monkeypatch, "httpx")

        coexists, curl_hits, _ = detect_curl_cffi_openssl_coexistence()
        assert coexists is True
        assert "curl_cffi.requests" in curl_hits


class TestWarnIfUnsafe:
    def test_silent_when_safe(
        self,
        clean_native_modules,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="cogtrix"):
            assert warn_if_unsafe() is False
        assert "NATIVE_TLS_COEXISTENCE" not in caplog.text

    def test_warns_when_both_present(
        self,
        clean_native_modules,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _inject(monkeypatch, "curl_cffi")
        _inject(monkeypatch, "httpx")

        with caplog.at_level(logging.WARNING, logger="cogtrix"):
            assert warn_if_unsafe(context="unit-test") is True
        # The warning must surface BOTH the curl_cffi side and the
        # OpenSSL side so operators know which import to investigate.
        assert "NATIVE_TLS_COEXISTENCE" in caplog.text
        assert "curl_cffi" in caplog.text
        assert "httpx" in caplog.text
        assert "unit-test" in caplog.text

    def test_only_warns_once(
        self,
        clean_native_modules,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _inject(monkeypatch, "curl_cffi")
        _inject(monkeypatch, "httpx")

        # First call emits; second call is suppressed so per-tool
        # dispatch on a misconfigured environment doesn't spam logs.
        with caplog.at_level(logging.WARNING, logger="cogtrix"):
            warn_if_unsafe()
            caplog.clear()
            warn_if_unsafe()
        assert "NATIVE_TLS_COEXISTENCE" not in caplog.text


class TestModuleHygiene:
    """Pins the runtime invariant: importing the package must NOT pull
    curl_cffi into sys.modules.

    If a future change accidentally adds ``import curl_cffi`` at
    module level — or imports a module that does — this test fails
    fast in CI rather than letting a heap-corruption regression ship.
    """

    def test_importing_web_search_does_not_load_curl_cffi(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Sweep ALL curl_cffi.* entries so a prior test's real-import
        # leftovers (e.g. tests/tools/test_ddg.py) don't poison the
        # baseline. The detection registry matches submodule names
        # exactly — a single stale key would mask a regression.
        _strip_native_modules(monkeypatch)

        # Re-import the web_search module from scratch so module-level
        # side effects re-execute — fully isolated so we don't decouple
        # other tests' bound references (see _isolated_reimport).
        with _isolated_reimport("cogtrix_core.tools.web_search"):
            assert "curl_cffi" not in sys.modules, (
                "Importing src.tools.web_search must NOT pull curl_cffi into the "
                "parent process — Bug D / cogtrix46 heap corruption regression. "
                "curl_cffi belongs ONLY in the DDG subprocess worker."
            )

    def test_importing_ddg_module_does_not_load_curl_cffi(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Same invariant for the lower-level _ddg primitive — importing
        # the module must NOT call fetch_ddg_html, which is the only
        # function that does the curl_cffi import. Sweep curl_cffi.*
        # before re-importing so a stale entry from another test in
        # the same xdist worker doesn't poison the assertion.
        _strip_native_modules(monkeypatch)
        with _isolated_reimport("cogtrix_core.tools._ddg"):
            assert "curl_cffi" not in sys.modules, (
                "Importing src.tools._ddg must NOT pull curl_cffi into the parent "
                "process — the import belongs strictly inside fetch_ddg_html."
            )
