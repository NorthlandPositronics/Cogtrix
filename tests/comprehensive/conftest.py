"""Setup for comprehensive-test harnesses.

Any test collected under ``tests/comprehensive/`` gets the dedicated config +
secrets loaded before it runs — and the process environment is **restored
afterwards** so the loaded secrets/`COGTRIX_CONFIG_FILE` can't leak into other
test files sharing the session (which previously flipped provider-default
assertions elsewhere in the suite).
"""

from __future__ import annotations

import os

import pytest

from tests.comprehensive.env_loader import load_comprehensive_env


@pytest.fixture(autouse=True)
def _comprehensive_env():
    """Load ``.env`` + pin ``COGTRIX_CONFIG_FILE`` per test, then restore env.

    Function-scoped with snapshot/restore so the real keys exist only while a
    comprehensive test runs and never persist into unrelated tests in the same
    pytest session (os.environ is process-global).
    """
    saved = dict(os.environ)
    load_comprehensive_env(override=True)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)
