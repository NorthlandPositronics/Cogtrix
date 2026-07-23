"""Agent complexity test fleet — multi-container Docker-based stress tests.

NOT pytest-collected: filenames deliberately avoid the ``test_*.py``
pattern so pytest auto-discovery ignores this directory.  These are
integration smoke tests that require a Docker daemon, LLM credentials,
and ~10–15 minutes of wall-clock time per run.

See README.md for usage.  Originated as the manual recipe documented
in ``CLAUDE.md`` and codified under #1930.
"""
