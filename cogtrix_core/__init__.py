"""Cogtrix — the application's Python package root.

Renamed from the former ``src/`` layout to a single ``cogtrix_core`` package
(#2465) so the built wheel exposes one clean top-level import namespace instead
of ~27 bare packages (agent, tools, auth, config, …). The name intentionally
avoids ``cogtrix`` to sidestep a collision with the root ``cogtrix.py`` CLI
entry, preserving the ``python cogtrix.py`` invocation.
"""
