"""Cogtrix plugin hook specifications.

Third-party packages that provide Cogtrix tools should:

1. Import ``hookimpl`` from this module (or from ``cogtrix.plugins``).
2. Decorate a ``cogtrix_tools`` method on a plugin class with ``@hookimpl``.
3. Declare an entry-point in the ``cogtrix.tools`` group pointing to that
   class::

       [project.entry-points."cogtrix.tools"]
       my_plugin = "my_package.tools:MyPlugin"

The hook must return a list of tool-config dicts.  Each dict must contain:

- ``name`` (str): unique tool name — lowercase, underscores
- ``description`` (str): one-sentence description shown in ``/tools``
- ``input_schema``: a Pydantic ``BaseModel`` subclass
- ``function``: the callable that implements the tool

Optional key:

- ``requires_confirmation`` (bool, default ``False``)

See ``docs/TOOLS_AUTHORING.md`` for the complete authoring guide.
"""

from __future__ import annotations

try:
    import pluggy as _pluggy

    hookspec = _pluggy.HookspecMarker("cogtrix")
    hookimpl = _pluggy.HookimplMarker("cogtrix")

    class CogtrixSpec:
        """Cogtrix plugin hook specifications."""

        @hookspec
        def cogtrix_tools(self) -> list[dict]:  # type: ignore[empty-body]
            """Return a list of TOOL_CONFIG dicts to register.

            Each dict must contain:
            - ``name`` (str): unique tool name
            - ``description`` (str): one-sentence summary shown in /tools
            - ``input_schema``: Pydantic BaseModel subclass
            - ``function``: the callable that implements the tool

            Optional:
            - ``requires_confirmation`` (bool, default False)
            """

except ImportError:  # pragma: no cover
    # pluggy is a declared dependency — this branch is unreachable in normal use
    hookspec = None  # type: ignore[assignment]
    hookimpl = None  # type: ignore[assignment]
    CogtrixSpec = None  # type: ignore[assignment]

__all__ = ["hookspec", "hookimpl", "CogtrixSpec"]
