"""Process-wide warning filters, installed before any langchain/langgraph import.

Import this module first from every Cogtrix entry point (CLI in
``cogtrix.py``; FastAPI app in ``src.api.app``).  Importing has the
side-effect of installing the filters; the module exports nothing.

Why this lives in its own module: ``langchain_core/__init__.py`` runs
``surface_langchain_deprecation_warnings()`` on import, which prepends a
``default`` filter for ``LangChainPendingDeprecationWarning``.  To make
our narrower ``ignore`` filter win, it must be prepended *after*
``langchain_core`` has loaded.  Co-locating the ``import langchain_core``
and the ``warnings.filterwarnings`` call in a single bootstrap module
keeps that ordering invariant explicit, and avoids littering entry points
with mid-import ``filterwarnings`` statements (which trip ruff's E402).
"""

from __future__ import annotations

import warnings

import langchain_core  # noqa: F401 — load to install its own deprecation filter

# Upstream gap: langgraph 1.1.3 imports JsonPlusSerializer, which constructs a
# langchain_core 1.3.3 Reviver() without an explicit ``allowed_objects=``,
# triggering LangChainPendingDeprecationWarning at import time.  Remove this
# block once langgraph passes ``allowed_objects='core'``.
warnings.filterwarnings(
    "ignore",
    message=r"The default value of `allowed_objects` will change",
)
