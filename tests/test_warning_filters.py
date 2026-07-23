"""Guard that the pytest LangChain warning filter is configured.

The filter mirrors the runtime suppression in ``src/_bootstrap.py``
for an upstream gap in ``langgraph 1.1.3`` that triggers
``LangChainPendingDeprecationWarning`` at import time.  Pytest never
loads ``_bootstrap`` so the filter must live in ``pyproject.toml``'s
``[tool.pytest.ini_options].filterwarnings`` list — this test fails
CI if that entry is removed or renamed.

Remove this test (and both filters) once langgraph passes
``allowed_objects='core'`` to its ``Reviver`` constructor.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_CONFTEST = _REPO_ROOT / "tests" / "conftest.py"


def test_langchain_warning_filter_configured() -> None:
    """``pyproject.toml`` must ignore the langgraph allowed_objects warning."""
    cfg = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    filters = cfg["tool"]["pytest"]["ini_options"].get("filterwarnings", [])
    assert any("allowed_objects" in entry for entry in filters), (
        "pyproject.toml [tool.pytest.ini_options].filterwarnings must include "
        "an ``ignore:The default value of `allowed_objects` will change`` "
        "entry — matching the runtime filter in src/_bootstrap.py."
    )


def test_conftest_imports_bootstrap() -> None:
    """``tests/conftest.py`` must import ``src._bootstrap`` at module level.

    The pyproject.toml ``filterwarnings`` config only catches warnings
    emitted inside test function bodies — warnings raised at module
    import / collection time (which is when langgraph triggers the
    PendingDeprecationWarning) bypass it.  The runtime filter from
    ``src._bootstrap`` is what actually suppresses the warning during
    collection, so conftest.py must force-import it before any other
    ``src.*`` import.
    """
    text = _CONFTEST.read_text(encoding="utf-8")
    assert "import src._bootstrap" in text, (
        "tests/conftest.py must ``import src._bootstrap`` at module "
        "level (before any other src.* import) to suppress the "
        "langgraph LangChainPendingDeprecationWarning at collection "
        "time.  pyproject.toml's filterwarnings only catches warnings "
        "during test execution, not at import time."
    )
