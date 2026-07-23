"""Tests for GitHub Issue #192 — multiple debug verbosity levels.

Covers:
- logging_config verbosity API (get/set/is_verbose/is_trace)
- Config file parsing (verbosity field, validation, legacy debug: true mapping)
- _apply_cli_args handling of --verbosity and --debug flags
- API schemas: SystemInfoOut.verbosity, DebugToggleRequest.verbosity
- toggle_debug endpoint applies set_verbosity()
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# logging_config verbosity API
# ---------------------------------------------------------------------------


def test_set_verbosity_clamps_to_zero():
    from src.logging_config import get_verbosity, set_verbosity

    set_verbosity(-5)
    assert get_verbosity() == 0


def test_set_verbosity_clamps_to_three():
    from src.logging_config import get_verbosity, set_verbosity

    set_verbosity(99)
    assert get_verbosity() == 3


def test_set_verbosity_valid_range():
    from src.logging_config import get_verbosity, set_verbosity

    for level in range(4):
        set_verbosity(level)
        assert get_verbosity() == level


def test_is_verbose_false_below_two():
    from src.logging_config import is_verbose, set_verbosity

    for level in (0, 1):
        set_verbosity(level)
        assert not is_verbose(), f"is_verbose() should be False at level {level}"


def test_is_verbose_true_at_two_and_three():
    from src.logging_config import is_verbose, set_verbosity

    for level in (2, 3):
        set_verbosity(level)
        assert is_verbose(), f"is_verbose() should be True at level {level}"


def test_is_trace_false_below_three():
    from src.logging_config import is_trace, set_verbosity

    for level in (0, 1, 2):
        set_verbosity(level)
        assert not is_trace(), f"is_trace() should be False at level {level}"


def test_is_trace_true_at_three():
    from src.logging_config import is_trace, set_verbosity

    set_verbosity(3)
    assert is_trace()


# ---------------------------------------------------------------------------
# Config file: verbosity field
# ---------------------------------------------------------------------------


def _make_config_from_dict(data: dict):  # type: ignore[return]
    import json
    import tempfile
    from pathlib import Path

    from src.config import Config
    from src.config import _apply_config_file as _acf  # noqa: PLC0415

    # Write a temp JSON config so _apply_config_file can parse it
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
        json.dump(data, fh)
        tmp_path = fh.name

    cfg = Config()
    _acf(cfg, Path(tmp_path))
    return cfg


def test_config_verbosity_zero():
    cfg = _make_config_from_dict({"verbosity": 0})
    assert cfg.verbosity == 0
    assert not cfg.debug
    assert not cfg.verbose


def test_config_verbosity_one_sets_debug():
    cfg = _make_config_from_dict({"verbosity": 1})
    assert cfg.verbosity == 1
    assert cfg.debug


def test_config_verbosity_two_sets_verbose():
    cfg = _make_config_from_dict({"verbosity": 2})
    assert cfg.verbosity == 2
    assert cfg.debug
    assert cfg.verbose


def test_config_verbosity_three():
    cfg = _make_config_from_dict({"verbosity": 3})
    assert cfg.verbosity == 3
    assert cfg.debug
    assert cfg.verbose


def test_config_verbosity_out_of_range_uses_default():
    cfg = _make_config_from_dict({"verbosity": 5})
    # Out-of-range value — should remain 0 (default)
    assert cfg.verbosity == 0


def test_config_legacy_debug_true_sets_verbosity_one():
    """debug: true in config file → verbosity = 1 (legacy mapping)."""
    cfg = _make_config_from_dict({"debug": True})
    assert cfg.debug
    assert cfg.verbosity == 1


def test_config_verbosity_takes_precedence_over_debug_key():
    """Explicit verbosity field overrides legacy debug: true."""
    cfg = _make_config_from_dict({"debug": True, "verbosity": 2})
    assert cfg.verbosity == 2
    assert cfg.verbose


# ---------------------------------------------------------------------------
# _apply_cli_args: --verbosity and --debug flags
# ---------------------------------------------------------------------------


def _args(**kwargs):
    """Build a simple namespace object for _apply_cli_args."""
    ns = types.SimpleNamespace(**kwargs)
    # Ensure all expected attrs are present to avoid AttributeError
    for attr in (
        "verbosity",
        "debug",
        "verbose",
        "log",
        "model",
        "session",
        "memory_mode",
        "data_dir",
        "allow_write_path",
    ):
        if not hasattr(ns, attr):
            setattr(ns, attr, None if attr != "debug" else False)
    return ns


def test_apply_cli_args_verbosity_zero():
    from src.config import Config, _apply_cli_args

    cfg = Config()
    _apply_cli_args(cfg, _args(verbosity=0))
    assert cfg.verbosity == 0
    assert not cfg.debug


def test_apply_cli_args_verbosity_one():
    from src.config import Config, _apply_cli_args

    cfg = Config()
    _apply_cli_args(cfg, _args(verbosity=1))
    assert cfg.verbosity == 1
    assert cfg.debug


def test_apply_cli_args_verbosity_two_enables_verbose():
    from src.config import Config, _apply_cli_args

    cfg = Config()
    _apply_cli_args(cfg, _args(verbosity=2))
    assert cfg.verbosity == 2
    assert cfg.debug
    assert cfg.verbose


def test_apply_cli_args_debug_flag_sets_verbosity_two():
    """--debug flag sets verbosity 2 (full debug + verbose output)."""
    from src.config import Config, _apply_cli_args

    cfg = Config()
    _apply_cli_args(cfg, _args(debug=True, verbosity=None))
    assert cfg.verbosity == 2
    assert cfg.debug
    assert cfg.verbose


def test_apply_cli_args_verbosity_takes_precedence_over_debug_flag():
    """--verbosity N beats --debug when both are provided."""
    from src.config import Config, _apply_cli_args

    cfg = Config()
    _apply_cli_args(cfg, _args(debug=True, verbosity=3))
    assert cfg.verbosity == 3
    assert cfg.debug
    assert cfg.verbose


def test_apply_cli_args_no_verbosity_flag_leaves_config_unchanged():
    from src.config import Config, _apply_cli_args

    cfg = Config()
    cfg.verbosity = 2
    _apply_cli_args(cfg, _args(verbosity=None, debug=False))
    assert cfg.verbosity == 2  # unchanged — no CLI override


def test_apply_cli_args_verbosity_one_autoenables_log():
    """--verbosity 1 turns on file logging (sets log_file to '' if None)."""
    from src.config import Config, _apply_cli_args

    cfg = Config()
    assert cfg.log_file is None
    _apply_cli_args(cfg, _args(verbosity=1))
    assert cfg.log_file is not None  # auto-enabled


# ---------------------------------------------------------------------------
# API schema: SystemInfoOut.verbosity
# ---------------------------------------------------------------------------


def test_system_info_out_has_verbosity_field():
    from datetime import UTC, datetime

    from src.api.schemas.system import SystemInfoOut

    info = SystemInfoOut(
        version="0.1.0",
        api_version="v1",
        platform="Linux",
        python_version="3.12",
        debug=False,
        verbose=False,
        verbosity=0,
        uptime_s=1.0,
        started_at=datetime.now(UTC),
    )
    assert info.verbosity == 0


def test_system_info_out_verbosity_range():
    from datetime import UTC, datetime

    import pydantic

    from src.api.schemas.system import SystemInfoOut

    now = datetime.now(UTC)
    # Valid values
    for v in range(4):
        info = SystemInfoOut(
            version="0.1.0",
            api_version="v1",
            platform="Linux",
            python_version="3.12",
            debug=v >= 1,
            verbose=v >= 2,
            verbosity=v,
            uptime_s=0.0,
            started_at=now,
        )
        assert info.verbosity == v

    # Out of range should fail validation
    with pytest.raises(pydantic.ValidationError):
        SystemInfoOut(
            version="0.1.0",
            api_version="v1",
            platform="Linux",
            python_version="3.12",
            debug=False,
            verbose=False,
            verbosity=4,
            uptime_s=0.0,
            started_at=now,
        )


# ---------------------------------------------------------------------------
# API schema: DebugToggleRequest.verbosity
# ---------------------------------------------------------------------------


def test_debug_toggle_request_verbosity_field():
    from src.api.schemas.system import DebugToggleRequest

    req = DebugToggleRequest(verbosity=2)
    assert req.verbosity == 2
    assert req.debug is None
    assert req.verbose is None


def test_debug_toggle_request_verbosity_none_by_default():
    from src.api.schemas.system import DebugToggleRequest

    req = DebugToggleRequest()
    assert req.verbosity is None


def test_debug_toggle_request_verbosity_out_of_range_rejected():
    import pydantic

    from src.api.schemas.system import DebugToggleRequest

    with pytest.raises(pydantic.ValidationError):
        DebugToggleRequest(verbosity=5)


# ---------------------------------------------------------------------------
# toggle_debug route: calls set_verbosity()
# ---------------------------------------------------------------------------


def _make_mock_request(verbosity: int = 0, debug: bool = False) -> MagicMock:
    cfg = MagicMock()
    cfg.debug = debug
    cfg.verbose = False
    cfg.verbosity = verbosity
    req = MagicMock()
    req.app.state.config = cfg
    return req


def test_toggle_debug_verbosity_field_takes_effect():
    """POST /system/debug with verbosity=2 calls set_verbosity(2)."""
    import asyncio

    from src.api.routes.system import toggle_debug
    from src.api.schemas.system import DebugToggleRequest

    request = _make_mock_request()
    body = DebugToggleRequest(verbosity=2)

    with (
        patch("src.api.routes.system.set_verbosity") as mock_sv,
        patch("src.api.routes.system._make_system_info") as mock_si,
    ):
        mock_si.return_value = MagicMock()
        asyncio.run(toggle_debug(request=request, current_user=MagicMock(), body=body))
        mock_sv.assert_called_once_with(2)


def test_toggle_debug_legacy_toggle_calls_set_verbosity():
    """POST /system/debug without verbosity — toggles debug flag and calls set_verbosity."""
    import asyncio

    from src.api.routes.system import toggle_debug
    from src.api.schemas.system import DebugToggleRequest

    request = _make_mock_request(debug=False)
    body = DebugToggleRequest()  # no verbosity, no debug

    with (
        patch("src.api.routes.system.set_verbosity") as mock_sv,
        patch("src.api.routes.system._make_system_info") as mock_si,
    ):
        mock_si.return_value = MagicMock()
        asyncio.run(toggle_debug(request=request, current_user=MagicMock(), body=body))
        # debug was False → toggled to True → verbosity=1
        mock_sv.assert_called_once_with(1)


def test_toggle_debug_verbosity_zero_turns_off():
    """POST /system/debug with verbosity=0 disables debug and calls set_verbosity(0)."""
    import asyncio

    from src.api.routes.system import toggle_debug
    from src.api.schemas.system import DebugToggleRequest

    request = _make_mock_request(debug=True, verbosity=1)
    body = DebugToggleRequest(verbosity=0)

    with (
        patch("src.api.routes.system.set_verbosity") as mock_sv,
        patch("src.api.routes.system._make_system_info") as mock_si,
    ):
        mock_si.return_value = MagicMock()
        asyncio.run(toggle_debug(request=request, current_user=MagicMock(), body=body))
        mock_sv.assert_called_once_with(0)
