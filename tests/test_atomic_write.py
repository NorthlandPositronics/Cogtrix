"""Tests for src.utils.atomic_write.atomic_write_json.

Covers:
- Happy path: destination file contains correct data, no temp file remains.
- Parent directory creation.
- os.fdopen raising: temp file cleaned up, raw fd closed.
- KeyboardInterrupt inside yield: temp file cleaned up.
- Exception inside yield: temp file cleaned up.
- Overwrite of an existing destination file.
- Non-default encoding.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from cogtrix_core.utils.atomic_write import atomic_write_json


class TestAtomicWriteJson:
    def test_happy_path(self, tmp_path: Path) -> None:
        """Destination file contains correct JSON; no temp file remains."""
        dest = tmp_path / "out.json"
        data = {"key": "value", "n": 42}
        with atomic_write_json(dest) as f:
            json.dump(data, f)
        assert dest.exists()
        assert json.loads(dest.read_text()) == data
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Parent directories are created if they don't exist."""
        dest = tmp_path / "sub" / "dir" / "out.json"
        with atomic_write_json(dest) as f:
            json.dump({}, f)
        assert dest.exists()

    def test_fdopen_raises_cleans_up_temp(self, tmp_path: Path) -> None:
        """If os.fdopen raises, the temp file is removed and raw fd is closed."""
        dest = tmp_path / "out.json"
        captured_fd: list[int] = []

        def fake_fdopen(fd: int, *args, **kwargs):  # type: ignore[no-untyped-def]
            captured_fd.append(fd)
            raise OSError("fdopen failed")

        with patch("os.fdopen", side_effect=fake_fdopen):
            with pytest.raises(OSError, match="fdopen failed"):
                with atomic_write_json(dest) as f:
                    json.dump({}, f)  # pragma: no cover — never reached

        # Destination must not exist
        assert not dest.exists()
        # No temp file should remain
        assert list(tmp_path.glob("*.tmp")) == []
        # The raw fd must have been closed by the cleanup path
        if captured_fd:
            with pytest.raises(OSError):
                os.close(captured_fd[0])

    def test_keyboard_interrupt_inside_yield_cleans_up(self, tmp_path: Path) -> None:
        """KeyboardInterrupt raised inside the yield block causes temp cleanup."""
        dest = tmp_path / "out.json"
        with pytest.raises(KeyboardInterrupt):
            with atomic_write_json(dest) as f:
                f.write("{}")
                raise KeyboardInterrupt

        assert not dest.exists()
        assert list(tmp_path.glob("*.tmp")) == []

    def test_exception_inside_yield_cleans_up(self, tmp_path: Path) -> None:
        """Regular exception inside the yield block causes temp cleanup."""
        dest = tmp_path / "out.json"
        with pytest.raises(ValueError, match="bad data"):
            with atomic_write_json(dest) as f:
                f.write("{}")
                raise ValueError("bad data")

        assert not dest.exists()
        assert list(tmp_path.glob("*.tmp")) == []

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        """Atomic replace works even when the destination already exists."""
        dest = tmp_path / "out.json"
        dest.write_text('{"old": true}')
        with atomic_write_json(dest) as f:
            json.dump({"new": True}, f)
        assert json.loads(dest.read_text()) == {"new": True}

    def test_encoding_parameter(self, tmp_path: Path) -> None:
        """Non-default encoding is respected."""
        dest = tmp_path / "out.txt"
        with atomic_write_json(dest, encoding="latin-1") as f:
            f.write("caf\xe9")  # 'café' in latin-1
        assert dest.read_bytes() == b"caf\xe9"

    def test_cleanup_catches_valueerror_on_double_close(self, tmp_path: Path) -> None:
        """BUG-097: The cleanup except block must catch ValueError as well as OSError so that
        a double-close (e.g. from a custom file object) does not produce a second exception
        that masks the original error.

        We simulate this by patching f.close() inside the except block to raise ValueError,
        verifying the original exception is still re-raised cleanly.
        """
        import json as _json
        from unittest.mock import patch

        dest = tmp_path / "out.json"
        original_error = RuntimeError("write failed")

        # We need f to be a real file-like so the context manager opens it,
        # but we intercept f.close() in the cleanup path to raise ValueError.
        real_f_holder: list = []
        original_fdopen = os.fdopen

        def patched_fdopen(fd: int, *args, **kwargs):  # type: ignore[no-untyped-def]
            f = original_fdopen(fd, *args, **kwargs)
            real_f_holder.append(f)
            return f

        with patch("os.fdopen", side_effect=patched_fdopen):
            with pytest.raises(RuntimeError, match="write failed"):
                with atomic_write_json(dest) as f:
                    _json.dump({"key": "value"}, f)
                    # Make f.close() raise ValueError to simulate the double-close edge case.
                    original_close = f.close
                    call_count = [0]

                    def close_that_raises():  # type: ignore[no-untyped-def]
                        call_count[0] += 1
                        if call_count[0] >= 2:
                            raise ValueError("I/O operation on closed file")
                        original_close()

                    f.close = close_that_raises  # type: ignore[method-assign]
                    raise original_error  # trigger the except BaseException branch

        # The destination must not exist (write was not completed atomically).
        assert not dest.exists()
        # No temp file should remain.
        assert list(tmp_path.glob("*.tmp")) == []
