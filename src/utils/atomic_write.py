"""Atomic file write utilities."""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import IO


@contextlib.contextmanager
def atomic_write_json(path: Path, encoding: str = "utf-8") -> Generator[IO[str]]:
    """Context manager for atomic text/JSON writes using tempfile.mkstemp.

    Yields an open text file handle. On successful exit, the temp file is
    renamed to *path* atomically. On any exception (including KeyboardInterrupt
    and SystemExit), the temp file is cleaned up and the exception re-raised.

    Ownership of the raw fd is transferred to the file object as soon as
    os.fdopen succeeds, so we track whether that transfer happened to avoid
    double-close.

    Args:
        path: Destination path. Parent directories are created if absent.
        encoding: Text encoding for the file object (default: ``"utf-8"``).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    tmp = Path(tmp_path)
    f: IO[str] | None = None
    try:
        f = os.fdopen(tmp_fd, "w", encoding=encoding)
        yield f
        f.close()
        f = None
        tmp.replace(path)
    except BaseException:
        if f is not None:
            try:
                f.close()
            except OSError:
                pass
        else:
            # os.fdopen never succeeded — fd is still raw
            try:
                os.close(tmp_fd)
            except OSError:
                pass
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
