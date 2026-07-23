"""Tests for file_ops: TOCTOU-safe error handling and core read/write behaviour."""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.tools.file_ops import (
    _APPEND_LOCK_MAX,
    _append_lock_guard,
    _append_locks,
    _get_append_lock,
    _RefLock,
    append_file,
    list_directory,
    patch_file,
    read_file,
    set_allowed_write_dirs,
    write_file,
)


@pytest.fixture()
def tmp_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Change cwd to a temp directory so _validate_path allows paths inside it."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


class TestReadFileTOCTOU:
    """Verify read_file handles missing/directory targets via exceptions, not pre-checks."""

    def test_file_not_found_returns_error(self, tmp_cwd: Path) -> None:
        result = read_file("nonexistent.txt")
        assert result.startswith("Error: File not found:")

    def test_directory_path_returns_not_a_file_error(self, tmp_cwd: Path) -> None:
        subdir = tmp_cwd / "adir"
        subdir.mkdir()
        result = read_file("adir")
        assert result.startswith("Error: Not a file:")

    def test_file_deleted_between_stat_and_open(self, tmp_cwd: Path) -> None:
        """Simulate a race: file disappears after stat() succeeds."""
        target = tmp_cwd / "race.txt"
        target.write_text("hello")

        original_open = open

        def vanishing_open(path, *args, **kwargs):  # type: ignore[no-untyped-def]
            target.unlink(missing_ok=True)
            return original_open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=vanishing_open):
            result = read_file("race.txt")

        assert result.startswith("Error: File not found:")

    def test_reads_existing_file(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "hello.txt").write_text("world")
        result = read_file("hello.txt")
        assert result == "world"

    def test_start_line_and_max_lines(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "multi.txt").write_text("a\nb\nc\nd\n")
        result = read_file("multi.txt", start_line=1, max_lines=2)
        assert "b\n" in result
        assert "c\n" in result

    def test_path_outside_cwd_rejected(self, tmp_cwd: Path) -> None:
        result = read_file("/etc/passwd")
        assert result.startswith("Error:")

    def test_unicode_decode_error(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "binary.bin").write_bytes(b"\xff\xfe")
        result = read_file("binary.bin", encoding="ascii")
        assert "decode" in result.lower() or "encoding" in result.lower()

    def test_large_file_rejected(self, tmp_cwd: Path) -> None:
        large = tmp_cwd / "big.txt"
        large.write_bytes(b"x" * (101 * 1024 * 1024))
        result = read_file("big.txt")
        assert "too large" in result.lower()


class TestWriteFile:
    """Basic write_file correctness — no TOCTOU pre-checks to verify here."""

    def test_creates_file(self, tmp_cwd: Path) -> None:
        result = write_file("new.txt", "content")
        assert "Successfully wrote" in result
        assert (tmp_cwd / "new.txt").read_text() == "content"

    def test_overwrites_existing(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "existing.txt").write_text("old")
        write_file("existing.txt", "new")
        assert (tmp_cwd / "existing.txt").read_text() == "new"

    def test_creates_parent_dirs(self, tmp_cwd: Path) -> None:
        result = write_file("deep/nested/file.txt", "hi")
        assert "Successfully wrote" in result
        assert (tmp_cwd / "deep" / "nested" / "file.txt").read_text() == "hi"

    def test_path_outside_cwd_rejected(self, tmp_cwd: Path) -> None:
        result = write_file("/tmp/cogtrix_test_escape.txt", "data")
        assert result.startswith("Error:")

    def test_path_as_dict_with_path_key(self, tmp_cwd: Path) -> None:
        """write_file should unwrap dict paths (e.g., {"path": "target.txt"})."""
        result = write_file({"path": "target.txt"}, "data")
        assert "Successfully wrote" in result
        assert (tmp_cwd / "target.txt").read_text() == "data"

    def test_path_as_dict_with_absolute_path_key(self, tmp_cwd: Path) -> None:
        result = write_file({"absolute_path": str(tmp_cwd / "abs.txt")}, "data")
        assert "Successfully wrote" in result
        assert (tmp_cwd / "abs.txt").read_text() == "data"

    def test_path_as_dict_with_file_path_key(self, tmp_cwd: Path) -> None:
        result = write_file({"file_path": "fp.txt"}, "data")
        assert "Successfully wrote" in result
        assert (tmp_cwd / "fp.txt").read_text() == "data"

    def test_path_as_dict_unknown_key_returns_error(self, tmp_cwd: Path) -> None:
        result = write_file({"unknown_key": "foo.txt"}, "data")
        assert result == "Error: Invalid arguments for write_file"


class TestAppendFile:
    """append_file correctness tests."""

    def test_appends_to_existing(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "log.txt").write_text("line1\n")
        append_file("log.txt", "line2\n")
        assert (tmp_cwd / "log.txt").read_text() == "line1\nline2\n"

    def test_creates_new_file(self, tmp_cwd: Path) -> None:
        result = append_file("fresh.txt", "data")
        assert "Successfully appended" in result
        assert (tmp_cwd / "fresh.txt").read_text() == "data"

    def test_path_as_dict(self, tmp_cwd: Path) -> None:
        """append_file should unwrap dict paths."""
        result = append_file({"path": "append_dict.txt"}, "hello")
        assert "Successfully appended" in result
        assert (tmp_cwd / "append_dict.txt").read_text() == "hello"


class TestPatchFile:
    """patch_file correctness tests."""

    def test_replaces_string(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "patchme.txt").write_text("hello world")
        result = patch_file("patchme.txt", "world", "cogtrix")
        assert "Patched patchme.txt" in result
        assert (tmp_cwd / "patchme.txt").read_text() == "hello cogtrix"

    def test_path_as_dict(self, tmp_cwd: Path) -> None:
        """patch_file should unwrap dict paths."""
        (tmp_cwd / "dict_patch.txt").write_text("before")
        result = patch_file({"path": "dict_patch.txt"}, "before", "after")
        assert "Patched dict_patch.txt" in result
        assert (tmp_cwd / "dict_patch.txt").read_text() == "after"

    def test_large_file_rejected(self, tmp_cwd: Path) -> None:
        large = tmp_cwd / "big.txt"
        large.write_bytes(b"x" * (101 * 1024 * 1024))
        result = patch_file("big.txt", "old", "new")
        assert "too large" in result.lower()


class TestListDirectory:
    """list_directory correctness — pre-checks on is_dir() are intentional, not TOCTOU."""

    def test_lists_files(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "a.txt").write_text("a")
        (tmp_cwd / "b.txt").write_text("b")
        result = list_directory(".")
        assert "a.txt" in result
        assert "b.txt" in result

    def test_missing_directory_returns_error(self, tmp_cwd: Path) -> None:
        result = list_directory("no_such_dir")
        assert result.startswith("Error: Directory not found:")

    def test_file_path_returns_not_a_directory(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "afile.txt").write_text("x")
        result = list_directory("afile.txt")
        assert result.startswith("Error: Not a directory:")

    def test_glob_pattern_filters(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "main.py").write_text("")
        (tmp_cwd / "notes.txt").write_text("")
        result = list_directory(".", pattern="*.py")
        assert "main.py" in result
        assert "notes.txt" not in result


class TestAllowedWritePaths:
    """Tests for --allow-write-path / set_allowed_write_dirs() feature."""

    @pytest.fixture(autouse=True)
    def _cleanup_extra_dirs(self) -> None:
        """Reset extra write dirs after each test."""
        yield
        set_allowed_write_dirs(None)

    def test_write_to_extra_dir_succeeds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Writing to an allowed extra dir should succeed."""
        work_dir = tmp_path / "work"
        extra_dir = tmp_path / "extra"
        work_dir.mkdir()
        extra_dir.mkdir()
        monkeypatch.chdir(work_dir)
        set_allowed_write_dirs([str(extra_dir)])
        result = write_file(str(extra_dir / "out.txt"), "hello")
        assert "Successfully wrote" in result
        assert (extra_dir / "out.txt").read_text() == "hello"

    def test_write_outside_all_dirs_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Writing outside cwd and extra dirs should be rejected."""
        work_dir = tmp_path / "work"
        extra_dir = tmp_path / "extra"
        forbidden_dir = tmp_path / "forbidden"
        work_dir.mkdir()
        extra_dir.mkdir()
        forbidden_dir.mkdir()
        monkeypatch.chdir(work_dir)
        set_allowed_write_dirs([str(extra_dir)])
        result = write_file(str(forbidden_dir / "evil.txt"), "bad")
        assert result.startswith("Error:")

    def test_read_from_extra_write_dir_succeeds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reading from an allowed extra write dir should work too."""
        work_dir = tmp_path / "work"
        extra_dir = tmp_path / "extra"
        work_dir.mkdir()
        extra_dir.mkdir()
        (extra_dir / "data.txt").write_text("content")
        monkeypatch.chdir(work_dir)
        set_allowed_write_dirs([str(extra_dir)])
        result = read_file(str(extra_dir / "data.txt"))
        assert result == "content"

    def test_set_allowed_write_dirs_none_clears(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Passing None should clear all extra dirs."""
        work_dir = tmp_path / "work"
        extra_dir = tmp_path / "extra"
        work_dir.mkdir()
        extra_dir.mkdir()
        monkeypatch.chdir(work_dir)
        set_allowed_write_dirs([str(extra_dir)])
        set_allowed_write_dirs(None)
        result = write_file(str(extra_dir / "file.txt"), "data")
        assert result.startswith("Error:")

    def test_set_allowed_write_dirs_empty_list_clears(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Passing an empty list should clear all extra dirs."""
        work_dir = tmp_path / "work"
        extra_dir = tmp_path / "extra"
        work_dir.mkdir()
        extra_dir.mkdir()
        monkeypatch.chdir(work_dir)
        set_allowed_write_dirs([str(extra_dir)])
        set_allowed_write_dirs([])
        result = write_file(str(extra_dir / "file.txt"), "data")
        assert result.startswith("Error:")

    def test_traversal_in_extra_dir_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Path traversal out of extra dir should be rejected."""
        work_dir = tmp_path / "work"
        extra_dir = tmp_path / "extra"
        work_dir.mkdir()
        extra_dir.mkdir()
        monkeypatch.chdir(work_dir)
        set_allowed_write_dirs([str(extra_dir)])
        result = write_file(str(extra_dir / ".." / "escape.txt"), "bad")
        assert result.startswith("Error:")


class TestGetAppendLock:
    """_get_append_lock() must evict across the full LRU, not just the first 32 entries."""

    def _fill_cache(self, n: int) -> list[str]:
        """Fill the cache with n free (ref_count=0) entries, returning their keys."""
        keys = [f"/tmp/fake_path_{i}.txt" for i in range(n)]
        with _append_lock_guard:
            for k in keys:
                _append_locks[k] = _RefLock()
        return keys

    def setup_method(self):
        """Clear the global cache before each test."""
        with _append_lock_guard:
            _append_locks.clear()

    def teardown_method(self):
        """Clear the global cache after each test."""
        with _append_lock_guard:
            _append_locks.clear()

    def test_evicts_entry_beyond_32_positions(self):
        """Cache must evict an entry that sits beyond position 32 in the LRU."""
        # Fill cache to the cap with free locks
        self._fill_cache(_APPEND_LOCK_MAX)
        assert len(_append_locks) == _APPEND_LOCK_MAX

        # Request a new lock — cache is at cap so eviction must occur
        new_key = "/tmp/new_entry.txt"
        lock = _get_append_lock(new_key)

        # The new entry must be present and the cache must not exceed cap+1
        with _append_lock_guard:
            assert new_key in _append_locks
            assert len(_append_locks) <= _APPEND_LOCK_MAX

        with lock:
            pass  # ensure lock is usable

    def test_busy_locks_not_evicted(self):
        """Locks with ref_count > 0 must not be evicted even when cache is full."""
        self._fill_cache(_APPEND_LOCK_MAX)

        # Mark the first 40 entries as busy
        busy_keys = list(_append_locks.keys())[:40]
        with _append_lock_guard:
            for k in busy_keys:
                _append_locks[k].ref_count = 1

        # Request a new lock — must still succeed (evict a non-busy entry)
        new_key = "/tmp/after_busy.txt"
        _get_append_lock(new_key)

        with _append_lock_guard:
            assert new_key in _append_locks
            # All busy entries must still be present
            for k in busy_keys:
                assert k in _append_locks

        # Clean up ref counts so teardown can clear
        with _append_lock_guard:
            for k in busy_keys:
                _append_locks[k].ref_count = 0


class TestPatchFileLocking:
    """patch_file additional correctness paths and per-file lock behaviour.

    Renamed from a second ``TestPatchFile`` (collided with the class at
    line 149 — the second definition shadowed the first, so the basic
    correctness tests never ran).  Split intentionally: this class
    covers locking and concurrent-patch semantics; basic correctness
    stays in TestPatchFile above.
    """

    def test_patches_existing_file(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "target.txt").write_text("hello old world")
        result = patch_file("target.txt", "old", "new")
        assert result.startswith("Patched")
        assert (tmp_cwd / "target.txt").read_text() == "hello new world"

    def test_old_str_not_found_returns_error(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "target.txt").write_text("hello world")
        result = patch_file("target.txt", "missing", "new")
        assert result.startswith("Error: old_str not found")

    def test_ambiguous_old_str_returns_error(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "target.txt").write_text("old old old")
        result = patch_file("target.txt", "old", "new")
        assert "found 3 times" in result

    def test_uses_file_lock(self, tmp_cwd: Path) -> None:
        """patch_file must acquire the per-file lock to prevent lost updates."""
        (tmp_cwd / "target.txt").write_text("hello old world")

        with patch("src.tools.file_ops._get_append_lock") as mock_get_lock:
            mock_lock = mock_get_lock.return_value
            mock_lock.__enter__ = lambda self: self  # type: ignore[no-untyped-def]
            mock_lock.__exit__ = lambda *args: None  # type: ignore[no-untyped-def]

            patch_file("target.txt", "old", "new")

            mock_get_lock.assert_called_once_with(str(tmp_cwd / "target.txt"))

    def test_concurrent_patches_are_serialized(self, tmp_cwd: Path) -> None:
        """Two concurrent patches to the same file must not lose updates."""
        import threading

        (tmp_cwd / "target.txt").write_text("hello old world")
        results: list[str] = []
        barrier = threading.Barrier(2)

        def patch_a() -> None:
            barrier.wait()
            results.append(patch_file("target.txt", "old", "NEW_A"))

        def patch_b() -> None:
            barrier.wait()
            results.append(patch_file("target.txt", "old", "NEW_B"))

        t1 = threading.Thread(target=patch_a)
        t2 = threading.Thread(target=patch_b)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # With proper locking, exactly one succeeds and one errors
        successes = [r for r in results if r.startswith("Patched")]
        errors = [r for r in results if r.startswith("Error")]
        assert len(successes) == 1, f"Expected 1 success, got: {results}"
        assert len(errors) == 1, f"Expected 1 error, got: {results}"

        # The file must contain exactly one replacement
        content = (tmp_cwd / "target.txt").read_text()
        assert "old" not in content
        assert ("NEW_A" in content) != ("NEW_B" in content)
