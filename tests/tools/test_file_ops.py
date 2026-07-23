"""Tests for file_ops: TOCTOU-safe error handling and core read/write behaviour."""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.tools.file_ops import append_file, list_directory, read_file, write_file


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
