"""Tests for _expand_at_references bounded directory traversal.

Regression tests for issue #911:
- Bounded traversal prevents OOM on large directories
- Depth limiting prevents deep recursive descent
- Symlink loop protection prevents infinite loops
"""

from __future__ import annotations

import os
from pathlib import Path

from cogtrix import _expand_at_references


class TestBoundedDirTraversal:
    """Test that _expand_at_references uses bounded directory traversal."""

    def test_small_directory_lists_correctly(self, tmp_path: Path) -> None:
        """Small directories should list all files up to the limit."""
        # Create 5 files in a temp directory
        (tmp_path / "file1.txt").write_text("content1")
        (tmp_path / "file2.txt").write_text("content2")
        (tmp_path / "file3.txt").write_text("content3")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "file4.txt").write_text("content4")

        text, injected = _expand_at_references(f"@{tmp_path}")
        assert str(tmp_path) in injected
        assert "file1.txt" in text
        assert "file2.txt" in text
        assert "file3.txt" in text
        assert "subdir/file4.txt" in text  # Subdirectory contents should be listed

    def test_large_directory_truncates_at_60(self, tmp_path: Path) -> None:
        """Directories with >60 files should be truncated."""
        # Create 65 files
        for i in range(65):
            (tmp_path / f"file{i:02d}.txt").write_text(f"content{i}")

        text, injected = _expand_at_references(f"@{tmp_path}")
        assert str(tmp_path) in injected
        # Should have truncated marker
        assert "... (truncated)" in text
        # Should have at most 60 entries (plus header)
        file_lines = [line for line in text.split("\n") if line.startswith("  file")]
        assert len(file_lines) <= 60

    def test_depth_limited_to_3_levels(self, tmp_path: Path) -> None:
        """Directory traversal should be limited to 3 levels."""
        # Create 4 levels of nested directories
        level1 = tmp_path / "level1"
        level2 = level1 / "level2"
        level3 = level2 / "level3"
        level4 = level3 / "level4"
        level4.mkdir(parents=True)

        (level1 / "file1.txt").write_text("level1")
        (level2 / "file2.txt").write_text("level2")
        (level3 / "file3.txt").write_text("level3")
        (level4 / "file4.txt").write_text("level4")

        text, _ = _expand_at_references(f"@{tmp_path}")
        # Only files up to level3 should appear (depth 3)
        assert "file1.txt" in text
        assert "file2.txt" in text
        assert "file3.txt" in text
        # file4 should NOT appear (depth 4, beyond limit)
        assert "file4.txt" not in text

    def test_symlink_loop_prevented(self, tmp_path: Path) -> None:
        """Symlink loops should not cause infinite traversal."""
        # Create a directory with a symlink back to parent
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "file.txt").write_text("content")

        # Create symlink back to tmp_path
        link = tmp_path / "link"
        link.symlink_to(tmp_path)

        # Should not hang or recurse infinitely
        text, _ = _expand_at_references(f"@{tmp_path}")
        # Should complete successfully
        assert "file.txt" in text
        # Symlinks to directories should not be listed as file entries
        # Check that there's no line starting with "  link" (symlink to directory)
        assert not any(
            line.strip().startswith("link") for line in text.split("\n") if line.startswith("  ")
        )

    def test_recursive_symlink_prevented(self, tmp_path: Path) -> None:
        """Recursive symlinks (A->B, B->A) should not cause infinite loop."""
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        (dir1 / "file1.txt").write_text("content1")
        (dir2 / "file2.txt").write_text("content2")

        # Create recursive symlinks
        (dir1 / "link_to_dir2").symlink_to(dir2)
        (dir2 / "link_to_dir1").symlink_to(dir1)

        # Should complete without hanging
        text, _ = _expand_at_references(f"@{tmp_path}")
        assert "file1.txt" in text
        assert "file2.txt" in text

    def test_hidden_files_excluded_by_default(self, tmp_path: Path) -> None:
        """Hidden files (dotfiles) should be excluded from listing."""
        (tmp_path / ".hidden").write_text("hidden")
        (tmp_path / "visible.txt").write_text("visible")

        text, _ = _expand_at_references(f"@{tmp_path}")
        assert "visible.txt" in text
        # Hidden files should be excluded
        assert ".hidden" not in text

    def test_at_reference_with_relative_path(self, tmp_path: Path) -> None:
        """Relative @path should be resolved relative to cwd."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "file.txt").write_text("content")

        # Change to subdir and use relative path
        old_cwd = os.getcwd()
        try:
            os.chdir(subdir)
            text, injected = _expand_at_references("@.")
            assert str(subdir) in injected
            assert "file.txt" in text
        finally:
            os.chdir(old_cwd)


class TestAtFileLimit:
    """Test the _AT_MAX_FILES limit for total @ references."""

    def test_max_files_limit_respected(self, tmp_path: Path) -> None:
        """Only _AT_MAX_FILES (5) @ references should be expanded."""
        # Create 7 temp directories
        for i in range(7):
            d = tmp_path / f"dir{i}"
            d.mkdir()
            (d / "file.txt").write_text(f"content{i}")

        # Use 7 @ references in one input
        text, injected = _expand_at_references(
            " ".join(
                f"@{tmp_path / dir_name}"
                for dir_name in ["dir0", "dir1", "dir2", "dir3", "dir4", "dir5", "dir6"]
            )
        )
        # Only 5 should be expanded
        assert len(injected) <= 5
        # The rest should remain as @ references
        assert "@dir5" in text or "dir5" not in injected
        assert "@dir6" in text or "dir6" not in injected


class TestEdgeCases:
    """Test edge cases for _expand_at_references."""

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Empty directories should produce just header."""
        text, _ = _expand_at_references(f"@{tmp_path}")
        assert f"[Directory: {tmp_path}]" in text
        assert "... (truncated)" not in text

    def test_nonexistent_path(self) -> None:
        """Nonexistent paths should be left as-is."""
        text, _ = _expand_at_references("@/nonexistent/path")
        assert "@/nonexistent/path" in text

    def test_file_reference(self, tmp_path: Path) -> None:
        """Single file references should be expanded."""
        f = tmp_path / "test.txt"
        f.write_text("hello world")

        text, injected = _expand_at_references(f"@{f}")
        assert str(f) in injected
        assert "hello world" in text

    def test_symlink_to_file(self, tmp_path: Path) -> None:
        """Symlinks to files should be listed with their metadata."""
        f = tmp_path / "original.txt"
        f.write_text("original content")
        link = tmp_path / "link.txt"
        link.symlink_to(f)

        text, _ = _expand_at_references(f"@{tmp_path}")
        # Symlinks to files should be listed with metadata
        assert "link.txt" in text
        assert "original.txt" in text
        # File content is NOT included for directory listings (only metadata)
        assert "original content" not in text
