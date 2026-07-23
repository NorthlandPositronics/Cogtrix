"""Tests for BUG-049 (_copy_with_content Pydantic v1/v2 compat) and
BUG-050 (RAG ingest subdirectory recursion)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# BUG-049: _copy_with_content
# ---------------------------------------------------------------------------


class TestCopyWithContent:
    """_copy_with_content must work with model_copy, copy, and plain objects."""

    def _fn(self):
        from cogtrix_core.agent.core import _copy_with_content

        return _copy_with_content

    def test_uses_model_copy_when_available(self):
        """Prefers model_copy (Pydantic v2) when present."""
        copy_with_content = self._fn()

        expected = object()
        msg = MagicMock(spec=["model_copy"])
        msg.model_copy.return_value = expected

        result = copy_with_content(msg, "new content")

        msg.model_copy.assert_called_once_with(update={"content": "new content"})
        assert result is expected

    def test_falls_back_to_copy_when_no_model_copy(self):
        """Falls back to copy (Pydantic v1) when model_copy is absent."""
        copy_with_content = self._fn()

        expected = object()
        msg = MagicMock(spec=["copy"])
        msg.copy.return_value = expected

        result = copy_with_content(msg, "hello")

        msg.copy.assert_called_once_with(update={"content": "hello"})
        assert result is expected

    def test_falls_back_to_shallow_copy_for_plain_objects(self):
        """Falls back to shallow copy + attribute set for plain objects."""
        copy_with_content = self._fn()

        class PlainMsg:
            def __init__(self, content: str) -> None:
                self.content = content

        original = PlainMsg("old content")
        result = copy_with_content(original, "replaced")

        assert result.content == "replaced"
        # Original must not be mutated
        assert original.content == "old content"

    def test_model_copy_takes_priority_over_copy(self):
        """model_copy is preferred even if copy also exists."""
        copy_with_content = self._fn()

        expected = object()
        msg = MagicMock(spec=["model_copy", "copy"])
        msg.model_copy.return_value = expected

        result = copy_with_content(msg, "text")

        msg.model_copy.assert_called_once()
        msg.copy.assert_not_called()
        assert result is expected


# ---------------------------------------------------------------------------
# BUG-050: _load_documents subdirectory recursion
# ---------------------------------------------------------------------------


class TestLoadDocumentsRecursion:
    """_load_documents must include files nested inside subdirectories."""

    def _fn(self):
        from cogtrix_core.rag.ingest import _load_documents

        return _load_documents

    def _make_loader(self, docs):
        loader = MagicMock()
        loader.load.return_value = docs
        return loader

    def test_loads_files_in_subdirectory(self, tmp_path: Path):
        """Files inside sub-dirs are loaded, not skipped."""
        load_documents = self._fn()

        subdir = tmp_path / "sub"
        subdir.mkdir()
        txt_file = subdir / "nested.txt"
        txt_file.write_text("hello")

        fake_doc = MagicMock()
        fake_loader = self._make_loader([fake_doc])

        with patch("cogtrix_core.rag.ingest._get_loader", return_value=fake_loader):
            docs, errors = load_documents(tmp_path)

        assert fake_doc in docs
        assert errors == []

    def test_loads_files_at_multiple_depths(self, tmp_path: Path):
        """Files at depth > 1 are also loaded."""
        load_documents = self._fn()

        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        (deep / "file.txt").write_text("deep")

        fake_doc = MagicMock()
        fake_loader = self._make_loader([fake_doc])

        with patch("cogtrix_core.rag.ingest._get_loader", return_value=fake_loader):
            docs, errors = load_documents(tmp_path)

        assert fake_doc in docs

    def test_skips_directories_themselves(self, tmp_path: Path):
        """No attempt is made to load directories as documents."""
        load_documents = self._fn()

        subdir = tmp_path / "empty_dir"
        subdir.mkdir()

        with patch("cogtrix_core.rag.ingest._get_loader") as mock_get:
            docs, errors = load_documents(tmp_path)

        # _get_loader should never be called with a directory
        for call in mock_get.call_args_list:
            path_arg = call.args[0]
            assert path_arg.is_file(), f"_get_loader called with non-file: {path_arg}"

        assert docs == []

    def test_error_message_uses_relative_path(self, tmp_path: Path):
        """Error messages show the path relative to docs_dir, not just the name."""
        load_documents = self._fn()

        subdir = tmp_path / "sub"
        subdir.mkdir()
        txt_file = subdir / "bad.txt"
        txt_file.write_text("x")

        broken_loader = MagicMock()
        broken_loader.load.side_effect = RuntimeError("disk error")

        with patch("cogtrix_core.rag.ingest._get_loader", return_value=broken_loader):
            _, errors = load_documents(tmp_path)

        assert any("sub/bad.txt" in e or "sub" in e for e in errors)

    def test_unsupported_file_error_uses_relative_path(self, tmp_path: Path):
        """'Skipped unsupported file' message includes the relative path."""
        load_documents = self._fn()

        subdir = tmp_path / "docs"
        subdir.mkdir()
        (subdir / "file.xyz").write_text("x")

        with patch("cogtrix_core.rag.ingest._get_loader", return_value=None):
            _, errors = load_documents(tmp_path)

        assert any("docs/file.xyz" in e or "docs" in e for e in errors)

    def test_top_level_files_still_loaded(self, tmp_path: Path):
        """Files directly in docs_dir continue to be loaded after the change."""
        load_documents = self._fn()

        (tmp_path / "top.txt").write_text("top level")

        fake_doc = MagicMock()
        fake_loader = self._make_loader([fake_doc])

        with patch("cogtrix_core.rag.ingest._get_loader", return_value=fake_loader):
            docs, errors = load_documents(tmp_path)

        assert fake_doc in docs
        assert errors == []
