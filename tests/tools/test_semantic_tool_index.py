"""Tests for src/tools/semantic_tool_index.py and the query= parameter on request_tools."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CATALOG = {
    "send_email": "Send an email to a recipient",
    "web_search": "Search the web for information",
    "write_file": "Write content to a file on disk",
    "read_file": "Read content from a file on disk",
    "calculator": "Evaluate mathematical expressions",
}


def _make_embeddings_mock(dim: int = 4) -> MagicMock:
    """Return a mock Embeddings object with deterministic fixed vectors."""
    import numpy as np

    rng = np.random.default_rng(0)
    vecs = rng.random((len(_CATALOG), dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms

    mock = MagicMock()
    mock.embed_documents.return_value = vecs.tolist()
    # embed_query returns the first description vector (send_email)
    mock.embed_query.return_value = vecs[0].tolist()
    return mock


# ---------------------------------------------------------------------------
# ToolIndex — is_ready / index building
# ---------------------------------------------------------------------------


class TestToolIndexIsReady:
    def test_is_ready_false_before_search(self):
        from src.tools.semantic_tool_index import ToolIndex

        idx = ToolIndex(_CATALOG)
        assert not idx.is_ready()

    def test_is_ready_true_after_successful_embed(self):
        from src.tools.semantic_tool_index import ToolIndex

        emb_mock = _make_embeddings_mock()
        with (
            patch("src.tools.semantic_tool_index._HAS_NUMPY", True),
            patch("src.tools.semantic_tool_index._np") as np_mock,
            patch("src.tools.semantic_tool_index.create_embeddings", return_value=emb_mock),
        ):
            import numpy as np

            # Wire _np mock to real numpy so array ops work
            np_mock.array = np.array
            np_mock.linalg = np.linalg
            np_mock.where = np.where
            np_mock.argsort = np.argsort
            np_mock.float32 = np.float32

            idx = ToolIndex(_CATALOG)
            idx.search("email")
            assert idx.is_ready()


# ---------------------------------------------------------------------------
# ToolIndex — semantic search
# ---------------------------------------------------------------------------


class TestToolIndexSemanticSearch:
    def _build_ready_index(self, catalog=None):
        """Build a ToolIndex with real numpy but mocked embeddings."""
        import numpy as np

        from src.tools.semantic_tool_index import ToolIndex

        cat = catalog if catalog is not None else _CATALOG
        emb_mock = MagicMock()
        dim = 4
        rng = np.random.default_rng(42)
        vecs = rng.random((len(cat), dim)).astype(np.float32)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        emb_mock.embed_documents.return_value = vecs.tolist()
        # query vector = first catalog vector (exact match for first tool)
        emb_mock.embed_query.return_value = vecs[0].tolist()

        with (
            patch("src.tools.semantic_tool_index._HAS_NUMPY", True),
            patch("src.tools.semantic_tool_index.create_embeddings", return_value=emb_mock),
        ):
            idx = ToolIndex(cat)
            idx.search("anything")  # trigger build

        # Manually replace _np reference so subsequent calls use real numpy
        import src.tools.semantic_tool_index as _mod

        _mod._np = np  # type: ignore[attr-defined]
        idx._embeddings = emb_mock
        return idx

    def test_search_returns_list_of_tool_names(self):
        import numpy as np

        from src.tools.semantic_tool_index import ToolIndex

        emb_mock = MagicMock()
        dim = 4
        rng = np.random.default_rng(7)
        vecs = rng.random((len(_CATALOG), dim)).astype(np.float32)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        emb_mock.embed_documents.return_value = vecs.tolist()
        emb_mock.embed_query.return_value = vecs[0].tolist()

        with (
            patch("src.tools.semantic_tool_index._HAS_NUMPY", True),
            patch("src.tools.semantic_tool_index.create_embeddings", return_value=emb_mock),
            patch("src.tools.semantic_tool_index._np", np),
        ):
            idx = ToolIndex(_CATALOG)
            results = idx.search("email")

        assert isinstance(results, list)
        assert all(r in _CATALOG for r in results)

    def test_search_respects_k_limit(self):
        import numpy as np

        from src.tools.semantic_tool_index import ToolIndex

        emb_mock = MagicMock()
        dim = 4
        rng = np.random.default_rng(99)
        vecs = rng.random((len(_CATALOG), dim)).astype(np.float32)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        emb_mock.embed_documents.return_value = vecs.tolist()
        emb_mock.embed_query.return_value = vecs[0].tolist()

        with (
            patch("src.tools.semantic_tool_index._HAS_NUMPY", True),
            patch("src.tools.semantic_tool_index.create_embeddings", return_value=emb_mock),
            patch("src.tools.semantic_tool_index._np", np),
        ):
            idx = ToolIndex(_CATALOG)
            results = idx.search("anything", k=2)

        assert len(results) <= 2

    def test_search_k_larger_than_catalog_returns_all(self):
        import numpy as np

        from src.tools.semantic_tool_index import ToolIndex

        small = {"tool_a": "does A", "tool_b": "does B"}
        emb_mock = MagicMock()
        dim = 4
        rng = np.random.default_rng(1)
        vecs = rng.random((2, dim)).astype(np.float32)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        emb_mock.embed_documents.return_value = vecs.tolist()
        emb_mock.embed_query.return_value = vecs[0].tolist()

        with (
            patch("src.tools.semantic_tool_index._HAS_NUMPY", True),
            patch("src.tools.semantic_tool_index.create_embeddings", return_value=emb_mock),
            patch("src.tools.semantic_tool_index._np", np),
        ):
            idx = ToolIndex(small)
            results = idx.search("something", k=100)

        assert len(results) == 2
        assert set(results) == {"tool_a", "tool_b"}

    def test_search_empty_catalog_returns_empty(self):
        from src.tools.semantic_tool_index import ToolIndex

        idx = ToolIndex({})
        results = idx.search("email")
        assert results == []

    def test_embeddings_cached_across_searches(self):
        import numpy as np

        from src.tools.semantic_tool_index import ToolIndex

        emb_mock = MagicMock()
        dim = 4
        rng = np.random.default_rng(5)
        vecs = rng.random((len(_CATALOG), dim)).astype(np.float32)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        emb_mock.embed_documents.return_value = vecs.tolist()
        emb_mock.embed_query.return_value = vecs[0].tolist()

        with (
            patch("src.tools.semantic_tool_index._HAS_NUMPY", True),
            patch("src.tools.semantic_tool_index.create_embeddings", return_value=emb_mock),
            patch("src.tools.semantic_tool_index._np", np),
        ):
            idx = ToolIndex(_CATALOG)
            idx.search("email")
            idx.search("file")

        # embed_documents must be called exactly once (index cached)
        emb_mock.embed_documents.assert_called_once()


# ---------------------------------------------------------------------------
# ToolIndex — keyword fallback
# ---------------------------------------------------------------------------


class TestToolIndexFallback:
    def test_keyword_fallback_when_numpy_unavailable(self):
        from src.tools.semantic_tool_index import ToolIndex

        with patch("src.tools.semantic_tool_index._HAS_NUMPY", False):
            idx = ToolIndex(_CATALOG)
            results = idx.search("email")

        assert not idx.is_ready()
        assert "send_email" in results

    def test_keyword_fallback_when_embedding_provider_raises(self):
        from src.tools.semantic_tool_index import ToolIndex

        with (
            patch("src.tools.semantic_tool_index._HAS_NUMPY", True),
            patch(
                "src.tools.semantic_tool_index.create_embeddings",
                side_effect=RuntimeError("provider down"),
            ),
        ):
            idx = ToolIndex(_CATALOG)
            results = idx.search("file")

        assert not idx.is_ready()
        assert any("file" in r for r in results)

    def test_keyword_fallback_caps_at_k(self):
        from src.tools.semantic_tool_index import ToolIndex

        with patch("src.tools.semantic_tool_index._HAS_NUMPY", False):
            idx = ToolIndex(_CATALOG)
            results = idx.search("file", k=1)

        assert len(results) <= 1


# ---------------------------------------------------------------------------
# configure() and TOOL_SETUP()
# ---------------------------------------------------------------------------


class TestConfigure:
    def test_configure_stores_values(self):
        import src.tools.semantic_tool_index as _mod
        from src.tools.semantic_tool_index import configure

        configure(provider="openai", model="text-embedding-3-small", api_key="sk-test")
        assert _mod._config["provider"] == "openai"
        assert _mod._config["model"] == "text-embedding-3-small"
        assert _mod._config["api_key"] == "sk-test"

    def test_tool_setup_calls_configure(self):
        from src.tools.semantic_tool_index import TOOL_SETUP, configure

        mock_config = MagicMock()
        mock_config.resolve_embedding_config.return_value = (
            "google",
            "text-embedding-004",
            None,
            "gkey",
        )

        with patch("src.tools.semantic_tool_index.configure", wraps=configure) as mock_cfg:
            TOOL_SETUP(mock_config)

        mock_cfg.assert_called_once_with(
            provider="google",
            model="text-embedding-004",
            base_url=None,
            api_key="gkey",
        )


# ---------------------------------------------------------------------------
# Config field
# ---------------------------------------------------------------------------


class TestConfigField:
    def test_semantic_tool_index_default_is_true(self):
        from src.config import Config

        cfg = Config()
        assert cfg.semantic_tool_index is True

    def test_semantic_tool_index_false_from_yaml(self, tmp_path):
        import yaml

        from src.config import Config, _apply_config_file

        cfg = Config()
        cfg_file = tmp_path / "cogtrix.yaml"
        cfg_file.write_text(yaml.dump({"semantic_tool_index": False}))
        _apply_config_file(cfg, cfg_file)

        assert cfg.semantic_tool_index is False


# ---------------------------------------------------------------------------
# create_request_tools_tool with query=
# ---------------------------------------------------------------------------


class TestRequestToolsQuery:
    def _make_tool(self, tool_index=None):
        from src.tools.configure import create_request_tools_tool

        available = {k: MagicMock() for k in _CATALOG}
        return create_request_tools_tool(
            available_tools=available,
            catalog=_CATALOG,
            active_names=set(),
            protected_names=set(),
            tool_index=tool_index,
        )

    def test_query_with_tool_index_returns_semantic_results(self):
        idx_mock = MagicMock()
        idx_mock.search.return_value = ["send_email", "web_search"]

        tool = self._make_tool(tool_index=idx_mock)
        result = tool.func(query="send an email")

        assert "send_email" in result
        assert "Semantic search results" in result
        idx_mock.search.assert_called_once_with("send an email", k=8)

    def test_query_with_no_tool_index_falls_back_to_full_catalog(self):
        tool = self._make_tool(tool_index=None)
        result = tool.func(query="send an email")

        # Should fall through to full catalog listing
        assert "Tools you can ADD" in result

    def test_add_wins_over_query(self):
        idx_mock = MagicMock()
        idx_mock.search.return_value = ["send_email"]

        tool = self._make_tool(tool_index=idx_mock)
        # add is non-empty → query is ignored, add is processed
        result = tool.func(add=["calculator"], query="email")

        idx_mock.search.assert_not_called()
        assert "calculator" in result.lower() or "loaded" in result.lower()

    def test_query_no_matches_returns_helpful_message(self):
        idx_mock = MagicMock()
        idx_mock.search.return_value = []  # nothing matches

        tool = self._make_tool(tool_index=idx_mock)
        result = tool.func(query="quantum teleportation")

        assert "No tools matched" in result
        assert "quantum teleportation" in result
