"""#2249 — knowledge-store embeddings must take credentials from Config, not env.

After #2223 unsets OPENAI_API_KEY post-startup, the old code built
``OpenAIEmbeddings(model=...)`` with no key (env-reliant → 401) and gated on a
non-existent ``config.embedding`` attribute. ``_setup_embeddings`` now resolves
through ``Config.resolve_embedding_config()`` + ``create_embeddings_from_config``
so the api_key/base_url come from Config (surviving the unset via the #2233
cache), with graceful Ollama → keyword fallback.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cogtrix_core.assistant.knowledge import SharedKnowledgeStore


def _make_store(tmp_path, resolve=None) -> SharedKnowledgeStore:
    """Build a store with construction I/O patched out and a controllable config."""
    config = SimpleNamespace(
        services={"assistant": {"knowledge": {"data_dir": str(tmp_path)}}},
        data_dir=str(tmp_path),
    )
    if resolve is not None:
        config.resolve_embedding_config = resolve
    with (
        patch.object(SharedKnowledgeStore, "_load", return_value=None),
        patch.object(SharedKnowledgeStore, "_setup_embeddings", return_value=None),
        patch("cogtrix_core.assistant.knowledge.threading.Thread"),
    ):
        store = SharedKnowledgeStore(config=config, llm=MagicMock())
    store._embeddings_ready.set()
    return store


@pytest.fixture
def store_with_openai(tmp_path):
    # resolve_embedding_config() returns (provider_type, model, base_url, api_key)
    resolve = MagicMock(return_value=("openai", "text-embedding-3-small", None, "sk-from-config"))
    return _make_store(tmp_path, resolve=resolve), resolve


class TestCredentialsFromConfig:
    def test_passes_resolved_api_key_to_factory(self, store_with_openai) -> None:
        store, resolve = store_with_openai
        fake_fn = MagicMock()
        with (
            patch(
                "cogtrix_core.providers.create_embeddings_from_config",
                return_value=(fake_fn, "openai/text-embedding-3-small"),
            ) as factory,
            patch.object(SharedKnowledgeStore, "_load_or_create_index", return_value=None),
        ):
            store._setup_embeddings()

        resolve.assert_called_once()
        factory.assert_called_once()
        kwargs = factory.call_args.kwargs
        # The key (and base_url) come from Config, NOT os.environ.
        assert kwargs["api_key"] == "sk-from-config"
        assert kwargs["model"] == "text-embedding-3-small"
        assert store._embedding_fn is fake_fn
        assert store._embedding_tag == "openai/text-embedding-3-small"

    def test_falls_back_to_ollama_when_configured_provider_fails(self, tmp_path) -> None:
        resolve = MagicMock(return_value=("openai", "m", "https://openrouter.ai/api/v1", "k"))
        store = _make_store(tmp_path, resolve=resolve)
        ollama_fn = MagicMock()
        with (
            patch(
                "cogtrix_core.providers.create_embeddings_from_config",
                side_effect=RuntimeError("openrouter is not an embeddings endpoint"),
            ),
            patch("langchain_ollama.OllamaEmbeddings", return_value=ollama_fn),
            patch.object(SharedKnowledgeStore, "_load_or_create_index", return_value=None),
        ):
            store._setup_embeddings()

        assert store._embedding_fn is ollama_fn
        assert store._embedding_tag == "ollama/nomic-embed-text"

    def test_keyword_recall_when_all_providers_fail(self, tmp_path) -> None:
        resolve = MagicMock(side_effect=Exception("no models configured"))
        store = _make_store(tmp_path, resolve=resolve)
        with (
            patch("langchain_ollama.OllamaEmbeddings", side_effect=Exception("no ollama")),
            patch.object(SharedKnowledgeStore, "_load_or_create_index") as load_idx,
        ):
            store._setup_embeddings()

        assert store._embedding_fn is None
        load_idx.assert_not_called()

    def test_config_without_resolve_method_uses_ollama(self, tmp_path) -> None:
        store = _make_store(tmp_path, resolve=None)  # SimpleNamespace, no resolve method
        ollama_fn = MagicMock()
        with (
            patch("langchain_ollama.OllamaEmbeddings", return_value=ollama_fn),
            patch.object(SharedKnowledgeStore, "_load_or_create_index", return_value=None),
        ):
            store._setup_embeddings()

        assert store._embedding_fn is ollama_fn
        assert store._embedding_tag == "ollama/nomic-embed-text"
