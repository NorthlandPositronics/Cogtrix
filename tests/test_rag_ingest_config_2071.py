"""Regression test for #2071 — agent rag_ingest must thread the full embedding +
chunk config into IngestConfig.

Previously `rag_ingest` passed only `embedding_provider`, dropping embedding_model
/ base_url / api_key / chunk_size / chunk_overlap. On OpenAI/Google providers this
built the index with the default embedding model and no API key — an index in a
different embedding space than `query_knowledge_base` reads (or a hard failure).
"""

from __future__ import annotations

from unittest.mock import patch

import src.tools.rag as rag_tool


def test_agent_rag_ingest_threads_embedding_and_chunk_config() -> None:
    orig = dict(rag_tool._rag_config)
    try:
        rag_tool.configure_rag(
            {
                "embedding_provider": "openai",
                "embedding_model": "text-embedding-3-small",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "sk-test-key",
                "chunk_size": 1234,
                "chunk_overlap": 77,
            }
        )
        captured: dict = {}

        def fake_ingest_many(paths, config):
            captured["config"] = config
            return {str(p): True for p in paths}

        with patch("src.rag.ingest.ingest_many", fake_ingest_many):
            rag_tool.rag_ingest("/tmp/some-doc.txt")

        cfg = captured["config"]
        # The index must be built with the SAME embedding the query side uses.
        assert cfg.embedding_provider == "openai"
        assert cfg.embedding_model == "text-embedding-3-small"
        assert cfg.base_url == "https://openrouter.ai/api/v1"
        assert cfg.api_key == "sk-test-key"
        # Operator chunk overrides must be honored, not the IngestConfig defaults.
        assert cfg.chunk_size == 1234
        assert cfg.chunk_overlap == 77
    finally:
        rag_tool._rag_config = orig
