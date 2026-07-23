# PM Role-Test Harness

A one-shot test harness that runs the Cogtrix agent through six Project Manager scenarios against a RAG-grounded synthetic project corpus.

**Issue:** [#1948](https://github.com/NorthlandPositronics/Cogtrix/issues/1948)
**Methodology, scorecard semantics, and interpretation guidance:** see [`docs/optional/testing/pm-role-test-harness.md`](../../docs/optional/testing/pm-role-test-harness.md) (internal documentation).

## Running

```
python -m tests.role_pm.run                           # all 6 scenarios
python -m tests.role_pm.run --scenario 01,03,06       # filter by id or numeric prefix
python -m tests.role_pm.run --model qwen3-coder       # explicit model override
python -m tests.role_pm.run --output role_pm_run.json # write JSON report alongside stdout
python -m tests.role_pm.run --verbose                 # debug logging
```

## Prerequisites

- `[rag]` and `[rag-rerank]` extras installed: `uv sync --extra rag --extra rag-rerank`.
  - `[rag-rerank]` pulls in `sentence-transformers` (transitively `torch` + `transformers`, ~hundreds of MB) for the cross-encoder re-ranker the harness opts into. If the extra is absent, the CE stage degrades gracefully — retrieval falls back to the un-re-ranked FAISS pool, never *worse* than the baseline. See `src/rag/reranker.py` for the contract and #1952 / #2004 for the rationale.
- An embedding-provider API key. By default OpenAI embeddings. Override via environment:
  - `ROLE_PM_EMBEDDING_PROVIDER` — `openai` (default), `ollama`, or `google`.
  - `ROLE_PM_EMBEDDING_MODEL` — model name; defaults match the provider's standard small embedding model.
  - `ROLE_PM_EMBEDDING_BASE_URL` — for Ollama.
  - `ROLE_PM_EMBEDDING_API_KEY` — falls back to `OPENAI_API_KEY` if unset.
- An LLM API key for the configured model (same priority order as Gate 2 — `OPENROUTER_API_KEY` first, then provider-specific keys).
- The model must exist in `tests/evaluation/models.yaml`. Add an entry there if you want to run against a model not currently registered.

## Layout

| Path | Contents |
|---|---|
| `tests/role_pm/run.py` | CLI entry point |
| `tests/role_pm/scorecard.py` | Scorecard dataclasses + measurable-signal computation |
| `tests/role_pm/corpus_ingest.py` | Idempotent corpus ingest (sha256-based skip) |
| `tests/role_pm/system_prompt.md` | System prompt the harness uses |
| `tests/role_pm/corpus/` | RAG corpus the harness ingests |
| `tests/role_pm/scenarios/` | Six scenario YAMLs |
| `tests/role_pm/rag/faiss_index/` | Created on first run; reused on subsequent runs when the corpus hash matches |

## Exit codes

- `0` — every scenario was a clean pass (no bugs, no failed criteria).
- `1` — at least one scenario surfaced a bug or failed criterion.
- `2` — scenario-filter argument matched no scenarios.

This convention lets the harness be wired into a future CI gate without changing it.

## Out of scope

Not wired into Gate 2. Not run by CI. Manual invocation only.
