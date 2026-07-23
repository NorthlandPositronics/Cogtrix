# Cogtrix RAG Guide

Turn your own documents into a searchable knowledge base that the agent can query during conversation. This feature uses Retrieval-Augmented Generation (RAG): your documents are split into chunks, converted into vector embeddings, and stored in a local FAISS index. When you ask a question, the most relevant chunks are retrieved and sent to the LLM alongside your query.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Document Preparation](#document-preparation)
- [Ingestion](#ingestion)
- [Querying](#querying)
- [Embedding Providers](#embedding-providers)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Overview

RAG (Retrieval-Augmented Generation) allows the agent to answer questions based on your documents. The process:

```
Documents (docs/)
       │
       ▼ Ingestion
┌─────────────────┐
│  Split into     │
│  chunks         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Create         │
│  embeddings     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Store in       │
│  FAISS index    │
└─────────────────┘
         │
         ▼ Query
┌─────────────────┐
│  Semantic       │
│  search         │
└────────┬────────┘
         │
         ▼
Relevant chunks → LLM → Answer
```

---

## Quick Start

### 1. Add Documents

```bash
mkdir -p docs
cp your-documents.pdf docs/
cp your-notes.md docs/
```

### 2. Build Vector Database

```bash
# Using Ollama embeddings (default — local, free)
python cogtrix.py --ingest

# Using OpenAI embeddings instead (requires API key)
python cogtrix.py --ingest --embedding-provider openai
```

### 3. Query

```bash
python cogtrix.py
You: What does the policy say about remote work?
```

---

## Document Preparation

### Supported Formats

| Format | Extensions | Notes |
|--------|------------|-------|
| PDF | `.pdf` | Text-based PDFs (not scanned images) |
| Markdown | `.md`, `.markdown` | Plain text with formatting |
| Text | `.txt` | Plain text files |
| CSV | `.csv` | Tabular data |

### Best Practices

1. **Use text-based PDFs** — Scanned documents won't work without OCR
2. **Structure documents** — Use headings, sections, lists
3. **Include context** — Document titles and sources help retrieval
4. **Keep files focused** — One topic per document improves relevance

### Directory Structure

Place files in the docs directory — **subdirectories are traversed recursively**, so any folder layout is supported.

```
docs/
├── remote-work-policy.pdf
├── expense-policy.pdf
├── onboarding-guide.md
├── tech-stack.md
└── employees.csv
```

You can organize files into subdirectories; they will all be ingested. Use `--docs-dir` to point at a specific subdirectory if you only want to ingest part of your document tree.

---

## Ingestion

### Basic Ingestion

```bash
python cogtrix.py --ingest
```

### With Options

```bash
# Custom documents directory
python cogtrix.py --ingest --docs-dir ./company-docs

# Custom output location
python cogtrix.py --ingest --vectordb-dir ./my-vectordb

# Use Ollama embeddings
python cogtrix.py --ingest --embedding-provider ollama

# Specific embedding model
python cogtrix.py --ingest --embedding-provider ollama --embedding-model mxbai-embed-large

# Full customization
python cogtrix.py --ingest \
  --docs-dir ./legal-docs \
  --vectordb-dir ./legal-vectordb \
  --embedding-provider ollama \
  --embedding-model nomic-embed-text
```

### Ingestion Output

```
📚 RAG Document Ingestion

  Documents directory: docs
  Vector DB output:    vectordb
  Embedding provider:  ollama

✓ Loaded 15 document(s)
✓ Created 234 chunk(s)
✓ Saved to vectordb/faiss_index
```

### Re-ingestion

To update the knowledge base after adding new documents:

```bash
# Re-run ingestion (overwrites existing index)
python cogtrix.py --ingest
```

---

## Querying

### Auto-Activation

When a knowledge base exists (either a global CLI index or per-document API indexes), the `query_knowledge_base` tool is **automatically pinned as active** at startup. The agent can use it immediately without loading it via `request_tools`. The tool description dynamically shows the number of indexes and their total size.

The tool searches all available FAISS indexes:
- **Global CLI index** — built via `--ingest`, stored at `data/vectordb/faiss_index/`
- **Per-document API indexes** — created when documents are uploaded via the API, stored at `data/api/uploads/{doc_id}/vectordb/faiss_index/`

Results from all indexes are merged and deduplicated by content (first 200 characters).

### In Conversation

The agent automatically uses the knowledge base when relevant:

```
You: What are the requirements for expense reports?

Agent: Based on the expense policy document, the requirements are:
1. Submit within 30 days of expense
2. Include receipts for amounts over $25
3. Use the standard expense form
[Source: expense-policy.pdf, page 3]
```

### Direct Tool Usage

The `query_knowledge_base` tool can be used explicitly:

```
You: Search the knowledge base for "vacation policy"
```

### Saving Notes

Use `save_to_knowledge_base` to persist a note, fact, or decision for later retrieval by the agent.

```
You: Save this: the deployment window is Friday at 18:00 UTC.
```

The tool accepts:

- `content` - required note text to store
- `source` - optional origin label, default `agent`
- `tags` - optional list of topic tags

Saved notes go to the dedicated agent-notes sub-index when FAISS is available. If FAISS is unavailable, Cogtrix falls back to a JSONL log so the information is still preserved.

### Query Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `question` | Required | Search query |
| `k` | 4 | Number of chunks to retrieve (1-10) |

### Hybrid Retrieval (BM25 + Vector)

Cogtrix supports an opt-in **hybrid retrieval** path that combines pure vector (dense) ranking with BM25 sparse ranking via Reciprocal Rank Fusion (RRF). This addresses two well-known weaknesses of pure-vector retrieval (catalogued in issue #1952):

- **Numeric / monetary token queries** — embedding models smooth tokens like `$2,400,000` toward generic semantics, so the document that actually contains the amount can fall out of the top-K. BM25 treats the amount as a distinct term and pins the right document.
- **One "central" document dominates unrelated queries** — when a corpus has a single dense reference document (a stakeholder register, a glossary), embeddings often pull every query toward its chunks. BM25's IDF weighting downweights such terms.

Hybrid is **off by default** — every existing pure-vector pipeline keeps working unchanged. Flip both flags in your `~/.cogtrix.yaml` to enable:

```yaml
rag:
  build_bm25_sidecar: true   # write bm25.pkl alongside the FAISS index at ingest time
  use_bm25_hybrid:    true   # fuse vector + BM25 ranks at query time
  bm25_rrf_k:         60     # RRF tuning constant (Cormack et al. 2009 standard)
```

Then re-ingest to build the BM25 sidecar:

```bash
python cogtrix.py --ingest
```

The sidecar (`bm25.pkl`) lives alongside `index.faiss`. The query path looks for it automatically:

- If both the flag is on AND the sidecar exists → hybrid retrieval runs.
- If the flag is on but no sidecar exists → graceful fallback to pure-vector.
- If the flag is off → pure-vector (regardless of sidecar presence).

**Cost:** the sidecar adds ~5-10% to ingest wall-clock time and a small additional disk file (tokens + chunk text). Hybrid queries add one extra in-memory BM25 scoring pass per index — typically < 10 ms for corpora under 10k chunks. `rank-bm25` is pure Python with only `numpy` as a dependency (already in the base install), so no native compilation is required.

---

## Embedding Providers

The default embedding provider is `ollama` (local, no API key required). OpenAI is also supported via `--embedding-provider openai`.

### Picking a model for retrieval quality (#1952 Option D)

The embedding model is the single biggest lever on retrieval quality.
A weaker model produces *diffuse* vectors — chunks on different topics
land near each other in vector space, and queries pull toward whichever
document happens to sit at the corpus centroid (typically a dense
stakeholder register or glossary).  A stronger model spreads the
corpus out and gives discriminative ranking even on hard queries
(numeric tokens, generic role words).

If you're seeing symptoms like:

- Exact-text queries fail to surface the document containing the
  text (e.g. searching for an amount that appears verbatim in one
  document and getting back unrelated chunks);
- Generic role-word queries (*"budget memo"*, *"schedule slip"*)
  consistently return chunks from a single dense reference document
  regardless of topic;

…the most leveraged single change is to switch to a more discriminative
embedding model.  Suggested upgrade path:

| If you're on… | Try… | Notes |
|---|---|---|
| Ollama `nomic-embed-text` (default) | Ollama `mxbai-embed-large` | Same provider; larger model with stronger discrimination.  `ollama pull mxbai-embed-large` then re-ingest. |
| OpenAI `text-embedding-3-small` (default) | OpenAI `text-embedding-3-large` | Same provider; ~3× the dimensionality.  Noticeably better on short-token / monetary-token queries.  Costs more per token but typical RAG corpora are small. |
| Any small open-model embedding | A bge-large / qwen3-embedding sized model | Larger open models that ship via Ollama or vLLM (set `provider.type: openai` with the appropriate `base_url`). |

After switching, run `python cogtrix.py --ingest` to rebuild the
FAISS index with the new vectors.  Existing chunks must be
re-embedded — the index is not portable across models.

This is the lightest-touch option for #1952's regime-B (monetary
tokens) and regime-C (one document dominates) failure modes.  Combine
with the BM25 hybrid path (above) and the lowered chunk-size defaults
(`chunk_size: 800`, `chunk_overlap: 100`) for compound effect.

### Ollama Embeddings (default)

**Pros:** Free, local, no API key
**Cons:** Requires Ollama running

```bash
# Make sure Ollama is running
ollama serve

# Pull embedding model
ollama pull nomic-embed-text

# Run ingestion (default — no flags needed)
python cogtrix.py --ingest
```

**Default model:** `nomic-embed-text`

### OpenAI Embeddings

**Pros:** High quality, fast
**Cons:** Requires API key, costs money

```bash
export OPENAI_API_KEY="sk-..."
python cogtrix.py --ingest --embedding-provider openai
```

**Default model:** `text-embedding-3-small`

### Google Embeddings

> **Note:** Google embeddings are supported via the config file (`rag.model` referencing a Google provider) but are NOT available via the `--embedding-provider` CLI flag.

**Pros:** High quality
**Cons:** Requires API key (`GEMINI_API_KEY`)

```bash
export GEMINI_API_KEY="..."
# Configure Google embeddings via .cogtrix.yaml (see "Using Named Providers" below)
```

**Default model:** `text-embedding-004`

Requires `langchain-google-genai`: `uv pip install "cogtrix[google]"`

### Using Named Providers

You can reference any named provider from your config for embeddings by defining a model entry in the `models` registry and pointing `rag.model` at it. The provider connection details (type, base_url, api_key) are resolved automatically.

```yaml
providers:
  gpu-server:
    type: ollama
    base_url: "http://192.168.1.100:11434"
  cloud-openai:
    type: openai
    api_key: "sk-..."

models:
  embed-local:
    provider: gpu-server
    model: nomic-embed-text
  embed-cloud:
    provider: cloud-openai
    model: text-embedding-3-small

rag:
  model: embed-local
```

Switch between embedding providers by changing the `rag.model` value — no need to touch the provider entries themselves.

### Available Ollama Embedding Models

| Model | Size | Quality |
|-------|------|---------|
| `nomic-embed-text` | 274M | Good |
| `mxbai-embed-large` | 670M | Better |
| `all-minilm` | 46M | Fast, smaller |
| `nomic-embed-text-v2-moe` | MoE | Advanced |

---

## Configuration

### Via Config File

```yaml
rag:
  docs_dir: docs
  vectordb_dir: vectordb
  chunk_size: 800
  chunk_overlap: 100
  model: embed-local
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `docs_dir` | `"docs"` | Source documents directory |
| `vectordb_dir` | `"vectordb"` | Vector database output |
| `chunk_size` | `800` | Characters per chunk |
| `chunk_overlap` | `100` | Overlap between chunks |
| `model` | `null` | Model name from the `models` registry for embeddings. Falls back to the active provider when not set. |

> **Default changed (#1952 Option C):** `chunk_size` lowered from `2000` → `800`,
> `chunk_overlap` from `200` → `100`.  Larger chunks span multiple topics and dilute
> the per-chunk semantic vector — the smaller defaults give each chunk tighter
> focus, partially relieving the *"one document dominates"* retrieval failure mode
> catalogued in #1952.  Operators with explicit values in `~/.cogtrix.yaml` keep
> their overrides; operators on defaults will see the new behaviour after
> re-running `python cogtrix.py --ingest`.

### Chunk Size Guidelines

| Document Type | Recommended Size | Overlap |
|---------------|------------------|---------|
| Technical docs | 600-1000 | 80-150 |
| Legal documents | 800-1200 | 200-300 |
| General text | 800-1200 | 100-150 |
| Short FAQs | 400-700 | 50-100 |

Smaller chunks = more precise retrieval, larger context window usage  
Larger chunks = more context per chunk, fewer chunks needed

---

## Troubleshooting

### "No vector store found"

```
Cause: Vector database hasn't been built
Solution: Run python cogtrix.py --ingest
```

### "Documents directory not found"

```
Cause: docs/ directory doesn't exist
Solution: mkdir -p docs && cp your-files.pdf docs/
```

### "No documents loaded"

```
Cause: No supported files in docs/
Solution: Add PDF, MD, TXT, or CSV files to docs/
```

### "Failed to create embeddings"

```
Cause: Missing or invalid API key (OpenAI/Google), or Ollama not running
Solutions:
  # For OpenAI
  export OPENAI_API_KEY="sk-..."
  python cogtrix.py --ingest --embedding-provider openai

  # For Google (config file only — not available via --embedding-provider)
  # See "Google Embeddings" section above for config-based setup

  # Use Ollama (default, no API key needed)
  python cogtrix.py --ingest
```

### "Failed to connect to Ollama"

```
Cause: Ollama not running
Solution:
  1. Start Ollama: ollama serve
  2. Pull model: ollama pull nomic-embed-text
  3. Retry ingestion
```

### "Out of memory during ingestion"

```
Cause: Too many documents or large files
Solutions:
  1. Process fewer documents at a time
  2. Use smaller embedding model
  3. Reduce chunk_size in config
```

### Poor retrieval quality

```
Causes & Solutions:
  1. Chunk size too large → Reduce chunk_size
  2. Wrong embedding model → Try different model
  3. Documents poorly structured → Improve formatting
  4. Query too vague → Be more specific
```

### Embedding mismatch error

```
Cause: Query uses different embedding model than index
Solution: Rebuild index with same model you'll use for queries
  python cogtrix.py --ingest --embedding-provider <same-provider>
```

---

## Advanced Usage

### Multiple Knowledge Bases

Create separate knowledge bases for different topics:

```bash
# Legal documents
python cogtrix.py --ingest --docs-dir ./legal --vectordb-dir ./data/legal-vectordb

# Technical docs
python cogtrix.py --ingest --docs-dir ./tech --vectordb-dir ./data/tech-vectordb
```

All available indexes (global CLI index and per-document API indexes) are searched automatically and results are merged.

### Programmatic Access

```python
from src.rag import ingest_documents, IngestConfig
from pathlib import Path

# ``vectordb_dir`` is the EXACT directory where ``index.faiss`` will be
# written.  The default CLI / query-side convention nests the index under
# a ``faiss_index/`` segment, so callers that want to mirror that layout
# should include the segment explicitly:
#     vectordb_dir=Path("./my-vectordb/faiss_index")

# Using Ollama (default)
config = IngestConfig(
    docs_dir=Path("./my-docs"),
    vectordb_dir=Path("./my-vectordb/faiss_index"),
    embedding_provider="ollama",
    embedding_model="nomic-embed-text",
)

# Using OpenAI or Google — pass the api_key explicitly
# config = IngestConfig(
#     docs_dir=Path("./my-docs"),
#     vectordb_dir=Path("./my-vectordb/faiss_index"),
#     embedding_provider="openai",
#     embedding_model="text-embedding-3-small",
#     api_key="sk-...",
# )

result = ingest_documents(config)

if result.success:
    print(f"Created {result.chunks_created} chunks")
else:
    print(f"Errors: {result.errors}")
```

---

## See Also

- [CONFIGURATION.md](CONFIGURATION.md) — Full configuration reference
- [TOOLS_REFERENCE.md](TOOLS_REFERENCE.md) — query_knowledge_base tool
- [PROVIDERS.md](PROVIDERS.md) — Embedding provider setup
