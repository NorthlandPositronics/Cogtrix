# Cogtrix RAG Guide

Complete guide to setting up and using the knowledge base (RAG) feature.

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
# Using OpenAI embeddings (requires API key)
python cogtrix.py --ingest

# Using Ollama embeddings (local, free)
python cogtrix.py --ingest --embedding-provider ollama
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

```
docs/
├── policies/
│   ├── remote-work-policy.pdf
│   └── expense-policy.pdf
├── guides/
│   ├── onboarding-guide.md
│   └── tech-stack.md
└── data/
    └── employees.csv
```

Note: Subdirectories are not scanned. Place all files directly in `docs/`.

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
  Vector DB output:    data/vectordb
  Embedding provider:  ollama

✓ Loaded 15 document(s)
✓ Created 234 chunk(s)
✓ Saved to data/vectordb/faiss_index
```

### Re-ingestion

To update the knowledge base after adding new documents:

```bash
# Re-run ingestion (overwrites existing index)
python cogtrix.py --ingest
```

---

## Querying

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

### Query Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `question` | Required | Search query |
| `k` | 4 | Number of chunks to retrieve (1-10) |
| `embedding_provider` | From config | Override embedding provider |

---

## Embedding Providers

### OpenAI Embeddings

**Pros:** High quality, fast  
**Cons:** Requires API key, costs money

```bash
export OPENAI_API_KEY="sk-..."
python cogtrix.py --ingest
```

**Default model:** `text-embedding-3-small`

### Ollama Embeddings

**Pros:** Free, local, no API key  
**Cons:** Requires Ollama running, slower

```bash
# Make sure Ollama is running
ollama serve

# Pull embedding model
ollama pull nomic-embed-text

# Run ingestion
python cogtrix.py --ingest --embedding-provider ollama
```

**Default model:** `nomic-embed-text`

### Using Named Providers

You can use any named Ollama provider from your config for embeddings:

```json
{
  "providers": {
    "gpu-server": {
      "type": "ollama",
      "base_url": "http://192.168.1.100:11434",
      "model": "llama3:70b"
    }
  },
  "rag": {
    "embedding_provider": "gpu-server",
    "embedding_model": "nomic-embed-text"
  }
}
```

The embedding provider will resolve to the named provider's type and base_url:

```
📚 RAG Document Ingestion

  Embedding provider:  gpu-server (ollama)
  Ollama URL:          http://192.168.1.100:11434
```

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

```json
{
  "rag": {
    "docs_dir": "docs",
    "vectordb_dir": "data/vectordb",
    "chunk_size": 1200,
    "chunk_overlap": 200,
    "embedding_provider": "ollama",
    "embedding_model": "nomic-embed-text"
  }
}
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `docs_dir` | `"docs"` | Source documents directory |
| `vectordb_dir` | `"data/vectordb"` | Vector database output |
| `chunk_size` | `1200` | Characters per chunk |
| `chunk_overlap` | `200` | Overlap between chunks |
| `embedding_provider` | `"openai"` | `"openai"`, `"ollama"`, or named provider |
| `embedding_model` | Auto | Embedding model name |

### Chunk Size Guidelines

| Document Type | Recommended Size | Overlap |
|---------------|------------------|---------|
| Technical docs | 1000-1500 | 150-200 |
| Legal documents | 800-1200 | 200-300 |
| General text | 1200-1500 | 150-200 |
| Short FAQs | 500-800 | 100-150 |

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

### "Failed to create embeddings" (OpenAI)

```
Cause: Missing or invalid API key
Solution: 
  export OPENAI_API_KEY="sk-..."
  Or use Ollama: --embedding-provider ollama
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

Note: Currently, only one knowledge base can be queried at a time (the one in config).

### Programmatic Access

```python
from src.rag import ingest_documents, IngestConfig
from pathlib import Path

config = IngestConfig(
    docs_dir=Path("./my-docs"),
    vectordb_dir=Path("./my-vectordb"),
    embedding_provider="ollama",
    embedding_model="nomic-embed-text",
)

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
