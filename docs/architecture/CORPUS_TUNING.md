# Corpus-Tuning Footprint

**Status:** Adopted 2026-06-05 (#2006 cycle-10 bias audit).

## What this is

Several production defaults in `src/orchestration/` were tuned empirically against the **PM-test corpus** (`tests/role_pm/corpus/` — a fabricated project-management corpus modelled on a typical software-delivery program with risks, decisions, change requests, and a stakeholder register).

The values are reasonable defaults for business-document corpora generally, but they are not corpus-agnostic. Operators deploying Cogtrix against a meaningfully different corpus (legal, medical, scientific, conversational) should audit and likely override these defaults.

This document inventories the tuning footprint so the bias is visible.

## Inventory

### Retry budgets (`src/orchestration/graph.py:677-706`)

| Constant | Default | Tuning origin |
|---|---:|---|
| `_MAX_UNSUPPORTED_QUOTE_RETRIES` | 2 | #2007 — qwen3-coder × PM cycle-3 |
| `_MAX_UNSUPPORTED_ATTRIBUTION_RETRIES` | 2 | #2007 — same empirical observation |
| `_MAX_ENTITY_OWNER_MISMATCH_RETRIES` | 2 | #2007 — cycle-3 PM run |
| `_MAX_CORPUS_ATTRIBUTION_MISMATCH_RETRIES` | 3 | #2006 cycle-10 — log evidence |
| `_MAX_TOPIC_SUBSTITUTION_RETRIES` | 2 | #2007 — same empirical reasoning |

Each constant has a "one extra LLM call per affected scenario per detector firing" cost ceiling. Reduce if production latency is a tighter constraint than mismatch repair quality.

### Structural-noun blocklists (`src/orchestration/verification.py`)

Both are overridable via kwargs as of #2006 cycle-10:

| Constant | Used by | Override |
|---|---|---|
| `_DEFAULT_STAKEHOLDER_NAME_BLOCKLIST` | `detect_entity_owner_mismatch` | `stakeholder_name_blocklist=` |
| `_DEFAULT_TITLE_PHRASE_STOPWORDS` | `detect_topic_substitution` (via `_extract_distinctive_subjects`) | `title_phrase_stopwords=` |

Both default sets contain English business-document structural nouns (`project`, `risk`, `register`, `report`, `decision`, `change`, `summary`, `status`, `update`, ...). In a non-PM corpus, a legitimate stakeholder name like `Risk Manager Smith` or a legitimate topic-subject like `Decision Theory` would be wrongly filtered. Override at the call site if your corpus has such names.

### RAG chunk size (`src/rag/ingest.py:135-143`)

| Constant | Default | Tuning origin |
|---|---:|---|
| `chunk_size` | 800 | #1952 Option C — qwen3-embedding diagnostics |
| `chunk_overlap` | 100 | same |

Calibrated for the qwen3-embedding model. The PM harness itself uses 500/50 (tighter, but harness-specific). For meaningfully different embedding models or corpora, re-run the chunk-size sweep before relying on these defaults.

### Entity-ID regex (`src/orchestration/verification.py:1684-1693`)

`_ENTITY_ID_RE` explicitly enumerates PM-corpus formats (`R-NN`, `DEC-YYYY-MM-DD-NN`, `CHG-NIMB-NN`, `NIMB-WBS-NN`) before falling through to a generic `[A-Z]{2,8}-\d{1,4}` catch-all. The explicit lines are redundant for matching (the catch-all covers them) but signal which formats were prioritized during PM-test work.

## What is NOT in the production code

Several earlier drafts of the recovery nudges contained literal PM-corpus identifiers (`R-13`, `Hyeon-Jin Park`, `05_risk_register.md`, `Migration Squad`). Those were removed in the #2006 cycle-10 bias audit:

- `src/orchestration/nodes/recovery.py` — `_format_corpus_attribution_mismatch_nudge` (cleaned in `d251012`)
- `src/orchestration/verification.py` — `_ENTITY_OWNER_MISMATCH_NUDGE` (cleaned in the same PR)

Future maintainers: do **not** re-introduce concrete corpus-specific examples in runtime LLM-facing strings. Use abstract failure-mode descriptions instead. Docstrings and code comments are fine — those never reach the LLM.

## When to revisit

Trigger an audit of this document when:

1. Cogtrix is deployed against a corpus whose section/document structure differs materially from a PM/business corpus (`docs/architecture/` legal corpora, `docs/medical/` notes, etc.).
2. The embedding model changes (chunk-size sweep may need to be re-run).
3. A new structured-identifier format appears regularly in tool results (consider extending `_ENTITY_ID_RE`).
4. The recovery-budget retry counts are observed to be load-bearing in production deployments where the agent profile differs from qwen3-coder.
