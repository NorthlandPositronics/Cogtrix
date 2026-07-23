"""
Shared knowledge store for Cogtrix assistant mode.

Extracts durable, entity-centric facts from conversation turns and recalls
them when relevant to other chats.  Provides cross-chat knowledge without
exposing per-chat history.
"""

from __future__ import annotations

import concurrent.futures as _cf
import hashlib
import heapq
import json
import logging
import os
import tempfile
import threading
import time
from copy import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("cogtrix")

_EXTRACTION_MAX_WORKERS = 2
_extraction_pool: _cf.ThreadPoolExecutor | None = None
_extraction_pool_lock = threading.Lock()


def _get_extraction_pool() -> _cf.ThreadPoolExecutor:
    global _extraction_pool
    with _extraction_pool_lock:
        if _extraction_pool is None or _extraction_pool._shutdown:
            _extraction_pool = _cf.ThreadPoolExecutor(
                max_workers=_EXTRACTION_MAX_WORKERS,
                thread_name_prefix="knowledge-extract",
            )
    return _extraction_pool


_DEFAULT_FACTS_SUBDIR = "knowledge/facts.json"
_DEFAULT_FAISS_SUBDIR = "vectordb/knowledge"

_EXTRACT_FACTS_SYSTEM = """Extract durable factual knowledge from this conversation exchange.

Rules:
- Extract ONLY facts that would be useful in future conversations with anyone.
- Facts must be entity-centric: about people, places, organizations, preferences, expertise.
- Do NOT extract: conversation flow, emotional states, specific requests, transient questions, or anything private to this conversation.
- Output a JSON array: [{"entity": "...", "fact": "..."}]
- If no durable facts are present, output: []

Good examples:
  {"entity": "Alice", "fact": "Is a veterinarian in Portland"}
  {"entity": "Project Aurora", "fact": "Uses PostgreSQL for persistence"}

Bad examples (do not extract):
  "Alice asked about the weather" (transient)
  "The user seems frustrated" (emotional state)
  "Alice wants me to write an email" (private request)
"""


@dataclass
class Fact:
    entity: str
    fact: str
    source_session: str
    timestamp: float
    fact_hash: str


def _compute_hash(entity: str, fact: str) -> str:
    normalized = f"{entity.lower().strip()}::{fact.lower().strip()}"
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


class SharedKnowledgeStore:
    """Cross-chat knowledge store — extracts durable facts from conversations
    and recalls them when relevant to other chats."""

    def __init__(self, config: Any, llm: Any, extraction_llm: Any = None) -> None:
        """
        Args:
            config: Full application Config object.
            llm: Main LLM instance (used for extraction if no dedicated model).
            extraction_llm: Optional dedicated LLM for fact extraction.
        """
        self._config = config
        self._llm = llm
        self._extraction_llm = extraction_llm or llm

        asst_cfg: dict[str, Any] = (
            config.services.get("assistant", {}) if hasattr(config, "services") else {}
        )
        know_cfg: dict[str, Any] = asst_cfg.get("knowledge", {})
        self._max_facts: int = int(know_cfg.get("max_facts", 10000))

        knowledge_data_dir = know_cfg.get("data_dir")
        if knowledge_data_dir is not None:
            data_dir = Path(knowledge_data_dir).resolve()
        else:
            top_level = getattr(config, "data_dir", "data")
            data_dir = Path(top_level).resolve()
        self._facts_path: Path = data_dir / _DEFAULT_FACTS_SUBDIR
        self._faiss_index_dir: str = str(data_dir / _DEFAULT_FAISS_SUBDIR)

        self._facts: list[Fact] = []
        self._fact_hashes: set[str] = set()
        self._lock = threading.Lock()
        self._index_lock = threading.Lock()

        self._vectorstore: Any = None
        self._embedding_fn: Any = None
        self._embedding_tag: str | None = None

        self._facts_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

        self._embeddings_ready = threading.Event()
        self._embedding_thread = threading.Thread(
            target=self._setup_embeddings_background,
            name="knowledge-embeddings",
            daemon=True,
        )
        self._embedding_thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_and_store(self, user_input: str, agent_response: str) -> None:
        """Submit fact extraction to the background pool and return immediately."""
        pool = _get_extraction_pool()
        pool.submit(self._extract_and_store_sync, user_input, agent_response)

    def _extract_and_store_sync(self, user_input: str, agent_response: str) -> None:
        """Synchronous fact extraction — runs inside the background pool."""
        try:
            facts = self._extract_facts(user_input, agent_response)
        except Exception as exc:
            log.debug("Fact extraction failed: %s", exc)
            return

        if not facts:
            return

        added: list[Fact] = []
        with self._lock:
            for raw in facts:
                entity = str(raw.get("entity", "")).strip()
                fact_text = str(raw.get("fact", "")).strip()
                if not entity or not fact_text:
                    continue

                fhash = _compute_hash(entity, fact_text)
                if fhash in self._fact_hashes:
                    continue

                if len(self._facts) >= self._max_facts:
                    log.debug("Knowledge store at capacity (%d facts)", self._max_facts)
                    break

                fact = Fact(
                    entity=entity,
                    fact=fact_text,
                    source_session="",
                    timestamp=time.time(),
                    fact_hash=fhash,
                )
                self._facts.append(fact)
                self._fact_hashes.add(fhash)
                added.append(fact)

            if added:
                log.debug("Knowledge store: added %d new fact(s)", len(added))
        if added:
            self._index_facts(added)
            self.save()

    def recall(self, query: str, k: int = 5) -> str | None:
        """Retrieve relevant facts as a formatted string for context injection.

        Returns None if no relevant facts found.
        """
        with self._lock:
            facts_snapshot = list(self._facts)

        if not facts_snapshot:
            return None

        if self._vectorstore is not None and self._embeddings_ready.is_set():
            results = self._recall_semantic(query, k, facts_snapshot)
        else:
            results = self._recall_keyword(query, k, facts_snapshot)

        if not results:
            return None

        lines = [f"- {f.entity}: {f.fact}" for f in results]
        return "\n".join(lines)

    def save(self) -> None:
        """Persist the fact store to disk."""
        with self._lock:
            facts_snapshot = list(self._facts)

        try:
            data = [asdict(f) for f in facts_snapshot]
            tmp_fd, tmp_path = tempfile.mkstemp(dir=str(self._facts_path.parent), suffix=".tmp")
            try:
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, self._facts_path)
            except Exception:
                try:
                    os.close(tmp_fd)
                except OSError:
                    pass
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            log.debug(
                "Knowledge store: saved %d facts to %s",
                len(facts_snapshot),
                self._facts_path,
            )
        except Exception as exc:
            log.warning("Knowledge store: save failed: %s", exc)

        if self._vectorstore is not None and self._embeddings_ready.is_set():
            try:
                idx_dir = Path(self._faiss_index_dir)
                idx_dir.mkdir(parents=True, exist_ok=True)
                self._vectorstore.save_local(str(idx_dir))
                meta = {"embedding_model": self._embedding_tag}
                (idx_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            except Exception as exc:
                log.warning("Knowledge store: FAISS save failed: %s", exc)

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_facts(self, user_input: str, agent_response: str) -> list[dict[str, str]]:
        """Call the extraction LLM and parse the JSON result."""
        from langchain_core.messages import HumanMessage, SystemMessage

        user_content = f"User: {user_input}\n\nAssistant: {agent_response}"
        messages = [
            SystemMessage(content=_EXTRACT_FACTS_SYSTEM),
            HumanMessage(content=user_content),
        ]

        response = self._extraction_llm.invoke(messages)
        raw_text: str = (
            response.content if hasattr(response, "content") else str(response)
        ).strip()

        # Extract JSON array from response (may have surrounding text)
        start = raw_text.find("[")
        end = raw_text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return []

        json_str = raw_text[start : end + 1]
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            log.debug("Knowledge extraction: JSON parse failed for: %s", json_str[:200])
            return []

        if not isinstance(parsed, list):
            return []

        return [item for item in parsed if isinstance(item, dict)]

    # ------------------------------------------------------------------
    # Recall helpers
    # ------------------------------------------------------------------

    def _recall_semantic(self, query: str, k: int, facts_snapshot: list[Fact]) -> list[Fact]:
        """Semantic recall via FAISS."""
        with self._index_lock:
            try:
                results = self._vectorstore.similarity_search(query, k=k)
            except Exception as exc:
                log.debug("FAISS recall failed: %s", exc)
                return self._recall_keyword(query, k, facts_snapshot)

        hash_to_fact = {f.fact_hash: f for f in facts_snapshot}
        matched: list[Fact] = []
        for doc in results:
            fhash = doc.metadata.get("fact_hash")
            if fhash and fhash in hash_to_fact:
                matched.append(hash_to_fact[fhash])

        return matched

    def _recall_keyword(self, query: str, k: int, facts: list[Fact]) -> list[Fact]:
        """Keyword overlap recall — fallback when FAISS is not available."""
        query_tokens = set(query.lower().split())
        if not query_tokens:
            return facts[:k]

        scored: list[tuple[int, Fact]] = []
        for fact in facts:
            text = f"{fact.entity} {fact.fact}".lower()
            overlap = sum(1 for tok in query_tokens if tok in text)
            if overlap > 0:
                scored.append((overlap, fact))

        top = heapq.nlargest(k, scored, key=lambda x: x[0])
        return [f for _, f in top]

    # ------------------------------------------------------------------
    # FAISS index
    # ------------------------------------------------------------------

    def _setup_embeddings_background(self) -> None:
        """Initialise embeddings in a background thread."""
        try:
            self._setup_embeddings()
        except Exception as exc:
            log.debug("Knowledge store: background embedding setup failed: %s", exc)
        finally:
            self._embeddings_ready.set()

    def _setup_embeddings(self) -> None:
        """Initialise the embedding function and load or create a FAISS index."""
        if not hasattr(self._config, "embedding"):
            return

        emb_provider = getattr(self._config.embedding, "provider", "ollama")
        emb_model = getattr(self._config.embedding, "model", None)

        fn: Any = None
        tag: str | None = None

        try:
            if emb_provider == "ollama":
                from langchain_ollama import OllamaEmbeddings

                model_name = emb_model or "nomic-embed-text"
                prov_cfg = None
                try:
                    prov_cfg = self._config.get_provider_config("ollama")
                except (ValueError, AttributeError):
                    pass
                base = (prov_cfg.get_base_url() if prov_cfg else None) or "http://localhost:11434"
                fn = OllamaEmbeddings(model=model_name, base_url=base)
                fn.embed_query("ping")
                tag = f"ollama/{model_name}"

            elif emb_provider == "openai":
                from langchain_openai import OpenAIEmbeddings

                model_name = emb_model or "text-embedding-3-small"
                fn = OpenAIEmbeddings(model=model_name)
                fn.embed_query("ping")
                tag = f"openai/{model_name}"

        except Exception as exc:
            log.debug("Knowledge store: embedding provider '%s' unavailable: %s", emb_provider, exc)

        if fn is None and emb_provider != "ollama":
            try:
                from langchain_ollama import OllamaEmbeddings

                fn = OllamaEmbeddings(model="nomic-embed-text")
                fn.embed_query("ping")
                tag = "ollama/nomic-embed-text"
            except Exception:
                pass

        if fn is None:
            log.debug("Knowledge store: no embedding provider — using keyword recall")
            return

        self._embedding_fn = fn
        self._embedding_tag = tag
        self._load_or_create_index()

    def _load_or_create_index(self) -> None:
        """Load an existing FAISS index or build one from stored facts."""
        idx_dir = Path(self._faiss_index_dir)
        meta_path = idx_dir / "meta.json"

        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}

            if meta.get("embedding_model") == self._embedding_tag:
                try:
                    from langchain_community.vectorstores import FAISS

                    self._vectorstore = FAISS.load_local(
                        str(idx_dir),
                        self._embedding_fn,
                        allow_dangerous_deserialization=True,
                    )
                    log.debug("Knowledge store: loaded FAISS index from %s", idx_dir)
                    return
                except Exception as exc:
                    log.debug("Knowledge store: failed to load FAISS index: %s", exc)
            else:
                log.debug(
                    "Knowledge store: embedding model changed (%s -> %s); rebuilding index",
                    meta.get("embedding_model"),
                    self._embedding_tag,
                )

        # Build from existing facts
        with self._lock:
            facts_snapshot = list(self._facts)

        if facts_snapshot:
            self._index_facts(facts_snapshot)

    def _index_facts(self, facts: list[Fact]) -> None:
        """Add facts to the FAISS index."""
        if self._embedding_fn is None:
            return

        with self._index_lock:
            try:
                from langchain_community.vectorstores import FAISS
                from langchain_core.documents import Document

                docs = [
                    Document(
                        page_content=f"{f.entity}: {f.fact}",
                        metadata={"fact_hash": f.fact_hash},
                    )
                    for f in facts
                ]

                if self._vectorstore is None:
                    self._vectorstore = FAISS.from_documents(docs, self._embedding_fn)
                else:
                    self._vectorstore.add_documents(docs)
            except Exception as exc:
                log.debug("Knowledge store: FAISS indexing failed: %s", exc)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load facts from disk."""
        if not self._facts_path.exists():
            return

        try:
            raw = json.loads(self._facts_path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Knowledge store: failed to load facts: %s", exc)
            return

        if not isinstance(raw, list):
            log.warning("Knowledge store: unexpected format in %s", self._facts_path)
            return

        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                fact = Fact(
                    entity=item["entity"],
                    fact=item["fact"],
                    source_session=item.get("source_session", ""),
                    timestamp=float(item.get("timestamp", 0.0)),
                    fact_hash=item["fact_hash"],
                )
                if fact.fact_hash not in self._fact_hashes:
                    self._facts.append(fact)
                    self._fact_hashes.add(fact.fact_hash)
            except (KeyError, TypeError, ValueError) as exc:
                log.debug("Knowledge store: skipping malformed fact entry: %s", exc)

        log.debug("Knowledge store: loaded %d facts from %s", len(self._facts), self._facts_path)


# ------------------------------------------------------------------
# Factory helper — mirrors _create_compression_llm pattern
# ------------------------------------------------------------------


def create_extraction_llm(model_ref: str, config: Any) -> Any | None:
    """Create a dedicated LLM for fact extraction.

    Resolves *model_ref* against config model_aliases and providers,
    then builds a LangChain LLM.  Returns None on any failure.
    """
    try:
        from src.agent.core import create_llm_from_provider_config

        models = getattr(config, "models", None) or {}
        provider_name: str | None = None
        model_name: str | None = None

        if model_ref in models:
            model_entry = models[model_ref]
            if hasattr(model_entry, "provider"):
                provider_name = model_entry.provider
                model_name = model_entry.model
            elif isinstance(model_entry, dict):
                provider_name = model_entry.get("provider", config.provider)
                model_name = model_entry.get("model")
            elif isinstance(model_entry, str) and "/" in model_entry:
                provider_name, model_name = model_entry.split("/", 1)
            else:
                provider_name = config.provider
                model_name = str(model_entry)
        elif "/" in model_ref:
            provider_name, model_name = model_ref.split("/", 1)
        else:
            provider_name = config.provider
            model_name = model_ref

        prov_cfg = copy(config.get_provider_config(provider_name))
        if model_name:
            prov_cfg.model = model_name

        llm = create_llm_from_provider_config(prov_cfg)
        log.info("Extraction LLM created: %s/%s", provider_name, model_name)
        return llm
    except Exception as exc:
        log.warning("Failed to create extraction LLM '%s': %s", model_ref, exc)
        return None
