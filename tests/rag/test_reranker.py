"""Unit tests for the RAG cross-encoder re-ranker (#1952 Option A).

Most of these tests do NOT install ``sentence-transformers`` — the module
must degrade gracefully without it, returning the input order intact.
The few tests that exercise the loaded-model path inject a stub
``CrossEncoder`` via ``monkeypatch`` so they run without the dep and
without the ~80 MB weight download.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from cogtrix_core.rag.reranker import (
    DEFAULT_RERANK_MODEL,
    _document_text,
    _reset_model_cache_for_tests,
    rerank,
)


@pytest.fixture(autouse=True)
def _reset_cache_between_tests() -> Iterator[None]:
    """Each test starts with a clean module-scope CE model cache.

    Without this, the singleton cache leaks across tests — a test that
    monkeypatches a stub CrossEncoder would poison every subsequent
    test, and the import-failure tests would never re-exercise the
    ``ImportError`` path.
    """
    _reset_model_cache_for_tests()
    yield
    _reset_model_cache_for_tests()


def _doc(content: str) -> SimpleNamespace:
    """Build a minimal document-shaped object the re-ranker accepts.

    The module only requires ``page_content``; we deliberately don't
    use ``langchain_core.documents.Document`` so the unit tests stay
    independent of LangChain version drift.
    """
    return SimpleNamespace(page_content=content)


class TestDocumentText:
    def test_extracts_page_content_when_string(self) -> None:
        assert _document_text(_doc("hello world")) == "hello world"

    def test_empty_when_missing_attribute(self) -> None:
        # Anything without ``page_content`` coerces to its ``str()`` form
        # rather than raising — the CE forward pass must always receive
        # a valid pair shape.
        class Bare:
            pass

        out = _document_text(Bare())
        # The class repr is acceptable — what matters is non-raising
        # behaviour and a string return.
        assert isinstance(out, str)

    def test_empty_when_content_is_none(self) -> None:
        assert _document_text(SimpleNamespace(page_content=None)) == ""

    def test_coerces_non_string_content(self) -> None:
        # Non-string content (list, dict, etc.) coerces via str().
        result = _document_text(SimpleNamespace(page_content=["a", "b"]))
        assert isinstance(result, str)
        assert "a" in result and "b" in result


class TestNoOpPaths:
    """``rerank`` short-circuits without loading the model in trivial cases."""

    def test_returns_empty_list_when_k_is_zero(self) -> None:
        assert rerank("any query", [_doc("a"), _doc("b")], k=0) == []

    def test_returns_empty_list_when_k_is_negative(self) -> None:
        assert rerank("any query", [_doc("a")], k=-3) == []

    def test_returns_empty_list_for_empty_input(self) -> None:
        assert rerank("any query", [], k=4) == []

    def test_short_circuits_on_empty_query(self) -> None:
        # No model load — single-element pass-through.
        docs = [_doc("a"), _doc("b"), _doc("c")]
        assert rerank("", docs, k=2) == docs[:2]

    def test_short_circuits_on_whitespace_query(self) -> None:
        docs = [_doc("a"), _doc("b")]
        assert rerank("   ", docs, k=2) == docs[:2]

    def test_short_circuits_on_single_candidate(self) -> None:
        # No re-rank needed for one candidate.
        only = _doc("just one")
        assert rerank("anything", [only], k=5) == [only]


class TestGracefulFallbackWhenSentenceTransformersMissing:
    """The library is optional — without it the re-ranker MUST return
    the input order intact rather than raising.  This is the documented
    contract: "re-ranking never makes retrieval worse than the baseline".
    """

    def test_falls_back_when_import_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Force ``import sentence_transformers`` to raise ImportError
        # by removing the module from sys.modules and any installed
        # version from the import path.
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)

        docs = [_doc("apple"), _doc("banana"), _doc("cherry")]
        result = rerank("fruit basket", docs, k=2)

        # Input order preserved, sliced to k.
        assert result == docs[:2]


class TestRerankWithStubbedModel:
    """Drive the loaded-model path with an injected stub CrossEncoder so
    we can exercise the scoring and ordering logic without installing
    sentence-transformers or downloading model weights.
    """

    @staticmethod
    def _install_stub_cross_encoder(
        monkeypatch: pytest.MonkeyPatch,
        score_by_text: dict[str, float],
        *,
        raise_on_predict: Exception | None = None,
    ) -> dict[str, Any]:
        """Install a fake ``sentence_transformers`` module with a
        controllable ``CrossEncoder``.  Returns a record dict that
        captures construction args + predict call payloads so tests
        can assert on them.
        """
        record: dict[str, Any] = {
            "init_calls": [],
            "predict_calls": [],
        }

        class _StubCE:
            def __init__(self, model_name: str, device: str | None = None) -> None:
                record["init_calls"].append({"model_name": model_name, "device": device})

            def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
                record["predict_calls"].append(list(pairs))
                if raise_on_predict is not None:
                    raise raise_on_predict
                return [score_by_text.get(text, 0.0) for _q, text in pairs]

        stub_module = SimpleNamespace(CrossEncoder=_StubCE)
        monkeypatch.setitem(sys.modules, "sentence_transformers", stub_module)
        return record

    def test_orders_by_descending_ce_score(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._install_stub_cross_encoder(
            monkeypatch,
            score_by_text={"low": -1.0, "mid": 0.5, "high": 2.3},
        )

        # Input order is INTENTIONALLY wrong so the test fails if the CE
        # pass is silently a no-op.
        docs = [_doc("low"), _doc("high"), _doc("mid")]
        result = rerank("any", docs, k=3)

        contents = [d.page_content for d in result]
        assert contents == ["high", "mid", "low"], (
            "Cross-encoder should reorder by descending score; " f"got {contents}"
        )

    def test_truncates_to_k(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._install_stub_cross_encoder(
            monkeypatch,
            score_by_text={"a": 5.0, "b": 4.0, "c": 3.0, "d": 2.0, "e": 1.0},
        )
        docs = [_doc(t) for t in ["e", "d", "c", "b", "a"]]

        result = rerank("any", docs, k=2)

        assert [d.page_content for d in result] == ["a", "b"]

    def test_falls_back_on_predict_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Predict raises — caller should still get a valid response.
        self._install_stub_cross_encoder(
            monkeypatch,
            score_by_text={},
            raise_on_predict=RuntimeError("torch crashed"),
        )

        docs = [_doc("first"), _doc("second"), _doc("third")]
        result = rerank("any", docs, k=2)

        # Input order preserved, sliced to k.
        assert result == docs[:2]

    def test_falls_back_when_predict_returns_wrong_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A buggy CE could return fewer / more scores than pairs.
        Defensive code falls back rather than mis-pairing.
        """
        record: dict[str, Any] = {}

        class _BadCE:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                record["constructed"] = True

            def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
                # Return one fewer score than expected.
                return [1.0] * (len(pairs) - 1)

        monkeypatch.setitem(
            sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=_BadCE)
        )

        docs = [_doc("a"), _doc("b"), _doc("c")]
        result = rerank("any", docs, k=2)

        assert record.get("constructed") is True
        assert result == docs[:2]

    def test_falls_back_when_predict_returns_non_numeric(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A buggy CE could return non-numeric values.  We must not
        propagate the TypeError out of ``rerank``.
        """

        class _NonNumericCE:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                pass

            def predict(self, pairs: list[tuple[str, str]]) -> list[Any]:
                # Strings that cannot be coerced via float().
                return ["nope"] * len(pairs)

        monkeypatch.setitem(
            sys.modules,
            "sentence_transformers",
            SimpleNamespace(CrossEncoder=_NonNumericCE),
        )

        docs = [_doc("a"), _doc("b")]
        result = rerank("any", docs, k=2)

        assert result == docs[:2]

    def test_falls_back_when_model_load_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A CrossEncoder constructor that raises (model not found,
        network blocked, etc.) must not propagate out of ``rerank``.
        """

        class _BrokenCE:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                raise OSError("Cannot download model")

        monkeypatch.setitem(
            sys.modules,
            "sentence_transformers",
            SimpleNamespace(CrossEncoder=_BrokenCE),
        )

        docs = [_doc("alpha"), _doc("beta")]
        result = rerank("any", docs, k=2)

        # Fallback — input order, sliced to k.
        assert result == docs[:2]

    def test_passes_default_model_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        record = self._install_stub_cross_encoder(monkeypatch, score_by_text={})

        rerank("any", [_doc("a"), _doc("b")], k=2)

        assert record["init_calls"], "CrossEncoder was not constructed"
        assert record["init_calls"][0]["model_name"] == DEFAULT_RERANK_MODEL

    def test_passes_custom_model_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        record = self._install_stub_cross_encoder(monkeypatch, score_by_text={})

        rerank(
            "any",
            [_doc("a"), _doc("b")],
            k=2,
            model_name="some-other/cross-encoder",
        )

        assert record["init_calls"][0]["model_name"] == "some-other/cross-encoder"

    def test_passes_device_when_supplied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        record = self._install_stub_cross_encoder(monkeypatch, score_by_text={})

        rerank("any", [_doc("a"), _doc("b")], k=2, device="cpu")

        assert record["init_calls"][0]["device"] == "cpu"

    def test_caches_model_load_within_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Model load is a singleton — two reranks against the same
        ``(model_name, device)`` pair share one CrossEncoder instance.
        """
        record = self._install_stub_cross_encoder(monkeypatch, score_by_text={"a": 1.0, "b": 0.5})

        rerank("any", [_doc("a"), _doc("b")], k=2)
        rerank("any", [_doc("a"), _doc("b")], k=2)

        assert len(record["init_calls"]) == 1, (
            "CE model should load exactly once per process per (model, device) — "
            f"got {len(record['init_calls'])} constructions"
        )

    def test_sends_query_in_every_pair(self, monkeypatch: pytest.MonkeyPatch) -> None:
        record = self._install_stub_cross_encoder(monkeypatch, score_by_text={})

        rerank("the user's actual question", [_doc("a"), _doc("b"), _doc("c")], k=3)

        sent_pairs = record["predict_calls"][0]
        assert all(pair[0] == "the user's actual question" for pair in sent_pairs)
        assert [p[1] for p in sent_pairs] == ["a", "b", "c"]
