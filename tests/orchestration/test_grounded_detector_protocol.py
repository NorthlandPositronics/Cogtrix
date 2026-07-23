"""Tests for the ``GroundedDetector`` protocol + ``GROUNDED_DETECTORS``
registry (#1964 Item B).

The protocol formalises the call shape every grounding-aware detector
must satisfy.  The registry is the declarative roster of all such
detectors with their grounding-source declarations.  These tests pin:

1. The registry currently lists the three known detectors with the
   correct handler-node names.
2. Each entry's ``detect`` callable accepts the protocol signature
   (``response_content`` positional, ``sources`` keyword-only) and
   returns ``list[str]``.
3. Each entry's ``handler_node`` follows the ``handle_<name>``
   convention so the router can resolve names mechanically.
4. The protocol is ``runtime_checkable`` — ``isinstance(obj,
   GroundedDetector)`` succeeds for the registered callables.
"""

from __future__ import annotations

import inspect

from src.orchestration.verification import (
    GROUNDED_DETECTORS,
    GroundedDetector,
    GroundedSources,
)

# ── Registry composition ───────────────────────────────────────────────


class TestRegistryComposition:
    """The roster as of #1964 Item B lists three detectors.  Adding a
    new one means updating this test in the same PR — that's the
    safety net that catches a registration omission."""

    def test_registry_roster(self) -> None:
        """Pins the registered detector set so a roster-drift in a future
        PR is caught.  Update this list — in the same PR — whenever a
        new grounding-aware detector is registered."""
        names = [spec.name for spec in GROUNDED_DETECTORS]
        assert names == [
            "unverified_entity",
            "unsupported_quote",
            "unsupported_attribution",
            "entity_owner_mismatch",
            "topic_substitution",
        ], (
            "GROUNDED_DETECTORS roster has drifted from the expected "
            "set.  If you added a new grounding-aware detector, update "
            "this test in the same PR."
        )

    def test_handler_node_follows_convention(self) -> None:
        """Every spec must declare ``handler_node="handle_<name>"`` so
        the router can resolve names mechanically — a typo here would
        break recovery routing silently."""
        for spec in GROUNDED_DETECTORS:
            assert spec.handler_node == f"handle_{spec.name}", (
                f"Spec {spec.name!r}: handler_node={spec.handler_node!r} "
                f"violates the ``handle_<name>`` convention."
            )

    def test_no_duplicate_names(self) -> None:
        names = [spec.name for spec in GROUNDED_DETECTORS]
        assert len(names) == len(set(names))

    def test_no_duplicate_callables(self) -> None:
        callables = [spec.detect for spec in GROUNDED_DETECTORS]
        # ``set`` on functions uses identity — duplicate registration
        # would mean the same callable is wired to two handler nodes.
        assert len(callables) == len(set(callables))


# ── Spec semantic invariants ───────────────────────────────────────────


class TestSpecSemantics:
    """Every grounding-aware detector must consume at least one
    grounding source (otherwise it's not a *grounding-aware* detector
    and should live in a different registry).  Also: when
    ``extracts_candidates_from_user_prompt`` is True,
    ``consumes_user_prompt`` for verification must be False — the
    two are mutually exclusive."""

    def test_consumes_at_least_one_grounding_source(self) -> None:
        for spec in GROUNDED_DETECTORS:
            consumes_anything = (
                spec.consumes_tool_results
                or spec.consumes_user_prompt
                or spec.consumes_system_prompt
            )
            assert consumes_anything, (
                f"Spec {spec.name!r} declares no grounding-source "
                f"consumption — that's not a grounding-aware detector."
            )

    def test_candidate_extraction_excludes_user_prompt_from_verification(self) -> None:
        for spec in GROUNDED_DETECTORS:
            if spec.extracts_candidates_from_user_prompt:
                assert not spec.consumes_user_prompt, (
                    f"Spec {spec.name!r}: a detector that extracts "
                    f"candidates FROM the user prompt cannot also "
                    f"VERIFY against the user prompt — every candidate "
                    f"would self-match.  See "
                    f"``detect_unverified_entities`` for the canonical "
                    f"example."
                )


# ── Protocol conformance ───────────────────────────────────────────────


class TestProtocolConformance:
    """Each registered detector callable must satisfy the
    ``GroundedDetector`` protocol — positional ``response_content``,
    keyword-only ``sources``, returns ``list[str]``."""

    def test_callable_signature_has_sources_kwarg(self) -> None:
        for spec in GROUNDED_DETECTORS:
            sig = inspect.signature(spec.detect)
            params = sig.parameters
            assert "sources" in params, (
                f"Spec {spec.name!r}: detector signature must accept a "
                f"``sources`` parameter — required by the GroundedDetector protocol."
            )
            sources_param = params["sources"]
            assert sources_param.kind == inspect.Parameter.KEYWORD_ONLY, (
                f"Spec {spec.name!r}: ``sources`` must be keyword-only "
                f"so callers cannot accidentally pass positional grounding "
                f"args — the bug class that drove this protocol."
            )

    def test_callable_accepts_response_content_positionally(self) -> None:
        """First non-self parameter must be ``response_content`` and accept
        a positional string."""
        for spec in GROUNDED_DETECTORS:
            sig = inspect.signature(spec.detect)
            first = next(iter(sig.parameters.values()))
            assert first.name == "response_content", (
                f"Spec {spec.name!r}: first parameter must be "
                f"``response_content`` (got {first.name!r})."
            )

    def test_runtime_check_passes_for_registered_callables(self) -> None:
        """``GroundedDetector`` is ``runtime_checkable`` — ``isinstance``
        works.  This catches a callable that drifts away from the
        protocol shape at runtime, not just static type-check time."""
        for spec in GROUNDED_DETECTORS:
            assert isinstance(spec.detect, GroundedDetector), (
                f"Spec {spec.name!r}: detector callable does not satisfy "
                f"the runtime GroundedDetector protocol check."
            )


# ── End-to-end: every registered detector actually runs ────────────────


class TestRegisteredDetectorsExecute:
    """Smoke test — calling each registered detector with a
    realistic ``GroundedSources`` does not crash and returns
    ``list[str]``.  Catches signature regressions that the static
    protocol check would miss."""

    def test_every_detector_returns_list_for_empty_sources(self) -> None:
        sources = GroundedSources()
        for spec in GROUNDED_DETECTORS:
            result = spec.detect("trivial response", sources=sources)
            assert isinstance(result, list), (
                f"Spec {spec.name!r}: detector returned {type(result).__name__} "
                f"instead of list[str]."
            )

    def test_every_detector_returns_list_for_grounded_response(self) -> None:
        # A bland response that should not flag any detector.
        sources = GroundedSources(
            tool_results=("Some tool output text.",),
            user_prompt="Hello, agent.",
            system_prompt="You are a helpful assistant.",
        )
        for spec in GROUNDED_DETECTORS:
            result = spec.detect(
                "Hi! How can I help?",  # benign response
                sources=sources,
            )
            assert isinstance(result, list)


# ── Type-only sanity: GroundedDetectorSpec is frozen ───────────────────


class TestSpecImmutability:
    def test_spec_is_frozen(self) -> None:
        spec = GROUNDED_DETECTORS[0]
        try:
            spec.name = "mutated"  # type: ignore[misc]
        except Exception:
            return
        raise AssertionError("GroundedDetectorSpec should be frozen (immutable)")
