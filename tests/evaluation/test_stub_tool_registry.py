"""Policy tests for the Gate 2 stub tool registry.

Enforces the C1-C7 guardrails from the stub-tool refactor plan so future
changes to ``stub_tool_registry`` cannot silently drift back toward the
"description-only" pathology that DeepSeek-V3 originally exposed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, get_args

import pytest
import yaml
from pydantic import BaseModel

from tests.evaluation.stub_tool_registry import (
    STUB_TOOL_REGISTRY,
    StubToolSpec,
)

_SCENARIOS_DIR = Path(__file__).parent / "scenarios"


def _all_smoke_scenarios() -> list[dict]:
    out: list[dict] = []
    for yml in sorted(_SCENARIOS_DIR.rglob("*.yaml")):
        with open(yml) as f:
            data = yaml.safe_load(f)
        if data is None:
            continue
        tags = data.get("tags") or []
        if "smoke" in tags or not tags:
            out.append(data)
    return out


def _required_field_names(schema: type[BaseModel]) -> list[str]:
    return [name for name, info in schema.model_fields.items() if info.is_required()]


def _all_success_criteria_substrings() -> set[str]:
    """Substrings that appear positively in any smoke scenario's success_criteria.

    Used by C1: a stub return must not contain any of these *except* values
    the agent passed in (which we'll verify field-by-field at runtime).
    """
    subs: set[str] = set()
    for scen in _all_smoke_scenarios():
        for crit in scen.get("success_criteria") or []:
            for prefix in ("contains:", "response_contains:"):
                if crit.startswith(prefix):
                    subs.add(crit[len(prefix) :].strip().lower())
    return subs


# ── C4: extra="forbid" on every schema ────────────────────────────────────────


@pytest.mark.parametrize("name,spec", list(STUB_TOOL_REGISTRY.items()))
def test_schema_forbids_extra_fields(name: str, spec: StubToolSpec) -> None:
    schema_cls = spec.input_schema
    config = getattr(schema_cls, "model_config", {})
    extra = config.get("extra") if isinstance(config, dict) else getattr(config, "extra", None)
    assert extra == "forbid", (
        f"{name}.input_schema must declare model_config = ConfigDict(extra='forbid') "
        f"to catch hallucinated tool fields; got extra={extra!r}"
    )


# ── C5: ≤ 2 required fields per schema ───────────────────────────────────────


@pytest.mark.parametrize("name,spec", list(STUB_TOOL_REGISTRY.items()))
def test_schema_has_at_most_two_required_fields(name: str, spec: StubToolSpec) -> None:
    required = _required_field_names(spec.input_schema)
    assert len(required) <= 2, (
        f"{name} has {len(required)} required fields ({required}); the policy "
        f"caps required fields at 2 so partial human prompts don't trip pydantic "
        f"before the agent can call the tool. Make extra fields Optional[T] = None."
    )


# ── C6: notes escape hatch on tools that should accept free text ─────────────
#
# Tools listed below are exempt because their semantics genuinely take no
# free-form context (e.g. get_current_datetime, request_tools where the
# list is the entire input).  Everything else must include a notes field.


_NOTES_EXEMPT: set[str] = set()  # adjust if a clearly note-free tool joins the registry


@pytest.mark.parametrize("name,spec", list(STUB_TOOL_REGISTRY.items()))
def test_schema_has_notes_escape_hatch(name: str, spec: StubToolSpec) -> None:
    if name in _NOTES_EXEMPT:
        pytest.skip(f"{name} explicitly exempt from notes escape hatch")
    fields = spec.input_schema.model_fields
    assert "notes" in fields, (
        f"{name} must include `notes: Optional[str] = None` so the agent can "
        f"surface off-schema context (e.g. payment terms) without inventing fields."
    )
    info = fields["notes"]
    assert not info.is_required(), f"{name}.notes must be optional"


# ── C2: no directive language in return templates ────────────────────────────


_DIRECTIVE_WORDS = ("next", "then", "now call", "you should", "must call", "please call")


@pytest.mark.parametrize("name,spec", list(STUB_TOOL_REGISTRY.items()))
def test_return_template_has_no_directive_language(name: str, spec: StubToolSpec) -> None:
    """A return value must not tell the agent what to do next."""
    # Synthesize a maximal-fields instance to exercise every echo path
    sample = _make_sample_instance(spec.input_schema)
    rendered = str(spec.return_template(sample)).lower()
    for w in _DIRECTIVE_WORDS:
        assert w not in rendered, f"{name} return contains directive phrase {w!r}: {rendered!r}"


# ── C3 + C7: returns echo inputs only; omitted fields stay omitted ────────────


@pytest.mark.parametrize("name,spec", list(STUB_TOOL_REGISTRY.items()))
def test_return_template_omits_unprovided_optional_fields(name: str, spec: StubToolSpec) -> None:
    """If the agent didn't pass an optional field, the return doesn't fabricate it."""
    schema = spec.input_schema
    required = _required_field_names(schema)
    # Build a minimal-required-only instance — every optional field stays default.
    minimal_kwargs = {f: _placeholder_for(schema.model_fields[f]) for f in required}
    inst = schema(**minimal_kwargs)
    out = spec.return_template(inst)
    for field_name in schema.model_fields:
        if field_name in required:
            continue
        # Optional field was NOT provided by the agent — it MUST NOT appear
        # in the return with a fabricated value.  Empty-list / empty-dict
        # *primary outputs* (e.g. search_web.results=[]) are not "fabricated
        # input echo"; they are the tool's own output.  We only enforce
        # absence for the echoed input names.
        assert field_name not in out, (
            f"{name} fabricates {field_name!r} when the agent didn't pass it; "
            f"return template must echo only provided inputs"
        )


# ── C1: return does not leak success_criteria substrings other than echoes ───


@pytest.mark.parametrize("name,spec", list(STUB_TOOL_REGISTRY.items()))
def test_return_template_does_not_leak_success_criteria_substring(
    name: str, spec: StubToolSpec
) -> None:
    """If the agent passes no input fields, the return must not by itself
    introduce any positive success_criteria substring across smoke scenarios.

    Substrings the agent DID pass (echoed back) are excluded from this check
    by construction — we build an instance with placeholder values that don't
    overlap real criteria.
    """
    required = _required_field_names(spec.input_schema)
    inst = spec.input_schema(
        **{f: _placeholder_for(spec.input_schema.model_fields[f]) for f in required}
    )
    rendered = str(spec.return_template(inst)).lower()
    for substring in _all_success_criteria_substrings():
        # Tool names appearing in returns is fine (they get into the haystack
        # via tool_call name anyway); skip them.
        if substring in STUB_TOOL_REGISTRY:
            continue
        # Common short tokens that legitimately appear in any structured
        # output (e.g. "error" appears in `errors: []` for validate_supplier_data).
        # We compare on whole-token boundaries to avoid false positives.
        if re.search(rf"\b{re.escape(substring)}\b", rendered):
            placeholder_values = [
                str(_placeholder_for(spec.input_schema.model_fields[f])) for f in required
            ]
            if any(substring in p.lower() for p in placeholder_values):
                continue
            pytest.fail(
                f"{name} return leaks success_criteria substring "
                f"{substring!r}: rendered={rendered!r}"
            )


# ── Status field present ──────────────────────────────────────────────────────


@pytest.mark.parametrize("name,spec", list(STUB_TOOL_REGISTRY.items()))
def test_return_template_has_status_field(name: str, spec: StubToolSpec) -> None:
    required = _required_field_names(spec.input_schema)
    inst = spec.input_schema(
        **{f: _placeholder_for(spec.input_schema.model_fields[f]) for f in required}
    )
    out = spec.return_template(inst)
    assert "status" in out, f"{name} return must include a 'status' field"
    assert isinstance(out["status"], str), f"{name}.status must be a string"


# ── Coverage: every smoke scenario's tools are in the registry ───────────────


def test_every_smoke_scenario_tool_is_registered() -> None:
    missing: dict[str, list[str]] = {}
    for scen in _all_smoke_scenarios():
        names: Iterable[str] = list(scen.get("tools_required") or []) + list(
            scen.get("tools_available") or []
        )
        gaps = [n for n in names if n not in STUB_TOOL_REGISTRY]
        if gaps:
            missing[scen["id"]] = gaps
    assert not missing, (
        f"Smoke scenarios reference tools not in stub_tool_registry: {missing}. "
        f"Add a StubToolSpec for each missing tool."
    )


# ── helpers ───────────────────────────────────────────────────────────────────


def _placeholder_for(field_info: Any) -> object:
    """Synthesise a benign value for the field's annotation.

    Unwraps ``Optional[T]`` / ``Union[T, None]`` to the inner type before
    matching, so an optional float field returns ``1.0`` not ``"x"``.
    """
    ann = field_info.annotation
    args = get_args(ann)
    # Only unwrap Optional / Union[..., None] — not list[str] or other generics
    if args and type(None) in args:
        non_none = tuple(a for a in args if a is not type(None))
        if len(non_none) == 1:
            ann = non_none[0]
            args = get_args(ann)
    if ann is float:
        return 1.0
    if ann is bool:
        return False
    if ann is int:
        return 1
    if getattr(ann, "__origin__", None) is list:
        return []
    return "x"


def _make_sample_instance(schema: type[BaseModel]) -> BaseModel:
    """Instantiate the schema populating every field with a benign value."""
    kwargs = {name: _placeholder_for(info) for name, info in schema.model_fields.items()}
    return schema(**kwargs)
