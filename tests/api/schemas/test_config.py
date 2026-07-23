"""Tests for cogtrix_core/api/schemas/config.py — providers, models, config view/edit, wizard."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cogtrix_core.api.schemas.config import (
    ConfigOut,
    ConfigPatchRequest,
    ConfigReloadResponse,
    ModelOut,
    ModelSwitchRequest,
    ProviderCreateRequest,
    ProviderHealthOut,
    ProviderOut,
    ProviderPatchRequest,
    WizardStartRequest,
    WizardStepOut,
    WizardStepRequest,
)

# ---------------------------------------------------------------------------
# ProviderOut — has_api_key is the only required bool; key value never returned
# ---------------------------------------------------------------------------


class TestProviderOut:
    def test_valid_minimal(self) -> None:
        p = ProviderOut(name="openai", type="openai", has_api_key=True)
        assert p.name == "openai"
        assert p.base_url is None
        assert p.has_api_key is True

    def test_valid_with_base_url(self) -> None:
        p = ProviderOut(
            name="my-openai",
            type="openai",
            base_url="https://api.openai.com/v1",
            has_api_key=False,
        )
        assert p.base_url == "https://api.openai.com/v1"

    def test_missing_required_field(self) -> None:
        with pytest.raises(ValidationError):
            ProviderOut(name="x", type="openai")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# ProviderCreateRequest — name pattern + bounds, api_key optional for Ollama
# ---------------------------------------------------------------------------


class TestProviderCreateRequest:
    def test_valid_minimal(self) -> None:
        req = ProviderCreateRequest(name="my-openai", type="openai")
        assert req.name == "my-openai"
        assert req.base_url is None
        assert req.api_key is None  # acceptable for Ollama

    def test_valid_full(self) -> None:
        req = ProviderCreateRequest(
            name="my-openai",
            type="openai",
            base_url="https://api.openai.com/v1",
            api_key="sk-...",
        )
        assert req.api_key == "sk-..."

    def test_name_at_min_length_1(self) -> None:
        assert ProviderCreateRequest(name="a", type="openai").name == "a"

    def test_name_at_max_length_64(self) -> None:
        assert ProviderCreateRequest(name="a" * 64, type="openai").name == "a" * 64

    def test_name_too_short_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ProviderCreateRequest(name="", type="openai")

    def test_name_too_long_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at most 64"):
            ProviderCreateRequest(name="a" * 65, type="openai")

    def test_name_pattern_alphanumeric_first_char(self) -> None:
        # The pattern requires the first char to be alphanumeric.
        for valid in ("a", "A", "1", "abc", "abc-1", "a_b", "abc-123_xyz"):
            assert ProviderCreateRequest(name=valid, type="openai").name == valid

    def test_name_leading_hyphen_rejected(self) -> None:
        with pytest.raises(ValidationError, match="pattern"):
            ProviderCreateRequest(name="-bad", type="openai")

    def test_name_leading_underscore_rejected(self) -> None:
        with pytest.raises(ValidationError, match="pattern"):
            ProviderCreateRequest(name="_bad", type="openai")

    def test_name_with_special_char_rejected(self) -> None:
        with pytest.raises(ValidationError, match="pattern"):
            ProviderCreateRequest(name="bad!name", type="openai")

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            ProviderCreateRequest(type="openai")  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            ProviderCreateRequest(name="x")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# ProviderPatchRequest — all optional
# ---------------------------------------------------------------------------


class TestProviderPatchRequest:
    def test_empty_patch(self) -> None:
        p = ProviderPatchRequest()
        assert p.base_url is None
        assert p.api_key is None

    def test_partial_patch(self) -> None:
        p = ProviderPatchRequest(base_url="https://new.example.com")
        assert p.base_url == "https://new.example.com"
        assert p.api_key is None

    def test_empty_string_api_key_to_clear(self) -> None:
        """Empty-string sentinel for clearing the key (per docstring)."""
        p = ProviderPatchRequest(api_key="")
        assert p.api_key == ""


# ---------------------------------------------------------------------------
# ProviderHealthOut
# ---------------------------------------------------------------------------


class TestProviderHealthOut:
    def test_reachable_ok(self) -> None:
        h = ProviderHealthOut(name="openai", reachable=True, latency_ms=142)
        assert h.latency_ms == 142
        assert h.error is None

    def test_unreachable_with_error(self) -> None:
        h = ProviderHealthOut(name="openai", reachable=False, error="connection timeout")
        assert h.latency_ms is None
        assert h.error == "connection timeout"

    def test_missing_required_field(self) -> None:
        with pytest.raises(ValidationError):
            ProviderHealthOut(name="openai")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# ModelOut
# ---------------------------------------------------------------------------


class TestModelOut:
    def test_valid_minimal(self) -> None:
        m = ModelOut(
            alias="gpt-4.1-mini",
            provider="openai",
            model_name="gpt-4.1-mini",
            is_active=True,
        )
        assert m.num_ctx is None
        assert m.temperature is None
        assert m.max_tokens is None

    def test_valid_full(self) -> None:
        m = ModelOut(
            alias="gpt-4.1-mini",
            provider="openai",
            model_name="gpt-4.1-mini",
            num_ctx=131072,
            temperature=0.7,
            max_tokens=4096,
            is_active=True,
        )
        assert m.num_ctx == 131072
        assert m.temperature == 0.7
        assert m.max_tokens == 4096

    def test_missing_required_field(self) -> None:
        with pytest.raises(ValidationError):
            ModelOut(provider="openai", model_name="x", is_active=True)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# ModelSwitchRequest
# ---------------------------------------------------------------------------


class TestModelSwitchRequest:
    def test_valid(self) -> None:
        assert ModelSwitchRequest(model="gpt-4.1-mini").model == "gpt-4.1-mini"

    def test_provider_slash_model_form(self) -> None:
        assert ModelSwitchRequest(model="openai/gpt-4o").model == "openai/gpt-4o"

    def test_missing_required(self) -> None:
        with pytest.raises(ValidationError):
            ModelSwitchRequest()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# ConfigOut — snapshot with all-optional nested lists + defaults
# ---------------------------------------------------------------------------


class TestConfigOut:
    def test_valid_minimal(self) -> None:
        c = ConfigOut(
            memory_mode="conversation",
            prompt_optimizer=True,
            parallel_tool_execution=True,
            context_compression=True,
            debug=False,
            verbose=False,
        )
        assert c.providers == []
        assert c.models == []
        assert c.delegate_enabled is True  # default

    def test_valid_with_providers_and_models(self) -> None:
        c = ConfigOut(
            memory_mode="conversation",
            prompt_optimizer=True,
            parallel_tool_execution=True,
            context_compression=True,
            debug=False,
            verbose=False,
            providers=[
                ProviderOut(name="openai", type="openai", has_api_key=True),
            ],
            models=[
                ModelOut(alias="gpt", provider="openai", model_name="gpt-4", is_active=True),
            ],
        )
        assert len(c.providers) == 1
        assert len(c.models) == 1

    def test_raw_yaml_admin_only_field(self) -> None:
        """raw_yaml carries the unredacted YAML — admin-only by API contract.
        The schema allows it as Optional; auth enforcement happens at the
        route layer."""
        c = ConfigOut(
            memory_mode="conversation",
            prompt_optimizer=False,
            parallel_tool_execution=False,
            context_compression=False,
            debug=False,
            verbose=False,
            raw_yaml="providers:\n  - name: openai\n",
        )
        assert c.raw_yaml is not None and "providers:" in c.raw_yaml

    def test_delegate_enabled_can_be_disabled(self) -> None:
        c = ConfigOut(
            memory_mode="conversation",
            prompt_optimizer=False,
            parallel_tool_execution=False,
            context_compression=False,
            debug=False,
            verbose=False,
            delegate_enabled=False,
        )
        assert c.delegate_enabled is False

    def test_missing_required_bool(self) -> None:
        with pytest.raises(ValidationError):
            ConfigOut(  # type: ignore[call-arg]
                memory_mode="conversation",
                prompt_optimizer=True,
                parallel_tool_execution=True,
                context_compression=True,
                debug=False,
                # verbose missing
            )


# ---------------------------------------------------------------------------
# ConfigPatchRequest — all-optional bool toggles
# ---------------------------------------------------------------------------


class TestConfigPatchRequest:
    def test_empty_patch(self) -> None:
        p = ConfigPatchRequest()
        assert p.debug is None
        assert p.verbose is None
        assert p.prompt_optimizer is None
        assert p.parallel_tool_execution is None
        assert p.context_compression is None

    def test_partial_patch(self) -> None:
        p = ConfigPatchRequest(debug=True, verbose=False)
        assert p.debug is True
        assert p.verbose is False

    def test_invalid_type_rejected(self) -> None:
        # Pydantic v2 coerces truthy strings like "yes"/"true" into bool,
        # so the test needs a value that genuinely cannot coerce.
        with pytest.raises(ValidationError):
            ConfigPatchRequest(debug={"nested": True})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ConfigReloadResponse
# ---------------------------------------------------------------------------


class TestConfigReloadResponse:
    def test_success_no_warnings(self) -> None:
        r = ConfigReloadResponse(reloaded=True, config_file_path="/home/x/.cogtrix.yaml")
        assert r.reloaded is True
        assert r.warnings == []

    def test_success_with_warnings(self) -> None:
        r = ConfigReloadResponse(reloaded=True, warnings=["deprecated key: foo"])
        assert r.warnings == ["deprecated key: foo"]

    def test_failure(self) -> None:
        r = ConfigReloadResponse(reloaded=False, warnings=["parse error"])
        assert r.reloaded is False
        assert r.config_file_path is None


# ---------------------------------------------------------------------------
# Wizard schemas
# ---------------------------------------------------------------------------


class TestWizardStartRequest:
    def test_defaults(self) -> None:
        r = WizardStartRequest()
        assert r.docs_url is None
        assert r.edit_existing is False

    def test_edit_existing_true(self) -> None:
        assert WizardStartRequest(edit_existing=True).edit_existing is True

    def test_with_docs_url(self) -> None:
        r = WizardStartRequest(docs_url="https://docs.example.com/cogtrix")
        assert r.docs_url == "https://docs.example.com/cogtrix"


class TestWizardStepRequest:
    def test_empty(self) -> None:
        r = WizardStepRequest()
        assert r.answer is None
        assert r.data is None

    def test_with_answer(self) -> None:
        r = WizardStepRequest(answer="yes")
        assert r.answer == "yes"

    def test_with_data(self) -> None:
        r = WizardStepRequest(data={"provider_type": "openai"})
        assert r.data == {"provider_type": "openai"}


class TestWizardStepOut:
    def test_valid_intermediate_step(self) -> None:
        s = WizardStepOut(
            wizard_id="wiz-1",
            step=1,
            total_steps=3,
            step_name="Connect to LLM",
            question="Which provider?",
            complete=False,
        )
        assert s.complete is False
        assert s.requires_acceptance is False
        assert s.warnings == []

    def test_final_step_with_yaml_preview(self) -> None:
        s = WizardStepOut(
            wizard_id="wiz-1",
            step=3,
            total_steps=3,
            step_name="Save",
            yaml_preview="providers:\n  - name: openai\n",
            complete=True,
            requires_acceptance=True,
        )
        assert s.yaml_preview is not None
        assert s.complete is True
        assert s.requires_acceptance is True

    def test_with_warnings(self) -> None:
        s = WizardStepOut(
            wizard_id="wiz-1",
            step=1,
            total_steps=2,
            step_name="step",
            complete=False,
            warnings=["api key looks invalid"],
        )
        assert s.warnings == ["api key looks invalid"]

    def test_missing_required_field(self) -> None:
        with pytest.raises(ValidationError):
            WizardStepOut(  # type: ignore[call-arg]
                step=1, total_steps=3, step_name="x", complete=False
            )
