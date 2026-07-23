"""Unit tests for the WorkflowRegistry (cogtrix_core/assistant/workflows.py).

Covers: CRUD operations, binding persistence, backward-compatible contact_prompts
fallback, auto-detect binding on first message, path containment enforcement,
resolution order, and thread-safety basics.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from cogtrix_core.assistant.workflows import (
    WorkflowDefinition,
    WorkflowRegistry,
    _load_workflow_yaml,
    _validate_workflow_id,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_workflow_yaml(data_dir: Path, wf_id: str, raw: dict) -> Path:
    """Write a workflow.yaml into the expected directory layout."""
    wf_dir = data_dir / "workflows" / wf_id
    wf_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = wf_dir / "workflow.yaml"
    yaml_path.write_text(yaml.dump(raw, allow_unicode=True), encoding="utf-8")
    return yaml_path


def _make_bike_sales(data_dir: Path) -> dict:
    """Create a minimal bike-sales workflow on disk and return the raw dict."""
    raw = {
        "id": "bike-sales",
        "name": "Bike Sales Assistant",
        "description": "Sells bikes",
        "system_prompt": "You are a bike sales advisor.",
        "knowledge_base": False,
        "tool_policy": {"excluded_tools": ["shell"], "additional_approved_tools": []},
        "auto_detect": {
            "enabled": True,
            "keywords": ["bike", "bicycle"],
            "patterns": [r"\bbike\b"],
            "min_confidence": 1,
        },
    }
    _write_workflow_yaml(data_dir, "bike-sales", raw)
    return raw


# ---------------------------------------------------------------------------
# _validate_workflow_id
# ---------------------------------------------------------------------------


class TestValidateWorkflowId:
    def test_valid_ids(self):
        for wf_id in ["foo", "bike-sales", "support_desk", "A1", "a"]:
            _validate_workflow_id(wf_id)

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="Invalid workflow ID"):
            _validate_workflow_id("")

    def test_rejects_leading_hyphen(self):
        with pytest.raises(ValueError, match="Invalid workflow ID"):
            _validate_workflow_id("-bad")

    def test_rejects_spaces(self):
        with pytest.raises(ValueError, match="Invalid workflow ID"):
            _validate_workflow_id("has space")

    def test_rejects_dots(self):
        with pytest.raises(ValueError, match="Invalid workflow ID"):
            _validate_workflow_id("has.dot")

    def test_rejects_slashes(self):
        with pytest.raises(ValueError, match="Invalid workflow ID"):
            _validate_workflow_id("path/traversal")


# ---------------------------------------------------------------------------
# _load_workflow_yaml
# ---------------------------------------------------------------------------


class TestLoadWorkflowYaml:
    def test_valid_yaml(self, tmp_path: Path):
        raw = {"id": "test", "name": "Test Workflow"}
        _write_workflow_yaml(tmp_path, "test", raw)
        wf = _load_workflow_yaml(tmp_path / "workflows" / "test" / "workflow.yaml")
        assert wf is not None
        assert wf.id == "test"
        assert wf.name == "Test Workflow"

    def test_missing_id_returns_none(self, tmp_path: Path):
        raw = {"name": "No ID"}
        _write_workflow_yaml(tmp_path, "noid", raw)
        wf = _load_workflow_yaml(tmp_path / "workflows" / "noid" / "workflow.yaml")
        assert wf is None

    def test_missing_name_returns_none(self, tmp_path: Path):
        raw = {"id": "noname"}
        _write_workflow_yaml(tmp_path, "noname", raw)
        wf = _load_workflow_yaml(tmp_path / "workflows" / "noname" / "workflow.yaml")
        assert wf is None

    def test_invalid_yaml_returns_none(self, tmp_path: Path):
        wf_dir = tmp_path / "workflows" / "bad"
        wf_dir.mkdir(parents=True)
        (wf_dir / "workflow.yaml").write_text("{{invalid yaml", encoding="utf-8")
        wf = _load_workflow_yaml(wf_dir / "workflow.yaml")
        assert wf is None

    def test_non_dict_yaml_returns_none(self, tmp_path: Path):
        wf_dir = tmp_path / "workflows" / "list"
        wf_dir.mkdir(parents=True)
        (wf_dir / "workflow.yaml").write_text("- item1\n- item2\n", encoding="utf-8")
        wf = _load_workflow_yaml(wf_dir / "workflow.yaml")
        assert wf is None

    def test_tool_policy_parsed(self, tmp_path: Path):
        raw = {
            "id": "tp",
            "name": "Tool Policy Test",
            "tool_policy": {
                "excluded_tools": ["shell", "write_file"],
                "additional_approved_tools": ["read_file"],
            },
        }
        _write_workflow_yaml(tmp_path, "tp", raw)
        wf = _load_workflow_yaml(tmp_path / "workflows" / "tp" / "workflow.yaml")
        assert wf is not None
        assert wf.tool_policy.excluded_tools == ["shell", "write_file"]
        assert wf.tool_policy.additional_approved_tools == ["read_file"]

    def test_auto_detect_parsed(self, tmp_path: Path):
        raw = {
            "id": "ad",
            "name": "Auto Detect Test",
            "auto_detect": {
                "enabled": True,
                "keywords": ["help", "support"],
                "patterns": [r"\bticket\b"],
                "min_confidence": 2,
            },
        }
        _write_workflow_yaml(tmp_path, "ad", raw)
        wf = _load_workflow_yaml(tmp_path / "workflows" / "ad" / "workflow.yaml")
        assert wf is not None
        assert wf.auto_detect.enabled is True
        assert wf.auto_detect.min_confidence == 2
        assert "help" in wf.auto_detect.keywords


# ---------------------------------------------------------------------------
# WorkflowRegistry — construction and reload
# ---------------------------------------------------------------------------


class TestRegistryInit:
    def test_empty_data_dir(self, tmp_path: Path):
        reg = WorkflowRegistry(tmp_path)
        assert reg.list_workflows() == []
        assert reg.list_bindings() == {}

    def test_loads_existing_workflows(self, tmp_path: Path):
        _make_bike_sales(tmp_path)
        reg = WorkflowRegistry(tmp_path)
        wfs = reg.list_workflows()
        assert len(wfs) == 1
        assert wfs[0].id == "bike-sales"

    def test_skips_mismatched_directory_name(self, tmp_path: Path):
        raw = {"id": "real-id", "name": "Mismatch"}
        _write_workflow_yaml(tmp_path, "wrong-dir", raw)
        reg = WorkflowRegistry(tmp_path)
        assert reg.list_workflows() == []

    def test_loads_existing_bindings(self, tmp_path: Path):
        _make_bike_sales(tmp_path)
        bindings_path = tmp_path / "workflows" / "bindings.json"
        bindings_path.write_text(
            json.dumps(
                {
                    "whatsapp::+123": {
                        "workflow_id": "bike-sales",
                        "assigned_at": "2026-01-01T00:00:00",
                        "assigned_by": "manual",
                    }
                }
            ),
            encoding="utf-8",
        )
        reg = WorkflowRegistry(tmp_path)
        bindings = reg.list_bindings()
        assert "whatsapp::+123" in bindings
        assert bindings["whatsapp::+123"]["workflow_id"] == "bike-sales"

    def test_reload_picks_up_new_workflow(self, tmp_path: Path):
        reg = WorkflowRegistry(tmp_path)
        assert len(reg.list_workflows()) == 0
        _make_bike_sales(tmp_path)
        reg.reload()
        assert len(reg.list_workflows()) == 1


# ---------------------------------------------------------------------------
# WorkflowRegistry — CRUD
# ---------------------------------------------------------------------------


class TestRegistryCRUD:
    def test_create_workflow(self, tmp_path: Path):
        reg = WorkflowRegistry(tmp_path)
        wf = WorkflowDefinition(id="new-wf", name="New Workflow")
        reg.create_workflow(wf)
        assert reg.get_workflow("new-wf") is not None
        assert (tmp_path / "workflows" / "new-wf" / "workflow.yaml").exists()

    def test_create_duplicate_raises(self, tmp_path: Path):
        reg = WorkflowRegistry(tmp_path)
        wf = WorkflowDefinition(id="dup", name="First")
        reg.create_workflow(wf)
        with pytest.raises(ValueError, match="already exists"):
            reg.create_workflow(WorkflowDefinition(id="dup", name="Second"))

    def test_create_invalid_id_raises(self, tmp_path: Path):
        reg = WorkflowRegistry(tmp_path)
        with pytest.raises(ValueError, match="Invalid workflow ID"):
            reg.create_workflow(WorkflowDefinition(id="bad/id", name="Bad"))

    def test_update_workflow(self, tmp_path: Path):
        reg = WorkflowRegistry(tmp_path)
        wf = WorkflowDefinition(id="upd", name="Original")
        reg.create_workflow(wf)
        wf.name = "Updated"
        reg.update_workflow(wf)
        updated = reg.get_workflow("upd")
        assert updated is not None
        assert updated.name == "Updated"
        assert updated.updated_at  # timestamp should be set

    def test_update_nonexistent_raises(self, tmp_path: Path):
        reg = WorkflowRegistry(tmp_path)
        with pytest.raises(ValueError, match="not found"):
            reg.update_workflow(WorkflowDefinition(id="ghost", name="Ghost"))

    def test_delete_workflow(self, tmp_path: Path):
        reg = WorkflowRegistry(tmp_path)
        wf = WorkflowDefinition(id="del", name="To Delete")
        reg.create_workflow(wf)
        assert reg.get_workflow("del") is not None
        reg.delete_workflow("del")
        assert reg.get_workflow("del") is None
        assert not (tmp_path / "workflows" / "del").exists()

    def test_delete_removes_associated_bindings(self, tmp_path: Path):
        reg = WorkflowRegistry(tmp_path)
        wf = WorkflowDefinition(id="bound", name="Bound WF")
        reg.create_workflow(wf)
        reg.bind("session::1", "bound")
        reg.bind("session::2", "bound")
        assert len(reg.list_bindings()) == 2
        reg.delete_workflow("bound")
        assert len(reg.list_bindings()) == 0

    def test_delete_nonexistent_raises(self, tmp_path: Path):
        reg = WorkflowRegistry(tmp_path)
        with pytest.raises(ValueError, match="not found"):
            reg.delete_workflow("nope")

    def test_get_workflow_returns_none_for_missing(self, tmp_path: Path):
        reg = WorkflowRegistry(tmp_path)
        assert reg.get_workflow("nonexistent") is None

    def test_create_sets_timestamps(self, tmp_path: Path):
        reg = WorkflowRegistry(tmp_path)
        wf = WorkflowDefinition(id="ts", name="Timestamp Test")
        reg.create_workflow(wf)
        assert wf.created_at != ""
        assert wf.updated_at != ""

    def test_update_preserves_created_at(self, tmp_path: Path):
        reg = WorkflowRegistry(tmp_path)
        wf = WorkflowDefinition(id="ts2", name="Original")
        reg.create_workflow(wf)
        original_created = wf.created_at
        # Create a new definition to simulate fresh input with empty created_at.
        wf2 = WorkflowDefinition(id="ts2", name="Modified", created_at="")
        reg.update_workflow(wf2)
        updated = reg.get_workflow("ts2")
        assert updated is not None
        assert updated.created_at == original_created


# ---------------------------------------------------------------------------
# WorkflowRegistry — bindings
# ---------------------------------------------------------------------------


class TestRegistryBindings:
    def test_bind_and_get(self, tmp_path: Path):
        reg = WorkflowRegistry(tmp_path)
        reg.create_workflow(WorkflowDefinition(id="wf1", name="WF1"))
        reg.bind("chat::abc", "wf1", "manual")
        binding = reg.get_binding("chat::abc")
        assert binding is not None
        assert binding["workflow_id"] == "wf1"
        assert binding["assigned_by"] == "manual"

    def test_bind_to_nonexistent_raises(self, tmp_path: Path):
        reg = WorkflowRegistry(tmp_path)
        with pytest.raises(ValueError, match="not found"):
            reg.bind("chat::abc", "ghost")

    def test_unbind_returns_true(self, tmp_path: Path):
        reg = WorkflowRegistry(tmp_path)
        reg.create_workflow(WorkflowDefinition(id="wf1", name="WF1"))
        reg.bind("chat::abc", "wf1")
        assert reg.unbind("chat::abc") is True
        assert reg.get_binding("chat::abc") is None

    def test_unbind_missing_returns_false(self, tmp_path: Path):
        reg = WorkflowRegistry(tmp_path)
        assert reg.unbind("nonexistent") is False

    def test_binding_persists_to_disk(self, tmp_path: Path):
        reg = WorkflowRegistry(tmp_path)
        reg.create_workflow(WorkflowDefinition(id="persist", name="Persist"))
        reg.bind("chat::x", "persist")
        bindings_file = tmp_path / "workflows" / "bindings.json"
        assert bindings_file.exists()
        raw = json.loads(bindings_file.read_text(encoding="utf-8"))
        assert "chat::x" in raw
        assert raw["chat::x"]["workflow_id"] == "persist"

    def test_binding_survives_reload(self, tmp_path: Path):
        reg = WorkflowRegistry(tmp_path)
        reg.create_workflow(WorkflowDefinition(id="surv", name="Survive"))
        reg.bind("chat::y", "surv")
        reg2 = WorkflowRegistry(tmp_path)
        binding = reg2.get_binding("chat::y")
        assert binding is not None
        assert binding["workflow_id"] == "surv"


# ---------------------------------------------------------------------------
# WorkflowRegistry — resolve()
# ---------------------------------------------------------------------------


class TestRegistryResolve:
    def test_explicit_binding_wins(self, tmp_path: Path):
        _make_bike_sales(tmp_path)
        reg = WorkflowRegistry(tmp_path)
        reg.bind("chat::1", "bike-sales")
        result = reg.resolve("chat::1", msg_text="hello")
        assert result.workflow_id == "bike-sales"
        assert result.system_prompt == "You are a bike sales advisor."
        assert result.assigned_by == "manual"

    def test_contact_prompts_fallback(self, tmp_path: Path):
        reg = WorkflowRegistry(
            tmp_path,
            contact_prompts={"Alice": "You are Alice's assistant."},
            phonebook={"+123": "Alice"},
        )
        result = reg.resolve("chat::alice", msg_text="hi", resolved_phone="+123")
        assert result.system_prompt == "You are Alice's assistant."
        assert result.assigned_by == "contact_prompt"
        assert result.workflow_id is None

    def test_auto_detect_binds_persistently(self, tmp_path: Path):
        _make_bike_sales(tmp_path)
        reg = WorkflowRegistry(tmp_path)
        result = reg.resolve("chat::new", msg_text="I want to buy a bike")
        assert result.workflow_id == "bike-sales"
        assert result.assigned_by == "auto"
        binding = reg.get_binding("chat::new")
        assert binding is not None
        assert binding["workflow_id"] == "bike-sales"

    def test_no_match_returns_all_none(self, tmp_path: Path):
        _make_bike_sales(tmp_path)
        reg = WorkflowRegistry(tmp_path)
        result = reg.resolve("chat::unrelated", msg_text="hello world")
        assert result.workflow_id is None
        assert result.system_prompt is None
        assert result.assigned_by is None

    def test_stale_binding_cleaned_up(self, tmp_path: Path):
        """Binding pointing to a deleted workflow gets cleaned up on resolve."""
        _make_bike_sales(tmp_path)
        reg = WorkflowRegistry(tmp_path)
        reg.bind("chat::stale", "bike-sales")
        reg.delete_workflow("bike-sales")
        # Re-add the binding manually to simulate stale state
        reg._bindings["chat::stale"] = {
            "workflow_id": "bike-sales",
            "assigned_at": "2026-01-01",
            "assigned_by": "manual",
        }
        result = reg.resolve("chat::stale", msg_text="")
        assert result.workflow_id is None
        assert reg.get_binding("chat::stale") is None

    def test_auto_detect_min_confidence(self, tmp_path: Path):
        """Auto-detect requires min_confidence matches."""
        raw = {
            "id": "strict",
            "name": "Strict Detect",
            "auto_detect": {
                "enabled": True,
                "keywords": ["alpha", "beta", "gamma"],
                "min_confidence": 3,
            },
        }
        _write_workflow_yaml(tmp_path, "strict", raw)
        reg = WorkflowRegistry(tmp_path)
        # Only 1 keyword matches — should not trigger
        result = reg.resolve("chat::s1", msg_text="I need alpha")
        assert result.workflow_id is None
        # All 3 match
        result = reg.resolve("chat::s2", msg_text="alpha beta gamma")
        assert result.workflow_id == "strict"

    def test_auto_detect_regex_pattern(self, tmp_path: Path):
        raw = {
            "id": "regex",
            "name": "Regex Test",
            "auto_detect": {
                "enabled": True,
                "keywords": [],
                "patterns": [r"order\s+#\d+"],
                "min_confidence": 1,
            },
        }
        _write_workflow_yaml(tmp_path, "regex", raw)
        reg = WorkflowRegistry(tmp_path)
        result = reg.resolve("chat::r1", msg_text="Check order #12345 status")
        assert result.workflow_id == "regex"

    def test_auto_detect_invalid_regex_skipped(self, tmp_path: Path):
        raw = {
            "id": "badpat",
            "name": "Bad Pattern",
            "auto_detect": {
                "enabled": True,
                "keywords": [],
                "patterns": ["[invalid"],  # broken regex
                "min_confidence": 1,
            },
        }
        _write_workflow_yaml(tmp_path, "badpat", raw)
        reg = WorkflowRegistry(tmp_path)
        result = reg.resolve("chat::bp", msg_text="anything")
        assert result.workflow_id is None

    def test_auto_detect_disabled_ignored(self, tmp_path: Path):
        raw = {
            "id": "disabled",
            "name": "Disabled Detect",
            "auto_detect": {
                "enabled": False,
                "keywords": ["trigger"],
                "min_confidence": 1,
            },
        }
        _write_workflow_yaml(tmp_path, "disabled", raw)
        reg = WorkflowRegistry(tmp_path)
        result = reg.resolve("chat::d1", msg_text="trigger word")
        assert result.workflow_id is None

    def test_resolution_order_binding_over_auto_detect(self, tmp_path: Path):
        """Explicit binding takes priority over auto-detect."""
        raw_support = {
            "id": "support",
            "name": "Support",
            "system_prompt": "Support prompt",
        }
        _write_workflow_yaml(tmp_path, "support", raw_support)
        _make_bike_sales(tmp_path)
        reg = WorkflowRegistry(tmp_path)
        reg.bind("chat::bound", "support")
        # Even though "bike" would auto-detect, binding wins
        result = reg.resolve("chat::bound", msg_text="I want a bike")
        assert result.workflow_id == "support"
        assert result.system_prompt == "Support prompt"

    def test_contact_prompt_over_auto_detect(self, tmp_path: Path):
        """contact_prompts takes priority over auto-detect."""
        _make_bike_sales(tmp_path)
        reg = WorkflowRegistry(
            tmp_path,
            contact_prompts={"Bob": "Bob's prompt"},
            phonebook={"+456": "Bob"},
        )
        result = reg.resolve("chat::bob", msg_text="bike", resolved_phone="+456")
        assert result.assigned_by == "contact_prompt"
        assert result.system_prompt == "Bob's prompt"

    def test_knowledge_base_dir(self, tmp_path: Path):
        raw = {
            "id": "kb",
            "name": "KB Test",
            "knowledge_base": True,
        }
        _write_workflow_yaml(tmp_path, "kb", raw)
        reg = WorkflowRegistry(tmp_path)
        reg.bind("chat::kb", "kb")
        result = reg.resolve("chat::kb")
        assert result.knowledge_base_dir is not None
        assert "kb" in str(result.knowledge_base_dir)
        assert "faiss_index" in str(result.knowledge_base_dir)

    def test_tool_policy_resolved(self, tmp_path: Path):
        _make_bike_sales(tmp_path)
        reg = WorkflowRegistry(tmp_path)
        reg.bind("chat::tp", "bike-sales")
        result = reg.resolve("chat::tp")
        assert result.tool_policy is not None
        assert "shell" in result.tool_policy.excluded_tools


# ---------------------------------------------------------------------------
# Path containment
# ---------------------------------------------------------------------------


class TestPathContainment:
    def test_create_rejects_traversal_id(self, tmp_path: Path):
        reg = WorkflowRegistry(tmp_path)
        with pytest.raises(ValueError):
            reg.create_workflow(WorkflowDefinition(id="../escape", name="Bad"))

    def test_symlink_outside_data_dir_skipped(self, tmp_path: Path):
        """Symlinks pointing outside data_dir are skipped during load."""
        import tempfile

        # Place the target truly outside data_dir (separate temp directory).
        with tempfile.TemporaryDirectory() as outside_str:
            outside = Path(outside_str)
            (outside / "workflow.yaml").write_text(
                yaml.dump({"id": "evil", "name": "Evil"}), encoding="utf-8"
            )
            data_dir = tmp_path / "data"
            data_dir.mkdir()
            wf_dir = data_dir / "workflows"
            wf_dir.mkdir(parents=True)
            link = wf_dir / "evil"
            try:
                link.symlink_to(outside)
            except OSError:
                pytest.skip("Cannot create symlinks on this platform")
            reg = WorkflowRegistry(data_dir)
            assert reg.get_workflow("evil") is None


# ---------------------------------------------------------------------------
# Thread safety (basic smoke)
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_bind_unbind(self, tmp_path: Path):
        """Multiple threads binding/unbinding should not corrupt state."""
        import threading

        reg = WorkflowRegistry(tmp_path)
        reg.create_workflow(WorkflowDefinition(id="conc", name="Concurrent"))
        errors: list[Exception] = []

        def bind_unbind(i: int) -> None:
            try:
                key = f"chat::{i}"
                reg.bind(key, "conc")
                reg.unbind(key)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=bind_unbind, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert not errors, f"Errors during concurrent bind/unbind: {errors}"
