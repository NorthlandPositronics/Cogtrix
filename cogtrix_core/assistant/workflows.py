"""Workflow registry for Cogtrix assistant mode.

Workflows bundle a system prompt, per-workflow FAISS knowledge base, and tool
policy into a named, reusable unit that can be bound to a chat session.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from cogtrix_core.utils.atomic_write import atomic_write_json

log = logging.getLogger("cogtrix")

_WORKFLOW_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class WorkflowToolPolicy:
    excluded_tools: list[str] = field(default_factory=list)
    additional_approved_tools: list[str] = field(default_factory=list)


@dataclass
class WorkflowAutoDetect:
    enabled: bool = False
    keywords: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    min_confidence: int = 1


@dataclass
class WorkflowDefinition:
    id: str
    name: str
    description: str = ""
    system_prompt: str | None = None
    system_prompt_file: str | None = None
    knowledge_base: bool = False
    tool_policy: WorkflowToolPolicy = field(default_factory=WorkflowToolPolicy)
    auto_detect: WorkflowAutoDetect = field(default_factory=WorkflowAutoDetect)
    created_at: str = ""
    updated_at: str = ""


@dataclass
class ResolvedWorkflow:
    workflow_id: str | None
    system_prompt: str | None
    knowledge_base_dir: Path | None
    tool_policy: WorkflowToolPolicy | None
    assigned_by: str | None  # "manual", "auto", "contact_prompt", or None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_workflow_id(workflow_id: str) -> None:
    if not _WORKFLOW_ID_RE.match(workflow_id):
        raise ValueError(
            f"Invalid workflow ID {workflow_id!r}: must match ^[a-zA-Z0-9][a-zA-Z0-9_-]*$"
        )


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _load_workflow_yaml(path: Path) -> WorkflowDefinition | None:
    """Parse a workflow.yaml file. Returns None and logs a warning on any error."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        log.warning("Failed to load workflow YAML at %s: %s", path, exc)
        return None

    if not isinstance(raw, dict):
        log.warning("Workflow YAML at %s is not a mapping — skipped", path)
        return None

    wf_id = raw.get("id", "")
    if not wf_id or not _WORKFLOW_ID_RE.match(str(wf_id)):
        log.warning("Workflow at %s has invalid or missing id %r — skipped", path, wf_id)
        return None

    name = raw.get("name", "")
    if not name:
        log.warning("Workflow at %s has no name — skipped", path)
        return None

    tp_raw = raw.get("tool_policy") or {}
    tool_policy = WorkflowToolPolicy(
        excluded_tools=list(tp_raw.get("excluded_tools") or []),
        additional_approved_tools=list(tp_raw.get("additional_approved_tools") or []),
    )

    ad_raw = raw.get("auto_detect") or {}
    auto_detect = WorkflowAutoDetect(
        enabled=bool(ad_raw.get("enabled", False)),
        keywords=list(ad_raw.get("keywords") or []),
        patterns=list(ad_raw.get("patterns") or []),
        min_confidence=int(ad_raw.get("min_confidence", 1)),
    )

    return WorkflowDefinition(
        id=str(wf_id),
        name=str(name),
        description=str(raw.get("description", "")),
        system_prompt=raw.get("system_prompt") or None,
        system_prompt_file=raw.get("system_prompt_file") or None,
        knowledge_base=bool(raw.get("knowledge_base", False)),
        tool_policy=tool_policy,
        auto_detect=auto_detect,
        created_at=str(raw.get("created_at", "")),
        updated_at=str(raw.get("updated_at", "")),
    )


def _workflow_to_dict(wf: WorkflowDefinition) -> dict[str, Any]:
    return {
        "id": wf.id,
        "name": wf.name,
        "description": wf.description,
        "system_prompt": wf.system_prompt,
        "system_prompt_file": wf.system_prompt_file,
        "knowledge_base": wf.knowledge_base,
        "tool_policy": {
            "excluded_tools": wf.tool_policy.excluded_tools,
            "additional_approved_tools": wf.tool_policy.additional_approved_tools,
        },
        "auto_detect": {
            "enabled": wf.auto_detect.enabled,
            "keywords": wf.auto_detect.keywords,
            "patterns": wf.auto_detect.patterns,
            "min_confidence": wf.auto_detect.min_confidence,
        },
        "created_at": wf.created_at,
        "updated_at": wf.updated_at,
    }


def _load_prompt_from_value(value: str, data_dir: Path) -> str:
    """Resolve a system_prompt_file value to text.

    If the value looks like a file path (absolute, home-relative, or relative),
    read it with containment check against *data_dir*. Otherwise return the value
    as inline text (only if it contains whitespace, suggesting multi-word prose
    rather than a path).
    """
    stripped = value.strip()
    if not stripped:
        return ""
    # Heuristic: if it looks like a path (starts with /, ~, ./, or has no spaces
    # and contains a slash or dot-extension), treat it as a file path.
    looks_like_path = (
        stripped.startswith("/")
        or stripped.startswith("~")
        or stripped.startswith("./")
        or stripped.startswith("../")
    )
    if looks_like_path:
        resolved_data = data_dir.resolve()
        path = Path(stripped).expanduser()
        if not path.is_absolute():
            path = resolved_data / path
        path = path.resolve()
        if not path.is_relative_to(resolved_data):
            log.warning(
                "Workflow system_prompt_file %s is outside data_dir %s — rejected",
                path,
                data_dir,
            )
            return ""
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            log.warning("Failed to read workflow system_prompt_file %s: %s", path, exc)
            return ""
    return stripped


# ---------------------------------------------------------------------------
# WorkflowRegistry
# ---------------------------------------------------------------------------


class WorkflowRegistry:
    """Load, cache, and resolve workflow definitions for assistant mode."""

    def __init__(
        self,
        data_dir: str | Path,
        contact_prompts: dict[str, str] | None = None,
        phonebook: dict[str, str] | None = None,
    ) -> None:
        self._data_dir = Path(data_dir).resolve()
        self._workflows_dir = self._data_dir / "workflows"
        self._bindings_path = self._workflows_dir / "bindings.json"
        self._contact_prompts: dict[str, str] = dict(contact_prompts or {})
        self._phonebook: dict[str, str] = dict(phonebook or {})
        self._lock = threading.RLock()
        self._workflows: dict[str, WorkflowDefinition] = {}
        self._bindings: dict[str, dict] = {}
        self.reload()

    # ------------------------------------------------------------------
    # Internal loaders
    # ------------------------------------------------------------------

    def _load_workflows(self) -> dict[str, WorkflowDefinition]:
        result: dict[str, WorkflowDefinition] = {}
        if not self._workflows_dir.is_dir():
            return result
        for yaml_path in sorted(self._workflows_dir.glob("*/workflow.yaml")):
            resolved = yaml_path.resolve()
            if not resolved.is_relative_to(self._data_dir):
                log.warning(
                    "Workflow path %s is outside data_dir %s — skipped",
                    yaml_path,
                    self._data_dir,
                )
                continue
            wf = _load_workflow_yaml(resolved)
            if wf is None:
                continue
            # The directory name must match the workflow id.
            dir_name = yaml_path.parent.name
            if wf.id != dir_name:
                log.warning(
                    "Workflow id %r does not match directory name %r at %s — skipped",
                    wf.id,
                    dir_name,
                    yaml_path,
                )
                continue
            result[wf.id] = wf
        return result

    def _load_bindings(self) -> dict[str, dict]:
        if not self._bindings_path.exists():
            return {}
        resolved = self._bindings_path.resolve()
        if not resolved.is_relative_to(self._data_dir):
            log.warning(
                "Bindings file %s is outside data_dir — ignored",
                self._bindings_path,
            )
            return {}
        try:
            raw = json.loads(resolved.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return raw
            log.warning("Bindings file is not a JSON object — reset to empty")
            return {}
        except Exception as exc:
            log.warning("Failed to load bindings.json: %s", exc)
            return {}

    def _save_bindings(self) -> None:
        """Write _bindings to disk atomically. Must be called under self._lock."""
        self._workflows_dir.mkdir(parents=True, exist_ok=True)
        with atomic_write_json(self._bindings_path) as f:
            json.dump(self._bindings, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reload(self) -> None:
        """Re-scan disk for workflow definitions and bindings."""
        with self._lock:
            self._workflows = self._load_workflows()
            self._bindings = self._load_bindings()

    def list_workflows(self) -> list[WorkflowDefinition]:
        with self._lock:
            return list(self._workflows.values())

    def get_workflow(self, workflow_id: str) -> WorkflowDefinition | None:
        with self._lock:
            return self._workflows.get(workflow_id)

    def create_workflow(self, definition: WorkflowDefinition) -> None:
        """Write a new workflow.yaml to disk. Raises ValueError if ID already exists."""
        _validate_workflow_id(definition.id)
        with self._lock:
            if definition.id in self._workflows:
                raise ValueError(f"Workflow {definition.id!r} already exists")
            now = _now_iso()
            if not definition.created_at:
                definition.created_at = now
            if not definition.updated_at:
                definition.updated_at = now
            wf_dir = self._workflows_dir / definition.id
            wf_dir_resolved = wf_dir.resolve()
            if not wf_dir_resolved.is_relative_to(self._data_dir):
                raise ValueError(
                    f"Workflow directory for {definition.id!r} would be outside data_dir"
                )
            wf_dir.mkdir(parents=True, exist_ok=True)
            yaml_path = wf_dir / "workflow.yaml"
            with atomic_write_json(yaml_path) as f:
                yaml.dump(_workflow_to_dict(definition), f, allow_unicode=True, sort_keys=False)
            self._workflows[definition.id] = definition

    def update_workflow(self, definition: WorkflowDefinition) -> None:
        """Overwrite an existing workflow.yaml. Raises ValueError if not found."""
        _validate_workflow_id(definition.id)
        with self._lock:
            if definition.id not in self._workflows:
                raise ValueError(f"Workflow {definition.id!r} not found")
            definition.updated_at = _now_iso()
            if not definition.created_at:
                definition.created_at = self._workflows[definition.id].created_at
            wf_dir = self._workflows_dir / definition.id
            wf_dir_resolved = (wf_dir / "workflow.yaml").resolve()
            if not wf_dir_resolved.is_relative_to(self._data_dir):
                raise ValueError(
                    f"Workflow directory for {definition.id!r} would be outside data_dir"
                )
            yaml_path = wf_dir / "workflow.yaml"
            with atomic_write_json(yaml_path) as f:
                yaml.dump(_workflow_to_dict(definition), f, allow_unicode=True, sort_keys=False)
            self._workflows[definition.id] = definition

    def delete_workflow(self, workflow_id: str) -> None:
        """Delete workflow directory + remove all bindings for this workflow."""
        with self._lock:
            if workflow_id not in self._workflows:
                raise ValueError(f"Workflow {workflow_id!r} not found")
            wf_dir = self._workflows_dir / workflow_id
            wf_dir_resolved = wf_dir.resolve()
            if not wf_dir_resolved.is_relative_to(self._data_dir):
                raise ValueError(
                    f"Workflow directory for {workflow_id!r} is outside data_dir — refusing delete"
                )
            import shutil

            shutil.rmtree(wf_dir, ignore_errors=True)
            del self._workflows[workflow_id]
            # Remove all bindings pointing to this workflow.
            keys_to_remove = [
                k for k, v in self._bindings.items() if v.get("workflow_id") == workflow_id
            ]
            for k in keys_to_remove:
                del self._bindings[k]
            if keys_to_remove:
                self._save_bindings()

    def bind(self, session_key: str, workflow_id: str, assigned_by: str = "manual") -> None:
        """Bind a session to a workflow. Saves to bindings.json atomically."""
        with self._lock:
            if workflow_id not in self._workflows:
                raise ValueError(f"Workflow {workflow_id!r} not found")
            self._bindings[session_key] = {
                "workflow_id": workflow_id,
                "assigned_at": _now_iso(),
                "assigned_by": assigned_by,
            }
            self._save_bindings()

    def unbind(self, session_key: str) -> bool:
        """Remove a binding. Returns True if one was removed."""
        with self._lock:
            if session_key not in self._bindings:
                return False
            del self._bindings[session_key]
            self._save_bindings()
            return True

    def list_bindings(self) -> dict[str, dict]:
        with self._lock:
            return dict(self._bindings)

    def get_binding(self, session_key: str) -> dict | None:
        with self._lock:
            return self._bindings.get(session_key)

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def _auto_detect(self, msg_text: str) -> str | None:
        """Score each auto-detect-enabled workflow against msg_text.

        Returns workflow_id of the highest-scoring workflow that meets its own
        min_confidence threshold. Ties are broken by alphabetical workflow ID
        for determinism.
        """
        text_lower = msg_text.lower()
        best_id: str | None = None
        best_score = 0
        for wf_id in sorted(self._workflows):
            wf = self._workflows[wf_id]
            if not wf.auto_detect.enabled:
                continue
            score = 0
            for kw in wf.auto_detect.keywords:
                if kw.lower() in text_lower:
                    score += 1
            for pat in wf.auto_detect.patterns:
                try:
                    if re.search(pat, msg_text, re.IGNORECASE):
                        score += 1
                except re.error as exc:
                    log.warning(
                        "Invalid auto_detect pattern %r in workflow %r: %s", pat, wf_id, exc
                    )
            if score >= wf.auto_detect.min_confidence and score > best_score:
                best_id = wf_id
                best_score = score
        return best_id

    def _resolve_system_prompt(self, wf: WorkflowDefinition) -> str | None:
        """Return the effective system prompt for a workflow (inline > file > None)."""
        if wf.system_prompt:
            return wf.system_prompt.strip() or None
        if wf.system_prompt_file:
            text = _load_prompt_from_value(wf.system_prompt_file, self._data_dir)
            return text or None
        return None

    def _knowledge_base_dir(self, wf: WorkflowDefinition) -> Path | None:
        if not wf.knowledge_base:
            return None
        kb_dir = self._workflows_dir / wf.id / "vectordb" / "faiss_index"
        resolved = kb_dir.resolve()
        if not resolved.is_relative_to(self._data_dir):
            log.warning(
                "Workflow KB dir %s is outside data_dir %s — rejected",
                kb_dir,
                self._data_dir,
            )
            return None
        return kb_dir

    def _resolved_from_workflow(self, wf: WorkflowDefinition, assigned_by: str) -> ResolvedWorkflow:
        return ResolvedWorkflow(
            workflow_id=wf.id,
            system_prompt=self._resolve_system_prompt(wf),
            knowledge_base_dir=self._knowledge_base_dir(wf),
            tool_policy=wf.tool_policy,
            assigned_by=assigned_by,
        )

    def _resolve_contact_prompt(
        self,
        sender_id: str,
        resolved_phone: str,
    ) -> str | None:
        """Look up sender in the phonebook, then find a contact_prompts entry."""
        if not self._contact_prompts:
            return None
        # Try to resolve a contact name via the phonebook.
        contact_name: str | None = None
        for identity in (resolved_phone, sender_id):
            if not identity:
                continue
            name = self._phonebook.get(identity)
            if name:
                contact_name = name
                break

        # Look up contact prompt by name or raw identity.
        for lookup_key in filter(None, [contact_name, resolved_phone, sender_id]):
            value = self._contact_prompts.get(lookup_key)
            if value:
                text = _load_prompt_from_value(value, self._data_dir)
                return text or None
        return None

    def resolve(
        self,
        session_key: str,
        msg_text: str = "",
        sender_id: str = "",
        resolved_phone: str = "",
    ) -> ResolvedWorkflow:
        """Resolve the effective workflow for a chat session.

        Resolution order:
        1. Explicit binding in bindings.json → return that workflow.
        2. contact_prompts lookup (backward compat) → ephemeral ResolvedWorkflow.
        3. Auto-detect on msg_text → bind persistently and return.
        4. No match → return ResolvedWorkflow with all-None fields.
        """
        with self._lock:
            # 1. Explicit binding.
            binding = self._bindings.get(session_key)
            if binding:
                wf_id = binding.get("workflow_id", "")
                wf = self._workflows.get(wf_id)
                if wf is not None:
                    return self._resolved_from_workflow(wf, binding.get("assigned_by", "manual"))
                # Binding points to a deleted workflow — clean it up.
                log.warning(
                    "Binding for %s points to unknown workflow %r — removing",
                    session_key,
                    wf_id,
                )
                del self._bindings[session_key]
                self._save_bindings()

            # 2. Legacy contact_prompts fallback (ephemeral, not persisted).
            contact_text = self._resolve_contact_prompt(sender_id, resolved_phone)
            if contact_text:
                return ResolvedWorkflow(
                    workflow_id=None,
                    system_prompt=contact_text,
                    knowledge_base_dir=None,
                    tool_policy=None,
                    assigned_by="contact_prompt",
                )

            # 3. Auto-detect.
            if msg_text:
                matched_id = self._auto_detect(msg_text)
                if matched_id:
                    wf = self._workflows[matched_id]
                    self._bindings[session_key] = {
                        "workflow_id": matched_id,
                        "assigned_at": _now_iso(),
                        "assigned_by": "auto",
                    }
                    self._save_bindings()
                    log.info("Auto-detected workflow %r for session %s", matched_id, session_key)
                    return self._resolved_from_workflow(wf, "auto")

            # 4. No match.
            return ResolvedWorkflow(
                workflow_id=None,
                system_prompt=None,
                knowledge_base_dir=None,
                tool_policy=None,
                assigned_by=None,
            )
