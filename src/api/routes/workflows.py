"""Workflow management endpoints.

Endpoints:
    GET    /api/v1/assistant/workflows                          — list workflows
    POST   /api/v1/assistant/workflows                          — create workflow
    GET    /api/v1/assistant/workflows/{workflow_id}            — get workflow
    PUT    /api/v1/assistant/workflows/{workflow_id}            — update workflow
    DELETE /api/v1/assistant/workflows/{workflow_id}            — delete workflow
    POST   /api/v1/assistant/workflows/{workflow_id}/documents  — upload document
    GET    /api/v1/assistant/workflows/{workflow_id}/documents  — list documents
    DELETE /api/v1/assistant/workflows/{workflow_id}/documents/{doc_id} — delete document
    GET    /api/v1/assistant/workflows/bindings                 — list bindings
    PUT    /api/v1/assistant/workflows/bindings/{session_key}   — bind workflow
    DELETE /api/v1/assistant/workflows/bindings/{session_key}   — unbind workflow
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from src.api.auth import TokenData, get_current_user, require_admin
from src.api.schemas.common import APIResponse, CursorPage
from src.api.schemas.workflow import (
    BindWorkflowRequest,
    WorkflowBindingOut,
    WorkflowCreate,
    WorkflowOut,
    WorkflowUpdate,
)

log = logging.getLogger("cogtrix.api.workflows")

router = APIRouter(prefix="/assistant/workflows", tags=["Workflows"])

_MAX_DOC_BYTES = 50 * 1024 * 1024
_ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt", ".md", ".markdown", ".csv"})
_WORKFLOW_ID_RE = __import__("re").compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_wf_id(workflow_id: str) -> None:
    """Validate workflow_id at the API boundary before it reaches filesystem ops."""
    if not _WORKFLOW_ID_RE.match(workflow_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "VALIDATION_ERROR",
                "message": (
                    f"Invalid workflow ID {workflow_id!r}. "
                    "Must match ^[a-zA-Z0-9][a-zA-Z0-9_-]*$."
                ),
            },
        )


def _get_registry(request: Request) -> object:
    registry = getattr(request.app.state, "workflow_registry", None)
    if registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "SERVICE_UNAVAILABLE", "message": "Workflow registry not initialized."},
        )
    return registry


def _wf_to_out(wf: object) -> WorkflowOut:
    from src.assistant.workflows import WorkflowDefinition

    w: WorkflowDefinition = wf  # type: ignore[assignment]
    return WorkflowOut(
        id=w.id,
        name=w.name,
        description=w.description,
        system_prompt=w.system_prompt,
        system_prompt_file=w.system_prompt_file,
        knowledge_base=w.knowledge_base,
        tool_policy={"excluded_tools": w.tool_policy.excluded_tools, "additional_approved_tools": w.tool_policy.additional_approved_tools},  # type: ignore[arg-type]
        auto_detect={"enabled": w.auto_detect.enabled, "keywords": w.auto_detect.keywords, "patterns": w.auto_detect.patterns, "min_confidence": w.auto_detect.min_confidence},  # type: ignore[arg-type]
        created_at=w.created_at,
        updated_at=w.updated_at,
    )


# ---------------------------------------------------------------------------
# Workflow CRUD
# ---------------------------------------------------------------------------


@router.get(
    "",
    summary="List workflows",
    response_model=APIResponse[CursorPage[WorkflowOut]],
)
async def list_workflows(
    request: Request,
    current_user: TokenData = Depends(get_current_user),
) -> APIResponse[CursorPage[WorkflowOut]]:
    registry = _get_registry(request)
    workflows = await asyncio.to_thread(registry.list_workflows)  # type: ignore[attr-defined]
    items = [_wf_to_out(w) for w in workflows]
    page: CursorPage[WorkflowOut] = CursorPage(
        items=items, next_cursor=None, has_more=False, total=len(items)
    )
    return APIResponse(data=page)


@router.post(
    "",
    summary="Create workflow",
    response_model=APIResponse[WorkflowOut],
    status_code=201,
)
async def create_workflow(
    body: WorkflowCreate,
    request: Request,
    current_user: TokenData = Depends(require_admin),
) -> APIResponse[WorkflowOut]:
    from src.assistant.workflows import WorkflowAutoDetect, WorkflowDefinition, WorkflowToolPolicy

    registry = _get_registry(request)
    wf = WorkflowDefinition(
        id=body.id,
        name=body.name,
        description=body.description,
        system_prompt=body.system_prompt,
        system_prompt_file=body.system_prompt_file,
        knowledge_base=body.knowledge_base,
        tool_policy=WorkflowToolPolicy(
            excluded_tools=list(body.tool_policy.excluded_tools),
            additional_approved_tools=list(body.tool_policy.additional_approved_tools),
        ),
        auto_detect=WorkflowAutoDetect(
            enabled=body.auto_detect.enabled,
            keywords=list(body.auto_detect.keywords),
            patterns=list(body.auto_detect.patterns),
            min_confidence=body.auto_detect.min_confidence,
        ),
    )
    try:
        await asyncio.to_thread(registry.create_workflow, wf)  # type: ignore[attr-defined]
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "CONFLICT", "message": str(exc)},
        ) from exc
    return APIResponse(data=_wf_to_out(wf))


@router.get(
    "/bindings",
    summary="List workflow bindings",
    response_model=APIResponse[list[WorkflowBindingOut]],
)
async def list_bindings(
    request: Request,
    current_user: TokenData = Depends(get_current_user),
) -> APIResponse[list[WorkflowBindingOut]]:
    registry = _get_registry(request)
    raw: dict = await asyncio.to_thread(registry.list_bindings)  # type: ignore[attr-defined]
    items = [
        WorkflowBindingOut(
            session_key=sk,
            workflow_id=v.get("workflow_id", ""),
            assigned_at=v.get("assigned_at", ""),
            assigned_by=v.get("assigned_by"),
        )
        for sk, v in raw.items()
    ]
    return APIResponse(data=items)


@router.get(
    "/{workflow_id}",
    summary="Get workflow",
    response_model=APIResponse[WorkflowOut],
)
async def get_workflow(
    workflow_id: str,
    request: Request,
    current_user: TokenData = Depends(get_current_user),
) -> APIResponse[WorkflowOut]:
    _validate_wf_id(workflow_id)
    registry = _get_registry(request)
    wf = await asyncio.to_thread(registry.get_workflow, workflow_id)  # type: ignore[attr-defined]
    if wf is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Workflow '{workflow_id}' not found."},
        )
    return APIResponse(data=_wf_to_out(wf))


@router.put(
    "/{workflow_id}",
    summary="Update workflow",
    response_model=APIResponse[WorkflowOut],
)
async def update_workflow(
    workflow_id: str,
    body: WorkflowUpdate,
    request: Request,
    current_user: TokenData = Depends(require_admin),
) -> APIResponse[WorkflowOut]:
    from src.assistant.workflows import WorkflowAutoDetect, WorkflowToolPolicy

    _validate_wf_id(workflow_id)
    registry = _get_registry(request)
    wf = await asyncio.to_thread(registry.get_workflow, workflow_id)  # type: ignore[attr-defined]
    if wf is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Workflow '{workflow_id}' not found."},
        )

    # BUG-213: Build a new definition instead of mutating the live registry object.
    import dataclasses

    updated = dataclasses.replace(wf)
    if body.name is not None:
        updated.name = body.name
    if body.description is not None:
        updated.description = body.description
    if body.system_prompt is not None:
        updated.system_prompt = body.system_prompt
    if body.system_prompt_file is not None:
        updated.system_prompt_file = body.system_prompt_file
    if body.knowledge_base is not None:
        updated.knowledge_base = body.knowledge_base
    if body.tool_policy is not None:
        updated.tool_policy = WorkflowToolPolicy(
            excluded_tools=list(body.tool_policy.excluded_tools),
            additional_approved_tools=list(body.tool_policy.additional_approved_tools),
        )
    if body.auto_detect is not None:
        updated.auto_detect = WorkflowAutoDetect(
            enabled=body.auto_detect.enabled,
            keywords=list(body.auto_detect.keywords),
            patterns=list(body.auto_detect.patterns),
            min_confidence=body.auto_detect.min_confidence,
        )

    try:
        await asyncio.to_thread(registry.update_workflow, updated)  # type: ignore[attr-defined]
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": str(exc)},
        ) from exc
    return APIResponse(data=_wf_to_out(updated))


@router.delete(
    "/{workflow_id}",
    summary="Delete workflow",
    response_model=APIResponse[None],
)
async def delete_workflow(
    workflow_id: str,
    request: Request,
    current_user: TokenData = Depends(require_admin),
) -> APIResponse[None]:
    _validate_wf_id(workflow_id)
    registry = _get_registry(request)
    try:
        await asyncio.to_thread(registry.delete_workflow, workflow_id)  # type: ignore[attr-defined]
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": str(exc)},
        ) from exc
    return APIResponse(data=None)


# ---------------------------------------------------------------------------
# Per-workflow document management
# ---------------------------------------------------------------------------


def _wf_kb_dir(registry: object, workflow_id: str) -> Path:
    from src.assistant.workflows import WorkflowRegistry

    reg: WorkflowRegistry = registry  # type: ignore[assignment]
    return reg._workflows_dir / workflow_id / "vectordb" / "faiss_index"


def _wf_docs_dir(registry: object, workflow_id: str) -> Path:
    from src.assistant.workflows import WorkflowRegistry

    reg: WorkflowRegistry = registry  # type: ignore[assignment]
    return reg._workflows_dir / workflow_id / "docs"


@router.post(
    "/{workflow_id}/documents",
    summary="Upload a document to workflow knowledge base",
    response_model=APIResponse[dict],
    status_code=202,
)
async def upload_workflow_document(
    workflow_id: str,
    request: Request,
    file: UploadFile = File(...),
    current_user: TokenData = Depends(require_admin),
) -> APIResponse[dict]:
    _validate_wf_id(workflow_id)
    registry = _get_registry(request)
    wf = await asyncio.to_thread(registry.get_workflow, workflow_id)  # type: ignore[attr-defined]
    if wf is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Workflow '{workflow_id}' not found."},
        )

    filename = Path(file.filename or "upload").name or "upload"
    suffix = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "code": "VALIDATION_ERROR",
                "message": (
                    f"Unsupported file type '{suffix}'. "
                    f"Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
                ),
            },
        )

    data = await file.read()
    if len(data) > _MAX_DOC_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "VALIDATION_ERROR",
                "message": f"File too large ({len(data)} bytes); maximum is 50 MB.",
            },
        )

    doc_id = str(uuid.uuid4())

    def _save_and_ingest() -> dict:
        from src.assistant.workflows import WorkflowRegistry

        reg: WorkflowRegistry = registry  # type: ignore[assignment]
        docs_dir = reg._workflows_dir / workflow_id / "docs" / doc_id
        docs_dir.mkdir(parents=True, exist_ok=True)
        file_path = docs_dir / filename
        # BUG-216: path containment — ensure resolved path stays inside data_dir.
        resolved_fp = file_path.resolve()
        if not resolved_fp.is_relative_to(reg._data_dir):
            raise ValueError("filename escapes workflow directory")
        file_path.write_bytes(data)
        log.info(
            "workflow_doc_upload: workflow=%s doc_id=%s file=%s size=%d",
            workflow_id,
            doc_id,
            filename,
            len(data),
        )
        return {"doc_id": doc_id, "filename": filename, "size_bytes": len(data), "status": "saved"}

    try:
        result = await asyncio.to_thread(_save_and_ingest)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "VALIDATION_ERROR", "message": str(exc)},
        ) from exc
    return APIResponse(data=result)


@router.get(
    "/{workflow_id}/documents",
    summary="List documents in workflow knowledge base",
    response_model=APIResponse[list[dict]],
)
async def list_workflow_documents(
    workflow_id: str,
    request: Request,
    current_user: TokenData = Depends(get_current_user),
) -> APIResponse[list[dict]]:
    _validate_wf_id(workflow_id)
    registry = _get_registry(request)
    wf = await asyncio.to_thread(registry.get_workflow, workflow_id)  # type: ignore[attr-defined]
    if wf is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Workflow '{workflow_id}' not found."},
        )

    def _list_docs() -> list[dict]:
        from src.assistant.workflows import WorkflowRegistry

        reg: WorkflowRegistry = registry  # type: ignore[assignment]
        docs_root = reg._workflows_dir / workflow_id / "docs"
        if not docs_root.is_dir():
            return []
        items = []
        for doc_dir in sorted(docs_root.iterdir()):
            if not doc_dir.is_dir():
                continue
            for f in doc_dir.iterdir():
                if f.is_file():
                    mime, _ = mimetypes.guess_type(f.name)
                    items.append(
                        {
                            "doc_id": doc_dir.name,
                            "filename": f.name,
                            "size_bytes": f.stat().st_size,
                            "content_type": mime or "application/octet-stream",
                        }
                    )
                    break
        return items

    items = await asyncio.to_thread(_list_docs)
    return APIResponse(data=items)


@router.delete(
    "/{workflow_id}/documents/{doc_id}",
    summary="Delete a document from workflow knowledge base",
    response_model=APIResponse[None],
)
async def delete_workflow_document(
    workflow_id: str,
    doc_id: str,
    request: Request,
    current_user: TokenData = Depends(require_admin),
) -> APIResponse[None]:
    _validate_wf_id(workflow_id)
    registry = _get_registry(request)
    wf = await asyncio.to_thread(registry.get_workflow, workflow_id)  # type: ignore[attr-defined]
    if wf is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Workflow '{workflow_id}' not found."},
        )

    def _delete_doc() -> bool:
        from src.assistant.workflows import WorkflowRegistry

        reg: WorkflowRegistry = registry  # type: ignore[assignment]
        doc_dir = reg._workflows_dir / workflow_id / "docs" / doc_id
        resolved = doc_dir.resolve()
        if not resolved.is_relative_to(reg._data_dir):
            raise ValueError("doc_id escapes workflow directory")
        if not doc_dir.is_dir():
            return False
        shutil.rmtree(doc_dir, ignore_errors=True)
        return True

    try:
        found = await asyncio.to_thread(_delete_doc)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "VALIDATION_ERROR", "message": str(exc)},
        ) from exc

    if not found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "Document not found."},
        )
    return APIResponse(data=None)


# ---------------------------------------------------------------------------
# Binding management
# ---------------------------------------------------------------------------


@router.put(
    "/bindings/{session_key:path}",
    summary="Bind a workflow to a chat session",
    response_model=APIResponse[WorkflowBindingOut],
)
async def bind_workflow(
    session_key: str,
    body: BindWorkflowRequest,
    request: Request,
    current_user: TokenData = Depends(require_admin),
) -> APIResponse[WorkflowBindingOut]:
    registry = _get_registry(request)
    try:
        await asyncio.to_thread(registry.bind, session_key, body.workflow_id, "manual")  # type: ignore[attr-defined]
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": str(exc)},
        ) from exc
    binding = await asyncio.to_thread(registry.get_binding, session_key)  # type: ignore[attr-defined]
    return APIResponse(
        data=WorkflowBindingOut(
            session_key=session_key,
            workflow_id=body.workflow_id,
            assigned_at=binding.get("assigned_at", "") if binding else "",
            assigned_by="manual",
        )
    )


@router.delete(
    "/bindings/{session_key:path}",
    summary="Remove a workflow binding from a chat session",
    response_model=APIResponse[None],
)
async def unbind_workflow(
    session_key: str,
    request: Request,
    current_user: TokenData = Depends(require_admin),
) -> APIResponse[None]:
    registry = _get_registry(request)
    removed = await asyncio.to_thread(registry.unbind, session_key)  # type: ignore[attr-defined]
    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"No binding found for '{session_key}'."},
        )
    return APIResponse(data=None)
