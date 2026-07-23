"""Configuration, provider, model management, and setup wizard endpoints.

Endpoints:
    GET    /api/v1/config                   — view current config snapshot
    PATCH  /api/v1/config                   — partial runtime config update
    POST   /api/v1/config/reload            — reload config from disk (admin)
    GET    /api/v1/config/providers         — list configured providers
    GET    /api/v1/config/providers/{name}  — get provider details
    POST   /api/v1/config/provider          — switch active provider
    POST   /api/v1/config/providers/{name}/health — provider connectivity check
    GET    /api/v1/config/models            — list available models
    POST   /api/v1/config/model             — switch active model
    POST   /api/v1/config/wizard            — start a setup wizard session
    POST   /api/v1/config/wizard/{id}/step  — advance the wizard one step
    DELETE /api/v1/config/wizard/{id}       — cancel and discard a wizard session
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Request, status

from src.api.auth import TokenData, get_current_user, require_admin
from src.api.schemas.common import APIResponse
from src.api.schemas.config import (
    ConfigOut,
    ConfigPatchRequest,
    ConfigReloadResponse,
    ModelOut,
    ModelSwitchRequest,
    ProviderHealthOut,
    ProviderOut,
    WizardStartRequest,
    WizardStepOut,
    WizardStepRequest,
)

log = logging.getLogger("cogtrix.api.config")

router = APIRouter(prefix="/config", tags=["Configuration"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_config(request: Request) -> Any:
    return getattr(request.app.state, "config", None)


def _provider_to_out(name: str, pc: Any) -> ProviderOut:
    return ProviderOut(
        name=name,
        type=getattr(pc, "type", name),
        base_url=getattr(pc, "base_url", None),
        has_api_key=bool(getattr(pc, "api_key", None)),
    )


def _model_to_out(alias: str, mc: Any, cfg: Any) -> ModelOut:
    active_alias = getattr(cfg, "active_model_alias", None)
    return ModelOut(
        alias=alias,
        provider=getattr(mc, "provider", ""),
        model_name=getattr(mc, "model", alias),
        num_ctx=getattr(mc, "context_window", None),
        temperature=getattr(mc, "temperature", None),
        max_tokens=getattr(mc, "max_tokens", None),
        is_active=(alias == active_alias),
    )


def _config_to_out(cfg: Any, raw_yaml: str | None = None) -> ConfigOut:
    providers_out: list[ProviderOut] = []
    if cfg is not None:
        cfg_providers = getattr(cfg, "providers", {}) or {}
        for name, pc in cfg_providers.items():
            providers_out.append(_provider_to_out(name, pc))

    models_out: list[ModelOut] = []
    if cfg is not None:
        for alias, mc in (getattr(cfg, "models", {}) or {}).items():
            models_out.append(_model_to_out(alias, mc, cfg))

    cfg_path = getattr(cfg, "config_file_path", None) if cfg else None
    active_alias = getattr(cfg, "active_model_alias", None) if cfg else None

    _raw_sys_prompt = getattr(cfg, "system_prompt", None) if cfg else None
    _sys_prompt = _raw_sys_prompt if isinstance(_raw_sys_prompt, str) and _raw_sys_prompt else None
    _services = getattr(cfg, "services", None) if cfg else None
    _guardrails: dict | None = None
    if isinstance(_services, dict):
        _asst = _services.get("assistant")
        if isinstance(_asst, dict):
            _gr = _asst.get("guardrails")
            if isinstance(_gr, dict):
                _guardrails = _gr

    return ConfigOut(
        active_model=active_alias,
        memory_mode=(getattr(cfg, "memory_mode", "conversation") if cfg else None)
        or "conversation",
        prompt_optimizer=bool(getattr(cfg, "prompt_optimizer", True) if cfg else True),
        parallel_tool_execution=bool(
            getattr(cfg, "parallel_tool_execution", True) if cfg else True
        ),
        context_compression=bool(getattr(cfg, "context_compression", True) if cfg else True),
        debug=bool(getattr(cfg, "debug", False) if cfg else False),
        verbose=bool(getattr(cfg, "verbose", False) if cfg else False),
        config_file_path=str(cfg_path) if cfg_path else None,
        providers=providers_out,
        models=models_out,
        raw_yaml=raw_yaml,
        system_prompt=_sys_prompt,
        guardrails=_guardrails,
    )


async def _read_raw_yaml(cfg: Any) -> str | None:
    """Read the raw YAML config file asynchronously (admin-only feature).

    Runs file I/O off the event loop thread to prevent stalls.
    Returns None if the config path is not set or the file cannot be read.
    """
    cfg_path = getattr(cfg, "config_file_path", None) if cfg else None
    if cfg_path is None:
        return None

    def _read() -> str | None:
        try:
            return cfg_path.read_text()
        except Exception:
            return None

    return await asyncio.to_thread(_read)


# ---------------------------------------------------------------------------
# Config view / edit
# ---------------------------------------------------------------------------


@router.get(
    "",
    summary="Get current configuration",
    description=(
        "Return a snapshot of the current application configuration. "
        "Sensitive values (API keys) are masked. "
        "Admin users additionally receive the raw YAML in the raw_yaml field."
    ),
    response_model=APIResponse[ConfigOut],
    responses={
        200: {"description": "Config snapshot returned."},
        401: {"description": "Not authenticated."},
    },
)
async def get_config(
    request: Request,
    current_user: TokenData = Depends(get_current_user),
) -> APIResponse[ConfigOut]:
    """Return the current application configuration (API keys masked).

    Auth: bearer token required. raw_yaml field populated for admin users only.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED.
    """
    cfg = _get_config(request)
    raw_yaml = await _read_raw_yaml(cfg) if current_user.is_admin else None
    return APIResponse(data=_config_to_out(cfg, raw_yaml=raw_yaml))


@router.patch(
    "",
    summary="Partial runtime config update",
    description=(
        "Update a subset of runtime-configurable settings without touching the config file. "
        "Changes persist only for the current process lifetime unless saved via the wizard. "
        "Admin only."
    ),
    response_model=APIResponse[ConfigOut],
    responses={
        200: {"description": "Config updated; new snapshot returned."},
        401: {"description": "Not authenticated."},
        403: {"description": "Admin required (FORBIDDEN)."},
        422: {"description": "Validation error (VALIDATION_ERROR, CONFIG_INVALID)."},
    },
)
async def patch_config(
    body: ConfigPatchRequest,
    request: Request,
    current_user: TokenData = Depends(require_admin),
) -> APIResponse[ConfigOut]:
    """Apply a partial runtime config update (admin only).

    Auth: admin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, VALIDATION_ERROR, CONFIG_INVALID.
    """
    import logging as _logging

    cfg = _get_config(request)
    if cfg is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "INTERNAL_ERROR", "message": "Config not available."},
        )

    if body.debug is not None:
        cfg.debug = body.debug
        level = _logging.DEBUG if body.debug else _logging.INFO
        _logging.getLogger("cogtrix").setLevel(level)
    if body.verbose is not None:
        cfg.verbose = body.verbose
    if body.prompt_optimizer is not None:
        cfg.prompt_optimizer = body.prompt_optimizer
    if body.parallel_tool_execution is not None:
        cfg.parallel_tool_execution = body.parallel_tool_execution
    if body.context_compression is not None:
        cfg.context_compression = body.context_compression

    raw_yaml = await _read_raw_yaml(cfg)
    return APIResponse(data=_config_to_out(cfg, raw_yaml=raw_yaml))


@router.post(
    "/reload",
    summary="Reload configuration from disk",
    description=(
        "Re-read the config file from disk and apply changes. "
        "Provider and model changes take effect on the next agent turn. "
        "Admin only."
    ),
    response_model=APIResponse[ConfigReloadResponse],
    responses={
        200: {"description": "Config reloaded."},
        401: {"description": "Not authenticated."},
        403: {"description": "Admin required (FORBIDDEN)."},
        422: {"description": "Config file invalid (CONFIG_INVALID)."},
    },
)
async def reload_config(
    request: Request,
    current_user: TokenData = Depends(require_admin),
) -> APIResponse[ConfigReloadResponse]:
    """Reload the config file from disk and apply changes (admin only).

    Auth: admin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, CONFIG_INVALID.
    """
    import asyncio

    warnings_list: list[str] = []
    try:
        from src.config import load_config

        # load_config() reads the config file, applies env vars, and resolves
        # model aliases — Config() alone would create an empty default config.
        new_cfg = await asyncio.to_thread(load_config)
        request.app.state.config = new_cfg
        cfg_path = getattr(new_cfg, "config_file_path", None)
        return APIResponse(
            data=ConfigReloadResponse(
                reloaded=True,
                config_file_path=str(cfg_path) if cfg_path else None,
                warnings=warnings_list,
            )
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "CONFIG_INVALID", "message": f"Config reload failed: {exc}"},
        ) from exc


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


@router.get(
    "/providers",
    summary="List configured providers",
    description="List all LLM providers defined in the config, with their settings and active status.",
    response_model=APIResponse[list[ProviderOut]],
    responses={
        200: {"description": "Provider list returned."},
        401: {"description": "Not authenticated."},
    },
)
async def list_providers(
    request: Request,
    current_user: TokenData = Depends(get_current_user),
) -> APIResponse[list[ProviderOut]]:
    """List all configured LLM providers.

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED.
    """
    cfg = _get_config(request)
    out = _config_to_out(cfg)
    return APIResponse(data=out.providers)


@router.get(
    "/providers/{provider_name}",
    summary="Get provider details",
    description="Return details for a single configured provider.",
    response_model=APIResponse[ProviderOut],
    responses={
        200: {"description": "Provider details returned."},
        401: {"description": "Not authenticated."},
        404: {"description": "Provider not found (NOT_FOUND)."},
    },
)
async def get_provider(
    provider_name: str,
    request: Request,
    current_user: TokenData = Depends(get_current_user),
) -> APIResponse[ProviderOut]:
    """Return details for a single provider.

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, NOT_FOUND.
    """
    cfg = _get_config(request)
    providers = (getattr(cfg, "providers", {}) if cfg else None) or {}
    if provider_name not in providers:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Provider '{provider_name}' not found."},
        )
    return APIResponse(data=_provider_to_out(provider_name, providers[provider_name]))


@router.post(
    "/provider",
    summary="Switch active provider — deprecated",
    description=(
        "Deprecated: provider switching is no longer supported. "
        "Use POST /config/model to switch models instead."
    ),
    response_model=APIResponse[ConfigOut],
    responses={
        410: {"description": "Endpoint removed (GONE)."},
        401: {"description": "Not authenticated."},
        403: {"description": "Admin required (FORBIDDEN)."},
    },
)
async def switch_provider(
    request: Request,
    body: Any = Body(default=None),
    current_user: TokenData = Depends(require_admin),
) -> APIResponse[ConfigOut]:
    """Switch provider endpoint — deprecated.

    Provider switching is no longer supported. Use POST /config/model instead.
    """
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail={
            "code": "GONE",
            "message": "Provider switching is no longer supported. Use POST /config/model to switch models.",
        },
    )


@router.post(
    "/providers/{provider_name}/health",
    summary="Provider connectivity check",
    description="Test connectivity to a provider and measure latency. Does not change any state.",
    response_model=APIResponse[ProviderHealthOut],
    responses={
        200: {"description": "Health check result returned (check the 'reachable' field)."},
        401: {"description": "Not authenticated."},
        404: {"description": "Provider not found (NOT_FOUND)."},
    },
)
async def check_provider_health(
    provider_name: str,
    request: Request,
    current_user: TokenData = Depends(get_current_user),
) -> APIResponse[ProviderHealthOut]:
    """Test connectivity to a provider and return reachability + latency.

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, NOT_FOUND.
    """
    import asyncio

    cfg = _get_config(request)
    providers = (getattr(cfg, "providers", {}) if cfg else None) or {}
    if provider_name not in providers:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Provider '{provider_name}' not found."},
        )
    pc = providers[provider_name]
    t0 = time.monotonic()
    try:
        from src.providers import create_chat_model, get_default_model

        prov_type = getattr(pc, "type", provider_name)
        default_model = get_default_model(prov_type)
        # create_chat_model may perform network I/O (e.g. Ollama API
        # introspection); run it off the event loop thread to prevent stalls.
        await asyncio.to_thread(
            create_chat_model,
            prov_type,
            model=default_model,
            api_key=getattr(pc, "api_key", None),
            base_url=getattr(pc, "get_base_url", lambda: None)(),
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        return APIResponse(
            data=ProviderHealthOut(
                name=provider_name, reachable=True, latency_ms=latency_ms, error=None
            )
        )
    except Exception as exc:
        latency_ms = int((time.monotonic() - t0) * 1000)
        return APIResponse(
            data=ProviderHealthOut(
                name=provider_name, reachable=False, latency_ms=latency_ms, error=str(exc)
            )
        )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@router.get(
    "/models",
    summary="List available models",
    description=(
        "List all named model registry entries from the config. "
        "For Ollama providers, also includes models returned by the /api/tags endpoint. "
        "The active field indicates the currently selected model."
    ),
    response_model=APIResponse[list[ModelOut]],
    responses={
        200: {"description": "Model list returned."},
        401: {"description": "Not authenticated."},
    },
)
async def list_models(
    request: Request,
    provider: str | None = None,
    current_user: TokenData = Depends(get_current_user),
) -> APIResponse[list[ModelOut]]:
    """List all available models from the registry and configured providers.

    Query parameters:
        provider — filter to models from a specific provider alias.

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED.
    """
    cfg = _get_config(request)
    out = _config_to_out(cfg)
    models = out.models
    if provider:
        models = [m for m in models if m.provider == provider]
    return APIResponse(data=models)


@router.post(
    "/model",
    summary="Switch active model",
    description=(
        "Switch the globally active model. "
        "Accepts a model alias from the models registry, or a 'provider/model_name' string. "
        "Invalidates the LLM bind-tools cache. Admin only."
    ),
    response_model=APIResponse[ConfigOut],
    responses={
        200: {"description": "Model switched; updated config snapshot returned."},
        401: {"description": "Not authenticated."},
        403: {"description": "Admin required (FORBIDDEN)."},
        404: {"description": "Model not found (NOT_FOUND)."},
        503: {"description": "Provider unreachable (PROVIDER_UNREACHABLE)."},
    },
)
async def switch_model(
    body: ModelSwitchRequest,
    request: Request,
    current_user: TokenData = Depends(require_admin),
) -> APIResponse[ConfigOut]:
    """Switch the globally active model (admin only).

    Auth: admin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, MODEL_UNAVAILABLE, PROVIDER_UNREACHABLE.
    """
    cfg = _get_config(request)
    if cfg is not None:
        cfg.active_model_alias = body.model
        from src.config import _resolve_model

        _resolve_model(cfg)
    try:
        from src.orchestration.runner import invalidate_llm_caches

        invalidate_llm_caches()
    except Exception as exc:
        log.debug("invalidate_llm_caches: %s", exc)
    raw_yaml = await _read_raw_yaml(cfg)
    return APIResponse(data=_config_to_out(cfg, raw_yaml=raw_yaml))


# ---------------------------------------------------------------------------
# Setup wizard — in-memory session store with 3-step flow
# ---------------------------------------------------------------------------


_wizard_sessions: dict[str, dict[str, Any]] = {}
_WIZARD_TTL = 1800  # 30 minutes


def _get_wizard(wizard_id: str) -> dict[str, Any] | None:
    """Return wizard session or None if expired/missing."""
    session = _wizard_sessions.get(wizard_id)
    if session is None:
        return None
    if time.monotonic() - session["created_mono"] > _WIZARD_TTL:
        _wizard_sessions.pop(wizard_id, None)
        return None
    return session


@router.post(
    "/wizard",
    summary="Start a setup wizard session",
    description=(
        "Initiate an interactive setup wizard that guides the user through "
        "provider selection, API key entry, model selection, and LLM-guided Q&A. "
        "Returns the first wizard step. Use the wizard_id to advance steps. "
        "Admin only."
    ),
    response_model=APIResponse[WizardStepOut],
    status_code=201,
    responses={
        201: {"description": "Wizard started; first step returned."},
        401: {"description": "Not authenticated."},
        403: {"description": "Admin required (FORBIDDEN)."},
    },
)
async def start_wizard(
    body: WizardStartRequest,
    request: Request,
    current_user: TokenData = Depends(require_admin),
) -> APIResponse[WizardStepOut]:
    """Start a setup wizard session and return the first step (admin only).

    Auth: admin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN.
    """
    wid = str(uuid.uuid4())

    # Detect environment (non-blocking)
    env = await asyncio.to_thread(_wizard_detect_env)

    # Load existing config if editing
    existing_yaml = ""
    if body.edit_existing:
        existing_yaml = await asyncio.to_thread(_wizard_load_existing)

    _wizard_sessions[wid] = {
        "created_mono": time.monotonic(),
        "step": 0,
        "env": env,
        "existing_yaml": existing_yaml,
        "bootstrap_info": None,
        "llm": None,
        "messages": [],
        "docs_url": body.docs_url,
    }

    return APIResponse(
        data=WizardStepOut(
            wizard_id=wid,
            step=0,
            total_steps=3,
            step_name="Connect to LLM",
            question=(
                "Select a provider and provide connection details. "
                "Send data: {provider_type, api_key?, base_url?, model}"
            ),
            yaml_preview=None,
            complete=False,
            warnings=[],
        )
    )


@router.post(
    "/wizard/{wizard_id}/step",
    summary="Advance the wizard by one step",
    description=(
        "Submit the user's answer to the current wizard question and advance to the next step. "
        "On the final step, the wizard validates the generated YAML and writes the config file."
    ),
    response_model=APIResponse[WizardStepOut],
    responses={
        200: {"description": "Step processed; next step (or completion) returned."},
        401: {"description": "Not authenticated."},
        403: {"description": "Admin required (FORBIDDEN)."},
        404: {"description": "Wizard session not found (NOT_FOUND)."},
        422: {"description": "Wizard step failed (WIZARD_STEP_ERROR)."},
    },
)
async def advance_wizard(
    wizard_id: str,
    body: WizardStepRequest,
    request: Request,
    current_user: TokenData = Depends(require_admin),
) -> APIResponse[WizardStepOut]:
    """Advance the wizard one step and return the next question or completion (admin only).

    Auth: admin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, NOT_FOUND, WIZARD_STEP_ERROR.
    """
    ws = _get_wizard(wizard_id)
    if ws is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "Wizard session not found or expired."},
        )

    step = ws["step"]

    # ── Step 0: Connect to LLM ──
    if step == 0:
        data = body.data or {}
        provider_type = data.get("provider_type", "openai")
        api_key = data.get("api_key")
        base_url = data.get("base_url")
        model = data.get("model")

        if not model:
            from src.providers import get_default_model

            model = get_default_model(provider_type)

        # Test connection off the event loop
        try:
            llm = await asyncio.to_thread(
                _wizard_test_connection, provider_type, model, api_key, base_url
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "PROVIDER_UNREACHABLE",
                    "message": str(exc),
                },
            ) from exc

        # Determine provider name
        provider_name = data.get("provider_name", provider_type)
        ws["bootstrap_info"] = {
            "provider": provider_name,
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
            "type": provider_type,
        }
        ws["llm"] = llm

        # Build system prompt and start conversation
        docs = await asyncio.to_thread(_wizard_load_docs, ws.get("docs_url"))
        from src.setup_wizard import _WIZARD_SYSTEM_PROMPT

        system_prompt = _WIZARD_SYSTEM_PROMPT.substitute(
            docs=docs,
            existing_config=ws["existing_yaml"] or "No existing configuration.",
            bootstrap_provider=provider_name,
            bootstrap_model=model,
        )

        # Get first LLM question
        from langchain_core.messages import AIMessage, SystemMessage

        messages = [SystemMessage(content=system_prompt)]
        try:
            ai_text = await asyncio.to_thread(_wizard_invoke_llm, llm, messages)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "PROVIDER_UNREACHABLE",
                    "message": str(exc),
                },
            ) from exc
        messages.append(AIMessage(content=ai_text))
        ws["messages"] = messages
        ws["step"] = 1

        return APIResponse(
            data=WizardStepOut(
                wizard_id=wizard_id,
                step=1,
                total_steps=3,
                step_name="Configure",
                question=ai_text,
                yaml_preview=None,
                complete=False,
                warnings=[],
            )
        )

    # ── Step 1: Configure (Q&A loop) ──
    if step == 1:
        from langchain_core.messages import AIMessage, HumanMessage

        user_answer = body.answer or ""
        if not user_answer.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "WIZARD_STEP_ERROR",
                    "message": "An answer is required for this step.",
                },
            )

        # Check if user wants to accept the config (data.accept = true)
        data = body.data or {}
        if data.get("accept") and ws["messages"]:
            # Look for YAML in the last AI message
            last_ai = ws["messages"][-1].content if hasattr(ws["messages"][-1], "content") else ""
            from src.setup_wizard import _has_yaml_block

            if _has_yaml_block(last_ai):
                ws["step"] = 2
                return await _wizard_save(wizard_id, ws, request)

        messages = ws["messages"]
        messages.append(HumanMessage(content=user_answer))

        llm = ws["llm"]
        try:
            ai_text = await asyncio.to_thread(_wizard_invoke_llm, llm, messages)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"code": "PROVIDER_UNREACHABLE", "message": str(exc)},
            ) from exc
        messages.append(AIMessage(content=ai_text))

        from src.setup_wizard import _has_yaml_block

        yaml_preview = None
        requires_acceptance = False
        question_text = ai_text
        if _has_yaml_block(ai_text):
            try:
                from src.setup_wizard import _extract_yaml, _mask_secrets

                raw_yaml = _extract_yaml(ai_text)
                yaml_preview = _mask_secrets(raw_yaml)
                # Strip the YAML code block from question to avoid duplication
                # (yaml_preview is the canonical place for config content)
                import re as _re

                question_text = _re.sub(r"```ya?ml\b.*?```", "", ai_text, flags=_re.DOTALL).strip()
            except Exception:
                pass
            requires_acceptance = True

        return APIResponse(
            data=WizardStepOut(
                wizard_id=wizard_id,
                step=1,
                total_steps=3,
                step_name="Configure",
                question=question_text,
                yaml_preview=yaml_preview,
                requires_acceptance=requires_acceptance,
                complete=False,
                warnings=[],
            )
        )

    # ── Step 2: Save ──
    if step == 2:
        return await _wizard_save(wizard_id, ws, request)

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail={"code": "WIZARD_STEP_ERROR", "message": "Invalid wizard step."},
    )


@router.delete(
    "/wizard/{wizard_id}",
    summary="Cancel a wizard session",
    description="Cancel and discard an in-progress wizard session. No config changes are saved.",
    response_model=APIResponse[None],
    responses={
        200: {"description": "Wizard cancelled."},
        401: {"description": "Not authenticated."},
        403: {"description": "Admin required (FORBIDDEN)."},
        404: {"description": "Wizard session not found (NOT_FOUND)."},
    },
)
async def cancel_wizard(
    wizard_id: str,
    current_user: TokenData = Depends(require_admin),
) -> APIResponse[None]:
    """Cancel a wizard session without saving (admin only).

    Auth: admin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, NOT_FOUND.
    """
    ws = _wizard_sessions.pop(wizard_id, None)
    if ws is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "Wizard session not found or expired."},
        )
    return APIResponse(data=None)


# ---------------------------------------------------------------------------
# Wizard helpers (blocking — called via asyncio.to_thread)
# ---------------------------------------------------------------------------


def _wizard_detect_env() -> dict[str, Any]:
    """Detect available providers from environment."""
    from src.setup_wizard import _detect_environment

    return _detect_environment()


def _wizard_load_existing() -> str:
    """Load existing config YAML if found."""
    from src.setup_wizard import _load_existing_config

    content, _path = _load_existing_config()
    return content


def _wizard_load_docs(url: str | None) -> str:
    """Load configuration docs."""
    from src.setup_wizard import _load_docs

    return _load_docs(url)


def _wizard_test_connection(
    provider_type: str, model: str, api_key: str | None, base_url: str | None
) -> Any:
    """Test LLM connection for API wizard. Returns LLM on success, raises on failure.

    Unlike the CLI _test_connection which swallows errors, this version propagates
    the original provider exception so callers can surface the real failure reason.
    """
    from langchain_core.messages import HumanMessage as _HumanMessage

    from src.agent.core import create_llm_from_provider_config
    from src.config import ModelConfig, ProviderConfig

    pc = ProviderConfig(name=provider_type, type=provider_type, api_key=api_key, base_url=base_url)
    mc = ModelConfig(provider=provider_type, model=model)
    llm = create_llm_from_provider_config(pc, mc)
    llm.invoke([_HumanMessage(content="Say 'ok' in one word.")])
    return llm


def _wizard_invoke_llm(llm: Any, messages: list[Any]) -> str:
    """Invoke the LLM with messages. Returns the AI response text."""
    response = llm.invoke(messages)
    return response.content if hasattr(response, "content") else str(response)


async def _wizard_save(
    wizard_id: str, ws: dict[str, Any], request: Request
) -> APIResponse[WizardStepOut]:
    """Extract YAML from conversation, validate, write, and reload config."""
    from src.setup_wizard import _extract_yaml, _has_yaml_block, _mask_secrets

    # Find the last AI message with YAML
    last_yaml_text = ""
    for msg in reversed(ws["messages"]):
        text = msg.content if hasattr(msg, "content") else str(msg)
        if _has_yaml_block(text):
            last_yaml_text = text
            break

    if not last_yaml_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "WIZARD_STEP_ERROR",
                "message": "No YAML configuration found in conversation.",
            },
        )

    warnings: list[str] = []
    try:
        raw_yaml = _extract_yaml(last_yaml_text)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "WIZARD_STEP_ERROR", "message": str(exc)},
        ) from exc

    # Validate and write
    try:
        await asyncio.to_thread(_wizard_validate_and_write, raw_yaml, ws["bootstrap_info"])
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "CONFIG_INVALID", "message": f"Config validation failed: {exc}"},
        ) from exc

    # Reload config into app state
    try:
        from src.config import load_config

        new_cfg = await asyncio.to_thread(load_config)
        request.app.state.config = new_cfg
        log.info("Config reloaded after wizard save")
    except Exception as exc:
        warnings.append(f"Config saved but reload failed: {exc}")

    # Cleanup wizard session
    _wizard_sessions.pop(wizard_id, None)

    return APIResponse(
        data=WizardStepOut(
            wizard_id=wizard_id,
            step=2,
            total_steps=3,
            step_name="Save",
            question=None,
            yaml_preview=_mask_secrets(raw_yaml),
            complete=True,
            warnings=warnings,
        )
    )


def _wizard_validate_and_write(raw_yaml: str, bootstrap_info: dict[str, Any]) -> str:
    """Validate YAML and write config file. Returns the masked YAML preview."""
    import tempfile
    from pathlib import Path

    import yaml

    from src.setup_wizard import _inject_bootstrap

    data = yaml.safe_load(raw_yaml)
    if not isinstance(data, dict):
        raise ValueError("Generated config is not a valid YAML mapping")

    _inject_bootstrap(data, bootstrap_info)

    # Validate via round-trip
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as tmp:
            tmp_path = Path(tmp.name)
            yaml.dump(data, tmp, default_flow_style=False, sort_keys=False)

        from src.config import Config, _apply_config_file

        test_config = Config()
        _apply_config_file(test_config, tmp_path)
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    # Write the final config
    output_path = Path.home() / ".cogtrix.yaml"
    yaml_text = yaml.dump(data, default_flow_style=False, sort_keys=False)
    output_path.write_text(yaml_text, encoding="utf-8")

    return yaml_text
