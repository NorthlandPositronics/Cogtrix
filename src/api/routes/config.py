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
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

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
    ProviderSwitchRequest,
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


def _provider_to_out(name: str, pc: Any, active_name: str) -> ProviderOut:
    return ProviderOut(
        name=name,
        type=getattr(pc, "type", name),
        base_url=getattr(pc, "base_url", None),
        model=getattr(pc, "model", None),
        has_api_key=bool(getattr(pc, "api_key", None)),
        temperature=getattr(pc, "temperature", None),
        max_tokens=getattr(pc, "max_tokens", None),
        is_active=(name == active_name),
    )


def _model_to_out(alias: str, mc: Any, cfg: Any) -> ModelOut:
    active_model = getattr(cfg, "model", None)
    return ModelOut(
        alias=alias,
        provider=getattr(mc, "provider", ""),
        model_name=getattr(mc, "model", alias),
        num_ctx=getattr(mc, "num_ctx", None),
        temperature=getattr(mc, "temperature", None),
        max_tokens=getattr(mc, "max_tokens", None),
        is_active=(alias == active_model),
    )


def _config_to_out(cfg: Any, raw_yaml: str | None = None) -> ConfigOut:
    providers_out: list[ProviderOut] = []
    active_provider = (getattr(cfg, "provider", "ollama") if cfg else None) or "ollama"
    if cfg is not None:
        cfg_providers = getattr(cfg, "providers", {}) or {}
        for name, pc in cfg_providers.items():
            providers_out.append(_provider_to_out(name, pc, active_provider))
        # Include active provider placeholder if it has no explicit config entry
        if active_provider not in cfg_providers:
            providers_out.insert(
                0,
                ProviderOut(
                    name=active_provider,
                    type=active_provider,
                    base_url=None,
                    model=getattr(cfg, "model", None),
                    has_api_key=False,
                    temperature=None,
                    max_tokens=None,
                    is_active=True,
                ),
            )

    models_out: list[ModelOut] = []
    if cfg is not None:
        for alias, mc in (getattr(cfg, "models", {}) or {}).items():
            models_out.append(_model_to_out(alias, mc, cfg))

    cfg_path = getattr(cfg, "config_file_path", None) if cfg else None

    return ConfigOut(
        provider=active_provider,
        model=getattr(cfg, "model", None) if cfg else None,
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
        from src.config import Config

        # Config() reads from disk; run it off the event loop thread.
        new_cfg = await asyncio.to_thread(Config)
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
    active_name = (getattr(cfg, "provider", "ollama") if cfg else None) or "ollama"
    providers = (getattr(cfg, "providers", {}) if cfg else None) or {}
    if provider_name not in providers:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Provider '{provider_name}' not found."},
        )
    return APIResponse(data=_provider_to_out(provider_name, providers[provider_name], active_name))


@router.post(
    "/provider",
    summary="Switch active provider",
    description=(
        "Switch the globally active LLM provider. "
        "All new sessions and existing sessions that use the global default will use "
        "the new provider from the next agent turn. "
        "Invalidates the LLM bind-tools cache. Admin only."
    ),
    response_model=APIResponse[ConfigOut],
    responses={
        200: {"description": "Provider switched; updated config snapshot returned."},
        401: {"description": "Not authenticated."},
        403: {"description": "Admin required (FORBIDDEN)."},
        404: {"description": "Provider not found (NOT_FOUND)."},
        503: {"description": "Provider unreachable (PROVIDER_UNREACHABLE)."},
    },
)
async def switch_provider(
    body: ProviderSwitchRequest,
    request: Request,
    current_user: TokenData = Depends(require_admin),
) -> APIResponse[ConfigOut]:
    """Switch the globally active LLM provider (admin only).

    Auth: admin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, NOT_FOUND, PROVIDER_UNREACHABLE.
    """
    cfg = _get_config(request)
    providers = (getattr(cfg, "providers", {}) if cfg else None) or {}
    _known_types = {"ollama", "openai", "anthropic", "google"}
    if body.provider not in providers and body.provider not in _known_types:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Provider '{body.provider}' not found."},
        )
    if cfg is not None:
        cfg.provider = body.provider
    try:
        from src.orchestration.runner import invalidate_llm_caches

        invalidate_llm_caches()
    except Exception as exc:
        log.debug("invalidate_llm_caches: %s", exc)
    raw_yaml = await _read_raw_yaml(cfg)
    return APIResponse(data=_config_to_out(cfg, raw_yaml=raw_yaml))


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
        from src.providers import create_chat_model_from_config

        # create_chat_model_from_config may perform network I/O (e.g. Ollama API
        # introspection); run it off the event loop thread to prevent stalls.
        await asyncio.to_thread(create_chat_model_from_config, pc)
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
        cfg.model = body.model
    try:
        from src.orchestration.runner import invalidate_llm_caches

        invalidate_llm_caches()
    except Exception as exc:
        log.debug("invalidate_llm_caches: %s", exc)
    raw_yaml = await _read_raw_yaml(cfg)
    return APIResponse(data=_config_to_out(cfg, raw_yaml=raw_yaml))


# ---------------------------------------------------------------------------
# Setup wizard — REST adaptation is non-trivial; return 501 for now
# ---------------------------------------------------------------------------


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
    current_user: TokenData = Depends(require_admin),
) -> APIResponse[WizardStepOut]:
    """Start a setup wizard session and return the first step (admin only).

    Auth: admin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={
            "code": "NOT_IMPLEMENTED",
            "message": "Setup wizard REST API is not yet implemented.",
        },
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
    current_user: TokenData = Depends(require_admin),
) -> APIResponse[WizardStepOut]:
    """Advance the wizard one step and return the next question or completion (admin only).

    Auth: admin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, NOT_FOUND, WIZARD_STEP_ERROR, CONFIG_INVALID.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={
            "code": "NOT_IMPLEMENTED",
            "message": "Setup wizard REST API is not yet implemented.",
        },
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
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={
            "code": "NOT_IMPLEMENTED",
            "message": "Setup wizard REST API is not yet implemented.",
        },
    )
