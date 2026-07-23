"""Configuration, provider, model management, and setup wizard endpoints.

Endpoints:
    GET    /api/v1/config                   — view current config snapshot
    PATCH  /api/v1/config                   — partial runtime config update
    POST   /api/v1/config/reload            — reload config from disk (admin)
    GET    /api/v1/config/providers         — list configured providers
    GET    /api/v1/config/providers/{name}  — get provider details
    POST   /api/v1/config/providers         — add a new provider (admin)
    PATCH  /api/v1/config/providers/{name}  — update provider api_key / base_url (admin)
    DELETE /api/v1/config/providers/{name}  — remove a provider (admin)
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
import pathlib
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
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
    ProviderCreateRequest,
    ProviderHealthOut,
    ProviderOut,
    ProviderPatchRequest,
    WizardStartRequest,
    WizardStepOut,
    WizardStepRequest,
)

log = logging.getLogger("cogtrix.api.config")

router = APIRouter(prefix="/config", tags=["Configuration"])

_WIZARD_LLM_TIMEOUT_SECONDS = 60


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
        delegate_enabled=bool(getattr(cfg, "delegate_enabled", True) if cfg else True),
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
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
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


def _write_providers_to_config(cfg: Any, providers_dict: dict[str, Any]) -> None:
    """Persist an updated providers dict to the config YAML file.

    Reads the current file, replaces the ``providers`` section, and writes back.
    Uses a temporary file for atomicity.  Raises on any I/O or parse error.
    """
    import yaml

    cfg_path = getattr(cfg, "config_file_path", None)
    if cfg_path is None:
        # No config file was loaded — the application is running from defaults or
        # environment variables only.  Silently writing to ~/.cogtrix.yaml would
        # create a file at the wrong location and diverge from the active config on
        # the next restart (BUG-243).  Callers must surface this as 503.
        raise RuntimeError(
            "No config file is loaded; provider changes cannot be persisted. "
            "Run the setup wizard to create a config file first."
        )

    cfg_path = pathlib.Path(cfg_path)

    try:
        data: dict = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        data = {}

    # Rebuild providers section preserving other config keys
    serialised: dict[str, Any] = {}
    for name, pc in providers_dict.items():
        entry: dict[str, Any] = {"type": getattr(pc, "type", name)}
        base_url = getattr(pc, "base_url", None)
        if base_url:
            entry["base_url"] = base_url
        api_key = getattr(pc, "api_key", None)
        if api_key:
            entry["api_key"] = api_key
        tool_instructions = getattr(pc, "tool_instructions", None)
        if tool_instructions:
            entry["tool_instructions"] = tool_instructions
        serialised[name] = entry

    data["providers"] = serialised

    tmp_path: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            dir=cfg_path.parent,
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp_path = pathlib.Path(tmp.name)
            yaml.dump(data, tmp, default_flow_style=False, sort_keys=False)
        tmp_path.replace(cfg_path)
        tmp_path = None
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


@router.post(
    "/providers",
    summary="Add a new provider",
    description=(
        "Add a new LLM provider entry to the config file at runtime. "
        "The change is persisted to disk and the in-memory config is reloaded immediately. "
        "Admin only."
    ),
    response_model=APIResponse[ProviderOut],
    status_code=201,
    responses={
        201: {"description": "Provider created."},
        401: {"description": "Not authenticated."},
        403: {"description": "Admin required (FORBIDDEN)."},
        409: {"description": "Provider name already exists (PROVIDER_EXISTS)."},
        422: {"description": "Validation error (VALIDATION_ERROR)."},
    },
)
async def create_provider(
    body: ProviderCreateRequest,
    request: Request,
    current_user: TokenData = Depends(require_admin),
) -> APIResponse[ProviderOut]:
    """Add a new provider to the config (admin only).

    Auth: admin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, PROVIDER_EXISTS, VALIDATION_ERROR.
    """
    from src.config import ProviderConfig
    from src.setup_wizard import _is_safe_ollama_url

    # SSRF guard: reject link-local / reserved base_url values (BUG-238).
    # Private RFC-1918 addresses are allowed (local Ollama, vLLM, LiteLLM).
    base_url = body.base_url or None
    if base_url is not None and not _is_safe_ollama_url(base_url, allow_private=True):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "code": "VALIDATION_ERROR",
                "message": "base_url resolves to a blocked address (link-local, reserved, or unspecified).",
            },
        )

    cfg = _get_config(request)

    # Serialise concurrent provider mutations to prevent TOCTOU on the config
    # file (read → modify → write race — BUG-237).
    async with _get_provider_write_lock():
        providers = dict((getattr(cfg, "providers", {}) or {}) if cfg else {})

        if body.name in providers:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "PROVIDER_EXISTS",
                    "message": f"Provider '{body.name}' already exists.",
                },
            )

        try:
            new_pc = ProviderConfig(
                name=body.name,
                type=body.type,
                base_url=base_url,
                api_key=body.api_key or None,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={"code": "VALIDATION_ERROR", "message": str(exc)},
            ) from exc

        providers[body.name] = new_pc

        try:
            await asyncio.to_thread(_write_providers_to_config, cfg, providers)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"code": "SERVICE_UNAVAILABLE", "message": str(exc)},
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"code": "INTERNAL_ERROR", "message": f"Failed to write config: {exc}"},
            ) from exc

        # Apply to in-memory config
        if cfg is not None and hasattr(cfg, "providers"):
            cfg.providers[body.name] = new_pc

    return APIResponse(data=_provider_to_out(body.name, new_pc))


@router.patch(
    "/providers/{provider_name}",
    summary="Update a provider",
    description=(
        "Update the base_url and/or api_key of an existing provider. "
        "Changes are persisted to disk and the in-memory config is updated immediately. "
        "Useful for rotating API keys without re-running the wizard. Admin only."
    ),
    response_model=APIResponse[ProviderOut],
    responses={
        200: {"description": "Provider updated."},
        401: {"description": "Not authenticated."},
        403: {"description": "Admin required (FORBIDDEN)."},
        404: {"description": "Provider not found (NOT_FOUND)."},
    },
)
async def update_provider(
    provider_name: str,
    body: ProviderPatchRequest,
    request: Request,
    current_user: TokenData = Depends(require_admin),
) -> APIResponse[ProviderOut]:
    """Update an existing provider's connection details (admin only).

    Auth: admin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, NOT_FOUND.
    """
    from src.config import ProviderConfig
    from src.setup_wizard import _is_safe_ollama_url

    # SSRF guard on the incoming base_url (BUG-238).
    if body.base_url is not None:
        new_base_url_candidate = body.base_url or None
        if new_base_url_candidate is not None and not _is_safe_ollama_url(
            new_base_url_candidate, allow_private=True
        ):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "code": "VALIDATION_ERROR",
                    "message": "base_url resolves to a blocked address (link-local, reserved, or unspecified).",
                },
            )

    cfg = _get_config(request)

    async with _get_provider_write_lock():  # BUG-237
        providers = dict((getattr(cfg, "providers", {}) or {}) if cfg else {})

        if provider_name not in providers:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "NOT_FOUND", "message": f"Provider '{provider_name}' not found."},
            )

        pc = providers[provider_name]
        changed = False
        new_base_url = getattr(pc, "base_url", None)
        new_api_key = getattr(pc, "api_key", None)
        new_tool_instructions = getattr(pc, "tool_instructions", None)

        if body.base_url is not None:
            new_base_url = body.base_url or None
            changed = True
        if body.api_key is not None:
            new_api_key = body.api_key or None
            changed = True

        if changed:
            pc = ProviderConfig(
                name=provider_name,
                type=getattr(pc, "type", provider_name),
                base_url=new_base_url,
                api_key=new_api_key,
                tool_instructions=new_tool_instructions,
            )
            providers[provider_name] = pc

            try:
                await asyncio.to_thread(_write_providers_to_config, cfg, providers)
            except RuntimeError as exc:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail={"code": "SERVICE_UNAVAILABLE", "message": str(exc)},
                ) from exc
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={"code": "INTERNAL_ERROR", "message": f"Failed to write config: {exc}"},
                ) from exc

            if cfg is not None and hasattr(cfg, "providers"):
                cfg.providers[provider_name] = pc

    return APIResponse(data=_provider_to_out(provider_name, pc))


@router.delete(
    "/providers/{provider_name}",
    summary="Remove a provider",
    description=(
        "Remove a provider from the config file. "
        "Fails with 409 if any active model references this provider. "
        "Admin only."
    ),
    response_model=APIResponse[None],
    responses={
        200: {"description": "Provider removed."},
        401: {"description": "Not authenticated."},
        403: {"description": "Admin required (FORBIDDEN)."},
        404: {"description": "Provider not found (NOT_FOUND)."},
        409: {"description": "Provider has an active model (PROVIDER_IN_USE)."},
    },
)
async def delete_provider(
    provider_name: str,
    request: Request,
    current_user: TokenData = Depends(require_admin),
) -> APIResponse[None]:
    """Remove a provider from the config (admin only).

    Auth: admin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, NOT_FOUND, PROVIDER_IN_USE.
    """
    cfg = _get_config(request)

    async with _get_provider_write_lock():  # BUG-237
        providers = dict((getattr(cfg, "providers", {}) or {}) if cfg else {})

        if provider_name not in providers:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "NOT_FOUND", "message": f"Provider '{provider_name}' not found."},
            )

        # Guard: refuse if any registered model references this provider
        models = (getattr(cfg, "models", {}) or {}) if cfg else {}
        referencing = [
            alias for alias, mc in models.items() if getattr(mc, "provider", None) == provider_name
        ]
        if referencing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "PROVIDER_IN_USE",
                    "message": (
                        f"Provider '{provider_name}' is referenced by model(s): "
                        + ", ".join(referencing)
                        + ". Remove or reassign those models first."
                    ),
                },
            )

        del providers[provider_name]

        try:
            await asyncio.to_thread(_write_providers_to_config, cfg, providers)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"code": "SERVICE_UNAVAILABLE", "message": str(exc)},
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"code": "INTERNAL_ERROR", "message": f"Failed to write config: {exc}"},
            ) from exc

        if cfg is not None and hasattr(cfg, "providers"):
            cfg.providers.pop(provider_name, None)

    return APIResponse(data=None)


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
            base_url=getattr(pc, "base_url", None),
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


# Serialises concurrent provider CRUD operations to prevent TOCTOU races on
# the config file (read-modify-write) between concurrent admin requests (BUG-237).
_provider_write_lock: asyncio.Lock | None = None
_provider_write_lock_guard = threading.Lock()


def _get_provider_write_lock() -> asyncio.Lock:
    """Return the module-level provider write lock.

    Uses double-checked locking with a ``threading.Lock`` guard to prevent
    the TOCTOU race that occurs when two concurrent requests both see
    ``_provider_write_lock is None`` and create different locks (BUG-237).
    """
    global _provider_write_lock
    if _provider_write_lock is None:
        with _provider_write_lock_guard:
            if _provider_write_lock is None:
                _provider_write_lock = asyncio.Lock()
    return _provider_write_lock


# ---------------------------------------------------------------------------
# Setup wizard — in-memory session store with 3-step flow
# ---------------------------------------------------------------------------


_wizard_sessions: dict[str, dict[str, Any]] = {}
_wizard_sessions_lock: asyncio.Lock | None = None
_wizard_sessions_lock_guard: threading.Lock | None = None
_WIZARD_TTL = 1800  # 30 minutes
# Shown when the first LLM call fails or returns no content (provider soft-fail).
_WIZARD_DEFAULT_FIRST_QUESTION = (
    "Welcome! I'll help you configure Cogtrix. "
    "What would you like to set up? For example: tools, memory settings, "
    "assistant mode, or any other option from the documentation."
)


def _get_wizard_sessions_lock() -> asyncio.Lock:
    """Return (lazily initialising) the module-level wizard sessions lock."""
    global _wizard_sessions_lock, _wizard_sessions_lock_guard
    if _wizard_sessions_lock_guard is None:
        _wizard_sessions_lock_guard = threading.Lock()
    if _wizard_sessions_lock is None:
        with _wizard_sessions_lock_guard:
            if _wizard_sessions_lock is None:
                _wizard_sessions_lock = asyncio.Lock()
    return _wizard_sessions_lock


def _get_wizard(wizard_id: str) -> dict[str, Any] | None:
    """Return wizard session or None if expired/missing.

    Caller must hold ``_get_wizard_sessions_lock()`` when reading and mutating
    wizard state to prevent concurrent advance_wizard calls from corrupting
    the conversation history (BUG-239).
    """
    session = _wizard_sessions.get(wizard_id)
    if session is None:
        return None
    if time.monotonic() - session["created_mono"] > _WIZARD_TTL:
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

    # Always load existing config — used both to pre-populate answers when editing
    # and to resolve stored API keys when reconnecting to an already-configured provider.
    existing_yaml = await asyncio.to_thread(_wizard_load_existing)

    async with _get_wizard_sessions_lock():
        _wizard_sessions[wid] = {
            "created_mono": time.monotonic(),
            "step": 0,
            "env": env,
            "existing_yaml": existing_yaml,
            "bootstrap_info": None,
            "llm": None,
            "messages": [],
            "docs_url": body.docs_url,
            # Per-session lock: serialises concurrent advance_wizard calls to prevent
            # duplicate LLM invocations and message history corruption (BUG-239).
            "lock": asyncio.Lock(),
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
    async with _get_wizard_sessions_lock():
        ws = _get_wizard(wizard_id)
        if ws is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "NOT_FOUND", "message": "Wizard session not found or expired."},
            )

        # Serialise concurrent advance_wizard calls for the same session to prevent
        # duplicate LLM calls and message history corruption (BUG-239).
        async with ws["lock"]:
            # Re-check after acquiring the lock — a concurrent call may have
            # expired or completed the session while we were waiting.
            ws = _get_wizard(wizard_id)
            if ws is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"code": "NOT_FOUND", "message": "Wizard session not found or expired."},
                )

            return await _advance_wizard_locked(wizard_id, ws, body, request)


async def _advance_wizard_locked(
    wizard_id: str,
    ws: dict[str, Any],
    body: WizardStepRequest,
    request: Request,
) -> APIResponse[WizardStepOut]:
    """Inner wizard step handler; called while holding ``ws['lock']``."""
    step = ws["step"]

    # ── Step 0: Connect to LLM ──
    if step == 0:
        data = body.data or {}
        provider_type = data.get("provider_type", "openai")
        api_key = data.get("api_key")
        base_url = data.get("base_url")
        model = data.get("model")
        provider_name = data.get("provider_name", provider_type)

        # Resolve presets/aliases (e.g. "groq", "xai") → native type + concrete base_url/key
        native_type, base_url, api_key = _resolve_wizard_provider(
            provider_type, api_key, base_url, ws.get("env") or {}
        )

        # If no api_key was submitted, try to resolve one from the existing config
        # (user is re-configuring and the key is already stored — no need to re-enter).
        if not api_key and ws.get("existing_yaml"):
            api_key = _resolve_api_key_from_existing(
                ws["existing_yaml"], provider_name=provider_name, base_url=base_url
            )

        if not model:
            from src.providers import get_default_model
            from src.providers.defaults import OPENAI_PRESETS

            preset_model = OPENAI_PRESETS.get(provider_type, {}).get("model")
            model = preset_model or get_default_model(native_type)

        # Test connection off the event loop
        try:
            llm, probe_warning = await asyncio.to_thread(
                _wizard_test_connection, native_type, model, api_key, base_url
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "code": "PROVIDER_UNREACHABLE",
                    "message": str(exc),
                },
            ) from exc

        ws["bootstrap_info"] = {
            "provider": provider_name,
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
            "type": native_type,
        }
        ws["llm"] = llm
        ws["probe_warning"] = probe_warning

        # Build system prompt and start conversation
        docs = await asyncio.to_thread(_wizard_load_docs, ws.get("docs_url"))
        from src.setup_wizard import (
            _WIZARD_SYSTEM_PROMPT,
            _index_docs,
            _sanitize_yaml_for_prompt,
        )

        # The API wizard does not offer a separate production model step; the
        # bootstrap model is also the active (production) model.
        production_context = f"Same as bootstrap: use {provider_name} / {model} as models.default."
        _raw_existing = ws["existing_yaml"]
        _safe_existing = (
            _sanitize_yaml_for_prompt(_raw_existing)
            if _raw_existing
            else "No existing configuration."
        )
        system_prompt = _WIZARD_SYSTEM_PROMPT.substitute(
            existing_config=_safe_existing,
            bootstrap_provider=provider_name,
            bootstrap_type=provider_type,
            bootstrap_base_url=base_url or "(default)",
            bootstrap_model=model,
            bootstrap_has_key="yes" if api_key else "no",
            production_context=production_context,
        )
        ws["docs_index"] = _index_docs(docs)

        # Strict OpenAI-compatible backends (vLLM, LiteLLM) reject a messages
        # list that contains only a SystemMessage — seed with HumanMessage("Start.")
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        messages: list[Any] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Start."),
        ]
        try:
            ai_text = await asyncio.to_thread(_wizard_invoke_llm, llm, messages)
        except Exception as exc:
            # Phase 1 already hard-fails on genuine misconfiguration (SDK init error).
            # If we reach here, the provider is reachable but the initial Q&A call failed
            # (e.g. context size overflow, transient error, or a flaky provider flagged by
            # the probe).  In all cases, fall back to the default question so the wizard
            # can advance — the user can still configure even with a limited-context model
            # (BUG-242, BUG-244).
            log.warning(
                "Wizard initial LLM call failed, using default question: %s",
                exc,
            )
            ai_text = ""  # falls through to default-question assignment below
        if not ai_text:
            ai_text = _WIZARD_DEFAULT_FIRST_QUESTION
        messages.append(AIMessage(content=ai_text))
        ws["messages"] = messages
        ws["step"] = 1

        warnings: list[str] = []
        if probe_warning:
            warnings.append(
                f"Connection probe returned a warning — provider may be unstable: {probe_warning}"
            )

        return APIResponse(
            data=WizardStepOut(
                wizard_id=wizard_id,
                step=1,
                total_steps=3,
                step_name="Configure",
                question=ai_text,
                yaml_preview=None,
                complete=False,
                warnings=warnings,
            )
        )

    # ── Step 1: Configure (Q&A loop) ──
    if step == 1:
        from langchain_core.messages import AIMessage, HumanMessage

        user_answer = body.answer or ""
        if not user_answer.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
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
        docs_index: dict[str, str] = ws.get("docs_index") or {}
        if docs_index:
            from src.setup_wizard import _retrieve_relevant_sections

            relevant = _retrieve_relevant_sections(user_answer, docs_index)
            if relevant:
                user_answer = (
                    f"[Relevant documentation]\n{relevant}\n\n[User question]\n{user_answer}"
                )
        messages.append(HumanMessage(content=user_answer))

        llm = ws["llm"]
        try:
            ai_text = await asyncio.to_thread(_wizard_invoke_llm, llm, messages)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
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
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
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
    async with _get_wizard_sessions_lock():
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


def _resolve_api_key_from_existing(
    existing_yaml: str, *, provider_name: str | None, base_url: str | None
) -> str | None:
    """Return the api_key from the existing config for a matching provider.

    Matches by provider name first, then by base_url.  Returns None if nothing
    is found so the caller can fall back gracefully.
    """
    try:
        import yaml as _yaml

        cfg = _yaml.safe_load(existing_yaml) or {}
        providers: dict[str, Any] = cfg.get("providers", {}) or {}
        # Match by name
        if provider_name and provider_name in providers:
            key = providers[provider_name].get("api_key")
            if key:
                return str(key)
        # Match by base_url
        if base_url:
            for entry in providers.values():
                if isinstance(entry, dict) and entry.get("base_url") == base_url:
                    key = entry.get("api_key")
                    if key:
                        return str(key)
    except Exception:
        pass
    return None


def _resolve_wizard_provider(
    provider_name: str,
    api_key: str | None,
    base_url: str | None,
    env: dict[str, str],
) -> tuple[str, str | None, str | None]:
    """Resolve a wizard provider name to ``(native_type, base_url, api_key)``.

    Resolution order:
    1. Native provider type (openai, ollama, anthropic, google) → unchanged.
    2. OpenAI-compatible preset (xai, groq) → type="openai" + preset base_url/key.
    3. Configured alias in the running config → use the alias's ProviderConfig.
    4. Fallback → return as-is (ProviderConfig validation will raise later).
    """
    from src.providers import PROVIDER_TYPES
    from src.providers.defaults import OPENAI_PRESETS

    # 1. Already a native type
    if provider_name in PROVIDER_TYPES:
        return provider_name, base_url, api_key

    # 2. OpenAI-compatible preset
    if provider_name in OPENAI_PRESETS:
        preset = OPENAI_PRESETS[provider_name]
        resolved_base_url = base_url or preset["base_url"]
        env_key = preset.get("env_key", "")
        resolved_key = api_key or (env.get(env_key) if env_key else None)
        return "openai", resolved_base_url, resolved_key

    # 3. Configured alias in the running config
    try:
        from src.config import Config

        cfg = Config()
        providers = cfg.providers or {}
        if provider_name in providers:
            pc = providers[provider_name]
            return pc.type, base_url or pc.base_url, api_key or pc.api_key
    except Exception:
        pass

    # 4. Fallback — return unchanged (will raise at ProviderConfig.__post_init__)
    return provider_name, base_url, api_key


def _wizard_test_connection(
    provider_type: str, model: str, api_key: str | None, base_url: str | None
) -> tuple[Any, str | None]:
    """Test LLM connection for API wizard. Returns (llm, probe_warning) on success, raises on failure.

    Two-phase test:
    - Phase 1 (hard fail): LLM object creation. A failure here means the config
      itself is invalid (bad provider type, malformed base_url, SDK init error).
      The exception propagates so the caller returns 422 PROVIDER_UNREACHABLE.
    - Phase 2 (soft fail): a live "Say ok" probe. Some valid providers return
      transient or setup-related errors (e.g. 'No connected db.') for cold pings
      while working fine for real conversations. A probe failure is logged and
      returned as ``probe_warning`` so callers can surface it to the user — but
      the LLM object is still returned so the wizard step can attempt the real
      initial Q&A call and raise a proper error if that also fails.
    """
    import logging as _logging

    from langchain_core.messages import HumanMessage as _HumanMessage

    from src.agent.core import create_llm_from_provider_config
    from src.config import ModelConfig, ProviderConfig

    # Phase 1: config/SDK validation — hard fail
    pc = ProviderConfig(name=provider_type, type=provider_type, api_key=api_key, base_url=base_url)
    mc = ModelConfig(provider=provider_type, model=model)
    llm = create_llm_from_provider_config(pc, mc)

    # Phase 2: live probe — soft fail; capture warning for callers to surface
    probe_warning: str | None = None
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(llm.invoke, [_HumanMessage(content="Say 'ok' in one word.")])
            try:
                future.result(timeout=_WIZARD_LLM_TIMEOUT_SECONDS)
            except FuturesTimeoutError:
                _logging.getLogger("cogtrix.api").warning(
                    "Wizard connection probe timed out after %ds (proceeding anyway)",
                    _WIZARD_LLM_TIMEOUT_SECONDS,
                )
                probe_warning = f"timeout after {_WIZARD_LLM_TIMEOUT_SECONDS}s"
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
    except Exception as exc:
        probe_warning = str(exc)
        _logging.getLogger("cogtrix.api").warning(
            "Wizard connection probe returned an error (proceeding anyway): %s", exc
        )

    return llm, probe_warning


def _wizard_invoke_llm(llm: Any, messages: list[Any]) -> str:
    """Invoke the LLM with messages. Returns the AI response text (never None)."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(llm.invoke, messages)
        try:
            response = future.result(timeout=_WIZARD_LLM_TIMEOUT_SECONDS)
        except FuturesTimeoutError:
            log.warning("Wizard LLM invoke timed out after %ds", _WIZARD_LLM_TIMEOUT_SECONDS)
            return ""
        except Exception:
            return ""
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
    content = response.content if hasattr(response, "content") else str(response)
    return content or ""


async def _wizard_save(
    wizard_id: str, ws: dict[str, Any], request: Request
) -> APIResponse[WizardStepOut]:
    """Extract YAML from conversation, validate, write, and reload config.

    .. note::
        The caller must already hold ``_get_wizard_sessions_lock()`` when
        calling this function.  Re-acquiring the lock here would deadlock
        because ``asyncio.Lock`` is not reentrant (BUG-239).
    """
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
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
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
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"code": "WIZARD_STEP_ERROR", "message": str(exc)},
        ) from exc

    # Validate and write
    try:
        await asyncio.to_thread(_wizard_validate_and_write, raw_yaml, ws["bootstrap_info"])
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
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

    # Cleanup wizard session — caller already holds _get_wizard_sessions_lock().
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

    from src.setup_wizard import _inject_bootstrap, _strip_nulls

    data = yaml.safe_load(raw_yaml)
    if not isinstance(data, dict):
        raise ValueError("Generated config is not a valid YAML mapping")

    _inject_bootstrap(data, bootstrap_info)
    # Remove None values and empty dicts so the round-trip validation and the
    # written config are clean — identical to what the CLI wizard does after
    # inject_bootstrap (BUG-239).
    data = _strip_nulls(data)

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

    # Write the final config atomically — write to a sibling temp file then
    # rename so a crash mid-write never leaves a half-written config.
    output_path = Path.home() / ".cogtrix.yaml"
    yaml_text = yaml.dump(data, default_flow_style=False, sort_keys=False)
    write_tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            dir=output_path.parent,
            delete=False,
            encoding="utf-8",
        ) as wt:
            write_tmp = Path(wt.name)
            wt.write(yaml_text)
        write_tmp.replace(output_path)
        write_tmp = None
    finally:
        if write_tmp is not None:
            try:
                write_tmp.unlink(missing_ok=True)
            except OSError:
                pass

    return yaml_text
