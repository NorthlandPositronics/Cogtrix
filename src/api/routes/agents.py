"""Named agent configuration endpoints.

Endpoints:
    GET  /api/v1/agents          — list all registered agents
    GET  /api/v1/agents/{name}   — get a single agent by name
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.auth import TokenData, get_current_user
from src.api.schemas.agent import AgentOut
from src.api.schemas.common import APIResponse

log = logging.getLogger("cogtrix.api.agents")

router = APIRouter(prefix="/agents", tags=["Agents"])


def _agent_config_to_out(agent: object) -> AgentOut:
    return AgentOut(
        name=str(getattr(agent, "name", "")),
        description=str(getattr(agent, "description", "")),
        system_prompt=str(getattr(agent, "system_prompt", "")),
        tools_include=list(getattr(agent, "tools_include", []) or []),
        tools_exclude=list(getattr(agent, "tools_exclude", []) or []),
        model_alias=str(getattr(agent, "model_alias", "")),
        memory_mode=str(getattr(agent, "memory_mode", "")),
        max_steps=int(getattr(agent, "max_steps", 20)),
        temperature=float(getattr(agent, "temperature", -1.0)),
    )


@router.get(
    "",
    summary="List registered agents",
    description="Return all named agent configurations loaded from the config file and AGENTS.md.",
    response_model=APIResponse[list[AgentOut]],
    responses={
        200: {"description": "Agent list returned."},
        401: {"description": "Not authenticated."},
    },
)
async def list_agents(
    current_user: TokenData = Depends(get_current_user),
) -> APIResponse[list[AgentOut]]:
    """List all named agents.

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED.
    """
    from src.agent import registry as _reg

    agents = [_agent_config_to_out(a) for a in _reg.list_agents()]
    return APIResponse(data=agents)


@router.get(
    "/{agent_name}",
    summary="Get agent by name",
    description="Return the configuration for a single named agent.",
    response_model=APIResponse[AgentOut],
    responses={
        200: {"description": "Agent returned."},
        401: {"description": "Not authenticated."},
        404: {"description": "Agent not found (AGENT_NOT_FOUND)."},
    },
)
async def get_agent(
    agent_name: str,
    current_user: TokenData = Depends(get_current_user),
) -> APIResponse[AgentOut]:
    """Return a single agent configuration by name.

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, AGENT_NOT_FOUND.
    """
    from src.agent import registry as _reg

    agent = _reg.get(agent_name)
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "AGENT_NOT_FOUND",
                "message": "No agent with that name is registered.",
            },
        )
    return APIResponse(data=_agent_config_to_out(agent))
