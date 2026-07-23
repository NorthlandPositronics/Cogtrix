"""Pydantic output schemas for the agents API."""

from __future__ import annotations

from pydantic import BaseModel


class AgentOut(BaseModel):
    """Serialized representation of a named agent configuration."""

    name: str
    description: str
    system_prompt: str
    tools_include: list[str]
    tools_exclude: list[str]
    model_alias: str
    memory_mode: str
    max_steps: int
    temperature: float
