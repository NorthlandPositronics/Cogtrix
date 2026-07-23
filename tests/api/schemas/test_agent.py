"""Tests for cogtrix_core/api/schemas/agent.py."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cogtrix_core.api.schemas.agent import AgentOut


class TestAgentOut:
    """AgentOut schema construction and validation."""

    def test_agent_out_valid(self) -> None:
        """Valid input constructs without error."""
        agent = AgentOut(
            name="researcher",
            description="A research-focused agent.",
            system_prompt="You are a helpful research assistant.",
            tools_include=["web_search", "read_file"],
            tools_exclude=["shell"],
            model_alias="gpt-4.1-mini",
            memory_mode="conversation",
            max_steps=25,
            temperature=0.7,
        )
        assert agent.name == "researcher"
        assert agent.tools_include == ["web_search", "read_file"]
        assert agent.temperature == 0.7

    def test_agent_out_empty_lists(self) -> None:
        """Empty tool lists are valid."""
        agent = AgentOut(
            name="minimal",
            description="Minimal agent.",
            system_prompt="Be helpful.",
            tools_include=[],
            tools_exclude=[],
            model_alias="gpt-4.1-mini",
            memory_mode="conversation",
            max_steps=10,
            temperature=0.5,
        )
        assert agent.tools_include == []
        assert agent.tools_exclude == []

    def test_agent_out_missing_required_field(self) -> None:
        """Missing required field raises ValidationError."""
        with pytest.raises(ValidationError):
            AgentOut(
                name="researcher",
                description="A research-focused agent.",
                system_prompt="You are a helpful research assistant.",
                tools_include=["web_search"],
                tools_exclude=[],
                model_alias="gpt-4.1-mini",
                memory_mode="conversation",
                # max_steps missing
                temperature=0.7,
            )

    def test_agent_out_negative_max_steps(self) -> None:
        """Negative max_steps is accepted (no ge constraint on AgentOut)."""
        agent = AgentOut(
            name="test",
            description="Test.",
            system_prompt="Test.",
            tools_include=[],
            tools_exclude=[],
            model_alias="gpt-4.1-mini",
            memory_mode="conversation",
            max_steps=-1,
            temperature=0.0,
        )
        assert agent.max_steps == -1

    def test_agent_out_serialization(self) -> None:
        """Serialization produces expected dict."""
        agent = AgentOut(
            name="researcher",
            description="A research-focused agent.",
            system_prompt="You are a helpful research assistant.",
            tools_include=["web_search", "read_file"],
            tools_exclude=["shell"],
            model_alias="gpt-4.1-mini",
            memory_mode="conversation",
            max_steps=25,
            temperature=0.7,
        )
        data = agent.model_dump()
        assert data["name"] == "researcher"
        assert data["max_steps"] == 25
        assert data["temperature"] == 0.7
