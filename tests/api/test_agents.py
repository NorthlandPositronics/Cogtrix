"""Tests for agent endpoints.

Coverage:
  - GET /api/v1/agents returns list of AgentOut with valid auth.
  - GET /api/v1/agents returns 401 without auth.
  - GET /api/v1/agents/{name} returns agent for valid name.
  - GET /api/v1/agents/{name} returns 404 with AGENT_NOT_FOUND for unknown name.
  - Response schemas match APIResponse[list[AgentOut]] / APIResponse[AgentOut].
"""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")

from src.api.auth import create_access_token  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user_token() -> str:
    """Return a valid non-admin JWT for test requests."""
    return create_access_token(
        user_id=str(uuid.uuid4()),
        role="user",
    )


def _make_agent_config(
    name: str,
    description: str = "",
    system_prompt: str = "",
    tools_include: list[str] | None = None,
    tools_exclude: list[str] | None = None,
    model_alias: str = "",
    memory_mode: str = "",
    max_steps: int = 20,
    temperature: float = -1.0,
):
    """Build a minimal AgentConfig-like object for mocking."""
    from src.agent.registry import AgentConfig

    return AgentConfig(
        name=name,
        description=description,
        system_prompt=system_prompt,
        tools_include=tools_include or [],
        tools_exclude=tools_exclude or [],
        model_alias=model_alias,
        memory_mode=memory_mode,
        max_steps=max_steps,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Tests — GET /api/v1/agents
# ---------------------------------------------------------------------------


class TestListAgentsAuth:
    def test_list_agents_returns_401_without_auth(self, client):
        response = client.get("/api/v1/agents")
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "UNAUTHORIZED"

    def test_list_agents_returns_200_with_valid_auth(self, client):
        token = _user_token()
        response = client.get(
            "/api/v1/agents",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200


class TestListAgentsData:
    def test_list_agents_returns_empty_list_when_no_agents(self, client):
        token = _user_token()
        with patch("src.agent.registry.list_agents", return_value=[]):
            response = client.get(
                "/api/v1/agents",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert response.status_code == 200
        data = response.json()["data"]
        assert data == []

    def test_list_agents_returns_agent_list(self, client):
        token = _user_token()
        agents = [
            _make_agent_config(name="alpha", description="Alpha agent"),
            _make_agent_config(name="beta", description="Beta agent"),
        ]
        with patch("src.agent.registry.list_agents", return_value=agents):
            response = client.get(
                "/api/v1/agents",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert response.status_code == 200
        body = response.json()
        assert "data" in body
        data = body["data"]
        assert len(data) == 2
        assert data[0]["name"] == "alpha"
        assert data[0]["description"] == "Alpha agent"
        assert data[1]["name"] == "beta"
        assert data[1]["description"] == "Beta agent"

    def test_list_agents_response_schema(self, client):
        """Verify the response envelope matches APIResponse[list[AgentOut]]."""
        token = _user_token()
        agents = [
            _make_agent_config(
                name="schema-test",
                description="Schema test agent",
                system_prompt="You are a test agent.",
                tools_include=["search_web", "read_file"],
                tools_exclude=["shell"],
                model_alias="gpt-4o",
                memory_mode="hybrid",
                max_steps=15,
                temperature=0.7,
            ),
        ]
        with patch("src.agent.registry.list_agents", return_value=agents):
            response = client.get(
                "/api/v1/agents",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert response.status_code == 200
        body = response.json()
        assert "data" in body
        data = body["data"]
        assert isinstance(data, list)
        assert len(data) == 1

        agent = data[0]
        assert agent["name"] == "schema-test"
        assert agent["description"] == "Schema test agent"
        assert agent["system_prompt"] == "You are a test agent."
        assert agent["tools_include"] == ["search_web", "read_file"]
        assert agent["tools_exclude"] == ["shell"]
        assert agent["model_alias"] == "gpt-4o"
        assert agent["memory_mode"] == "hybrid"
        assert agent["max_steps"] == 15
        assert agent["temperature"] == 0.7

    def test_list_agents_sorted_by_name(self, client):
        """Agents should be returned in the order provided by the registry."""
        token = _user_token()
        agents = [
            _make_agent_config(name="zebra"),
            _make_agent_config(name="apple"),
            _make_agent_config(name="mango"),
        ]
        with patch("src.agent.registry.list_agents", return_value=agents):
            response = client.get(
                "/api/v1/agents",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert response.status_code == 200
        names = [a["name"] for a in response.json()["data"]]
        assert names == ["zebra", "apple", "mango"]


# ---------------------------------------------------------------------------
# Tests — GET /api/v1/agents/{name}
# ---------------------------------------------------------------------------


class TestGetAgentAuth:
    def test_get_agent_returns_401_without_auth(self, client):
        response = client.get("/api/v1/agents/some-agent")
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "UNAUTHORIZED"

    def test_get_agent_returns_200_with_valid_auth(self, client):
        token = _user_token()
        agent = _make_agent_config(name="valid-agent")
        with patch("src.agent.registry.get", return_value=agent):
            response = client.get(
                "/api/v1/agents/valid-agent",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert response.status_code == 200


class TestGetAgentData:
    def test_get_agent_by_name(self, client):
        token = _user_token()
        agent = _make_agent_config(
            name="my-agent",
            description="My special agent",
            system_prompt="Be helpful.",
            model_alias="claude-sonnet",
            max_steps=25,
            temperature=0.5,
        )
        with patch("src.agent.registry.get", return_value=agent):
            response = client.get(
                "/api/v1/agents/my-agent",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert response.status_code == 200
        body = response.json()
        assert "data" in body
        data = body["data"]
        assert data["name"] == "my-agent"
        assert data["description"] == "My special agent"
        assert data["system_prompt"] == "Be helpful."
        assert data["model_alias"] == "claude-sonnet"
        assert data["max_steps"] == 25
        assert data["temperature"] == 0.5

    def test_get_agent_not_found(self, client):
        token = _user_token()
        with patch("src.agent.registry.get", return_value=None):
            response = client.get(
                "/api/v1/agents/nonexistent-agent",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert response.status_code == 404
        body = response.json()
        assert "error" in body
        assert body["error"]["code"] == "AGENT_NOT_FOUND"
        assert "registered" in body["error"]["message"].lower()

    def test_get_agent_response_schema(self, client):
        """Verify the response envelope matches APIResponse[AgentOut]."""
        token = _user_token()
        agent = _make_agent_config(
            name="schema-agent",
            description="Schema validation agent",
            system_prompt="Validate schemas.",
            tools_include=["http_request"],
            tools_exclude=["python_exec", "shell"],
            model_alias="gpt-4o-mini",
            memory_mode="short",
            max_steps=10,
            temperature=0.3,
        )
        with patch("src.agent.registry.get", return_value=agent):
            response = client.get(
                "/api/v1/agents/schema-agent",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert response.status_code == 200
        body = response.json()
        assert "data" in body
        data = body["data"]
        assert data["name"] == "schema-agent"
        assert data["description"] == "Schema validation agent"
        assert data["system_prompt"] == "Validate schemas."
        assert data["tools_include"] == ["http_request"]
        assert data["tools_exclude"] == ["python_exec", "shell"]
        assert data["model_alias"] == "gpt-4o-mini"
        assert data["memory_mode"] == "short"
        assert data["max_steps"] == 10
        assert data["temperature"] == 0.3

    def test_get_agent_with_default_values(self, client):
        """AgentConfig with default values should serialize correctly."""
        token = _user_token()
        agent = _make_agent_config(name="defaults-agent")
        with patch("src.agent.registry.get", return_value=agent):
            response = client.get(
                "/api/v1/agents/defaults-agent",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["name"] == "defaults-agent"
        assert data["description"] == ""
        assert data["system_prompt"] == ""
        assert data["tools_include"] == []
        assert data["tools_exclude"] == []
        assert data["model_alias"] == ""
        assert data["memory_mode"] == ""
        assert data["max_steps"] == 20
        assert data["temperature"] == -1.0

    def test_get_agent_name_with_special_characters(self, client):
        """Agent names with hyphens and underscores should work."""
        token = _user_token()
        agent = _make_agent_config(name="my-special_agent.v2")
        with patch("src.agent.registry.get", return_value=agent):
            response = client.get(
                "/api/v1/agents/my-special_agent.v2",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert response.status_code == 200
        assert response.json()["data"]["name"] == "my-special_agent.v2"

    def test_get_agent_name_url_encoded(self, client):
        """Agent names with spaces should work when URL-encoded."""
        token = _user_token()
        agent = _make_agent_config(name="My Agent")
        with patch("src.agent.registry.get", return_value=agent):
            response = client.get(
                "/api/v1/agents/My%20Agent",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert response.status_code == 200
        assert response.json()["data"]["name"] == "My Agent"
