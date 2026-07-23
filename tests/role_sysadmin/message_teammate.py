"""The ``message_teammate`` tool — the agent's collaboration surface.

The agent-under-test talks to the simulated ops lead by calling
``message_teammate(role, message)``; the harness backs the tool with the
deterministic :class:`~tests.role_sysadmin.personas.PersonaChannel`, so the
lead's reply comes straight back as the tool result. Used for scope questions
and the final hand-off ("done — here's what I changed and verified").
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from tests.role_sysadmin.personas import VALID_ROLES, PersonaChannel

_DESCRIPTION = (
    "Send a message to a teammate and get their reply. Use this to collaborate:\n"
    "- role='lead' — ask the ops lead scope/requirement questions before guessing, "
    "and hand off when you are finished. To finish, send a short report saying the "
    "work is DONE and listing exactly what you changed and how you VERIFIED each "
    "change (the commands you ran and their output).\n"
    "Returns the teammate's reply as text."
)


class MessageTeammateInput(BaseModel):
    """Input schema for ``message_teammate``."""

    role: str = Field(description=f"Who to message: one of {sorted(VALID_ROLES)}.")
    message: str = Field(description="The message to send to the teammate.")


def make_message_teammate_callable(channel: PersonaChannel) -> Any:
    """Return a ``(role, message) -> str`` callable backed by *channel*."""

    def message_teammate(role: str, message: str) -> str:
        """Send a message to a teammate (the ops lead) and return their reply."""
        return channel.message(role, message)

    return message_teammate


def build_message_teammate_tool(channel: PersonaChannel) -> Any:
    """Build the ``message_teammate`` ``StructuredTool`` bound to *channel*."""
    from langchain_core.tools import StructuredTool

    return StructuredTool.from_function(
        func=make_message_teammate_callable(channel),
        name="message_teammate",
        description=_DESCRIPTION,
        args_schema=MessageTeammateInput,
    )
