"""The ``message_teammate`` tool — the agent's collaboration surface.

The agent-under-test talks to the simulated manager / reviewer / QA by calling
``message_teammate(role, message)``; the harness backs the tool with the
deterministic :class:`~tests.role_swe.personas.PersonaChannel`, so the persona's
reply comes straight back as the tool result and the agent can react within the
same run (ask → patch → submit → revise → ship).

``build_message_teammate_tool`` returns a LangChain ``StructuredTool`` wired to a
specific channel; ``make_message_teammate_callable`` exposes the raw callable for
tests that don't need the LangChain wrapper.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from tests.role_swe.personas import VALID_ROLES, PersonaChannel

_DESCRIPTION = (
    "Send a message to a teammate and get their reply. Use this to collaborate:\n"
    "- role='manager' — ask scope/requirement questions before guessing.\n"
    "- role='reviewer' — submit your change for review (say 'ready for review'); "
    "they reply 'Approved' or 'CHANGES_REQUESTED: <points>'. Address every point.\n"
    "- role='qa' — after approval, hand off to QA; they reply 'QA passed' or file a "
    "DEFECT. Fix real defects; if a defect looks spurious, investigate and push back "
    "with evidence.\n"
    "Returns the teammate's reply as text."
)


class MessageTeammateInput(BaseModel):
    """Input schema for ``message_teammate``."""

    role: str = Field(description=f"Who to message: one of {sorted(VALID_ROLES)}.")
    message: str = Field(description="The message to send to the teammate.")


def make_message_teammate_callable(channel: PersonaChannel) -> Any:
    """Return a ``(role, message) -> str`` callable backed by *channel*."""

    def message_teammate(role: str, message: str) -> str:
        """Send a message to a teammate (manager/reviewer/qa) and return their reply."""
        return channel.message(role, message)

    return message_teammate


def build_message_teammate_tool(channel: PersonaChannel) -> Any:
    """Build the ``message_teammate`` ``StructuredTool`` bound to *channel*.

    Args:
        channel: The persona channel the tool routes messages through.

    Returns:
        A LangChain ``StructuredTool`` ready to add to the agent's active tools.
    """
    from langchain_core.tools import StructuredTool

    return StructuredTool.from_function(
        func=make_message_teammate_callable(channel),
        name="message_teammate",
        description=_DESCRIPTION,
        args_schema=MessageTeammateInput,
    )
