"""Agent safety tests included in the cogtrix_core/agent coverage suite.

Re-exports the test classes from tests/test_safety.py so they run
when the coverage command includes this file path, contributing to
coverage of cogtrix_core/agent/safety.py.
"""

from tests.test_safety import (  # noqa: F401
    TestConfirmationResult,
    TestCreateSafeToolWrapper,
    TestUserCancelledRun,
)

# ---------------------------------------------------------------------------
# create_safe_tool — additional coverage
# ---------------------------------------------------------------------------


class TestCreateSafeTool:
    def test_creates_tool_with_correct_name(self):
        from cogtrix_core.agent.safety import create_safe_tool

        tool = create_safe_tool(lambda: "ok", name="my_tool", description="does stuff")
        assert tool.name == "my_tool"

    def test_creates_tool_with_correct_description(self):
        from cogtrix_core.agent.safety import create_safe_tool

        tool = create_safe_tool(lambda: "ok", name="t", description="A tool description")
        assert tool.description == "A tool description"

    def test_confirm_false_by_default(self):
        from cogtrix_core.agent.safety import create_safe_tool

        tool = create_safe_tool(lambda: "ok", name="t", description="d")
        assert tool.requires_confirmation is False

    def test_confirm_true_sets_flag(self):
        from cogtrix_core.agent.safety import create_safe_tool

        tool = create_safe_tool(lambda: "ok", name="t", description="d", confirm=True)
        assert tool.requires_confirmation is True
        assert tool.metadata.get("requires_confirmation") is True

    def test_tool_is_callable(self):
        from cogtrix_core.agent.safety import create_safe_tool

        tool = create_safe_tool(lambda x: f"result:{x}", name="t", description="d")
        assert callable(tool.func)
