"""Tests for /tools category grouping logic (issue #299)."""

from __future__ import annotations

from cogtrix import _SECONDARY_CATEGORY_ORDER, _TOOL_CATEGORY_MAP, _categorize_tools


def _groups(tool_names: list[str]) -> dict[str, list[str]]:
    """Return category → tool list mapping from _categorize_tools."""
    return dict(_categorize_tools(tool_names))


def test_git_tools_grouped_under_git():
    """git_* tools appear under 'Git' category, not 'Other'."""
    git_tools = [n for n, cat in _TOOL_CATEGORY_MAP.items() if cat == "Git"]
    assert git_tools, "No Git tools in _TOOL_CATEGORY_MAP"
    groups = _groups(git_tools)
    assert "Git" in groups
    assert set(groups["Git"]) == set(git_tools)
    assert "Other" not in groups


def test_github_tools_grouped_under_github():
    """gh_* tools appear under 'GitHub' category, not 'Other'."""
    gh_tools = [n for n, cat in _TOOL_CATEGORY_MAP.items() if cat == "GitHub"]
    assert gh_tools, "No GitHub tools in _TOOL_CATEGORY_MAP"
    groups = _groups(gh_tools)
    assert "GitHub" in groups
    assert set(groups["GitHub"]) == set(gh_tools)
    assert "Other" not in groups


def test_whatsapp_tools_grouped():
    """whatsapp_* tools appear under 'WhatsApp' category, not 'Other'."""
    wa_tools = [n for n, cat in _TOOL_CATEGORY_MAP.items() if cat == "WhatsApp"]
    assert wa_tools, "No WhatsApp tools in _TOOL_CATEGORY_MAP"
    groups = _groups(wa_tools)
    assert "WhatsApp" in groups
    assert set(groups["WhatsApp"]) == set(wa_tools)
    assert "Other" not in groups


def test_tasks_agents_tools_grouped():
    """Task and agent tools appear under 'Tasks & Agents' category, not 'Other'."""
    ta_tools = [n for n, cat in _TOOL_CATEGORY_MAP.items() if cat == "Tasks & Agents"]
    assert ta_tools, "No 'Tasks & Agents' tools in _TOOL_CATEGORY_MAP"
    groups = _groups(ta_tools)
    assert "Tasks & Agents" in groups
    assert set(groups["Tasks & Agents"]) == set(ta_tools)
    assert "Other" not in groups


def test_other_category_excludes_mapped_tools():
    """No tool in _TOOL_CATEGORY_MAP appears under 'Other'."""
    all_mapped = list(_TOOL_CATEGORY_MAP.keys())
    # Also add a truly unknown tool to ensure "Other" still works when needed
    unknown = "unknown_tool_xyz"
    groups = _groups(all_mapped + [unknown])
    other = set(groups.get("Other", []))
    for tool in all_mapped:
        assert tool not in other, f"{tool!r} should not be in 'Other'"
    # The unknown tool should land in Other
    assert unknown in other


def test_secondary_category_order_after_primary():
    """Secondary categories appear after primary ones in the output."""
    # Mix one primary-category tool with secondary-category tools
    tools = ["search_web", "git_commit", "gh_create_issue", "cron_add"]
    result = _categorize_tools(tools)
    cat_names = [cat for cat, _ in result]
    # Primary category "Search" must come before any secondary categories
    assert "Search" in cat_names
    search_idx = cat_names.index("Search")
    for sec_cat in ["Git", "GitHub", "Cron"]:
        if sec_cat in cat_names:
            assert (
                cat_names.index(sec_cat) > search_idx
            ), f"Secondary category {sec_cat!r} should come after primary 'Search'"


def test_other_omitted_when_empty():
    """'Other' section is absent when all tools are mapped."""
    all_mapped = list(_TOOL_CATEGORY_MAP.keys())
    groups = _groups(all_mapped)
    assert "Other" not in groups


def test_secondary_category_display_order():
    """Secondary categories follow _SECONDARY_CATEGORY_ORDER."""
    tools = list(_TOOL_CATEGORY_MAP.keys())
    result = _categorize_tools(tools)
    cat_names = [cat for cat, _ in result]
    # Extract only secondary categories that appear
    present_secondary = [c for c in cat_names if c in _SECONDARY_CATEGORY_ORDER]
    expected_order = [c for c in _SECONDARY_CATEGORY_ORDER if c in present_secondary]
    assert present_secondary == expected_order
