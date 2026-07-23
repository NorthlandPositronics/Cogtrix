"""Unit tests for classify_task_complexity() in src.orchestration.intent."""

from __future__ import annotations

import pytest

from src.orchestration.intent import TaskComplexity, classify_task_complexity

# ---------------------------------------------------------------------------
# COMPLEX_ACTION
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt",
    [
        "build binutils from source in a container",
        "compile and install gcc locally",
        "deploy the server using docker",
        "set up a CI/CD pipeline for the project",
        "bootstrap the toolchain from source",
        "migrate the database schema",
        "provision the server with the new configuration",
        "install cmake and configure the build",
    ],
)
def test_complex_action_detected(prompt: str) -> None:
    assert (
        classify_task_complexity(prompt) == TaskComplexity.COMPLEX_ACTION
    ), f"Expected COMPLEX_ACTION for: {repr(prompt)}"


# ---------------------------------------------------------------------------
# COMPLEX_RESEARCH
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt",
    [
        "provide a holistic analysis of the current AI landscape",
        "compare and contrast from multiple perspectives the Python frameworks",
        "write a comprehensive review of all aspects of cloud providers",
        "in-depth analysis of the market dynamics",
        "thorough investigation into the root causes of climate change",
        "please give an in-depth research into renewable energy",
    ],
)
def test_complex_research_detected(prompt: str) -> None:
    assert (
        classify_task_complexity(prompt) == TaskComplexity.COMPLEX_RESEARCH
    ), f"Expected COMPLEX_RESEARCH for: {repr(prompt)}"


# ---------------------------------------------------------------------------
# SIMPLE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt",
    [
        "hello",
        "what is 2+2",
        "hi there",
        "thanks",
        "who are you",
    ],
)
def test_simple_not_misclassified(prompt: str) -> None:
    assert (
        classify_task_complexity(prompt) == TaskComplexity.SIMPLE
    ), f"Expected SIMPLE for: {repr(prompt)}"


# ---------------------------------------------------------------------------
# MODERATE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt",
    [
        "search the web for latest news about AI",
        "read the file and summarize it",
        "explain Python decorators",
        "what is a binary search tree",
        "list the files in the current directory",
    ],
)
def test_moderate_default(prompt: str) -> None:
    assert (
        classify_task_complexity(prompt) == TaskComplexity.MODERATE
    ), f"Expected MODERATE for: {repr(prompt)}"


# ---------------------------------------------------------------------------
# Proximity guard — verb and target too far apart must NOT be COMPLEX_ACTION
# ---------------------------------------------------------------------------


def test_action_proximity_guard_list_ideas() -> None:
    """'build a list of ideas for ...' should not be COMPLEX_ACTION.

    'build' is the verb and 'container' is the target but they are > 80 chars
    apart in this sentence, so the proximity guard must suppress the match.
    """
    prompt = (
        "build a list of ideas for improving user experience on the landing page "
        "and then describe the container we should use"
    )
    result = classify_task_complexity(prompt)
    assert (
        result != TaskComplexity.COMPLEX_ACTION
    ), f"Proximity guard failed — expected NOT COMPLEX_ACTION for: {repr(prompt)}, got {result}"


def test_action_proximity_guard_verb_and_target_distant() -> None:
    """Verb and target separated by more than 80 characters should not fire."""
    # 'build' near the start; 'docker' far away after 90+ chars of filler
    filler = "a" * 90
    prompt = f"build {filler} docker"
    result = classify_task_complexity(prompt)
    assert (
        result != TaskComplexity.COMPLEX_ACTION
    ), f"Proximity guard failed — expected NOT COMPLEX_ACTION, got {result}"


def test_action_proximity_guard_close_pair_fires() -> None:
    """Verb and target within 80 chars must still be detected as COMPLEX_ACTION."""
    prompt = "compile the compiler binary"
    assert classify_task_complexity(prompt) == TaskComplexity.COMPLEX_ACTION


# ---------------------------------------------------------------------------
# Return type is always TaskComplexity
# ---------------------------------------------------------------------------


def test_return_type() -> None:
    for prompt in ("hello", "build from source", "holistic analysis", "read the docs"):
        result = classify_task_complexity(prompt)
        assert isinstance(
            result, TaskComplexity
        ), f"Expected TaskComplexity instance, got {type(result)} for {repr(prompt)}"
