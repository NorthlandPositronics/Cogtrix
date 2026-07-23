"""Unit tests for ReasoningMemoryManager."""

import threading
from datetime import datetime

import pytest

from src.memory.factory import MemoryFactory
from src.memory.json_store import JsonFileMemoryStore
from src.memory.modes.reasoning import ReasoningMemoryManager


def _ensure_registration():
    """Ensure reasoning mode is registered."""
    if not MemoryFactory.is_registered("reasoning"):
        MemoryFactory.register("reasoning", ReasoningMemoryManager)


class MockStore:
    """Mock storage for testing."""

    def __init__(self):
        self.data = {}

    def load_history(self, session_id: str):
        return self.data.get(session_id, [])

    def save_history(self, session_id: str, messages):
        self.data[session_id] = list(messages)


class TestReasoningMemoryManager:
    """Tests for ReasoningMemoryManager."""

    def setup_method(self):
        """Ensure reasoning mode is registered before each test."""
        _ensure_registration()

    def test_mode_name(self):
        """Test that mode_name returns 'reasoning'."""
        manager = ReasoningMemoryManager(MockStore(), "test")
        assert manager.mode_name == "reasoning"

    def test_factory_registration(self):
        """Test that reasoning mode is registered with factory."""
        assert MemoryFactory.is_registered("reasoning")

        store = MockStore()
        manager = MemoryFactory.create("reasoning", store, "session")
        assert isinstance(manager, ReasoningMemoryManager)

    def test_default_config(self):
        """Test default configuration values."""
        manager = ReasoningMemoryManager(MockStore(), "test")
        assert manager._mode_config["working_memory_size"] == 30
        assert manager._mode_config["vector_recall_k"] == 2
        assert manager._mode_config["track_reasoning"] is True
        assert manager._mode_config["track_decisions"] is True
        assert manager._mode_config["max_decisions"] == 20
        assert manager._mode_config["summary_max_age_hours"] == 24

    def test_custom_config(self):
        """Test custom configuration overrides defaults."""
        config = {"working_memory_size": 4, "max_decisions": 10}
        manager = ReasoningMemoryManager(MockStore(), "test", config)

        assert manager._mode_config["working_memory_size"] == 4
        assert manager._mode_config["max_decisions"] == 10

    def test_system_prompt_additions(self):
        """Test that reasoning mode adds system prompt."""
        manager = ReasoningMemoryManager(MockStore(), "test")
        prompt = manager.get_system_prompt_additions()

        assert prompt is not None
        assert "strategic" in prompt.lower()
        assert "reasoning" in prompt.lower()


class TestProblemAndHypothesis:
    """Tests for problem and hypothesis tracking."""

    def test_set_problem(self):
        """Test setting current problem."""
        manager = ReasoningMemoryManager(MockStore(), "test")
        manager.load()

        manager.set_problem("How to scale the database?")

        assert manager._current_problem == "How to scale the database?"

    def test_set_hypothesis(self):
        """Test setting hypothesis."""
        manager = ReasoningMemoryManager(MockStore(), "test")
        manager.load()

        manager.set_hypothesis("Sharding will solve the scaling issue")

        assert manager._active_hypothesis is not None
        assert "Sharding" in manager._active_hypothesis

    def test_problem_in_context(self):
        """Test that problem appears in context prefix."""
        manager = ReasoningMemoryManager(MockStore(), "test")
        manager.load()

        manager.set_problem("Design the API architecture")

        context = manager.prepare_context("next")

        assert context.has_context_prefix()
        assert "Design the API architecture" in (context.context_prefix or "")


class TestReasoningChain:
    """Tests for reasoning chain tracking."""

    def test_add_reasoning_step(self):
        """Test adding reasoning steps."""
        manager = ReasoningMemoryManager(MockStore(), "test")
        manager.load()

        manager.add_reasoning_step("First, identify the requirements")
        manager.add_reasoning_step("Then, evaluate options")

        assert len(manager._reasoning_chain) == 2

    def test_clear_reasoning_chain(self):
        """Test clearing reasoning chain."""
        manager = ReasoningMemoryManager(MockStore(), "test")
        manager.load()

        manager.add_reasoning_step("Step 1")
        manager.set_hypothesis("Test hypothesis")
        manager.clear_reasoning_chain()

        assert len(manager._reasoning_chain) == 0
        assert manager._active_hypothesis is None

    def test_reasoning_chain_in_context(self):
        """Test that reasoning chain appears in context."""
        manager = ReasoningMemoryManager(MockStore(), "test")
        manager.load()

        manager.add_reasoning_step("Analyze the problem")
        manager.add_reasoning_step("Consider alternatives")

        context = manager.prepare_context("next")

        assert context.has_context_prefix()
        assert "Analyze the problem" in (context.context_prefix or "")


class TestAlternativesTracking:
    """Tests for alternatives tracking."""

    def test_add_alternative(self):
        """Test adding alternatives."""
        manager = ReasoningMemoryManager(MockStore(), "test")
        manager.load()

        manager.add_alternative("Option A", "exploring")
        manager.add_alternative("Option B", "promising")

        assert "Option A" in manager._alternatives
        assert manager._alternatives["Option A"] == "exploring"

    def test_update_alternative(self):
        """Test updating alternative status."""
        manager = ReasoningMemoryManager(MockStore(), "test")
        manager.load()

        manager.add_alternative("Option A", "exploring")
        manager.update_alternative("Option A", "selected")

        assert manager._alternatives["Option A"] == "selected"

    def test_mark_dead_end(self):
        """Test marking dead ends."""
        manager = ReasoningMemoryManager(MockStore(), "test")
        manager.load()

        manager.add_alternative("Bad approach", "exploring")
        manager.mark_dead_end("Bad approach won't work")

        assert "Bad approach won't work" in manager._dead_ends
        assert manager._alternatives["Bad approach"] == "dead_end"

    def test_max_alternatives_limit(self):
        """Test max alternatives limit."""
        config = {"max_alternatives": 3}
        manager = ReasoningMemoryManager(MockStore(), "test", config)
        manager.load()

        for i in range(5):
            manager.add_alternative(f"Option {i}")

        assert len(manager._alternatives) == 3

    def test_add_assumption(self):
        """Test adding assumptions."""
        manager = ReasoningMemoryManager(MockStore(), "test")
        manager.load()

        manager.add_assumption("Users have stable internet")

        assert "Users have stable internet" in manager._assumptions


class TestGoalTracking:
    """Tests for goal hierarchy tracking."""

    def test_set_objective(self):
        """Test setting primary objective."""
        manager = ReasoningMemoryManager(MockStore(), "test")
        manager.load()

        manager.set_objective("Launch product by Q2")

        assert manager._primary_objective == "Launch product by Q2"

    def test_add_goal(self):
        """Test adding goals."""
        manager = ReasoningMemoryManager(MockStore(), "test")
        manager.load()

        manager.add_goal("g1", "Complete backend", success_criteria=["API done"])

        assert len(manager._goals) == 1
        assert manager._goals[0].description == "Complete backend"

    def test_update_goal_status(self):
        """Test updating goal status."""
        manager = ReasoningMemoryManager(MockStore(), "test")
        manager.load()

        manager.add_goal("g1", "Complete backend")
        manager.update_goal_status("g1", "in_progress")

        assert manager._goals[0].status == "in_progress"

    def test_set_current_phase(self):
        """Test setting current phase."""
        manager = ReasoningMemoryManager(MockStore(), "test")
        manager.load()

        manager.set_current_phase("Design Phase")

        assert manager._current_phase == "Design Phase"

    def test_objective_in_context(self):
        """Test that objective appears in context."""
        manager = ReasoningMemoryManager(MockStore(), "test")
        manager.load()

        manager.set_objective("Build scalable system")

        context = manager.prepare_context("next")

        assert context.has_context_prefix()
        assert "Build scalable system" in (context.context_prefix or "")


class TestDecisionLogging:
    """Tests for decision logging."""

    def test_log_decision(self):
        """Test logging a decision."""
        manager = ReasoningMemoryManager(MockStore(), "test")
        manager.load()

        manager.log_decision(
            decision_id="d1",
            decision="Use PostgreSQL",
            rationale="Better for complex queries",
            alternatives_rejected=["MongoDB", "MySQL"],
            trade_offs=["More setup required"],
        )

        assert len(manager._decisions) == 1
        assert manager._decisions[0].decision == "Use PostgreSQL"

    def test_max_decisions_limit(self):
        """Test max decisions limit."""
        config = {"max_decisions": 3}
        manager = ReasoningMemoryManager(MockStore(), "test", config)
        manager.load()

        for i in range(5):
            manager.log_decision(f"d{i}", f"Decision {i}", "Reason")

        assert len(manager._decisions) == 3

    def test_get_decisions(self):
        """Test getting decisions."""
        manager = ReasoningMemoryManager(MockStore(), "test")
        manager.load()

        manager.log_decision("d1", "Decision 1", "Reason 1")
        manager.log_decision("d2", "Decision 2", "Reason 2")

        decisions = manager.get_decisions()
        assert len(decisions) == 2

    def test_decisions_in_context(self):
        """Test that decisions appear in context."""
        manager = ReasoningMemoryManager(MockStore(), "test")
        manager.load()

        manager.log_decision("d1", "Use microservices", "Better scaling")

        context = manager.prepare_context("next")

        assert context.has_context_prefix()
        assert "microservices" in (context.context_prefix or "")


class TestConstraintTracking:
    """Tests for constraint tracking."""

    def test_add_constraint(self):
        """Test adding constraints."""
        manager = ReasoningMemoryManager(MockStore(), "test")
        manager.load()

        manager.add_constraint("business", "Budget is $100k")
        manager.add_constraint("technical", "Must use Python")
        manager.add_constraint("non_negotiables", "Security first")

        assert "Budget is $100k" in manager._constraints["business"]
        assert "Must use Python" in manager._constraints["technical"]

    def test_get_constraints(self):
        """Test getting constraints."""
        manager = ReasoningMemoryManager(MockStore(), "test")
        manager.load()

        manager.add_constraint("business", "Budget limit")

        all_constraints = manager.get_constraints()
        assert "business" in all_constraints

        business_only = manager.get_constraints("business")
        assert "Budget limit" in business_only["business"]

    def test_constraints_in_context(self):
        """Test that constraints appear in context."""
        manager = ReasoningMemoryManager(MockStore(), "test")
        manager.load()

        manager.add_constraint("technical", "Python 3.10+")

        context = manager.prepare_context("next")

        assert context.has_context_prefix()
        assert "Python 3.10+" in (context.context_prefix or "")


class TestPersistence:
    """Tests for save/load functionality."""

    def test_save_and_load_messages(self):
        """Test persistence of messages."""
        store = MockStore()

        manager1 = ReasoningMemoryManager(store, "test-session")
        manager1.load()
        manager1.update("Question", "Answer")
        manager1.save()

        manager2 = ReasoningMemoryManager(store, "test-session")
        manager2.load()

        assert manager2.get_message_count() == 2

    def test_to_dict(self):
        """Test serialization to dictionary."""
        manager = ReasoningMemoryManager(MockStore(), "test")
        manager.load()

        manager.update("Q", "A")
        manager.set_objective("Main goal")
        manager.set_problem("Current problem")
        manager.add_goal("g1", "Sub-goal")
        manager.log_decision("d1", "Decision", "Rationale")
        manager.add_constraint("business", "Budget")

        data = manager.to_dict()

        assert data["mode"] == "reasoning"
        assert len(data["messages"]) == 2
        assert data["primary_objective"] == "Main goal"
        assert data["current_problem"] == "Current problem"
        assert len(data["goals"]) == 1
        assert len(data["decisions"]) == 1

    def test_from_dict(self):
        """Test restoration from dictionary."""
        manager1 = ReasoningMemoryManager(MockStore(), "test")
        manager1.load()
        manager1.update("Q", "A")
        manager1.set_objective("Main goal")
        manager1.add_goal("g1", "Sub-goal")
        manager1.log_decision("d1", "Decision", "Rationale")

        data = manager1.to_dict()

        manager2 = ReasoningMemoryManager(MockStore(), "test")
        manager2.from_dict(data)

        assert manager2.get_message_count() == 2
        assert manager2._primary_objective == "Main goal"
        assert len(manager2._goals) == 1
        assert len(manager2._decisions) == 1


class TestReasoningModeWithJsonStore:
    """Test reasoning mode with actual JsonFileMemoryStore."""

    def test_full_workflow(self, tmp_path):
        """Test complete workflow with real storage."""
        store = JsonFileMemoryStore(str(tmp_path))

        # Session 1
        m1 = ReasoningMemoryManager(store, "reasoning-test")
        m1.load()
        m1.set_objective("Design new architecture")
        m1.update("How should we proceed?", "Let's analyze options.")
        m1.log_decision("d1", "Start with requirements", "Foundation first")
        m1.save()

        # Verify file created
        session_file = tmp_path / "reasoning-test.json"
        assert session_file.exists()

        # Session 2 - Continue
        m2 = ReasoningMemoryManager(store, "reasoning-test")
        m2.load()

        context = m2.prepare_context("What's next?")
        assert context.total_messages_stored == 2


class TestMetadata:
    """Tests for metadata in context."""

    def test_metadata_includes_reasoning_info(self):
        """Test that metadata includes reasoning-specific information."""
        manager = ReasoningMemoryManager(MockStore(), "test")
        manager.load()

        manager.set_objective("Goal")
        manager.set_problem("Problem")
        manager.add_goal("g1", "Sub-goal")
        manager.log_decision("d1", "Decision", "Reason")
        manager.add_reasoning_step("Step 1")
        manager.add_alternative("Option A")

        context = manager.prepare_context("next")

        assert context.metadata["has_objective"] is True
        assert context.metadata["has_problem"] is True
        assert context.metadata["goal_count"] == 1
        assert context.metadata["decision_count"] == 1
        assert context.metadata["reasoning_steps"] == 1
        assert context.metadata["alternative_count"] == 1


class TestStats:
    """Tests for statistics."""

    def test_get_stats(self):
        """Test get_stats returns comprehensive info."""
        manager = ReasoningMemoryManager(MockStore(), "test")
        manager.load()

        manager.update("Q", "A")
        manager.set_objective("Objective")
        manager.add_goal("g1", "Goal")
        manager.log_decision("d1", "Decision", "Reason")
        manager.add_reasoning_step("Step")
        manager.add_alternative("Alt")
        manager.mark_dead_end("Dead end")
        manager.add_assumption("Assumption")

        stats = manager.get_stats()

        assert stats["mode"] == "reasoning"
        assert stats["total_messages"] == 2
        assert stats["has_objective"] is True
        assert stats["goal_count"] == 1
        assert stats["decision_count"] == 1
        assert stats["reasoning_steps"] == 1
        assert stats["alternative_count"] == 1
        assert stats["dead_end_count"] == 1
        assert stats["assumption_count"] == 1


class TestClear:
    """Tests for clear functionality."""

    def test_clear_resets_all(self):
        """Test that clear resets all state."""
        manager = ReasoningMemoryManager(MockStore(), "test")
        manager.load()

        manager.update("Q", "A")
        manager.set_objective("Objective")
        manager.set_problem("Problem")
        manager.add_goal("g1", "Goal")
        manager.log_decision("d1", "Decision", "Reason")
        manager.add_reasoning_step("Step")
        manager.add_alternative("Alt")
        manager.add_constraint("business", "Constraint")

        manager.clear()

        assert manager.get_message_count() == 0
        assert manager._primary_objective is None
        assert manager._current_problem is None
        assert len(manager._goals) == 0
        assert len(manager._decisions) == 0
        assert len(manager._reasoning_chain) == 0
        assert len(manager._alternatives) == 0


class TestReasoningTimestamps:
    """Tests for timestamp support in reasoning memory mode."""

    def setup_method(self):
        _ensure_registration()

    def test_update_stamps_messages(self):
        """Test that update attaches timestamps to messages."""
        manager = ReasoningMemoryManager(MockStore(), "test")
        manager.load()
        manager.update("Analyze options", "Here is my analysis.")

        for msg in manager._messages:
            ts = manager._get_msg_ts(msg)
            assert ts is not None
            datetime.fromisoformat(ts)

    def test_prepare_context_injects_timestamps(self):
        """Timestamps are prepended to HumanMessages only (not AI — the LLM mimics them)."""
        manager = ReasoningMemoryManager(MockStore(), "test")
        manager.load()
        manager.update("Question", "Answer")

        context = manager.prepare_context("next")

        for msg in context.messages:
            content = msg.content if hasattr(msg, "content") else msg["content"]
            if type(msg).__name__ == "HumanMessage":
                assert content.startswith("[")

    def test_to_dict_from_dict_roundtrip(self):
        """Test that timestamps survive to_dict / from_dict round-trip."""
        m1 = ReasoningMemoryManager(MockStore(), "test")
        m1.load()
        m1.update("Q", "A")

        data = m1.to_dict()
        for msg_data in data["messages"]:
            assert "timestamp" in msg_data

        m2 = ReasoningMemoryManager(MockStore(), "test")
        m2.from_dict(data)

        for msg in m2._messages:
            assert m2._get_msg_ts(msg) is not None


class TestReasoningClearDeadlockRegression:
    """Regression tests for issue #1402: AB/BA deadlock in clear().

    The bug: reasoning.py:clear() acquired _hybrid_lock via super().clear()
    before acquiring _mode_lock — the reverse of the standard nesting order
    (_mode_lock -> _hybrid_lock) used by every other method in the memory
    subsystem. This created a deadlock when clear() ran concurrently with
    prepare_context() on the same manager instance.

    The fix: clear() now acquires _mode_lock first, then calls super().clear(),
    matching the standard nesting order.
    """

    @pytest.fixture
    def manager(self):
        store = MockStore()
        manager = ReasoningMemoryManager(store, "test_deadlock")
        manager.load()
        return manager

    def test_clear_and_prepare_context_no_deadlock(self, manager):
        """clear() and prepare_context() running concurrently must not deadlock."""
        clear_exc = []
        clear_done = threading.Event()
        barrier = threading.Barrier(2)

        def clear_worker():
            try:
                barrier.wait()
                manager.clear()
            except Exception as exc:
                clear_exc.append(exc)
            finally:
                clear_done.set()

        def context_worker():
            try:
                barrier.wait()
                manager.prepare_context()
            except Exception:
                pass

        ct = threading.Thread(target=clear_worker)
        pt = threading.Thread(target=context_worker)
        ct.start()
        pt.start()
        done = clear_done.wait(timeout=10)
        assert done, "clear() timed out — possible deadlock"
        ct.join(timeout=2)
        pt.join(timeout=2)
        assert clear_exc == [], f"clear() raised: {clear_exc}"

    def test_clear_blocks_on_mode_lock_then_completes(self, manager):
        """clear() must block when _mode_lock is held, then complete when released."""
        clear_exc = []
        clear_done = threading.Event()
        start_gate = threading.Barrier(2)

        def holder():
            with manager._mode_lock:
                start_gate.wait()

        def clear_worker():
            start_gate.wait()
            try:
                manager.clear()
            except Exception as exc:
                clear_exc.append(exc)
            finally:
                clear_done.set()

        ht = threading.Thread(target=holder)
        ct = threading.Thread(target=clear_worker)
        ht.start()
        ct.start()
        done = clear_done.wait(timeout=10)
        assert done, "clear() timed out — possible deadlock"
        ht.join(timeout=2)
        ct.join(timeout=2)
        assert not clear_exc, f"clear() raised: {clear_exc}"
        assert clear_done.is_set(), "clear() did not complete after lock release — deadlock"


class TestTokensSinceSummaryLockRegression:
    """Regression tests for #1295: _tokens_since_summary race between update()
    and background summarizer. Token counter increments must be serialized by
    _hybrid_lock to prevent lost updates when reset_summary() races with update().
    """

    def test_tokens_increment_protected_by_hybrid_lock(self):
        """update() must increment _tokens_since_summary under _hybrid_lock."""
        _ensure_registration()
        manager = ReasoningMemoryManager(MockStore(), "test")
        manager.load()
        initial = manager._tokens_since_summary
        manager.update("What is 2+2?", "4")
        assert manager._tokens_since_summary > initial

    def test_reset_summary_state_zeros_counter(self):
        """_reset_summary_state() must reset _tokens_since_summary to 0."""
        _ensure_registration()
        manager = ReasoningMemoryManager(MockStore(), "test")
        manager.load()
        manager.update("Q", "A")
        assert manager._tokens_since_summary > 0
        manager._reset_summary_state()
        assert manager._tokens_since_summary == 0

    def test_concurrent_updates_serialized_by_hybrid_lock(self):
        """Two threads calling update() concurrently must not raise or corrupt counter."""
        _ensure_registration()
        manager = ReasoningMemoryManager(MockStore(), "test")
        manager.load()
        errors = []
        barrier = threading.Barrier(2)

        def updater():
            try:
                barrier.wait()
                for _ in range(20):
                    manager.update("Q", "A")
            except Exception as exc:
                errors.append(("updater", exc))

        t1 = threading.Thread(target=updater)
        t2 = threading.Thread(target=updater)
        t1.start()
        t2.start()
        t1.join(timeout=20)
        t2.join(timeout=20)
        assert t1.is_alive() is False, "thread t1 did not finish"
        assert t2.is_alive() is False, "thread t2 did not finish"
        assert errors == [], f"Concurrent access raised: {errors}"


class TestGoalDataDir2160:
    """#2160 — the reasoning prefix must read goals from the store's configured
    data_dir (base_path = <data_dir>/history), not a hardcoded 'data', so it
    sees goals the goal tools persisted under config.data_dir."""

    def test_goal_prefix_reads_from_store_data_dir(self, tmp_path) -> None:
        import src.tasks.goal_tracker as gt
        from src.tasks.goal_tracker import GoalStack

        session_id = "reasoning-2160"
        # Persist a goal under the CONFIGURED data_dir (tmp_path), as the goal
        # tools would. A direct GoalStack avoids polluting the module cache.
        GoalStack(session_id, tmp_path).push("ship the widget")
        gt._stacks.clear()  # ensure reasoning is the first cache user

        store = JsonFileMemoryStore(str(tmp_path / "history"))
        manager = ReasoningMemoryManager(store, session_id)
        manager.load()
        try:
            context = manager.prepare_context("next")
            prefix = context.context_prefix or ""
        finally:
            gt._stacks.clear()

        # Pre-fix this looked in ./data/goals (empty) and missed the goal.
        assert "ship the widget" in prefix

    def test_falls_back_to_data_without_base_path(self, tmp_path) -> None:
        """A store without base_path doesn't crash the prefix build (#2160)."""
        import src.tasks.goal_tracker as gt

        gt._stacks.clear()
        manager = ReasoningMemoryManager(MockStore(), "reasoning-2160-fallback")
        manager.load()
        try:
            context = manager.prepare_context("next")  # must not raise
            assert context is not None
        finally:
            gt._stacks.clear()
