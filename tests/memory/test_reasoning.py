"""Unit tests for ReasoningMemoryManager."""

from datetime import datetime

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
