"""Tests for the delegate tool."""

import threading
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest

from src.agent.safety import UserCancelledRun  # noqa: E402
from src.tools.delegate import (
    _MAX_CIRCUIT_BREAKERS,
    TOOL_CONFIGS,
    DelegateInput,
    DelegateParallelInput,
    DelegateResult,
    ModelCircuitBreaker,
    _build_prompt,
    _check_allowed_model,
    _circuit_breaker_lock,
    _evict_stale_breakers,
    _extract_content,
    _get_circuit_breaker,
    _validate_json_response,
    configure_delegate,
    delegate_parallel,
    delegate_task,
    get_model_status,
    reset_model_status,
    set_delegate_tools,
)
from src.tools.delegate import (
    resolve_delegate_defaults as _resolve_defaults,
)
from src.tools.delegate import (
    resolve_model_alias as _resolve_model_alias,
)


class TestConfiguration:
    """Tests for delegate configuration."""

    def test_configure_delegate_updates_config(self):
        """Test that configure_delegate updates the module config."""
        configure_delegate(
            {
                "enabled": False,
                "default_provider": "openai",
                "models": {"fast": "ollama/gemma3:12b"},
            }
        )

        # Config should be updated (we test via _resolve_model_alias)
        provider, model, _ = _resolve_model_alias(None, "fast")
        assert provider == "ollama"
        assert model == "gemma3:12b"

    @pytest.mark.parametrize(
        "models_cfg, input_provider, alias, expected_provider, expected_model",
        [
            (
                {"smart": "openai/gpt-4", "code": "ollama/qwen3-coder:30b-a3b"},
                None,
                "smart",
                "openai",
                "gpt-4",
            ),
            (
                {"smart": "openai/gpt-4", "code": "ollama/qwen3-coder:30b-a3b"},
                None,
                "code",
                "ollama",
                "qwen3-coder:30b-a3b",
            ),
            ({"default": "gemma3:12b"}, "ollama", "default", "ollama", "gemma3:12b"),
        ],
    )
    def test_resolve_alias(
        self,
        models_cfg: dict,
        input_provider: str | None,
        alias: str,
        expected_provider: str,
        expected_model: str,
    ) -> None:
        """Test resolving model aliases with various provider/model configurations."""
        configure_delegate({"models": models_cfg})
        provider, model, _ = _resolve_model_alias(input_provider, alias)
        assert provider == expected_provider
        assert model == expected_model

    def test_no_alias_passthrough(self):
        """Test that non-alias values pass through unchanged."""
        provider, model, alias_config = _resolve_model_alias("openai", "gpt-4")
        assert provider == "openai"
        assert model == "gpt-4"
        assert alias_config == {}


class TestJsonValidation:
    """Tests for JSON response validation."""

    def test_valid_json(self):
        """Test validation of valid JSON."""
        valid, parsed = _validate_json_response('{"key": "value"}')
        assert valid is True
        assert parsed == {"key": "value"}

    def test_valid_json_array(self):
        """Test validation of JSON array."""
        valid, parsed = _validate_json_response("[1, 2, 3]")
        assert valid is True
        assert parsed == [1, 2, 3]

    def test_invalid_json(self):
        """Test validation of invalid JSON."""
        valid, parsed = _validate_json_response("not valid json")
        assert valid is False
        assert parsed is None

    def test_json_in_markdown_block(self):
        """Test extraction of JSON from markdown code block."""
        response = """```json
{"result": "success"}
```"""
        valid, parsed = _validate_json_response(response)
        assert valid is True
        assert parsed == {"result": "success"}

    def test_json_in_plain_code_block(self):
        """Test extraction of JSON from plain code block."""
        response = """```
{"data": 123}
```"""
        valid, parsed = _validate_json_response(response)
        assert valid is True
        assert parsed == {"data": 123}


class TestPromptBuilding:
    """Tests for prompt building."""

    def test_build_text_prompt(self):
        """Test building a text format prompt."""
        messages = _build_prompt(
            task="Summarize this text",
            context="Some long text here",
            response_format="text",
            json_schema=None,
        )

        assert len(messages) == 2
        # Check system message
        system_content = (
            messages[0].content if hasattr(messages[0], "content") else messages[0]["content"]
        )
        assert "delegated task" in system_content.lower()

        # Check user message
        user_content = (
            messages[1].content if hasattr(messages[1], "content") else messages[1]["content"]
        )
        assert "Summarize this text" in user_content
        assert "Some long text here" in user_content

    def test_build_json_prompt(self):
        """Test building a JSON format prompt."""
        messages = _build_prompt(
            task="Extract data",
            context="Source data",
            response_format="json",
            json_schema='{"name": "string", "age": "number"}',
        )

        system_content = (
            messages[0].content if hasattr(messages[0], "content") else messages[0]["content"]
        )
        assert "valid JSON" in system_content
        assert "name" in system_content

    def test_build_code_prompt(self):
        """Test building a code format prompt."""
        messages = _build_prompt(
            task="Write a function",
            context="",
            response_format="code",
            json_schema=None,
        )

        system_content = (
            messages[0].content if hasattr(messages[0], "content") else messages[0]["content"]
        )
        assert "code only" in system_content.lower()


class TestDelegateResult:
    """Tests for DelegateResult dataclass."""

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = DelegateResult(
            success=True,
            response="Hello",
            format_valid=True,
            parsed_json={"greeting": "Hello"},
            model_used="gpt-4",
            provider="openai",
            duration_seconds=1.5,
            error=None,
        )

        d = result.to_dict()
        assert d["success"] is True
        assert d["response"] == "Hello"
        assert d["model_used"] == "gpt-4"
        assert d["duration_seconds"] == 1.5


class TestInputSchemas:
    """Tests for Pydantic input schemas."""

    def test_delegate_input_defaults(self):
        """Test DelegateInput default values."""
        input_data = DelegateInput(task="Test task")
        assert input_data.task == "Test task"
        assert input_data.context == ""
        assert input_data.use_tools is True
        assert input_data.response_format == "text"
        assert input_data.timeout == 60
        assert input_data.temperature == 0.7

    def test_delegate_parallel_input(self):
        """Test DelegateParallelInput schema."""
        input_data = DelegateParallelInput(
            tasks=[
                {"task": "Task 1"},
                {"task": "Task 2", "provider": "openai"},
            ],
            timeout=120,
        )
        assert len(input_data.tasks) == 2
        assert input_data.timeout == 120


class TestToolConfig:
    """Tests for TOOL_CONFIGS structure."""

    def test_tool_configs_has_required_fields(self):
        """Test that TOOL_CONFIGS has all required fields."""
        assert len(TOOL_CONFIGS) == 2

        tool_names = [c["name"] for c in TOOL_CONFIGS]
        assert "delegate_task" in tool_names
        assert "delegate_parallel" in tool_names

        for config in TOOL_CONFIGS:
            assert "name" in config
            assert "description" in config
            assert "input_schema" in config
            assert "function" in config
            assert "requires_confirmation" in config
            assert config["requires_confirmation"] is False  # No confirmation required


class TestDelegateTaskWithMock:
    """Tests for delegate_task with mocked LLM."""

    def setup_method(self):
        """Reset config, circuit breakers, and delegate tools before each test."""
        reset_model_status()
        configure_delegate(
            {
                "enabled": True,
                "default_provider": "ollama",
                "allowed_providers": ["openai", "ollama"],
            }
        )
        # Clear thread-local delegate tools so prior tests don't bleed through
        set_delegate_tools([], {})
        # Also patch get_delegate_tools at the module level: if tools somehow
        # leak via thread-local from prior tests, the agent path is bypassed.
        self._tools_patcher = mock.patch("src.tools.delegate.get_delegate_tools", return_value=[])
        self._tools_patcher.start()

    def teardown_method(self, _method=None):
        self._tools_patcher.stop()

    @patch("src.tools.delegate.create_delegate_llm")
    def test_delegate_task_success(self, mock_create_llm):
        """Test successful task delegation."""
        # Mock LLM
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "This is the summary."
        mock_llm.invoke.return_value = mock_response
        mock_llm.model = "test-model"
        mock_create_llm.return_value = mock_llm

        result = delegate_task(
            task="Summarize this",
            context="Long text here",
            provider="ollama",
            timeout=30,
            use_tools=False,
        )

        assert "Delegated to:" in result
        assert "Response:" in result
        assert "This is the summary." in result

    @patch("src.tools.delegate.create_delegate_llm")
    def test_delegate_task_json_valid(self, mock_create_llm):
        """Test delegation with valid JSON response."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"result": "success"}'
        mock_llm.invoke.return_value = mock_response
        mock_create_llm.return_value = mock_llm

        result = delegate_task(
            task="Extract data",
            context="Source data to extract from",
            response_format="json",
            provider="ollama",
            use_tools=False,
        )

        assert "JSON Valid:** ✓" in result

    @patch("src.tools.delegate.create_delegate_llm")
    def test_delegate_task_json_invalid(self, mock_create_llm):
        """Test delegation with invalid JSON response."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Not valid JSON at all"
        mock_llm.invoke.return_value = mock_response
        mock_create_llm.return_value = mock_llm

        result = delegate_task(
            task="Extract data",
            context="Source data to extract from",
            response_format="json",
            provider="ollama",
            use_tools=False,
        )

        assert "JSON Valid:** ✗" in result

    def test_delegate_task_disabled(self):
        """Test that delegation fails when disabled."""
        configure_delegate({"enabled": False})

        result = delegate_task(task="Test")
        assert "Delegation disabled" in result

    def test_delegate_task_rejects_empty_context_no_tools(self):
        """Test that LLM-only delegation with empty context is rejected."""
        result = delegate_task(task="Summarize", use_tools=False, context="")
        assert "rejected" in result.lower()
        assert "context is empty" in result.lower()

    def test_delegate_task_rejects_whitespace_context_no_tools(self):
        """Test that LLM-only delegation with whitespace-only context is rejected."""
        result = delegate_task(task="Summarize", use_tools=False, context="   \n  ")
        assert "rejected" in result.lower()

    @patch("src.tools.delegate.create_delegate_llm")
    def test_delegate_task_allows_empty_context_with_tools(self, mock_create_llm):
        """Test that tool-capable delegation with empty context is allowed."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Found results"
        mock_llm.invoke.return_value = mock_response
        mock_llm.model = "test-model"
        mock_create_llm.return_value = mock_llm

        result = delegate_task(
            task="Search for info",
            context="",
            use_tools=True,
            provider="ollama",
        )
        assert "rejected" not in result.lower()
        assert "Found results" in result

    @patch("src.tools.delegate.create_delegate_llm")
    def test_delegate_task_error_handling(self, mock_create_llm):
        """Test error handling in delegation."""
        mock_create_llm.side_effect = Exception("Connection failed")

        result = delegate_task(task="Test", provider="ollama")
        assert "failed" in result.lower()

    @patch("src.tools.delegate.create_delegate_llm")
    def test_delegate_task_user_cancelled_propagates(self, mock_create_llm):
        """UserCancelledRun raised during delegation must propagate to caller."""
        mock_create_llm.side_effect = UserCancelledRun("User cancelled")

        with pytest.raises(UserCancelledRun):
            delegate_task(task="Test", provider="ollama")


class TestDelegateParallelWithMock:
    """Tests for delegate_parallel with mocked LLM."""

    def setup_method(self):
        """Reset config before each test."""
        configure_delegate(
            {
                "enabled": True,
                "default_provider": "ollama",
                "allowed_providers": ["openai", "ollama"],
            }
        )

    @patch("src.tools.delegate.create_delegate_llm")
    def test_parallel_delegation(self, mock_create_llm):
        """Test parallel task delegation."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Result"
        mock_llm.invoke.return_value = mock_response
        mock_llm.model = "test-model"
        mock_create_llm.return_value = mock_llm

        result = delegate_parallel(
            tasks=[
                {"task": "Task 1"},
                {"task": "Task 2"},
                {"task": "Task 3"},
            ],
            timeout=60,
        )

        assert "Parallel Delegation:** 3 tasks" in result
        assert "Task 1" in result
        assert "Task 2" in result
        assert "Task 3" in result

    @patch("src.tools.delegate._emit_status")
    def test_parallel_honors_per_task_timeouts(self, mock_emit_status):
        """Test that each task uses its own timeout instead of the batch timeout."""
        from src.tools import delegate as delegate_mod

        recorded_timeouts: list[float | None] = []

        class FakeFuture:
            def __init__(self, value):
                self._value = value

            def result(self, timeout=None):
                recorded_timeouts.append(timeout)
                return self._value

            def cancel(self):
                return True

        class FakeExecutor:
            def __init__(self, *args, **kwargs):
                self._futures = []

            def submit(self, fn, *args, **kwargs):
                future = FakeFuture(fn(*args, **kwargs))
                self._futures.append(future)
                return future

            def shutdown(self, wait=False, cancel_futures=False):
                return None

        mock_result = DelegateResult(
            success=True,
            response="ok",
            format_valid=True,
            parsed_json=None,
            model_used="test-model",
            provider="ollama",
            duration_seconds=0,
            error=None,
        )

        with (
            patch.object(delegate_mod, "ThreadPoolExecutor", FakeExecutor),
            patch.object(delegate_mod, "_execute_single_task", return_value=mock_result),
            patch.object(delegate_mod.time, "time", return_value=1000.0),
        ):
            result = delegate_parallel(
                tasks=[
                    {"task": "Task 1", "timeout": 10},
                    {"task": "Task 2", "timeout": 25},
                ],
                timeout=120,
            )

        assert recorded_timeouts == [10, 25]
        assert "Parallel Delegation:** 2 tasks" in result
        mock_emit_status.assert_called()

    def test_parallel_empty_tasks(self):
        """Test parallel with empty task list."""
        result = delegate_parallel(tasks=[])
        assert "No tasks provided" in result

    def test_parallel_rejects_empty_context_no_tools(self):
        """Test that parallel rejects LLM-only tasks with empty context."""
        result = delegate_parallel(
            tasks=[
                {"task": "Task 1", "use_tools": False, "context": ""},
            ],
        )
        assert "rejected" in result.lower()
        assert "task 1" in result.lower()

    def test_parallel_disabled(self):
        """Test parallel when delegation is disabled."""
        configure_delegate({"enabled": False})

        result = delegate_parallel(tasks=[{"task": "Test"}])
        assert "Delegation disabled" in result

    @patch("src.tools.delegate._execute_single_task")
    def test_parallel_user_cancelled_propagates(self, mock_execute):
        """UserCancelledRun raised during parallel delegation must propagate."""
        mock_execute.side_effect = UserCancelledRun("User cancelled")

        with pytest.raises(UserCancelledRun):
            delegate_parallel(tasks=[{"task": "Task 1"}])


class TestCircuitBreaker:
    """Tests for the circuit breaker functionality."""

    def setup_method(self):
        """Reset circuit breakers, config, and delegate tools before each test."""
        reset_model_status()
        configure_delegate(
            {
                "enabled": True,
                "default_provider": "ollama",
                "allowed_providers": ["openai", "ollama"],
                "allowed_models": None,
                "models": {},
                "default_model_alias": None,
                "max_consecutive_failures": 5,
                "circuit_breaker_cooldown": 300,
            }
        )
        set_delegate_tools([], {})
        self._tools_patcher = mock.patch("src.tools.delegate.get_delegate_tools", return_value=[])
        self._tools_patcher.start()

    def teardown_method(self, _method=None):
        self._tools_patcher.stop()

    def test_circuit_breaker_initial_state(self):
        """Test that circuit breaker starts in available state."""
        breaker = ModelCircuitBreaker()
        assert breaker.consecutive_failures == 0
        assert breaker.is_unavailable is False
        available, reason = breaker.check_availability()
        assert available is True
        assert reason is None

    def test_circuit_breaker_records_failure(self):
        """Test that failures are recorded correctly."""
        breaker = ModelCircuitBreaker()
        breaker.record_failure("Connection error", max_failures=5)

        assert breaker.consecutive_failures == 1
        assert breaker.is_unavailable is False
        assert breaker.last_error == "Connection error"

    def test_circuit_breaker_trips_after_max_failures(self):
        """Test that circuit breaker trips after max consecutive failures."""
        breaker = ModelCircuitBreaker()

        for i in range(5):
            breaker.record_failure(f"Error {i + 1}", max_failures=5)

        assert breaker.consecutive_failures == 5
        assert breaker.is_unavailable is True

        available, reason = breaker.check_availability(cooldown=300)
        assert available is False
        assert reason is not None and "consecutive failures" in reason

    def test_circuit_breaker_resets_on_success(self):
        """Test that success resets the circuit breaker."""
        breaker = ModelCircuitBreaker()

        # Record some failures
        for i in range(3):
            breaker.record_failure(f"Error {i + 1}", max_failures=5)

        assert breaker.consecutive_failures == 3

        # Record success
        breaker.record_success()

        assert breaker.consecutive_failures == 0
        assert breaker.is_unavailable is False
        assert breaker.last_error == ""

    def test_get_model_status_empty(self):
        """Test get_model_status when no models tracked."""
        reset_model_status()
        status = get_model_status()
        assert status == {}

    def test_reset_model_status_all(self):
        """Test resetting all model statuses."""
        # Create some circuit breakers by making calls
        breaker = ModelCircuitBreaker()
        breaker.record_failure("Error", max_failures=5)

        reset_model_status()
        status = get_model_status()
        assert status == {}

    @patch("src.tools.delegate.create_delegate_llm")
    def test_delegate_records_failure_in_circuit_breaker(self, mock_create_llm):
        """Test that delegate_task records failures."""
        reset_model_status()
        mock_create_llm.side_effect = Exception("Connection refused")

        # Make 5 failed calls
        for _ in range(5):
            delegate_task(task="Test", provider="ollama", model="test-model")

        # Check model status
        status = get_model_status()
        assert "ollama/test-model" in status
        assert status["ollama/test-model"]["available"] is False
        assert status["ollama/test-model"]["consecutive_failures"] == 5

    @patch("src.tools.delegate.create_delegate_llm")
    def test_delegate_blocked_when_unavailable(self, mock_create_llm):
        """Test that delegate_task is blocked when model is unavailable."""
        reset_model_status()
        mock_create_llm.side_effect = Exception("Connection refused")

        # Trip the circuit breaker
        for _ in range(5):
            delegate_task(task="Test", provider="ollama", model="blocked-model")

        # Reset mock to not raise
        mock_create_llm.side_effect = None
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Success"
        mock_llm.invoke.return_value = mock_response
        mock_create_llm.return_value = mock_llm

        # Try to call again - should be blocked
        result = delegate_task(task="Test", provider="ollama", model="blocked-model")

        assert "unavailable" in result.lower()
        assert "consecutive failures" in result.lower()

    @patch("src.tools.delegate.create_delegate_llm")
    def test_delegate_success_resets_circuit_breaker(self, mock_create_llm):
        """Test that successful delegation resets circuit breaker."""
        reset_model_status()

        # First, create some failures (but not enough to trip)
        mock_create_llm.side_effect = Exception("Temporary error")
        for _ in range(3):
            delegate_task(task="Test", provider="ollama", model="recover-model")

        status = get_model_status()
        assert status["ollama/recover-model"]["consecutive_failures"] == 3

        # Now succeed
        mock_create_llm.side_effect = None
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Success"
        mock_llm.invoke.return_value = mock_response
        mock_create_llm.return_value = mock_llm

        result = delegate_task(task="Test", provider="ollama", model="recover-model")

        assert "Success" in result

        # Circuit breaker should be reset
        status = get_model_status()
        assert status["ollama/recover-model"]["consecutive_failures"] == 0
        assert status["ollama/recover-model"]["available"] is True


class TestCircuitBreakerEviction:
    """Tests for the circuit breaker eviction mechanism."""

    def setup_method(self):
        reset_model_status()

    def teardown_method(self):
        reset_model_status()

    def test_eviction_trims_dict_when_over_cap(self):
        """Creating more than _MAX_CIRCUIT_BREAKERS entries should trigger eviction."""
        import src.tools.delegate as mod

        reset_model_status()
        for i in range(_MAX_CIRCUIT_BREAKERS + 5):
            _get_circuit_breaker("provider", f"model-{i}")

        assert len(mod._circuit_breakers) <= _MAX_CIRCUIT_BREAKERS

    def test_eviction_removes_stale_idle_entries(self):
        """Entries with zero failures and old last_used should be evicted first."""
        import time

        import src.tools.delegate as mod

        reset_model_status()
        old_ts = time.time() - 7200.0  # 2 hours ago — beyond the idle window

        for i in range(10):
            mod._circuit_breakers[f"stale/model-{i}"] = ModelCircuitBreaker(last_used=old_ts)

        hot_key = "hot/model"
        mod._circuit_breakers[hot_key] = ModelCircuitBreaker(
            consecutive_failures=3, last_used=time.time()
        )

        _evict_stale_breakers()

        assert hot_key in mod._circuit_breakers
        for i in range(10):
            assert f"stale/model-{i}" not in mod._circuit_breakers

    def test_eviction_keeps_entries_with_active_failures(self):
        """Entries with non-zero consecutive failures must not be evicted by idle pass."""
        import time

        import src.tools.delegate as mod

        reset_model_status()
        old_ts = time.time() - 7200.0

        failing_key = "failing/model"
        mod._circuit_breakers[failing_key] = ModelCircuitBreaker(
            consecutive_failures=3, last_used=old_ts
        )

        _evict_stale_breakers()

        assert failing_key in mod._circuit_breakers

    def test_last_used_updated_on_repeated_access(self):
        """Accessing an existing breaker should update its last_used timestamp."""
        import time

        import src.tools.delegate as mod

        reset_model_status()
        breaker = _get_circuit_breaker("p", "m")
        first_ts = breaker.last_used

        time.sleep(0.01)
        _get_circuit_breaker("p", "m")

        assert mod._circuit_breakers["p/m"].last_used >= first_ts


class TestCheckAllowedModel:
    """Tests for _check_allowed_model — the allowed-models gate."""

    def test_no_restriction_when_allowed_models_is_none(self):
        """When allowed_models is not configured, all models should pass."""
        configure_delegate({"allowed_models": None})
        assert _check_allowed_model("anything") is None

    def test_allowed_model_passes(self):
        """An explicitly allowed model should pass."""
        configure_delegate({"allowed_models": ["coder", "gpt4o", "fast"]})
        assert _check_allowed_model("coder") is None

    def test_disallowed_model_returns_error(self):
        """A model not in the allowed list should return an error message."""
        configure_delegate({"allowed_models": ["coder", "gpt4o"]})
        err = _check_allowed_model("unknown-model")
        assert err is not None
        assert "not in the allowed" in err

    def test_none_model_passes_when_allowed_list_set(self):
        """When model is None (agent uses defaults), it should pass."""
        configure_delegate({"allowed_models": ["coder", "gpt4o"]})
        assert _check_allowed_model(None) is None

    def test_empty_string_model_passes_when_allowed_list_set(self):
        """When model is empty string (agent omitted it), it should pass."""
        configure_delegate({"allowed_models": ["coder", "gpt4o"]})
        assert _check_allowed_model("") is None


class TestResolveDefaults:
    """Tests for _resolve_defaults — filling in missing provider/model."""

    def setup_method(self):
        """Reset delegate config to a clean baseline before each test."""
        configure_delegate(
            {
                "default_provider": "ollama",
                "default_model": None,
                "default_model_alias": None,
                "models": {},
            }
        )

    def test_both_provided(self):
        """When both provider and model are given, they should pass through."""
        p, m = _resolve_defaults("openai", "gpt-4.1")
        assert p == "openai"
        assert m == "gpt-4.1"

    def test_provider_none_uses_default(self):
        """When provider is None, the configured default should be used."""
        configure_delegate({"default_provider": "spark-cluster"})
        p, m = _resolve_defaults(None, "some-model")
        assert p == "spark-cluster"
        assert m == "some-model"

    def test_model_none_uses_default(self):
        """When model is None, the configured default should be used."""
        configure_delegate({"default_model": "nemotron-nano"})
        p, m = _resolve_defaults("ollama", None)
        assert p == "ollama"
        assert m == "nemotron-nano"

    def test_both_none_uses_defaults(self):
        """When both are None, both defaults should be applied."""
        configure_delegate({"default_provider": "ollama", "default_model": "qwen3:8b"})
        p, m = _resolve_defaults(None, None)
        assert p == "ollama"
        assert m == "qwen3:8b"

    def test_model_none_no_default_configured(self):
        """When model is None and no default_model configured, fallback to 'default'."""
        configure_delegate({"default_model": None})
        _, m = _resolve_defaults("ollama", None)
        assert m == "default"


class TestSetDelegateTools:
    """Tests for set_delegate_tools — registering tools for delegates."""

    def teardown_method(self):
        """Clean up module-level _delegate_tools after each test."""
        set_delegate_tools([])

    def test_filters_excluded_tools(self):
        """Recursion and destructive tools are excluded; safe tools pass through."""
        tools = []
        for name in (
            "read_file",
            "delegate_task",
            "deep_think",
            "request_tools",
            "execute_shell_command",
            "write_file",
            "execute_python",
        ):
            tool = MagicMock()
            tool.name = name
            tools.append(tool)

        set_delegate_tools(tools)

        import src.tools.delegate as mod

        names = [t.name for t in mod._delegate_tools]
        # Safe research tool passes through
        assert "read_file" in names
        # Recursion / meta tools are excluded
        assert "delegate_task" not in names
        assert "deep_think" not in names
        assert "request_tools" not in names
        # Destructive tools are sandboxed
        assert "execute_shell_command" not in names
        assert "write_file" not in names
        assert "execute_python" not in names

    def test_empty_input_clears_tools(self):
        """Passing an empty list should clear delegate tools."""
        tool = MagicMock()
        tool.name = "read_file"
        set_delegate_tools([tool])

        import src.tools.delegate as mod

        assert len(mod._delegate_tools) == 1

        set_delegate_tools([])
        assert len(mod._delegate_tools) == 0

    def test_tools_without_name_attribute_are_skipped(self):
        """Objects without a 'name' attribute get empty-string name → kept."""
        tool = MagicMock(spec=[])  # no attributes
        set_delegate_tools([tool])

        import src.tools.delegate as mod

        assert len(mod._delegate_tools) == 1

    def test_merges_active_and_available_tools(self):
        """Active + available on-demand tools should be merged for delegates."""
        active = MagicMock()
        active.name = "read_file"
        ondemand_search = MagicMock()
        ondemand_search.name = "search_web"
        ondemand_calc = MagicMock()
        ondemand_calc.name = "calculate"

        set_delegate_tools(
            [active],
            available_tools={
                "search_web": ondemand_search,
                "calculate": ondemand_calc,
            },
        )

        import src.tools.delegate as mod

        names = {t.name for t in mod._delegate_tools}
        assert names == {"read_file", "search_web", "calculate"}

    def test_deduplicates_across_active_and_available(self):
        """A tool in both active and available should appear only once."""
        tool_a = MagicMock()
        tool_a.name = "read_file"
        tool_b = MagicMock()
        tool_b.name = "read_file"

        set_delegate_tools([tool_a], available_tools={"read_file": tool_b})

        import src.tools.delegate as mod

        assert len(mod._delegate_tools) == 1
        assert mod._delegate_tools[0] is tool_a

    def test_excludes_recursion_tools_from_available(self):
        """Excluded tools in available_tools should be filtered out."""
        active = MagicMock()
        active.name = "read_file"
        deep = MagicMock()
        deep.name = "deep_think"

        set_delegate_tools([active], available_tools={"deep_think": deep})

        import src.tools.delegate as mod

        names = {t.name for t in mod._delegate_tools}
        assert "deep_think" not in names
        assert "read_file" in names

    def test_excludes_destructive_tools(self):
        """High-risk tools (shell, file write, code exec) are sandboxed."""
        tools = []
        for name in (
            "read_file",
            "execute_shell_command",
            "execute_python",
            "write_file",
            "patch_file",
            "append_file",
            "http_post",
            "send_email",
            "self_improve",
        ):
            tool = MagicMock()
            tool.name = name
            tools.append(tool)

        set_delegate_tools(tools)

        import src.tools.delegate as mod

        names = {t.name for t in mod._delegate_tools}
        assert "read_file" in names
        assert "execute_shell_command" not in names
        assert "execute_python" not in names
        assert "write_file" not in names
        assert "patch_file" not in names
        assert "append_file" not in names
        assert "http_post" not in names
        assert "send_email" not in names
        assert "self_improve" not in names

    def test_excludes_destructive_tools_from_available(self):
        """Sandbox applies to available_tools as well as active tools."""
        active = MagicMock()
        active.name = "read_file"
        shell = MagicMock()
        shell.name = "execute_shell_command"
        write = MagicMock()
        write.name = "write_file"

        set_delegate_tools(
            [active],
            available_tools={
                "execute_shell_command": shell,
                "write_file": write,
            },
        )

        import src.tools.delegate as mod

        names = {t.name for t in mod._delegate_tools}
        assert names == {"read_file"}

    def test_safe_tools_remain_available_to_delegates(self):
        """Non-destructive research tools are still passed to delegates."""
        safe_tools = []
        for name in (
            "read_file",
            "search_web",
            "search_news",
            "http_get",
            "git_status",
            "git_diff",
            "list_directory",
            "file_info",
            "calculate",
            "get_current_datetime",
        ):
            tool = MagicMock()
            tool.name = name
            safe_tools.append(tool)

        set_delegate_tools(safe_tools)

        import src.tools.delegate as mod

        names = {t.name for t in mod._delegate_tools}
        assert names == {
            "read_file",
            "search_web",
            "search_news",
            "http_get",
            "git_status",
            "git_diff",
            "list_directory",
            "file_info",
            "calculate",
            "get_current_datetime",
        }


class TestExtractContent:
    """Tests for _extract_content — text extraction from LLM responses."""

    def test_string_content(self):
        """Simple string .content attribute."""
        msg = MagicMock()
        msg.content = "Hello world"
        assert _extract_content(msg) == "Hello world"

    def test_list_content_strings(self):
        """List of plain strings."""
        msg = MagicMock()
        msg.content = ["Part 1", "Part 2"]
        assert _extract_content(msg) == "Part 1\nPart 2"

    def test_list_content_dicts(self):
        """List of text-content dicts (Anthropic-style)."""
        msg = MagicMock()
        msg.content = [{"type": "text", "text": "Hello"}, {"type": "text", "text": "World"}]
        assert _extract_content(msg) == "Hello\nWorld"

    def test_none_content(self):
        """None .content should return empty string."""
        msg = MagicMock()
        msg.content = None
        assert _extract_content(msg) == ""

    def test_no_content_attribute(self):
        """Object without .content falls back to str()."""
        result = _extract_content(42)
        assert result == "42"

    def test_empty_list_content(self):
        """Empty list .content falls back to str(content)."""
        msg = MagicMock()
        msg.content = []
        result = _extract_content(msg)
        assert result == "[]"


class TestCircuitBreakerThreadSafety:
    """Tests for thread-safe access to _circuit_breakers."""

    def setup_method(self):
        reset_model_status()

    def teardown_method(self):
        reset_model_status()

    def test_circuit_breaker_lock_is_lock(self):
        """_circuit_breaker_lock is a reentrant lock (RLock)."""
        assert isinstance(_circuit_breaker_lock, type(threading.RLock()))

    def test_get_circuit_breaker_concurrent_no_crash(self):
        """Concurrent _get_circuit_breaker calls must not raise or corrupt state."""
        errors: list[Exception] = []

        def worker(idx: int) -> None:
            try:
                for j in range(20):
                    _get_circuit_breaker("provider", f"model-{idx}-{j}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"

    def test_reset_model_status_concurrent_no_crash(self):
        """Concurrent reset_model_status calls must not raise."""
        for i in range(50):
            _get_circuit_breaker("p", f"m-{i}")

        errors: list[Exception] = []

        def resetter() -> None:
            try:
                for _ in range(10):
                    reset_model_status()
            except Exception as exc:
                errors.append(exc)

        def getter() -> None:
            try:
                for i in range(10):
                    _get_circuit_breaker("p", f"m-{i}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=resetter) for _ in range(5)] + [
            threading.Thread(target=getter) for _ in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
