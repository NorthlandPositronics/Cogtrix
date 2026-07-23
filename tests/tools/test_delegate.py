"""Tests for the delegate tool."""

from unittest.mock import MagicMock, patch

from src.tools.delegate import (
    TOOL_CONFIGS,
    DelegateInput,
    DelegateParallelInput,
    DelegateResult,
    ModelCircuitBreaker,
    _build_prompt,
    _resolve_model_alias,
    _validate_json_response,
    configure_delegate,
    delegate_parallel,
    delegate_task,
    get_model_status,
    reset_model_status,
)


class TestConfiguration:
    """Tests for delegate configuration."""

    def test_configure_delegate_updates_config(self):
        """Test that configure_delegate updates the module config."""
        configure_delegate(
            {
                "enabled": False,
                "default_provider": "openai",
                "model_aliases": {"fast": "ollama/llama3:8b"},
            }
        )

        # Config should be updated (we test via _resolve_model_alias)
        provider, model, alias_config = _resolve_model_alias(None, "fast")
        assert provider == "ollama"
        assert model == "llama3:8b"

    def test_resolve_alias_with_provider_model(self):
        """Test resolving alias with provider/model format."""
        configure_delegate(
            {
                "model_aliases": {
                    "smart": "openai/gpt-4",
                    "code": "ollama/codellama:13b",
                }
            }
        )

        provider, model, alias_config = _resolve_model_alias(None, "smart")
        assert provider == "openai"
        assert model == "gpt-4"

        provider, model, alias_config = _resolve_model_alias(None, "code")
        assert provider == "ollama"
        assert model == "codellama:13b"

    def test_resolve_alias_model_only(self):
        """Test resolving alias that's just a model name."""
        configure_delegate(
            {
                "model_aliases": {
                    "default": "llama3:8b",
                }
            }
        )

        provider, model, alias_config = _resolve_model_alias("ollama", "default")
        assert provider == "ollama"
        assert model == "llama3:8b"

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
        """Reset config before each test."""
        configure_delegate(
            {
                "enabled": True,
                "default_provider": "ollama",
                "allowed_providers": ["openai", "ollama"],
            }
        )

    @patch("src.tools.delegate._create_llm")
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
        )

        assert "Delegated to:" in result
        assert "Response:" in result
        assert "This is the summary." in result

    @patch("src.tools.delegate._create_llm")
    def test_delegate_task_json_valid(self, mock_create_llm):
        """Test delegation with valid JSON response."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"result": "success"}'
        mock_llm.invoke.return_value = mock_response
        mock_create_llm.return_value = mock_llm

        result = delegate_task(
            task="Extract data",
            response_format="json",
            provider="ollama",
        )

        assert "JSON Valid:** ✓" in result

    @patch("src.tools.delegate._create_llm")
    def test_delegate_task_json_invalid(self, mock_create_llm):
        """Test delegation with invalid JSON response."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Not valid JSON at all"
        mock_llm.invoke.return_value = mock_response
        mock_create_llm.return_value = mock_llm

        result = delegate_task(
            task="Extract data",
            response_format="json",
            provider="ollama",
        )

        assert "JSON Valid:** ✗" in result

    def test_delegate_task_disabled(self):
        """Test that delegation fails when disabled."""
        configure_delegate({"enabled": False})

        result = delegate_task(task="Test")
        assert "Delegation disabled" in result

    @patch("src.tools.delegate._create_llm")
    def test_delegate_task_error_handling(self, mock_create_llm):
        """Test error handling in delegation."""
        mock_create_llm.side_effect = Exception("Connection failed")

        result = delegate_task(task="Test", provider="ollama")
        assert "failed" in result.lower()


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

    @patch("src.tools.delegate._create_llm")
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

    def test_parallel_empty_tasks(self):
        """Test parallel with empty task list."""
        result = delegate_parallel(tasks=[])
        assert "No tasks provided" in result

    def test_parallel_disabled(self):
        """Test parallel when delegation is disabled."""
        configure_delegate({"enabled": False})

        result = delegate_parallel(tasks=[{"task": "Test"}])
        assert "Delegation disabled" in result


class TestCircuitBreaker:
    """Tests for the circuit breaker functionality."""

    def setup_method(self):
        """Reset circuit breakers and config before each test."""
        reset_model_status()
        configure_delegate(
            {
                "enabled": True,
                "default_provider": "ollama",
                "allowed_providers": ["openai", "ollama"],
                "max_consecutive_failures": 5,
                "circuit_breaker_cooldown": 300,
            }
        )

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
        assert "consecutive failures" in reason

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

    @patch("src.tools.delegate._create_llm")
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

    @patch("src.tools.delegate._create_llm")
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

    @patch("src.tools.delegate._create_llm")
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
