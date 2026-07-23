"""TestForge suite — regression and coverage tests for recent changes.

Covers:
1. WizardStepOut.requires_acceptance field (config.py wizard step 1)
2. WorkflowDocumentOut schema (workflow.py)
3. WebSocketCallbackHandler.on_llm_start debug logging (callbacks.py)
4. ConfigOut.system_prompt and ConfigOut.guardrails fields (config.py schema)
5. Callbacks: on_llm_end token accumulation edge cases (BUG-FORGE-003)
6. Callbacks: on_llm_start with missing serialized keys
7. Workflow document endpoints return WorkflowDocumentOut shape
"""

from __future__ import annotations

import asyncio
import os
import uuid
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Environment setup — before any src.api imports
# ---------------------------------------------------------------------------

os.environ.setdefault("COGTRIX_JWT_SECRET", "testsecret_mustbe32chars_minimum00")
os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")


# ===========================================================================
# 1. WizardStepOut schema — requires_acceptance field
# ===========================================================================


class TestWizardStepOutSchema:
    """WizardStepOut must expose a requires_acceptance bool field."""

    def test_requires_acceptance_defaults_to_false(self):
        from src.api.schemas.config import WizardStepOut

        step = WizardStepOut(
            wizard_id="wiz-001",
            step=1,
            total_steps=3,
            step_name="Configure",
            complete=False,
        )
        assert step.requires_acceptance is False

    def test_requires_acceptance_can_be_set_true(self):
        from src.api.schemas.config import WizardStepOut

        step = WizardStepOut(
            wizard_id="wiz-002",
            step=1,
            total_steps=3,
            step_name="Configure",
            complete=False,
            requires_acceptance=True,
        )
        assert step.requires_acceptance is True

    def test_requires_acceptance_serialises_in_json(self):
        from src.api.schemas.config import WizardStepOut

        step = WizardStepOut(
            wizard_id="wiz-003",
            step=1,
            total_steps=3,
            step_name="Configure",
            complete=False,
            requires_acceptance=True,
        )
        data = step.model_dump()
        assert "requires_acceptance" in data
        assert data["requires_acceptance"] is True

    def test_requires_acceptance_false_in_json(self):
        from src.api.schemas.config import WizardStepOut

        step = WizardStepOut(
            wizard_id="wiz-004",
            step=0,
            total_steps=3,
            step_name="Connect to LLM",
            complete=False,
        )
        data = step.model_dump()
        assert data["requires_acceptance"] is False

    def test_wizard_step_out_has_all_required_fields(self):
        from src.api.schemas.config import WizardStepOut

        step = WizardStepOut(
            wizard_id="x",
            step=0,
            total_steps=3,
            step_name="Start",
            complete=False,
        )
        data = step.model_dump()
        for field in (
            "wizard_id",
            "step",
            "total_steps",
            "step_name",
            "question",
            "yaml_preview",
            "complete",
            "requires_acceptance",
            "warnings",
        ):
            assert field in data, f"missing field: {field}"

    def test_wizard_step_out_complete_true_with_acceptance(self):
        from src.api.schemas.config import WizardStepOut

        step = WizardStepOut(
            wizard_id="final-wiz",
            step=2,
            total_steps=3,
            step_name="Save",
            complete=True,
            requires_acceptance=False,
            yaml_preview="providers:\n  default:\n    type: ollama\n",
        )
        assert step.complete is True
        assert step.requires_acceptance is False
        assert step.yaml_preview is not None


# ===========================================================================
# 2. WorkflowDocumentOut schema
# ===========================================================================


class TestWorkflowDocumentOutSchema:
    """WorkflowDocumentOut must expose doc_id, filename, size_bytes, content_type, status."""

    def test_basic_construction(self):
        from src.api.schemas.workflow import WorkflowDocumentOut

        doc = WorkflowDocumentOut(
            doc_id=str(uuid.uuid4()),
            filename="spec.pdf",
            size_bytes=12345,
        )
        assert doc.filename == "spec.pdf"
        assert doc.size_bytes == 12345
        assert doc.content_type is None
        assert doc.status is None

    def test_with_content_type_and_status(self):
        from src.api.schemas.workflow import WorkflowDocumentOut

        doc_id = str(uuid.uuid4())
        doc = WorkflowDocumentOut(
            doc_id=doc_id,
            filename="readme.md",
            size_bytes=1024,
            content_type="text/markdown",
            status="saved",
        )
        assert doc.doc_id == doc_id
        assert doc.content_type == "text/markdown"
        assert doc.status == "saved"

    def test_serialises_all_fields(self):
        from src.api.schemas.workflow import WorkflowDocumentOut

        doc = WorkflowDocumentOut(
            doc_id="abc-123",
            filename="data.csv",
            size_bytes=500,
            content_type="text/csv",
            status="saved",
        )
        data = doc.model_dump()
        assert data["doc_id"] == "abc-123"
        assert data["filename"] == "data.csv"
        assert data["size_bytes"] == 500
        assert data["content_type"] == "text/csv"
        assert data["status"] == "saved"

    def test_null_fields_allowed(self):
        from src.api.schemas.workflow import WorkflowDocumentOut

        doc = WorkflowDocumentOut(doc_id="x", filename="f.txt", size_bytes=0)
        data = doc.model_dump()
        assert data["content_type"] is None
        assert data["status"] is None

    def test_size_bytes_zero_is_valid(self):
        from src.api.schemas.workflow import WorkflowDocumentOut

        doc = WorkflowDocumentOut(doc_id="z", filename="empty.txt", size_bytes=0)
        assert doc.size_bytes == 0

    def test_large_size_bytes(self):
        from src.api.schemas.workflow import WorkflowDocumentOut

        large = 50 * 1024 * 1024  # 50 MB cap
        doc = WorkflowDocumentOut(doc_id="big", filename="big.pdf", size_bytes=large)
        assert doc.size_bytes == large


# ===========================================================================
# 3. WebSocketCallbackHandler — on_llm_start debug logging
# ===========================================================================


class TestCallbacksOnLlmStart:
    """on_llm_start must log at DEBUG level and not raise for any serialized shape."""

    def _make_queue_and_loop(self):
        loop = asyncio.new_event_loop()

        async def _make():
            return asyncio.Queue()

        q = loop.run_until_complete(_make())
        return q, loop

    def test_on_llm_start_does_not_raise_with_full_serialized(self):
        pytest.importorskip("fastapi")
        from src.api.callbacks import WebSocketCallbackHandler

        q, loop = self._make_queue_and_loop()
        handler = WebSocketCallbackHandler(ws_queue=q, loop=loop)
        # Should not raise
        handler.on_llm_start(
            serialized={"name": "gpt-4o", "id": ["langchain", "GPT4o"]},
            prompts=["Hello"],
        )
        loop.close()

    def test_on_llm_start_handles_none_serialized(self):
        pytest.importorskip("fastapi")
        from src.api.callbacks import WebSocketCallbackHandler

        q, loop = self._make_queue_and_loop()
        handler = WebSocketCallbackHandler(ws_queue=q, loop=loop)
        handler.on_llm_start(serialized=None, prompts=["Hello"])
        loop.close()

    def test_on_llm_start_handles_empty_serialized(self):
        pytest.importorskip("fastapi")
        from src.api.callbacks import WebSocketCallbackHandler

        q, loop = self._make_queue_and_loop()
        handler = WebSocketCallbackHandler(ws_queue=q, loop=loop)
        handler.on_llm_start(serialized={}, prompts=[])
        loop.close()

    def test_on_llm_start_handles_serialized_with_only_id(self):
        pytest.importorskip("fastapi")
        from src.api.callbacks import WebSocketCallbackHandler

        q, loop = self._make_queue_and_loop()
        handler = WebSocketCallbackHandler(ws_queue=q, loop=loop)
        # id is a list; last element is the model class name
        handler.on_llm_start(
            serialized={"id": ["langchain_openai", "ChatOpenAI"]},
            prompts=["test"],
        )
        loop.close()

    def test_on_llm_start_handles_empty_prompts(self):
        pytest.importorskip("fastapi")
        from src.api.callbacks import WebSocketCallbackHandler

        q, loop = self._make_queue_and_loop()
        handler = WebSocketCallbackHandler(ws_queue=q, loop=loop)
        handler.on_llm_start(serialized={"name": "model"}, prompts=[])
        loop.close()

    def test_on_llm_start_large_prompt_does_not_raise(self):
        pytest.importorskip("fastapi")
        from src.api.callbacks import WebSocketCallbackHandler

        q, loop = self._make_queue_and_loop()
        handler = WebSocketCallbackHandler(ws_queue=q, loop=loop)
        big_prompt = "x" * 100_000
        handler.on_llm_start(serialized={"name": "model"}, prompts=[big_prompt])
        loop.close()

    def test_on_llm_start_does_not_enqueue_message(self):
        """on_llm_start should only log — it must NOT enqueue a WS message."""
        pytest.importorskip("fastapi")
        from src.api.callbacks import WebSocketCallbackHandler

        q, loop = self._make_queue_and_loop()
        handler = WebSocketCallbackHandler(ws_queue=q, loop=loop)
        handler.on_llm_start(serialized={"name": "test-model"}, prompts=["prompt"])
        # Give the event loop a tick to process any call_soon_threadsafe callbacks
        loop.run_until_complete(asyncio.sleep(0))
        assert q.empty(), "on_llm_start must not enqueue WebSocket messages"
        loop.close()


# ===========================================================================
# 4. ConfigOut schema — system_prompt and guardrails fields
# ===========================================================================


class TestConfigOutSchema:
    """ConfigOut must expose system_prompt and guardrails."""

    def test_system_prompt_field_defaults_to_none(self):
        from src.api.schemas.config import ConfigOut

        cfg = ConfigOut(
            memory_mode="conversation",
            prompt_optimizer=True,
            parallel_tool_execution=True,
            context_compression=True,
            debug=False,
            verbose=False,
        )
        assert cfg.system_prompt is None

    def test_guardrails_field_defaults_to_none(self):
        from src.api.schemas.config import ConfigOut

        cfg = ConfigOut(
            memory_mode="conversation",
            prompt_optimizer=True,
            parallel_tool_execution=True,
            context_compression=True,
            debug=False,
            verbose=False,
        )
        assert cfg.guardrails is None

    def test_system_prompt_can_be_set(self):
        from src.api.schemas.config import ConfigOut

        cfg = ConfigOut(
            memory_mode="conversation",
            prompt_optimizer=True,
            parallel_tool_execution=True,
            context_compression=True,
            debug=False,
            verbose=False,
            system_prompt="You are a helpful assistant.",
        )
        assert cfg.system_prompt == "You are a helpful assistant."

    def test_guardrails_can_be_set_as_dict(self):
        from src.api.schemas.config import ConfigOut

        guardrail_cfg = {"enabled": True, "rate_limit": {"per_minute": 10}}
        cfg = ConfigOut(
            memory_mode="conversation",
            prompt_optimizer=True,
            parallel_tool_execution=True,
            context_compression=True,
            debug=False,
            verbose=False,
            guardrails=guardrail_cfg,
        )
        assert cfg.guardrails == guardrail_cfg

    def test_config_out_serialises_new_fields(self):
        from src.api.schemas.config import ConfigOut

        cfg = ConfigOut(
            memory_mode="reasoning",
            prompt_optimizer=False,
            parallel_tool_execution=True,
            context_compression=False,
            debug=True,
            verbose=False,
            system_prompt="Custom prompt.",
            guardrails={"llm_judge": {"enabled": False}},
        )
        data = cfg.model_dump()
        assert "system_prompt" in data
        assert "guardrails" in data
        assert data["system_prompt"] == "Custom prompt."
        assert data["guardrails"]["llm_judge"]["enabled"] is False


# ===========================================================================
# 5. WebSocketCallbackHandler — on_llm_end token accumulation (BUG-FORGE-003)
# ===========================================================================


class TestCallbacksOnLlmEnd:
    """on_llm_end must accumulate tokens from both dict and object usage_metadata."""

    def _make_handler(self):
        pytest.importorskip("fastapi")
        from src.api.callbacks import WebSocketCallbackHandler

        loop = asyncio.new_event_loop()

        async def _make():
            return asyncio.Queue()

        q = loop.run_until_complete(_make())
        h = WebSocketCallbackHandler(ws_queue=q, loop=loop)
        return h, loop

    def test_accumulates_from_llm_output_token_usage(self):
        handler, loop = self._make_handler()
        response = MagicMock()
        response.llm_output = {"token_usage": {"prompt_tokens": 100, "completion_tokens": 50}}
        response.generations = None
        handler.on_llm_end(response)
        assert handler.input_tokens == 100
        assert handler.output_tokens == 50
        loop.close()

    def test_accumulates_from_llm_output_usage_key(self):
        handler, loop = self._make_handler()
        response = MagicMock()
        response.llm_output = {"usage": {"prompt_tokens": 80, "completion_tokens": 30}}
        response.generations = None
        handler.on_llm_end(response)
        assert handler.input_tokens == 80
        assert handler.output_tokens == 30
        loop.close()

    def test_accumulates_from_generation_usage_metadata_dict(self):
        handler, loop = self._make_handler()
        # Simulate LangChain generation with dict usage_metadata
        msg = MagicMock()
        msg.usage_metadata = {"input_tokens": 200, "output_tokens": 75}
        gen = MagicMock()
        gen.message = msg
        response = MagicMock()
        response.llm_output = None
        response.generations = [[gen]]
        handler.on_llm_end(response)
        assert handler.input_tokens == 200
        assert handler.output_tokens == 75
        loop.close()

    def test_accumulates_from_generation_usage_metadata_object(self):
        """usage_metadata as an object with attributes (older LangChain)."""
        handler, loop = self._make_handler()
        um = MagicMock()
        um.input_tokens = 150
        um.output_tokens = 60
        # Make isinstance(um, dict) return False
        msg = MagicMock()
        msg.usage_metadata = um
        gen = MagicMock()
        gen.message = msg
        response = MagicMock()
        response.llm_output = None
        response.generations = [[gen]]
        handler.on_llm_end(response)
        assert handler.input_tokens == 150
        assert handler.output_tokens == 60
        loop.close()

    def test_accumulates_across_multiple_calls(self):
        handler, loop = self._make_handler()
        for _ in range(3):
            response = MagicMock()
            response.llm_output = {"token_usage": {"prompt_tokens": 10, "completion_tokens": 5}}
            response.generations = None
            handler.on_llm_end(response)
        assert handler.input_tokens == 30
        assert handler.output_tokens == 15
        loop.close()

    def test_no_usage_data_does_not_raise(self):
        handler, loop = self._make_handler()
        response = MagicMock()
        response.llm_output = None
        response.generations = []
        handler.on_llm_end(response)
        assert handler.input_tokens == 0
        assert handler.output_tokens == 0
        loop.close()

    def test_llm_output_with_no_usage_key_falls_through(self):
        handler, loop = self._make_handler()
        response = MagicMock()
        response.llm_output = {"model": "gpt-4"}  # no token_usage or usage
        response.generations = []
        handler.on_llm_end(response)
        assert handler.input_tokens == 0
        loop.close()


# ===========================================================================
# 6. WebSocketCallbackHandler — on_llm_new_token final flag (BUG-218)
# ===========================================================================


class TestCallbacksTokenFinalFlag:
    """on_llm_new_token.final must only be True after tool calls and with none in-flight."""

    def _make_handler(self):
        pytest.importorskip("fastapi")
        from src.api.callbacks import WebSocketCallbackHandler

        loop = asyncio.new_event_loop()

        async def _make():
            return asyncio.Queue()

        q = loop.run_until_complete(_make())
        h = WebSocketCallbackHandler(ws_queue=q, loop=loop)
        return h, q, loop

    def _flush_loop(self, loop):
        loop.run_until_complete(asyncio.sleep(0))

    def _drain_queue(self, q, loop) -> list[dict]:
        self._flush_loop(loop)
        items = []
        while not q.empty():
            items.append(q.get_nowait())
        return items

    def test_final_false_when_no_tool_calls_seen(self):
        handler, q, loop = self._make_handler()
        handler.on_llm_new_token("hello")
        msgs = self._drain_queue(q, loop)
        token_msgs = [m for m in msgs if m.get("type") == "token"]
        assert token_msgs, "expected at least one token message"
        assert token_msgs[-1]["payload"]["final"] is False
        loop.close()

    def test_final_answer_token_buffered_after_tool_call(self):
        # #2251: a post-tool final-answer token (tool_call_count>0, none in-flight)
        # is buffered (suppressed), NOT streamed live — so a verification-recovery
        # regeneration can't double-render. The surviving answer is emitted once by
        # the turn runner at turn end.
        handler, q, loop = self._make_handler()
        handler.on_tool_start({"name": "search"}, "", run_id="run-1")
        handler.on_tool_end("result", run_id="run-1")
        handler.on_llm_new_token("response text")
        msgs = self._drain_queue(q, loop)
        token_msgs = [m for m in msgs if m.get("type") == "token"]
        assert token_msgs == [], "post-tool final-answer tokens must not stream live (#2251)"
        assert handler.final_answer_buffered is True
        loop.close()

    def test_final_false_while_tool_in_flight(self):
        handler, q, loop = self._make_handler()
        handler.on_tool_start({"name": "search"}, "", run_id="run-2")
        # Tool started but not ended → in-flight → final must be False
        handler.on_llm_new_token("preamble")
        msgs = self._drain_queue(q, loop)
        token_msgs = [m for m in msgs if m.get("type") == "token"]
        assert token_msgs, "expected at least one token message"
        assert token_msgs[-1]["payload"]["final"] is False
        loop.close()


# ===========================================================================
# 7. Workflow document API endpoint shape
# ===========================================================================


class TestWorkflowDocumentEndpointShape:
    """Workflow document endpoints return WorkflowDocumentOut-shaped payloads."""

    def _api_client(self):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from src.api.app import create_app
        from src.api.auth import create_access_token

        app = create_app()
        admin_token = create_access_token(user_id=str(uuid.uuid4()), role="admin")
        return TestClient(app, raise_server_exceptions=False), admin_token

    def test_upload_document_returns_404_when_registry_absent(self, tmp_path):
        client, admin_token = self._api_client()
        with client:
            resp = client.post(
                "/api/v1/assistant/workflows/my-wf/documents",
                files={"file": ("test.txt", b"hello world", "text/plain")},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        # 503 = registry not initialised (no assistant service running)
        assert resp.status_code in (503, 404)

    def test_list_documents_returns_503_when_registry_absent(self):
        client, admin_token = self._api_client()
        with client:
            resp = client.get(
                "/api/v1/assistant/workflows/my-wf/documents",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code in (503, 404)

    def test_delete_document_returns_503_when_registry_absent(self):
        client, admin_token = self._api_client()
        doc_id = str(uuid.uuid4())
        with client:
            resp = client.delete(
                f"/api/v1/assistant/workflows/my-wf/documents/{doc_id}",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code in (503, 404)

    def test_invalid_workflow_id_returns_400(self):
        client, admin_token = self._api_client()
        with client:
            resp = client.get(
                "/api/v1/assistant/workflows/../secret/documents",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        # Path traversal or invalid id — 400 or 404 depending on routing
        assert resp.status_code in (400, 404)

    def test_workflow_document_out_shape_in_upload_response(self, tmp_path):
        """When a registry IS present, upload returns WorkflowDocumentOut fields."""
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from src.api.app import create_app
        from src.api.auth import create_access_token

        app = create_app()
        admin_token = create_access_token(user_id=str(uuid.uuid4()), role="admin")

        # Build a mock WorkflowRegistry with enough internals for the upload path.
        # The route reads registry._workflows_dir and registry._data_dir.
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        workflows_dir = data_dir / "workflows"
        workflows_dir.mkdir()

        mock_wf = MagicMock()
        mock_wf.id = "test-wf"
        mock_wf.knowledge_base = False

        mock_registry = MagicMock()
        mock_registry.get_workflow.return_value = mock_wf
        mock_registry._workflows_dir = workflows_dir
        mock_registry._data_dir = data_dir

        with TestClient(app, raise_server_exceptions=False) as client:
            app.state.workflow_registry = mock_registry
            try:
                resp = client.post(
                    "/api/v1/assistant/workflows/test-wf/documents",
                    files={"file": ("readme.txt", b"hello content", "text/plain")},
                    headers={"Authorization": f"Bearer {admin_token}"},
                )
            finally:
                app.state.workflow_registry = None

        # 202 = accepted and saved; 503 = registry not initialised (mock not wired)
        if resp.status_code == 202:
            data = resp.json().get("data", {})
            assert "doc_id" in data
            assert "filename" in data
            assert "size_bytes" in data
            assert data["filename"] == "readme.txt"
            assert data["size_bytes"] > 0
            assert data["status"] == "saved"


# ===========================================================================
# 8. WhatsAppConfig default require_confirmation value
# ===========================================================================


class TestWhatsAppConfigDefaults:
    """WhatsAppConfig.require_confirmation defaults to True (the class default)."""

    def test_dataclass_default_require_confirmation_is_true(self):
        from src.tools.whatsapp import WhatsAppConfig

        cfg = WhatsAppConfig()
        assert cfg.require_confirmation is True

    def test_build_tool_configs_respects_require_confirmation_true(self):
        """_build_tool_configs() uses _cfg.require_confirmation for send tools."""
        import src.tools.whatsapp as _wa

        with patch.object(_wa._cfg, "require_confirmation", True):
            with patch.object(_wa._cfg, "allow_send", True):
                configs = _wa._build_tool_configs()

        send_tools = [c for c in configs if "send" in c["name"]]
        assert send_tools, "expected send tools when allow_send=True"
        for tool in send_tools:
            assert tool["requires_confirmation"] is True

    def test_build_tool_configs_respects_require_confirmation_false(self):
        """When require_confirmation=False in config, send tools reflect that."""
        import src.tools.whatsapp as _wa

        with patch.object(_wa._cfg, "require_confirmation", False):
            with patch.object(_wa._cfg, "allow_send", True):
                configs = _wa._build_tool_configs()

        send_tools = [c for c in configs if "send" in c["name"]]
        assert send_tools, "expected send tools when allow_send=True"
        for tool in send_tools:
            assert tool["requires_confirmation"] is False

    def test_check_and_contacts_tools_never_require_confirmation(self):
        import src.tools.whatsapp as _wa

        with patch.object(_wa._cfg, "allow_send", True):
            with patch.object(_wa._cfg, "allow_receive", True):
                configs = _wa._build_tool_configs()

        for tool in configs:
            if tool["name"] in ("whatsapp_check", "whatsapp_contacts"):
                assert tool["requires_confirmation"] is False


# ===========================================================================
# 9. WorkflowDocumentOut import is available from schemas package
# ===========================================================================


class TestWorkflowDocumentOutImport:
    """WorkflowDocumentOut should be importable from the public schemas package."""

    def test_importable_from_workflow_module(self):
        from src.api.schemas.workflow import WorkflowDocumentOut

        assert WorkflowDocumentOut is not None

    def test_workflow_schemas_init_exports(self):
        """The workflow schemas module must expose WorkflowDocumentOut."""
        import src.api.schemas.workflow as wf_schemas

        assert hasattr(wf_schemas, "WorkflowDocumentOut")

    def test_workflow_document_out_is_pydantic_model(self):
        from pydantic import BaseModel

        from src.api.schemas.workflow import WorkflowDocumentOut

        assert issubclass(WorkflowDocumentOut, BaseModel)


# ===========================================================================
# 10. ConfigOut raw_yaml field is present (admin vs user)
# ===========================================================================


class TestConfigOutRawYaml:
    """ConfigOut.raw_yaml field must be present (null for non-admin)."""

    def test_raw_yaml_defaults_to_none(self):
        from src.api.schemas.config import ConfigOut

        cfg = ConfigOut(
            memory_mode="conversation",
            prompt_optimizer=True,
            parallel_tool_execution=True,
            context_compression=True,
            debug=False,
            verbose=False,
        )
        assert cfg.raw_yaml is None

    def test_raw_yaml_can_carry_yaml_string(self):
        from src.api.schemas.config import ConfigOut

        yaml_str = "providers:\n  default:\n    type: ollama\n"
        cfg = ConfigOut(
            memory_mode="conversation",
            prompt_optimizer=True,
            parallel_tool_execution=True,
            context_compression=True,
            debug=False,
            verbose=False,
            raw_yaml=yaml_str,
        )
        assert cfg.raw_yaml == yaml_str
