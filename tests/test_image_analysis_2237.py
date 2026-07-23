"""Tests for #2237 — inbound-image vision plumbing."""

from __future__ import annotations

import base64
import time
from unittest.mock import MagicMock, patch

from src.assistant.channel import IncomingMessage
from src.assistant.channels.whatsapp import WhatsAppChannel
from src.tools._whatsapp_client import (
    ChatOverview,
    Message,
    WahaClient,
)

# ---------------------------------------------------------------------------
# WahaClient.download_media
# ---------------------------------------------------------------------------


class TestDownloadMedia:
    """Unit tests for WahaClient.download_media."""

    def _make_client(self) -> WahaClient:
        return WahaClient(base_url="http://localhost:3000", api_key="test-key", session="default")

    def _mock_response(
        self,
        *,
        status_code: int = 200,
        content_type: str = "image/jpeg",
        content_length: str | None = None,
        body: bytes = b"\xff\xd8\xff" * 10,
    ) -> MagicMock:
        resp = MagicMock()
        resp.status_code = status_code
        resp.headers = {"Content-Type": content_type}
        if content_length is not None:
            resp.headers["Content-Length"] = content_length
        resp.raw = MagicMock()
        resp.raw.read.return_value = body
        return resp

    def test_image_jpeg_returns_bytes_and_mimetype(self):
        client = self._make_client()
        body = b"\xff\xd8\xff" * 5
        resp = self._mock_response(content_type="image/jpeg", body=body)
        with patch("src.tools._whatsapp_client.requests.get", return_value=resp) as mock_get:
            result = client.download_media("http://waha/media/abc123")

        assert result is not None
        data, mimetype = result
        assert data == body
        assert mimetype == "image/jpeg"
        # Auth header only — no Content-Type
        call_kwargs = mock_get.call_args
        sent_headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers", {})
        assert sent_headers.get("X-Api-Key") == "test-key"
        assert "Content-Type" not in sent_headers

    def test_image_png_returns_correct_mimetype(self):
        client = self._make_client()
        body = b"\x89PNG" + b"\x00" * 8
        resp = self._mock_response(content_type="image/png", body=body)
        with patch("src.tools._whatsapp_client.requests.get", return_value=resp):
            result = client.download_media("http://waha/media/png")

        assert result is not None
        _, mimetype = result
        assert mimetype == "image/png"

    def test_non_image_content_type_returns_none(self):
        client = self._make_client()
        resp = self._mock_response(content_type="application/pdf", body=b"%PDF")
        with patch("src.tools._whatsapp_client.requests.get", return_value=resp):
            result = client.download_media("http://waha/media/doc")

        assert result is None

    def test_video_content_type_returns_none(self):
        client = self._make_client()
        resp = self._mock_response(content_type="video/mp4", body=b"\x00\x00\x00\x00")
        with patch("src.tools._whatsapp_client.requests.get", return_value=resp):
            result = client.download_media("http://waha/media/vid")

        assert result is None

    def test_http_error_returns_none(self):
        client = self._make_client()
        resp = self._mock_response(status_code=404, content_type="image/jpeg", body=b"Not Found")
        with patch("src.tools._whatsapp_client.requests.get", return_value=resp):
            result = client.download_media("http://waha/media/missing")

        assert result is None

    def test_oversize_via_content_length_returns_none(self):
        client = self._make_client()
        big = 9 * 1024 * 1024
        resp = self._mock_response(
            content_type="image/jpeg",
            content_length=str(big),
            body=b"\xff" * 100,
        )
        with patch("src.tools._whatsapp_client.requests.get", return_value=resp):
            result = client.download_media("http://waha/media/big", max_bytes=8 * 1024 * 1024)

        assert result is None

    def test_oversize_via_read_returns_none(self):
        client = self._make_client()
        max_bytes = 100
        # read() returns more than max_bytes + 1
        resp = self._mock_response(content_type="image/jpeg", body=b"\xff" * 200)
        with patch("src.tools._whatsapp_client.requests.get", return_value=resp):
            result = client.download_media("http://waha/media/big2", max_bytes=max_bytes)

        assert result is None

    def test_connection_error_returns_none(self):
        import requests as _requests

        client = self._make_client()
        with patch(
            "src.tools._whatsapp_client.requests.get",
            side_effect=_requests.exceptions.ConnectionError("refused"),
        ):
            result = client.download_media("http://waha/media/noconn")

        assert result is None

    def test_timeout_returns_none(self):
        import requests as _requests

        client = self._make_client()
        with patch(
            "src.tools._whatsapp_client.requests.get",
            side_effect=_requests.exceptions.Timeout("timed out"),
        ):
            result = client.download_media("http://waha/media/slow")

        assert result is None

    def test_generic_exception_returns_none(self):
        client = self._make_client()
        with patch(
            "src.tools._whatsapp_client.requests.get",
            side_effect=RuntimeError("unexpected"),
        ):
            result = client.download_media("http://waha/media/boom")

        assert result is None

    def test_no_api_key_sends_no_auth_header(self):
        client = WahaClient(base_url="http://localhost:3000", api_key=None)
        body = b"\xff\xd8\xff"
        resp = self._mock_response(content_type="image/jpeg", body=body)
        with patch("src.tools._whatsapp_client.requests.get", return_value=resp) as mock_get:
            result = client.download_media("http://waha/media/noauth")

        assert result is not None
        call_kwargs = mock_get.call_args
        sent_headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers", {})
        assert "X-Api-Key" not in sent_headers

    def test_content_type_with_charset_param_parsed_correctly(self):
        """mimetype strips the ;charset= suffix before checking image/*."""
        client = self._make_client()
        body = b"\x89PNG" + b"\x00" * 4
        resp = self._mock_response(content_type="image/png; charset=utf-8", body=body)
        with patch("src.tools._whatsapp_client.requests.get", return_value=resp):
            result = client.download_media("http://waha/media/charset")

        assert result is not None
        _, mimetype = result
        assert mimetype == "image/png"


# ---------------------------------------------------------------------------
# WhatsAppChannel._process_message — image handling
# ---------------------------------------------------------------------------


def _make_whatsapp_channel(
    analyze_media: bool = True,
) -> WhatsAppChannel:
    return WhatsAppChannel(
        {
            "waha_url": "http://localhost:3000",
            "api_key": "key",
            "session": "default",
            "analyze_media": analyze_media,
        }
    )


def _make_waha_message(
    *,
    msg_id: str = "msg-1",
    body: str = "hello",
    has_media: bool = False,
    media_url: str | None = None,
    from_me: bool = False,
    timestamp: int = 1_000_000,
    from_number: str = "15551234567@c.us",
) -> Message:
    return Message(
        id=msg_id,
        timestamp=timestamp,
        from_number=from_number,
        body=body,
        from_me=from_me,
        has_media=has_media,
        media_url=media_url,
    )


def _make_chat_overview(chat_id: str = "15551234567@c.us") -> ChatOverview:
    return ChatOverview(id=chat_id, name="Test Contact")


class TestWhatsAppChannelImageHandling:
    """Tests for _process_message image download and IncomingMessage.images."""

    def test_text_only_message_has_empty_images(self):
        ch = _make_whatsapp_channel()
        msg = _make_waha_message(body="just text", has_media=False)
        chat = _make_chat_overview()

        with patch.object(ch._client, "download_media") as mock_dl:
            result = ch._process_message(msg, chat, now=1_000_100.0)

        mock_dl.assert_not_called()
        assert result is not None
        assert result.images == []

    def test_image_message_populates_images_field(self):
        ch = _make_whatsapp_channel(analyze_media=True)
        raw_bytes = b"\xff\xd8\xff" * 4
        msg = _make_waha_message(
            body="look at this",
            has_media=True,
            media_url="http://waha/media/abc",
        )
        chat = _make_chat_overview()

        with patch.object(ch._client, "download_media", return_value=(raw_bytes, "image/jpeg")):
            result = ch._process_message(msg, chat, now=1_000_100.0)

        assert result is not None
        assert len(result.images) == 1
        expected_uri = f"data:image/jpeg;base64,{base64.b64encode(raw_bytes).decode()}"
        assert result.images[0] == expected_uri

    def test_media_only_message_not_skipped_when_image_downloaded(self):
        """Empty body + image = message must NOT be discarded."""
        ch = _make_whatsapp_channel(analyze_media=True)
        raw_bytes = b"\x89PNG" + b"\x00" * 8
        msg = _make_waha_message(body="", has_media=True, media_url="http://waha/media/img")
        chat = _make_chat_overview()

        with patch.object(ch._client, "download_media", return_value=(raw_bytes, "image/png")):
            result = ch._process_message(msg, chat, now=1_000_100.0)

        assert result is not None
        assert len(result.images) == 1

    def test_media_only_message_skipped_when_download_fails(self):
        """Empty body + failed download = message is still discarded (no text, no images)."""
        ch = _make_whatsapp_channel(analyze_media=True)
        msg = _make_waha_message(body="", has_media=True, media_url="http://waha/media/bad")
        chat = _make_chat_overview()

        with patch.object(ch._client, "download_media", return_value=None):
            result = ch._process_message(msg, chat, now=1_000_100.0)

        assert result is None

    def test_analyze_media_false_skips_download(self):
        ch = _make_whatsapp_channel(analyze_media=False)
        msg = _make_waha_message(body="caption", has_media=True, media_url="http://waha/m")
        chat = _make_chat_overview()

        with patch.object(ch._client, "download_media") as mock_dl:
            result = ch._process_message(msg, chat, now=1_000_100.0)

        mock_dl.assert_not_called()
        assert result is not None
        assert result.images == []

    def test_has_media_false_skips_download(self):
        ch = _make_whatsapp_channel(analyze_media=True)
        msg = _make_waha_message(body="no media here", has_media=False, media_url=None)
        chat = _make_chat_overview()

        with patch.object(ch._client, "download_media") as mock_dl:
            ch._process_message(msg, chat, now=1_000_100.0)

        mock_dl.assert_not_called()

    def test_media_url_none_skips_download(self):
        ch = _make_whatsapp_channel(analyze_media=True)
        msg = _make_waha_message(body="", has_media=True, media_url=None)
        chat = _make_chat_overview()

        with patch.object(ch._client, "download_media") as mock_dl:
            result = ch._process_message(msg, chat, now=1_000_100.0)

        mock_dl.assert_not_called()
        # No body, no images → skipped
        assert result is None


# ---------------------------------------------------------------------------
# prepare_messages_with_context — multimodal content
# ---------------------------------------------------------------------------


class TestPrepareMessagesWithContext:
    """Tests for the user_images parameter of prepare_messages_with_context."""

    def test_without_images_returns_plain_string_content(self):
        from src.agent.core import prepare_messages_with_context

        msgs = prepare_messages_with_context(
            history_messages=[],
            user_input="hello",
        )
        assert len(msgs) == 1
        last = msgs[-1]
        assert last.content == "hello"

    def test_with_images_returns_multimodal_list(self):
        from src.agent.core import prepare_messages_with_context

        uri = "data:image/png;base64,AAAA"
        msgs = prepare_messages_with_context(
            history_messages=[],
            user_input="describe this",
            user_images=[uri],
        )
        last = msgs[-1]
        assert isinstance(last.content, list)
        assert last.content[0] == {"type": "text", "text": "describe this"}
        assert last.content[1] == {"type": "image_url", "image_url": {"url": uri}}

    def test_multiple_images_all_included(self):
        from src.agent.core import prepare_messages_with_context

        uris = [
            "data:image/jpeg;base64,AAA=",
            "data:image/png;base64,BBB=",
        ]
        msgs = prepare_messages_with_context(
            history_messages=[],
            user_input="two images",
            user_images=uris,
        )
        last = msgs[-1]
        assert isinstance(last.content, list)
        assert len(last.content) == 3  # 1 text + 2 images
        assert last.content[0]["type"] == "text"
        assert last.content[1]["image_url"]["url"] == uris[0]
        assert last.content[2]["image_url"]["url"] == uris[1]

    def test_empty_images_list_returns_plain_string(self):
        from src.agent.core import prepare_messages_with_context

        msgs = prepare_messages_with_context(
            history_messages=[],
            user_input="no images",
            user_images=[],
        )
        last = msgs[-1]
        assert last.content == "no images"

    def test_none_images_returns_plain_string(self):
        from src.agent.core import prepare_messages_with_context

        msgs = prepare_messages_with_context(
            history_messages=[],
            user_input="no images",
            user_images=None,
        )
        last = msgs[-1]
        assert last.content == "no images"

    def test_context_prefix_still_injected_with_images(self):
        from src.agent.core import prepare_messages_with_context

        uri = "data:image/png;base64,AAAA"
        msgs = prepare_messages_with_context(
            history_messages=[],
            user_input="with prefix",
            context_prefix="some context",
            user_images=[uri],
        )
        # First message is context, last is the multimodal user message
        assert len(msgs) == 2
        assert "some context" in msgs[0].content
        assert isinstance(msgs[-1].content, list)


# ---------------------------------------------------------------------------
# run_agent threads user_images to prepare_messages_with_context
# ---------------------------------------------------------------------------


class TestRunAgentThreadsUserImages:
    """Verify that run_agent passes user_images to prepare_messages_with_context."""

    def test_user_images_forwarded_to_prepare_messages(self):
        from src.orchestration.runner import run_agent

        captured: dict = {}

        def fake_prepare(
            history_messages,
            user_input,
            context_prefix=None,
            max_context_tokens=None,
            user_images=None,
        ):
            captured["user_images"] = user_images
            try:
                from langchain_core.messages import HumanMessage

                return [HumanMessage(content=user_input)]
            except ImportError:
                return [{"type": "human", "content": user_input}]

        uri = "data:image/jpeg;base64,AAAA"

        mock_config = MagicMock()
        mock_config.system_prompt = "sys"
        mock_config.max_context_tokens = None
        mock_config.llm_timeout = 30
        mock_config.compression_min_age = None
        mock_config.compression_min_chars = None
        mock_config.task_ownership_classifier_enabled = False
        mock_config.bound_cache = None
        mock_config.compression_cache = None
        mock_config.active_tools_list = []
        mock_config.available_tools = {}
        mock_config.parallel_tool_execution = False
        mock_config.memory_manager = None

        with patch(
            "src.orchestration.runner.prepare_messages_with_context", side_effect=fake_prepare
        ) as mock_pmc:
            with patch("src.orchestration.runner.build_agent_graph") as mock_graph_fn:
                mock_graph = MagicMock()
                mock_graph.stream.return_value = iter(
                    [{"messages": [MagicMock(content="ok", type="ai")]}]
                )
                mock_graph_fn.return_value = mock_graph
                with patch("src.orchestration.runner._drain_background_compression_jobs"):
                    with patch("src.orchestration.runner.classify_task_complexity") as mock_cls:
                        mock_cls.return_value = MagicMock()
                        with patch("src.orchestration.runner.classify_task_ownership") as mock_own:
                            mock_own.return_value = MagicMock(
                                mode=MagicMock(name="EXECUTE"),
                                confidence=1.0,
                                raw_signal="execute",
                                inferred_action=None,
                            )
                            try:
                                run_agent(
                                    user_input="describe",
                                    history_messages=[],
                                    registry=MagicMock(),
                                    approvals=set(),
                                    config=mock_config,
                                    user_images=[uri],
                                )
                            except Exception:
                                pass

        assert mock_pmc.called, "prepare_messages_with_context was not called"
        assert captured.get("user_images") == [uri]


# ---------------------------------------------------------------------------
# handler._prepare_agent_call returns images; _run_agent forwards user_images
# ---------------------------------------------------------------------------


class TestHandlerImageThreading:
    """Verify handler plumbing passes images through to the runner."""

    def _make_handler(self) -> MagicMock:
        from src.assistant.handler import MessageHandler

        handler = object.__new__(MessageHandler)
        handler._session_mgr = MagicMock()
        handler._llm = MagicMock()
        handler._system_prompt = "sys"
        handler._registry = MagicMock()
        handler._approvals = set()
        handler._max_context_tokens = None
        handler._llm_timeout = 30
        handler._compression_llm = None
        handler._knowledge_store = None
        handler._guardrails = MagicMock()
        handler._guardrails.check_tool_call = MagicMock()
        handler._agent_runner = MagicMock(return_value="response")
        handler._parallel_tool_execution = False
        handler._config = {}
        handler._services_config = {}
        handler._github_default_repo = ""
        handler._max_response_length = 4000
        handler._scheduler = None
        handler._deferral_mgr = None
        handler._workflow_registry = None
        handler._campaign_mgr = None
        handler._datamarking_enabled = False
        handler._excluded_tools = frozenset()
        handler._available_tools = {}
        handler._active_tools = []
        return handler  # type: ignore[return-value]

    def _make_msg(self, images: list[str] | None = None) -> IncomingMessage:
        return IncomingMessage(
            channel="whatsapp",
            chat_id="123@c.us",
            message_id="m1",
            sender_id="sender",
            sender_name="Alice",
            text="caption",
            timestamp=float(time.time()),
            images=images or [],
        )

    def test_prepare_agent_call_returns_images(self):
        handler = self._make_handler()
        uri = "data:image/jpeg;base64,AAAA"
        msg = self._make_msg(images=[uri])
        context = MagicMock()
        context.messages = []

        result = handler._prepare_agent_call(msg, context)

        assert len(result) == 6
        images = result[5]
        assert images == [uri]

    def test_prepare_agent_call_returns_empty_images_when_none(self):
        handler = self._make_handler()
        msg = self._make_msg(images=[])
        context = MagicMock()
        context.messages = []

        result = handler._prepare_agent_call(msg, context)

        assert result[5] == []

    def test_run_agent_forwards_user_images_to_runner(self):
        """_run_agent passes user_images kwarg to the agent_runner callable."""
        handler = self._make_handler()
        session = MagicMock()
        session.memory_manager = MagicMock()
        session.session_key = "whatsapp::123"

        captured_kwargs: dict = {}

        def fake_runner(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return "ok"

        handler._agent_runner = fake_runner

        uri = "data:image/png;base64,BBBB"
        with patch(
            "src.assistant.handler.SessionState", return_value=MagicMock(loaded_tools=set())
        ):
            with patch("src.orchestration.run_config.AgentRunConfig"):
                handler._run_agent(
                    user_input="describe",
                    history_messages=[],
                    context_prefix=None,
                    effective_prompt="sys",
                    active_tools=[],
                    session=session,
                    user_images=[uri],
                )

        assert captured_kwargs.get("user_images") == [uri]

    def test_run_agent_passes_none_user_images_by_default(self):
        handler = self._make_handler()
        session = MagicMock()
        session.memory_manager = MagicMock()
        session.session_key = "whatsapp::123"

        captured_kwargs: dict = {}

        def fake_runner(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return "ok"

        handler._agent_runner = fake_runner

        with patch(
            "src.assistant.handler.SessionState", return_value=MagicMock(loaded_tools=set())
        ):
            with patch("src.orchestration.run_config.AgentRunConfig"):
                handler._run_agent(
                    user_input="text only",
                    history_messages=[],
                    context_prefix=None,
                    effective_prompt="sys",
                    active_tools=[],
                    session=session,
                )

        assert captured_kwargs.get("user_images") is None
