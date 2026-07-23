"""OpenAI and OpenAI-compatible LLM provider (xAI, vLLM, Groq, Together, etc.)."""

from __future__ import annotations

from typing import Any

from src.providers.defaults import CHAT_MODELS, EMBEDDING_MODELS

# ── Lazy imports ─────────────────────────────────────────────────────

try:
    from langchain_openai import ChatOpenAI

    CHAT_AVAILABLE = True
except ImportError:
    ChatOpenAI = None  # type: ignore[misc, assignment]
    CHAT_AVAILABLE = False


# ── DeepSeek reasoning-model wrapper ─────────────────────────────────
#
# DeepSeek's reasoning models (deepseek-reasoner / any model that uses
# thinking mode) require that the assistant's `reasoning_content` field
# be echoed back verbatim in the next API call.  LangChain's ChatOpenAI
# serialisation silently drops it, causing a 400 error on the second
# tool-round or any multi-turn exchange.
#
# This subclass injects `reasoning_content` back into assistant message
# dicts whenever the original AIMessage carried it in additional_kwargs.

if CHAT_AVAILABLE:

    class _DeepSeekChatModel(ChatOpenAI):  # type: ignore[misc]
        """ChatOpenAI subclass that preserves reasoning_content for DeepSeek.

        DeepSeek reasoning models require the assistant's `reasoning_content`
        to be echoed back in subsequent API calls.  LangChain's two-step failure:
          1. _convert_dict_to_message() never extracts reasoning_content from the
             API response dict, so AIMessage.additional_kwargs never has it.
          2. _get_request_payload() therefore has nothing to re-inject.

        Fix — Part A (_create_chat_result): extract reasoning_content from the
        raw API response *before* LangChain discards it, then store it in
        AIMessage.additional_kwargs so Part B can re-inject it.

        Fix — Part B (_get_request_payload): on the next API call, walk message
        pairs in lockstep and inject reasoning_content into assistant dicts.
        """

        def _create_chat_result(self, response: Any, generation_info: Any = None) -> Any:
            # ── Part A: capture reasoning_content from raw API response ──────
            # Extract BEFORE calling super(), which discards it via
            # _convert_dict_to_message(). OpenAI SDK stores unknown fields in
            # __pydantic_extra__ (extra="allow"), so model_dump() includes them.
            rc_by_index: dict[int, str] = {}
            try:
                choices = (
                    response.get("choices", [])
                    if isinstance(response, dict)
                    else getattr(response, "choices", [])
                )
                for i, choice in enumerate(choices):
                    msg = (
                        choice.get("message", {})
                        if isinstance(choice, dict)
                        else getattr(choice, "message", None)
                    )
                    if msg is None:
                        continue
                    rc = (
                        msg.get("reasoning_content")
                        if isinstance(msg, dict)
                        else (
                            getattr(msg, "reasoning_content", None)
                            or (getattr(msg, "__pydantic_extra__", None) or {}).get(
                                "reasoning_content"
                            )
                        )
                    )
                    if rc:
                        rc_by_index[i] = rc
            except (AttributeError, KeyError, TypeError, IndexError):
                pass  # introspection failure — super() path still runs
            except Exception as _exc:
                import logging as _logging

                _logging.getLogger("cogtrix.providers.openai").warning(
                    "DeepSeek _create_chat_result: unexpected error capturing "
                    "reasoning_content: %s",
                    _exc,
                    exc_info=True,
                )

            result = super()._create_chat_result(response, generation_info)

            # Store captured reasoning_content in the corresponding AIMessage
            for i, gen in enumerate(result.generations):
                if i in rc_by_index and hasattr(gen, "message"):
                    gen.message.additional_kwargs["reasoning_content"] = rc_by_index[i]

            return result

        def _convert_chunk_to_generation_chunk(
            self, chunk: Any, default_chunk_class: Any, base_generation_info: Any
        ) -> Any:
            # ── Part C: capture reasoning_content from streaming delta chunks ──
            # Streaming uses _convert_chunk_to_generation_chunk instead of
            # _create_chat_result, so reasoning_content must be captured here.
            result = super()._convert_chunk_to_generation_chunk(
                chunk, default_chunk_class, base_generation_info
            )
            if result is None:
                return result
            try:
                # chunk is the full streaming response dict: {"choices": [{"delta": {...}}]}
                # reasoning_content lives at choices[0]["delta"], NOT at the top-level chunk.
                if isinstance(chunk, dict):
                    choices = (
                        chunk.get("choices") or (chunk.get("chunk") or {}).get("choices") or []
                    )
                    if choices:
                        delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
                        rc = delta.get("reasoning_content") if isinstance(delta, dict) else None
                        if rc and hasattr(result, "message"):
                            result.message.additional_kwargs["reasoning_content"] = rc
            except (AttributeError, KeyError, TypeError, IndexError):
                pass
            return result

        def _get_request_payload(self, input_: Any, *, stop: Any = None, **kwargs: Any) -> dict:
            # ── Part B: re-inject reasoning_content into outgoing message dicts ─
            payload = super()._get_request_payload(input_, stop=stop, **kwargs)
            try:
                msgs = getattr(self, "_convert_input", lambda x: x)(input_)
                if hasattr(msgs, "to_messages"):
                    msgs = msgs.to_messages()
                if isinstance(msgs, list):
                    # _DeepSeekChatModel is only instantiated for api.deepseek.com.
                    # Reasoning-capable DeepSeek models require reasoning_content on
                    # EVERY assistant message in the history — always, not only when
                    # some messages already carry it.  Two cases require the fill:
                    #
                    #   1. /think result messages: created as plain AIMessage(content=…)
                    #      via memory_manager.update legacy path — no reasoning_content.
                    #   2. Sessions loaded from old JSON (before rc serialization was
                    #      added): ALL messages lack the field.
                    #
                    # Using "" satisfies the DeepSeek API for messages where the
                    # original reasoning chain is unavailable.  Non-reasoning models
                    # (deepseek-chat) ignore the extra field.
                    for key in ("messages", "input"):
                        msg_dicts = payload.get(key, [])
                        if msg_dicts:
                            for msg, msg_dict in zip(msgs, msg_dicts, strict=False):
                                if not (
                                    isinstance(msg_dict, dict)
                                    and msg_dict.get("role") == "assistant"
                                ):
                                    continue
                                rc = getattr(msg, "additional_kwargs", {}).get("reasoning_content")
                                if rc:
                                    msg_dict["reasoning_content"] = rc
                                else:
                                    msg_dict.setdefault("reasoning_content", "")
                            break  # only process the first key that has entries
            except (AttributeError, KeyError, TypeError):
                pass  # introspection failure — payload still sent without reasoning_content
            except Exception as _exc:
                import logging as _logging

                _logging.getLogger("cogtrix.providers.openai").warning(
                    "DeepSeek _get_request_payload: unexpected error re-injecting "
                    "reasoning_content: %s",
                    _exc,
                    exc_info=True,
                )
            return payload

else:
    _DeepSeekChatModel = None  # type: ignore[assignment,misc]

_DEEPSEEK_BASE_URL = "api.deepseek.com"

try:
    from langchain_openai import OpenAIEmbeddings

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    OpenAIEmbeddings = None  # type: ignore[misc, assignment]
    EMBEDDINGS_AVAILABLE = False


def create_chat_model(
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.5,
    **kwargs: Any,
) -> Any:
    """Create an OpenAI (or compatible) chat model.

    Args:
        model: Model name (default: gpt-4.1-mini).
        api_key: API key (``None`` → falls back to ``OPENAI_API_KEY`` env var).
        base_url: Custom endpoint (``None`` → OpenAI default).
        temperature: Sampling temperature.
        **kwargs: Extra keyword arguments forwarded to ``ChatOpenAI``.

    Returns:
        ``ChatOpenAI`` instance.

    Raises:
        ImportError: If ``langchain-openai`` is not installed.
    """
    if not CHAT_AVAILABLE:
        raise ImportError("langchain-openai not installed. Run: pip install langchain-openai")

    llm_kwargs: dict[str, Any] = {
        "model": model or CHAT_MODELS["openai"],
        "temperature": temperature,
        "max_retries": 3,
    }
    if base_url:
        llm_kwargs["base_url"] = base_url
        # OpenAI-compatible endpoints (vLLM, LM Studio, etc.) often require no
        # authentication, but the SDK unconditionally rejects a missing api_key.
        # Use the caller-supplied key when present; fall back to a placeholder so
        # the SDK's client-option check passes without forcing users to invent a
        # key.  "not-required" is deliberately descriptive so it never appears as
        # a confusing literal in SDK error messages (BUG-231).
        llm_kwargs["api_key"] = api_key or "not-required"
    elif api_key:
        llm_kwargs["api_key"] = api_key
    llm_kwargs.update(kwargs)
    cls = (
        _DeepSeekChatModel
        if (base_url and _DEEPSEEK_BASE_URL in base_url and _DeepSeekChatModel is not None)
        else ChatOpenAI
    )
    return cls(**llm_kwargs)


def create_embeddings(
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> Any:
    """Create OpenAI embeddings.

    Args:
        model: Embedding model name (default: text-embedding-3-small).
        api_key: API key.
        base_url: Custom endpoint.

    Returns:
        ``OpenAIEmbeddings`` instance.

    Raises:
        ImportError: If ``langchain-openai`` is not installed.
    """
    if not EMBEDDINGS_AVAILABLE:
        raise ImportError("langchain-openai not installed. Run: pip install langchain-openai")

    emb_kwargs: dict[str, Any] = {"model": model or EMBEDDING_MODELS["openai"]}
    if api_key:
        emb_kwargs["api_key"] = api_key
    if base_url:
        emb_kwargs["openai_api_base"] = base_url
    emb_kwargs.update(kwargs)
    return OpenAIEmbeddings(**emb_kwargs)
