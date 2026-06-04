"""HTTPX-enabled LangChain model wrapper."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import openai
from any_llm.types.completion import ChatCompletion
from langchain_anyllm import ChatAnyLLM
from langchain_anyllm.utils import _convert_delta_to_message_chunk, _convert_message_to_dict
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.messages import AIMessageChunk, BaseMessage, BaseMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from pydantic import ConfigDict, Field


class ChatAnyLLMWithHttpx(ChatAnyLLM):
    """ChatAnyLLM variant that can use caller-provided HTTPX clients."""

    http_client: httpx.Client | None = Field(default=None, exclude=True)
    async_http_client: httpx.AsyncClient | None = Field(default=None, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _make_openai_client(self) -> openai.OpenAI:
        return openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            http_client=self.http_client,
        )

    def _make_async_openai_client(self) -> openai.AsyncOpenAI:
        return openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            http_client=self.async_http_client,
        )

    def _build_create_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Strip constructor-only keys before passing to completions.create()."""
        return {
            key: value
            for key, value in params.items()
            if key not in ("api_key", "api_base", "provider", "model")
        }

    def _call_completion(
        self,
        messages: list[dict[str, Any]],
        params: dict[str, Any],
    ) -> ChatCompletion:
        if self.http_client is None:
            return super()._call_completion(messages, params)

        create_params = self._build_create_params(params)
        client = self._make_openai_client()
        return client.chat.completions.create(
            model=self.model.split(":")[-1],
            messages=messages,
            **create_params,
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if self.http_client is None:
            yield from super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs)
            return

        message_dicts = [_convert_message_to_dict(message) for message in messages]
        params = self._create_params(stop, **kwargs)
        params["stream"] = True
        if not self._is_anthropic_model() and "stream_options" not in params and self.stream_options:
            params["stream_options"] = self.stream_options

        create_params = self._build_create_params(params)
        client = self._make_openai_client()
        stream = client.chat.completions.create(
            model=self.model.split(":")[-1],
            messages=message_dicts,
            **create_params,
        )

        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        for stream_chunk in stream:
            if len(stream_chunk.choices) == 0:
                if getattr(stream_chunk, "usage", None):
                    usage = stream_chunk.usage.model_dump()
                    usage_metadata = self._extract_usage_metadata(usage)
                    if usage_metadata:
                        usage_chunk = AIMessageChunk(
                            content="",
                            response_metadata={"model_name": self.model},
                            usage_metadata=usage_metadata,
                        )
                        yield ChatGenerationChunk(message=usage_chunk)
                continue

            for choice in stream_chunk.choices:
                message_chunk = _convert_delta_to_message_chunk(choice.delta, default_chunk_class)

                if choice.finish_reason and getattr(stream_chunk, "usage", None):
                    if isinstance(message_chunk, AIMessageChunk):
                        usage = stream_chunk.usage.model_dump()
                        message_chunk.usage_metadata = self._extract_usage_metadata(usage)
                        message_chunk.response_metadata = {"model_name": self.model}

                default_chunk_class = message_chunk.__class__
                generation_chunk = ChatGenerationChunk(message=message_chunk)
                if run_manager:
                    content = message_chunk.content
                    if isinstance(content, str):
                        run_manager.on_llm_new_token(content, chunk=generation_chunk)
                yield generation_chunk

    async def _acall_completion(
        self,
        messages: list[dict[str, Any]],
        params: dict[str, Any],
    ) -> ChatCompletion:
        if self.async_http_client is None:
            return await super()._acall_completion(messages, params)

        create_params = self._build_create_params(params)
        client = self._make_async_openai_client()
        return await client.chat.completions.create(
            model=self.model.split(":")[-1],
            messages=messages,
            **create_params,
        )

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        if self.async_http_client is None:
            async for chunk in super()._astream(messages, stop=stop, run_manager=run_manager, **kwargs):
                yield chunk
            return

        message_dicts = [_convert_message_to_dict(message) for message in messages]
        params = self._create_params(stop, **kwargs)
        params["stream"] = True
        if not self._is_anthropic_model() and "stream_options" not in params and self.stream_options:
            params["stream_options"] = self.stream_options

        create_params = self._build_create_params(params)
        client = self._make_async_openai_client()
        stream = await client.chat.completions.create(
            model=self.model.split(":")[-1],
            messages=message_dicts,
            **create_params,
        )

        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        async for stream_chunk in stream:
            if len(stream_chunk.choices) == 0:
                if getattr(stream_chunk, "usage", None):
                    usage = stream_chunk.usage.model_dump()
                    usage_metadata = self._extract_usage_metadata(usage)
                    if usage_metadata:
                        usage_chunk = AIMessageChunk(
                            content="",
                            response_metadata={"model_name": self.model},
                            usage_metadata=usage_metadata,
                        )
                        yield ChatGenerationChunk(message=usage_chunk)
                continue

            for choice in stream_chunk.choices:
                message_chunk = _convert_delta_to_message_chunk(choice.delta, default_chunk_class)

                if choice.finish_reason and getattr(stream_chunk, "usage", None):
                    if isinstance(message_chunk, AIMessageChunk):
                        usage = stream_chunk.usage.model_dump()
                        message_chunk.usage_metadata = self._extract_usage_metadata(usage)
                        message_chunk.response_metadata = {"model_name": self.model}

                default_chunk_class = message_chunk.__class__
                generation_chunk = ChatGenerationChunk(message=message_chunk)
                if run_manager:
                    content = message_chunk.content
                    if isinstance(content, str):
                        await run_manager.on_llm_new_token(content, chunk=generation_chunk)
                yield generation_chunk
