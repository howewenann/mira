"""HTTPX-enabled LangChain model wrapper."""

from __future__ import annotations

from typing import Any

import httpx
import openai
from any_llm.types.completion import ChatCompletion
from langchain_anyllm import ChatAnyLLM
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
