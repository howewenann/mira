"""Tests for dynamic model metadata inference."""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import patch

import httpx

from config.metadata import ModelMetadata, apply_model_metadata, infer_model_metadata


class ProfileModel:
    """Tiny model double exposing the LangChain profile field MIRA writes."""

    def __init__(self, profile: dict[str, object] | None = None) -> None:
        self.profile = profile


class FakeAsyncClient:
    """Async HTTPX client double that records direct-network settings."""

    calls: list[dict[str, Any]] = []
    payload: dict[str, Any] = {}
    error: Exception | None = None

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        FakeAsyncClient.calls.append(kwargs)

    async def __aenter__(self) -> "FakeAsyncClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def get(self, url: str, headers: dict[str, str]) -> httpx.Response:
        FakeAsyncClient.calls.append({"url": url, "headers": headers})
        if FakeAsyncClient.error is not None:
            raise FakeAsyncClient.error
        return httpx.Response(200, json=FakeAsyncClient.payload, request=httpx.Request("GET", url))


def lmstudio_config(model: str = "gemma-4-e4b") -> dict[str, Any]:
    """Return the minimal LM Studio config used by metadata tests."""
    return {
        "llm_provider": "lmstudio",
        "llm_model": model,
        "llm_base_url": "http://localhost:1234/v1",
        "llm_api_key": "lm-studio",
    }


class MetadataTests(unittest.IsolatedAsyncioTestCase):
    """Tests for model metadata discovery and profile application."""

    def setUp(self) -> None:
        FakeAsyncClient.calls = []
        FakeAsyncClient.error = None
        FakeAsyncClient.payload = {
            "models": [
                {
                    "key": "other-model",
                    "max_context_length": 999999,
                    "loaded_instances": [{"id": "other-model", "config": {"context_length": 4096}}],
                },
                {
                    "key": "gemma-4-e4b",
                    "max_context_length": 131072,
                    "loaded_instances": [
                        {"id": "gemma-4-e4b@q4", "config": {"context_length": 10000}},
                    ],
                },
            ]
        }

    async def test_lmstudio_metadata_uses_loaded_instance_context(self) -> None:
        """LM Studio metadata should use loaded config.context_length, not max_context_length."""
        with patch("config.metadata.httpx.AsyncClient", FakeAsyncClient):
            metadata = await infer_model_metadata(lmstudio_config())

        self.assertEqual(metadata, ModelMetadata(10000, "lmstudio.api.v1.loaded_instance"))
        self.assertEqual(FakeAsyncClient.calls[0]["trust_env"], False)
        self.assertEqual(FakeAsyncClient.calls[0]["verify"], False)
        self.assertIsNone(FakeAsyncClient.calls[0]["timeout"])
        self.assertEqual(FakeAsyncClient.calls[1]["url"], "http://localhost:1234/api/v1/models")

    async def test_lmstudio_metadata_selects_configured_model(self) -> None:
        """The configured model name should choose the matching LM Studio entry."""
        with patch("config.metadata.httpx.AsyncClient", FakeAsyncClient):
            metadata = await infer_model_metadata(lmstudio_config("other-model"))

        self.assertEqual(metadata.context_tokens, 4096)

    async def test_configured_context_override_wins(self) -> None:
        """Manual context overrides should avoid provider metadata calls."""
        metadata = await infer_model_metadata({**lmstudio_config(), "llm_context_tokens": 8192})

        self.assertEqual(metadata, ModelMetadata(8192, "MIRA_LLM_CONTEXT_TOKENS"))
        self.assertEqual(FakeAsyncClient.calls, [])

    async def test_unavailable_metadata_returns_unknown(self) -> None:
        """Provider metadata failures should not crash startup or turns."""
        FakeAsyncClient.error = httpx.ConnectError("offline")

        with patch("config.metadata.httpx.AsyncClient", FakeAsyncClient):
            metadata = await infer_model_metadata(lmstudio_config())

        self.assertEqual(metadata, ModelMetadata())

    async def test_profile_is_used_when_provider_metadata_is_missing(self) -> None:
        """A supplied LangChain profile remains a fallback for non-LM Studio models."""
        model = ProfileModel({"max_input_tokens": 32000})

        metadata = await infer_model_metadata({"llm_provider": "openai"}, model=model)

        self.assertEqual(metadata, ModelMetadata(32000, "model_profile.max_input_tokens"))

    def test_apply_metadata_sets_profile_for_deepagents(self) -> None:
        """The model profile should be populated before summarization middleware is built."""
        model = ProfileModel()

        self.assertIs(apply_model_metadata(model, ModelMetadata(10000, "test")), model)

        self.assertEqual(model.profile, {"max_input_tokens": 10000})


if __name__ == "__main__":
    unittest.main()
