"""Tests for dynamic model metadata inference."""

from __future__ import annotations

import unittest
from copy import deepcopy
from typing import Any
from unittest.mock import patch

import httpx

from config.metadata import ModelMetadata, apply_model_metadata, infer_model_metadata
from config.llm import ModelProfile, ModelRegistry
from config.settings import DEFAULT_SETTINGS


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


def profile_config(
    provider: str,
    model: str,
    *,
    context_limit_tokens: int = 32768,
    api_base: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    settings = deepcopy(DEFAULT_SETTINGS)
    settings["models"]["main"] = "main"
    settings["models"]["context_limit_tokens"] = context_limit_tokens
    values: dict[str, Any] = {"provider": provider, "model": model}
    if api_base is not None:
        values["api_base"] = api_base
    if api_key is not None:
        values["api_key"] = api_key
    return {
        "settings": settings,
        "model_registry": ModelRegistry({"main": ModelProfile("main", values)}),
    }


def lmstudio_config(model: str = "gemma-4-e4b", *, context_limit_tokens: int = 32768) -> dict[str, Any]:
    """Return the minimal LM Studio config used by metadata tests."""
    return profile_config(
        "lmstudio",
        model,
        context_limit_tokens=context_limit_tokens,
        api_base="http://localhost:1234/v1",
        api_key="lm-studio",
    )


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
        self.assertEqual(FakeAsyncClient.calls[0]["timeout"], 2.0)
        self.assertEqual(FakeAsyncClient.calls[1]["url"], "http://localhost:1234/api/v1/models")

    async def test_lmstudio_metadata_timeout_can_be_configured(self) -> None:
        """LM Studio metadata probing should use a finite configurable timeout."""
        with patch("config.metadata.httpx.AsyncClient", FakeAsyncClient):
            await infer_model_metadata({**lmstudio_config(), "lmstudio_metadata_timeout": 0.75})

        self.assertEqual(FakeAsyncClient.calls[0]["timeout"], 0.75)

    async def test_lmstudio_metadata_selects_configured_model(self) -> None:
        """The configured model name should choose the matching LM Studio entry."""
        with patch("config.metadata.httpx.AsyncClient", FakeAsyncClient):
            metadata = await infer_model_metadata(lmstudio_config("other-model"))

        self.assertEqual(metadata.context_tokens, 4096)

    async def test_configured_context_caps_provider_metadata(self) -> None:
        """Manual context limits should cap provider metadata without raising it."""
        with patch("config.metadata.httpx.AsyncClient", FakeAsyncClient):
            metadata = await infer_model_metadata(lmstudio_config(context_limit_tokens=8192))

        self.assertEqual(metadata, ModelMetadata(8192, "settings.models.context_limit_tokens"))

    async def test_provider_metadata_wins_when_below_configured_cap(self) -> None:
        """Provider metadata should remain the effective limit when below the env cap."""
        with patch("config.metadata.httpx.AsyncClient", FakeAsyncClient):
            metadata = await infer_model_metadata(
                lmstudio_config("other-model", context_limit_tokens=8192)
            )

        self.assertEqual(metadata, ModelMetadata(4096, "lmstudio.api.v1.loaded_instance"))

    async def test_unavailable_metadata_returns_unknown(self) -> None:
        """Provider metadata failures should not crash startup or turns."""
        FakeAsyncClient.error = httpx.ConnectError("offline")

        with patch("config.metadata.httpx.AsyncClient", FakeAsyncClient):
            metadata = await infer_model_metadata(lmstudio_config())

        self.assertEqual(metadata, ModelMetadata(32768, "settings.models.context_limit_tokens"))

    async def test_profile_is_used_when_provider_metadata_is_missing(self) -> None:
        """A supplied LangChain profile remains a fallback for non-LM Studio models."""
        model = ProfileModel({"max_input_tokens": 32000})

        metadata = await infer_model_metadata(profile_config("openai", "gpt-test"), model=model)

        self.assertEqual(metadata, ModelMetadata(32000, "model_profile.max_input_tokens"))

    async def test_configured_context_caps_profile_metadata(self) -> None:
        """Manual context limits should cap LangChain profile context."""
        model = ProfileModel({"max_input_tokens": 64000})

        metadata = await infer_model_metadata(profile_config("openai", "gpt-test"), model=model)

        self.assertEqual(metadata, ModelMetadata(32768, "settings.models.context_limit_tokens"))

    async def test_lmstudio_failure_uses_env_fallback_not_model_profile(self) -> None:
        """LM Studio should use its API or env/default cap, not an incidental profile."""
        FakeAsyncClient.error = httpx.ConnectError("offline")
        model = ProfileModel({"max_input_tokens": 16000})

        with patch("config.metadata.httpx.AsyncClient", FakeAsyncClient):
            metadata = await infer_model_metadata(lmstudio_config(), model=model)

        self.assertEqual(metadata, ModelMetadata(32768, "settings.models.context_limit_tokens"))

    async def test_default_context_is_used_when_env_key_is_missing(self) -> None:
        """The backend should provide a default profile limit when no source reports one."""
        metadata = await infer_model_metadata(profile_config("custom", "model"))

        self.assertEqual(metadata, ModelMetadata(32768, "settings.models.context_limit_tokens"))

    def test_apply_metadata_sets_profile_for_deepagents(self) -> None:
        """The model profile should be populated before summarization middleware is built."""
        model = ProfileModel()

        self.assertIs(apply_model_metadata(model, ModelMetadata(10000, "test")), model)

        self.assertEqual(model.profile, {"max_input_tokens": 10000})


if __name__ == "__main__":
    unittest.main()
