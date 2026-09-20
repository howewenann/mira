"""Focused model-boundary compatibility tests."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from deepagents.backends import StateBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, ToolMessage
from langchain_anyllm import ChatAnyLLM

from agent.middleware import ModelCompatibilityMiddleware

IMAGE_BLOCK = {
    "type": "image",
    "base64": "cG5nLWRhdGE=",
    "mime_type": "image/png",
}
IMAGE_URL_BLOCK = {
    "type": "image_url",
    "image_url": {"url": "data:image/png;base64,cG5nLWRhdGE="},
}


def model_with_capability(image_inputs: bool) -> ChatAnyLLM:
    model = ChatAnyLLM(provider="openai", model="probe-model", api_key="probe-key")
    model.profile = {"image_inputs": image_inputs}
    return model


def response() -> ModelResponse[Any]:
    return ModelResponse(result=[AIMessage(content="done")])


class ModelCompatibilityRequestTests(unittest.TestCase):
    def test_canonical_images_are_copied_and_converted_in_order(self) -> None:
        text = {"type": "text", "text": "before"}
        audio = {"type": "audio", "base64": "audio-data", "mime_type": "audio/wav"}
        file_block = {"type": "file", "file_id": "file-1"}
        unknown = {"type": "custom", "value": 1}
        content = [text, IMAGE_BLOCK, audio, IMAGE_URL_BLOCK, file_block, unknown]
        message = ToolMessage(name="read_file", tool_call_id="read-1", content=content)
        original_text = message.content[0]
        original_audio = message.content[2]
        original_image_url = message.content[3]
        state = {"messages": [message]}
        request = ModelRequest(
            model=model_with_capability(True),
            messages=[message],
            state=state,
        )
        original_content = copy.deepcopy(message.content)
        captured: dict[str, ModelRequest[Any]] = {}

        def handler(bound: ModelRequest[Any]) -> ModelResponse[Any]:
            captured["request"] = bound
            return response()

        ModelCompatibilityMiddleware(Path(".")).wrap_model_call(request, handler)

        bound = captured["request"]
        bound_message = bound.messages[0]
        self.assertIsNot(bound, request)
        self.assertIsNot(bound_message, message)
        self.assertEqual(
            bound_message.content,
            [text, IMAGE_URL_BLOCK, audio, IMAGE_URL_BLOCK, file_block, unknown],
        )
        self.assertIs(bound_message.content[0], original_text)
        self.assertIs(bound_message.content[2], original_audio)
        self.assertIs(bound_message.content[3], original_image_url)
        self.assertEqual(message.content, original_content)
        self.assertIs(request.messages[0], message)
        self.assertIs(request.state, state)
        self.assertIs(bound.state, state)
        self.assertIs(state["messages"][0], message)
        self.assertEqual(state["messages"][0].content[1]["type"], "image")

    def test_noop_content_reuses_request_and_message_objects(self) -> None:
        blocks = [
            {"type": "text", "text": "hello"},
            IMAGE_URL_BLOCK,
            {"type": "audio", "base64": "audio-data", "mime_type": "audio/wav"},
            {"type": "video", "url": "https://example.test/video"},
            {"type": "file", "file_id": "file-1"},
            {"type": "custom", "value": 1},
        ]
        message = ToolMessage(name="read_file", tool_call_id="read-2", content=blocks)
        original_content = message.content
        request = ModelRequest(model=model_with_capability(True), messages=[message])
        captured: dict[str, ModelRequest[Any]] = {}

        def handler(bound: ModelRequest[Any]) -> ModelResponse[Any]:
            captured["request"] = bound
            return response()

        ModelCompatibilityMiddleware(Path(".")).wrap_model_call(request, handler)

        self.assertIs(captured["request"], request)
        self.assertIs(captured["request"].messages[0], message)
        self.assertIs(message.content, original_content)

    def test_plain_text_reuses_original_request(self) -> None:
        message = ToolMessage(name="read_file", tool_call_id="read-3", content="plain text")
        request = ModelRequest(model=model_with_capability(True), messages=[message])
        captured: list[ModelRequest[Any]] = []

        ModelCompatibilityMiddleware(Path(".")).wrap_model_call(
            request,
            lambda bound: captured.append(bound) or response(),
        )

        self.assertEqual(captured, [request])
        self.assertIs(captured[0].messages[0], message)


class AsyncModelCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_request_normalization_matches_sync_behavior(self) -> None:
        message = ToolMessage(
            name="read_file",
            tool_call_id="read-async",
            content=[IMAGE_BLOCK],
        )
        request = ModelRequest(model=model_with_capability(True), messages=[message])
        captured: list[ModelRequest[Any]] = []

        async def handler(bound: ModelRequest[Any]) -> ModelResponse[Any]:
            captured.append(bound)
            return response()

        await ModelCompatibilityMiddleware(Path(".")).awrap_model_call(request, handler)

        self.assertEqual(captured[0].messages[0].content, [IMAGE_URL_BLOCK])
        self.assertEqual(message.content, [IMAGE_BLOCK])

    async def test_deepagents_filters_before_compatibility_conversion(self) -> None:
        filesystem = FilesystemMiddleware(backend=StateBackend())
        compatibility = ModelCompatibilityMiddleware(Path("."))

        for supported in (False, True):
            with self.subTest(image_inputs=supported):
                message = ToolMessage(
                    name="read_file",
                    tool_call_id=f"read-{supported}",
                    content=[IMAGE_BLOCK],
                )
                state = {"messages": [message]}
                request = ModelRequest(
                    model=model_with_capability(supported),
                    messages=[message],
                    state=state,
                )
                captured: list[ModelRequest[Any]] = []

                async def capture(bound: ModelRequest[Any]) -> ModelResponse[Any]:
                    captured.append(bound)
                    return response()

                async def apply_compatibility(bound: ModelRequest[Any]) -> ModelResponse[Any]:
                    return await compatibility.awrap_model_call(bound, capture)

                await filesystem.awrap_model_call(request, apply_compatibility)

                bound_message = captured[0].messages[0]
                if supported:
                    self.assertEqual(bound_message.content, [IMAGE_URL_BLOCK])
                else:
                    self.assertIn("does not support image content", bound_message.text)
                    self.assertNotEqual(bound_message.content, [IMAGE_URL_BLOCK])
                self.assertEqual(message.content, [IMAGE_BLOCK])
                self.assertIs(state["messages"][0], message)

    async def test_normalized_image_url_survives_chat_anyllm_outbound_conversion(self) -> None:
        message = ToolMessage(
            name="read_file",
            tool_call_id="read-outbound",
            content=[IMAGE_BLOCK],
        )
        request = ModelRequest(model=model_with_capability(True), messages=[message])
        captured_request: list[ModelRequest[Any]] = []

        async def handler(bound: ModelRequest[Any]) -> ModelResponse[Any]:
            captured_request.append(bound)
            return response()

        await ModelCompatibilityMiddleware(Path(".")).awrap_model_call(request, handler)
        outbound: dict[str, Any] = {}

        class OutboundCaptured(Exception):
            pass

        async def capture_anyllm(*_args: Any, **kwargs: Any) -> Any:
            outbound["messages"] = copy.deepcopy(kwargs["messages"])
            raise OutboundCaptured

        with patch("langchain_anyllm.chat_models.acompletion", new=capture_anyllm):
            with self.assertRaises(OutboundCaptured):
                await request.model.ainvoke([captured_request[0].messages[0]])

        self.assertEqual(outbound["messages"][0]["content"], [IMAGE_URL_BLOCK])


if __name__ == "__main__":
    unittest.main()
