"""Focused local-file reference parser and middleware tests."""

from __future__ import annotations

import asyncio
import unittest
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.middleware.file_references import FileReferenceMiddleware, local_file_references


class FakeModelRequest:
    def __init__(self, messages: list[Any], system_message: SystemMessage | None = None) -> None:
        self.state = {"messages": messages}
        self.system_message = system_message

    def override(self, **kwargs: Any) -> "FakeModelRequest":
        return FakeModelRequest(
            kwargs.get("state", self.state)["messages"],
            kwargs.get("system_message", self.system_message),
        )


class FileReferenceParserTests(unittest.TestCase):
    def test_quoted_multiple_duplicate_and_normalized_paths(self) -> None:
        self.assertEqual(
            local_file_references(
                'Compare @src\\auth.py with @"docs/design notes.md" and @src/auth.py.'
            ),
            ["/src/auth.py", "/docs/design notes.md"],
        )

    def test_punctuation_is_not_part_of_plain_references(self) -> None:
        self.assertEqual(
            local_file_references("Review (@src/auth.py), @README.md; then @tests/test_auth.py!"),
            ["/src/auth.py", "/README.md", "/tests/test_auth.py"],
        )

    def test_email_like_text_is_ignored(self) -> None:
        self.assertEqual(
            local_file_references("Email name@example.com, then inspect @config/settings.py"),
            ["/config/settings.py"],
        )

    def test_missing_paths_are_returned_without_filesystem_access(self) -> None:
        self.assertEqual(local_file_references("Open @does/not/exist.py"), ["/does/not/exist.py"])


class FileReferenceMiddlewareTests(unittest.TestCase):
    def setUp(self) -> None:
        self.middleware = FileReferenceMiddleware()

    def capture(self, messages: list[Any], system: str | None = None) -> FakeModelRequest:
        request = FakeModelRequest(
            messages,
            SystemMessage(content=system) if system is not None else None,
        )
        captured: dict[str, FakeModelRequest] = {}

        def handler(updated: FakeModelRequest) -> str:
            captured["request"] = updated
            return "ok"

        self.assertEqual(self.middleware.wrap_model_call(request, handler), "ok")
        return captured["request"]

    def test_no_references_passes_request_through(self) -> None:
        request = FakeModelRequest([HumanMessage(content="hello")])
        captured: list[FakeModelRequest] = []
        self.middleware.wrap_model_call(request, lambda updated: captured.append(updated))
        self.assertIs(captured[0], request)

    def test_one_and_multiple_references_use_tool_neutral_guidance(self) -> None:
        updated = self.capture(
            [HumanMessage(content='Compare @src/auth.py and @"docs/design notes.md"')],
            system="base",
        )
        text = str(updated.system_message.text)
        self.assertIn("base", text)
        self.assertIn("- /src/auth.py", text)
        self.assertIn("- /docs/design notes.md", text)
        self.assertIn("Follow any file-reading tool explicitly requested", text)
        self.assertNotIn("read_file", text)

    def test_explicit_specialized_reader_is_not_contradicted(self) -> None:
        updated = self.capture(
            [HumanMessage(content="Use read_file_as_bytes to read @asset.bin")]
        )
        text = str(updated.system_message.text)
        self.assertIn("- /asset.bin", text)
        self.assertNotIn("read_file", text)

    def test_duplicate_references_are_injected_once(self) -> None:
        updated = self.capture([HumanMessage(content="Use @README.md then @README.md")])
        self.assertEqual(str(updated.system_message.text).count("- /README.md"), 1)

    def test_latest_human_turn_prevents_reference_leakage(self) -> None:
        updated = self.capture(
            [
                HumanMessage(content="Read @secret.txt"),
                HumanMessage(content="Now answer without a file reference"),
            ]
        )
        self.assertIsNone(updated.system_message)

    def test_async_hook_uses_same_ephemeral_guidance(self) -> None:
        request = FakeModelRequest([HumanMessage(content="Read @README.md")])
        captured: list[FakeModelRequest] = []

        async def handler(updated: FakeModelRequest) -> str:
            captured.append(updated)
            return "ok"

        self.assertEqual(asyncio.run(self.middleware.awrap_model_call(request, handler)), "ok")
        self.assertIn("- /README.md", str(captured[0].system_message.text))
        self.assertNotIn("read_file", str(captured[0].system_message.text))


if __name__ == "__main__":
    unittest.main()
