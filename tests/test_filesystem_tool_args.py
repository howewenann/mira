"""Regression tests for MIRA filesystem tool-call argument normalization."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from deepagents import FilesystemPermission, create_deep_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from agent.factory import _write_interrupts
from agent.middleware import FilesystemToolCallArgsMiddleware
from agent.resources import build_resources
from runtime.runner import run_turn
from session.checkpoint import make_checkpointer


class BindableFakeMessagesListChatModel(FakeMessagesListChatModel):
    """Fake model compatible with LangChain tool binding."""

    def bind_tools(self, tools: list[Any], *, tool_choice: Any = None, **kwargs: Any) -> Any:
        return self


class ApprovalRenderer:
    """Minimal renderer that auto-approves write interrupts."""

    def __init__(self) -> None:
        self.tool_calls: list[tuple[str, Any, str]] = []

    def __getattr__(self, name: str) -> Any:
        def ignore(*args: Any, **kwargs: Any) -> None:
            return None

        return ignore

    async def ask_approvals(self, interrupts: list[Any]) -> list[dict[str, str]]:
        actions = interrupts[0].value["action_requests"]
        return [{"type": "approve"} for _ in actions]

    async def ask_user(self, interrupt: Any) -> str:
        return ""

    def tool_call(self, name: str, args: Any, result: str = "", call_id: str = "") -> None:
        self.tool_calls.append((name, args, call_id))


class FilesystemToolArgTests(unittest.IsolatedAsyncioTestCase):
    """Tests for file tool arg repair before execution."""

    def test_normalizer_renames_path_arg_and_virtualizes_workspace_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            target = workspace / "nested" / "note.txt"
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"path": str(target), "content": "hello"},
                        "id": "call-write",
                    }
                ],
            )

            update = FilesystemToolCallArgsMiddleware(workspace).after_model({"messages": [message]}, None)

            self.assertIsNotNone(update)
            self.assertEqual(message.tool_calls[0]["args"], {"content": "hello", "file_path": "/nested/note.txt"})

    async def test_write_and_read_succeed_when_model_uses_path_arg(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            resources = build_resources(workspace, create_examples=False)
            model = BindableFakeMessagesListChatModel(
                responses=[
                    AIMessage(
                        content=[
                            {"type": "reasoning", "reasoning": "Need to create a file."},
                            {"type": "text", "text": "\n\n"},
                            {
                                "type": "tool_call",
                                "id": "call-write",
                                "name": "write_file",
                                "args": {"path": "/smoke.txt", "content": "hello smoke"},
                            },
                        ],
                        tool_calls=[
                            {
                                "name": "write_file",
                                "args": {"path": "/smoke.txt", "content": "hello smoke"},
                                "id": "call-write",
                            }
                        ],
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "read_file",
                                "args": {"path": "/smoke.txt"},
                                "id": "call-read",
                            }
                        ],
                    ),
                    AIMessage(content="done"),
                ]
            )
            agent = create_deep_agent(
                model=model,
                backend=resources.backend,
                middleware=[FilesystemToolCallArgsMiddleware(workspace)],
                tools=resources.tools,
                skills=resources.skills,
                memory=resources.memory,
                subagents=resources.subagents,
                permissions=[FilesystemPermission(operations=["read", "write"], paths=["/**"], mode="allow")],
                interrupt_on=_write_interrupts(),
                checkpointer=make_checkpointer(),
            )
            renderer = ApprovalRenderer()

            await run_turn(agent, "write then read", renderer, "filesystem-path-arg")

            self.assertEqual((workspace / "smoke.txt").read_text(), "hello smoke")
            self.assertIn(("write_file", {"content": "hello smoke", "file_path": "/smoke.txt"}, "call-write"), renderer.tool_calls)
            self.assertIn(("read_file", {"file_path": "/smoke.txt"}, "call-read"), renderer.tool_calls)


if __name__ == "__main__":
    unittest.main()
