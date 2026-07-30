"""Behavioral compatibility checks for the pinned DeepAgents 0.7 stack."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from deepagents.backends import FilesystemBackend
from langchain_quickjs import CodeInterpreterMiddleware

from agent.factory import _write_interrupts
from agent.tools.specs import collect_tool_specs
from runtime.runner import annotate_filesystem_approvals
from ui.interrupts import APPROVAL_CONSEQUENCE, action_preview, action_text


class DeepAgentsFilesystemCompatibilityTests(unittest.TestCase):
    """Keep MIRA's documented file semantics aligned with DeepAgents 0.7."""

    def test_write_replaces_and_edit_preserves_unmatched_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = FilesystemBackend(root_dir=Path(directory), virtual_mode=True)

            self.assertIsNone(backend.write("/notes.txt", "alpha\nbeta\n").error)
            self.assertIsNone(backend.write("/notes.txt", "replacement\n").error)
            self.assertEqual((Path(directory) / "notes.txt").read_text(encoding="utf-8"), "replacement\n")

            self.assertIsNone(backend.edit("/notes.txt", "replacement", "targeted").error)
            self.assertEqual((Path(directory) / "notes.txt").read_text(encoding="utf-8"), "targeted\n")

    def test_delete_is_recursive_and_uses_configurable_hitl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = FilesystemBackend(root_dir=Path(directory), virtual_mode=True)
            backend.write("/tree/child.txt", "content")

            approval_interrupts = _write_interrupts(
                {
                    "settings": {
                        "hitl": {
                            "tools": {
                                "delete": {"enabled": True, "always_allow": False},
                                "write_file": {"enabled": True, "always_allow": False},
                            }
                        }
                    }
                }
            )
            automatic_interrupts = _write_interrupts(
                {
                    "settings": {
                        "hitl": {
                            "tools": {
                                "delete": {"enabled": True, "always_allow": True},
                                "write_file": {"enabled": True, "always_allow": True},
                            }
                        }
                    }
                }
            )
            self.assertIn("delete", approval_interrupts)
            self.assertIn("write_file", approval_interrupts)
            self.assertNotIn("delete", automatic_interrupts)
            self.assertNotIn("write_file", automatic_interrupts)
            rejected_target = Path(directory) / "tree"
            self.assertTrue(rejected_target.exists(), "a rejected request must not execute the backend operation")
            self.assertIsNone(backend.delete("/tree").error)
            self.assertFalse((Path(directory) / "tree").exists())

    def test_backend_without_delete_does_not_advertise_it(self) -> None:
        specs = collect_tool_specs(object(), [], [], [], ())

        self.assertNotIn("delete", [item["name"] for item in specs])

    def test_approval_copy_distinguishes_create_replace_edit_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = FilesystemBackend(root_dir=Path(directory), virtual_mode=True)
            backend.write("/existing.txt", "old")
            actions = [
                {"name": "write_file", "args": {"file_path": "/new.txt", "content": "new"}},
                {"name": "write_file", "args": {"file_path": "/existing.txt", "content": "new"}},
                {
                    "name": "edit_file",
                    "args": {"file_path": "/existing.txt", "old_string": "old", "new_string": "new"},
                },
                {"name": "delete", "args": {"file_path": "/tree"}},
            ]
            annotate_filesystem_approvals([{"action_requests": actions}], backend)

        self.assertEqual(actions[0][APPROVAL_CONSEQUENCE], "Creates a new file.")
        self.assertEqual(actions[1][APPROVAL_CONSEQUENCE], "Replaces the entire existing file.")
        self.assertIn("targeted replacements", actions[2][APPROVAL_CONSEQUENCE])
        self.assertIn("Recursively deletes", actions[3][APPROVAL_CONSEQUENCE])
        self.assertIn("delete", action_text(actions[3]))
        self.assertIn("destructive", action_text(actions[3]))
        self.assertIn("Recursively deletes", action_preview(actions[3]))


class QuickJSCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    """Exercise the exact QuickJS 0.3.5 behavior MIRA relies on."""

    def runtime(self, call_id: str = "eval-call") -> SimpleNamespace:
        return SimpleNamespace(
            tool_call_id=call_id,
            config={"configurable": {"thread_id": "compat-thread"}},
            state={},
            context=None,
            store=None,
            stream_writer=lambda _event: None,
        )

    async def test_basic_async_eval_thread_persistence_and_isolation(self) -> None:
        middleware = CodeInterpreterMiddleware(timeout=1, mode="thread")
        self.addCleanup(middleware._registry.close)
        tool = middleware.tools[0]
        runtime = self.runtime()

        first = await tool.coroutine(runtime=runtime, code="globalThis.saved = 41; saved")
        second = await tool.coroutine(runtime=runtime, code="saved + 1")
        isolated = await tool.coroutine(
            runtime=runtime,
            code='[typeof process, typeof require, typeof fetch].join("/")',
        )

        self.assertIn("41", str(first.content))
        self.assertIn("42", str(second.content))
        self.assertIn("undefined/undefined/undefined", str(isolated.content))

    async def test_errors_timeout_configuration_and_cancellation_propagate(self) -> None:
        middleware = CodeInterpreterMiddleware(timeout=0.05, mode="call")
        self.addCleanup(middleware._registry.close)
        tool = middleware.tools[0]
        runtime = self.runtime()

        syntax = await tool.coroutine(runtime=runtime, code="const =")
        runtime_error = await tool.coroutine(runtime=runtime, code='throw new Error("boom")')

        self.assertIn("SyntaxError", str(syntax.content))
        self.assertIn("boom", str(runtime_error.content))
        self.assertEqual(middleware._timeout, 0.05)

        task = asyncio.create_task(tool.coroutine(runtime=runtime, code="1 + 1"))
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task


if __name__ == "__main__":
    unittest.main()
