"""Behavioral compatibility checks for the pinned DeepAgents 0.7 stack."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend
from langchain_quickjs import CodeInterpreterMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from agent.factory import _action_permissions, _write_interrupts
from agent.resources import build_resources
from agent.tools.specs import collect_tool_specs
from core.execution.runner import annotate_filesystem_approvals, run_turn
from session.checkpoint import make_checkpointer
from ui.shared.interrupts import APPROVAL_CONSEQUENCE, action_preview, action_text


class BindableFakeMessagesListChatModel(FakeMessagesListChatModel):
    """Fake model compatible with DeepAgents tool binding."""

    def bind_tools(self, tools: list[object], *, tool_choice: object = None, **kwargs: object) -> object:
        return self


class ApprovalRenderer:
    """Minimal renderer that returns one configured HITL decision."""

    def __init__(self, decision: dict[str, object]) -> None:
        self.decision = decision
        self.approvals: list[list[object]] = []

    def __getattr__(self, name: str) -> object:
        def ignore(*args: object, **kwargs: object) -> None:
            return None

        return ignore

    async def ask_approvals(self, interrupts: list[object]) -> list[dict[str, object]]:
        self.approvals.append(interrupts)
        return [self.decision]


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

    def test_degenerate_read_windows_are_safe_through_mira_backend_routing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            resources = build_resources(Path(directory), create_examples=False)
            self.assertIsNone(resources.backend.write("/notes.txt", "alpha\nbeta\n").error)

            negative_offset = resources.backend.read("/notes.txt", offset=-5, limit=1)
            empty_window = resources.backend.read("/notes.txt", offset=0, limit=0)

            self.assertIsNone(negative_offset.error)
            self.assertEqual(negative_offset.file_data["content"], "alpha\n")
            self.assertEqual(negative_offset.start_line, 1)
            self.assertIsNone(empty_window.error)
            self.assertEqual(empty_window.file_data["content"], "")
            self.assertTrue(empty_window.no_lines_requested)

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


class DeepAgentsPermissionCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    """Exercise MIRA's permission ordering through DeepAgents tools and HITL."""

    @staticmethod
    def backends(directory: str) -> tuple[FilesystemBackend, FilesystemBackend, CompositeBackend]:
        root = Path(directory)
        project = FilesystemBackend(root_dir=root / "project", virtual_mode=True)
        defaults = FilesystemBackend(root_dir=root / "defaults", virtual_mode=True)
        combined = CompositeBackend(
            default=project,
            routes={"/mira-defaults/": defaults},
            artifacts_root="/.mira",
        )
        return project, defaults, combined

    @staticmethod
    def delete_agent(backend: CompositeBackend, file_path: str) -> object:
        model = BindableFakeMessagesListChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "delete",
                            "args": {"file_path": file_path},
                            "id": "call-delete",
                        }
                    ],
                ),
                AIMessage(content="done"),
            ]
        )
        return create_deep_agent(
            model=model,
            backend=backend,
            permissions=_action_permissions(),
            interrupt_on=_write_interrupts(),
            checkpointer=make_checkpointer(),
        )

    async def test_exact_project_delete_is_approved_and_defaults_delete_is_denied(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project, defaults, combined = self.backends(directory)
            self.assertIsNone(project.write("/project.txt", "project").error)
            self.assertIsNone(defaults.write("/protected.txt", "protected").error)

            project_renderer = ApprovalRenderer({"type": "approve"})
            await run_turn(
                self.delete_agent(combined, "/project.txt"),
                "delete project file",
                project_renderer,
                "delete-project-file",
            )

            defaults_renderer = ApprovalRenderer({"type": "approve"})
            await run_turn(
                self.delete_agent(combined, "/mira-defaults/protected.txt"),
                "delete protected file",
                defaults_renderer,
                "delete-default-file",
            )

            self.assertEqual(len(project_renderer.approvals), 1)
            self.assertEqual(len(defaults_renderer.approvals), 1)
            self.assertIsNotNone(project.read("/project.txt").error)
            self.assertEqual(defaults.read("/protected.txt").file_data["content"], "protected")

    async def test_recursive_delete_remains_approval_gated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project, _defaults, combined = self.backends(directory)
            self.assertIsNone(project.write("/tree/child.txt", "content").error)

            reject_renderer = ApprovalRenderer({"type": "reject"})
            await run_turn(
                self.delete_agent(combined, "/tree"),
                "delete tree",
                reject_renderer,
                "reject-recursive-delete",
            )
            self.assertEqual(project.read("/tree/child.txt").file_data["content"], "content")

            approve_renderer = ApprovalRenderer({"type": "approve"})
            await run_turn(
                self.delete_agent(combined, "/tree"),
                "delete tree",
                approve_renderer,
                "approve-recursive-delete",
            )

            self.assertEqual(len(reject_renderer.approvals), 1)
            self.assertEqual(len(approve_renderer.approvals), 1)
            self.assertIsNotNone(project.read("/tree/child.txt").error)

    async def test_edited_delete_path_is_rechecked_against_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project, defaults, combined = self.backends(directory)
            self.assertIsNone(project.write("/safe.txt", "safe").error)
            self.assertIsNone(defaults.write("/protected.txt", "protected").error)
            renderer = ApprovalRenderer(
                {
                    "type": "edit",
                    "edited_action": {
                        "name": "delete",
                        "args": {"file_path": "/mira-defaults/protected.txt"},
                    },
                }
            )

            await run_turn(
                self.delete_agent(combined, "/safe.txt"),
                "delete safe file",
                renderer,
                "edit-delete-target",
            )

            self.assertEqual(len(renderer.approvals), 1)
            self.assertEqual(project.read("/safe.txt").file_data["content"], "safe")
            self.assertEqual(defaults.read("/protected.txt").file_data["content"], "protected")


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
