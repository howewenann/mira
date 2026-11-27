"""Consumer operations exercised with native graphs and disposable persistence."""

from __future__ import annotations

import asyncio
import ast
import importlib.abc
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage

from mira import MiraApplication
from mira.api import MessageEvent, RuntimeEvent, WorkflowEvent
from mira.settings import set_planning_todos
from tests.test_checkpoint import BindableFakeMessagesListChatModel


ROOT = Path(__file__).resolve().parents[3]


class Consumer:
    """Only emit/request exist; Core cannot inspect renderer capabilities."""

    __slots__ = ("events",)

    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)

    async def request(self, request):
        raise AssertionError(f"Unexpected request: {request}")


class PublicOperationsTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_session_and_configuration_operations(self):
        from config.metadata import ModelMetadata
        from config.settings import normalize_settings

        model = BindableFakeMessagesListChatModel(responses=[AIMessage(content="Complete.")])
        frontend = Consumer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workflows = root / ".mira" / "workflows"
            workflows.mkdir(parents=True)
            (workflows / "upper.py").write_text('''from typing import TypedDict
from langgraph.graph import START, END, StateGraph
class Input(TypedDict):
    value: str
class State(Input):
    pass
def workflow(mira):
    graph = StateGraph(State, input_schema=Input)
    graph.add_node("upper", lambda state: {"value": state["value"].upper()})
    graph.add_edge(START, "upper")
    graph.add_edge("upper", END)
    return graph.compile()
''', encoding="utf-8")
            config = {"session_dir": str(root / ".mira" / "_sessions"),
                      "settings": normalize_settings({}), "settings_valid": True, "llm_direct": False}
            with (
                patch("agent.llm.active_model_issues", return_value=[]),
                patch("agent.llm.get_llm", return_value=model),
                patch("agent.factory.get_llm", return_value=model),
                patch("agent.factory.get_rubric_model_name", return_value="test"),
                patch("agent.llm.get_model_name", return_value="test"),
                patch("config.metadata.infer_model_metadata", AsyncMock(return_value=ModelMetadata(context_tokens=32768))),
            ):
                application = await MiraApplication.start(workspace=root, config=config, frontend=frontend)
                try:
                    session = await application.open_session("public")
                    self.assertTrue(any(item.get("group") == "Skill Validation" for item in session.snapshot().resources["tools"]))
                    await session.prompt("hello")
                    self.assertTrue(any(isinstance(event, MessageEvent) for event in frontend.events))
                    self.assertEqual(session.snapshot(include_transcript=False).transcript, ())
                    self.assertEqual(session.snapshot().pending, "none")
                    application.update_session(session.id, title="Renamed", pinned=True)
                    summary = application.list_sessions()[0]
                    self.assertEqual(summary["custom_title"], "Renamed")
                    summary["custom_title"] = "changed copy"
                    self.assertEqual(application.list_sessions()[0]["custom_title"], "Renamed")
                    application.record_prompt("hello")
                    self.assertEqual(application.prompt_history(), ["hello"])
                    application.clear_prompt_history()
                    self.assertEqual(application.prompt_history(), [])
                    source = root / "sample.txt"
                    source.write_text("sample", encoding="utf-8")
                    references, failures = await session.upload_files([source])
                    self.assertEqual(failures, [])
                    self.assertEqual(len(references), 1)
                    self.assertIn("sample.txt", await application.list_files())
                    await session.clear_uploads()
                    report = await session.context_report()
                    self.assertIsNotNone(report)
                    await session.compact()
                    await session.run_workflow("/workflow-demo", record_user=False)
                    self.assertTrue(any(isinstance(event, WorkflowEvent) and event.phase == "run_finish"
                                        for event in frontend.events))
                    descriptor, inputs = application.workflow_invocation('/workflow__upper value="hello"')
                    self.assertFalse(hasattr(descriptor, "factory"))
                    self.assertEqual(inputs, {"value": "hello"})
                    self.assertEqual(await session.run_workflow('/workflow__upper value="hello"'), {"value": "HELLO"})
                    self.assertEqual(await session.run_workflow('/workflow__upper', inputs={"value": "second"}), {"value": "SECOND"})
                    history = [event for event in session.snapshot().transcript if event["type"] == "workflow"]
                    self.assertEqual(len(history), 2)
                    self.assertNotEqual(history[0]["turn_id"], history[1]["turn_id"])
                    self.assertTrue(all(event["status"] == "DONE" for event in history))
                    self.assertFalse(any(event["type"] == "user" and event.get("text", "").startswith("/workflow__")
                                         for event in session.snapshot().transcript))
                    ok, message = await application.apply_settings(set_planning_todos(application.settings(), True))
                    self.assertTrue(ok, message)
                    self.assertTrue(any(isinstance(event, RuntimeEvent) and event.kind == "configuration"
                                        for event in frontend.events))
                    previous_updates = sum(isinstance(event, RuntimeEvent) and event.kind == "configuration" for event in frontend.events)
                    with patch("config.runtime.load_effective_config", return_value=config):
                        await application.reload(restart_mcp=True)
                    self.assertEqual(sum(isinstance(event, RuntimeEvent) and event.kind == "configuration" for event in frontend.events), previous_updates + 1)
                    self.assertEqual(application.mcp.servers, {})
                    await application.mcp.discover_resources()
                    await application.mcp.discover_prompts()
                    self.assertEqual(application.mcp.resources, ())
                    self.assertTrue(all(not hasattr(command, "factory") and not hasattr(command, "resolver")
                                        for command in application.commands()))
                    await session.close()
                    session = await application.open_session("public")
                    self.assertEqual(session.snapshot().history["custom_title"], "Renamed")
                    await session.clear()
                    self.assertEqual(session.snapshot().transcript, ())
                    self.assertEqual(session.inspections(), [])
                    self.assertTrue(await application.delete_session("public"))
                finally:
                    await application.shutdown()


class IsolationTests(unittest.TestCase):
    def test_textual_uses_only_the_supported_core_boundary(self):
        offenders = []
        for root in (ROOT / "ui" / "textual", ROOT / "ui" / "shared"):
            for path in root.rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    names = [node.module or ""] if isinstance(node, ast.ImportFrom) else (
                        [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
                    if any(name.split(".")[0] in {"agent", "core", "config", "session", "tracing"} for name in names):
                        offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
                    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
                        if node.value.attr in {"application", "core_session"} and node.attr.startswith("_"):
                            offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
        self.assertEqual(offenders, [])

    def test_public_operations_with_textual_and_all_ui_imports_blocked(self):
        script = '''
import importlib.abc, sys, unittest
class NoUI(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'textual', 'ui'}:
            raise ImportError('UI imports blocked: ' + fullname)
sys.meta_path.insert(0, NoUI())
suite = unittest.defaultTestLoader.loadTestsFromName(
    'tests.core.interface.test_operations.PublicOperationsTests')
result = unittest.TextTestRunner().run(suite)
sys.exit(not result.wasSuccessful())
'''
        completed = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                                   text=True, capture_output=True, timeout=90)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
