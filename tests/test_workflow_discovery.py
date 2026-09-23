"""Focused discovery and invocation tests for launchable local Workflows."""

from __future__ import annotations

import tempfile
import unittest
from io import StringIO
from pathlib import Path

from rich.console import Console

from agent.workflows.discovery import (
    discover_workflows,
    validate_runtime_graph,
    workflow_arguments,
)
from core.application import MiraApplication
from core.interface import NullFrontend
from ui.textual.runtime_report import workflows_table


VALID_WORKFLOW = '''
from typing import NotRequired, TypedDict
from langgraph.graph import END, START, StateGraph

class InputState(TypedDict):
    topic: str
    depth: NotRequired[int]
    enabled: NotRequired[bool]
    threshold: NotRequired[float]
    tags: NotRequired[list[str]]
    options: NotRequired[dict[str, int]]

class State(InputState):
    report: str

def workflow(mira):
    graph = StateGraph(State, input_schema=InputState)
    graph.add_node("done", lambda state: {"report": state["topic"]})
    graph.add_edge(START, "done")
    graph.add_edge("done", END)
    return graph.compile()
'''


class WorkflowDiscoveryTests(unittest.TestCase):
    def _write(self, root: Path, relative: str, text: str) -> Path:
        path = root / ".mira" / "workflows" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_direct_file_registers_deterministic_command_and_usage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            path = self._write(workspace, "research.py", VALID_WORKFLOW)
            self._write(workspace, "Alpha.py", VALID_WORKFLOW)
            self._write(workspace, "nested/ignored.py", VALID_WORKFLOW)

            registry = discover_workflows(workspace, object())

            self.assertEqual(list(registry.specs), ["Alpha", "research"])
            spec = registry.specs["research"]
            self.assertEqual(spec.path, path)
            self.assertEqual(spec.command, "/workflow__research")
            self.assertEqual(
                spec.usage,
                "/workflow__research topic=<str> [depth=<int>] [enabled=<bool>] "
                "[threshold=<float/number>] [tags=<list>] [options=<dict>]",
            )
            self.assertEqual(registry.issues, ())

    def test_not_required_and_messages_state_inputs_are_valid(self) -> None:
        source = '''
from typing import NotRequired
from langgraph.graph import END, START, MessagesState, StateGraph
class InputState(MessagesState):
    depth: NotRequired[int]
class State(InputState):
    report: str
def workflow(mira):
    graph = StateGraph(State, input_schema=InputState)
    graph.add_node("done", lambda state: {"report": "done"})
    graph.add_edge(START, "done")
    graph.add_edge("done", END)
    return graph.compile()
'''
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            self._write(workspace, "chat.py", source)

            registry = discover_workflows(workspace, object())

            spec = registry.specs["chat"]
            self.assertEqual(spec.usage, "/workflow__chat messages=<list> [depth=<int>]")
            self.assertNotIn("$defs", spec.usage)

    def test_invalid_files_become_issues_without_aborting_discovery(self) -> None:
        cases = {
            "import.py": "raise ImportError('missing workflow dependency')\n",
            "missing.py": "value = 1\n",
            "signature.py": "def workflow(mira, topic):\n    return None\n",
            "wrong_name.py": "def workflow(api):\n    return None\n",
            "default.py": "def workflow(mira=None):\n    return None\n",
            "variadic.py": "def workflow(*mira):\n    return None\n",
            "async.py": "async def workflow(mira):\n    return None\n",
            "construction.py": "def workflow(mira):\n    raise RuntimeError('cannot build')\n",
            "return.py": "def workflow(mira):\n    return object()\n",
            "implicit.py": '''
from typing import TypedDict
from langgraph.graph import END, START, StateGraph
class State(TypedDict):
    topic: str
def workflow(mira):
    graph = StateGraph(State)
    graph.add_node("done", lambda state: {})
    graph.add_edge(START, "done")
    graph.add_edge("done", END)
    return graph.compile()
''',
            "empty.py": '''
from typing import TypedDict
from langgraph.graph import END, START, StateGraph
class InputState(TypedDict):
    pass
class State(InputState):
    result: str
def workflow(mira):
    graph = StateGraph(State, input_schema=InputState)
    graph.add_node("done", lambda state: {"result": "done"})
    graph.add_edge(START, "done")
    graph.add_edge("done", END)
    return graph.compile()
''',
            "non_object.py": '''
from typing import TypedDict
from langgraph.graph import END, START, StateGraph
class InputState(TypedDict):
    topic: str
class State(InputState):
    result: str
def workflow(mira):
    graph = StateGraph(State, input_schema=InputState)
    graph.add_node("done", lambda state: {"result": "done"})
    graph.add_edge(START, "done")
    graph.add_edge("done", END)
    compiled = graph.compile()
    compiled.get_input_jsonschema = lambda: {"type": "array", "items": {"type": "string"}}
    return compiled
''',
        }
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            for name, source in cases.items():
                self._write(workspace, name, source)
            self._write(workspace, "valid.py", VALID_WORKFLOW)

            registry = discover_workflows(workspace, object())

            self.assertEqual(list(registry.specs), ["valid"])
            self.assertEqual(len(registry.issues), len(cases))
            self.assertEqual(
                {issue.summary for issue in registry.issues},
                {f"Workflow: {Path(name).stem}" for name in cases},
            )
            details = "\n".join(issue.details for issue in registry.issues)
            self.assertIn("workflow(mira) was not found", details)
            self.assertIn("signature must be exactly", details)
            self.assertIn("cannot build", details)
            self.assertIn("must return a compiled LangGraph", details)
            self.assertIn("dedicated LangGraph input_schema", details)
            self.assertIn("top-level object with named properties", details)

    def test_runtime_graph_must_keep_registered_shallow_contract(self) -> None:
        changed = VALID_WORKFLOW.replace("topic: str", "subject: str").replace(
            'state["topic"]', 'state["subject"]'
        )
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            original = self._write(workspace, "research.py", VALID_WORKFLOW)
            spec = discover_workflows(workspace, object()).specs["research"]
            changed_path = self._write(workspace, "changed.py", changed)
            changed_spec = discover_workflows(workspace, object()).specs["changed"]

            with self.assertRaisesRegex(ValueError, "schema changed since discovery"):
                validate_runtime_graph(spec, changed_spec.factory(object()))
            self.assertEqual(spec.path, original)
            self.assertEqual(changed_spec.path, changed_path)

    def test_application_reload_replaces_registry_and_workflow_issues(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            path = self._write(workspace, "broken.py", "value = 1\n")
            application = MiraApplication(
                frontend=NullFrontend(),
                workspace=workspace,
                agent=object(),
                issues=[],
            )
            first_registry = application.workflow_registry
            self.assertNotIn("broken", first_registry.specs)
            self.assertEqual(
                [issue.summary for issue in application.issues],
                ["Workflow: broken"],
            )

            path.write_text(VALID_WORKFLOW, encoding="utf-8")
            second_registry = application.reload_workflows()

            self.assertIsNot(second_registry, first_registry)
            self.assertIn("broken", second_registry.specs)
            self.assertEqual(application.issues, [])

    def test_argument_parser_preserves_json_types_and_shell_strings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            self._write(workspace, "research.py", VALID_WORKFLOW)
            spec = discover_workflows(workspace, object()).specs["research"]

            values = workflow_arguments(
                spec,
                '/workflow__research topic="Cubaris isopods" depth=3 enabled=true '
                'threshold=0.75 tags=["a", "b"] options={"limit": 10}',
            )

            self.assertEqual(
                values,
                {
                    "topic": "Cubaris isopods",
                    "depth": 3,
                    "enabled": True,
                    "threshold": 0.75,
                    "tags": ["a", "b"],
                    "options": {"limit": 10},
                },
            )
            self.assertEqual(
                workflow_arguments(spec, r"/workflow__research topic=Cubaris\ isopods"),
                {"topic": "Cubaris isopods"},
            )

    def test_argument_errors_are_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            self._write(workspace, "research.py", VALID_WORKFLOW)
            spec = discover_workflows(workspace, object()).specs["research"]
            cases = {
                "/workflow__research": "missing required workflow arguments",
                "/workflow__research topic=x nope=1": "unknown workflow argument",
                "/workflow__research topic=x topic=y": "duplicate workflow argument",
                "/workflow__research topic": "malformed workflow argument",
                "/workflow__research topic=x depth=bad": "invalid workflow input",
                '/workflow__research topic="unterminated': "invalid workflow arguments",
                "/workflow__research topic=x tags=[1, 2": "invalid workflow arguments",
            }
            for invocation, expected in cases.items():
                with self.subTest(invocation=invocation):
                    with self.assertRaisesRegex(ValueError, expected) as raised:
                        workflow_arguments(spec, invocation)
                    self.assertIn(f"usage: {spec.usage}", str(raised.exception))

    def test_workflows_table_renders_complete_generated_usage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            self._write(workspace, "research.py", VALID_WORKFLOW)
            spec = discover_workflows(workspace, object()).specs["research"]
            output = StringIO()
            Console(file=output, width=160, force_terminal=False).print(
                workflows_table([spec])
            )

            rendered = " ".join(output.getvalue().split())
            self.assertIn(spec.command, rendered)
            self.assertIn("topic=<str>", rendered)
            self.assertIn("[options=<dict>]", rendered)


if __name__ == "__main__":
    unittest.main()
