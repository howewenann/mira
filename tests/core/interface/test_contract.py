"""Headless and dependency-direction proofs for MIRA's consumer boundary."""

from __future__ import annotations

import ast
import asyncio
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage

from core.application import MiraApplication, MiraSession
from core.interface import (
    ApprovalRequest,
    ArtifactEvent,
    AskUserRequest,
    FrontendEmitter,
    FrontendEvent,
    FrontendRequest,
    MCPEvent,
    MessageEvent,
    RubricEvent,
    RuntimeEvent,
    ToolEvent,
)
from session.dashboard import normalize_dashboard


ROOT = Path(__file__).resolve().parents[3]


class RecordingFrontend:
    """Tiny contract test double; it has no Textual dependency."""

    def __init__(self, responses: list[Any] | None = None) -> None:
        self.events: list[FrontendEvent] = []
        self.requests: list[FrontendRequest] = []
        self.responses = list(responses or [])

    def emit(self, event: FrontendEvent) -> None:
        self.events.append(event)

    async def request(self, request: FrontendRequest) -> Any:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError(f"no response supplied for {type(request).__name__}")
        return self.responses.pop(0)


class AsyncItems:
    def __init__(self, items: list[Any]) -> None:
        self.items = items

    async def __aiter__(self) -> Any:
        for item in self.items:
            yield item


class StreamedMessage:
    """Native-shaped LangChain message stream with identity and provenance."""

    def __init__(self) -> None:
        self.message_id = "message-7"
        self.namespace = ("coordinator",)
        self.metadata = {"provider": "test"}
        self.reasoning = AsyncItems([" ", "visible thought"])
        self.text = AsyncItems(["Hello", " world"])
        self.tool_calls: list[Any] = []


class ToolCall:
    completed = True

    def __init__(self) -> None:
        self.name = "read_file"
        self.args = {"file_path": "README.md", "line_start": 1}
        self.id = "tool-9"
        self.output = "contents"
        self.namespace = ("tools:read_file", "worker")
        self.metadata = {"graph": "subgraph"}


class FakeStream:
    def __init__(
        self,
        *,
        messages: list[Any] | None = None,
        tools: list[Any] | None = None,
        output: dict[str, Any] | None = None,
        interrupts: list[Any] | None = None,
    ) -> None:
        self.messages = AsyncItems(messages or [])
        self.tool_calls = AsyncItems(tools or [])
        self.subagents = AsyncItems([])
        self.custom = AsyncItems([])
        self._output = output or {"messages": []}
        self._interrupts = interrupts or []

    async def __aiter__(self) -> Any:
        if False:
            yield None

    async def output(self) -> dict[str, Any]:
        return self._output

    def interrupts(self) -> list[Any]:
        return self._interrupts


class FakeAgent:
    def __init__(self, streams: list[FakeStream]) -> None:
        self.streams = list(streams)
        self.payloads: list[Any] = []

    async def astream_events(
        self,
        payload: Any,
        config: dict[str, Any],
        version: str,
        **kwargs: Any,
    ) -> FakeStream:
        del config, version, kwargs
        self.payloads.append(payload)
        return self.streams.pop(0)


class MemoryStore:
    def __init__(self) -> None:
        self.saves: list[dict[str, Any]] = []

    def save(self, record: dict[str, Any]) -> None:
        self.saves.append(dict(record))


def session_record() -> dict[str, Any]:
    return {
        "id": "mira-session-1",
        "title": "Untitled session",
        "workspace": str(ROOT),
        "turns": 0,
        "dashboard": normalize_dashboard(None),
        "current_plan": None,
        "current_goal": None,
        "events": [],
    }


def application_for(
    frontend: RecordingFrontend,
    agent: FakeAgent,
) -> tuple[MiraApplication, MiraSession]:
    store = MemoryStore()
    application = MiraApplication(
        frontend=frontend,
        workspace=ROOT,
        agent=agent,
        plan_agent=agent,
        config={"settings": {}},
        model_name="test-model",
        context_limit_tokens=4096,
        context_limit_source="test",
        store=store,
        checkpointer=None,
        mcp_manager=None,
        tool_failures=[],
        issues=[],
        resource_metadata={},
        project_backend=None,
        agent_unavailable_message="",
    )
    session = MiraSession(application, session_record())
    return application, session


class FrontendContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_headless_session_preserves_native_stream_structure(self) -> None:
        frontend = RecordingFrontend()
        tool = ToolCall()
        stream = FakeStream(
            messages=[StreamedMessage()],
            tools=[tool],
            output={"messages": [AIMessage(content="Hello world")]},
        )
        application, session = application_for(frontend, FakeAgent([stream]))

        result = await session.prompt("Inspect the README")

        self.assertEqual(result.final_text, "Hello world")
        messages = [event for event in frontend.events if isinstance(event, MessageEvent)]
        reasoning = [event for event in messages if event.phase == "reasoning"]
        content = [event for event in messages if event.phase == "content"]
        self.assertEqual([event.text for event in reasoning], [" ", "visible thought"])
        self.assertEqual([event.text for event in content], ["Hello", " world"])
        self.assertTrue(all(event.message_id == "message-7" for event in reasoning + content))
        self.assertTrue(all(event.namespace == ("coordinator",) for event in reasoning + content))
        self.assertEqual(content[0].content_blocks, ({"type": "text", "text": "Hello"},))

        tools = [event for event in frontend.events if isinstance(event, ToolEvent)]
        started = next(event for event in tools if event.phase == "start")
        finished = next(event for event in tools if event.phase == "completed_result")
        self.assertEqual(started.tool_call_id, "tool-9")
        self.assertEqual(started.arguments, {"file_path": "README.md", "line_start": 1})
        self.assertEqual(started.namespace, ("tools:read_file", "worker"))
        self.assertEqual(started.metadata["graph"], "subgraph")
        self.assertEqual(finished.result, "contents")

        snapshot = session.snapshot()
        self.assertEqual(snapshot.session_id, "mira-session-1")
        self.assertEqual(snapshot.mode, "action")
        self.assertEqual(snapshot.turns, 1)
        self.assertEqual(snapshot.transcript[0]["type"], "user")
        self.assertEqual(snapshot.current_goal, None)
        self.assertEqual(snapshot.current_plan, None)
        await application.shutdown()

    async def test_mira_lifecycle_events_are_first_class_contract_values(self) -> None:
        frontend = RecordingFrontend()
        emitter = FrontendEmitter(frontend, session_id="mira-session-1", turn_id="turn-2")
        artifact = {"id": "goal-1", "objective": "Separate the frontend"}

        emitter.session_state("opened")
        emitter.artifact("goal", "implement", artifact, {"action": "implement"})
        emitter.rubric_evaluation_started("rubric-1", 1, 3, grader_model="grader")
        emitter.mcp("initialized", server="docs", detail={"tools": 2})

        session_event = next(event for event in frontend.events if isinstance(event, RuntimeEvent))
        artifact_event = next(event for event in frontend.events if isinstance(event, ArtifactEvent))
        rubric_event = next(event for event in frontend.events if isinstance(event, RubricEvent))
        mcp_event = next(event for event in frontend.events if isinstance(event, MCPEvent))
        self.assertEqual((session_event.kind, session_event.state), ("session", "opened"))
        self.assertEqual(artifact_event.artifact_id, "goal-1")
        self.assertEqual(artifact_event.decision, {"action": "implement"})
        self.assertEqual(rubric_event.run_id, "rubric-1")
        self.assertEqual((mcp_event.phase, mcp_event.server), ("initialized", "docs"))

    async def test_native_hitl_decisions_round_trip_without_new_interrupt_model(self) -> None:
        interrupt = {
            "action_requests": [
                {"name": "write_file", "args": {"file_path": "note.txt", "content": "hi"}}
            ],
            "review_configs": [
                {"action_name": "write_file", "allowed_decisions": ["approve", "edit", "reject"]}
            ],
        }
        decisions = [
            {"type": "approve"},
            {
                "type": "edit",
                "edited_action": {
                    "name": "write_file",
                    "args": {"file_path": "note.txt", "content": "edited"},
                },
            },
            {"type": "reject", "message": "Use a different file."},
        ]
        for decision in decisions:
            with self.subTest(decision=decision["type"]):
                frontend = RecordingFrontend([[decision]])
                agent = FakeAgent(
                    [
                        FakeStream(interrupts=[interrupt]),
                        FakeStream(output={"messages": [AIMessage(content="done")]}),
                    ]
                )
                application, session = application_for(frontend, agent)

                result = await session.prompt("Write a note")

                self.assertEqual(result.final_text, "done")
                self.assertEqual(len(frontend.requests), 1)
                request = frontend.requests[0]
                self.assertIsInstance(request, ApprovalRequest)
                self.assertIs(request.interrupts[0], interrupt)
                self.assertEqual(agent.payloads[1].resume, {"decisions": [decision]})
                await application.shutdown()

    async def test_ask_user_and_formal_plan_run_headlessly(self) -> None:
        ask_interrupt = {
            "type": "ask_user",
            "question": "Which path?",
            "options": ["A", "B"],
        }
        ask_call = {
            "name": "ask_user",
            "id": "ask-1",
            "args": {"question": "Which path?", "options": ["A", "B"]},
            "completed": False,
        }
        ask_frontend = RecordingFrontend(["B"])
        ask_agent = FakeAgent(
            [
                FakeStream(tools=[ask_call], interrupts=[ask_interrupt]),
                FakeStream(output={"messages": [AIMessage(content="Chose B")]}),
            ]
        )
        _application, ask_session = application_for(ask_frontend, ask_agent)

        ask_result = await ask_session.prompt("Choose")

        self.assertEqual(ask_result.final_text, "Chose B")
        self.assertIsInstance(ask_frontend.requests[0], AskUserRequest)
        self.assertIs(ask_frontend.requests[0].interrupt, ask_interrupt)
        self.assertEqual(ask_agent.payloads[1].resume, "B")

        plan_interrupt = {
            "type": "finalize_plan",
            "title": "Frontend separation",
            "objective": "Separate MIRA Core from presentation.",
            "context_and_constraints": "Keep native graph behavior.",
            "key_changes": ["Add one frontend contract."],
            "test_plan": ["Run headless and Textual tests."],
            "assumptions": ["No ACP implementation."],
            "success_criteria": "- Core runs headlessly.",
        }
        plan_call = {
            "name": "finalize_plan",
            "id": "plan-call-1",
            "args": {key: value for key, value in plan_interrupt.items() if key != "type"},
            "completed": False,
        }
        plan_frontend = RecordingFrontend([{"action": "close"}])
        plan_agent = FakeAgent(
            [
                FakeStream(tools=[plan_call], interrupts=[plan_interrupt]),
                FakeStream(),
            ]
        )
        _application, plan_session = application_for(plan_frontend, plan_agent)
        await plan_session.set_mode("plan")

        await plan_session.prompt("Create a separation plan")

        artifact_events = [
            event for event in plan_frontend.events if isinstance(event, ArtifactEvent)
        ]
        self.assertEqual([event.phase for event in artifact_events], ["proposed", "close"])
        self.assertIs(plan_frontend.requests[0].interrupt, plan_interrupt)
        self.assertEqual(plan_session.snapshot().current_plan["objective"], plan_interrupt["objective"])
        self.assertNotEqual(plan_session.id, plan_session.mode["plan_thread_id"])


class ArchitectureTests(unittest.TestCase):
    def test_core_packages_do_not_import_ui(self) -> None:
        offenders: list[str] = []
        for package in ("agent", "core"):
            for path in (ROOT / package).rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in ast.walk(tree):
                    module = node.module if isinstance(node, ast.ImportFrom) else ""
                    names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
                    if (module or "").split(".", 1)[0] == "ui" or any(
                        name.split(".", 1)[0] == "ui" for name in names
                    ):
                        offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])

    def test_core_does_not_import_external_protocol_adapters(self) -> None:
        offenders: list[str] = []
        for path in (ROOT / "core").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                module = node.module if isinstance(node, ast.ImportFrom) else ""
                names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
                if (module or "").split(".", 1)[0] == "protocols" or any(
                    name.split(".", 1)[0] == "protocols" for name in names
                ):
                    offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])

    def test_core_interface_has_no_framework_or_acp_dependency(self) -> None:
        forbidden_imports = ("textual", "pyside", "qt", "deepagents_acp")
        offenders: list[str] = []
        for path in (ROOT / "core" / "interface").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                module = node.module if isinstance(node, ast.ImportFrom) else ""
                names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
                imported = [module, *names]
                if any(
                    name.lower().split(".", 1)[0].startswith(forbidden_imports)
                    for name in imported
                    if name
                ):
                    offenders.append(str(path.relative_to(ROOT)))
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
        self.assertEqual(offenders, [])
        self.assertNotIn("deepagents-acp", pyproject)
        self.assertNotIn("deepagents_acp", pyproject)

    def test_runtime_package_is_eliminated(self) -> None:
        self.assertFalse((ROOT / "runtime").exists())

    def test_headless_core_import_does_not_construct_textual(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import core.application, sys; "
                "assert not any(name == 'textual' or name.startswith('textual.') for name in sys.modules)",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
