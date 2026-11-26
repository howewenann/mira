"""Inspection ownership through the production typed frontend boundary."""

from __future__ import annotations

import asyncio
import io
import runpy
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, ToolMessage

from mira import MiraSession
from mira.api import InspectionEvent, RuntimeEvent, SubagentEvent, ToolEvent
from tests.core.interface.test_contract import (
    AsyncItems, FakeAgent, FakeStream, RecordingFrontend, StreamedMessage,
    application_for, session_record,
)
from tests.test_rubric_persistence import phase_start, phase_end
from tests.core.interface.test_contract import ROOT


def child(call_id: str, output=None):
    async def completed():
        return f"answer {call_id}"

    return SimpleNamespace(
        name="worker", task_input="identical request", id=call_id,
        cause={"type": "toolCall", "tool_call_id": call_id},
        trigger_call_id=f"native-{call_id}", path=(f"tools:native-{call_id}",),
        messages=AsyncItems([StreamedMessage()]), tool_calls=AsyncItems([]),
        output=output if output is not None else completed(),
    )


def parent_call(call_id: str):
    return SimpleNamespace(
        name="task", args={"description": "identical request", "subagent_type": "worker"},
        id=call_id, output=f"answer {call_id}", completed=True,
    )


def delegation(*call_ids):
    message = StreamedMessage()
    message.tool_calls = [{"name": "task", "id": call_id,
                           "args": {"description": "identical request", "subagent_type": "worker"}}
                          for call_id in call_ids]
    return message


def final_output(*call_ids):
    return {"messages": [
        AIMessage(content="", tool_calls=delegation(*call_ids).tool_calls),
        *(ToolMessage(content=f"answer {key}", tool_call_id=key, name="task") for key in call_ids),
        AIMessage(content="done"),
    ]}


class InspectionContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_request_arriving_after_child_updates_captured_input(self):
        captured = asyncio.Event()

        class Frontend(RecordingFrontend):
            def emit(self, event):
                super().emit(event)
                if isinstance(event, RuntimeEvent) and event.kind == "inspection":
                    captured.set()

        class LateTaskCalls:
            async def __aiter__(self):
                await captured.wait()
                yield parent_call("late")

        handle = child("late")
        handle.task_input = ""
        stream = FakeStream(output=final_output("late"))
        stream.subagents = AsyncItems([handle])
        stream.tool_calls = LateTaskCalls()
        app, session = application_for(Frontend(), FakeAgent([stream]))
        await session.prompt("delegate")
        self.assertEqual(session.inspection("subagent:late").events[0].text, "identical request")
        await app.shutdown()

    async def test_minimal_example_uses_public_lifecycle_and_refuses_requests(self):
        example = runpy.run_path(str(ROOT / "examples/mira_api/minimal_frontend.py"))
        session = SimpleNamespace(prompt=AsyncMock())
        application = SimpleNamespace(open_session=AsyncMock(return_value=session), shutdown=AsyncMock())
        with patch.object(example["MiraApplication"], "start", new=AsyncMock(return_value=application)):
            await example["main"]()
        session.prompt.assert_awaited_once_with("Reply only with PONG")
        application.shutdown.assert_awaited_once()
        with self.assertRaisesRegex(RuntimeError, "Interaction not supported"):
            await example["MinimalFrontend"]().request(object())

    async def test_full_example_restores_and_selects_inspection_without_internal_imports(self):
        from mira.api import LiveInspection

        example = runpy.run_path(str(ROOT / "examples/mira_api/full_frontend.py"))
        session = SimpleNamespace(
            snapshot=lambda: SimpleNamespace(session_id="saved"),
            inspections=lambda: [{"id": "run-1", "status": "DONE"}],
            inspection=lambda key: LiveInspection(key, "child", events=[InspectionEvent("assistant", "captured answer")]),
            resume_pending=AsyncMock(), continue_incomplete=AsyncMock(),
        )
        application = SimpleNamespace(open_session=AsyncMock(return_value=session), shutdown=AsyncMock())
        output = io.StringIO()
        with patch.object(example["MiraApplication"], "start", new=AsyncMock(return_value=application)), \
                patch("sys.argv", ["full_frontend.py", "--resume", "--session", "saved"]), \
                patch("asyncio.to_thread", new=AsyncMock(side_effect=[
                    "/inspections", "/inspect run-1", "/resume-pending", "/continue-incomplete", "y", "/quit",
                ])), redirect_stdout(output):
            await example["main"]()
        application.open_session.assert_awaited_once_with(session_id="saved", resume=True)
        session.resume_pending.assert_awaited_once()
        session.continue_incomplete.assert_awaited_once()
        application.shutdown.assert_awaited_once()
        self.assertIn("captured answer", output.getvalue())

    async def test_full_example_cancellation_waits_for_turn_start_and_joins_cleanup(self):
        example = runpy.run_path(str(ROOT / "examples/mira_api/full_frontend.py"))
        application = SimpleNamespace(shutdown=AsyncMock())
        session = SimpleNamespace()
        active = None

        async def start(**kwargs):
            frontend = kwargs["frontend"]

            async def prompt(text):
                nonlocal active
                active = asyncio.current_task()
                frontend.emit(RuntimeEvent(kind="session", state="turn_started"))
                await asyncio.Event().wait()

            async def cancel():
                self.assertIsNotNone(active)
                active.cancel()

            session.prompt = prompt
            session.cancel = cancel
            application.open_session = AsyncMock(return_value=session)
            return application

        with patch.object(example["MiraApplication"], "start", side_effect=start), \
                patch("sys.argv", ["full_frontend.py", "--cancel-after", "0", "work"]), \
                redirect_stdout(io.StringIO()):
            await example["main"]()
        self.assertTrue(active.cancelled())
        application.shutdown.assert_awaited_once()

    async def test_protocol_only_frontend_captures_parallel_children_and_durable_origins(self):
        frontend = RecordingFrontend()
        stream = FakeStream(messages=[delegation("a", "b")], tools=[parent_call("a"), parent_call("b")],
                            output=final_output("a", "b"))
        stream.subagents = AsyncItems([child("a"), child("b")])
        app, session = application_for(frontend, FakeAgent([stream]))
        await session.prompt("delegate")
        self.assertEqual(set(vars(frontend)), {"events", "requests", "responses"})
        runs = session.inspections()
        self.assertEqual(len(runs), 2)
        self.assertEqual({run["origin_call_id"] for run in runs}, {"a", "b"})
        starts = [event for event in frontend.events if isinstance(event, SubagentEvent)
                  and event.phase == "start"]
        self.assertEqual(len({event.metadata["inspection_id"] for event in starts}), 2)
        for run in runs:
            self.assertNotIn("events", run)
            # Standalone task delegation can have no durable tool-call entry;
            # the run still preserves the exact native call, without inventing one.
            self.assertIsNone(run["origin_event_id"])
            inspection = session.inspection(run["inspection_id"])
            self.assertEqual(inspection.status, "DONE")
            self.assertEqual(inspection.events[-1].text, f"answer {run['origin_call_id']}")
        updates = [event for event in frontend.events if isinstance(event, RuntimeEvent)
                   and event.kind == "inspection"]
        self.assertTrue(updates)
        self.assertEqual({event.session_id for event in updates}, {session.id})
        self.assertEqual({event.turn_id for event in updates}, {starts[0].turn_id})
        self.assertTrue(any(event.state == "delta" for event in updates))
        await app.shutdown()

    async def test_queries_and_incremental_payloads_are_detached_and_session_scoped(self):
        frontend = RecordingFrontend()
        app, first = application_for(frontend, FakeAgent([]))
        other_record = session_record()
        other_record["id"] = "other-session"
        second = MiraSession(app, other_record)
        for session in (first, second):
            session._inspections.start("same-id", session.id, "task")
            session._inspections.append("same-id", InspectionEvent(
                "tool_call", name="read_file", args={"paths": [session.id]}, call_id="tool",
            ))
        observed = first.inspection("same-id")
        observed.events[-1].args["paths"].append("mutated")
        update = next(event for event in frontend.events if isinstance(event, RuntimeEvent)
                      and event.kind == "inspection" and event.state == "append")
        update.detail["update"].event.args["paths"].append("event mutation")
        self.assertEqual(first.inspection("same-id").events[-1].args["paths"], [first.id])
        self.assertEqual(second.inspection("same-id").events[-1].args["paths"], [second.id])
        self.assertIsNone(first.inspection("unknown"))
        self.assertEqual(first.inspections(origin_event_id=999), [])
        await first.close()
        self.assertIsNone(first.inspection("same-id"))
        self.assertEqual(first._inspections._items, {})
        self.assertIsNotNone(second.inspection("same-id"))
        await app.shutdown()

    async def test_historical_queries_need_no_live_store_or_graph_execution(self):
        stream = FakeStream(messages=[delegation("a")], tools=[parent_call("a")], output=final_output("a"))
        stream.subagents = AsyncItems([child("a")])
        app, session = application_for(RecordingFrontend(), FakeAgent([stream]))
        await session.prompt("delegate")
        run = session.inspections()[0]
        record = deepcopy(session.record)
        await session.close()
        restored = MiraSession(app, record)
        self.assertEqual(restored._inspections._items, {})
        historical = restored.inspection(run["id"])
        self.assertEqual(historical.events[-1].text, "answer a")
        self.assertEqual(restored.inspection(run["inspection_id"]), historical)
        historical.events.clear()
        run["task"] = "changed"
        self.assertTrue(restored.inspection(run["id"]).events)
        self.assertEqual(restored.inspections()[0]["task"], "identical request")
        await app.shutdown()

    async def test_rubric_verifier_and_grader_use_session_capture_through_custom_events(self):
        stream = FakeStream()
        stream.custom = AsyncItems([
            phase_start("rubric-run", 0, "verifier", [{"kind": "user", "text": "verify input"}]),
            {"type": "rubric_inspection_delta", "grading_run_id": "rubric-run",
             "iteration": 0, "phase": "verifier", "kind": "assistant", "text": "verified"},
            phase_end("rubric-run", 0, "verifier", final_response="verified"),
            phase_start("rubric-run", 0, "grader", [{"kind": "user", "text": "grade input"}]),
            {"type": "rubric_inspection_delta", "grading_run_id": "rubric-run",
             "iteration": 0, "phase": "grader", "kind": "assistant", "text": "graded"},
            phase_end("rubric-run", 0, "grader", final_response="graded"),
        ])
        frontend = RecordingFrontend()
        app, session = application_for(frontend, FakeAgent([stream]))
        await session.prompt("evaluate")
        verifier = session.inspection("rubric:rubric-run:0:verifier")
        grader = session.inspection("rubric:rubric-run:0:grader")
        self.assertEqual(verifier.events[0].text, "verify input")
        self.assertEqual(grader.events[0].text, "grade input")
        self.assertEqual(verifier.events[-1].text, "verified")
        self.assertEqual(grader.events[-1].text, "graded")
        self.assertEqual(len(session.inspections()), 2)
        self.assertTrue(all(run["origin_event_id"] is None for run in session.inspections()))
        await app.shutdown()

    async def test_cancellation_retains_captured_children_and_shutdown_releases_live_data(self):
        running = asyncio.Event()

        async def blocked():
            running.set()
            await asyncio.Event().wait()

        stream = FakeStream(messages=[delegation("a")], tools=[parent_call("a")], output=final_output("a"))
        stream.subagents = AsyncItems([child("a", blocked())])
        frontend = RecordingFrontend()
        app, session = application_for(frontend, FakeAgent([stream]))
        task = asyncio.create_task(session.prompt("delegate"))
        await running.wait()
        await session.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        inspection = session.inspection("subagent:a")
        self.assertEqual(inspection.status, "CANCELLED")
        self.assertTrue(inspection.events)
        self.assertEqual(session.inspections()[0]["status"], "CANCELLED", session.inspections())
        await app.shutdown()
        self.assertEqual(session._inspections._items, {})

    async def test_explicit_recovery_retains_inspection_and_run_identity(self):
        stream = FakeStream(messages=[delegation("a")], tools=[parent_call("a")], output=final_output("a"))
        stream.subagents = AsyncItems([child("a")])
        frontend = RecordingFrontend()
        app, session = application_for(frontend, FakeAgent([stream]))
        await session.prompt("delegate")
        run = session.inspections()[0]
        original_turn = session.record["events"][0]["turn_id"]
        record = deepcopy(session.record)
        await session.close()
        resumed_stream = FakeStream(messages=[delegation("a")], tools=[parent_call("a")], output=final_output("a"))
        resumed_stream.subagents = AsyncItems([child("a")])
        agent = FakeAgent([resumed_stream])
        resumed = MiraSession(app, record, agent=agent, plan_agent=agent)
        resumed.recovery = {"incomplete": True, "interrupts": [], "agent": agent,
                            "thread_id": resumed.id, "turn_id": original_turn}
        await resumed.continue_incomplete()
        self.assertEqual(len(resumed.inspections()), 1)
        self.assertEqual(resumed.inspections()[0]["id"], run["id"])
        self.assertEqual(resumed.inspections()[0]["inspection_id"], run["inspection_id"])
        self.assertEqual(resumed.inspections()[0]["origin_event_id"], run["origin_event_id"])
        await app.shutdown()

    async def test_text_projection_batches_content_and_caches_known_origin(self):
        from core.execution.inspection.persistence import PersistentSubagentRuns
        from core.interface import FrontendEmitter

        app, session = application_for(RecordingFrontend(), FakeAgent([]))
        session.record["events"] = [{"type": "tool_call", "id": 1, "name": "task", "call_id": "a"}]
        store = session._inspections
        store.start("subagent:a", "worker", "request")
        observer = PersistentSubagentRuns(FrontendEmitter(app.frontend), session.record,
                                          app.store, store)
        observer.subagent_started("worker", "request", inspection_id="subagent:a", row_id="a")
        with patch.object(observer, "_origin_event_id", side_effect=AssertionError("rescanned origin")), \
                patch.object(observer, "_snapshot", wraps=observer._snapshot) as snapshot:
            for _ in range(100):
                store.append_delta("subagent:a", "assistant", "x")
            snapshot.assert_not_called()
            observer.flush()
            self.assertEqual(snapshot.call_count, 1)
        self.assertEqual(session.inspection("subagent:a").events[-1].text, "x" * 100)
        observer.close()
        await app.shutdown()
