"""Inspect effective provider messages through real MIRA turns and checkpoints."""

from __future__ import annotations

import asyncio
import gc
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from deepagents.backends import FilesystemBackend
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.runtime import Runtime
from pydantic import Field

from agent.middleware.mode_transition import MODE_NOTICES, MODE_TRANSITION_SOURCE, ModeTransitionMiddleware
from agent.middleware.compaction import create_mira_summarization_middleware
from tests import test_sqlite_sessions as sqlite_tests
from core.execution.compaction import commit_forced_compaction
from core.execution.handoff import effective_conversation
from core.execution.turns import plan_command_prompt
from core.interface.requests import ApprovalRequest
from tests.test_workflows import WorkflowModel
from ui.textual.commands.dispatcher import handle_command


class RecordingModeModel(WorkflowModel):
    requests: list[list] = Field(default_factory=list, exclude=True)

    def bind_tools(self, tools, **kwargs):
        clone = super().bind_tools(tools, **kwargs)
        clone.requests = self.requests
        return clone

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.requests.append(list(messages))
        last_user = next(m.text for m in reversed(messages) if isinstance(m, HumanMessage))
        if last_user == "Write probe.txt" and not isinstance(messages[-1], ToolMessage):
            response = AIMessage(content="", tool_calls=[{
                "name": "write_file", "id": "mode-write", "type": "tool_call",
                "args": {"file_path": "/probe.txt", "content": "done"},
            }])
        else:
            response = AIMessage(content="Done.\nRESPONSE_STATUS: COMPLETE")
        return ChatResult(generations=[ChatGeneration(message=response)])


class ApprovalFrontend:
    def __init__(self, disconnect=False):
        self.disconnect = disconnect

    def emit(self, event):
        pass

    async def request(self, request):
        if not isinstance(request, ApprovalRequest):
            raise AssertionError(f"Unexpected request: {request}")
        if self.disconnect:
            raise RuntimeError("frontend disconnected")
        return [{"type": "approve"}]


class ModeTransitionTests(unittest.IsolatedAsyncioTestCase):
    start = sqlite_tests.SqliteSessionTests.start
    stop = sqlite_tests.SqliteSessionTests.stop

    async def asyncTearDown(self):
        gc.collect()

    def assert_notice(self, messages, mode=None, prompt=None):
        notices = [m for m in messages if m.name == MODE_TRANSITION_SOURCE]
        self.assertEqual(len(notices), int(mode is not None))
        if mode is not None:
            notice = notices[0]
            self.assertIsInstance(notice, HumanMessage)
            self.assertEqual(notice.content, MODE_NOTICES[mode])
            if prompt is not None:
                self.assertEqual(messages[messages.index(notice) + 1].content, prompt)

    async def assert_clean(self, app, session):
        for agent, thread in ((session.agent, session.id), (session.plan_agent, f"{session.id}:plan")):
            config = {"configurable": {"thread_id": thread}}
            async for state in agent.aget_state_history(config):
                self.assert_notice(state.values.get("messages", []))
                self.assert_notice(effective_conversation(agent, state.values))
        serialized = app.store.path(session.id).read_text(encoding="utf-8")
        self.assertNotIn(MODE_TRANSITION_SOURCE, serialized)
        for notice in MODE_NOTICES.values():
            self.assertNotIn(notice.splitlines()[0], serialized)

    async def test_repeated_roundtrips_and_no_subsequent_notice(self):
        with tempfile.TemporaryDirectory() as directory:
            model = RecordingModeModel()
            app = await self.start(Path(directory), model=model)
            try:
                session = await app.open_session("roundtrips")
                await session.set_mode("act")
                await session.prompt("initial")
                self.assert_notice(model.requests[-1])
                for index, mode in enumerate(("plan", "act", "plan", "act", "plan")):
                    await session.set_mode(mode)
                    await session.set_mode(mode)
                    prompt = f"turn {index}"
                    before = len(model.requests)
                    await session.prompt(prompt)
                    self.assertEqual(len(model.requests), before + 1)
                    self.assert_notice(model.requests[-1], mode, prompt)
                    await session.set_mode(mode)
                    await session.prompt(f"followup {index}")
                    self.assert_notice(model.requests[-1])
                await self.assert_clean(app, session)
            finally:
                await self.stop(app)

    async def test_commands_pending_latest_and_inline_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            model = RecordingModeModel()
            app = await self.start(Path(directory), model=model)
            try:
                session = await app.open_session("commands")
                renderer = SimpleNamespace(core_session=session, system_message=lambda *a, **k: None)
                for command in ("/plan", "/act", "/plan", "/act"):
                    await handle_command(command, renderer, session.record, "test", session.mode)
                self.assertFalse(model.requests)
                await session.prompt("latest wins")
                self.assert_notice(model.requests[-1], "act", "latest wins")
                # Match the TUI's existing /plan [prompt] dispatch exactly.
                prompt = plan_command_prompt('/plan Please inspect "palindrome.py".')
                await handle_command("/plan", renderer, session.record, "test", session.mode)
                await session.prompt(prompt)
                self.assert_notice(model.requests[-1], "plan", prompt)
                await self.assert_clean(app, session)
            finally:
                await self.stop(app)

    async def test_pending_transition_survives_cold_restart_and_session_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            app = await self.start(workspace)
            session = await app.open_session("pending")
            await session.set_mode("plan")
            await self.stop(app)
            model = RecordingModeModel()
            app = await self.start(workspace, model=model)
            try:
                restored = await app.open_session("pending")
                other = await app.open_session("other")
                await other.prompt("unrelated")
                self.assert_notice(model.requests[-1])
                await restored.prompt("after restart")
                self.assert_notice(model.requests[-1], "plan", "after restart")
                await restored.prompt("later")
                self.assert_notice(model.requests[-1])
                await self.assert_clean(app, restored)
                await self.assert_clean(app, other)
            finally:
                await self.stop(app)

    async def test_tools_and_live_hitl_keep_notice_on_every_model_call(self):
        with tempfile.TemporaryDirectory() as directory:
            model = RecordingModeModel()
            app = await self.start(Path(directory), model=model, frontend=ApprovalFrontend())
            try:
                session = await app.open_session("tools")
                await session.set_mode("plan")
                await session.set_mode("act")
                await session.prompt("Write probe.txt")
                self.assertEqual(len(model.requests), 2)
                for messages in model.requests:
                    self.assert_notice(messages, "act", "Write probe.txt")
                self.assertIsInstance(model.requests[-1][-1], ToolMessage)
                self.assertEqual(Path(directory, "probe.txt").read_text(), "done")
                await session.prompt("later")
                self.assert_notice(model.requests[-1])
                await self.assert_clean(app, session)
            finally:
                await self.stop(app)

    async def test_interrupted_tool_turn_resumes_after_cold_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            first_model = RecordingModeModel()
            app = await self.start(workspace, model=first_model, frontend=ApprovalFrontend(disconnect=True))
            try:
                session = await app.open_session("resume")
                await session.set_mode("plan")
                await session.set_mode("act")
                with self.assertRaisesRegex(RuntimeError, "disconnected"):
                    await session.prompt("Write probe.txt")
                self.assert_notice(first_model.requests[-1], "act", "Write probe.txt")
                self.assertIsNotNone(session.recovery)
                self.assertFalse((workspace / "probe.txt").exists())
            finally:
                await self.stop(app)
            model = RecordingModeModel()
            app = await self.start(workspace, model=model, frontend=ApprovalFrontend())
            try:
                session = await app.open_session("resume")
                await session.resume_pending()
                self.assertEqual(len(model.requests), 1)
                self.assert_notice(model.requests[0], "act", "Write probe.txt")
                self.assertIsInstance(model.requests[0][-1], ToolMessage)
                await session.prompt("next")
                self.assert_notice(model.requests[-1])
                await self.assert_clean(app, session)
            finally:
                await self.stop(app)

    async def test_incomplete_model_turn_resumes_with_original_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            model = RecordingModeModel()
            app = await self.start(Path(directory), model=model)
            try:
                session = await app.open_session("incomplete")
                await session.set_mode("plan")
                with patch.object(RecordingModeModel, "_generate", side_effect=RuntimeError("provider down")):
                    with self.assertRaisesRegex(RuntimeError, "provider down"):
                        await session.prompt("interrupted prompt")
                self.assertTrue(session.recovery["incomplete"])
                await session.continue_incomplete()
                self.assert_notice(model.requests[-1], "plan", "interrupted prompt")
                await session.prompt("next")
                self.assert_notice(model.requests[-1])
                await self.assert_clean(app, session)
            finally:
                await self.stop(app)

    async def test_cancelled_turn_keeps_binding_for_resume(self):
        entered = asyncio.Event()

        class WaitingFrontend(ApprovalFrontend):
            async def request(self, request):
                entered.set()
                await asyncio.Event().wait()

        with tempfile.TemporaryDirectory() as directory:
            model = RecordingModeModel()
            app = await self.start(Path(directory), model=model, frontend=WaitingFrontend())
            try:
                session = await app.open_session("cancelled")
                await session.set_mode("plan")
                await session.set_mode("act")
                task = asyncio.create_task(session.prompt("Write probe.txt"))
                await asyncio.wait_for(entered.wait(), timeout=30)
                await session.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                app.frontend = ApprovalFrontend()
                await session.resume_pending()
                self.assertEqual(len(model.requests), 2)
                for messages in model.requests:
                    self.assert_notice(messages, "act", "Write probe.txt")
                await self.assert_clean(app, session)
            finally:
                await self.stop(app)

    async def test_completion_before_semantic_save_does_not_requeue_notice(self):
        with tempfile.TemporaryDirectory() as directory:
            model = RecordingModeModel()
            app = await self.start(Path(directory), model=model)
            try:
                session = await app.open_session("completed")
                await session.set_mode("plan")
                await session.prompt("finished")
                state = await session.plan_agent.aget_state({"configurable": {"thread_id": "completed:plan"}})
                anchor = next(m.id for m in state.values["messages"] if isinstance(m, HumanMessage))
                # Reproduce a crash between graph completion and semantic clear.
                session.mode["mode_transition"] = {"mode": "plan", "message_id": anchor,
                                                   "thread_id": "completed:plan"}
                await session.persist_graph_state()
                await session.restore_graph_state()
                self.assertIsNone(session.mode["mode_transition"])
                await session.prompt("next")
                self.assert_notice(model.requests[-1])
            finally:
                await self.stop(app)

    async def test_prepared_messages_keep_exact_relative_order(self):
        with tempfile.TemporaryDirectory() as directory:
            model = RecordingModeModel()
            app = await self.start(Path(directory), model=model)
            try:
                first = await app.open_session("first")
                second = await app.open_session("second")
                await first.set_mode("plan")
                await asyncio.gather(
                    first.prompt("first", prepared_messages=[HumanMessage(content="one"), HumanMessage(content="two")]),
                    second.prompt("independent"),
                )
                transitioned = next(batch for batch in model.requests if any(m.text == "one" for m in batch))
                unrelated = next(batch for batch in model.requests if any(m.text == "independent" for m in batch))
                self.assert_notice(transitioned, "plan", "one")
                self.assertEqual([m.text for m in transitioned if isinstance(m, HumanMessage)
                                  and m.name != MODE_TRANSITION_SOURCE], ["one", "two"])
                self.assert_notice(unrelated)
            finally:
                await self.stop(app)

    async def test_retained_plan_resume_and_revision_queue_mode_notices(self):
        from session.plans import replace_current_plan
        from tests.test_current_plan import artifact

        with tempfile.TemporaryDirectory() as directory:
            model = RecordingModeModel()
            app = await self.start(Path(directory), model=model)
            try:
                session = await app.open_session("artifact")
                retained = replace_current_plan(session.record, artifact())
                await session.set_mode("plan")
                await session.start_artifact("plan")
                await session.prompt("Proceed")
                self.assert_notice(model.requests[-1], "act")
                await session.begin_artifact_revision("plan", retained, "Clarify the result")
                await session.prompt("Clarify the result")
                self.assert_notice(model.requests[-1], "plan")
                await self.assert_clean(app, session)
            finally:
                await self.stop(app)

    async def test_compaction_and_handoff_do_not_archive_notice(self):
        with tempfile.TemporaryDirectory() as directory:
            model = RecordingModeModel()
            app = await self.start(Path(directory), model=model)
            try:
                session = await app.open_session("compact")
                for index in range(12):
                    await session.prompt(f"history {index}")
                model.profile = {"max_input_tokens": 100}
                snapshot = await session.agent.aget_state({"configurable": {"thread_id": session.id}})
                plan = await session.agent.mira_compaction.aplan_forced_compaction_update(snapshot.values)
                result = await commit_forced_compaction(session.agent, session.id, snapshot, plan)
                self.assertTrue(result.compacted)
                model.profile = {"max_input_tokens": 32768}
                await session.set_mode("plan")
                await session.prompt("post compaction")
                self.assert_notice(model.requests[-1], "plan", "post compaction")
                await session.set_mode("act")
                await session.prompt("handoff")
                self.assert_notice(model.requests[-1], "act", "handoff")
                archive = (await session.agent.mira_backend.adownload_files([result.file_path]))[0].content
                self.assertNotIn(MODE_TRANSITION_SOURCE.encode(), archive)
                self.assertNotIn(b"You are now in", archive)
                await self.assert_clean(app, session)
            finally:
                await self.stop(app)

    async def test_sync_async_wrappers_preserve_order_and_do_not_accumulate(self):
        middleware = ModeTransitionMiddleware()
        history = [HumanMessage(content="old", id="old"), AIMessage(content="old reply"),
                   HumanMessage(content="real", id="anchor"), AIMessage(content="", id="call", tool_calls=[{
                       "name": "read", "args": {}, "id": "tool", "type": "tool_call",
                   }]), ToolMessage(content="result", tool_call_id="tool", id="result")]
        request = ModelRequest(model=RecordingModeModel(), messages=history, state={"messages": history},
                               runtime=Runtime(context=SimpleNamespace(mode_transition={
                                   "mode": "plan", "message_id": "anchor",
                               })))
        seen = []

        def handler(value):
            seen.append(value.messages)
            return ModelResponse(result=[AIMessage(content="done")])

        async def ahandler(value):
            return handler(value)

        middleware.wrap_model_call(request, handler)
        await middleware.awrap_model_call(request.override(messages=seen[-1]), ahandler)
        for messages in seen:
            self.assert_notice(messages, "plan", "real")
            self.assertEqual([m for m in messages if m.name != MODE_TRANSITION_SOURCE], history)
        self.assertEqual(request.messages, history)
        # Simulate native projection after the anchor has been summarized.
        projected = [HumanMessage(content="summary"), *history[-2:]]
        middleware.wrap_model_call(request.override(messages=projected), handler)
        self.assert_notice(seen[-1], "plan")
        self.assertEqual([m for m in seen[-1] if m.name != MODE_TRANSITION_SOURCE], projected)

    async def test_native_compaction_of_active_anchor_retains_transient_notice(self):
        with tempfile.TemporaryDirectory() as directory:
            model = RecordingModeModel(profile={"max_input_tokens": 1000})
            backend = FilesystemBackend(root_dir=Path(directory), virtual_mode=True)
            summarization = create_mira_summarization_middleware(
                model, backend, FakeMessagesListChatModel(responses=[AIMessage(content="summarized history")]),
            )
            summarization._lc_helper.keep = ("messages", 2)
            history = [HumanMessage(content="older context " * 1500, id="old"),
                       AIMessage(content="older response", id="reply"),
                       HumanMessage(content="current request", id="anchor")]
            for index in range(3):
                history.extend([AIMessage(content="", id=f"ai-{index}", tool_calls=[{
                    "name": "read_file", "args": {"file_path": "/test"}, "id": f"tool-{index}",
                    "type": "tool_call",
                }]), ToolMessage(content="file contents " * 50, tool_call_id=f"tool-{index}", id=f"result-{index}")])
            request = ModelRequest(model=model, messages=history, state={"messages": history},
                                   runtime=Runtime(context=SimpleNamespace(mode_transition={
                                       "mode": "plan", "message_id": "anchor",
                                   })))

            async def invoke(value):
                return ModelResponse(result=[await model.ainvoke(value.messages)])

            transition = ModeTransitionMiddleware()
            result = await summarization.awrap_model_call(
                request, lambda value: transition.awrap_model_call(value, invoke),
            )
            effective = model.requests[-1]
            self.assertFalse(any(m.id == "anchor" for m in effective))
            self.assert_notice(effective, "plan")
            self.assertEqual(effective[-2:], history[-2:])
            self.assertEqual(request.messages, history)
            self.assert_notice([result.command.update["_summarization_event"]["summary_message"]])
            archive_path = result.command.update["_summarization_event"]["file_path"]
            archive = (await backend.adownload_files([archive_path]))[0].content
            self.assertNotIn(b"You are now in", archive)


if __name__ == "__main__":
    unittest.main()
