"""Integration checks for MIRA's shared DeepAgents compaction engine."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import FileDownloadResponse
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain.tools import ToolRuntime
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from agent.factory import build_agent
from agent.middleware.compaction import MiraCompactionMiddleware, create_mira_summarization_middleware
from agent.middleware.context_report import ContextReportMiddleware
from core.context.observation import context_usage_scope
from core.execution.compaction import commit_forced_compaction
from session.checkpoint import make_checkpointer


class BindableModel(FakeMessagesListChatModel):
    def bind_tools(self, tools: list[object], *, tool_choice: object = None, **kwargs: object) -> object:
        return self


class NativeCompactionTests(unittest.IsolatedAsyncioTestCase):
    async def make_mira_conversation(self, workspace: Path, thread_id: str, saver: object = None) -> tuple:
        """Create completed turns through MIRA's full agent and middleware stack."""
        saver = saver or make_checkpointer()
        model = BindableModel(responses=[AIMessage(content="main reply")])
        model.profile = {"max_input_tokens": 5000}
        with patch("agent.factory.get_llm", return_value=model):
            agent = build_agent({}, workspace, saver)
        config = {"configurable": {"thread_id": thread_id}}
        for index in range(12):
            await agent.ainvoke({"messages": [HumanMessage(content=f"history {index}")]}, config)
        model.profile = {"max_input_tokens": 100}
        return agent, model, saver, config

    async def force_compact(self, agent: object, thread_id: str) -> object:
        snapshot = await agent.aget_state({"configurable": {"thread_id": thread_id}})
        plan = await agent.mira_compaction.aplan_forced_compaction_update(snapshot.values)
        return await commit_forced_compaction(agent, thread_id, snapshot, plan)

    def test_native_name_replaces_default_summarizer_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = FilesystemBackend(root_dir=Path(directory), virtual_mode=True)
            main = BindableModel(responses=[AIMessage(content="main")])
            summarization = create_mira_summarization_middleware(main, backend)
            tool = MiraCompactionMiddleware(summarization)
            with patch("deepagents.graph.create_agent", return_value=MagicMock()) as build:
                create_deep_agent(model=main, backend=backend, middleware=[summarization, tool])
            stack = build.call_args.kwargs["middleware"]
            self.assertEqual([item for item in stack if item.name == "SummarizationMiddleware"], [summarization])
            self.assertIs(tool._summarization, summarization)

    async def test_automatic_and_tool_compaction_share_native_engine(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = FilesystemBackend(root_dir=Path(directory), virtual_mode=True)
            main = BindableModel(responses=[AIMessage(content="main")])
            main.profile = {"max_input_tokens": 5000}
            summary_model = BindableModel(responses=[AIMessage(content="dedicated summary")])
            summarization = create_mira_summarization_middleware(main, backend, summary_model)
            tool = MiraCompactionMiddleware(summarization)
            messages = [HumanMessage(content="history " * 150) for _ in range(25)]
            request = ModelRequest(model=main, messages=messages, tools=[], state={"messages": messages})
            observed: list[dict] = []
            seen: list[list] = []

            async def handle(value: ModelRequest) -> ModelResponse:
                seen.append(value.messages)
                return ModelResponse(result=[AIMessage(content="done")])

            with context_usage_scope(observed.append):
                response = await summarization.awrap_model_call(
                    request, lambda value: ContextReportMiddleware(summarization).awrap_model_call(value, handle),
                )
            self.assertIn("dedicated summary", response.command.update["_summarization_event"]["summary_message"].content)
            self.assertLess(len(seen[0]), len(messages))
            self.assertEqual(observed[-1]["context_tokens"], summarization._count_tokens(seen[0], None, []))

            provider = main._get_ls_params()["ls_provider"]
            runtime = ToolRuntime(
                state={"messages": [
                    *messages,
                    AIMessage(
                        content="previous reply",
                        usage_metadata={"input_tokens": 2500, "output_tokens": 1, "total_tokens": 2501},
                        response_metadata={"model_provider": provider},
                    ),
                ]},
                context=None, config={}, stream_writer=lambda _: None,
                tool_call_id="compact-call", store=None,
            )
            tool_result = await tool._arun_compact(runtime)
            self.assertIn("dedicated summary", tool_result.update["_summarization_event"]["summary_message"].content)

    async def test_manual_compaction_on_mira_graph_survives_reload_and_repeats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            agent, model, saver, config = await self.make_mira_conversation(workspace, "mira-compact")
            first = await self.force_compact(agent, "mira-compact")
            self.assertTrue(first.compacted)
            snapshot = await agent.aget_state(config)
            self.assertFalse(snapshot.next)
            self.assertEqual(len(snapshot.values["messages"]), 13)
            self.assertEqual(snapshot.values["_summarization_event"]["file_path"], first.file_path)
            session_id = snapshot.values["_summarization_session_id"]
            self.assertEqual((await self.force_compact(agent, "mira-compact")).reason, "nothing_to_compact")

            model.profile = {"max_input_tokens": 5000}
            with patch("agent.factory.get_llm", return_value=model):
                reloaded = build_agent({}, workspace, saver)
            for index in range(12, 20):
                await reloaded.ainvoke({"messages": [HumanMessage(content=f"history {index}")]}, config)
            model.profile = {"max_input_tokens": 100}
            second = await self.force_compact(reloaded, "mira-compact")
            self.assertEqual(second.file_path, first.file_path)
            state = (await reloaded.aget_state(config)).values
            self.assertEqual(state["_summarization_session_id"], session_id)
            self.assertEqual(len(state["messages"]), 21)
            archive = (await reloaded.mira_backend.adownload_files([first.file_path]))[0].content
            self.assertEqual(archive.count(b"## Summarized at"), 2)
            self.assertEqual(archive.count(b"history 0"), 1)
            self.assertIn(b"history 12", archive)

    async def test_automatic_and_tool_paths_append_to_real_checkpoint_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent, model, _, config = await self.make_mira_conversation(Path(directory), "auto-and-tool")
            self.assertIs(agent.mira_compaction._summarization, agent.mira_summarization)

            # The attached automatic middleware updates this real checkpoint.
            snapshot = await agent.aget_state(config)
            request = ModelRequest(
                model=model, messages=snapshot.values["messages"], tools=[], state=snapshot.values,
            )

            async def handle(_request: ModelRequest) -> ModelResponse:
                return ModelResponse(result=[AIMessage(content="done")])

            response = await agent.mira_summarization.awrap_model_call(request, handle)
            await agent.aupdate_state(snapshot.config, response.command.update)
            state = (await agent.aget_state(config)).values
            first_event = state["_summarization_event"]
            path = first_event["file_path"]
            session_id = state["_summarization_session_id"]
            self.assertTrue(path)

            model.profile = {"max_input_tokens": 5000}
            for index in range(12, 20):
                await agent.ainvoke({"messages": [HumanMessage(content=f"later {index}")]}, config)
            provider = model._get_ls_params()["ls_provider"]
            await agent.aupdate_state(config, {"messages": [AIMessage(
                content="recent reply",
                usage_metadata={"input_tokens": 2500, "output_tokens": 1, "total_tokens": 2501},
                response_metadata={"model_provider": provider},
            )]})
            model.profile = {"max_input_tokens": 100}
            snapshot = await agent.aget_state(config)
            runtime = ToolRuntime(
                state=snapshot.values, context=None, config=config,
                stream_writer=lambda _: None, tool_call_id="compact-tool", store=None,
            )
            command = await agent.mira_compaction._arun_compact(runtime)
            update = {key: command.update[key] for key in ("_summarization_event", "_summarization_session_id")}
            await agent.aupdate_state(snapshot.config, update)

            state = (await agent.aget_state(config)).values
            self.assertEqual(state["_summarization_session_id"], session_id)
            self.assertEqual(state["_summarization_event"]["file_path"], path)
            self.assertGreater(state["_summarization_event"]["cutoff_index"], first_event["cutoff_index"])
            archive = (await agent.mira_backend.adownload_files([path]))[0].content
            self.assertEqual(archive.count(b"## Summarized at"), 2)

    async def test_new_checkpoint_during_summary_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent, model, _, config = await self.make_mira_conversation(Path(directory), "summary-race")

            async def advance(_messages: list[object]) -> str:
                model.profile = {"max_input_tokens": 5000}
                await agent.ainvoke({"messages": [HumanMessage(content="newer turn")]}, config)
                model.profile = {"max_input_tokens": 100}
                return "stale summary"

            with patch.object(agent.mira_summarization, "_acreate_summary", advance):
                with self.assertRaisesRegex(RuntimeError, "Conversation changed"):
                    await self.force_compact(agent, "summary-race")
            state = (await agent.aget_state(config)).values
            self.assertIn("newer turn", [message.content for message in state["messages"]])
            self.assertNotIn("_summarization_event", state)
            self.assertFalse(list((Path(directory) / ".mira" / "conversation_history").glob("*.md")))

    async def test_archive_write_failures_leave_checkpoint_and_archive_retryable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent, _, _, config = await self.make_mira_conversation(Path(directory), "archive-failure")
            backend = agent.mira_backend
            original_write = backend.awrite

            async def write_then_fail(*args: object, **kwargs: object) -> object:
                await original_write(*args, **kwargs)
                raise OSError("response lost after write")

            with patch.object(backend, "awrite", write_then_fail):
                failed = await self.force_compact(agent, "archive-failure")
            self.assertEqual(failed.reason, "archive_failed")
            self.assertNotIn("_summarization_event", (await agent.aget_state(config)).values)
            self.assertFalse(list((Path(directory) / ".mira" / "conversation_history").glob("*.md")))
            recovered = await self.force_compact(agent, "archive-failure")
            archive = (await backend.adownload_files([recovered.file_path]))[0].content
            self.assertEqual(archive.count(b"## Summarized at"), 1)

    async def test_checkpoint_write_failure_rolls_back_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent, _, _, config = await self.make_mira_conversation(Path(directory), "checkpoint-failure")
            with patch.object(agent, "aupdate_state", side_effect=RuntimeError("checkpoint failed")):
                with self.assertRaisesRegex(RuntimeError, "checkpoint failed"):
                    await self.force_compact(agent, "checkpoint-failure")
            self.assertNotIn("_summarization_event", (await agent.aget_state(config)).values)
            self.assertFalse(list((Path(directory) / ".mira" / "conversation_history").glob("*.md")))
            recovered = await self.force_compact(agent, "checkpoint-failure")
            archive = (await agent.mira_backend.adownload_files([recovered.file_path]))[0].content
            self.assertEqual(archive.count(b"## Summarized at"), 1)

    async def test_checkpoint_commit_reported_as_failure_keeps_archive_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent, _, _, config = await self.make_mira_conversation(Path(directory), "commit-response")
            original_update = agent.aupdate_state

            async def commit_then_fail(*args: object, **kwargs: object) -> object:
                await original_update(*args, **kwargs)
                raise RuntimeError("response lost after checkpoint write")

            with patch.object(agent, "aupdate_state", commit_then_fail):
                result = await self.force_compact(agent, "commit-response")
            self.assertTrue(result.compacted)
            self.assertEqual((await agent.aget_state(config)).values["_summarization_event"]["file_path"], result.file_path)
            self.assertEqual((await self.force_compact(agent, "commit-response")).reason, "nothing_to_compact")

    async def test_new_checkpoint_after_archive_write_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent, model, _, config = await self.make_mira_conversation(Path(directory), "archive-race")
            backend = agent.mira_backend
            original_write = backend.awrite

            async def write_then_advance(*args: object, **kwargs: object) -> object:
                result = await original_write(*args, **kwargs)
                model.profile = {"max_input_tokens": 5000}
                await agent.ainvoke({"messages": [HumanMessage(content="newer turn")]}, config)
                model.profile = {"max_input_tokens": 100}
                return result

            with patch.object(backend, "awrite", write_then_advance):
                with self.assertRaisesRegex(RuntimeError, "Conversation changed"):
                    await self.force_compact(agent, "archive-race")
            state = (await agent.aget_state(config)).values
            self.assertIn("newer turn", [message.content for message in state["messages"]])
            self.assertNotIn("_summarization_event", state)
            self.assertFalse(list((Path(directory) / ".mira" / "conversation_history").glob("*.md")))

    async def test_archive_read_error_preserves_existing_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent, model, _, config = await self.make_mira_conversation(Path(directory), "archive-read")
            first = await self.force_compact(agent, "archive-read")
            backend = agent.mira_backend
            before = (await backend.adownload_files([first.file_path]))[0].content
            model.profile = {"max_input_tokens": 5000}
            for index in range(12, 20):
                await agent.ainvoke({"messages": [HumanMessage(content=f"history {index}")]}, config)
            model.profile = {"max_input_tokens": 100}
            with patch.object(
                backend, "adownload_files",
                return_value=[FileDownloadResponse(path=first.file_path, error="permission_denied")],
            ), self.assertRaisesRegex(RuntimeError, "Could not read conversation history"):
                await self.force_compact(agent, "archive-read")
            self.assertEqual((await backend.adownload_files([first.file_path]))[0].content, before)
            self.assertEqual((await agent.aget_state(config)).values["_summarization_event"]["file_path"], first.file_path)


if __name__ == "__main__":
    unittest.main()
