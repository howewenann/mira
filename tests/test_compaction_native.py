"""Integration checks for MIRA's DeepAgents 0.7 summarization engine."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import FileDownloadResponse
from deepagents.middleware.summarization import SummarizationToolMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain.tools import ToolRuntime
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent.middleware.compaction import compact_after_turn, create_mira_summarization_middleware
from agent.middleware.context_report import ContextReportMiddleware
from core.context.observation import context_usage_scope


class BindableModel(FakeMessagesListChatModel):
    def bind_tools(self, tools: list[object], *, tool_choice: object = None, **kwargs: object) -> object:
        return self


class NativeCompactionTests(unittest.IsolatedAsyncioTestCase):
    def test_native_name_replaces_default_summarizer_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = FilesystemBackend(root_dir=Path(directory), virtual_mode=True)
            main = BindableModel(responses=[AIMessage(content="main")])
            summarization = create_mira_summarization_middleware(main, backend)
            tool = SummarizationToolMiddleware(summarization)
            with patch("deepagents.graph.create_agent", return_value=MagicMock()) as build:
                create_deep_agent(model=main, backend=backend, middleware=[summarization, tool])
            stack = build.call_args.kwargs["middleware"]
            summarizers = [item for item in stack if item.name == "SummarizationMiddleware"]
            self.assertEqual(summarizers, [summarization])
            self.assertIs(tool._summarization, summarization)
            subagents = next(item for item in stack if item.name == "SubAgentMiddleware")
            general = next(spec for spec in subagents._subagents if spec["name"] == "general-purpose")
            self.assertEqual(
                [item for item in general["middleware"] if item.name == "SummarizationMiddleware"],
                [summarization],
            )

    async def test_automatic_and_tool_compaction_share_native_engine(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = FilesystemBackend(root_dir=Path(directory), virtual_mode=True)
            main = BindableModel(responses=[AIMessage(content="main")])
            main.profile = {"max_input_tokens": 5000}
            summary_model = BindableModel(responses=[AIMessage(content="dedicated summary")])
            summarization = create_mira_summarization_middleware(main, backend, summary_model)
            tool = SummarizationToolMiddleware(summarization)
            messages = [HumanMessage(content="history " * 150) for _ in range(25)]
            request = ModelRequest(model=main, messages=messages, tools=[], state={"messages": messages})
            seen = []
            observed = []

            async def handle(value: ModelRequest) -> ModelResponse:
                seen.append(value.messages)
                return ModelResponse(result=[AIMessage(content="done")])

            observer = ContextReportMiddleware(summarization)
            with context_usage_scope(observed.append):
                response = await summarization.awrap_model_call(
                    request, lambda value: observer.awrap_model_call(value, handle),
                )
            update = response.command.update
            self.assertIn("dedicated summary", update["_summarization_event"]["summary_message"].content)
            self.assertTrue(update["_summarization_session_id"])
            self.assertLess(len(seen[0]), len(messages))
            self.assertEqual(
                observed[-1]["context_tokens"], summarization._count_tokens(seen[0], None, []),
            )
            self.assertLess(observed[-1]["context_tokens"], summarization._count_tokens(messages, None, []))

            provider = main._get_ls_params()["ls_provider"]
            tool_messages = [
                *messages,
                AIMessage(
                    content="previous reply",
                    usage_metadata={"input_tokens": 2500, "output_tokens": 1, "total_tokens": 2501},
                    response_metadata={"model_provider": provider},
                ),
            ]
            runtime = ToolRuntime(
                state={"messages": tool_messages}, context=None, config={},
                stream_writer=lambda _: None, tool_call_id="compact-call", store=None,
            )
            tool_result = await tool._arun_compact(runtime)
            self.assertIn("_summarization_event", tool_result.update)
            self.assertIn("dedicated summary", tool_result.update["_summarization_event"]["summary_message"].content)

    async def test_shared_engine_compacts_checkpoint_and_appends_after_reload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = FilesystemBackend(root_dir=Path(directory), virtual_mode=True)
            saver = InMemorySaver()
            main = BindableModel(responses=[AIMessage(content="main response")])
            main.profile = {"max_input_tokens": 100}
            summary_model = BindableModel(responses=[AIMessage(content="dedicated summary")])
            summarization = create_mira_summarization_middleware(main, backend, summary_model)
            tool = SummarizationToolMiddleware(summarization)
            self.assertIs(tool._summarization, summarization)
            self.assertEqual(summarization.name, "SummarizationMiddleware")
            self.assertIs(summarization.model, main)
            self.assertEqual(summarization._get_profile_limits(), 100)

            agent = create_deep_agent(
                model=main, backend=backend, middleware=[summarization, tool], checkpointer=saver,
            )
            agent.mira_summarization = summarization
            config = {"configurable": {"thread_id": "native-compact"}}
            original = [
                AIMessage(content=[
                    {"type": "reasoning", "reasoning": "private reasoning"},
                    {"type": "text", "text": "visible answer"},
                ]),
                *[HumanMessage(content=f"message {index}") for index in range(1, 25)],
            ]
            await agent.aupdate_state(config, {"messages": original}, as_node="model")

            first = await compact_after_turn(agent, "native-compact")
            self.assertTrue(first.compacted)
            self.assertEqual(first.summary, "dedicated summary")
            self.assertTrue(first.file_path)
            first_snapshot = await agent.aget_state(config)
            self.assertFalse(first_snapshot.next)
            state = first_snapshot.values
            self.assertEqual(len(state["messages"]), len(original))
            first_cutoff = state["_summarization_event"]["cutoff_index"]
            self.assertGreater(first_cutoff, 0)
            session_id = state["_summarization_session_id"]
            archive = (await backend.adownload_files([first.file_path]))[0].content
            self.assertIn(b"visible answer", archive)
            self.assertNotIn(b"private reasoning", archive)
            self.assertFalse(state["_summarization_event"]["summary_message"].additional_kwargs)
            self.assertFalse((await compact_after_turn(agent, "native-compact")).compacted)

            reloaded_summary = create_mira_summarization_middleware(main, backend, summary_model)
            reloaded = create_deep_agent(
                model=main,
                backend=backend,
                middleware=[reloaded_summary, SummarizationToolMiddleware(reloaded_summary)],
                checkpointer=saver,
            )
            reloaded.mira_summarization = reloaded_summary
            await reloaded.aupdate_state(
                config, {"messages": [HumanMessage(content=f"new {index}") for index in range(6)]},
                as_node="model",
            )
            second = await compact_after_turn(reloaded, "native-compact")
            self.assertTrue(second.compacted)
            self.assertEqual(second.file_path, first.file_path)
            state = (await reloaded.aget_state(config)).values
            self.assertEqual(state["_summarization_session_id"], session_id)
            self.assertGreater(state["_summarization_event"]["cutoff_index"], first_cutoff)
            archive = (await backend.adownload_files([first.file_path]))[0].content
            self.assertEqual(archive.count(b"## Summarized at"), 2)
            self.assertIn(b"new 0", archive)
            self.assertNotIn(b"dedicated summary", archive)

    async def test_checkpoint_failure_does_not_append_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = FilesystemBackend(root_dir=Path(directory), virtual_mode=True)
            main = BindableModel(responses=[AIMessage(content="summary")])
            main.profile = {"max_input_tokens": 100}
            summarization = create_mira_summarization_middleware(main, backend)
            graph = create_deep_agent(model=main, backend=backend, middleware=[summarization], checkpointer=InMemorySaver())
            config = {"configurable": {"thread_id": "write-failure"}}
            await graph.aupdate_state(
                config, {"messages": [HumanMessage(content=f"old {index}") for index in range(25)]},
                as_node="model",
            )

            class FailingWrite:
                mira_summarization = summarization

                async def aget_state(self, requested: dict) -> object:
                    return await graph.aget_state(requested)

                async def aupdate_state(self, *args: object, **kwargs: object) -> None:
                    raise RuntimeError("checkpoint failed")

            with self.assertRaisesRegex(RuntimeError, "checkpoint failed"):
                await compact_after_turn(FailingWrite(), "write-failure")
            self.assertNotIn("_summarization_event", (await graph.aget_state(config)).values)
            self.assertFalse(list(Path(directory).rglob("*.md")))

    async def test_changed_checkpoint_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = FilesystemBackend(root_dir=Path(directory), virtual_mode=True)
            main = BindableModel(responses=[AIMessage(content="summary")])
            main.profile = {"max_input_tokens": 100}
            summarization = create_mira_summarization_middleware(main, backend)
            graph = create_deep_agent(model=main, backend=backend, middleware=[summarization], checkpointer=InMemorySaver())
            config = {"configurable": {"thread_id": "changed-checkpoint"}}
            await graph.aupdate_state(
                config, {"messages": [HumanMessage(content=f"old {index}") for index in range(25)]},
                as_node="model",
            )

            class AdvancingGraph:
                mira_summarization = summarization
                reads = 0

                async def aget_state(self, requested: dict) -> object:
                    self.reads += 1
                    if self.reads == 2:
                        await graph.aupdate_state(
                            requested, {"messages": [HumanMessage(content="newer checkpoint")]},
                            as_node="model",
                        )
                    return await graph.aget_state(requested)

                async def aupdate_state(self, *args: object, **kwargs: object) -> None:
                    raise AssertionError("stale summary was committed")

            with self.assertRaisesRegex(RuntimeError, "Conversation changed"):
                await compact_after_turn(AdvancingGraph(), "changed-checkpoint")
            state = (await graph.aget_state(config)).values
            self.assertEqual(state["messages"][-1].content, "newer checkpoint")
            self.assertNotIn("_summarization_event", state)
            self.assertFalse(list(Path(directory).rglob("*.md")))

    async def test_archive_read_error_preserves_existing_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = FilesystemBackend(root_dir=Path(directory), virtual_mode=True)
            main = BindableModel(responses=[AIMessage(content="summary")])
            main.profile = {"max_input_tokens": 100}
            summarization = create_mira_summarization_middleware(main, backend)
            graph = create_deep_agent(model=main, backend=backend, middleware=[summarization], checkpointer=InMemorySaver())
            graph.mira_summarization = summarization
            config = {"configurable": {"thread_id": "archive-error"}}
            await graph.aupdate_state(
                config, {"messages": [HumanMessage(content=f"old {index}") for index in range(25)]},
                as_node="model",
            )
            first = await compact_after_turn(graph, "archive-error")
            before = (await backend.adownload_files([first.file_path]))[0].content
            await graph.aupdate_state(
                config, {"messages": [HumanMessage(content=f"new {index}") for index in range(10)]},
                as_node="model",
            )
            with patch.object(
                backend, "adownload_files",
                return_value=[FileDownloadResponse(path=first.file_path, error="permission_denied")],
            ), self.assertLogs("agent.middleware.compaction", level="ERROR"):
                failed = await compact_after_turn(graph, "archive-error")
            self.assertTrue(failed.compacted)
            self.assertEqual(failed.reason, "archive_failed")
            self.assertEqual((await backend.adownload_files([first.file_path]))[0].content, before)
            state = (await graph.aget_state(config)).values
            self.assertIsNone(state["_summarization_event"]["file_path"])

    async def test_archive_link_failure_preserves_checkpoint_and_later_append(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = FilesystemBackend(root_dir=Path(directory), virtual_mode=True)
            main = BindableModel(responses=[AIMessage(content="summary")])
            main.profile = {"max_input_tokens": 100}
            summarization = create_mira_summarization_middleware(main, backend)
            graph = create_deep_agent(model=main, backend=backend, middleware=[summarization], checkpointer=InMemorySaver())
            graph.mira_summarization = summarization
            config = {"configurable": {"thread_id": "link-failure"}}
            original = [HumanMessage(content=f"old {index}") for index in range(25)]
            await graph.aupdate_state(config, {"messages": original}, as_node="model")

            class FailingLink:
                mira_summarization = summarization
                writes = 0

                async def aget_state(self, requested: dict) -> object:
                    return await graph.aget_state(requested)

                async def aupdate_state(self, *args: object, **kwargs: object) -> object:
                    self.writes += 1
                    if self.writes == 2:
                        raise RuntimeError("archive link failed")
                    return await graph.aupdate_state(*args, **kwargs)

            with self.assertRaisesRegex(RuntimeError, "archive link failed"):
                await compact_after_turn(FailingLink(), "link-failure")
            state = (await graph.aget_state(config)).values
            self.assertEqual(len(state["messages"]), len(original))
            self.assertIsNone(state["_summarization_event"]["file_path"])
            path = summarization._get_history_path(state["_summarization_session_id"])
            self.assertEqual((await backend.adownload_files([path]))[0].content.count(b"## Summarized at"), 1)

            await graph.aupdate_state(
                config, {"messages": [HumanMessage(content=f"new {index}") for index in range(10)]},
                as_node="model",
            )
            recovered = await compact_after_turn(graph, "link-failure")
            self.assertEqual(recovered.file_path, path)
            self.assertEqual((await backend.adownload_files([path]))[0].content.count(b"## Summarized at"), 2)


if __name__ == "__main__":
    unittest.main()
