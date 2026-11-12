"""Production evidence retrieval, isolation, and fail-closed grading."""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import AsyncMock, patch

from deepagents.middleware import rubric as stock
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent.rubric.middleware import MiraRubricMiddleware
from agent.rubric.offload import EVIDENCE_TOOL, coverage_gaps, offload_transcript, selected_indices
from tests.test_rubric_middleware import FixedFakeChatModel, grader_message


def read_call(path: str, offset: int = 0, limit: int = 2000) -> AIMessage:
    return AIMessage(content="", tool_calls=[{
        "name": EVIDENCE_TOOL, "id": f"read-{offset}",
        "args": {"file_path": path, "offset": offset, "limit": limit},
    }])


def parent_state(text: str | None = None) -> dict:
    return {"messages": [HumanMessage(content="Review the evidence."),
                         AIMessage(content=text or ("source evidence " * 300))],
            "rubric": "The response is grounded in the source evidence."}


class RubricOffloadTests(unittest.TestCase):
    def test_probe_backup_closes_both_sqlite_connections(self):
        probe_path = Path(__file__).parent / "probes" / "mira_rubric_live_probe_fixed.py"
        spec = importlib.util.spec_from_file_location("rubric_live_probe", probe_path)
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
        connect = sqlite3.connect
        connections = []

        def tracked_connect(*args, **kwargs):
            connection = connect(*args, **kwargs)
            connections.append(connection)
            return connection

        with tempfile.TemporaryDirectory() as directory:
            source, destination = Path(directory) / "source.db", Path(directory) / "backup.db"
            with closing(connect(source)) as connection:
                connection.execute("CREATE TABLE evidence (value TEXT)")
                connection.execute("INSERT INTO evidence VALUES ('original')")
                connection.commit()
            original = source.read_bytes()
            with patch.object(probe.sqlite3, "connect", tracked_connect):
                probe._copy_db_read_only(source, destination)
            self.assertEqual(len(connections), 2)
            for connection in connections:
                with self.assertRaises(sqlite3.ProgrammingError):
                    connection.execute("SELECT 1")
            self.assertEqual(source.read_bytes(), original)
            with closing(connect(destination)) as connection:
                self.assertEqual(connection.execute("SELECT value FROM evidence").fetchone(), ("original",))

    def test_equal_original_request_in_tail_does_not_reappear_truncated(self):
        original = HumanMessage(content="request " * 600)
        messages = [original, *[AIMessage(content="short") for _ in range(30)],
                    original.model_copy(deep=True)]
        modified, _, manifest = offload_transcript(messages)
        self.assertEqual(len(manifest), 1)
        transcript = stock._build_grader_transcript(modified)
        self.assertNotIn("...(truncated)", transcript)
        self.assertEqual(transcript.count(EVIDENCE_TOOL), 1)

    def test_only_stock_selected_oversized_messages_are_offloaded(self):
        messages = [HumanMessage(content="request " * 600), AIMessage(content="omitted " * 600)]
        messages.extend(AIMessage(content="short") for _ in range(30))
        original = copy.deepcopy(messages)
        modified, files, manifest = offload_transcript(messages)
        self.assertEqual(selected_indices(messages), [0, *range(2, 32)])
        self.assertEqual(manifest, [{"path": "/rubric_offload/message-1.txt", "total_lines": 1}])
        self.assertEqual(files[manifest[0]["path"]]["content"], messages[0].content)
        self.assertIn(EVIDENCE_TOOL, modified[0].content)
        self.assertEqual(messages, original)

    def test_threshold_and_tool_arguments_preserve_parent(self):
        self.assertEqual(offload_transcript([HumanMessage(content="x" * 4000)])[2], [])
        message = AIMessage(content="", tool_calls=[{
            "name": "write_file", "id": "write", "args": {"content": "x" * 5000},
        }])
        modified, files, manifest = offload_transcript([message])
        self.assertTrue(manifest)
        self.assertIn("x" * 5000, files[manifest[0]["path"]]["content"])
        self.assertFalse(modified[0].tool_calls)
        self.assertTrue(message.tool_calls)

    def test_coverage_rejects_unlinked_errors_headers_and_wrong_bodies(self):
        path = "/rubric_offload/message-1.txt"
        manifest = [{"path": path, "total_lines": 3}]
        files = {path: {"content": "alpha\nbeta\ngamma"}}
        call = read_call(path, limit=2)
        partial = ToolMessage(content="@@ lines 1-2 of 3 @@\nalpha\nbeta", tool_call_id="read-0")
        self.assertEqual(coverage_gaps([call, partial], manifest, files), {path: [3]})
        for invalid in [
            partial.model_copy(update={"status": "error"}),
            partial.model_copy(update={"tool_call_id": "unmatched"}),
            partial.model_copy(update={"content": "@@ lines 1-2 of 3 @@\nalpha\nwrong"}),
            partial.model_copy(update={"content": "@@ lines 1-3 of 3 @@"}),
            partial.model_copy(update={"content": "@@ lines 1-2 of 4 @@\nalpha\nbeta"}),
        ]:
            with self.subTest(content=invalid.content):
                self.assertEqual(coverage_gaps([call, invalid], manifest, files), {path: [1, 2, 3]})
        final = ToolMessage(content="@@ lines 3-3 of 3 @@\ngamma", tool_call_id="read-2")
        self.assertEqual(coverage_gaps([call, partial, read_call(path, 2, 1), final], manifest, files), {})

    def test_sync_grading_retries_then_forwards_full_native_evidence(self):
        state = parent_state()
        original = copy.deepcopy(state)
        model = FixedFakeChatModel(messages=iter([
            AIMessage(content="VERIFICATION_COMPLETE"),
            read_call("/rubric_offload/message-2.txt"),
            AIMessage(content="VERIFICATION_COMPLETE"), grader_message(),
        ]))
        middleware = MiraRubricMiddleware(model=model)
        stream_grader = middleware._stream_final_grader

        def verify_grader_input(grader, payload, **kwargs):
            evidence = [message for message in payload["messages"] if isinstance(message, ToolMessage)]
            self.assertEqual(len(evidence), 1)
            self.assertEqual(evidence[0].content.split("@@\n", 1)[1], state["messages"][1].content)
            self.assertNotIn("files", payload)
            return stream_grader(grader, payload, **kwargs)

        middleware._stream_final_grader = verify_grader_input
        result = middleware._invoke_grader(state, 0)
        self.assertEqual(result.result, "satisfied")
        self.assertEqual(state, original)
        self.assertIn(EVIDENCE_TOOL, model.bindings[0]["tools"])
        self.assertEqual(model.bindings[-1]["tools"], [])


class AsyncRubricOffloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_invocations_share_no_evidence_or_project_access(self):
        class ReaderModel(FixedFakeChatModel):
            def _generate(self, messages, **kwargs):
                response = (
                    AIMessage(content="VERIFICATION_COMPLETE")
                    if any(isinstance(message, ToolMessage) for message in messages)
                    else read_call("/rubric_offload/message-2.txt")
                )
                return ChatResult(generations=[ChatGeneration(message=response)])

        middleware = MiraRubricMiddleware(model=ReaderModel(messages=iter([])))
        verifier = middleware._ensure_verifier(offloaded=True)
        payloads = [middleware._verifier_input(parent_state((word + " ") * 1000), 0)
                    for word in ("first", "second")]
        results = await asyncio.gather(*[
            middleware._astream_verifier(verifier, payload, config={}, context=None)
            for payload in payloads
        ])
        for word, result in zip(("first", "second"), results):
            evidence = next(message for message in result["messages"] if isinstance(message, ToolMessage))
            self.assertEqual(evidence.content.split("@@\n", 1)[1], (word + " ") * 1000)

        # StateBackend has no route to a project filesystem, even if the
        # verifier asks its dedicated reader for a project-looking path.
        model = FixedFakeChatModel(messages=iter([
            read_call("/project-secret.txt"), AIMessage(content="VERIFICATION_COMPLETE"),
        ]))
        isolated = MiraRubricMiddleware(model=model)
        result = await isolated._astream_verifier(
            isolated._ensure_verifier(offloaded=True),
            {"messages": [HumanMessage(content="Inspect")], "files": {}},
            config={}, context=None,
        )
        returned = next(message for message in result["messages"] if isinstance(message, ToolMessage))
        self.assertIn("not found", returned.content)

    async def test_async_grading_preserves_newlines_and_parent_state(self):
        for ending in ["", "\n", "\r\n", "\r", "\n\n"]:
            with self.subTest(ending=ending):
                source = ("evidence " * 500) + ending
                state = parent_state(source)
                original = copy.deepcopy(state)
                model = FixedFakeChatModel(messages=iter([
                    read_call("/rubric_offload/message-2.txt"),
                    AIMessage(content="VERIFICATION_COMPLETE"), grader_message(),
                ]))
                middleware = MiraRubricMiddleware(model=model)
                result = await middleware._ainvoke_grader(state, 0)
                self.assertEqual(result.result, "satisfied")
                self.assertEqual(state, original)

    async def test_exhaustion_blocks_grader_and_resets_on_next_invocation(self):
        model = FixedFakeChatModel(messages=iter([AIMessage(content="VERIFICATION_COMPLETE") for _ in range(6)]))
        middleware = MiraRubricMiddleware(model=model)
        middleware._astream_final_grader = AsyncMock()
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "offload-read coverage incomplete"):
                await middleware._ainvoke_grader(parent_state(), 0)
        middleware._astream_final_grader.assert_not_called()
        self.assertIsNone(middleware._grader)
        self.assertEqual(len(model.bindings), 6)

    async def test_cached_native_verifier_keeps_files_local_to_each_invocation(self):
        model = FixedFakeChatModel(messages=iter([
            read_call("/rubric_offload/message-2.txt"), AIMessage(content="VERIFICATION_COMPLETE"),
            read_call("/rubric_offload/message-2.txt"), AIMessage(content="VERIFICATION_COMPLETE"),
        ]))
        middleware = MiraRubricMiddleware(model=model)
        verifier = middleware._ensure_verifier(offloaded=True)
        for word in ["first", "second"]:
            state = parent_state((word + " ") * 1000)
            payload = middleware._verifier_input(state, 0)
            result = await middleware._astream_verifier(verifier, payload, config={}, context=None)
            self.assertEqual(coverage_gaps(result["messages"], payload["offloaded_message_manifest"], payload["files"]), {})
            evidence = middleware._verification_evidence(result)
            tool_result = next(message for message in evidence if isinstance(message, ToolMessage))
            self.assertIn(word, tool_result.content)
            self.assertNotIn("files", state)
            self.assertNotIn("offloaded_message_manifest", state)

    async def test_cancellation_does_not_invoke_grader_or_mutate_parent(self):
        state = parent_state()
        original = copy.deepcopy(state)
        model = FixedFakeChatModel(messages=iter([]))
        middleware = MiraRubricMiddleware(model=model)
        middleware._astream_verifier = AsyncMock(side_effect=asyncio.CancelledError)
        middleware._astream_final_grader = AsyncMock()
        with self.assertRaises(asyncio.CancelledError):
            await middleware._ainvoke_grader(state, 0)
        middleware._astream_final_grader.assert_not_called()
        self.assertEqual(state, original)


if __name__ == "__main__":
    unittest.main()
