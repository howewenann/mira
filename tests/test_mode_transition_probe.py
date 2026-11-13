"""Check reasoning attribution and evidence analysis without a live model."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from tests.probes import probe_mode_transition as probe
from core.interface import FrontendEmitter
from core.interface.requests import ArtifactReviewRequest, ArtifactDisplayRequest
from session.recorder import SessionEventEmitter, SessionRecorder


class ModeTransitionProbeTests(unittest.TestCase):
    def test_recorded_reasoning_is_linked_to_the_actual_model_invocation(self):
        record = {"events": []}
        frontend = probe.RecordedFrontend()
        frontend.session = SimpleNamespace(record=record)
        recorder = SessionRecorder(record, SimpleNamespace(save=lambda value: None), "planning")
        emitter = SessionEventEmitter(FrontendEmitter(frontend, turn_id="1"), recorder)
        emitter.reasoning_delta("I am in Plan mode.", message_id="native-1")
        emitter.text_delta("Read-only reply.", message_id="native-1")
        recorder.finish_main()

        observer = probe.ModelInputs()
        run_id = uuid4()
        observer.on_chat_model_start({}, [[HumanMessage(content="prompt")]], run_id=run_id)
        observer.on_llm_end(LLMResult(generations=[[
            ChatGeneration(message=AIMessage(content="Read-only reply.", id="native-1")),
        ]]), run_id=run_id)
        probe.link_invocations(observer.requests, frontend.links)
        invocation = observer.requests[0]
        self.assertEqual(invocation["reasoning_event_ids"], [1])
        self.assertEqual(invocation["assistant_event_ids"], [2])
        self.assertEqual(invocation["reasoning_availability"], "available")
        self.assertEqual(record["events"][0]["mode"], "planning")
        self.assertEqual(record["events"][0]["text"], "I am in Plan mode.")
        self.assertEqual(frontend.links[1]["message_id"], "native-1")

    def test_unavailable_reasoning_does_not_borrow_another_invocations_event(self):
        requests = [{"response_message_ids": ["one"]}, {"response_message_ids": ["two"]},
                    {"response_message_ids": []}]
        probe.link_invocations(requests, {7: {"message_id": "one", "type": "reasoning"}})
        self.assertEqual(requests[0]["reasoning_availability"], "available")
        self.assertEqual(requests[1]["reasoning_availability"], "unavailable")
        self.assertEqual(requests[1]["reasoning_event_ids"], [])
        self.assertEqual(requests[2]["reasoning_availability"], "unmapped: no response ID")

    def test_cancelled_invocation_can_link_actual_reasoning_by_native_run_metadata(self):
        requests = [{"response_message_ids": [], "callback_run_id": "cancelled-run"}]
        probe.link_invocations(requests, {27: {"message_id": "native", "type": "reasoning",
                                              "metadata": {"run_id": "cancelled-run"}}})
        self.assertEqual(requests[0]["reasoning_event_ids"], [27])
        self.assertEqual(requests[0]["reasoning_availability"], "available")
        self.assertFalse(requests[0]["response_finished"])

    def test_identical_calls_include_all_arguments_and_preserve_event_ids(self):
        events = [{"id": i, "type": "tool_call", "mode": "planning", "name": "read_file",
                   "args": args} for i, args in (
                       (11, {"file_path": "/test"}), (17, {"file_path": "/test"}),
                       (19, {"file_path": "/test", "offset": 5}),
                   )]
        summary = probe.inspect_events(events)
        self.assertEqual(summary["identical_call_groups"], [{
            "name": "read_file", "args": {"file_path": "/test"}, "event_ids": [11, 17],
        }])

    def test_replayed_call_is_not_counted_as_a_new_identical_model_call(self):
        events = [{"id": i, "type": "tool_call", "mode": "planning", "name": "show_plan",
                   "args": {}, "call_id": call_id} for i, call_id in ((16, "one"), (18, "one"), (24, "two"))]
        summary = probe.inspect_events(events)
        self.assertEqual(summary["identical_call_groups"][0]["event_ids"], [16, 24])
        self.assertEqual(summary["replayed_call_groups"], [{"call_id": "one", "event_ids": [16, 18]}])

    def test_mentions_are_candidates_and_repetition_is_not_a_compliance_verdict(self):
        paragraph = "I am currently in Act mode, since the previous turn wrote the file. " * 2
        events = [{"id": 10, "type": "reasoning", "mode": "planning", "text": paragraph + "\n\n" + paragraph}]
        summary = probe.inspect_events(events)
        self.assertEqual(summary["mode_mention_candidates"][0]["event_id"], 10)
        self.assertEqual(summary["mode_mention_candidates"][0]["mode"], "planning")
        self.assertEqual(summary["repeated_reasoning_paragraphs"][0]["event_ids"], [10, 10])
        self.assertNotIn("correct_mode", summary)

    def test_input_observer_keeps_returned_results_and_notice_count(self):
        observer = probe.ModelInputs()
        observer.on_chat_model_start({}, [[
            HumanMessage(content="notice", name=probe.MODE_TRANSITION_SOURCE),
            HumanMessage(content="real prompt"), ToolMessage(content="returned value", tool_call_id="call-1"),
        ]], run_id=uuid4())
        self.assertEqual(observer.requests[0]["notice_count"], 1)
        self.assertEqual(observer.requests[0]["returned_tools"][0]["text"], "returned value")


class ModeTransitionProbeAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_unsolicited_formal_plan_is_rejected_without_implementing(self):
        frontend = probe.RecordedFrontend()
        answer = await frontend.request(ArtifactReviewRequest("plan", None, {"id": "proposal"}))
        self.assertEqual(answer, {"action": "clear"})

    async def test_display_answer_reflects_actual_retained_artifact(self):
        frontend = probe.RecordedFrontend()
        frontend.session = SimpleNamespace(record={"current_plan": None})
        self.assertEqual(await frontend.request(ArtifactDisplayRequest("plan")), "No current Plan.")
        frontend.session.record["current_plan"] = {"id": "retained"}
        self.assertEqual(await frontend.request(ArtifactDisplayRequest("plan")), "Current Plan rendered.")

    async def test_display_resume_uses_a_native_command_and_restores_stream_method(self):
        received = []

        async def native_stream(payload, **kwargs):
            received.append(payload)
            return "native stream"

        graph = SimpleNamespace(astream_events=native_stream)

        async def real_runner(**kwargs):
            return await kwargs["agent"].astream_events({"messages": []})

        async def prompt(text, **kwargs):
            self.assertEqual(text, "")
            self.assertFalse(kwargs["record_user"])
            return await kwargs["turn_runner"](agent=graph)

        session = SimpleNamespace(prompt=prompt)
        with patch.object(probe, "run_turn", real_runner):
            self.assertEqual(await probe.resume_display(session, graph, "No current Plan."), "native stream")
        self.assertIsInstance(received[0], probe.Command)
        self.assertEqual(received[0].resume, "No current Plan.")
        self.assertIs(graph.astream_events, native_stream)


if __name__ == "__main__":
    unittest.main()
