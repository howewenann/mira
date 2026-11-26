"""Regression evidence for outer-interaction identity and frontend projections."""

from __future__ import annotations

import asyncio
import io
import runpy
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from core.execution.inspection.live import LiveInspectionStore
from core.execution.inspection.subagents import live_inspection_store
from core.interface import ArtifactEvent, FrontendEmitter, MessageEvent, RuntimeEvent, ToolEvent
from core.execution.runner import TurnResult
from session.context import normalize_events
from tests.core.interface.test_contract import (
    FakeAgent, FakeStream, MemoryStore, RecordingFrontend, StreamedMessage,
    ToolCall, application_for, session_record,
    ROOT,
)
from session.recorder import SessionEventEmitter, SessionRecorder
from ui.shared.adapter import RendererAdapter


class ContractCorrectnessTests(unittest.IsolatedAsyncioTestCase):
    async def test_plan_and_goal_revision_implementation_and_rubric_share_outer_identity(self) -> None:
        for kind in ('plan', 'goal'):
            with self.subTest(kind=kind):
                frontend = RecordingFrontend([
                    {'action': 'revise', 'feedback': 'Make the criterion precise'},
                    {'action': 'implement'},
                ])
                application, session = application_for(frontend, FakeAgent([]))
                phases = []

                async def runner(**kwargs):
                    renderer = kwargs['renderer']
                    phases.append(kwargs['planning_stage'])
                    if kwargs['planning_stage']:
                        artifact = {
                            'type': f'finalize_{kind}', 'title': 'Reviewed work',
                            'objective': 'Deliver the requested change',
                            'success_criteria': '- A focused check passes.',
                            'context_and_constraints': 'Preserve existing functionality.',
                            'key_changes': ['Make the focused change.'],
                            'test_plan': ['Run the focused check.'],
                            'assumptions': ['No additional assumptions.'],
                        }
                        review = await getattr(renderer, f'finalize_{kind}')(artifact)
                        return TurnResult(formal_review=review)
                    renderer.text_delta('done', message_id='final-model')
                    renderer.rubric_evaluation_started('grading', 1, 3)
                    renderer.rubric_evaluation_finished({'grading_run_id': 'grading', 'iteration': 0, 'result': 'satisfied'}, 3)
                    return TurnResult(final_text='done', rubric_status='satisfied')

                with patch('core.application.session.synchronize_conversation', new=AsyncMock()):
                    if kind == 'plan':
                        await session.set_mode('plan')
                    else:
                        await session.begin_goal('Deliver the requested change')
                    await session.prompt('Deliver the requested change', turn_runner=runner)
                observed = [e for e in frontend.events if e.turn_id]
                self.assertEqual(len({e.turn_id for e in observed}), 1)
                artifacts = [e for e in observed if isinstance(e, ArtifactEvent)]
                self.assertTrue(all(e.event_id > 0 for e in artifacts))
                self.assertEqual(phases[-1], None)
                self.assertEqual(len(phases), 3)
                self.assertEqual(session.record['turns'], 1)
                self.assertEqual(len({e['turn_id'] for e in session.record['events']}), 1)
                await application.shutdown()

    async def test_new_attempts_after_error_and_cancel_have_distinct_outer_ids(self) -> None:
        frontend = RecordingFrontend()
        application, session = application_for(frontend, FakeAgent([]))

        async def fail(**kwargs):
            kwargs['renderer'].text_delta('partial')
            raise ValueError('provider failed')

        with self.assertRaisesRegex(ValueError, 'provider failed'):
            await session.prompt('first', turn_runner=fail)

        started = asyncio.Event()

        async def block(**kwargs):
            kwargs['renderer'].text_delta('partial')
            started.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(session.prompt('second', turn_runner=block))
        await started.wait()
        await session.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        users = [e for e in frontend.events if isinstance(e, MessageEvent) and e.phase == 'user']
        self.assertEqual(len({e.turn_id for e in users}), 2)
        lifecycle = [e for e in frontend.events if isinstance(e, RuntimeEvent) and e.kind == 'session']
        self.assertTrue(all(e.turn_id for e in lifecycle))
        self.assertEqual([e.turn_id for e in lifecycle if e.state == 'turn_started'], [e.turn_id for e in users])
        await application.shutdown()

    async def test_live_events_correlate_with_normalized_durable_events(self) -> None:
        frontend = RecordingFrontend()
        application, session = application_for(frontend, FakeAgent([
            FakeStream(messages=[StreamedMessage()], tools=[ToolCall()]),
        ]))
        await session.prompt('inspect')
        durable = {e['id']: e for e in normalize_events(session.record['events'])}
        observed = [e for e in frontend.events if isinstance(e, (MessageEvent, ToolEvent)) and e.phase in {'user', 'content', 'reasoning', 'start', 'completed_result'}]
        self.assertTrue(observed)
        for event in observed:
            saved = durable[event.event_id]
            self.assertEqual(saved['turn_id'], event.turn_id)
            self.assertEqual(saved.get('message_id', ''), event.message_id)
            self.assertEqual(tuple(saved.get('namespace', ())), event.namespace)
        content = [e for e in observed if isinstance(e, MessageEvent) and e.phase == 'content']
        self.assertEqual(content[0].event_id, content[1].event_id)
        result = next(e for e in observed if isinstance(e, ToolEvent) and e.phase == 'completed_result')
        call = next(e for e in observed if isinstance(e, ToolEvent) and e.phase == 'start')
        self.assertEqual(result.origin_event_id, call.event_id)
        await application.shutdown()

    async def test_distinct_native_messages_keep_distinct_durable_entries_in_one_turn(self) -> None:
        frontend = RecordingFrontend()
        first, second = StreamedMessage(), StreamedMessage()
        first.message_id, second.message_id = 'first-native', 'second-native'
        first.reasoning.items.clear()
        second.reasoning.items.clear()
        application, session = application_for(frontend, FakeAgent([FakeStream(messages=[first, second])]))
        await session.prompt('inspect')
        content = [e for e in frontend.events if isinstance(e, MessageEvent) and e.phase == 'content']
        ids = {native: {e.event_id for e in content if e.message_id == native}
               for native in ('first-native', 'second-native')}
        self.assertTrue(ids['first-native'].isdisjoint(ids['second-native']))
        self.assertEqual(len({e.turn_id for e in content}), 1)
        await application.shutdown()

    async def test_snapshot_nested_payloads_are_detached(self) -> None:
        application, session = application_for(RecordingFrontend(), FakeAgent([]))
        session.record['events'] = [{'id': 1, 'type': 'tool_call', 'args': {'nested': ['original']}}]
        before = deepcopy(session.record)
        snapshot = session.snapshot()
        snapshot.transcript[0]['args']['nested'].append('changed')
        snapshot.dashboard['tokens']['in'] = 999
        self.assertEqual(session.record, before)
        await application.shutdown()

    async def test_whitespace_removed_by_persistence_does_not_orphan_later_reasoning(self) -> None:
        class NormalizingStore(MemoryStore):
            def save(self, record):
                record['events'] = normalize_events(record.get('events'))
                super().save(record)

        frontend = RecordingFrontend()
        application, session = application_for(frontend, FakeAgent([FakeStream(messages=[StreamedMessage()])]))
        application.store = NormalizingStore()
        await session.prompt('inspect')
        visible = next(e for e in frontend.events if isinstance(e, MessageEvent) and e.text == 'visible thought')
        saved = next(e for e in session.record['events'] if e['type'] == 'reasoning')
        self.assertEqual(visible.event_id, saved['id'])
        await application.shutdown()

    async def test_individual_stopped_tool_outcome_crosses_typed_boundary(self) -> None:
        frontend = RecordingFrontend()
        record = session_record()
        clock = iter([10.0, 12.5])
        recorder = SessionEventEmitter(
            FrontendEmitter(frontend, session_id=record['id'], turn_id='outer'),
            SessionRecorder(record, MemoryStore(), 'action'),
            clock=lambda: next(clock),
        )
        recorder.tool_call('read_file', {'file_path': 'x'}, call_id='call')
        recorder.stop_active_tools('interrupted')
        stopped = [e for e in frontend.events if isinstance(e, ToolEvent) and e.phase == 'stop']
        self.assertEqual(len(stopped), 1)
        self.assertEqual(stopped[0].tool_call_id, 'call')
        self.assertEqual(stopped[0].duration_ms, 2500)
        self.assertEqual(stopped[0].status, 'interrupted')
        self.assertEqual(stopped[0].event_id, record['events'][0]['id'])

    async def test_idless_result_origin_does_not_bind_to_an_earlier_failed_turn(self) -> None:
        frontend = RecordingFrontend()
        record = session_record()
        for turn in ("failed", "new"):
            recorder = SessionEventEmitter(
                FrontendEmitter(frontend, session_id=record['id'], turn_id=turn),
                SessionRecorder(record, MemoryStore(), 'action', turn_id=turn),
            )
            recorder.tool_call('read_file', {'file_path': turn})
            if turn == 'failed':
                recorder.stop_active_tools('interrupted')
            else:
                recorder.completed_tool_result('read_file', 'new result')
        result = next(e for e in frontend.events if isinstance(e, ToolEvent) and e.phase == 'completed_result')
        call = next(e for e in record['events'] if e['type'] == 'tool_call' and e['turn_id'] == 'new')
        self.assertEqual(result.origin_event_id, call['id'])

    async def test_bootstrap_accepts_the_selected_frontend(self) -> None:
        from cli.commands import _bootstrap
        frontend = RecordingFrontend()
        application, session = application_for(frontend, FakeAgent([]))
        application.open_session = AsyncMock(return_value=session)
        with patch('core.application.MiraApplication.start', new=AsyncMock(return_value=application)) as start:
            await _bootstrap(application.workspace, None, False, {}, frontend=frontend)
        self.assertIs(start.await_args.kwargs['frontend'], frontend)


class AdapterCorrectnessTests(unittest.TestCase):
    def test_recovered_start_reaches_consumers_without_a_separate_recovery_renderer(self) -> None:
        calls = []
        RendererAdapter(SimpleNamespace(tool_call=lambda *a, **kw: calls.append(a))).emit(
            ToolEvent(phase="recovered_start", name="read_file", arguments={"file_path": "x"}, tool_call_id="call"),
        )
        self.assertEqual(calls, [("read_file", {"file_path": "x"})])

    def test_message_projection_retains_the_complete_typed_event(self) -> None:
        calls = []
        event = MessageEvent(phase='content', text='hello', session_id='s', turn_id='t', message_id='m', namespace=('node',), metadata={'provider': 'test'})
        RendererAdapter(SimpleNamespace(text_delta=lambda *a, **kw: calls.append(kw))).emit(event)
        self.assertIs(calls[0]['frontend_event'], event)

    def test_session_notifications_are_delivered(self) -> None:
        events = []
        event = RuntimeEvent(kind='session', state='ready', session_id='s', turn_id='t')
        RendererAdapter(SimpleNamespace(session_state_changed=events.append)).emit(event)
        self.assertEqual(events, [event])

    def test_error_phase_is_not_projected_as_success(self) -> None:
        calls = []
        renderer = SimpleNamespace(tool_error=lambda *a, **kw: calls.append('error'), tool_result=lambda *a, **kw: calls.append('success'))
        RendererAdapter(renderer).emit(ToolEvent(phase='error', name='read_file', result='failed'))
        self.assertEqual(calls, ['error'])

    def test_historical_events_do_not_acquire_invented_turn_identity(self) -> None:
        event = normalize_events([{'id': 1, 'type': 'assistant', 'text': 'old'}])[0]
        self.assertNotIn('turn_id', event)
        self.assertNotIn('message_id', event)

    def test_inspection_capture_still_requires_a_hidden_frontend_capability(self) -> None:
        # Evidence for the next ownership slice, not a new public capability.
        frontend = RecordingFrontend()
        emitter = FrontendEmitter(frontend)
        self.assertIsNone(live_inspection_store(emitter))
        frontend.live_inspections = LiveInspectionStore()
        self.assertIs(live_inspection_store(emitter), frontend.live_inspections)


class FrontendExampleTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_example_answers_mcp_form_and_url_elicitations(self) -> None:
        from mira.api import MCPElicitationRequest

        example = runpy.run_path(str(ROOT / 'examples/mira_api/full_frontend.py'))
        frontend = example['FullFrontend']()
        request = MCPElicitationRequest({'requests': [
            {'key': 'form', 'mode': 'form', 'requested_schema': {'type': 'object'}},
            {'key': 'url', 'mode': 'url', 'url': 'https://example.test/consent'},
        ]})
        with patch('asyncio.to_thread', new=AsyncMock(side_effect=['accept', '{"name":"test"}', 'decline'])), redirect_stdout(io.StringIO()):
            response = await frontend.request(request)
        self.assertEqual(response, {'responses': {
            'form': {'action': 'accept', 'content': {'name': 'test'}},
            'url': {'action': 'decline'},
        }})

    async def test_full_example_preserves_mira_approval_decisions(self) -> None:
        from mira.api import ApprovalRequest

        example = runpy.run_path(str(ROOT / 'examples/mira_api/full_frontend.py'))
        frontend = example['FullFrontend']()
        request = ApprovalRequest(({'action_requests': [
            {'name': 'read_file', 'args': {'file_path': 'x'}} for _ in range(4)
        ]},))
        answers = ['allow_once', 'allow_always', 'edit', '{"name":"read_file","args":{"file_path":"y"}}', '']
        with patch('asyncio.to_thread', new=AsyncMock(side_effect=answers)), redirect_stdout(io.StringIO()):
            response = await frontend.request(request)
        self.assertEqual([item['type'] for item in response], ['allow_once', 'allow_always', 'edit', 'reject_once'])
        self.assertEqual(response[2]['edited_action']['args'], {'file_path': 'y'})
