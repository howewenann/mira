"""Mounted Textual checks for the public-event grouped transcript."""

import unittest
import asyncio
from io import StringIO
from unittest.mock import AsyncMock, Mock
from rich.console import Console

from mira.api import MessageEvent, RuntimeEvent, ToolEvent, SubagentEvent, RubricEvent, AskUserRequest, ApprovalRequest, MCPElicitationRequest
from tests.test_textual_app import make_app, wait_until, plan_artifact, goal_artifact
from ui.textual.widgets.assistant_turn import AssistantTurn, ActivityRow
from ui.textual.widgets.chat_log import ChatLog, AssistantBubble


async def settle(app, pilot):
    await wait_until(lambda: all(turn._composed and not turn._moving and
        all((row.widget.parent is turn.details if row is turn.selected else row.widget.parent is None)
            if row.completed and not row.attention else row.widget.parent is turn.active
            for row in turn.rows.values()) for turn in app.query(AssistantTurn)), timeout=5)
    await pilot.pause()


def rendered(widget, width=100):
    if isinstance(widget, ActivityRow):
        return ' '.join(str(child.content) for child in (widget.glyph, widget.label, widget.preview, widget.duration))
    output = StringIO()
    Console(file=output, width=width, color_system=None).print(widget.content)
    return output.getvalue()


class GroupedTranscriptTests(unittest.IsolatedAsyncioTestCase):
    async def test_workflow_completion_is_counted_and_failed_rubric_stays_visible(self):
        app = make_app()
        async with app.run_test(size=(110, 44)) as pilot:
            frontend = app.application.frontend
            identity = dict(session_id=app.core_session.id, turn_id='categories')
            frontend.emit(RuntimeEvent(kind='session', state='turn_started', **identity))
            chat = app.query_one(ChatLog)
            workflow = chat.workflow_started('workflow', 'Native workflow')
            await settle(app, pilot)
            turn = chat.query_one(AssistantTurn)
            self.assertIs(workflow.parent, turn.active)
            self.assertEqual(turn.completed_counts['Workflows'], 0)
            chat.workflow_finished('workflow', {'result': 'complete'}, available=True)
            frontend.emit(RubricEvent(phase='lifecycle', lifecycle={
                'type': 'rubric_verification_end', 'grading_run_id': 'failed', 'iteration': 0,
                'succeeded': False, 'duration_ms': 1200}, **identity))
            await settle(app, pilot)
            self.assertEqual(turn.completed_counts['Workflows'], 1)
            self.assertIsNone(workflow.parent)
            rubric = next(row for row in turn.rows.values() if row.category == 'Rubric')
            self.assertTrue(rubric.attention)
            self.assertIs(rubric.widget.parent, turn.active)
            turn.toggle_history()
            turn.select_activity(turn.rows[workflow])
            await settle(app, pilot)
            self.assertIs(workflow.parent, turn.details)
            self.assertTrue(list(workflow.query('Tree')))
            self.assertIs(workflow.final_button.parent, workflow.footer)
            self.assertTrue(workflow.final_button.display)
            chat.clear_log()
            await pilot.pause()
            chat.restore_session({'events': [{'type': 'workflow', 'command': '/workflow saved', 'workflow_id': 'saved',
                'workflow_name': 'Saved workflow', 'status': 'DONE', 'turn_id': 'saved-workflow',
                'duration_ms': 2000, 'final_state_available': True, 'final_state': {'result': 'complete'}}]})
            await settle(app, pilot)
            restored = chat.query_one(AssistantTurn)
            self.assertEqual(restored.completed_counts['Workflows'], 1)
            restored.toggle_history()
            restored.select_activity(restored.completed_rows[0])
            await settle(app, pilot)
            self.assertEqual(restored.completed_rows[0].duration.content, '00:02')

    async def test_final_response_has_exactly_one_blank_row_in_both_display_modes(self):
        app = make_app()
        async with app.run_test(size=(110, 44)) as pilot:
            chat = app.query_one(ChatLog)
            chat.restore_session({'events': [
                {'type': 'reasoning', 'text': 'Full thinking detail.', 'turn_id': 'spacing'},
                {'type': 'tool_call', 'name': 'read_file', 'args': {'file_path': '/README.md'}, 'call_id': 'c', 'turn_id': 'spacing'},
                {'type': 'tool_result', 'name': 'read_file', 'output': 'Full result.', 'call_id': 'c', 'turn_id': 'spacing'},
                {'type': 'assistant', 'text': '**Final** response.', 'turn_id': 'spacing'},
            ]})
            await settle(app, pilot)
            turn = chat.query_one(AssistantTurn)
            final = turn.candidate.widget
            self.assertEqual(final._body_switcher.current, 'assistant-text')
            for expanded in (False, True):
                if expanded:
                    turn.toggle_history()
                for selected in ([None, *turn.completed_rows] if expanded else [None]):
                    if selected is not None:
                        turn.select_activity(selected)
                    await settle(app, pilot)
                    for _ in range(2):
                        preceding = turn.details if turn.selected else turn.history if expanded else turn.summary
                        self.assertEqual(final.region.y - preceding.region.bottom, 1)
                        final.toggle_display()
                        await pilot.pause()
            chat.clear_log()
            await pilot.pause()
            chat.restore_session({'events': [{'type': 'reasoning', 'text': 'No response', 'turn_id': 'empty'}]})
            await settle(app, pilot)
            self.assertFalse(list(chat.query('.turn-final-response')))

    async def test_native_list_cursor_mouse_and_parallel_arrival_preserve_reading_position(self):
        app = make_app()
        async with app.run_test(size=(110, 40)) as pilot:
            frontend = app.application.frontend
            identity = dict(session_id=app.core_session.id, turn_id='reading')
            frontend.emit(RuntimeEvent(kind='session', state='turn_started', **identity))
            frontend.emit(ToolEvent(phase='start', name='read_file', arguments={'file_path': '/slow'}, tool_call_id='slow', **identity))
            for i in range(50):
                frontend.emit(MessageEvent(phase='reasoning', text=f'Thinking {i}', message_id=f'm-{i}', **identity))
                frontend.emit(MessageEvent(phase='end', message_id=f'm-{i}', **identity))
            await settle(app, pilot)
            chat = app.query_one(ChatLog)
            turn = chat.query_one(AssistantTurn)
            self.assertFalse(turn.history.children)
            self.assertEqual(len(turn.active.children), 1)
            turn.toggle_history()
            await settle(app, pilot)
            row = turn.completed_rows[25]
            chat.scroll_to_widget(row, top=True, animate=False, immediate=True)
            app.action_focus_prompt()
            await pilot.pause()
            position = chat.scroll_y
            await pilot.click(row.label)
            await settle(app, pilot)
            self.assertIs(turn.selected, row)
            self.assertTrue(turn.history.has_focus)
            self.assertEqual(chat.scroll_y, position)
            frontend.emit(ToolEvent(phase='completed_result', name='read_file', result='read complete', tool_call_id='slow', duration_ms=1000, **identity))
            await settle(app, pilot)
            self.assertIs(turn.history.highlighted_child, row)
            self.assertIs(turn.selected, row)
            self.assertEqual(chat.scroll_y, position)
            self.assertEqual(turn.history.region.height, 103)
            await pilot.press(*(['down'] * 40))
            await pilot.pause()
            self.assertTrue(chat.content_region.overlaps(turn.history.highlighted_child.region))
            self.assertEqual(turn.history.scroll_y, 0)
            self.assertEqual(turn.history.max_scroll_y, 0)

    async def test_natural_history_height_at_five_twenty_and_fifty_rows(self):
        for width in (60, 150):
            for count in (5, 20, 50):
                with self.subTest(width=width, count=count):
                    app = make_app()
                    async with app.run_test(size=(width, 40)) as pilot:
                        chat = app.query_one(ChatLog)
                        events = [{"type": "reasoning", "text": f"thought {index}", "turn_id": "layout"}
                                  for index in range(count)]
                        events.append({"type": "assistant", "text": "Final response", "turn_id": "layout"})
                        chat.restore_session({"events": events})
                        await settle(app, pilot)
                        turn = chat.query_one(AssistantTurn)
                        self.assertEqual(turn.summary.size.height, 2)
                        self.assertFalse(turn.history.display)
                        self.assertNotIn("00:", rendered(turn.summary))
                        turn.toggle_history()
                        await settle(app, pilot)
                        self.assertEqual(turn.history.size.height, 2 * count + 1)
                        self.assertTrue(all(row.region.height == 1 for row in turn.completed_rows))
                        self.assertFalse(turn.history.allow_vertical_scroll)
                        self.assertGreater(chat.max_scroll_y, 0)
                        chat.scroll_to_widget(turn.completed_rows[-1], animate=False)
                        await pilot.pause()
                        self.assertTrue(chat.content_region.overlaps(turn.completed_rows[-1].region))

    async def test_large_selected_details_are_full_and_below_all_rows(self):
        app = make_app()
        thinking = "\n".join(f"Thought line {index}" for index in range(400))
        command = "python -m pytest " + "tests/long_path/" * 160 + "\n--verbose\n--capture=no"
        output = "\n".join(f"Output line {index}" for index in range(350)) + "\nOUTPUT_END"
        async with app.run_test(size=(100, 40)) as pilot:
            chat = app.query_one(ChatLog)
            chat.restore_session({"events": [
                {"type": "reasoning", "text": thinking, "turn_id": "details"},
                {"type": "tool_call", "name": "execute", "args": {"command": command}, "call_id": "long", "turn_id": "details"},
                {"type": "tool_result", "name": "execute", "output": output, "call_id": "long", "duration_ms": 8000, "turn_id": "details"},
                {"type": "assistant", "text": "Final response", "turn_id": "details"},
            ]})
            await settle(app, pilot)
            turn = chat.query_one(AssistantTurn)
            turn.toggle_history()
            thought, tool = turn.completed_rows
            turn.select_activity(thought)
            await settle(app, pilot)
            self.assertGreaterEqual(thought.widget.size.height, 400)
            self.assertGreaterEqual(turn.details.region.y, turn.history.region.bottom)
            self.assertFalse(turn.details.allow_vertical_scroll)
            chat.scroll_to_widget(tool, animate=False)
            await pilot.pause()
            turn.history.index = list(turn.history.children).index(tool)
            turn.history.focus(scroll_visible=False)
            await pilot.press('enter')
            await settle(app, pilot)
            self.assertIsNone(thought.widget.parent)
            self.assertIs(tool.widget.parent, turn.details)
            self.assertEqual([row.region.height for row in turn.completed_rows], [1, 1])
            self.assertEqual(tool.widget.args_editor.text.count('--verbose'), 1)
            self.assertEqual(tool.widget.output.content.plain, output)
            self.assertTrue(tool.widget.args_collapsible.collapsed)
            self.assertTrue(tool.widget.output_collapsible.collapsed)
            tool.widget.args_collapsible.collapsed = False
            tool.widget.output_collapsible.collapsed = False
            await settle(app, pilot)
            await wait_until(lambda: tool.widget.args_editor.size.height > 12)
            self.assertGreaterEqual(tool.widget.output.size.height, 351)
            self.assertGreater(tool.widget.args_editor.size.height, 12)
            self.assertFalse(tool.widget.args_editor.allow_vertical_scroll)
            for width in (60, 150):
                await pilot.resize_terminal(width, 40)
                await settle(app, pilot)
                self.assertEqual(tool.content_size.height, 1)
                duration = tool.duration.render_lines(tool.duration.size.region)[0].text
                self.assertEqual(duration.strip(), '00:08')
                self.assertEqual(tool.duration.region.right, tool.content_region.right)
                preview = ''.join(strip.text for strip in tool.preview.render_lines(tool.preview.region.at_offset((0, 0))))
                self.assertIn('…', preview)
            turn.close_button.press()
            await settle(app, pilot)
            self.assertIsNone(turn.selected)
            self.assertFalse(turn.details.display)
            turn.select_activity(thought)
            await settle(app, pilot)
            turn.toggle_history()
            await settle(app, pilot)
            self.assertIsNone(turn.selected)
            self.assertIsNone(thought.widget.parent)

    async def test_row_glyph_and_geometry_stay_fixed_during_hover_focus_and_selection(self):
        app = make_app()
        async with app.run_test(size=(110, 40)) as pilot:
            chat = app.query_one(ChatLog)
            chat.restore_session({"events": [
                {"type": "tool_call", "name": "read_file", "args": {"file_path": "/README.md"}, "call_id": "c", "turn_id": "glyph"},
                {"type": "tool_result", "name": "read_file", "output": "result", "call_id": "c", "duration_ms": 1000, "turn_id": "glyph"},
                {"type": "assistant", "text": "Final", "turn_id": "glyph"},
            ]})
            await settle(app, pilot)
            turn = chat.query_one(AssistantTurn)
            chat.scroll_to_widget(turn.summary, animate=False)
            await pilot.pause()
            await pilot.click('.turn-summary', offset=(20, 1))
            await settle(app, pilot)
            row = turn.completed_rows[0]
            region = row.region
            def glyph():
                strip = row.glyph.render_lines(row.glyph.size.region)[0]
                text = strip.text
                return text.index('▾' if row.selected else '▸'), text
            initial_x, initial = glyph()
            for focus in (False, True):
                if focus:
                    turn.history.index = 0
                    turn.history.focus(scroll_visible=False)
                await pilot.hover('.activity-row', offset=(4, 0))
                await pilot.pause()
                self.assertEqual(row.region, region)
                self.assertEqual(glyph()[0], initial_x)
                self.assertFalse(row.styles.text_style.bold)
                self.assertFalse(row.styles.text_style.reverse)
            await pilot.press('enter')
            await settle(app, pilot)
            self.assertTrue(row.selected)
            self.assertEqual(row.region, region)
            self.assertEqual(glyph()[0], initial_x)
            self.assertEqual(glyph()[1].replace('▾', '▸'), initial)
            tool = row.widget
            title = tool.args_collapsible.query_one('CollapsibleTitle')
            chat.scroll_to_widget(title, animate=False)
            await pilot.pause()
            before = title.region
            title.focus(scroll_visible=False)
            await pilot.hover(title)
            await pilot.pause()
            self.assertEqual(title.region, before)
            self.assertFalse(title.styles.text_style.bold)
            self.assertFalse(title.styles.text_style.reverse)
            await pilot.press('enter')
            await pilot.pause()
            self.assertEqual(title.region.size, before.size)

    async def test_turn_clock_starts_before_model_output_and_user_stays_outside(self):
        app = make_app()
        async with app.run_test(size=(110, 40)) as pilot:
            frontend = app.application.frontend
            identity = dict(session_id=app.core_session.id, turn_id='starting')
            frontend.emit(RuntimeEvent(kind='session', state='turn_started', **identity))
            await settle(app, pilot)
            chat = app.query_one(ChatLog)
            turn = chat.query_one(AssistantTurn)
            self.assertGreater(turn.region.height, 0)
            self.assertIn('Working', rendered(turn.summary))
            frontend.emit(MessageEvent(phase='user', text='question', **identity))
            await settle(app, pilot)
            user = chat.query_one('.user')
            self.assertIs(user.parent, chat)
            self.assertLess(list(chat.children).index(user), list(chat.children).index(turn))

    async def test_archived_message_keeps_markdown_mode_and_copy_feedback(self):
        app = make_app()
        async with app.run_test(size=(120, 44)) as pilot:
            frontend = app.application.frontend
            identity = dict(session_id=app.core_session.id, turn_id='display')
            frontend.emit(RuntimeEvent(kind='session', state='turn_started', **identity))
            frontend.emit(MessageEvent(phase='user', text='work', **identity))
            frontend.emit(MessageEvent(phase='content', text='**Intermediate**', message_id='first', **identity))
            frontend.emit(MessageEvent(phase='end', message_id='first', **identity))
            await settle(app, pilot)
            turn = app.query_one(AssistantTurn)
            message = turn.candidate.widget
            message.FEEDBACK_SECONDS = 0.3
            await wait_until(lambda: message._body_switcher.current == 'assistant-text')
            message.toggle_display()
            self.assertEqual(message._body_switcher.current, 'assistant-markdown')
            message.show_copied()
            frontend.emit(ToolEvent(phase='start', name='read_file', arguments={}, tool_call_id='call', **identity))
            await settle(app, pilot)
            self.assertIsNone(message.parent)
            turn.toggle_history()
            turn.select_activity(turn.rows[message])
            await settle(app, pilot)
            self.assertEqual(message._body_switcher.current, 'assistant-markdown')
            await asyncio.sleep(0.35)
            await pilot.pause()
            self.assertEqual(str(message.query_one('.assistant-copy').label), 'Copy')

    async def test_formal_controls_stay_outside_history_until_resolved(self):
        app = make_app()
        async with app.run_test(size=(120, 44)) as pilot:
            frontend = app.application.frontend
            identity = dict(session_id=app.core_session.id, turn_id="formal")
            frontend.emit(RuntimeEvent(kind="session", state="turn_started", **identity))
            frontend.emit(MessageEvent(phase="user", text="formal work", **identity))
            chat = app.query_one(ChatLog)
            plan = plan_artifact(plan_id='group-plan', title='Plan', objective='Build',
                context_and_constraints='Existing behavior', key_changes=['Build'], test_plan=['Check'],
                assumptions=['Existing setup'], success_criteria='- Done', rubric_enabled=False, rubric_iterations=1)
            chat.present_plan(plan, active=True)
            await settle(app, pilot)
            control = chat.query_one('#plan-implement-group-plan')
            await wait_until(lambda: control.has_focus)
            turn = chat.query_one(AssistantTurn)
            self.assertIn(turn.active, control.ancestors)
            self.assertEqual(turn.completed_counts['Plans'], 0)
            chat.resolve_plan('group-plan', 'completed')
            await settle(app, pilot)
            self.assertEqual(turn.completed_counts['Plans'], 1)
            goal = goal_artifact(goal_id='group-goal', title='Goal', objective='Build',
                success_criteria='- Done', rubric_enabled=False, rubric_iterations=1)
            chat.present_goal(goal, active=True)
            await settle(app, pilot)
            control = chat.query_one('#goal-implement-group-goal')
            await wait_until(lambda: control.has_focus)
            frontend.emit(RuntimeEvent(kind='session', state='ready', **identity))
            await settle(app, pilot)
            self.assertIn(turn.active, control.ancestors)
            self.assertEqual(turn.completed_counts['Goals'], 0)

    async def test_waiting_controls_errors_and_final_after_rubric_remain_visible(self):
        app = make_app()
        async with app.run_test(size=(120, 44)) as pilot:
            frontend = app.application.frontend
            identity = dict(session_id=app.core_session.id, turn_id="attention")
            frontend.emit(RuntimeEvent(kind="session", state="turn_started", **identity))
            frontend.emit(MessageEvent(phase="user", text="work", **identity))
            frontend.emit(ToolEvent(phase="start", name="execute", arguments={}, tool_call_id="bad", **identity))
            frontend.emit(ToolEvent(phase="error", name="execute", result="permission denied", tool_call_id="bad", **identity))
            frontend.emit(MessageEvent(phase="content", text="Final response", message_id="answer", **identity))
            frontend.emit(MessageEvent(phase="end", message_id="answer", **identity))
            for phase in ("rubric_verification_start", "rubric_verification_end"):
                frontend.emit(RubricEvent(phase="lifecycle", lifecycle={"type": phase, "grading_run_id": "run", "iteration": 0, "succeeded": True}, **identity))
            await settle(app, pilot)
            turn = app.query_one(AssistantTurn)
            async def answer(*args):
                self.assertIs(turn.candidate.widget.parent, turn.active)
                self.assertEqual(turn.completed_counts['Messages'], 0)
                self.assertEqual(turn.state, "waiting")
                self.assertIn("Waiting for you", rendered(turn.summary))
                return "answered"
            for request, method in ((AskUserRequest({}), "ask_user"), (ApprovalRequest(()), "ask_approvals"), (MCPElicitationRequest({}), "answer_mcp_elicitation")):
                setattr(app, method, AsyncMock(side_effect=answer))
                self.assertEqual(await frontend.request(request), "answered")
                self.assertEqual(turn.state, "running")
            frontend.emit(RuntimeEvent(kind="session", state="ready", **identity))
            await settle(app, pilot)
            self.assertEqual(turn.completed_counts["Messages"], 0)
            self.assertEqual(turn.completed_counts["Tools"], 1)
            self.assertIn(turn.active, turn.candidate.widget.ancestors)
            self.assertIn(turn.active, app.query_one(ChatLog)._tool_blocks["id:bad"]["widget"].ancestors)

    async def test_legacy_interleaving_and_eval_history_use_only_durable_relationships(self):
        app = make_app()
        async with app.run_test(size=(110, 40)) as pilot:
            chat = app.query_one(ChatLog)
            chat.restore_session({"events": [
                {"type": "assistant", "text": "first", "turn_id": "ambiguous"},
                {"type": "info", "text": "old unassigned entry"},
                {"type": "assistant", "text": "last", "turn_id": "ambiguous"},
            ]})
            await pilot.pause()
            self.assertFalse(list(chat.query(AssistantTurn)))
            chat.clear_log()
            run = {"id": "saved-child", "inspection_id": "subagent:row", "inspection_type": "subagent",
                   "origin_tool": "eval", "origin_event_id": 4, "row_id": "row", "display_name": "worker",
                   "status": "DONE", "task": "check", "output": "done", "duration_ms": 40}
            chat.subagent_runs_provider = lambda: [run]
            chat.restore_session({"events": [
                {"type": "tool_call", "name": "eval", "args": {}, "call_id": "eval-call", "id": 4, "turn_id": "saved"},
                {"type": "tool_result", "name": "eval", "output": "done", "call_id": "eval-call", "id": 5, "turn_id": "saved"},
                {"type": "assistant", "text": "answer", "id": 6, "turn_id": "saved"},
            ]})
            await settle(app, pilot)
            turn = chat.query_one(AssistantTurn)
            self.assertEqual(turn.completed_counts["Subagents"], 1)
            self.assertTrue(all('\\[' not in rendered(row) for row in turn.completed_rows))
            self.assertFalse(turn.inspect_button.is_mounted)
            turn.toggle_history()
            child = next(row for row in turn.completed_rows if row.category == 'Subagents')
            turn.select_activity(child)
            await settle(app, pilot)
            self.assertTrue(turn.inspect_button.is_mounted)
            chat.scroll_to_widget(turn.inspect_button, animate=False)
            await pilot.pause()
            self.assertGreater(turn.inspect_button.region.height, 0)

    async def test_parallel_tools_children_rubric_and_recovery_keep_identity(self):
        app = make_app()
        async with app.run_test(size=(120, 44)) as pilot:
            frontend = app.application.frontend
            identity = dict(session_id=app.core_session.id, turn_id="parallel")
            frontend.emit(RuntimeEvent(kind="session", state="turn_started", **identity))
            frontend.emit(MessageEvent(phase="user", text="parallel work", **identity))
            for call in ("one", "two"):
                frontend.emit(ToolEvent(phase="start", name="read_file", arguments={}, tool_call_id=call, **identity))
                frontend.emit(SubagentEvent(phase="eval_start", name="worker", row_id="shared", eval_id=call,
                    metadata={"inspection_id": f"child-{call}"}, **identity))
            await settle(app, pilot)
            chat = app.query_one(ChatLog)
            turn = chat.query_one(AssistantTurn)
            self.assertEqual(len(turn.active.children), 4)
            self.assertTrue(all(widget.region.height > 0 for widget in turn.active.children))
            self.assertFalse(turn.completed_counts["Tools"])
            frontend.emit(ToolEvent(phase="result", name="read_file", result="ok", tool_call_id="two", **identity))
            frontend.emit(SubagentEvent(phase="eval_finish", name="worker", row_id="shared", eval_id="two", result="done",
                metadata={"inspection_id": "child-two"}, **identity))
            await settle(app, pilot)
            self.assertEqual(turn.completed_counts["Tools"], 1)
            self.assertEqual(turn.completed_counts["Subagents"], 1)
            self.assertEqual(len(turn.active.children), 2)
            frontend.emit(ToolEvent(phase="stop", name="read_file", tool_call_id="one", status="interrupted", duration_ms=20, **identity))
            frontend.emit(RuntimeEvent(kind="session", state="ready", **identity))
            await settle(app, pilot)
            original = chat._tool_blocks["id:one"]["widget"]
            frontend.emit(RuntimeEvent(kind="session", state="turn_started", **identity))
            frontend.emit(ToolEvent(phase="recovered_start", name="read_file", tool_call_id="one", arguments={}, **identity))
            await settle(app, pilot)
            self.assertIs(chat._tool_blocks["id:one"]["widget"], original)
            self.assertEqual(len(chat.query(AssistantTurn)), 1)
            self.assertEqual(turn.completed_counts["Tools"], 1)
            self.assertIn(turn.active, original.ancestors)
            for phase in ("rubric_verification_start", "rubric_verification_end", "rubric_grading_start", "rubric_grading_end"):
                frontend.emit(RubricEvent(phase="lifecycle", lifecycle={"type": phase, "grading_run_id": "grade", "iteration": 0, "succeeded": True}, **identity))
            await settle(app, pilot)
            self.assertEqual(turn.completed_counts["Rubric"], 2)

    async def test_live_phases_share_one_container_and_final_stays_visible(self):
        app = make_app()
        async with app.run_test(size=(110, 40)) as pilot:
            frontend = app.application.frontend
            identity = dict(session_id=app.core_session.id, turn_id="outer")
            frontend.emit(RuntimeEvent(kind="session", state="turn_started", **identity))
            frontend.emit(MessageEvent(phase="user", text="question", **identity))
            frontend.emit(MessageEvent(phase="reasoning", text="thinking", message_id="one", **identity))
            frontend.emit(MessageEvent(phase="content", text="I will check.", message_id="one", **identity))
            frontend.emit(MessageEvent(phase="end", message_id="one", **identity))
            frontend.emit(ToolEvent(phase="arguments_delta", name="read_file", arguments={"file_path": "a"}, tool_call_id="call", **identity))
            frontend.emit(ToolEvent(phase="start", name="read_file", arguments={"file_path": "a"}, tool_call_id="call", **identity))
            await settle(app, pilot)
            chat = app.query_one(ChatLog)
            turn = app.query_one(AssistantTurn)
            self.assertEqual(turn.completed_counts["Thinking"], 1)
            self.assertEqual(turn.completed_counts["Messages"], 1)
            self.assertEqual(turn.completed_counts["Tools"], 0)
            self.assertFalse(turn.expanded)
            self.assertEqual(len(chat._tool_blocks), 1)
            frontend.emit(ToolEvent(phase="result", name="read_file", result="data", tool_call_id="call", duration_ms=30, **identity))
            frontend.emit(MessageEvent(phase="content", text="Final answer", message_id="two", **identity))
            frontend.emit(MessageEvent(phase="end", message_id="two", **identity))
            frontend.emit(RuntimeEvent(kind="session", state="ready", **identity))
            await settle(app, pilot)
            self.assertEqual(len(chat.query(AssistantTurn)), 1)
            self.assertEqual(turn.completed_counts["Tools"], 1)
            self.assertEqual(turn.candidate.widget.text, "Final answer")
            self.assertIn(turn.active, turn.candidate.widget.ancestors)
            self.assertTrue(turn.candidate.widget.visible)
            self.assertIn("Worked", rendered(turn.summary))
            self.assertEqual(len(turn.completed_rows), 3)
            self.assertFalse(list(turn.history.children))
            self.assertGreater(turn.candidate.widget.region.height, 0)
            app.copy_to_clipboard = Mock()
            turn.candidate.widget.query_one('.assistant-copy').press()
            await pilot.pause()
            app.copy_to_clipboard.assert_called_once_with('Final answer')
            frontend.emit(MessageEvent(phase="content", text="another session", session_id="foreign", turn_id="foreign"))
            self.assertEqual(len(chat.query(AssistantTurn)), 1)

    async def test_history_retains_all_rows_and_legacy_records_stay_flat(self):
        app = make_app()
        async with app.run_test(size=(110, 40)) as pilot:
            chat = app.query_one(ChatLog)
            events = [{"type": "user", "text": "old question"},
                      {"type": "assistant", "text": "legacy answer"}]
            for index in range(15):
                events.append({"type": "reasoning", "text": f"thought {index}", "turn_id": "saved", "id": index+1})
            events.append({"type": "assistant", "text": "saved answer", "turn_id": "saved", "id": 20})
            chat.restore_session({"events": events})
            await settle(app, pilot)
            turn = chat.query_one(AssistantTurn)
            self.assertEqual(turn.completed_counts["Thinking"], 15)
            self.assertFalse(turn.expanded)
            self.assertEqual(len(chat.query(AssistantBubble)), 2)
            title = turn.summary
            before = title.region
            focused = app.focused
            chat.scroll_to_widget(title, animate=False)
            await pilot.hover('.turn-summary')
            await pilot.pause()
            self.assertEqual(title.region.size, before.size)
            self.assertEqual(title.region.x, before.x)
            self.assertIs(app.focused, focused)
            turn.toggle_history()
            await settle(app, pilot)
            self.assertEqual(title.region.height, before.height)
            self.assertEqual(len(turn.history.query(ActivityRow)), 15)
            self.assertEqual(turn.history.size.height, 31)
            chat.scroll_home(animate=False, immediate=True)
            await settle(app, pilot)
            position = chat.scroll_y
            rows = list(turn.history.query(ActivityRow))
            turn.select_activity(rows[0])
            await settle(app, pilot)
            self.assertTrue(rows[0].selected)
            self.assertFalse(rows[1].selected)
            self.assertEqual(chat.scroll_y, position)
            self.assertFalse(chat.follow_tail)
            frontend = app.application.frontend
            identity = dict(session_id=app.core_session.id, turn_id="saved")
            frontend.emit(RuntimeEvent(kind="session", state="turn_started", **identity))
            frontend.emit(ToolEvent(phase="start", name="read_file", tool_call_id="new", arguments={}, **identity))
            frontend.emit(ToolEvent(phase="result", name="read_file", tool_call_id="new", result="done", **identity))
            await settle(app, pilot)
            self.assertTrue(turn.expanded)
            self.assertTrue(rows[0].selected)
            self.assertEqual(chat.scroll_y, position)
            self.assertIn('since recovery', rendered(turn.summary))


if __name__ == "__main__":
    unittest.main()
