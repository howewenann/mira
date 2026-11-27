"""Mounted Textual checks for the public-event grouped transcript."""

import unittest
import asyncio
from unittest.mock import AsyncMock, Mock

from mira.api import MessageEvent, RuntimeEvent, ToolEvent, SubagentEvent, RubricEvent, AskUserRequest, ApprovalRequest, MCPElicitationRequest
from tests.test_textual_app import make_app, wait_until, plan_artifact, goal_artifact
from ui.textual.widgets.assistant_turn import AssistantTurn, ActivityRow
from ui.textual.widgets.chat_log import ChatLog, AssistantBubble


async def settle(app, pilot):
    await wait_until(lambda: all(turn._composed and not turn._moving and
        all(widget.is_mounted and all(child.is_mounted for child in widget.query('*'))
            for widget in turn.rows) for turn in app.query(AssistantTurn)), timeout=5)
    await pilot.pause()


class GroupedTranscriptTests(unittest.IsolatedAsyncioTestCase):
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
            self.assertIn('Working', str(turn.summary.content))
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
                self.assertEqual(turn.state, "waiting")
                self.assertIn("Waiting for you", str(turn.summary.content))
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
            self.assertTrue(all('\\[' not in str(row.title) for row in turn.completed_rows))
            self.assertEqual(len(turn.query(".rubric-inspect")), 1)
            turn.history.collapsed = False
            child = next(row for row in turn.completed_rows if row.category == 'Subagents')
            child.collapsed = False
            await settle(app, pilot)
            turn.viewport.scroll_end(animate=False, force=True, immediate=True)
            await pilot.pause()
            self.assertGreater(child.query_one('.rubric-inspect').region.height, 0)

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
            self.assertEqual(len(turn.active.query(ActivityRow)), 4)
            self.assertTrue(all(row.widget.region.height > 0 for row in turn.active.query(ActivityRow)))
            self.assertFalse(turn.completed_counts["Tools"])
            frontend.emit(ToolEvent(phase="result", name="read_file", result="ok", tool_call_id="two", **identity))
            frontend.emit(SubagentEvent(phase="eval_finish", name="worker", row_id="shared", eval_id="two", result="done",
                metadata={"inspection_id": "child-two"}, **identity))
            await settle(app, pilot)
            self.assertEqual(turn.completed_counts["Tools"], 1)
            self.assertEqual(turn.completed_counts["Subagents"], 1)
            self.assertEqual(len(turn.active.query(ActivityRow)), 2)
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
            self.assertTrue(turn.history.collapsed)
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
            self.assertIn("Worked for", str(turn.summary.content))
            self.assertEqual(len(turn.viewport.query(ActivityRow)), 3)
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
            self.assertTrue(turn.history.collapsed)
            self.assertEqual(len(chat.query(AssistantBubble)), 2)
            title = turn.history.query_one('CollapsibleTitle')
            before = title.region
            focused = app.focused
            await pilot.hover('.completed-activity > CollapsibleTitle')
            await pilot.pause()
            self.assertEqual(title.region.size, before.size)
            self.assertEqual(title.region.x, before.x)
            self.assertIs(app.focused, focused)
            turn.history.collapsed = False
            await settle(app, pilot)
            self.assertEqual(title.region.height, before.height)
            self.assertEqual(len(turn.viewport.query(ActivityRow)), 15)
            self.assertEqual(turn.viewport.size.height, 5)
            self.assertTrue(turn.viewport.is_vertical_scroll_end)
            turn.viewport.scroll_home(animate=False, immediate=True)
            await settle(app, pilot)
            position = turn.viewport.scroll_y
            rows = list(turn.viewport.query(ActivityRow))
            rows[0].collapsed = False
            await settle(app, pilot)
            self.assertFalse(rows[0].collapsed)
            self.assertTrue(rows[1].collapsed)
            self.assertEqual(turn.viewport.scroll_y, position)
            self.assertFalse(chat.follow_tail)
            frontend = app.application.frontend
            identity = dict(session_id=app.core_session.id, turn_id="saved")
            frontend.emit(RuntimeEvent(kind="session", state="turn_started", **identity))
            frontend.emit(ToolEvent(phase="start", name="read_file", tool_call_id="new", arguments={}, **identity))
            frontend.emit(ToolEvent(phase="result", name="read_file", tool_call_id="new", result="done", **identity))
            await settle(app, pilot)
            self.assertFalse(turn.history.collapsed)
            self.assertFalse(rows[0].collapsed)
            self.assertEqual(turn.viewport.scroll_y, position)
            self.assertIn('since recovery', str(turn.summary.content))


if __name__ == "__main__":
    unittest.main()
