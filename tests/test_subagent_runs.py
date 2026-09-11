"""Focused coverage for durable subagent capture and reconstruction."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from core.execution.runner import EvalSubagentRenderer
from core.execution.streams.subagents import SubagentProtocolCapture, SubagentTranscriptCapture
from session.recorder import SessionEventEmitter, SessionRecorder
from session.store import SessionStore
from session.subagent_runs import (
    CANCELLED,
    DONE,
    ERROR,
    INTERRUPTED,
    RUNNING,
    append_run_event,
    finish_run,
    get_run,
    reconcile_stale_runs,
    runs_for_anchor,
    start_run,
)
from ui.textual.widgets import ChatLog, SubagentAnchor, SubagentInspector, SubagentsPanel
from ui.textual.subagent_replay import persisted_session, replay_run, restore_anchor
from ui.textual.widgets.tool_bubble import ToolBubble


class Store:
    def __init__(self) -> None:
        self.saved = False

    def save(self, _record: dict) -> None:
        self.saved = True


def record() -> dict:
    return {"events": [], "runs": [], "turns": 2}


def values_protocol_event(namespace: tuple[str, ...], messages: list[object]) -> dict:
    return {
        "method": "values",
        "params": {"namespace": namespace, "data": {"messages": messages}},
    }


def messages_protocol_event(
    namespace: tuple[str, ...],
    payload: dict,
    *,
    run_id: str = "model-run",
) -> dict:
    return {
        "method": "messages",
        "params": {
            "namespace": namespace,
            "data": (payload, {"run_id": run_id}),
        },
    }


class SubagentRunTests(unittest.TestCase):
    def test_standalone_task_creates_durable_run_with_tool_event_anchor(self) -> None:
        session = record()
        recorder = SessionRecorder(session, Store(), "action")
        event = recorder.delegation_started(
            [{"id": "task-1", "name": "task", "args": {"description": "inspect code"}}]
        )
        _, run = recorder.subagent_started(
            "researcher [fox]",
            "inspect code",
            task_call_id="task-1",
        )

        self.assertEqual(run["anchor_id"], event["calls"][0]["anchor_id"])
        self.assertEqual(run["task_call_id"], "task-1")
        self.assertEqual(run["turn_id"], "3")
        self.assertNotEqual(run["id"], run["anchor_id"])

    def test_eval_task_provenance_uses_outer_eval_anchor(self) -> None:
        session = record()
        recorder = SessionRecorder(session, Store(), "action")
        eval_event = recorder.tool_call("eval", {"code": "task(...)"}, call_id="eval-1")
        run = recorder.eval_subagent_started(
            "researcher [one]",
            "research one",
            eval_id="eval-1",
            task_call_id="ptc-task-1",
        )

        self.assertEqual(run["anchor_id"], str(eval_event["id"]))
        self.assertEqual(run["eval_id"], "eval-1")
        self.assertEqual(run["task_call_id"], "ptc-task-1")
        self.assertEqual(len(session["runs"]), 1, "eval itself must not be represented as a run")

    def test_multiple_eval_tasks_keep_separate_run_identities(self) -> None:
        session = record()
        recorder = SessionRecorder(session, Store(), "action")
        recorder.tool_call("eval", {}, call_id="eval-1")
        first = recorder.eval_subagent_started(
            "researcher [one]", "one", eval_id="eval-1", task_call_id="ptc-1"
        )
        second = recorder.eval_subagent_started(
            "researcher [two]", "two", eval_id="eval-1", task_call_id="ptc-2"
        )

        self.assertNotEqual(first["id"], second["id"])
        self.assertEqual(first["anchor_id"], second["anchor_id"])

    def test_parallel_transcripts_never_mix(self) -> None:
        session = record()
        first = start_run(session, anchor_id="a", turn_id="1", task_call_id="task-1")
        second = start_run(session, anchor_id="b", turn_id="1", task_call_id="task-2")
        append_run_event(session, first["id"], {"type": "assistant", "text": "first"})
        append_run_event(session, second["id"], {"type": "assistant", "text": "second"})

        self.assertEqual(first["events"][0]["text"], "first")
        self.assertEqual(second["events"][0]["text"], "second")

    def test_sequential_runs_preserve_creation_order_and_identity(self) -> None:
        session = record()
        first = start_run(session, anchor_id="a", turn_id="1", task_call_id="task-1")
        append_run_event(session, first["id"], {"type": "assistant", "text": "first"})
        finish_run(session, first["id"], status=DONE)
        second = start_run(session, anchor_id="b", turn_id="1", task_call_id="task-2")
        append_run_event(session, second["id"], {"type": "assistant", "text": "second"})
        finish_run(session, second["id"], status=DONE)

        self.assertEqual([run["id"] for run in session["runs"]], [first["id"], second["id"]])
        self.assertEqual([run["task_call_id"] for run in session["runs"]], ["task-1", "task-2"])

    def test_capture_persists_reasoning_messages_tools_and_results(self) -> None:
        class Sink:
            def __init__(self) -> None:
                self.events: list[dict] = []

            def subagent_run_event(self, _task_call_id: str, event: dict) -> None:
                self.events.append(dict(event))

        sink = Sink()
        capture = SubagentTranscriptCapture(sink, "task-1")
        capture.reasoning_delta("think")
        capture.text_delta("answer")
        capture.tool_call("read_file", {"path": "README.md"}, call_id="tool-1")
        capture.completed_tool_result("read_file", "contents", call_id="tool-1")

        self.assertEqual(
            [event["type"] for event in sink.events],
            ["reasoning", "assistant", "tool_call", "tool_result"],
        )

    def test_eval_protocol_snapshots_capture_every_parallel_child(self) -> None:
        session = record()
        recorder = SessionRecorder(session, Store(), "action")
        recorder.tool_call("eval", {}, call_id="eval-1")
        for letter in "ABC":
            recorder.eval_subagent_started(
                f"worker [{letter}]",
                f"Return {letter}.",
                eval_id="eval-1",
                task_call_id=f"ptc-{letter}",
            )

        emitter = SessionEventEmitter(object(), recorder)
        capture = SubagentProtocolCapture(emitter)
        human_a = HumanMessage("Return A.")
        capture.handle(values_protocol_event(("tools:one",), [human_a]))
        for letter in "ABC":
            capture.run_started(
                {"id": f"ptc-{letter}", "description": f"Return {letter}."}
            )

        tool_call = AIMessage(
            content="",
            additional_kwargs={"reasoning_content": "inspect first"},
            tool_calls=[{"name": "read_file", "args": {"path": "README.md"}, "id": "tool-1"}],
        )
        tool_result = ToolMessage("contents", tool_call_id="tool-1", name="read_file")
        final_a = AIMessage("A")
        capture.handle(values_protocol_event(("tools:one",), [human_a, tool_call]))
        capture.handle(
            values_protocol_event(("tools:one",), [human_a, tool_call, tool_result])
        )
        capture.handle(
            values_protocol_event(
                ("tools:one",), [human_a, tool_call, tool_result, final_a]
            )
        )
        for index, letter in enumerate("BC", start=1):
            capture.handle(
                values_protocol_event(
                    ("tools:one", str(index)),
                    [HumanMessage(f"Return {letter}."), AIMessage(letter)],
                )
            )

        runs = {run["task_call_id"]: run for run in session["runs"]}
        self.assertEqual(runs["ptc-A"]["stream_path"], ["tools:one"])
        self.assertEqual(
            [event["type"] for event in runs["ptc-A"]["events"]],
            ["reasoning", "tool_call", "tool_result", "assistant"],
        )
        self.assertEqual(runs["ptc-A"]["output"], "A")
        self.assertEqual(runs["ptc-B"]["events"][0]["text"], "B")
        self.assertEqual(runs["ptc-C"]["events"][0]["text"], "C")

    def test_protocol_message_deltas_capture_reasoning_and_restore_full_task(self) -> None:
        session = record()
        recorder = SessionRecorder(session, Store(), "action")
        recorder.tool_call("eval", {}, call_id="eval-1")
        full_task = "Investigate the native protocol carefully. " + "Full detail. " * 20
        shortened = full_task[:200]
        run = recorder.eval_subagent_started(
            "worker [one]",
            shortened,
            eval_id="eval-1",
            task_call_id="ptc-one",
        )
        capture = SubagentProtocolCapture(SessionEventEmitter(object(), recorder))
        capture.run_started({"id": "ptc-one", "description": shortened})
        namespace = ("tools:eval",)
        capture.handle(values_protocol_event(namespace, [HumanMessage(full_task)]))
        capture.handle(
            messages_protocol_event(
                namespace,
                {
                    "event": "content-block-delta",
                    "delta": {
                        "type": "reasoning-delta",
                        "reasoning": "check evidence",
                    },
                },
            )
        )
        capture.handle(
            messages_protocol_event(
                namespace,
                {
                    "event": "content-block-delta",
                    "delta": {"type": "text-delta", "text": "final answer"},
                },
            )
        )
        capture.handle(
            values_protocol_event(
                namespace,
                [
                    HumanMessage(full_task),
                    AIMessage(
                        "final answer",
                        id="message-one",
                        additional_kwargs={"reasoning_content": "check evidence"},
                    ),
                ],
            )
        )

        self.assertEqual(run["task_input"], full_task)
        self.assertEqual(
            [(event["type"], event["text"]) for event in run["events"]],
            [("reasoning", "check evidence"), ("assistant", "final answer")],
        )

    def test_high_level_eval_child_reuses_protocol_run_identity(self) -> None:
        session = record()
        recorder = SessionRecorder(session, Store(), "action")
        recorder.tool_call("eval", {}, call_id="eval-1")
        run = recorder.eval_subagent_started(
            "worker [one]",
            "inspect",
            eval_id="eval-1",
            task_call_id="ptc-one",
        )
        capture = SubagentProtocolCapture(SessionEventEmitter(object(), recorder))
        capture.run_started(
            {"id": "ptc-one", "description": "inspect", "eval_id": "eval-1"}
        )
        namespace = ("tools:eval-1",)

        claimed = capture.claim_high_level_subagent(
            SimpleNamespace(task_input="", path=namespace)
        )
        capture.handle(
            values_protocol_event(
                namespace,
                [HumanMessage("inspect with the complete native task input")],
            )
        )
        capture.handle(
            messages_protocol_event(
                namespace,
                {
                    "event": "content-block-delta",
                    "delta": {"type": "text-delta", "text": "duplicate"},
                },
            )
        )

        self.assertEqual(claimed, "ptc-one")
        self.assertEqual(session["runs"], [run])
        self.assertEqual(run["task_input"], "inspect with the complete native task input")
        self.assertEqual(run["events"], [])

    def test_eval_description_fallback_receives_a_cool_name(self) -> None:
        class Renderer:
            def __init__(self) -> None:
                self.started: list[str] = []

            def subagent_label(self, subagent: object) -> str:
                return f"{getattr(subagent, 'name')} [mauve-mammoth]"

            def eval_subagent_started(self, name: str, *_args: object, **_kwargs: object) -> None:
                self.started.append(name)

        renderer = Renderer()
        description = "Write a story that is exactly 20 words long."
        EvalSubagentRenderer(renderer).handle(
            {
                "type": "subagent",
                "phase": "start",
                "id": "ptc-one",
                "subagent_type": "general-purpose",
                "description": description,
                "label": "final",
            }
        )

        self.assertEqual(renderer.started, ["general-purpose [mauve-mammoth]"])

    def test_eval_failure_persists_visible_error_transcript_event(self) -> None:
        session = record()
        recorder = SessionRecorder(session, Store(), "action")
        recorder.tool_call("eval", {}, call_id="eval-1")
        run = recorder.eval_subagent_started(
            "worker [one]",
            "fail clearly",
            eval_id="eval-1",
            task_call_id="ptc-one",
        )

        SessionEventEmitter(object(), recorder).eval_subagent_cancelled(
            "worker [one]",
            "context limit exceeded",
            eval_id="eval-1",
            row_id="ptc-one",
        )

        self.assertEqual(run["status"], ERROR)
        self.assertEqual(run["output"], "context limit exceeded")
        self.assertEqual(
            [(event["type"], event["text"]) for event in run["events"]],
            [("system_error", "context limit exceeded")],
        )

    def test_generated_display_name_is_normalized_without_regeneration(self) -> None:
        session = record()
        run = start_run(
            session,
            anchor_id="anchor",
            turn_id="1",
            task_call_id="task-1",
            name="researcher [fox]",
            display_name="researcher [fox]",
        )
        enriched = start_run(
            session,
            anchor_id="anchor",
            turn_id="1",
            task_call_id="task-1",
            name="researcher [different-runtime-label]",
        )

        self.assertIs(enriched, run)
        self.assertEqual(enriched["display_name"], "researcher [fox]")

    def test_terminal_statuses_and_stale_reconciliation_are_durable(self) -> None:
        for status in (DONE, ERROR, CANCELLED, INTERRUPTED):
            session = record()
            run = start_run(session, anchor_id="a", turn_id="1", task_call_id=status)
            finish_run(session, run["id"], status=status, output="result", duration_ms=12)
            self.assertEqual(get_run(session, run["id"])["status"], status)

        stale = record()
        run = start_run(stale, anchor_id="a", turn_id="1", task_call_id="stale")
        self.assertTrue(reconcile_stale_runs(stale))
        self.assertEqual(run["status"], INTERRUPTED)

    def test_full_store_reload_reconciles_stale_run_from_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            session = store.new("session-1", Path(directory))
            run = start_run(session, anchor_id="a", turn_id="1", task_call_id="task-1")
            append_run_event(session, run["id"], {"type": "assistant", "text": "durable"})
            store.save(session)

            loaded = store.read(store.path("session-1"))

        self.assertEqual(loaded["runs"][0]["status"], INTERRUPTED)
        self.assertEqual(loaded["runs"][0]["events"][0]["text"], "durable")
        anchors = [event for event in loaded["events"] if event.get("type") == "subagent_anchor"]
        self.assertEqual([event["anchor_id"] for event in anchors], [run["anchor_id"]])

    def test_runs_for_anchor_returns_only_that_origin(self) -> None:
        session = record()
        start_run(session, anchor_id="a", turn_id="1", task_call_id="one")
        start_run(session, anchor_id="b", turn_id="1", task_call_id="two")
        self.assertEqual([run["task_call_id"] for run in runs_for_anchor(session, "a")], ["one"])


class SubagentReplayTests(unittest.TestCase):
    def test_terminal_task_callback_appends_and_emits_anchor(self) -> None:
        class Renderer:
            def __init__(self) -> None:
                self.anchors: list[str] = []

            def delegation_started(self, _calls: list[dict], **_kwargs: object) -> None:
                pass

            def subagent_started(self, *_args: object, **_kwargs: object) -> None:
                pass

            def subagent_finished(self, *_args: object, **_kwargs: object) -> None:
                pass

            def completed_tool_result(self, *_args: object, **_kwargs: object) -> None:
                pass

            def subagent_anchor(self, anchor_id: str, **_kwargs: object) -> None:
                self.anchors.append(anchor_id)

        session = record()
        renderer = Renderer()
        emitter = SessionEventEmitter(renderer, SessionRecorder(session, Store(), "action"))
        emitter.delegation_started(
            [{"id": "task-1", "name": "task", "args": {"description": "one"}}]
        )
        emitter.subagent_started("worker [one]", "one", task_call_id="task-1")
        emitter.subagent_finished("worker [one]", "done", task_call_id="task-1")
        emitter.completed_tool_result("task", "done", call_id="task-1")

        anchors = [event for event in session["events"] if event.get("type") == "subagent_anchor"]
        self.assertEqual(renderer.anchors, [anchors[0]["anchor_id"]])

    def test_standalone_and_eval_anchors_count_only_owned_runs(self) -> None:
        session = record()
        recorder = SessionRecorder(session, Store(), "action")
        task_event = recorder.delegation_started(
            [{"id": "task-1", "name": "task", "args": {"description": "one"}}]
        )
        _, task_run = recorder.subagent_started("worker [one]", "one", task_call_id="task-1")
        recorder.subagent_finished("worker [one]", "done", task_call_id="task-1")
        task_anchor = recorder.subagent_anchor("task", "task-1")

        eval_event = recorder.tool_call("eval", {}, call_id="eval-1")
        for index in range(3):
            run = recorder.eval_subagent_started(
                f"worker [{index}]",
                str(index),
                eval_id="eval-1",
                task_call_id=f"ptc-{index}",
            )
            finish_run(session, run["id"], status=DONE)
        eval_anchor = recorder.subagent_anchor("eval", "eval-1")

        self.assertEqual(task_anchor["anchor_id"], task_event["calls"][0]["anchor_id"])
        self.assertEqual(eval_anchor["anchor_id"], str(eval_event["id"]))
        self.assertEqual(runs_for_anchor(session, task_anchor["anchor_id"]), [task_run])
        self.assertEqual(len(runs_for_anchor(session, eval_anchor["anchor_id"])), 3)

    def test_two_anchor_replays_restore_different_ordered_subsets(self) -> None:
        session = record()
        one = start_run(session, anchor_id="one", turn_id="1", task_call_id="task-1")
        two = start_run(session, anchor_id="two", turn_id="1", task_call_id="task-2")
        three = start_run(session, anchor_id="two", turn_id="1", task_call_id="task-3")

        class Panel:
            def __init__(self) -> None:
                self.runs: list[dict] = []

            def restore(self, runs: list[dict]) -> None:
                self.runs = runs

        panel = Panel()
        restore_anchor(panel, session, "one")
        self.assertEqual(panel.runs, [one])
        restore_anchor(panel, session, "two")
        self.assertEqual(panel.runs, [two, three])

    def test_restored_panel_preserves_run_group_order_and_status_notation(self) -> None:
        session = record()
        regular = start_run(session, anchor_id="a", turn_id="1", task_call_id="regular")
        failed = start_run(
            session,
            anchor_id="b",
            turn_id="1",
            task_call_id="failed",
            eval_id="eval-1",
        )
        interrupted = start_run(
            session,
            anchor_id="b",
            turn_id="1",
            task_call_id="interrupted",
            eval_id="eval-1",
        )
        finish_run(session, regular["id"], status=DONE)
        finish_run(session, failed["id"], status=ERROR)
        finish_run(session, interrupted["id"], status=INTERRUPTED)

        panel = SubagentsPanel()
        panel.restore(session["runs"])

        self.assertEqual(list(panel._records), [regular["id"], failed["id"], interrupted["id"]])
        self.assertEqual(panel._regular_order, [regular["id"]])
        self.assertEqual(panel._groups[panel._group_order[0]].order, [failed["id"], interrupted["id"]])
        self.assertEqual(
            [panel._row_cells(panel._records[run["id"]], 30)[0].plain[:3] for run in session["runs"]],
            [" v ", " x ", " - "],
        )

    def test_terminal_failure_cancellation_and_interruption_still_anchor(self) -> None:
        for terminal in (ERROR, CANCELLED, INTERRUPTED):
            session = record()
            run = start_run(session, anchor_id=terminal, turn_id="1", task_call_id="task")
            finish_run(session, run["id"], status=terminal)
            event = SessionRecorder(session, Store(), "action").subagent_anchor(
                "task", status=terminal.lower(), anchor_id=terminal
            )
            self.assertIsNotNone(event)

    def test_session_without_subagents_has_no_anchor(self) -> None:
        session = record()
        event = SessionRecorder(session, Store(), "action").subagent_anchor(
            "task", anchor_id="missing"
        )
        self.assertIsNone(event)

    def test_replay_run_reads_only_supplied_json_record(self) -> None:
        first = record()
        second = record()
        run = start_run(first, anchor_id="a", turn_id="1", task_call_id="task")
        self.assertIs(replay_run(first, run["id"]), run)
        self.assertIsNone(replay_run(second, run["id"]))


class SubagentInspectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_panel_run_id_opens_shared_inspector_from_session_data(self) -> None:
        from tests.test_textual_app import make_app, renderable_plain

        session = {
            "id": "thread-1",
            "workspace": ".",
            "created_at": "2026-01-01T00:00:00+00:00",
            "turns": 0,
            "events": [],
            "runs": [],
        }
        full_task = "Inspect every relevant implementation detail without truncation. " * 8
        run = start_run(
            session,
            anchor_id="anchor",
            turn_id="1",
            task_call_id="task-1",
            name="researcher [fox]",
            display_name="researcher [fox]",
            task_input=full_task,
        )
        append_run_event(session, run["id"], {"type": "assistant", "text": "child answer"})
        app = make_app(session=session)

        async with app.run_test(size=(100, 32)) as pilot:
            panel = app.query_one(SubagentsPanel)
            panel.restore(session["runs"])
            await pilot.pause()
            self.assertIn(run["id"], panel._records)
            table = panel.query_one("#subagents-tasks")
            self.assertTrue(table.show_cursor)
            self.assertFalse(table.can_focus)
            self.assertEqual(table.styles.padding.right, 1)
            completion = app.query_one("#autocomplete-input")
            prompt = completion.query_one("#prompt")
            panel_ceiling = completion.safe_prompt_height
            self.assertGreater(panel_ceiling, prompt.region.height)
            prompt.styles.height = panel_ceiling
            await pilot.pause()
            self.assertEqual(prompt.region.height, panel_ceiling)
            prompt.styles.height = 3
            await pilot.pause()

            await pilot.click("#subagents-tasks", offset=(4, 0))
            await pilot.pause()

            inspector = app.query_one(SubagentInspector)
            self.assertTrue(inspector.display)
            self.assertEqual(inspector.run_id, run["id"])
            self.assertFalse(app.query_one("#chat-log", ChatLog).display)
            self.assertTrue(panel.display)
            self.assertTrue(app.query_one("#autocomplete-input").display)
            self.assertTrue(app.query_one("#telemetry-row").display)
            self.assertIsNone(app.focused)
            inspector_ceiling = completion.safe_prompt_height
            self.assertGreater(inspector_ceiling, prompt.region.height)
            prompt.styles.height = inspector_ceiling
            await pilot.pause()
            self.assertEqual(prompt.region.height, inspector_ceiling)
            task = inspector.query_one("#subagent-inspector-task")
            self.assertIn(full_task, renderable_plain(task))
            self.assertGreater(task.region.height, 3)
            inner_titles = [
                str(getattr(item, "border_title", ""))
                for item in inspector.query_one(ChatLog).children
            ]
            self.assertIn("mira", inner_titles)
            self.assertTrue(app.query_one("#status-row").display)

            app.open_subagent_inspector(run["id"])
            await pilot.pause()

            await pilot.click("#subagent-inspector-close")
            await pilot.pause()
            self.assertFalse(inspector.display)
            self.assertTrue(app.query_one("#chat-log", ChatLog).display)

    async def test_eval_anchor_is_an_unfocused_tool_bubble_action(self) -> None:
        from tests.test_textual_app import make_app

        session = {
            "id": "thread-anchor",
            "title": "Anchor test",
            "workspace": ".",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "turns": 0,
            "dashboard": {},
            "current_plan": None,
            "current_goal": None,
            "events": [],
            "runs": [],
        }
        recorder = SessionRecorder(session, Store(), "action")
        recorder.tool_call("eval", {"code": "await task(...)"}, call_id="eval-one")
        run = recorder.eval_subagent_started(
            "worker [one]",
            "inspect",
            eval_id="eval-one",
            task_call_id="ptc-one",
        )
        finish_run(session, run["id"], status=DONE)
        recorder.completed_tool_result("eval", "done", call_id="eval-one")
        recorder.subagent_anchor("eval", "eval-one")
        app = make_app(session=session)
        async with app.run_test(size=(100, 28)) as pilot:
            chat = app.query_one(ChatLog)
            anchor = chat.query_one(SubagentAnchor)
            ancestors = []
            parent = anchor.parent
            while parent is not None:
                ancestors.append(parent)
                parent = parent.parent
            self.assertTrue(any(isinstance(widget, ToolBubble) for widget in ancestors))
            self.assertFalse(anchor.can_focus)
            anchor.press()
            await pilot.pause()
            await pilot.pause()
            self.assertIn(run["id"], app.query_one(SubagentsPanel)._records)

    async def test_task_anchor_is_inside_its_tool_bubble_and_opens_panel(self) -> None:
        from tests.test_textual_app import make_app

        session = {
            "id": "thread-task-anchor",
            "title": "Task anchor test",
            "custom_title": False,
            "pinned": False,
            "workspace": ".",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "turns": 0,
            "dashboard": {},
            "current_plan": None,
            "current_goal": None,
            "events": [],
            "runs": [],
        }
        recorder = SessionRecorder(session, Store(), "action")
        recorder.tool_call(
            "task",
            {"description": "inspect", "subagent_type": "general-purpose"},
            call_id="task-one",
        )
        recorder.delegation_started(
            [{"id": "task-one", "name": "task", "args": {"description": "inspect"}}]
        )
        _, run = recorder.subagent_started(
            "worker [one]", "inspect", task_call_id="task-one"
        )
        recorder.subagent_finished("worker [one]", "done", task_call_id="task-one")
        recorder.completed_tool_result("task", "done", call_id="task-one")
        recorder.subagent_anchor("task", "task-one")

        app = make_app(session=session)
        async with app.run_test(size=(100, 28)) as pilot:
            anchor = app.query_one(ChatLog).query_one(SubagentAnchor)
            self.assertTrue(any(isinstance(parent, ToolBubble) for parent in anchor.ancestors))
            anchor.press()
            await pilot.pause()
            await pilot.pause()
            self.assertIn(run["id"], app.query_one(SubagentsPanel)._records)

    async def test_legacy_error_output_is_visible_in_inspector(self) -> None:
        from tests.test_textual_app import make_app, renderable_plain

        session = {
            "id": "thread-error",
            "workspace": ".",
            "created_at": "2026-01-01T00:00:00+00:00",
            "turns": 0,
            "events": [],
            "runs": [],
        }
        run = start_run(
            session,
            anchor_id="anchor",
            turn_id="1",
            task_call_id="task-error",
            task_input="fail",
        )
        finish_run(session, run["id"], status=ERROR, output="context limit exceeded")
        app = make_app(session=session)

        async with app.run_test(size=(100, 28)) as pilot:
            inspector = app.query_one(SubagentInspector)
            inspector.show_run(run)
            await pilot.pause()
            transcript = "\n".join(
                renderable_plain(widget) for widget in inspector.query_one(ChatLog).children
            )
            self.assertIn("context limit exceeded", transcript)

    async def test_restart_anchor_reconstructs_panel_and_inspector_from_file(self) -> None:
        from tests.test_textual_app import make_app, renderable_plain

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SessionStore(root)
            session = store.new("restart", root)
            recorder = SessionRecorder(session, store, "action")
            recorder.delegation_started(
                [{"id": "task-1", "name": "task", "args": {"description": "persist me"}}]
            )
            _, run = recorder.subagent_started(
                "researcher [fox]", "persist me", task_call_id="task-1"
            )
            recorder.subagent_run_event(
                "task-1", {"type": "assistant", "text": "restored answer"}
            )
            recorder.subagent_run_event(
                "task-1",
                {
                    "type": "delegation",
                    "calls": [
                        {"id": "nested", "name": "task", "args": {"description": "nested"}}
                    ],
                },
            )
            recorder.subagent_finished(
                "researcher [fox]", "restored answer", task_call_id="task-1"
            )
            recorder.subagent_anchor("task", "task-1")
            restarted = store.read(store.path("restart"))
            app = make_app(workspace=root, session=restarted, store=store)
            restarted["runs"][0]["status"] = RUNNING
            restarted["runs"][0]["events"][0]["text"] = "cached wrong answer"

            async with app.run_test(size=(100, 32)) as pilot:
                anchors = list(app.query(SubagentAnchor))
                self.assertEqual(len(anchors), 1)
                self.assertEqual(str(anchors[0].label), "Subagents · 1")

                anchors[0].press()
                await pilot.pause()
                panel = app.query_one(SubagentsPanel)
                self.assertEqual(list(panel._records), [run["id"]])
                self.assertEqual(panel._records[run["id"]].status, DONE)

                panel.post_message(panel.RunSelected(run["id"]))
                await pilot.pause()
                inspector = app.query_one(SubagentInspector)
                self.assertTrue(inspector.display)
                self.assertEqual(inspector.run_id, run["id"])
                self.assertEqual(len(inspector.query(SubagentAnchor)), 0)
                transcript = "\n".join(
                    renderable_plain(widget) for widget in inspector.query_one(ChatLog).children
                )
                self.assertIn("restored answer", transcript)
                self.assertNotIn("cached wrong answer", transcript)

                loaded_again = persisted_session(store, restarted)
                self.assertEqual(replay_run(loaded_again, run["id"])["events"][0]["text"], "restored answer")


if __name__ == "__main__":
    unittest.main()
