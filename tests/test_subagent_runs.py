"""Focused coverage for durable subagent capture and reconstruction."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

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
        from tests.test_textual_app import make_app

        session = {
            "id": "thread-1",
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
            task_call_id="task-1",
            name="researcher [fox]",
            display_name="researcher [fox]",
            task_input="inspect code",
        )
        append_run_event(session, run["id"], {"type": "assistant", "text": "child answer"})
        app = make_app(session=session)

        async with app.run_test(size=(100, 32)) as pilot:
            panel = app.query_one(SubagentsPanel)
            panel.restore(session["runs"])
            self.assertIn(run["id"], panel._records)

            panel.post_message(panel.RunSelected(run["id"]))
            await pilot.pause()

            inspector = app.query_one(SubagentInspector)
            self.assertTrue(inspector.display)
            self.assertEqual(inspector.run_id, run["id"])
            self.assertFalse(app.query_one("#chat-log", ChatLog).display)
            inner_titles = [str(getattr(item, "border_title", "")) for item in inspector.query_one(ChatLog).children]
            self.assertIn("mira", inner_titles)
            self.assertTrue(app.query_one("#status-row").display)

            await pilot.click("#subagent-inspector-close")
            await pilot.pause()
            self.assertFalse(inspector.display)
            self.assertTrue(app.query_one("#chat-log", ChatLog).display)

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
