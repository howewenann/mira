#!/usr/bin/env python
"""MIRA live Rubric offload probe using the actual MIRA Textual Inspector widget.

Run from the MIRA repository in its environment:

    python tests/probes/mira_rubric_inspector_probe.py
    python tests/probes/mira_rubric_inspector_probe.py --session-id SESSION_ID

1: Verifier Inspector, 2: Grader Inspector, Ctrl+Q: exit.
The existing source session is backed up read-only, never resumed or modified.
Production offloading and its read guard run in evaluation-local memory.
Temporary integrity-check copies are cleaned up on exit.
The displayed events come from actual nested MIRA Verifier/Grader executions.
"""


from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

from mira_rubric_live_probe_fixed import (
    _checkpoint_messages,
    _complete_tool_pages,
    _copy_db_read_only,
    _default_rubric,
    _effective_messages,
    _offload,
    _repo,
    _selected_indices,
    _session_candidates,
)


# Inspector UI runner (uses MIRA's actual Inspector and RubricInspectionProjector)
# ============================================================================


def _ui_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the live Rubric offloading probe inside MIRA's real Textual Inspector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo", type=Path, default=None, help="MIRA source repository, if not current directory")
    parser.add_argument("--workspace", type=Path, default=None, help="MIRA workspace with .mira settings and saved sessions")
    parser.add_argument("--session-id", default=None, help="Saved MIRA session id (source is read-only)")
    parser.add_argument("--thread-id", default=None, help="Exact Act/Plan/Goal thread within --session-id")
    parser.add_argument("--scan-sessions", type=int, default=25, help="Most recent sessions to scan")
    parser.add_argument("--rubric-file", type=Path, default=None, help="Optional rubric text file")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic evidence only, or append it to an explicit --session-id if needed")
    parser.add_argument("--max-total-chars", type=int, default=250000, help="Safety cap for offloaded evidence")
    parser.add_argument("--tool-output-chars", type=int, default=5000, help="Displayed tool-result preview budget; raw tool evidence stays intact")
    parser.add_argument("--headless", action="store_true", help="Run the real Inspector without a terminal and exit when the live probe completes")
    return parser.parse_args()


def inspector_main() -> int:
    opts = _ui_args()
    repo = _repo(opts.repo)
    workspace = (opts.workspace or repo).expanduser().resolve()
    if not (workspace / ".mira").is_dir():
        raise RuntimeError(f"Not a MIRA workspace (missing .mira/): {workspace}")
    if opts.thread_id and not opts.session_id:
        raise RuntimeError("--thread-id requires --session-id")
    sys.path.insert(0, str(repo))

    import threading
    from textual import on
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.widgets import Button, Footer, Header, Static

    from core.execution.inspection.live import LiveInspectionStore
    from core.execution.inspection.rubric import RubricInspectionProjector, rubric_inspection_id
    from ui.textual.widgets.inspector import Inspector
    from ui.textual.platform.windows.input import driver_class_for_platform

    RUN_ID = "mira-offload-probe"
    VERIFIER_ID = rubric_inspection_id(RUN_ID, 0, "verifier")
    GRADER_ID = rubric_inspection_id(RUN_ID, 0, "grader")
    styles = [
        str(repo / "ui" / "textual" / "styles" / "mira.tcss"),
        str(repo / "ui" / "textual" / "styles" / "workspace.tcss"),
    ]

    class RubricInspectorProbe(App[None]):
        AUTO_FOCUS = None
        TITLE = "MIRA · Rubric Offload Live Probe"
        CSS_PATH = [p for p in styles if Path(p).is_file()]
        CSS = """
        Screen { layout: vertical; }
        #probe-top { height: 3; width: 100%; align: left middle; padding: 0 1; }
        #probe-top Button { margin-right: 1; min-width: 13; }
        #probe-title { width: 1fr; content-align: right middle; text-style: bold; }
        #probe-inspector { height: 1fr; width: 100%; }
        #probe-status { height: 3; padding: 0 2; content-align: left middle; }
        #probe-coverage { height: 3; padding: 0 2; content-align: left middle; }
        """
        BINDINGS = [
            Binding("1", "verifier", "Verifier"),
            Binding("2", "grader", "Grader"),
            Binding("ctrl+q", "quit", "Quit"),
        ]

        def __init__(self) -> None:
            super().__init__(driver_class=driver_class_for_platform(sys.platform))
            self.live_inspections = LiveInspectionStore()
            self.projection = RubricInspectionProjector(self)
            self.active_phase = "verifier"
            self.ui_thread: int | None = None
            self.done = False
            self.completed = False
            self.last_error = ""
            self.stats: dict[str, Any] = {}

        def compose(self) -> ComposeResult:
            yield Header(show_clock=False)
            with Horizontal(id="probe-top"):
                yield Button("1 · Verifier", id="probe-verifier", variant="primary")
                yield Button("2 · Grader", id="probe-grader")
                yield Static("Read-only · live model calls", id="probe-title")
            yield Inspector(
                self.live_inspections,
                tool_output_chars=max(240, opts.tool_output_chars),
                embedded=True,
                id="probe-inspector",
            )
            yield Static("Preparing read-only checkpoint snapshot...", id="probe-status")
            yield Static("Coverage: not yet evaluated", id="probe-coverage")
            yield Footer()

        def on_mount(self) -> None:
            self.set_focus(None)
            self.ui_thread = threading.get_ident()
            self.call_after_refresh(self._launch)

        def _launch(self) -> None:
            self.run_worker(self._execute_probe(), name="rubric-offload-probe", exclusive=True)

        @on(Button.Pressed, "#probe-verifier")
        def verifier_pressed(self, event: Button.Pressed) -> None:
            event.stop()
            self.action_verifier()

        @on(Button.Pressed, "#probe-grader")
        def grader_pressed(self, event: Button.Pressed) -> None:
            event.stop()
            self.action_grader()

        def action_verifier(self) -> None:
            self._show_phase("verifier")

        def action_grader(self) -> None:
            self._show_phase("grader")

        def _show_phase(self, phase: str) -> None:
            self.active_phase = phase
            inspection_id = VERIFIER_ID if phase == "verifier" else GRADER_ID
            self.query_one("#probe-verifier", Button).variant = "primary" if phase == "verifier" else "default"
            self.query_one("#probe-grader", Button).variant = "primary" if phase == "grader" else "default"
            inspector = self.query_one("#probe-inspector", Inspector)
            if not inspector.open(inspection_id):
                self._status(f"{phase.title()} has not started yet. Select 1 or 2 after the phase begins.")

        def _status(self, message: str) -> None:
            self.query_one("#probe-status", Static).update(message)

        def _coverage(self, message: str) -> None:
            self.query_one("#probe-coverage", Static).update(message)

        def _event_writer(self, value: dict[str, Any]) -> None:
            # MIRA's nested tool events may originate in a worker thread.
            # All Textual widgets and LiveInspectionStore subscriptions are
            # mutated on the UI thread only.
            if threading.get_ident() == self.ui_thread:
                self._apply_event(value)
            else:
                self.call_from_thread(self._apply_event, value)

        def _apply_event(self, value: dict[str, Any]) -> None:
            kind = str(value.get("type") or "")
            if kind == "rubric_verification_start":
                self.projection.start(value, "verifier")
                if self.active_phase == "verifier":
                    self._show_phase("verifier")
                self._status("Verifier running · real model + read_file tool calls")
            elif kind == "rubric_grading_start":
                self.projection.start(value, "grader")
                if self.active_phase == "grader":
                    self._show_phase("grader")
                self._status("Grader running · tool-free structured output")
            elif kind == "rubric_inspection_delta":
                self.projection.append_delta(value)
            elif kind == "rubric_tool_start":
                self.projection.tool_call(value)
            elif kind == "rubric_tool_end":
                self.projection.tool_completion(value)
            elif kind == "rubric_verification_end":
                self.projection.finish(value, "verifier")
                if not value.get("succeeded", False):
                    self._status("Verifier failed · select 1 to inspect the actual error")
            elif kind == "rubric_grading_end":
                self.projection.finish(value, "grader")
                if not value.get("succeeded", False):
                    self._status("Grader failed · select 2 to inspect the actual error")

        async def _execute_probe(self) -> None:
            import tempfile
            from config.loader import load_config
            from deepagents.backends import FilesystemBackend
            from deepagents.middleware import rubric as stock
            from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
            from agent.llm import get_llm, get_rubric_model_name
            from agent.rubric.middleware import MiraRubricMiddleware, _RUBRIC_RUN
            from config.settings import RUBRIC_MODEL

            try:
                config = load_config(workspace)
                candidates = [] if opts.synthetic and not opts.session_id else _session_candidates(config, opts.session_id, opts.scan_sessions)
                with tempfile.TemporaryDirectory(prefix="mira-rubric-inspector-probe-") as tmp:
                    temp = Path(tmp)
                    selected = ("synthetic", "synthetic", [], {}) if opts.synthetic and not opts.session_id else None
                    fallback = None
                    for idx, (sid, db) in enumerate(candidates):
                        self._status(f"Scanning checkpoint {idx+1}/{len(candidates)} · {sid}")
                        backup = temp / f"session-{idx}-backup.db"
                        try:
                            _copy_db_read_only(db, backup)
                            snapshots = await _checkpoint_messages(backup, sid, requested_thread=opts.thread_id)
                        except Exception:
                            if opts.session_id:
                                raise
                            continue
                        best = None
                        for tid, messages, values in snapshots:
                            effective = _effective_messages(messages, values)
                            count = sum(
                                len(stock._coerce_text(effective[i])) > stock._MAX_TRANSCRIPT_CHARS_PER_MESSAGE
                                for i in _selected_indices(effective)
                            )
                            if best is None or count > best[3]:
                                best = (tid, effective, values, count)
                        if best is None:
                            continue
                        tid, effective, values, count = best
                        row = (sid, tid, effective, values)
                        if fallback is None:
                            fallback = row
                        if count or (opts.synthetic and opts.session_id):
                            selected = row
                            break
                        if opts.session_id:
                            break
                    if selected is None and opts.synthetic:
                        selected = fallback
                    if selected is None:
                        raise RuntimeError(
                            "No eligible >4,000-char message found in the stock Rubric window. "
                            "Try --session-id or --synthetic."
                        )

                    sid, tid, effective, values = selected
                    large = [i for i in _selected_indices(effective)
                             if len(stock._coerce_text(effective[i])) > stock._MAX_TRANSCRIPT_CHARS_PER_MESSAGE]
                    if not large and opts.synthetic:
                        effective.append(AIMessage(
                            content="[PROBE-ONLY synthetic assistant evidence]\n"
                            + "Verified line content from the session.\n" * 230
                            + "PROBE_END_SENTINEL_7F32F4D1\n"
                        ))
                    backend = FilesystemBackend(root_dir=temp / "workspace", virtual_mode=True)
                    surrogates, required = _offload(effective, backend, opts.max_total_chars)
                    self.stats = {"session": sid, "thread": tid, "required": required}
                    self._status(
                        f"Session {sid} · {len(required)} offloaded message(s) · "
                        f"{sum(map(len, required.values())):,} chars · "
                        f"Rubric model: {get_rubric_model_name(config)}"
                    )
                    self._coverage("Offloaded: " + ", ".join(f"{path} ({len(value):,} chars)" for path, value in required.items()))

                    try:
                        rubric_text = opts.rubric_file.read_text(encoding="utf-8") if opts.rubric_file else _default_rubric(values, required)
                        model = get_llm(config, role=RUBRIC_MODEL)
                        middleware = MiraRubricMiddleware(
                            model=model,
                            verifier_tools=[],
                            verifier_middleware=[],
                            max_iterations=1,
                        )
                        state = {"messages": list(effective), "rubric": rubric_text, "_rubric_criteria": []}
                        verify_base = middleware._astream_verifier
                        grade_base = middleware._astream_final_grader
                        evidence_holder: dict[str, Any] = {}

                        async def verifier_run(
                            verifier: Any, nested_input: dict[str, Any], *, config: dict[str, Any], context: Any
                        ) -> dict[str, Any]:
                            result = await verify_base(verifier, nested_input, config=config, context=context)
                            evidence = middleware._verification_evidence(result)
                            evidence_holder["evidence"] = evidence
                            coverage = _complete_tool_pages(evidence, required)
                            evidence_holder["coverage"] = coverage
                            summary = "; ".join(
                                f"{Path(path).name}: {item['covered_lines']}/{item['total_lines']} lines"
                                for path, item in coverage.items()
                            )
                            self._coverage("Actual Verifier coverage · " + summary)
                            if not all(item["complete"] for item in coverage.values()):
                                raise AssertionError(
                                    "INCOMPLETE RETRIEVAL: grader intentionally BLOCKED. " + summary
                                )
                            return result

                        async def grader_run(
                            grader: Any, nested_input: dict[str, Any], *, config: dict[str, Any], context: Any
                        ) -> dict[str, Any]:
                            expected = evidence_holder.get("evidence") or []
                            actual_messages = nested_input.get("messages") or []
                            actual_ids = {id(item) for item in actual_messages}
                            lost = [item for item in expected if (
                                id(item) not in actual_ids if isinstance(item, ToolMessage) else item not in actual_messages
                            )]
                            if lost:
                                raise AssertionError(f"GRADER INPUT LOST {len(lost)} REAL verifier evidence messages")
                            count = sum(isinstance(item, ToolMessage) for item in expected)
                            if not count:
                                raise AssertionError("Grader input has no real ToolMessages")
                            self._coverage(
                                f"PASS Verifier coverage and forwarding · {count} original ToolMessage(s) in Grader input"
                            )
                            return await grade_base(grader, nested_input, config=config, context=context)

                        middleware._astream_verifier = verifier_run
                        middleware._astream_final_grader = grader_run
                        token = _RUBRIC_RUN.set((RUN_ID, 0, self._event_writer))
                        try:
                            graded = await middleware._ainvoke_grader(state, iteration=0, context=None)
                            self.completed = True
                            self._status(
                                f"PASS probe execution · Grader verdict: {graded.result} · "
                                "Press 1/2 to inspect actual Verifier/Grader events"
                            )
                        finally:
                            _RUBRIC_RUN.reset(token)
                    finally:
                        cleanup = backend.delete("/rubric_offload")
                        if cleanup.error:
                            self._coverage(f"WARNING: explicit cleanup failed ({cleanup.error}); temp directory still removed")
                self.done = True
            except asyncio.CancelledError:
                self._status("Probe cancelled; original session unchanged")
                raise
            except Exception as exc:
                self.done = True
                self.last_error = f"{type(exc).__name__}: {exc}"
                self._status(f"FAIL: {self.last_error} · original session unchanged · press 1/2 to inspect")
                # A setup failure occurs before any native Rubric events. Show it
                # as a diagnostic error (not a fake grader/tool result).
                if self.live_inspections.get(VERIFIER_ID) is None:
                    self.live_inspections.start(
                        VERIFIER_ID, "Probe setup", "Diagnostic probe initialization", inspection_type="Rubrics"
                    )
                    self.live_inspections.finish(VERIFIER_ID, status="ERROR", error=self.last_error)
                    if self.active_phase == "verifier":
                        self._show_phase("verifier")
                if self.live_inspections.get(GRADER_ID) is None:
                    self._coverage("Grader was not invoked; no verdict is available")
            finally:
                if opts.headless:
                    self.exit()

    app = RubricInspectorProbe()
    app.run(headless=opts.headless)
    if opts.headless:
        print("PASS: live Inspector probe" if app.completed else f"FAIL: {app.last_error}")
    return 0 if app.completed else 1


if __name__ == "__main__":
    try:
        sys.exit(inspector_main())
    except Exception as exc:
        print(f"Probe could not start: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc(limit=3)
        sys.exit(1)
