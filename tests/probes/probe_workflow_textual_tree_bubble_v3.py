"""UI-only probe for the proposed MIRA Workflow Tree bubble.

Run from the MIRA repo root:

    python tests/probes/probe_workflow_textual_tree.py

The probe deliberately uses fake runtime events. It is only meant to answer:
"does this Textual interaction model actually feel good?"

It reuses MIRA's real Inspector, ChatLog bubbles, LiveInspectionStore and
mira.tcss. The entire tree lives inside one transcript bubble and is built incrementally over ~11 seconds.

Interaction contract under test
-------------------------------
- Step rows are structural only.
- Arrow glyphs alone expand/collapse parent rows.
- Clicking Workflow-node text opens the node Inspector without changing expansion.
- Running Workflow nodes may temporarily keep a disclosure glyph because agent children
  can still appear later.
- Once a Workflow node is terminal, it keeps a disclosure glyph only if it actually
  has child agent rows.
- Workflow nodes are selectable.
- Agent executions observed inside a Workflow node are child leaves.
- Click a Workflow node -> node Inspector with bubbles:
      Input state
      agent summary bubble(s), if applicable
      Result, once available
- Agent bubbles inside that node Inspector are read-only summaries.
- Close the Inspector to return to the main Tree.
- Click an agent leaf in the main Tree -> existing full agent Inspector.
- There is no Inspector -> Inspector navigation.
- HITL reuses the same Workflow Tree node.

Keys
----
q       quit
escape  close Inspector / return to Tree
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.pretty import Pretty
from rich.style import Style
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, ContentSwitcher, Static, Tree
from textual.widgets.tree import TreeNode

from core.execution.inspection.live import InspectionEvent, LiveInspectionStore
from core.execution.streams.rubric import format_elapsed
from ui.shared.terminal.colors import (
    TOOL_CANCELLED_COLOR,
    TOOL_COMPLETED_COLOR,
    TOOL_DURATION_COLOR,
    TOOL_FAILED_COLOR,
    TOOL_RUNNING_COLOR,
)
from ui.shared.terminal.spinners import SPINNER_FRAMES
from ui.textual.widgets.chat_log import ChatLog
from ui.textual.widgets.inspector import Inspector


REPO_ROOT = Path(__file__).resolve().parents[2]
MIRA_CSS = REPO_ROOT / "ui" / "textual" / "styles" / "mira.tcss"
_MISSING = object()
SIMULATION_SPEED = 5.0
STATUS_WIDTH = 31
STATUS_VERB_WIDTH = 17


@dataclass(slots=True)
class AgentView:
    inspection_id: str
    name: str
    task: str
    summary: str = ""
    status: str = "RUNNING"
    started_at: float = field(default_factory=time.monotonic)
    duration_ms: int | None = None
    tree_node: TreeNode[Any] | None = None


@dataclass(slots=True)
class WorkflowNodeView:
    task_id: str
    name: str
    step: int
    input_state: Any
    result: Any = _MISSING
    status: str = "RUNNING"
    started_at: float = field(default_factory=time.monotonic)
    duration_ms: int | None = None
    agents: list[AgentView] = field(default_factory=list)
    tree_node: TreeNode[Any] | None = None


@dataclass(frozen=True, slots=True)
class StepView:
    step: int


RunView = WorkflowNodeView | AgentView


def logical_elapsed_ms(view: RunView) -> int:
    real_seconds = max(0.0, time.monotonic() - view.started_at)
    return int(real_seconds * SIMULATION_SPEED * 1000)


def lifecycle_text(view: RunView, frame: int) -> Text:
    """ToolBubble vocabulary, with the elapsed token in one stable column."""
    text = Text()

    if view.duration_ms is not None:
        elapsed = format_elapsed(view.duration_ms)
        if view.status == "CANCELLED":
            verb = "Cancelled after"
            verb_color = TOOL_CANCELLED_COLOR
        elif view.status == "ERROR":
            verb = "Failed after"
            verb_color = TOOL_FAILED_COLOR
        else:
            verb = "Completed in"
            verb_color = TOOL_COMPLETED_COLOR
        text.append(verb, style=verb_color)
        text.append(" " * max(1, STATUS_VERB_WIDTH - len(verb)))
        text.append(elapsed, style=TOOL_DURATION_COLOR)
        return text

    elapsed = format_elapsed(logical_elapsed_ms(view))
    if view.status == "WAITING":
        verb = "Waiting ·"
        text.append(verb, style="#e2b44f")
        text.append(" " * max(1, STATUS_VERB_WIDTH - len(verb)))
        text.append(elapsed, style=TOOL_DURATION_COLOR)
        text.append(" elapsed", style=TOOL_DURATION_COLOR)
        return text

    spinner = SPINNER_FRAMES[frame % len(SPINNER_FRAMES)]
    text.append(f"{spinner} ", style=TOOL_DURATION_COLOR)
    verb = "Running ·"
    text.append(verb, style=TOOL_RUNNING_COLOR)
    used = 2 + len(verb)
    text.append(" " * max(1, STATUS_VERB_WIDTH - used))
    text.append(elapsed, style=TOOL_DURATION_COLOR)
    text.append(" elapsed", style=TOOL_DURATION_COLOR)
    return text


def tree_depth(node: TreeNode[Any]) -> int:
    depth = 0
    parent = node.parent
    while parent is not None:
        depth += 1
        parent = parent.parent
    return max(0, depth - 1)


class WorkflowTree(Tree[Any]):
    """Tree with one aligned lifecycle/timer column on the right."""

    def __init__(self, label: str, **kwargs: Any) -> None:
        super().__init__(label, **kwargs)
        self._spinner_frame = 0

    def advance_spinner(self) -> None:
        self._spinner_frame = (self._spinner_frame + 1) % len(SPINNER_FRAMES)
        self.refresh()

    def render_label(
        self,
        node: TreeNode[Any],
        base_style: Style,
        style: Style,
    ) -> Text:
        left = super().render_label(node, base_style, style)
        data = node.data
        if not isinstance(data, (WorkflowNodeView, AgentView)):
            return left

        status = lifecycle_text(data, self._spinner_frame)
        guide_width = tree_depth(node) * int(self.guide_depth)
        available = max(1, self.size.width - guide_width - 1)
        desired_left = max(left.cell_len + 2, available - STATUS_WIDTH)
        if left.cell_len < desired_left:
            left.append(" " * (desired_left - left.cell_len))
        else:
            left.append("  ")
        left.append_text(status)
        return left


class WorkflowTreeBubble(Vertical):
    """One MIRA-style transcript bubble containing the entire live Workflow."""

    def __init__(self, workflow_name: str) -> None:
        super().__init__(classes="message workflow-tree-bubble")
        self.border_title = f"workflow - {workflow_name}"

    def compose(self) -> ComposeResult:
        tree = WorkflowTree("execution", id="workflow-tree")
        tree.show_root = False
        tree.show_guides = True

        # Important interaction split:
        #
        #   click arrow glyph -> expand / collapse
        #   click node text   -> select / open Inspector
        #
        # Textual's Tree handles arrow clicks separately via its native
        # "toggle" metadata. Disabling auto_expand prevents a normal node
        # selection from also expanding/collapsing the parent.
        tree.auto_expand = False

        yield tree
        with Horizontal(id="workflow-footer"):
            yield Button(
                "Final state",
                id="workflow-final-state-probe",
                classes="workflow-final-state",
                compact=True,
            )
            yield Static("", id="probe-status")


class ProbeInspector(Inspector):
    """Existing Inspector plus one probe-only Workflow-node projection."""

    def open_workflow_node(self, node: WorkflowNodeView) -> None:
        """Render Input -> observed-agent bubbles -> Result."""
        self.stop_inspection()
        self.query_one("#inspector-title", Static).update(
            f"Inspector · workflow · {node.name}"
        )
        log = self.query_one("#inspector-log", ChatLog)
        log.clear_log()

        self._value_bubble(log, "Input state", node.input_state)

        # These are visual summaries only. They deliberately have no click-through.
        for agent in node.agents:
            if agent.status == "RUNNING":
                log.subagent_started(agent.name, agent.task)
            elif agent.status == "CANCELLED":
                log.subagent_cancelled(agent.name, agent.summary, agent.task)
            else:
                log.subagent_finished(agent.name, agent.summary, agent.task)

        if node.result is not _MISSING:
            self._value_bubble(log, "Result", node.result)

    def open_probe_final_state(self, value: Any) -> None:
        self.stop_inspection()
        self.query_one("#inspector-title", Static).update(
            "Inspector · workflow · Final state"
        )
        log = self.query_one("#inspector-log", ChatLog)
        log.clear_log()
        self._value_bubble(log, "Final state", value)

    @staticmethod
    def _value_bubble(log: ChatLog, title: str, value: Any) -> None:
        """Use MIRA's existing bubble machinery; no timestamps are injected."""
        renderable = Pretty(value, expand_all=False)
        workflow_value = getattr(log, "workflow_node_data", None)
        if callable(workflow_value):
            workflow_value(title, renderable)
            return

        # Compatibility fallback for a local refactor that removed
        # workflow_node_data(). This is probe-only and still uses ChatLog's
        # normal message-bubble primitive.
        add_block = getattr(log, "_add_block", None)
        if callable(add_block):
            log.finish_stream_phase()
            add_block(title, renderable, "message command")
            return

        log.command_output(renderable)


class WorkflowTreeProbe(App[None]):
    CSS_PATH = str(MIRA_CSS)

    CSS = """
    #workflow-probe-view {
        width: 1fr;
        height: 1fr;
    }

    #workflow-probe-scroll {
        width: 1fr;
        height: 1fr;
        padding: 1;
        scrollbar-color: #5bb8b1;
        scrollbar-background: #122023;
    }

    .workflow-tree-bubble {
        width: 1fr;
        height: auto;
        border: solid #5BB8B1;
        padding: 0 1;
        margin: 0 0 1 0;
        background: #0c0f10;
    }

    #workflow-tree {
        width: 1fr;
        height: 2;
        background: transparent;
        border: none;
        padding: 0;
        margin: 0;
        scrollbar-size: 0 0;
    }

    #workflow-tree > .tree--cursor {
        background: #17313a;
        color: #eef7f8;
    }

    #workflow-footer {
        width: 1fr;
        height: 1;
        min-height: 1;
        margin-top: 1;
        align-horizontal: left;
        display: none;
    }

    #workflow-final-state-probe {
        width: 14;
        min-width: 14;
        height: 1;
        min-height: 1;
        padding: 0 1;
    }

    #probe-status {
        width: 1fr;
        height: 1;
        color: #82909a;
        padding-left: 2;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "close_inspector", "Back", priority=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.store = LiveInspectionStore()
        self.step_nodes: dict[int, TreeNode[Any]] = {}
        self.workflow_nodes: dict[str, WorkflowNodeView] = {}
        self.agent_views: dict[str, AgentView] = {}
        self.active_node_inspector = ""
        self.final_state = {
            "topic": "isopods",
            "audience": "adults",
            "analysis": {
                "research": "complete",
                "critique": "complete",
                "verified": True,
            },
            "approved": True,
            "answer": "Final decision brief",
        }

    def compose(self) -> ComposeResult:
        with ContentSwitcher(initial="workflow-probe-view", id="transcript-viewport"):
            with Vertical(id="workflow-probe-view"):
                with VerticalScroll(id="workflow-probe-scroll"):
                    yield WorkflowTreeBubble("decision_brief")
            yield ProbeInspector(self.store, id="inspector")

    def on_mount(self) -> None:
        self.workflow_started_at = time.monotonic()
        self.set_interval(0.10, self._tick)
        self._schedule_demo()

    # ------------------------------------------------------------------
    # Fake progressive runtime
    # ------------------------------------------------------------------

    def _schedule_demo(self) -> None:
        events = [
            (0.30, self._prepare_start),
            (1.00, self._prepare_done),
            (1.50, self._parallel_nodes_start),
            (2.20, self._researcher_start),
            (2.55, self._critic_start),
            (2.90, self._checker_start),
            (4.00, self._researcher_done),
            (4.35, self._checker_done),
            (4.80, self._critic_done),
            # Parent nodes deliberately remain RUNNING after inner agents finish.
            (5.70, self._parallel_nodes_done),
            (6.30, self._merge_start),
            (6.85, self._merge_done),
            (7.30, self._approval_start),
            (7.90, self._approval_waiting),
            (9.00, self._approval_resume),
            (9.70, self._approval_done),
            (10.10, self._finalize_start),
            (10.65, self._finalize_done),
            (11.00, self._complete),
        ]
        for delay, callback in events:
            self.set_timer(delay, callback)

    def _prepare_start(self) -> None:
        self._start_node(
            "prepare-1",
            "prepare",
            1,
            {"topic": "isopods", "audience": "adults"},
        )

    def _prepare_done(self) -> None:
        self._finish_node("prepare-1", {"prepared": True})

    def _parallel_nodes_start(self) -> None:
        state = {"topic": "isopods", "audience": "adults", "prepared": True}
        self._start_node("analyse-1", "analyse", 2, state)
        self._start_node("verify-1", "verify", 2, state)

    def _researcher_start(self) -> None:
        self._start_agent(
            "analyse-1",
            "probe:researcher",
            "researcher",
            "Research the strongest evidence about the topic.",
        )

    def _critic_start(self) -> None:
        self._start_agent(
            "analyse-1",
            "probe:critic",
            "critic",
            "Challenge the assumptions and identify weak claims.",
        )

    def _checker_start(self) -> None:
        self._start_agent(
            "verify-1",
            "probe:checker",
            "checker",
            "Independently verify the important factual claims.",
        )

    def _researcher_done(self) -> None:
        self._finish_agent(
            "probe:researcher",
            "Found the strongest supporting evidence and two caveats.",
        )

    def _checker_done(self) -> None:
        self._finish_agent(
            "probe:checker",
            "Verified the main factual claims.",
        )

    def _critic_done(self) -> None:
        self._finish_agent(
            "probe:critic",
            "Identified one weak assumption and suggested a safer framing.",
        )

    def _parallel_nodes_done(self) -> None:
        self._finish_node(
            "analyse-1",
            {"analysis": "Research + critique merged by trailing Python."},
        )
        self._finish_node("verify-1", {"verified": True})

    def _merge_start(self) -> None:
        self._start_node(
            "merge-1",
            "merge",
            3,
            {
                "analysis": "Research + critique merged by trailing Python.",
                "verified": True,
            },
        )

    def _merge_done(self) -> None:
        self._finish_node("merge-1", {"draft_ready": True})

    def _approval_start(self) -> None:
        self._start_node(
            "approval-1",
            "approval",
            4,
            {"draft_ready": True, "approved": False},
        )

    def _approval_waiting(self) -> None:
        self._set_node_status("approval-1", "WAITING")

    def _approval_resume(self) -> None:
        # Same native task identity -> same Tree node.
        self._set_node_status("approval-1", "RUNNING")

    def _approval_done(self) -> None:
        self._finish_node("approval-1", {"approved": True})

    def _finalize_start(self) -> None:
        self._start_node(
            "finalize-1",
            "finalize",
            5,
            {"approved": True, "draft_ready": True},
        )

    def _finalize_done(self) -> None:
        self._finish_node(
            "finalize-1",
            {"answer": "Final decision brief"},
        )

    def _complete(self) -> None:
        footer = self.query_one("#workflow-footer", Horizontal)
        footer.styles.display = "block"
        elapsed_ms = int(
            (time.monotonic() - self.workflow_started_at)
            * SIMULATION_SPEED
            * 1000
        )
        self.query_one("#probe-status", Static).update(
            Text.assemble(
                ("Completed in", TOOL_COMPLETED_COLOR),
                (f" {format_elapsed(elapsed_ms)}", TOOL_DURATION_COLOR),
            )
        )

    # ------------------------------------------------------------------
    # Tree projection
    # ------------------------------------------------------------------

    def _step(self, step: int) -> TreeNode[Any]:
        if step in self.step_nodes:
            return self.step_nodes[step]
        tree = self.query_one("#workflow-tree", WorkflowTree)
        node = tree.root.add(
            Text(f"Step {step}", style="bold #B7A4E8"),
            data=StepView(step),
            expand=True,
        )
        tree.root.expand()
        self.step_nodes[step] = node
        self._fit_tree_height()
        return node

    def _start_node(self, task_id: str, name: str, step: int, input_state: Any) -> None:
        existing = self.workflow_nodes.get(task_id)
        if existing is not None:
            existing.status = "RUNNING"
            self._refresh_node(existing)
            return

        view = WorkflowNodeView(task_id, name, step, input_state)
        view.tree_node = self._step(step).add(
            Text(name, style="bold #dce7ea"),
            data=view,
            expand=True,
        )
        self.workflow_nodes[task_id] = view
        self._fit_tree_height()
        self._refresh_open_node_inspector(task_id)

    def _finish_node(self, task_id: str, result: Any) -> None:
        view = self.workflow_nodes[task_id]
        view.result = result
        view.status = "DONE"
        view.duration_ms = logical_elapsed_ms(view)
        self._refresh_node(view)

    def _set_node_status(self, task_id: str, status: str) -> None:
        view = self.workflow_nodes[task_id]
        view.status = status
        self._refresh_node(view)

    def _start_agent(
        self,
        task_id: str,
        inspection_id: str,
        name: str,
        task: str,
    ) -> None:
        owner = self.workflow_nodes[task_id]
        if owner.tree_node is None:
            return

        agent = AgentView(inspection_id, name, task)
        agent.tree_node = owner.tree_node.add_leaf(
            Text(name, style="#c9d5d8"),
            data=agent,
        )

        # A real child now exists, so this Workflow node is meaningfully
        # expandable.
        owner.tree_node.allow_expand = True
        owner.tree_node.expand()
        owner.agents.append(agent)
        self.agent_views[inspection_id] = agent

        # Populate the actual LiveInspectionStore used by the normal Inspector.
        self.store.start(inspection_id, name, task, inspection_type="subagent")
        self.store.append(
            inspection_id,
            InspectionEvent("reasoning", text="Working through the assigned task..."),
        )
        self._fit_tree_height()
        self._refresh_open_node_inspector(task_id)

    def _finish_agent(self, inspection_id: str, summary: str) -> None:
        agent = self.agent_views[inspection_id]
        agent.summary = summary
        agent.status = "DONE"
        agent.duration_ms = logical_elapsed_ms(agent)

        self.store.finish(
            inspection_id,
            status="DONE",
            final_response=summary,
        )

        owner = self._owner_of_agent(inspection_id)
        if owner is not None:
            self._refresh_open_node_inspector(owner.task_id)
        self.query_one("#workflow-tree", WorkflowTree).refresh()

    def _owner_of_agent(self, inspection_id: str) -> WorkflowNodeView | None:
        for view in self.workflow_nodes.values():
            if any(a.inspection_id == inspection_id for a in view.agents):
                return view
        return None

    def _refresh_node(self, view: WorkflowNodeView) -> None:
        self._sync_node_disclosure(view)
        self.query_one("#workflow-tree", WorkflowTree).refresh()
        self._refresh_open_node_inspector(view.task_id)

    @staticmethod
    def _sync_node_disclosure(view: WorkflowNodeView) -> None:
        """Remove meaningless arrows from terminal Workflow leaves.

        While a node is RUNNING or WAITING, it remains expandable because agent
        children may still appear later. Once terminal, the arrow is retained
        only when the node actually has child agent rows.
        """
        tree_node = view.tree_node
        if tree_node is None:
            return

        terminal = view.status in {"DONE", "ERROR", "CANCELLED"}
        has_children = bool(tree_node.children)
        tree_node.allow_expand = (not terminal) or has_children

    def _refresh_open_node_inspector(self, task_id: str) -> None:
        if self.active_node_inspector != task_id:
            return
        view = self.workflow_nodes.get(task_id)
        if view is not None:
            self.query_one("#inspector", ProbeInspector).open_workflow_node(view)

    def _fit_tree_height(self) -> None:
        """Grow the Tree with the transcript bubble; no inner tree scroller."""
        rows = (
            len(self.step_nodes)
            + len(self.workflow_nodes)
            + len(self.agent_views)
        )
        self.query_one("#workflow-tree", WorkflowTree).styles.height = max(2, rows + 1)

    def _tick(self) -> None:
        self.query_one("#workflow-tree", WorkflowTree).advance_spinner()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    @on(Tree.NodeSelected, "#workflow-tree")
    def tree_selected(self, event: Tree.NodeSelected[Any]) -> None:
        data = event.node.data

        if isinstance(data, StepView):
            # Step rows are structural only.
            # Clicking their text does nothing; the native arrow glyph owns
            # expand/collapse. Keyboard Space still toggles the selected row.
            return

        if isinstance(data, WorkflowNodeView):
            self.active_node_inspector = data.task_id
            self.query_one("#inspector", ProbeInspector).open_workflow_node(data)
            self.query_one("#transcript-viewport", ContentSwitcher).current = "inspector"
            return

        if isinstance(data, AgentView):
            self.active_node_inspector = ""
            inspector = self.query_one("#inspector", ProbeInspector)
            if inspector.open(data.inspection_id):
                self.query_one("#transcript-viewport", ContentSwitcher).current = (
                    "inspector"
                )

    @on(Inspector.Closed)
    def inspector_closed(self, event: Inspector.Closed) -> None:
        event.stop()
        self._return_to_tree()

    @on(Button.Pressed, "#workflow-final-state-probe")
    def final_state_selected(self, event: Button.Pressed) -> None:
        event.stop()
        self.active_node_inspector = ""
        self.query_one("#inspector", ProbeInspector).open_probe_final_state(
            self.final_state
        )
        self.query_one("#transcript-viewport", ContentSwitcher).current = "inspector"

    def action_close_inspector(self) -> None:
        switcher = self.query_one("#transcript-viewport", ContentSwitcher)
        if switcher.current == "inspector":
            self._return_to_tree()

    def _return_to_tree(self) -> None:
        self.query_one("#inspector", ProbeInspector).stop_inspection()
        self.active_node_inspector = ""
        self.query_one("#transcript-viewport", ContentSwitcher).current = (
            "workflow-probe-view"
        )
        self.call_after_refresh(lambda: self.query_one("#workflow-tree", WorkflowTree).focus())


if __name__ == "__main__":
    WorkflowTreeProbe().run()
