"""One incremental native Workflow execution tree for the chat transcript."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from rich.markup import escape
from rich.pretty import Pretty, pretty_repr
from rich.style import Style
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Static, Tree
from textual.widgets.tree import TreeNode

from core.execution.streams.rubric import elapsed_ms
from ui.shared.terminal.spinners import SPINNER_FRAMES
from ui.textual.widgets.tool_bubble import tool_lifecycle_status


STATUS_WIDTH = 31
WAITING_COLOR = "#e2b44f"


@dataclass(slots=True)
class WorkflowAgentView:
    """Process-local display state for one observed agent execution."""

    inspection_id: str
    name: str
    task: str
    status: str = "RUNNING"
    result: str = ""
    error: str = ""
    started_at: float = field(default_factory=time.monotonic)
    duration_ms: int | None = None
    tree_node: TreeNode[Any] | None = None


@dataclass(slots=True)
class WorkflowTaskView:
    """Process-local display state for one native root task invocation."""

    task_id: str
    name: str
    step: int
    input_state: Any
    status: str = "RUNNING"
    result: Any = None
    result_available: bool = False
    error: str = ""
    started_at: float = field(default_factory=time.monotonic)
    duration_ms: int | None = None
    agents: list[WorkflowAgentView] = field(default_factory=list)
    tree_node: TreeNode[Any] | None = None


@dataclass(frozen=True, slots=True)
class WorkflowStepView:
    """Structural native execution-batch row."""

    step: int


WorkflowRunView = WorkflowTaskView | WorkflowAgentView


def _tree_depth(node: TreeNode[Any]) -> int:
    depth = 0
    parent = node.parent
    while parent is not None:
        depth += 1
        parent = parent.parent
    return max(0, depth - 1)


def _lifecycle_text(view: WorkflowRunView, frame: int) -> Text:
    waiting = view.status == "WAITING"
    return tool_lifecycle_status(
        draft=False,
        started_at=view.started_at,
        duration_ms=view.duration_ms,
        is_error=view.status == "ERROR",
        terminal_status="cancelled" if view.status == "CANCELLED" else "",
        active_status="Waiting" if waiting else "",
        active_color=WAITING_COLOR if waiting else "",
        show_spinner=not waiting,
        frame=frame,
    )


class WorkflowTree(Tree[Any]):
    """Native Tree with aligned ToolBubble-style lifecycle text."""

    def __init__(self) -> None:
        super().__init__("execution", id="workflow-tree")
        self.show_root = False
        self.show_guides = True
        self.auto_expand = False
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
        if not isinstance(data, (WorkflowTaskView, WorkflowAgentView)):
            return left

        status = _lifecycle_text(data, self._spinner_frame)
        guide_width = _tree_depth(node) * int(self.guide_depth)
        available = max(1, self.size.width - guide_width - 1)
        status_column = max(1, available - STATUS_WIDTH)
        if left.cell_len >= status_column:
            left.truncate(max(1, status_column - 2), overflow="ellipsis")
        left.append(" " * max(2, status_column - left.cell_len))
        left.append_text(status)
        return left


class WorkflowNodeSelected(Message):
    """Request the Workflow-node summary Inspector."""

    def __init__(self, view: WorkflowTaskView) -> None:
        super().__init__()
        self.view = view


class WorkflowAgentSelected(Message):
    """Request the ordinary full agent Inspector."""

    def __init__(self, inspection_id: str) -> None:
        super().__init__()
        self.inspection_id = inspection_id


class WorkflowFinalStateSelected(Message):
    """Request the Workflow final-state Inspector."""

    def __init__(self, workflow_name: str, final_state: Any) -> None:
        super().__init__()
        self.workflow_name = workflow_name
        self.final_state = final_state


class WorkflowValueBubble(Vertical):
    """One retained Workflow value with its own clipboard action."""

    FEEDBACK_SECONDS = 1.5

    def __init__(self, title: str, value: Any) -> None:
        super().__init__(classes="message command workflow-value")
        self.border_title = escape(title)
        self.copy_text = pretty_repr(value, expand_all=True)
        self._body = Static(
            Pretty(value, expand_all=False),
            classes="workflow-value-body",
        )
        self._copy_button = Button(
            "Copy",
            classes="workflow-value-copy",
            compact=True,
        )
        self._feedback_version = 0

    def compose(self) -> ComposeResult:
        yield self._body
        with Horizontal(classes="workflow-value-actions"):
            yield self._copy_button

    @on(Button.Pressed, ".workflow-value-copy")
    def copy_value(self, event: Button.Pressed) -> None:
        """Copy the complete value and show transient feedback."""
        event.stop()
        self.app.copy_to_clipboard(self.copy_text)
        self._feedback_version += 1
        version = self._feedback_version
        self._copy_button.label = "Copied"
        self.set_timer(
            self.FEEDBACK_SECONDS,
            lambda: self._restore_copy_label(version),
        )

    def _restore_copy_label(self, version: int) -> None:
        if self._feedback_version == version and self._copy_button.is_mounted:
            self._copy_button.label = "Copy"


class WorkflowTreeBubble(Vertical):
    """One growing main-chat bubble for a complete Workflow execution."""

    def __init__(self, workflow_id: str, workflow_name: str) -> None:
        super().__init__(classes="message workflow-tree-bubble")
        self.workflow_id = workflow_id
        self.workflow_name = workflow_name or "workflow"
        self.border_title = f"workflow - {self.workflow_name}"
        self.workflow_tree = WorkflowTree()
        self.final_button = Button(
            "Final state",
            classes="workflow-final-state",
            compact=True,
        )
        self.overall_status = Static(classes="workflow-overall-status")
        self.footer = Horizontal(
            self.final_button,
            self.overall_status,
            classes="workflow-footer",
        )
        self.footer.display = False
        self.final_button.display = False
        self.started_at = time.monotonic()
        self.duration_ms: int | None = None
        self.status = "RUNNING"
        self.final_state: Any = None
        self.final_state_available = False
        self.step_nodes: dict[int, TreeNode[Any]] = {}
        self.task_views: dict[str, WorkflowTaskView] = {}
        self.agent_views: dict[str, WorkflowAgentView] = {}
        self._pending_agents: dict[str, list[tuple[Any, ...]]] = {}

    def compose(self) -> ComposeResult:
        yield self.workflow_tree
        yield self.footer

    def start_task(
        self,
        task_id: str,
        name: str,
        step: int,
        input_state: Any,
    ) -> WorkflowTaskView:
        view = self.task_views.get(task_id)
        if view is not None:
            view.status = "RUNNING"
            view.error = ""
            self._refresh_task(view)
            return view

        view = WorkflowTaskView(task_id, name, step, input_state)
        view.tree_node = self._step(step).add(
            Text(name, style="bold #dce7ea"),
            data=view,
            expand=True,
            allow_expand=True,
        )
        self.task_views[task_id] = view
        for pending in self._pending_agents.pop(task_id, []):
            self.start_agent(task_id, *pending)
        self._fit_tree_height()
        self.workflow_tree.refresh(layout=True)
        return view

    def wait_task(self, task_id: str) -> WorkflowTaskView | None:
        return self._set_task_status(task_id, "WAITING")

    def resume_task(self, task_id: str) -> WorkflowTaskView | None:
        return self._set_task_status(task_id, "RUNNING")

    def finish_task(
        self,
        task_id: str,
        *,
        status: str,
        error: str,
        result: Any,
        result_available: bool,
    ) -> WorkflowTaskView | None:
        view = self.task_views.get(task_id)
        if view is None:
            return None
        view.status = status
        view.error = error
        view.result = result
        view.result_available = result_available
        view.duration_ms = elapsed_ms(view.started_at)
        self._refresh_task(view)
        return view

    def start_agent(
        self,
        task_id: str,
        inspection_id: str,
        name: str,
        task: str,
        resumed: bool = False,
    ) -> WorkflowAgentView | None:
        owner = self.task_views.get(task_id)
        if owner is None:
            self._pending_agents.setdefault(task_id, []).append(
                (inspection_id, name, task, resumed)
            )
            return None
        agent = self.agent_views.get(inspection_id)
        if agent is not None:
            agent.status = "RUNNING"
            agent.error = ""
            agent.duration_ms = None
            self.workflow_tree.refresh()
            return agent

        agent = WorkflowAgentView(inspection_id, name, task)
        assert owner.tree_node is not None
        agent.tree_node = owner.tree_node.add_leaf(
            Text(name, style="#c9d5d8"),
            data=agent,
        )
        owner.tree_node.allow_expand = True
        owner.tree_node.expand()
        owner.agents.append(agent)
        self.agent_views[inspection_id] = agent
        self._fit_tree_height()
        self.workflow_tree.refresh(layout=True)
        return agent

    def wait_agent(self, inspection_id: str) -> WorkflowAgentView | None:
        agent = self.agent_views.get(inspection_id)
        if agent is not None:
            agent.status = "WAITING"
            self.workflow_tree.refresh()
        return agent

    def finish_agent(
        self,
        inspection_id: str,
        *,
        status: str,
        result: str,
        error: str,
    ) -> WorkflowAgentView | None:
        agent = self.agent_views.get(inspection_id)
        if agent is None:
            return None
        agent.status = status
        agent.result = result
        agent.error = error
        agent.duration_ms = elapsed_ms(agent.started_at)
        self.workflow_tree.refresh()
        return agent

    def complete(self, final_state: Any, *, available: bool) -> None:
        self.status = "DONE"
        self.duration_ms = elapsed_ms(self.started_at)
        self.final_state = final_state
        self.final_state_available = available
        self._finish_footer()

    def cancel(self, error: str = "") -> None:
        self.status = "ERROR" if error else "CANCELLED"
        self.duration_ms = elapsed_ms(self.started_at)
        for task in self.task_views.values():
            if task.status in {"RUNNING", "WAITING"}:
                task.status = self.status
                task.error = error
                task.duration_ms = elapsed_ms(task.started_at)
                self._sync_task_disclosure(task)
        for agent in self.agent_views.values():
            if agent.status in {"RUNNING", "WAITING"}:
                agent.status = self.status
                agent.error = error
                agent.duration_ms = elapsed_ms(agent.started_at)
        self._finish_footer()
        self.workflow_tree.refresh()

    def tick(self) -> None:
        if self.duration_ms is None:
            self.workflow_tree.advance_spinner()

    def _step(self, step: int) -> TreeNode[Any]:
        node = self.step_nodes.get(step)
        if node is not None:
            return node
        node = self.workflow_tree.root.add(
            Text(f"Step {step}", style="bold #B7A4E8"),
            data=WorkflowStepView(step),
            expand=True,
        )
        self.workflow_tree.root.expand()
        self.step_nodes[step] = node
        return node

    def _set_task_status(
        self,
        task_id: str,
        status: str,
    ) -> WorkflowTaskView | None:
        view = self.task_views.get(task_id)
        if view is None:
            return None
        view.status = status
        view.error = ""
        self._refresh_task(view)
        return view

    def _refresh_task(self, view: WorkflowTaskView) -> None:
        self._sync_task_disclosure(view)
        self.workflow_tree.refresh()

    @staticmethod
    def _sync_task_disclosure(view: WorkflowTaskView) -> None:
        if view.tree_node is None:
            return
        terminal = view.status in {"DONE", "ERROR", "CANCELLED"}
        view.tree_node.allow_expand = not terminal or bool(view.tree_node.children)

    def _finish_footer(self) -> None:
        self.footer.display = True
        self.final_button.display = self.final_state_available
        self.overall_status.update(
            tool_lifecycle_status(
                draft=False,
                started_at=self.started_at,
                duration_ms=self.duration_ms,
                is_error=self.status == "ERROR",
                terminal_status="cancelled" if self.status == "CANCELLED" else "",
            )
        )

    def _fit_tree_height(self) -> None:
        self.workflow_tree.styles.height = max(2, self._visible_rows() + 1)

    def _visible_rows(self) -> int:
        def count_visible(node: TreeNode[Any]) -> int:
            total = 0
            for child in node.children:
                total += 1
                if child.is_expanded:
                    total += count_visible(child)
            return total

        return count_visible(self.workflow_tree.root)

    @on(Tree.NodeExpanded, "#workflow-tree")
    @on(Tree.NodeCollapsed, "#workflow-tree")
    def tree_shape_changed(self, _event: Message | None = None) -> None:
        self._fit_tree_height()

    @on(Tree.NodeSelected, "#workflow-tree")
    def tree_selected(self, event: Tree.NodeSelected[Any]) -> None:
        data = event.node.data
        if isinstance(data, WorkflowTaskView):
            self.post_message(WorkflowNodeSelected(data))
        elif isinstance(data, WorkflowAgentView):
            self.post_message(WorkflowAgentSelected(data.inspection_id))

    @on(Button.Pressed, ".workflow-final-state")
    def final_state_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self.final_state_available:
            self.post_message(
                WorkflowFinalStateSelected(
                    self.workflow_name,
                    self.final_state,
                )
            )


__all__ = [
    "WorkflowAgentSelected",
    "WorkflowAgentView",
    "WorkflowFinalStateSelected",
    "WorkflowNodeSelected",
    "WorkflowStepView",
    "WorkflowTaskView",
    "WorkflowTree",
    "WorkflowTreeBubble",
    "WorkflowValueBubble",
]
