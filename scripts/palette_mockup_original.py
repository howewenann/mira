r"""Textual mockup of MIRA's current lifecycle and MCP palette.

Run from the repository root with:
    conda run --no-capture-output -n ai_agents python scripts\palette_mockup_original.py

The production status bar uses inline Rich styles for its semantic spans.  This
preview splits those spans into small widgets so the comparison variant can
change colors with TCSS alone while preserving MIRA's layout and wording.
"""

from __future__ import annotations

import sys
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Footer, Static


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ui.spinners import SPINNER_FRAMES  # noqa: E402


ACTIVE_STATES = {"starting", "running", "cancelling"}
STATE_SYMBOLS = {
    "ready": "●",
    "error": "×",
}

TOOL_ACTIVE_STATES = {"preparing", "running"}
TOOL_LABELS = {
    "preparing": "Preparing",
    "running": "Running",
    "completed": "Completed in",
    "cancelled": "Cancelled after",
    "failed": "Failed after",
}


class AnimatedState(Static):
    """One fixed-width lifecycle label driven entirely by CSS classes."""

    def __init__(self, state: str, *, prefix: str = "", **kwargs: object) -> None:
        classes = f"lifecycle {state} {kwargs.pop('classes', '')}".strip()
        super().__init__(classes=classes, **kwargs)
        self.state = state
        self.prefix = prefix
        self.spinner_index = 3

    def on_mount(self) -> None:
        self.refresh_label()

    def tick(self) -> None:
        if self.state in ACTIVE_STATES:
            self.spinner_index = (self.spinner_index + 1) % len(SPINNER_FRAMES)
            self.refresh_label()

    def refresh_label(self) -> None:
        symbol = (
            SPINNER_FRAMES[self.spinner_index]
            if self.state in ACTIVE_STATES
            else STATE_SYMBOLS[self.state]
        )
        self.update(f"{self.prefix}{symbol} {self.state.upper()}")


class ToolSample(Vertical):
    """MIRA-shaped tool bubble with one lifecycle state."""

    def __init__(self, state: str, elapsed: str, output: str = "") -> None:
        super().__init__(classes=f"message tool-call {state}")
        self.state = state
        self.elapsed = elapsed
        self.output_text = output
        self.border_title = "tool · eval"

    def compose(self) -> ComposeResult:
        yield Static(
            'call: {"description": "Check the implementation and tests"}',
            classes="tool-call-text",
        )
        if self.output_text:
            yield Static(self.output_text, classes="tool-output")
        with Horizontal(classes="tool-status-row"):
            yield ToolState(self.state, classes="tool-state")
            yield Static(self.elapsed, classes="tool-duration")


class ToolState(Static):
    """One tool lifecycle label using the same semantic color classes."""

    def __init__(self, state: str, **kwargs: object) -> None:
        classes = f"lifecycle {state} {kwargs.pop('classes', '')}".strip()
        super().__init__(classes=classes, **kwargs)
        self.state = state
        self.spinner_index = 3

    def on_mount(self) -> None:
        self.refresh_label()

    def tick(self) -> None:
        if self.state in TOOL_ACTIVE_STATES:
            self.spinner_index = (self.spinner_index + 1) % len(SPINNER_FRAMES)
            self.refresh_label()

    def refresh_label(self) -> None:
        prefix = f"{SPINNER_FRAMES[self.spinner_index]} " if self.state in TOOL_ACTIVE_STATES else ""
        self.update(f"{prefix}{TOOL_LABELS[self.state]}")


class StatusExamples(Vertical):
    """All user-visible lifecycle states on the production header background."""

    def compose(self) -> ComposeResult:
        yield Static("HEADER LIFECYCLE", classes="section-title")
        with Horizontal(classes="state-strip"):
            for state in ("starting", "ready", "running", "cancelling", "error"):
                yield AnimatedState(state, classes="state-sample")


class HeaderControlExamples(Vertical):
    """Goal and Plan controls shown against the real header background."""

    def compose(self) -> ComposeResult:
        yield Static("HEADER CONTROLS", classes="section-title")
        with Horizontal(classes="control-preview"):
            yield Static("MIRA | ACT | ● READY | Ctx  9%", classes="control-status")
            yield Button("Goal Draft", classes="artifact-control goal-control")
            yield Static("│", classes="control-separator")
            yield Button("! MCP 1/5", classes="mcp-control")
        with Horizontal(classes="control-preview"):
            yield Static("MIRA | PLAN | ● READY | Ctx  9%", classes="control-status")
            yield Button("Plan Draft", classes="artifact-control plan-control")
            yield Static("│", classes="control-separator")
            yield Button("! MCP 1/5", classes="mcp-control")


class TranscriptExamples(Vertical):
    """Representative consumers of the same semantic states."""

    def compose(self) -> ComposeResult:
        yield Static("TOOL LIFECYCLE", classes="section-title")
        with Horizontal(classes="tool-grid"):
            yield ToolSample("preparing", "· 00:02 elapsed")
            yield ToolSample("running", "· 00:06 elapsed")
            yield ToolSample("completed", "00:31", "Tests passed")
            yield ToolSample("cancelled", "00:12")
            yield ToolSample("failed", "00:12", "AssertionError")
        yield Static("OTHER ACTIVITY", classes="section-title activity-title")
        with Horizontal(classes="activity-row"):
            yield AnimatedState(
                "running",
                prefix="working...  ",
                classes="activity-status",
            )
            yield AnimatedState(
                "running",
                prefix="subagent researcher  ",
                classes="subagent-status",
            )
            yield AnimatedState(
                "cancelling",
                prefix="subagent reviewer  ",
                classes="subagent-status",
            )


class McpExamples(Vertical):
    """The existing MCP badge palette used as the proposed color source."""

    def compose(self) -> ComposeResult:
        yield Static("MCP REFERENCE PALETTE", classes="section-title")
        with Horizontal(classes="mcp-strip"):
            yield Static("AVAILABLE", classes="mcp-status-badge available")
            yield Static("WARNING", classes="mcp-status-badge warning")
            yield Static("FAILED", classes="mcp-status-badge failed")
            yield Static("DISABLED", classes="mcp-status-badge disabled")
            yield Static("TRANSIENT", classes="mcp-status-badge transient")


class PaletteMockup(App[None]):
    """Standalone visual comparison built from MIRA's TUI vocabulary."""

    TITLE = "MIRA · original palette"
    SUB_TITLE = "Press Q to close"
    BINDINGS = [("q", "quit", "Quit")]

    CSS = """
    Screen {
        background: #0c0f10;
        color: #e8edef;
    }

    #status-row {
        height: 1;
        width: 1fr;
        padding: 0 1;
        background: #17313a;
        color: #eef7f8;
    }

    #brand {
        width: 6;
        color: #d6fff6;
        text-style: bold;
    }

    #mode {
        width: 6;
        color: #d7dee2;
    }

    #header-state {
        width: 17;
    }

    #context {
        width: 1fr;
        color: #70d77a;
        text-style: bold;
    }

    #mcp-status-button {
        width: 13;
        min-width: 13;
        max-width: 13;
        height: 1;
        padding: 0;
        border: none;
        background: #5bb8b1;
        color: #081112;
        text-style: bold;
    }

    #mcp-status-button:focus {
        background: #7ce3dc;
        color: #081112;
    }

    #artifact-status-button {
        width: 13;
        min-width: 13;
        max-width: 13;
        height: 1;
        padding: 0;
        border: none;
        background: #10191b;
        color: #c58fd6;
        text-style: bold;
    }

    #header-control-separator {
        display: none;
        width: 1;
        height: 1;
    }

    #preview-scroll {
        height: 1fr;
        margin: 1;
        padding: 0 1;
        border: solid #2d5661;
        scrollbar-color: #5bb8b1;
        scrollbar-background: #122023;
    }

    .preview-section {
        height: auto;
        margin-bottom: 1;
    }

    .section-title {
        height: 1;
        margin-bottom: 1;
        color: #8fa3b8;
        text-style: bold;
    }

    .state-strip,
    .mcp-strip,
    .activity-row {
        height: 3;
        padding: 1;
        background: #17313a;
    }

    .control-preview {
        height: 1;
        margin-bottom: 1;
        padding: 0 1;
        background: #17313a;
        color: #eef7f8;
    }

    .control-status {
        width: 1fr;
        height: 1;
    }

    .artifact-control,
    .mcp-control {
        width: 13;
        min-width: 13;
        max-width: 13;
        height: 1;
        padding: 0;
        border: none;
        text-style: bold;
    }

    .artifact-control {
        background: #10191b;
    }

    .artifact-control.goal-control { color: #c58fd6; }
    .artifact-control.plan-control { color: #a0b86b; }

    .mcp-control {
        background: #5bb8b1;
        color: #081112;
    }

    .control-separator {
        display: none;
        width: 1;
        height: 1;
    }

    .state-sample {
        width: 1fr;
        text-align: center;
    }

    .lifecycle {
        height: 1;
        text-style: bold;
    }

    /* Current MIRA lifecycle colors from status_bar.py/terminal_colors.py. */
    .lifecycle.starting { color: #5bb8b1; }
    .lifecycle.running { color: #579a96; }
    .lifecycle.ready { color: #6fa884; }
    .lifecycle.cancelling { color: #b89b59; }
    .lifecycle.error { color: #b96562; }

    .lifecycle.preparing { color: #b89b59; }
    .lifecycle.completed { color: #6fa884; }
    .lifecycle.cancelled { color: #b89b59; }
    .lifecycle.failed { color: #b96562; }

    .tool-grid {
        height: auto;
    }

    .message.tool-call {
        width: 1fr;
        height: 8;
        margin-right: 1;
        padding: 0 1;
        border: solid #8fa3b8;
        background: #0c0f10;
    }

    .message.tool-call.error {
        margin-right: 0;
    }

    .tool-call-text {
        height: 2;
        color: #e8edef;
    }

    .tool-output {
        height: 1;
        color: #b8c1c7;
    }

    .tool-status-row {
        height: 1;
        margin-top: 1;
    }

    .tool-state {
        width: 17;
    }

    .tool-duration {
        width: 1fr;
        color: #98a5ac;
    }

    .activity-status,
    .subagent-status {
        width: 1fr;
    }

    .activity-title {
        margin-top: 1;
    }

    /* These currently use Rich's named yellow rather than the tool palette. */
    .activity-status,
    .subagent-status.running,
    .subagent-status.cancelling {
        color: yellow;
    }

    .mcp-status-badge {
        width: 1fr;
        height: 1;
        margin-right: 1;
        text-align: center;
        text-style: bold;
    }

    .mcp-status-badge.transient {
        margin-right: 0;
    }

    .mcp-status-badge.available { background: #194b32; color: #a9f0b1; }
    .mcp-status-badge.warning { background: #5a451e; color: #fff0ad; }
    .mcp-status-badge.failed { background: #5b272a; color: #ffb3b3; }
    .mcp-status-badge.disabled { background: #30383d; color: #aeb8be; }
    .mcp-status-badge.transient { background: #263f60; color: #bcd1f5; }

    #telemetry-row {
        height: 1;
        width: 1fr;
        padding: 0 1;
        background: #101b1f;
        color: #b8c1c7;
    }

    #model {
        width: 1fr;
        color: #7ce3dc;
    }

    #telemetry {
        width: auto;
        color: #b8c1c7;
    }

    Footer {
        display: none;
    }
    """

    def compose(self) -> ComposeResult:
        with Horizontal(id="status-row"):
            yield Static("MIRA", id="brand")
            yield Static("| Act |", id="mode")
            yield AnimatedState("running", id="header-state")
            yield Static("| Ctx █░░░░░░░░░  9%  (5.7k/64.0k)", id="context")
            yield Button("Goal Draft", id="artifact-status-button")
            yield Static("│", id="header-control-separator")
            yield Button("! MCP 1/5", id="mcp-status-button")
        with VerticalScroll(id="preview-scroll"):
            yield HeaderControlExamples(classes="preview-section")
            yield StatusExamples(classes="preview-section")
            yield TranscriptExamples(classes="preview-section")
            yield McpExamples(classes="preview-section")
        with Horizontal(id="telemetry-row"):
            yield Static("model: [lmstudio-gemma] openai:google/gemma-4-12b", id="model")
            yield Static("In 0 Out 0  |  Turns 0  |  00:47", id="telemetry")
        yield Footer()

    def on_mount(self) -> None:
        self.set_interval(0.12, self.tick_spinners)

    def tick_spinners(self) -> None:
        for badge in self.query(AnimatedState):
            badge.tick()
        for state in self.query(ToolState):
            state.tick()


if __name__ == "__main__":
    PaletteMockup().run()
