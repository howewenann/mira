"""Rich-based terminal renderer for MIRA."""

from __future__ import annotations

import asyncio
import json
import re
import sys
from itertools import cycle
from pathlib import Path
from typing import Any

from faker import Faker
from prompt_toolkit import PromptSession
from prompt_toolkit.shortcuts import choice
from pyfiglet import Figlet
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

DEFAULT_TOOL_OUTPUT_CHARS = 240
SUBAGENT_COLOURS = ["magenta", "yellow", "green", "#00FFFF", "blue"]
SPINNER_FRAMES = ["-", "\\", "|", "/"]
MIRA_CYAN = "cyan"
MIRA_TITLE = "bold white"


class Renderer:
    """Render all terminal output for MIRA.

    The runner sends small events to this class while DeepAgents streams. The
    renderer keeps the terminal-specific state here so the runtime can stay
    focused on agent events rather than Rich panels, live displays, and colors.
    """

    def __init__(self, tool_output_chars: int = DEFAULT_TOOL_OUTPUT_CHARS) -> None:
        """Create a renderer with an optional tool-output truncation limit."""
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")

        self.tool_output_chars = tool_output_chars
        self.console = Console(stderr=True)
        self.fake = Faker()
        self.subagent_colours: dict[str, str] = {}
        self.subagent_labels: dict[int, str] = {}
        self.subagent_suffixes: set[str] = set()
        self.subagent_blocks: dict[str, dict[str, str]] = {}
        self.live: Live | None = None
        self.spinner_index = 0
        self.colours = cycle(SUBAGENT_COLOURS)

        # Live panels are updated token-by-token. Keep the source text separate
        # from Rich renderables so streamed content can be re-rendered safely.
        self._thinking_text = ""
        self._thinking_live: Live | None = None
        self._response_text = ""
        self._response_live: Live | None = None

    def splash(self, model_name: str, session_id: str, workspace: str | Path) -> None:
        """Print the startup banner, session metadata, and first-use hints."""
        wordmark = Figlet(font="blocky").renderText("MIRA").rstrip()
        logo_width = max((len(line.rstrip()) for line in wordmark.splitlines()), default=0)
        border = Text("=" * logo_width, style=MIRA_CYAN)
        divider = Text("-" * logo_width, style=MIRA_CYAN)

        self.console.print(border)
        self.console.print(Text(wordmark, style=MIRA_CYAN))
        self.console.print()
        self.console.print(Text("Minimal Iterative Reasoning Agent", style=MIRA_TITLE))
        self.console.print(divider)
        self.console.print(self._label_text("session", session_id))
        self.console.print(self._label_text("model", model_name))
        self.console.print(self._label_text("workspace", workspace))
        self.console.print()
        self.console.print(self._hint_text())
        self.console.print()

    def newline(self) -> None:
        """Print a blank line after Ctrl+C/EOF so the shell prompt is clean."""
        self.console.print()

    # Thinking (streaming)

    def reasoning_delta(self, delta: str) -> None:
        """Append a streamed reasoning token to the live thinking panel."""
        cleaned = re.sub(r"</?[^>]+>", "", delta)
        if not cleaned:
            return

        self._stop_response_live()

        if self._thinking_live is None:
            self._thinking_text = ""
            self._thinking_live = Live(
                self._render_thinking(),
                console=self.console,
                refresh_per_second=12,
                transient=False,
            )
            self._thinking_live.start()

        self._thinking_text += cleaned
        self._thinking_live.update(self._render_thinking())

    def _render_thinking(self) -> Panel:
        """Build the current thinking panel as literal text."""
        return Panel(
            Text(self._thinking_text),
            title=f"[bold {MIRA_CYAN}]Thinking[/bold {MIRA_CYAN}]",
            title_align="left",
            border_style=f"dim {MIRA_CYAN}",
        )

    def _stop_thinking_live(self) -> None:
        """Finalize and clear the thinking live display if it is active."""
        if self._thinking_live is None:
            return
        self._thinking_live.update(self._render_thinking())
        self._thinking_live.stop()
        self._thinking_live = None
        self._thinking_text = ""

    # Response (streaming)

    def text_delta(self, delta: str) -> None:
        """Append a streamed answer token to the live response panel.

        Rich markup from the model is treated as plain text by
        ``_render_response``. This protects file contents such as
        ``[tool]`` or ``[red]`` from being interpreted as terminal styling.
        """
        if not delta:
            return

        self._stop_thinking_live()

        if self._response_live is None:
            self._response_text = ""
            self._response_live = Live(
                self._render_response(),
                console=self.console,
                refresh_per_second=12,
                transient=False,
            )
            self._response_live.start()

        self._response_text += delta
        self._response_live.update(self._render_response())

    def _render_response(self) -> Panel:
        """Build the current response panel as literal text."""
        return Panel(
            Text(self._response_text),
            title=f"[bold {MIRA_CYAN}]mira - response[/bold {MIRA_CYAN}]",
            title_align="left",
            border_style=MIRA_CYAN,
        )

    def _stop_response_live(self) -> None:
        """Finalize and clear the response live display if it is active."""
        if self._response_live is None:
            return
        self._response_live.update(self._render_response())
        self._response_live.stop()
        self._response_live = None
        self._response_text = ""

    # Tool calls

    def tool_call(self, name: str, args: Any) -> None:
        """Render a non-task tool call and its arguments."""
        self._stop_thinking_live()
        self._stop_response_live()

        text = Text()
        text.append("args: ", style=MIRA_CYAN)
        text.append(self.truncate(args), style="white")
        self.console.print(
            Panel(
                text,
                title=f"[bold {MIRA_CYAN}]mira - {name}[/bold {MIRA_CYAN}]",
                title_align="left",
                border_style=MIRA_CYAN,
            )
        )

    def tool_result(self, name: str, result: str) -> None:
        """Print a compact tool result for coordinator-level tool calls."""
        text = Text("  output: ", style="dim white")
        text.append(self.truncate(result), style="dim white")
        self.console.print(text)

    def plan(self, plan_id: int, text: str) -> None:
        """Render a saved planning-mode result."""
        self.console.print(
            Panel(
                Text(text),
                title=f"[bold {MIRA_CYAN}]mira - plan #{plan_id}[/bold {MIRA_CYAN}]",
                title_align="left",
                border_style=MIRA_CYAN,
            )
        )

    def no_plans(self) -> None:
        """Render the empty state for the saved-plan list."""
        self.console.print(
            Panel(
                Text("no saved plans"),
                title=f"[bold {MIRA_CYAN}]mira - plans[/bold {MIRA_CYAN}]",
                title_align="left",
                border_style=MIRA_CYAN,
            )
        )

    def delegation_started(self, calls: list[dict[str, Any]]) -> None:
        """Render the compact summary shown when MIRA delegates to subagents."""
        if not calls:
            return

        self._stop_thinking_live()
        self._stop_response_live()

        valid: list[str] = []
        errors: list[str] = []
        for call in calls:
            raw_args = call.get("args", {}) if isinstance(call, dict) else {}
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except (TypeError, json.JSONDecodeError):
                    errors.append(f"could not parse args: {str(raw_args)[:60]}")
                    continue
            args = raw_args if isinstance(raw_args, dict) else {}
            description = args.get("description")
            if description:
                valid.append(str(description))
            else:
                errors.append(f"missing description in args: {str(args)[:60]}")

        if not valid and not errors:
            return

        count = len(valid)
        label = "subagent" if count == 1 else "subagents"

        body = Text()
        body.append(f"delegating to {count} {label}...\n", style="bold yellow")
        for description in valid:
            body.append("  request: ", style=MIRA_CYAN)
            body.append(self.truncate(description) + "\n", style="white")
        for error in errors:
            body.append(f"  failed: {error}\n", style="dim red")

        self.console.print(
            Panel(
                body,
                title=f"[bold {MIRA_CYAN}]mira - task[/bold {MIRA_CYAN}]",
                title_align="left",
                border_style=MIRA_CYAN,
            )
        )

    # Subagents

    def subagent_started(self, subagent: str, task_input: str = "") -> None:
        """Create or replace the live block for a running subagent."""
        self.subagent_blocks[subagent] = {
            "request": task_input,
            "status": "RUNNING",
            "output": "",
            "colour": self.subagent_colour(subagent),
        }
        self.refresh_subagents()

    def subagent_finished(self, subagent: str, result: str = "") -> None:
        """Mark a subagent block as done and attach its final output."""
        block = self.subagent_blocks.setdefault(
            subagent,
            {
                "request": "",
                "status": "RUNNING",
                "output": "",
                "colour": self.subagent_colour(subagent),
            },
        )
        block["status"] = "DONE"
        block["output"] = result
        self.refresh_subagents()

    def finish_main(self) -> None:
        """Stop all live displays at the end of a top-level agent turn."""
        self._stop_thinking_live()
        self.stop_subagent_live()
        self._stop_response_live()

    async def ask_approvals(self, interrupts: list[Any]) -> list[dict[str, Any]]:
        """Ask the user to approve, edit, or reject interrupted tool actions."""
        decisions: list[dict[str, Any]] = []

        for interrupt in interrupts:
            for action in self.action_requests(interrupt):
                self.console.print()
                self.console.print(Panel(Text(self.action_text(action)), title="Approval", border_style=MIRA_CYAN))
                answer = await self._choice()

                if answer == "e":
                    decisions.append(await self.edit_decision(action))
                elif answer == "r":
                    decisions.append({"type": "reject"})
                else:
                    decisions.append({"type": "approve"})

        return decisions

    def action_requests(self, interrupt: Any) -> list[Any]:
        """Extract action requests from a LangGraph interrupt payload."""
        value = getattr(interrupt, "value", interrupt)

        if isinstance(value, dict) and value.get("action_requests"):
            return list(value["action_requests"])

        return [value]

    def action_text(self, action: Any) -> str:
        """Format an approval action as readable text."""
        if isinstance(action, dict):
            name = action.get("name", "tool")
            args = action.get("args", {})
            return f"{name}\n\n{json.dumps(args, indent=2)}"

        return str(action)

    async def edit_decision(self, action: Any) -> dict[str, Any]:
        """Prompt for edited JSON args and return a LangGraph decision."""
        if not isinstance(action, dict):
            return {"type": "reject"}

        edited_args = await self.prompt_json(action.get("args", {}))
        if edited_args is None:
            return {"type": "reject"}

        return {
            "type": "edit",
            "edited_action": {
                "name": action.get("name", "tool"),
                "args": edited_args,
            },
        }

    async def prompt_json(self, original: dict[str, Any]) -> dict[str, Any] | None:
        """Prompt for replacement action arguments and parse them as JSON."""
        session = PromptSession()
        prompt = "edited args JSON> "
        text = await asyncio.to_thread(session.prompt, prompt, default=json.dumps(original))

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            self.console.print("[dim]invalid JSON; rejecting action[/dim]")
            return None

        return parsed if isinstance(parsed, dict) else None

    async def ask_create_git_repo(self, message: str) -> bool:
        """Ask whether MIRA should initialize Git for the workspace."""
        answer = await self._prompt_choice(message, [("y", "yes"), ("n", "no")])
        return answer == "y"

    async def ask_continue_without_git(self, message: str) -> bool:
        """Ask whether startup should continue without Git protection."""
        answer = await self._prompt_choice(message, [("c", "continue"), ("e", "exit")])
        return answer == "c"

    async def _choice(self) -> str:
        """Run the blocking prompt-toolkit choice widget in a worker thread."""
        return await self._prompt_choice(
            "Approve this action?",
            [("y", "approve"), ("e", "edit"), ("r", "reject")],
        )

    async def _prompt_choice(self, message: str, options: list[tuple[str, str]]) -> str:
        """Run a blocking prompt-toolkit choice widget in a worker thread."""
        return await asyncio.to_thread(
            choice,
            message,
            options=options,
            show_frame=True,
        )

    def start_subagent_live(self) -> None:
        """Start the live group that contains subagent status panels."""
        if self.live is not None:
            return

        self.subagent_blocks = {}
        self.spinner_index = 0
        self.live = Live(
            self.render_subagents(),
            console=self.console,
            refresh_per_second=8,
            transient=False,
        )
        self.live.start()

    def stop_subagent_live(self) -> None:
        """Finalize and stop the subagent live group."""
        if self.live is None:
            return

        self.live.update(self.render_subagents())
        self.live.stop()
        self.live = None

    def tick_subagents(self) -> None:
        """Advance the spinner for running subagents."""
        if not self.has_running_subagents():
            return

        self.spinner_index = (self.spinner_index + 1) % len(SPINNER_FRAMES)
        self.refresh_subagents()

    def has_running_subagents(self) -> bool:
        """Return whether any subagent panel is still running."""
        return any(block["status"] == "RUNNING" for block in self.subagent_blocks.values())

    def refresh_subagents(self) -> None:
        """Update the live subagent group or print a static snapshot."""
        if self.live is not None:
            self.live.update(self.render_subagents())
            return

        self.console.print(self.render_subagents())

    def render_subagents(self) -> Group | str:
        """Build the renderable group containing all current subagent blocks."""
        if not self.subagent_blocks:
            return ""

        return Group(*(self.render_subagent(label, block) for label, block in self.subagent_blocks.items()))

    def render_subagent(self, label: str, block: dict[str, str]) -> Panel:
        """Build one subagent panel from its stored status block."""
        colour = block["colour"]
        status = block["status"]
        text = Text()

        if block.get("request"):
            text.append("request: ", style=MIRA_CYAN)
            text.append(self.truncate(block["request"]), style="white")
            text.append("\n")

        text.append("status: ", style=MIRA_CYAN)
        if status == "RUNNING":
            frame = SPINNER_FRAMES[self.spinner_index]
            text.append(f"{frame} RUNNING", style="bold yellow")
        else:
            text.append("DONE", style="bold green")

        if block.get("output"):
            text.append("\noutput: ", style=MIRA_CYAN)
            text.append(self.truncate_output(block.get("output", "")))

        return Panel(
            text,
            title=Text(f"subagent - {label}", style=f"bold {colour}"),
            title_align="left",
            border_style=colour,
        )

    def truncate_output(self, value: Any) -> Text:
        """Return tool output as literal Rich text, truncated when configured."""
        text = self.single_line(value)

        if self.tool_output_chars == 0 or len(text) <= self.tool_output_chars:
            return Text(text, style="white")

        rendered = Text(text[: self.tool_output_chars], style="white")
        rendered.append(" ... truncated ...", style="yellow")
        return rendered

    def truncate(self, value: Any) -> str:
        """Return a single-line string shortened to the configured display size."""
        text = self.single_line(value)

        if self.tool_output_chars == 0 or len(text) <= self.tool_output_chars:
            return text

        return text[: self.tool_output_chars] + " ... truncated ..."

    def single_line(self, value: Any) -> str:
        """Collapse whitespace so terminal panels stay compact."""
        return re.sub(r"\s+", " ", str(value or "")).strip()

    def subagent_label(self, subagent: Any) -> str:
        """Return a stable readable label for a subagent object."""
        key = id(subagent)
        if key not in self.subagent_labels:
            name = getattr(subagent, "name", "subagent")
            self.subagent_labels[key] = f"{name} [{self.subagent_suffix()}]"

        return self.subagent_labels[key]

    def subagent_suffix(self) -> str:
        """Generate a short unique suffix for a subagent label."""
        for _ in range(20):
            suffix = self.single_line(self.fake.word()).lower()
            if suffix and suffix not in self.subagent_suffixes:
                self.subagent_suffixes.add(suffix)
                return suffix

        suffix = self.single_line(self.fake.uuid4())[:8]
        self.subagent_suffixes.add(suffix)
        return suffix

    def subagent_colour(self, name: str) -> str:
        """Assign each subagent label a stable color for this process."""
        if name not in self.subagent_colours:
            self.subagent_colours[name] = next(self.colours)

        return self.subagent_colours[name]

    def _label_text(
        self,
        label: str,
        value: str | Path,
        *,
        label_style: str = "dim",
        value_style: str = "white",
    ) -> Text:
        """Build a metadata line without parsing the value as Rich markup."""
        text = Text()
        text.append(f"{label}: ", style=label_style)
        text.append(str(value), style=value_style)
        return text

    def _hint_text(self) -> Text:
        """Build the compact command hints shown below the splash metadata."""
        text = Text()
        text.append("enter", style="bold white")
        text.append(" to send  |  ", style="dim")
        text.append("/help", style=MIRA_CYAN)
        text.append("  |  ", style="dim")
        text.append("/tools", style=MIRA_CYAN)
        text.append("  |  ", style="dim")
        text.append("/plan", style=MIRA_CYAN)
        text.append("  |  ", style="dim")
        text.append("/act", style=MIRA_CYAN)
        text.append("  |  ", style="dim")
        text.append("/plans", style=MIRA_CYAN)
        text.append("  |  ", style="dim")
        text.append("↑/↓", style="bold white")
        text.append(" history  |  ctrl+c to quit", style="dim")
        return text
