import json
import re
import sys
from itertools import cycle

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


class Renderer:
    def __init__(self, tool_output_chars: int = DEFAULT_TOOL_OUTPUT_CHARS) -> None:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")

        self.tool_output_chars = tool_output_chars
        self.console = Console(stderr=True)
        self.fake = Faker()
        self.subagent_colours = {}
        self.subagent_labels = {}
        self.subagent_suffixes = set()
        self.subagent_sections = set()
        self.subagent_blocks = {}
        self.live = None
        self.spinner_index = 0
        self.colours = cycle(SUBAGENT_COLOURS)
        self.section = None

        # live panels for thinking and response
        self._thinking_text = ""
        self._thinking_live = None
        self._response_text = ""
        self._response_live = None

    def splash(self, model_name: str, session_id: str) -> None:
        wordmark = Figlet(font="blocky").renderText("mira").rstrip()
        border = f"[{MIRA_CYAN}]===[/{MIRA_CYAN}]"
        divider = f"[{MIRA_CYAN}]---[/{MIRA_CYAN}]"

        self.console.print(border)
        self.console.print(f"[{MIRA_CYAN}]{wordmark}[/{MIRA_CYAN}]")
        self.console.print()
        self.console.print(f"[bold {MIRA_CYAN}]Minimal Iterative Reasoning Agent[/bold {MIRA_CYAN}]")
        self.console.print(divider)
        self.console.print(f"[dim]model:[/dim] {model_name}")
        self.console.print(f"[dim]session:[/dim] {session_id}")
        self.console.print(border)
        self.console.print()

    # ── Thinking (streaming) ─────────────────────────────────────────────────

    def reasoning_delta(self, delta: str) -> None:
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
        return Panel(
            self._thinking_text,
            title=f"[bold {MIRA_CYAN}]Thinking[/bold {MIRA_CYAN}]",
            title_align="left",
            border_style=f"dim {MIRA_CYAN}",
        )

    def _stop_thinking_live(self) -> None:
        if self._thinking_live is None:
            return
        self._thinking_live.update(self._render_thinking())
        self._thinking_live.stop()
        self._thinking_live = None
        self._thinking_text = ""

    # ── Response (streaming) ─────────────────────────────────────────────────

    def text_delta(self, delta: str) -> None:
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
        return Panel(
            self._response_text,
            title=f"[bold {MIRA_CYAN}]mira - response[/bold {MIRA_CYAN}]",
            title_align="left",
            border_style=MIRA_CYAN,
        )

    def _stop_response_live(self) -> None:
        if self._response_live is None:
            return
        self._response_live.update(self._render_response())
        self._response_live.stop()
        self._response_live = None
        self._response_text = ""

    # ── Tool calls ───────────────────────────────────────────────────────────

    def tool_call(self, name: str, args) -> None:
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
        # results are shown inside the tool panel via subagent blocks;
        # for coordinator tool calls just print quietly
        self.console.print(f"  output: {self.truncate(result)}", style="dim white")

    def delegation_started(self, calls: list[dict]) -> None:
        import json

        if not calls:
            return

        self._stop_thinking_live()
        self._stop_response_live()

        # Parse each call, separating valid delegations from malformed chunks
        valid = []
        errors = []
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
                valid.append(description)
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

    # ── Subagents ────────────────────────────────────────────────────────────

    def subagent_started(self, subagent: str, task_input: str = "") -> None:
        self.subagent_blocks[subagent] = {
            "request": task_input,
            "status": "RUNNING",
            "tool": None,
            "args": None,
            "output": "",
            "colour": self.subagent_colour(subagent),
        }
        self.refresh_subagents()

    def subagent_finished(
        self,
        subagent: str,
        tool: str | None = None,
        args=None,
        result: str = "",
    ) -> None:
        block = self.subagent_blocks.setdefault(
            subagent,
            {
                "request": "",
                "status": "RUNNING",
                "tool": None,
                "args": None,
                "output": "",
                "colour": self.subagent_colour(subagent),
            },
        )
        block["status"] = "DONE"
        block["tool"] = tool
        block["args"] = args
        block["output"] = result
        self.refresh_subagents()

    def subagent_tool_result(self, subagent: str, tool: str, result: str) -> None:
        self.subagent_finished(subagent, tool, {}, result)

    def finish_main(self) -> None:
        self._stop_thinking_live()
        self.stop_subagent_live()
        self._stop_response_live()

    async def ask_approvals(self, interrupts: list) -> list[dict]:
        decisions = []

        for interrupt in interrupts:
            for action in self.action_requests(interrupt):
                self.console.print()
                self.console.print(Panel(self.action_text(action), title="Approval", border_style=MIRA_CYAN))
                answer = await self._choice()

                if answer == "e":
                    decisions.append(await self.edit_decision(action))
                elif answer == "r":
                    decisions.append({"type": "reject"})
                else:
                    decisions.append({"type": "approve"})

        return decisions

    def action_requests(self, interrupt) -> list:
        value = getattr(interrupt, "value", interrupt)

        if isinstance(value, dict) and value.get("action_requests"):
            return value["action_requests"]

        return [value]

    def action_text(self, action) -> str:
        if isinstance(action, dict):
            name = action.get("name", "tool")
            args = action.get("args", {})
            return f"{name}\n\n{json.dumps(args, indent=2)}"

        return str(action)

    async def edit_decision(self, action) -> dict:
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

    async def prompt_json(self, original: dict) -> dict | None:
        import asyncio

        session = PromptSession()
        prompt = "edited args JSON> "
        text = await asyncio.to_thread(session.prompt, prompt, default=json.dumps(original))

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            self.console.print("[dim]invalid JSON; rejecting action[/dim]")
            return None

    async def _choice(self) -> str:
        import asyncio

        return await asyncio.to_thread(
            choice,
            "Approve this action?",
            [("y", "approve"), ("e", "edit"), ("r", "reject")],
            show_frame=True,
        )

    def write_dim(self, value: str) -> None:
        self.console.print(value, style="dim", end="")

    def start_subagent_live(self) -> None:
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
        if self.live is None:
            return

        self.live.update(self.render_subagents())
        self.live.stop()
        self.live = None

    def tick_subagents(self) -> None:
        if not self.has_running_subagents():
            return

        self.spinner_index = (self.spinner_index + 1) % len(SPINNER_FRAMES)
        self.refresh_subagents()

    def has_running_subagents(self) -> bool:
        return any(block["status"] == "RUNNING" for block in self.subagent_blocks.values())

    def refresh_subagents(self) -> None:
        if self.live is not None:
            self.live.update(self.render_subagents())
            return

        self.console.print(self.render_subagents())

    def render_subagents(self):
        if not self.subagent_blocks:
            return ""

        return Group(*(self.render_subagent(label, block) for label, block in self.subagent_blocks.items()))

    def render_subagent(self, label: str, block: dict) -> Panel:
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

        if block.get("tool"):
            text.append("\nfinal tool: ", style=MIRA_CYAN)
            text.append(str(block["tool"]), style="bold white")
            text.append("\nargs: ", style=MIRA_CYAN)
            text.append(self.format_args(block.get("args")), style="white")
            text.append("\noutput: ", style=MIRA_CYAN)
            text.append(self.truncate_output(block.get("output", "")))

        return Panel(
            text,
            title=Text(f"subagent - {label}", style=f"bold {colour}"),
            title_align="left",
            border_style=colour,
        )

    def format_args(self, value) -> str:
        if value is None:
            return "{}"

        try:
            return self.truncate(json.dumps(value, ensure_ascii=False, sort_keys=True))
        except TypeError:
            return self.truncate(value)

    def truncate_output(self, value) -> Text:
        text = self.single_line(value)

        if self.tool_output_chars == 0 or len(text) <= self.tool_output_chars:
            return Text(text, style="white")

        rendered = Text(text[: self.tool_output_chars], style="white")
        rendered.append(" ... truncated ...", style="yellow")
        return rendered

    def truncate(self, value: str) -> str:
        text = self.single_line(value)

        if self.tool_output_chars == 0 or len(text) <= self.tool_output_chars:
            return text

        return text[: self.tool_output_chars] + " ... truncated ..."

    def single_line(self, value) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    def subagent_label(self, subagent) -> str:
        key = id(subagent)
        if key not in self.subagent_labels:
            name = getattr(subagent, "name", "subagent")
            self.subagent_labels[key] = f"{name} [{self.subagent_suffix()}]"

        return self.subagent_labels[key]

    def subagent_suffix(self) -> str:
        for _ in range(20):
            suffix = self.single_line(self.fake.word()).lower()
            if suffix and suffix not in self.subagent_suffixes:
                self.subagent_suffixes.add(suffix)
                return suffix

        suffix = self.single_line(self.fake.uuid4())[:8]
        self.subagent_suffixes.add(suffix)
        return suffix

    def subagent_colour(self, name: str) -> str:
        if name not in self.subagent_colours:
            self.subagent_colours[name] = next(self.colours)

        return self.subagent_colours[name]
