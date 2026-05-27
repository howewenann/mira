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
SUBAGENT_COLOURS = ["magenta", "yellow", "green", "cyan", "blue"]
SPINNER_FRAMES = ["-", "\\", "|", "/"]


class Renderer:
    def __init__(self, tool_output_chars: int = DEFAULT_TOOL_OUTPUT_CHARS) -> None:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")

        self.tool_output_chars = tool_output_chars
        self.console = Console()
        self.fake = Faker()
        self.reasoning = []
        self.subagent_colours = {}
        self.subagent_labels = {}
        self.subagent_suffixes = set()
        self.subagent_sections = set()
        self.subagent_blocks = {}
        self.live = None
        self.spinner_index = 0
        self.colours = cycle(SUBAGENT_COLOURS)
        self.section = None

    def splash(self, model_name: str, session_id: str) -> None:
        wordmark = Figlet(font="blocky").renderText("mira").rstrip()
        border = "[cyan]===[/cyan]"
        divider = "[cyan]---[/cyan]"

        self.console.print(border)
        self.console.print(f"[cyan]{wordmark}[/cyan]")
        self.console.print()
        self.console.print("[bold cyan]Minimal Iterative Reasoning Agent[/bold cyan]")
        self.console.print(divider)
        self.console.print(f"[dim]model:[/dim] {model_name}")
        self.console.print(f"[dim]session:[/dim] {session_id}")
        self.console.print(border)
        self.console.print()

    def text(self, value: str) -> None:
        self.flush_reasoning()
        self.enter_main()
        sys.stdout.write(value)
        sys.stdout.flush()

    def newline(self) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()

    def add_reasoning(self, value: str) -> None:
        cleaned = re.sub(r"</?[^>]+>", "", value).strip()
        if cleaned:
            self.reasoning.append(cleaned)

    def flush_reasoning(self) -> None:
        if not self.reasoning:
            return

        text = "\n\n".join(self.reasoning)
        self.console.print(Panel(text, title="Thinking", border_style="dim cyan"))
        self.reasoning.clear()

    def tool_call(self, name: str, args) -> None:
        self.flush_reasoning()
        self.enter_main()
        self.console.print()
        self.console.print(f"  mira tool call: {name}", style="cyan")
        self.console.print(f"  args: {self.truncate(args)}", style="bright_black")

    def tool_result(self, name: str, result: str) -> None:
        self.enter_main()
        self.console.print(f"  output: {self.truncate(result)}", style="white")

    def delegation_started(self, calls: list[dict]) -> None:
        if not calls:
            return

        self.flush_reasoning()
        self.enter_main()
        count = len(calls)
        label = "subagent" if count == 1 else "subagents"
        self.console.print(f"  delegating to {count} {label}...", style="bold yellow")

        for call in calls:
            args = call.get("args", {}) if isinstance(call, dict) else {}
            description = args.get("description")
            if description:
                self.console.print(f"  request: {self.truncate(description)}", style="cyan")

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
        self.flush_reasoning()
        self.stop_subagent_live()
        self.enter_main()
        self.console.print("  mira done", style="green")

    async def ask_approvals(self, interrupts: list) -> list[dict]:
        decisions = []

        for interrupt in interrupts:
            for action in self.action_requests(interrupt):
                self.console.print()
                self.console.print(Panel(self.action_text(action), title="Approval", border_style="cyan"))
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
            text.append("request: ", style="cyan")
            text.append(self.truncate(block["request"]), style="white")
            text.append("\n")

        text.append("status: ", style="cyan")
        if status == "RUNNING":
            frame = SPINNER_FRAMES[self.spinner_index]
            text.append(f"{frame} RUNNING", style="bold yellow")
        else:
            text.append("DONE", style="bold green")

        if block.get("tool"):
            text.append("\nfinal tool: ", style="cyan")
            text.append(str(block["tool"]), style="bold white")
            text.append("\nargs: ", style="cyan")
            text.append(self.format_args(block.get("args")), style="white")
            text.append("\noutput: ", style="cyan")
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

    def enter_main(self) -> None:
        if self.section == "main":
            return

        self.section = "main"
        self.console.print()
        self.console.print("mira [main]", style="bold cyan")

    def enter_subagent(self, label: str) -> None:
        if self.section == label:
            return

        self.section = label
        if label in self.subagent_sections:
            return

        self.subagent_sections.add(label)
        colour = self.subagent_colour(label)
        self.console.print()
        self.console.print(f"subagent - {label}", style=f"bold {colour}")

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
