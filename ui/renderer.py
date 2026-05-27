import json
import re
import sys
from itertools import cycle

from faker import Faker
from prompt_toolkit import PromptSession
from prompt_toolkit.shortcuts import choice
from pyfiglet import Figlet
from rich.console import Console
from rich.panel import Panel

DEFAULT_TOOL_OUTPUT_CHARS = 240
SUBAGENT_COLOURS = ["35", "33", "32", "36", "34"]


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
        self.colours = cycle(SUBAGENT_COLOURS)
        self.section = None

    def splash(self, model_name: str, session_id: str) -> None:
        wordmark = Figlet(font="blocky").renderText("mira").rstrip()
        border = "[cyan]═══[/cyan]"
        divider = "[cyan]───[/cyan]"

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
        self.write_dim(f"\n  mira tool call: {name}  {self.truncate(args)}\n")

    def tool_result(self, name: str, result: str) -> None:
        self.enter_main()
        self.write_dim(f"  └ {self.truncate(result)}\n")

    def subagent_tool_result(self, subagent: str, tool: str, result: str) -> None:
        self.enter_subagent(subagent)
        sys.stdout.write(f"  \033[2mtool call: {tool}\033[0m")
        sys.stdout.write(f"\n  \033[2m└ {self.truncate(result)}\033[0m\n")
        sys.stdout.flush()

    def finish_main(self) -> None:
        self.flush_reasoning()
        self.enter_main()
        self.write_dim("  mira done\n")

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
        sys.stdout.write(f"\033[2m{value}\033[0m")
        sys.stdout.flush()

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
        sys.stdout.write("\n\033[1;36mmira [main]\033[0m\n")
        sys.stdout.flush()

    def enter_subagent(self, label: str) -> None:
        if self.section == label:
            return

        self.section = label
        colour = self.subagent_colour(label)
        sys.stdout.write(f"\n\033[1;{colour}msubagent - {label}\033[0m\n")
        sys.stdout.flush()

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
