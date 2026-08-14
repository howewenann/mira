"""Standalone native-Textual mockups for possible MIRA UI changes.

Run with: python textual_ui_mockup.py
"""

from __future__ import annotations

import json
import math
from time import monotonic

from rich.cells import cell_len
from rich.markup import escape
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.theme import Theme
from textual.widgets import (
    Button,
    Collapsible,
    ContentSwitcher,
    DataTable,
    Footer,
    Input,
    Label,
    LoadingIndicator,
    OptionList,
    Select,
    Static,
    Switch,
    TextArea,
)


MIRA_THEME = Theme(
    name="mira-mockup",
    primary="#5bb8b1",
    secondary="#d2a957",
    accent="#7ce3dc",
    warning="#d2a957",
    error="#d96b66",
    success="#7fbe72",
    foreground="#e8edef",
    background="#0c0f10",
    surface="#101516",
    panel="#151f22",
    boost="#eef7f8",
    dark=True,
    variables={
        "border": "#5bb8b1",
        "border-blurred": "#56616a",
        "block-cursor-background": "#14524f",
        "block-cursor-foreground": "#e8fffb",
        "block-cursor-text-style": "bold",
        "block-cursor-blurred-background": "#1b3036",
        "block-cursor-blurred-foreground": "#eef7f8",
        "block-cursor-blurred-text-style": "none",
        "block-hover-background": "#17313a",
        "input-selection-background": "#14524f",
        "scrollbar": "#5bb8b1",
        "scrollbar-background": "#122023",
        "scrollbar-background-hover": "#17313a",
        "scrollbar-background-active": "#1b3036",
    },
)


class DemoPage(VerticalScroll):
    """Scrollable base for a single mockup page."""


class ArgumentTextArea(TextArea):
    """Read-only argument viewer that grows to content, then scrolls."""

    MAX_ROWS = 12
    WRAP_ESTIMATE = 96

    def __init__(self, text: str) -> None:
        super().__init__(
            text,
            language=None,
            soft_wrap=True,
            read_only=True,
            show_line_numbers=False,
            highlight_cursor_line=False,
            tab_behavior="focus",
            classes="args-json",
        )
        self._estimated_rows = sum(
            max(1, math.ceil(len(line) / self.WRAP_ESTIMATE))
            for line in text.splitlines()
        )
        self.styles.height = min(self.MAX_ROWS, max(1, self._estimated_rows))

    def on_mount(self) -> None:
        self.call_after_refresh(self.fit_to_rendered_content)

    def on_resize(self) -> None:
        self.call_after_refresh(self.fit_to_rendered_content)

    def fit_to_rendered_content(self) -> None:
        """Use Textual's actual wrapped height once layout width is known."""
        content_rows = (
            self.wrapped_document.height
            if self.wrap_width > 0
            else self._estimated_rows
        )
        rows = min(
            self.MAX_ROWS,
            max(1, content_rows),
        )
        if self.size.height != rows:
            self.styles.height = rows


class ClosableCollapsible(Collapsible):
    """Native Collapsible with a close control sharing its title row."""

    def compose(self) -> ComposeResult:
        with Horizontal(classes="subagent-title-row"):
            yield self._title
            yield Button("×", id="close-subagents", tooltip="Close subagents panel")
        with self.Contents():
            yield from self._contents_list


class HitlDemo(DemoPage):
    def compose(self) -> ComposeResult:
        yield Label("HITL / Ask User choices", classes="page-title")
        with Container(classes="card"):
            yield Label("Short confirmation", classes="section-title")
            yield Static(
                "Allow MIRA to run [bold]git status --short[/bold] in this workspace?",
                classes="explanation",
            )
            with Horizontal(classes="button-row"):
                yield Button("Allow", id="hitl-allow", variant="success")
                yield Button("Always allow", id="hitl-always")
                yield Button("Deny", id="hitl-deny", variant="error")
            yield Label("No choice made.", id="hitl-result", classes="result")

        with Container(classes="card"):
            yield Label("Ask User", classes="section-title")
            yield Static(
                "How should MIRA handle tool arguments that exceed the compact "
                "display limit?",
                classes="question",
            )
            yield OptionList(
                "Keep a short summary in the bubble and reveal the complete, "
                "pretty-printed arguments when I expand it.",
                "Show all arguments inline even when they make the conversation "
                "substantially longer.",
                "Write oversized arguments to an artifact and show only its path "
                "and a brief preview in the conversation.",
                "Use the compact view by default, but automatically expand failed "
                "tool calls so debugging context is immediately visible.",
                id="ask-options",
            )
            yield Label("Choose an option above.", id="ask-result", classes="result")

    @on(Button.Pressed, "#hitl-allow, #hitl-always, #hitl-deny")
    def show_hitl_choice(self, event: Button.Pressed) -> None:
        self.query_one("#hitl-result", Label).update(
            f"Selected: {event.button.label}"
        )

    @on(OptionList.OptionSelected, "#ask-options")
    def show_ask_choice(self, event: OptionList.OptionSelected) -> None:
        self.query_one("#ask-result", Label).update(
            f"Selected option {event.option_index + 1}"
        )


def setting_row(label: str, control: Input | Select | Switch) -> Horizontal:
    return Horizontal(Label(label, classes="setting-label"), control, classes="setting-row")


class SettingsDemo(DemoPage):
    PAGES = {
        "settings-general-button": "settings-general",
        "settings-models-button": "settings-models",
        "settings-tools-button": "settings-tools",
        "settings-mcp-button": "settings-mcp",
    }

    def compose(self) -> ComposeResult:
        yield Label("Settings navigation + ContentSwitcher", classes="page-title")
        with Horizontal(id="settings-nav", classes="button-row nav-row"):
            yield Button("General", id="settings-general-button", classes="active")
            yield Button("Models", id="settings-models-button")
            yield Button("Custom Tools", id="settings-tools-button")
            yield Button("MCP", id="settings-mcp-button")

        with ContentSwitcher(initial="settings-general", id="settings-switcher"):
            with Vertical(id="settings-general", classes="settings-page card"):
                yield Label("General", classes="section-title")
                yield setting_row("Theme", Select((("MIRA dark", "dark"), ("Textual", "textual")), value="dark"))
                yield setting_row("Show planning todos", Switch(value=True))
                yield setting_row("Default workspace", Input("D:\\Projects\\mira"))
            with Vertical(id="settings-models", classes="settings-page card"):
                yield Label("Models", classes="section-title")
                yield setting_row("Provider", Select((("OpenAI", "openai"), ("Anthropic", "anthropic")), value="openai"))
                yield setting_row("Action model", Input("gpt-5.2"))
                yield setting_row("Planning model", Input("gpt-5.2"))
            with Vertical(id="settings-tools", classes="settings-page card"):
                yield Label("Custom Tools", classes="section-title")
                yield setting_row("Tool directory", Input(".mira/tools"))
                yield setting_row("Load workspace tools", Switch(value=True))
                yield Static("3 discovered tools: build_docs, inspect_logs, format_report", classes="hint")
            with Vertical(id="settings-mcp", classes="settings-page card"):
                yield Label("MCP", classes="section-title")
                yield setting_row("Enable MCP servers", Switch(value=True))
                yield setting_row("Startup timeout", Input("15 seconds"))
                yield Static("DuckDuckGo, Playwright, and Docling are configured.", classes="hint")

    @on(Button.Pressed, "#settings-nav Button")
    def switch_settings_page(self, event: Button.Pressed) -> None:
        page = self.PAGES[event.button.id or ""]
        self.query_one("#settings-switcher", ContentSwitcher).current = page
        for button in self.query("#settings-nav Button"):
            button.set_class(button is event.button, "active")


class ExecuteDemo(DemoPage):
    def compose(self) -> ComposeResult:
        yield Label("Execute environment Select + ContentSwitcher", classes="page-title")
        with Container(classes="card"):
            with Horizontal(classes="environment-row"):
                yield Label("Run commands in", classes="setting-label")
                yield Select(
                    (
                        ("Current environment", "exec-current"),
                        ("Conda environment name", "exec-conda-name"),
                        ("Conda prefix", "exec-conda-prefix"),
                        ("Virtual environment path", "exec-venv"),
                    ),
                    value="exec-current",
                    allow_blank=False,
                    id="execute-mode",
                )
            with ContentSwitcher(initial="exec-current", id="execute-switcher"):
                with Vertical(id="exec-current", classes="execute-page"):
                    yield Static(
                        "Commands inherit the environment used to launch MIRA. "
                        "No additional configuration is needed.",
                        classes="hint",
                    )
                with Vertical(id="exec-conda-name", classes="execute-page"):
                    yield setting_row("Conda environment", Input("my_project_env"))
                with Vertical(id="exec-conda-prefix", classes="execute-page"):
                    yield setting_row("Conda prefix", Input("C:\\envs\\project"))
                with Vertical(id="exec-venv", classes="execute-page"):
                    yield setting_row("Venv location", Input(".venv"))

    @on(Select.Changed, "#execute-mode")
    def change_execute_mode(self, event: Select.Changed) -> None:
        if isinstance(event.value, str):
            self.query_one("#execute-switcher", ContentSwitcher).current = event.value


def detail(label: str, value: str) -> Static:
    text = Text()
    text.append(f"{label:<12}", style="bold #b8c1c7")
    text.append(value, style="#eef7f8")
    return Static(text, classes="server-detail")


class McpDemo(DemoPage):
    def compose(self) -> ComposeResult:
        yield Label("MCP servers using native Collapsible", classes="page-title")
        with Collapsible(
            detail("Transport", "stdio"),
            detail("Command", "uvx duckduckgo-mcp-server"),
            detail("Tools", "2"),
            detail("Resources", "0"),
            detail("Prompts", "0"),
            Horizontal(
                Button("Stop", variant="error"),
                Button("Restart"),
                classes="button-row server-actions",
            ),
            title="DuckDuckGo  •  Running",
            collapsed=True,
            classes="server running",
        ):
            pass
        with Collapsible(
            detail("Transport", "stdio"),
            detail("Command", "npx @playwright/mcp@latest"),
            detail("Tools", "23"),
            detail("Resources", "0"),
            detail("Prompts", "1"),
            Horizontal(Button("Start", variant="success"), classes="button-row server-actions"),
            title="Playwright  •  Stopped",
            collapsed=True,
            classes="server stopped",
        ):
            pass
        with Collapsible(
            detail("Transport", "HTTP"),
            detail("URL", "http://127.0.0.1:5001/mcp"),
            detail("Tools", "Unavailable"),
            Static("Connection refused while discovering server capabilities.", classes="error-text"),
            Horizontal(
                Button("Start", variant="success"),
                Button("Restart"),
                classes="button-row server-actions",
            ),
            title="Docling  •  Failed",
            collapsed=False,
            classes="server failed",
        ):
            pass


class SubagentsDemo(DemoPage):
    MAX_BODY_ROWS = 12
    SCENARIO_GROUPS = {
        "dynamic": ("Group 1", "Group 2", "Group 3"),
        "regular": (),
        "mixed": ("Tasks", "Group 1", "Group 2"),
    }
    TASKS = {
        "Tasks": [
            ("regular-search", "researcher [silver-tern]", "Search Textual documentation", "running", None),
            ("regular-inspect", "general-purpose [cream-mole]", "Inspect MIRA widgets", "done", 8.4),
            ("regular-build", "coder [amber-fox]", "Build the native widget prototype", "waiting", 0.0),
            ("regular-review", "general-purpose [lilac-owl]", "Review the final visual hierarchy", "done", 11.2),
        ],
        "Group 1": [
            ("group-1-haiku", "general-purpose [cream-mole]", "Generate 8 unique breakfast haikus", "running", None),
            ("group-1-rank", "general-purpose [silver-tern]", "Rank imagery and rhythm", "done", 4.7),
            ("group-1-check", "general-purpose [amber-fox]", "Check the winning haiku", "waiting", 0.0),
        ],
        "Group 2": [
            ("group-2-compare", "general-purpose [cobalt-finch]", "Compare two finalist haikus", "running", None),
            ("group-2-tone", "general-purpose [moss-gecko]", "Judge tone and originality", "done", 6.1),
        ],
        "Group 3": [
            (
                f"group-3-judge-{index}",
                f"general-purpose [judge-{index:02d}]",
                f"Score candidate haiku {index} against the final rubric",
                "done" if index <= 9 else "waiting",
                index * 0.8 if index <= 9 else 0.0,
            )
            for index in range(1, 15)
        ],
    }
    SCENARIO_TASK_KEYS = {
        "dynamic": ("Group 1", "Group 2", "Group 3"),
        "regular": ("Tasks",),
        "mixed": ("Tasks", "Group 1", "Group 2"),
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scenario = "mixed"
        self.group = "Tasks"
        self.started_at = monotonic()

    def compose(self) -> ComposeResult:
        yield Label("Subagents panel using native primitives", classes="page-title")
        yield Static(
            "Regular-only, eval-only, and mixed workloads use MIRA's current "
            "grouping rules. DataTable text is not application-selectable; terminal "
            "screen-copy may require Shift-drag.",
            classes="subagent-demo-note",
        )
        with Horizontal(classes="subagent-scenario-row"):
            yield Label("Sample workload", classes="subagent-scenario-label")
            yield Select(
                (
                    ("Mixed regular + eval", "mixed"),
                    ("Dynamic eval groups only", "dynamic"),
                    ("Regular task subagents only", "regular"),
                ),
                value="mixed",
                allow_blank=False,
                id="subagent-scenario",
            )
        yield Button("Reopen subagents panel", id="reopen-subagents", classes="hidden")
        with ClosableCollapsible(
            Horizontal(
                Label("GROUPS", id="subagent-groups-label", classes="subagent-groups-label"),
                Label("TASK", classes="subagent-task-label"),
                Label("STATUS", classes="subagent-status-label"),
                Label("TIME", classes="subagent-time-label"),
                classes="subagent-toolbar",
            ),
            Horizontal(
                Vertical(
                    OptionList(id="subagent-groups", compact=True),
                    id="subagent-groups-column",
                    classes="groups-column",
                ),
                Vertical(
                    DataTable(
                        id="subagent-tasks",
                        cursor_type="none",
                        zebra_stripes=False,
                        show_header=False,
                        show_cursor=False,
                    ),
                    classes="tasks-column",
                ),
                id="subagent-columns",
                classes="subagent-columns",
            ),
            title="subagents",
            collapsed=False,
            id="subagents-panel",
        ):
            pass

    def on_mount(self) -> None:
        table = self.query_one("#subagent-tasks", DataTable)
        table.add_column("TASK", key="task", width=40)
        table.add_column("STATUS", key="status", width=10)
        table.add_column("TIME", key="time", width=8)
        table.cell_padding = 1
        self.load_scenario("mixed")
        self.set_interval(0.1, self.update_elapsed_times)

    def on_resize(self) -> None:
        self.call_after_refresh(self.align_task_columns)

    def load_scenario(self, scenario: str) -> None:
        self.scenario = scenario
        group_keys = self.SCENARIO_GROUPS[scenario]
        self.group = group_keys[0] if group_keys else "Tasks"
        self.started_at = monotonic()

        groups = self.query_one("#subagent-groups", OptionList)
        groups.set_options(self.group_prompt(key, 0.0) for key in group_keys)
        if group_keys:
            groups.highlighted = 0
        show_groups = bool(group_keys)
        self.query_one("#subagent-groups-label").display = show_groups
        self.query_one("#subagent-groups-column").display = show_groups
        self.refresh_tasks()
        self.update_panel_title()

    def refresh_tasks(self) -> None:
        self.populate_task_rows()
        self.size_panel_to_content()
        self.call_after_refresh(self.align_task_columns)

    def populate_task_rows(self) -> None:
        table = self.query_one("#subagent-tasks", DataTable)
        table.clear(columns=False)
        elapsed = monotonic() - self.started_at
        task_width = table.ordered_columns[0].width if table.ordered_columns else 40
        for key, name, task, status, duration in self.TASKS[self.group]:
            shown_time = elapsed if duration is None else duration
            task_text = Text("/ ", style="bold #d2a957")
            task_text.append(name, style="#b7a4e8")
            task_text.append(f"  {task}", style="#eef7f8")
            if cell_len(task_text.plain) > task_width:
                if task_width > 3:
                    task_text.truncate(task_width - 3, overflow="crop")
                    task_text.append("...", style="#b8c1c7")
                else:
                    task_text = Text("." * max(0, task_width), style="#b8c1c7")
            status_text = Text(status.upper(), style=self.status_style(status))
            table.add_row(task_text, status_text, f"{shown_time:.1f}s", key=key)

    def size_panel_to_content(self) -> None:
        group_rows = len(self.SCENARIO_GROUPS[self.scenario])
        task_rows = len(self.TASKS[self.group])
        rows = min(self.MAX_BODY_ROWS, max(1, group_rows, task_rows))
        self.query_one("#subagent-columns").styles.height = rows

    def align_task_columns(self) -> None:
        table = self.query_one("#subagent-tasks", DataTable)
        width = table.content_region.width or table.size.width
        if width <= 0:
            return
        fixed_width = (10 + 2 * table.cell_padding) + (8 + 2 * table.cell_padding)
        # Reserve one cell for a possible vertical scrollbar. The task column
        # yields first so STATUS and TIME always remain visible.
        task_width = max(2, width - fixed_width - 2 * table.cell_padding - 1)
        if not table.ordered_columns:
            return
        column = table.ordered_columns[0]
        if column.width != task_width:
            column.width = task_width
            self.populate_task_rows()
            table.refresh(layout=True)

    def update_panel_title(self) -> None:
        task_keys = self.SCENARIO_TASK_KEYS[self.scenario]
        records = [record for key in task_keys for record in self.TASKS[key]]
        done = sum(1 for _key, _name, _task, status, _duration in records if status == "done")
        eval_groups = sum(1 for key in task_keys if key.startswith("Group "))
        label = "dynamic subagents" if self.scenario == "dynamic" else "subagents"
        title = f"{label}    {done}/{len(records)} done"
        if eval_groups:
            title += f"    {eval_groups} {'group' if eval_groups == 1 else 'groups'}"
        self.query_one("#subagents-panel", Collapsible).title = title

    def update_elapsed_times(self) -> None:
        table = self.query_one("#subagent-tasks", DataTable)
        elapsed = monotonic() - self.started_at
        self.refresh_group_prompts(elapsed)
        for key, _name, _task, _status, duration in self.TASKS[self.group]:
            if duration is None:
                try:
                    table.update_cell(key, "time", f"{elapsed:.1f}s")
                except Exception:
                    return

    def refresh_group_prompts(self, elapsed: float = 0.0) -> None:
        groups = self.query_one("#subagent-groups", OptionList)
        for index, group in enumerate(self.SCENARIO_GROUPS[self.scenario]):
            group_elapsed = elapsed + (index * 2.3)
            groups.replace_option_prompt_at_index(
                index,
                self.group_prompt(group, group_elapsed),
            )

    @staticmethod
    def status_style(status: str) -> str:
        return {
            "running": "bold #7ce3dc",
            "done": "#7fbe72",
            "waiting": "#b8c1c7",
        }[status]

    def group_prompt(self, group: str, elapsed: float) -> Text:
        records = self.TASKS[group]
        done = sum(1 for _key, _name, _task, status, _duration in records if status == "done")
        prompt = Text("> " if group == self.group else "  ", style="#eef7f8")
        prompt.append("* ", style="bold #d2a957")
        prompt.append(group, style="bold #eef7f8")
        prompt.append(f"  {done}/{len(records)}  ", style="#b8c1c7")
        prompt.append(f"{elapsed:.1f}s", style="#eef7f8")
        return prompt

    @on(Select.Changed, "#subagent-scenario")
    def select_scenario(self, event: Select.Changed) -> None:
        if isinstance(event.value, str):
            self.load_scenario(event.value)

    @on(OptionList.OptionSelected, "#subagent-groups")
    def select_group(self, event: OptionList.OptionSelected) -> None:
        self.group = self.SCENARIO_GROUPS[self.scenario][event.option_index]
        self.started_at = monotonic()
        self.refresh_group_prompts()
        self.refresh_tasks()

    @on(Button.Pressed, "#close-subagents")
    def close_panel(self) -> None:
        self.query_one("#subagents-panel", Collapsible).display = False
        self.query_one("#reopen-subagents", Button).display = True

    @on(Button.Pressed, "#reopen-subagents")
    def reopen_panel(self) -> None:
        self.query_one("#subagents-panel", Collapsible).display = True
        self.query_one("#reopen-subagents", Button).display = False


class LoadingDemo(DemoPage):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.started_at = monotonic()

    def compose(self) -> ComposeResult:
        yield Label("Native LoadingIndicator beside explanatory text", classes="page-title")
        with Container(classes="card loading-card"):
            yield self.loading_row("Generating success criteria...", "criteria-time", 0.0)
            yield self.loading_row("Creating plan...", "plan-time", 3.6)
            yield self.loading_row("Calling read_file...", "tool-time", 6.2)
        with Container(classes="card comparison-card"):
            yield Label("Small visual comparison", classes="section-title")
            with Horizontal(classes="comparison-row"):
                yield Static("⠹  Creating plan...", classes="braille-sample")
                yield Static("Existing-style static approximation", classes="hint")
            with Horizontal(classes="comparison-row"):
                yield LoadingIndicator()
                yield Static("Creating plan...", classes="loading-label")
                yield Static("Native Textual indicator", classes="hint")

    @staticmethod
    def loading_row(label: str, timer_id: str, offset: float) -> Horizontal:
        return Horizontal(
            LoadingIndicator(),
            Static(label, classes="loading-label"),
            Static(f"{offset:.1f}s", id=timer_id, classes="elapsed"),
            classes="loading-row",
        )

    def on_mount(self) -> None:
        self.set_interval(0.1, self.update_times)

    def update_times(self) -> None:
        elapsed = monotonic() - self.started_at
        self.query_one("#criteria-time", Static).update(f"{elapsed:.1f}s")
        self.query_one("#plan-time", Static).update(f"{elapsed + 3.6:.1f}s")
        self.query_one("#tool-time", Static).update(f"{elapsed + 6.2:.1f}s")


def compact_args(arguments: dict, limit: int = 112) -> str:
    rendered = json.dumps(arguments, ensure_ascii=False, separators=(", ", ": "))
    if len(rendered) <= limit:
        return rendered
    return rendered[: limit - 1] + "…"


def tool_bubble(
    tool_name: str,
    arguments: dict,
    output: str,
    duration: str,
    *,
    collapsed: bool,
    timestamp: str,
) -> Container:
    title = Text("call: ", style="bold cyan")
    title.append(compact_args(arguments), style="#e8edef")
    args_editor = ArgumentTextArea(
        json.dumps(arguments, ensure_ascii=False, indent=2)
    )
    output_text = Text()
    output_text.append("-" * 12, style="dim")
    output_text.append("\noutput:\n", style="bold cyan")
    output_text.append(output, style="dim")
    duration_text = Text()
    duration_text.append("Completed in", style="#6fa884")
    duration_text.append(f" {duration}", style="#98a5ac")
    bubble = Container(
        Collapsible(
            args_editor,
            title=title,
            collapsed=collapsed,
            classes="args-collapsible",
        ),
        Static(output_text, classes="tool-output"),
        Static(duration_text, classes="tool-duration"),
        classes="chat-message tool-bubble",
    )
    bubble.border_title = escape(f"tool - {tool_name}")
    bubble.border_subtitle = escape(timestamp)
    return bubble


def transcript_bubble(
    title: str,
    body: str,
    timestamp: str,
    classes: str,
) -> Static:
    bubble = Static(body, classes=f"chat-message {classes}")
    bubble.border_title = escape(title)
    bubble.border_subtitle = escape(timestamp)
    return bubble


class ToolArgsDemo(DemoPage):
    def compose(self) -> ComposeResult:
        yield Label("Expandable tool-call args", classes="page-title")
        yield Static(
            "These are rendered in a representative MIRA main-chat transcript. "
            "Only the call/arguments row expands; output remains compact. In an "
            "expanded call, click the JSON editor, drag or use Shift+arrows to "
            "select, Ctrl+C to copy, and scroll independently.",
            classes="chat-demo-note",
        )
        with Vertical(id="tool-chat-log"):
            yield transcript_bubble(
                "you",
                "Search for the latest Nvidia stock price and save the results to "
                "/nvidia_stock_price.md.",
                "2026-08-13 11:16",
                "chat-user",
            )

            search_args = {"query": "latest Nvidia stock price"}
            search_output = (
                "NVIDIA Corporation (NVDA) — market summary, recent price, day "
                "range, related news, and analyst coverage… [truncated]"
            )
            yield tool_bubble(
                "mcp_Duckduckgo_Search",
                search_args,
                search_output,
                "1.3s",
                collapsed=True,
                timestamp="2026-08-13 11:16",
            )

            markdown_content = """## Search Results for 'latest Nvidia stock price'

```json
{
  "results": [
    {
      "title": "NVIDIA Corporation (NVDA) Stock Price, News, Quote & History",
      "url": "https://finance.yahoo.com/quote/NVDA/",
      "snippet": "View the latest NVIDIA market data, company news, and historical performance."
    },
    {
      "title": "NVDA Stock Quote",
      "url": "https://www.nasdaq.com/market-activity/stocks/nvda",
      "snippet": "Real-time quote, charts, institutional holdings, and company information."
    }
  ]
}
```

Prices can change quickly. Verify the quote with a live market source before relying on it.
"""
            write_args = {
                "file_path": "/nvidia_stock_price.md",
                "content": markdown_content,
                "options": {
                    "encoding": "utf-8",
                    "create_parents": True,
                    "audit": {
                        "source_tools": ["mcp_Duckduckgo_Search"],
                        "tags": ["finance", "search-results", "prototype"],
                    },
                },
            }
            yield tool_bubble(
                "writeFile",
                write_args,
                "File written successfully",
                "0.4s",
                collapsed=False,
                timestamp="2026-08-13 11:17",
            )

            yield transcript_bubble(
                "mira",
                "I found the latest market references and saved the results to "
                "/nvidia_stock_price.md.",
                "2026-08-13 11:17",
                "chat-assistant",
            )

            yield transcript_bubble(
                "you",
                'Use eval to ask two subagents whether "quiet pond" or "bright '
                'market" is better for careful research.',
                "2026-08-13 11:18",
                "chat-user",
            )

            eval_code = """const candidates = [
  {
    name: "quiet pond",
    qualities: ["calm", "focused", "reflective"]
  },
  {
    name: "bright market",
    qualities: ["energetic", "social", "varied"]
  }
];

const judgments = await Promise.all(
  candidates.map((candidate) =>
    task({
      subagent_type: "general-purpose",
      description: `Judge this setting as a place for careful research: ${JSON.stringify(candidate)}`,
      responseSchema: {
        type: "object",
        properties: {
          score: { type: "number" },
          reason: { type: "string" }
        },
        required: ["score", "reason"]
      }
    })
  )
);

return {
  candidates,
  judgments,
  winner: judgments[0].score >= judgments[1].score
    ? candidates[0].name
    : candidates[1].name
};"""
            eval_args = {"code": eval_code}
            eval_output = (
                '<result>{"winner":"quiet pond","judgments":['
                '{"score":9,"reason":"Calm surroundings support sustained focus."},'
                '{"score":6,"reason":"Stimulating, but frequent activity may distract."}'
                "]}</result>\n<stdout>Completed 2 parallel judge tasks.</stdout>… [truncated]"
            )
            yield tool_bubble(
                "eval",
                eval_args,
                eval_output,
                "6.8s",
                collapsed=True,
                timestamp="2026-08-13 11:18",
            )

            yield transcript_bubble(
                "you",
                "Create a formal implementation Plan for replacing MIRA's custom "
                "tool-argument expansion with native Textual widgets.",
                "2026-08-13 11:20",
                "chat-user",
            )
            prepare_plan_args = {
                "objective": (
                    "Create an implementation-ready Plan for native expandable "
                    "tool-call arguments in MIRA's main chat."
                ),
                "context_and_constraints": (
                    "Preserve the current MIRA transcript bubble, keep compact "
                    "arguments visible by default, reveal complete selectable "
                    "arguments on demand, and never make tool output expandable. "
                    "Prefer native Textual Collapsible and TextArea behavior."
                ),
            }
            yield tool_bubble(
                "prepare_plan",
                prepare_plan_args,
                "Success Criteria generated; Plan finalization is ready.",
                "2.6s",
                collapsed=True,
                timestamp="2026-08-13 11:20",
            )
            finalize_plan_args = {
                "title": "Use native expandable arguments in MIRA tool bubbles",
                "key_changes": [
                    "Embed a native Collapsible call row inside each existing tool bubble.",
                    "Render full pretty JSON in a bounded read-only TextArea with selection and scrolling.",
                    "Keep result truncation, completion timing, and transcript timestamps unchanged.",
                ],
                "test_plan": [
                    "Cover collapsed, expanded, long, nested, and malformed argument values.",
                    "Verify mouse and keyboard scrolling without moving the outer chat unexpectedly.",
                    "Smoke-test prepare_plan, finalize_plan, prepare_goal, finalize_goal, and eval bubbles.",
                ],
                "assumptions": [
                    "Textual's read-only TextArea selection is acceptable for copying arguments.",
                    "The expanded region may use a fixed height to protect transcript readability.",
                ],
            }
            yield tool_bubble(
                "finalize_plan",
                finalize_plan_args,
                "Plan presented and retained for implementation.",
                "0.8s",
                collapsed=True,
                timestamp="2026-08-13 11:20",
            )

            yield transcript_bubble(
                "you",
                "Create a formal Goal for making long MIRA tool calls inspectable "
                "without overwhelming the conversation.",
                "2026-08-13 11:22",
                "chat-user",
            )
            prepare_goal_args = {
                "objective": (
                    "Make complete tool-call arguments easy to inspect, select, "
                    "copy, and scroll while preserving a compact main transcript."
                ),
                "context_and_constraints": (
                    "Use native Textual behavior, preserve current MIRA bubble "
                    "identity, and keep outputs truncated and non-expandable."
                ),
                "research_evidence": (
                    "The visual prototype showed that an auto-height pretty JSON "
                    "block hides navigation and lacks application-level selection; "
                    "a bounded read-only TextArea addresses both issues."
                ),
            }
            yield tool_bubble(
                "prepare_goal",
                prepare_goal_args,
                "Success Criteria generated; Goal finalization is ready.",
                "2.2s",
                collapsed=True,
                timestamp="2026-08-13 11:22",
            )
            finalize_goal_args = {
                "title": "Inspectable, compact tool calls in MIRA chat",
            }
            yield tool_bubble(
                "finalize_goal",
                finalize_goal_args,
                "Goal presented and retained.",
                "0.5s",
                collapsed=True,
                timestamp="2026-08-13 11:22",
            )

    @on(Collapsible.Expanded, ".args-collapsible")
    def refit_expanded_arguments(self, event: Collapsible.Expanded) -> None:
        editor = event.collapsible.query_one(ArgumentTextArea)
        editor.call_after_refresh(editor.fit_to_rendered_content)


class MiraNativeMockup(App):
    TITLE = "MIRA · Native Textual UI explorations"
    SUB_TITLE = "Throwaway standalone prototype"
    BINDINGS = [("ctrl+q", "quit", "Quit")]

    CSS = """
    Screen {
        background: #0c0f10;
        color: #e8edef;
    }

    #topbar {
        height: 3;
        padding: 0 2;
        content-align: left middle;
        background: #101b1f;
        color: #d6fff6;
        text-style: bold;
        border-bottom: solid #2d5661;
    }

    #body {
        height: 1fr;
    }

    #sidebar {
        width: 31;
        height: 1fr;
        padding: 1;
        background: #101516;
        border-right: solid #2d5661;
    }

    #sidebar-title {
        height: 2;
        color: #b8c1c7;
        text-style: bold;
    }

    #demo-nav {
        height: 1fr;
        border: none;
        background: #101516;
        scrollbar-color: #5bb8b1;
        scrollbar-background: #122023;
    }

    #demo-nav > .option-list--option-highlighted {
        background: #1b3036;
        color: #eef7f8;
    }

    #demo-switcher {
        width: 1fr;
        height: 1fr;
    }

    DemoPage {
        width: 1fr;
        height: 1fr;
        padding: 1 2 3 2;
        scrollbar-color: #5bb8b1;
        scrollbar-background: #122023;
    }

    .page-title {
        height: 3;
        padding: 0 1;
        content-align: left middle;
        color: #eef7f8;
        text-style: bold;
        background: #17313a;
        border-left: heavy #5bb8b1;
        margin-bottom: 1;
    }

    .card {
        height: auto;
        padding: 1 2;
        margin-bottom: 1;
        border: solid #3e5962;
        background: #151f22;
    }

    .section-title, .column-title {
        height: 2;
        color: #eef7f8;
        text-style: bold;
    }

    .question {
        margin-bottom: 1;
        color: #eef7f8;
        text-style: bold;
    }

    .explanation, .hint {
        color: #b8c1c7;
    }

    .result {
        height: 2;
        padding-top: 1;
        color: #7ce3dc;
    }

    .button-row {
        height: 3;
        margin-top: 1;
    }

    .button-row Button {
        width: auto;
        min-width: 12;
        margin-right: 1;
    }

    Button {
        background: #1b3036;
        color: #eef7f8;
        border: none;
    }

    Button:hover, Button:focus {
        background: #25434b;
    }

    Button.active {
        background: #5bb8b1;
        color: #081112;
        text-style: bold;
    }

    #ask-options {
        height: 13;
        border: solid #3c4a50;
        background: #101719;
    }

    #ask-options > .option-list--option-highlighted {
        background: #1b3036;
        color: #eef7f8;
    }

    .nav-row {
        margin: 0 0 1 0;
    }

    #settings-switcher {
        height: 18;
    }

    .settings-page {
        height: auto;
    }

    .setting-row, .environment-row {
        height: 4;
        align: left middle;
    }

    .setting-label {
        width: 24;
        content-align: left middle;
        color: #b8c1c7;
    }

    .setting-row Input, .setting-row Select, .environment-row Select {
        width: 48;
    }

    Input, Select {
        background: #101719;
        border: solid #56616a;
        color: #eef7f8;
    }

    Input:focus, Select:focus {
        border: solid #5bb8b1;
    }

    #execute-switcher {
        height: 7;
        margin-top: 1;
    }

    .execute-page {
        height: auto;
        padding: 1;
        background: #101719;
        border-left: solid #d2a957;
    }

    Collapsible.server {
        height: auto;
        margin-bottom: 1;
        padding: 0 1;
        border: solid #3e5962;
        background: #151f22;
    }

    Collapsible.server.running { border-left: heavy #7fbe72; }
    Collapsible.server.stopped { border-left: heavy #7a858c; }
    Collapsible.server.failed { border-left: heavy #d96b66; }

    .server-detail {
        height: 1;
        margin-left: 2;
    }

    .server-actions {
        margin-left: 1;
    }

    .error-text {
        height: auto;
        margin: 1 2;
        padding: 1;
        background: #301d1f;
        color: #ffb3b3;
        border-left: solid #d95757;
    }

    #reopen-subagents {
        width: auto;
        height: 1;
        min-height: 1;
        padding: 0 1;
        margin-bottom: 1;
    }

    .hidden {
        display: none;
    }

    .subagent-demo-note {
        height: auto;
        margin: 0 0 1 0;
        color: #b8c1c7;
    }

    .subagent-scenario-row {
        height: 3;
        margin-bottom: 1;
        align: left middle;
    }

    .subagent-scenario-label {
        width: 18;
        height: 3;
        margin-right: 1;
        content-align: left middle;
        color: #b8c1c7;
    }

    #subagent-scenario {
        width: 32;
        height: 3;
        border: none;
        background: transparent;
    }

    #subagent-scenario > SelectCurrent {
        border: solid #56616a;
        background: #151a1d;
        color: #eef7f8;
    }

    #subagent-scenario:focus > SelectCurrent {
        border: solid #5bb8b1;
    }

    #subagents-panel {
        height: auto;
        padding: 0 1 1 1;
        background: #101516;
        border-top: solid #b7a4e8;
        color: #ece7ff;
        overflow-x: hidden;
    }

    #subagents-panel:focus-within {
        background-tint: transparent;
    }

    #subagents-panel > CollapsibleTitle {
        width: 1fr;
        padding: 0;
        color: #ece7ff;
        text-style: bold;
    }

    .subagent-title-row {
        height: 1;
        width: 1fr;
    }

    .subagent-title-row > CollapsibleTitle {
        width: 1fr;
        height: 1;
        padding: 0;
        color: #ece7ff;
        text-style: bold;
    }

    .subagent-title-row > CollapsibleTitle:focus {
        background: transparent;
        color: #ece7ff;
        text-style: bold;
    }

    .subagent-title-row > CollapsibleTitle:hover,
    .subagent-title-row > CollapsibleTitle:focus:hover {
        background: #17313a;
        color: #eef7f8;
    }

    .subagent-toolbar {
        height: 1;
        align: left middle;
    }

    .subagent-groups-label {
        width: 27%;
        min-width: 10;
        max-width: 34;
        height: 1;
        padding-left: 1;
        border-right: solid #4a5360;
        color: #b8c1c7;
        text-style: bold;
    }

    .subagent-task-label {
        width: 1fr;
        padding-left: 2;
        color: #b8c1c7;
        text-style: bold;
    }

    .subagent-status-label {
        width: 12;
        color: #b8c1c7;
        text-style: bold;
    }

    .subagent-time-label {
        width: 10;
        color: #b8c1c7;
        text-align: right;
        text-style: bold;
    }

    #close-subagents {
        width: 3;
        min-width: 3;
        height: 1;
        min-height: 1;
        padding: 0;
        margin: 0;
        background: transparent;
        color: #b8c1c7;
    }

    .subagent-columns {
        height: 10;
        overflow-x: hidden;
    }

    .groups-column {
        width: 27%;
        min-width: 10;
        max-width: 34;
        padding-right: 1;
        margin-right: 1;
        border-right: solid #4a5360;
    }

    .tasks-column {
        width: 1fr;
        padding-left: 1;
    }

    #subagent-groups {
        height: 1fr;
        padding: 0;
        border: none;
        background: transparent;
        scrollbar-color: #5bb8b1;
        scrollbar-background: #122023;
    }

    #subagent-groups:focus {
        background: transparent;
        background-tint: transparent;
    }

    #subagent-groups > .option-list--option {
        padding: 0;
    }

    #subagent-groups > .option-list--option-highlighted {
        background: transparent;
        color: #b8c1c7;
    }

    #subagent-groups:focus > .option-list--option-highlighted {
        background: transparent;
        color: #b8c1c7;
        text-style: none;
    }

    #subagent-groups > .option-list--option-hover,
    #subagent-groups:focus > .option-list--option-hover {
        background: #17313a;
    }

    #subagent-tasks {
        height: 1fr;
        border: none;
        background: transparent;
        overflow-x: hidden;
        scrollbar-size-horizontal: 0;
        scrollbar-color: #5bb8b1;
        scrollbar-background: #122023;
    }

    #subagent-tasks:focus {
        background: transparent;
        background-tint: transparent;
    }

    #subagent-tasks > .datatable--cursor {
        background: transparent;
        color: #e8edef;
        text-style: none;
    }

    #subagent-tasks:focus > .datatable--cursor {
        background: transparent;
        color: #e8edef;
        text-style: none;
    }

    #subagent-tasks > .datatable--hover {
        background: transparent;
    }

    .loading-card {
        padding-top: 2;
        padding-bottom: 2;
    }

    .loading-row, .comparison-row {
        height: 3;
        align: left middle;
    }

    .loading-row LoadingIndicator, .comparison-row LoadingIndicator {
        width: 7;
        height: 1;
        color: #5bb8b1;
        margin-right: 1;
    }

    .loading-label, .braille-sample {
        width: 36;
        color: #eef7f8;
    }

    .elapsed {
        width: 10;
        color: #98a5ac;
        text-align: right;
    }

    .comparison-card {
        margin-top: 1;
    }

    .comparison-row .hint {
        width: 1fr;
    }

    .braille-sample {
        width: 44;
        color: #d2a957;
    }

    .chat-demo-note {
        height: auto;
        margin: 0 1 1 1;
        color: #b8c1c7;
    }

    #tool-chat-log {
        height: auto;
        padding: 1;
        border: solid #2d5661;
        background: #0c0f10;
    }

    .chat-message {
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
        background: transparent;
        color: #e8edef;
    }

    .chat-user {
        border: solid #d6b35a;
    }

    .chat-assistant {
        border: solid #5bb8b1;
    }

    .tool-bubble {
        border: solid #8fa3b8;
    }

    .args-collapsible {
        height: auto;
        padding: 0;
        margin: 0;
        background: transparent;
        border: none;
    }

    .args-collapsible > CollapsibleTitle {
        width: 1fr;
        height: auto;
        padding: 0;
        color: #e8edef;
        text-wrap: wrap;
    }

    .args-collapsible > CollapsibleTitle:focus {
        background: transparent;
        color: #e8edef;
        text-style: none;
    }

    .args-collapsible > CollapsibleTitle:hover,
    .args-collapsible > CollapsibleTitle:focus:hover {
        background: #17313a;
        color: #eef7f8;
    }

    .args-collapsible > Contents {
        padding: 0 0 0 3;
    }

    .args-json {
        padding: 0 1;
        margin: 0 0 0 2;
        background: #101719;
        border-left: solid #d2a957;
        border-right: none;
        border-top: none;
        border-bottom: none;
        scrollbar-color: #d2a957;
        scrollbar-background: #20272b;
    }

    .args-json:focus {
        border-left: solid #7ce3dc;
        border-right: none;
        border-top: none;
        border-bottom: none;
    }

    .tool-output {
        height: auto;
        max-height: 6;
        padding: 0;
        background: transparent;
        color: #b8c1c7;
        overflow: hidden;
    }

    .tool-duration {
        height: 1;
        padding: 0;
        color: #98a5ac;
    }

    Footer {
        background: #101b1f;
        color: #b8c1c7;
    }
    """

    DEMOS = (
        "HITL / Ask User",
        "Settings navigation",
        "Execute environment",
        "MCP servers",
        "Subagents panel",
        "Loading indicators",
        "Expandable tool args",
    )

    DEMO_IDS = (
        "demo-hitl",
        "demo-settings",
        "demo-execute",
        "demo-mcp",
        "demo-subagents",
        "demo-loading",
        "demo-tool-args",
    )

    def __init__(self) -> None:
        super().__init__()
        self.register_theme(MIRA_THEME)
        self.theme = MIRA_THEME.name

    def compose(self) -> ComposeResult:
        yield Static("MIRA  /  Native Textual UI explorations", id="topbar")
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield Label("DEMOS", id="sidebar-title")
                yield OptionList(*self.DEMOS, id="demo-nav")
            with ContentSwitcher(initial="demo-hitl", id="demo-switcher"):
                yield HitlDemo(id="demo-hitl")
                yield SettingsDemo(id="demo-settings")
                yield ExecuteDemo(id="demo-execute")
                yield McpDemo(id="demo-mcp")
                yield SubagentsDemo(id="demo-subagents")
                yield LoadingDemo(id="demo-loading")
                yield ToolArgsDemo(id="demo-tool-args")
        yield Footer()

    def on_mount(self) -> None:
        navigation = self.query_one("#demo-nav", OptionList)
        navigation.highlighted = 0
        navigation.focus()

    @on(OptionList.OptionSelected, "#demo-nav")
    def switch_demo(self, event: OptionList.OptionSelected) -> None:
        self.query_one("#demo-switcher", ContentSwitcher).current = self.DEMO_IDS[
            event.option_index
        ]


if __name__ == "__main__":
    MiraNativeMockup().run()
