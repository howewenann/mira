"""Focused tests for current-context composition and its modal."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import HumanMessage
from textual.app import App, ComposeResult
from textual.color import Color
from textual.widgets import Button, Collapsible, ProgressBar, Static

from agent.middleware.context_report import ContextReportMiddleware
from agent.mcp.manager import MCPManager
from agent.mcp.models import MCPServerState
from runtime.context_report import (
    ContextReportMCPServer,
    ContextReportObservation,
    build_context_report,
    contributor_share,
    current_context_values,
    estimate_conversation_tokens,
    estimate_text_tokens,
    estimated_token_text,
    format_memory_prompt,
    format_skills_prompt,
    mcp_tool_metadata,
    split_system_prompt,
)
from runtime.context_usage import context_usage_scope
from ui.command_help import command_help_entries
from ui.widgets.context_report import ContextReportRow, ContextReportScreen, ContextReportTools


def schema_tool(
    name: str,
    description: str = "tool",
    *,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    tool: dict[str, object] = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}},
        },
    }
    if metadata is not None:
        tool["metadata"] = metadata
    return tool


def contributor(report: object, label: str):
    return next(row for row in report.rows if row.label == label)  # type: ignore[attr-defined]


def tool_child(report: object, label: str):
    tools = contributor(report, "Tools")
    return next(row for row in tools.children if row.label == label)


def populated_report():
    duck_tools = (
        schema_tool(
            "mcp__duckduckgo__search",
            metadata={"mira_mcp": {"server": "duckduckgo", "tool": "search"}},
        ),
        schema_tool(
            "mcp__duckduckgo__news",
            metadata={"mira_mcp": {"server": "duckduckgo", "tool": "news"}},
        ),
    )
    return build_context_report(
        observation=ContextReportObservation(
            "instructions",
            (HumanMessage(content="hello"),),
            (schema_tool("read_file"), schema_tool("project_search"), *duck_tools),
        ),
        current_tokens=10_500,
        limit_tokens=64_000,
        memory_prompt="memory",
        memory_files=1,
        skills_prompt="skills",
        skills_count=1,
        tool_metadata=(
            {"name": "read_file", "source": "built-in"},
            {"name": "project_search", "source": "project"},
        ),
        mcp_servers=(
            ContextReportMCPServer("duckduckgo", "Available"),
            ContextReportMCPServer("github", "Disabled"),
            ContextReportMCPServer("oauth-test", "Failed", "login failed"),
        ),
    )


class ContextReportTests(unittest.TestCase):
    def test_capture_middleware_observes_effective_request_without_changing_it(self) -> None:
        calls: list[dict[str, object]] = []
        request = SimpleNamespace(
            messages=[HumanMessage(content="hello")],
            system_message="system",
            tools=[schema_tool("visible")],
        )

        with context_usage_scope(calls.append):
            result = ContextReportMiddleware().wrap_model_call(request, lambda value: value)

        self.assertIs(result, request)
        observation = calls[0]["context_report_observation"]
        self.assertIsInstance(observation, ContextReportObservation)
        self.assertEqual(observation.system_prompt, "system")  # type: ignore[union-attr]
        self.assertEqual(len(observation.tools), 1)  # type: ignore[union-attr]

    def test_text_estimate_uses_four_characters_and_rounds_up(self) -> None:
        self.assertEqual(estimate_text_tokens(""), 0)
        self.assertEqual(estimate_text_tokens("1234"), 1)
        self.assertEqual(estimate_text_tokens("12345"), 2)
        self.assertEqual(estimated_token_text(0), "0")

    def test_report_has_five_mutually_exclusive_contributors(self) -> None:
        report = populated_report()

        self.assertEqual(
            [row.label for row in report.rows],
            ["Instructions", "Memory", "Skills", "Tools", "Conversation"],
        )
        self.assertEqual(report.estimated_tokens, sum(row.tokens or 0 for row in report.rows))
        self.assertEqual(report.accounted_tokens, report.estimated_tokens)
        self.assertEqual(
            report.injected_tokens,
            sum(row.tokens or 0 for row in report.rows if row.label != "Conversation"),
        )

    def test_summary_distinguishes_live_usage_from_estimated_share(self) -> None:
        report = populated_report()
        self.assertEqual(current_context_values(report), ("10.5k / 64.0k", "16%"))
        self.assertEqual(len(report.rows), 5)
        self.assertIsNotNone(report.share_total)

    def test_pending_current_context_does_not_claim_zero_or_use_io_totals(self) -> None:
        report = build_context_report(
            observation=None,
            current_tokens=0,
            limit_tokens=64_000,
            memory_prompt="",
            skills_prompt="",
        )

        self.assertEqual(current_context_values(report), ("pending / 64.0k", "—"))
        self.assertIsNone(report.current_tokens)

    def test_conversation_ignores_cumulative_usage_metadata(self) -> None:
        base = {"role": "assistant", "content": "answer"}
        with_usage = {**base, "usage_metadata": {"input_tokens": 999_999, "output_tokens": 999_999}}

        self.assertEqual(
            estimate_conversation_tokens([base]),
            estimate_conversation_tokens([with_usage]),
        )

    def test_contributor_shares_use_overall_estimated_total(self) -> None:
        report = populated_report()
        shares = [contributor_share(row.tokens, report.estimated_tokens) or 0 for row in report.rows]

        self.assertAlmostEqual(sum(shares), 100.0)
        mcp = tool_child(report, "MCP")
        duck = next(row for row in mcp.children if row.label.startswith("duckduckgo"))
        self.assertEqual(
            contributor_share(duck.tokens, report.estimated_tokens),
            contributor_share(mcp.tokens, report.estimated_tokens),
        )

    def test_live_tool_metadata_detects_two_duckduckgo_tools(self) -> None:
        report = populated_report()
        mcp = tool_child(report, "MCP")
        duck = next(row for row in mcp.children if row.label.startswith("duckduckgo"))

        self.assertEqual(duck.label, "duckduckgo · 2 tools")
        self.assertGreater(duck.tokens or 0, 0)

    def test_disabled_and_failed_mcp_servers_contribute_no_tokens(self) -> None:
        mcp = tool_child(populated_report(), "MCP")
        states = {row.label: row for row in mcp.children}

        self.assertIsNone(states["github · disabled"].tokens)
        self.assertIsNone(states["oauth-test · failed"].tokens)
        self.assertIn("login failed", states["oauth-test · failed"].detail)

    def test_builtin_custom_and_mcp_tools_are_categorized_from_live_surface(self) -> None:
        report = populated_report()

        self.assertGreater(tool_child(report, "Built-in").tokens or 0, 0)
        self.assertGreater(tool_child(report, "Custom").tokens or 0, 0)
        self.assertGreater(tool_child(report, "MCP").tokens or 0, 0)

    def test_pre_request_mcp_fallback_uses_mode_visible_server_tools(self) -> None:
        duck = schema_tool(
            "mcp__duckduckgo__search",
            metadata={"mira_mcp": {"server": "duckduckgo", "tool": "search"}},
        )
        report = build_context_report(
            observation=None,
            current_tokens=None,
            limit_tokens=64_000,
            memory_prompt="",
            skills_prompt="",
            mcp_servers=(ContextReportMCPServer("duckduckgo", "Available", tools=(duck,)),),
        )
        server = tool_child(report, "MCP").children[0]

        self.assertEqual(server.label, "duckduckgo · 1 tool")
        self.assertGreater(server.tokens or 0, 0)

    def test_duckduckgo_count_flows_through_mcp_manager_mode_selection(self) -> None:
        duck_tools = [
            schema_tool(
                f"mcp__duckduckgo__{name}",
                metadata={"mira_mcp": {"server": "duckduckgo", "tool": name}},
            )
            for name in ("search", "news")
        ]
        state = MCPServerState(
            name="duckduckgo",
            transport="stdio",
            config={},
            fingerprint="test",
            status="Available",
            tools=duck_tools,
            tool_metadata=[
                {
                    "name": f"mcp__duckduckgo__{name}",
                    "original_name": name,
                    "server": "duckduckgo",
                    "source": "mcp",
                }
                for name in ("search", "news")
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            manager = MCPManager(Path(directory))
            manager.servers = {state.name: state}
            selected, _ = manager.tools_for_mode({}, planning=False)
        selected_duck = tuple(
            tool
            for tool in selected
            if (mcp_tool_metadata(tool) or {}).get("server") == "duckduckgo"
        )
        report = build_context_report(
            observation=None,
            current_tokens=None,
            limit_tokens=64_000,
            memory_prompt="",
            skills_prompt="",
            mcp_servers=(
                ContextReportMCPServer(
                    "duckduckgo",
                    "Available",
                    tools=selected_duck,
                ),
            ),
        )

        self.assertEqual(tool_child(report, "MCP").children[0].label, "duckduckgo · 2 tools")

    def test_memory_and_skills_are_removed_from_instructions_once(self) -> None:
        memory = format_memory_prompt({"/AGENTS.md": "Memory"}, ["/AGENTS.md"])
        skills = format_skills_prompt([], ["/skills"])
        system = f"base\n\n{skills}\n\n{memory}"
        instructions, observed_memory, observed_skills = split_system_prompt(
            system,
            memory_prompt=memory,
            skills_prompt=skills,
        )
        report = build_context_report(
            observation=ContextReportObservation(system, (), ()),
            current_tokens=None,
            limit_tokens=None,
            memory_prompt=memory,
            memory_files=1,
            skills_prompt=skills,
        )

        self.assertNotIn("<agent_memory>", instructions)
        self.assertNotIn("## Skills System", instructions)
        self.assertEqual(len(instructions) + len(memory) + len(skills), len(system))
        self.assertEqual(contributor(report, "Instructions").tokens, estimate_text_tokens(instructions))
        self.assertIsNotNone(observed_memory)
        self.assertIsNotNone(observed_skills)
        self.assertEqual(contributor(report, "Memory").tokens, estimate_text_tokens(memory))
        self.assertEqual(contributor(report, "Skills").tokens, estimate_text_tokens(skills))


class ContextReportHost(App[None]):
    CSS_PATH = "../ui/styles/mira.tcss"

    def __init__(self, report=None) -> None:
        super().__init__()
        self.report = report or populated_report()

    def compose(self) -> ComposeResult:
        yield Static("host")

    def on_mount(self) -> None:
        self.push_screen(ContextReportScreen(self.report))


class ContextReportScreenTests(unittest.IsolatedAsyncioTestCase):
    async def test_feature_name_command_and_summary_content(self) -> None:
        commands = {usage for usage, _description in command_help_entries()}
        self.assertIn("/context-report", commands)
        retired_command = "/context-" + "doc" + "tor"
        self.assertNotIn(retired_command, commands)

        app = ContextReportHost()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            screen = app.screen

            self.assertIsInstance(screen, ContextReportScreen)
            self.assertEqual(
                screen.query_one("#context-report-title", Static).render().plain,
                "Context Report",
            )
            title = screen.query_one("#context-report-title", Static)
            close = screen.query_one("#context-report-close", Button)
            self.assertIsNone(screen.focused)
            self.assertFalse(close.has_focus)
            self.assertEqual(close.label.plain, "x")
            self.assertTrue(close.has_class("panel-close"))
            self.assertLessEqual(title.region.right, close.region.x)
            self.assertNotIn("Close", [button.label.plain for button in screen.query(Button)])
            self.assertEqual(len(screen.query(ProgressBar)), 0)
            visible_text = " ".join(widget.render().plain for widget in screen.query(Static))
            self.assertNotIn("Estimated accounted context", visible_text)
            self.assertNotIn("Estimation delta", visible_text)
            self.assertNotIn("memory files", visible_text)
            self.assertNotIn("loaded skill", visible_text)
            await pilot.click("#context-report-close")
            await pilot.pause()
            self.assertNotIsInstance(app.screen, ContextReportScreen)

    async def test_only_tools_and_mcp_are_collapsible(self) -> None:
        app = ContextReportHost()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            screen = app.screen

            self.assertEqual(len(screen.query(Collapsible)), 2)
            self.assertIsInstance(
                screen.query_one("#context-report-row-tools"),
                ContextReportTools,
            )
            self.assertIsInstance(screen.query_one("#context-report-tool-mcp"), Collapsible)
            for label in ("instructions", "memory", "skills", "conversation"):
                self.assertIsInstance(
                    screen.query_one(f"#context-report-row-{label}"),
                    ContextReportRow,
                )

    async def test_tools_and_mcp_expand_to_groups_then_server_rows(self) -> None:
        app = ContextReportHost()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.click(".context-report-tools-header > CollapsibleTitle")
            await pilot.pause()
            tools = app.screen.query_one("#context-report-row-tools", Collapsible)
            self.assertFalse(tools.collapsed)

            for label in ("built-in", "custom", "mcp"):
                self.assertIsNotNone(app.screen.query_one(f"#context-report-tool-{label}"))
            self.assertEqual(len(app.screen.query("#context-report-row-tools Collapsible")), 1)
            mcp = app.screen.query_one("#context-report-tool-mcp", Collapsible)
            self.assertTrue(mcp.collapsed)

            await pilot.click(".context-report-mcp-header > CollapsibleTitle")
            await pilot.pause()
            self.assertFalse(mcp.collapsed)
            duck = app.screen.query_one(
                "#context-report-server-duckduckgo---2-tools .context-report-label",
                Static,
            ).render().plain
            self.assertEqual(duck, "duckduckgo · 2 tools")

    async def test_disclosure_hover_highlights_without_moving_the_glyph(self) -> None:
        app = ContextReportHost()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            tools_title = app.screen.query_one(".context-report-tools-header > CollapsibleTitle")

            async def assert_hover_is_stable(title, label: str) -> None:
                await pilot.hover("#context-report-close")
                await pilot.pause()
                plain = title.render().plain
                before = (
                    title.region,
                    title.styles.width,
                    title.styles.height,
                    title.styles.padding,
                    title.styles.text_style,
                    plain.index(label),
                    len(plain.partition(" ")[0]),
                )
                self.assertEqual(title.styles.background.a, 0)
                await pilot.hover(title)
                await pilot.pause()
                hovered = title.render().plain
                after = (
                    title.region,
                    title.styles.width,
                    title.styles.height,
                    title.styles.padding,
                    title.styles.text_style,
                    hovered.index(label),
                    len(hovered.partition(" ")[0]),
                )
                self.assertEqual(after, before)
                self.assertEqual(title.styles.background, Color.parse("#1b3036"))
                self.assertEqual(title.styles.color, Color.parse("#eef7f8"))

            await assert_hover_is_stable(tools_title, "Tools")
            await pilot.click(tools_title)
            await pilot.pause()
            await assert_hover_is_stable(tools_title, "Tools")
            mcp_title = app.screen.query_one(".context-report-mcp-header > CollapsibleTitle")
            await assert_hover_is_stable(mcp_title, "MCP")
            await pilot.click(mcp_title)
            await pilot.pause()
            await assert_hover_is_stable(mcp_title, "MCP")

    async def test_primary_and_detail_numbers_use_the_intended_color_hierarchy(self) -> None:
        app = ContextReportHost()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.click(".context-report-tools-header > CollapsibleTitle")
            await pilot.pause()
            await pilot.click(".context-report-mcp-header > CollapsibleTitle")
            await pilot.pause()
            white = Color.parse("#eef7f8")
            gray = Color.parse("#9fb0b6")

            for label in ("instructions", "memory", "skills", "conversation"):
                row = app.screen.query_one(f"#context-report-row-{label}")
                self.assertEqual(row.query_one(".context-report-value").styles.color, white)
                self.assertEqual(row.query_one(".context-report-percent").styles.color, white)
            tools = app.screen.query_one(".context-report-tools-header")
            self.assertEqual(tools.query_one(".context-report-value").styles.color, white)
            self.assertEqual(tools.query_one(".context-report-percent").styles.color, white)
            mcp = app.screen.query_one("#context-report-tool-mcp")
            self.assertEqual(mcp.query_one(".context-report-value").styles.color, white)
            self.assertEqual(mcp.query_one(".context-report-percent").styles.color, white)

            for selector in (
                "#context-report-tool-built-in",
                "#context-report-tool-custom",
                ".context-report-level-2",
            ):
                row = app.screen.query_one(selector)
                self.assertEqual(row.query_one(".context-report-value").styles.color, gray)
                self.assertEqual(row.query_one(".context-report-percent").styles.color, gray)

    async def test_numeric_columns_keep_fixed_positions_when_tools_expands(self) -> None:
        app = ContextReportHost()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()

            def positions(selector: str) -> tuple[int, int]:
                row = app.screen.query_one(selector)
                return (
                    row.query_one(".context-report-value").region.x,
                    row.query_one(".context-report-percent").region.x,
                )

            selectors = (
                "#context-report-current-header",
                "#context-report-current",
                "#context-report-contributor-header",
                "#context-report-row-instructions",
                ".context-report-tools-header",
            )
            collapsed = [positions(selector) for selector in selectors]
            self.assertEqual(len(set(collapsed)), 1)

            await pilot.click(".context-report-tools-header > CollapsibleTitle")
            await pilot.pause()
            expanded = [positions(selector) for selector in selectors]
            expanded.extend(
                positions(f"#context-report-tool-{label}")
                for label in ("built-in", "custom", "mcp")
            )
            self.assertEqual(len(set(expanded)), 1)
            self.assertEqual(expanded[: len(selectors)], collapsed)

    async def test_long_labels_cannot_intrude_into_numeric_columns(self) -> None:
        long_name = "server-" + ("very-long-name-" * 12)
        report = build_context_report(
            observation=ContextReportObservation("instructions", (), ()),
            current_tokens=1,
            limit_tokens=64_000,
            memory_prompt="",
            skills_prompt="",
            mcp_servers=(ContextReportMCPServer(long_name, "Disabled"),),
        )
        app = ContextReportHost(report)
        async with app.run_test(size=(76, 30)) as pilot:
            await pilot.click(".context-report-tools-header > CollapsibleTitle")
            await pilot.pause()
            await pilot.click(".context-report-mcp-header > CollapsibleTitle")
            await pilot.pause()
            row = app.screen.query_one(".context-report-level-2")
            label = row.query_one(".context-report-label")
            value = row.query_one(".context-report-value")
            percent = row.query_one(".context-report-percent")

            self.assertLessEqual(label.region.right, value.region.x)
            self.assertLessEqual(value.region.right, percent.region.x)
            self.assertEqual(value.region.width, 18)
            self.assertEqual(percent.region.width, 10)

    async def test_used_limit_and_usage_align_with_tokens_and_share(self) -> None:
        app = ContextReportHost()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            current_header = app.screen.query_one("#context-report-current-header")
            contributor_header = app.screen.query_one("#context-report-contributor-header")

            self.assertEqual(
                current_header.query_one(".context-report-value").region.x,
                contributor_header.query_one(".context-report-value").region.x,
            )
            self.assertEqual(
                current_header.query_one(".context-report-percent").region.x,
                contributor_header.query_one(".context-report-percent").region.x,
            )

    async def test_current_context_value_fits_inline_without_wrapping(self) -> None:
        app = ContextReportHost()
        async with app.run_test(size=(76, 30)) as pilot:
            await pilot.pause()
            row = app.screen.query_one("#context-report-current")
            value = row.query_one(".context-report-value", Static)
            percent = row.query_one(".context-report-percent", Static)

            self.assertEqual(value.render().plain, "10.5k / 64.0k")
            self.assertEqual(percent.render().plain, "16%")
            self.assertEqual(value.styles.text_wrap, "nowrap")
            self.assertEqual(value.region.height, 1)


if __name__ == "__main__":
    unittest.main()
