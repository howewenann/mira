"""Textual checks for model navigation, fresh-state labels, and Issues."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent.mcp.manager import MCPManager
from config.llm import ModelProfile, ModelRegistry
from config.settings import load_settings, model_assignment, set_model_assignment
from runtime.issues import Issue
from tests.test_textual_app import make_app, renderable_plain, wait_until
from ui.command_help import command_help_entries
from ui.widgets import IssuesScreen, PromptBox, SettingsPanel
from ui.widgets.settings_panel import INHERIT_VALUE
from textual.widgets import Button, Collapsible, Select


class ModelManagementUITests(unittest.IsolatedAsyncioTestCase):
    async def test_no_main_keeps_local_commands_and_blocks_only_model_turns(self) -> None:
        app = make_app(agent=None, plan_agent=None, model_name="unset")
        async with app.run_test() as pilot:
            await pilot.pause()
            prompt = app.query_one(PromptBox)
            await app.submit_prompt(PromptBox.Submitted(prompt, "/help"))
            await pilot.pause()
            output = "\n".join(renderable_plain(item) for item in app.query_one("#chat-log").children)
            self.assertIn("Commands", output)

            await app.submit_prompt(PromptBox.Submitted(prompt, "do model work"))
            await pilot.pause()
            output = "\n".join(renderable_plain(item) for item in app.query_one("#chat-log").children)
            self.assertIn("Main model is not configured. Run /models.", output)
            self.assertNotIn("error report:", output)

    async def test_models_command_and_footer_share_models_tab(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            registry = ModelRegistry(
                {"claude": ModelProfile("claude", {"provider": "anthropic", "model": "sonnet"})}
            )
            config = {"settings": load_settings(workspace), "model_registry": registry}
            app = make_app(workspace, config=config, model_name="unset")
            async with app.run_test(size=(110, 36)) as pilot:
                await pilot.pause()
                self.assertEqual(str(app.query_one("#model-settings-button", Button).label), "model: unset")
                app.query_one("#model-settings-button", Button).press()
                await wait_until(lambda: len(app.query(SettingsPanel)) == 1)
                panel = app.query_one(SettingsPanel)
                await wait_until(lambda: panel._model_controls_ready)
                self.assertFalse(panel.query_one("#settings-body").display)
                self.assertTrue(panel.query_one("#settings-models-body").display)
                self.assertFalse(panel.query_one("#settings-tab-general", Button).has_class("active"))
                self.assertTrue(panel.query_one("#settings-tab-models", Button).has_class("active"))
                await wait_until(lambda: len(panel.query("#settings-model-main")) == 1)
                await wait_until(lambda: panel.query_one("#settings-model-main", Select).value == INHERIT_VALUE)
                await wait_until(lambda: panel.query_one("#settings-models-body").display)
                self.assertTrue(panel.query_one("#settings-models-body").display)
                self.assertEqual(panel.query_one("#settings-model-main", Select).value, INHERIT_VALUE)
                self.assertEqual(panel._role_options("rubric")[0], ("unset (default)", INHERIT_VALUE))

                panel.query_one("#settings-close", Button).press()
                await wait_until(lambda: len(app.query(SettingsPanel)) == 0)
                prompt = app.query_one(PromptBox)
                await app.submit_prompt(PromptBox.Submitted(prompt, "/models"))
                await wait_until(lambda: len(app.query(SettingsPanel)) == 1)
                command_panel = app.query_one(SettingsPanel)
                await wait_until(lambda: command_panel._model_controls_ready)
                self.assertFalse(command_panel.query_one("#settings-body").display)
                self.assertTrue(command_panel.query_one("#settings-models-body").display)
                self.assertTrue(command_panel.query_one("#settings-tab-models", Button).has_class("active"))

            commands = {usage for usage, _description in command_help_entries()}
            self.assertIn("/models", commands)
            self.assertNotIn("/model", commands)

    async def test_disabling_mcp_without_main_keeps_runtime_available(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            mcp_directory = workspace / ".mira" / "mcp"
            mcp_directory.mkdir(parents=True)
            (mcp_directory / "mcp.json").write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "docs": {"type": "http", "url": "https://example.test/mcp"},
                        }
                    }
                ),
                encoding="utf-8",
            )
            manager = MCPManager(workspace, token_root=workspace / "tokens")
            config = {
                "settings": load_settings(workspace),
                "settings_valid": True,
                "model_registry": ModelRegistry(),
            }
            app = make_app(
                workspace,
                config=config,
                agent=None,
                plan_agent=None,
                model_name="unset",
                mcp_manager=manager,
            )

            async with app.run_test() as pilot:
                await pilot.pause()
                self.assertTrue(await manager.set_server_enabled("docs", False))
                await pilot.pause()

                self.assertEqual(manager.servers["docs"].status, "Disabled")
                self.assertIsNone(app.agent)
                self.assertIsNone(app.plan_agent)
                self.assertEqual(app.agent_unavailable_message, "Main model is not configured. Run /models.")
                self.assertTrue(app.ready)

    async def test_models_layout_uses_aligned_full_height_rows(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            config = {"settings": load_settings(workspace), "model_registry": ModelRegistry()}
            subagents = [
                {"name": "general-purpose", "kind": "raw", "description": "General work."},
                {"name": "example-project-guide", "kind": "raw", "description": "Project guidance."},
            ]
            app = make_app(
                workspace,
                config=config,
                model_name="unset",
                resource_metadata={"subagents": subagents},
            )

            async with app.run_test(size=(110, 48)) as pilot:
                await pilot.pause()
                app.query_one("#model-settings-button", Button).press()
                await wait_until(lambda: len(app.query(SettingsPanel)) == 1)
                panel = app.query_one(SettingsPanel)
                await wait_until(lambda: panel._model_controls_ready)
                await pilot.pause()

                context_section = panel.query_one(".settings-section.model-context")
                context_row = panel.query_one(".settings-context-row")
                context_input = panel.query_one("#settings-model-context-limit")
                assignments_section = panel.query_one(".settings-section.model-assignments")
                subagents_section = panel.query_one(".settings-section.subagents")
                header = panel.query_one(".settings-subagent-header")
                enable_header = header.query_one(".settings-column-label.enabled")
                model_header = header.query_one(".settings-column-label.model")
                rows = list(panel.query(".settings-subagent-row"))

                self.assertEqual(context_section.styles.margin.top, 2)
                self.assertEqual(assignments_section.styles.margin.top, 1)
                self.assertEqual(header.styles.margin.top, 1)
                self.assertEqual(context_row.region.height, 3)
                self.assertEqual(context_input.region.y, context_row.region.y)
                self.assertEqual(context_input.region.height, context_row.region.height)
                self.assertEqual(len(rows), 2)

                for row in rows:
                    toggle = row.query_one(".settings-toggle")
                    selector = row.query_one(".settings-subagent-model-select")
                    self.assertEqual(selector.region.width, 28)
                    self.assertEqual(toggle.region.x, enable_header.region.x)
                    self.assertEqual(toggle.region.y, selector.region.y + 1)
                    self.assertEqual(
                        selector.region.x * 2 + selector.region.width,
                        model_header.region.x * 2 + model_header.region.width,
                    )

                self.assertLess(subagents_section.region.y, header.region.y)

    async def test_inherited_labels_follow_main_while_explicit_assignments_stay_pinned(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            settings = load_settings(Path(directory))
        settings = set_model_assignment(settings, "main", "claude")
        settings = set_model_assignment(settings, "summarization", "gpt")
        registry = ModelRegistry(
            {
                "claude": ModelProfile("claude", {"provider": "anthropic", "model": "sonnet"}),
                "gpt": ModelProfile("gpt", {"provider": "openai", "model": "gpt-4.1"}),
            }
        )
        panel = SettingsPanel(
            settings=settings,
            tool_metadata=[],
            apply_change=lambda _settings: None,
            model_registry=registry,
        )

        self.assertEqual(panel._role_options("main"), [("claude", "claude"), ("gpt", "gpt")])
        self.assertEqual(panel._role_options("rubric")[0], ("claude (default)", INHERIT_VALUE))
        self.assertEqual(model_assignment(settings, "summarization"), "gpt")

        panel.settings = set_model_assignment(settings, "main", "gpt")
        self.assertEqual(panel._role_options("rubric")[0], ("gpt (default)", INHERIT_VALUE))
        self.assertEqual(model_assignment(panel.settings, "summarization"), "gpt")
        source_options = panel._subagent_options(
            {"name": "reviewer", "kind": "raw", "source_model": "[defined] anthropic:haiku"}
        )
        self.assertEqual(str(source_options[0][0]), "[defined] anthropic:haiku")
        self.assertEqual(str(panel._subagent_options({"name": "graph", "kind": "compiled"})[0][0]), "[compiled]")
        self.assertEqual(
            str(panel._subagent_options({"name": "async", "kind": "async", "graph_id": "jobs"})[0][0]),
            "[async] jobs",
        )

    async def test_issues_screen_is_flat_collapsed_and_view_only(self) -> None:
        issue = Issue("MODEL", "Main model is not configured", ".mira/settings.yml", "Unset", "Run /models.")
        app = make_app(issues=[issue], model_name="unset")
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("#issues-button", Button).press()
            await wait_until(lambda: isinstance(app.screen, IssuesScreen))
            await wait_until(lambda: len(app.screen.query(Collapsible)) == 1)
            rows = list(app.screen.query(Collapsible))
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0].collapsed)
            self.assertEqual(list(app.screen.query("Input")), [])


if __name__ == "__main__":
    unittest.main()
