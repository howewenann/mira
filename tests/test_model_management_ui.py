"""Textual checks for model navigation, fresh-state labels, and Issues."""

from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agent.mcp.manager import MCPManager
from config.llm import ConfigError, ModelProfile, ModelRegistry
from config.metadata import ModelMetadata
from config.settings import (
    DEFAULT_SETTINGS,
    context_limit_tokens,
    load_settings,
    model_assignment,
    save_settings,
    set_context_limit_tokens,
    set_model_assignment,
    set_subagent_enabled,
    set_subagent_model_assignment,
)
from runtime.issues import Issue
from tests.test_textual_app import make_app, renderable_plain, wait_until
from ui.command_help import command_help_entries
from ui.widgets import IssuesScreen, PromptBox, SettingsPanel
from ui.widgets.settings_panel import INHERIT_VALUE
from textual.widgets import Button, Collapsible, Select


class ModelManagementUITests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _mcp_manager() -> SimpleNamespace:
        return SimpleNamespace(
            reload=AsyncMock(),
            shutdown=AsyncMock(),
            set_change_handler=lambda _handler: None,
            issues=[],
            show_status=False,
            servers={},
            prompt_registry=SimpleNamespace(issues=[]),
        )

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

    async def test_models_changes_rebuild_agents_without_reloading_mcp(self) -> None:
        """Every Models control should preserve MCP and share the rebuild status."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            settings = deepcopy(DEFAULT_SETTINGS)
            settings["models"]["main"] = "old"
            settings["models"]["context_limit_tokens"] = 8192
            registry = ModelRegistry(
                {
                    "old": ModelProfile("old", {"provider": "openai", "model": "old-model"}),
                    "new": ModelProfile("new", {"provider": "openai", "model": "new-model"}),
                }
            )
            manager = self._mcp_manager()
            app = make_app(
                workspace,
                config={"settings": settings, "settings_valid": True, "model_registry": registry},
                model_name="[old] openai:old-model",
                mcp_manager=manager,
            )

            async def infer_metadata(config: dict, model: object | None = None) -> ModelMetadata:
                return ModelMetadata(
                    context_limit_tokens(config),
                    "settings.models.context_limit_tokens",
                )

            with (
                patch("agent.llm.get_llm", return_value=SimpleNamespace(profile={})),
                patch(
                    "config.metadata.infer_model_metadata",
                    new_callable=AsyncMock,
                    side_effect=infer_metadata,
                ) as infer,
            ):
                async with app.run_test(size=(110, 36)) as pilot:
                    await pilot.pause()
                    app._reload_runtime = AsyncMock()  # type: ignore[method-assign]
                    app._rebuild_agents = AsyncMock()  # type: ignore[method-assign]

                    updates = [
                        lambda current: set_model_assignment(current, "main", "new"),
                        lambda current: set_model_assignment(current, "rubric", "old"),
                        lambda current: set_model_assignment(current, "summarization", "new"),
                        lambda current: set_subagent_enabled(current, "project-guide", True),
                        lambda current: set_subagent_model_assignment(current, "project-guide", "old"),
                        lambda current: set_context_limit_tokens(current, 4096),
                    ]
                    for update in updates:
                        ok, message = await app._apply_settings(update(app.config["settings"]))
                        self.assertTrue(ok)
                        self.assertEqual(message, "settings saved; agents rebuilt")

                    footer = str(app.query_one("#model-settings-button", Button).label)

            app._reload_runtime.assert_not_awaited()
            app._rebuild_agents.assert_awaited()
            self.assertEqual(app._rebuild_agents.await_count, 6)
            manager.reload.assert_not_awaited()
            self.assertEqual(infer.await_count, 2)
            self.assertEqual(
                app._rebuild_agents.await_args_list[1].kwargs["metadata"],
                ModelMetadata(8192, "settings.models.context_limit_tokens"),
            )
            self.assertEqual(app.model_name, "[new] openai:new-model")
            self.assertEqual(app.context_limit_tokens, 4096)
            self.assertEqual(app.runtime_snapshot.provider, "openai")
            self.assertEqual(footer, "model: [new] openai:new-model")

    async def test_plain_agent_rebuild_preserves_active_model_metadata(self) -> None:
        """General settings rebuilds should keep the provider-derived context."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            app = make_app(
                Path(directory),
                context_limit_tokens=4096,
                context_limit_source="provider.profile.max_input_tokens",
            )

            async with app.run_test() as pilot:
                await pilot.pause()
                with patch.object(
                    app,
                    "_build_agent_pair",
                    return_value=("rebuilt-agent", "rebuilt-plan-agent"),
                ) as build_pair:
                    await app._rebuild_agents()

            self.assertEqual(
                build_pair.call_args.kwargs["metadata"],
                ModelMetadata(4096, "provider.profile.max_input_tokens"),
            )
            self.assertEqual(app.agent, "rebuilt-agent")
            self.assertEqual(app.plan_agent, "rebuilt-plan-agent")

    async def test_models_change_without_main_keeps_expected_model_issue(self) -> None:
        """Subagent settings should rebuild locally even while Main is unset."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            manager = self._mcp_manager()
            app = make_app(
                workspace,
                config={
                    "settings": load_settings(workspace),
                    "settings_valid": True,
                    "model_registry": ModelRegistry(),
                },
                agent=None,
                plan_agent=None,
                model_name="unset",
                mcp_manager=manager,
            )

            async with app.run_test() as pilot:
                await pilot.pause()
                app._reload_runtime = AsyncMock()  # type: ignore[method-assign]
                updated = set_subagent_enabled(app.config["settings"], "project-guide", True)
                ok, message = await app._apply_settings(updated)

            self.assertTrue(ok)
            self.assertEqual(message, "settings saved; agents rebuilt")
            app._reload_runtime.assert_not_awaited()
            manager.reload.assert_not_awaited()
            self.assertIsNone(app.agent)
            self.assertIsNone(app.plan_agent)
            self.assertTrue(any(issue.summary == "Main model is not configured" for issue in app.issues))

    async def test_rejected_model_selection_restores_visible_value(self) -> None:
        """A failed immediate save should put the selector back on its prior assignment."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            settings = set_model_assignment(load_settings(workspace), "main", "old")
            self.assertTrue(save_settings(workspace, settings))
            registry = ModelRegistry(
                {
                    "old": ModelProfile("old", {"provider": "openai", "model": "old-model"}),
                    "new": ModelProfile("new", {"provider": "openai", "model": "new-model"}),
                }
            )
            app = make_app(
                workspace,
                config={"settings": settings, "model_registry": registry},
                model_name="[old] openai:old-model",
            )

            async with app.run_test(size=(110, 36)) as pilot:
                await pilot.pause()
                app._apply_settings = AsyncMock(  # type: ignore[method-assign]
                    return_value=(False, "settings not saved: rebuild failed")
                )
                app._handle_settings_command("models")
                await wait_until(lambda: len(app.query(SettingsPanel)) == 1)
                panel = app.query_one(SettingsPanel)
                await wait_until(lambda: panel._model_controls_ready)
                selector = panel.query_one("#settings-model-main", Select)
                await wait_until(lambda: selector.value == "old")
                selector.value = "new"
                event = SimpleNamespace(stop=lambda: None, select=selector, value="new")
                await panel.change_model_assignment(event)

                status = renderable_plain(panel.query_one("#settings-status"))

            self.assertEqual(app._apply_settings.await_count, 1)
            self.assertIn("settings not saved: rebuild failed", status)

    async def test_failed_model_rebuild_restores_settings_without_full_reload(self) -> None:
        """A rejected Models change should leave disk, config, and agents unchanged."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            manager = self._mcp_manager()
            app = make_app(workspace, mcp_manager=manager)

            async with app.run_test() as pilot:
                await pilot.pause()
                old_agent = app.agent
                old_plan_agent = app.plan_agent
                app._reload_runtime = AsyncMock()  # type: ignore[method-assign]
                app._rebuild_agents = AsyncMock(  # type: ignore[method-assign]
                    side_effect=ConfigError("rubric model is unavailable")
                )
                updated = set_model_assignment(app.config["settings"], "rubric", "test")
                ok, message = await app._apply_settings(updated)

            self.assertFalse(ok)
            self.assertEqual(message, "settings not saved: rubric model is unavailable")
            self.assertIsNone(model_assignment(load_settings(workspace), "rubric"))
            self.assertIsNone(model_assignment(app.config, "rubric"))
            self.assertIs(app.agent, old_agent)
            self.assertIs(app.plan_agent, old_plan_agent)
            app._reload_runtime.assert_not_awaited()
            manager.reload.assert_not_awaited()

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
