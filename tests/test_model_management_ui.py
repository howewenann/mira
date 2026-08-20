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
    subagent_model_assignment,
)
from runtime.issues import Issue
from tests.test_textual_app import make_app, renderable_plain, wait_until
from ui.command_help import command_help_entries
from ui.widgets import IssuesScreen, PromptBox, SettingsPanel
from ui.widgets.settings_panel import INHERIT_VALUE
from textual.color import Color
from textual.geometry import Region
from textual.widgets import Button, Collapsible, ContentSwitcher, Select, Static


def rendered_lines(widget: object) -> list[str]:
    """Return the exact terminal-cell lines rendered by a mounted widget."""
    region = getattr(widget, "region")
    return [
        strip.text
        for strip in widget.render_lines(Region(0, 0, region.width, region.height))
    ]


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
                self.assertEqual(
                    panel.query_one("#settings-switcher", ContentSwitcher).current,
                    "settings-models-body",
                )
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
                self.assertEqual(
                    command_panel.query_one("#settings-switcher", ContentSwitcher).current,
                    "settings-models-body",
                )
                self.assertFalse(command_panel.query_one("#settings-body").display)
                self.assertTrue(command_panel.query_one("#settings-models-body").display)
                self.assertTrue(command_panel.query_one("#settings-tab-models", Button).has_class("active"))

            commands = {usage for usage, _description in command_help_entries()}
            self.assertIn("/models", commands)
            self.assertNotIn("/model", commands)

    async def test_footer_model_button_is_bright_left_aligned_and_responsive(self) -> None:
        """The footer shortcut should preserve its template and ellipsize only when needed."""
        long_identity = "[lmstudio-gemma] openai:google/gemma-4-12b"
        rendered_by_width: dict[int, str] = {}
        button_widths: list[int] = []

        for width in (80, 110, 140):
            app = make_app(model_name=long_identity)
            async with app.run_test(size=(width, 20)) as pilot:
                await pilot.pause()
                row = app.query_one("#telemetry-row")
                button = app.query_one("#model-settings-button", Button)
                telemetry = app.query_one("#telemetry-values")
                rendered = rendered_lines(button)[0].strip()

                self.assertEqual(str(button.label), f"model: {long_identity}")
                self.assertEqual(button.styles.background, Color.parse("#8897E8"))
                self.assertEqual(button.styles.color, Color.parse("#0C0F10"))
                self.assertEqual(button.styles.content_align_horizontal, "left")
                self.assertEqual(button.styles.text_align, "left")
                self.assertLessEqual(button.region.width, int(row.region.width * 0.6) + 1)
                self.assertGreater(telemetry.region.width, 0)
                self.assertEqual(button.region.right, telemetry.region.x)

                if width == 110:
                    button.focus()
                    await pilot.pause()
                    self.assertEqual(button.styles.background, Color.parse("#A8B3FF"))

                rendered_by_width[width] = rendered
                button_widths.append(button.region.width)

        self.assertTrue(rendered_by_width[80].startswith("model: [lmstudio"))
        self.assertTrue(rendered_by_width[80].endswith("…"))
        self.assertTrue(rendered_by_width[110].endswith("…"))
        self.assertEqual(rendered_by_width[140], f"model: {long_identity}")
        self.assertEqual(button_widths, sorted(button_widths))

        app = make_app(model_name="unset")
        async with app.run_test(size=(80, 20)) as pilot:
            await pilot.pause()
            button = app.query_one("#model-settings-button", Button)
            self.assertEqual(rendered_lines(button)[0].strip(), "model: unset")

    async def test_footer_model_button_reflows_immediately_when_identity_changes(self) -> None:
        """A runtime model update should resize the mounted footer button immediately."""
        identity = "[lmstudio-gemma] openai:google/gemma-4-12b"
        app = make_app(model_name="unset")

        async with app.run_test(size=(140, 20)) as pilot:
            await pilot.pause()
            button = app.query_one("#model-settings-button", Button)
            unset_width = button.region.width

            app.model_name = identity
            app._set_status(state=app.status_state)
            await pilot.pause()

            self.assertGreater(button.region.width, unset_width)
            self.assertEqual(rendered_lines(button)[0].strip(), f"model: {identity}")

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
            profile = "lmstudio-gemma-profile-name-that-is-deliberately-long"
            settings = set_model_assignment(load_settings(workspace), "main", profile)
            self.assertTrue(save_settings(workspace, settings))
            config = {
                "settings": settings,
                "model_registry": ModelRegistry(
                    {
                        profile: ModelProfile(
                            profile,
                            {"provider": "openai", "model": "google/gemma-4-12b"},
                        )
                    }
                ),
            }
            subagents = [
                {"name": "general-purpose", "kind": "raw", "description": "General work."},
                {
                    "name": "extremely-long-user-defined-subagent-name-that-needs-ellipsis",
                    "kind": "raw",
                    "description": "Project guidance.",
                },
            ]
            app = make_app(
                workspace,
                config=config,
                model_name=f"[{profile}] openai:google/gemma-4-12b",
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
                main_selector = panel.query_one("#settings-model-main", Select)
                assignments_section = panel.query_one(".settings-section.model-assignments")
                subagents_section = panel.query_one(".settings-section.subagents")
                header = panel.query_one(".settings-subagent-header")
                name_header = header.query_one(".settings-column-label.name", Static)
                enable_header = header.query_one(".settings-column-label.enabled")
                model_header = header.query_one(".settings-column-label.model")
                rows = list(panel.query(".settings-subagent-row"))

                self.assertEqual(context_section.styles.margin.top, 2)
                self.assertEqual(assignments_section.styles.margin.top, 1)
                self.assertEqual(header.styles.margin.top, 0)
                self.assertEqual(context_row.region.height, 3)
                self.assertEqual(context_input.region.y, context_row.region.y)
                self.assertEqual(context_input.region.height, context_row.region.height)
                self.assertEqual(context_input.region.x, main_selector.region.x)
                self.assertEqual(renderable_plain(name_header), "")
                self.assertEqual(header.region.y, subagents_section.region.bottom)
                self.assertEqual(len(rows), 2)

                for row in rows:
                    name = row.query_one(".settings-label", Static)
                    toggle = row.query_one(".settings-toggle")
                    selector = row.query_one(".settings-subagent-model-select")
                    selector_label = selector.query_one("SelectCurrent").query_one("#label", Static)
                    self.assertEqual(toggle.region.x, main_selector.region.x)
                    self.assertEqual(toggle.region.x, enable_header.region.x)
                    self.assertEqual(toggle.region.x - name.region.right, 2)
                    self.assertEqual(toggle.region.y, selector.region.y + 1)
                    self.assertEqual(row.region.height, 3)
                    self.assertGreaterEqual(selector.region.width, 40)
                    self.assertGreater(selector.region.width, name.region.width)
                    self.assertEqual(selector_label.region.height, 1)
                    self.assertEqual(
                        selector.region.x * 2 + selector.region.width,
                        model_header.region.x * 2 + model_header.region.width,
                    )

                long_row = rows[1]
                long_name = long_row.query_one(".settings-label", Static)
                long_model = long_row.query_one("SelectCurrent").query_one("#label", Static)
                self.assertEqual([line for line in rendered_lines(long_name) if line.strip()][0][-1], "…")
                self.assertEqual(len(rendered_lines(long_model)), 1)
                self.assertEqual(rendered_lines(long_model)[0].rstrip()[-1], "…")

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

    async def test_mounted_inherited_labels_repaint_immediately_after_main_changes(self) -> None:
        """Null assignments should repaint without closing and reopening Settings."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            settings = set_model_assignment(load_settings(workspace), "summarization", "old")
            settings = set_subagent_model_assignment(settings, "reviewer", "old")
            self.assertTrue(save_settings(workspace, settings))
            registry = ModelRegistry(
                {
                    "old": ModelProfile("old", {"provider": "openai", "model": "old-model"}),
                    "new": ModelProfile("new", {"provider": "openai", "model": "new-model"}),
                }
            )
            subagents = [
                {"name": "general-purpose", "kind": "raw", "description": "General work."},
                {"name": "reviewer", "kind": "raw", "description": "Review work."},
            ]
            app = make_app(
                workspace,
                config={"settings": settings, "model_registry": registry},
                model_name="unset",
                resource_metadata={"subagents": subagents},
            )

            async with app.run_test(size=(110, 42)) as pilot:
                await pilot.pause()
                app._apply_settings = AsyncMock(  # type: ignore[method-assign]
                    return_value=(True, "settings saved; agents rebuilt; no reload required")
                )
                app._handle_settings_command("models")
                await wait_until(lambda: len(app.query(SettingsPanel)) == 1)
                panel = app.query_one(SettingsPanel)
                await wait_until(lambda: panel._model_controls_ready)

                main = panel.query_one("#settings-model-main", Select)
                rubric = panel.query_one("#settings-model-rubric", Select)
                summarization = panel.query_one("#settings-model-summarization", Select)
                general = panel.query_one("#settings-subagent-model-general-purpose", Select)
                reviewer = panel.query_one("#settings-subagent-model-reviewer", Select)
                self.assertEqual(
                    renderable_plain(rubric.query_one("SelectCurrent").query_one("#label")),
                    "unset (default)",
                )
                self.assertEqual(
                    renderable_plain(general.query_one("SelectCurrent").query_one("#label")),
                    "unset (default)",
                )

                main.value = "new"
                await wait_until(lambda: model_assignment(panel.settings, "main") == "new")
                await wait_until(
                    lambda: renderable_plain(
                        rubric.query_one("SelectCurrent").query_one("#label")
                    )
                    == "new (default)"
                )

                self.assertEqual(
                    renderable_plain(rubric.query_one("SelectCurrent").query_one("#label")),
                    "new (default)",
                )
                self.assertEqual(
                    renderable_plain(general.query_one("SelectCurrent").query_one("#label")),
                    "new (default)",
                )
                self.assertEqual(
                    renderable_plain(summarization.query_one("SelectCurrent").query_one("#label")),
                    "old",
                )
                self.assertEqual(
                    renderable_plain(reviewer.query_one("SelectCurrent").query_one("#label")),
                    "old",
                )

            self.assertEqual(app._apply_settings.await_count, 1)
            self.assertIsNone(model_assignment(settings, "main"))
            self.assertEqual(model_assignment(panel.settings, "main"), "new")
            self.assertEqual(model_assignment(panel.settings, "summarization"), "old")
            self.assertEqual(subagent_model_assignment(panel.settings, "reviewer"), "old")

    async def test_keyboard_main_selection_is_saved_once_without_reverting(self) -> None:
        """Repainting Main after a real selection must not persist its stale prior value."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            current = "lmstudio-gemma"
            target = "lmstudio-gemma-no-reasoning"
            settings = set_model_assignment(load_settings(workspace), "main", current)
            self.assertTrue(save_settings(workspace, settings))
            registry = ModelRegistry(
                {
                    current: ModelProfile(
                        current,
                        {"provider": "openai", "model": "google/gemma-4-12b"},
                    ),
                    target: ModelProfile(
                        target,
                        {
                            "provider": "openai",
                            "model": "google/gemma-4-12b",
                            "model_kwargs": {"reasoning_effort": "none"},
                        },
                    ),
                }
            )
            subagents = [
                {"name": "general-purpose", "kind": "raw", "description": "General work."},
            ]
            app = make_app(
                workspace,
                config={"settings": settings, "settings_valid": True, "model_registry": registry},
                model_name=f"[{current}] openai:google/gemma-4-12b",
                resource_metadata={"subagents": subagents},
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
                ),
            ):
                async with app.run_test(size=(110, 42)) as pilot:
                    await pilot.pause()
                    app._rebuild_agents = AsyncMock()  # type: ignore[method-assign]
                    app._apply_settings = AsyncMock(  # type: ignore[method-assign]
                        wraps=app._apply_settings
                    )
                    app._handle_settings_command("models")
                    await wait_until(lambda: len(app.query(SettingsPanel)) == 1)
                    panel = app.query_one(SettingsPanel)
                    await wait_until(lambda: panel._model_controls_ready)

                    main = panel.query_one("#settings-model-main", Select)
                    rubric = panel.query_one("#settings-model-rubric", Select)
                    summarization = panel.query_one("#settings-model-summarization", Select)
                    general = panel.query_one("#settings-subagent-model-general-purpose", Select)
                    main.focus()
                    await pilot.press("enter")
                    await wait_until(lambda: main.expanded)
                    await pilot.press("down", "enter")
                    await wait_until(lambda: app._apply_settings.await_count >= 1)
                    await pilot.pause(0.2)

                    self.assertEqual(app._apply_settings.await_count, 1)
                    self.assertEqual(main.value, target)
                    self.assertEqual(model_assignment(panel.settings, "main"), target)
                    self.assertEqual(model_assignment(app.config, "main"), target)
                    self.assertEqual(model_assignment(load_settings(workspace), "main"), target)
                    self.assertEqual(
                        str(app.query_one("#model-settings-button", Button).label),
                        f"model: [{target}] openai:google/gemma-4-12b",
                    )
                    for selector in (rubric, summarization, general):
                        self.assertEqual(
                            renderable_plain(selector.query_one("SelectCurrent").query_one("#label")),
                            f"{target} (default)",
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
                        self.assertEqual(message, "settings saved; agents rebuilt; no reload required")

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
            self.assertEqual(message, "settings saved; agents rebuilt; no reload required")
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
                selector.focus()
                await pilot.press("enter")
                await wait_until(lambda: selector.expanded)
                await pilot.press("down", "enter")
                await wait_until(lambda: app._apply_settings.await_count == 1)
                await pilot.pause(0.1)

                status = renderable_plain(panel.query_one("#settings-status"))

            self.assertEqual(app._apply_settings.await_count, 1)
            self.assertEqual(selector.value, "old")
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
