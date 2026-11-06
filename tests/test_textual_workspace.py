"""Rendered production workspace regressions against the approved v34 layout."""

import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from textual.widgets import Button, OptionList, Static, TextArea

from tests.test_textual_app import (
    make_app, compositor_plain, wait_until, renderable_plain, move_captured_mouse,
    start_artifact_review, final_plan_interrupt,
)
from config.llm import ModelProfile, ModelRegistry
from config.metadata import ModelMetadata
from config.settings import set_model_assignment
from ui.textual.widgets import ChatLog, Inspector, PromptBox
from ui.textual.widgets.autocomplete_input import AutocompleteInput
from ui.textual.widgets.session_history import SessionActionMenuScreen, SessionItem
from ui.textual.widgets.workspace import Workspace
from ui.textual.widgets.workspace_controls import ActionTab, TabOverflowDropdown, ModelAliasPopup


class WorkspaceTests(unittest.IsolatedAsyncioTestCase):
    async def test_initial_layout(self):
        app = make_app()
        async with app.run_test(size=(160, 48)) as pilot:
            await pilot.pause()
            workspace = app.query_one(Workspace)
            self.assertEqual(workspace.active["main"], "chat")
            self.assertFalse(app.query_one("#session-sidebar").display)
            self.assertFalse(workspace.dock_visible)
            self.assertEqual(app.query_one("#navigation-rail").region.width, 9)
            self.assertEqual(app.query_one("#status-row").region.width, 160)
            self.assertEqual(app.query_one("#telemetry-row").region.y, 47)
            self.assertEqual(app.query_one("#primary-pane").region.y, 1)
            self.assertEqual(app.query_one("#primary-pane").region.bottom, 47)
            self.assertEqual(app.query_one(PromptBox).region.height, 3)
            self.assertIn("main: test", compositor_plain(app))
            self.assertFalse(list(app.query("#model-settings-button")))

    async def test_dock_final_file_and_repeated_moves_preserve_widgets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.py"
            path.write_text("print('retained')\n", encoding="utf-8")
            app = make_app(workspace=Path(directory))
            async with app.run_test(size=(160, 48)) as pilot:
                workspace = app.query_one(Workspace)
                await workspace.open_file(path)
                await pilot.pause()
                view = next(v for v in workspace.views.values() if v.path == path)
                editor = view.surface.query_one(TextArea)
                for _ in range(4):
                    await workspace.move(view.key, "right")
                    await pilot.pause()
                    self.assertEqual(workspace.active["main"], "chat")
                    self.assertTrue(workspace.dock_visible)
                    self.assertEqual(app.query_one("#right-pane").region.width, 42)
                    await workspace.move(view.key, "main")
                    await pilot.pause()
                    self.assertIs(editor, view.surface.query_one(TextArea))
                await workspace.close_view(view.key)
                await pilot.pause()
                self.assertEqual(list(workspace.views), ["chat"])

    async def test_sidebar_switch_resize_and_full_panel_alignment(self):
        app = make_app()
        async with app.run_test(size=(200, 48)) as pilot:
            await pilot.click("#rail-chat")
            sidebar = app.query_one("#session-sidebar")
            self.assertTrue(sidebar.display)
            self.assertEqual(sidebar.region.width, 42)
            app.drag_width("sidebar", sidebar.region.x + 100)
            await pilot.pause()
            self.assertEqual(sidebar.region.width, 66)
            app.drag_width("sidebar", sidebar.region.x)
            await pilot.pause()
            self.assertEqual(sidebar.region.width, 42)
            await pilot.click("#rail-files")
            self.assertEqual(app.sidebar_kind, "files")
            self.assertTrue(sidebar.display)
            self.assertTrue(app.query_one("#files-panel").display)
            await wait_until(lambda: not app.query_one("#rail-files").has_class("-active"))
            await pilot.click("#rail-files")
            await wait_until(lambda: not sidebar.display)
            self.assertFalse(sidebar.display)
            await pilot.click("#rail-chat")
            workspace = app.query_one(Workspace)
            await workspace.open_btw()
            await workspace.move("btw", "right")
            await pilot.pause()
            for selector in ("#navigation-rail", "#session-sidebar", "#primary-pane", "#right-pane"):
                region = app.query_one(selector).region
                self.assertEqual((region.y, region.bottom), (1, 47))
            app.drag_width("right", workspace.region.right - 200)
            await pilot.pause()
            self.assertEqual(app.query_one("#right-pane").region.width, 76)
            app.drag_width("right", workspace.region.right)
            await pilot.pause()
            self.assertEqual(app.query_one("#right-pane").region.width, 28)
            self.assertEqual(sidebar.region.width, 42)

    async def test_resize_handle_is_invisible_except_hover_and_drag(self):
        app = make_app()
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.click("#rail-chat")
            handle = app.query_one("#side-divider")
            self.assertEqual(handle.region.width, 1)
            self.assertEqual(str(handle.render()), "")
            self.assertEqual(handle.styles.background.hex, "#0C0F10")
            await pilot.hover("#side-divider")
            self.assertEqual(handle.styles.background.hex, "#5BB8B1")
            await pilot.mouse_down("#side-divider")
            self.assertFalse(app.ALLOW_SELECT)
            await move_captured_mouse(pilot, handle.region.x + 8, handle.region.y + 2)
            await pilot.mouse_up(offset=(handle.region.x + 8, handle.region.y + 2))
            await pilot.hover("#status-row")
            self.assertFalse(handle.has_class("dragging"))
            self.assertTrue(app.ALLOW_SELECT)
            self.assertEqual(handle.styles.background.hex, "#0C0F10")

    async def test_narrow_resize_folds_and_restores_dock_without_losing_content(self):
        app = make_app()
        async with app.run_test(size=(160, 48)) as pilot:
            workspace = app.query_one(Workspace)
            await workspace.open_btw()
            surface = workspace.views["btw"].surface
            await workspace.move("btw", "right")
            for width, height in ((110, 32), (80, 24), (60, 20), (200, 48)):
                await pilot.resize_terminal(width, height)
                await pilot.pause()
                self.assertIs(workspace.views["btw"].surface, surface)
                self.assertEqual(workspace.views["btw"].group, "right" if width >= 107 else "main")
                for view in workspace.views.values():
                    if view.surface.display:
                        self.assertGreater(view.surface.region.width, 0)
                        self.assertLessEqual(view.surface.region.right, width)
                        self.assertLessEqual(view.surface.region.bottom, height - 1)

    async def test_btw_placeholder_never_submits_or_claims_a_model(self):
        app = make_app()
        async with app.run_test(size=(160, 40)) as pilot:
            workspace = app.query_one(Workspace)
            with patch.object(app, "submit_prompt", AsyncMock()) as submit:
                await workspace.open_btw()
                await pilot.pause()
                self.assertTrue(app.query_one("#btw-prompt").disabled)
                self.assertTrue(app.query_one("#attach-btw").disabled)
                self.assertTrue(app.query_one("#model-trigger-btw").disabled)
                await pilot.press("enter")
                submit.assert_not_awaited()
                self.assertIn("btw: unset", compositor_plain(app))
                await workspace.move("btw", "right")
                await workspace.close_view("btw")
                await pilot.pause()
                self.assertFalse(app.query_one("#effective-model-btw").display)
                self.assertFalse(app.query_one("#effective-model-separator").display)

    async def test_overflow_dropdowns_truncate_one_row_keep_identity_and_dismiss(self):
        app = make_app()
        async with app.run_test(size=(160, 40)) as pilot:
            workspace = app.query_one(Workspace)
            for index in range(18):
                await workspace.open_view(f"long-{index}", f"{index}-" + "long_filename_" * 8, "file", Static(str(index)))
            await pilot.pause()
            await pilot.click("#main-overflow")
            await pilot.pause()
            self.assertIsInstance(app.screen, TabOverflowDropdown)
            self.assertIsNone(app.screen.focused)
            menu = app.screen.query_one(OptionList)
            self.assertEqual(len(menu._lines), menu.option_count)
            self.assertTrue(str(menu.get_option_at_index(1).prompt).endswith("…"))
            menu.highlighted = 1
            await pilot.press("enter")
            await pilot.pause()
            self.assertEqual(workspace.active["main"], "long-0")
            await workspace.move("long-0", "right")
            await pilot.pause()
            await pilot.click("#right-overflow")
            await pilot.pause()
            self.assertIsInstance(app.screen, TabOverflowDropdown)
            self.assertIsNone(app.screen.focused)
            await pilot.click(offset=(0, 0))
            self.assertEqual(len(app.screen_stack), 1)
            await pilot.click("#right-overflow")
            await pilot.press("escape")
            self.assertEqual(len(app.screen_stack), 1)

    async def test_stale_tab_events_and_concurrent_commands_are_safe(self):
        from textual.widgets import Tabs
        app = make_app()
        async with app.run_test(size=(160, 40)) as pilot:
            workspace = app.query_one(Workspace)
            await workspace.open_btw()
            stale = workspace.header("main").get_tab("head-main-btw")
            chat_tab = workspace.header("main").get_tab("head-main-chat")
            workspace.tab_activated(Tabs.TabActivated(workspace.header("main"), chat_tab))
            self.assertEqual(workspace.active["main"], "btw")
            await asyncio.gather(workspace.move("btw", "right"), workspace.move("btw", "main"), workspace.move("btw", "right"))
            workspace.tab_activated(Tabs.TabActivated(workspace.header("main"), stale))
            await workspace.close_view("btw")
            workspace.tab_activated(Tabs.TabActivated(workspace.header("main"), stale))
            await pilot.pause()
            self.assertEqual(workspace.active["main"], "chat")
            self.assertEqual(workspace.keys("right"), [])
            await workspace.move("chat", "right")
            await workspace.close_view("chat")
            self.assertIn("chat", workspace.views)
            self.assertEqual(workspace.query_one("#head-main-chat", ActionTab).label_text, "Chat")

    async def test_inspector_retains_live_subscription_and_has_no_internal_header(self):
        app = make_app()
        async with app.run_test(size=(160, 40)) as pilot:
            workspace = app.query_one(Workspace)
            inspector = app.query_one(Inspector)
            inspection_id = app.live_inspections.allocate_id("workspace-live")
            app.live_inspections.start(inspection_id, "researcher", "Live workspace inspection")
            self.assertTrue(inspector.open(inspection_id))
            await workspace.open_inspector()
            await pilot.pause()
            self.assertFalse(list(inspector.query("#inspector-header")))
            self.assertFalse(list(inspector.query("#inspector-close")))
            log = inspector.query_one(ChatLog)
            log.system_message("retained inspection")
            await workspace.move("inspector", "right")
            app.live_inspections.append_delta(inspection_id, "assistant", "Update while docked")
            await pilot.pause()
            self.assertIn("Update while docked", "\n".join(renderable_plain(child) for child in log.children))
            await workspace.move("inspector", "main")
            self.assertIs(inspector.query_one(ChatLog), log)
            inspector.focus()
            await pilot.press("escape")
            await pilot.pause()
            self.assertNotIn("inspector", workspace.views)
            self.assertTrue(app.query_one(PromptBox).has_focus)

    async def test_model_popup_aliases_above_prompt_and_effective_state(self):
        app = make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.click("#model-trigger-main")
            self.assertIsInstance(app.screen, ModelAliasPopup)
            self.assertIsNone(app.screen.focused)
            menu = app.screen.query_one(OptionList)
            self.assertEqual(str(menu.get_option_at_index(0).prompt), "test")
            self.assertLessEqual(menu.region.bottom, app.query_one("#model-trigger-main").region.y)
            self.assertLessEqual(menu.region.right, app.size.width)
            await pilot.press("escape")
            app.config["settings"] = set_model_assignment(app.config, "main", "new-selection")
            app._set_status(state="ready")
            self.assertEqual(app.effective_model_alias, "test")
            self.assertIn("main: test", compositor_plain(app))
            self.assertIn("new-selection", app.query_one("#actions-main").alias)

    async def test_successful_and_failed_model_activation_use_production_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            app = make_app(workspace=Path(directory))
            registry = app.prebuilt["config"]["model_registry"]
            registry.profiles["next-alias"] = ModelProfile("next-alias", {"provider": "openai", "model": "private-provider-id"})
            async with app.run_test(size=(120, 32)) as pilot:
                with (
                    patch("agent.llm.get_llm", return_value=object()),
                    patch("config.metadata.infer_model_metadata", AsyncMock(return_value=ModelMetadata(8192, "test"))),
                    patch.object(app, "_build_agent_pair", side_effect=RuntimeError("activation failed")),
                ):
                    await app._activate_model_alias("next-alias")
                self.assertEqual(app.effective_model_alias, "test")
                with (
                    patch("agent.llm.get_llm", return_value=object()),
                    patch("config.metadata.infer_model_metadata", AsyncMock(return_value=ModelMetadata(8192, "test"))),
                    patch.object(app, "_build_agent_pair", return_value=("next-agent", "next-plan")),
                ):
                    await app._activate_model_alias("next-alias")
                self.assertEqual(app.effective_model_alias, "next-alias")
                await pilot.pause()
                self.assertIn("main: next-alias", compositor_plain(app))
                self.assertNotIn("private-provider-id", renderable_plain(app.query_one("#effective-model-main")))
                await app._apply_settings(set_model_assignment(app.config, "main", None))
                await pilot.pause()
                self.assertIsNone(app.agent)
                self.assertEqual(app.effective_model_alias, "")
                self.assertIn("main: unset", compositor_plain(app))

    async def test_model_popup_clamps_to_minimum_dock_bounds(self):
        app = make_app()
        async with app.run_test(size=(160, 40)) as pilot:
            workspace = app.query_one(Workspace)
            await workspace.open_btw()
            await workspace.move("btw", "right")
            workspace.right_width = 28
            workspace.sync_layout()
            await pilot.pause()
            trigger = app.query_one("#model-trigger-btw")
            pane = app.query_one("#right-pane").region
            app.push_screen(ModelAliasPopup("btw", tuple(f"long-model-alias-{i}" for i in range(25)), "",
                            (trigger.region.right, trigger.region.y), (pane.x, pane.y, pane.right, pane.bottom), tuple(app.size)))
            await pilot.pause()
            menu = app.screen.query_one(OptionList)
            self.assertGreaterEqual(menu.region.x, pane.x)
            self.assertLessEqual(menu.region.right, pane.right)
            self.assertLessEqual(menu.region.bottom, trigger.region.y)
            self.assertLessEqual(menu.region.height, 12)
            self.assertIsNone(app.screen.focused)
            await pilot.click(offset=(0, 0))
            self.assertEqual(len(app.screen_stack), 1)

    async def test_settings_rail_always_opens_general(self):
        app = make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await app._run_settings_command("models")
            await pilot.pause()
            self.assertTrue(app._settings_panel.query_one("#settings-tab-models").has_class("active"))
            app._settings_panel._close()
            await pilot.pause()
            await pilot.click("#rail-settings")
            await wait_until(lambda: app._settings_panel is not None)
            await pilot.pause()
            self.assertTrue(app._settings_panel.query_one("#settings-tab-general").has_class("active"))
            self.assertIsNone(app.screen.focused)

    async def test_history_menu_keeps_right_of_row_anchor_and_click_away(self):
        app = make_app()
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.click("#rail-chat")
            await app.query_one("#sessions").append(SessionItem({"id": "menu-row", "custom_title": "Saved chat"}))
            await pilot.pause()
            row = app.query_one(SessionItem)
            await pilot.click(row.query_one(".session-menu-action"))
            await pilot.pause()
            self.assertIsInstance(app.screen, SessionActionMenuScreen)
            self.assertIsNone(app.screen.focused)
            menu = app.screen.query_one("#session-action-popup")
            await wait_until(lambda: menu.region.x == row.region.right)
            self.assertEqual((menu.region.x, menu.region.y), (row.region.right, row.region.y))
            await pilot.click(offset=(0, 0))
            self.assertEqual(len(app.screen_stack), 1)

    async def test_tab_glyph_actions_and_hover_reserve_the_same_width(self):
        app = make_app()
        async with app.run_test(size=(160, 40)) as pilot:
            workspace = app.query_one(Workspace)
            await workspace.open_view("file", "example.py", "file", Static("preview"))
            await pilot.pause()
            tab = app.query_one("#head-main-file", ActionTab)
            width = tab.region.width
            workspace.activate("chat")
            await pilot.hover("#status-row")
            await pilot.pause()
            self.assertEqual(tab.region.width, width)
            self.assertNotIn("×", tab.label_text)
            await pilot.hover(tab)
            self.assertEqual(tab.region.width, width)
            await pilot.click(tab, offset=(tab.dock_x, 0))
            await wait_until(lambda: workspace.views["file"].group == "right")
            await pilot.pause()
            dock_tab = app.query_one("#head-right-file", ActionTab)
            await pilot.click(dock_tab, offset=(dock_tab.close_x, 0))
            await wait_until(lambda: "file" not in workspace.views)
            self.assertEqual(workspace.active["main"], "chat")

    async def test_tab_arrows_only_light_on_hover_and_stay_muted_at_edges(self):
        app = make_app()
        async with app.run_test(size=(120, 35)) as pilot:
            workspace = app.query_one(Workspace)
            for index in range(10):
                await workspace.open_view(f"view-{index}", f"Example preview {index}", "file", Static("preview"))
            await pilot.pause()
            left = app.query_one("#main-scroll-left", Button)
            right = app.query_one("#main-scroll-right", Button)
            header = workspace.header("main")
            header.scroll_strip(10000)
            await wait_until(lambda: right.disabled)
            self.assertFalse(left.disabled)
            self.assertTrue(right.disabled)
            left.focus()
            await pilot.hover("#status-row")
            self.assertEqual(left.styles.background.hex, "#0C0F10")
            await pilot.hover(left)
            self.assertEqual(left.styles.background.hex, "#21424B")
            await pilot.hover(right)
            self.assertEqual(right.styles.background.hex, "#0C0F10")
            header.scroll_strip(-10000)
            await pilot.hover("#status-row")
            await pilot.pause()
            self.assertTrue(left.disabled)
            self.assertFalse(right.disabled)
            self.assertEqual(right.styles.background.hex, "#0C0F10")

    async def test_file_tree_expands_real_directories_and_opens_preview(self):
        from textual.widgets import DirectoryTree
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "source"
            folder.mkdir()
            target = folder / "example.py"
            target.write_text("print('real file')", encoding="utf-8")
            app = make_app(workspace=root)
            async with app.run_test(size=(160, 40)) as pilot:
                await pilot.click("#rail-files")
                tree = app.query_one(DirectoryTree)
                tree.root.expand()
                await wait_until(lambda: bool(tree.root.children))
                node = next(n for n in tree.root.children if n.data.path == folder)
                node.expand()
                await wait_until(lambda: bool(node.children))
                tree.move_cursor(node.children[0])
                tree.focus()
                await pilot.press("enter")
                await wait_until(lambda: any(v.path == target for v in app.query_one(Workspace).views.values()))
                view = next(v for v in app.query_one(Workspace).views.values() if v.path == target)
                self.assertIn("real file", view.surface.query_one(TextArea).text)

    async def test_telemetry_keeps_full_alias_when_wide_and_metrics_when_narrow(self):
        from ui.textual.widgets.status_bar import TelemetryBar
        app = make_app()
        async with app.run_test(size=(200, 40)) as pilot:
            telemetry = app.query_one(TelemetryBar)
            alias = "local-model-alias-with-a-long-readable-name"
            telemetry.set_state(model_name=alias, btw_name="", dashboard={}, turns=3)
            await pilot.pause()
            main = telemetry.query_one("#effective-model-main")
            self.assertGreaterEqual(main.region.width, len(alias) + 6)
            self.assertIn(alias, compositor_plain(app))
            await pilot.resize_terminal(80, 24)
            await pilot.pause()
            values = telemetry.query_one("#telemetry-values")
            self.assertLessEqual(values.region.right, app.size.width)
            self.assertIn("Turns", compositor_plain(app))
            self.assertLessEqual(main.region.right, values.region.x)
            self.assertFalse(list(telemetry.query(Button)))

    async def test_chat_attachments_and_completion_survive_switching_views(self):
        app = make_app()
        async with app.run_test(size=(160, 40)) as pilot:
            workspace = app.query_one(Workspace)
            prompt = app.query_one(PromptBox)
            prompt.value = "Review "
            prompt.insert_file_references(["@/uploads/report.txt"])
            prompt.styles.height = 8
            await pilot.pause()
            draft = prompt.value
            await workspace.open_btw()
            await pilot.pause()
            await workspace.move("btw", "right")
            workspace.activate("chat")
            await pilot.pause()
            self.assertIs(app.query_one(PromptBox), prompt)
            self.assertEqual(prompt.value, draft)
            self.assertEqual(prompt.region.height, 8)
            prompt.value = "/hel"
            prompt.focus()
            await pilot.pause()
            self.assertTrue(app.query_one(AutocompleteInput).active)
            await pilot.press("enter")
            self.assertTrue(prompt.value.startswith("/help"))

    async def test_foreground_prompts_and_formal_review_reveal_chat(self):
        from ui.textual.widgets import PromptPanel
        app = make_app()
        async with app.run_test(size=(160, 40)) as pilot:
            workspace = app.query_one(Workspace)
            await workspace.open_view("preview", "example.py", "file", Static("preview"))
            choice = asyncio.create_task(app._prompt_choice("Approval", "Continue?", [("a", "Allow")]))
            await wait_until(lambda: app.query_one(PromptPanel).active)
            self.assertEqual(workspace.active["main"], "chat")
            await pilot.press("a")
            self.assertEqual(await asyncio.wait_for(choice, 2), "a")
            workspace.activate("preview")
            review = start_artifact_review(app, "plan", final_plan_interrupt())
            await wait_until(lambda: app._pending_artifact_review is not None)
            self.assertEqual(workspace.active["main"], "chat")
            artifact = app._pending_artifact_review.artifact
            await app._handle_plan_action("close", artifact["id"])
            decision = await asyncio.wait_for(review, 2)
            self.assertEqual(decision["action"], "close")
            self.assertEqual(app.session["current_plan"]["id"], artifact["id"])

    async def test_queued_inspector_close_does_not_stop_a_new_inspection(self):
        app = make_app()
        async with app.run_test(size=(160, 40)) as pilot:
            workspace = app.query_one(Workspace)
            inspector = app.query_one(Inspector)
            for key in ("old", "new"):
                app.live_inspections.start(key, "researcher", key)
            inspector.open("old")
            await workspace.open_inspector()
            app._restore_chat_viewport()
            inspector.open("new")
            app._show_inspector(inspector)
            await wait_until(lambda: "inspector" in workspace.views and workspace.active["main"] == "inspector")
            await pilot.pause()
            self.assertEqual(inspector.inspection_id, "new")
            app.live_inspections.append_delta("new", "assistant", "New inspection remains live")
            await pilot.pause()
            log = inspector.query_one(ChatLog)
            self.assertIn("New inspection remains live", "\n".join(renderable_plain(child) for child in log.children))


if __name__ == "__main__":
    unittest.main()
