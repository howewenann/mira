"""Native tab navigation over stable, independently dockable workspace views.

Views share one layout parent so docking never unmounts live transcripts,
subscriptions, editors, or attachments. Only the native header tabs move.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.geometry import Offset, Region, Size, Spacing
from textual.layout import Layout, WidgetPlacement
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Tabs
from textual.widgets.option_list import Option

from ui.textual.widgets.workspace_controls import (
    ActionTab, DragDivider, ScrollableHeaderTabs, TabCommand, TabOverflowDropdown,
)
from ui.textual.widgets.workspace_surfaces import BTWSurface, FileSurface

Group = Literal["main", "right"]


@dataclass
class WorkspaceView:
    key: str
    title: str
    kind: str
    surface: Widget
    group: Group = "main"
    path: Path | None = None


class WorkspaceLayout(Layout):
    """Place two complete frames and their retained content in the same parent."""

    name = "workspace"

    def arrange(self, parent: Workspace, children: list[Widget], size: Size, greedy: bool = True):
        right = parent.dock_visible
        dock_width = max(36, min(parent.right_width, size.width - 29)) if right else 0
        main_width = max(0, size.width - dock_width - int(right))
        frames = {
            "main": Region(0, 0, main_width, size.height),
            "right": Region(main_width + 1, 0, dock_width, size.height),
        }
        placements = []
        for child in children:
            if child.id == "primary-pane":
                region = frames["main"]
            elif child.id == "right-pane":
                region = frames["right"]
            elif child.id == "right-divider":
                region = Region(main_width, 0, 1, size.height)
            else:
                view = next((v for v in parent.views.values() if v.surface is child), None)
                if view is None:
                    continue
                frame = frames[view.group]
                region = Region(frame.x + 1, 3, max(0, frame.width - 2), max(0, size.height - 4))
            placements.append(WidgetPlacement(region, Offset(), Spacing(), child))
        return placements


class WorkspacePane(Vertical):
    def __init__(self, group: Group):
        super().__init__(id="primary-pane" if group == "main" else "right-pane")
        self.group = group

    def compose(self) -> ComposeResult:
        group = self.group
        with Horizontal(classes="workspace-top-actions"):
            tabs = [ActionTab("Chat", group="main", key="chat")] if group == "main" else []
            yield ScrollableHeaderTabs(*tabs, id=f"{group}-header-tabs", classes="workspace-header-tabs")
            with Horizontal(classes="workspace-header-scroll"):
                yield Button("‹", id=f"{group}-scroll-left", tooltip="Scroll tabs left")
                yield Button("›", id=f"{group}-scroll-right", tooltip="Scroll tabs right")
                yield Button("⋯", id=f"{group}-overflow", tooltip="All tabs")


class Workspace(Widget):
    """Own view placement and tab transitions; never execute agent work."""

    class ViewsChanged(Message):
        pass

    def __init__(self, chat: Widget, inspector: Widget):
        super().__init__(id="editor-workspace")
        self._workspace_layout = WorkspaceLayout()
        self.views = {"chat": WorkspaceView("chat", "Chat", "chat", chat)}
        self.inspector = inspector
        self.active: dict[Group, str] = {"main": "chat", "right": ""}
        self.right_width = 42
        self.dock_visible = False
        self._folded: list[str] = []
        self._mutating = False
        self._lock = asyncio.Lock()
        self._counter = 0

    @property
    def layout(self) -> Layout:
        return self._workspace_layout

    def compose(self) -> ComposeResult:
        yield WorkspacePane("main")
        yield DragDivider("right", id="right-divider")
        yield WorkspacePane("right")
        yield self.views["chat"].surface
        # The Inspector remains mounted even while closed, for production's
        # inspection lookup and event handlers. Closing detaches its subscription.
        yield self.inspector

    def on_mount(self) -> None:
        self.inspector.display = False
        self.sync_layout()

    def header(self, group: Group) -> ScrollableHeaderTabs:
        return self.query_one(f"#{group}-header-tabs", ScrollableHeaderTabs)

    def keys(self, group: Group) -> list[str]:
        return [key for key, view in self.views.items() if view.group == group]

    def activate(self, key: str) -> None:
        view = self.views.get(key)
        if view is None or not view.surface.is_mounted:
            return
        header = self.header(view.group)
        tab_id = f"head-{view.group}-{key}"
        if header.get_tab(tab_id) is None:
            return
        self.active[view.group] = key
        if header.active != tab_id:
            header.active = tab_id
        self.sync_layout()

    @on(Tabs.TabActivated)
    def tab_activated(self, event: Tabs.TabActivated) -> None:
        if event.tabs.id not in {"main-header-tabs", "right-header-tabs"}:
            return
        event.stop()
        if self._mutating or not isinstance(event.tab, ActionTab):
            return
        if event.tab.id != event.tabs.active:
            # Even a surviving tab can have an older activation queued behind
            # a newer selection. The native strip is the current authority.
            return
        view = self.views.get(event.tab.key)
        # A removed native Tab may still have queued an activation message.
        if view and view.group == event.tab.group and event.tab.is_mounted:
            self.activate(view.key)

    async def open_view(self, key: str, title: str, kind: str, surface: Widget, path: Path | None = None) -> None:
        async with self._lock:
            if path is not None:
                existing = next((v for v in self.views.values() if v.path == path), None)
                if existing:
                    self.activate(existing.key)
                    return
            if key in self.views:
                self.activate(key)
                return
            self._mutating = True
            try:
                if not surface.is_mounted:
                    await self.mount(surface)
                self.views[key] = WorkspaceView(key, title, kind, surface, path=path)
                await self.header("main").add_tab(ActionTab(title, group="main", key=key))
                self.activate(key)
            finally:
                self._mutating = False
        self.post_message(self.ViewsChanged())

    async def open_inspector(self) -> None:
        await self.open_view("inspector", "Inspector", "inspector", self.inspector)

    async def open_btw(self) -> None:
        await self.open_view("btw", "BTW", "btw", BTWSurface())

    async def open_file(self, path: Path) -> None:
        path = path.resolve()
        existing = next((v for v in self.views.values() if v.path == path), None)
        if existing:
            self.activate(existing.key)
            return
        self._counter += 1
        await self.open_view(f"file-{self._counter}", path.name, "file", FileSurface(path), path)

    async def move(self, key: str, dest: Group, *, responsive: bool = False) -> None:
        async with self._lock:
            view = self.views.get(key)
            if key == "chat" or view is None or view.group == dest:
                return
            if dest == "right" and self.app.size.width < 107 and not responsive:
                self.notify("Expand the terminal to dock a tab on the right.")
                return
            self._mutating = True
            try:
                source = view.group
                remaining = [k for k in self.keys(source) if k != key]
                # Switch to a surviving view BEFORE removing its selected tab.
                if self.active[source] == key:
                    self.active[source] = remaining[-1] if remaining else ""
                    if remaining:
                        self.activate(remaining[-1])
                view.group = dest
                await self.header(source).remove_tab(f"head-{source}-{key}")
                await self.header(dest).add_tab(ActionTab(view.title, group=dest, key=key))
                self.activate(key)
                if not responsive and key in self._folded:
                    self._folded.remove(key)
            finally:
                self._mutating = False
        self.sync_layout()

        self.post_message(self.ViewsChanged())

    async def close_view(self, key: str, *, stop_inspection: bool = True) -> None:
        async with self._lock:
            view = self.views.get(key)
            if key == "chat" or view is None:
                return
            self._mutating = True
            try:
                remaining = [k for k in self.keys(view.group) if k != key]
                if self.active[view.group] == key:
                    self.active[view.group] = remaining[-1] if remaining else ""
                    if remaining:
                        self.activate(remaining[-1])
                del self.views[key]
                if key == "inspector":
                    if stop_inspection:
                        self.inspector.stop_inspection()
                    self.inspector.display = False
                await self.header(view.group).remove_tab(f"head-{view.group}-{key}")
                if key != "inspector":
                    await view.surface.remove()
                if key in self._folded:
                    self._folded.remove(key)
            finally:
                self._mutating = False
        self.sync_layout()
        self.post_message(self.ViewsChanged())

    @on(ScrollableHeaderTabs.Scrolled)
    def tabs_scrolled(self, event: ScrollableHeaderTabs.Scrolled) -> None:
        event.stop()
        self.call_after_refresh(self.sync_tab_actions)

    @on(TabCommand)
    async def tab_command(self, event: TabCommand) -> None:
        event.stop()
        if event.command == "close":
            await self.close_view(event.key)
        else:
            await self.move(event.key, "right" if event.group == "main" else "main")

    @on(Button.Pressed, ".workspace-header-scroll Button")
    def header_button(self, event: Button.Pressed) -> None:
        event.stop()
        group: Group = "main" if event.button.id.startswith("main-") else "right"
        if event.button.id.endswith("overflow"):
            self.open_overflow(group, event.button)
        else:
            self.header(group).scroll_strip(-1 if event.button.id.endswith("left") else 1)
            self.call_after_refresh(self.sync_tab_actions)

    def open_overflow(self, group: Group, button: Button) -> None:
        options = [Option(self.views[k].title, id=k) for k in self.keys(group)]
        if group == "main":
            if "btw" not in self.views:
                options.append(Option("Open BTW", id="open-btw"))
        pane = self.query_one("#primary-pane" if group == "main" else "#right-pane")
        def picked(key: str | None) -> None:
            if key == "open-btw":
                self.run_worker(self.open_btw())
            elif key in self.views and self.views[key].group == group:
                self.activate(key)
        self.app.push_screen(TabOverflowDropdown(group, options, button.region.right,
                             button.region.bottom, pane.region, tuple(self.app.size),
                             active_key=self.active[group]), picked)

    def sync_tab_actions(self) -> None:
        for group in ("main", "right"):
            header = self.header(group)
            view = self.views.get(self.active[group])
            for kind in ("chat", "file", "inspector", "btw"):
                header.set_class(view is not None and view.kind == kind, "accent-" + kind)
            for tab in header.query(ActionTab):
                tab.sync_actions()
            scroll = header.query_one("#tabs-scroll")
            for side in ("left", "right"):
                self.query_one(f"#{group}-scroll-{side}", Button).disabled = (
                    scroll.max_scroll_x <= 0 or
                    (scroll.scroll_x <= 0 if side == "left" else scroll.scroll_x >= scroll.max_scroll_x)
                )

    def sync_layout(self) -> None:
        if not self.is_mounted:
            return
        self.dock_visible = bool(self.keys("right")) and self.app.size.width >= 107
        # Child layout invalidation also invalidates this parent's arrangement
        # cache in Textual 8.2.7. Width is otherwise state in our custom layout.
        self.query_one("#right-pane").styles.width = self.right_width
        self.query_one("#primary-pane").refresh(layout=True)
        for selector in ("#right-pane", "#right-divider"):
            self.query_one(selector).display = self.dock_visible
        for view in self.views.values():
            view.surface.display = self.active[view.group] == view.key and (view.group == "main" or self.dock_visible)
        self.refresh(layout=True)
        self.call_after_refresh(self.sync_tab_actions)

    async def responsive(self) -> None:
        if self.app.size.width < 107:
            for key in list(self.keys("right")):
                if key not in self._folded:
                    self._folded.append(key)
                await self.move(key, "main", responsive=True)
        else:
            for key in list(self._folded):
                if key in self.views:
                    await self.move(key, "right", responsive=True)
                if key in self._folded:
                    self._folded.remove(key)
        self.sync_layout()
