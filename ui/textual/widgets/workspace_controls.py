"""Workspace controls adapted from the approved v34 interaction reference."""
from __future__ import annotations
from typing import Any, Literal
from rich.cells import cell_len, set_cell_size
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.events import Click, MouseDown, MouseMove, MouseUp, MouseRelease, Resize
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Tab, Tabs, Static, OptionList
from textual.widgets.option_list import Option

class TabCommand(Message):
    """A click on a tab's reserved Dock/Close character cell."""

    def __init__(self, group: str, key: str, command: str):
        super().__init__()
        self.group = group
        self.key = key
        self.command = command

class ActionTab(Tab):
    """Native Textual tab with fixed-width, per-tab clickable actions.

    Reusing Tabs retains Textual 8.2.7's built-in scrolling / activation.
    The visual label is ALWAYS the same width. On hover/selection only the
    dock / close glyphs appear, replacing reserved spaces in the tab itself.
    No detached toolbar, no tab-width jitter and no Font dependency.
    """

    def __init__(self, title: str, *, group: str, key: str):
        self.title = title
        self.group = group
        self.key = key
        self.hovered = False
        self._shown = False
        super().__init__(Text(title + "     "), id=f"head-{group}-{key}")
        self._set_actions(False)
        self.tooltip = "Click tab to select; hover for Dock and Close actions"

    @property
    def dock_x(self) -> int:
        # CSS defines exactly one cell of horizontal padding.
        return 1 + cell_len(self.title) + 2

    @property
    def close_x(self) -> int:
        return self.dock_x + 2

    def _set_actions(self, visible: bool) -> None:
        if self.key == "chat":
            # Chat has no reserved docking/close cells: permanently main.
            label = self.title
            visible = False
        else:
            dock = ("›" if self.group == "main" else "‹") if visible else " "
            close = "×" if visible else " "
            label = f"{self.title}  {dock} {close}"
        if self.label_text != label:
            self.label = Text(label)
        self._shown = visible

    def sync_actions(self) -> None:
        self._set_actions(self.hovered or self.has_class("-active"))

    def on_enter(self) -> None:
        self.hovered = True
        self.sync_actions()

    def on_leave(self) -> None:
        self.hovered = False
        self.sync_actions()

    def _on_click(self, event: Click) -> None:
        # Textual's Tab click dispatch calls _on_click; intercept only the
        # reserved glyph cells, otherwise keep native tab activation.
        if event.button == 1 and self._shown:
            if event.x == self.dock_x and self.key != "chat":
                event.stop()
                self.post_message(TabCommand(self.group, self.key, "dock"))
                return
            if event.x == self.close_x and self.key != "chat":
                event.stop()
                self.post_message(TabCommand(self.group, self.key, "close"))
                return
        super()._on_click()

class ScrollableHeaderTabs(Tabs):
    """Native Tabs plus wheel-to-scroll behavior restricted to the tab bar."""

    class Scrolled(Message):
        pass

    def on_mount(self) -> None:
        scroller = self.query_one("#tabs-scroll")
        self.watch(scroller, "scroll_x", self._scroll_changed, init=False)
        self.watch(scroller, "virtual_size", self._scroll_changed, init=False)

    def _scroll_changed(self) -> None:
        self.post_message(self.Scrolled())

    def on_resize(self, _event: Resize) -> None:
        self._scroll_changed()

    def scroll_strip(self, direction: int) -> None:
        scroller = self.query_one("#tabs-scroll")
        target = min(max(0, scroller.scroll_x + direction * 12), scroller.max_scroll_x)
        scroller.scroll_to(x=target, animate=False, force=True)
        self.post_message(self.Scrolled())

    def on_mouse_scroll_down(self, event) -> None:
        self.scroll_strip(+1)
        event.stop()

    def on_mouse_scroll_up(self, event) -> None:
        self.scroll_strip(-1)
        event.stop()

class DragDivider(Static):
    """Invisible, one-cell draggable separator BETWEEN two solid panel borders.

    Idle: background matches workspace (no third line). Hover/drag: teal.
    Mouse capture retains resize when cursor travels over adjacent panels.
    """

    def __init__(self, purpose: Literal["sidebar", "right"], **kwargs):
        super().__init__("", classes="workspace-divider", **kwargs)
        self.purpose = purpose
        self.dragging = False
        self.tooltip = (
            "Drag this line to resize the history / files sidebar"
            if purpose == "sidebar"
            else "Drag this line to resize the right dock"
        )

    def render(self) -> str:
        # The CSS background is the only hover indicator. Render no glyphs.
        return ""

    def on_resize(self, _event: Resize) -> None:
        self.refresh()

    def on_mouse_down(self, event: MouseDown) -> None:
        if event.button != 1:
            return
        self.dragging = True
        self.add_class("dragging")
        self.refresh()
        # Textual's arbitrary text selection is undesirable *while resizing*.
        # Keep it enabled for normal transcript copying outside of a drag.
        self.app.begin_divider_drag()
        self.capture_mouse()
        event.stop()
        event.prevent_default()

    def on_mouse_move(self, event: MouseMove) -> None:
        if self.dragging:
            self.app.drag_width(self.purpose, event.screen_x)
            self.app.clear_selection()
            event.stop()
            event.prevent_default()

    def _end_drag(self) -> None:
        if not self.dragging:
            return
        self.dragging = False
        self.remove_class("dragging")
        self.refresh()
        self.app.end_divider_drag()

    def on_mouse_up(self, event: MouseUp) -> None:
        if self.dragging and event.button == 1:
            self._end_drag()
            self.release_mouse()
            event.stop()
            event.prevent_default()

    def on_mouse_release(self, _event: MouseRelease) -> None:
        # Terminal focus/capture loss must restore normal text selection.
        self._end_drag()

class TabOverflowDropdown(ModalScreen[str | None]):
    """Shared MAIN/RIGHT tab overflow menu, anchored to the triggering ellipsis.

    Transparent, dismissible screen rather than a focus-only in-pane overlay:
    clicks anywhere outside the menu reliably close it, just like the model
    alias chooser. Both tab groups have identical styles and behaviour.
    """

    CSS = """
    TabOverflowDropdown { align: left top; background: rgba(0,0,0,0.0); }
    TabOverflowDropdown #tab-overflow-options {
        position: absolute; width: 24; height: auto; max-height: 15;
        padding: 0; margin: 0; border: solid #5bb8b1;
        background: #101b1f; color: #e8edef;
        scrollbar-color: #5bb8b1; scrollbar-background: #122023;
    }
    TabOverflowDropdown #tab-overflow-options > .option-list--option {
        padding: 0 1;
    }
    TabOverflowDropdown #tab-overflow-options:focus > .option-list--option-highlighted {
        background: #1b3036; color: #eef7f8;
    }
    """
    AUTO_FOCUS = ""
    BINDINGS = [Binding("escape", "cancel", "Close tab menu", show=False)]

    def __init__(self, group: Literal["main", "right"], options: list[Option],
                 anchor_right: int, anchor_bottom: int,
                 pane_region: Any, screen_size: tuple[int, int], *,
                 active_key: str | None = None) -> None:
        super().__init__()
        self.group = group
        screen_w, screen_h = screen_size
        self.menu_width = min(24, max(8, pane_region.width - 2), screen_w - 2)
        # One visible row per option. The menu has two border cells, two
        # horizontal option-padding cells and may reserve a scrollbar cell.
        # Reserve one further cell rather than relying on OptionList wrapping
        # long filenames into a second row (which broke the menu geometry).
        self.label_columns = max(1, self.menu_width - 8)
        self.options = []
        for option in options:
            active = option.id == active_key and option.id not in {"open-inspector", "open-btw"}
            label = Text()
            label.append("● " if active else "  ", style="#5bb8b1")
            label.append(self._single_line_label(str(option.prompt), self.label_columns))
            self.options.append(Option(label, id=option.id, disabled=option.disabled))
        self.menu_height = min(max(3, len(options) + 2), 15,
                               max(3, pane_region.height - 2), screen_h - 2)
        self.menu_x = max(pane_region.x + 1,
                          min(anchor_right - self.menu_width,
                              pane_region.right - self.menu_width - 1))
        self.menu_x = max(0, min(self.menu_x, screen_w - self.menu_width))
        below = anchor_bottom
        if below + self.menu_height > pane_region.bottom - 1:
            below = max(pane_region.y + 1, anchor_bottom - self.menu_height - 1)
        self.menu_y = max(0, min(below, screen_h - self.menu_height))

    @staticmethod
    def _single_line_label(label: str, width: int) -> str:
        """Keep the OptionList row height at one cell for any tab title.

        Rich's cell helpers account for wide Unicode characters. The option
        ID is unchanged, so selecting a shortened label opens the real tab.
        """
        label = " ".join(label.splitlines())
        if cell_len(label) <= width:
            return label
        if width <= 1:
            return "…"
        return set_cell_size(label, width - 1).rstrip() + "…"

    def compose(self) -> ComposeResult:
        yield OptionList(*self.options, id="tab-overflow-options", compact=True)

    def on_mount(self) -> None:
        menu = self.query_one("#tab-overflow-options", OptionList)
        menu.styles.width = self.menu_width
        menu.styles.height = self.menu_height
        menu.styles.offset = (self.menu_x, self.menu_y)
        menu.highlighted = 0 if self.options else None
        self.set_focus(None)
        self.call_after_refresh(self.set_focus, None)

    @on(OptionList.OptionSelected, "#tab-overflow-options")
    def select(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.dismiss(str(event.option.id or ""))

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_key(self, event) -> None:
        options = self.query_one(OptionList)
        actions = {"up": options.action_cursor_up, "down": options.action_cursor_down,
                   "enter": options.action_select, "pageup": options.action_page_up,
                   "pagedown": options.action_page_down, "home": options.action_first,
                   "end": options.action_last}
        if event.key in actions:
            event.stop()
            event.prevent_default()
            actions[event.key]()

    def on_click(self, event: Click) -> None:
        # Same click-away dismissal pattern as MIRA v29's ModelAliasPopup.
        if event.widget is self:
            self.dismiss(None)

class ModelAliasPopup(ModalScreen[str | None]):
    """Compact model menu anchored UPWARD to the active composer.

    Unlike the stock SelectOverlay (which may appear beneath a bottom-pinned
    composer), the menu's geometry is clamped inside the owning Chat/BTW pane
    and strictly above the triggering model button.  This is NOT Settings.
    """

    CSS = """
    ModelAliasPopup { background: rgba(0, 0, 0, 0.0); }
    ModelAliasPopup #model-alias-options {
        position: absolute; width: 20; height: 5;
        border: solid #51696c; padding: 0; margin: 0;
        background: #152529; color: #e8edef;
        scrollbar-color: #5bb8b1; scrollbar-background: #122023;
    }
    ModelAliasPopup #model-alias-options > .option-list--option { padding: 0 1; }
    """
    AUTO_FOCUS = ""
    BINDINGS = [Binding("escape", "dismiss_aliases", "Close model menu", show=False)]

    def __init__(self, kind: Literal["main", "btw"], options: tuple[str, ...],
                 current: str, anchor: tuple[int, int],
                 bounds: tuple[int, int, int, int], screen_size: tuple[int, int]):
        super().__init__()
        self.kind = kind
        self.aliases = options
        self.current = current
        ax_right, ay = anchor
        pane_x, pane_y, pane_right, _pane_bottom = bounds
        screen_width, screen_height = screen_size
        room = max(10, min(screen_width - 2, pane_right - pane_x - 2))
        self.menu_width = min(max(18, max(map(len, options), default=10) + 5), room)
        self.menu_height = min(max(3, len(options) + 2), 12,
                               max(3, ay - pane_y - 1), max(3, screen_height - 3))
        self.menu_x = max(pane_x + 1,
                          min(ax_right - self.menu_width, pane_right - self.menu_width - 1))
        self.menu_x = max(0, min(self.menu_x, screen_width - self.menu_width))
        self.menu_y = max(pane_y + 1, ay - self.menu_height)
        self.menu_y = max(0, min(self.menu_y, screen_height - self.menu_height))

    def compose(self) -> ComposeResult:
        # No two-character current marker: it caused names to truncate in a
        # narrow dock. Highlight the current option instead.
        yield OptionList(*[
            Option(alias, id=f"alias-{i}")
            for i, alias in enumerate(self.aliases)
        ], id="model-alias-options")

    def on_mount(self) -> None:
        options = self.query_one("#model-alias-options", OptionList)
        options.styles.width = self.menu_width
        options.styles.height = self.menu_height
        options.styles.offset = (self.menu_x, self.menu_y)
        if self.current in self.aliases:
            options.highlighted = self.aliases.index(self.current)
        self.set_focus(None)
        self.call_after_refresh(self.set_focus, None)

    @on(OptionList.OptionSelected, "#model-alias-options")
    def alias_selected(self, event: OptionList.OptionSelected) -> None:
        index = event.option_index
        selected = self.aliases[index] if 0 <= index < len(self.aliases) else None
        self.dismiss(selected)
        event.stop()

    def action_dismiss_aliases(self) -> None:
        self.dismiss(None)

    def on_key(self, event) -> None:
        options = self.query_one(OptionList)
        actions = {"up": options.action_cursor_up, "down": options.action_cursor_down,
                   "enter": options.action_select, "pageup": options.action_page_up,
                   "pagedown": options.action_page_down, "home": options.action_first,
                   "end": options.action_last}
        if event.key in actions:
            event.stop()
            event.prevent_default()
            actions[event.key]()

    def on_click(self, event: Click) -> None:
        if event.widget is self:
            self.dismiss(None)
