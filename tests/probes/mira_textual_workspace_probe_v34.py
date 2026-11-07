"""MIRA UI FIDELITY PROBE v34 — compact composers, fixed Chat and anchored dropdown menus.

Target: MIRA v2.12.0 / Textual 8.2.7.  This is NOT a replacement for MIRA.
Save this file as tests/probes/mira_textual_workspace_probe_v34.py
in the MIRA repository, then run from the repository root:

    conda activate mira
    cd D:\\Projects\\mira
    python tests/probes/mira_textual_workspace_probe_v34.py --workspace .
    python tests/probes/mira_textual_workspace_probe_v34.py --workspace . --self-test
    python tests/probes/mira_textual_workspace_probe_v34.py --workspace . --screenshots .\\ui-probe-shots

The probe imports MIRA's REAL ChatLog, AutocompleteInput/PromptBox,
SessionHistory, Inspector, StatusBar and TelemetryBar and loads its REAL
ui/textual/styles/mira.tcss. Probe-only CSS supplies four separate solid-bordered panels, invisible
hover-only resize handles and per-type tab accents. It preserves MIRA's transcript widgets. The agent isn't started.

Only reading file previews and creating requested screenshots can touch disk.
Chats, prompts, sessions, inspector events and BTW answers are simulated.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from rich.cells import cell_len, set_cell_size

# The probe is intentionally runnable from MIRA's project root, independently
# of any package installation or CLI entrypoint. Never import a second checkout.
ROOT = Path.cwd().resolve()
if (ROOT / "ui" / "textual" / "styles" / "mira.tcss").exists():
    sys.path.insert(0, str(ROOT))

try:
    from textual import on
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.css.query import NoMatches
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.events import Key, MouseDown, MouseMove, MouseUp, MouseRelease, Resize, Click
    from textual.screen import ModalScreen
    from textual.message import Message
    from textual.widgets import (
        Button, ContentSwitcher, DirectoryTree, ListView, Markdown, OptionList,
        Static, Tabs, Tab, TabbedContent, TabPane, TextArea, Select,
    )
    from textual.widget import Widget
    from textual.widgets.option_list import Option
    from ui.textual.widgets import (
        AutocompleteInput, ChatLog, Inspector, PromptBox,
        SessionHistory, StatusBar, TelemetryBar,
    )
    from ui.textual.widgets.session_history import SessionItem, SessionActionMenuScreen
    from ui.textual.widgets.autocomplete_input import PromptResizeHandle
    from core.execution.inspection.live import LiveInspectionStore
    from ui.textual.widgets.status_bar import telemetry_values
    from config.llm import load_model_registry
    from config.settings import load_settings, model_assignment, MAIN_MODEL
except ImportError as exc:
    raise SystemExit(
        "Run this probe from the MIRA repository root using the `mira` Conda "
        f"environment. Could not import MIRA/Textual: {exc}"
    ) from exc

MIRA_STYLESHEET = ROOT / "ui" / "textual" / "styles" / "mira.tcss"
if not MIRA_STYLESHEET.is_file():
    raise SystemExit(f"MIRA's actual stylesheet not found: {MIRA_STYLESHEET}")


@dataclass(frozen=True)
class View:
    key: str
    title: str
    kind: Literal["chat", "file", "inspector", "btw"]
    path: Path | None = None


SAMPLE_HISTORY = [
    ("demo-1", "Implement MCP Skills and simplify the resource panel", True),
    ("demo-2", "Investigate workflow inspector focus and frozen timers", False),
    ("demo-3", "Explain effective conversation handoff after compaction", False),
    ("demo-4", "Refactor Access settings and reset to defaults", False),
]

SAMPLE_CHAT = [
    ("you", "Explain how conversation handoff works between Act, Plan, and Goal."),
    ("mira", "When modes change, MIRA replaces the destination's messages with the source's effective conversation. This preserves the post-compaction context without duplicating older messages."),
    ("you", "Does the destination retain its own checkpoint state?"),
    ("mira", "Yes. Only the effective conversation is transferred. The destination thread keeps its independent checkpoint state.\n\nYou can inspect the handoff implementation in the current codebase."),
]

DEMO_MODEL_ALIASES = ("lmstudio-gemma-4-12b", "qwen3.5-27b", "gemma-fast")


def available_model_aliases(workspace: Path) -> tuple[tuple[str, ...], str]:
    """Read exactly the configured profile ALIASES, never provider/model IDs.

    Settings -> Models uses the registry's profile names. If a probe is run
    without valid user profiles, use clearly simulated demo choices.
    Never write .mira/models.yml or .mira/settings.yml.
    """
    registry = load_model_registry(workspace)
    options = tuple(registry.profiles)
    settings = load_settings(workspace)
    selected = model_assignment(settings, MAIN_MODEL)
    if not options:
        # A profile can be present but invalid only because an environment
        # variable is unavailable. The Settings UI still knows its alias;
        # read only the YAML *keys*, never the provider/model values.
        registry_path = workspace / ".mira" / "models.yml"
        if registry_path.is_file():
            try:
                import yaml
                raw = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
                raw_models = raw.get("models", {}) if isinstance(raw, dict) else {}
                if isinstance(raw_models, dict):
                    options = tuple(str(name) for name in raw_models if isinstance(name, str))
            except (OSError, ValueError, UnicodeError, yaml.YAMLError):
                pass
    if not options:
        options = DEMO_MODEL_ALIASES
    if not selected or selected not in options:
        selected = options[0]
    return options, selected


class ProbeTelemetryBar(TelemetryBar):
    """Read-only *effective* model aliases and right-aligned MIRA metrics.

    The model labels are intrinsically sized, NOT percentage-capped against
    an auto-sized container (which truncated them even on wide terminals).
    """

    def compose(self) -> ComposeResult:
        with Horizontal(id="probe-effective-models"):
            yield Static("main: unset", id="probe-model-main")
            yield Static("│", id="probe-model-separator")
            yield Static("btw: unset", id="probe-model-btw")
        yield Static(id="telemetry-values")

    def set_state(self, *, model_name: str, btw_name: str | None = None,
                  dashboard: dict | None = None, turns: int = 0) -> None:
        self.query_one("#probe-model-main", Static).update(f"main: {model_name or 'unset'}")
        has_btw = btw_name is not None
        divider = self.query_one("#probe-model-separator", Static)
        btw_label = self.query_one("#probe-model-btw", Static)
        divider.display = has_btw
        btw_label.display = has_btw
        if has_btw:
            btw_label.update(f"btw: {btw_name}")
        self.query_one("#telemetry-values", Static).update(telemetry_values(dashboard, turns))


class ComposerActions(Horizontal):
    """Right-pinned model trigger within the single outlined prompt composer."""

    def __init__(self, kind: Literal["main", "btw"]):
        super().__init__(classes="probe-composer-actions", id=f"actions-{kind}")
        self.kind = kind

    def compose(self) -> ComposeResult:
        yield Button("+ Add file", classes="probe-add-file", id=f"attach-{self.kind}",
                     tooltip=f"Simulate attaching a file to {self.kind.upper()} conversation")
        # This one flexible item takes all spare room. The model is NOT 1fr.
        yield Static("sample.txt" if self.app.attached_mock[self.kind] else "",
                     classes="probe-attached-name", id=f"attached-{self.kind}")
        yield Button("", classes="probe-model-trigger", id=f"model-trigger-{self.kind}",
                     tooltip=f"Choose {self.kind.upper()} model alias")

    def on_mount(self) -> None:
        self.call_after_refresh(self.sync_model_button)

    def on_resize(self, _event: Resize) -> None:
        self.call_after_refresh(self.sync_model_button)

    def sync_model_button(self) -> None:
        """Fit the alias at the far right, leaving room for Add file and arrow.

        At the dock's minimum width the alias is shortened but the ▼ stays
        visible. The pop-up itself always offers the complete alias options.
        """
        if not self.is_mounted:
            return
        button = self.query_one(f"#model-trigger-{self.kind}", Button)
        add = self.query_one(f"#attach-{self.kind}", Button)
        alias = self.app.model_configured[self.kind]
        available = max(0, self.content_region.width)
        add_width = add.region.width or 13
        maximum = max(7, available - add_width - 1)
        desired = min(len(alias) + 5, maximum)
        button.styles.width = desired
        # Textual Button has one line-pad cell on each side; budget for them,
        # the gap and the final down-arrow.
        budget = max(1, desired - 4)
        shown = alias if len(alias) <= budget else alias[:max(1, budget - 1)] + "…"
        button.label = f"{shown} ▾"
        button.tooltip = f"Select {self.kind.upper()} model: {alias}"


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
        # minimum-width 28-column dock. Highlight the current option instead.
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
        options.focus()

    @on(OptionList.OptionSelected, "#model-alias-options")
    def alias_selected(self, event: OptionList.OptionSelected) -> None:
        index = event.option_index
        selected = self.aliases[index] if 0 <= index < len(self.aliases) else None
        self.dismiss(selected)
        event.stop()

    def action_dismiss_aliases(self) -> None:
        self.dismiss(None)

    def on_click(self, event: Click) -> None:
        if event.widget is self:
            self.dismiss(None)


class SettingsPreview(ModalScreen[None]):
    """Read-only settings preview: rail wrench always opens GENERAL.

    Model choices are the same registry aliases as the composer dropdowns.
    Settings model changes are simulated in this probe, never persisted.
    """

    CSS = """
    SettingsPreview { align: center middle; background: rgba(4,7,8,0.76); }
    #settings-preview-panel { width: 68; max-width: 90%; height: auto;
       background: #10191c; border: solid #5bb8b1; padding: 1 2; }
    #settings-preview-header { width: 1fr; height: 2; }
    #settings-preview-heading { width: 1fr; text-style: bold; color: #eef7f8; }
    #settings-preview-close { min-width: 0; width: 4; height: 1; border: none;
        background: transparent; color: #d7e8e9; }
    #settings-preview-tabs { height: 2; width: 1fr; }
    #settings-preview-note { height: auto; padding: 1 0; color: #b8c1c7; }
    #settings-preview-model-main, #settings-preview-model-btw {
        width: 1fr; height: 3; }
    .settings-preview-model-label { width: 12; height: 1; color: #d9e3e6; }
    .settings-preview-select { width: 1fr; min-width: 18; height: 1; }
    """
    BINDINGS = [Binding("escape", "dismiss_preview", "Close", show=False)]

    def compose(self) -> ComposeResult:
        with Vertical(id="settings-preview-panel"):
            with Horizontal(id="settings-preview-header"):
                yield Static("MIRA Settings · UI preview", id="settings-preview-heading")
                yield Button("×", id="settings-preview-close")
            yield Tabs(Tab("General", id="tab-general"),
                       Tab("Models", id="tab-models"),
                       Tab("Access", id="tab-access"),
                       id="settings-preview-tabs")
            yield Static("Theme: Dark\nDefault mode: ACT", id="settings-preview-note")
            with Vertical(id="settings-preview-models"):
                with Horizontal(id="settings-preview-model-main"):
                    yield Static("Main", classes="settings-preview-model-label")
                    yield Select([(x, x) for x in self.app.model_aliases],
                                 value=self.app.model_configured["main"],
                                 allow_blank=False, compact=True,
                                 id="settings-model-main", classes="settings-preview-select")
                with Horizontal(id="settings-preview-model-btw"):
                    yield Static("BTW", classes="settings-preview-model-label")
                    yield Select([(x, x) for x in self.app.model_aliases],
                                 value=self.app.model_configured["btw"],
                                 allow_blank=False, compact=True,
                                 id="settings-model-btw", classes="settings-preview-select")
            yield Static("Aliases only · no settings are saved by the probe.",
                         id="settings-preview-footer")

    def on_mount(self) -> None:
        self.query_one("#settings-preview-tabs", Tabs).active = "tab-general"
        self._show_tab("General")

    def _show_tab(self, name: str) -> None:
        self.query_one("#settings-preview-models", Vertical).display = name == "Models"
        self.query_one("#settings-preview-note", Static).display = name != "Models"
        if name == "General":
            self.query_one("#settings-preview-note", Static).update("Theme: Dark\nDefault mode: ACT")
        elif name == "Access":
            self.query_one("#settings-preview-note", Static).update("Access controls (preview)")

    @on(Tabs.TabActivated, "#settings-preview-tabs")
    def tab_changed(self, event: Tabs.TabActivated) -> None:
        if event.tab is not None:
            self._show_tab(event.tab.label_text)

    @on(Select.Changed, ".settings-preview-select")
    def settings_model_changed(self, event: Select.Changed) -> None:
        """Mirror the same aliases as the composer; probe-only application."""
        if event.value is Select.NULL:
            return
        kind = "btw" if event.select.id == "settings-model-btw" else "main"
        alias = str(event.value)
        if alias not in self.app.model_aliases:
            return
        self.app.model_configured[kind] = alias
        self.app.model_effective[kind] = alias  # simulate confirmed switch
        for composer in list(self.app.query(ComposerActions)):
            if composer.kind == kind:
                composer.call_after_refresh(composer.sync_model_button)
        self.app._render_status()
        event.stop()

    @on(Button.Pressed, "#settings-preview-close")
    def close_clicked(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    def action_dismiss_preview(self) -> None:
        self.dismiss(None)



class DemoStore:
    """Read-only in-memory session store, shaped like MIRA's store interface."""

    def __init__(self):
        self.records: dict[str, dict] = {}
        stamp = datetime.now(timezone.utc).isoformat()
        for i, (key, label, pinned) in enumerate(SAMPLE_HISTORY):
            self.records[key] = {
                "id": key, "custom_title": label, "pinned": pinned,
                "created_at": stamp, "updated_at": stamp,
                "events": [{"type": "user", "text": label}],
            }

    def session_paths(self):
        return list(self.records)

    def read(self, path):
        return self.records[str(path)]


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
        super().__init__(title + "     ", id=f"head-{group}-{key}")
        self._set_actions(False)
        self.tooltip = "Click tab to select; hover for Dock and Close actions"

    @property
    def dock_x(self) -> int:
        # CSS defines exactly one cell of horizontal padding.
        return 1 + len(self.title) + 2

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
            self.label = label
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

    def scroll_strip(self, direction: int) -> None:
        scroller = self.query_one("#tabs-scroll")
        target = min(max(0, scroller.scroll_x + direction * 12), scroller.max_scroll_x)
        scroller.scroll_to(x=target, animate=False, force=True)

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
        super().__init__("", classes="probe-divider", **kwargs)
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


class CompactPromptResizeHandle(PromptResizeHandle):
    """Probe-only copy of the native resize clamp with a 3-row minimum.

    Production MIRA's MIN_PROMPT_HEIGHT remains untouched. This composer
    consists of two editable content rows and one model/attachment row;
    its upper and lower border strokes are additional frame rows.
    """

    def on_mouse_move(self, event: MouseMove) -> None:
        if self._drag_start_y is None or self._prompt_start_height is None:
            return
        requested = self._prompt_start_height + self._drag_start_y - event.screen_y
        shell = self.parent
        prompt = shell.query_one(PromptBox)
        prompt.styles.height = min(shell.safe_prompt_height, max(3, requested))
        event.stop()
        event.prevent_default()


class ProbeAutocompleteInput(AutocompleteInput):
    """Use MIRA's existing prompt/drag handle in a tabbed chat surface.

    The production safe_prompt_height looks for #transcript-viewport and
    #inspector in MiraApp. Neither exists in this standalone tabbed probe,
    which otherwise limits vertical dragging to the current prompt height.
    Only the geometry calculation is adapted; the real PromptResizeHandle
    and PromptBox still handle the mouse and input.
    """

    def compose(self) -> ComposeResult:
        # Preserve MIRA's real PromptBox, autocomplete popup and native grip,
        # replacing ONLY the grip's probe-local five-row minimum.
        yield OptionList(id="autocomplete-options", compact=True)
        yield PromptBox()
        yield CompactPromptResizeHandle()

    @property
    def safe_prompt_height(self) -> int:
        prompt = self.query_one(PromptBox)
        current = max(3, prompt.region.height)
        if self.parent is None:
            return current
        try:
            log = self.parent.parent.query_one("#chat-log", ChatLog)
        except NoMatches:
            return current
        # Real ChatLog has no outer border in this dockable probe.
        # Retain three visible lines for the transcript during a resize.
        spare = max(0, log.region.height - 3)
        return current + spare


class ChatSurface(Vertical):
    """Real MIRA transcript + prompt, with integrated footer controls."""

    def compose(self) -> ComposeResult:
        yield ChatLog(id="chat-log")
        with Vertical(id="chat-composer", classes="probe-composer"):
            yield ProbeAutocompleteInput()
            yield ComposerActions("main")


class BTWPrompt(TextArea):
    """A second input with MIRA's prompt styling, without duplicate #prompt IDs."""

    class Submitted(Message):
        def __init__(self, value: str):
            super().__init__()
            self.value = value

    def __init__(self):
        super().__init__("", id="btw-prompt", placeholder="Ask a side question",
                         show_line_numbers=False)

    @property
    def value(self) -> str:
        return self.text

    @value.setter
    def value(self, text: str) -> None:
        self.text = text

    def on_key(self, event: Key) -> None:
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self.value))
        elif event.key == "shift+enter":
            event.stop()
            event.prevent_default()
            self.insert("\n")


class BTWResizeHandle(Static):
    """MIRA's exact [=] full-width prompt grip, adapted to BTWPrompt.

    Captures the pointer while resizing, so moving over transcript text does
    not select text.  A single BTW height is maintained by the probe across
    right-dock / main-pane recreations of the surface.
    """

    ALLOW_SELECT = False

    def __init__(self):
        super().__init__("", id="btw-resize-handle")
        self._btw_drag_y: int | None = None
        self._btw_initial_height: int | None = None
        self.tooltip = "Drag to resize the BTW prompt (main or right dock)"

    def render(self) -> str:
        # Exact full-width [=] top-border cue from MIRA's PromptResizeHandle.
        width = self.size.width
        grip = "[=]"
        if width < len(grip):
            return "━" * width
        left = (width - len(grip)) // 2
        return "━" * left + grip + "━" * (width - left - len(grip))

    def on_mouse_down(self, event: MouseDown) -> None:
        if event.button != 1:
            return
        prompt = self.parent.query_one(BTWPrompt)
        self._btw_drag_y = event.screen_y
        self._btw_initial_height = prompt.region.height
        self.app.begin_divider_drag()
        self.capture_mouse()
        event.stop()
        event.prevent_default()

    def on_mouse_move(self, event: MouseMove) -> None:
        if self._btw_drag_y is None or self._btw_initial_height is None:
            return
        input_shell = self.parent
        requested = self._btw_initial_height + self._btw_drag_y - event.screen_y
        target = max(3, min(input_shell.safe_prompt_height, requested))
        prompt = input_shell.query_one(BTWPrompt)
        prompt.styles.height = target
        self.app.btw_prompt_height = target
        self.app.clear_selection()
        event.stop()
        event.prevent_default()

    def _stop_btw_drag(self) -> None:
        if self._btw_drag_y is None:
            return
        self._btw_drag_y = None
        self._btw_initial_height = None
        self.app.end_divider_drag()

    def on_mouse_up(self, event: MouseUp) -> None:
        if self._btw_drag_y is not None and event.button == 1:
            self._stop_btw_drag()
            self.release_mouse()
            event.stop()
            event.prevent_default()

    def on_mouse_release(self, _event: MouseRelease) -> None:
        self._stop_btw_drag()


class BTWInput(Vertical):
    """Independent BTW input; the resize affordance follows it between panes."""

    def compose(self) -> ComposeResult:
        yield BTWPrompt()
        yield BTWResizeHandle()

    def on_mount(self) -> None:
        self.query_one(BTWPrompt).styles.height = self.app.btw_prompt_height
        self.call_after_refresh(self._sync_handle)
        self.call_after_refresh(self._shrink_to_fit)

    def on_resize(self, _event: Resize) -> None:
        self.call_after_refresh(self._sync_handle)
        self.call_after_refresh(self._shrink_to_fit)

    def _sync_handle(self) -> None:
        prompt_width = self.query_one(BTWPrompt).region.width
        if prompt_width:
            self.query_one(BTWResizeHandle).styles.width = max(1, prompt_width - 2)

    @property
    def safe_prompt_height(self) -> int:
        prompt = self.query_one(BTWPrompt)
        current = max(3, prompt.region.height)
        composer = self.parent
        surface = composer.parent if composer is not None else None
        if surface is None:
            return current
        try:
            log = surface.query_one("#btw-log", ChatLog)
        except NoMatches:
            return current
        # Chat transcript retains at least three visible rows. Use the local
        # pane rather than a global viewport: right and main both work.
        pane_limit = max(3, surface.content_region.height - 3)
        reclaimable = max(0, log.region.height - 3)
        return max(3, min(pane_limit, current + reclaimable))

    def _shrink_to_fit(self) -> None:
        prompt = self.query_one(BTWPrompt)
        limit = self.safe_prompt_height
        if prompt.region.height > limit:
            prompt.styles.height = limit
            self.app.btw_prompt_height = limit


class BTWSurface(Vertical):
    def compose(self) -> ComposeResult:
        yield ChatLog(id="btw-log")
        with Vertical(id="btw-composer", classes="probe-composer"):
            yield BTWInput(id="btw-input-shell")
            yield ComposerActions("btw")


class FileSurface(Vertical):
    def __init__(self, path: Path):
        super().__init__(classes="probe-file")
        self.path = path

    def compose(self) -> ComposeResult:
        yield Static(f" {self.path.name}  ·  READ ONLY", classes="probe-file-heading")
        try:
            if not self.path.is_file():
                raise OSError("Not a regular file")
            if self.path.stat().st_size > 350_000:
                raise OSError("Preview limited to 350 KB")
            data = self.path.read_bytes()
            if b"\x00" in data[:4096]:
                raise OSError("Binary preview not supported")
            content = data.decode("utf-8", errors="replace")
        except OSError as error:
            yield Static(str(error), classes="probe-file-error")
            return
        if self.path.suffix.lower() in {".md", ".markdown"}:
            with VerticalScroll(classes="probe-markdown-scroll"):
                yield Markdown(content)
        else:
            # MIRA does not currently include a full code editor. Rendering with
            # the native Textual TextArea avoids introducing an alien theme.
            yield TextArea(
                content, read_only=True, language=None,
                show_line_numbers=True, soft_wrap=False,
                classes="probe-code", tab_behavior="focus",
            )


class HelpScreen(ModalScreen[None]):
    CSS = """
    HelpScreen { align: center middle; background: rgba(4,7,8,0.70); }
    HelpScreen #probe-help { width: 77; max-width: 92%; height: auto;
      padding: 1 2; border: heavy #5bb8b1; background: #101516; }
    HelpScreen #help-title { height: 2; color: #d6fff6; text-style: bold; }
    HelpScreen #help-body { height: auto; color: #bdc8cd; }
    HelpScreen #help-close { width: 12; min-width: 12; margin-top: 1; }
    """
    BINDINGS = [Binding("escape", "close", "Close", show=False)]

    def compose(self) -> ComposeResult:
        with Vertical(id="probe-help"):
            yield Static("MIRA UI PROBE · REAL MIRA WIDGETS", id="help-title")
            yield Static(
                "Chat / File rail: Ctrl+1 / Ctrl+2 (click again to collapse)\n"
                "Change chat icon: F4 cycles standard Unicode characters\n"
                "Open 10 example file tabs: F5 (test horizontal overflow)\n"
                "New chat: Ctrl+N     Inspector: Ctrl+I     BTW tab: Ctrl+G\n"
                "BTW modal: Ctrl+B   Dock active tab: Ctrl+Right\n"
                "Return active docked tab: Ctrl+Left; per-tab ‹/› and × on hover\n"
                "Arrows beside tabs scroll; ⋯ lists ALL tabs / opens BTW, Inspector\n"
                "Drag the INVISIBLE 1-cell gap between solid panel borders;\n"
                "hover/drag turns that cell teal, with no third idle line.\n"
                "All four panels own complete solid borders. No extra\n"
                "top/bottom frame rules or bottom docking.\n\n"
                "Actual MIRA widgets: ChatLog, AutocompleteInput/PromptBox,\n"
                "SessionHistory, StatusBar, TelemetryBar, Inspector.\n"
                "MIRA's original mira.tcss is loaded from this checkout.\n"
                "Chat and BTW composers include Add file and model dropdown.\n"
                "Footer model text is read-only; BTW only when open.\n"
                "Settings wrench opens General. Aliases load from models.yml.\n\n"
                "All chat data is simulated. No model calls, writes, or\n"
                "session persistence.  F1 opens help. Ctrl+Q exits.",
                id="help-body",
            )
            yield Button("Close", id="help-close")

    @on(Button.Pressed, "#help-close")
    def click_close(self, event: Button.Pressed) -> None:
        self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)


class BTWModal(ModalScreen[None]):
    """Modal comparison using the very same MIRA ChatLog and prompt widgets."""
    CSS = """
    BTWModal { align: center middle; background: rgba(4,7,8,0.72); }
    BTWModal #btw-dialog { width: 88; max-width: 94%; height: 80%;
       border: heavy #5bb8b1; padding: 0 1; background: #101516; }
    BTWModal #btw-dialog-title { height: 2; color: #d6fff6;
       text-style: bold; padding: 0 1; }
    BTWModal #btw-modal-close { dock: right; width: 3; height: 1;
       border: none; padding: 0; background: transparent; }
    BTWModal #btw-modal-log { height: 1fr; width: 1fr; border: solid #2d5661;
       background: #0c0f10; padding: 0 1; }
    """
    BINDINGS = [Binding("escape", "dismiss_modal", "Close", show=False)]

    def compose(self) -> ComposeResult:
        with Vertical(id="btw-dialog"):
            yield Static("BTW  ·  Independent side conversation", id="btw-dialog-title")
            yield Button("×", id="btw-modal-close", tooltip="Close modal")
            yield ChatLog(id="btw-modal-log")
            yield BTWPrompt()

    def on_mount(self) -> None:
        log = self.query_one("#btw-modal-log", ChatLog)
        for role, body in self.app.btw_messages:
            if role == "you":
                log.user_message(body)
            else:
                log.assistant_message(body)
        self.query_one("#btw-prompt", BTWPrompt).focus()

    # Custom Submitted messages have no `control`; do not use CSS selectors.
    @on(BTWPrompt.Submitted)
    def submit_question(self, event: BTWPrompt.Submitted) -> None:
        event.stop()
        question = event.value.strip()
        if not question:
            return
        self.app.btw_messages.extend([
            ("you", question),
            ("mira", "Mock BTW response; the main conversation is unchanged."),
        ])
        log = self.query_one("#btw-modal-log", ChatLog)
        log.user_message(question)
        log.assistant_message("Mock BTW response; the main conversation is unchanged.")
        self.query_one("#btw-prompt", BTWPrompt).value = ""

    @on(Button.Pressed, "#btw-modal-close")
    def close_modal(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    def action_dismiss_modal(self) -> None:
        self.dismiss(None)


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
    BINDINGS = [Binding("escape", "cancel", "Close tab menu", show=False)]

    def __init__(self, group: Literal["main", "right"], options: list[Option],
                 anchor_right: int, anchor_bottom: int,
                 pane_region: Any, screen_size: tuple[int, int]) -> None:
        super().__init__()
        self.group = group
        screen_w, screen_h = screen_size
        self.menu_width = min(24, max(8, pane_region.width - 2), screen_w - 2)
        # One visible row per option. The menu has two border cells, two
        # horizontal option-padding cells and may reserve a scrollbar cell.
        # Reserve one further cell rather than relying on OptionList wrapping
        # long filenames into a second row (which broke the menu geometry).
        self.label_columns = max(1, self.menu_width - 6)
        self.options = [
            Option(
                self._single_line_label(str(option.prompt), self.label_columns),
                id=option.id, disabled=option.disabled,
            )
            for option in options
        ]
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
        menu.focus()

    @on(OptionList.OptionSelected, "#tab-overflow-options")
    def select(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.dismiss(str(event.option.id or ""))

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_click(self, event: Click) -> None:
        # Same click-away dismissal pattern as MIRA v29's ModelAliasPopup.
        if event.widget is self:
            self.dismiss(None)


class HistoryActionDropdown(SessionActionMenuScreen):
    """MIRA's EXISTING History action menu with click-outside dismissal.

    Inherited placement is unchanged: immediately RIGHT of the originating
    History row and aligned with the row TOP (not below the ellipsis).
    It also preserves the exact Rename / Pin / Delete / Cancel options.
    """

    CSS = """
    HistoryActionDropdown { align: left top; background: transparent; }
    """

    def on_click(self, event: Click) -> None:
        if event.widget is self:
            self.dismiss(None)


class MiraFidelityProbe(App[None]):
    """Real-MIRA visual shell with a simulated, movable workspace."""

    TITLE = "MIRA · Textual UI fidelity probe"
    CSS_PATH = [MIRA_STYLESHEET]
    CSS = """
    /* LOCKED MIRA WORKSPACE DESIGN (v32):
       - Real MIRA status/footer are full-width, one terminal row, teal.
       - Rail, sidebar, editor and right dock each own a SOLID four-edge border.
       - No fake junctions, custom horizontal rules or added top/bottom rows.
       - Between pairs of panel borders: one *invisible* draggable cell, teal
         only on hover/drag. Rail/sidebar use a plain 1-cell dark separation.
       - History/Explorer list body remains charcoal, selected row stays teal
         with the original gold indicator.
       - Tab glow accent per active content type: Chat teal; Files soft blue;
         Inspector purple; BTW soft coral pink (never warning amber).
       - Startup: sidebar collapsed, no dock, so just rail + main chat.
       - v24 refinement: rail 9 cells; extra 1-cell side inset for
         History/Explorer contents and header.
       - v25/v26 fix: override Textual Button's default 16-cell min-width
         while retaining the default 1-cell line padding. Textual 8.2.7
         rejects `line-pad: 0`, so do not specify it here.
       - v27: hide redundant Inspector header.
       - v30: compact Chat/BTW input to two rows plus one action row;
         Chat has no dock affordance and cannot move; right overflow and
         History ... originally used in-pane dropdown OptionLists.
       - v32: SAME anchored popup for both main/dock ellipses, click-away and
         Escape to close. History uses its ORIGINAL MIRA right-of-row,
         top-aligned menu, with click-away dismissal added.
       - v30: continuously bordered Chat/BTW composers, with independent
         Add file actions and model-alias model alias buttons at the bottom.
         Footer is passive runtime alias telemetry, BTW only if open.
    */
    #probe-root { height: 100%; width: 100%; layout: vertical;
       background: #0c0f10; }
    #status-row {
      width: 100%; height: 1; min-height: 1; max-height: 1;
      padding: 0 1; margin: 0; border: none; background: #17313a;
    }
    #app-shell { height: 1fr; min-height: 0; width: 100%;
      layout: horizontal; padding: 0 1; margin: 0;
      background: #0c0f10; }
    #telemetry-row {
      width: 100%; height: 1; min-height: 1; max-height: 1;
      padding: 0 1; margin: 0; border: none; background: #17313a;
    }
    /* Model telemetry is NONCLICKABLE; aliases only, not full provider IDs. */
    /* 1fr is the FLEXIBLE SPACE before telemetry, not a max-width on labels. */
    #probe-effective-models { height: 1; width: 1fr; min-width: 0;
      layout: horizontal; background: transparent; }
    #probe-model-main { height: 1; width: auto; min-width: 0;
      text-wrap: nowrap; color: #5bb8b1; }
    #probe-model-separator { height: 1; width: 3; min-width: 3;
      content-align: center middle; color: #83999e; }
    #probe-model-btw { height: 1; width: auto; min-width: 0;
      text-wrap: nowrap; color: #e58faa; }
    #telemetry-row #telemetry-values { height: 1; width: auto;
      min-width: 25; content-align: right middle; }
    /* Side rail is fully bordered, including both horizontal edges. */
    #navigation-rail { width: 9; min-width: 9; max-width: 9; height: 100%;
      padding: 0; margin: 0; border: solid #51696c;
      background: #0c0f10; overflow: hidden; }
    /* Textual 8.2.7 Button defaults to min-width: 16 and line-pad: 1.
       Override only min-width: the default line-pad: 1 is fine for 7 inner
       columns; specifying line-pad: 0 causes a CSS parse error. */
    #navigation-rail Button { width: 100%; min-width: 0; max-width: 100%;
      height: 3; min-height: 3;
      padding: 0; margin: 0; border: none;
      content-align: center middle; text-align: center;
      background: transparent; color: #aeb8be;
      text-style: not reverse; background-tint: transparent; tint: transparent; }
    #navigation-rail Button:hover, #navigation-rail Button:focus,
    #navigation-rail Button.-active { color: #d6fff6;
      text-style: not reverse; background-tint: transparent; tint: transparent; }
    #navigation-rail Button:hover { background: #21424b; color: #7ce3dc;
      text-style: not reverse; background-tint: transparent; tint: transparent; }
    #navigation-rail Button.selected, #navigation-rail Button.selected:hover,
    #navigation-rail Button.selected:focus,
    #navigation-rail Button.selected.-active {
      background: #21424b; color: #7ce3dc; text-style: bold not reverse;
      background-tint: transparent; tint: transparent; }
    #rail-space { height: 1fr; }
    /* The dark cell after rail is intentional breathing room, not a border. */
    #rail-panel-gap { width: 1; min-width: 1; max-width: 1;
      height: 100%; padding: 0; margin: 0; background: #0c0f10; }

    #session-sidebar { display: none; width: 42; min-width: 42;
      height: 100%; border: solid #51696c; margin: 0; padding: 0;
      background: #0c0f10; }
    #session-sidebar-header { height: 2; min-height: 2; max-height: 2;
      margin: 0; padding: 0 1; background: #0c0f10; border: none; }
    #session-sidebar-title { color: #c2cbd0; text-style: bold; }
    #collapse-sidebar { width: 3; min-width: 3; height: 1; min-height: 1;
      border: none; padding: 0; background: transparent; color: #7ce3dc; }
    #sidebar-content, #history-panel, #files-panel { height: 1fr; width: 1fr;
      background: #1e1e1e; }
    #sessions { height: 1fr; width: 1fr; padding: 0 2;
      background: #1e1e1e; }
    #sessions > .session-row { background: #1e1e1e; }
    #sessions > .session-row.active,
    #sessions > .session-row.--highlight { background: #1b3036; }
    #files-panel DirectoryTree { height: 1fr; width: 1fr; border: none;
      padding: 0 2; background: #1e1e1e; color: #c4cdd2;
      scrollbar-color: #5bb8b1; scrollbar-background: #122023; }
    #files-panel DirectoryTree > .tree--cursor { background: #1b3036; }
    #files-panel DirectoryTree:focus > .tree--cursor { background: #21424b; }

    /* One 1-cell HIT TARGET between the side-by-side border strokes.
       It consumes exactly one column in both hover and idle states, so
       hovering and dragging cannot shift panel positions. No border glyph. */
    .probe-divider { display: none;
      width: 1; min-width: 1; max-width: 1;
      height: 100%; min-height: 100%;
      padding: 0; margin: 0; border: none;
      background: #0c0f10; color: #0c0f10;
      pointer: ew-resize; overflow: hidden; }
    .probe-divider:hover, .probe-divider.dragging {
      background: #5bb8b1; color: #5bb8b1; }

    #main-panel { height: 100%; width: 1fr; min-width: 24;
      padding: 0; margin: 0; border: none; background: #0c0f10; }
    #editor-workspace { height: 1fr; width: 1fr; layout: horizontal;
      padding: 0; margin: 0; }
    #primary-pane { height: 100%; width: 1fr; min-width: 24;
      margin: 0; padding: 0; border: solid #51696c;
      background: #0c0f10; }
    #right-pane { height: 100%; width: 42; min-width: 28; display: none;
      margin: 0; padding: 0; border: solid #51696c;
      background: #0c0f10; }
    /* Each panel has its own native tab strip inside its outer border. */
    .probe-top-actions { width: 1fr; height: 2; min-height: 2; max-height: 2;
      margin: 0; padding: 0; background: #0c0f10; layout: horizontal; }
    .probe-header-tabs { width: 1fr; height: 2; min-width: 1;
      background: #0c0f10; }
    .probe-header-tabs Tab { background: #0c0f10; color: #aeb8be;
      height: 1; padding: 0 1; text-align: left; text-style: not reverse; }
    .probe-header-tabs Tab:hover { background: #1b3036; color: #e8edef; }
    /* Active tab stays part of the tab strip, rather than sharing the
       STATUS BAR's #17313a background.  The teal underline marks activity. */
    .probe-header-tabs Tab.-active,
    .probe-header-tabs:focus Tab.-active { background: #0c0f10;
      color: #eef7f8; text-style: bold not reverse; }
    /* The inactive underline track is the same colour as the tab strip;
       only the active tab segment gets a teal accent. */
    .probe-header-tabs Underline { background: #0c0f10; }
    .probe-header-tabs Underline > .underline--bar {
      background: #0c0f10; color: #5bb8b1; }
    .probe-header-tabs.accent-file Underline > .underline--bar {
      color: #78a9e8; }
    .probe-header-tabs.accent-inspector Underline > .underline--bar {
      color: #b785ff; }
    .probe-header-tabs.accent-btw Underline > .underline--bar {
      color: #e58faa; }
    .probe-header-tabs.accent-chat Underline > .underline--bar {
      color: #5bb8b1; }
    .probe-header-scroll { width: 10; min-width: 10; max-width: 10;
      height: 2; background: #0c0f10; }
    /* The main and dock tab-scroll arrows have ONLY a transient mouse-hover
       highlight. Keyboard focus, click/pressed state and reachability must
       NEVER produce a persistent teal block or highlight the sibling arrow.
       Keep navigation and boundary disabling unchanged. */
    .probe-header-scroll Button { width: 3; min-width: 3; max-width: 3;
      height: 1; min-height: 1; border: none; padding: 0;
      margin: 0; background: #0c0f10; color: #b8c1c7;
      text-style: not reverse; content-align: center middle;
      background-tint: transparent; tint: transparent; }
    /* Explicitly neutralise the built-in Button focus / pressed appearance.
       Focus remains usable for keyboard navigation, but has NO hover glow. */
    .probe-header-scroll Button:focus,
    .probe-header-scroll Button.-active,
    .probe-header-scroll Button:focus.-active {
      background: #0c0f10; color: #b8c1c7;
      border: none; text-style: not reverse;
      background-tint: transparent; tint: transparent; }
    /* A focused button still lights up if (and ONLY if) the pointer is over it. */
    .probe-header-scroll Button:hover:enabled,
    .probe-header-scroll Button:focus:hover:enabled {
      background: #21424b; color: #eef7f8;
      border: none; text-style: not reverse;
      background-tint: transparent; tint: transparent; }
    /* Disable state wins over both focus and hover, including at strip edges. */
    .probe-header-scroll Button:disabled,
    .probe-header-scroll Button:disabled:focus,
    .probe-header-scroll Button:disabled:hover {
      background: #0c0f10; color: #56646a;
      border: none; text-style: not reverse;
      background-tint: transparent; tint: transparent; }

    .probe-tabs { height: 1fr; width: 1fr; background: #0c0f10; }
    /* Only hide the INTERNAL TabbedContent tab bar, not the visible header
       Tabs that sit in the adjacent #probe-top-actions row. */
    .probe-tabs > ContentTabs { display: none; }
    .probe-tabs TabPane { padding: 0; height: 1fr; width: 1fr; }
    /* ONE owner of each horizontal gutter: the Chat / BTW surface.
       v14 applied transcript padding and prompt-wrapper margins separately,
       offsetting their borders, resize handle, and lower corners. Avoid any
       child margin on layers: base resize / dock: left. */
    .probe-tabs ChatSurface, .probe-tabs BTWSurface {
      height: 1fr; width: 1fr; padding: 0 1;
    }
    .probe-tabs #chat-log, .probe-tabs #btw-log {
      width: 1fr; height: 1fr; padding: 0; border: none;
      scrollbar-color: #5bb8b1; scrollbar-background: #122023;
    }
    .probe-tabs #autocomplete-input {
      width: 1fr; height: auto; margin: 0; padding: 0;
    }
    .probe-tabs .probe-composer { height: auto; width: 1fr;
      padding: 0; margin: 0; background: #0c0f10; }
    /* Native prompt owns the TOP and SIDES; the row underneath completes
       the SAME outline. No extra independent toolbar border or gutter. */
    .probe-tabs #prompt { height: 3; border-top: heavy #5bb8b1;
      border-left: heavy #5bb8b1; border-right: heavy #5bb8b1;
      border-bottom: none; padding: 0 1; background: transparent; }
    .probe-composer-actions { height: 2; min-height: 2; max-height: 2;
      layout: horizontal; margin: 0; padding: 0;
      border-left: heavy #5bb8b1; border-right: heavy #5bb8b1;
      border-bottom: heavy #5bb8b1; border-top: none;
      background: #0c0f10; }
    .probe-composer-actions .probe-add-file {
      height: 1; min-height: 1; min-width: 9; width: auto;
      padding: 0 1; margin: 0; border: none;
      background: transparent; color: #a9d9d5;
      text-style: not reverse; }
    .probe-composer-actions .probe-add-file:hover,
    .probe-composer-actions .probe-add-file:focus {
      background: #183037; color: #eafffc;
      text-style: not reverse; }
    .probe-composer-actions .probe-attached-name {
      width: 1fr; height: 1; min-width: 0; color: #80cec8;
      text-wrap: nowrap; text-overflow: ellipsis; }
    /* Fixed-to-content trigger follows the flexible attached-name spacer.
       No 1fr width / percentage cap: full alias at wide widths; clipped alias
       plus a persistent arrow at narrow widths via ComposerActions. */
    .probe-composer-actions .probe-model-trigger {
      width: auto; min-width: 0; max-width: 100%; height: 1; min-height: 1;
      margin: 0; padding: 0; border: none; background: transparent;
      color: #b8d0e1; content-align: right middle; text-align: right;
      text-style: not reverse; }
    .probe-composer-actions .probe-model-trigger:hover,
    .probe-composer-actions .probe-model-trigger:focus {
      background: #182c31; color: #eef7f8; border: none; }

    .probe-tabs #btw-input-shell {
      height: auto; width: 1fr; margin: 0; padding: 0;
      layers: base resize;
    }
    #btw-prompt { height: 3; border-top: heavy #5bb8b1;
      border-left: heavy #5bb8b1; border-right: heavy #5bb8b1;
      border-bottom: none; padding: 0 1; background: transparent; }
    /* Same full-width [=] resize line as the original MIRA Chat prompt. */
    #btw-resize-handle {
      layer: resize; dock: left;
      width: 1; height: 1; offset: 1 0;
      padding: 0; margin: 0; border: none;
      background: transparent; color: #5bb8b1;
      pointer: ns-resize;
    }
    #btw-resize-handle:hover { color: #7ce3dc; }

    /* File previews are editor surfaces, not framed dialog boxes. */
    .probe-file { height: 1fr; width: 1fr; background: #0c0f10;
      border: none; padding: 0 1; }
    .probe-file-heading { height: 2; padding: 0 1;
      color: #b8c1c7; background: #0c0f10; }
    .probe-file-error { padding: 1 2; color: #d96b66; height: auto; }
    /* Inspector tab is the only visible title / close affordance.
       The real MIRA Inspector still composes #inspector-header and updates
       #inspector-title dynamically, so hide the whole row (and its close
       button) without removing those widgets or reserving blank space.
       The Inspector transcript consequently begins directly below the tab. */
    .probe-tabs #inspector { border: none; background: #0c0f10; }
    .probe-tabs #inspector-header { display: none; }
    .probe-code { height: 1fr; width: 1fr; border: none;
      padding: 0 1; background: #0c0f10; }
    .probe-markdown-scroll { height: 1fr; width: 1fr; padding: 1 2; }
    """
    BINDINGS = [
        Binding("ctrl+1", "history", "History", show=False),
        Binding("f4", "cycle_chat_icon", "Next chat icon", show=False),
        Binding("f5", "stress_tabs", "Open example tabs", show=False),
        Binding("ctrl+2", "files", "Files", show=False),
        Binding("ctrl+n", "new_chat", "New chat", show=False),
        Binding("ctrl+i", "inspector", "Inspector", show=False),
        Binding("ctrl+g", "btw", "BTW", show=False),
        Binding("ctrl+b", "btw_modal", "BTW modal", show=False),
        Binding("ctrl+right", "dock", "Dock right", show=False),
        Binding("ctrl+left", "undock", "Return to main", show=False),
        Binding("f1", "help", "Help", show=False),
        Binding("ctrl+q", "quit", "Quit", show=False),
    ]
    RAIL_WIDTH = 9
    # Emoji selectors (FE0F) in v16's 🗨️ and ⚙️ were rendered one
    # terminal cell wider/narrower on the exact two icon rows in Windows.
    # Use a single code point per rail symbol: Textual and terminal agree on
    # the cell width for these default-emoji characters more consistently.
    # F4 alternatives are also single Unicode scalars (never VS16 sequences).
    CHAT_GLYPHS = ("💬", "🗨", "🗩", "🗪")
    MIN_DOCK_WIDTH = 107
    MIN_SIDEBAR_WIDTH = 85

    def __init__(self, workspace: Path, *, ascii_icons: bool = False):
        super().__init__()
        self.workspace = workspace.expanduser().resolve()
        self.ascii_icons = ascii_icons
        self.chat_glyph_index = 0
        self.history_store = DemoStore()
        self.sidebar_kind: Literal["history", "files"] = "history"
        self.sidebar_open = False
        self.sidebar_width = 42
        self.right_width = 42
        self.model_aliases, initial_alias = available_model_aliases(self.workspace)
        self.model_configured = {"main": initial_alias, "btw": initial_alias}
        # In production, effective comes from ACTUAL runtime confirmation;
        # the standalone probe simulates a successful activation on selection.
        self.model_effective = dict(self.model_configured)
        self.attached_mock = {"main": False, "btw": False}
        self.views = {"chat": View("chat", "Chat", "chat")}
        self.main_keys = ["chat"]
        self.right_keys: list[str] = []
        self.folded_keys: list[str] = []
        self.counter = 1
        self.chat_messages = list(SAMPLE_CHAT)
        self.chat_session = "demo-1"
        self.chat_draft = ""
        self.btw_messages: list[tuple[str, str]] = [
            ("mira", "Side questions are isolated from the main conversation.")
        ]
        self.btw_draft = ""
        self.btw_prompt_height = 3
        self._mounted_ready = False
        self._reflow_busy = False
        # Textual posts activation messages as a side effect of removing tabs.
        # Ignore those transient messages until both tab strips are consistent.
        self._tab_mutation_depth = 0
        self._selection_before_drag: bool | None = None

    def compose(self) -> ComposeResult:
        # One real, screen-wide status strip is the FIRST root child. All
        # other columns begin below it; nothing is positioned absolutely.
        glyphs = (
            ("C", "F", "S") if self.ascii_icons else
            (self.CHAT_GLYPHS[self.chat_glyph_index], DirectoryTree.ICON_NODE.strip(), "🔧")
        )
        with Vertical(id="probe-root"):
            with Horizontal(id="status-row"):
                yield StatusBar(id="status")
                yield Button("MCP 3/3", id="mcp-status-button")
            with Horizontal(id="app-shell"):
                with Vertical(id="navigation-rail"):
                    yield Button(glyphs[0], id="rail-chat", tooltip="History · Ctrl+1")
                    yield Button(glyphs[1], id="rail-files", tooltip="Files · Ctrl+2")
                    yield Static("", id="rail-space")
                    yield Button(glyphs[2], id="rail-help", tooltip="Settings · General")
                yield Static("", id="rail-panel-gap")
                with Vertical(id="session-sidebar"):
                    with Horizontal(id="session-sidebar-header"):
                        yield Static("HISTORY", id="session-sidebar-title")
                        yield Button("+ New", id="new-chat", tooltip="New mock chat")
                        yield Button("×", id="collapse-sidebar", tooltip="Collapse sidebar")
                    with ContentSwitcher(initial="history-panel", id="sidebar-content"):
                        with Vertical(id="history-panel"):
                            yield SessionHistory(id="sessions")
                        with Vertical(id="files-panel"):
                            yield DirectoryTree(self.workspace, id="files-tree")
                yield DragDivider("sidebar", id="side-divider")
                with Vertical(id="main-panel"):
                    with Horizontal(id="editor-workspace"):
                        with Vertical(id="primary-pane"):
                            with Horizontal(classes="probe-top-actions"):
                                yield ScrollableHeaderTabs(
                                    ActionTab("Chat", group="main", key="chat"),
                                    id="main-header-tabs", classes="probe-header-tabs")
                                with Horizontal(classes="probe-header-scroll"):
                                    yield Button("‹", id="main-scroll-left", tooltip="Scroll main tabs left")
                                    yield Button("›", id="main-scroll-right", tooltip="Scroll main tabs right")
                                    yield Button("⋯", id="main-overflow", tooltip="All main tabs / Open BTW or Inspector")
                            with TabbedContent(id="main-tabs", classes="probe-tabs"):
                                with TabPane("Chat", id="main-chat"):
                                    yield ChatSurface()
                        yield DragDivider("right", id="right-divider")
                        with Vertical(id="right-pane"):
                            with Horizontal(classes="probe-top-actions"):
                                yield ScrollableHeaderTabs(id="right-header-tabs", classes="probe-header-tabs")
                                with Horizontal(classes="probe-header-scroll"):
                                    yield Button("‹", id="right-scroll-left", tooltip="Scroll docked tabs left")
                                    yield Button("›", id="right-scroll-right", tooltip="Scroll docked tabs right")
                                    yield Button("⋯", id="right-overflow", tooltip="All docked tabs")
                            yield TabbedContent(id="right-tabs", classes="probe-tabs")
            yield ProbeTelemetryBar(id="telemetry-row")

    def on_mount(self) -> None:
        self._mounted_ready = True
        self._refresh_history()
        self._render_chat()
        self._render_status()
        self._layout()

    def _render_status(self) -> None:
        # Show profile ALIASES only. The footer deliberately does not read
        # the Select's pending/configured value: it reads effective runtime.
        self.query_one(StatusBar).set_state(
            mode="ACT", model_name=self.model_effective["main"], state="ready",
            dashboard={}, turns=len(self.chat_messages) // 2, busy=False,
        )
        self.query_one(ProbeTelemetryBar).set_state(
            model_name=self.model_effective["main"],
            btw_name=self.model_effective["btw"] if "btw" in self.views else None,
            dashboard={}, turns=len(self.chat_messages) // 2,
        )

    def _refresh_history(self) -> None:
        self.query_one(SessionHistory).refresh_sessions(
            self.history_store, current_id=self.chat_session
        )

    def _render_chat(self) -> None:
        for surface in list(self.query(ChatSurface)):
            log = surface.query_one("#chat-log", ChatLog)
            log.clear_log()
            log.startup(model_name=self.model_effective["main"], session_id=self.chat_session,
                        workspace=str(self.workspace))
            for role, content in self.chat_messages:
                if role == "you":
                    log.user_message(content)
                else:
                    log.assistant_message(content)
            surface.query_one(PromptBox).value = self.chat_draft
        self._render_status()

    def _render_btw(self) -> None:
        for surface in list(self.query(BTWSurface)):
            log = surface.query_one("#btw-log", ChatLog)
            log.clear_log()
            for role, content in self.btw_messages:
                if role == "you":
                    log.user_message(content)
                else:
                    log.assistant_message(content)
            surface.query_one("#btw-prompt", BTWPrompt).value = self.btw_draft

    def _render_inspector(self) -> None:
        for inspector in list(self.query(Inspector)):
            log = inspector.query_one("#inspector-log", ChatLog)
            log.clear_log()
            log.user_message("Workflow node: classify_documents")
            log.assistant_message("Input state\n  document_count: 12\n  mode: ACT\n\nResult\n  Classified 12 documents\n\nOutput state\n  status: completed")

    def _make_view(self, view: View) -> Widget:
        if view.kind == "chat":
            return ChatSurface()
        if view.kind == "btw":
            return BTWSurface()
        if view.kind == "inspector":
            return Inspector(LiveInspectionStore(), id="inspector")
        assert view.path is not None
        return FileSurface(view.path)

    def _tabs(self, group: Literal["main", "right"]) -> TabbedContent:
        return self.query_one("#main-tabs" if group == "main" else "#right-tabs", TabbedContent)

    def _header_tabs(self, group: Literal["main", "right"]) -> ScrollableHeaderTabs:
        return self.query_one("#main-header-tabs" if group == "main" else "#right-header-tabs", ScrollableHeaderTabs)

    def _sync_tab_accents(self) -> None:
        """Apply the active view's *type*, not the tab label, to its glow bar.

        File tabs can have arbitrary filenames, so determining their colour
        by the title would be wrong. Each group is styled independently.
        """
        for group in ("main", "right"):
            header = self._header_tabs(group)
            key = self._active(group)
            kind = self.views[key].kind if key in self.views else None
            for tab_kind in ("chat", "file", "inspector", "btw"):
                header.set_class(kind == tab_kind, "accent-" + tab_kind)

    def _sync_tab_actions(self) -> None:
        self._sync_tab_accents()
        for tab in self.query(ActionTab):
            tab.sync_actions()
        for group in ("main", "right"):
            try:
                header = self._header_tabs(group)
                scroll = header.query_one("#tabs-scroll")
                for side in ("left", "right"):
                    button = self.query_one(f"#{group}-scroll-{side}", Button)
                    button.disabled = (scroll.max_scroll_x <= 0 or
                        (side == "left" and scroll.scroll_x <= 0) or
                        (side == "right" and scroll.scroll_x >= scroll.max_scroll_x))
            except NoMatches:
                pass

    def _pane_is_present(self, group: Literal["main", "right"], key: str) -> bool:
        """A key in our registry alone is NOT proof Textual still has its pane.

        remove_pane() can post activation messages before the separately
        managed header tab has been removed. Guard both sides of the sync.
        """
        try:
            self._tabs(group).get_pane(f"{group}-{key}")
        except (NoMatches, ValueError):
            return False
        return True

    def _activate(self, group: Literal["main", "right"], key: str) -> None:
        """Activate only after BOTH the header and the content pane exist."""
        keys = self.main_keys if group == "main" else self.right_keys
        if key not in keys or not self._pane_is_present(group, key):
            return
        content_id = f"{group}-{key}"
        header_id = f"head-{group}-{key}"
        if self._header_tabs(group).get_tab(header_id) is None:
            return
        if self._tabs(group).active != content_id:
            self._tabs(group).active = content_id
        if self._header_tabs(group).active != header_id:
            self._header_tabs(group).active = header_id
        self.call_after_refresh(self._sync_tab_actions)

    @on(Tabs.TabActivated)
    def header_tab_activated(self, event: Tabs.TabActivated) -> None:
        # During add/move/close, Textual can emit activation from a tab that
        # is being removed. Never synchronize it back to a missing pane.
        if self._tab_mutation_depth:
            return
        if event.tabs.id not in {"main-header-tabs", "right-header-tabs"} or event.tab is None:
            return
        group: Literal["main", "right"] = (
            "main" if event.tabs.id == "main-header-tabs" else "right"
        )
        prefix = f"head-{group}-"
        tab_id = event.tab.id or ""
        key = tab_id[len(prefix):] if tab_id.startswith(prefix) else ""
        keys = self.main_keys if group == "main" else self.right_keys
        if key in keys and self._pane_is_present(group, key):
            target = f"{group}-{key}"
            if self._tabs(group).active != target:
                self._tabs(group).active = target
            self._layout()
            self.call_after_refresh(self._sync_tab_actions)

    @on(TabbedContent.TabActivated)
    def content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        if self._tab_mutation_depth:
            return
        if event.tabbed_content.id not in {"main-tabs", "right-tabs"}:
            return
        group: Literal["main", "right"] = (
            "main" if event.tabbed_content.id == "main-tabs" else "right"
        )
        pane_id = event.pane.id or ""
        prefix = f"{group}-"
        key = pane_id[len(prefix):] if pane_id.startswith(prefix) else ""
        keys = self.main_keys if group == "main" else self.right_keys
        target = f"head-{group}-{key}"
        if (key in keys and self._pane_is_present(group, key)
                and self._header_tabs(group).get_tab(target) is not None):
            if self._header_tabs(group).active != target:
                self._header_tabs(group).active = target
            self._layout()
            self.call_after_refresh(self._sync_tab_actions)

    def _where(self, key: str) -> Literal["main", "right"] | None:
        if key in self.main_keys:
            return "main"
        if key in self.right_keys:
            return "right"
        return None

    def _active(self, group: Literal["main", "right"]) -> str | None:
        active = self._tabs(group).active
        prefix = group + "-"
        return active[len(prefix):] if active and active.startswith(prefix) else None

    def _capture_draft(self, key: str) -> None:
        if key == "chat":
            for view in self.query(ChatSurface):
                self.chat_draft = view.query_one(PromptBox).value
        elif key == "btw":
            for view in self.query(BTWSurface):
                prompt = view.query_one("#btw-prompt", BTWPrompt)
                self.btw_draft = prompt.value
                self.btw_prompt_height = max(3, prompt.region.height)

    async def add(self, view: View) -> None:
        if view.key in self.views:
            group = self._where(view.key)
            if group is not None:
                self._activate(group, view.key)
            return
        self._tab_mutation_depth += 1
        try:
            self.views[view.key] = view
            await self._tabs("main").add_pane(
                TabPane(view.title, self._make_view(view), id="main-" + view.key)
            )
            self.main_keys.append(view.key)
            await self._header_tabs("main").add_tab(
                ActionTab(view.title, group="main", key=view.key)
            )
            self._activate("main", view.key)
        finally:
            self._tab_mutation_depth -= 1
        if view.kind == "btw":
            self._render_btw()
        if view.kind == "inspector":
            self._render_inspector()
        self._layout()
        self._render_status()

    async def move(self, key: str, dest: Literal["main", "right"], *, responsive: bool = False) -> None:
        source = self._where(key)
        if key == "chat" and dest != "main":
            self.notify("Chat is fixed to the main workspace.")
            return
        if source is None or source == dest:
            return
        if source == "main" and len(self.main_keys) < 2:
            self.notify("At least one tab must remain in the main pane.")
            return
        if dest == "right" and self.size.width < self.MIN_DOCK_WIDTH and not responsive:
            self.notify("Not enough width for right dock; expand the terminal.")
            return

        self._capture_draft(key)
        self._tab_mutation_depth += 1
        try:
            source_keys = self.main_keys if source == "main" else self.right_keys
            dest_keys = self.main_keys if dest == "main" else self.right_keys
            remaining = [candidate for candidate in source_keys if candidate != key]
            if self._active(source) == key and remaining:
                # Select a surviving tab BEFORE removing the selected pane.
                self._activate(source, remaining[-1])

            # Mark ownership changed before awaiting Textual's removals.
            # A queued TabActivated for this key must not resolve to source.
            source_keys.remove(key)
            await self._tabs(source).remove_pane(f"{source}-{key}")
            await self._header_tabs(source).remove_tab(f"head-{source}-{key}")

            view = self.views[key]
            await self._tabs(dest).add_pane(
                TabPane(view.title, self._make_view(view), id=f"{dest}-{key}")
            )
            dest_keys.append(key)
            await self._header_tabs(dest).add_tab(
                ActionTab(view.title, group=dest, key=key)
            )
            self._activate(dest, key)
        finally:
            self._tab_mutation_depth -= 1

        if view.kind == "chat":
            self._render_chat()
        if view.kind == "btw":
            self._render_btw()
        if view.kind == "inspector":
            self._render_inspector()
        self._layout()
        self._render_status()

    async def close_view(self, group: Literal["main", "right"], key: str | None = None) -> None:
        key = key or self._active(group)
        if key is None:
            return
        if key == "chat":
            self.notify("The main chat cannot be closed.")
            return
        keys = self.main_keys if group == "main" else self.right_keys
        if key not in keys:
            return
        self._tab_mutation_depth += 1
        try:
            remaining = [candidate for candidate in keys if candidate != key]
            if self._active(group) == key and remaining:
                self._activate(group, remaining[-1])
            keys.remove(key)
            await self._tabs(group).remove_pane(f"{group}-{key}")
            await self._header_tabs(group).remove_tab(f"head-{group}-{key}")
            self.views.pop(key, None)
            if key in self.folded_keys:
                self.folded_keys.remove(key)
        finally:
            self._tab_mutation_depth -= 1
        self._layout()
        self._render_status()

    async def open_file(self, path: Path) -> None:
        path = path.resolve()
        if not path.is_file():
            return
        for item in self.views.values():
            if item.kind == "file" and item.path == path:
                location = self._where(item.key)
                if location is not None:
                    self._activate(location, item.key)
                return
        self.counter += 1
        await self.add(View(f"file-{self.counter}", path.name, "file", path))

    @on(DirectoryTree.FileSelected, "#files-tree")
    async def select_file(self, event: DirectoryTree.FileSelected) -> None:
        event.stop()
        await self.open_file(event.path)

    @on(ListView.Selected, "#sessions")
    def select_history(self, event: ListView.Selected) -> None:
        item = event.item
        if not isinstance(item, SessionItem):
            return
        self.chat_session = item.session_id
        self.chat_messages = [
            ("you", item.record.get("custom_title", "Session")),
            ("mira", "This is a simulated restored conversation. The visuals use MIRA's real chat widgets."),
        ]
        self.chat_draft = ""
        self._render_chat()
        group = self._where("chat")
        if group:
            self._activate(group, "chat")

    @on(SessionItem.PinRequested)
    def pin_chat(self, event: SessionItem.PinRequested) -> None:
        record = self.history_store.records.get(event.item.session_id)
        if record:
            record["pinned"] = not record.get("pinned", False)
            self._refresh_history()

    @on(SessionItem.MenuRequested)
    def history_menu(self, event: SessionItem.MenuRequested) -> None:
        """Open MIRA's real RIGHT-of-row, TOP-aligned action menu."""
        event.stop()
        item = event.item
        session_id = item.session_id
        row = item.region
        self.push_screen(
            HistoryActionDropdown(
                pinned=item.pinned,
                anchor_x=row.right,
                anchor_y=row.y,
            ),
            callback=lambda action: self._apply_history_action(session_id, action),
        )

    def _apply_history_action(self, session_id: str, action: str | None) -> None:
        if action is None:
            return  # Clicked away, pressed Escape, or selected Cancel.
        item = next((it for it in self.query(SessionItem)
                     if it.session_id == session_id), None)
        if item is None:
            return
        if action == "rename":
            item.begin_rename()
        elif action == "pin":
            self.pin_chat(SessionItem.PinRequested(item))
        elif action == "delete":
            # DemoStore is in-memory only; never delete a real chat.
            self.history_store.records.pop(session_id, None)
            self._refresh_history()

    @on(SessionItem.RenameSubmitted)
    def rename_history_action(self, event: SessionItem.RenameSubmitted) -> None:
        event.stop()
        record = self.history_store.records.get(event.item.session_id)
        if record is not None and event.value.strip():
            record["custom_title"] = event.value.strip()
        event.item.cancel_rename()
        self._refresh_history()

    def _open_tab_overflow(self, group: Literal["main", "right"]) -> None:
        """MAIN and RIGHT ellipses use the identical anchored dropdown."""
        keys = self.main_keys if group == "main" else self.right_keys
        choices = [
            Option(("● " if key == self._active(group) else "  ")
                   + self.views[key].title, id=f"select:{key}")
            for key in keys
        ]
        if group == "main":
            choices.extend((
                Option("＋ Open BTW", id="create:btw"),
                Option("＋ Open Inspector", id="create:inspector"),
            ))
        anchor = self.query_one(f"#{group}-overflow", Button).region
        pane = self.query_one("#primary-pane" if group == "main"
                              else "#right-pane", Vertical)
        self.push_screen(
            TabOverflowDropdown(
                group, choices, anchor.right, anchor.bottom,
                pane.region, (self.size.width, self.size.height),
            ),
            callback=lambda selected: self._on_picked_tab(group, selected),
        )

    # Production PromptBox.Submitted exposes `prompt`, not `control`.
    @on(PromptBox.Submitted)
    def chat_submitted(self, event: PromptBox.Submitted) -> None:
        msg = event.value.strip()
        if not msg:
            return
        self.chat_messages.append(("you", msg))
        self.chat_messages.append(("mira", "UI fidelity probe only. No LLM or tool was called."))
        self.chat_draft = ""
        self._render_chat()

    @on(BTWPrompt.Submitted)
    def btw_submitted(self, event: BTWPrompt.Submitted) -> None:
        msg = event.value.strip()
        if not msg:
            return
        self.btw_messages.append(("you", msg))
        self.btw_messages.append(("mira", "Mock BTW response; main chat is unchanged."))
        self.btw_draft = ""
        self._render_btw()

    @on(Inspector.Closed)
    def inspector_closed(self, event: Inspector.Closed) -> None:
        event.stop()
        group = self._where("inspector")
        if group is not None:
            self._activate(group, "inspector")
            self.run_worker(self.close_view(group), exclusive=False)

    @on(Button.Pressed, "#mcp-status-button")
    def click_mcp(self, event: Button.Pressed) -> None:
        self.notify("MCP status is a visual sample; no servers are started.")

    def mock_activate_model(self, kind: Literal["main", "btw"], alias: str) -> None:
        """Simulate successful runtime switch, then update read-only footer."""
        if alias not in self.model_aliases:
            return
        self.model_configured[kind] = alias
        self.model_effective[kind] = alias
        for composer in list(self.query(ComposerActions)):
            if composer.kind == kind:
                composer.call_after_refresh(composer.sync_model_button)
        self._render_status()
        self.notify(f"Mock {kind.upper()} runtime switched to {alias!r}.", timeout=2)

    @on(Button.Pressed, ".probe-model-trigger")
    def open_model_dropdown(self, event: Button.Pressed) -> None:
        """Open a screen-space constrained drop-UP instead of native Select."""
        event.stop()
        kind: Literal["main", "btw"] = (
            "btw" if event.button.id == "model-trigger-btw" else "main"
        )
        pane_id = "right-pane" if kind == "btw" and self._where("btw") == "right" else "primary-pane"
        pane = self.query_one(f"#{pane_id}", Vertical)
        anchor = event.button.region
        bounds = pane.region
        popup = ModelAliasPopup(
            kind, self.model_aliases, self.model_configured[kind],
            (anchor.right, anchor.y),
            (bounds.x, bounds.y, bounds.right, bounds.bottom),
            (self.size.width, self.size.height),
        )
        # Apply after the modal closes so normal MIRA widgets are mounted
        # again when the footer and composer triggers refresh.
        self.push_screen(
            popup,
            callback=lambda alias: self.mock_activate_model(kind, alias) if alias else None,
        )

    @on(Button.Pressed, ".probe-add-file")
    def click_add_file(self, event: Button.Pressed) -> None:
        # Probe-only attachment state. The production native picker will be
        # wired per-conversation during integration; no file is read here.
        kind = "btw" if event.button.id == "attach-btw" else "main"
        self.attached_mock[kind] = not self.attached_mock[kind]
        for item in self.query(f"#attached-{kind}"):
            item.update("sample.txt" if self.attached_mock[kind] else "")
        self.notify(f"Mock {'attached' if self.attached_mock[kind] else 'removed'}: {kind} sample.txt", timeout=1.3)

    @on(Button.Pressed, "#rail-chat")
    def click_history(self, event: Button.Pressed) -> None:
        self.action_history()

    @on(Button.Pressed, "#rail-files")
    def click_files(self, event: Button.Pressed) -> None:
        self.action_files()

    @on(Button.Pressed, "#rail-help")
    def click_help(self, event: Button.Pressed) -> None:
        # Production requirement: the rail wrench always enters GENERAL.
        self.push_screen(SettingsPreview())

    @on(Button.Pressed, "#collapse-sidebar")
    def click_collapse(self, event: Button.Pressed) -> None:
        self.sidebar_open = False
        self._layout()

    @on(Button.Pressed, "#new-chat")
    def click_new(self, event: Button.Pressed) -> None:
        self.action_new_chat()

    @on(TabCommand)
    def tab_command(self, event: TabCommand) -> None:
        event.stop()
        if event.command == "dock":
            target = "right" if event.group == "main" else "main"
            self.run_worker(self.move(event.key, target), exclusive=False)
        elif event.command == "close":
            self.run_worker(self.close_view(event.group, event.key), exclusive=False)

    @on(Button.Pressed, ".probe-header-scroll Button")
    def scroll_tabs(self, event: Button.Pressed) -> None:
        event.stop()
        button_id = event.button.id or ""
        group = "right" if button_id.startswith("right-") else "main"
        if button_id.endswith("-overflow"):
            self._open_tab_overflow(group)
            return
        direction = -1 if button_id.endswith("-left") else +1
        self._header_tabs(group).scroll_strip(direction)
        self.call_after_refresh(self._sync_tab_actions)

    def _on_picked_tab(self, group: str, selection: str | None) -> None:
        if not selection:
            return
        if selection.startswith("select:"):
            key = selection.partition(":")[2]
            if key in (self.main_keys if group == "main" else self.right_keys):
                self._activate(group, key)
        elif selection == "create:btw":
            self.action_btw()
        elif selection == "create:inspector":
            self.action_inspector()

    def action_cycle_chat_icon(self) -> None:
        if self.ascii_icons:
            self.notify("Icon cycling uses Unicode; restart without --ascii-icons.")
            return
        self.chat_glyph_index = (self.chat_glyph_index + 1) % len(self.CHAT_GLYPHS)
        glyph = self.CHAT_GLYPHS[self.chat_glyph_index]
        self.query_one("#rail-chat", Button).label = glyph
        self.notify(f"Chat glyph {self.chat_glyph_index + 1}/{len(self.CHAT_GLYPHS)}: {glyph!r}", timeout=1.3)

    def action_stress_tabs(self) -> None:
        """Open multiple actual repository files to exercise tab overflow."""
        self.run_worker(self._stress_tabs(), exclusive=False)

    async def _stress_tabs(self) -> None:
        paths = [
            "ui/textual/app.py", "ui/textual/styles/mira.tcss",
            "ui/textual/widgets/chat_log.py", "ui/textual/widgets/session_history.py",
            "ui/textual/widgets/status_bar.py", "ui/textual/widgets/prompt_box.py",
            "ui/textual/widgets/autocomplete_input.py", "ui/textual/widgets/inspector.py",
            "ui/textual/widgets/prompt_panel.py", "ui/textual/splash.py",
        ]
        for relative in paths:
            candidate = self.workspace / relative
            if candidate.is_file():
                await self.open_file(candidate)
        self.call_after_refresh(self._sync_tab_actions)

    def action_history(self) -> None:
        self.sidebar_open = not (self.sidebar_open and self.sidebar_kind == "history")
        self.sidebar_kind = "history"
        self._layout()

    def action_files(self) -> None:
        self.sidebar_open = not (self.sidebar_open and self.sidebar_kind == "files")
        self.sidebar_kind = "files"
        self._layout()

    def action_new_chat(self) -> None:
        self.chat_session = "new-demo"
        self.chat_messages = []
        self.chat_draft = ""
        self._render_chat()
        group = self._where("chat")
        if group:
            self._activate(group, "chat")

    def action_inspector(self) -> None:
        self.run_worker(self.add(View("inspector", "Inspector", "inspector")), exclusive=False)

    def action_btw(self) -> None:
        self.run_worker(self.add(View("btw", "BTW", "btw")), exclusive=False)

    def action_btw_modal(self) -> None:
        self.push_screen(BTWModal())

    def action_dock(self) -> None:
        key = self._active("main")
        if key:
            self.run_worker(self.move(key, "right"), exclusive=False)

    def action_undock(self) -> None:
        key = self._active("right")
        if key:
            self.run_worker(self.move(key, "main"), exclusive=False)

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def begin_divider_drag(self) -> None:
        """Suppress non-input text selection only for an active resize drag."""
        if self._selection_before_drag is None:
            self._selection_before_drag = self.ALLOW_SELECT
        self.ALLOW_SELECT = False
        self.clear_selection()

    def end_divider_drag(self) -> None:
        if self._selection_before_drag is not None:
            self.ALLOW_SELECT = self._selection_before_drag
            self._selection_before_drag = None
        self.clear_selection()

    def drag_width(self, which: str, screen_x: int) -> None:
        """Resize using the actual bordered widget positions.

        Avoid hardcoded screen origins: panels sit inside the shell's
        horizontal padding, and both resize cells have their own 1-col width.
        Always leave at least 28 cells for the main workspace.
        """
        shell = self.query_one("#app-shell", Horizontal)
        visible_sidebar = self.query_one("#session-sidebar", Vertical).display
        visible_right = self.query_one("#right-pane", Vertical).display
        min_editor_width = 28
        inner_width = max(0, shell.content_region.width)
        rail_and_gap = self.RAIL_WIDTH + 1
        if which == "sidebar":
            origin = self.query_one("#session-sidebar", Vertical).region.x
            reserved_right = self.right_width + 1 if visible_right else 0
            capacity = inner_width - rail_and_gap - 1 - reserved_right - min_editor_width
            max_sidebar = max(42, min(66, capacity))
            self.sidebar_width = max(42, min(max_sidebar, screen_x - origin))
        else:
            reserved_left = rail_and_gap + (
                self.sidebar_width + 1 if visible_sidebar else 0
            )
            capacity = inner_width - reserved_left - 1 - min_editor_width
            max_right = max(28, min(76, capacity))
            # Cursor is ON the 1-cell splitter, immediately before dock.
            requested = shell.content_region.right - screen_x - 1
            self.right_width = max(28, min(max_right, requested))
        self._layout()

    def _layout(self) -> None:
        if not self._mounted_ready:
            return
        show_sidebar = self.sidebar_open and self.size.width >= self.MIN_SIDEBAR_WIDTH
        if self.right_keys and self.size.width < 145:
            show_sidebar = False
        self.query_one("#session-sidebar", Vertical).display = show_sidebar
        self.query_one("#session-sidebar", Vertical).styles.width = self.sidebar_width
        self.query_one("#side-divider", DragDivider).display = show_sidebar
        self.query_one("#session-sidebar-title", Static).update(
            "HISTORY" if self.sidebar_kind == "history" else "EXPLORER"
        )
        self.query_one("#sidebar-content", ContentSwitcher).current = (
            "history-panel" if self.sidebar_kind == "history" else "files-panel"
        )
        self.query_one("#new-chat", Button).display = self.sidebar_kind == "history"
        for rid, kind in (("rail-chat", "history"), ("rail-files", "files")):
            self.query_one("#" + rid, Button).set_class(show_sidebar and self.sidebar_kind == kind, "selected")
        show_right = bool(self.right_keys) and self.size.width >= self.MIN_DOCK_WIDTH
        self.query_one("#right-pane", Vertical).display = show_right
        self.query_one("#right-divider", DragDivider).display = show_right
        shell = self.query_one("#app-shell", Horizontal)
        # Screen padding (1 cell each side), rail, permanent rail-to-panel
        # gap, optional sidebar+handle, and optional right dock+handle.
        rail_total = self.RAIL_WIDTH + 1
        inner_width = max(0, shell.content_region.width)
        available = inner_width - rail_total - (
            self.sidebar_width + 1 if show_sidebar else 0
        )
        self.query_one("#right-pane", Vertical).styles.width = max(
            28, min(self.right_width, available - 29)
        )
        # Per-tab Dock / Close replaces the separate top-row command buttons.
        self.call_after_refresh(self._sync_tab_actions)
        self.query_one("#mcp-status-button", Button).display = self.size.width >= 80

    def on_resize(self, _event=None) -> None:
        if not self._mounted_ready or self._reflow_busy:
            return
        self.run_worker(self._responsive(), group="responsive", exclusive=True)

    async def _responsive(self) -> None:
        self._reflow_busy = True
        try:
            if self.size.width < self.MIN_DOCK_WIDTH:
                for key in list(self.right_keys):
                    if key not in self.folded_keys:
                        self.folded_keys.append(key)
                    await self.move(key, "main", responsive=True)
            else:
                for key in list(self.folded_keys):
                    if key in self.main_keys and len(self.main_keys) > 1:
                        await self.move(key, "right", responsive=True)
                    self.folded_keys.remove(key)
        finally:
            self._reflow_busy = False
            self._layout()


async def smoke(workspace: Path, screenshots: Path | None) -> None:
    """Validate v34 geometry, single-line dropdowns, click-away menus and footer.

    Export REAL Textual SVG screenshots from the user's own MIRA checkout.
    No extra fake frame rows are rendered at any terminal size.
    """
    if screenshots:
        screenshots.mkdir(parents=True, exist_ok=True)
    for width, height in ((145, 42), (102, 32), (78, 27)):
        app = MiraFidelityProbe(workspace)
        async with app.run_test(size=(width, height)) as pilot:
            await pilot.pause()
            assert app.main_keys == ["chat"] and not app.right_keys
            status = app.query_one("#status-row", Horizontal)
            shell = app.query_one("#app-shell", Horizontal)
            footer = app.query_one("#telemetry-row", ProbeTelemetryBar)
            rail = app.query_one("#navigation-rail", Vertical)
            rail_gap = app.query_one("#rail-panel-gap", Static)
            sidebar = app.query_one("#session-sidebar", Vertical)
            primary = app.query_one("#primary-pane", Vertical)
            side_handle = app.query_one("#side-divider", DragDivider)
            right_handle = app.query_one("#right-divider", DragDivider)
            right = app.query_one("#right-pane", Vertical)
            assert status.region.y == 0 and status.region.height == 1
            assert status.region.width == width, "Header must span the entire terminal"
            assert footer.region.width == width and footer.region.height == 1
            assert footer.region.bottom == height, "Footer must reach screen bottom"
            assert app.sidebar_width == 42 and app.right_width == 42
            assert app.query_one("#probe-model-main", Static) is not None
            assert not app.query_one("#probe-model-btw", Static).display
            assert not app.query_one("#probe-model-separator", Static).display
            assert app.query_one("#model-trigger-main", Button) is not None
            assert not list(app.query("#model-settings-button, #add-file-button")), (
                "Footer must have no clickable model or Add file buttons"
            )
            chat_composer = app.query_one("#chat-composer", Vertical)
            assert chat_composer.query_one("#attach-main", Button)
            assert chat_composer.query_one("#model-trigger-main", Button) is not None
            assert chat_composer.query_one("#actions-main", ComposerActions)
            assert "max-width: 43%" not in MiraFidelityProbe.CSS, (
                "Model telemetry must not truncate aliases against auto width"
            )
            assert footer.query_one("#probe-effective-models", Horizontal).region.width > 0
            if width >= 145:
                label = footer.query_one("#probe-model-main", Static)
                assert label.region.width >= len("main: " + app.model_effective["main"]), (
                    "Main alias truncated despite ample footer width"
                )
                assert footer.query_one("#telemetry-values", Static).region.right <= footer.region.right
            assert "width: 1fr; max-width: 55%" not in MiraFidelityProbe.CSS, (
                "The composer trigger must not expand toward the centre"
            )
            assert status.region.bottom == shell.region.y, "Top gutter accidentally added"
            assert shell.region.bottom == footer.region.y, "Bottom gutter accidentally added"
            assert not list(app.query("#frame-top, #frame-bottom, #frame-gap-top, #frame-gap-bottom"))
            assert rail.region.width == app.RAIL_WIDTH == 9, "Rail must provide nine cells"
            assert rail.region.y == primary.region.y == shell.region.y
            assert rail.region.bottom == primary.region.bottom == shell.region.bottom
            assert rail_gap.region.width == 1 and rail_gap.region.x == rail.region.right
            assert primary.region.x == rail_gap.region.right, (
                "Initial Chat should follow separately bordered rail with one dark cell"
            )
            assert not sidebar.display and not side_handle.display
            assert not right.display and not right_handle.display
            assert app._header_tabs("main").active == "head-main-chat"
            assert app._header_tabs("main").has_class("accent-chat")
            chat_icon = app.query_one("#rail-chat", Button)
            files_icon = app.query_one("#rail-files", Button)
            assert files_icon.region.y == chat_icon.region.bottom, "Unexpected icon gap"
            assert chat_icon.region.x == rail.region.x + 1, "Rail's left border missing"
            assert chat_icon.region.right == rail.region.right - 1, "Rail's right border missing"
            assert chat_icon.region.width == rail.content_region.width == 7, (
                "Rail button was widened by Textual's default min-width: 16; icon will be clipped"
            )
            assert files_icon.region.width == rail.content_region.width
            assert chat_icon.render().plain.strip(), "Chat icon label must not be empty"
            assert files_icon.render().plain.strip(), "Files icon label must not be empty"
            assert chat_icon.region.y == rail.region.y + 1, "Rail's top border missing"
            assert len(app.CHAT_GLYPHS[0]) == 1
            assert all("\ufe0f" not in c for c in app.CHAT_GLYPHS)
            if screenshots and width == 145:
                app.export_screenshot(path=str(screenshots / "01-mira-startup.svg"))

            app.action_history()
            await pilot.pause()
            if width >= app.MIN_SIDEBAR_WIDTH:
                assert sidebar.display and side_handle.display
                assert sidebar.region.y == primary.region.y == rail.region.y
                assert sidebar.region.bottom == primary.region.bottom == rail.region.bottom
                assert sidebar.region.x == rail_gap.region.right, (
                    "History must have a separately bordered panel after the dark rail gap"
                )
                assert side_handle.region.width == 1
                assert side_handle.region.x == sidebar.region.right
                assert primary.region.x == side_handle.region.right
                assert side_handle.region.y == shell.region.y
                assert side_handle.region.bottom == shell.region.bottom
                # Invisible by default: no border, no fake glyph, matched dark background.
                assert side_handle.render() == ""
                assert app.sidebar_width >= 42, "History min width is 42"
                header = app.query_one("#session-sidebar-header", Horizontal)
                assert header.styles.padding.left == header.styles.padding.right == 1, (
                    "History/Explorer heading needs one-cell side gutters"
                )
                history = app.query_one("#sessions", SessionHistory)
                assert history.styles.padding.left == history.styles.padding.right == 2, (
                    "History list needs one additional side cell over v23"
                )
                assert history.region.y > sidebar.region.y, "Sidebar top border/header missing"
                assert history.region.x >= sidebar.region.x + 1
                assert history.region.right <= sidebar.region.right - 1
            if width == 145:
                first_row = next(iter(app.query(SessionItem)))
                app.history_menu(SessionItem.MenuRequested(first_row))
                await pilot.pause()
                assert isinstance(app.screen, HistoryActionDropdown)
                history_dropdown = app.screen.query_one("#session-action-popup", Vertical)
                assert history_dropdown.region.x == min(
                    first_row.region.right, width - HistoryActionDropdown.POPUP_WIDTH
                ), "History menu must open right of row like current production MIRA"
                assert history_dropdown.region.y == min(
                    first_row.region.y, height - HistoryActionDropdown.POPUP_HEIGHT
                ), "History menu must align with the History row top"
                if screenshots:
                    app.export_screenshot(path=str(screenshots / "02-mira-history-menu.svg"))
                # Clicking OUTSIDE closes without invoking Rename/Pin/Delete.
                await pilot.click(offset=(0, 0))
                await pilot.pause()
                assert not isinstance(app.screen, HistoryActionDropdown), (
                    "History menu must dismiss when clicking outside"
                )
            if screenshots and width == 145:
                app.export_screenshot(path=str(screenshots / "02-mira-history.svg"))

            app.action_files()
            await pilot.pause()
            if width >= app.MIN_SIDEBAR_WIDTH:
                files_tree = app.query_one("#files-tree", DirectoryTree)
                assert files_tree.styles.padding.left == files_tree.styles.padding.right == 2, (
                    "File Explorer tree should match the History list inset"
                )
            if screenshots and width == 145:
                app.export_screenshot(path=str(screenshots / "03-mira-files.svg"))
            file_path = workspace / "ui" / "textual" / "app.py"
            await app.open_file(file_path if file_path.is_file() else Path(__file__))
            await pilot.pause()
            assert app._header_tabs("main").has_class("accent-file"), (
                "An active file tab must glow blue"
            )
            if screenshots and width == 145:
                app.export_screenshot(path=str(screenshots / "04-mira-file-preview.svg"))

            await app.add(View("inspector", "Inspector", "inspector"))
            await pilot.pause()
            assert app._header_tabs("main").has_class("accent-inspector"), (
                "Inspector must glow purple"
            )
            inspector = app.query_one(Inspector)
            inspector_header = inspector.query_one("#inspector-header", Horizontal)
            assert not inspector_header.display, (
                "Inspector's redundant title/close row must be hidden without occupying space"
            )
            assert inspector.query_one("#inspector-title", Static), (
                "Keep real Inspector title widget mounted for dynamic updates"
            )
            assert inspector.query_one("#inspector-log", ChatLog).region.y == inspector.region.y, (
                "Inspector transcript should begin at the top, without a hidden-header gap"
            )
            if width >= app.MIN_DOCK_WIDTH:
                await app.move("inspector", "right")
                await pilot.pause()
                assert right.display and right_handle.display
                assert right_handle.region.width == 1
                assert right_handle.region.x == primary.region.right
                assert right.region.x == right_handle.region.right
                assert right.region.y == primary.region.y == rail.region.y
                assert right.region.bottom == primary.region.bottom == rail.region.bottom
                assert right_handle.render() == ""
                if width == 145:
                    # BOTH main and dock use the same dropdown class/style.
                    for group, owner in (("main", primary), ("right", right)):
                        app._open_tab_overflow(group)
                        await pilot.pause()
                        assert isinstance(app.screen, TabOverflowDropdown)
                        dropdown = app.screen
                        popup = dropdown.query_one("#tab-overflow-options", OptionList)
                        # Long filenames must be truncated, never wrapped to
                        # extra visual rows (which overflowed the small menu).
                        assert all(
                            "\n" not in str(option.prompt)
                            and cell_len(str(option.prompt)) <= dropdown.label_columns
                            for option in dropdown.options
                        ), f"{group}: tab menu option wraps beyond one row"
                        assert all(option.id for option in dropdown.options)
                        assert TabOverflowDropdown._single_line_label(
                            "ARCHITECTURE_DECISIONS.md", 18
                        ).endswith("…"), "long filenames should have an ellipsis"
                        assert popup.region.x >= owner.region.x
                        assert popup.region.right <= owner.region.right
                        assert popup.region.y >= owner.region.y
                        assert popup.region.bottom <= owner.region.bottom
                        if screenshots:
                            app.export_screenshot(path=str(
                                screenshots / f"05-mira-{group}-dropdown.svg"
                            ))
                        await pilot.click(offset=(0, 0))
                        await pilot.pause()
                        assert not isinstance(app.screen, TabOverflowDropdown), (
                            f"{group} tab dropdown must dismiss on outside click"
                        )
                assert app._header_tabs("right").has_class("accent-inspector"), (
                    "Docked Inspector must remain purple"
                )
                inspector = app.query_one(Inspector)
                assert not inspector.query_one("#inspector-header", Horizontal).display, (
                    "Inspector title/close row reappeared after docking"
                )
                assert inspector.query_one("#inspector-log", ChatLog).region.y == inspector.region.y, (
                    "Docked Inspector transcript should start immediately below the tab bar"
                )
                if width == 145:
                    # Show every independently bordered panel at once.
                    app.action_history()
                    await pilot.pause()
                    assert sidebar.display and side_handle.display and right.display
                    assert (rail.region.y, sidebar.region.y, primary.region.y, right.region.y) == (
                        shell.region.y,
                    ) * 4, "Four independent panel borders must align along top"
                    assert (rail.region.bottom, sidebar.region.bottom, primary.region.bottom,
                            right.region.bottom) == (shell.region.bottom,) * 4, (
                        "Four independent panel borders must align along bottom"
                    )
                    if screenshots:
                        app.export_screenshot(path=str(screenshots / "05-mira-four-panels.svg"))
            await app.add(View("btw", "BTW", "btw"))
            await pilot.pause()
            assert app.query_one("#probe-model-btw", Static).display, (
                "BTW telemetry should appear whenever BTW is open"
            )
            assert app.query_one("#probe-model-separator", Static).display
            assert app.query_one("#btw-composer", Vertical).query_one("#attach-btw", Button)
            assert app.query_one("#btw-composer", Vertical).query_one("#model-trigger-btw", Button)
            changed = app.model_aliases[-1]
            app.mock_activate_model("btw", changed)
            await pilot.pause()
            assert app.model_effective["btw"] == changed
            assert app.query_one("#probe-model-btw", Static) is not None

            assert app._header_tabs("main").has_class("accent-btw"), (
                "BTW must glow coral pink"
            )
            if width >= app.MIN_DOCK_WIDTH:
                await app.move("btw", "right")
                await pilot.pause()
                assert app._header_tabs("right").has_class("accent-btw")
                if width == 145:
                    btw = app.query_one(BTWSurface)
                    prompt = btw.query_one(BTWPrompt)
                    log = btw.query_one("#btw-log", ChatLog)
                    assert (log.region.x, log.region.right) == (
                        prompt.region.x, prompt.region.right
                    ), "BTW prompt and transcript horizontal edges differ"
                    assert btw.query_one(BTWInput).safe_prompt_height > prompt.region.height
                    old = prompt.region.height
                    assert old == 3, "BTW prompt should show two input rows"
                    assert btw.query_one("#actions-btw", ComposerActions).region.height == 2
                    prompt.styles.height = old + 2
                    app.btw_prompt_height = old + 2
                    await pilot.pause()
                    assert prompt.region.height > old, "BTW prompt resize failed"
                    if screenshots:
                        app.export_screenshot(path=str(screenshots / "06-mira-btw-docked.svg"))
                    # Test an anchored dropdown at the dock's MIN width.
                    # Popup must open UPWARD, fit within the dock, and never
                    # extend below the footer or beyond the terminal edge.
                    app.right_width = 28
                    app._layout()
                    await pilot.pause()
                    narrow_right = app.query_one("#right-pane", Vertical)
                    anchor = app.query_one("#model-trigger-btw", Button).region
                    bounds = narrow_right.region
                    menu = ModelAliasPopup(
                        "btw", app.model_aliases, app.model_configured["btw"],
                        (anchor.right, anchor.y),
                        (bounds.x, bounds.y, bounds.right, bounds.bottom),
                        (app.size.width, app.size.height),
                    )
                    assert menu.menu_x >= bounds.x and menu.menu_x + menu.menu_width <= bounds.right
                    assert menu.menu_y + menu.menu_height <= anchor.y, (
                        "Model menu must open above the bottom-pinned BTW prompt"
                    )
                    app.push_screen(menu)
                    await pilot.pause()
                    if screenshots:
                        app.export_screenshot(path=str(screenshots / "07-mira-btw-min-width-dropdown.svg"))
                    menu.dismiss(None)
                    await pilot.pause()
                    app.right_width = 42
                    app._layout()
                    await pilot.pause()
                    await app.move("btw", "main")
                    await pilot.pause()
                    assert app.query_one(BTWPrompt).region.height == app.btw_prompt_height, (
                        "Undocking lost BTW prompt height"
                    )
                    assert changed in str(app.query_one("#btw-composer", Vertical).query_one("#model-trigger-btw", Button).tooltip), (
                        "Undocking lost BTW model selection"
                    )
                    await app.move("btw", "right")
                    await pilot.pause()
            if width == 145:
                # The Chat prompt uses MIRA's own resize widget.
                app._activate("main", "chat")
                await pilot.pause()
                surface = app.query_one(ChatSurface)
                log = surface.query_one("#chat-log", ChatLog)
                prompt_shell = surface.query_one(ProbeAutocompleteInput)
                prompt = prompt_shell.query_one(PromptBox)
                assert (log.region.x, log.region.right) == (
                    prompt.region.x, prompt.region.right
                ), "Original Chat transcript/prompt alignment regressed"
                assert prompt_shell.safe_prompt_height > prompt.region.height
                # Compact composer: PromptBox 3 rows (top border + 2 input
                # rows); action block 2 rows (controls + bottom border).
                assert prompt.region.height == 3, "Chat prompt should show two input rows"
                assert surface.query_one("#actions-main", ComposerActions).region.height == 2
                previous = app.ALLOW_SELECT
                app.begin_divider_drag()
                assert not app.ALLOW_SELECT
                app.end_divider_drag()
                assert app.ALLOW_SELECT == previous
                # Verify both splitters resize the desired panels without
                # moving the neighboring panel borders out of alignment.
                old_sidebar = app.sidebar_width
                app.drag_width("sidebar", side_handle.region.x + 2)
                await pilot.pause()
                assert app.sidebar_width > old_sidebar
                old_dock = app.right_width
                app.drag_width("right", right_handle.region.x - 2)
                await pilot.pause()
                assert app.right_width > old_dock
                assert right.region.bottom == shell.region.bottom
                await app._stress_tabs()
                await pilot.pause()
                assert len(list(app._header_tabs("main").query(ActionTab))) >= 8
                assert app._header_tabs("main").query_one("#tabs-scroll").max_scroll_x > 0
                # v32: the left/right buttons are DISABLED based solely on
                # scroll position; the enabled sibling must not gain a
                # persistent selected/hover class at either end.
                for group in ("main", "right"):
                    scroller = app._header_tabs(group).query_one("#tabs-scroll")
                    left = app.query_one(f"#{group}-scroll-left", Button)
                    right_arrow = app.query_one(f"#{group}-scroll-right", Button)
                    scroller.scroll_to(x=0, animate=False, force=True)
                    await pilot.pause()
                    app._sync_tab_actions()
                    assert left.disabled, f"{group}: left arrow enabled at start"
                    assert right_arrow.disabled == (scroller.max_scroll_x <= 0), (
                        f"{group}: right arrow boundary handling changed"
                    )
                    assert not right_arrow.has_class("selected"), (
                        f"{group}: opposite arrow must not be selected at start"
                    )
                    if scroller.max_scroll_x > 0:
                        scroller.scroll_to(x=scroller.max_scroll_x, animate=False, force=True)
                        await pilot.pause()
                        app._sync_tab_actions()
                        assert right_arrow.disabled, f"{group}: right arrow enabled at end"
                        assert not left.disabled, f"{group}: left arrow disabled at end"
                        assert not left.has_class("selected"), (
                            f"{group}: opposite arrow must not be selected at end"
                        )
                # v33 regression: docking the LAST visible main tab used to
                # deliver a stale header activation (e.g. file-10) AFTER its
                # TabPane was removed and crash with "No Tab with id".
                last_main_key = app.main_keys[-1]
                assert last_main_key != "chat"
                app._activate("main", last_main_key)
                await pilot.pause()
                assert app._active("main") == last_main_key
                await app.move(last_main_key, "right")
                await pilot.pause()
                assert last_main_key not in app.main_keys
                assert last_main_key in app.right_keys
                assert not app._pane_is_present("main", last_main_key)
                assert app._pane_is_present("right", last_main_key)
                surviving = app._active("main")
                assert surviving in app.main_keys and surviving != last_main_key
                assert app._header_tabs("main").active == f"head-main-{surviving}"
                assert app._active("right") == last_main_key
                assert app._header_tabs("right").active == f"head-right-{last_main_key}"
                assert app._tab_mutation_depth == 0
                # Reversing the move must not reawaken stale activation events.
                await app.move(last_main_key, "main")
                await pilot.pause()
                assert app._active("main") == last_main_key
                assert app._header_tabs("main").active == f"head-main-{last_main_key}"
                assert app._tab_mutation_depth == 0
                # Closing the selected last tab takes the same protected path.
                await app.close_view("main", last_main_key)
                await pilot.pause()
                assert last_main_key not in app.main_keys
                assert not app._pane_is_present("main", last_main_key)
                assert app._active("main") in app.main_keys
                assert app._header_tabs("main").active == f"head-main-{app._active('main')}"
                assert app._tab_mutation_depth == 0
                if screenshots:
                    app.export_screenshot(path=str(screenshots / "07-mira-tab-overflow.svg"))
            if screenshots and width == 78:
                app.export_screenshot(path=str(screenshots / "08-mira-narrow.svg"))
            assert app._where("chat") == "main", "Chat must stay in main workspace"
            chat_header = next(tab for tab in app.query(ActionTab) if tab.key == "chat")
            assert "›" not in chat_header.label_text and "×" not in chat_header.label_text
            await app.move("chat", "right")
            assert app._where("chat") == "main", "move() must block Chat docking"
            assert "chat" in app.main_keys
            # Close BTW: its model label and divider MUST disappear, even if
            # another view (such as Inspector) remains docked.
            btw_group = app._where("btw")
            if btw_group is not None:
                await app.close_view(btw_group, "btw")
                await pilot.pause()
                assert not app.query_one("#probe-model-btw", Static).display
                assert not app.query_one("#probe-model-separator", Static).display
            print(f"PASS v33 {width}x{height}")
    if screenshots:
        print("Real Textual-rendered SVG frames:", screenshots.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--ascii-icons", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--screenshots", type=Path)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    if workspace != ROOT:
        parser.error("Run from the MIRA repo root with --workspace . (this is intentional).")
    # Fast regression guard for the malformed CSS declaration in v25.
    assert "line-pad: 0;" not in MiraFidelityProbe.CSS, "Invalid Textual 8.2.7 CSS"
    if args.self_test or args.screenshots:
        asyncio.run(smoke(workspace, args.screenshots))
    else:
        MiraFidelityProbe(workspace, ascii_icons=args.ascii_icons).run()


if __name__ == "__main__":
    main()
