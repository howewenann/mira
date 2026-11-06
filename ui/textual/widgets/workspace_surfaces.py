"""File previews, the integrated composer controls, and the future BTW surface."""

from __future__ import annotations

from pathlib import Path

from rich.cells import cell_len, set_cell_size
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Resize, MouseDown, MouseMove, MouseUp, MouseRelease
from textual.css.query import NoMatches
from textual.widgets import Button, Markdown, Static, TextArea


class ComposerActions(Horizontal):
    """A flexible spacer pins the alias selector to the prompt's right edge."""

    def __init__(self, kind: str = "main"):
        super().__init__(classes="workspace-composer-actions", id=f"actions-{kind}")
        self.kind = kind
        self.alias = "unset"

    def compose(self) -> ComposeResult:
        yield Button("+ Add file", id="add-file-button" if self.kind == "main" else "attach-btw",
                     classes="workspace-add-file", disabled=self.kind == "btw")
        yield Static("", classes="workspace-attached-name")
        yield Button("unset ▾", id=f"model-trigger-{self.kind}",
                     classes="workspace-model-trigger", disabled=self.kind == "btw")

    def on_mount(self) -> None:
        self.call_after_refresh(self.sync_model_button)

    def on_resize(self, _event: Resize) -> None:
        self.call_after_refresh(self.sync_model_button)

    def set_alias(self, alias: str | None) -> None:
        self.alias = alias or "unset"
        self.sync_model_button()

    def sync_model_button(self) -> None:
        if not self.is_mounted:
            return
        try:
            button = self.query_one(".workspace-model-trigger", Button)
            add = self.query_one(".workspace-add-file", Button)
        except NoMatches:
            # A resize callback can outlive a tab being closed this frame.
            return
        maximum = max(5, self.content_region.width - (add.region.width or 13) - 1)
        desired = min(cell_len(self.alias) + 5, maximum)
        button.styles.width = desired
        budget = max(1, desired - 4)
        shown = self.alias if cell_len(self.alias) <= budget else set_cell_size(self.alias, max(0, budget - 1)).rstrip() + "…"
        button.label = f"{shown} ▾"
        button.tooltip = f"Select model: {self.alias}" if self.kind == "main" else "BTW model selection is not available yet"


class ChatSurface(Vertical):
    """Layout marker for the existing production transcript and prompt widgets."""


class BTWResizeHandle(Static):
    """Full-width resize grip for the future BTW composer."""
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
        prompt = self.parent.query_one(TextArea)
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
        prompt = input_shell.query_one(TextArea)
        prompt.styles.height = target
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
    def __init__(self):
        super().__init__(id="btw-input-shell", classes="workspace-composer")

    def compose(self) -> ComposeResult:
        yield TextArea("", id="btw-prompt", disabled=True)
        yield ComposerActions("btw")
        yield BTWResizeHandle()

    def on_mount(self) -> None:
        self.screen.screen_layout_refresh_signal.subscribe(self, self.sync_geometry)
        self.call_after_refresh(self.sync_geometry)

    @property
    def safe_prompt_height(self) -> int:
        return max(3, self.parent.content_region.height - 5)

    def sync_geometry(self, _screen=None) -> None:
        if not self.is_mounted or not self.query(TextArea):
            return
        prompt = self.query_one(TextArea)
        self.query_one(BTWResizeHandle).styles.width = max(1, prompt.region.width - 2)
        if prompt.region.height > self.safe_prompt_height:
            prompt.styles.height = self.safe_prompt_height


class BTWSurface(Vertical):
    """UI only: no agent, messages, checkpoint, uploads, or selected model."""

    def compose(self) -> ComposeResult:
        yield Static(
            "BTW is coming in a future release.\n\n"
            "Independent conversations are not available yet. "
            "The composer, attachments, and model selection are disabled.",
            id="btw-placeholder", markup=False,
        )
        yield BTWInput()


class FileSurface(Vertical):
    """Bounded read-only preview; the DirectoryTree remains the navigation owner."""

    def __init__(self, path: Path):
        super().__init__(classes="workspace-file")
        self.path = path

    def compose(self) -> ComposeResult:
        yield Static(f" {self.path.name}  ·  READ ONLY", classes="workspace-file-heading", markup=False)
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
            yield Static(str(error), classes="workspace-file-error", markup=False)
            return
        if self.path.suffix.lower() in {".md", ".markdown"}:
            with VerticalScroll(classes="workspace-markdown-scroll"):
                yield Markdown(content)
        else:
            yield TextArea(content, read_only=True, show_line_numbers=True, soft_wrap=False,
                           classes="workspace-code", tab_behavior="focus")
