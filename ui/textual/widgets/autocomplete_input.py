"""Slash-command and local-file autocomplete prompt surface."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.events import Key, MouseDown, MouseMove, MouseUp, Resize
from textual.widgets import OptionList, Static, TextArea
from textual.widgets.option_list import Option

from ui.textual.commands.help import command_help_entries, command_insertion
from ui.textual.widgets.prompt_box import PromptBox


MAX_COMPLETIONS = 5
MIN_PROMPT_HEIGHT = 5
FILE_DISCOVERY_CONCURRENCY = 32
POPUP_BORDER_HEIGHT = 2
EXCLUDED_FILE_COMPONENTS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "build",
    "dist",
}


@dataclass(frozen=True)
class CompletionItem:
    """The source-neutral data needed to render and insert one completion."""

    kind: Literal["tool", "subagent", "file", "mcp_resource", "native_command", "prompt_command", "status"]
    display: str
    insertion: str
    description: str = ""
    selectable: bool = True
    metadata: Any = None


@dataclass(frozen=True)
class _CompletionFragment:
    kind: Literal["command", "file"]
    start: int
    end: int
    query: str


class PromptResizeHandle(Static):
    """Drag target that redraws the prompt's top border."""

    ALLOW_SELECT = False

    def __init__(self) -> None:
        super().__init__("", id="prompt-resize-handle")
        self._drag_start_y: int | None = None
        self._prompt_start_height: int | None = None

    def render(self) -> str:
        width = self.size.width
        grip = "[=]"
        if width < len(grip):
            return "━" * width
        left = (width - len(grip)) // 2
        return "━" * left + grip + "━" * (width - left - len(grip))

    def on_mouse_down(self, event: MouseDown) -> None:
        if event.button != 1:
            return
        prompt = self.parent.query_one(PromptBox)
        self._drag_start_y = event.screen_y
        self._prompt_start_height = prompt.region.height
        self.capture_mouse()
        event.stop()
        event.prevent_default()

    def on_mouse_move(self, event: MouseMove) -> None:
        if self._drag_start_y is None or self._prompt_start_height is None:
            return
        delta = self._drag_start_y - event.screen_y
        requested_height = self._prompt_start_height + delta
        autocomplete = self.parent
        prompt = autocomplete.query_one(PromptBox)
        new_height = min(
            autocomplete.safe_prompt_height,
            max(MIN_PROMPT_HEIGHT, requested_height),
        )
        prompt.styles.height = new_height
        event.stop()
        event.prevent_default()

    def on_mouse_up(self, event: MouseUp) -> None:
        if self._drag_start_y is None or event.button != 1:
            return
        self._drag_start_y = None
        self._prompt_start_height = None
        self.release_mouse()
        event.stop()
        event.prevent_default()


class AutocompleteInput(Vertical):
    """PromptBox wrapper with one native popup for commands and files."""

    def __init__(
        self,
        *,
        project_backend: Any = None,
        tool_provider: Callable[[], list[dict[str, str]]] | None = None,
        subagent_provider: Callable[[], list[dict[str, str]]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(id="autocomplete-input", **kwargs)
        self.project_backend = project_backend
        self.tool_provider = tool_provider
        self.subagent_provider = subagent_provider
        self._items: list[CompletionItem] = []
        self._file_paths: list[str] | None = None
        self._fragment: _CompletionFragment | None = None
        self._interaction_start: int | None = None
        self._dismissed_start: int | None = None
        self._generation = 0
        self._file_worker: Any = None
        self._resource_worker: Any = None
        self._prompt_worker: Any = None
        self._mcp_resources: list[Any] = []
        self.mcp_manager: Any = None

    @property
    def active(self) -> bool:
        """Return whether completion choices are currently visible."""
        return bool(self._items) and self.query_one(OptionList).display

    @property
    def items(self) -> tuple[CompletionItem, ...]:
        """Expose the displayed completion items for focused UI tests."""
        return tuple(self._items)

    def compose(self) -> ComposeResult:
        yield OptionList(id="autocomplete-options", compact=True)
        yield PromptBox()
        yield PromptResizeHandle()

    def on_mount(self) -> None:
        self.query_one(OptionList).display = False
        self.screen.screen_layout_refresh_signal.subscribe(
            self,
            self._shrink_prompt_to_fit,
            immediate=True,
        )
        self.call_after_refresh(self._sync_resize_handle_width)

    def on_resize(self, _event: Resize) -> None:
        self.call_after_refresh(self._sync_resize_handle_width)

    def _sync_resize_handle_width(self) -> None:
        """Keep the border handle inset from the prompt's two corners."""
        width = self.query_one(PromptBox).region.width
        if width:
            self.query_one(PromptResizeHandle).styles.width = max(1, width - 2)

    @property
    def safe_prompt_height(self) -> int:
        """Return the live prompt ceiling while preserving the main layout."""
        prompt = self.query_one(PromptBox)
        try:
            chat = self.screen.query_one("#chat-log")
            main_panel = self.screen.query_one("#main-panel")
            telemetry = self.screen.query_one("#telemetry-row")
        except NoMatches:
            return max(MIN_PROMPT_HEIGHT, prompt.region.height)

        bottom_space = main_panel.content_region.bottom - telemetry.region.bottom
        reclaimable_chat = chat.content_region.height - 1
        return max(
            MIN_PROMPT_HEIGHT,
            prompt.region.height + reclaimable_chat + bottom_space,
        )

    def _shrink_prompt_to_fit(self, _screen: Any) -> None:
        """Shrink an oversized prompt after layout changes, without auto-growing."""
        prompt = self.query_one(PromptBox)
        safe_height = self.safe_prompt_height
        if prompt.region.height > safe_height:
            prompt.styles.height = safe_height

    def set_project_backend(self, backend: Any) -> None:
        """Switch discovery to a rebuilt project backend and clear old state."""
        self.dismiss()
        self.project_backend = backend

    def set_mcp_manager(self, manager: Any) -> None:
        """Use the manager's canonical prompt and resource registries."""
        self.dismiss()
        self.mcp_manager = manager

    def prompt_disabled_changed(self, disabled: bool) -> None:
        """Mirror prompt availability in the handle without disabling resizing."""
        self.query_one(PromptResizeHandle).set_class(disabled, "-prompt-disabled")
        if disabled:
            self.dismiss()

    def dismiss(self, *, suppress_current: bool = False) -> None:
        """Close the popup and invalidate any pending discovery result."""
        if suppress_current and self._fragment is not None:
            self._dismissed_start = self._fragment.start
        self._cancel_file_worker()
        self._generation += 1
        self._items = []
        self._file_paths = None
        self._fragment = None
        self._interaction_start = None
        if self.is_mounted:
            options = self.query_one(OptionList)
            options.clear_options()
            options.display = False
            self.query_one(PromptResizeHandle).styles.offset = (1, 0)

    def handle_prompt_key(self, event: Key) -> bool:
        """Route popup controls while leaving keyboard focus on PromptBox."""
        if event.key == "escape" and self._fragment is not None:
            event.stop()
            event.prevent_default()
            self.dismiss(suppress_current=True)
            return True
        if not self.active:
            return False
        options = self.query_one(OptionList)
        if event.key == "up":
            event.stop()
            event.prevent_default()
            options.action_cursor_up()
            return True
        if event.key == "down":
            event.stop()
            event.prevent_default()
            options.action_cursor_down()
            return True
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self._accept(options.highlighted)
            return True
        return False

    @on(TextArea.Changed, "#prompt")
    def prompt_changed(self, _event: TextArea.Changed) -> None:
        self._refresh_from_prompt()

    @on(TextArea.SelectionChanged, "#prompt")
    def prompt_selection_changed(self, _event: TextArea.SelectionChanged) -> None:
        self._refresh_from_prompt()

    @on(OptionList.OptionSelected, "#autocomplete-options")
    def option_selected(self, event: OptionList.OptionSelected) -> None:
        """Accept native mouse selection and return focus to the prompt."""
        event.stop()
        self._accept(event.option_index)

    def _refresh_from_prompt(self) -> None:
        prompt = self.query_one(PromptBox)
        if prompt.disabled:
            self.dismiss()
            return
        if prompt.displaying_untouched_history_entry:
            self.dismiss()
            return
        cursor = _offset_from_location(prompt.value, prompt.cursor_location)
        fragment = completion_fragment(prompt.value, cursor)
        if fragment is None:
            self._dismissed_start = None
            self.dismiss()
            return
        if self._dismissed_start == fragment.start:
            self._hide_options()
            return
        if fragment.kind == "command":
            self._cancel_file_worker()
            self._generation += 1
            self._file_paths = None
            self._interaction_start = None
            self._fragment = fragment
            self._show_items(command_items(fragment.query, self._prompt_registry()))
            if self.mcp_manager is not None:
                generation = self._generation
                self._prompt_worker = self.run_worker(
                    self._load_mcp_prompts(generation, fragment.start),
                    name=f"autocomplete-prompts-{generation}",
                    exclusive=False,
                )
            return

        same_interaction = self._interaction_start == fragment.start
        self._fragment = fragment
        if same_interaction:
            if self._file_paths is not None:
                self._show_items(
                    attachment_items(
                        self._file_paths,
                        self._mcp_resources,
                        fragment.query,
                        self._resource_errors(),
                        tools=self._active_tools(),
                        subagents=self._active_subagents(),
                    )
                )
            return

        self._cancel_file_worker()
        self._generation += 1
        self._interaction_start = fragment.start
        self._file_paths = None
        self._mcp_resources = []
        generation = self._generation
        if self.project_backend is not None:
            self._file_worker = self.run_worker(
                self._load_project_files(generation, fragment.start),
                name=f"autocomplete-files-{generation}",
                exclusive=False,
            )
        else:
            self._file_paths = []
        if self.mcp_manager is not None:
            self._mcp_resources = list(self.mcp_manager.resource_registry.values())
            self._resource_worker = self.run_worker(
                self._load_mcp_resources(generation, fragment.start),
                name=f"autocomplete-resources-{generation}",
                exclusive=False,
            )
        self._show_items(
            attachment_items(
                [],
                self._mcp_resources,
                fragment.query,
                self._resource_errors(),
                tools=self._active_tools(),
                subagents=self._active_subagents(),
            )
        )

    async def _load_project_files(self, generation: int, interaction_start: int) -> None:
        try:
            paths = await discover_project_files(self.project_backend)
        except Exception:
            return
        if generation != self._generation or interaction_start != self._interaction_start:
            return
        self._file_worker = None
        self._file_paths = paths
        fragment = self._fragment
        if fragment is None or fragment.kind != "file" or fragment.start != interaction_start:
            return
        self._show_items(
            attachment_items(
                self._file_paths,
                self._mcp_resources,
                fragment.query,
                self._resource_errors(),
                tools=self._active_tools(),
                subagents=self._active_subagents(),
            )
        )

    async def _load_mcp_resources(self, generation: int, interaction_start: int) -> None:
        await self.mcp_manager.discover_resources()
        if generation != self._generation or interaction_start != self._interaction_start:
            return
        self._resource_worker = None
        self._mcp_resources = list(self.mcp_manager.resource_registry.values())
        fragment = self._fragment
        if fragment is not None and fragment.kind == "file" and fragment.start == interaction_start:
            self._show_items(
                attachment_items(
                    self._file_paths or [],
                    self._mcp_resources,
                    fragment.query,
                    self._resource_errors(),
                    tools=self._active_tools(),
                    subagents=self._active_subagents(),
                )
            )

    async def _load_mcp_prompts(self, generation: int, interaction_start: int) -> None:
        await self.mcp_manager.discover_prompts()
        if generation != self._generation:
            return
        self._prompt_worker = None
        fragment = self._fragment
        if fragment is not None and fragment.kind == "command" and fragment.start == interaction_start:
            self._show_items(command_items(fragment.query, self._prompt_registry()))

    def _prompt_registry(self) -> Any:
        return self.mcp_manager.prompt_registry if self.mcp_manager is not None else None

    def _resource_errors(self) -> list[str]:
        return self.mcp_manager.resource_errors() if self.mcp_manager is not None else []

    def _active_tools(self) -> list[dict[str, str]]:
        """Read the current agent tool projection without caching mode state."""
        return self.tool_provider() if self.tool_provider is not None else []

    def _active_subagents(self) -> list[dict[str, str]]:
        """Read effective enabled subagents without caching reload state."""
        return self.subagent_provider() if self.subagent_provider is not None else []

    def _show_items(self, items: list[CompletionItem]) -> None:
        statuses = [item for item in items if not item.selectable]
        selectable = [item for item in items if item.selectable]
        if statuses:
            self._items = [*selectable[: MAX_COMPLETIONS - 1], statuses[0]]
        else:
            self._items = selectable[:MAX_COMPLETIONS]
        options = self.query_one(OptionList)
        options.set_options([Option(_completion_row(item), disabled=not item.selectable) for item in self._items])
        options.display = bool(self._items)
        options_height = 0
        if self._items:
            options.highlighted = 0
            options_height = min(len(self._items), MAX_COMPLETIONS) + POPUP_BORDER_HEIGHT
            options.styles.height = options_height
        self.query_one(PromptResizeHandle).styles.offset = (1, options_height)

    def _hide_options(self) -> None:
        self._items = []
        if self.is_mounted:
            options = self.query_one(OptionList)
            options.clear_options()
            options.display = False
            self.query_one(PromptResizeHandle).styles.offset = (1, 0)

    def _accept(self, index: int | None) -> None:
        fragment = self._fragment
        if fragment is None or index is None or not (0 <= index < len(self._items)):
            return
        item = self._items[index]
        if not item.selectable:
            return
        prompt = self.query_one(PromptBox)
        start = _location_from_offset(prompt.value, fragment.start)
        end = _location_from_offset(prompt.value, fragment.end)
        prompt.replace(item.insertion, start, end)
        self.dismiss(suppress_current=True)
        self.call_after_refresh(prompt.focus)

    def _cancel_file_worker(self) -> None:
        for attribute in ("_file_worker", "_resource_worker", "_prompt_worker"):
            worker = getattr(self, attribute)
            if worker is not None:
                worker.cancel()
                setattr(self, attribute, None)


def command_items(query: str, prompt_registry: Any = None) -> list[CompletionItem]:
    """Merge native and prompt command matches while preserving their kinds."""
    items = [*native_command_items(query), *prompt_command_items(query, prompt_registry)]
    return sorted(items, key=lambda item: (item.display.casefold(), item.display))


def native_command_items(query: str) -> list[CompletionItem]:
    """Return literal substring matches from the shared native command registry."""
    folded = query.casefold()
    return [
        CompletionItem(
            kind="native_command",
            display=usage,
            insertion=command_insertion(usage),
            description=description,
        )
        for usage, description in command_help_entries()
        if folded in usage.casefold()
    ]


def prompt_command_items(query: str, prompt_registry: Any = None) -> list[CompletionItem]:
    """Return literal substring matches from the canonical prompt registry."""
    if prompt_registry is None:
        return []
    folded = query.casefold()
    specs = getattr(prompt_registry, "specs", {})
    return [
        CompletionItem(
            kind="prompt_command",
            display=spec.usage,
            insertion=command_insertion(spec.usage),
            description=spec.description,
            metadata=spec,
        )
        for spec in specs.values()
        if folded in spec.usage.casefold()
    ]


def file_items(paths: list[str], query: str) -> list[CompletionItem]:
    """Return alphabetized literal substring matches for one file interaction."""
    folded = query.casefold()
    matches = sorted(
        (path for path in paths if folded in path.casefold()),
        key=lambda path: (path.casefold(), path),
    )[:MAX_COMPLETIONS]
    return [
        CompletionItem(
            kind="file",
            display=path,
            insertion=f'@"{path}"' if " " in path else f"@{path}",
            description="local file",
        )
        for path in matches
    ]


def attachment_items(
    paths: list[str],
    resources: list[Any],
    query: str,
    errors: list[str] | None = None,
    *,
    tools: list[dict[str, str]] | None = None,
    subagents: list[dict[str, str]] | None = None,
) -> list[CompletionItem]:
    """Merge files, resources, tools, and enabled subagents before limiting."""
    folded = query.casefold()
    local_matches = sorted(
        (path for path in paths if folded in path.casefold()),
        key=lambda path: (path.casefold(), path),
    )
    items = [
        CompletionItem(
            kind="file",
            display=path,
            insertion=f'@"{path}"' if " " in path else f"@{path}",
            description="local file",
        )
        for path in local_matches
    ]
    for resource in resources:
        if folded not in str(resource.token).casefold() and folded not in str(resource.name).casefold():
            continue
        insertion = f'@"{resource.token}"' if any(char.isspace() for char in resource.token) else f"@{resource.token}"
        items.append(
            CompletionItem(
                kind="mcp_resource",
                display=resource.token,
                insertion=insertion,
                description=resource.description or resource.name,
                metadata=resource,
            )
        )
    for tool in tools or []:
        name = str(tool.get("name") or "")
        if not name or folded not in name.casefold():
            continue
        items.append(
            CompletionItem(
                kind="tool",
                display=name,
                insertion=name,
                description=str(tool.get("description") or ""),
                metadata=tool,
            )
        )
    for subagent in subagents or []:
        name = str(subagent.get("name") or "")
        if not name or folded not in name.casefold():
            continue
        items.append(
            CompletionItem(
                kind="subagent",
                display=f"{name} subagent",
                insertion=f"{name} subagent",
                description=str(subagent.get("description") or ""),
                metadata=subagent,
            )
        )
    items.sort(key=lambda item: (item.display.casefold(), item.display))
    for error in errors or []:
        items.append(CompletionItem("status", f"MCP resources unavailable: {error}", "", selectable=False))
    return items


async def discover_project_files(backend: Any) -> list[str]:
    """Enumerate project files through backend ls calls while pruning noisy trees."""
    pending = deque(["/"])
    visited: set[str] = set()
    files: set[str] = set()
    while pending:
        directories: list[str] = []
        while pending and len(directories) < FILE_DISCOVERY_CONCURRENCY:
            directory = pending.popleft()
            if directory in visited:
                continue
            visited.add(directory)
            directories.append(directory)
        if not directories:
            continue

        results = await asyncio.gather(*(backend.als(directory) for directory in directories))
        for result in results:
            for entry in getattr(result, "entries", None) or []:
                if not isinstance(entry, dict):
                    continue
                path = str(entry.get("path") or "").replace("\\", "/")
                relative = path.strip("/")
                if not relative or _excluded_project_path(relative):
                    continue
                if entry.get("is_dir"):
                    pending.append(f"/{relative}")
                else:
                    files.add(relative)
    return sorted(files, key=lambda path: (path.casefold(), path))


def _excluded_project_path(path: str) -> bool:
    components = {component.casefold() for component in path.split("/")}
    return bool(components & EXCLUDED_FILE_COMPONENTS)


def completion_fragment(text: str, cursor: int) -> _CompletionFragment | None:
    """Locate the active command token or @ fragment at the cursor."""
    if text.startswith("/") and not any(character.isspace() for character in text):
        if 1 <= cursor <= len(text):
            return _CompletionFragment("command", 0, len(text), text[1:cursor])

    prefix = text[:cursor]
    at = prefix.rfind("@")
    if at < 0 or any(character.isspace() for character in prefix[at + 1 :]):
        return None
    return _CompletionFragment("file", at, cursor, prefix[at + 1 :])


def _completion_row(item: CompletionItem) -> Text:
    row = Text(no_wrap=True, overflow="ellipsis")
    if not item.selectable:
        row.append(item.display, style="#b8c1c7")
        return row
    labels = {
        "tool": ("TOOL", "#78d5cf"),
        "subagent": ("SUBA", "#93A4C3"),
        "mcp_resource": ("RSRC", "#c7a0e8"),
        "file": ("FILE", "#aeb8be"),
        "native_command": ("CMND", "#d2a957"),
        "prompt_command": ("PRMT", "#8fb9e8"),
    }
    label, color = labels[item.kind]
    row.append(label, style=f"bold {color}")
    row.append("  ")
    row.append(item.display, style="#e8edef")
    if item.description:
        row.append(f"  {item.description}", style="#b8c1c7")
    return row


def _offset_from_location(text: str, location: tuple[int, int]) -> int:
    row, column = location
    lines = text.splitlines(keepends=True)
    return sum(len(line) for line in lines[:row]) + column


def _location_from_offset(text: str, offset: int) -> tuple[int, int]:
    before = text[:offset]
    row = before.count("\n")
    column = len(before.rsplit("\n", 1)[-1])
    return row, column


__all__ = [
    "AutocompleteInput",
    "CompletionItem",
    "attachment_items",
    "command_items",
    "completion_fragment",
    "discover_project_files",
    "file_items",
    "native_command_items",
    "prompt_command_items",
]
