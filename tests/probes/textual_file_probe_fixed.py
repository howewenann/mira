"""
MIRA Textual file-input probe.

Tests:
1. Multi-file drag/drop from Explorer into the Textual input.
2. Multi-file Explorer Ctrl+C -> Ctrl+V into the Textual input.
3. [Add file] -> native Windows multi-file picker.

Run:
    pip install textual
    python tests/probes/textual_file_probe_fixed.py

Recommended:
- Test in Windows Terminal.
- Repeat in VS Code integrated terminal.
- Include filenames with spaces.
- Select multiple mixed file types.

The probe does NOT upload into a real MIRA backend. For [Add file], it inserts
fake MIRA-style references such as:

    @/.mira_uploads/probe/report.pdf
"""

from __future__ import annotations

import asyncio
import ctypes
import os
import shlex
import time
from pathlib import Path
from typing import Iterable

from textual import events
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Header, RichLog, Static, TextArea


CF_HDROP = 15


def get_windows_clipboard_files() -> list[Path]:
    """Return every file currently present in Windows CF_HDROP.

    Explicit Win32 ctypes signatures are required on 64-bit Python.
    Without them, ctypes assumes c_int return values and can truncate
    pointer-sized clipboard handles returned by GetClipboardData().
    """
    if os.name != "nt":
        return []

    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)

    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL

    user32.CloseClipboard.argtypes = []
    user32.CloseClipboard.restype = wintypes.BOOL

    user32.IsClipboardFormatAvailable.argtypes = [wintypes.UINT]
    user32.IsClipboardFormatAvailable.restype = wintypes.BOOL

    user32.GetClipboardData.argtypes = [wintypes.UINT]
    user32.GetClipboardData.restype = wintypes.HANDLE

    shell32.DragQueryFileW.argtypes = [
        wintypes.HANDLE,
        wintypes.UINT,
        wintypes.LPWSTR,
        wintypes.UINT,
    ]
    shell32.DragQueryFileW.restype = wintypes.UINT

    if not user32.OpenClipboard(None):
        return []

    try:
        if not user32.IsClipboardFormatAvailable(CF_HDROP):
            return []

        hdrop = user32.GetClipboardData(CF_HDROP)
        if not hdrop:
            return []

        count = shell32.DragQueryFileW(hdrop, 0xFFFFFFFF, None, 0)
        files: list[Path] = []

        for index in range(count):
            length = shell32.DragQueryFileW(hdrop, index, None, 0)
            if length == 0:
                continue

            buffer = ctypes.create_unicode_buffer(length + 1)
            copied = shell32.DragQueryFileW(
                hdrop,
                index,
                buffer,
                length + 1,
            )
            if copied:
                files.append(Path(buffer.value))

        return files
    finally:
        user32.CloseClipboard()


def choose_files(initial_dir: Path) -> list[Path]:
    """Open a Windows multi-file picker using only the Python stdlib."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    try:
        selected = filedialog.askopenfilenames(
            title="Add files to MIRA",
            initialdir=str(initial_dir),
        )
        return [Path(p) for p in selected]
    finally:
        root.destroy()


def parse_existing_files_from_text(text: str) -> list[Path]:
    """Best-effort parser for one or more pasted/dropped path strings."""
    text = text.strip()
    if not text:
        return []

    try:
        tokens = shlex.split(text, posix=False)
    except ValueError:
        tokens = [text]

    files: list[Path] = []

    for token in tokens:
        token = token.strip().strip('"').strip("'")
        if not token:
            continue

        path = Path(token)

        try:
            resolved = path.resolve()
        except OSError:
            continue

        if resolved.exists() and resolved.is_file():
            files.append(resolved)

    return files


def fake_backend_refs(paths: Iterable[Path]) -> str:
    return "\n".join(
        f"@/.mira_uploads/probe/{path.name}"
        for path in paths
    )


class ProbeTextArea(TextArea):
    """TextArea that logs Paste vs Key delivery."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_key_time: float | None = None

    def _log(self, message: str) -> None:
        if isinstance(self.app, FileProbeApp):
            self.app.log_event(message)

    async def _on_paste(self, event: events.Paste) -> None:
        self._log(
            f"PASTE len={len(event.text)} text={event.text!r}"
        )

        clipboard_files = get_windows_clipboard_files()
        if clipboard_files:
            self._log(
                "CF_HDROP "
                f"count={len(clipboard_files)} "
                f"files={[str(p) for p in clipboard_files]!r}"
            )

        parsed = await asyncio.to_thread(
            parse_existing_files_from_text,
            event.text,
        )
        if parsed:
            self._log(
                "PARSED_PASTE_FILES "
                f"count={len(parsed)} "
                f"files={[str(p) for p in parsed]!r}"
            )

        await super()._on_paste(event)

    async def _on_key(self, event: events.Key) -> None:
        now = time.perf_counter()

        delta_ms: float | None = None
        if self._last_key_time is not None:
            delta_ms = (now - self._last_key_time) * 1000

        self._last_key_time = now

        if event.is_printable or event.key in {"enter", "space"}:
            delta = "first" if delta_ms is None else f"{delta_ms:.1f}ms"
            self._log(
                f"KEY key={event.key!r} "
                f"char={event.character!r} "
                f"dt={delta}"
            )

        await super()._on_key(event)


class FileProbeApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }

    #title {
        height: auto;
        padding: 0 1;
    }

    #composer {
        height: 10;
        border: round $accent;
        margin: 0 1;
    }

    #chat-input {
        height: 1fr;
    }

    #controls {
        height: 3;
        padding: 0 1;
    }

    #add-file {
        margin-right: 1;
    }

    #log {
        height: 1fr;
        border: round $secondary;
        margin: 0 1 1 1;
    }
    """

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+l", "clear_log", "Clear log"),
        ("ctrl+h", "inspect_clipboard", "Inspect CF_HDROP"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()

        yield Static(
            "MIRA file-input probe | "
            "multi-file drag/drop, Explorer Ctrl+C/Ctrl+V, Add file",
            id="title",
        )

        with Vertical(id="composer"):
            yield ProbeTextArea(id="chat-input")

            with Horizontal(id="controls"):
                yield Button("Add file", id="add-file")
                yield Button("Model: probe", id="model-button", disabled=True)

        yield RichLog(
            id="log",
            highlight=True,
            markup=False,
            wrap=True,
        )

        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#chat-input", ProbeTextArea).focus()
        self.log_event(f"PROJECT_DIR {Path.cwd()}")
        self.log_event(f"OS {os.name}")
        self.log_event(
            "READY: drag multiple files, Ctrl+C/Ctrl+V multiple Explorer files, "
            "or click Add file"
        )

    def log_event(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.query_one("#log", RichLog).write(f"[{timestamp}] {message}")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "add-file":
            return

        self.log_event("ADD_FILE_OPEN start")

        paths = await asyncio.to_thread(
            choose_files,
            Path.cwd(),
        )

        text_area = self.query_one("#chat-input", ProbeTextArea)

        if not paths:
            self.log_event("ADD_FILE_OPEN cancelled")
            text_area.focus()
            return

        self.log_event(
            "ADD_FILE_RESULT "
            f"count={len(paths)} "
            f"files={[str(p) for p in paths]!r}"
        )

        refs = fake_backend_refs(paths)

        if text_area.text and not text_area.text.endswith(("\n", " ")):
            text_area.insert("\n")

        text_area.insert(refs)
        text_area.focus()

        self.log_event(
            f"FAKE_UPLOAD_RESULT refs={refs!r}"
        )

    def action_clear_log(self) -> None:
        self.query_one("#log", RichLog).clear()
        self.log_event("LOG_CLEARED")

    def action_inspect_clipboard(self) -> None:
        files = get_windows_clipboard_files()

        if files:
            self.log_event(
                "MANUAL_CF_HDROP "
                f"count={len(files)} "
                f"files={[str(p) for p in files]!r}"
            )
        else:
            self.log_event("MANUAL_CF_HDROP empty")


if __name__ == "__main__":
    FileProbeApp().run()
