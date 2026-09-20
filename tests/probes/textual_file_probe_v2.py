"""
MIRA file-upload probe v2.

This probes the complete intended flow:

1. Explorer multi-file Ctrl+C -> Ctrl+V
2. Explorer multi-file drag/drop into the Textual input
3. [Add file] native Windows multi-file picker

All three converge to:

    list[Path]
        ->
    DeepAgents FilesystemBackend.aupload_files(...)
        ->
    /.mira_uploads/probe/<session>/<filename>
        ->
    backend.adownload_files(...) byte-for-byte verification
        ->
    insert @/.mira_uploads/... references into the composer

The physical files should appear under:

    <project>/.mira_uploads/probe/<session>/

Dependencies:
    textual
    deepagents

Run from the MIRA project directory:

    python tests/probes/textual_file_probe_v2.py

Recommended tests:
    - Add file: select 3 files
    - Explorer: select 3 files -> Ctrl+C -> Ctrl+V
    - Explorer: select 3 files -> drag/drop into the input
    - include at least one filename containing spaces
    - include mixed extensions (.md, .jpg, .pdf, .zip, etc.)

The event log should show:
    INPUT ...
    UPLOAD ...
    VERIFY OK ...
    INSERTED_REFS ...
"""

from __future__ import annotations

import asyncio
import ctypes
import hashlib
import os
import re
import shlex
import time
from pathlib import Path
from typing import Iterable

from deepagents.backends import FilesystemBackend
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Header, RichLog, Static, TextArea


CF_HDROP = 15

PASTE_BURST_CHAR_GAP_SECONDS = 0.03
PASTE_BURST_FLUSH_SECONDS = 0.08
PASTE_BURST_MIN_CHARS = 3

WINDOWS_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:[\\/]")


# ---------------------------------------------------------------------------
# Windows clipboard
# ---------------------------------------------------------------------------

def get_windows_clipboard_files() -> list[Path]:
    """Return every file currently present in Windows CF_HDROP."""
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


# ---------------------------------------------------------------------------
# Native file picker
# ---------------------------------------------------------------------------

def choose_files(initial_dir: Path) -> list[Path]:
    """Open a Windows multi-file picker using the Python stdlib."""
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


# ---------------------------------------------------------------------------
# Dropped-path parsing
# ---------------------------------------------------------------------------

def looks_like_dropped_path_prefix(text: str) -> bool:
    """Pure string check used to promote rapid key streams into one payload."""
    value = text.strip().lstrip("<'\"")

    return bool(
        WINDOWS_DRIVE_PREFIX.match(value)
        or value.startswith(("\\\\", "/", "~/", "file://"))
    )


def parse_existing_files_from_text(text: str) -> list[Path]:
    """Parse a complete terminal payload as one or more existing files.

    This is intentionally strict: every token must resolve to an existing file.

    On Windows we use shlex(posix=False) so backslashes are preserved:
        D:\\foo\\a.txt D:\\foo\\b.txt

    Quoted filenames with spaces are also supported.
    """
    payload = text.strip()
    if not payload:
        return []

    tokens: list[str] = []

    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        try:
            if os.name == "nt":
                line_tokens = shlex.split(line, posix=False)
            else:
                line_tokens = shlex.split(line, posix=True)
        except ValueError:
            return []

        if not line_tokens:
            return []

        tokens.extend(line_tokens)

    if not tokens:
        return []

    files: list[Path] = []

    for token in tokens:
        value = token.strip().strip('"').strip("'")
        if not value:
            return []

        if value.startswith("file://"):
            # Minimal Windows file:// support for the probe.
            value = value.removeprefix("file:///").removeprefix("file://")
            value = value.replace("/", os.sep)

        path = Path(value).expanduser()

        try:
            resolved = path.resolve()
        except OSError:
            return []

        if not resolved.exists() or not resolved.is_file():
            return []

        files.append(resolved)

    return files


# ---------------------------------------------------------------------------
# Backend upload helpers
# ---------------------------------------------------------------------------

def read_file_bytes(paths: list[Path]) -> list[bytes]:
    return [path.read_bytes() for path in paths]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def mention_for_backend_path(path: str) -> str:
    """Render a backend path using the existing @file-style convention.

    Spaces are escaped so the reference remains one token for @ parsers that
    use backslash escaping.
    """
    return "@" + path.replace(" ", r"\ ")


# ---------------------------------------------------------------------------
# Textual input
# ---------------------------------------------------------------------------

class ProbeTextArea(TextArea):
    """TextArea with dcode-style rapid-paste reconstruction for file drops."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._last_key_time: float | None = None
        self._rapid_run_text = ""

        self._burst_buffer = ""
        self._burst_generation = 0

    def _log(self, message: str) -> None:
        if isinstance(self.app, FileProbeApp):
            self.app.log_event(message)

    # ----- normal/bracketed paste -----------------------------------------

    async def _on_paste(self, event: events.Paste) -> None:
        self._log(
            f"PASTE len={len(event.text)} text={event.text!r}"
        )

        # Explorer Ctrl+C -> Ctrl+V.
        #
        # In the observed VS Code terminal this arrives as Paste(text=""),
        # while the real file list remains available in CF_HDROP.
        clipboard_files = await asyncio.to_thread(get_windows_clipboard_files)

        if clipboard_files:
            self._log(
                "CF_HDROP "
                f"count={len(clipboard_files)} "
                f"files={[str(p) for p in clipboard_files]!r}"
            )

            event.prevent_default()
            event.stop()

            await self.app.handle_input_files(
                clipboard_files,
                source="clipboard",
            )
            return

        # Some terminals may paste/drop actual path text in one Paste event.
        parsed = await asyncio.to_thread(
            parse_existing_files_from_text,
            event.text,
        )

        if parsed:
            self._log(
                "PASTE_PATHS "
                f"count={len(parsed)} "
                f"files={[str(p) for p in parsed]!r}"
            )

            event.prevent_default()
            event.stop()

            await self.app.handle_input_files(
                parsed,
                source="paste-paths",
            )
            return

        # Ordinary text paste.
        await super()._on_paste(event)

    # ----- dcode-style rapid key burst ------------------------------------

    async def _on_key(self, event: events.Key) -> None:
        now = time.perf_counter()

        delta: float | None = None
        if self._last_key_time is not None:
            delta = now - self._last_key_time

        self._last_key_time = now

        # Once a path-shaped rapid run has been promoted, keep all following
        # printable characters in the hidden buffer until the terminal goes idle.
        if self._burst_buffer:
            if event.is_printable and event.character is not None:
                self._burst_buffer += event.character
                self._schedule_burst_flush()

                event.prevent_default()
                event.stop()
                return

            if event.key == "enter":
                self._burst_buffer += "\n"
                self._schedule_burst_flush()

                event.prevent_default()
                event.stop()
                return

            # A non-printable key ends the terminal-generated burst first.
            await self._flush_burst()

        if event.is_printable and event.character is not None:
            character = event.character

            if delta is not None and delta <= PASTE_BURST_CHAR_GAP_SECONDS:
                self._rapid_run_text += character
            else:
                self._rapid_run_text = character

            delta_text = "first" if delta is None else f"{delta * 1000:.1f}ms"
            self._log(
                f"KEY key={event.key!r} "
                f"char={character!r} "
                f"dt={delta_text}"
            )

            # Let Textual insert the character normally first. If the run is
            # subsequently confirmed as a dropped path, we remove that already
            # inserted prefix and transfer ownership to _burst_buffer.
            await super()._on_key(event)

            if (
                len(self._rapid_run_text) >= PASTE_BURST_MIN_CHARS
                and looks_like_dropped_path_prefix(self._rapid_run_text)
            ):
                payload = self._rapid_run_text

                if self._remove_recent_inserted_text(payload):
                    self._burst_buffer = payload
                    self._rapid_run_text = ""

                    self._log(
                        f"BURST_PROMOTED prefix={payload!r}"
                    )

                    self._schedule_burst_flush()

            return

        # Non-printable human interaction resets unpromoted rapid-run tracking.
        self._rapid_run_text = ""
        await super()._on_key(event)

    def _remove_recent_inserted_text(self, payload: str) -> bool:
        """Remove a just-inserted rapid run immediately before the cursor."""
        if not payload or not self.selection.is_empty:
            return False

        cursor = self.cursor_location
        cursor_offset = self.document.get_index_from_location(cursor)
        start_offset = cursor_offset - len(payload)

        if start_offset < 0:
            return False

        if self.text[start_offset:cursor_offset] != payload:
            return False

        start = self.document.get_location_from_index(start_offset)
        self.delete(start, cursor)
        return True

    def _schedule_burst_flush(self) -> None:
        self._burst_generation += 1
        generation = self._burst_generation
        asyncio.create_task(self._flush_after_idle(generation))

    async def _flush_after_idle(self, generation: int) -> None:
        await asyncio.sleep(PASTE_BURST_FLUSH_SECONDS)

        if generation != self._burst_generation:
            return

        if self._burst_buffer:
            await self._flush_burst()

    async def _flush_burst(self) -> None:
        payload = self._burst_buffer
        self._burst_buffer = ""
        self._burst_generation += 1

        if not payload:
            return

        self._log(
            f"BURST_FLUSH len={len(payload)} text={payload!r}"
        )

        parsed = await asyncio.to_thread(
            parse_existing_files_from_text,
            payload,
        )

        if not parsed:
            self._log(
                "BURST_NOT_FILES -> reinserting payload verbatim"
            )
            self.insert(payload)
            return

        self._log(
            "BURST_PATHS "
            f"count={len(parsed)} "
            f"files={[str(p) for p in parsed]!r}"
        )

        await self.app.handle_input_files(
            parsed,
            source="drag-drop-burst",
        )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

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
        height: 11;
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

    def __init__(self) -> None:
        super().__init__()

        self.project_dir = Path.cwd().resolve()

        # This is the actual DeepAgents backend we are probing.
        self.backend = FilesystemBackend(
            root_dir=self.project_dir,
            virtual_mode=True,
        )

        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        self.upload_root = f"/.mira_uploads/probe/{self.session_id}"

        # Prevent same-name collisions across repeated upload actions.
        self._used_backend_paths: set[str] = set()

    def compose(self) -> ComposeResult:
        yield Header()

        yield Static(
            "MIRA file-upload probe | "
            "drag/drop, Explorer Ctrl+C/Ctrl+V, Add file",
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

        physical_upload_dir = (
            self.project_dir
            / ".mira_uploads"
            / "probe"
            / self.session_id
        )

        self.log_event(f"PROJECT_DIR {self.project_dir}")
        self.log_event(
            "BACKEND FilesystemBackend "
            f"root={self.project_dir} virtual_mode=True"
        )
        self.log_event(f"UPLOAD_ROOT {self.upload_root}")
        self.log_event(f"PHYSICAL_UPLOAD_DIR {physical_upload_dir}")
        self.log_event(
            "READY: test Add file, multi-file Ctrl+C/Ctrl+V, "
            "and multi-file drag/drop"
        )

    def log_event(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.query_one("#log", RichLog).write(
            f"[{timestamp}] {message}"
        )

    # ----- common convergence point ---------------------------------------

    async def handle_input_files(
        self,
        paths: Iterable[Path],
        *,
        source: str,
    ) -> None:
        """Upload local source files into the backend and insert @ refs."""
        source_paths = [Path(p).resolve() for p in paths]

        if not source_paths:
            return

        self.log_event(
            f"INPUT source={source} count={len(source_paths)} "
            f"files={[str(p) for p in source_paths]!r}"
        )

        backend_paths = [
            self._allocate_backend_path(path)
            for path in source_paths
        ]

        try:
            contents = await asyncio.to_thread(
                read_file_bytes,
                source_paths,
            )
        except Exception as exc:
            self.log_event(
                f"READ_SOURCE_FAILED {type(exc).__name__}: {exc}"
            )
            return

        upload_items = list(zip(backend_paths, contents, strict=True))

        self.log_event(
            "UPLOAD_START "
            f"count={len(upload_items)} "
            f"destinations={backend_paths!r}"
        )

        try:
            responses = await self.backend.aupload_files(upload_items)
        except Exception as exc:
            self.log_event(
                f"UPLOAD_CALL_FAILED {type(exc).__name__}: {exc}"
            )
            return

        successful_paths: list[str] = []
        expected_by_path = dict(zip(backend_paths, contents, strict=True))

        for response in responses:
            if response.error is not None:
                self.log_event(
                    f"UPLOAD_FAILED path={response.path!r} "
                    f"error={response.error!r}"
                )
                continue

            successful_paths.append(response.path)
            self.log_event(
                f"UPLOAD_OK path={response.path!r}"
            )

        if not successful_paths:
            self.log_event("NO_SUCCESSFUL_UPLOADS")
            return

        # Verify through the backend API rather than checking host files directly.
        try:
            downloads = await self.backend.adownload_files(
                successful_paths
            )
        except Exception as exc:
            self.log_event(
                f"VERIFY_CALL_FAILED {type(exc).__name__}: {exc}"
            )
            return

        verified_paths: list[str] = []

        for download in downloads:
            if download.error is not None:
                self.log_event(
                    f"VERIFY_FAILED path={download.path!r} "
                    f"error={download.error!r}"
                )
                continue

            expected = expected_by_path[download.path]
            actual = download.content

            if actual != expected:
                self.log_event(
                    f"VERIFY_MISMATCH path={download.path!r} "
                    f"expected_sha256={sha256_bytes(expected)} "
                    f"actual_sha256={sha256_bytes(actual or b'')}"
                )
                continue

            verified_paths.append(download.path)

            physical = (
                self.project_dir
                / download.path.lstrip("/").replace("/", os.sep)
            )

            self.log_event(
                f"VERIFY_OK path={download.path!r} "
                f"bytes={len(expected)} "
                f"sha256={sha256_bytes(expected)[:12]} "
                f"physical={str(physical)!r}"
            )

        if not verified_paths:
            self.log_event("NO_VERIFIED_UPLOADS")
            return

        self._insert_backend_refs(verified_paths)

    def _allocate_backend_path(self, source_path: Path) -> str:
        """Allocate a collision-safe backend path preserving the filename."""
        name = source_path.name
        stem = source_path.stem
        suffix = source_path.suffix

        candidate = f"{self.upload_root}/{name}"
        counter = 2

        while candidate in self._used_backend_paths:
            name = f"{stem}_{counter}{suffix}"
            candidate = f"{self.upload_root}/{name}"
            counter += 1

        self._used_backend_paths.add(candidate)
        return candidate

    def _insert_backend_refs(self, backend_paths: list[str]) -> None:
        text_area = self.query_one("#chat-input", ProbeTextArea)

        refs = [
            mention_for_backend_path(path)
            for path in backend_paths
        ]

        text = "\n".join(refs)

        if text_area.text and not text_area.text.endswith(("\n", " ")):
            text_area.insert("\n")

        text_area.insert(text)
        text_area.focus()

        self.log_event(
            f"INSERTED_REFS refs={refs!r}"
        )

    # ----- Add file --------------------------------------------------------

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "add-file":
            return

        self.log_event("ADD_FILE_OPEN")

        paths = await asyncio.to_thread(
            choose_files,
            self.project_dir,
        )

        text_area = self.query_one("#chat-input", ProbeTextArea)

        if not paths:
            self.log_event("ADD_FILE_CANCELLED")
            text_area.focus()
            return

        await self.handle_input_files(
            paths,
            source="add-file",
        )

        text_area.focus()

    # ----- utilities -------------------------------------------------------

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
