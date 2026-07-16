"""Textual Windows driver that preserves Shift+Enter console records."""

from __future__ import annotations

import asyncio
from asyncio import AbstractEventLoop, run_coroutine_threadsafe
from ctypes import byref, wintypes
from threading import Event, Thread
from typing import TYPE_CHECKING, Callable

from textual import constants
from textual._xterm_parser import XTermParser
from textual.drivers import win32
from textual.drivers._writer_thread import WriterThread
from textual.drivers.windows_driver import WindowsDriver
from textual.events import Event as TextualEvent
from textual.events import Resize
from textual.geometry import Size

from ui.windows_input import normalize_windows_key_record

if TYPE_CHECKING:
    from textual.app import App


class MiraWindowsEventMonitor(Thread):
    """Read Win32 input records before Textual discards their modifier state."""

    def __init__(
        self,
        loop: AbstractEventLoop,
        app: App,
        exit_event: Event,
        process_event: Callable[[TextualEvent], None],
    ) -> None:
        self.loop = loop
        self.app = app
        self.exit_event = exit_event
        self.process_event = process_event
        super().__init__(name="textual-input")

    def run(self) -> None:
        """Read console records and feed normalized text to Textual's parser."""
        exit_requested = self.exit_event.is_set
        parser = XTermParser(debug=constants.DEBUG)

        try:
            read_count = wintypes.DWORD(0)
            input_handle = win32.GetStdHandle(win32.STD_INPUT_HANDLE)
            max_events = 1024
            key_event_type = 0x0001
            window_size_event_type = 0x0004
            input_records = (win32.INPUT_RECORD * max_events)()
            read_console_input = win32.KERNEL32.ReadConsoleInputW
            keys: list[str] = []

            while not exit_requested():
                for event in parser.tick():
                    self.process_event(event)

                if win32.wait_for_handles([input_handle], 100) is None:
                    continue

                read_console_input(
                    input_handle,
                    byref(input_records),
                    max_events,
                    byref(read_count),
                )
                read_input_records = input_records[: read_count.value]

                keys.clear()
                new_size: tuple[int, int] | None = None
                for input_record in read_input_records:
                    if input_record.EventType == key_event_type:
                        key_event = input_record.Event.KeyEvent
                        key_text = normalize_windows_key_record(
                            key_down=bool(key_event.bKeyDown),
                            repeat_count=int(key_event.wRepeatCount),
                            virtual_key_code=int(key_event.wVirtualKeyCode),
                            unicode_character=key_event.uChar.UnicodeChar,
                            control_key_state=int(key_event.dwControlKeyState),
                        )
                        if key_text:
                            keys.append(key_text)
                    elif input_record.EventType == window_size_event_type:
                        size = input_record.Event.WindowBufferSizeEvent.dwSize
                        new_size = (size.X, size.Y)

                if keys:
                    # Match Textual's surrogate-pair preserving conversion.
                    normalized_input = (
                        "".join(keys).encode("utf-16", "surrogatepass").decode("utf-16")
                    )
                    for event in parser.feed(normalized_input):
                        self.process_event(event)

                if new_size is not None:
                    self.on_size_change(*new_size)
        except Exception as error:
            self.app.log.error("EVENT MONITOR ERROR", error)

    def on_size_change(self, width: int, height: int) -> None:
        """Send a Textual resize event from the input thread."""
        size = Size(width, height)
        event = Resize(size, size)
        run_coroutine_threadsafe(self.app._post_message(event), loop=self.loop)


class MiraWindowsDriver(WindowsDriver):
    """Use MIRA's raw-record monitor with Textual's Windows display driver."""

    def start_application_mode(self) -> None:
        """Start Textual application mode with MIRA's input monitor."""
        loop = asyncio.get_running_loop()
        self._restore_console = win32.enable_application_mode()
        self._writer_thread = WriterThread(self._file)
        self._writer_thread.start()

        self.write("\x1b[?1049h")
        self._enable_mouse_support()
        self.write("\x1b[?25l")
        self.write("\x1b[?1004h")
        self.write("\x1b[>1u")
        self.flush()
        self._enable_bracketed_paste()

        self._event_thread = MiraWindowsEventMonitor(
            loop,
            self._app,
            self.exit_event,
            self.process_message,
        )
        self._event_thread.start()
