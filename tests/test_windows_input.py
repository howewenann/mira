"""Tests for MIRA's Windows input-record normalization."""

from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from textual._xterm_parser import XTermParser

from ui.textual.platform.windows.input import (
    SHIFT_ENTER_SEQUENCE,
    SHIFT_PRESSED,
    VK_RETURN,
    driver_class_for_platform,
    normalize_windows_key_record,
)
from ui.textual.platform.windows import clipboard as native_clipboard


class FakeNativeFunction:
    """Callable ctypes stand-in that accepts signature attributes."""

    def __init__(self, callback: object) -> None:
        self.callback = callback
        self.argtypes: object = None
        self.restype: object = None
        self.calls: list[tuple[object, ...]] = []

    def __call__(self, *args: object) -> object:
        self.calls.append(args)
        return self.callback(*args)  # type: ignore[operator]


def normalize(
    *,
    key_down: bool = True,
    repeat_count: int = 1,
    virtual_key_code: int = VK_RETURN,
    unicode_character: str = "\r",
    control_key_state: int = 0,
) -> str:
    """Normalize a synthetic Windows key record with concise defaults."""
    return normalize_windows_key_record(
        key_down=key_down,
        repeat_count=repeat_count,
        virtual_key_code=virtual_key_code,
        unicode_character=unicode_character,
        control_key_state=control_key_state,
    )


class WindowsInputTests(unittest.TestCase):
    """Keep Windows-specific normalization below the PromptBox event layer."""

    def test_plain_return_keeps_textual_enter_input(self) -> None:
        self.assertEqual(normalize(), "\r")

    def test_shift_return_encodes_shift_enter(self) -> None:
        self.assertEqual(normalize(control_key_state=SHIFT_PRESSED), SHIFT_ENTER_SEQUENCE)

    def test_shift_return_key_up_emits_nothing(self) -> None:
        self.assertEqual(normalize(key_down=False, control_key_state=SHIFT_PRESSED), "")

    def test_unrelated_printable_key_keeps_existing_processing(self) -> None:
        self.assertEqual(
            normalize(
                repeat_count=4,
                virtual_key_code=ord("A"),
                unicode_character="a",
                control_key_state=0,
            ),
            "a",
        )

    def test_shift_return_does_not_also_emit_plain_enter(self) -> None:
        events = list(XTermParser().feed(normalize(control_key_state=SHIFT_PRESSED)))

        self.assertEqual(
            [(event.key, event.character) for event in events],
            [("shift+enter", None)],
        )

    def test_shift_return_repeat_count_emits_one_event_per_repeat(self) -> None:
        events = list(
            XTermParser().feed(
                normalize(repeat_count=3, control_key_state=SHIFT_PRESSED)
            )
        )

        self.assertEqual([event.key for event in events], ["shift+enter", "shift+enter", "shift+enter"])

    def test_encoded_vt_shift_enter_passes_through_once(self) -> None:
        encoded = "".join(
            normalize(
                virtual_key_code=0,
                unicode_character=character,
                control_key_state=0,
            )
            for character in SHIFT_ENTER_SEQUENCE
        )
        events = list(XTermParser().feed(encoded))

        self.assertEqual(encoded, SHIFT_ENTER_SEQUENCE)
        self.assertEqual([event.key for event in events], ["shift+enter"])

    def test_synthetic_control_record_keeps_textual_filter(self) -> None:
        self.assertEqual(
            normalize(
                virtual_key_code=0,
                unicode_character="x",
                control_key_state=SHIFT_PRESSED,
            ),
            "",
        )

    def test_non_windows_uses_textual_default_driver(self) -> None:
        self.assertIsNone(driver_class_for_platform("linux"))
        self.assertIsNone(driver_class_for_platform("darwin"))

    def test_clipboard_reader_returns_all_hdrop_files_and_closes_clipboard(self) -> None:
        values = [r"C:\files\one.txt", r"C:\files\annual report.pdf"]
        open_clipboard = FakeNativeFunction(lambda _owner: True)
        close_clipboard = FakeNativeFunction(lambda: True)
        available = FakeNativeFunction(lambda _format: True)
        get_data = FakeNativeFunction(lambda _format: 1234567890123)

        def drag_query(_handle: object, index: int, buffer: object, _size: int) -> int:
            if index == 0xFFFFFFFF:
                return len(values)
            value = values[index]
            if buffer is None:
                return len(value)
            buffer.value = value
            return len(value)

        query_files = FakeNativeFunction(drag_query)
        user32 = SimpleNamespace(
            OpenClipboard=open_clipboard,
            CloseClipboard=close_clipboard,
            IsClipboardFormatAvailable=available,
            GetClipboardData=get_data,
        )
        shell32 = SimpleNamespace(DragQueryFileW=query_files)

        with patch.object(
            native_clipboard.ctypes,
            "WinDLL",
            side_effect=[user32, shell32],
            create=True,
        ):
            paths = native_clipboard.get_windows_clipboard_files()

        self.assertEqual([str(path) for path in paths], values)
        self.assertEqual(len(close_clipboard.calls), 1)
        self.assertIs(get_data.restype, native_clipboard.wintypes.HANDLE)
        self.assertEqual(query_files.restype, native_clipboard.wintypes.UINT)

    def test_clipboard_reader_fails_gracefully_when_clipboard_cannot_open(self) -> None:
        open_clipboard = FakeNativeFunction(lambda _owner: False)
        close_clipboard = FakeNativeFunction(lambda: True)
        user32 = SimpleNamespace(
            OpenClipboard=open_clipboard,
            CloseClipboard=close_clipboard,
            IsClipboardFormatAvailable=FakeNativeFunction(lambda _format: True),
            GetClipboardData=FakeNativeFunction(lambda _format: 1),
        )
        shell32 = SimpleNamespace(DragQueryFileW=FakeNativeFunction(lambda *_args: 0))

        with patch.object(
            native_clipboard.ctypes,
            "WinDLL",
            side_effect=[user32, shell32],
            create=True,
        ):
            self.assertEqual(native_clipboard.get_windows_clipboard_files(), [])

        self.assertEqual(close_clipboard.calls, [])

    @unittest.skipUnless(sys.platform == "win32", "MIRA's Windows driver imports only on Windows")
    def test_windows_selects_mira_driver(self) -> None:
        from ui.textual.platform.windows.driver import MiraWindowsDriver

        self.assertIs(driver_class_for_platform("win32"), MiraWindowsDriver)


if __name__ == "__main__":
    unittest.main()
