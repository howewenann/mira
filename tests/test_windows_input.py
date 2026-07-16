"""Tests for MIRA's Windows input-record normalization."""

from __future__ import annotations

import sys
import unittest

from textual._xterm_parser import XTermParser

from ui.windows_input import (
    SHIFT_ENTER_SEQUENCE,
    SHIFT_PRESSED,
    VK_RETURN,
    driver_class_for_platform,
    normalize_windows_key_record,
)


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

    @unittest.skipUnless(sys.platform == "win32", "MIRA's Windows driver imports only on Windows")
    def test_windows_selects_mira_driver(self) -> None:
        from ui.windows_driver import MiraWindowsDriver

        self.assertIs(driver_class_for_platform("win32"), MiraWindowsDriver)


if __name__ == "__main__":
    unittest.main()
