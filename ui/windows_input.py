"""Windows key-record normalization used by MIRA's Textual driver."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from textual.driver import Driver

VK_RETURN = 0x0D
SHIFT_PRESSED = 0x0010
SHIFT_ENTER_SEQUENCE = "\x1b[13;2u"


def normalize_windows_key_record(
    *,
    key_down: bool,
    repeat_count: int,
    virtual_key_code: int,
    unicode_character: str,
    control_key_state: int,
) -> str:
    """Return the text Textual should parse for one Windows key record.

    Textual 8.2.7 normally keeps only ``UnicodeChar`` from a key-down record.
    Preserve that behavior except for Shift+Return, where the discarded control
    state is needed to distinguish a newline request from prompt submission.
    """
    if not key_down:
        return ""

    # Preserve Textual's existing filtering of synthetic control-state records.
    if control_key_state and virtual_key_code == 0:
        return ""

    if virtual_key_code == VK_RETURN and control_key_state & SHIFT_PRESSED:
        return SHIFT_ENTER_SEQUENCE * repeat_count

    return unicode_character


def driver_class_for_platform(platform: str) -> type[Driver] | None:
    """Return MIRA's Windows driver only for native Windows launches."""
    if platform != "win32":
        return None

    from ui.windows_driver import MiraWindowsDriver

    return MiraWindowsDriver
