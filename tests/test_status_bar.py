"""Tests for status-bar formatting helpers."""

from __future__ import annotations

import unittest

from ui.widgets.status_bar import context_bar, truncate


def mojibake(text: str) -> str:
    """Return the common result of decoding UTF-8 bytes as Windows-1252."""
    return text.encode("utf-8").decode("windows-1252")


class StatusBarFormattingTests(unittest.TestCase):
    """Protect terminal glyphs from accidental source mojibake."""

    def test_context_bar_uses_unicode_block_glyphs(self) -> None:
        bar = context_bar(40)

        self.assertEqual(bar, "████░░░░░░")
        self.assertNotIn(mojibake("█"), bar)
        self.assertNotIn(mojibake("░"), bar)

    def test_truncate_uses_unicode_ellipsis(self) -> None:
        shortened = truncate("alpha beta", 6)

        self.assertEqual(shortened, "alpha…")
        self.assertNotIn(mojibake("…"), shortened)


if __name__ == "__main__":
    unittest.main()
