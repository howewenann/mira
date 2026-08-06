"""Tests for status-bar formatting helpers."""

from __future__ import annotations

from io import StringIO
import unittest

from rich.console import Console

from ui.widgets.status_bar import context_bar, telemetry_row, truncate


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

    def test_usage_and_duration_are_right_aligned(self) -> None:
        output = StringIO()
        Console(file=output, width=80, force_terminal=False).print(
            telemetry_row(
                "model",
                {
                    "context": {"used_tokens": 100, "limit_tokens": 1000},
                    "tokens": {"in": 12, "out": 3},
                    "duration_seconds": 65,
                },
                1,
            )
        )

        line = output.getvalue().rstrip("\n")
        self.assertEqual(len(line), 80)
        self.assertTrue(line.endswith("In 12 Out 3 | Turns 1 | 01:05"))


if __name__ == "__main__":
    unittest.main()
