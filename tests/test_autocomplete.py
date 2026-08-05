"""Focused unit tests for autocomplete matching and project glob projection."""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from typing import Any

from ui.command_help import COMMAND_HELP_SECTIONS, command_help_entries
from ui.repl import COMMAND_HELP_SECTIONS as REPL_COMMAND_HELP_SECTIONS
from ui.widgets.autocomplete_input import command_items, completion_fragment, discover_project_files, file_items


class TreeBackend:
    """Backend ls double that exposes requested directories."""

    def __init__(self, entries: dict[str, list[dict[str, Any]]]) -> None:
        self.entries = entries
        self.requests: list[str] = []

    async def als(self, path: str) -> Any:
        self.requests.append(path)
        return SimpleNamespace(entries=self.entries.get(path, []))


class AutocompleteModelTests(unittest.TestCase):
    def test_help_and_autocomplete_share_one_command_registry(self) -> None:
        self.assertIs(REPL_COMMAND_HELP_SECTIONS, COMMAND_HELP_SECTIONS)

    def test_commands_come_from_help_source_and_are_alphabetical(self) -> None:
        help_entries = dict(command_help_entries())
        items = command_items("he")
        self.assertTrue(items)
        self.assertEqual([item.display for item in items], sorted(item.display for item in items))
        for item in items:
            self.assertEqual(item.description, help_entries[item.display])

    def test_command_insertion_space_depends_on_declared_arguments(self) -> None:
        self.assertEqual(next(item for item in command_items("goal") if item.display == "/goal <prompt>").insertion, "/goal ")
        self.assertEqual(next(item for item in command_items("plan") if item.display == "/plan [prompt]").insertion, "/plan")
        self.assertEqual(command_items("help")[0].insertion, "/help")

    def test_command_description_matches_do_not_displace_the_typed_command(self) -> None:
        self.assertEqual([item.display for item in command_items("too")], ["/tools"])

    def test_filtering_is_literal_substring_not_fuzzy(self) -> None:
        self.assertEqual(command_items("hlep"), [])
        self.assertTrue(any(item.display == "/help" for item in command_items("elp")))

    def test_slash_activation_requires_first_space_free_token_and_cursor_inside(self) -> None:
        self.assertIsNotNone(completion_fragment("/help", 3))
        self.assertIsNone(completion_fragment("/help text", 3))
        self.assertIsNone(completion_fragment("Review /he", 10))

    def test_file_filtering_is_case_insensitive_alphabetical_and_capped(self) -> None:
        paths = [f"src/Auth{index:02}.py" for index in range(12)] + ["SRC/auth.py"]
        items = file_items(paths, "aUtH")
        self.assertEqual(len(items), 10)
        self.assertEqual(
            [item.display for item in items],
            sorted((item.display for item in items), key=lambda value: (value.casefold(), value)),
        )

    def test_project_walk_prunes_noisy_directories_before_descending(self) -> None:
        backend = TreeBackend(
            {
                "/": [
                    {"path": "/src/", "is_dir": True},
                    {"path": "/build/", "is_dir": True},
                    {"path": "/.git/", "is_dir": True},
                    {"path": "/README.md", "is_dir": False},
                ],
                "/src": [
                    {"path": "/src/auth.py", "is_dir": False},
                    {"path": "/src/building.py", "is_dir": False},
                ],
                "/build": [{"path": "/build/generated.py", "is_dir": False}],
                "/.git": [{"path": "/.git/config", "is_dir": False}],
            }
        )

        self.assertEqual(
            asyncio.run(discover_project_files(backend)),
            ["README.md", "src/auth.py", "src/building.py"],
        )
        self.assertEqual(backend.requests, ["/", "/src"])

    def test_paths_with_spaces_use_quoted_insertion(self) -> None:
        item = file_items(["docs/design notes.md"], "notes")[0]
        self.assertEqual(item.insertion, '@"docs/design notes.md"')


if __name__ == "__main__":
    unittest.main()
