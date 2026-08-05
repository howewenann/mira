"""Focused unit tests for autocomplete matching and project glob projection."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from ui.command_help import COMMAND_HELP_SECTIONS, command_help_entries
from ui.repl import COMMAND_HELP_SECTIONS as REPL_COMMAND_HELP_SECTIONS
from ui.widgets.autocomplete_prompt import command_items, completion_fragment, file_items, project_file_paths


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
        self.assertEqual(command_items("durable Objective")[0].insertion, "/goal ")
        self.assertEqual(next(item for item in command_items("show commands") if item.display == "/help").insertion, "/help")

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

    def test_project_paths_exclude_directories_and_noisy_components(self) -> None:
        result = SimpleNamespace(
            matches=[
                {"path": "/src/auth.py", "is_dir": False},
                {"path": "/src/building.py", "is_dir": False},
                {"path": "/build/generated.py", "is_dir": False},
                {"path": "/.git/config", "is_dir": False},
                {"path": "/src", "is_dir": True},
            ]
        )
        self.assertEqual(project_file_paths(result), ["src/auth.py", "src/building.py"])

    def test_paths_with_spaces_use_quoted_insertion(self) -> None:
        item = file_items(["docs/design notes.md"], "notes")[0]
        self.assertEqual(item.insertion, '@"docs/design notes.md"')


if __name__ == "__main__":
    unittest.main()
