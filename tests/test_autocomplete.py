"""Focused unit tests for autocomplete matching and project glob projection."""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from typing import Any

from agent.mcp.models import MCPResource, PromptArgument, PromptSpec
from ui.command_help import COMMAND_HELP_SECTIONS, command_help_entries
from ui.repl import COMMAND_HELP_SECTIONS as REPL_COMMAND_HELP_SECTIONS
from ui.widgets.autocomplete_input import (
    CompletionItem,
    _completion_row,
    attachment_items,
    command_items,
    completion_fragment,
    discover_project_files,
    file_items,
)


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

    def test_command_insertion_is_always_the_exact_command_token(self) -> None:
        self.assertEqual(next(item for item in command_items("goal") if item.display == "/goal <prompt>").insertion, "/goal")
        self.assertEqual(next(item for item in command_items("plan") if item.display == "/plan [prompt]").insertion, "/plan")
        self.assertEqual(command_items("help")[0].insertion, "/help")

    def test_native_and_prompt_commands_keep_explicit_kinds_and_full_usage(self) -> None:
        async def resolve(_values: list[str]) -> list[Any]:
            return []

        prompt = PromptSpec(
            command="/prompt__review",
            description="Review code",
            arguments=(PromptArgument("file"), PromptArgument("focus", required=False)),
            source="local",
            resolver=resolve,
        )
        registry = SimpleNamespace(specs={prompt.command: prompt})

        native = next(item for item in command_items("goal", registry) if item.display.startswith("/goal "))
        reusable = next(item for item in command_items("review", registry))

        self.assertEqual(native.kind, "native_command")
        self.assertEqual(native.insertion, "/goal")
        self.assertEqual(reusable.kind, "prompt_command")
        self.assertEqual(reusable.display, "/prompt__review <file> [focus]")
        self.assertEqual(reusable.insertion, "/prompt__review")

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
        self.assertEqual(len(items), 5)
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

    def test_attachment_matches_merge_and_sort_by_exact_display_identifier(self) -> None:
        resource = MCPResource(
            token="mcp__docs__api_reference",
            server="docs",
            uri="docs://api",
            name="API reference",
            description="API documentation",
        )
        tools = [
            {"name": "read_file", "description": "Read a file"},
            {"name": "mcp__github__create_issue", "description": "Create an issue"},
        ]

        items = attachment_items(
            ["src/read_file_notes.py"],
            [resource],
            "",
            tools=tools,
        )

        self.assertEqual(
            [item.display for item in items],
            sorted((item.display for item in items), key=lambda value: (value.casefold(), value)),
        )
        mcp_tool = next(item for item in items if item.kind == "tool" and item.display.startswith("mcp__github"))
        file_item = next(item for item in items if item.kind == "file")
        resource_item = next(item for item in items if item.kind == "mcp_resource")
        self.assertEqual(mcp_tool.insertion, "mcp__github__create_issue")
        self.assertEqual(file_item.insertion, "@src/read_file_notes.py")
        self.assertEqual(resource_item.insertion, "@mcp__docs__api_reference")

    def test_tool_matching_is_literal_substring_without_category_behavior(self) -> None:
        tools = [
            {"name": "read_file", "description": "Read"},
            {"name": "search_tool", "description": "Search"},
        ]

        self.assertEqual(
            [item.display for item in attachment_items([], [], "too", tools=tools)],
            ["search_tool"],
        )
        self.assertEqual(attachment_items([], [], "rfl", tools=tools), [])

    def test_completion_rows_use_exact_label_styles_without_coloring_content(self) -> None:
        expected = {
            "tool": ("TOOL", "#78d5cf"),
            "mcp_resource": ("RSRC", "#c7a0e8"),
            "file": ("FILE", "#aeb8be"),
            "native_command": ("CMND", "#d2a957"),
            "prompt_command": ("PRMT", "#8fb9e8"),
        }

        for kind, (label, color) in expected.items():
            with self.subTest(kind=kind):
                row = _completion_row(CompletionItem(kind, "identifier", "value", "description"))
                self.assertEqual(row.plain, f"{label}  identifier  description")
                self.assertTrue(row.no_wrap)
                self.assertEqual(row.overflow, "ellipsis")
                self.assertEqual(str(row.spans[0].style), f"bold {color}")
                self.assertEqual(str(row.spans[1].style), "#e8edef")
                self.assertEqual(str(row.spans[2].style), "#b8c1c7")

    def test_status_rows_are_unlabelled_muted_and_single_line(self) -> None:
        row = _completion_row(
            CompletionItem("status", "MCP resources unavailable", "", selectable=False)
        )

        self.assertEqual(row.plain, "MCP resources unavailable")
        self.assertEqual(str(row.spans[0].style), "#b8c1c7")
        self.assertTrue(row.no_wrap)
        self.assertEqual(row.overflow, "ellipsis")


if __name__ == "__main__":
    unittest.main()
