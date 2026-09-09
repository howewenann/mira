"""Tests for Rich runtime report tables."""

from __future__ import annotations

import unittest
from io import StringIO

from rich.console import Console

from agent.mcp.models import PromptArgument, PromptSpec
from ui.textual.runtime_report import prompts_table


async def resolve(_values):
    return []


class PromptTableTests(unittest.TestCase):
    def render(self, table, width: int = 120) -> str:
        output = StringIO()
        Console(file=output, force_terminal=False, width=width).print(table)
        return output.getvalue()

    def test_local_and_mcp_rows_separate_command_source_and_arguments(self) -> None:
        local = PromptSpec("/prompt__test_prompt", "Local prompt", (), "local", resolve)
        mcp = PromptSpec(
            "/mcp__github__issue_to_fix_workflow",
            "Create a detailed issue workflow description that wraps naturally.",
            (
                PromptArgument("owner"),
                PromptArgument("repo"),
                PromptArgument("title"),
                PromptArgument("description"),
                PromptArgument("labels", required=False),
            ),
            "mcp",
            resolve,
            server="github",
        )
        table = prompts_table([local, mcp])

        self.assertEqual([column.header for column in table.columns], ["Prompt", "Source", "Arguments", "Description"])
        self.assertEqual(str(table.columns[0]._cells[0]), "/prompt__test_prompt")
        self.assertEqual(str(table.columns[2]._cells[0]), "-")
        self.assertEqual(str(table.columns[1]._cells[1]), "mcp:github")
        self.assertNotIn("<owner>", str(table.columns[0]._cells[1]))
        self.assertIn("<owner> <repo>", str(table.columns[2]._cells[1]))
        rendered = self.render(table, width=72)
        self.assertIn("issue_to_fix", rendered)
        self.assertGreater(len(rendered.splitlines()), 5)

    def test_empty_registry_has_explicit_row(self) -> None:
        table = prompts_table([])
        self.assertIn("none loaded", self.render(table))
