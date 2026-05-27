import unittest
from io import StringIO

from rich.console import Console

from runtime import runner
from ui.renderer import Renderer


class AsyncItems:
    def __init__(self, items):
        self.items = items

    async def __aiter__(self):
        for item in self.items:
            yield item


class Message:
    def __init__(self, tool_calls=None):
        self.tool_calls = tool_calls or []


class ToolCall:
    def __init__(self, name, args, output):
        self.name = name
        self.args = args
        self.output = output


class Subagent:
    def __init__(self, name, tool_calls):
        self.name = name
        self.task_input = "look around"
        self.tool_calls = AsyncItems(tool_calls)


class RecordingRenderer:
    def __init__(self):
        self.events = []

    def add_reasoning(self, value):
        self.events.append(("reasoning", value))

    def text(self, value):
        self.events.append(("text", value))

    def tool_call(self, name, args):
        self.events.append(("tool_call", name, args))

    def delegation_started(self, calls):
        self.events.append(("delegation_started", calls))

    def subagent_label(self, subagent):
        return subagent.name

    def subagent_started(self, name, task_input=""):
        self.events.append(("subagent_started", name, task_input))

    def subagent_finished(self, name, tool=None, args=None, result=""):
        self.events.append(("subagent_finished", name, tool, args, result))


class RecordingConsole:
    def __init__(self):
        self.lines = []

    def print(self, *values, **kwargs):
        self.lines.append(" ".join(str(value) for value in values))


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_task_tool_calls_are_hidden(self):
        renderer = RecordingRenderer()
        messages = AsyncItems(
            [
                Message(
                    [
                        {"name": "task", "args": {"description": "delegate"}},
                        {"name": "read_file", "args": {"path": "README.md"}},
                    ]
                )
            ]
        )

        await runner._consume_messages(messages, renderer)

        self.assertEqual(
            renderer.events,
            [
                (
                    "delegation_started",
                    [{"name": "task", "args": {"description": "delegate"}}],
                ),
                ("tool_call", "read_file", {"path": "README.md"}),
            ],
        )

    async def test_two_task_calls_produce_one_delegation_event(self):
        renderer = RecordingRenderer()
        messages = AsyncItems(
            [
                Message(
                    [
                        {"name": "task", "args": {"description": "one"}},
                        {"name": "task", "args": {"description": "two"}},
                    ]
                )
            ]
        )

        await runner._consume_messages(messages, renderer)

        self.assertEqual(len(renderer.events), 1)
        self.assertEqual(renderer.events[0][0], "delegation_started")
        self.assertEqual(len(renderer.events[0][1]), 2)

    async def test_subagent_prints_one_header_and_final_call(self):
        renderer = RecordingRenderer()
        subagent = Subagent(
            "general-purpose [one]",
            [
                ToolCall("grep", {"pattern": "TODO"}, "first output"),
                ToolCall("read_file", {"path": "ui/renderer.py"}, "final output"),
            ],
        )

        await runner._consume_subagent(subagent, renderer)

        self.assertEqual(
            renderer.events,
            [
                ("subagent_started", "general-purpose [one]", "look around"),
                (
                    "subagent_finished",
                    "general-purpose [one]",
                    "read_file",
                    {"path": "ui/renderer.py"},
                    "final output",
                ),
            ],
        )

    async def test_two_subagents_print_two_headers(self):
        renderer = RecordingRenderer()

        await runner._consume_subagents(
            AsyncItems(
                [
                    Subagent("general-purpose [one]", [ToolCall("grep", {}, "one")]),
                    Subagent("general-purpose [two]", [ToolCall("grep", {}, "two")]),
                ]
            ),
            renderer,
        )

        headers = [event for event in renderer.events if event[0] == "subagent_started"]
        self.assertEqual(len(headers), 2)

    def test_renderer_truncates_final_subagent_output(self):
        renderer = Renderer(tool_output_chars=5)

        self.assertEqual(renderer.truncate("abcdefgh"), "abcde ... truncated ...")

    def test_renderer_prints_each_subagent_header_once(self):
        renderer = Renderer()
        renderer.console = RecordingConsole()

        renderer.subagent_started("general-purpose [one]")
        renderer.subagent_started("general-purpose [two]")
        renderer.subagent_finished("general-purpose [one]", "grep", {}, "done")

        console = Console(record=True, force_terminal=False, width=100, file=StringIO())
        console.print(renderer.render_subagents())
        output = console.export_text()

        self.assertEqual(output.count("subagent - general-purpose [one]"), 1)
        self.assertEqual(output.count("subagent - general-purpose [two]"), 1)

    def test_renderer_renders_running_and_finished_blocks(self):
        renderer = Renderer(tool_output_chars=8)
        renderer.console = RecordingConsole()
        renderer.subagent_started("general-purpose [one]", "inspect files")
        renderer.subagent_finished(
            "general-purpose [one]",
            "read_file",
            {"path": "runtime/runner.py"},
            "abcdefghijklmnopqrstuvwxyz",
        )

        console = Console(record=True, force_terminal=False, width=100, file=StringIO())
        console.print(renderer.render_subagents())
        output = console.export_text()

        self.assertIn("subagent - general-purpose [one]", output)
        self.assertIn("request:", output)
        self.assertIn("DONE", output)
        self.assertIn("final tool:", output)
        self.assertIn("args:", output)
        self.assertIn("output:", output)
        self.assertIn("truncated", output)
        self.assertNotIn("\033", output)

    def test_renderer_renders_running_status(self):
        renderer = Renderer()
        renderer.console = RecordingConsole()
        renderer.subagent_started("general-purpose [one]", "inspect files")

        console = Console(record=True, force_terminal=False, width=100, file=StringIO())
        console.print(renderer.render_subagents())
        output = console.export_text()

        self.assertIn("RUNNING", output)
        self.assertIn("request:", output)


if __name__ == "__main__":
    unittest.main()
