"""Tests for planning-mode policy and interactive routing."""

from __future__ import annotations

import unittest
from io import StringIO
from typing import Any
from unittest.mock import patch

from rich.console import Console

from agent import factory
from agent.plan_policy import PLAN_PROJECT_WRITE_TOOLS, project_write_tools_text, plan_system_prompt
from runtime import runner
from ui import repl


class RecordingConsole:
    """Console double that stores printed lines for assertions."""

    def __init__(self) -> None:
        """Create empty line storage."""
        self.lines: list[str] = []

    def print(self, *values: Any, **kwargs: Any) -> None:
        """Record printed values as a single string."""
        output = StringIO()
        console = Console(file=output, force_terminal=False, width=120)
        console.print(*values, **kwargs)
        self.lines.append(output.getvalue().rstrip("\n"))

    def clear(self) -> None:
        """Record that the screen was cleared."""
        self.lines.append("clear")


class RecordingRenderer:
    """Renderer double for interactive and plan-mode tests."""

    def __init__(self) -> None:
        """Create console and plan-panel storage."""
        self.console = RecordingConsole()
        self.plan_panels: list[tuple[int, str]] = []
        self.no_plans_called = False

    def splash(self, model_name: str, session_id: str, workspace: str) -> None:
        """Record splash metadata."""
        self.console.print(f"splash {model_name} {session_id} {workspace}")

    def newline(self) -> None:
        """Record a blank line."""
        self.console.print("")

    def plan(self, plan_id: int, text: str) -> None:
        """Record a rendered saved-plan panel."""
        self.plan_panels.append((plan_id, text))

    def no_plans(self) -> None:
        """Record that the no-plans empty state was rendered."""
        self.no_plans_called = True


class FakeModelRequest:
    """Small request object used to test PlanningToolFilter."""

    def __init__(self, tools: list[Any]) -> None:
        """Store the available tools on the request test double."""
        self.tools = tools

    def override(self, **kwargs: Any) -> "FakeModelRequest":
        """Return a new request with replacement tools."""
        return FakeModelRequest(kwargs.get("tools", self.tools))


class FakeStore:
    """Store double that accepts session saves from interactive turns."""

    def save(self, record: dict[str, Any]) -> None:
        """Ignore saved records; tests assert routing instead."""
        return None


def sample_tool() -> None:
    """Run a sample operation for tests."""
    return None


class PlanModeTests(unittest.IsolatedAsyncioTestCase):
    """Tests for planning-mode policy, interactive routing, and plan storage."""

    def test_plan_permissions_deny_writes(self) -> None:
        """Planning permissions should deny all filesystem writes."""
        permissions = factory._plan_permissions()

        self.assertEqual(len(permissions), 1)
        self.assertEqual(permissions[0].operations, ["write"])
        self.assertEqual(permissions[0].paths, ["/**"])
        self.assertEqual(permissions[0].mode, "deny")

    def test_action_permissions_allow_reads_and_writes(self) -> None:
        """Action permissions should allow filesystem reads and writes."""
        permissions = factory._action_permissions()

        self.assertEqual(len(permissions), 2)
        self.assertEqual(permissions[0].operations, ["write"])
        self.assertEqual(permissions[0].paths, ["/mira-defaults/**"])
        self.assertEqual(permissions[0].mode, "deny")
        self.assertEqual(permissions[1].operations, ["read", "write"])
        self.assertEqual(permissions[1].paths, ["/**"])
        self.assertEqual(permissions[1].mode, "allow")

    def test_plan_agent_disables_write_interrupts_and_denies_writes(self) -> None:
        """The planning agent should hide writes instead of requesting approval."""
        with (
            patch("agent.factory.get_llm", return_value="model"),
            patch("agent.factory.CodeInterpreterMiddleware", return_value="code"),
            patch("agent.factory.create_summarization_tool_middleware", return_value="summary"),
            patch("agent.factory.create_deep_agent", return_value="agent") as create_deep_agent,
        ):
            agent = factory.build_plan_agent({}, ".", "checkpointer")

        self.assertEqual(agent, "agent")
        kwargs = create_deep_agent.call_args.kwargs
        self.assertIsNone(kwargs["interrupt_on"])
        self.assertIn("planning mode", kwargs["system_prompt"])
        self.assertEqual(kwargs["permissions"][0].operations, ["write"])
        self.assertEqual(kwargs["permissions"][0].mode, "deny")
        self.assertTrue(any(isinstance(item, factory.PlanningToolFilter) for item in kwargs["middleware"]))

    def test_action_agent_keeps_write_interrupts(self) -> None:
        """The action agent should keep write approval interrupts enabled."""
        with (
            patch("agent.factory.get_llm", return_value="model"),
            patch("agent.factory.CodeInterpreterMiddleware", return_value="code"),
            patch("agent.factory.create_summarization_tool_middleware", return_value="summary"),
            patch("agent.factory.create_deep_agent", return_value="agent") as create_deep_agent,
        ):
            agent = factory.build_agent({}, ".", "checkpointer")

        self.assertEqual(agent, "agent")
        kwargs = create_deep_agent.call_args.kwargs
        self.assertEqual(kwargs["interrupt_on"], factory._write_interrupts())
        self.assertIsNone(kwargs["system_prompt"])
        self.assertEqual(kwargs["permissions"][0].paths, ["/mira-defaults/**"])
        self.assertEqual(kwargs["permissions"][0].mode, "deny")
        self.assertEqual(kwargs["permissions"][1].operations, ["read", "write"])
        self.assertEqual(kwargs["permissions"][1].mode, "allow")
        self.assertTrue(any(factory._tool_name(tool) == "grep" for tool in kwargs["tools"]))

    def test_agent_build_attaches_tool_metadata(self) -> None:
        """Built agents should expose tool metadata for the UI."""
        with (
            patch("agent.factory.get_llm", return_value="model"),
            patch("agent.factory.CodeInterpreterMiddleware", return_value="code"),
            patch("agent.factory.create_summarization_tool_middleware", return_value="summary"),
            patch("agent.factory.create_deep_agent", return_value=type("Agent", (), {})()),
        ):
            agent = factory.build_agent({}, ".", "checkpointer")

        names = [tool["name"] for tool in agent.mira_tool_specs]
        self.assertIn("ask_user", names)
        self.assertIn("read_file", names)
        self.assertIn("write_file", names)
        self.assertIn("edit_file", names)
        self.assertIn("grep", names)
        grep = next(tool for tool in agent.mira_tool_specs if tool["name"] == "grep")
        self.assertEqual(grep["source"], "default")
        self.assertEqual(grep["replaces"], "built-in")

    def test_plan_agent_metadata_hides_write_tools(self) -> None:
        """Plan agents should expose only planning-available tool metadata."""
        with (
            patch("agent.factory.get_llm", return_value="model"),
            patch("agent.factory.CodeInterpreterMiddleware", return_value="code"),
            patch("agent.factory.create_summarization_tool_middleware", return_value="summary"),
            patch("agent.factory.create_deep_agent", return_value=type("Agent", (), {})()),
        ):
            agent = factory.build_plan_agent({}, ".", "checkpointer")

        names = [tool["name"] for tool in agent.mira_tool_specs]
        self.assertIn("ask_user", names)
        self.assertIn("read_file", names)
        self.assertNotIn("write_file", names)
        self.assertNotIn("edit_file", names)

    def test_plan_tool_filter_hides_write_tools_from_model(self) -> None:
        """PlanningToolFilter should remove write/edit tools from requests."""
        middleware = factory.PlanningToolFilter(PLAN_PROJECT_WRITE_TOOLS)
        request = FakeModelRequest(
            [
                {"name": "read_file"},
                {"name": "write_file"},
                type("Tool", (), {"name": "edit_file"})(),
                type("Tool", (), {"name": "grep"})(),
            ]
        )

        filtered = middleware._filter_request(request)

        names = [factory._tool_name(tool) for tool in filtered.tools]
        self.assertEqual(names, ["read_file", "grep"])

    async def test_plan_and_act_commands_toggle_mode(self) -> None:
        """Slash commands should switch between planning and action modes."""
        renderer = RecordingRenderer()
        mode: dict[str, Any] = {"planning": False}
        session = {"id": "thread-1", "workspace": ".", "turns": 0}

        handled = await repl.handle_command("/plan", renderer, session, "model", mode)
        self.assertTrue(handled)
        self.assertTrue(mode["planning"])
        self.assertIn("planning mode", renderer.console.lines[-1])
        self.assertIn(f"{project_write_tools_text()} disabled", renderer.console.lines[-1])

        handled = await repl.handle_command("/act", renderer, session, "model", mode)
        self.assertTrue(handled)
        self.assertFalse(mode["planning"])
        self.assertIn("action mode", renderer.console.lines[-1])

    async def test_act_announces_pending_plan_when_plan_exists(self) -> None:
        """Leaving planning mode should queue the last saved plan once."""
        renderer = RecordingRenderer()
        mode = {"planning": True, "last_plan": "Do the thing", "plan_pending": False}

        handled = await repl.handle_command("/act", renderer, {}, "model", mode)

        self.assertTrue(handled)
        self.assertFalse(mode["planning"])
        self.assertTrue(mode["plan_pending"])
        self.assertIn("last plan will be included", renderer.console.lines[-1])

    async def test_help_includes_plan_commands(self) -> None:
        """The help command should describe available commands."""
        renderer = RecordingRenderer()

        handled = await repl.handle_command("/help", renderer, {}, "model", {"planning": False})

        self.assertTrue(handled)
        self.assertEqual(len(renderer.console.lines), 1)
        output = "\n".join(renderer.console.lines)
        self.assertIn("Commands", output)
        self.assertIn("/plan", output)
        self.assertIn("enter planning mode", output)
        self.assertIn("/act", output)
        self.assertIn("return to action mode", output)
        self.assertIn("/tools", output)
        self.assertIn("list tools", output)
        self.assertIn("/memories", output)
        self.assertIn("/skills", output)
        self.assertIn("/subagents", output)

    def test_help_table_returns_rich_table(self) -> None:
        """The help table should render all commands together."""
        table = repl.help_table()

        output = StringIO()
        Console(file=output, force_terminal=False, width=100).print(table)
        rendered = output.getvalue()

        self.assertIn("Commands", rendered)
        self.assertIn("Command", rendered)
        self.assertIn("Description", rendered)
        self.assertIn("/help", rendered)
        self.assertIn("/subagents", rendered)

    async def test_tools_command_lists_action_tools(self) -> None:
        """The tools command should show action-mode tools."""
        renderer = RecordingRenderer()

        handled = await repl.handle_command("/tools", renderer, {}, "model", {"planning": False})

        self.assertTrue(handled)
        output = "\n".join(renderer.console.lines)
        self.assertIn("Tools (action)", output)
        self.assertIn("ask_user", output)
        self.assertIn("read_file", output)
        self.assertIn("write_file", output)
        self.assertIn("edit_file", output)
        self.assertIn("task", output)
        self.assertIn("Description", output)

    async def test_tools_command_hides_write_tools_in_planning_mode(self) -> None:
        """The tools command should reflect planning-mode tool restrictions."""
        renderer = RecordingRenderer()

        handled = await repl.handle_command("/tools", renderer, {}, "model", {"planning": True})

        self.assertTrue(handled)
        output = "\n".join(renderer.console.lines)
        self.assertIn("Tools (planning)", output)
        self.assertIn("ask_user", output)
        self.assertIn("read_file", output)
        self.assertNotIn("write_file", output)
        self.assertNotIn("edit_file", output)

    async def test_resource_commands_show_loaded_resources(self) -> None:
        """Resource commands should print attached resource metadata."""
        renderer = RecordingRenderer()
        mode = {
            "resources": {
                "memories": [
                    {
                        "name": "AGENTS.md",
                        "source": "project",
                        "path": "/.mira/memories/AGENTS.md",
                        "replaces": "default",
                    }
                ],
                "skills": [
                    {
                        "name": "codebase-orientation",
                        "source": "default",
                        "path": "/mira-defaults/skills/codebase-orientation/SKILL.md",
                        "replaces": "",
                    }
                ],
                "subagents": [
                    {
                        "name": "code-reviewer",
                        "source": "default",
                        "path": "/mira-defaults/subagents/code_reviewer.py",
                        "replaces": "",
                    }
                ],
            }
        }

        self.assertTrue(await repl.handle_command("/memories", renderer, {}, "model", mode))
        self.assertTrue(await repl.handle_command("/skills", renderer, {}, "model", mode))
        self.assertTrue(await repl.handle_command("/subagents", renderer, {}, "model", mode))

        output = "\n".join(renderer.console.lines)
        self.assertIn("Memories", output)
        self.assertIn("AGENTS.md", output)
        self.assertIn("project", output)
        self.assertIn("Replaces", output)
        self.assertIn("Skills", output)
        self.assertIn("codebase-orientation", output)
        self.assertIn("default", output)
        self.assertIn("Subagents", output)
        self.assertIn("code-reviewer", output)

    def test_tool_specs_use_agent_metadata(self) -> None:
        """Tool specs should come from agent metadata when available."""
        agent = type(
            "Agent",
            (),
            {"mira_tool_specs": [{"name": "custom_tool", "description": "custom description"}]},
        )()

        self.assertEqual(repl.tool_specs(agent), [{"name": "custom_tool", "description": "custom description"}])

    def test_tool_specs_use_docstrings_for_callables(self) -> None:
        """Callable tool descriptions should fall back to docstrings."""
        agent = type("Agent", (), {"tools": [sample_tool]})()

        self.assertEqual(
            repl.tool_specs(agent),
            [{"name": "sample_tool", "description": "Run a sample operation for tests."}],
        )

    def test_tool_descriptions_use_first_sentence(self) -> None:
        """Tool display descriptions should be shortened for the table."""
        tools = [{"name": "read_file", "description": "Reads a file from the filesystem.\n\nUse this after listing files."}]

        self.assertEqual(
            repl.normalize_tool_specs(tools),
            [{"name": "read_file", "description": "Reads a file from the filesystem."}],
        )

    def test_tools_table_returns_rich_table(self) -> None:
        """The tools table should expose tool metadata in a Rich table."""
        table = repl.tools_table(
            "Tools (action)",
            [{"name": "long_tool", "description": "This description should wrap when the width is narrow."}],
        )

        output = StringIO()
        Console(file=output, force_terminal=False, width=80).print(table)
        rendered = output.getvalue()
        self.assertIn("Tools (action)", rendered)
        self.assertIn("long_tool", rendered)
        self.assertIn("Description", rendered)

    def test_tool_table_shows_source_and_replacement(self) -> None:
        """The tools table should show custom tool source and replacement info."""
        table = repl.tools_table(
            "Tools (action)",
            [
                {
                    "name": "grep",
                    "description": "Search with regex.",
                    "source": "default",
                    "replaces": "built-in",
                }
            ],
        )

        output = StringIO()
        Console(file=output, force_terminal=False, width=80).print(table)
        rendered = output.getvalue()
        self.assertIn("grep", rendered)
        self.assertIn("default", rendered)
        self.assertIn("built-in", rendered)

    def test_resources_table_returns_rich_table(self) -> None:
        """The resources table should expose source and replacement columns."""
        table = repl.resources_table(
            "Memories",
            [
                {
                    "name": "AGENTS.md",
                    "source": "project",
                    "replaces": "default",
                    "path": "/.mira/memories/AGENTS.md",
                }
            ],
        )

        output = StringIO()
        Console(file=output, force_terminal=False, width=100).print(table)
        rendered = output.getvalue()
        self.assertIn("Memories", rendered)
        self.assertIn("AGENTS.md", rendered)
        self.assertIn("project", rendered)
        self.assertIn("default", rendered)

    async def test_clear_command_clears_console(self) -> None:
        """The clear command should call the console clear method."""
        renderer = RecordingRenderer()

        handled = await repl.handle_command("/clear", renderer, {}, "model", {"planning": False})

        self.assertTrue(handled)
        self.assertEqual(renderer.console.lines, ["clear"])

    async def test_run_user_turn_routes_to_plan_agent_while_planning(self) -> None:
        """User text in planning mode should go to the planning agent."""
        renderer = RecordingRenderer()
        session = {"id": "thread-1", "workspace": ".", "turns": 0}
        store = FakeStore()
        mode = repl.initial_mode("action-agent", "plan-agent")
        calls: list[tuple[Any, str, str]] = []

        async def fake_run_turn(agent: Any, text: str, renderer: Any, thread_id: str) -> runner.TurnResult:
            """Record agent invocations from interactive routing."""
            calls.append((agent, text, thread_id))
            return runner.TurnResult()

        with patch("ui.repl.run_turn", fake_run_turn):
            await repl.handle_command("/plan", renderer, session, "model", mode)
            await repl.run_user_turn(
                agent="action-agent",
                plan_agent="plan-agent",
                renderer=renderer,
                store=store,
                session=session,
                mode=mode,
                text="write a file",
            )
            await repl.handle_command("/act", renderer, session, "model", mode)
            await repl.run_user_turn(
                agent="action-agent",
                plan_agent="plan-agent",
                renderer=renderer,
                store=store,
                session=session,
                mode=mode,
                text="write it now",
            )

        self.assertEqual(calls[0][0], "plan-agent")
        self.assertIn("You are in planning mode.", calls[0][1])
        self.assertIn("Do not call write_file, edit_file", calls[0][1])
        self.assertIn("User request:\nwrite a file", calls[0][1])
        self.assertEqual(calls[0][2], "thread-1:plan:1")
        self.assertEqual(calls[1], ("action-agent", "write it now", "thread-1"))
        self.assertEqual(session["turns"], 2)

    async def test_run_user_turn_injects_valid_plan_once_after_act(self) -> None:
        """A clean saved plan should be injected into one action request."""
        renderer = RecordingRenderer()
        session = {"id": "thread-1", "workspace": ".", "turns": 0}
        store = FakeStore()
        mode = repl.initial_mode("action-agent", "plan-agent")
        calls: list[tuple[Any, str, str]] = []
        results = [
            runner.TurnResult(final_text="Create test.txt with hello world."),
            runner.TurnResult(final_text="done"),
            runner.TurnResult(final_text="done again"),
        ]

        async def fake_run_turn(agent: Any, text: str, renderer: Any, thread_id: str) -> runner.TurnResult:
            """Record agent invocations and return scripted results."""
            calls.append((agent, text, thread_id))
            return results.pop(0)

        with patch("ui.repl.run_turn", fake_run_turn):
            await repl.handle_command("/plan", renderer, session, "model", mode)
            await repl.run_user_turn(
                agent="action-agent",
                plan_agent="plan-agent",
                renderer=renderer,
                store=store,
                session=session,
                mode=mode,
                text="plan the write",
            )
            await repl.handle_command("/act", renderer, session, "model", mode)
            await repl.run_user_turn(
                agent="action-agent",
                plan_agent="plan-agent",
                renderer=renderer,
                store=store,
                session=session,
                mode=mode,
                text="do it",
            )
            await repl.run_user_turn(
                agent="action-agent",
                plan_agent="plan-agent",
                renderer=renderer,
                store=store,
                session=session,
                mode=mode,
                text="do another thing",
            )

        self.assertEqual(calls[0][0], "plan-agent")
        self.assertIn("User request:\nplan the write", calls[0][1])
        self.assertEqual(calls[0][2], "thread-1:plan:1")
        self.assertEqual(calls[1][0], "action-agent")
        self.assertIn("Previous planning context:", calls[1][1])
        self.assertIn("Create test.txt with hello world.", calls[1][1])
        self.assertIn("You are now in action mode.", calls[1][1])
        self.assertIn("Write/edit tools are available again", calls[1][1])
        self.assertIn("User request:\ndo it", calls[1][1])
        self.assertEqual(calls[2], ("action-agent", "do another thing", "thread-1"))

    async def test_run_user_turn_does_not_save_blocked_plan(self) -> None:
        """A plan that tried to write should not be reused in action mode."""
        renderer = RecordingRenderer()
        session = {"id": "thread-1", "workspace": ".", "turns": 0}
        store = FakeStore()
        mode = repl.initial_mode("action-agent", "plan-agent")
        calls: list[tuple[Any, str, str]] = []
        results = [
            runner.TurnResult(
                final_text="I cannot write files.",
                tool_calls=["write_file"],
                tool_results=["Error: permission denied for write on /test.txt"],
            ),
            runner.TurnResult(final_text="done"),
        ]

        async def fake_run_turn(agent: Any, text: str, renderer: Any, thread_id: str) -> runner.TurnResult:
            """Record agent invocations and return scripted results."""
            calls.append((agent, text, thread_id))
            return results.pop(0)

        with patch("ui.repl.run_turn", fake_run_turn):
            await repl.handle_command("/plan", renderer, session, "model", mode)
            await repl.run_user_turn(
                agent="action-agent",
                plan_agent="plan-agent",
                renderer=renderer,
                store=store,
                session=session,
                mode=mode,
                text="write a file",
            )
            await repl.handle_command("/act", renderer, session, "model", mode)
            await repl.run_user_turn(
                agent="action-agent",
                plan_agent="plan-agent",
                renderer=renderer,
                store=store,
                session=session,
                mode=mode,
                text="do it",
            )

        self.assertTrue(any("no plan was saved" in line for line in renderer.console.lines))
        self.assertEqual(calls[1], ("action-agent", "do it", "thread-1"))

    async def test_session_command_shows_current_mode(self) -> None:
        """The session command should print mode and saved-plan count."""
        renderer = RecordingRenderer()
        session = {"id": "thread-1", "workspace": ".", "turns": 3}

        handled = await repl.handle_command("/session", renderer, session, "model", {"planning": True, "plans": []})

        self.assertTrue(handled)
        self.assertIn("session: thread-1", renderer.console.lines)
        self.assertIn("mode: planning", renderer.console.lines)
        self.assertIn("saved plans: 0", renderer.console.lines)

    def test_plan_thread_id_is_separate_from_action_thread(self) -> None:
        """Planning threads should be isolated from action memory."""
        self.assertEqual(repl.plan_thread_id({"id": "thread-1"}), "thread-1:plan")
        self.assertEqual(repl.plan_thread_id({"id": "thread-1"}, 2), "thread-1:plan:2")

    def test_action_request_text_clears_pending_plan(self) -> None:
        """The saved plan should be consumed only once."""
        mode = {"planning": False, "last_plan": "Plan text", "plan_pending": True}

        text = repl.action_request_text(mode, "Implement")

        self.assertIn("Previous planning context:\nPlan text", text)
        self.assertIn("Do not assume planning-mode permission errors still apply.", text)
        self.assertIn("User request:\nImplement", text)
        self.assertEqual(mode["last_plan"], "")
        self.assertFalse(mode["plan_pending"])

    def test_invalid_plan_result_when_write_tool_was_used(self) -> None:
        """A plan is invalid if the write tool was called."""
        result = runner.TurnResult(final_text="Nope", tool_calls=["write_file"])

        self.assertFalse(repl.has_clean_plan(result))

    def test_invalid_plan_result_when_project_write_was_blocked(self) -> None:
        """A plan is invalid if a write was blocked by permissions."""
        result = runner.TurnResult(
            final_text="Nope",
            tool_calls=["write_file"],
            tool_results=["Error: permission denied for write on /test.txt"],
        )

        self.assertFalse(repl.has_clean_plan(result))

    def test_invalid_plan_result_when_final_text_mentions_permission_denied(self) -> None:
        """A plan is invalid if final text reports a blocked write."""
        result = runner.TurnResult(final_text="I hit permission denied for write on /test.txt.")

        self.assertFalse(repl.has_clean_plan(result))

    def test_plan_request_text_wraps_user_request(self) -> None:
        """Planning requests should include the non-mutating instructions."""
        text = repl.plan_request_text("write a file")

        self.assertIn("You are in planning mode.", text)
        self.assertIn("Respond only with a concrete plan", text)
        self.assertIn("User request:\nwrite a file", text)

    def test_save_plan_result_records_memory_plan(self) -> None:
        """A clean plan should be saved in the interactive mode dictionary."""
        renderer = RecordingRenderer()
        mode: dict[str, Any] = {"plans": [], "last_plan": "", "plan_pending": False}

        repl.save_clean_plan(mode, runner.TurnResult(final_text="1. Create test.txt."), renderer)

        self.assertEqual(mode["last_plan"], "1. Create test.txt.")
        self.assertEqual(mode["plans"], [{"id": 1, "text": "1. Create test.txt."}])

    async def test_plans_command_shows_empty_state(self) -> None:
        """The plans command should render an empty state when no plans exist."""
        renderer = RecordingRenderer()

        handled = await repl.handle_command("/plans", renderer, {}, "model", {"plans": []})

        self.assertTrue(handled)
        self.assertTrue(renderer.no_plans_called)

    async def test_plans_command_renders_each_saved_plan_once(self) -> None:
        """The plans command should render each saved plan exactly once."""
        renderer = RecordingRenderer()
        mode = {"plans": [{"id": 1, "text": "First plan\nmore"}, {"id": 2, "text": "Latest plan"}]}

        handled = await repl.handle_command("/plans", renderer, {}, "model", mode)

        self.assertTrue(handled)
        self.assertEqual(renderer.plan_panels, [(1, "First plan\nmore"), (2, "Latest plan")])
        self.assertEqual(renderer.console.lines, [])

    def test_plan_policy_drives_prompt_and_repl_validation(self) -> None:
        """The policy constants should appear in the planning prompt."""
        prompt = plan_system_prompt()

        for tool in PLAN_PROJECT_WRITE_TOOLS:
            self.assertIn(tool, prompt)

        self.assertIn("Never call disabled tools", prompt)


if __name__ == "__main__":
    unittest.main()
