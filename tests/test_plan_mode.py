"""Tests for planning-mode policy and interactive routing."""

from __future__ import annotations

from copy import deepcopy
import unittest
from io import StringIO
from typing import Any
from unittest.mock import patch

from langchain_core.exceptions import ContextOverflowError
from rich.console import Console
from rich.text import Text

from agent.context_overflow import context_overflow_error, set_context_overflow_notice
from agent import factory
from agent.middleware import ModelToolVisibilityMiddleware, PlanningStageMiddleware
from agent.planning.policy import PLAN_DISABLED_TOOLS, plan_disabled_tools_text, plan_system_prompt
from agent.tools.specs import tool_name
from config.metadata import ModelMetadata
from config.runtime import RuntimeSnapshot
from runtime import runner
from runtime.context_usage import record_deepagents_context_tokens
from ui import repl
from ui.interrupts import research_summary_request
from ui.runtime_snapshot import resources_table, runtime_report, tools_table


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
        self.usage_updates = 0

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

    def text_delta(self, value: str) -> None:
        """Record streamed assistant text."""
        self.console.print(value)

    def reasoning_delta(self, value: str) -> None:
        """Record streamed reasoning text."""
        self.console.print(value)

    def tool_call(self, name: str, args: Any, call_id: str = "") -> None:
        """Record a visible tool call."""
        self.console.print(f"{name}: {args}")

    def tool_result(self, name: str, result: str, call_id: str = "") -> None:
        """Record a visible tool result."""
        self.console.print(f"{name}: {result}")

    def delegation_started(self, calls: list[dict[str, Any]]) -> None:
        """Record delegation."""
        self.console.print(str(calls))

    def subagent_started(self, name: str, task_input: str = "", *, origin: str = "") -> None:
        """Record subagent start."""
        self.console.print(f"{name}: {task_input}")

    def subagent_finished(self, name: str, result: str = "") -> None:
        """Record subagent finish."""
        self.console.print(f"{name}: {result}")

    def compaction_started(self) -> None:
        """Record compaction start."""
        self.console.print("compacting context...")

    def compaction_finished(self) -> None:
        """Record compaction completion."""
        self.console.print("context compacted")

    def finish_main(self) -> None:
        """Record stream finalization."""
        return None

    def usage_updated(self) -> None:
        """Record a live dashboard refresh."""
        self.usage_updates += 1


class FakeModelRequest:
    """Small request object used to test ModelToolVisibilityMiddleware."""

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


class CapturingStore:
    """Store double that snapshots every save."""

    def __init__(self) -> None:
        self.saves: list[dict[str, Any]] = []

    def save(self, record: dict[str, Any]) -> None:
        self.saves.append(deepcopy(record))


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
            patch("agent.middleware.CodeInterpreterMiddleware", return_value="code"),
            patch("agent.middleware.create_mira_summarization_middleware", return_value="auto-summary"),
            patch("agent.middleware.create_mira_summarization_tool_middleware", return_value="summary"),
            patch("agent.factory.create_deep_agent", return_value="agent") as create_deep_agent,
        ):
            agent = factory.build_plan_agent({}, ".", "checkpointer")

        self.assertEqual(agent, "agent")
        kwargs = create_deep_agent.call_args.kwargs
        self.assertIsNone(kwargs["interrupt_on"])
        self.assertIn("planning mode", kwargs["system_prompt"])
        self.assertEqual(kwargs["permissions"][0].operations, ["write"])
        self.assertEqual(kwargs["permissions"][0].mode, "deny")
        self.assertTrue(any(isinstance(item, ModelToolVisibilityMiddleware) for item in kwargs["middleware"]))

    def test_action_agent_keeps_write_interrupts(self) -> None:
        """The action agent should keep write approval interrupts enabled."""
        with (
            patch("agent.factory.get_llm", return_value="model"),
            patch("agent.middleware.CodeInterpreterMiddleware", return_value="code"),
            patch("agent.middleware.create_mira_summarization_middleware", return_value="auto-summary"),
            patch("agent.middleware.create_mira_summarization_tool_middleware", return_value="summary"),
            patch("agent.factory.create_deep_agent", return_value="agent") as create_deep_agent,
        ):
            agent = factory.build_agent({}, ".", "checkpointer")

        self.assertEqual(agent, "agent")
        kwargs = create_deep_agent.call_args.kwargs
        self.assertEqual(kwargs["interrupt_on"], factory._write_interrupts())
        self.assertEqual(kwargs["interrupt_on"]["write_file"]["allowed_decisions"], ["approve", "edit", "reject"])
        self.assertEqual(kwargs["interrupt_on"]["edit_file"]["allowed_decisions"], ["approve", "edit", "reject"])
        self.assertIsNone(kwargs["system_prompt"])
        self.assertEqual(kwargs["permissions"][0].paths, ["/mira-defaults/**"])
        self.assertEqual(kwargs["permissions"][0].mode, "deny")
        self.assertEqual(kwargs["permissions"][1].operations, ["read", "write"])
        self.assertEqual(kwargs["permissions"][1].mode, "allow")
        self.assertTrue(any(tool_name(tool) == "grep" for tool in kwargs["tools"]))

    def test_action_agent_respects_tool_always_allow_settings(self) -> None:
        """Configured always-allow tools should be removed from interrupts."""
        config = {
            "settings": {
                "hitl": {
                    "tools": {
                        "write_file": {"always_allow": True},
                        "edit_file": {"always_allow": False},
                        "web_search": {"always_allow": False},
                    }
                }
            }
        }

        with (
            patch("agent.factory.get_llm", return_value="model"),
            patch("agent.middleware.CodeInterpreterMiddleware", return_value="code"),
            patch("agent.middleware.create_mira_summarization_middleware", return_value="auto-summary"),
            patch("agent.middleware.create_mira_summarization_tool_middleware", return_value="summary"),
            patch("agent.factory.create_deep_agent", return_value="agent") as create_deep_agent,
        ):
            factory.build_agent(config, ".", "checkpointer")

        interrupts = create_deep_agent.call_args.kwargs["interrupt_on"]
        self.assertNotIn("write_file", interrupts)
        self.assertIn("edit_file", interrupts)
        self.assertIn("web_search", interrupts)

    def test_action_agent_disables_dynamic_subagents_by_default(self) -> None:
        """QuickJS should not expose eval-internal task() unless the setting is enabled."""
        with (
            patch("agent.factory.get_llm", return_value="model"),
            patch("agent.middleware.CodeInterpreterMiddleware", return_value="code") as code_middleware,
            patch("agent.middleware.create_mira_summarization_middleware", return_value="auto-summary"),
            patch("agent.middleware.create_mira_summarization_tool_middleware", return_value="summary"),
            patch("agent.factory.create_deep_agent", return_value="agent"),
        ):
            factory.build_agent({}, ".", "checkpointer")

        self.assertFalse(code_middleware.call_args.kwargs["subagents"])

    def test_action_agent_enables_dynamic_subagents_from_settings(self) -> None:
        """The system setting should pass through to CodeInterpreterMiddleware."""
        config = {"settings": {"system": {"dynamic_subagents": {"enabled": True}}}}

        with (
            patch("agent.factory.get_llm", return_value="model"),
            patch("agent.middleware.CodeInterpreterMiddleware", return_value="code") as code_middleware,
            patch("agent.middleware.create_mira_summarization_middleware", return_value="auto-summary"),
            patch("agent.middleware.create_mira_summarization_tool_middleware", return_value="summary"),
            patch("agent.factory.create_deep_agent", return_value="agent"),
        ):
            factory.build_agent(config, ".", "checkpointer")

        self.assertTrue(code_middleware.call_args.kwargs["subagents"])

    def test_rubric_middleware_is_action_only_and_uses_configured_cap(self) -> None:
        """Enabled rubric grading should not leak into the planning agent."""
        config = {"settings": {"system": {"rubric": {"enabled": True, "max_iterations": 5}}}}
        with (
            patch("agent.factory.get_llm", return_value="model"),
            patch("agent.middleware.CodeInterpreterMiddleware", return_value="code"),
            patch("agent.middleware.create_mira_summarization_middleware", return_value="auto-summary"),
            patch("agent.middleware.create_mira_summarization_tool_middleware", return_value="summary"),
            patch("agent.factory.RubricMiddleware", return_value="rubric") as rubric,
            patch("agent.factory.create_deep_agent", return_value=type("Agent", (), {})()) as create,
        ):
            factory.build_agent(config, ".", "checkpointer")
            action_middleware = create.call_args.kwargs["middleware"]
            factory.build_plan_agent(config, ".", "checkpointer")
            plan_kwargs = create.call_args.kwargs

        rubric.assert_called_once_with(model="model", tools=None, max_iterations=5)
        self.assertIn("rubric", action_middleware)
        self.assertNotIn("rubric", plan_kwargs["middleware"])
        self.assertIn("prepare_goal", [tool_name(tool) for tool in plan_kwargs["tools"]])
        self.assertTrue(any(isinstance(item, PlanningStageMiddleware) for item in plan_kwargs["middleware"]))

    def test_disabled_plan_prompt_and_tools_remain_unchanged(self) -> None:
        """The default planning agent must retain its existing prompt and tools."""
        with (
            patch("agent.factory.get_llm", return_value="model"),
            patch("agent.middleware.CodeInterpreterMiddleware", return_value="code"),
            patch("agent.middleware.create_mira_summarization_middleware", return_value="auto-summary"),
            patch("agent.middleware.create_mira_summarization_tool_middleware", return_value="summary"),
            patch("agent.factory.RubricMiddleware") as rubric,
            patch("agent.factory.create_deep_agent", return_value=type("Agent", (), {})()) as create,
        ):
            factory.build_plan_agent({}, ".", "checkpointer")

        rubric.assert_not_called()
        self.assertEqual(create.call_args.kwargs["system_prompt"], factory.PLAN_SYSTEM_PROMPT)
        self.assertNotIn("prepare_goal", [tool_name(tool) for tool in create.call_args.kwargs["tools"]])
        self.assertFalse(
            any(isinstance(item, PlanningStageMiddleware) for item in create.call_args.kwargs["middleware"])
        )

    def test_action_agent_compiles_subagents_when_response_schemas_are_disabled(self) -> None:
        """Schema-free dynamic mode should pass compiled workers to DeepAgents."""
        config = {
            "settings": {
                "system": {
                    "dynamic_subagents": {"enabled": True, "response_schema": False},
                }
            }
        }
        compiled = [{"name": "general-purpose", "description": "Compiled", "runnable": object()}]

        with (
            patch("agent.factory.get_llm", return_value="model"),
            patch("agent.middleware.CodeInterpreterMiddleware", return_value="code"),
            patch("agent.middleware.create_mira_summarization_middleware", return_value="auto-summary"),
            patch("agent.middleware.create_mira_summarization_tool_middleware", return_value="summary"),
            patch("agent.factory.compile_dynamic_subagents", return_value=compiled) as compile_subagents,
            patch("agent.factory.create_deep_agent", return_value="agent") as create_deep_agent,
        ):
            factory.build_agent(config, ".", "checkpointer")

        compile_subagents.assert_called_once()
        self.assertIs(create_deep_agent.call_args.kwargs["subagents"], compiled)

    def test_action_agent_leaves_subagents_raw_when_response_schemas_are_enabled(self) -> None:
        """The compatibility default should leave DeepAgents construction unchanged."""
        config = {
            "settings": {
                "system": {
                    "dynamic_subagents": {"enabled": True, "response_schema": True},
                }
            }
        }

        with (
            patch("agent.factory.get_llm", return_value="model"),
            patch("agent.middleware.CodeInterpreterMiddleware", return_value="code"),
            patch("agent.middleware.create_mira_summarization_middleware", return_value="auto-summary"),
            patch("agent.middleware.create_mira_summarization_tool_middleware", return_value="summary"),
            patch("agent.factory.compile_dynamic_subagents") as compile_subagents,
            patch("agent.factory.create_deep_agent", return_value="agent"),
        ):
            factory.build_agent(config, ".", "checkpointer")

        compile_subagents.assert_not_called()

    def test_agent_build_passes_metadata_before_summarization_middleware(self) -> None:
        """Model metadata should be applied before DeepAgents summarization middleware is created."""
        metadata = ModelMetadata(10000, "test")
        model = type("Model", (), {"profile": {"max_input_tokens": 10000}})()

        with (
            patch("agent.factory.get_llm", return_value=model) as get_llm,
            patch("agent.middleware.CodeInterpreterMiddleware", return_value="code"),
            patch("agent.middleware.create_mira_summarization_middleware", return_value="auto-summary") as auto_summary,
            patch("agent.middleware.create_mira_summarization_tool_middleware", return_value="summary") as summary,
            patch("agent.factory.create_deep_agent", return_value="agent"),
        ):
            factory.build_agent({}, ".", "checkpointer", metadata=metadata)

        get_llm.assert_called_once_with({}, metadata=metadata)
        auto_summary.assert_called_once()
        summary.assert_called_once()
        self.assertIs(auto_summary.call_args.kwargs["model"], model)
        self.assertIs(summary.call_args.kwargs["model"], model)
        self.assertEqual(auto_summary.call_args.kwargs["model"].profile["max_input_tokens"], 10000)

    def test_agent_build_attaches_tool_metadata(self) -> None:
        """Built agents should expose tool metadata for the UI."""
        with (
            patch("agent.factory.get_llm", return_value="model"),
            patch("agent.middleware.CodeInterpreterMiddleware", return_value="code"),
            patch("agent.middleware.create_mira_summarization_middleware", return_value="auto-summary"),
            patch("agent.middleware.create_mira_summarization_tool_middleware", return_value="summary"),
            patch("agent.factory.create_deep_agent", return_value=type("Agent", (), {})()),
        ):
            agent = factory.build_agent({}, ".", "checkpointer")

        names = [tool["name"] for tool in agent.mira_tool_specs]
        self.assertIn("ask_user", names)
        self.assertIn("read_file", names)
        self.assertIn("write_file", names)
        self.assertIn("edit_file", names)
        self.assertIn("grep", names)
        self.assertNotIn("prepare_goal", names)
        self.assertNotIn("present_plan", names)
        grep = next(tool for tool in agent.mira_tool_specs if tool["name"] == "grep")
        self.assertEqual(grep["source"], "default")
        self.assertEqual(grep["replaces"], "built-in")

    def test_plan_agent_metadata_hides_write_tools(self) -> None:
        """Plan agents should hide mutating and delegation tool metadata."""
        with (
            patch("agent.factory.get_llm", return_value="model"),
            patch("agent.middleware.CodeInterpreterMiddleware", return_value="code"),
            patch("agent.middleware.create_mira_summarization_middleware", return_value="auto-summary"),
            patch("agent.middleware.create_mira_summarization_tool_middleware", return_value="summary"),
            patch("agent.factory.create_deep_agent", return_value=type("Agent", (), {})()),
        ):
            agent = factory.build_plan_agent({}, ".", "checkpointer")

        names = [tool["name"] for tool in agent.mira_tool_specs]
        self.assertIn("ask_user", names)
        self.assertIn("present_plan", names)
        self.assertNotIn("prepare_goal", names)
        self.assertIn("read_file", names)
        self.assertNotIn("write_file", names)
        self.assertNotIn("edit_file", names)
        self.assertNotIn("execute", names)
        self.assertNotIn("task", names)
        self.assertNotIn("eval", names)

    def test_action_agent_metadata_hides_disabled_inbuilt_tools(self) -> None:
        """Disabled inbuilt tools should be excluded from action-mode metadata."""
        config = {"settings": {"hitl": {"tools": {"edit_file": {"enabled": False, "always_allow": False}}}}}
        with (
            patch("agent.factory.get_llm", return_value="model"),
            patch("agent.middleware.CodeInterpreterMiddleware", return_value="code"),
            patch("agent.middleware.create_mira_summarization_middleware", return_value="auto-summary"),
            patch("agent.middleware.create_mira_summarization_tool_middleware", return_value="summary"),
            patch("agent.factory.create_deep_agent", return_value=type("Agent", (), {})()),
        ):
            agent = factory.build_agent(config, ".", "checkpointer")

        names = [tool["name"] for tool in agent.mira_tool_specs]
        self.assertIn("write_file", names)
        self.assertNotIn("edit_file", names)
        self.assertIn("edit_file", factory.effective_excluded_tools(config, (), True))

    def test_plan_tool_filter_hides_mutating_and_delegation_tools_from_model(self) -> None:
        """Plan requests should omit every tool disabled by planning policy."""
        middleware = ModelToolVisibilityMiddleware(PLAN_DISABLED_TOOLS)
        request = FakeModelRequest(
            [
                {"name": "read_file"},
                {"name": "write_file"},
                type("Tool", (), {"name": "edit_file"})(),
                {"name": "execute"},
                {"name": "task"},
                {"name": "eval"},
                type("Tool", (), {"name": "grep"})(),
            ]
        )

        filtered = middleware._filter_request(request)

        names = [tool_name(tool) for tool in filtered.tools]
        self.assertEqual(names, ["read_file", "grep"])

    def test_action_tool_filter_hides_present_plan_from_model(self) -> None:
        """The action agent should not expose the structured planning tool."""
        middleware = ModelToolVisibilityMiddleware(factory.ACTION_EXCLUDED_TOOLS)
        request = FakeModelRequest(
            [
                {"name": "read_file"},
                {"name": "present_plan"},
                {"name": "write_file"},
            ]
        )

        filtered = middleware._filter_request(request)

        names = [tool_name(tool) for tool in filtered.tools]
        self.assertEqual(names, ["read_file", "write_file"])

    def test_available_tools_are_mode_specific_for_present_plan_and_execute(self) -> None:
        """Fallback tool display should keep plan-only and execute-only boundaries."""
        action_names = [tool["name"] for tool in repl.available_tools({}, planning=False)]
        planning_names = [tool["name"] for tool in repl.available_tools({}, planning=True)]

        self.assertNotIn("present_plan", action_names)
        self.assertIn("present_plan", planning_names)
        for tool in PLAN_DISABLED_TOOLS:
            self.assertNotIn(tool, planning_names)

    def test_rubric_planning_tool_display_matches_current_stage(self) -> None:
        mode = {
            "rubric_enabled": True,
            "planning_tools": [
                {"name": "read_file"},
                {"name": "ask_user"},
                {"name": "prepare_goal"},
                {"name": "present_plan"},
            ],
            "planning_stage": "research",
        }

        research = [tool["name"] for tool in repl.available_tools(mode, planning=True)]
        mode["planning_stage"] = "finalize"
        finalize = [tool["name"] for tool in repl.available_tools(mode, planning=True)]

        self.assertEqual(research, ["read_file", "ask_user", "prepare_goal"])
        self.assertEqual(finalize, ["present_plan"])

    async def test_plan_and_act_commands_toggle_mode(self) -> None:
        """Slash commands should switch between planning and action modes."""
        renderer = RecordingRenderer()
        mode: dict[str, Any] = {"planning": False}
        session = {"id": "thread-1", "workspace": ".", "turns": 0}

        handled = await repl.handle_command("/plan", renderer, session, "model", mode)
        self.assertTrue(handled)
        self.assertTrue(mode["planning"])
        self.assertIn("planning mode", renderer.console.lines[-1])
        self.assertIn(f"{plan_disabled_tools_text()} disabled", renderer.console.lines[-1])

        handled = await repl.handle_command("/act", renderer, session, "model", mode)
        self.assertTrue(handled)
        self.assertFalse(mode["planning"])
        self.assertIn("action mode", renderer.console.lines[-1])

    async def test_plan_command_primes_resume_context_for_saved_plans(self) -> None:
        """Starting a fresh planning thread should carry recent plan context."""
        renderer = RecordingRenderer()
        mode: dict[str, Any] = {"planning": False}
        session = {
            "id": "thread-1",
            "workspace": ".",
            "turns": 1,
            "events": [
                {
                    "id": 1,
                    "type": "plan",
                    "status": "approved for implementation",
                    "plan": {
                        "id": "plan-1",
                        "title": "Palindrome Plan",
                        "summary": ["Create palindrome.py."],
                        "key_changes": ["Add is_palindrome."],
                        "test_plan": ["Run python palindrome.py."],
                        "assumptions": ["Use Python."],
                    },
                }
            ],
        }
        calls: list[str] = []

        async def fake_run_turn(
            agent: Any,
            text: str,
            renderer: Any,
            thread_id: str,
            **kwargs: Any,
        ) -> runner.TurnResult:
            calls.append(text)
            return runner.TurnResult(final_text="done")

        handled = await repl.handle_command("/plan", renderer, session, "model", mode)
        self.assertTrue(handled)
        self.assertTrue(session["resume_context_pending"])

        with patch("ui.repl.run_turn", fake_run_turn):
            await repl.run_user_turn(
                agent="action-agent",
                plan_agent="plan-agent",
                renderer=renderer,
                store=FakeStore(),
                session=session,
                mode=mode,
                text="show me the previous plan",
            )

        self.assertIn("Recent structured plans:", calls[0])
        self.assertIn("plan-1 (approved for implementation): Palindrome Plan", calls[0])
        self.assertIn("Current user request:", calls[0])
        self.assertFalse(session.get("resume_context_pending"))

    async def test_act_does_not_queue_current_plan(self) -> None:
        """Leaving planning mode should not queue plan execution."""
        renderer = RecordingRenderer()
        mode = {"planning": True, "current_plan": {"title": "Do the thing"}}

        handled = await repl.handle_command("/act", renderer, {}, "model", mode)

        self.assertTrue(handled)
        self.assertFalse(mode["planning"])
        self.assertNotIn("approved_plan", mode)
        self.assertIn("action mode", renderer.console.lines[-1])

    async def test_help_includes_plan_commands(self) -> None:
        """The help command should describe available commands."""
        renderer = RecordingRenderer()

        handled = await repl.handle_command("/help", renderer, {}, "model", {"planning": False})

        self.assertTrue(handled)
        self.assertEqual(len(renderer.console.lines), 1)
        output = "\n".join(renderer.console.lines)
        self.assertIn("Commands", output)
        self.assertIn("/plan", output)
        self.assertIn("enter safe planning mode", output)
        self.assertIn("/act", output)
        self.assertIn("return to action mode", output)
        self.assertIn("/settings", output)
        self.assertIn("/runtime", output)
        self.assertIn("/tools", output)
        self.assertIn("/memories", output)
        self.assertIn("/skills", output)
        self.assertIn("/subagents", output)
        self.assertIn("/session", output)
        self.assertIn("/reload", output)
        for section in ("General", "Inspect", "Workflow", "Configuration", "Chat & history"):
            self.assertIn(section, output)
        self.assertNotIn("/config", output)
        self.assertNotIn("/model", output)

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
        self.assertIn("Inspect", rendered)
        self.assertIn("/runtime", rendered)
        self.assertIn("/tools", rendered)
        self.assertIn("/session", rendered)
        self.assertNotIn("/model", rendered)
        section_cells = [cell for cell in table.columns[0]._cells if isinstance(cell, Text)]
        self.assertTrue(section_cells)
        self.assertTrue(all(str(cell.style) == repl.HELP_SECTION_STYLE for cell in section_cells))

    async def test_runtime_command_is_recognized_as_textual_only(self) -> None:
        """The shared router should advertise TUI-only runtime inspection."""
        renderer = RecordingRenderer()

        handled = await repl.handle_command("/runtime", renderer, {}, "test-model", {"planning": False})

        self.assertTrue(handled)
        self.assertEqual(renderer.console.lines, ["/runtime is available in the Textual app"])

    async def test_model_command_is_removed(self) -> None:
        """The model alias should not duplicate the focused runtime command."""
        renderer = RecordingRenderer()

        handled = await repl.handle_command("/model", renderer, {}, "model", {"planning": False})

        self.assertTrue(handled)
        self.assertEqual(renderer.console.lines, ["unknown command: /model"])

    async def test_tools_command_lists_current_mode_tools(self) -> None:
        """The tools command should render one table for the active mode."""
        renderer = RecordingRenderer()

        handled = await repl.handle_command("/tools", renderer, {}, "model", {"planning": False})

        self.assertTrue(handled)
        self.assertEqual(len(renderer.console.lines), 1)
        output = renderer.console.lines[0]
        self.assertIn("Tools (action)", output)
        self.assertIn("read_file", output)
        self.assertIn("write_file", output)

    async def test_resource_commands_render_one_section_each(self) -> None:
        """Each resource command should produce one table with an explicit empty state."""
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
                "skills": [],
                "subagents": [],
            }
        }
        outputs = {}
        for command in ("/memories", "/skills", "/subagents"):
            renderer = RecordingRenderer()
            self.assertTrue(await repl.handle_command(command, renderer, {}, "model", mode))
            self.assertEqual(len(renderer.console.lines), 1)
            outputs[command] = renderer.console.lines[0]

        self.assertIn("AGENTS.md", outputs["/memories"])
        self.assertIn("none loaded", outputs["/skills"])
        self.assertIn("none loaded", outputs["/subagents"])

    def test_available_tools_lists_action_tools(self) -> None:
        """Runtime reporting should receive the action-mode tool projection."""
        names = {tool["name"] for tool in repl.available_tools({"planning": False}, planning=False)}

        self.assertTrue({"ask_user", "read_file", "write_file", "edit_file", "task"} <= names)

    def test_available_tools_hides_write_tools_in_planning_mode(self) -> None:
        """Runtime reporting should receive the planning-mode tool projection."""
        names = {tool["name"] for tool in repl.available_tools({"planning": True}, planning=True)}

        self.assertIn("ask_user", names)
        self.assertIn("read_file", names)
        for tool in PLAN_DISABLED_TOOLS:
            self.assertNotIn(tool, names)

    def test_runtime_report_shows_sanitized_runtime_details(self) -> None:
        """The runtime report should stay focused on model and connection state."""
        report = runtime_report(
            RuntimeSnapshot(
                model_name="test-model",
                provider="lmstudio",
                endpoint="http://localhost:1234/v1",
                direct_effective=False,
                direct_requested=False,
            )
        )
        rendered = StringIO()
        Console(file=rendered, force_terminal=False, width=120).print(report)
        output = rendered.getvalue()

        self.assertIn("Runtime", output)
        self.assertIn("test-model", output)
        self.assertIn("lmstudio", output)
        self.assertIn("http://localhost:1234/v1", output)
        self.assertIn("-d / --direct", output)
        self.assertNotIn("Tools", output)
        self.assertNotIn("Memories", output)

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
        table = tools_table(
            [{"name": "long_tool", "description": "This description should wrap when the width is narrow."}],
            planning=False,
        )

        output = StringIO()
        Console(file=output, force_terminal=False, width=80).print(table)
        rendered = output.getvalue()
        self.assertIn("Tools (action)", rendered)
        self.assertIn("long_tool", rendered)
        self.assertIn("Description", rendered)

    def test_tool_table_shows_source_and_replacement(self) -> None:
        """The tools table should show custom tool source and replacement info."""
        table = tools_table(
            [
                {
                    "name": "grep",
                    "description": "Search with regex.",
                    "source": "default",
                    "replaces": "built-in",
                }
            ],
            planning=False,
        )

        output = StringIO()
        Console(file=output, force_terminal=False, width=80).print(table)
        rendered = output.getvalue()
        self.assertIn("grep", rendered)
        self.assertIn("default", rendered)
        self.assertIn("built-in", rendered)

    def test_resources_table_returns_rich_table(self) -> None:
        """The resources table should expose source and replacement columns."""
        table = resources_table(
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

    async def test_destructive_clear_commands_are_textual_only_in_repl_handler(self) -> None:
        """Durable clear commands should not run without Textual confirmation support."""
        for command in ("/clear-chat", "/clear-all-chats", "/clear-errors", "/clear-prompts"):
            renderer = RecordingRenderer()

            handled = await repl.handle_command(command, renderer, {}, "model", {"planning": False})

            self.assertTrue(handled)
            self.assertIn(f"{command} is available in the Textual app with confirmation", renderer.console.lines)

    async def test_run_user_turn_routes_to_plan_agent_while_planning(self) -> None:
        """User text in planning mode should go to the planning agent."""
        renderer = RecordingRenderer()
        session = {"id": "thread-1", "workspace": ".", "turns": 0}
        store = FakeStore()
        mode = repl.initial_mode("action-agent", "plan-agent")
        calls: list[tuple[Any, str, str]] = []

        async def fake_run_turn(
            agent: Any,
            text: str,
            renderer: Any,
            thread_id: str,
            **kwargs: Any,
        ) -> runner.TurnResult:
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
        self.assertIn(f"The following tools are disabled: {plan_disabled_tools_text()}.", calls[0][1])
        self.assertIn("User request:\nwrite a file", calls[0][1])
        self.assertEqual(calls[0][2], "thread-1:plan:1")
        self.assertEqual(calls[1], ("action-agent", "write it now", "thread-1"))
        self.assertEqual(session["turns"], 2)

    async def test_run_user_turn_injects_approved_plan_once(self) -> None:
        """An explicitly approved plan should be injected into one action request."""
        renderer = RecordingRenderer()
        session = {"id": "thread-1", "workspace": ".", "turns": 0}
        store = FakeStore()
        mode = repl.initial_mode("action-agent", "plan-agent")
        mode["approved_plan"] = {
            "title": "Create file",
            "summary": ["Create test.txt with hello world."],
            "key_changes": ["Write the file."],
            "test_plan": ["Run the focused file creation check."],
            "assumptions": ["Use the root directory."],
        }
        calls: list[tuple[Any, str, str]] = []
        results = [
            runner.TurnResult(final_text="done"),
            runner.TurnResult(final_text="done again"),
        ]

        async def fake_run_turn(
            agent: Any,
            text: str,
            renderer: Any,
            thread_id: str,
            **kwargs: Any,
        ) -> runner.TurnResult:
            """Record agent invocations and return scripted results."""
            calls.append((agent, text, thread_id))
            return results.pop(0)

        with patch("ui.repl.run_turn", fake_run_turn):
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

        self.assertEqual(calls[0][0], "action-agent")
        self.assertIn("Previous planning context:", calls[0][1])
        self.assertIn("Create test.txt with hello world.", calls[0][1])
        self.assertIn("You are now in action mode.", calls[0][1])
        self.assertIn("Write/edit tools are available again", calls[0][1])
        self.assertIn("User request:\ndo it", calls[0][1])
        self.assertEqual(calls[1], ("action-agent", "do another thing", "thread-1"))

    async def test_run_user_turn_distinguishes_omitted_cleared_and_supplied_rubric(self) -> None:
        renderer = RecordingRenderer()
        session = {"id": "thread-1", "workspace": ".", "turns": 0}
        store = FakeStore()
        disabled = repl.initial_mode("action-agent", "plan-agent")
        enabled = repl.initial_mode(
            "action-agent",
            "plan-agent",
            {"system": {"rubric": {"enabled": True, "max_iterations": 4}}},
        )
        calls: list[dict[str, Any]] = []

        async def fake_run_turn(**kwargs: Any) -> runner.TurnResult:
            calls.append(kwargs)
            return runner.TurnResult()

        with patch("ui.repl.run_turn", fake_run_turn):
            await repl.run_user_turn(
                agent="action-agent",
                plan_agent="plan-agent",
                renderer=renderer,
                store=store,
                session=session,
                mode=disabled,
                text="ordinary disabled turn",
            )
            await repl.handle_command("/plan", renderer, session, "model", enabled)
            await repl.run_user_turn(
                agent="action-agent",
                plan_agent="plan-agent",
                renderer=renderer,
                store=store,
                session=session,
                mode=enabled,
                text="research a plan",
            )
            enabled["planning"] = False
            await repl.run_user_turn(
                agent="action-agent",
                plan_agent="plan-agent",
                renderer=renderer,
                store=store,
                session=session,
                mode=enabled,
                text="ordinary enabled turn",
            )
            session["active_goal"] = {
                "proposal_id": "proposal-1",
                "id": "proposal-1",
                "objective": "Build it.",
                "criteria": "- It works.",
                "plan": {"title": "Build", "summary": ["Build it."]},
                "origin": "goal_command",
                "rubric_iterations": 4,
                "status": "active",
                "last_rubric_status": "",
            }
            await repl.run_user_turn(
                agent="action-agent",
                plan_agent="plan-agent",
                renderer=renderer,
                store=store,
                session=session,
                mode=enabled,
                text="Build it.",
            )

        self.assertFalse(calls[0]["include_rubric_state"])
        self.assertFalse(calls[1]["include_rubric_state"])
        self.assertEqual(calls[1]["planning_stage"], "research")
        self.assertIn("call prepare_goal", calls[1]["text"])
        self.assertNotIn("Fill every present_plan section", calls[1]["text"])
        self.assertTrue(calls[2]["include_rubric_state"])
        self.assertIsNone(calls[2]["rubric"])
        self.assertEqual(calls[3]["rubric"], "- It works.")
        self.assertEqual(calls[3]["rubric_max_iterations"], 4)

    async def test_rubric_plan_revision_starts_in_finalize_stage(self) -> None:
        renderer = RecordingRenderer()
        session = {"id": "thread-1", "workspace": ".", "turns": 0}
        mode = repl.initial_mode(
            "action-agent",
            "plan-agent",
            {"system": {"rubric": {"enabled": True, "max_iterations": 3}}},
        )
        mode.update(
            {
                "planning": True,
                "planning_stage": "finalize",
                "plan_thread_id": "thread-1:plan:revision",
            }
        )
        captured: dict[str, Any] = {}

        async def fake_run_turn(**kwargs: Any) -> runner.TurnResult:
            captured.update(kwargs)
            return runner.TurnResult()

        with patch("ui.repl.run_turn", fake_run_turn):
            await repl.run_user_turn(
                agent="action-agent",
                plan_agent="plan-agent",
                renderer=renderer,
                store=FakeStore(),
                session=session,
                mode=mode,
                text="revise with existing criteria",
            )

        self.assertEqual(captured["agent"], "plan-agent")
        self.assertEqual(captured["planning_stage"], "finalize")

    async def test_run_user_turn_applies_live_usage_once(self) -> None:
        """Interactive usage callbacks should refresh dashboard without final double-counting."""
        renderer = RecordingRenderer()
        session = {"id": "thread-1", "workspace": ".", "turns": 0, "events": []}
        store = CapturingStore()
        mode = repl.initial_mode("action-agent", "plan-agent")
        pre_generation_state: dict[str, Any] = {}

        async def fake_run_turn(
            agent: Any,
            text: str,
            renderer: Any,
            thread_id: str,
            usage_callback: Any | None = None,
            **kwargs: Any,
        ) -> runner.TurnResult:
            pre_generation_state.update(deepcopy(session["dashboard"]))
            usage = {
                "input_tokens": 8200,
                "output_tokens": 1424,
                "total_tokens": 9624,
                "context_tokens": 9624,
                "source": "usage_metadata",
            }
            if usage_callback is not None:
                usage_callback(usage)
            record_deepagents_context_tokens(9624)
            result = runner.TurnResult()
            result.add_usage(usage)
            return result

        with patch("ui.repl.run_turn", fake_run_turn):
            await repl.run_user_turn(
                agent="action-agent",
                plan_agent="plan-agent",
                renderer=renderer,
                store=store,
                session=session,
                mode=mode,
                text="use tokens",
                model_name="lmstudio:test",
                context_limit_tokens=1000,
            )

        self.assertEqual(pre_generation_state["tokens"], {"in": 0, "out": 0})
        self.assertEqual(pre_generation_state["context"]["used_tokens"], 0)
        self.assertEqual(pre_generation_state["context"]["limit_tokens"], 1000)
        self.assertEqual(session["dashboard"]["tokens"], {"in": 8200, "out": 1424})
        self.assertEqual(session["dashboard"]["context"]["used_tokens"], 9624)
        self.assertEqual(session["turns"], 1)
        self.assertEqual(renderer.usage_updates, 2)
        self.assertEqual(store.saves[-1]["turns"], 1)

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

        async def fake_run_turn(
            agent: Any,
            text: str,
            renderer: Any,
            thread_id: str,
            **kwargs: Any,
        ) -> runner.TurnResult:
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

        self.assertEqual(calls[1], ("action-agent", "do it", "thread-1"))

    async def test_run_user_turn_persists_visible_events_before_failure(self) -> None:
        """In-flight user, assistant, and tool events should survive failed turns."""
        renderer = RecordingRenderer()
        session = {"id": "thread-1", "workspace": ".", "turns": 0, "events": []}
        store = CapturingStore()
        mode = repl.initial_mode("action-agent", "plan-agent")

        async def fake_run_turn(
            agent: Any,
            text: str,
            renderer: Any,
            thread_id: str,
            **kwargs: Any,
        ) -> runner.TurnResult:
            renderer.text_delta("working")
            renderer.tool_call("read_file", {"path": "missing.txt"}, call_id="call-read")
            renderer.completed_tool_error("read_file", "file not found", call_id="call-read")
            raise RuntimeError("model stopped")

        with patch("ui.repl.run_turn", fake_run_turn):
            with self.assertRaisesRegex(RuntimeError, "model stopped"):
                await repl.run_user_turn(
                    agent="action-agent",
                    plan_agent="plan-agent",
                    renderer=renderer,
                    store=store,
                    session=session,
                    mode=mode,
                    text="inspect the repo",
                )

        self.assertEqual(session["turns"], 0)
        self.assertGreaterEqual(len(store.saves), 4)
        event_types = [event["type"] for event in store.saves[-1]["events"]]
        self.assertEqual(event_types, ["user", "assistant", "tool_call", "tool_result", "system_error"])
        self.assertEqual(store.saves[0]["events"][0]["text"], "inspect the repo")
        self.assertEqual(store.saves[-1]["events"][1]["text"], "working")
        self.assertEqual(store.saves[-1]["events"][2]["name"], "read_file")
        self.assertEqual(store.saves[-1]["events"][3]["status"], "error")
        self.assertIn("file not found", store.saves[-1]["events"][3]["output"])
        self.assertIn("model stopped", store.saves[-1]["events"][4]["text"])

    async def test_run_user_turn_records_context_overflow_as_info(self) -> None:
        """Escaped context overflow should persist as info instead of system_error."""
        renderer = RecordingRenderer()
        session = {"id": "thread-1", "workspace": ".", "turns": 0, "events": []}
        store = CapturingStore()
        mode = repl.initial_mode("action-agent", "plan-agent")
        notice = "Context limit pressure detected. Compacting older context and retrying."

        async def fake_run_turn(
            agent: Any,
            text: str,
            renderer: Any,
            thread_id: str,
            **kwargs: Any,
        ) -> runner.TurnResult:
            raise context_overflow_error("provider context limit reached", notice)

        with patch("ui.repl.run_turn", fake_run_turn):
            with self.assertRaises(ContextOverflowError):
                await repl.run_user_turn(
                    agent="action-agent",
                    plan_agent="plan-agent",
                    renderer=renderer,
                    store=store,
                    session=session,
                    mode=mode,
                    text="inspect the repo",
                )

        event_types = [event["type"] for event in store.saves[-1]["events"]]
        self.assertEqual(event_types, ["user", "info"])
        self.assertEqual(store.saves[-1]["events"][1]["text"], notice)
        self.assertNotIn("MIRA simulated a context overflow", "\n".join(renderer.console.lines))

    async def test_run_user_turn_continues_after_compaction_notice(self) -> None:
        """A successful DeepAgents compaction should not stop the visible answer."""
        renderer = RecordingRenderer()
        session = {"id": "thread-1", "workspace": ".", "turns": 0, "events": []}
        store = CapturingStore()
        mode = repl.initial_mode("action-agent", "plan-agent")
        notice = "Provider context limit reached. Compacting older context and retrying."

        async def fake_run_turn(
            agent: Any,
            text: str,
            renderer: Any,
            thread_id: str,
            **kwargs: Any,
        ) -> runner.TurnResult:
            set_context_overflow_notice(notice)
            renderer.compaction_started()
            renderer.compaction_finished()
            renderer.text_delta("The story continued.")
            return runner.TurnResult(
                final_text="The story continued.",
                input_tokens=8200,
                output_tokens=50,
                total_tokens=8250,
                context_tokens=8250,
                usage_source="usage_metadata",
            )

        with patch("ui.repl.run_turn", fake_run_turn):
            result = await repl.run_user_turn(
                agent="action-agent",
                plan_agent="plan-agent",
                renderer=renderer,
                store=store,
                session=session,
                mode=mode,
                text="hello",
            )

        self.assertEqual(result.final_text, "The story continued.")
        event_types = [event["type"] for event in session["events"]]
        self.assertEqual(event_types, ["user", "info", "assistant"])
        self.assertEqual(sum(1 for event in session["events"] if event["type"] == "info"), 1)
        self.assertEqual(session["events"][-1]["text"], "The story continued.")
        self.assertEqual(session["turns"], 1)
        rendered = "\n".join(renderer.console.lines)
        self.assertEqual(rendered.count("Provider context limit reached"), 1)
        self.assertIn("context compacted", rendered)

    async def test_session_command_shows_current_mode(self) -> None:
        """The session command should print conversation-derived state."""
        renderer = RecordingRenderer()
        session = {"id": "thread-1", "workspace": ".", "turns": 3}

        handled = await repl.handle_command("/session", renderer, session, "model", {"planning": True})

        self.assertTrue(handled)
        self.assertEqual(len(renderer.console.lines), 1)
        self.assertIn("session: thread-1", renderer.console.lines[0])
        self.assertIn("mode: planning", renderer.console.lines[0])
        self.assertIn("active goal: no", renderer.console.lines[0])
        self.assertIn("active plan: no", renderer.console.lines[0])

    def test_session_summary_detects_goals_and_both_plan_shapes(self) -> None:
        """Goals and plans should follow the active conversation proposal state."""
        session = {
            "id": "thread-1",
            "title": "Runtime cleanup",
            "workspace": ".",
            "turns": 3,
        }
        cases = (
            ({"planning": False}, None, "no", "no"),
            ({"planning": True, "current_plan": {"title": "Ordinary"}}, None, "no", "yes"),
            ({"planning": True, "current_proposal": {"plan": {"title": "Proposal"}}}, None, "no", "yes"),
            (
                {"planning": False},
                {
                    "proposal_id": "proposal-1",
                    "objective": "Build it.",
                    "criteria": "- Done.",
                    "plan": {"title": "Build"},
                    "status": "active",
                },
                "yes",
                "yes",
            ),
        )

        for mode, active_goal, expected_goal, expected_plan in cases:
            with self.subTest(mode=mode):
                session["active_goal"] = active_goal
                summary = repl.session_summary_text(session, mode)
                self.assertIn(f"active goal: {expected_goal}", summary)
                self.assertIn(f"active plan: {expected_plan}", summary)
                self.assertIn("title: Runtime cleanup", summary)
                self.assertIn("workspace: .", summary)
                self.assertIn("turns: 3", summary)

    def test_plan_thread_id_is_separate_from_action_thread(self) -> None:
        """Planning threads should be isolated from action memory."""
        self.assertEqual(repl.plan_thread_id({"id": "thread-1"}), "thread-1:plan")
        self.assertEqual(repl.plan_thread_id({"id": "thread-1"}, 2), "thread-1:plan:2")

    def test_action_request_text_clears_approved_plan(self) -> None:
        """The approved plan should be consumed only once."""
        mode = {
            "planning": False,
            "approved_plan": {
                "title": "Plan text",
                "summary": ["Do the thing."],
                "key_changes": ["Update the implementation."],
                "test_plan": ["Run focused checks."],
                "assumptions": ["No extra assumptions."],
            },
        }

        text = repl.action_request_text(mode, "Implement")

        self.assertIn("Previous planning context:", text)
        self.assertIn("Title: Plan text", text)
        self.assertIn("- Do the thing.", text)
        self.assertIn("Test Plan:\n- Run focused checks.", text)
        self.assertIn("Do not assume planning-mode permission errors still apply.", text)
        self.assertIn("Use a todo/checklist", text)
        self.assertIn("Run every feasible Test Plan command/check after implementation.", text)
        self.assertIn("state exactly which one was skipped and why", text)
        self.assertIn("User request:\nImplement", text)
        self.assertIsNone(mode["approved_plan"])

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
        self.assertIn("use normal assistant messages", text)
        self.assertIn("Never ask a user-facing question in a normal assistant message", text)
        self.assertIn("not limited to software or coding work", text)
        self.assertIn("outcomes, scope, audience, priorities, behavior, presentation", text)
        self.assertIn("separate discoverable facts from user preferences", text)
        self.assertIn("multiple reasonable meanings", text)
        self.assertIn("call ask_user immediately", text)
        self.assertIn("A recommendation does not remove the need to call ask_user", text)
        self.assertIn("ask_user is an intermediate planning step, never the final outcome", text)
        self.assertIn("treat that answer as context for the original IMPLEMENTATION request", text)
        self.assertIn("Never present alternatives in prose", text)
        self.assertIn("preserve them as separate, mutually exclusive ask_user options", text)
        self.assertIn("Do not merge, replace, or invent alternatives", text)
        self.assertIn("classify the current user request as exactly one", text)
        self.assertIn("SAFE_CONVERSATION", text)
        self.assertIn("IMPLEMENTATION", text)
        self.assertIn("A normal assistant message is not a valid final outcome", text)
        self.assertIn("Do not wait for the user to say 'show me the plan'", text)
        self.assertIn("you must call present_plan", text)
        self.assertIn(f"The following tools are disabled: {plan_disabled_tools_text()}.", text)
        self.assertIn("User request:\nwrite a file", text)
        self.assertGreater(text.index("Before returning, check the intent you classified"), text.index("User request:\nwrite a file"))
        self.assertIn("repository research and prose findings are intermediate work", text)
        self.assertIn("move the decision plus 1-3 choices into ask_user", text)
        self.assertIn("does not change IMPLEMENTATION into SAFE_CONVERSATION", text)
        self.assertIn("Never end an IMPLEMENTATION turn with assistant prose or a user-facing question", text)
        self.assertIn("Fill every present_plan section", text)
        self.assertIn("Use this exact content template when calling present_plan", text)
        self.assertIn("as JSON arrays of strings, never as single strings", text)
        self.assertIn("Goal: the user-visible outcome", text)
        self.assertIn("Run: exact command/check to execute after implementation.", text)
        self.assertIn("Do not use vague Test Plan items", text)
        self.assertIn("If execute is unavailable", text)

    def test_rubric_research_request_ends_with_prepare_goal_contract(self) -> None:
        text = repl.plan_request_text("write a file", rubric_research=True)

        self.assertIn("rubric-enabled planning research mode", text)
        self.assertIn("call prepare_goal", text)
        self.assertIn("present_plan is unavailable during this research stage", text)
        self.assertNotIn("Fill every present_plan section", text)

    def test_plan_revision_text_includes_old_plan_and_feedback(self) -> None:
        """Revision requests should not depend on planning-thread memory."""
        text = repl.plan_revision_text(
            {
                "title": "Palindrome Plan",
                "summary": ["Create a helper."],
                "key_changes": ["Add palindrome.py."],
                "test_plan": ["Add unit tests."],
                "assumptions": ["Use Python."],
            },
            "include a testing plan",
        )

        self.assertIn("Revise this structured plan.", text)
        self.assertIn("Title: Palindrome Plan", text)
        self.assertIn("Summary:\n- Create a helper.", text)
        self.assertIn("Key Changes:\n- Add palindrome.py.", text)
        self.assertIn("Test Plan:\n- Add unit tests.", text)
        self.assertIn("Assumptions:\n- Use Python.", text)
        self.assertIn("User feedback:\ninclude a testing plan", text)
        self.assertIn("Create a revised plan using present_plan", text)

    def test_rubric_plan_revision_keeps_fields_delimited_and_feedback_identical(self) -> None:
        text = repl.rubric_plan_revision_text(
            {
                "objective": "Build search.",
                "criteria": "- Ranked results are returned.",
                "plan": {"title": "Search", "summary": ["Add ranking."]},
            },
            "Make the plan shorter.",
        )

        self.assertIn("<objective>\nBuild search.\n</objective>", text)
        self.assertIn("<previous_plan>\nTitle: Search", text)
        self.assertIn("<previous_definition_of_done>\n- Ranked results are returned.", text)
        self.assertIn("<user_feedback>\nMake the plan shorter.\n</user_feedback>", text)
        self.assertIn("then call prepare_goal", text)

    def test_prepare_goal_research_handoff_is_bounded(self) -> None:
        summary = research_summary_request(
            {"type": "prepare_goal", "research_summary": "x" * 5000}
        )

        self.assertEqual(len(summary), 4000)

    async def test_explicit_goal_proposal_uses_plan_agent_without_persistent_plan_mode(self) -> None:
        renderer = RecordingRenderer()
        session = {"id": "thread-1", "workspace": ".", "turns": 0}
        mode = repl.initial_mode(
            "action-agent",
            "plan-agent",
            {"system": {"rubric": {"enabled": True, "max_iterations": 3}}},
        )
        mode["proposal_run"] = {
            "origin": "goal_command",
            "thread_id": "thread-1:plan:1",
            "stage": "research",
            "explicit": True,
        }
        captured: dict[str, Any] = {}

        async def fake_run_turn(**kwargs: Any) -> runner.TurnResult:
            captured.update(kwargs)
            return runner.TurnResult()

        with patch("ui.repl.run_turn", fake_run_turn):
            await repl.run_user_turn(
                agent="action-agent",
                plan_agent="plan-agent",
                renderer=renderer,
                store=FakeStore(),
                session=session,
                mode=mode,
                text="Write an announcement.",
            )

        self.assertFalse(mode["planning"])
        self.assertEqual(captured["agent"], "plan-agent")
        self.assertIn("Do not classify it as SAFE_CONVERSATION", captured["text"])
        self.assertIn("call prepare_goal immediately", captured["text"])

    async def test_active_goal_continues_after_exhaustion_and_completes_when_satisfied(self) -> None:
        renderer = RecordingRenderer()
        session = {
            "id": "thread-1",
            "workspace": ".",
            "turns": 0,
            "active_goal": {
                "proposal_id": "proposal-1",
                "objective": "Build search.",
                "criteria": "- Search works.",
                "plan": {"title": "Search", "summary": ["Add search."]},
                "origin": "goal_command",
                "rubric_iterations": 2,
                "status": "active",
                "last_rubric_status": "",
            },
            "events": [],
        }
        mode = repl.initial_mode(
            "action-agent",
            "plan-agent",
            {"system": {"rubric": {"enabled": True, "max_iterations": 2}}},
        )
        statuses = iter(("max_iterations_reached", "satisfied", ""))
        calls: list[dict[str, Any]] = []

        async def fake_run_turn(**kwargs: Any) -> runner.TurnResult:
            calls.append(kwargs)
            return runner.TurnResult(rubric_status=next(statuses))

        with patch("ui.repl.run_turn", fake_run_turn):
            await repl.run_user_turn(
                agent="action-agent",
                plan_agent="plan-agent",
                renderer=renderer,
                store=FakeStore(),
                session=session,
                mode=mode,
                text="Continue where we left off.",
            )
            self.assertEqual(session["active_goal"]["status"], "active")
            self.assertEqual(session["active_goal"]["last_rubric_status"], "max_iterations_reached")
            await repl.run_user_turn(
                agent="action-agent",
                plan_agent="plan-agent",
                renderer=renderer,
                store=FakeStore(),
                session=session,
                mode=mode,
                text="Continue.",
            )
            self.assertEqual(session["active_goal"]["status"], "complete")
            await repl.run_user_turn(
                agent="action-agent",
                plan_agent="plan-agent",
                renderer=renderer,
                store=FakeStore(),
                session=session,
                mode=mode,
                text="Unrelated question.",
            )

        self.assertEqual(calls[0]["rubric"], "- Search works.")
        self.assertEqual(calls[1]["rubric"], "- Search works.")
        self.assertIn("<approved_plan>", calls[1]["text"])
        self.assertIsNone(calls[2]["rubric"])
        self.assertEqual(calls[2]["agent"], "action-agent")

    async def test_plans_command_is_removed(self) -> None:
        """The old saved-plan command should no longer be handled specially."""
        renderer = RecordingRenderer()

        handled = await repl.handle_command("/plans", renderer, {}, "model", {})

        self.assertTrue(handled)
        self.assertIn("unknown command: /plans", renderer.console.lines[-1])

    def test_plan_policy_drives_prompt_and_repl_validation(self) -> None:
        """The policy constants should appear in the planning prompt."""
        prompt = plan_system_prompt()

        for tool in PLAN_DISABLED_TOOLS:
            self.assertIn(tool, prompt)

        self.assertIn("Never call disabled tools", prompt)
        self.assertIn("Never ask a user-facing question in a normal assistant message", prompt)
        self.assertIn("before using tools or answering, classify", prompt)
        self.assertIn("Do not classify by punctuation, keywords alone, or regex-style text matching", prompt)
        self.assertNotIn("find a way to make the code base neater", prompt.lower())
        self.assertNotIn("make this neater", prompt.lower())
        self.assertIn("IMPLEMENTATION must call ask_user", prompt)
        self.assertIn("Test Plan bullets", prompt)
        self.assertIn("Use this exact content template when calling present_plan", prompt)
        self.assertIn("Success criteria", prompt)
        self.assertIn("Run: exact command/check to execute after implementation.", prompt)
        self.assertIn("Do not use vague Test Plan items", prompt)
        self.assertIn("If execute is unavailable", prompt)


if __name__ == "__main__":
    unittest.main()
