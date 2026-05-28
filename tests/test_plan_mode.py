import unittest
from unittest.mock import patch

from agent import factory
from agent.plan_policy import PLAN_PROJECT_WRITE_TOOLS, project_write_tools_text, plan_system_prompt
from runtime import runner
from ui import repl


class RecordingConsole:
    def __init__(self):
        self.lines = []

    def print(self, *values, **kwargs):
        self.lines.append(" ".join(str(value) for value in values))

    def clear(self):
        self.lines.append("clear")


class RecordingRenderer:
    def __init__(self):
        self.console = RecordingConsole()
        self.plan_panels = []
        self.no_plans_called = False

    def splash(self, model_name, session_id):
        self.console.print(f"splash {model_name} {session_id}")

    def newline(self):
        self.console.print("")

    def plan(self, plan_id, text):
        self.plan_panels.append((plan_id, text))

    def no_plans(self):
        self.no_plans_called = True


class FakePromptSession:
    inputs = []

    def __init__(self, *args, **kwargs):
        self.inputs = list(self.__class__.inputs)

    async def prompt_async(self, prompt):
        if not self.inputs:
            raise EOFError
        return self.inputs.pop(0)


class FakeModelRequest:
    def __init__(self, tools):
        self.tools = tools

    def override(self, **kwargs):
        return FakeModelRequest(kwargs.get("tools", self.tools))


class PlanModeTests(unittest.IsolatedAsyncioTestCase):
    def test_plan_permissions_deny_writes(self):
        permissions = factory._plan_permissions()

        self.assertEqual(len(permissions), 1)
        self.assertEqual(permissions[0].operations, ["write"])
        self.assertEqual(permissions[0].paths, ["/**"])
        self.assertEqual(permissions[0].mode, "deny")

    def test_action_permissions_allow_reads_and_writes(self):
        permissions = factory._action_permissions()

        self.assertEqual(len(permissions), 1)
        self.assertEqual(permissions[0].operations, ["read", "write"])
        self.assertEqual(permissions[0].paths, ["/**"])
        self.assertEqual(permissions[0].mode, "allow")

    def test_plan_agent_disables_write_interrupts_and_denies_writes(self):
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
        self.assertTrue(any(isinstance(item, factory.ToolNameFilterMiddleware) for item in kwargs["middleware"]))

    def test_action_agent_keeps_write_interrupts(self):
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
        self.assertEqual(kwargs["permissions"][0].operations, ["read", "write"])
        self.assertEqual(kwargs["permissions"][0].mode, "allow")

    def test_plan_tool_filter_hides_write_tools_from_model(self):
        middleware = factory.ToolNameFilterMiddleware(PLAN_PROJECT_WRITE_TOOLS)
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

    async def test_plan_and_act_commands_toggle_mode(self):
        renderer = RecordingRenderer()
        mode = {"planning": False}
        session = {"id": "thread-1", "workspace": ".", "turns": 0}

        handled = await repl.handle_command("/plan", renderer, None, session, "model", mode)
        self.assertTrue(handled)
        self.assertTrue(mode["planning"])
        self.assertIn("planning mode", renderer.console.lines[-1])
        self.assertIn(f"{project_write_tools_text()} disabled", renderer.console.lines[-1])

        handled = await repl.handle_command("/act", renderer, None, session, "model", mode)
        self.assertTrue(handled)
        self.assertFalse(mode["planning"])
        self.assertIn("action mode", renderer.console.lines[-1])

    async def test_act_announces_pending_plan_when_plan_exists(self):
        renderer = RecordingRenderer()
        mode = {"planning": True, "last_plan": "Do the thing", "plan_pending": False}

        handled = await repl.handle_command("/act", renderer, None, {}, "model", mode)

        self.assertTrue(handled)
        self.assertFalse(mode["planning"])
        self.assertTrue(mode["plan_pending"])
        self.assertIn("last plan will be included", renderer.console.lines[-1])

    async def test_help_includes_plan_commands(self):
        renderer = RecordingRenderer()

        handled = await repl.handle_command("/help", renderer, None, {}, "model", {"planning": False})

        self.assertTrue(handled)
        self.assertIn("/plan", renderer.console.lines[-1])
        self.assertIn("/act", renderer.console.lines[-1])
        self.assertIn("/plans", renderer.console.lines[-1])
        self.assertIn("/clear", renderer.console.lines[-1])

    async def test_clear_command_clears_console(self):
        renderer = RecordingRenderer()

        handled = await repl.handle_command("/clear", renderer, None, {}, "model", {"planning": False})

        self.assertTrue(handled)
        self.assertEqual(renderer.console.lines, ["clear"])

    async def test_start_repl_routes_turns_to_plan_agent_while_planning(self):
        renderer = RecordingRenderer()
        session = {"id": "thread-1", "workspace": ".", "turns": 0}
        store = type("Store", (), {"save": lambda self, record: None})()
        calls = []

        async def fake_run_turn(agent, text, renderer, thread_id):
            calls.append((agent, text, thread_id))

        FakePromptSession.inputs = [
            "/plan",
            "write a file",
            "/act",
            "write it now",
        ]

        with patch("ui.repl.PromptSession", FakePromptSession), patch("ui.repl.run_turn", fake_run_turn):
            await repl.start_repl(
                agent="action-agent",
                plan_agent="plan-agent",
                renderer=renderer,
                store=store,
                session=session,
                model_name="model",
            )

        self.assertEqual(calls[0][0], "plan-agent")
        self.assertIn("You are in planning mode.", calls[0][1])
        self.assertIn("Do not call write_file, edit_file", calls[0][1])
        self.assertIn("User request:\nwrite a file", calls[0][1])
        self.assertEqual(calls[0][2], "thread-1:plan:1")
        self.assertEqual(calls[1], ("action-agent", "write it now", "thread-1"))
        self.assertEqual(session["turns"], 2)

    async def test_start_repl_injects_valid_plan_once_after_act(self):
        renderer = RecordingRenderer()
        session = {"id": "thread-1", "workspace": ".", "turns": 0}
        store = type("Store", (), {"save": lambda self, record: None})()
        calls = []
        results = [
            runner.TurnResult(final_text="Create test.txt with hello world."),
            runner.TurnResult(final_text="done"),
            runner.TurnResult(final_text="done again"),
        ]

        async def fake_run_turn(agent, text, renderer, thread_id):
            calls.append((agent, text, thread_id))
            return results.pop(0)

        FakePromptSession.inputs = [
            "/plan",
            "plan the write",
            "/act",
            "do it",
            "do another thing",
        ]

        with patch("ui.repl.PromptSession", FakePromptSession), patch("ui.repl.run_turn", fake_run_turn):
            await repl.start_repl(
                agent="action-agent",
                plan_agent="plan-agent",
                renderer=renderer,
                store=store,
                session=session,
                model_name="model",
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

    async def test_start_repl_does_not_save_blocked_plan(self):
        renderer = RecordingRenderer()
        session = {"id": "thread-1", "workspace": ".", "turns": 0}
        store = type("Store", (), {"save": lambda self, record: None})()
        calls = []
        results = [
            runner.TurnResult(
                final_text="I cannot write files.",
                tool_calls=["write_file"],
                tool_results=["Error: permission denied for write on /test.txt"],
            ),
            runner.TurnResult(final_text="done"),
        ]

        async def fake_run_turn(agent, text, renderer, thread_id):
            calls.append((agent, text, thread_id))
            return results.pop(0)

        FakePromptSession.inputs = [
            "/plan",
            "write a file",
            "/act",
            "do it",
        ]

        with patch("ui.repl.PromptSession", FakePromptSession), patch("ui.repl.run_turn", fake_run_turn):
            await repl.start_repl(
                agent="action-agent",
                plan_agent="plan-agent",
                renderer=renderer,
                store=store,
                session=session,
                model_name="model",
            )

        self.assertTrue(any("no plan was saved" in line for line in renderer.console.lines))
        self.assertEqual(calls[1], ("action-agent", "do it", "thread-1"))

    async def test_session_command_shows_current_mode(self):
        renderer = RecordingRenderer()
        session = {"id": "thread-1", "workspace": ".", "turns": 3}

        handled = await repl.handle_command("/session", renderer, None, session, "model", {"planning": True, "plans": []})

        self.assertTrue(handled)
        self.assertIn("session: thread-1", renderer.console.lines)
        self.assertIn("mode: planning", renderer.console.lines)
        self.assertIn("saved plans: 0", renderer.console.lines)

    def test_plan_thread_id_is_separate_from_action_thread(self):
        self.assertEqual(repl.plan_thread_id({"id": "thread-1"}), "thread-1:plan")
        self.assertEqual(repl.plan_thread_id({"id": "thread-1"}, 2), "thread-1:plan:2")

    def test_action_request_text_clears_pending_plan(self):
        mode = {"planning": False, "last_plan": "Plan text", "plan_pending": True}

        text = repl.action_request_text(mode, "Implement")

        self.assertIn("Previous planning context:\nPlan text", text)
        self.assertIn("Do not assume planning-mode permission errors still apply.", text)
        self.assertIn("User request:\nImplement", text)
        self.assertEqual(mode["last_plan"], "")
        self.assertFalse(mode["plan_pending"])

    def test_invalid_plan_result_when_write_tool_was_used(self):
        result = runner.TurnResult(final_text="Nope", tool_calls=["write_file"])

        self.assertFalse(repl.is_valid_plan_result(result))

    def test_invalid_plan_result_when_project_write_was_blocked(self):
        result = runner.TurnResult(
            final_text="Nope",
            tool_calls=["write_file"],
            tool_results=["Error: permission denied for write on /test.txt"],
        )

        self.assertFalse(repl.is_valid_plan_result(result))

    def test_invalid_plan_result_when_final_text_mentions_permission_denied(self):
        result = runner.TurnResult(final_text="I hit permission denied for write on /test.txt.")

        self.assertFalse(repl.is_valid_plan_result(result))

    def test_plan_request_text_wraps_user_request(self):
        text = repl.plan_request_text("write a file")

        self.assertIn("You are in planning mode.", text)
        self.assertIn("Respond only with a concrete plan", text)
        self.assertIn("User request:\nwrite a file", text)

    def test_save_plan_result_records_memory_plan(self):
        renderer = RecordingRenderer()
        mode = {"plans": [], "last_plan": "", "plan_pending": False}

        repl.save_plan_result(mode, runner.TurnResult(final_text="1. Create test.txt."), renderer)

        self.assertEqual(mode["last_plan"], "1. Create test.txt.")
        self.assertEqual(mode["plans"], [{"id": 1, "text": "1. Create test.txt."}])

    async def test_plans_command_shows_empty_state(self):
        renderer = RecordingRenderer()

        handled = await repl.handle_command("/plans", renderer, None, {}, "model", {"plans": []})

        self.assertTrue(handled)
        self.assertTrue(renderer.no_plans_called)

    async def test_plans_command_renders_each_saved_plan_once(self):
        renderer = RecordingRenderer()
        mode = {"plans": [{"id": 1, "text": "First plan\nmore"}, {"id": 2, "text": "Latest plan"}]}

        handled = await repl.handle_command("/plans", renderer, None, {}, "model", mode)

        self.assertTrue(handled)
        self.assertEqual(renderer.plan_panels, [(1, "First plan\nmore"), (2, "Latest plan")])
        self.assertEqual(renderer.console.lines, [])

    def test_plan_policy_drives_prompt_and_repl_validation(self):
        prompt = plan_system_prompt()

        for tool in PLAN_PROJECT_WRITE_TOOLS:
            self.assertIn(tool, prompt)

        self.assertIn("Never call disabled tools", prompt)


if __name__ == "__main__":
    unittest.main()
