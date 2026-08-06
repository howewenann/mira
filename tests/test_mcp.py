"""Focused MCP configuration, lifecycle, registry, and attachment tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import ToolException
from langchain_mcp_adapters.callbacks import Callbacks
from mcp.types import ReadResourceResult, TextResourceContents
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from agent.mcp.configuration import approval_preview, configuration_fingerprint, load_mcp_configuration
from agent.mcp.manager import MCPManager
from agent.mcp.models import MCPResource
from agent.mcp.prompts import PromptRegistry, mustache_variables
from config.settings import (
    ToolPolicy,
    load_settings,
    mcp_server_always_allow,
    mcp_server_approved_fingerprint,
    mcp_server_enabled,
    mcp_tool_policy,
    save_settings,
    set_mcp_server_always_allow,
    set_mcp_server_approved_fingerprint,
    set_mcp_server_enabled,
    set_mcp_tool_policy_value,
    set_tool_plan_access,
    tool_plan_access,
)
from session.context import normalize_events, session_mcp_attachments
from ui.spinners import SPINNER_FRAMES
from ui.widgets.mcp_panel import MCPPanelScreen, controls_for, status_class
from ui.widgets import PromptBox


def write_config(root: Path, servers: dict) -> None:
    directory = root / ".mira"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "mcp.json").write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")


class MCPConfigurationTests(unittest.TestCase):
    def test_no_file_is_valid_and_unconfigured(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            loaded = load_mcp_configuration(Path(directory))
        self.assertFalse(loaded.exists)
        self.assertTrue(loaded.valid)
        self.assertEqual(loaded.servers, {})

    def test_valid_stdio_and_http_are_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_config(
                root,
                {
                    "local": {"command": "python", "args": ["server.py"], "env": {"TOKEN": "secret"}},
                    "docs": {"type": "http", "url": "https://example.test/mcp", "headers": {"Authorization": "secret"}},
                },
            )
            loaded = load_mcp_configuration(root)
        self.assertTrue(loaded.valid)
        self.assertEqual(loaded.servers["local"].transport, "stdio")
        self.assertEqual(loaded.servers["docs"].transport, "http")

    def test_invalid_json_and_top_level_are_whole_file_issues(self) -> None:
        for text in ("{", "[]", '{"servers": {}}'):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / ".mira").mkdir()
                (root / ".mira" / "mcp.json").write_text(text, encoding="utf-8")
                loaded = load_mcp_configuration(root)
                self.assertFalse(loaded.valid)
                self.assertIsNotNone(loaded.issue)
                self.assertEqual(loaded.servers, {})

    def test_invalid_named_server_is_retained_beside_valid_server(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_config(root, {"bad": {"args": []}, "good": {"command": "python", "args": []}})
            loaded = load_mcp_configuration(root)
        self.assertEqual(loaded.servers["bad"].status, "Failed")
        self.assertIn("command", loaded.servers["bad"].error)
        self.assertEqual(loaded.servers["good"].status, "Disabled")

    def test_fingerprint_and_preview_never_expose_secret_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_config(root, {"x": {"command": "run", "args": [], "env": {"TOKEN": "top-secret"}}})
            state = load_mcp_configuration(root).servers["x"]
        preview = approval_preview(state)
        fingerprint = configuration_fingerprint(state.config)
        self.assertIn("TOKEN", preview)
        self.assertNotIn("top-secret", preview)
        self.assertEqual(len(fingerprint), 64)
        self.assertNotIn("top-secret", fingerprint)


class MCPSettingsTests(unittest.TestCase):
    def test_defaults_and_policy_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = load_settings(root)
            self.assertTrue(mcp_server_enabled(settings, "github"))
            self.assertFalse(mcp_server_always_allow(settings, "github"))
            self.assertEqual(mcp_tool_policy(settings, "github", "search"), ToolPolicy())
            settings = set_mcp_server_enabled(settings, "github", False)
            settings = set_mcp_tool_policy_value(settings, "github", "search", "plan_access", True)
            self.assertTrue(save_settings(root, settings))
            loaded = load_settings(root)
            self.assertFalse(mcp_server_enabled(loaded, "github"))
            self.assertTrue(mcp_tool_policy(loaded, "github", "search").plan_access)

    def test_custom_plan_access_defaults_no_and_can_be_explicit(self) -> None:
        settings = load_settings(Path("missing-workspace-for-defaults"))
        self.assertFalse(tool_plan_access(settings, "custom_search"))
        self.assertTrue(tool_plan_access(set_tool_plan_access(settings, "custom_search", True), "custom_search"))

    def test_approval_fingerprint_is_hash_only_and_changes_invalidate_match(self) -> None:
        settings = load_settings(Path("missing-workspace-for-defaults"))
        first = configuration_fingerprint({"transport": "stdio", "command": "a", "args": [], "env": {"X": "one"}})
        second = configuration_fingerprint({"transport": "stdio", "command": "a", "args": [], "env": {"X": "two"}})
        settings = set_mcp_server_always_allow(settings, "x", True)
        settings = set_mcp_server_approved_fingerprint(settings, "x", first)
        self.assertEqual(mcp_server_approved_fingerprint(settings, "x"), first)
        self.assertNotEqual(mcp_server_approved_fingerprint(settings, "x"), second)


class LocalPromptTests(unittest.IsolatedAsyncioTestCase):
    async def test_top_level_utf8_files_variables_and_quoted_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts = root / ".mira" / "prompts"
            prompts.mkdir(parents=True)
            (prompts / "review.anything").write_text("Review {{path}} for {{focus}} and {{path}}", encoding="utf-8")
            (prompts / "ignored").mkdir()
            registry = PromptRegistry(root)
            prepared = await registry.resolve('/prompt__review "src/auth module.py" "security and correctness"')
        self.assertEqual(mustache_variables("{{a}} {{#items}}{{b}}{{/items}} {{a}}"), ("a", "items", "b"))
        self.assertIsNotNone(prepared)
        self.assertEqual(len(prepared.messages), 1)
        self.assertIsInstance(prepared.messages[0], HumanMessage)
        self.assertIn("src/auth module.py", str(prepared.messages[0].content))

    async def test_missing_excess_unclosed_and_duplicate_names_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts = root / ".mira" / "prompts"
            prompts.mkdir(parents=True)
            (prompts / "one.md").write_text("{{value}}", encoding="utf-8")
            registry = PromptRegistry(root)
            with self.assertRaises(ValueError):
                await registry.resolve("/prompt__one")
            with self.assertRaises(ValueError):
                await registry.resolve("/prompt__one a b")
            with self.assertRaises(ValueError):
                await registry.resolve('/prompt__one "bad')
            with self.assertRaises(ValueError):
                await registry.resolve("/prompt__missing value")
            (prompts / "one.txt").write_text("duplicate", encoding="utf-8")
            registry.reload_local()
            self.assertNotIn("/prompt__one", registry.specs)
            self.assertEqual(registry.warnings, ["local prompt collision excluded: /prompt__one"])


class Page:
    def __init__(self, field: str, values: list, cursor: str | None = None) -> None:
        setattr(self, field, values)
        self.nextCursor = cursor


def tool_descriptor(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, description=f"{name} description", inputSchema={"type": "object", "properties": {}}, annotations=None, meta=None)


class FakeSession:
    def __init__(self, tools: list[str] | None = None) -> None:
        self.tool_names = tools or []
        self.tool_pages = 0
        self.resource_pages = 0
        self.calls: list[tuple[str, dict]] = []

    async def list_tools(self, cursor: str | None = None) -> Page:
        self.tool_pages += 1
        return Page("tools", [tool_descriptor(name) for name in self.tool_names])

    async def list_resources(self, cursor: str | None = None) -> Page:
        self.resource_pages += 1
        if cursor is None:
            return Page("resources", [SimpleNamespace(uri="repo://one", title="One", name="one", description="first", mimeType="text/plain")], "next")
        return Page("resources", [SimpleNamespace(uri="repo://two", title="Two", name="two", description="second", mimeType="text/plain")])

    async def list_prompts(self, cursor: str | None = None) -> Page:
        argument = SimpleNamespace(name="topic", required=True)
        return Page("prompts", [SimpleNamespace(name="review", description="Review", arguments=[argument])])

    async def get_prompt(self, name: str, arguments: dict | None = None) -> SimpleNamespace:
        from mcp.types import GetPromptResult, PromptMessage, TextContent

        return GetPromptResult(messages=[PromptMessage(role="user", content=TextContent(type="text", text=arguments["topic"])), PromptMessage(role="assistant", content=TextContent(type="text", text="draft"))])

    async def call_tool(self, name: str, arguments: dict, **kwargs) -> SimpleNamespace:
        self.calls.append((name, arguments))
        return SimpleNamespace(content=[], structuredContent=None, isError=False)

    async def read_resource(self, uri: str) -> ReadResourceResult:
        return ReadResourceResult(contents=[TextResourceContents(uri=uri, text="first"), TextResourceContents(uri=uri, text="second")])


class FakeClient:
    def __init__(self, sessions: dict[str, FakeSession]) -> None:
        self.sessions = sessions
        self.callbacks = Callbacks()
        self.opened: list[str] = []
        self.closed: list[str] = []

    @asynccontextmanager
    async def session(self, name: str):
        self.opened.append(name)
        try:
            yield self.sessions[name]
        finally:
            self.closed.append(name)


class MCPManagerTests(unittest.IsolatedAsyncioTestCase):
    async def make_manager(self, root: Path, servers: dict[str, dict], sessions: dict[str, FakeSession]) -> MCPManager:
        write_config(root, servers)
        manager = MCPManager(root)
        manager.client = FakeClient(sessions)

        async def allow(_state, _preview):
            return "allow"

        await manager.initialize(allow)
        return manager

    async def test_persistent_runtime_namespaced_tools_lazy_caches_and_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = FakeSession(["search"])
            manager = await self.make_manager(root, {"github": {"command": "fake", "args": []}}, {"github": session})
            self.assertEqual(manager.client.opened, ["github"])
            self.assertEqual([tool.name for tool in manager.servers["github"].tools], ["mcp__github__search"])
            self.assertIsNone(manager.servers["github"].resources)
            await manager.discover_resources()
            await manager.discover_resources()
            self.assertEqual(session.resource_pages, 2)
            self.assertEqual(set(manager.resource_registry), {"mcp__github__repo://one", "mcp__github__repo://two"})
            await manager.set_server_enabled("github", False)
            self.assertEqual(manager.client.closed, ["github"])
            self.assertEqual(manager.servers["github"].status, "Disabled")
            self.assertEqual(manager.resource_registry, {})

    async def test_failure_isolation_denial_and_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_config(root, {"a": {"command": "a", "args": []}, "b": {"command": "b", "args": []}})
            manager = MCPManager(root)
            client = FakeClient({"a": FakeSession(), "b": FakeSession()})
            manager.client = client

            async def approve(state, _preview):
                return "deny" if state.name == "a" else "allow"

            await manager.initialize(approve)
            self.assertEqual(manager.servers["a"].status, "Approval required")
            self.assertEqual(manager.servers["b"].status, "Available")
            self.assertTrue(mcp_server_enabled(load_settings(root), "a"))
            await manager.restart_server("b")
            self.assertEqual(client.opened.count("b"), 2)
            self.assertEqual(client.closed.count("b"), 1)
            await manager.shutdown()

    async def test_tool_collisions_and_conversion_failures_are_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collided = await self.make_manager(
                root,
                {"x": {"command": "x", "args": []}},
                {"x": FakeSession(["same", "same"])},
            )
            self.assertEqual(collided.servers["x"].tools, [])
            self.assertEqual(collided.servers["x"].status, "Partially available")
            self.assertIn("ambiguous tool name", collided.servers["x"].error)
            await collided.shutdown()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def convert(_session, descriptor, **_kwargs):
                if descriptor.name == "bad":
                    raise ValueError("cannot convert")
                return SimpleNamespace(name=descriptor.name, description="works", metadata={})

            with patch("agent.mcp.manager.convert_mcp_tool_to_langchain_tool", side_effect=convert):
                manager = await self.make_manager(
                    root,
                    {"x": {"command": "x", "args": []}},
                    {"x": FakeSession(["good", "bad"])},
                )
            self.assertEqual([item.name for item in manager.servers["x"].tools], ["mcp__x__good"])
            self.assertEqual(manager.servers["x"].status, "Partially available")
            self.assertIn("tool bad", manager.servers["x"].error)
            await manager.shutdown()

    async def test_complete_tool_list_failure_is_server_local(self) -> None:
        class BrokenTools(FakeSession):
            async def list_tools(self, cursor=None):
                raise ValueError("bad tools payload")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = await self.make_manager(
                root,
                {
                    "bad": {"command": "bad", "args": []},
                    "good": {"command": "good", "args": []},
                },
                {"bad": BrokenTools(), "good": FakeSession(["search"])},
            )
            self.assertEqual(manager.servers["bad"].status, "Partially available")
            self.assertEqual(manager.servers["bad"].tools, [])
            self.assertEqual(manager.servers["good"].status, "Available")
            self.assertEqual(len(manager.servers["good"].tools), 1)
            await manager.shutdown()

    async def test_prompt_roles_attachments_and_exact_resource_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = await self.make_manager(root, {"x": {"command": "x", "args": []}}, {"x": FakeSession()})
            await manager.discover_prompts()
            prepared = await manager.prompt_registry.resolve('/mcp__x__review "quoted topic"')
            self.assertIsInstance(prepared.messages[0], HumanMessage)
            self.assertIsInstance(prepared.messages[1], AIMessage)
            await manager.discover_resources()
            text = "Use @mcp__x__repo://one"
            attachments = manager.attachments_from_text(text)
            self.assertEqual(attachments[0]["uri"], "repo://one")
            messages = [HumanMessage(content=text, additional_kwargs={"mira_mcp_attachments": attachments})]
            runtime = SimpleNamespace(state={"messages": messages})
            result = await manager.read_tool.coroutine(server="x", uri="repo://one", runtime=runtime)
            self.assertEqual(result, "first\n\nsecond")
            with self.assertRaisesRegex(ToolException, "not attached"):
                await manager.read_tool.coroutine(server="x", uri="repo://invented", runtime=runtime)
            await manager.shutdown()

    async def test_binary_and_resource_read_failures_are_normal_tool_errors(self) -> None:
        from mcp.types import BlobResourceContents

        class BinarySession(FakeSession):
            async def read_resource(self, uri: str) -> ReadResourceResult:
                return ReadResourceResult(contents=[BlobResourceContents(uri=uri, blob="AA==")])

        class BrokenReadSession(FakeSession):
            async def read_resource(self, uri: str) -> ReadResourceResult:
                raise RuntimeError("offline")

        for session, expected in ((BinarySession(), "binary or unsupported"), (BrokenReadSession(), "read failed")):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manager = await self.make_manager(root, {"x": {"command": "x", "args": []}}, {"x": session})
                messages = [
                    HumanMessage(
                        content="attached",
                        additional_kwargs={
                            "mira_mcp_attachments": [MCPResource("token", "x", "repo://one").attachment()]
                        },
                    )
                ]
                runtime = SimpleNamespace(state={"messages": messages})
                with self.assertRaisesRegex(ToolException, expected):
                    await manager.read_tool.coroutine(server="x", uri="repo://one", runtime=runtime)
                await manager.shutdown()

    async def test_session_attachment_metadata_survives_normalization(self) -> None:
        resource = MCPResource("mcp__x__repo://one", "x", "repo://one")
        record = {"events": [{"id": 1, "type": "user", "created_at": "now", "text": "attached", "attachments": [resource.attachment()]}]}
        normalized = normalize_events(record["events"])
        self.assertEqual(normalized[0]["attachments"][0]["uri"], "repo://one")
        self.assertEqual(session_mcp_attachments(record)[0]["server"], "x")


class PanelManager:
    def __init__(self) -> None:
        self.servers = {
            "one": SimpleNamespace(
                name="one",
                transport="stdio",
                status="Available",
                transient=False,
                tools=[],
                tool_metadata=[],
                prompts=None,
                resources=None,
                error="",
                prompt_error="",
                resource_error="",
            ),
            "two": SimpleNamespace(
                name="two",
                transport="http",
                status="Disabled",
                transient=False,
                tools=[],
                tool_metadata=[],
                prompts=None,
                resources=None,
                error="",
                prompt_error="",
                resource_error="",
            ),
        }
        self.discovered: list[str] = []
        self.prompt_registry = SimpleNamespace(warnings=[], rows=lambda: [])
        self.resource_registry = {}
        self.config_issue = None
        self.change_handler = None
        self.shutdown_calls = 0

    @property
    def show_status(self) -> bool:
        return True

    @property
    def usable_count(self) -> int:
        return sum(state.status in {"Available", "Partially available"} for state in self.servers.values())

    @property
    def configured_count(self) -> int:
        return len(self.servers)

    def set_change_handler(self, handler) -> None:
        self.change_handler = handler

    async def shutdown(self) -> None:
        self.shutdown_calls += 1

    async def discover_prompts(self, name: str) -> None:
        self.discovered.append(f"prompts:{name}")
        self.servers[name].prompts = []

    async def discover_resources(self, name: str) -> None:
        self.discovered.append(f"resources:{name}")
        self.servers[name].resources = []

    async def set_server_enabled(self, name: str, enabled: bool) -> bool:
        self.servers[name].status = "Available" if enabled else "Disabled"
        return True

    async def restart_server(self, name: str) -> bool:
        self.servers[name].status = "Available"
        return True


class PanelApp(App[None]):
    def __init__(self, manager: PanelManager) -> None:
        super().__init__()
        self.manager = manager

    def compose(self) -> ComposeResult:
        yield Static("host")

    def on_mount(self) -> None:
        self.push_screen(MCPPanelScreen(self.manager))


class MCPPanelTests(unittest.IsolatedAsyncioTestCase):
    async def test_focused_header_alone_expands_and_uses_shared_caches(self) -> None:
        manager = PanelManager()
        app = PanelApp(manager)
        async with app.run_test(size=(100, 35)) as pilot:
            await pilot.pause()
            screen = app.screen
            self.assertIsInstance(screen, MCPPanelScreen)
            first = screen.query_one("#mcp-header-one", Button)
            second = screen.query_one("#mcp-header-two", Button)
            self.assertTrue(first.has_focus)
            await pilot.press("enter")
            await pilot.pause()
            self.assertIn("one", screen.expanded)
            self.assertNotIn("two", screen.expanded)
            self.assertEqual(manager.discovered, ["prompts:one", "resources:one"])
            self.assertFalse(second.has_focus)

    def test_controls_status_colours_and_spinner_are_consistent(self) -> None:
        self.assertEqual(controls_for("Disabled"), ("Enable",))
        for status in ("Available", "Partially available", "Approval required", "Failed"):
            self.assertEqual(controls_for(status), ("Disable", "Restart"))
        self.assertEqual(status_class("Available"), "available")
        self.assertEqual(status_class("Failed"), "failed")
        self.assertEqual(SPINNER_FRAMES, ("|", "/", "-", "\\"))

    async def test_app_button_and_slash_open_the_same_panel_pathway(self) -> None:
        from tests.test_textual_app import make_app

        manager = PanelManager()
        app = make_app(mcp_manager=manager)
        async with app.run_test(size=(100, 35)) as pilot:
            await pilot.pause()
            button = app.query_one("#mcp-status-button", Button)
            self.assertTrue(button.display)
            self.assertEqual(str(button.label), "MCP 1/2")
            await pilot.click("#mcp-status-button")
            await pilot.pause()
            self.assertIsInstance(app.screen, MCPPanelScreen)
            app.screen.dismiss()
            await pilot.pause()
            prompt = app.query_one(PromptBox)
            await app.submit_prompt(PromptBox.Submitted(prompt, "/mcp"))
            await pilot.pause()
            self.assertIsInstance(app.screen, MCPPanelScreen)


if __name__ == "__main__":
    unittest.main()
