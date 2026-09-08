"""Focused MCP configuration, lifecycle, registry, and attachment tests."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import anyio
import httpx2 as httpx
from fastmcp import Context, FastMCP
from fastmcp.client.transports import StdioTransport, StreamableHttpTransport
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool, ToolException
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from mcp.types import (
    PromptsCapability,
    ElicitRequest,
    ElicitRequestFormParams,
    InputRequiredResult,
    ReadResourceResult,
    ResourcesCapability,
    ServerCapabilities,
    TextResourceContents,
    ToolsCapability,
)
from textual.app import App, ComposeResult
from textual.widgets import Button, Collapsible, Static

from agent.mcp.auth import (
    create_http_oauth,
    discover_token_endpoint_auth_method,
    select_token_endpoint_auth_method,
)
from agent.mcp.configuration import (
    approval_preview,
    configuration_fingerprint,
    load_mcp_configuration,
    mcp_path,
)
from agent.mcp.errors import sanitized_error
from agent.mcp.manager import MCPManager
from agent.mcp.integration import MCPAdapter, MiraMCPIntegration
from agent.mcp.models import MCPResource, MCPServerState, PromptArgument, PromptSpec
from agent.mcp.prompts import PromptRegistry, mustache_variables
from agent.resources.project_setup import ensure_project_examples
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
from config.interpolation import resolve_environment
from session.context import normalize_events, session_mcp_attachments
from ui.shared.terminal.spinners import SPINNER_FRAMES
from ui.textual.widgets.mcp_panel import (
    MCPPanelScreen,
    capability_metric,
    capability_summary,
    controls_for,
    mcp_summary_symbol,
    status_badge,
    status_class,
)
from ui.textual.widgets import PromptBox, PromptPanel


def write_config(root: Path, servers: dict) -> None:
    directory = root / ".mira" / "mcp"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "mcp.json").write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")


class MCPConfigurationTests(unittest.TestCase):
    def test_mcp_path_resolves_only_the_active_nested_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(mcp_path(root), root.resolve() / ".mira" / "mcp" / "mcp.json")

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
        self.assertNotIn("auth", loaded.servers["docs"].connection_config)

    def test_environment_references_resolve_only_in_runtime_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_config(
                root,
                {
                    "remote": {
                        "type": "http",
                        "url": "https://${MCP_HOST}/mcp",
                        "headers": {"Authorization": "Bearer ${MCP_TOKEN}"},
                    },
                    "local": {
                        "command": "${PYTHON_COMMAND}",
                        "args": ["--token=${MCP_TOKEN}", "literal"],
                        "env": {"${KEY_NAME}": "${MCP_TOKEN}"},
                    },
                },
            )
            first = load_mcp_configuration(
                root,
                environ={
                    "MCP_HOST": "example.test",
                    "MCP_TOKEN": "first-secret",
                    "PYTHON_COMMAND": "python",
                },
            )
            second = load_mcp_configuration(
                root,
                environ={
                    "MCP_HOST": "example.test",
                    "MCP_TOKEN": "rotated-secret",
                    "PYTHON_COMMAND": "python",
                },
            )

        remote = first.servers["remote"]
        self.assertEqual(remote.config["url"], "https://${MCP_HOST}/mcp")
        self.assertEqual(remote.config["headers"]["Authorization"], "Bearer ${MCP_TOKEN}")
        self.assertEqual(remote.connection_config["url"], "https://example.test/mcp")
        self.assertEqual(
            remote.connection_config["headers"]["Authorization"],
            "Bearer first-secret",
        )
        self.assertEqual(
            remote.connection_config["headers"]["Authorization"],
            "Bearer first-secret",
        )
        self.assertNotIn("first-secret", approval_preview(remote))
        self.assertEqual(remote.fingerprint, second.servers["remote"].fingerprint)

        local = first.servers["local"]
        self.assertEqual(local.connection_config["command"], "python")
        self.assertEqual(
            local.connection_config["args"],
            ["--token=first-secret", "literal"],
        )
        self.assertEqual(local.connection_config["env"], {"${KEY_NAME}": "first-secret"})

    def test_environment_resolution_is_single_pass_and_missing_values_are_server_local(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_config(
                root,
                {
                    "missing": {
                        "type": "http",
                        "url": "https://example.test/mcp",
                        "headers": {"Authorization": "Bearer ${MISSING_TOKEN}"},
                    },
                    "available": {"command": "python", "args": []},
                },
            )
            loaded = load_mcp_configuration(root, environ={})

        self.assertEqual(loaded.servers["missing"].status, "Failed")
        self.assertEqual(
            loaded.servers["missing"].error,
            "environment variable MISSING_TOKEN is not set; define it before starting MIRA "
            "or in the workspace .env",
        )
        self.assertEqual(loaded.servers["available"].status, "Disabled")
        self.assertEqual(
            resolve_environment(
                "${FIRST}",
                environ={"FIRST": "${SECOND}", "SECOND": "secret"},
            ),
            "${SECOND}",
        )

    def test_invalid_or_empty_environment_references_fail_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid environment variable name"):
            resolve_environment("${NOT-VALID}", environ={})
        with self.assertRaisesRegex(ValueError, "malformed environment reference"):
            resolve_environment("${UNCLOSED", environ={})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_config(root, {"empty": {"command": "${COMMAND}"}})
            loaded = load_mcp_configuration(root, environ={"COMMAND": ""})
        self.assertEqual(loaded.servers["empty"].status, "Failed")
        self.assertEqual(
            loaded.servers["empty"].error,
            "stdio server command resolved to an empty value",
        )

    def test_invalid_json_and_top_level_are_whole_file_issues(self) -> None:
        for text in ("{", "[]", '{"servers": {}}'):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config_path = root / ".mira" / "mcp" / "mcp.json"
                config_path.parent.mkdir(parents=True)
                config_path.write_text(text, encoding="utf-8")
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

    def test_loader_ignores_old_root_file_and_schema_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_path = root / ".mira" / "mcp.json"
            old_path.parent.mkdir(parents=True)
            old_path.write_text(
                json.dumps({"mcpServers": {"legacy": {"command": "legacy"}}}),
                encoding="utf-8",
            )

            missing = load_mcp_configuration(root)
            self.assertFalse(missing.exists)
            self.assertEqual(missing.servers, {})

            write_config(root, {})
            active_path = mcp_path(root)
            active_path.write_text(
                json.dumps({"$schema": "./schema.json", "mcpServers": {}}),
                encoding="utf-8",
            )
            loaded = load_mcp_configuration(root)
            self.assertTrue(loaded.exists)
            self.assertTrue(loaded.valid)
            self.assertEqual(loaded.servers, {})

    def test_generated_active_and_example_configurations_load_as_intended(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ensure_project_examples(root)
            active_path = mcp_path(root)

            loaded = load_mcp_configuration(root)
            self.assertTrue(loaded.exists)
            self.assertTrue(loaded.valid)
            self.assertEqual(loaded.servers, {})

            example_path = root / ".mira" / "examples" / "mcp" / "example.json"
            active_path.write_text(example_path.read_text(encoding="utf-8"), encoding="utf-8")
            example = load_mcp_configuration(root, environ={"REMOTE_MCP_TOKEN": "resolved-secret"})
            self.assertEqual(set(example.servers), {"fetch", "remote"})
            self.assertEqual(example.servers["fetch"].transport, "stdio")
            self.assertEqual(
                example.servers["fetch"].config,
                {
                    "transport": "stdio",
                    "command": "uv",
                    "args": [
                        "run",
                        "--project",
                        ".mira/mcp/servers/fetch",
                        "mcp-server-fetch",
                    ],
                    "env": {},
                },
            )
            self.assertEqual(example.servers["remote"].transport, "http")
            self.assertEqual(
                example.servers["remote"].config,
                {
                    "transport": "http",
                    "url": "https://example.com/mcp",
                    "headers": {"Authorization": "Bearer ${REMOTE_MCP_TOKEN}"},
                },
            )
            self.assertEqual(
                example.servers["remote"].connection_config["headers"],
                {"Authorization": "Bearer resolved-secret"},
            )

    def test_example_is_inert_and_schema_matches_public_configuration_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ensure_project_examples(root)
            loaded = load_mcp_configuration(root)
            self.assertEqual(loaded.servers, {})

            schema = json.loads((root / ".mira" / "mcp" / "schema.json").read_text(encoding="utf-8"))
            self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
            self.assertEqual(set(schema["properties"]), {"$schema", "mcpServers"})
            self.assertEqual(schema["required"], ["mcpServers"])
            self.assertFalse(schema["additionalProperties"])
            servers = schema["properties"]["mcpServers"]
            self.assertEqual(servers["propertyNames"]["minLength"], 1)
            stdio, http = servers["additionalProperties"]["oneOf"]
            self.assertEqual(set(stdio["properties"]), {"type", "command", "args", "env"})
            self.assertEqual(stdio["required"], ["command"])
            self.assertFalse(stdio["additionalProperties"])
            self.assertEqual(set(http["properties"]), {"type", "url", "headers"})
            self.assertEqual(http["required"], ["type", "url"])
            self.assertFalse(http["additionalProperties"])
            self.assertIn("${NAME}", schema["description"])
            self.assertIn("${NAME}", stdio["properties"]["env"]["description"])
            self.assertIn("${NAME}", http["properties"]["headers"]["description"])
            self.assertNotIn('"transport"', json.dumps(schema))
            self.assertNotIn("servers", schema["properties"])

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


class MCPErrorRenderingTests(unittest.TestCase):
    def test_nested_secret_text_is_redacted(self) -> None:
        error = ExceptionGroup(
            "outer",
            [RuntimeError("access_token=token-value refresh_token=refresh-value Bearer bearer-value")],
        )
        rendered = sanitized_error(error)
        self.assertNotIn("token-value", rendered)
        self.assertNotIn("refresh-value", rendered)
        self.assertNotIn("bearer-value", rendered)
        direct = sanitized_error(RuntimeError("access_token=token-value Bearer bearer-value"))
        self.assertIn("[redacted]", direct)

    def test_nested_http_failure_surfaces_response_body(self) -> None:
        request = httpx.Request("POST", "https://example.test/mcp")
        response = httpx.Response(
            400,
            request=request,
            text="IDE authentication failed: invalid token",
        )
        failure = httpx.HTTPStatusError("request failed", request=request, response=response)

        rendered = sanitized_error(ExceptionGroup("unhandled errors in a TaskGroup", [failure]))

        self.assertEqual(rendered, "HTTP 400 Bad Request: IDE authentication failed: invalid token")
        self.assertNotIn("ExceptionGroup", rendered)
        self.assertNotIn("TaskGroup", rendered)

    def test_unread_http_failure_surfaces_authentication_header(self) -> None:
        request = httpx.Request("POST", "https://example.test/mcp")
        response = httpx.Response(
            400,
            request=request,
            headers={
                "WWW-Authenticate": 'Bearer error="invalid_token", error_description="Invalid token"'
            },
            stream=httpx.ByteStream(b"unread response body"),
        )
        failure = httpx.HTTPStatusError("request failed", request=request, response=response)

        rendered = sanitized_error(
            ExceptionGroup("unhandled errors in a TaskGroup", [failure, anyio.WouldBlock()])
        )

        self.assertEqual(rendered, "HTTP 400 Bad Request: invalid_token: Invalid token")
        self.assertNotIn("WouldBlock", rendered)

    def test_multiple_nested_failures_are_concise_and_redacted(self) -> None:
        error = ExceptionGroup(
            "outer",
            [RuntimeError("first failure"), ValueError("Bearer secret-token")],
        )

        rendered = sanitized_error(error)

        self.assertEqual(rendered, "RuntimeError: first failure; ValueError: Bearer [redacted]")

    def test_actionable_failure_is_not_hidden_by_async_context(self) -> None:
        try:
            raise anyio.WouldBlock
        except anyio.WouldBlock:
            error = RuntimeError("Token exchange failed: invalid client")

        self.assertEqual(
            sanitized_error(error),
            "RuntimeError: Token exchange failed: invalid client",
        )


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
            settings = set_mcp_tool_policy_value(settings, "github", "search", "ptc", True)
            self.assertTrue(save_settings(root, settings))
            loaded = load_settings(root)
            self.assertFalse(mcp_server_enabled(loaded, "github"))
            self.assertTrue(mcp_tool_policy(loaded, "github", "search").plan_access)
            self.assertTrue(mcp_tool_policy(loaded, "github", "search").ptc)

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
            self.assertEqual(len(registry.issues), 1)
            self.assertEqual(registry.issues[0].category, "STARTUP")
            self.assertIn("/prompt__one", registry.issues[0].summary)

    async def test_prompt_usage_distinguishes_required_and_optional_arguments(self) -> None:
        calls: list[dict[str, str]] = []

        async def resolve(values: dict[str, str]) -> list[HumanMessage]:
            calls.append(values)
            return [HumanMessage(content="resolved")]

        with tempfile.TemporaryDirectory() as directory:
            registry = PromptRegistry(Path(directory))
            spec = PromptSpec(
                command="/mcp__github__review_pr",
                description="Review a pull request",
                arguments=(
                    PromptArgument("repo"),
                    PromptArgument("pr"),
                    PromptArgument("focus", required=False),
                ),
                source="mcp",
                resolver=resolve,
                server="github",
            )
            registry.mcp[spec.command] = spec

            self.assertEqual(spec.usage, "/mcp__github__review_pr <repo> <pr> [focus]")
            with self.assertRaisesRegex(
                ValueError,
                r"^missing required prompt arguments: pr; usage: ",
            ):
                await registry.resolve("/mcp__github__review_pr repo=owner/repo")
            self.assertEqual(calls, [])

            with self.assertRaisesRegex(ValueError, r"use name=value for every argument$"):
                await registry.resolve("/mcp__github__review_pr owner/repo 42")

            prepared_without_optional = await registry.resolve(
                "/mcp__github__review_pr repo=owner/repo pr=42"
            )
            prepared = await registry.resolve(
                '/mcp__github__review_pr pr=42 repo=owner/repo focus="security and correctness"'
            )

        self.assertIsNotNone(prepared_without_optional)
        self.assertIsNotNone(prepared)
        self.assertEqual(
            calls,
            [
                {"repo": "owner/repo", "pr": "42"},
                {"pr": "42", "repo": "owner/repo", "focus": "security and correctness"},
            ],
        )

    async def test_optional_prompt_arguments_reject_mixed_unknown_and_duplicate_names(self) -> None:
        async def resolve(_values: dict[str, str]) -> list[HumanMessage]:
            return [HumanMessage(content="resolved")]

        with tempfile.TemporaryDirectory() as directory:
            registry = PromptRegistry(Path(directory))
            spec = PromptSpec(
                command="/mcp__github__review_pr",
                description="Review a pull request",
                arguments=(PromptArgument("repo"), PromptArgument("focus", required=False)),
                source="mcp",
                resolver=resolve,
                server="github",
            )
            registry.mcp[spec.command] = spec

            with self.assertRaisesRegex(ValueError, r"use name=value for every argument$"):
                await registry.resolve("/mcp__github__review_pr repo=owner/repo security")
            with self.assertRaisesRegex(ValueError, r"^unknown prompt argument: typo;"):
                await registry.resolve("/mcp__github__review_pr repo=owner/repo typo=value")
            with self.assertRaisesRegex(ValueError, r"^duplicate prompt argument: repo;"):
                await registry.resolve("/mcp__github__review_pr repo=one repo=two")


class MCPNativeIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def test_token_endpoint_auth_method_selection(self) -> None:
        cases = (
            (["client_secret_basic", "client_secret_post"], "client_secret_basic"),
            (["client_secret_post"], "client_secret_post"),
            (["none"], "none"),
            ([], "none"),
            (None, None),
        )
        for supported, expected in cases:
            with self.subTest(supported=supported):
                self.assertEqual(
                    select_token_endpoint_auth_method(supported),
                    expected,
                )

    async def test_http_oauth_receives_selected_token_endpoint_auth_method(self) -> None:
        with (
            patch(
                "agent.mcp.auth.discover_token_endpoint_auth_method",
                new=AsyncMock(return_value="client_secret_post"),
            ),
            patch("agent.mcp.auth.OAuth") as oauth_type,
        ):
            oauth = await create_http_oauth("https://example.test/mcp")

        self.assertIs(oauth, oauth_type.return_value)
        oauth_type.assert_called_once_with(
            "https://example.test/mcp",
            additional_client_metadata={
                "token_endpoint_auth_method": "client_secret_post"
            },
        )

    async def test_http_oauth_discovers_metadata_with_public_sdk_request(self) -> None:
        response = httpx.Response(
            200,
            json={
                "issuer": "https://example.test",
                "authorization_endpoint": "https://example.test/authorize",
                "token_endpoint": "https://example.test/token",
                "token_endpoint_auth_methods_supported": ["client_secret_post"],
            },
        )
        with patch("agent.mcp.auth.httpx.AsyncClient") as client_type:
            client = client_type.return_value.__aenter__.return_value
            client.send = AsyncMock(return_value=response)
            method = await discover_token_endpoint_auth_method(
                "https://example.test/mcp"
            )

        self.assertEqual(method, "client_secret_post")
        request = client.send.await_args.args[0]
        self.assertEqual(
            str(request.url),
            "https://example.test/.well-known/oauth-authorization-server",
        )
        self.assertIn("mcp-protocol-version", request.headers)

    async def test_http_oauth_uses_sdk_default_when_metadata_is_unavailable(self) -> None:
        with (
            patch(
                "agent.mcp.auth.discover_token_endpoint_auth_method",
                new=AsyncMock(return_value=None),
            ),
            patch("agent.mcp.auth.OAuth") as oauth_type,
        ):
            await create_http_oauth("https://example.test/mcp")

        oauth_type.assert_called_once_with(
            "https://example.test/mcp",
            additional_client_metadata=None,
        )

    async def test_ordinary_http_clients_enable_native_oauth_and_preserve_headers(self) -> None:
        state = MCPServerState(
            name="remote",
            transport="http",
            config={
                "transport": "http",
                "url": "https://example.test/mcp",
                "headers": {"X-Project": "mira"},
            },
            fingerprint="test",
        )
        oauth = object()

        with (
            patch(
                "agent.mcp.integration.create_http_oauth",
                new=AsyncMock(return_value=oauth),
            ) as create_oauth,
            patch("agent.mcp.integration.Client") as client_type,
            patch("agent.mcp.integration.MCPAdapter") as adapter_type,
        ):
            integration = MiraMCPIntegration(state)
            await integration.__aenter__()

        create_oauth.assert_awaited_once_with("https://example.test/mcp")
        transport = client_type.call_args.args[0]
        self.assertIsInstance(transport, StreamableHttpTransport)
        self.assertEqual(transport.url, "https://example.test/mcp")
        self.assertEqual(transport.headers, {"X-Project": "mira"})
        self.assertIs(client_type.call_args.kwargs["auth"], oauth)
        adapter_type.return_value.__aenter__.assert_awaited_once_with()

    def test_stdio_clients_do_not_receive_http_oauth(self) -> None:
        state = MCPServerState(
            name="local",
            transport="stdio",
            config={"transport": "stdio", "command": "python", "args": [], "env": {}},
            fingerprint="test",
        )

        with (
            patch("agent.mcp.integration.Client") as client_type,
            patch("agent.mcp.integration.MCPAdapter"),
        ):
            MiraMCPIntegration(state)

        self.assertIsInstance(client_type.call_args.args[0], StdioTransport)
        self.assertNotIn("auth", client_type.call_args.kwargs)

    async def test_manager_starts_executes_and_stops_a_real_stdio_server(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            server_path = root / "stdio_server.py"
            server_path.write_text(
                """from fastmcp import FastMCP

server = FastMCP(\"stdio test\")

@server.tool
def ping(value: str) -> str:
    return f\"pong: {value}\"

if __name__ == \"__main__\":
    server.run(show_banner=False)
""",
                encoding="utf-8",
            )
            write_config(
                root,
                {"local": {"command": sys.executable, "args": [str(server_path)]}},
            )
            manager = MCPManager(root)

            async def approve(_state: object, _preview: str) -> str:
                return "allow"

            await manager.initialize(approve)
            try:
                state = manager.servers["local"]
                self.assertEqual(state.status, "Available")
                self.assertEqual([tool.name for tool in state.tools], ["mcp__local__ping"])
                result = await state.tools[0].ainvoke({"value": "ready"})
                self.assertEqual(
                    "".join(
                        block.get("text", "")
                        for block in result
                        if block.get("type") == "text"
                    ),
                    "pong: ready",
                )
            finally:
                await manager.shutdown()
            self.assertEqual(state.status, "Disabled")

    async def test_tools_execute_and_elicitation_interrupts_resume_or_decline(self) -> None:
        server = FastMCP("MIRA native MCP test")

        @server.tool
        async def greet(name: str) -> str:
            return f"Hello, {name}."

        @server.tool
        async def guarded_greet(ctx: Context) -> str | InputRequiredResult:
            responses = ctx.input_responses
            if responses is None:
                return InputRequiredResult(
                    inputRequests={
                        "identity": ElicitRequest(
                            params=ElicitRequestFormParams(
                                message="Who should be greeted?",
                                requestedSchema={
                                    "type": "object",
                                    "properties": {"name": {"type": "string"}},
                                    "required": ["name"],
                                },
                            )
                        )
                    },
                    requestState="greeting-round",
                )
            answer = responses["identity"]
            if answer.action == "accept":
                return f"Hello, {answer.content['name']}."
            return f"Greeting {answer.action}d."

        state = MCPServerState(
            name="native",
            transport="stdio",
            config={"command": "unused", "args": []},
            fingerprint="test",
        )
        integration = MiraMCPIntegration(state)
        integration.adapter = MCPAdapter(server)

        async with integration:
            descriptors = await integration.list_tool_descriptors()
            tools = {
                descriptor.name: await integration.as_tool(descriptor)
                for descriptor in descriptors
            }

            def tool_text(value) -> str:
                return "".join(
                    str(block.get("text") or "")
                    for block in value
                    if isinstance(block, dict) and block.get("type") == "text"
                )

            self.assertEqual(
                tool_text(await tools["greet"].ainvoke({"name": "MIRA"})),
                "Hello, MIRA.",
            )

            async def call_guarded(_state: dict[str, object]) -> dict[str, object]:
                return {"result": await tools["guarded_greet"].ainvoke({})}

            builder = StateGraph(dict)
            builder.add_node("call", call_guarded)
            builder.add_edge(START, "call")
            builder.add_edge("call", END)
            graph = builder.compile(checkpointer=InMemorySaver())

            accepted_config = {"configurable": {"thread_id": "mcp-accept"}}
            interrupted = await graph.ainvoke({}, accepted_config)
            payload = interrupted["__interrupt__"][0].value
            self.assertEqual(payload["type"], "mcp_elicitation")
            self.assertEqual(payload["tool_name"], "guarded_greet")
            self.assertEqual(payload["requests"][0]["key"], "identity")
            accepted = await graph.ainvoke(
                Command(
                    resume={
                        "responses": {
                            "identity": {
                                "action": "accept",
                                "content": {"name": "Ada"},
                            }
                        }
                    }
                ),
                accepted_config,
            )
            self.assertEqual(tool_text(accepted["result"]), "Hello, Ada.")

            declined_config = {"configurable": {"thread_id": "mcp-decline"}}
            await graph.ainvoke({}, declined_config)
            declined = await graph.ainvoke(
                Command(resume={"responses": {"identity": {"action": "decline"}}}),
                declined_config,
            )
            self.assertEqual(tool_text(declined["result"]), "Greeting declined.")


class Page:
    def __init__(self, field: str, values: list, cursor: str | None = None) -> None:
        setattr(self, field, values)
        self.nextCursor = cursor


def tool_descriptor(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, description=f"{name} description", inputSchema={"type": "object", "properties": {}}, annotations=None, meta=None)


class FakeSession:
    def __init__(
        self,
        tools: list[str] | None = None,
        *,
        capabilities: ServerCapabilities | None = None,
    ) -> None:
        self.tool_names = tools or []
        self.tool_pages = 0
        self.prompt_pages = 0
        self.resource_pages = 0
        self.calls: list[tuple[str, dict]] = []
        self.capabilities = capabilities or ServerCapabilities(
            tools=ToolsCapability(),
            prompts=PromptsCapability(),
            resources=ResourcesCapability(),
        )

    def get_server_capabilities(self) -> ServerCapabilities:
        return self.capabilities

    async def list_tools(self, cursor: str | None = None) -> Page:
        self.tool_pages += 1
        return Page("tools", [tool_descriptor(name) for name in self.tool_names])

    async def list_resources(self, cursor: str | None = None) -> Page:
        self.resource_pages += 1
        if cursor is None:
            return Page("resources", [SimpleNamespace(uri="repo://one", title="One", name="one", description="first", mimeType="text/plain")], "next")
        return Page("resources", [SimpleNamespace(uri="repo://two", title="Two", name="two", description="second", mimeType="text/plain")])

    async def list_prompts(self, cursor: str | None = None) -> Page:
        self.prompt_pages += 1
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
        self.opened: list[str] = []
        self.closed: list[str] = []
        self.entered_tasks: list[tuple[str, asyncio.Task | None]] = []
        self.exited_tasks: list[tuple[str, asyncio.Task | None]] = []

    def integration(self, name: str) -> FakeIntegration:
        return FakeIntegration(self, name)


class FakeIntegration:
    def __init__(self, owner: FakeClient, name: str) -> None:
        self.owner = owner
        self.name = name
        self.session = owner.sessions[name]

    async def __aenter__(self):
        name = self.name
        self.owner.opened.append(name)
        self.owner.entered_tasks.append((name, asyncio.current_task()))
        return self

    async def __aexit__(self, *_args) -> None:
        name = self.name
        self.owner.closed.append(name)
        self.owner.exited_tasks.append((name, asyncio.current_task()))

    def get_server_capabilities(self) -> ServerCapabilities:
        return self.session.get_server_capabilities()

    async def list_tool_descriptors(self) -> list:
        return list((await self.session.list_tools()).tools)

    async def as_tool(self, descriptor) -> StructuredTool:
        async def call(**arguments):
            return await self.session.call_tool(descriptor.name, arguments)

        return StructuredTool(
            name=descriptor.name,
            description=descriptor.description,
            args_schema=descriptor.inputSchema,
            coroutine=call,
        )

    async def list_prompts(self) -> list:
        return list((await self.session.list_prompts()).prompts)

    async def get_prompt_messages(self, name: str, arguments: dict | None = None) -> list:
        response = await self.session.get_prompt(name, arguments)
        messages = []
        for message in response.messages:
            cls = HumanMessage if message.role == "user" else AIMessage
            messages.append(cls(message.content.text))
        return messages

    async def list_resources(self) -> list:
        resources = []
        cursor = None
        while True:
            page = await self.session.list_resources(cursor)
            resources.extend(page.resources)
            cursor = page.nextCursor
            if not cursor:
                return resources

    async def read_resource(self, uri: str) -> list:
        return list((await self.session.read_resource(uri)).contents)

class MCPManagerTests(unittest.IsolatedAsyncioTestCase):
    async def make_manager(self, root: Path, servers: dict[str, dict], sessions: dict[str, FakeSession]) -> MCPManager:
        write_config(root, servers)
        manager = MCPManager(root)
        manager.client = FakeClient(sessions)
        manager._new_integration = lambda state: manager.client.integration(state.name)

        async def allow(_state, _preview):
            return "allow"

        await manager.initialize(allow)
        return manager

    async def test_all_stdio_connections_disable_sdk_stderr_logging(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_config(
                root,
                {
                    "implicit": {"command": "first", "args": []},
                    "explicit": {"type": "stdio", "command": "second", "args": []},
                    "remote": {"type": "http", "url": "https://example.test/mcp"},
                },
            )
            manager = MCPManager(root)
            transports = [
                manager._new_integration(state).client.transport
                for state in manager.servers.values()
                if state.transport == "stdio"
            ]

        self.assertEqual([item.command for item in transports], ["first", "second"])
        self.assertTrue(all(isinstance(item, StdioTransport) for item in transports))
        self.assertTrue(all(item.log_file == Path(os.devnull) for item in transports))

    async def test_persistent_runtime_eager_caches_and_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = FakeSession(["search"])
            manager = await self.make_manager(root, {"github": {"command": "fake", "args": []}}, {"github": session})
            self.assertEqual(manager.client.opened, ["github"])
            self.assertEqual([tool.name for tool in manager.servers["github"].tools], ["mcp__github__search"])
            self.assertEqual(len(manager.servers["github"].prompts or []), 1)
            self.assertEqual(len(manager.servers["github"].resources or []), 2)
            self.assertEqual(session.prompt_pages, 1)
            self.assertEqual(session.resource_pages, 2)
            await manager.discover_resources()
            await manager.discover_resources()
            self.assertEqual(session.resource_pages, 2)
            self.assertEqual(set(manager.resource_registry), {"mcp__github__repo://one", "mcp__github__repo://two"})
            await manager.set_server_enabled("github", False)
            self.assertEqual(manager.client.closed, ["github"])
            self.assertEqual(manager.servers["github"].status, "Disabled")
            self.assertEqual(manager.resource_registry, {})

    async def test_starting_status_remains_until_advertised_discovery_finishes(self) -> None:
        class GatedPrompts(FakeSession):
            def __init__(self) -> None:
                super().__init__(["search"])
                self.discovery_started = asyncio.Event()
                self.release_discovery = asyncio.Event()

            async def list_prompts(self, cursor=None):
                self.prompt_pages += 1
                self.discovery_started.set()
                await self.release_discovery.wait()
                argument = SimpleNamespace(name="topic", required=True)
                return Page(
                    "prompts",
                    [SimpleNamespace(name="review", description="Review", arguments=[argument])],
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_config(root, {"gated": {"command": "fake", "args": []}})
            session = GatedPrompts()
            manager = MCPManager(root)
            manager.client = FakeClient({"gated": session})
            manager._new_integration = lambda state: manager.client.integration(state.name)

            async def allow(_state, _preview):
                return "allow"

            initialization = asyncio.create_task(manager.initialize(allow))
            await session.discovery_started.wait()
            state = manager.servers["gated"]
            self.assertEqual(state.status, "Starting")
            self.assertIsNone(state.prompts)
            self.assertIsNone(state.resources)

            session.release_discovery.set()
            await initialization
            self.assertEqual(state.status, "Available")
            self.assertEqual(len(state.prompts or []), 1)
            self.assertEqual(len(state.resources or []), 2)
            await manager.shutdown()

    async def test_unadvertised_capabilities_are_empty_without_degrading_health(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions = {
                "tools-only": FakeSession(
                    ["search"],
                    capabilities=ServerCapabilities(tools=ToolsCapability()),
                ),
                "prompts-only": FakeSession(
                    capabilities=ServerCapabilities(prompts=PromptsCapability()),
                ),
                "resources-only": FakeSession(
                    capabilities=ServerCapabilities(resources=ResourcesCapability()),
                ),
            }
            manager = await self.make_manager(
                root,
                {name: {"command": "fake", "args": []} for name in sessions},
                sessions,
            )

            for name, session in sessions.items():
                with self.subTest(server=name):
                    state = manager.servers[name]
                    self.assertEqual(state.status, "Available")
                    self.assertEqual(state.error, "")
                    self.assertEqual(state.prompt_error, "")
                    self.assertEqual(state.resource_error, "")

                    if name != "tools-only":
                        self.assertEqual(state.tools, [])
                        self.assertEqual(session.tool_pages, 0)
                    if name != "prompts-only":
                        self.assertEqual(state.prompts, [])
                        self.assertEqual(session.prompt_pages, 0)
                    if name != "resources-only":
                        self.assertEqual(state.resources, [])
                        self.assertEqual(session.resource_pages, 0)

            self.assertEqual(len(manager.servers["tools-only"].tools), 1)
            self.assertEqual(len(manager.servers["prompts-only"].prompts or []), 1)
            self.assertEqual(len(manager.servers["resources-only"].resources or []), 2)
            await manager.shutdown()

    async def test_advertised_capability_failure_is_partial_during_startup(self) -> None:
        class BrokenPrompts(FakeSession):
            async def list_prompts(self, cursor=None):
                self.prompt_pages += 1
                raise ValueError("bad prompts payload")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = BrokenPrompts(["search"])
            manager = await self.make_manager(
                root,
                {"broken": {"command": "fake", "args": []}},
                {"broken": session},
            )
            state = manager.servers["broken"]
            self.assertEqual(state.status, "Partially available")
            self.assertEqual(state.prompts, [])
            self.assertIn("prompts: ValueError: bad prompts payload", state.error)
            self.assertEqual(session.prompt_pages, 1)
            await manager.shutdown()

    async def test_failure_isolation_denial_and_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_config(root, {"a": {"command": "a", "args": []}, "b": {"command": "b", "args": []}})
            manager = MCPManager(root)
            client = FakeClient({"a": FakeSession(), "b": FakeSession()})
            manager.client = client
            manager._new_integration = lambda state: client.integration(state.name)

            async def approve(state, _preview):
                return "deny" if state.name == "a" else "allow"

            await manager.initialize(approve)
            self.assertEqual(manager.servers["a"].status, "Approval required")
            self.assertEqual(manager.servers["b"].status, "Available")
            self.assertTrue(mcp_server_enabled(load_settings(root), "a"))
            await manager.restart_server("b")
            self.assertEqual(client.opened.count("b"), 2)
            self.assertEqual(client.closed.count("b"), 1)
            self.assertEqual(client.sessions["b"].tool_pages, 2)
            self.assertEqual(client.sessions["b"].prompt_pages, 2)
            self.assertEqual(client.sessions["b"].resource_pages, 4)
            entered = [task for name, task in client.entered_tasks if name == "b"]
            exited = [task for name, task in client.exited_tasks if name == "b"]
            self.assertIs(exited[0], entered[0])
            self.assertNotIn("a", client.closed)
            await manager.shutdown()

    async def test_restart_preserves_other_server_session_and_capabilities(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = await self.make_manager(
                root,
                {"a": {"command": "a", "args": []}, "b": {"command": "b", "args": []}},
                {"a": FakeSession(["replace"]), "b": FakeSession(["keep"])},
            )
            other = manager.servers["b"]
            other_session = other.session
            other_tools = list(other.tools)

            self.assertTrue(await manager.restart_server("a"))

            self.assertIs(other.session, other_session)
            self.assertEqual(other.tools, other_tools)
            self.assertEqual(other.status, "Available")
            self.assertNotIn("b", manager.client.closed)
            await manager.shutdown()

    async def test_cancelled_restart_request_still_finishes_transition(self) -> None:
        class GatedSession(FakeSession):
            def __init__(self) -> None:
                super().__init__(["search"])
                self.tool_lists = 0
                self.restart_started = asyncio.Event()
                self.release_restart = asyncio.Event()

            async def list_tools(self, cursor=None):
                self.tool_lists += 1
                if self.tool_lists == 2:
                    self.restart_started.set()
                    await self.release_restart.wait()
                return await super().list_tools(cursor)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = GatedSession()
            manager = await self.make_manager(
                root,
                {"a": {"command": "a", "args": []}},
                {"a": session},
            )
            restart = asyncio.create_task(manager.restart_server("a"))
            await session.restart_started.wait()
            restart.cancel()
            session.release_restart.set()

            self.assertTrue(await restart)
            self.assertEqual(manager.servers["a"].status, "Available")
            self.assertEqual(manager.client.opened.count("a"), 2)
            self.assertEqual(manager.client.closed.count("a"), 1)
            await manager.shutdown()

    async def test_startup_preserves_actionable_native_oauth_errors(self) -> None:
        class FailedIntegration:
            async def __aenter__(self):
                try:
                    raise anyio.WouldBlock
                except anyio.WouldBlock:
                    raise RuntimeError("Token exchange failed: invalid client")

            async def __aexit__(self, *_args):
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_config(root, {"remote": {"type": "http", "url": "https://example.test/mcp"}})
            manager = MCPManager(root)
            manager._new_integration = lambda _state: FailedIntegration()

            async def approve(_state, _preview):
                return "allow"

            await manager.initialize(approve)

            self.assertEqual(manager.servers["remote"].status, "Failed")
            self.assertEqual(
                manager.servers["remote"].error,
                "RuntimeError: Token exchange failed: invalid client",
            )

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

            async def convert(_integration, descriptor):
                if descriptor.name == "bad":
                    raise ValueError("cannot convert")

                async def works():
                    return "works"

                return StructuredTool.from_function(
                    coroutine=works,
                    name=descriptor.name,
                    description="works",
                )

            with patch.object(FakeIntegration, "as_tool", autospec=True, side_effect=convert):
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
                prompts=[],
                resources=[],
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
                prompts=[],
                resources=[],
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
    CSS_PATH = "../ui/textual/styles/mira.tcss"

    def __init__(self, manager: PanelManager) -> None:
        super().__init__()
        self.manager = manager

    def compose(self) -> ComposeResult:
        yield Static("host")

    async def reload_runtime(self) -> None:
        """Stand in for MiraApp's shared /reload-runtime pathway."""

    def on_mount(self) -> None:
        self.push_screen(MCPPanelScreen(self.manager, self.reload_runtime))


class MCPPanelTests(unittest.IsolatedAsyncioTestCase):
    async def test_generated_empty_configuration_exposes_no_mcp_status_or_issues(self) -> None:
        from tests.test_textual_app import make_app

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ensure_project_examples(root)
            manager = MCPManager(root)
            self.assertFalse(manager.show_status)
            self.assertEqual(manager.issues, [])

            app = make_app(root, mcp_manager=manager)
            async with app.run_test():
                self.assertFalse(app.query_one("#mcp-status-button", Button).display)
                self.assertFalse(app.query_one("#issues-button", Button).display)
                self.assertEqual(app.issues, [])

    async def test_native_expansion_is_presentation_only_and_preserves_focus(self) -> None:
        manager = PanelManager()
        app = PanelApp(manager)
        async with app.run_test(size=(100, 35)) as pilot:
            await pilot.pause()
            screen = app.screen
            self.assertIsInstance(screen, MCPPanelScreen)
            first = screen.query_one("#mcp-details-one", Collapsible)
            second = screen.query_one("#mcp-details-two", Collapsible)
            first_title = first.query_one("CollapsibleTitle")
            first_title.focus()
            state = manager.servers["one"]
            before = (
                state.status,
                tuple(state.tools),
                tuple(state.prompts or []),
                tuple(state.resources or []),
                state.error,
                state.prompt_error,
                state.resource_error,
            )
            await pilot.press("enter")
            await pilot.pause()
            self.assertFalse(first.collapsed)
            self.assertTrue(second.collapsed)
            self.assertEqual(manager.discovered, [])
            self.assertEqual(
                (
                    state.status,
                    tuple(state.tools),
                    tuple(state.prompts or []),
                    tuple(state.resources or []),
                    state.error,
                    state.prompt_error,
                    state.resource_error,
                ),
                before,
            )
            self.assertTrue(first_title.has_focus)

    async def test_server_card_uses_header_counts_actions_structure(self) -> None:
        manager = PanelManager()
        app = PanelApp(manager)
        async with app.run_test(size=(62, 26)) as pilot:
            await pilot.pause()
            screen = app.screen
            card = screen.query_one("#mcp-card-one")
            header = screen.query_one("#mcp-header-one", Static)
            badge = screen.query_one("#mcp-card-one .mcp-status-badge", Static)
            capabilities = screen.query_one("#mcp-details-one", Collapsible)
            title = capabilities.query_one("CollapsibleTitle", Static)
            controls = screen.query_one("#mcp-card-one .mcp-controls")
            control_buttons = list(controls.query(Button))

            self.assertEqual(header.region.y, badge.region.y)
            self.assertEqual(header.region.height, 1)
            self.assertEqual(badge.region.height, 1)
            self.assertIn("[STDIO]", str(header.render()))
            self.assertEqual(
                capability_summary(manager.servers["one"]).plain,
                "Tools  0   ·   Prompts  0   ·   Resources  0",
            )
            self.assertIn("Tools  0", str(title.render()))
            self.assertEqual(capabilities.region.height, 1)
            self.assertEqual(capabilities.region.y, header.region.bottom)
            self.assertEqual(controls.region.height, 1)
            self.assertEqual(controls.region.y, capabilities.region.bottom)
            self.assertTrue(control_buttons)
            self.assertTrue(all(button.region.height == 1 for button in control_buttons))
            self.assertLessEqual(header.region.right, badge.region.x)
            self.assertLessEqual(card.region.right, screen.query_one("#mcp-scroll").region.right)
            self.assertEqual(card.styles.margin.right, 1)

    async def test_available_server_without_capabilities_needs_no_attention(self) -> None:
        manager = PanelManager()
        app = PanelApp(manager)
        async with app.run_test(size=(62, 26)) as pilot:
            await pilot.pause()
            screen = app.screen
            await pilot.click("#mcp-details-one > CollapsibleTitle", offset=(2, 0))
            await pilot.pause()

            badge = screen.query_one("#mcp-card-one .mcp-status-badge", Static)
            details = "\n".join(str(section.render()) for section in screen.query(".mcp-detail-section"))
            self.assertEqual(str(badge.render()), "Available")
            self.assertIn("No tools", details)
            self.assertIn("No prompts", details)
            self.assertIn("No resources", details)
            self.assertEqual(len(screen.query("#mcp-card-one .mcp-attention")), 0)
            self.assertEqual(mcp_summary_symbol([manager.servers["one"]]), "✓")

    async def test_diagnostics_stay_visible_in_title_and_lead_expanded_details(self) -> None:
        manager = PanelManager()
        state = manager.servers["one"]
        state.prompt_error = "Prompt discovery timed out."
        app = PanelApp(manager)

        async with app.run_test(size=(76, 30)) as pilot:
            await pilot.pause()
            screen = app.screen
            card = screen.query_one("#mcp-card-one")
            details = screen.query_one("#mcp-details-one", Collapsible)
            title = details.query_one("CollapsibleTitle", Static)
            attention = details.query_one(".mcp-attention", Static)

            self.assertTrue(details.collapsed)
            self.assertIn("▶ [!] Tools", str(title.render()))
            self.assertNotIn("Needs attention", str(title.render()))
            self.assertIn("available", card.classes)
            self.assertIn("warning", attention.classes)
            self.assertEqual(str(attention.render()), "Prompt discovery timed out.")

            await pilot.click("#mcp-details-one > CollapsibleTitle", offset=(2, 0))
            await pilot.pause()
            first_section = details.query_one(".mcp-detail-section", Static)
            self.assertLess(attention.region.y, first_section.region.y)

    async def test_failed_diagnostic_uses_error_severity(self) -> None:
        manager = PanelManager()
        state = manager.servers["one"]
        state.status = "Failed"
        state.error = "Connection refused."
        app = PanelApp(manager)

        async with app.run_test(size=(76, 30)):
            details = app.screen.query_one("#mcp-details-one", Collapsible)
            attention = details.query_one(".mcp-attention", Static)
            self.assertIn("failed", app.screen.query_one("#mcp-card-one").classes)
            self.assertIn("failed", attention.classes)
            self.assertIn("▶ [!] Tools", str(details.query_one("CollapsibleTitle", Static).render()))

    async def test_open_panel_refreshes_unnotified_startup_transitions(self) -> None:
        manager = PanelManager()
        state = manager.servers["one"]
        state.status = "Starting"
        state.transient = True
        app = PanelApp(manager)

        async with app.run_test(size=(62, 26)) as pilot:
            await pilot.pause()
            screen = app.screen
            self.assertIn("Starting", str(screen.query_one("#mcp-card-one .mcp-status-badge", Static).render()))

            state.status = "Available"
            state.transient = False
            state.tools.append(object())
            await pilot.pause(0.25)

            badge = screen.query_one("#mcp-card-one .mcp-status-badge", Static)
            title = screen.query_one("#mcp-details-one > CollapsibleTitle", Static)
            self.assertEqual(str(badge.render()), "Available")
            self.assertIn("Tools  1", str(title.render()))

    def test_capability_metrics_colour_labels_and_keep_counts_bold_white(self) -> None:
        expected = {
            "Tools": "#78d5cf",
            "Prompts": "#8fb9e8",
            "Resources": "#c7a0e8",
        }
        for label, colour in expected.items():
            metric = capability_metric(label, "3")
            self.assertEqual(metric.plain, f"{label}  3")
            self.assertEqual(str(metric.spans[0].style), colour)
            self.assertEqual(str(metric.spans[-1].style), "bold #eef7f8")

    async def test_expanding_scrolled_server_uses_native_collapsible_in_place(self) -> None:
        from tests.test_textual_app import make_app

        manager = PanelManager()
        template = vars(manager.servers["one"])
        manager.servers = {
            f"server-{index}": SimpleNamespace(
                **{
                    **template,
                    "name": f"server-{index}",
                    "tools": [],
                    "tool_metadata": [],
                    "prompts": None,
                    "resources": None,
                }
            )
            for index in range(8)
        }
        app = make_app(mcp_manager=manager)
        async with app.run_test(size=(76, 24)) as pilot:
            await pilot.click("#mcp-status-button")
            await pilot.pause()
            screen = app.screen
            scroll = screen.query_one("#mcp-scroll")
            scroll.scroll_to(y=scroll.max_scroll_y, animate=False, force=True, immediate=True)
            await pilot.pause()
            self.assertGreater(scroll.scroll_y, 0)

            details = screen.query_one("#mcp-details-server-7", Collapsible)
            await pilot.click("#mcp-details-server-7 > CollapsibleTitle", offset=(2, 0))
            await pilot.pause()

            self.assertFalse(details.collapsed)
            self.assertIs(details, screen.query_one("#mcp-details-server-7", Collapsible))
            self.assertEqual(manager.discovered, [])
            self.assertTrue(details.query_one("CollapsibleTitle").has_focus)

    def test_controls_status_colours_and_spinner_are_consistent(self) -> None:
        self.assertEqual(controls_for("Disabled"), ("Enable",))
        for status in ("Available", "Partially available", "Approval required", "Failed"):
            self.assertEqual(controls_for(status), ("Restart", "Disable"))
        self.assertEqual(status_class("Available"), "available")
        self.assertEqual(status_class("Failed"), "failed")
        self.assertEqual(status_badge("Available"), "Available")
        self.assertEqual(status_badge("Partially available"), "Partial")
        self.assertEqual(status_badge("Failed"), "Failed")
        self.assertEqual(SPINNER_FRAMES, ("|", "/", "-", "\\"))

    def test_summary_symbol_uses_explicit_worst_state(self) -> None:
        states = lambda *statuses: [SimpleNamespace(status=status) for status in statuses]
        self.assertEqual(mcp_summary_symbol(states("Available", "Available")), "✓")
        self.assertEqual(mcp_summary_symbol(states("Available", "Disabled")), "!")
        self.assertEqual(mcp_summary_symbol(states("Disabled", "Disabled")), "–")
        self.assertEqual(mcp_summary_symbol(states("Starting"), spinner=1), "/")
        self.assertEqual(mcp_summary_symbol(states("Starting", "Failed")), "x")

    async def test_app_button_and_slash_open_the_same_panel_pathway(self) -> None:
        from tests.test_textual_app import make_app

        manager = PanelManager()
        app = make_app(mcp_manager=manager)
        async with app.run_test(size=(100, 35)) as pilot:
            await pilot.pause()
            button = app.query_one("#mcp-status-button", Button)
            self.assertTrue(button.display)
            self.assertEqual(str(button.label), "! MCP 1/2")
            self.assertEqual(button.parent.id, "status-row")
            self.assertTrue(
                {"available", "warning", "failed", "transient"}.isdisjoint(button.classes)
            )
            await pilot.click("#mcp-status-button")
            await pilot.pause()
            self.assertIsInstance(app.screen, MCPPanelScreen)
            self.assertEqual(str(app.screen.query_one("#mcp-title-close", Button).label), "x")
            actions = list(app.screen.query_one("#mcp-actions").query(Button))
            self.assertEqual([button.id for button in actions], ["mcp-reload", "mcp-close"])
            self.assertEqual([str(button.label) for button in actions], ["Reload Runtime", "Close"])
            header = app.screen.query_one("#mcp-header-one", Static)
            self.assertIn("one", str(header.render()))
            self.assertIn("[STDIO]", str(header.render()))
            app.screen.dismiss()
            await pilot.pause()
            prompt = app.query_one(PromptBox)
            await app.submit_prompt(PromptBox.Submitted(prompt, "/mcp"))
            await pilot.pause()
            self.assertIsInstance(app.screen, MCPPanelScreen)

    async def test_reload_button_uses_full_runtime_path_and_close_dismisses(self) -> None:
        """MCP footer actions should reload the full runtime and close the panel."""
        from tests.test_textual_app import make_app, renderable_plain, wait_until
        from ui.textual.widgets import ChatLog

        manager = PanelManager()
        app = make_app(mcp_manager=manager)
        app._reload_agents = AsyncMock()  # type: ignore[method-assign]
        app._reload_runtime = AsyncMock()  # type: ignore[method-assign]

        async with app.run_test(size=(100, 35)) as pilot:
            await pilot.click("#mcp-status-button")
            await pilot.pause()
            screen = app.screen
            self.assertIsInstance(screen, MCPPanelScreen)

            await pilot.click("#mcp-reload")
            await wait_until(lambda: app._reload_runtime.await_count == 1)
            await wait_until(
                lambda: bool(buttons := list(screen.query("#mcp-reload"))) and buttons[0].has_focus
            )
            app._reload_runtime.assert_awaited_once_with()
            app._reload_agents.assert_not_awaited()
            rendered = "\n".join(
                renderable_plain(block) for block in app.query_one(ChatLog).children
            )
            self.assertIn("runtime reloaded", rendered)
            self.assertNotIn("agent reloaded", rendered)

            await pilot.click("#mcp-close")
            await pilot.pause()
            self.assertNotIsInstance(app.screen, MCPPanelScreen)

    async def test_closing_panel_reveals_approval_and_same_reload_completes(self) -> None:
        """Closing MCP should expose its pending approval without cancelling reload."""
        from tests.test_textual_app import make_app, wait_until

        manager = PanelManager()
        app = make_app(mcp_manager=manager)
        decisions: list[str] = []
        reload_completed = asyncio.Event()

        async def reload_runtime() -> None:
            decision = await app.approve_mcp_server(
                SimpleNamespace(name="new-server"),
                "command: example-mcp-server",
            )
            decisions.append(decision)
            reload_completed.set()

        app._reload_runtime = reload_runtime  # type: ignore[method-assign]

        async with app.run_test(size=(100, 35)) as pilot:
            await pilot.click("#mcp-status-button")
            await pilot.pause()
            screen = app.screen
            self.assertIsInstance(screen, MCPPanelScreen)

            await pilot.click("#mcp-reload")
            prompt_panel = app.query_one(PromptPanel)
            prompt = app.query_one(PromptBox)
            await wait_until(lambda: prompt_panel.active)

            self.assertTrue(screen.reloading)
            self.assertTrue(screen.query_one("#mcp-reload", Button).disabled)
            self.assertFalse(screen.query_one("#mcp-close", Button).disabled)
            self.assertFalse(screen.query_one("#mcp-title-close", Button).disabled)
            self.assertTrue(prompt.disabled)

            await pilot.click("#mcp-close")
            await wait_until(lambda: not isinstance(app.screen, MCPPanelScreen))
            self.assertTrue(prompt_panel.active)
            self.assertTrue(prompt_panel.display)

            await pilot.click("#prompt-choice-0")
            await wait_until(reload_completed.is_set)
            await wait_until(lambda: not app._reload_in_progress)

            self.assertEqual(decisions, ["allow"])
            self.assertFalse(prompt.disabled)
            self.assertNotIsInstance(app.screen, MCPPanelScreen)

    async def test_reload_during_existing_approval_reveals_prompt_without_reloading(self) -> None:
        """Reload should not create a second prompt over an existing MCP approval."""
        from tests.test_textual_app import make_app, renderable_plain, wait_until
        from ui.textual.widgets import ChatLog

        manager = PanelManager()
        app = make_app(mcp_manager=manager)
        app._reload_runtime = AsyncMock()  # type: ignore[method-assign]

        async with app.run_test(size=(100, 35)) as pilot:
            approval = asyncio.create_task(
                app.approve_mcp_server(
                    SimpleNamespace(name="existing-server"),
                    "command: existing-mcp-server",
                )
            )
            prompt_panel = app.query_one(PromptPanel)
            await wait_until(lambda: prompt_panel.active)

            await pilot.click("#mcp-status-button")
            await pilot.pause()
            self.assertIsInstance(app.screen, MCPPanelScreen)

            await pilot.click("#mcp-reload")
            await wait_until(lambda: not isinstance(app.screen, MCPPanelScreen))

            app._reload_runtime.assert_not_awaited()
            self.assertTrue(prompt_panel.active)
            self.assertTrue(prompt_panel.display)
            await pilot.click("#prompt-choice-0")
            self.assertEqual(await approval, "allow")
            rendered = "\n".join(
                renderable_plain(block) for block in app.query_one(ChatLog).children
            )
            self.assertIn("answer the current prompt before reloading", rendered)

    async def test_escape_dismisses_panel_while_app_owned_reload_continues(self) -> None:
        """Every panel dismissal path should remain available during reload."""
        from tests.test_textual_app import make_app, wait_until

        manager = PanelManager()
        app = make_app(mcp_manager=manager)
        reload_started = asyncio.Event()
        release_reload = asyncio.Event()

        async def reload_runtime() -> None:
            reload_started.set()
            await release_reload.wait()

        app._reload_runtime = reload_runtime  # type: ignore[method-assign]

        async with app.run_test(size=(100, 35)) as pilot:
            await pilot.click("#mcp-status-button")
            await pilot.pause()
            screen = app.screen
            self.assertIsInstance(screen, MCPPanelScreen)

            await pilot.click("#mcp-reload")
            await wait_until(reload_started.is_set)
            self.assertFalse(screen.query_one("#mcp-close", Button).disabled)
            self.assertFalse(screen.query_one("#mcp-title-close", Button).disabled)

            await pilot.press("escape")
            await wait_until(lambda: not isinstance(app.screen, MCPPanelScreen))
            self.assertTrue(app._reload_in_progress)

            release_reload.set()
            await wait_until(lambda: not app._reload_in_progress)


if __name__ == "__main__":
    unittest.main()
