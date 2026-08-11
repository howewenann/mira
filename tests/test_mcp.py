"""Focused MCP configuration, lifecycle, registry, and attachment tests."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import anyio
import httpx
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import ToolException
from langchain_mcp_adapters.callbacks import Callbacks
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from mcp.types import (
    PromptsCapability,
    ReadResourceResult,
    ResourcesCapability,
    ServerCapabilities,
    TextResourceContents,
    ToolsCapability,
)
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from agent.mcp.configuration import (
    adapter_connection,
    approval_preview,
    configuration_fingerprint,
    load_mcp_configuration,
    mcp_path,
)
from agent.mcp.auth import (
    FileTokenStorage,
    MiraOAuthProvider,
    OAuthLoginRequired,
    has_persisted_login,
    is_oauth_login_required,
    sanitized_error,
    server_token_directory,
)
from agent.mcp.manager import MCPManager
from agent.mcp.models import MCPResource, PromptArgument, PromptSpec
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
from ui.spinners import SPINNER_FRAMES
from ui.widgets.mcp_panel import (
    MCPPanelScreen,
    capability_metric,
    capability_summary,
    controls_for,
    mcp_summary_symbol,
    status_badge,
    status_class,
)
from ui.widgets import PromptBox


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
            adapter_connection(remote)["headers"]["Authorization"],
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

            example_path = active_path.parent / "example.json"
            active_path.write_text(example_path.read_text(encoding="utf-8"), encoding="utf-8")
            example = load_mcp_configuration(root, environ={"REMOTE_MCP_TOKEN": "resolved-secret"})
            self.assertEqual(set(example.servers), {"local-server", "remote-server"})
            self.assertEqual(example.servers["local-server"].transport, "stdio")
            self.assertEqual(
                example.servers["local-server"].config,
                {
                    "transport": "stdio",
                    "command": "python",
                    "args": ["/absolute/path/to/server.py"],
                    "env": {},
                },
            )
            self.assertEqual(example.servers["remote-server"].transport, "http")
            self.assertEqual(
                example.servers["remote-server"].config,
                {
                    "transport": "http",
                    "url": "https://example.com/mcp",
                    "headers": {"Authorization": "Bearer ${REMOTE_MCP_TOKEN}"},
                },
            )
            self.assertEqual(
                example.servers["remote-server"].connection_config["headers"],
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


def oauth_response_error(status: int, *, authenticate: str = "") -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.test/mcp")
    headers = {"WWW-Authenticate": authenticate} if authenticate else {}
    response = httpx.Response(status, request=request, headers=headers)
    return httpx.HTTPStatusError("request failed", request=request, response=response)


class MCPOAuthClassificationTests(unittest.IsolatedAsyncioTestCase):
    def state(self, *, transport: str = "http", headers: dict[str, str] | None = None):
        definition = (
            {"type": "http", "url": "https://example.test/mcp", "headers": headers or {}}
            if transport == "http"
            else {"command": "fake", "args": []}
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_config(root, {"linear": definition})
            return load_mcp_configuration(root).servers["linear"]

    async def test_valid_nested_mcp_oauth_challenge_requires_login(self) -> None:
        challenge = 'Bearer realm="mcp", resource_metadata="https://example.test/.well-known/oauth-protected-resource"'
        nested = ExceptionGroup("connection", [RuntimeError("wrapper", oauth_response_error(401, authenticate=challenge))])
        self.assertTrue(await is_oauth_login_required(nested, self.state()))

    async def test_plain_401_static_authorization_and_stdio_remain_failures(self) -> None:
        plain = oauth_response_error(401)
        with patch("agent.mcp.auth._discover_oauth_metadata", return_value=False):
            self.assertFalse(await is_oauth_login_required(plain, self.state()))
        self.assertFalse(
            await is_oauth_login_required(
                oauth_response_error(
                    401,
                    authenticate='Bearer resource_metadata="https://example.test/metadata"',
                ),
                self.state(headers={"Authorization": "Bearer configured"}),
            )
        )
        self.assertFalse(await is_oauth_login_required(plain, self.state(transport="stdio")))

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


class MCPOAuthStorageTests(unittest.IsolatedAsyncioTestCase):
    def state(self, name: str = "linear", url: str = "https://example.test/mcp"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_config(root, {name: {"type": "http", "url": url}})
            return load_mcp_configuration(root).servers[name]

    async def test_user_level_identity_storage_is_atomic_distinct_and_forgettable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            token_root = Path(directory) / "profile" / ".mira" / "_state" / "mcp-tokens"
            first = self.state("linear", "https://example.test/mcp/")
            equivalent = self.state("linear", "https://EXAMPLE.test:443/mcp")
            other = self.state("linear", "https://example.test/other")
            colliding_name = self.state("linear/a", "https://example.test/mcp")
            same_component = self.state("linear-a", "https://example.test/mcp")
            self.assertEqual(
                server_token_directory(first, token_root=token_root),
                server_token_directory(equivalent, token_root=token_root),
            )
            self.assertNotEqual(
                server_token_directory(first, token_root=token_root),
                server_token_directory(other, token_root=token_root),
            )
            self.assertNotEqual(
                server_token_directory(same_component, token_root=token_root),
                server_token_directory(colliding_name, token_root=token_root),
            )
            storage = FileTokenStorage(server_token_directory(first, token_root=token_root))
            await storage.set_tokens(OAuthToken(access_token="stored-access", refresh_token="stored-refresh", expires_in=3600))
            loaded = await FileTokenStorage(storage.directory).get_tokens()
            self.assertEqual(loaded.access_token, "stored-access")
            self.assertTrue(has_persisted_login(first, token_root=token_root))
            self.assertFalse((Path(directory) / ".mira" / "mcp" / "mcp.json").exists())
            await storage.clear()
            self.assertFalse(storage.directory.exists())

    async def test_stored_token_is_used_silently_and_missing_token_cannot_open_browser(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            token_root = Path(directory)
            state = self.state()
            opened: list[str] = []
            provider = MiraOAuthProvider(
                state,
                interactive=False,
                token_root=token_root,
                browser_opener=lambda url: opened.append(url) or True,
            )
            await provider.storage.set_tokens(OAuthToken(access_token="stored", expires_in=3600))

            async def respond(request: httpx.Request) -> httpx.Response:
                self.assertEqual(request.headers.get("Authorization"), "Bearer stored")
                return httpx.Response(200, request=request)

            async with httpx.AsyncClient(transport=httpx.MockTransport(respond), auth=provider) as client:
                response = await client.get("https://example.test/mcp")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(opened, [])
            with self.assertRaises(OAuthLoginRequired):
                await provider._redirect("https://auth.example/authorize?code=secret")
            self.assertEqual(opened, [])

    async def test_expired_stored_token_refreshes_without_browser(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = self.state()
            opened: list[str] = []
            provider = MiraOAuthProvider(
                state,
                interactive=False,
                token_root=Path(directory),
                browser_opener=lambda url: opened.append(url) or True,
            )
            await provider.storage.set_tokens(
                OAuthToken(access_token="expired", refresh_token="refresh", expires_in=-60)
            )
            await provider.storage.set_client_info(
                OAuthClientInformationFull(
                    client_id="dynamic-client",
                    redirect_uris=["http://127.0.0.1/callback"],
                    token_endpoint_auth_method="none",
                )
            )
            requests: list[str] = []

            async def respond(request: httpx.Request) -> httpx.Response:
                requests.append(str(request.url))
                if request.url.path == "/token":
                    return httpx.Response(
                        200,
                        request=request,
                        headers={"Content-Type": "application/json"},
                        json={"access_token": "fresh", "token_type": "Bearer", "expires_in": 3600},
                    )
                self.assertEqual(request.headers.get("Authorization"), "Bearer fresh")
                return httpx.Response(200, request=request)

            async with httpx.AsyncClient(transport=httpx.MockTransport(respond), auth=provider) as client:
                response = await client.get("https://example.test/mcp")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(requests, ["https://example.test/token", "https://example.test/mcp"])
            self.assertEqual(opened, [])

    async def test_static_authorization_header_ignores_stale_oauth_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            token_root = root / "profile-tokens"
            write_config(root, {"server": {"type": "http", "url": "https://example.test/mcp"}})
            state = load_mcp_configuration(root).servers["server"]
            storage = FileTokenStorage(server_token_directory(state, token_root=token_root))
            await storage.set_tokens(OAuthToken(access_token="stale", expires_in=3600))
            write_config(
                root,
                {
                    "server": {
                        "type": "http",
                        "url": "https://example.test/mcp",
                        "headers": {"Authorization": "Bearer configured"},
                    }
                },
            )
            manager = MCPManager(root, token_root=token_root)
            connection = manager.client.connections["server"]
            self.assertNotIn("auth", connection)
            self.assertEqual(connection["headers"]["Authorization"], "Bearer configured")


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
        self.callbacks = Callbacks()
        self.opened: list[str] = []
        self.closed: list[str] = []
        self.entered_tasks: list[tuple[str, asyncio.Task | None]] = []
        self.exited_tasks: list[tuple[str, asyncio.Task | None]] = []

    @asynccontextmanager
    async def session(self, name: str):
        self.opened.append(name)
        self.entered_tasks.append((name, asyncio.current_task()))
        try:
            yield self.sessions[name]
        finally:
            self.closed.append(name)
            self.exited_tasks.append((name, asyncio.current_task()))


class MCPManagerTests(unittest.IsolatedAsyncioTestCase):
    async def make_manager(self, root: Path, servers: dict[str, dict], sessions: dict[str, FakeSession]) -> MCPManager:
        write_config(root, servers)
        manager = MCPManager(root, token_root=root / "profile-tokens")
        manager.client = FakeClient(sessions)

        async def allow(_state, _preview):
            return "allow"

        await manager.initialize(allow)
        return manager

    async def test_all_stdio_connections_disable_sdk_stderr_logging(self) -> None:
        calls: list[tuple[str, object]] = []
        omitted = object()

        @asynccontextmanager
        async def fake_stdio_client(server, *, errlog=omitted):
            calls.append((server.command, errlog))
            yield object(), object()

        class FakeClientSession:
            def __init__(self, _read, _write, **_kwargs) -> None:
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args) -> None:
                pass

            async def initialize(self) -> None:
                pass

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
            manager = MCPManager(root, token_root=root / "profile-tokens")
            with (
                patch("agent.mcp.client.stdio_client", fake_stdio_client),
                patch("agent.mcp.client.ClientSession", FakeClientSession),
            ):
                for name, connection in manager.client.connections.items():
                    if connection["transport"] == "stdio":
                        async with manager.client.session(name):
                            pass

        self.assertEqual(calls, [("first", None), ("second", None)])

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
            manager = MCPManager(root, token_root=root / "profile-tokens")
            manager.client = FakeClient({"gated": session})

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

    async def test_approval_precedes_oauth_classification(self) -> None:
        class ChallengedClient:
            callbacks = Callbacks()
            connections = {
                "linear": {
                    "transport": "streamable_http",
                    "url": "https://example.test/mcp",
                    "headers": {},
                }
            }

            @asynccontextmanager
            async def session(self, _name: str):
                challenge = 'Bearer resource_metadata="https://example.test/.well-known/oauth-protected-resource"'
                raise oauth_response_error(401, authenticate=challenge)
                yield  # pragma: no cover

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_config(root, {"linear": {"type": "http", "url": "https://example.test/mcp"}})
            manager = MCPManager(root, token_root=root / "profile-tokens")
            manager.client = ChallengedClient()
            events: list[str] = []

            async def approve(_state, _preview):
                events.append("approved")
                return "allow"

            await manager.initialize(approve)
            self.assertEqual(events, ["approved"])
            self.assertEqual(manager.servers["linear"].status, "Login required")

    async def test_explicit_login_targets_one_mapping_and_notifies_registry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_config(
                root,
                {
                    "linear": {"type": "http", "url": "https://example.test/mcp"},
                    "other": {"type": "http", "url": "https://other.test/mcp"},
                },
            )
            manager = MCPManager(root, token_root=root / "profile-tokens")
            state = manager.servers["linear"]
            state.status = "Login required"
            other_mapping = manager.client.connections["other"]
            transitions: list[str] = []
            changes: list[str] = []

            async def start(target, *, restarting, force_approval=False):
                transitions.append(target.status)
                target.status = "Starting"
                transitions.append(target.status)
                target.status = "Available"

            async def changed():
                changes.append(state.status)

            manager._start_server = start
            manager.set_change_handler(changed)
            self.assertTrue(await manager.login_server("linear"))
            self.assertEqual(transitions, ["Authenticating", "Starting"])
            self.assertEqual(state.status, "Available")
            self.assertIs(manager.client.connections["other"], other_mapping)
            self.assertIn("auth", manager.client.connections["linear"])
            self.assertEqual(changes, ["Available"])

    async def test_failed_explicit_login_returns_to_login_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_config(root, {"linear": {"type": "http", "url": "https://example.test/mcp"}})
            manager = MCPManager(root, token_root=root / "profile-tokens")
            state = manager.servers["linear"]
            state.status = "Login required"

            async def fail(target, *, restarting, force_approval=False):
                target.status = "Failed"
                target.error = "Browser authorization was cancelled or denied."

            manager._start_server = fail
            self.assertFalse(await manager.login_server("linear"))
            self.assertEqual(state.status, "Login required")
            self.assertNotIn("token", state.error.casefold())

    async def test_later_oauth_failure_removes_only_target_capabilities(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = await self.make_manager(
                root,
                {
                    "linear": {"type": "http", "url": "https://example.test/mcp"},
                    "other": {"command": "other", "args": []},
                },
                {"linear": FakeSession(["search"]), "other": FakeSession(["keep"])},
            )
            target = manager.servers["linear"]
            manager._oauth_providers["linear"] = SimpleNamespace()
            changes: list[bool] = []

            async def changed():
                changes.append(True)

            manager.set_change_handler(changed)
            self.assertTrue(await manager._handle_later_auth_failure(target, oauth_response_error(401)))
            self.assertEqual(target.status, "Login required")
            self.assertEqual(target.tools, [])
            self.assertEqual(manager.client.closed, ["linear"])
            self.assertEqual(manager.servers["other"].status, "Available")
            self.assertEqual(len(manager.servers["other"].tools), 1)
            self.assertEqual(changes, [True])
            await manager.shutdown()

    async def test_forget_login_preserves_config_and_disabled_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            token_root = root / "profile-tokens"
            write_config(root, {"linear": {"type": "http", "url": "https://example.test/mcp"}})
            settings = set_mcp_server_enabled(load_settings(root), "linear", False)
            self.assertTrue(save_settings(root, settings))
            state = load_mcp_configuration(root).servers["linear"]
            storage = FileTokenStorage(server_token_directory(state, token_root=token_root))
            await storage.set_tokens(OAuthToken(access_token="stored", expires_in=3600))
            original = mcp_path(root).read_text(encoding="utf-8")
            manager = MCPManager(root, token_root=token_root)
            self.assertTrue(await manager.forget_server_login("linear"))
            self.assertEqual(manager.servers["linear"].status, "Disabled")
            self.assertFalse(storage.directory.exists())
            self.assertEqual(mcp_path(root).read_text(encoding="utf-8"), original)

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
        self.login_calls: list[str] = []
        self.forgotten: set[str] = set()

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

    def has_persisted_login(self, name: str) -> bool:
        return name in self.forgotten

    async def login_server(self, name: str) -> bool:
        self.login_calls.append(name)
        self.servers[name].status = "Available"
        return True

    async def forget_server_login(self, name: str) -> bool:
        self.forgotten.discard(name)
        self.servers[name].status = "Login required"
        return True


class PanelApp(App[None]):
    CSS_PATH = "../ui/styles/mira.tcss"

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
            manager = MCPManager(root, token_root=root / "profile-tokens")
            self.assertFalse(manager.show_status)
            self.assertEqual(manager.issues, [])

            app = make_app(root, mcp_manager=manager)
            async with app.run_test():
                self.assertFalse(app.query_one("#mcp-status-button", Button).display)
                self.assertFalse(app.query_one("#issues-button", Button).display)
                self.assertEqual(app.issues, [])

    async def test_expansion_is_presentation_only_and_preserves_clicked_focus(self) -> None:
        manager = PanelManager()
        app = PanelApp(manager)
        async with app.run_test(size=(100, 35)) as pilot:
            await pilot.pause()
            screen = app.screen
            self.assertIsInstance(screen, MCPPanelScreen)
            first = screen.query_one("#mcp-header-one", Button)
            second = screen.query_one("#mcp-header-two", Button)
            self.assertTrue(first.has_focus)
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
            self.assertIn("one", screen.expanded)
            self.assertNotIn("two", screen.expanded)
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
            self.assertFalse(second.has_focus)
            self.assertTrue(screen.query_one("#mcp-header-one", Button).has_focus)

    async def test_server_card_uses_header_counts_actions_structure(self) -> None:
        manager = PanelManager()
        app = PanelApp(manager)
        async with app.run_test(size=(62, 26)) as pilot:
            await pilot.pause()
            screen = app.screen
            card = screen.query_one("#mcp-card-one")
            header = screen.query_one("#mcp-header-one", Button)
            badge = screen.query_one("#mcp-card-one .mcp-status-badge", Static)
            counts = screen.query_one("#mcp-card-one .mcp-counts", Static)
            controls = screen.query_one("#mcp-card-one .mcp-controls")
            control_buttons = list(controls.query(Button))

            self.assertEqual(header.region.y, badge.region.y)
            self.assertEqual(header.region.height, 1)
            self.assertEqual(badge.region.height, 1)
            self.assertIn("[STDIO]", str(header.label))
            self.assertEqual(
                capability_summary(manager.servers["one"]).plain,
                "Tools  0   ·   Prompts  0   ·   Resources  0",
            )
            self.assertEqual(counts.styles.content_align[0], "left")
            self.assertEqual(counts.region.height, 1)
            self.assertEqual(counts.region.y, header.region.bottom)
            self.assertEqual(controls.region.height, 1)
            self.assertEqual(controls.region.y, counts.region.bottom)
            self.assertTrue(control_buttons)
            self.assertTrue(all(button.region.height == 1 for button in control_buttons))
            self.assertLessEqual(header.region.right, badge.region.x)
            self.assertLessEqual(card.region.right, screen.query_one("#mcp-scroll").region.right)

    async def test_available_server_without_capabilities_needs_no_attention(self) -> None:
        manager = PanelManager()
        app = PanelApp(manager)
        async with app.run_test(size=(62, 26)) as pilot:
            await pilot.pause()
            screen = app.screen
            await pilot.click("#mcp-header-one", offset=(2, 0))
            await pilot.pause()

            badge = screen.query_one("#mcp-card-one .mcp-status-badge", Static)
            details = "\n".join(str(section.render()) for section in screen.query(".mcp-detail-section"))
            self.assertEqual(str(badge.render()), "Available")
            self.assertIn("No tools", details)
            self.assertIn("No prompts", details)
            self.assertIn("No resources", details)
            self.assertEqual(len(screen.query(".mcp-error-block")), 0)
            self.assertEqual(mcp_summary_symbol([manager.servers["one"]]), "✓")

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
            counts = screen.query_one("#mcp-card-one .mcp-counts", Static)
            self.assertEqual(str(badge.render()), "Available")
            self.assertIn("Tools  1", str(counts.render()))

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

    async def test_expanding_scrolled_server_preserves_scroll_position(self) -> None:
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
            before = scroll.scroll_y
            self.assertGreater(before, 0)

            await pilot.click("#mcp-header-server-7", offset=(2, 0))
            manager.servers["server-0"].status = "Restarting"
            manager.servers["server-0"].transient = True
            for _ in range(5):
                await pilot.pause()

            self.assertIn("server-7", screen.expanded)
            self.assertEqual(manager.discovered, [])
            self.assertEqual(screen.query_one("#mcp-scroll").scroll_y, before)
            self.assertTrue(screen.query_one("#mcp-header-server-7", Button).has_focus)

    def test_controls_status_colours_and_spinner_are_consistent(self) -> None:
        self.assertEqual(controls_for("Disabled"), ("Enable",))
        self.assertEqual(controls_for("Disabled", persisted_login=True), ("Enable", "Forget login"))
        for status in ("Available", "Partially available", "Approval required", "Failed"):
            self.assertEqual(controls_for(status), ("Restart", "Disable"))
        self.assertEqual(controls_for("Login required"), ("Login", "Disable"))
        self.assertEqual(controls_for("Authenticating"), ("Login", "Disable"))
        self.assertEqual(
            controls_for("Login required", persisted_login=True),
            ("Login", "Disable", "Forget login"),
        )
        self.assertEqual(status_class("Available"), "available")
        self.assertEqual(status_class("Login required"), "warning")
        self.assertEqual(status_class("Authenticating"), "transient")
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

    async def test_login_control_runs_in_worker_and_authenticating_row_spins(self) -> None:
        manager = PanelManager()
        manager.servers["two"].status = "Login required"
        app = PanelApp(manager)
        async with app.run_test(size=(100, 35)) as pilot:
            await pilot.pause()
            await pilot.click("#mcp-login-two")
            await pilot.pause()
            self.assertEqual(manager.login_calls, ["two"])
            self.assertEqual(manager.servers["two"].status, "Available")

            manager.servers["two"].status = "Authenticating"
            manager.servers["two"].transient = True
            await app.screen.refresh_from_manager()
            login = app.screen.query_one("#mcp-login-two", Button)
            disable = app.screen.query_one("#mcp-disable-two", Button)
            header = app.screen.query_one("#mcp-header-two", Button)
            self.assertTrue(login.disabled)
            self.assertTrue(disable.disabled)
            self.assertIn("transient", header.classes)
            badge = app.screen.query_one("#mcp-card-two .mcp-status-badge", Static)
            self.assertIn("Authenticating", str(badge.render()))

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
            self.assertIn("one", str(app.screen.query_one("#mcp-header-one", Button).label))
            self.assertIn("[STDIO]", str(app.screen.query_one("#mcp-header-one", Button).label))
            app.screen.dismiss()
            await pilot.pause()
            prompt = app.query_one(PromptBox)
            await app.submit_prompt(PromptBox.Submitted(prompt, "/mcp"))
            await pilot.pause()
            self.assertIsInstance(app.screen, MCPPanelScreen)

    async def test_reload_button_uses_full_runtime_path_and_close_dismisses(self) -> None:
        """MCP footer actions should reload the full runtime and close the panel."""
        from tests.test_textual_app import make_app, renderable_plain, wait_until
        from ui.widgets import ChatLog

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
            await wait_until(lambda: screen.query_one("#mcp-reload", Button).has_focus)
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


if __name__ == "__main__":
    unittest.main()
