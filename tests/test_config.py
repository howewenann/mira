"""Tests for environment configuration loading."""

from __future__ import annotations

import os
import ssl
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import httpx
import typer
from typer.testing import CliRunner

from agent.llm import get_llm, get_model_name
from agent.llm_httpx import ChatAnyLLMWithHttpx
from cli import commands
from cli.main import app as cli_app
from config.llm import ConfigError, load_llm_config
from config.loader import load_config


class LLMConfigTests(unittest.TestCase):
    """Tests for normalized LLM provider configuration."""

    def test_default_config_uses_lmstudio(self) -> None:
        """Missing LLM env vars should keep the local default."""
        config = load_llm_config({})

        self.assertEqual(config["llm_provider"], "lmstudio")
        self.assertEqual(config["llm_model"], "local-model")
        self.assertEqual(config["llm_base_url"], "http://localhost:1234/v1")
        self.assertEqual(config["llm_api_key"], "lm-studio")

    def test_canonical_config_loads_provider_values(self) -> None:
        """Canonical MIRA_LLM_* values should define the provider."""
        config = load_llm_config(
            {
                "MIRA_LLM_PROVIDER": "openai",
                "MIRA_LLM_MODEL": "gpt-4.1-mini",
                "MIRA_LLM_API_KEY": "key",
                "MIRA_LLM_BASE_URL": "https://example.test/v1",
                "MIRA_LLM_TEMPERATURE": "0.2",
                "MIRA_LLM_MAX_TOKENS": "1024",
                "MIRA_LLM_TOP_P": "0.9",
            }
        )

        self.assertEqual(config["llm_provider"], "openai")
        self.assertEqual(config["llm_model"], "gpt-4.1-mini")
        self.assertEqual(config["llm_api_key"], "key")
        self.assertEqual(config["llm_base_url"], "https://example.test/v1")
        self.assertEqual(config["llm_temperature"], 0.2)
        self.assertEqual(config["llm_max_tokens"], 1024)
        self.assertEqual(config["llm_top_p"], 0.9)

    def test_claude_provider_alias_uses_anthropic(self) -> None:
        """Claude should map to the Anthropic provider."""
        config = load_llm_config({"MIRA_LLM_PROVIDER": "claude", "MIRA_LLM_MODEL": "claude-sonnet"})

        self.assertEqual(config["llm_provider"], "anthropic")
        self.assertEqual(config["llm_model"], "claude-sonnet")

    def test_non_default_provider_requires_model(self) -> None:
        """Cloud providers should not silently inherit the local model name."""
        with self.assertRaisesRegex(ConfigError, "MIRA_LLM_MODEL is required"):
            load_llm_config({"MIRA_LLM_PROVIDER": "openai"})

    def test_canonical_values_require_provider(self) -> None:
        """A model without an explicit provider should be rejected."""
        with self.assertRaisesRegex(ConfigError, "MIRA_LLM_PROVIDER is required"):
            load_llm_config({"MIRA_LLM_MODEL": "gpt-4.1-mini"})

    def test_legacy_lmstudio_values_still_work(self) -> None:
        """Existing LM Studio env files should remain usable."""
        config = load_llm_config(
            {
                "MIRA_LMSTUDIO_MODEL": "old-model",
                "MIRA_LMSTUDIO_BASE_URL": "http://localhost:1234/v1",
                "MIRA_LMSTUDIO_API_KEY": "old-key",
            }
        )

        self.assertEqual(config["llm_provider"], "lmstudio")
        self.assertEqual(config["llm_model"], "old-model")
        self.assertEqual(config["llm_base_url"], "http://localhost:1234/v1")
        self.assertEqual(config["llm_api_key"], "old-key")

    def test_canonical_and_legacy_values_conflict(self) -> None:
        """Users should choose either canonical or old LM Studio names."""
        with self.assertRaisesRegex(ConfigError, "Use either MIRA_LLM"):
            load_llm_config(
                {
                    "MIRA_LLM_PROVIDER": "openai",
                    "MIRA_LLM_MODEL": "gpt-4.1-mini",
                    "MIRA_LMSTUDIO_MODEL": "local-model",
                }
            )

    def test_provider_specific_blocks_need_canonical_selector(self) -> None:
        """Non-LMStudio provider-specific blocks should direct users to canonical config."""
        with self.assertRaisesRegex(ConfigError, "MIRA_LLM_PROVIDER"):
            load_llm_config({"MIRA_OPENAI_MODEL": "gpt-4.1-mini", "MIRA_ANTHROPIC_MODEL": "claude-sonnet"})

    def test_invalid_generation_values_raise_config_error(self) -> None:
        """Generation values should fail clearly when malformed."""
        with self.assertRaisesRegex(ConfigError, "MIRA_LLM_TEMPERATURE"):
            load_llm_config(
                {
                    "MIRA_LLM_PROVIDER": "lmstudio",
                    "MIRA_LLM_TEMPERATURE": "warm",
                }
            )

    def test_load_config_reads_workspace_dotenv(self) -> None:
        """Workspace .env values should be loaded by the main config loader."""
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            workspace = Path(directory)
            (workspace / ".env").write_text(
                "\n".join(
                    [
                        "MIRA_LLM_PROVIDER=lmstudio",
                        "MIRA_LLM_MODEL=from-dotenv",
                        "MIRA_SESSION_MAX_CHARS=12345",
                        "MIRA_SESSION_RECENT_MESSAGES=7",
                        "MIRA_SESSION_SUMMARY_MAX_CHARS=2345",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(workspace)

        self.assertEqual(config["llm_provider"], "lmstudio")
        self.assertEqual(config["llm_model"], "from-dotenv")
        self.assertEqual(config["session_dir"], str(workspace / ".mira" / "_sessions"))
        self.assertEqual(config["session_max_chars"], 12345)
        self.assertEqual(config["session_recent_messages"], 7)
        self.assertEqual(config["session_summary_max_chars"], 2345)

    def test_get_llm_passes_normalized_config_to_chat_anyllm(self) -> None:
        """The LangChain model should be created from normalized LLM keys."""
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4.1-mini",
            "llm_api_key": "key",
            "llm_base_url": "https://example.test/v1",
            "llm_temperature": 0.2,
            "llm_max_tokens": 1024,
            "llm_top_p": 0.9,
        }

        with patch("agent.llm.ChatAnyLLM", return_value="llm") as chat:
            self.assertEqual(get_llm(config), "llm")

        chat.assert_called_once_with(
            model="gpt-4.1-mini",
            provider="openai",
            api_base="https://example.test/v1",
            api_key="key",
            temperature=0.2,
            max_tokens=1024,
            top_p=0.9,
        )

    def test_get_llm_requests_lmstudio_stream_usage(self) -> None:
        """LM Studio streaming should ask for final usage chunks."""
        config = {
            "llm_provider": "lmstudio",
            "llm_model": "gemma-4-e4b",
            "llm_api_key": "lm-studio",
            "llm_base_url": "http://localhost:1234/v1",
        }

        with patch("agent.llm.ChatAnyLLM", return_value="llm") as chat:
            self.assertEqual(get_llm(config), "llm")

        chat.assert_called_once_with(
            model="gemma-4-e4b",
            provider="lmstudio",
            api_base="http://localhost:1234/v1",
            api_key="lm-studio",
            stream_options={"include_usage": True},
        )

    def test_httpx_llm_wrapper_uses_pydantic_v2_config_without_warning(self) -> None:
        """The HTTPX-enabled wrapper should not use deprecated class-based config."""
        sync_client = httpx.Client()
        async_client = httpx.AsyncClient()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                llm = ChatAnyLLMWithHttpx(
                    model="test-model",
                    provider="lmstudio",
                    api_key="key",
                    api_base="http://localhost:1234/v1",
                    http_client=sync_client,
                    async_http_client=async_client,
                )

            self.assertIs(llm.http_client, sync_client)
            self.assertIs(llm.async_http_client, async_client)
            self.assertFalse(any("class-based" in str(warning.message).lower() for warning in caught))
        finally:
            sync_client.close()
            import asyncio

            asyncio.run(async_client.aclose())

    def test_get_llm_uses_insecure_direct_httpx_clients(self) -> None:
        """The insecure-direct config should build direct clients with TLS verification off."""
        llm = get_llm(
            {
                "llm_provider": "lmstudio",
                "llm_model": "gemma-4-e4b",
                "llm_api_key": "lm-studio",
                "llm_base_url": "http://localhost:1234/v1",
                "llm_insecure_direct": True,
            }
        )
        try:
            self.assertIsInstance(llm, ChatAnyLLMWithHttpx)
            self.assertIsNotNone(llm.http_client)
            self.assertIsNotNone(llm.async_http_client)
            self.assertFalse(llm.http_client.trust_env)
            self.assertFalse(llm.async_http_client.trust_env)
            self.assertEqual(llm.http_client._transport._pool._ssl_context.verify_mode, ssl.CERT_NONE)
            self.assertEqual(llm.async_http_client._transport._pool._ssl_context.verify_mode, ssl.CERT_NONE)
            self.assertFalse(llm.http_client._transport._pool._ssl_context.check_hostname)
            self.assertFalse(llm.async_http_client._transport._pool._ssl_context.check_hostname)
            self.assertEqual(llm.stream_options, {"include_usage": True})
        finally:
            if isinstance(llm, ChatAnyLLMWithHttpx):
                if llm.http_client is not None:
                    llm.http_client.close()
                if llm.async_http_client is not None:
                    import asyncio

                    asyncio.run(llm.async_http_client.aclose())

    def test_get_model_name_includes_provider(self) -> None:
        """The UI should identify both provider and model."""
        self.assertEqual(
            get_model_name({"llm_provider": "anthropic", "llm_model": "claude-sonnet"}),
            "anthropic:claude-sonnet",
        )


class CLIConfigTests(unittest.TestCase):
    """Tests for user-facing CLI config errors."""

    def test_help_includes_insecure_direct_flag_only(self) -> None:
        """The CLI should expose only the clean insecure-direct flag."""
        result = CliRunner().invoke(cli_app, ["--help"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("--insecure-direct", result.output)

    def test_run_prints_config_errors_without_traceback(self) -> None:
        """Config errors should exit cleanly through Typer."""
        with (
            patch("cli.commands._suppress_known_warnings"),
            patch("config.loader.load_config", side_effect=ConfigError("choose one provider")),
            patch("typer.echo") as echo,
        ):
            with self.assertRaises(typer.Exit) as raised:
                commands.run(prompt=None, resume=False, workspace=Path("."), session=None)

        self.assertEqual(raised.exception.exit_code, 2)
        echo.assert_called_once_with("Configuration error: choose one provider", err=True)


class CLIStartupTests(unittest.IsolatedAsyncioTestCase):
    """Tests for startup ordering around the Git safety guard."""

    async def test_run_checks_git_before_bootstrap(self) -> None:
        """The Git guard should run before sessions, resources, or agents are created."""
        events: list[str] = []
        config = {
            "tool_output_chars": 123,
            "session_dir": "unused",
            "llm_provider": "lmstudio",
            "llm_model": "local-model",
        }
        renderer = object()
        session_record = {"id": "thread-1"}
        case = self

        class Store:
            """Store double that records saves."""

            def save(self, record: dict[str, str]) -> None:
                """Record that the one-shot session was saved."""
                events.append("save")
                case.assertEqual(record["id"], session_record["id"])
                case.assertEqual(record["messages"][0]["content"], "hello")

        def load_config(workspace: Path) -> dict[str, object]:
            events.append("config")
            return config

        def make_renderer(tool_output_chars: int) -> object:
            events.append("renderer")
            self.assertEqual(tool_output_chars, 123)
            return renderer

        async def ensure_git_repository(workspace: Path, guard_renderer: object) -> bool:
            events.append("guard")
            self.assertIs(guard_renderer, renderer)
            return True

        def bootstrap(
            workspace: Path,
            session: str | None,
            resume: bool,
            config: dict[str, object] | None = None,
            renderer: object | None = None,
        ) -> dict[str, object]:
            events.append("bootstrap")
            self.assertIs(config, config_data)
            self.assertIs(renderer, renderer_obj)
            return {
                "agent": "agent",
                "renderer": renderer_obj,
                "session": session_record,
                "store": Store(),
            }

        async def run_turn(
            agent: object,
            text: str,
            renderer: object,
            thread_id: str,
            token_counter: object | None = None,
        ) -> None:
            events.append("run_turn")
            self.assertEqual((agent, text, renderer, thread_id), ("agent", "hello", renderer_obj, "thread-1"))
            self.assertIsNone(token_counter)

        config_data = config
        renderer_obj = renderer

        with (
            patch("config.loader.load_config", load_config),
            patch("ui.renderer.Renderer", make_renderer),
            patch("cli.git_guard.ensure_git_repository", ensure_git_repository),
            patch("cli.commands._bootstrap", bootstrap),
            patch("runtime.runner.run_turn", run_turn),
        ):
            await commands._run(prompt="hello", resume=False, workspace=Path("."), session=None)

        self.assertEqual(events, ["config", "renderer", "guard", "bootstrap", "run_turn", "save"])

    async def test_run_sets_insecure_direct_config_flag(self) -> None:
        """The CLI flag should be carried into bootstrap config."""
        config = {
            "tool_output_chars": 123,
            "session_dir": "unused",
            "llm_provider": "lmstudio",
            "llm_model": "local-model",
        }

        async def ensure_git_repository(workspace: Path, guard_renderer: object) -> bool:
            return True

        def bootstrap(
            workspace: Path,
            session: str | None,
            resume: bool,
            config: dict[str, object] | None = None,
            renderer: object | None = None,
        ) -> dict[str, object]:
            self.assertIsNotNone(config)
            self.assertTrue(config["llm_insecure_direct"])
            return {
                "agent": "agent",
                "renderer": renderer,
                "session": {"id": "thread-1"},
                "store": type("Store", (), {"save": lambda self, record: None})(),
            }

        async def run_turn(*args: object, **kwargs: object) -> object:
            return type("Result", (), {"final_text": "done"})()

        with (
            patch("config.loader.load_config", return_value=config),
            patch("ui.renderer.Renderer", return_value=object()),
            patch("cli.git_guard.ensure_git_repository", ensure_git_repository),
            patch("cli.commands._bootstrap", bootstrap),
            patch("runtime.runner.run_turn", run_turn),
        ):
            await commands._run(
                prompt="hello",
                resume=False,
                workspace=Path("."),
                session=None,
                insecure_direct=True,
            )

    async def test_run_exits_when_git_guard_blocks_startup(self) -> None:
        """Choosing exit after a Git failure should stop before bootstrap."""
        renderer = object()

        async def ensure_git_repository(workspace: Path, guard_renderer: object) -> bool:
            return False

        with (
            patch("config.loader.load_config", return_value={"tool_output_chars": 123}),
            patch("ui.renderer.Renderer", return_value=renderer),
            patch("cli.git_guard.ensure_git_repository", ensure_git_repository),
            patch("cli.commands._bootstrap") as bootstrap,
        ):
            with self.assertRaises(typer.Exit) as raised:
                await commands._run(prompt="hello", resume=False, workspace=Path("."), session=None)

        self.assertEqual(raised.exception.exit_code, 1)
        bootstrap.assert_not_called()


if __name__ == "__main__":
    unittest.main()
