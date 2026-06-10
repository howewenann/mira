"""Tests for environment configuration loading."""

from __future__ import annotations

import asyncio
import os
import ssl
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import typer
from langchain_anyllm import ChatAnyLLM
from typer.testing import CliRunner

from agent.llm import get_llm, get_model_name
from cli import commands
from cli.main import app as cli_app
from config.llm import ConfigError, load_llm_config
from config.loader import load_config
from session.dashboard import context_limit_for_config, context_limit_for_model, token_counter_for_model


class ProfileModel:
    """Tiny model double exposing the LangChain profile field MIRA reads."""

    def __init__(self, profile: dict[str, object] | None = None) -> None:
        self.profile = profile


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
                "MIRA_LLM_CONTEXT_TOKENS": "8192",
            }
        )

        self.assertEqual(config["llm_provider"], "openai")
        self.assertEqual(config["llm_model"], "gpt-4.1-mini")
        self.assertEqual(config["llm_api_key"], "key")
        self.assertEqual(config["llm_base_url"], "https://example.test/v1")
        self.assertEqual(config["llm_temperature"], 0.2)
        self.assertEqual(config["llm_max_tokens"], 1024)
        self.assertEqual(config["llm_top_p"], 0.9)
        self.assertEqual(config["llm_context_tokens"], 8192)

    def test_non_default_provider_requires_model(self) -> None:
        """Cloud providers should not silently inherit the local model name."""
        with self.assertRaisesRegex(ConfigError, "MIRA_LLM_MODEL is required"):
            load_llm_config({"MIRA_LLM_PROVIDER": "openai"})

    def test_canonical_values_require_provider(self) -> None:
        """A model without an explicit provider should be rejected."""
        with self.assertRaisesRegex(ConfigError, "MIRA_LLM_PROVIDER is required"):
            load_llm_config({"MIRA_LLM_MODEL": "gpt-4.1-mini"})

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
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(workspace)

        self.assertEqual(config["llm_provider"], "lmstudio")
        self.assertEqual(config["llm_model"], "from-dotenv")
        self.assertEqual(config["session_dir"], str(workspace / ".mira" / "_sessions"))

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

    def test_get_llm_direct_uses_anyllm_client_args(self) -> None:
        """Direct mode should use AnyLLM client_args with a direct async HTTPX client."""
        llm = get_llm(
            {
                "llm_provider": "lmstudio",
                "llm_model": "gemma-4-e4b",
                "llm_api_key": "lm-studio",
                "llm_base_url": "http://localhost:1234/v1",
                "llm_direct": True,
            }
        )
        client: httpx.AsyncClient | None = None
        try:
            self.assertIsInstance(llm, ChatAnyLLM)
            client = llm.model_kwargs["client_args"]["http_client"]
            self.assertIsInstance(client, httpx.AsyncClient)
            self.assertFalse(client.trust_env)
            self.assertEqual(client._transport._pool._ssl_context.verify_mode, ssl.CERT_NONE)
            self.assertFalse(client._transport._pool._ssl_context.check_hostname)
            self.assertEqual(llm.stream_options, {"include_usage": True})
            self.assertFalse(llm.disable_streaming)
        finally:
            if client is not None:
                asyncio.run(client.aclose())

    def test_get_model_name_includes_provider(self) -> None:
        """The UI should identify both provider and model."""
        self.assertEqual(
            get_model_name({"llm_provider": "anthropic", "llm_model": "claude-sonnet"}),
            "anthropic:claude-sonnet",
        )

    def test_context_limit_prefers_langchain_model_profile(self) -> None:
        """Dashboard limits should use the same profile field DeepAgents reads."""
        self.assertEqual(
            context_limit_for_model(ProfileModel({"max_input_tokens": 8192})),
            (8192, "model_profile.max_input_tokens"),
        )

    def test_context_limit_prefers_explicit_config(self) -> None:
        """Local models can provide the real context window when profiles are absent."""
        self.assertEqual(
            context_limit_for_config({"llm_context_tokens": 4096}, ProfileModel({"max_input_tokens": 8192})),
            (4096, "MIRA_LLM_CONTEXT_TOKENS"),
        )

    def test_context_limit_falls_back_to_deepagents_trigger(self) -> None:
        """Profile-less models should still show the effective compaction limit."""
        self.assertEqual(
            context_limit_for_model(ProfileModel(None)),
            (170000, "deepagents.compaction_trigger"),
        )

    def test_token_counter_uses_langchain_approximation(self) -> None:
        """Context estimates should not depend on the LM Studio SDK."""
        counter = token_counter_for_model(ProfileModel(None))

        self.assertGreater(counter("hello world"), 0)


class CLIConfigTests(unittest.TestCase):
    """Tests for user-facing CLI config errors."""

    def test_help_includes_short_flags_without_workspace_default_path(self) -> None:
        """The CLI should expose short aliases without leaking cwd as a default."""
        result = CliRunner().invoke(cli_app, ["--help"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("-d", result.output)
        self.assertIn("--direct", result.output)
        self.assertIn("-h", result.output)
        self.assertIn("--help", result.output)
        self.assertIn("-w", result.output)
        self.assertIn("--workspace", result.output)
        self.assertNotIn("[default: ", result.output)

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
                case.assertEqual(record["events"][0]["text"], "hello")

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
            self.assertEqual((agent, text, thread_id), ("agent", "hello", "thread-1"))
            self.assertIs(getattr(renderer, "renderer", None), renderer_obj)
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

        self.assertEqual(events, ["config", "renderer", "guard", "bootstrap", "save", "run_turn", "save"])

    async def test_run_sets_direct_config_flag(self) -> None:
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
            self.assertTrue(config["llm_direct"])
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
                direct=True,
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
