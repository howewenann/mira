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

from agent.llm import chat_anyllm_transport_kwargs, get_llm, get_model_name
from cli import commands
from cli.main import app as cli_app
from config.llm import ConfigError, DEFAULT_CONTEXT_TOKENS, load_llm_config
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
        self.assertEqual(config["llm_context_tokens"], DEFAULT_CONTEXT_TOKENS)

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
                        "MIRA_TOOL_OUTPUT_CHARS=321",
                        "MIRA_LMSTUDIO_METADATA_TIMEOUT=0.5",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(workspace)

        self.assertEqual(config["llm_provider"], "lmstudio")
        self.assertEqual(config["llm_model"], "from-dotenv")
        self.assertEqual(config["tool_output_chars"], 321)
        self.assertEqual(config["lmstudio_metadata_timeout"], 0.5)
        self.assertEqual(config["llm_context_tokens"], DEFAULT_CONTEXT_TOKENS)
        self.assertEqual(config["session_dir"], str(workspace / ".mira" / "_sessions"))

    def test_load_config_can_override_loaded_environment_from_dotenv(self) -> None:
        """Explicit reloads should let workspace .env replace existing process values."""
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
            os.environ["MIRA_LLM_PROVIDER"] = "lmstudio"
            os.environ["MIRA_LLM_MODEL"] = "already-loaded"

            config = load_config(workspace, override_dotenv=True)

        self.assertEqual(config["llm_model"], "from-dotenv")

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
        """LM Studio should use OpenAI-compatible transport and request final usage chunks."""
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
            provider="openai",
            api_base="http://localhost:1234/v1",
            api_key="lm-studio",
            stream_options={"include_usage": True},
        )

    def test_lmstudio_transport_preserves_config_identity(self) -> None:
        """LM Studio should keep MIRA identity while using OpenAI-compatible transport."""
        config = {
            "llm_provider": "lmstudio",
            "llm_model": "gemma-4-e4b",
        }

        self.assertEqual(config["llm_provider"], "lmstudio")
        self.assertEqual(
            chat_anyllm_transport_kwargs(config),
            {"model": "gemma-4-e4b", "provider": "openai"},
        )
        self.assertEqual(get_model_name(config), "lmstudio:gemma-4-e4b")

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
            self.assertEqual(llm.provider, "openai")
            self.assertEqual(get_model_name({"llm_provider": "lmstudio", "llm_model": "gemma-4-e4b"}), "lmstudio:gemma-4-e4b")
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

class CLIConfigTests(unittest.TestCase):
    """Tests for user-facing CLI config errors."""

    def test_help_includes_short_flags_without_workspace_default_path(self) -> None:
        """The CLI should expose short aliases without leaking cwd as a default."""
        result = CliRunner().invoke(cli_app, ["--help"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("-d", result.output)
        self.assertIn("--direct", result.output)
        self.assertIn("-f", result.output)
        self.assertIn("--file", result.output)
        self.assertIn("-h", result.output)
        self.assertIn("--help", result.output)
        self.assertIn("-w", result.output)
        self.assertIn("--workspace", result.output)
        self.assertNotIn("[default: ", result.output)

    def test_cli_file_options_pass_prompt_file_to_run(self) -> None:
        """The public file options should map to the internal prompt_file argument."""
        for option in ("--file", "-f"):
            with self.subTest(option=option), patch("cli.main.run") as run:
                result = CliRunner().invoke(cli_app, [option, "prompt.markdown"])

                self.assertEqual(result.exit_code, 0)
                run.assert_called_once()
                self.assertEqual(run.call_args.kwargs["prompt"], None)
                self.assertEqual(run.call_args.kwargs["prompt_file"], Path("prompt.markdown"))

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

    async def _captured_one_shot_text(
        self,
        *,
        prompt: str | None = None,
        prompt_file: Path | None = None,
        workspace: Path,
    ) -> tuple[str, str]:
        config = {
            "tool_output_chars": 123,
            "session_dir": "unused",
            "llm_provider": "lmstudio",
            "llm_model": "local-model",
        }
        session_record = {"id": "thread-1", "events": [], "turns": 0, "dashboard": {}}
        captured: list[str] = []

        async def ensure_git_repository(workspace: Path, guard_renderer: object) -> bool:
            return True

        async def bootstrap(
            workspace: Path,
            session: str | None,
            resume: bool,
            config: dict[str, object] | None = None,
            renderer: object | None = None,
        ) -> dict[str, object]:
            return {
                "agent": "agent",
                "renderer": renderer,
                "session": session_record,
                "store": type("Store", (), {"save": lambda self, record: None})(),
            }

        async def run_turn(agent: object, text: str, renderer: object, thread_id: str) -> object:
            captured.append(text)
            return type("Result", (), {"final_text": "done"})()

        with (
            patch("config.loader.load_config", return_value=config),
            patch("ui.renderer.Renderer", return_value=object()),
            patch("cli.git_guard.ensure_git_repository", ensure_git_repository),
            patch("cli.commands._bootstrap", bootstrap),
            patch("runtime.runner.run_turn", run_turn),
        ):
            await commands._run(
                prompt=prompt,
                resume=False,
                workspace=workspace,
                session=None,
                prompt_file=prompt_file,
            )

        return captured[0], session_record["events"][0]["text"]

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

        async def bootstrap(
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
        ) -> None:
            events.append("run_turn")
            self.assertEqual((agent, text, thread_id), ("agent", "hello", "thread-1"))
            self.assertIs(getattr(renderer, "renderer", None), renderer_obj)

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

        async def bootstrap(
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

    async def test_one_shot_records_system_error_when_turn_fails(self) -> None:
        """One-shot prompt failures should be visible in the saved session."""
        config = {
            "tool_output_chars": 123,
            "session_dir": "unused",
            "llm_provider": "lmstudio",
            "llm_model": "local-model",
        }
        session_record = {"id": "thread-1", "events": [], "turns": 0, "dashboard": {}}
        saved: list[dict[str, object]] = []

        async def ensure_git_repository(workspace: Path, guard_renderer: object) -> bool:
            return True

        async def bootstrap(
            workspace: Path,
            session: str | None,
            resume: bool,
            config: dict[str, object] | None = None,
            renderer: object | None = None,
        ) -> dict[str, object]:
            return {
                "agent": "agent",
                "renderer": renderer,
                "session": session_record,
                "store": type("Store", (), {"save": lambda self, record: saved.append(record.copy())})(),
            }

        async def run_turn(*args: object, **kwargs: object) -> object:
            raise RuntimeError("unexecuted tool call")

        with (
            patch("config.loader.load_config", return_value=config),
            patch("ui.renderer.Renderer", return_value=object()),
            patch("cli.git_guard.ensure_git_repository", ensure_git_repository),
            patch("cli.commands._bootstrap", bootstrap),
            patch("runtime.runner.run_turn", run_turn),
        ):
            with self.assertRaisesRegex(RuntimeError, "unexecuted tool call"):
                await commands._run(prompt="hello", resume=False, workspace=Path("."), session=None)

        self.assertEqual([event["type"] for event in session_record["events"]], ["user", "system_error"])
        self.assertIn("unexecuted tool call", session_record["events"][-1]["text"])
        self.assertTrue(saved)

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

    async def test_short_file_flag_reads_markdown_prompt(self) -> None:
        """The file prompt path should be read and sent to one-shot mode."""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "prompt.md").write_text("# Task\nDo the thing.\n", encoding="utf-8")

            run_text, recorded_text = await self._captured_one_shot_text(
                prompt_file=Path("prompt.md"),
                workspace=workspace,
            )

        self.assertEqual(run_text, "# Task\nDo the thing.\n")
        self.assertEqual(recorded_text, "# Task\nDo the thing.\n")

    async def test_long_file_flag_accepts_markdown_extension(self) -> None:
        """The long Markdown extension should be accepted for file prompts."""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "prompt.markdown").write_text("Use this prompt.\n", encoding="utf-8")

            run_text, recorded_text = await self._captured_one_shot_text(
                prompt_file=Path("prompt.markdown"),
                workspace=workspace,
            )

        self.assertEqual(run_text, "Use this prompt.\n")
        self.assertEqual(recorded_text, "Use this prompt.\n")

    async def test_prompt_text_remains_literal_when_markdown_file_exists(self) -> None:
        """The -p prompt text should not auto-read matching workspace files."""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "README.md").write_text("file contents\n", encoding="utf-8")

            run_text, recorded_text = await self._captured_one_shot_text(
                prompt="README.md",
                workspace=workspace,
            )

        self.assertEqual(run_text, "README.md")
        self.assertEqual(recorded_text, "README.md")

    async def test_prompt_and_file_flags_cannot_be_combined(self) -> None:
        """One-shot startup should reject ambiguous prompt input."""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "prompt.md").write_text("file contents\n", encoding="utf-8")

            with self.assertRaises(typer.Exit) as raised:
                await commands._run(
                    prompt="literal",
                    resume=False,
                    workspace=workspace,
                    session=None,
                    prompt_file=Path("prompt.md"),
                )

        self.assertEqual(raised.exception.exit_code, 2)

    async def test_file_flag_rejects_missing_directory_and_non_markdown_paths(self) -> None:
        """The file prompt input should fail before model startup for invalid paths."""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "notes.txt").write_text("not markdown\n", encoding="utf-8")
            (workspace / "folder.md").mkdir()

            cases = [Path("missing.md"), Path("folder.md"), Path("notes.txt")]
            for prompt_file in cases:
                with self.subTest(prompt_file=prompt_file):
                    with self.assertRaises(typer.Exit) as raised:
                        await commands._run(
                            prompt=None,
                            resume=False,
                            workspace=workspace,
                            session=None,
                            prompt_file=prompt_file,
                        )

                    self.assertEqual(raised.exception.exit_code, 2)


if __name__ == "__main__":
    unittest.main()
