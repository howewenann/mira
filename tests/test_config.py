"""Tests for environment configuration loading."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import typer

from agent.llm import get_llm, get_model_name
from cli import commands
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
                "MIRA_LLM_PROVIDER=lmstudio\nMIRA_LLM_MODEL=from-dotenv\n",
                encoding="utf-8",
            )

            config = load_config(workspace)

        self.assertEqual(config["llm_provider"], "lmstudio")
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

    def test_get_model_name_includes_provider(self) -> None:
        """The REPL splash should identify both provider and model."""
        self.assertEqual(
            get_model_name({"llm_provider": "anthropic", "llm_model": "claude-sonnet"}),
            "anthropic:claude-sonnet",
        )


class CLIConfigTests(unittest.TestCase):
    """Tests for user-facing CLI config errors."""

    def test_run_prints_config_errors_without_traceback(self) -> None:
        """Config errors should exit cleanly through Typer."""
        with (
            patch("cli.commands._suppress_known_warnings"),
            patch("cli.commands._bootstrap", side_effect=ConfigError("choose one provider")),
            patch("typer.echo") as echo,
        ):
            with self.assertRaises(typer.Exit) as raised:
                commands.run(prompt=None, resume=False, workspace=Path("."), session=None)

        self.assertEqual(raised.exception.exit_code, 2)
        echo.assert_called_once_with("Configuration error: choose one provider", err=True)


if __name__ == "__main__":
    unittest.main()
