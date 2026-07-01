"""Tests for workspace settings persistence."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config import settings


class SettingsTests(unittest.TestCase):
    """Tests for .mira/settings.yml loading and normalization."""

    def test_missing_settings_yields_defaults(self) -> None:
        """Missing settings should keep Git protection and write approvals enabled."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            loaded = settings.load_settings(Path(directory))

        self.assertTrue(settings.git_protection_enabled(loaded))
        self.assertFalse(settings.tool_always_allow(loaded, "write_file"))
        self.assertFalse(settings.tool_always_allow(loaded, "edit_file"))
        self.assertFalse(settings.tool_always_allow(loaded, "eval"))
        self.assertFalse(settings.tool_always_allow(loaded, "task"))
        self.assertFalse(settings.tool_always_allow(loaded, "execute"))
        self.assertTrue(settings.tool_enabled(loaded, "write_file"))
        self.assertFalse(settings.tool_enabled(loaded, "execute"))
        self.assertFalse(settings.dynamic_subagents_enabled(loaded))
        self.assertTrue(settings.tool_always_allow(loaded, "web_search"))
        self.assertEqual(
            settings.execute_env_settings(loaded),
            {"mode": "system", "name": "", "prefix": "", "path": "", "allow": []},
        )

    def test_execute_env_settings_normalize_supported_modes(self) -> None:
        """Execute environment settings should keep names, paths, and allowlists only."""
        loaded = settings.normalize_settings(
            {
                "hitl": {
                    "execute_env": {
                        "mode": "conda_name",
                        "name": "project_env",
                        "prefix": r"C:\Users\me\.conda\envs\ignored",
                        "path": ".venv",
                        "allow": ["CUDA_HOME", "HF_HOME", "CUDA_HOME", "BAD=VALUE", "*", "bad-name"],
                    }
                }
            }
        )

        execute_env = settings.execute_env_settings(loaded)
        self.assertEqual(execute_env["mode"], "conda_name")
        self.assertEqual(execute_env["name"], "project_env")
        self.assertEqual(execute_env["prefix"], r"C:\Users\me\.conda\envs\ignored")
        self.assertEqual(execute_env["path"], ".venv")
        self.assertEqual(execute_env["allow"], ["CUDA_HOME", "HF_HOME"])

        self.assertEqual(
            settings.execute_env_settings({"hitl": {"execute_env": {"mode": "conda_prefix", "prefix": "C:/env"}}})[
                "mode"
            ],
            "conda_prefix",
        )
        self.assertEqual(
            settings.execute_env_settings({"hitl": {"execute_env": {"mode": "venv", "path": ".venv"}}})["mode"],
            "venv",
        )

    def test_execute_env_allow_accepts_comma_separated_names(self) -> None:
        """The UI helper should save env var names without saving values."""
        updated = settings.set_execute_env_allow(
            settings.DEFAULT_SETTINGS,
            "CUDA_HOME, HF_HOME, REQUESTS_CA_BUNDLE, TOKEN=value",
        )

        execute_env = settings.execute_env_settings(updated)
        self.assertEqual(execute_env["allow"], ["CUDA_HOME", "HF_HOME", "REQUESTS_CA_BUNDLE"])

    def test_dynamic_subagents_setting_defaults_off_and_can_toggle(self) -> None:
        """Dynamic eval subagents should be disabled unless explicitly enabled."""
        loaded = settings.normalize_settings({})

        self.assertFalse(settings.dynamic_subagents_enabled(loaded))

        updated = settings.set_dynamic_subagents(loaded, True)
        self.assertTrue(settings.dynamic_subagents_enabled(updated))

        updated = settings.set_dynamic_subagents(updated, False)
        self.assertFalse(settings.dynamic_subagents_enabled(updated))

    def test_partial_and_malformed_yaml_falls_back_safely(self) -> None:
        """Partial settings should merge with defaults; malformed YAML should not crash."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            path = settings.settings_path(workspace)
            path.parent.mkdir()
            path.write_text(
                "system:\n"
                "  dynamic_subagents:\n"
                "    enabled: true\n"
                "hitl:\n"
                "  git_protection:\n"
                "    enabled: false\n"
                "  tools:\n"
                "    write_file:\n"
                "      enabled: false\n"
                "      always_allow: true\n",
                encoding="utf-8",
            )

            loaded = settings.load_settings(workspace)
            self.assertTrue(settings.dynamic_subagents_enabled(loaded))
            self.assertFalse(settings.git_protection_enabled(loaded))
            self.assertFalse(settings.tool_enabled(loaded, "write_file"))
            self.assertTrue(settings.tool_always_allow(loaded, "write_file"))

            updated = settings.set_tool_enabled(loaded, "write_file", True)
            self.assertTrue(settings.tool_enabled(updated, "write_file"))
            updated = settings.set_tool_enabled(updated, "write_file", False)
            self.assertFalse(settings.tool_enabled(updated, "write_file"))

            updated = settings.set_tool_enabled(loaded, "execute", True)
            self.assertTrue(settings.tool_enabled(updated, "execute"))
            updated = settings.set_tool_enabled(updated, "execute", False)
            self.assertFalse(settings.tool_enabled(updated, "execute"))

            path.write_text("hitl: [", encoding="utf-8")
            loaded = settings.load_settings(workspace)
            self.assertTrue(settings.git_protection_enabled(loaded))
            self.assertFalse(settings.tool_always_allow(loaded, "write_file"))

    def test_save_settings_writes_expected_yaml(self) -> None:
        """Saving toggles should persist the normalized schema."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            updated = settings.set_git_protection(settings.load_settings(workspace), False)
            updated = settings.set_tool_always_allow(updated, "web_search", False)
            updated = settings.set_tool_enabled(updated, "web_search", False)

            self.assertTrue(settings.save_settings(workspace, updated))
            text = settings.settings_path(workspace).read_text(encoding="utf-8")
            loaded = settings.load_settings(workspace)

        self.assertIn("settings.yml", str(settings.settings_path(workspace)))
        self.assertIn("git_protection", text)
        self.assertIn("web_search", text)
        self.assertIn("enabled: false", text)
        self.assertFalse(settings.git_protection_enabled(loaded))
        self.assertFalse(settings.tool_always_allow(loaded, "web_search"))
        self.assertFalse(settings.tool_enabled(loaded, "web_search"))


if __name__ == "__main__":
    unittest.main()
