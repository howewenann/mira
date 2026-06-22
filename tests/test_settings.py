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
        self.assertTrue(settings.tool_enabled(loaded, "write_file"))
        self.assertTrue(settings.tool_always_allow(loaded, "web_search"))

    def test_partial_and_malformed_yaml_falls_back_safely(self) -> None:
        """Partial settings should merge with defaults; malformed YAML should not crash."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            path = settings.settings_path(workspace)
            path.parent.mkdir()
            path.write_text(
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
            self.assertFalse(settings.git_protection_enabled(loaded))
            self.assertTrue(settings.tool_enabled(loaded, "write_file"))
            self.assertTrue(settings.tool_always_allow(loaded, "write_file"))

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
