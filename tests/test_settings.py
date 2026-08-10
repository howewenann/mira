"""Tests for workspace settings persistence."""

from __future__ import annotations

from copy import deepcopy
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
        self.assertTrue(settings.dynamic_subagent_response_schema_enabled(loaded))
        self.assertFalse(settings.planning_todos_enabled(loaded))
        self.assertEqual(settings.planning_response_status_max_retries(loaded), 2)
        self.assertFalse(settings.rubric_enabled(loaded))
        self.assertEqual(settings.rubric_max_iterations(loaded), 3)
        self.assertFalse(settings.tool_always_allow(loaded, "custom_search"))
        self.assertTrue(settings.tool_enabled(loaded, "custom_search"))
        self.assertTrue(settings.tool_enabled(loaded, "delete"))
        self.assertFalse(settings.tool_always_allow(loaded, "delete"))
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

    def test_planning_todos_default_off_and_can_toggle(self) -> None:
        """Planning todos should be an explicit, reversible opt-in."""
        loaded = settings.normalize_settings({})

        self.assertFalse(settings.planning_todos_enabled(loaded))
        updated = settings.set_planning_todos(loaded, True)
        self.assertTrue(settings.planning_todos_enabled(updated))
        updated = settings.set_planning_todos(updated, False)
        self.assertFalse(settings.planning_todos_enabled(updated))

    def test_recursive_delete_uses_configurable_approval_policy(self) -> None:
        """Delete should keep the conservative default while respecting opt-out."""
        default = settings.normalize_settings({})
        self.assertFalse(settings.tool_always_allow(default, "delete"))

        loaded = settings.normalize_settings(
            {"hitl": {"tools": {"delete": {"enabled": True, "always_allow": True}}}}
        )

        self.assertTrue(settings.tool_enabled(loaded, "delete"))
        self.assertTrue(settings.tool_always_allow(loaded, "delete"))
        updated = settings.set_tool_always_allow(loaded, "delete", False)
        self.assertFalse(settings.tool_always_allow(updated, "delete"))

    def test_dynamic_response_schema_defaults_on_and_can_toggle(self) -> None:
        """Dynamic response schemas should stay enabled for compatibility."""
        loaded = settings.normalize_settings({})

        self.assertTrue(settings.dynamic_subagent_response_schema_enabled(loaded))

        updated = settings.set_dynamic_subagent_response_schema(loaded, False)
        self.assertFalse(settings.dynamic_subagent_response_schema_enabled(updated))
        self.assertFalse(settings.dynamic_subagents_enabled(updated))

        updated = settings.set_dynamic_subagent_response_schema(updated, True)
        self.assertTrue(settings.dynamic_subagent_response_schema_enabled(updated))

    def test_rubric_settings_normalize_and_preserve_invalid_iteration_values(self) -> None:
        """Rubric settings should default safely and reject unsupported caps."""
        loaded = settings.normalize_settings(
            {"system": {"rubric": {"enabled": True, "max_iterations": 5}}}
        )
        self.assertTrue(settings.rubric_enabled(loaded))
        self.assertEqual(settings.rubric_max_iterations(loaded), 5)

        for value in (0, -1, 21, True, "4"):
            updated = settings.set_rubric_max_iterations(loaded, value)
            self.assertEqual(settings.rubric_max_iterations(updated), 5)

        malformed = settings.normalize_settings(
            {"system": {"rubric": {"enabled": "yes", "max_iterations": 0}}}
        )
        self.assertFalse(settings.rubric_enabled(malformed))
        self.assertEqual(settings.rubric_max_iterations(malformed), 3)

    def test_planning_response_status_retries_normalize_and_preserve_invalid_values(self) -> None:
        """The shared Plan/Goal retry cap should follow rubric-style bounds."""
        loaded = settings.normalize_settings(
            {"system": {"planning_response_status": {"max_retries": 5}}}
        )
        self.assertEqual(settings.planning_response_status_max_retries(loaded), 5)

        for value in (1, 20):
            updated = settings.set_planning_response_status_max_retries(loaded, value)
            self.assertEqual(settings.planning_response_status_max_retries(updated), value)

        for value in (0, -1, 21, True, "4"):
            updated = settings.set_planning_response_status_max_retries(loaded, value)
            self.assertEqual(settings.planning_response_status_max_retries(updated), 5)

        malformed = settings.normalize_settings(
            {"system": {"planning_response_status": {"max_retries": 0}}}
        )
        self.assertEqual(settings.planning_response_status_max_retries(malformed), 2)

    def test_retired_planning_next_action_key_becomes_startup_issue(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            retired = deepcopy(settings.DEFAULT_SETTINGS)
            retired["system"]["planning_next_action"] = retired["system"].pop(
                "planning_response_status"
            )
            self.assertTrue(settings.save_settings(workspace, retired))
            path = settings.settings_path(workspace)
            text = path.read_text(encoding="utf-8").replace(
                "planning_response_status:", "planning_next_action:"
            )
            path.write_text(text, encoding="utf-8")

            result = settings.load_settings_result(workspace)
            self.assertFalse(result.valid)
            self.assertIn("planning_next_action", result.issues[0].summary)

    def test_partial_settings_use_defaults_and_invalid_shapes_become_issues(self) -> None:
        """Omitted fields use defaults while malformed or unknown fields remain visible."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            path = settings.settings_path(workspace)
            path.parent.mkdir()
            path.write_text(
                "system:\n"
                "  dynamic_subagents:\n"
                "    enabled: true\n"
                "    response_schema: false\n"
                "hitl:\n"
                "  git_protection:\n"
                "    enabled: false\n"
                "  tools:\n"
                "    write_file:\n"
                "      enabled: false\n"
                "      always_allow: true\n",
                encoding="utf-8",
            )

            partial = settings.load_settings_result(workspace)
            self.assertTrue(partial.valid)
            self.assertEqual(partial.issues, ())
            self.assertEqual(partial.settings["models"]["context_limit_tokens"], 32768)

            path.write_text("hitl: [", encoding="utf-8")
            malformed = settings.load_settings_result(workspace)
            self.assertFalse(malformed.valid)
            self.assertIn("Invalid settings.yml", malformed.issues[0].summary)

            self.assertTrue(settings.save_settings(workspace, settings.DEFAULT_SETTINGS))
            exact = path.read_text(encoding="utf-8")
            path.write_text(f"{exact}unknown: true\n", encoding="utf-8")
            unknown = settings.load_settings_result(workspace)
            self.assertFalse(unknown.valid)
            self.assertIn("Unsupported setting: unknown", unknown.issues[0].summary)

            path.write_text(exact.replace("max_retries: 2", "max_retries: 0"), encoding="utf-8")
            invalid = settings.load_settings_result(workspace)
            self.assertFalse(invalid.valid)
            self.assertIn("planning_response_status.max_retries", invalid.issues[0].summary)

    def test_save_settings_writes_expected_yaml(self) -> None:
        """Saving toggles should persist the normalized schema."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            updated = settings.set_git_protection(settings.load_settings(workspace), False)
            updated = settings.set_planning_todos(updated, True)
            updated = settings.set_planning_response_status_max_retries(updated, 4)
            updated = settings.set_tool_always_allow(updated, "delete", True)
            updated = settings.set_tool_always_allow(updated, "web_search", False)
            updated = settings.set_tool_enabled(updated, "web_search", False)

            self.assertTrue(settings.save_settings(workspace, updated))
            text = settings.settings_path(workspace).read_text(encoding="utf-8")
            loaded = settings.load_settings(workspace)

        self.assertIn("settings.yml", str(settings.settings_path(workspace)))
        self.assertIn("git_protection", text)
        self.assertIn("response_schema: true", text)
        self.assertIn("web_search", text)
        self.assertIn("enabled: false", text)
        self.assertNotIn("llm_direct", text)
        self.assertNotIn("llm_direct", loaded)
        self.assertFalse(settings.git_protection_enabled(loaded))
        self.assertTrue(settings.planning_todos_enabled(loaded))
        self.assertEqual(settings.planning_response_status_max_retries(loaded), 4)
        self.assertTrue(settings.tool_always_allow(loaded, "delete"))
        self.assertFalse(settings.tool_always_allow(loaded, "web_search"))
        self.assertFalse(settings.tool_enabled(loaded, "web_search"))


if __name__ == "__main__":
    unittest.main()
