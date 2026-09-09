"""Tests for MIRA's protected read-only skill authoring validator."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from deepagents.backends import FilesystemBackend

from agent import factory
from agent.resources import build_resources
from agent.tools.skill_validation import build_validate_skill_tool
from config.settings import READ_ONLY_BUILTIN_TOOLS, load_settings


class SkillValidationTests(unittest.TestCase):
    def tool_for(self, root: Path):
        return build_validate_skill_tool(FilesystemBackend(root_dir=root, virtual_mode=True))

    def write(self, root: Path, name: str, content: str) -> None:
        path = root / ".mira" / "skills" / name / "SKILL.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def test_valid_directory_and_skill_file_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root, "review", "---\nname: review\ndescription: Review code.\n---\n")
            tool = self.tool_for(root)
            self.assertEqual(tool.invoke({"path": ".mira/skills/review"}), {"valid": True, "errors": []})
            self.assertEqual(tool.invoke({"path": "/.mira/skills/review/SKILL.md"}), {"valid": True, "errors": []})

    def test_missing_file_and_invalid_yaml_are_structured_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tool = self.tool_for(root)
            missing = tool.invoke({"path": ".mira/skills/missing"})
            self.write(root, "broken", "---\nname: [broken\ndescription: nope\n---\n")
            broken = tool.invoke({"path": ".mira/skills/broken"})

        self.assertFalse(missing["valid"])
        self.assertIn("SKILL.md not found", missing["errors"][0])
        self.assertFalse(broken["valid"])
        self.assertIn("Invalid YAML", broken["errors"][0])

    def test_frontmatter_must_be_present_and_a_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root, "plain", "# No frontmatter\n")
            self.write(root, "list", "---\n- name\n- description\n---\n")
            tool = self.tool_for(root)
            plain = tool.invoke({"path": ".mira/skills/plain"})
            sequence = tool.invoke({"path": ".mira/skills/list"})

        self.assertIn("No valid YAML frontmatter", plain["errors"][0])
        self.assertIn("must be a YAML mapping", sequence["errors"][0])

    def test_required_fields_name_rules_mismatch_and_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(
                root,
                "parent",
                "---\nname: Bad--Name\ndescription: '<invalid>'\ncompatibility: '"
                + "x" * 501
                + "'\nunexpected: yes\n---\n",
            )
            result = self.tool_for(root).invoke({"path": ".mira/skills/parent"})
            self.write(root, "missing", "---\nlicense: MIT\n---\n")
            required = self.tool_for(root).invoke({"path": ".mira/skills/missing"})
            long_name = "a" * 65
            self.write(root, long_name, f"---\nname: {long_name}\ndescription: " + "d" * 1025 + "\n---\n")
            long_result = self.tool_for(root).invoke({"path": f".mira/skills/{long_name}"})

        joined = "\n".join(result["errors"])
        self.assertIn("lowercase letters", joined)
        self.assertIn("must match parent directory", joined)
        self.assertIn("angle brackets", joined)
        self.assertIn("Compatibility is too long", joined)
        self.assertIn("Unexpected frontmatter fields", joined)
        self.assertIn("Missing required 'name'", "\n".join(required["errors"]))
        self.assertIn("Missing required 'description'", "\n".join(required["errors"]))
        self.assertIn("Name is too long", "\n".join(long_result["errors"]))
        self.assertIn("Description is too long", "\n".join(long_result["errors"]))

    def test_validator_is_protected_read_only_and_has_all_safe_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            custom = root / ".mira" / "tools"
            custom.mkdir(parents=True)
            (custom / "replacement.py").write_text(
                "from langchain_core.tools import tool\n\n@tool\ndef validate_skill(path: str) -> str:\n"
                "    \"\"\"Replace validation.\"\"\"\n    return 'mutated'\n",
                encoding="utf-8",
            )
            resources = build_resources(root, create_examples=False)
            validator = next(tool for tool in resources.tools if tool.name == "validate_skill")
            result = validator.invoke({"path": ".mira/skills/missing"})
            settings = load_settings(root)
            rubric_tools, rubric_interrupts = factory.effective_rubric_tools(
                {"settings": settings},
                resources.backend,
                resources.tools,
                resources.metadata["tools"],
            )

        self.assertIsInstance(result, dict)
        self.assertEqual([tool.name for tool in resources.tools].count("validate_skill"), 1)
        self.assertIn("validate_skill", READ_ONLY_BUILTIN_TOOLS)
        self.assertEqual(
            settings["hitl"]["tools"]["validate_skill"],
            {"enabled": True, "plan_access": True, "ptc": True, "rubric": True},
        )
        self.assertNotIn("validate_skill", factory._write_interrupts({"settings": settings}))
        self.assertIn("validate_skill", [tool.name for tool in rubric_tools])
        self.assertNotIn("validate_skill", rubric_interrupts)
