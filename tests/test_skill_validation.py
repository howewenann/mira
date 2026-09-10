"""Tests for MIRA's protected read-only skill authoring validator."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path, PurePosixPath

from deepagents.backends import FilesystemBackend
from deepagents.middleware.skills import MAX_SKILL_FILE_SIZE

from agent import factory
from agent.resources import build_resources
from agent.tools.skill_validation import build_validate_skill_tool
from config.settings import READ_ONLY_BUILTIN_TOOLS, load_settings


class SkillValidationTests(unittest.TestCase):
    def tool_for(self, root: Path):
        return build_validate_skill_tool(FilesystemBackend(root_dir=root, virtual_mode=True))

    def write_path(self, root: Path, path: str, content: str) -> None:
        parts = PurePosixPath(path).parts
        target = root.joinpath(*(parts[1:] if parts[:1] == ("/",) else parts))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content.encode("utf-8"))

    def write(self, root: Path, name: str, content: str) -> None:
        self.write_path(root, f"/.mira/skills/{name}/SKILL.md", content)

    @staticmethod
    def valid_content(name: str = "foo") -> str:
        return (
            f"---\nname: {name}\n"
            "description: Review code when the user requests a focused review.\n"
            "---\n\nReview the code and report concrete findings.\n"
        )

    @staticmethod
    def categories(result: dict[str, object]) -> list[str]:
        return [str(error).split("]", 1)[0] + "]" for error in result["errors"]]  # type: ignore[index]

    def test_valid_directory_and_skill_file_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root, "foo", self.valid_content())
            tool = self.tool_for(root)

            self.assertEqual(
                tool.invoke({"path": ".mira/skills/foo"}),
                {"valid": True, "errors": []},
            )
            self.assertEqual(
                tool.invoke({"path": ".mira/skills/foo/SKILL.md"}),
                {"valid": True, "errors": []},
            )

    def test_frontmatter_only_skill_requires_markdown_body(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(
                root,
                "foo",
                "---\nname: foo\ndescription: Review code when requested.\n---\n",
            )
            result = self.tool_for(root).invoke({"path": ".mira/skills/foo"})

        self.assertEqual(
            result,
            {
                "valid": False,
                "errors": ["[BODY] Add Markdown instructions after the YAML frontmatter."],
            },
        )

    def test_wrong_mira_location_is_read_and_content_is_still_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_path(root, "/mira/skills/foo/SKILL.md", self.valid_content())
            result = self.tool_for(root).invoke({"path": "/mira/skills/foo"})

        self.assertEqual(self.categories(result), ["[LOCATION]"])
        self.assertIn("/mira/skills/foo/SKILL.md", result["errors"][0])
        self.assertIn("/.mira/skills/foo/SKILL.md", result["errors"][0])

    def test_multiple_simultaneous_failures_have_stable_category_order(self) -> None:
        content = (
            "---\n"
            "name: Foo\n"
            "description: ''\n"
            "compatibility: []\n"
            "unexpected: true\n"
            "---\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_path(root, "/mira/skills/foo/SKILL.md", content)
            result = self.tool_for(root).invoke({"path": "/mira/skills/foo/SKILL.md"})

        self.assertEqual(
            self.categories(result),
            [
                "[LOCATION]",
                "[FRONTMATTER]",
                "[NAME]",
                "[NAME]",
                "[DESCRIPTION]",
                "[COMPATIBILITY]",
                "[BODY]",
            ],
        )

    def test_nested_skill_location_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_path(
                root,
                "/.mira/skills/group/foo/SKILL.md",
                self.valid_content(),
            )
            result = self.tool_for(root).invoke(
                {"path": "/.mira/skills/group/foo/SKILL.md"}
            )

        self.assertEqual(self.categories(result), ["[LOCATION]"])

    def test_parent_traversal_is_not_a_valid_skill_location(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root, "bar", self.valid_content("bar"))
            result = self.tool_for(root).invoke(
                {"path": "/.mira/skills/foo/../bar/SKILL.md"}
            )

        self.assertEqual(self.categories(result), ["[FILE]", "[LOCATION]"])
        self.assertIn("foo/../bar", result["errors"][1])

    def test_invalid_yaml_still_reports_location_size_and_body(self) -> None:
        content = "---\nname: [broken\n---\n" + " " * MAX_SKILL_FILE_SIZE
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_path(root, "/mira/skills/foo/SKILL.md", content)
            result = self.tool_for(root).invoke({"path": "/mira/skills/foo"})

        self.assertEqual(
            self.categories(result),
            ["[LOCATION]", "[SIZE]", "[FRONTMATTER]", "[BODY]"],
        )

    def test_over_limit_uses_deepagents_character_limit(self) -> None:
        prefix = self.valid_content()
        content = prefix + "x" * (MAX_SKILL_FILE_SIZE - len(prefix) + 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root, "foo", content)
            result = self.tool_for(root).invoke({"path": ".mira/skills/foo"})

        self.assertEqual(self.categories(result), ["[SIZE]"])
        self.assertIn(str(MAX_SKILL_FILE_SIZE + 1), result["errors"][0])

    def test_missing_file_and_invalid_utf8_are_file_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tool = self.tool_for(root)
            missing = tool.invoke({"path": ".mira/skills/missing"})

            invalid = root / ".mira" / "skills" / "invalid" / "SKILL.md"
            invalid.parent.mkdir(parents=True)
            invalid.write_bytes(b"\xff\xfe")
            undecodable = tool.invoke({"path": ".mira/skills/invalid"})

        self.assertEqual(self.categories(missing), ["[FILE]"])
        self.assertIn("/.mira/skills/missing/SKILL.md", missing["errors"][0])
        self.assertEqual(self.categories(undecodable), ["[FILE]"])
        self.assertIn("UTF-8", undecodable["errors"][0])

    def test_frontmatter_must_be_present_valid_yaml_and_a_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root, "plain", "# No frontmatter\n")
            self.write(root, "broken", "---\nname: [broken\n---\n\nInstructions.\n")
            self.write(root, "list", "---\n- name\n- description\n---\n\nInstructions.\n")
            tool = self.tool_for(root)
            plain = tool.invoke({"path": ".mira/skills/plain"})
            broken = tool.invoke({"path": ".mira/skills/broken"})
            sequence = tool.invoke({"path": ".mira/skills/list"})

        self.assertIn("valid YAML frontmatter", plain["errors"][0])
        self.assertIn("Fix invalid YAML", broken["errors"][0])
        self.assertIn("YAML frontmatter to a mapping", sequence["errors"][0])

    def test_required_fields_name_rules_mismatch_and_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(
                root,
                "parent",
                "---\nname: Bad--Name\ndescription: '<invalid>'\ncompatibility: '"
                + "x" * 501
                + "'\nunexpected: yes\n---\n\nInstructions.\n",
            )
            result = self.tool_for(root).invoke({"path": ".mira/skills/parent"})
            self.write(root, "missing", "---\nlicense: MIT\n---\n\nInstructions.\n")
            required = self.tool_for(root).invoke({"path": ".mira/skills/missing"})
            long_name = "a" * 65
            self.write(
                root,
                long_name,
                f"---\nname: {long_name}\ndescription: "
                + "d" * 1025
                + "\n---\n\nInstructions.\n",
            )
            long_result = self.tool_for(root).invoke({"path": f".mira/skills/{long_name}"})

        joined = "\n".join(result["errors"])
        self.assertIn("lowercase letters", joined)
        self.assertIn("matches the parent directory", joined)
        self.assertIn("angle brackets", joined)
        self.assertIn("Shorten 'compatibility'", joined)
        self.assertIn("Remove unsupported fields", joined)
        self.assertIn("required 'name'", "\n".join(required["errors"]))
        self.assertIn("required 'description'", "\n".join(required["errors"]))
        self.assertIn("Shorten 'name'", "\n".join(long_result["errors"]))
        self.assertIn("Shorten 'description'", "\n".join(long_result["errors"]))

    def test_validator_is_protected_read_only_and_has_all_safe_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            custom = root / ".mira" / "tools"
            custom.mkdir(parents=True)
            (custom / "replacement.py").write_text(
                "from langchain_core.tools import tool\n\n@tool\ndef validate_skill(path: str) -> str:\n"
                '    """Replace validation."""\n    return \'mutated\'\n',
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
