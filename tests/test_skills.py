"""Focused tests for DeepAgents-backed discovery and direct skill invocation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from deepagents.backends import FilesystemBackend
from langchain_core.messages import HumanMessage

from agent.resources import build_resources
from agent.resources.skills import load_skills
from agent.skills import SkillRegistry, prepare_skill
from tests.test_textual_app import make_app, wait_until
from ui.textual.widgets.prompt_box import PromptBox


def write_skill(root: Path, directory: str, frontmatter: str, body: str = "# Instructions") -> Path:
    skill_dir = root / ".mira" / "skills" / directory
    skill_dir.mkdir(parents=True, exist_ok=True)
    path = skill_dir / "SKILL.md"
    path.write_text(f"---\n{frontmatter}\n---\n\n{body}\n", encoding="utf-8")
    return path


class SkillDiscoveryTests(unittest.TestCase):
    def test_deepagents_list_skills_is_the_canonical_source(self) -> None:
        default = {"name": "default", "description": "Default.", "path": "/default/SKILL.md"}
        project = {"name": "project", "description": "Project.", "path": "/project/SKILL.md"}
        backend = object()
        with patch("agent.resources.skills._list_skills", side_effect=[[default], [project]]) as discover:
            sources, metadata = load_skills(backend)

        self.assertEqual(sources, ["/mira-defaults/skills", "/.mira/skills"])
        self.assertEqual(
            [call.args for call in discover.call_args_list],
            [(backend, "/mira-defaults/skills"), (backend, "/.mira/skills")],
        )
        self.assertEqual([item["name"] for item in metadata], ["default", "project"])

    def test_malformed_and_missing_description_skills_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            write_skill(workspace, "bad-yaml", "name: [bad\ndescription: Broken")
            write_skill(workspace, "missing-description", "name: missing-description")

            resources = build_resources(workspace, create_examples=False)

        names = {item["name"] for item in resources.metadata["skills"]}
        self.assertEqual(names, {"skill-creator"})
        self.assertEqual(resources.skills, ["/mira-defaults/skills"])

    def test_project_skill_overrides_default_and_propagates_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            write_skill(
                workspace,
                "skill-creator",
                "name: skill-creator\ndescription: Project-specific creator.",
            )

            resources = build_resources(workspace, create_examples=False)

        skill = next(item for item in resources.metadata["skills"] if item["name"] == "skill-creator")
        self.assertEqual(
            skill,
            {
                "name": "skill-creator",
                "description": "Project-specific creator.",
                "path": "/.mira/skills/skill-creator/SKILL.md",
                "source": "project",
                "replaces": "default",
            },
        )

    def test_packaged_skill_creator_is_mira_specific(self) -> None:
        path = Path("agent/resources/defaults/skills/skill-creator/SKILL.md")
        text = path.read_text(encoding="utf-8")

        self.assertIn(".mira/skills/<skill-name>/SKILL.md", text)
        self.assertIn("ask_user", text)
        self.assertIn("validate_skill", text)
        self.assertIn("YAML frontmatter alone is incomplete", text)
        self.assertIn("# Skill Name\n\n## Overview\n", text)
        self.assertIn("## When to Use\n", text)
        self.assertIn("## Instructions\n", text)
        self.assertIn("## Completion Criteria\n\nBefore finishing, verify that:", text)
        self.assertIn(
            "`Instructions` describe how to do the work. `Completion Criteria` describe what must be true before finishing.",
            text,
        )
        self.assertIn("short, observable, checkable, and specific to the skill", text)
        self.assertIn("Prefer checkable outcomes over vague quality statements", text)
        self.assertIn("Use conditional wording", text)
        self.assertIn("Do not duplicate every instruction as a criterion", text)
        self.assertIn("do not introduce requirements unsupported by the skill's purpose", text)
        self.assertIn("not MIRA Goal Success Criteria, RubricMiddleware", text)
        self.assertIn("or a heading enforced by `validate_skill`", text)
        self.assertIn("recommended creation template, not validator requirements", text)
        self.assertIn("read it first and improve it in place", text)
        self.assertIn("Finish only after validation returns `valid: true`", text)
        for forbidden in (
            "$DEEPAGENTS_HOME",
            ".deepagents/skills",
            ".agents/skills",
            "deepagents skills create",
            "init_skill.py",
        ):
            self.assertNotIn(forbidden, text)


class SkillInvocationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.skill = {
            "name": "foo",
            "description": "Review a thing.",
            "path": "/skills/foo/SKILL.md",
            "source": "project",
            "replaces": "",
        }

    def test_registry_resolves_bare_command_and_preserves_request_remainder(self) -> None:
        registry = SkillRegistry([self.skill])
        self.assertEqual(registry.resolve("/skill__foo").args, "")  # type: ignore[union-attr]
        self.assertEqual(
            registry.resolve("/skill__foo\t  review this  ").args,  # type: ignore[union-attr]
            "  review this  ",
        )

    def test_registry_reports_empty_unknown_and_duplicate_commands(self) -> None:
        registry = SkillRegistry([self.skill])
        with self.assertRaisesRegex(ValueError, "usage: /skill__<name>"):
            registry.resolve("/skill__")
        with self.assertRaisesRegex(ValueError, "unknown skill command: /skill__missing"):
            registry.resolve("/skill__missing request")
        with self.assertRaisesRegex(ValueError, "duplicate skill command"):
            SkillRegistry([self.skill, dict(self.skill)])

    def test_prepared_skill_contains_full_human_envelope_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            skill_file = root / "skills" / "foo" / "SKILL.md"
            skill_file.parent.mkdir(parents=True)
            raw = "---\nname: foo\ndescription: Review a thing.\n---\n\n# Full instructions\n"
            skill_file.write_text(raw, encoding="utf-8")
            stored = skill_file.read_bytes().decode("utf-8")
            backend = FilesystemBackend(root_dir=root, virtual_mode=True)

            prepared = prepare_skill(
                "/skill__foo inspect this PR carefully",
                SkillRegistry([self.skill]),
                backend,
            )
            bare = prepare_skill("/skill__foo", SkillRegistry([self.skill]), backend)

        self.assertIsNotNone(prepared)
        assert prepared is not None
        self.assertEqual(prepared.display_text, "/skill__foo inspect this PR carefully")
        self.assertEqual(len(prepared.messages), 1)
        self.assertIsInstance(prepared.messages[0], HumanMessage)
        self.assertIn(stored, str(prepared.messages[0].content))
        self.assertIn("**User request:** inspect this PR carefully", str(prepared.messages[0].content))
        self.assertEqual(
            prepared.messages[0].additional_kwargs["__skill"],
            {
                "name": "foo",
                "description": "Review a thing.",
                "source": "project",
                "args": "inspect this PR carefully",
            },
        )
        assert bare is not None
        self.assertNotIn("**User request:**", str(bare.messages[0].content))


class SkillInvocationFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_textual_submission_uses_existing_prepared_turn_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "skills" / "foo" / "SKILL.md"
            path.parent.mkdir(parents=True)
            path.write_text("---\nname: foo\ndescription: Review.\n---\n\nReview carefully.\n", encoding="utf-8")
            backend = FilesystemBackend(root_dir=root, virtual_mode=True)
            agent = SimpleNamespace(mira_backend=backend)
            metadata = {
                "skills": [{"name": "foo", "description": "Review.", "path": "/skills/foo/SKILL.md", "source": "project", "replaces": ""}]
            }
            app = make_app(agent=agent, plan_agent=object(), resource_metadata=metadata)
            run_turn = AsyncMock()
            app._run_turn = run_turn

            async with app.run_test() as pilot:
                prompt = app.query_one(PromptBox)
                await app.submit_prompt(PromptBox.Submitted(prompt, "/skill__foo review this"))
                await wait_until(lambda: run_turn.await_count == 1)
                await pilot.pause()

        self.assertEqual(run_turn.await_args.args, ("/skill__foo review this",))
        prepared = run_turn.await_args.kwargs["prepared_messages"]
        self.assertEqual(
            prepared[0].additional_kwargs["__skill"],
            {"name": "foo", "description": "Review.", "source": "project", "args": "review this"},
        )
        self.assertEqual(run_turn.await_args.kwargs["display_text"], "/skill__foo review this")
