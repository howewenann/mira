"""Real MCP protocol, native SkillsMiddleware, and hostile manifest coverage."""

from __future__ import annotations

import base64
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from deepagents import create_deep_agent
from deepagents.middleware.skills import SkillsMiddleware
from fastmcp.skills.models import Skill
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent.mcp.manager import MCPManager
from agent.mcp.models import MCPServerState
from agent.mcp.skills import (
    MAX_FILE_BYTES, MCPSkillsBackend, _read_bytes, discover_skill, skill_identity,
)
from agent.resources import build_resources
from agent.skills import SkillRegistry, prepare_skill
from core.application.app import mcp_snapshot
from core.execution.runner import run_turn
from tests.test_mcp import write_config
from tests.test_runner import RunTurnRenderer
from ui.textual.widgets.mcp_panel import capability_counts, capability_summary, server_detail_sections, visible_fixed_resources


def write_skill(root: Path, name: str = "review", description: str = "Review files.") -> bytes:
    folder = root / name
    (folder / "references").mkdir(parents=True, exist_ok=True)
    body = f"---\nname: {name}\ndescription: {description}\n---\n\nRead references/guide.md before reviewing.\n"
    raw = body.encode()
    (folder / "SKILL.md").write_bytes(raw)
    (folder / "references" / "guide.md").write_bytes(b"Check the evidence.\n")
    return raw


class BindableModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class MCPNativeSkillsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.served = self.root / "served"
        self.raw = write_skill(self.served)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()

    async def manager(self, names=("docs",), mode="skills", approve=True):
        write_config(self.workspace, {
            name: {"command": sys.executable, "args": [
                str(Path("tests/fixtures/mcp_skills_server.py").resolve()), str(self.served), mode,
            ]} for name in names
        })
        manager = MCPManager(self.workspace)
        self.addAsyncCleanup(manager.shutdown)
        await manager.initialize(AsyncMock(return_value="allow") if approve else None)
        return manager

    async def test_stdio_native_discovery_reads_pin_and_ordinary_capabilities(self):
        manager = await self.manager()
        state = manager.servers["docs"]
        self.assertEqual(state.status, "Available", state.error)
        self.assertEqual(len(state.skills), 1)
        self.assertEqual(len(state.tools), 1)
        self.assertEqual(len(state.prompts), 1)
        self.assertTrue(state.resources)
        registry = dict(manager.resource_registry)
        manifest_uris = {resource.uri for skill in state.skills for resource in skill.manifest.resources}
        self.assertTrue(any(resource.uri in manifest_uris for resource in state.resources))
        visible = visible_fixed_resources(state)
        self.assertTrue(all(resource.uri not in manifest_uris for resource in visible))
        self.assertEqual(dict(capability_counts(state))["Resources"], str(len(visible)))
        self.assertEqual((await state.tools[0].ainvoke({"value": "ok"}))[0]["text"], "ok")
        self.assertIn("Hello Ada", str(await manager.prompt_registry.mcp["/mcp__docs__greeting"].resolver({"name": "Ada"})))
        resources = build_resources(self.workspace, create_examples=False, mcp_manager=manager)
        native = SkillsMiddleware(backend=resources.backend, sources=resources.skills)
        sync = native.before_agent({}, None, {})["skills_metadata"]
        asynchronous = (await native.abefore_agent({}, None, {}))["skills_metadata"]
        self.assertEqual(sync, asynchronous)
        self.assertEqual([(s["name"], s["path"]) for s in sync],
                         [(s["name"], s["path"]) for s in resources.metadata["skills"]])
        item = next(s for s in resources.metadata["skills"] if s["source"] == "mcp")
        self.assertEqual(item["server"], "docs")
        self.assertEqual(item["name"], skill_identity("docs", "review"))
        result = await resources.backend.aread(item["path"])
        self.assertFalse(result.error)
        self.assertIn("Read references/guide.md", result.file_data["content"])
        agent = create_deep_agent(
            model=BindableModel(responses=[
                AIMessage(content="", tool_calls=[{
                    "name": "read_file", "args": {"file_path": item["path"]}, "id": "read-skill",
                }]),
                AIMessage(content="done"),
            ]),
            backend=resources.backend, skills=resources.skills,
        )
        output = await agent.ainvoke({"messages": [{"role": "user", "content": "Read the skill"}]})
        tool_output = next(m for m in output["messages"] if isinstance(m, ToolMessage))
        self.assertIn("Read references/guide.md", str(tool_output.content))
        helper = item["path"].replace("SKILL.md", "references/guide.md")
        self.assertEqual((await resources.backend.adownload_files([helper]))[0].content, b"Check the evidence.\n")
        (self.served / "review/references/guide.md").write_bytes(b"Altered after discovery.\n")
        self.assertIn("SHA-256", (await resources.backend.aread(helper)).error)
        command = prepare_skill(f"/skill__{item['name']} review this", SkillRegistry(resources.metadata["skills"]))
        pinned = await native.abefore_model({"skills_metadata": sync, "pinned_skills": command.pinned_skills}, None)
        self.assertIn("Read references/guide.md", pinned["messages"][0].content)
        self.assertEqual((self.served / "review/SKILL.md").read_bytes(), self.raw)
        self.assertFalse((self.workspace / ".mira/skills").exists())
        self.assertTrue((await resources.backend.awrite(item["path"], "overwrite")).error)
        self.assertTrue((await resources.backend.aedit(item["path"], "Read", "Ignore")).error)
        self.assertIn("Skills  1", capability_summary(state).plain)
        self.assertIn(item["name"], dict(server_detail_sections(state))["Skills"])
        self.assertEqual(mcp_snapshot(manager)["capabilities"]["skills"], 1)
        self.assertEqual(manager.resource_registry, registry)

    async def test_unapproved_and_no_extension_servers(self):
        manager = await self.manager(mode="no-skills", approve=False)
        self.assertEqual(manager.servers["docs"].status, "Approval required")
        self.assertIsNone(manager.servers["docs"].skills)
        manager._approval_handler = AsyncMock(return_value="allow")
        self.assertTrue(await manager.restart_server("docs"))
        self.assertEqual(manager.servers["docs"].skills, [])
        self.assertEqual(len(manager.servers["docs"].tools), 1)

    async def test_duplicates_local_collision_disable_restart_reload_and_disconnect(self):
        manager = await self.manager(names=("one", "two"))
        write_skill(self.workspace / ".mira/skills")
        resources = build_resources(self.workspace, create_examples=False, mcp_manager=manager)
        remote = [s for s in resources.metadata["skills"] if s["source"] == "mcp"]
        self.assertEqual(len(remote), 2)
        self.assertEqual(len({s["name"] for s in remote}), 2)
        self.assertIn("review", {s["name"] for s in resources.metadata["skills"]})
        old = next(s for s in remote if s["server"] == "one")
        await manager.set_server_enabled("one", False)
        self.assertTrue((await resources.backend.aread(old["path"])).error)
        await manager.set_server_enabled("one", True)
        self.assertFalse((await resources.backend.aread(old["path"])).error)
        self.assertTrue(await manager.restart_server("one"))
        generation = manager.skills_generation
        await manager.reload()
        self.assertGreater(manager.skills_generation, generation)
        self.assertFalse((await resources.backend.aread(old["path"])).error)
        write_skill(self.workspace / ".mira/skills", old["name"], "Local wins.")
        with patch("core.diagnostics.logging.get_diagnostics_logger") as logger:
            collision = build_resources(self.workspace, create_examples=False, mcp_manager=manager)
        logger.return_value.warning.assert_called_once()
        item = next(s for s in collision.metadata["skills"] if s["name"] == old["name"])
        self.assertEqual(item["source"], "project")
        await manager._runtimes["two"].stop()
        other = next(s for s in remote if s["server"] == "two")
        self.assertTrue((await resources.backend.aread(other["path"])).error)

    async def test_runner_refreshes_existing_checkpoint_after_capability_change(self):
        manager = await self.manager()
        resources = build_resources(self.workspace, create_examples=False, mcp_manager=manager)
        agent = create_deep_agent(model=BindableModel(responses=[AIMessage(content="done")]),
                                  backend=resources.backend, skills=resources.skills, checkpointer=InMemorySaver())
        agent.mira_skills_refreshed_threads = set()
        agent.mira_mcp_manager = manager
        agent.mira_skills_generation = manager.skills_generation
        thread = "retained"
        config = {"configurable": {"thread_id": thread}}
        await run_turn(agent, "First", RunTurnRenderer(), thread)
        before = (await agent.aget_state(config)).values
        self.assertTrue(any(s["name"].startswith("mcp-") for s in before["skills_metadata"]))
        await manager.set_server_enabled("docs", False)
        await run_turn(agent, "Second", RunTurnRenderer(), thread)
        after = (await agent.aget_state(config)).values
        self.assertFalse(any(s["name"].startswith("mcp-") for s in after["skills_metadata"]))
        self.assertGreater(len(after["messages"]), len(before["messages"]))
        write_skill(self.served, description="Updated description.")
        write_skill(self.served, "new-guidance")
        await manager.set_server_enabled("docs", True)
        await run_turn(agent, "Third", RunTurnRenderer(), thread)
        updated = (await agent.aget_state(config)).values
        self.assertEqual(next(s for s in updated["skills_metadata"] if s["name"] == skill_identity("docs", "review"))["description"], "Updated description.")
        self.assertIn(skill_identity("docs", "new-guidance"), {s["name"] for s in updated["skills_metadata"]})


class MCPManifestTests(unittest.IsolatedAsyncioTestCase):
    def manifest(self, *, dynamic=False):
        self.raw = b"---\nname: review\ndescription: Review files.\n---\nBody\n"
        uri = "skill://library/review/SKILL.md"
        return Skill.model_validate({"uri": uri, "frontmatter": {"name": "review", "description": "Review files."},
            "resources": "dynamic" if dynamic else [{"uri": uri, "size": len(self.raw), "digest": "sha256:" + hashlib.sha256(self.raw).hexdigest()}]})

    def session(self, manifest):
        return SimpleNamespace(get_skill=AsyncMock(return_value=manifest),
            read_resource=AsyncMock(return_value=[SimpleNamespace(uri=manifest.uri, text=self.raw.decode())]),
            is_connected=lambda: True)

    async def test_bad_hash_size_content_and_frontmatter(self):
        manifest = self.manifest()
        for content in (self.raw + b"corrupt", self.raw.replace(b"Body", b"Evil")):
            session = self.session(manifest)
            session.read_resource.return_value[0].text = content.decode()
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                await discover_skill(session, "docs", manifest)
        for raw in (b"not frontmatter", self.raw.replace(b"review", b"other!")):
            dynamic = self.manifest(dynamic=True)
            session = self.session(dynamic)
            session.read_resource.return_value[0].text = raw.decode()
            with self.assertRaises(ValueError):
                await discover_skill(session, "docs", dynamic)
        manifest.resources[0].size = MAX_FILE_BYTES + 1
        session = self.session(manifest)
        with self.assertRaisesRegex(ValueError, "size limit"):
            await discover_skill(session, "docs", manifest)
        session.read_resource.assert_not_awaited()

    async def test_dynamic_reads_scoped_files_without_invented_hashes(self):
        manifest = self.manifest(dynamic=True)
        session = self.session(manifest)
        skill = await discover_skill(session, "docs", manifest)
        state = MCPServerState("docs", "stdio", {}, "test", status="Available", session=session, skills=[skill])
        backend = MCPSkillsBackend(SimpleNamespace(servers={"docs": state}), set())
        uri = "skill://library/review/references/live.md"
        session.read_resource.return_value = [SimpleNamespace(uri=uri, text="Current guidance")]
        prefix = f"/skills/{skill.name}/"
        self.assertEqual((await backend.aread(prefix + "references/live.md")).file_data["content"], "Current guidance")
        session.read_resource.assert_awaited_with(uri)
        for path in ("../other/SKILL.md", "%2e%2e/other", "/secret", "a//b", "a\\b", "a/%252e%252e", "a?query", "a/%0ab"):
            session.read_resource.reset_mock()
            self.assertTrue((await backend.aread(prefix + path)).error, path)
            session.read_resource.assert_not_awaited()
        self.assertTrue((await backend.aread(prefix + "SKILL.md", limit=0)).no_lines_requested)
        state.status = "Disabled"
        self.assertTrue((await backend.aread(prefix + "SKILL.md")).error)

    async def test_manifest_traversal_cross_skill_and_duplicate_paths(self):
        manifest = self.manifest()
        for uri in ("skill://library/other/notes.md", "skill://elsewhere/review/notes.md",
                    "skill://library/review/%2e%2e/notes.md", "skill://library/review/a%2fb", "skill://library/review/a?x=1"):
            broken = manifest.model_copy(deep=True)
            broken.resources[0].uri = uri
            with self.assertRaises(ValueError):
                await discover_skill(self.session(broken), "docs", broken)

    async def test_blob_bytes_and_read_failure_partial_discovery(self):
        manifest = self.manifest()
        session = self.session(manifest)
        session.read_resource.return_value = [SimpleNamespace(uri=manifest.uri, blob=base64.b64encode(self.raw).decode())]
        self.assertEqual(await _read_bytes(session, manifest.uri, manifest.resources[0]), self.raw)
        session.supports_skills = lambda: True
        session.list_skills = AsyncMock(return_value=[manifest, manifest])
        session.get_skill.side_effect = [ValueError("broken entry"), manifest]
        with tempfile.TemporaryDirectory() as directory:
            manager = MCPManager(Path(directory))
            state = MCPServerState("docs", "stdio", {}, "test", status="Available", session=session)
            await manager._discover_server_skills(state)
        self.assertEqual(len(state.skills), 1)
        self.assertEqual(state.status, "Partially available")
        self.assertIn("broken entry", state.skill_error)

    async def test_resource_errors_and_discovery_failures_remain_visible(self):
        manifest = self.manifest(dynamic=True)
        session = self.session(manifest)
        for contents in ([], [SimpleNamespace(uri="skill://elsewhere/SKILL.md", text="wrong")],
                         [SimpleNamespace(uri=manifest.uri, blob="invalid base64")]):
            session.read_resource.return_value = contents
            with self.assertRaises(ValueError):
                await _read_bytes(session, manifest.uri)
        session.supports_skills = lambda: True
        session.list_skills = AsyncMock(side_effect=RuntimeError("invalid skills/list response"))
        with tempfile.TemporaryDirectory() as directory:
            manager = MCPManager(Path(directory))
            state = MCPServerState("docs", "stdio", {}, "test", status="Available", session=session, tools=["working tool"])
            await manager._discover_server_skills(state)
        self.assertEqual(state.skills, [])
        self.assertEqual(state.tools, ["working tool"])
        self.assertEqual(state.status, "Partially available")
        self.assertIn("invalid skills/list response", state.skill_error)


if __name__ == "__main__":
    unittest.main()
