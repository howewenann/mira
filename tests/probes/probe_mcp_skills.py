r"""MIRA MCP Skills -> DeepAgents SkillsMiddleware compatibility probe.

No production imports or edits, no LLM calls, and no external server needed.
Requires the project's installed dependencies, including FastMCP >= 4.1.2.

From the repository root with the MIRA environment activated:
    python tests/probes/probe_mcp_skills.py

Contributor verification with the shared Conda environment:
    conda run -n mira python tests/probes/probe_mcp_skills.py

Do not use Python's -O flag: verification relies on assertions. Expected result:
five PASS sections and ALL FIVE TESTS PASSED, with exit code 0. The probe prints
dependency versions; the original verified set was DeepAgents 0.7.32, FastMCP
4.1.2, LangChain 1.4.7, and LangGraph 1.2.18.

This very file is also launched as a REAL local stdio MCP server in a child
process using sys.executable, so it works across platforms and uses the same
environment as its parent. Sample skills and the workspace are temporary and
cleaned up on exit. No credentials or user configuration are read.

This is the original async backend compatibility proof, limited to a static
manifest. It does not test MIRA's prefetch/synchronous assembly boundary,
lifecycle, identities, UI, checkpoint refresh, or dynamic resources. Integrated
coverage lives in tests/test_mcp_skills.py and tests/test_sqlite_sessions.py.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib.metadata
import sys
import tempfile
import traceback
from pathlib import Path, PurePosixPath
from typing import Any

SKILL_NAME = "review-documents"
SKILL_DESCRIPTION = "Review documents using their supporting references."
SKILL_BODY = """---
name: review-documents
description: Review documents using their supporting references.
---
# Document Review

Read the document and check its claims against the references.
See references/guide.md for more guidance.
"""
HELPER_BODY = "Cite the evidence for each correction.\n"
ROOT = "/skills"
MOUNT = "/mcp-probe/"


def serve(skills_root: Path) -> None:
    """Child process: serve real MCP protocol over stdin/stdout."""
    from fastmcp import FastMCP
    from fastmcp.server.providers.skills import SkillsDirectoryProvider
    from fastmcp.skills.extension import SkillsExtension

    server = FastMCP("MIRA MCP Skills Probe")
    provider = SkillsDirectoryProvider(
        roots=skills_root,
        experimental_skills=True,
        reload=True,
    )
    server.add_provider(provider)
    server.add_extension(SkillsExtension(providers=[provider]))

    @server.tool
    def probe_echo(value: str) -> str:
        """Return a value for ordinary MCP tool-discovery verification."""
        return value

    server.run(show_banner=False)  # stdio; keep stdout reserved for the protocol


def write_sample_skills(directory: Path) -> None:
    skill_dir = directory / SKILL_NAME
    (skill_dir / "references").mkdir(parents=True)
    # Use exact LF bytes on Windows too. Default text-mode writes translate
    # "\n" to "\r\n" there, making a raw-byte assertion fail even though
    # MCP manifests and backend reads are working correctly.
    (skill_dir / "SKILL.md").write_text(SKILL_BODY, encoding="utf-8", newline="\n")
    (skill_dir / "references" / "guide.md").write_text(
        HELPER_BODY, encoding="utf-8", newline="\n"
    )


def extract_bytes(contents: list[Any]) -> bytes:
    """Decode one text/blob MCP resource; keep source bytes for SHA-256."""
    if len(contents) != 1:
        raise ValueError(f"Expected one MCP resource content block, got {len(contents)}")
    resource = contents[0]
    text = getattr(resource, "text", None)
    if isinstance(text, str):
        return text.encode("utf-8")
    blob = getattr(resource, "blob", None)
    if isinstance(blob, str):
        return base64.b64decode(blob, validate=True)
    raise TypeError(f"Unexpected MCP resource type: {type(resource).__name__}")


async def run_probe() -> None:
    from deepagents.backends import CompositeBackend, FilesystemBackend
    from deepagents.backends.protocol import BackendProtocol, FileDownloadResponse, LsResult, ReadResult
    from deepagents.middleware.skills import SkillsMiddleware
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport
    from fastmcp.skills import SkillsClientExtension

    print("VERSIONS")
    for package in ("deepagents", "fastmcp", "langchain", "langgraph"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "not installed"
        print(f"  {package}: {version}")

    with tempfile.TemporaryDirectory(prefix="mira-mcp-skills-probe-") as temp:
        temp_root = Path(temp)
        skills_root = temp_root / "served-skills"
        workspace = temp_root / "workspace"
        skills_root.mkdir()
        workspace.mkdir()
        write_sample_skills(skills_root)

        # Uses the SAME stdio transport style as MIRA's MCP integration.
        transport = StdioTransport(
            command=sys.executable,
            args=[str(Path(__file__).resolve()), "--serve", str(skills_root)],
            keep_alive=False,
        )
        async with Client(transport, extensions=[SkillsClientExtension()]) as client:
            print("\nTEST 1: MCP connection and tool discovery")
            tools = await client.list_tools()
            assert any(getattr(t, "name", "") == "probe_echo" for t in tools), tools
            print("  PASS: stdio server connected and its ordinary MCP tool is available")

            print("\nTEST 2: native MCP Skills discovery")
            summaries = await client.list_skills()
            assert len(summaries) == 1, f"Expected one skill; got: {summaries!r}"
            summary = summaries[0]
            assert summary.frontmatter.name == SKILL_NAME
            assert summary.frontmatter.description == SKILL_DESCRIPTION
            skill = await client.get_skill(summary.uri)
            assert skill.uri == summary.uri
            assert isinstance(skill.resources, list), (
                "Probe needs an MCP skill with a static resource manifest"
            )
            print(f"  PASS: skills/list and skills/get => {SKILL_NAME!r}")

            class MCPReadOnlySkillsBackend(BackendProtocol):
                """PROBE ONLY: translate MCP skill resources to DeepAgents async file ops.

                This is a minimal proof of the backend boundary, not production
                implementation. It is deliberately read-only and async-only.
                """

                def __init__(self, connected_client: Any, entries: list[Any]):
                    self.client = connected_client
                    self.filenames: dict[str, Any] = {}
                    self.skill_names: set[str] = set()
                    for entry in entries:
                        if not isinstance(entry.resources, list):
                            raise ValueError(f"Dynamic manifest unsupported in probe: {entry.uri}")
                        name = entry.frontmatter.name
                        self.skill_names.add(name)
                        base_uri = entry.uri.rsplit("/", 1)[0] + "/"
                        for resource in entry.resources:
                            if not resource.uri.startswith(base_uri):
                                raise ValueError("Manifest contains a resource outside its skill URI")
                            relative = resource.uri[len(base_uri):]
                            parts = PurePosixPath(relative).parts
                            if not parts or any(part in {"..", "."} for part in parts):
                                raise ValueError(f"Unsafe resource path: {relative}")
                            target = f"{ROOT}/{name}/" + "/".join(parts)
                            self.filenames[target] = resource

                async def als(self, path: str) -> LsResult:
                    normalized = path.rstrip("/") or "/"
                    if normalized == ROOT:
                        return LsResult(entries=[
                            {"path": f"{ROOT}/{name}", "is_dir": True}
                            for name in sorted(self.skill_names)
                        ])
                    prefix = normalized + "/"
                    entries: dict[str, dict[str, Any]] = {}
                    for name in self.filenames:
                        if not name.startswith(prefix):
                            continue
                        remainder = name[len(prefix):]
                        child = remainder.split("/", 1)[0]
                        child_path = prefix + child
                        entries[child_path] = {
                            "path": child_path,
                            "is_dir": "/" in remainder,
                        }
                    return LsResult(entries=list(entries.values()))

                async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
                    results: list[FileDownloadResponse] = []
                    for path in paths:
                        resource = self.filenames.get(path)
                        if resource is None:
                            results.append(FileDownloadResponse(path=path, error="file_not_found"))
                            continue
                        try:
                            contents = await self.client.read_resource(resource.uri)
                            data = extract_bytes(list(contents))
                            expected = str(resource.digest)
                            actual = "sha256:" + hashlib.sha256(data).hexdigest()
                            if actual != expected or len(data) != resource.size:
                                raise ValueError(
                                    f"manifest verification failed for {resource.uri}: "
                                    f"expected {expected}, actual {actual}"
                                )
                            results.append(FileDownloadResponse(path=path, content=data))
                        except Exception as exc:
                            results.append(FileDownloadResponse(path=path, error=str(exc)))
                    return results

                async def aread(
                    self, file_path: str, offset: int = 0, limit: int = 2000
                ) -> ReadResult:
                    if limit <= 0:
                        return ReadResult(no_lines_requested=True)
                    response = (await self.adownload_files([file_path]))[0]
                    if response.error or response.content is None:
                        return ReadResult(error=response.error or "file_not_found")
                    try:
                        lines = response.content.decode("utf-8").splitlines(keepends=True)
                    except UnicodeDecodeError:
                        return ReadResult(error="non-utf8 skill file")
                    start = max(offset, 0)
                    selected = lines[start:start + limit]
                    if not selected:
                        return ReadResult(file_data={"content": "", "encoding": "utf-8"})
                    end = start + len(selected)
                    return ReadResult(
                        file_data={"content": "".join(selected), "encoding": "utf-8"},
                        start_line=start + 1,
                        end_line=end,
                        next_offset=end if end < len(lines) else None,
                        total_lines=len(lines),
                    )

            remote_backend = MCPReadOnlySkillsBackend(client, [skill])
            combined = CompositeBackend(
                default=FilesystemBackend(root_dir=workspace, virtual_mode=True),
                routes={MOUNT: remote_backend},
            )

            print("\nTEST 3: MIRA-style CompositeBackend + DeepAgents SkillsMiddleware")
            middleware = SkillsMiddleware(backend=combined, sources=[MOUNT + "skills"])
            update = await middleware.abefore_agent({}, None, {})
            assert update is not None, "Middleware returned no metadata"
            metadata = update["skills_metadata"]
            assert len(metadata) == 1, f"Unexpected discovered skills: {metadata!r}"
            assert metadata[0]["name"] == SKILL_NAME
            assert metadata[0]["path"] == f"{MOUNT}skills/{SKILL_NAME}/SKILL.md"
            print(f"  PASS: discovered {metadata[0]['name']} at {metadata[0]['path']}")

            print("\nTEST 4: skill instructions can be read through backend")
            read_result = await combined.aread(metadata[0]["path"])
            assert not read_result.error, f"read_file backend failure: {read_result.error}"
            assert "Read the document" in read_result.file_data["content"]
            helper_path = f"{MOUNT}skills/{SKILL_NAME}/references/guide.md"
            helper_result = (await combined.adownload_files([helper_path]))[0]
            assert not helper_result.error, f"supporting file failure: {helper_result.error}"
            expected_helper = HELPER_BODY.encode("utf-8")
            assert helper_result.content == expected_helper, (
                "Supporting file content differs from the authored sample. "
                f"Expected {expected_helper!r}, got {helper_result.content!r}"
            )
            print("  PASS: SKILL.md and supporting file returned via MCP resources/read")
            print("  PASS: both matched their MCP SHA-256 manifest entries")

            print("\nTEST 5: existing middleware pins full skill without an LLM")
            pinned = await middleware.abefore_model(
                {"skills_metadata": metadata, "pinned_skills": [SKILL_NAME]}, None
            )
            assert pinned is not None and pinned.get("messages")
            message = pinned["messages"][0]
            assert "Read the document" in str(message.content)
            assert message.additional_kwargs.get("skill", {}).get("name") == SKILL_NAME
            assert (await middleware.abefore_agent({"skills_metadata": metadata}, None, {})) is None
            print("  PASS: real skill content pinned as a LangChain message")
            print("  PASS: middleware honors its existing per-thread discovery cache")

            print("\nALL FIVE TESTS PASSED")
            print("PROVEN: An MCP-read-only backend can supply skills to the existing")
            print("        async SkillsMiddleware without a new skills middleware.")
            print("NOT PROVEN: integration into MIRA's app lifecycle, permissions,")
            print("            UI, checkpoint refresh, or a full LLM tool-calling turn.")


def main() -> int:
    if not __debug__:
        print("Run without -O: this probe requires Python assertions.", file=sys.stderr)
        return 2
    if len(sys.argv) >= 2 and sys.argv[1] == "--serve":
        serve(Path(sys.argv[2]))
        return 0
    try:
        asyncio.run(run_probe())
        return 0
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"DEPENDENCY ERROR: {exc}", file=sys.stderr)
        print("Run in the installed MIRA environment (see script header).", file=sys.stderr)
        return 2
    except Exception:
        print("\nPROBE FAILED: see traceback below", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
