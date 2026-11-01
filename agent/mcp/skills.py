"""Read-only SEP-2640 files mounted into the ordinary DeepAgents backend.

Connection-time discovery verifies and snapshots SKILL.md, allowing synchronous
resource assembly without blocking an event loop. Supporting files stay remote.
Only the backend's frontmatter presentation changes names; server bytes are
verified before that projection and are never written back or copied to disk.
"""

from __future__ import annotations

import base64
import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, unquote, urlsplit

import yaml
from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    FileDownloadResponse,
    FileUploadResponse,
    LsResult,
    ReadResult,
    WriteResult,
)
from fastmcp.skills.models import Skill, SkillResource

from agent.mcp.errors import sanitized_error
from agent.mcp.models import MCPServerState

MCP_SKILLS_ROUTE = "/mira-mcp/"
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_DISCOVERY_BYTES = 16 * 1024 * 1024


def skill_identity(server: str, name: str) -> str:
    """Fit an unambiguous server/skill identity into Agent Skills' 64 chars."""
    label = re.sub(r"[^a-z0-9]+", "-", server.lower()).strip("-")[:12].rstrip("-") or "server"
    digest = hashlib.sha256(f"{server}\0{name}".encode()).hexdigest()[:16]
    return f"mcp-{label}-{digest}-{name[:29].rstrip('-')}"


def _segments(path: str) -> tuple[str, ...]:
    parts = tuple(unquote(part) for part in path.split("/"))
    if any(
        not part or part in {".", ".."} or any(c in part for c in "\\/%?#")
        or any(ord(c) < 32 or ord(c) == 127 for c in part)
        for part in parts
    ):
        raise ValueError("Unsafe skill path")
    return parts


def _relative_uri(root: str, uri: str) -> str:
    if any(ord(c) < 32 or ord(c) == 127 for c in root + uri):
        raise ValueError("Invalid skill resource URI")
    base, target = urlsplit(root), urlsplit(uri)
    if not base.scheme or base.query or base.fragment or target.query or target.fragment:
        raise ValueError("Invalid skill resource URI")
    parent = _segments(base.path.lstrip("/"))[:-1]
    parts = _segments(target.path.lstrip("/"))
    if (
        (target.scheme, target.netloc) != (base.scheme, base.netloc)
        or parts[:len(parent)] != parent
        or len(parts) <= len(parent)
    ):
        raise ValueError("Resource is outside its skill directory")
    return "/".join(parts[len(parent):])


async def _read_bytes(session: Any, uri: str, resource: SkillResource | None = None) -> bytes:
    if resource is not None and resource.size > MAX_FILE_BYTES:
        raise ValueError("Skill file exceeds size limit")
    contents = await session.read_resource(uri)
    if len(contents) != 1 or str(contents[0].uri) != uri:
        raise ValueError("Expected one resource with the requested URI")
    content = contents[0]
    text = getattr(content, "text", None)
    blob = getattr(content, "blob", None)
    if isinstance(text, str):
        if len(text) > MAX_FILE_BYTES:
            raise ValueError("Skill file exceeds size limit")
        data = text.encode("utf-8")
    elif isinstance(blob, str):
        if len(blob) > ((MAX_FILE_BYTES + 2) // 3) * 4:
            raise ValueError("Skill file exceeds size limit")
        data = base64.b64decode(blob, validate=True)
    else:
        raise ValueError("Unsupported skill resource content")
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("Skill file exceeds size limit")
    if resource is not None and (
        len(data) != resource.size
        or "sha256:" + hashlib.sha256(data).hexdigest() != resource.digest
    ):
        raise ValueError("Skill manifest size or SHA-256 mismatch")
    return data


@dataclass(frozen=True)
class MCPSkill:
    name: str
    description: str
    manifest: Skill
    files: dict[str, SkillResource]
    instructions: bytes


async def discover_skill(session: Any, server: str, summary: Skill) -> MCPSkill:
    manifest = Skill.model_validate(await session.get_skill(summary.uri))
    if manifest.uri != summary.uri or manifest.frontmatter.name != summary.frontmatter.name:
        raise ValueError("skills/get changed the requested skill identity")
    _relative_uri(manifest.uri, manifest.uri)
    files: dict[str, SkillResource] = {}
    if isinstance(manifest.resources, list):
        if len(manifest.resources) > 10000:
            raise ValueError("Skill manifest has too many files")
        for resource in manifest.resources:
            relative = _relative_uri(manifest.uri, resource.uri)
            if relative in files:
                raise ValueError("Duplicate skill resource path")
            files[relative] = resource
    raw = await _read_bytes(session, manifest.uri, files.get("SKILL.md"))
    text = raw.decode("utf-8")
    match = re.match(r"\A---\r?\n(.*?)\r?\n---(?:\r?\n|$)", text, re.DOTALL)
    if match is None:
        raise ValueError("SKILL.md has no frontmatter")
    frontmatter = yaml.safe_load(match[1])
    if (
        not isinstance(frontmatter, dict)
        or frontmatter.get("name") != manifest.frontmatter.name
        or frontmatter.get("description") != manifest.frontmatter.description
    ):
        raise ValueError("SKILL.md frontmatter disagrees with manifest")
    name = skill_identity(server, manifest.frontmatter.name)
    frontmatter["name"] = name
    # The original instructions and all other frontmatter values are preserved.
    projected = "---\n" + yaml.safe_dump(frontmatter, sort_keys=False) + "---\n" + text[match.end():]
    return MCPSkill(name, manifest.frontmatter.description, manifest, files, projected.encode("utf-8"))


class MCPSkillsBackend(BackendProtocol):
    """One lifecycle-bound view of connected servers; no independent registry."""

    def __init__(self, manager: Any, reserved_names: set[str]) -> None:
        self.manager = manager
        self.reserved_names = reserved_names

    def _skills(self) -> Iterator[tuple[MCPServerState, MCPSkill]]:
        for state in self.manager.servers.values():
            if state.usable and state.session is not None and state.session.is_connected():
                for skill in state.skills or ():
                    if skill.name not in self.reserved_names:
                        yield state, skill

    def _find(self, path: str) -> tuple[MCPServerState, MCPSkill, str]:
        parts = _segments(path.removeprefix("/"))
        if len(parts) < 3 or parts[0] != "skills":
            raise ValueError("Unknown skill file")
        for state, skill in self._skills():
            if parts[1] == skill.name:
                relative = "/".join(parts[2:])
                if skill.manifest.resources != "dynamic" and relative not in skill.files:
                    break
                return state, skill, relative
        raise ValueError("Skill is unavailable or file is not in its manifest")

    def ls(self, path: str) -> LsResult:
        prefix = path.rstrip("/") + "/"
        entries = {}
        for _, skill in self._skills():
            for relative in set(skill.files) | {"SKILL.md"}:
                target = f"/skills/{skill.name}/{relative}"
                if target.startswith(prefix):
                    child, _, rest = target[len(prefix):].partition("/")
                    entries[prefix + child] = {"path": prefix + child, "is_dir": bool(rest)}
        return LsResult(entries=list(entries.values()))

    async def als(self, path: str) -> LsResult:
        return self.ls(path)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        results = []
        for path in paths:
            try:
                _, skill, relative = self._find(path)
                if relative != "SKILL.md":
                    raise ValueError("Remote supporting files require asynchronous reads")
                results.append(FileDownloadResponse(path=path, content=skill.instructions))
            except ValueError as error:
                results.append(FileDownloadResponse(path=path, error=str(error)))
        return results

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        results = []
        for path in paths:
            try:
                state, skill, relative = self._find(path)
                session = state.session
                if relative == "SKILL.md":
                    data = skill.instructions
                else:
                    resource = skill.files.get(relative)
                    uri = (
                        resource.uri if resource else
                        skill.manifest.uri.rsplit("/", 1)[0] + "/" + quote(relative, safe="/")
                    )
                    _relative_uri(skill.manifest.uri, uri)
                    data = await _read_bytes(session, uri, resource)
                if state.session is not session or not state.usable or not session.is_connected():
                    raise ValueError("MCP server disconnected during read")
                results.append(FileDownloadResponse(path=path, content=data))
            except Exception as error:
                results.append(FileDownloadResponse(path=path, error=sanitized_error(error)))
        return results

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        if limit <= 0:
            return ReadResult(no_lines_requested=True)
        result = (await self.adownload_files([file_path]))[0]
        if result.error or result.content is None:
            return ReadResult(error=result.error or "File unavailable")
        try:
            lines = result.content.decode("utf-8").splitlines(keepends=True)
        except UnicodeDecodeError:
            return ReadResult(error="Skill file is not UTF-8 text")
        start = max(0, offset)
        selected = lines[start:start + limit]
        end = start + len(selected)
        return ReadResult(
            file_data={"content": "".join(selected), "encoding": "utf-8"},
            start_line=start + 1 if selected else None,
            end_line=end if selected else None,
            total_lines=len(lines),
            next_offset=end if end < len(lines) else None,
        )

    def write(self, file_path: str, content: str) -> WriteResult:
        return WriteResult(error="MCP skills are read-only")

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        return EditResult(error="MCP skills are read-only")

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return [FileUploadResponse(path=path, error="permission_denied") for path, _ in files]
