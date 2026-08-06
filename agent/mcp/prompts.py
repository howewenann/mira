"""One shared local and MCP prompt command registry."""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from agent.mcp.models import PreparedPrompt, PromptArgument, PromptSpec

_MUSTACHE = re.compile(r"{{{?\s*([#^/!>&]?)\s*([A-Za-z_][\w.]*)\s*}?}}")


class PromptRegistry:
    """Canonical registry used by invocation, help, autocomplete, and panels."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.local: dict[str, PromptSpec] = {}
        self.mcp: dict[str, PromptSpec] = {}
        self.warnings: list[str] = []
        self.reload_local()

    @property
    def specs(self) -> dict[str, PromptSpec]:
        return {**self.local, **self.mcp}

    def reload_local(self) -> None:
        self.local = {}
        self.warnings = []
        root = self.workspace / ".mira" / "prompts"
        if not root.is_dir():
            return
        grouped: dict[str, list[tuple[Path, str]]] = {}
        for path in sorted((item for item in root.iterdir() if item.is_file()), key=lambda item: item.name.casefold()):
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            command = f"/prompt__{path.stem}"
            grouped.setdefault(command, []).append((path, text))
        for command, items in grouped.items():
            if len(items) != 1:
                self.warnings.append(f"local prompt collision excluded: {command}")
                continue
            _path, template_text = items[0]
            names = mustache_variables(template_text)
            template = ChatPromptTemplate.from_template(template_text, template_format="mustache")

            async def resolve(values: dict[str, str], *, _template: Any = template) -> list[BaseMessage]:
                return list(_template.format_messages(**values))

            self.local[command] = PromptSpec(
                command=command,
                description="Local prompt",
                arguments=tuple(PromptArgument(name) for name in names),
                source="local",
                resolver=resolve,
            )

    def replace_server(self, server: str, specs: list[PromptSpec]) -> None:
        self.remove_server(server)
        self.mcp.update({spec.command: spec for spec in specs})

    def remove_server(self, server: str) -> None:
        self.mcp = {key: value for key, value in self.mcp.items() if value.server != server}

    async def resolve(self, invocation: str) -> PreparedPrompt | None:
        if not invocation.startswith("/"):
            return None
        try:
            parts = shlex.split(invocation, posix=True)
        except ValueError as error:
            token = invocation.split(maxsplit=1)[0]
            if token in self.specs or token.startswith(("/prompt__", "/mcp__")):
                raise ValueError(f"invalid prompt arguments: {error}") from error
            return None
        if not parts:
            return None
        spec = self.specs.get(parts[0])
        if spec is None:
            if parts[0].startswith(("/prompt__", "/mcp__")):
                raise ValueError(f"unknown prompt command: {parts[0]}")
            return None
        values = prompt_arguments(spec, parts[1:])
        return PreparedPrompt(invocation, await spec.resolver(values))

    def rows(self) -> list[tuple[str, str]]:
        return [(spec.usage, spec.description) for spec in self.specs.values()]


def mustache_variables(text: str) -> tuple[str, ...]:
    names: list[str] = []
    for marker, name in _MUSTACHE.findall(text):
        if marker in {"/", "!", ">", "&"}:
            continue
        root = name.split(".", 1)[0]
        if root not in names:
            names.append(root)
    return tuple(names)


def prompt_arguments(spec: PromptSpec, tokens: list[str]) -> dict[str, str]:
    """Parse one invocation using the prompt's unambiguous argument style."""
    if all(argument.required for argument in spec.arguments):
        if len(tokens) != len(spec.arguments):
            raise ValueError(f"usage: {spec.usage}")
        return {
            argument.name: value
            for argument, value in zip(spec.arguments, tokens, strict=True)
        }

    usage = f"usage: {spec.usage}; use name=value for every argument"
    arguments = {argument.name: argument for argument in spec.arguments}
    values: dict[str, str] = {}
    for token in tokens:
        name, separator, value = token.partition("=")
        if not separator or not name:
            raise ValueError(usage)
        if name not in arguments:
            raise ValueError(f"unknown prompt argument: {name}; {usage}")
        if name in values:
            raise ValueError(f"duplicate prompt argument: {name}; {usage}")
        values[name] = value

    missing = [
        argument.name
        for argument in spec.arguments
        if argument.required and argument.name not in values
    ]
    if missing:
        raise ValueError(f"missing required prompt arguments: {', '.join(missing)}; {usage}")
    return values
