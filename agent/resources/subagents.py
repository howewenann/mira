"""Discovery and effective selection for DeepAgents subagents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT
from langchain_core.language_models import BaseChatModel

from agent.resources.paths import (
    SUBAGENTS_DIR,
    default_dir,
    default_virtual_dir,
    project_dir,
    project_virtual_dir,
)
from agent.resources.python_files import import_python_file
from config.settings import subagent_enabled, subagent_model_assignment
from core.diagnostics.issues import Issue


@dataclass(frozen=True, slots=True)
class DiscoveredSubagent:
    name: str
    description: str
    spec: Any
    path: str
    source: str
    replaces: str = ""
    kind: str = "raw"
    source_model_label: str = ""

    def display_item(self) -> dict[str, str]:
        item = {
            "name": self.name,
            "description": self.description,
            "path": self.path,
            "source": self.source,
            "replaces": self.replaces,
            "kind": self.kind,
            "source_model": self.source_model_label,
        }
        graph_id = self.spec.get("graph_id") if isinstance(self.spec, dict) else getattr(self.spec, "graph_id", None)
        if graph_id:
            item["graph_id"] = str(graph_id)
        return item


@dataclass(frozen=True, slots=True)
class SubagentDiscovery:
    items: tuple[DiscoveredSubagent, ...]
    issues: tuple[Issue, ...] = ()
    complete: bool = True


def load_subagents(workspace: Path) -> tuple[list[Any], list[dict[str, str]]]:
    """Compatibility projection of all discovered definitions."""
    discovery = discover_subagents(workspace)
    return [item.spec for item in discovery.items if item.source != "built-in"], [item.display_item() for item in discovery.items]


def discover_subagents(workspace: Path) -> SubagentDiscovery:
    defaults, default_issues, default_complete = _discover_tier(
        default_dir(SUBAGENTS_DIR), default_virtual_dir(SUBAGENTS_DIR), "default", workspace
    )
    projects, project_issues, project_complete = _discover_tier(
        project_dir(workspace, SUBAGENTS_DIR), project_virtual_dir(SUBAGENTS_DIR), "project", workspace
    )
    issues = [*default_issues, *project_issues]
    defaults = _exclude_duplicates(defaults, issues)
    projects = _exclude_duplicates(projects, issues)

    merged: dict[str, DiscoveredSubagent] = {item.name: item for item in defaults}
    for item in projects:
        replaced = merged.get(item.name)
        merged[item.name] = DiscoveredSubagent(
            name=item.name,
            description=item.description,
            spec=item.spec,
            path=item.path,
            source=item.source,
            replaces=replaced.source if replaced is not None else "",
            kind=item.kind,
            source_model_label=item.source_model_label,
        )

    items = list(merged.values())
    general = next((item for item in items if item.name == "general-purpose"), None)
    if general is None:
        general = _descriptor(dict(GENERAL_PURPOSE_SUBAGENT), "deepagents", "built-in")
    else:
        items.remove(general)
    return SubagentDiscovery(
        (general, *items),
        tuple(issues),
        default_complete and project_complete,
    )


def effective_subagent_specs(discovery: SubagentDiscovery, config: dict[str, Any]) -> list[Any]:
    """Return enabled definitions with MIRA overrides applied to raw copies."""
    from agent.llm import get_profile_model

    selected: list[Any] = []
    for item in discovery.items:
        if not subagent_enabled(config, item.name):
            continue
        spec = item.spec
        if item.kind != "raw":
            selected.append(spec)
            continue
        effective = dict(spec)
        override = subagent_model_assignment(config, item.name)
        if override:
            effective["model"] = get_profile_model(config, override)
        selected.append(effective)
    return selected


def subagent_model_issues(discovery: SubagentDiscovery, config: dict[str, Any]) -> list[Issue]:
    from agent.llm import model_registry

    registry = model_registry(config)
    issues: list[Issue] = []
    for item in discovery.items:
        if item.kind != "raw" or not subagent_enabled(config, item.name):
            continue
        override = subagent_model_assignment(config, item.name)
        if not override or registry.profile(override) is not None:
            continue
        issues.append(
            Issue(
                "MODEL",
                f"Missing subagent model profile: {override}",
                f".mira/settings.yml: models.subagents.{item.name}.model",
                f"Enabled subagent '{item.name}' explicitly requires '{override}'.",
                f"Select a valid model for {item.name}, clear its override, or disable it via /models.",
            )
        )
    return issues


def _discover_tier(
    root: Path,
    virtual_root: str,
    source: str,
    workspace: Path,
) -> tuple[list[DiscoveredSubagent], list[Issue], bool]:
    if not root.exists():
        return [], [], True
    items: list[DiscoveredSubagent] = []
    issues: list[Issue] = []
    complete = True
    for path in sorted(root.glob("*.py"), key=lambda item: (item.name.casefold(), item.name)):
        try:
            specs = subagents_from_file(path)
            items.extend(_descriptor(spec, f"{virtual_root}/{path.name}", source) for spec in specs)
        except Exception as error:
            complete = False
            try:
                location = path.resolve().relative_to(workspace.resolve()).as_posix()
            except ValueError:
                location = str(path)
            issues.append(
                Issue(
                    "STARTUP",
                    f"Could not load subagent file: {path.name}",
                    location,
                    f"{type(error).__name__}: {error}",
                    "Correct the subagent definition or its imports and run /reload.",
                )
            )
    return items, issues, complete


def _exclude_duplicates(items: list[DiscoveredSubagent], issues: list[Issue]) -> list[DiscoveredSubagent]:
    grouped: dict[str, list[DiscoveredSubagent]] = {}
    for item in items:
        grouped.setdefault(item.name, []).append(item)
    unique: list[DiscoveredSubagent] = []
    for item in items:
        conflicts = grouped[item.name]
        if len(conflicts) == 1:
            unique.append(item)
            continue
        if item is not conflicts[0]:
            continue
        locations = "\n".join(f"- {conflict.path}" for conflict in conflicts)
        issues.append(
            Issue(
                "STARTUP",
                f"Duplicate subagent name: {item.name}",
                item.path,
                f"Conflicting definitions:\n{locations}",
                "Rename one definition and run /reload.",
            )
        )
    return unique


def _descriptor(spec: Any, path: str, source: str) -> DiscoveredSubagent:
    name = subagent_name(spec)
    description = str(spec.get("description") or "") if isinstance(spec, dict) else str(getattr(spec, "description", "") or "")
    kind = "raw"
    if isinstance(spec, dict) and "graph_id" in spec:
        kind = "async"
    elif isinstance(spec, dict) and "runnable" in spec:
        kind = "compiled"
    model = spec.get("model") if isinstance(spec, dict) else getattr(spec, "model", None)
    return DiscoveredSubagent(
        name,
        description,
        spec,
        path,
        source,
        kind=kind,
        source_model_label=_source_model_label(model),
    )


def _source_model_label(model: Any) -> str:
    if model is None:
        return ""
    if isinstance(model, str):
        return f"[defined] {model}"
    provider = _model_provider(model)
    identifier = _model_identifier(model)
    if provider and identifier:
        return f"[defined] {provider}:{identifier}"
    return f"[defined] {type(model).__name__}"


def _model_identifier(model: Any) -> str:
    for attribute in ("model_name", "model", "model_id"):
        value = getattr(model, attribute, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _model_provider(model: Any) -> str:
    if not isinstance(model, BaseChatModel):
        return ""
    getter = getattr(model, "_get_ls_params", None)
    if not callable(getter):
        return ""
    try:
        params = getter()
    except Exception:
        return ""
    value = params.get("ls_provider") if isinstance(params, dict) else None
    return value.strip() if isinstance(value, str) else ""


def subagents_from_file(path: Path) -> list[Any]:
    module = import_python_file(path, "mira_resource_subagents")
    subagents = getattr(module, "SUBAGENTS", [])
    if not isinstance(subagents, list | tuple):
        raise TypeError(f"{path} must define SUBAGENTS as a list")
    return list(subagents)


def subagent_name(subagent: Any) -> str:
    name = subagent.get("name") if isinstance(subagent, dict) else getattr(subagent, "name", None)
    if not name:
        raise ValueError(f"Subagent is missing a name: {subagent!r}")
    return str(name)


__all__ = [
    "DiscoveredSubagent",
    "SubagentDiscovery",
    "discover_subagents",
    "effective_subagent_specs",
    "load_subagents",
    "subagent_model_issues",
]
