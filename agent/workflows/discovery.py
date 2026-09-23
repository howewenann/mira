"""Discovery and invocation parsing for launchable workspace workflows."""

from __future__ import annotations

import inspect
import json
import shlex
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from agent.resources.paths import WORKFLOWS_DIR, project_dir
from agent.resources.python_files import import_python_file
from core.diagnostics.issues import Issue


@dataclass(frozen=True, slots=True)
class WorkflowInput:
    """One top-level launch input exposed by a native LangGraph schema."""

    name: str
    required: bool
    type_label: str


@dataclass(frozen=True, slots=True)
class WorkflowSpec:
    """The process-local metadata needed to launch one workspace workflow."""

    name: str
    command: str
    path: Path
    factory: Callable[[Any], Any]
    input_model: Any
    inputs: tuple[WorkflowInput, ...]

    @property
    def usage(self) -> str:
        arguments = " ".join(
            (
                f"{item.name}=<{item.type_label}>"
                if item.required
                else f"[{item.name}=<{item.type_label}>]"
            )
            for item in self.inputs
        )
        return f"{self.command} {arguments}" if arguments else self.command

    @property
    def contract(self) -> tuple[tuple[str, bool, str], ...]:
        return tuple((item.name, item.required, item.type_label) for item in self.inputs)


@dataclass(frozen=True, slots=True)
class WorkflowRegistry:
    """Valid workspace workflows plus their non-fatal discovery issues."""

    specs: Mapping[str, WorkflowSpec]
    issues: tuple[Issue, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "specs", MappingProxyType(dict(self.specs)))
        object.__setattr__(self, "issues", tuple(self.issues))

    @property
    def commands(self) -> dict[str, WorkflowSpec]:
        return {spec.command: spec for spec in self.specs.values()}

    def resolve(self, invocation: str) -> tuple[WorkflowSpec, dict[str, Any]] | None:
        command = invocation.lstrip().split(maxsplit=1)[0]
        spec = self.commands.get(command)
        if spec is None:
            if command.startswith("/workflow__"):
                raise ValueError(f"unknown workflow command: {command}")
            return None
        return spec, workflow_arguments(spec, invocation)


def discover_workflows(workspace: Path, mira: Any) -> WorkflowRegistry:
    """Discover valid direct Python workflow files without aborting startup."""
    root = project_dir(Path(workspace), WORKFLOWS_DIR)
    if not root.is_dir():
        return WorkflowRegistry({})

    specs: dict[str, WorkflowSpec] = {}
    issues: list[Issue] = []
    paths = sorted(root.glob("*.py"), key=lambda item: (item.name.casefold(), item.name))
    for path in paths:
        name = path.stem
        try:
            module = import_python_file(path, "mira_resource_workflow")
            factory = getattr(module, "workflow", None)
            _validate_factory(factory)
            graph = factory(mira)
            input_model, inputs = validate_workflow_graph(graph)
            spec = WorkflowSpec(
                name=name,
                command=f"/workflow__{name}",
                path=path,
                factory=factory,
                input_model=input_model,
                inputs=inputs,
            )
            specs[name] = spec
        except BaseException as error:
            issues.append(_workflow_issue(name, path, workspace, error))
    return WorkflowRegistry(specs, tuple(issues))


def validate_runtime_graph(spec: WorkflowSpec, graph: Any) -> Any:
    """Validate a fresh invocation graph against its registered public contract."""
    input_model, inputs = validate_workflow_graph(graph)
    contract = tuple((item.name, item.required, item.type_label) for item in inputs)
    if contract != spec.contract:
        raise ValueError(
            f"{spec.command} input schema changed since discovery; run /reload"
        )
    return input_model


def validate_workflow_graph(graph: Any) -> tuple[Any, tuple[WorkflowInput, ...]]:
    """Return the public validation model and shallow input contract."""
    if not isinstance(graph, CompiledStateGraph):
        raise TypeError("workflow(mira) must return a compiled LangGraph.")
    if graph.builder.input_schema is graph.builder.state_schema:
        raise TypeError(
            "MIRA workflows require a dedicated LangGraph input_schema.\n\n"
            "Use:\n\nStateGraph(State, input_schema=InputState)"
        )
    schema = graph.get_input_jsonschema()
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if (
        not isinstance(schema, dict)
        or schema.get("type") != "object"
        or not isinstance(properties, dict)
        or not properties
    ):
        raise TypeError(
            "workflow input_schema must be a top-level object with named properties."
        )
    required_names = schema.get("required", [])
    if not isinstance(required_names, list):
        required_names = []
    required = {
        str(name)
        for name in required_names
        if isinstance(name, str)
    }
    inputs = tuple(
        WorkflowInput(str(name), str(name) in required, _type_label(value))
        for name, value in properties.items()
    )
    return graph.get_input_schema(), inputs


def validate_workflow_input(input_model: Any, values: dict[str, Any], usage: str) -> None:
    """Validate decoded values through LangGraph's public Pydantic input model."""
    try:
        input_model.model_validate(values)
    except Exception as error:
        detail = str(error).splitlines()[0]
        raise ValueError(f"invalid workflow input: {detail}; usage: {usage}") from error


def workflow_arguments(spec: WorkflowSpec, invocation: str) -> dict[str, Any]:
    """Parse uniform name=value workflow arguments with JSON-first values."""
    usage = f"usage: {spec.usage}"
    try:
        tokens = _raw_shell_tokens(invocation)
    except ValueError as error:
        raise ValueError(f"invalid workflow arguments: {error}; {usage}") from error
    if not tokens or tokens[0] != spec.command:
        raise ValueError(usage)

    fields = {item.name: item for item in spec.inputs}
    values: dict[str, Any] = {}
    for token in tokens[1:]:
        name, separator, raw_value = token.partition("=")
        if not separator or not name or not raw_value:
            raise ValueError(f"malformed workflow argument: {token}; {usage}")
        if name not in fields:
            raise ValueError(f"unknown workflow argument: {name}; {usage}")
        if name in values:
            raise ValueError(f"duplicate workflow argument: {name}; {usage}")
        values[name] = _decode_value(raw_value)

    missing = [item.name for item in spec.inputs if item.required and item.name not in values]
    if missing:
        raise ValueError(
            f"missing required workflow arguments: {', '.join(missing)}; {usage}"
        )
    validate_workflow_input(spec.input_model, values, spec.usage)
    return values


def _validate_factory(factory: Any) -> None:
    if factory is None:
        raise TypeError("workflow(mira) was not found.")
    if not callable(factory):
        raise TypeError("workflow(mira) must be callable.")
    if inspect.iscoroutinefunction(factory):
        raise TypeError("workflow(mira) must be a synchronous graph-construction factory.")
    parameters = list(inspect.signature(factory).parameters.values())
    if not (
        len(parameters) == 1
        and parameters[0].name == "mira"
        and parameters[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        and parameters[0].default is inspect.Parameter.empty
    ):
        raise TypeError("workflow factory signature must be exactly: def workflow(mira):")


def _workflow_issue(
    name: str,
    path: Path,
    workspace: Path,
    error: BaseException,
) -> Issue:
    try:
        location = path.resolve().relative_to(Path(workspace).resolve()).as_posix()
    except ValueError:
        location = str(path)
    detail = str(error).strip() or type(error).__name__
    return Issue(
        "STARTUP",
        f"Workflow: {name}",
        location,
        detail,
        "Correct the workflow definition or its imports and run /reload.",
    )


def _type_label(schema: Any) -> str:
    if not isinstance(schema, dict):
        return "json"
    value_type = schema.get("type")
    if isinstance(value_type, list):
        non_null = [item for item in value_type if item != "null"]
        value_type = non_null[0] if len(non_null) == 1 else None
    if value_type is None and isinstance(schema.get("anyOf"), list):
        variants = [
            item.get("type")
            for item in schema["anyOf"]
            if isinstance(item, dict) and item.get("type") != "null"
        ]
        value_type = variants[0] if len(variants) == 1 else None
    return {
        "string": "str",
        "integer": "int",
        "number": "float/number",
        "boolean": "bool",
        "array": "list",
        "object": "dict",
    }.get(value_type, "json")


def _decode_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            parsed = shlex.split(value, posix=True)
        except ValueError as error:
            raise ValueError(str(error)) from error
        if len(parsed) != 1:
            raise ValueError(f"invalid value: {value}")
        return parsed[0]


def _raw_shell_tokens(text: str) -> list[str]:
    """Split assignments while retaining quotes and JSON container whitespace."""
    tokens: list[str] = []
    current: list[str] = []
    quote = ""
    escaped = False
    containers: list[str] = []
    closing = {"[": "]", "{": "}"}
    for character in text.strip():
        if escaped:
            current.extend(("\\", character))
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if quote:
            current.append(character)
            if character == quote:
                quote = ""
            continue
        if character in {'"', "'"}:
            quote = character
            current.append(character)
            continue
        if character in closing:
            containers.append(closing[character])
            current.append(character)
            continue
        if character in {"]", "}"}:
            if containers and character == containers[-1]:
                containers.pop()
            current.append(character)
            continue
        if character.isspace() and not containers:
            if current:
                tokens.append("".join(current))
                current = []
            continue
        current.append(character)
    if escaped:
        current.append("\\")
    if quote:
        raise ValueError("No closing quotation")
    if containers:
        raise ValueError("No closing JSON container")
    if current:
        tokens.append("".join(current))
    return tokens


__all__ = [
    "WorkflowInput",
    "WorkflowRegistry",
    "WorkflowSpec",
    "discover_workflows",
    "validate_runtime_graph",
    "validate_workflow_graph",
    "validate_workflow_input",
    "workflow_arguments",
]
