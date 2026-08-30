"""Workspace tracing-profile registry and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from core.diagnostics.issues import Issue

TRACING_FILE = "tracing.yml"
SpanAttributeValue = (
    str | bool | int | float | list[str] | list[bool] | list[int] | list[float]
)
TRACING_REGISTRY_TEMPLATE = """# MIRA tracing configuration
#
# Each profile defines an OTLP/HTTP tracing destination.
#
# Profile fields:
#   endpoint:         OTLP/HTTP trace endpoint
#   headers:          Optional HTTP headers sent with trace exports
#   span_attributes:  Optional attributes added to every exported span
#
# Environment variables can be referenced with ${NAME}.

profiles:
  phoenix:
    endpoint: http://127.0.0.1:6006/v1/traces
    headers: {}

  mlflow:
    endpoint: http://127.0.0.1:5000/v1/traces
    headers:
      x-mlflow-experiment-id: "0"
    span_attributes:
      mlflow.message.format: langchain

  langsmith:
    endpoint: https://api.smith.langchain.com/otel/v1/traces
    headers:
      x-api-key: ${LANGSMITH_API_KEY}
      X-Tenant-Id: ${LANGSMITH_WORKSPACE_ID}
      Langsmith-Project: MIRA
"""


@dataclass(frozen=True, slots=True)
class TracingProfile:
    """One validated tracing profile with unresolved environment references."""

    name: str
    endpoint: str
    headers: dict[str, str]
    span_attributes: dict[str, SpanAttributeValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TracingRegistry:
    """Ordered valid profiles plus diagnostics for invalid entries."""

    profiles: dict[str, TracingProfile] = field(default_factory=dict)
    invalid_names: tuple[str, ...] = ()
    issues: tuple[Issue, ...] = ()

    def profile(self, name: str | None) -> TracingProfile | None:
        return self.profiles.get(str(name or ""))


def tracing_path(workspace: Path) -> Path:
    return workspace.expanduser().resolve() / ".mira" / TRACING_FILE


def bootstrap_tracing_registry(workspace: Path) -> Path:
    """Create the initial registry once, preserving every existing user file."""
    path = tracing_path(workspace)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TRACING_REGISTRY_TEMPLATE, encoding="utf-8")
    return path


def load_tracing_registry(workspace: Path) -> TracingRegistry:
    """Load structurally valid profiles without resolving secret references."""
    try:
        path = bootstrap_tracing_registry(workspace)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        return TracingRegistry(issues=(_registry_issue("Invalid tracing registry", _error_text(error)),))

    if not isinstance(raw, dict) or set(raw) != {"profiles"}:
        return TracingRegistry(
            issues=(
                _registry_issue(
                    "Invalid tracing registry",
                    "Top level must contain only the profiles mapping.",
                ),
            )
        )
    entries = raw.get("profiles")
    if not isinstance(entries, dict):
        return TracingRegistry(
            issues=(
                _registry_issue(
                    "Invalid tracing profiles mapping",
                    "profiles must be a mapping of profile names to profile data.",
                ),
            )
        )

    profiles: dict[str, TracingProfile] = {}
    invalid: list[str] = []
    issues: list[Issue] = []
    for raw_name, raw_profile in entries.items():
        name = raw_name if isinstance(raw_name, str) else ""
        try:
            endpoint, headers, span_attributes = _validate_profile(name, raw_profile)
        except ValueError as error:
            shown = name or repr(raw_name)
            invalid.append(name)
            issues.append(
                Issue(
                    "STARTUP",
                    f"Invalid tracing profile: {shown}",
                    f".mira/tracing.yml: profiles.{shown}",
                    str(error),
                    "Correct the profile or remove it, then run /reload-runtime.",
                )
            )
            continue
        profiles[name] = TracingProfile(name, endpoint, headers, span_attributes)
    return TracingRegistry(profiles, tuple(invalid), tuple(issues))


def _validate_profile(
    name: str,
    raw: Any,
) -> tuple[str, dict[str, str], dict[str, SpanAttributeValue]]:
    if not name.strip():
        raise ValueError("profile name must be a non-empty string")
    if not isinstance(raw, dict):
        raise ValueError("profile must be a mapping")
    required = {"endpoint", "headers"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"profile is missing required field(s): {', '.join(sorted(missing))}")
    unknown = set(raw) - required - {"span_attributes"}
    if unknown:
        raise ValueError(f"unsupported profile field(s): {', '.join(sorted(map(str, unknown)))}")
    endpoint = raw.get("endpoint")
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("endpoint is required and must be a non-empty string")
    headers = raw.get("headers")
    if not isinstance(headers, dict):
        raise ValueError("headers must be a mapping")
    if not all(
        isinstance(key, str) and key.strip() and isinstance(value, str)
        for key, value in headers.items()
    ):
        raise ValueError("headers must use non-empty string names and string values")
    span_attributes = raw.get("span_attributes", {})
    if not isinstance(span_attributes, dict):
        raise ValueError("span_attributes must be a mapping")
    if not all(isinstance(key, str) and key.strip() for key in span_attributes):
        raise ValueError("span_attributes must use non-empty string names")
    invalid = [key for key, value in span_attributes.items() if not _is_span_attribute(value)]
    if invalid:
        raise ValueError(
            "span_attributes values must be OpenTelemetry scalars or homogeneous scalar lists; "
            f"invalid: {', '.join(invalid)}"
        )
    return endpoint.strip(), dict(headers), dict(span_attributes)


def _is_span_attribute(value: Any) -> bool:
    if isinstance(value, str | bool | int | float):
        return True
    if not isinstance(value, list):
        return False
    if not value:
        return True
    return all(type(item) is type(value[0]) for item in value) and all(
        isinstance(item, str | bool | int | float) for item in value
    )


def _registry_issue(summary: str, details: str) -> Issue:
    return Issue(
        "STARTUP",
        summary,
        ".mira/tracing.yml",
        details,
        "Fix .mira/tracing.yml and run /reload-runtime.",
    )


def _error_text(error: BaseException) -> str:
    message = str(error).strip()
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


__all__ = [
    "TRACING_FILE",
    "TRACING_REGISTRY_TEMPLATE",
    "TracingProfile",
    "TracingRegistry",
    "bootstrap_tracing_registry",
    "load_tracing_registry",
    "tracing_path",
]
