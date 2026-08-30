"""Ordered model-profile registry and validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from config.interpolation import EnvironmentInterpolationError, resolve_environment
from core.diagnostics.issues import Issue

MODELS_FILE = "models.yml"
DEFAULT_CONTEXT_TOKENS = 32768
PROFILE_FIELDS = {
    "provider",
    "model",
    "api_key",
    "api_base",
    "temperature",
    "max_tokens",
    "top_p",
    "model_kwargs",
}
RESERVED_MODEL_KWARGS = {
    "cache",
    "callbacks",
    "custom_get_token_ids",
    "metadata",
    "model",
    "model_id",
    "name",
    "output_version",
    "profile",
    "provider",
    "api_key",
    "api_base",
    "base_url",
    "temperature",
    "max_tokens",
    "top_p",
    "model_kwargs",
    "messages",
    "tools",
    "tool_choice",
    "response_format",
    "stream",
    "streaming",
    "stream_options",
    "disable_streaming",
    "rate_limiter",
    "tags",
    "verbose",
    "client_args",
    "client",
    "async_client",
}


class ConfigError(ValueError):
    """Expected configuration failure that must not create a crash report."""


class DuplicateKeyError(ValueError):
    """Raised when a YAML mapping contains duplicate keys."""


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """One validated, unresolved-name model profile."""

    name: str
    values: dict[str, Any]

    @property
    def provider(self) -> str:
        return str(self.values["provider"])

    @property
    def model(self) -> str:
        return str(self.values["model"])

    @property
    def identity(self) -> str:
        return f"[{self.name}] {self.provider}:{self.model}"


@dataclass(frozen=True, slots=True)
class ModelRegistry:
    """Ordered valid profiles plus diagnostics for invalid entries."""

    profiles: dict[str, ModelProfile] = field(default_factory=dict)
    invalid_names: tuple[str, ...] = ()
    issues: tuple[Issue, ...] = ()

    def profile(self, name: str | None) -> ModelProfile | None:
        return self.profiles.get(str(name or ""))


def models_path(workspace: Path) -> Path:
    return workspace.expanduser().resolve() / ".mira" / MODELS_FILE


def load_model_registry(
    workspace: Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> ModelRegistry:
    """Load valid profiles while retaining independent profile diagnostics."""
    path = models_path(workspace)
    if not path.exists():
        return ModelRegistry()
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, UnicodeError, yaml.YAMLError, DuplicateKeyError) as error:
        return ModelRegistry(
            issues=(
                Issue(
                    "MODEL",
                    "Invalid model registry",
                    ".mira/models.yml",
                    _error_text(error),
                    "Fix the registry YAML and run /reload.",
                ),
            )
        )
    if not isinstance(raw, dict) or set(raw) != {"models"}:
        return ModelRegistry(
            issues=(
                Issue(
                    "MODEL",
                    "Invalid model registry",
                    ".mira/models.yml",
                    "Top level must contain only the models mapping.",
                    "Use a top-level models: key and run /reload.",
                ),
            )
        )
    entries = raw.get("models")
    if entries is None:
        entries = {}
    if not isinstance(entries, dict):
        return ModelRegistry(
            issues=(
                Issue(
                    "MODEL",
                    "Invalid models mapping",
                    ".mira/models.yml",
                    "models must be a mapping of profile names to profile data.",
                    "Fix .mira/models.yml and run /reload.",
                ),
            )
        )

    profiles: dict[str, ModelProfile] = {}
    invalid: list[str] = []
    issues: list[Issue] = []
    for raw_name, raw_profile in entries.items():
        name = str(raw_name) if isinstance(raw_name, str) else ""
        try:
            values = _validate_profile(name, raw_profile, environ=environ)
        except (ValueError, EnvironmentInterpolationError) as error:
            shown = name or repr(raw_name)
            invalid.append(name)
            issues.append(
                Issue(
                    "MODEL",
                    f"Invalid model profile: {shown}",
                    f".mira/models.yml: models.{shown}",
                    str(error),
                    "Correct the profile or remove it, then run /reload.",
                )
            )
            continue
        profiles[name] = ModelProfile(name, values)
    return ModelRegistry(profiles, tuple(invalid), tuple(issues))


def _validate_profile(
    name: str,
    raw: Any,
    *,
    environ: Mapping[str, str] | None,
) -> dict[str, Any]:
    if not name.strip():
        raise ValueError("profile name must be a non-empty string")
    if not isinstance(raw, dict):
        raise ValueError("profile must be a mapping")
    unknown = [str(key) for key in raw if key not in PROFILE_FIELDS]
    if unknown:
        raise ValueError(f"unsupported profile field(s): {', '.join(unknown)}")
    values = resolve_environment(raw, environ=environ)
    for required in ("provider", "model"):
        value = values.get(required)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{required} is required and must be a non-empty string")
        values[required] = value.strip()
    for key in ("api_key", "api_base"):
        value = values.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
    for key in ("temperature", "top_p"):
        value = values.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int | float)):
            raise ValueError(f"{key} must be a number")
    max_tokens = values.get("max_tokens")
    if max_tokens is not None and (
        isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0
    ):
        raise ValueError("max_tokens must be a positive integer")
    model_kwargs = values.get("model_kwargs", {})
    if model_kwargs is None:
        model_kwargs = {}
    if not isinstance(model_kwargs, dict):
        raise ValueError("model_kwargs must be a mapping")
    if not all(isinstance(key, str) and key for key in model_kwargs):
        raise ValueError("model_kwargs keys must be non-empty strings")
    reserved = sorted(str(key) for key in model_kwargs if str(key).lower() in RESERVED_MODEL_KWARGS)
    if reserved:
        raise ValueError(f"model_kwargs cannot set runtime-owned key(s): {', '.join(reserved)}")
    values["model_kwargs"] = dict(model_kwargs)
    return values


def _construct_mapping(loader: yaml.Loader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise DuplicateKeyError(f"duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def _error_text(error: BaseException) -> str:
    message = str(error).strip()
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


__all__ = [
    "ConfigError",
    "DEFAULT_CONTEXT_TOKENS",
    "MODELS_FILE",
    "ModelProfile",
    "ModelRegistry",
    "RESERVED_MODEL_KWARGS",
    "load_model_registry",
    "models_path",
]
