"""Single model-profile factory and assignment resolver."""

from __future__ import annotations

from typing import Any, Literal

from langchain_anyllm import ChatAnyLLM

from config.llm import ConfigError, ModelProfile, ModelRegistry
from config.metadata import ModelMetadata, apply_model_metadata
from config.settings import (
    MAIN_MODEL,
    RUBRIC_MODEL,
    SUMMARIZATION_MODEL,
    context_limit_tokens,
    model_assignment,
    rubric_enabled,
)
from runtime.issues import Issue

ModelRole = Literal["main", "rubric", "summarization"]
STREAM_USAGE_PROVIDERS = {"lmstudio"}


def model_registry(config: dict[str, Any]) -> ModelRegistry:
    value = config.get("model_registry")
    return value if isinstance(value, ModelRegistry) else ModelRegistry()


def assigned_profile_name(config: dict[str, Any], role: ModelRole) -> str | None:
    """Return the effective profile name for a role, including Main inheritance."""
    explicit = model_assignment(config, role)
    if explicit:
        return explicit
    if role != MAIN_MODEL:
        return model_assignment(config, MAIN_MODEL)
    return None


def resolve_model_profile(config: dict[str, Any], role: ModelRole = MAIN_MODEL) -> ModelProfile:
    """Resolve one required effective assignment without silent fallback."""
    name = assigned_profile_name(config, role)
    if not name:
        if role == MAIN_MODEL:
            raise ConfigError("Main model is not configured. Run /models.")
        raise ConfigError(f"{role.title()} model cannot inherit because Main is not configured. Run /models.")
    profile = model_registry(config).profile(name)
    if profile is None:
        raise ConfigError(f"{role.title()} model profile '{name}' is unavailable. Open Issues or run /models.")
    return profile


def get_llm(
    config: dict[str, Any],
    metadata: ModelMetadata | None = None,
    *,
    role: ModelRole = MAIN_MODEL,
) -> ChatAnyLLM:
    """Create a model through the sole registry-backed ChatAnyLLM factory."""
    profile = resolve_model_profile(config, role)
    return _create_profile_model(config, profile, metadata)


def get_profile_model(
    config: dict[str, Any],
    profile_name: str,
    metadata: ModelMetadata | None = None,
) -> ChatAnyLLM:
    """Create one explicitly named registry profile through the shared factory."""
    profile = model_registry(config).profile(profile_name)
    if profile is None:
        raise ConfigError(f"Model profile '{profile_name}' is unavailable. Open Issues or run /models.")
    return _create_profile_model(config, profile, metadata)


def _create_profile_model(
    config: dict[str, Any],
    profile: ModelProfile,
    metadata: ModelMetadata | None,
) -> ChatAnyLLM:
    values = dict(profile.values)
    model_kwargs = dict(values.pop("model_kwargs", {}) or {})
    if profile.provider.lower() in STREAM_USAGE_PROVIDERS:
        values["stream_options"] = {"include_usage": True}
    if config.get("llm_direct"):
        import httpx

        model_kwargs["client_args"] = {
            "http_client": httpx.AsyncClient(trust_env=False, verify=False),
        }
    if model_kwargs:
        values["model_kwargs"] = model_kwargs
    model = ChatAnyLLM(**values)
    selected_metadata = metadata or ModelMetadata(context_tokens=context_limit_tokens(config))
    return apply_model_metadata(model, selected_metadata)


def get_model_name(config: dict[str, Any]) -> str:
    """Return Main's profile-qualified identity, or the fresh-workspace label."""
    name = model_assignment(config, MAIN_MODEL)
    profile = model_registry(config).profile(name)
    if profile is None:
        return "unset" if not name else f"[{name}] unavailable"
    return profile.identity


def get_rubric_model_name(config: dict[str, Any]) -> str:
    profile = resolve_model_profile(config, RUBRIC_MODEL)
    return profile.identity


def model_unavailable_message(config: dict[str, Any]) -> str:
    """Return the concise execution-boundary error for the current Main assignment."""
    name = model_assignment(config, MAIN_MODEL)
    if not name:
        return "Main model is not configured. Run /models."
    if model_registry(config).profile(name) is None:
        return f"Main model profile '{name}' is unavailable. Run /models."
    return "Model-backed execution is unavailable. Open Issues or run /models."


def active_model_issues(config: dict[str, Any]) -> list[Issue]:
    """Validate only assignments required by active core components."""
    issues: list[Issue] = []
    _append_assignment_issue(config, MAIN_MODEL, issues)
    _append_assignment_issue(config, SUMMARIZATION_MODEL, issues)
    if rubric_enabled(config):
        _append_assignment_issue(config, RUBRIC_MODEL, issues)
    return issues


def _append_assignment_issue(config: dict[str, Any], role: ModelRole, issues: list[Issue]) -> None:
    explicit = model_assignment(config, role)
    effective = assigned_profile_name(config, role)
    if effective and model_registry(config).profile(effective) is not None:
        return
    if role != MAIN_MODEL and not explicit and not model_assignment(config, MAIN_MODEL):
        return  # Main owns the single missing-root diagnostic.
    if role == MAIN_MODEL and not effective:
        issues.append(
            Issue(
                "MODEL",
                "Main model is not configured",
                ".mira/settings.yml: models.main",
                "Model-backed MIRA is unavailable until Main is selected.",
                "Define a profile in .mira/models.yml, run /reload, then select it with /models.",
            )
        )
        return
    name = effective or "unset"
    issues.append(
        Issue(
            "MODEL",
            f"Missing {role} model profile: {name}",
            f".mira/settings.yml: models.{role}",
            f"The active {role} assignment explicitly requires '{name}'.",
            "Add the profile back to .mira/models.yml or select a valid profile with /models.",
        )
    )


__all__ = [
    "active_model_issues",
    "assigned_profile_name",
    "get_llm",
    "get_model_name",
    "get_profile_model",
    "model_unavailable_message",
    "get_rubric_model_name",
    "model_registry",
    "resolve_model_profile",
]
