"""Configured model metadata discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class ModelMetadata:
    """Metadata MIRA needs before building an agent."""

    context_tokens: int | None = None
    context_source: str = "unknown"


async def infer_model_metadata(config: dict[str, Any], model: Any | None = None) -> ModelMetadata:
    """Infer metadata for the configured model."""
    configured_limit = positive_int(config.get("llm_context_tokens"))
    if configured_limit:
        return ModelMetadata(configured_limit, "MIRA_LLM_CONTEXT_TOKENS")

    provider = str(config.get("llm_provider") or "").lower()
    if provider == "lmstudio":
        metadata = await infer_lmstudio_metadata(config)
        if metadata.context_tokens:
            return metadata

    profile_limit = profile_context_tokens(model)
    if profile_limit:
        return ModelMetadata(profile_limit, "model_profile.max_input_tokens")

    return ModelMetadata()


def apply_model_metadata(model: Any, metadata: ModelMetadata) -> Any:
    """Attach inferred metadata to a LangChain model object."""
    if not metadata.context_tokens:
        return model

    profile = getattr(model, "profile", None)
    profile = dict(profile) if isinstance(profile, dict) else {}
    profile["max_input_tokens"] = metadata.context_tokens
    model.profile = profile
    return model


async def infer_lmstudio_metadata(config: dict[str, Any]) -> ModelMetadata:
    """Read the loaded LM Studio context length from the REST API."""
    base_url = str(config.get("llm_base_url") or "").rstrip("/")
    if not base_url:
        return ModelMetadata()

    root_url = base_url[:-3] if base_url.endswith("/v1") else base_url
    url = f"{root_url}/api/v1/models"
    headers = {"Authorization": f"Bearer {config.get('llm_api_key') or 'lm-studio'}"}
    try:
        async with httpx.AsyncClient(trust_env=False, verify=False, timeout=None) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            payload = response.json()
    except Exception:
        return ModelMetadata()

    context_tokens = lmstudio_context_tokens(payload, str(config.get("llm_model") or ""))
    if not context_tokens:
        return ModelMetadata()
    return ModelMetadata(context_tokens, "lmstudio.api.v1.loaded_instance")


def lmstudio_context_tokens(payload: Any, model_name: str) -> int | None:
    """Return the loaded instance context length for a configured LM Studio model."""
    if not isinstance(payload, dict):
        return None

    models = payload.get("models")
    if not isinstance(models, list):
        models = payload.get("data")
    if not isinstance(models, list):
        return None

    for model in models:
        if not isinstance(model, dict) or not lmstudio_model_matches(model, model_name):
            continue
        context_tokens = loaded_instance_context_tokens(model.get("loaded_instances"), model_name)
        if context_tokens:
            return context_tokens
    return None


def lmstudio_model_matches(model: dict[str, Any], model_name: str) -> bool:
    """Return whether one LM Studio model entry describes the configured model."""
    candidates = {
        str(model.get("key") or ""),
        str(model.get("id") or ""),
        str(model.get("model") or ""),
        str(model.get("selected_variant") or ""),
    }
    variants = model.get("variants")
    if isinstance(variants, list):
        candidates.update(str(item or "") for item in variants)

    if model_name in candidates:
        return True
    return any(candidate.startswith(f"{model_name}@") for candidate in candidates if candidate)


def loaded_instance_context_tokens(instances: Any, model_name: str) -> int | None:
    """Return context length from the matching loaded instance."""
    if not isinstance(instances, list):
        return None

    fallback: int | None = None
    for instance in instances:
        if not isinstance(instance, dict):
            continue
        context_tokens = positive_int((instance.get("config") or {}).get("context_length"))
        if not context_tokens:
            continue
        instance_id = str(instance.get("id") or "")
        if instance_id == model_name or instance_id.startswith(f"{model_name}@"):
            return context_tokens
        fallback = fallback or context_tokens
    return fallback


def profile_context_tokens(model: Any | None) -> int | None:
    """Return `model.profile.max_input_tokens` when available."""
    profile = getattr(model, "profile", None)
    if not isinstance(profile, dict):
        return None
    return positive_int(profile.get("max_input_tokens")) or None


def positive_int(value: Any) -> int:
    """Return a non-negative integer from loose provider metadata."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0
