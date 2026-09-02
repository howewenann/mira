"""Configure LangSmith LangChain tracing for generic OTLP/HTTP export."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Awaitable, Callable, Mapping, MutableMapping
from functools import wraps
from typing import Any, ParamSpec, TypeVar

from config.interpolation import EnvironmentInterpolationError, resolve_environment
from config.settings import tracing_enabled, tracing_settings
from config.tracing import TracingRegistry
from core.diagnostics.issues import Issue

_FLUSH_TIMEOUT_MILLIS = 1_000
_OWNED_ENVIRONMENT_KEYS = (
    "LANGSMITH_TRACING",
    "LANGSMITH_TRACING_MODE",
    "LANGCHAIN_REVISION_ID",
)
_active_langsmith: Any = None
_active_client: Any = None
_active_provider: Any = None
_active_environment: MutableMapping[str, str] | None = None
_saved_environment: dict[str, str | None] | None = None

_P = ParamSpec("_P")
_T = TypeVar("_T")


def trace_user_turn(
    function: Callable[_P, Awaitable[_T]],
) -> Callable[_P, Awaitable[_T]]:
    """Wrap one complete user turn in the active LangSmith trace context."""

    @wraps(function)
    async def traced(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        langsmith, client = _active_langsmith, _active_client
        if langsmith is None or client is None:
            return await function(*args, **kwargs)
        visible_text = kwargs.get("display_text")
        if visible_text is None:
            visible_text = kwargs.get("text", "")
        async with langsmith.trace(
            "MIRA Turn",
            run_type="chain",
            inputs={"input": visible_text},
            client=client,
        ) as turn:
            result = await function(*args, **kwargs)
            turn.set(
                outputs={"output": getattr(result, "final_text", "")},
                usage_metadata={
                    "input_tokens": getattr(result, "input_tokens", 0),
                    "output_tokens": getattr(result, "output_tokens", 0),
                    "total_tokens": getattr(result, "total_tokens", 0),
                },
            )
            turn.end()
            return result

    return traced


def configure_tracing(
    settings: dict[str, Any] | None,
    registry: TracingRegistry | None = None,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> list[Issue]:
    """Apply the current tracing settings without making tracing a startup dependency."""
    target = os.environ if environ is None else environ
    if not tracing_enabled(settings):
        shutdown_tracing()
        return []

    selected = str(tracing_settings(settings)["profile"])
    if registry is None or registry.profile(selected) is None:
        shutdown_tracing()
        if registry is not None and selected in registry.invalid_names:
            return [
                _tracing_issue(
                    f"Tracing profile '{selected}' is invalid",
                    f"Correct profile '{selected}' in .mira/tracing.yml.",
                )
            ]
        return [
            _tracing_issue(
                f"Tracing profile '{selected}' was not found in .mira/tracing.yml.",
                "Select an available profile or add it to .mira/tracing.yml.",
            )
        ]

    profile = registry.profile(selected)
    assert profile is not None
    try:
        resolved = resolve_environment(
            {
                "endpoint": profile.endpoint,
                "headers": profile.headers,
                "span_attributes": profile.span_attributes,
            },
            environ=target,
        )
    except EnvironmentInterpolationError as exc:
        shutdown_tracing()
        reason = str(exc).partition(";")[0]
        details = (
            f"{reason}; define referenced values in the process environment "
            "before starting or reloading MIRA"
        )
        return [_tracing_issue("Tracing configuration could not be resolved", details)]

    shutdown_tracing()
    try:
        (
            langsmith,
            Resource,
            ResourceAttributes,
            TracerProvider,
            LangSmithOpenInferenceProcessor,
            BatchSpanProcessor,
            OTLPSpanExporter,
        ) = _load_runtime()
    except (ImportError, ModuleNotFoundError):
        return [
            _tracing_issue(
                "Tracing dependencies are not installed",
                "The optional OpenInference OTLP tracing runtime is unavailable.",
            )
        ]

    endpoint = str(resolved["endpoint"])
    headers = dict(resolved["headers"])
    span_attributes = dict(resolved["span_attributes"])
    client = None
    provider = None
    try:
        resource = Resource.create(
            {
                "service.name": "MIRA",
                ResourceAttributes.PROJECT_NAME: "MIRA",
            }
        )
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)
        provider.add_span_processor(LangSmithOpenInferenceProcessor(span_attributes))
        provider.add_span_processor(BatchSpanProcessor(exporter))
        _remember_environment(target)
        if "LANGCHAIN_REVISION_ID" not in target:
            target["LANGCHAIN_REVISION_ID"] = _safe_revision_id() or "unknown"
        target.update(
            {
                "LANGSMITH_TRACING": "true",
                "LANGSMITH_TRACING_MODE": "otel",
            }
        )
        client = langsmith.Client(tracing_mode="otel", otel_tracer_provider=provider)
        langsmith.configure(client=client, enabled=True)
    except Exception as exc:
        _teardown_runtime(langsmith, client, provider, target)
        return [
            _tracing_issue(
                "Tracing could not be initialized",
                f"{type(exc).__name__}: optional OpenInference OTLP runtime initialization failed.",
            )
        ]

    global _active_langsmith, _active_client, _active_provider, _active_environment
    _active_langsmith = langsmith
    _active_client = client
    _active_provider = provider
    _active_environment = target
    return []


def shutdown_tracing() -> None:
    """Idempotently disable callbacks and drain the MIRA-owned provider."""
    global _active_langsmith, _active_client, _active_provider, _active_environment
    langsmith = _active_langsmith
    client, provider, environ = _active_client, _active_provider, _active_environment
    _active_langsmith = None
    _active_client = None
    _active_provider = None
    _active_environment = None
    _teardown_runtime(langsmith, client, provider, environ)


def tracing_yaml_fragment(
    *,
    enabled: bool,
    profile: str,
    middleware_spans: str,
    endpoint: str,
    headers: Mapping[str, str],
    span_attributes: Mapping[str, Any],
) -> str:
    """Render the unresolved effective tracing fragment used by the preview."""
    import yaml

    return yaml.safe_dump(
        {
            "tracing": {
                "enabled": bool(enabled),
                "profile": profile,
                "middleware_spans": middleware_spans,
                "endpoint": endpoint.strip(),
                "headers": dict(headers),
                "span_attributes": dict(span_attributes),
            }
        },
        sort_keys=False,
    ).rstrip()


def _load_runtime() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Import optional tracing dependencies only after tracing is enabled."""
    import langsmith
    from openinference.semconv.resource import ResourceAttributes
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from tracing.semantic_processor import LangSmithOpenInferenceProcessor

    return (
        langsmith,
        Resource,
        ResourceAttributes,
        TracerProvider,
        LangSmithOpenInferenceProcessor,
        BatchSpanProcessor,
        OTLPSpanExporter,
    )


def _safe_revision_id() -> str | None:
    """Read the Git revision without allowing the subprocess to inherit stdin."""
    try:
        revision = subprocess.check_output(
            ["git", "describe", "--tags", "--always", "--dirty"],
            stdin=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            encoding="utf-8",
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return revision.strip() or None


def _teardown_runtime(
    langsmith: Any,
    client: Any,
    provider: Any,
    environ: MutableMapping[str, str] | None,
) -> None:
    if langsmith is not None:
        try:
            langsmith.configure(client=None, enabled=None)
        except Exception:
            pass
    if client is not None:
        try:
            client.close(timeout=1.0)
        except Exception:
            pass
    if provider is not None:
        try:
            provider.force_flush(timeout_millis=_FLUSH_TIMEOUT_MILLIS)
        except Exception:
            pass
        try:
            provider.shutdown()
        except Exception:
            pass
    if environ is not None:
        _restore_environment(environ)


def _remember_environment(environ: Mapping[str, str]) -> None:
    global _saved_environment
    _saved_environment = {key: environ.get(key) for key in _OWNED_ENVIRONMENT_KEYS}


def _restore_environment(environ: MutableMapping[str, str]) -> None:
    global _saved_environment
    if _saved_environment is None:
        return
    for key, value in _saved_environment.items():
        if value is None:
            environ.pop(key, None)
        else:
            environ[key] = value
    _saved_environment = None


def _tracing_issue(summary: str, details: str) -> Issue:
    if "dependencies" in summary.lower():
        guidance = (
            'Install the tracing extra with: pip install "mira[tracing]", '
            "then run /reload-runtime."
        )
    elif "initialized" in summary.lower():
        guidance = "Review the tracing error and run /reload-runtime; MIRA remains usable."
    else:
        guidance = "Correct the tracing profile and run /reload-runtime."
    return Issue(
        "STARTUP",
        summary,
        ".mira/tracing.yml",
        details,
        guidance,
    )
