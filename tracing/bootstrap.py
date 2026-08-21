"""Configure OpenInference LangChain tracing for generic OTLP/HTTP export."""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from typing import Any

from config.interpolation import EnvironmentInterpolationError, resolve_environment
from config.settings import tracing_enabled, tracing_settings
from config.tracing import TracingRegistry
from runtime.issues import Issue

_FLUSH_TIMEOUT_MILLIS = 1_000
_LANGSMITH_TRACING_KEYS = (
    "LANGSMITH_TRACING",
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING",
    "LANGCHAIN_TRACING_V2",
)
_active_instrumentor: Any = None
_active_provider: Any = None


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
            {"endpoint": profile.endpoint, "headers": profile.headers},
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
            LangChainInstrumentor,
            Resource,
            ResourceAttributes,
            TracerProvider,
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
    instrumentor = None
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
        provider.add_span_processor(BatchSpanProcessor(exporter))
        instrumentor = LangChainInstrumentor()
        instrumentor.instrument(tracer_provider=provider)
    except Exception as exc:
        _teardown_runtime(instrumentor, provider)
        return [
            _tracing_issue(
                "Tracing could not be initialized",
                f"{type(exc).__name__}: optional OpenInference OTLP runtime initialization failed.",
            )
        ]

    global _active_instrumentor, _active_provider
    _active_instrumentor = instrumentor
    _active_provider = provider
    return _external_langsmith_issues(target)


def shutdown_tracing() -> None:
    """Idempotently detach instrumentation and drain the MIRA-owned provider."""
    global _active_instrumentor, _active_provider
    instrumentor, provider = _active_instrumentor, _active_provider
    _active_instrumentor = None
    _active_provider = None
    _teardown_runtime(instrumentor, provider)


def tracing_yaml_fragment(
    *,
    enabled: bool,
    profile: str,
    endpoint: str,
    headers: Mapping[str, str],
) -> str:
    """Render the unresolved effective tracing fragment used by the preview."""
    import yaml

    return yaml.safe_dump(
        {
            "tracing": {
                "enabled": bool(enabled),
                "profile": profile,
                "endpoint": endpoint.strip(),
                "headers": dict(headers),
            }
        },
        sort_keys=False,
    ).rstrip()


def _load_runtime() -> tuple[Any, Any, Any, Any, Any, Any]:
    """Import optional tracing dependencies only after tracing is enabled."""
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.semconv.resource import ResourceAttributes
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    return (
        LangChainInstrumentor,
        Resource,
        ResourceAttributes,
        TracerProvider,
        BatchSpanProcessor,
        OTLPSpanExporter,
    )


def _teardown_runtime(instrumentor: Any, provider: Any) -> None:
    if instrumentor is not None:
        try:
            instrumentor.uninstrument()
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


def _external_langsmith_issues(environ: Mapping[str, str]) -> list[Issue]:
    enabled_keys = [key for key in _LANGSMITH_TRACING_KEYS if _is_true(environ.get(key))]
    if not enabled_keys:
        return []
    names = " or ".join(enabled_keys)
    return [
        Issue(
            "STARTUP",
            "External LangSmith tracing is also enabled",
            "process environment",
            (
                f"External setting(s) {names} enable LangSmith's native tracing while "
                "MIRA's OpenInference tracing is active, so duplicate traces may be produced."
            ),
            (
                "Disable external LangSmith tracing if MIRA's configured OpenInference "
                "path should be the only trace source."
            ),
        )
    ]


def _is_true(value: str | None) -> bool:
    return bool(value and value.strip().lower() in {"true", "1"})


def _tracing_issue(summary: str, details: str) -> Issue:
    if "dependencies" in summary.lower():
        guidance = 'Install the tracing extra with: pip install "mira[tracing]", then run /reload-runtime.'
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
