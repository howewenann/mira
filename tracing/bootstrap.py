"""Configure LangChain's existing LangSmith tracer for generic OTLP/HTTP."""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from typing import Any
from urllib.parse import quote

import yaml

from config.interpolation import EnvironmentInterpolationError, resolve_environment
from config.settings import tracing_enabled, tracing_settings
from runtime.issues import Issue

_ENVIRONMENT_KEYS = (
    "LANGSMITH_TRACING",
    "LANGSMITH_TRACING_MODE",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_HEADERS",
)
_active_client: Any = None
_active_provider: Any = None
_saved_environment: dict[str, str | None] | None = None


def configure_tracing(
    settings: dict[str, Any] | None,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> list[Issue]:
    """Apply the current tracing settings without making tracing a startup dependency."""
    target = os.environ if environ is None else environ
    if not tracing_enabled(settings):
        _disable_tracing(target)
        return []

    values = tracing_settings(settings)
    try:
        resolved = resolve_environment(values, environ=target)
    except EnvironmentInterpolationError as exc:
        _disable_tracing(target)
        reason = str(exc).partition(";")[0]
        details = (
            f"{reason}; define referenced values in the process environment "
            "before starting or reloading MIRA"
        )
        return [_tracing_issue("Tracing configuration could not be resolved", details)]

    try:
        langsmith, TracerProvider, BatchSpanProcessor, OTLPSpanExporter = _load_runtime()
    except (ImportError, ModuleNotFoundError):
        _disable_tracing(target)
        return [
            _tracing_issue(
                "Tracing dependencies are not installed",
                "The optional generic OTLP tracing runtime is unavailable.",
            )
        ]

    _disable_tracing(target)
    _remember_environment(target)
    endpoint = str(resolved["endpoint"])
    headers = dict(resolved["headers"])
    target.update(
        {
            "LANGSMITH_TRACING": "true",
            "LANGSMITH_TRACING_MODE": "otel",
            "OTEL_EXPORTER_OTLP_ENDPOINT": endpoint,
            "OTEL_EXPORTER_OTLP_HEADERS": serialize_headers(headers),
        }
    )

    global _active_client, _active_provider
    client = None
    provider = None
    try:
        exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(exporter))
        client = langsmith.Client(tracing_mode="otel", otel_tracer_provider=provider)
        langsmith.configure(client=client, enabled=True)
    except Exception as exc:
        try:
            if client is not None:
                client.close(timeout=1.0)
        except Exception:
            pass
        try:
            if provider is not None:
                provider.shutdown()
        except Exception:
            pass
        _restore_environment(target)
        return [
            _tracing_issue(
                "Tracing could not be initialized",
                f"{type(exc).__name__}: optional OTLP runtime initialization failed.",
            )
        ]
    _active_client = client
    _active_provider = provider
    return []


def serialize_headers(headers: Mapping[str, str]) -> str:
    """Serialize headers using the URL-encoded OTLP environment format."""
    return ",".join(
        f"{quote(key, safe='-._~')}={quote(value, safe='-._~')}"
        for key, value in headers.items()
    )


def parse_headers_yaml(text: str) -> dict[str, str]:
    """Parse a YAML mapping suitable for OTLP HTTP headers."""
    if not text.strip():
        return {}
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError("headers must be valid YAML") from exc
    if not isinstance(parsed, dict):
        raise ValueError("headers must be a YAML mapping")
    if not all(
        isinstance(key, str) and key.strip() and isinstance(value, str)
        for key, value in parsed.items()
    ):
        raise ValueError("header names and values must be strings")
    return dict(parsed)


def tracing_yaml_fragment(*, enabled: bool, endpoint: str, headers: Mapping[str, str]) -> str:
    """Render the exact unresolved settings fragment used by the preview."""
    return yaml.safe_dump(
        {
            "tracing": {
                "enabled": bool(enabled),
                "endpoint": endpoint.strip(),
                "headers": dict(headers),
            }
        },
        sort_keys=False,
    ).rstrip()


def _load_runtime() -> tuple[Any, Any, Any, Any]:
    """Import optional tracing dependencies only after tracing is enabled."""
    import langsmith
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    return langsmith, TracerProvider, BatchSpanProcessor, OTLPSpanExporter


def _disable_tracing(environ: MutableMapping[str, str]) -> None:
    global _active_client, _active_provider
    if _active_client is None and _active_provider is None:
        return
    client, provider = _active_client, _active_provider
    _active_client = None
    _active_provider = None
    try:
        import langsmith

        langsmith.configure(client=None, enabled=None)
    except Exception:
        pass
    try:
        client.close(timeout=1.0)
    except Exception:
        pass
    try:
        provider.shutdown()
    except Exception:
        pass
    _restore_environment(environ)


def _remember_environment(environ: Mapping[str, str]) -> None:
    global _saved_environment
    _saved_environment = {key: environ.get(key) for key in _ENVIRONMENT_KEYS}


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
    return Issue(
        "STARTUP",
        summary,
        ".mira/settings.yml / tracing",
        details,
        'Install the tracing extra with: pip install "mira[tracing]", then run /reload-runtime.',
    )
