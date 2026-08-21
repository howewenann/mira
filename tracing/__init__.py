"""Optional generic OTLP tracing support."""

from tracing.bootstrap import configure_tracing, shutdown_tracing, tracing_yaml_fragment

__all__ = ["configure_tracing", "shutdown_tracing", "tracing_yaml_fragment"]
